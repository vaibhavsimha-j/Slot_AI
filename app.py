"""SLOT AI — General Purpose Scheduling Agent
pip install streamlit langchain-groq langgraph langchain-core ortools pydantic pandas openpyxl fpdf2
streamlit run app.py
"""
import os, json, copy, uuid, threading
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, TypedDict
import streamlit as st
import pandas as pd
from io import BytesIO

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from ortools.sat.python import cp_model

# ── Config ──────────────────────────────────────────────────────────────────────
MODEL = "llama-3.1-8b-instant"
st.set_page_config(page_title="SLOT AI", page_icon="📅", layout="wide")

# ── Session Init ────────────────────────────────────────────────────────────────
_DEFAULTS = dict(
    api_key="",
    constraints={}, timetable={}, timetable_history=[],
    display_messages=[], tt_updated=False,
    thread_id=str(uuid.uuid4()),
    page="chat",
    chat_sessions=[],
    active_thread_id="",
    renaming_chat_tid=None,
)
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)
if not st.session_state.api_key and os.getenv("GROQ_API_KEY"):
    st.session_state.api_key = os.getenv("GROQ_API_KEY", "")
if "thread_memories" not in st.session_state:
    st.session_state.thread_memories = {}
if "thread_agents" not in st.session_state:
    st.session_state.thread_agents = {}

# Bootstrap the first chat session on a fresh load
if not st.session_state.chat_sessions:
    _init_tid = st.session_state.thread_id
    st.session_state.chat_sessions = [{
        "thread_id": _init_tid, "title": "New Chat",
        "display_messages": [], "constraints": {},
        "timetable": {}, "timetable_history": [],
    }]
    st.session_state.active_thread_id = _init_tid
elif not st.session_state.active_thread_id:
    # Recover active pointer after a hot-reload
    st.session_state.active_thread_id = st.session_state.chat_sessions[0]["thread_id"]
    st.session_state.thread_id = st.session_state.active_thread_id

# ── Chat session helpers ──────────────────────────────────────────────────────
def _chat_title(msgs: list) -> str:
    """Derive a short title from the first user message."""
    for m in msgs:
        if m["role"] == "user":
            t = m["content"].strip().replace("\n", " ")
            return (t[:42] + "…") if len(t) > 42 else t
    return "New Chat"

def _save_current_chat():
    """Flush active session_state back into chat_sessions."""
    tid = st.session_state.active_thread_id
    for s in st.session_state.chat_sessions:
        if s["thread_id"] == tid:
            s["display_messages"]  = list(st.session_state.display_messages)
            s["constraints"]       = copy.deepcopy(st.session_state.constraints)
            s["timetable"]         = copy.deepcopy(st.session_state.timetable)
            s["timetable_history"] = list(st.session_state.timetable_history)
            s["title"]             = _chat_title(s["display_messages"])
            break

def _load_chat(tid: str):
    """Load a stored chat into active session_state (does NOT rerun)."""
    for s in st.session_state.chat_sessions:
        if s["thread_id"] == tid:
            st.session_state.active_thread_id  = tid
            st.session_state.thread_id         = tid
            st.session_state.display_messages  = list(s["display_messages"])
            st.session_state.constraints       = copy.deepcopy(s["constraints"])
            st.session_state.timetable         = copy.deepcopy(s["timetable"])
            st.session_state.timetable_history = list(s["timetable_history"])
            st.session_state.page              = "chat"
            break

def _new_chat():
    """Save current, create a blank chat, and set it active."""
    _save_current_chat()
    new_tid = str(uuid.uuid4())
    st.session_state.chat_sessions.insert(0, {
        "thread_id": new_tid, "title": "New Chat",
        "display_messages": [], "constraints": {},
        "timetable": {}, "timetable_history": [],
    })
    _load_chat(new_tid)

def _get_thread_memory(tid: str) -> MemorySaver:
    if tid not in st.session_state.thread_memories:
        st.session_state.thread_memories[tid] = MemorySaver()
    return st.session_state.thread_memories[tid]

def _get_agent(tid: str):
    if tid not in st.session_state.thread_agents:
        st.session_state.thread_agents[tid] = _build_graph(_get_thread_memory(tid))
    return st.session_state.thread_agents[tid]

def _clear_thread_memory(tid: str):
    st.session_state.thread_memories.pop(tid, None)
    st.session_state.thread_agents.pop(tid, None)
    _STORE.pop(tid, None)

def _delete_chat(tid: str):
    """Remove a chat; switch to another if it was the active one."""
    _clear_thread_memory(tid)
    st.session_state.chat_sessions = [
        s for s in st.session_state.chat_sessions if s["thread_id"] != tid
    ]
    if st.session_state.active_thread_id == tid:
        if st.session_state.chat_sessions:
            _load_chat(st.session_state.chat_sessions[0]["thread_id"])
        else:
            _new_chat()

# ── Session store — tools cannot access st.session_state, so we use a thread-local
# to carry the active session id into tool calls (LangGraph invoke is synchronous,
# so the tool always runs on the same OS thread as the Streamlit script).
_tl    = threading.local()          # carries session_id for the current thread
_STORE: dict[str, dict] = {}        # per-session state dicts, keyed by thread_id

def _s() -> dict:
    """Return the mutable state dict for the current session."""
    sid = getattr(_tl, "session_id", "")
    if not sid or sid not in _STORE:
        sid = "_default"
    return _STORE.setdefault(sid, {
        "api_key": "", "model": MODEL,
        "constraints": {}, "timetable": {}, "timetable_history": [], "tt_updated": False,
    })

# ── Utilities ───────────────────────────────────────────────────────────────────
def _get_llm(t: float = 0.0) -> ChatGroq:
    # api_key is injected into os.environ["GROQ_API_KEY"] before every agent invocation
    # so ChatGroq picks it up automatically — avoids asyncio context propagation issues
    return ChatGroq(
        model=MODEL,
        temperature=t,
    )

def _cell(e):
    if isinstance(e, dict):
        s = e.get("subject", "")
        return f"{s} ({e.get('professor', '')})" if s not in ("FREE", "", None) else "—"
    return str(e).strip() or "—"

def _to_dfs(tt: dict) -> dict:
    out = {}
    for day, data in tt.items():
        if not data:
            continue
        rooms = list(next(iter(data.values())))
        slots = list(data)
        out[day] = pd.DataFrame(
            {r: [_cell(data[s].get(r, "—")) for s in slots] for r in rooms},
            index=pd.Index(slots, name="Time"),
        )
    return out

def _export_day_pdf(day: str, df: pd.DataFrame) -> bytes | None:
    try:
        from fpdf import FPDF
        _enc = lambda t: str(t).encode("latin-1", "replace").decode("latin-1")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, _enc(day))
        pdf.ln()
        pdf.set_font("Helvetica", size=7)
        cols = ["Time"] + list(df.columns)
        w = 190 / len(cols)
        pdf.set_fill_color(220, 220, 220)
        for col in cols:
            pdf.cell(w, 7, _enc(col), border=1, fill=True)
        pdf.ln()
        for idx, row in df.iterrows():
            pdf.cell(w, 7, _enc(str(idx)), border=1)
            for v in row:
                pdf.cell(w, 7, _enc(str(v))[:20], border=1)
            pdf.ln()
        return bytes(pdf.output())
    except Exception:
        return None


def _export_day_xlsx(day: str, df: pd.DataFrame) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name=day[:31])
    return buf.getvalue()


def _show_tt(tt: dict, key_prefix: str = "tt"):
    dfs = _to_dfs(tt)
    for day, df in dfs.items():
        safe = day.replace(" ", "_")
        hdr_col, pdf_col, xl_col = st.columns([5, 1, 1])
        hdr_col.markdown(f"**{day}**")
        pdf_bytes = _export_day_pdf(day, df)
        if pdf_bytes:
            pdf_col.download_button(
                "⬇️ PDF", pdf_bytes, f"{safe}.pdf", "application/pdf",
                key=f"{key_prefix}_pdf_{safe}", use_container_width=True,
            )
        xl_col.download_button(
            "⬇️ Excel", _export_day_xlsx(day, df), f"{safe}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{key_prefix}_xl_{safe}", use_container_width=True,
        )
        st.dataframe(df, use_container_width=True)

def _missing(c: dict) -> list:
    return [m for k, m in [
        ("teachers", "professor→subject mapping"),
        ("slots",    "time slots"),
        ("rooms",    "room names"),
        ("days",     "working days"),
    ] if not c.get(k)]

# ── OR-Tools CP-SAT Solver ──────────────────────────────────────────────────────
def _ortools_solve(constraints: dict) -> dict:
    teachers  = constraints.get("teachers", {})
    rooms     = constraints.get("rooms", [])
    slots     = constraints.get("slots", [])
    days      = constraints.get("days", [])

    sp       = {s: p for p, ss in teachers.items() for s in ss}
    subjects = list(sp.keys())
    n_d, n_s, n_r, n_sub = len(days), len(slots), len(rooms), len(subjects)

    if not all([n_d, n_s, n_r, n_sub]):
        return {"status": "error", "reason": "Missing core constraints (teachers/rooms/slots/days)."}

    model = cp_model.CpModel()
    x = {
        (d, s, r, sub): model.NewBoolVar(f"x{d}{s}{r}{sub}")
        for d in range(n_d) for s in range(n_s)
        for r in range(n_r) for sub in range(n_sub)
    }

    # Each room has exactly one subject per slot
    for d in range(n_d):
        for s in range(n_s):
            for r in range(n_r):
                model.AddExactlyOne(x[d, s, r, sub] for sub in range(n_sub))

    # No professor double-booked in the same slot
    for d in range(n_d):
        for s in range(n_s):
            for prof, psubjs in teachers.items():
                ids = [subjects.index(sub) for sub in psubjs if sub in subjects]
                if ids:
                    model.Add(
                        sum(x[d, s, r, sid] for r in range(n_r) for sid in ids) <= 1
                    )

    # Professor unavailability
    for prof, unavail_days in constraints.get("professor_unavailability", {}).items():
        ids = [subjects.index(s) for s in teachers.get(prof, []) if s in subjects]
        for d, day in enumerate(days):
            if day in unavail_days:
                for s in range(n_s):
                    for r in range(n_r):
                        for sid in ids:
                            model.Add(x[d, s, r, sid] == 0)

    # Room restrictions: subject forbidden from certain rooms
    for subj, forbidden in constraints.get("room_restrictions", {}).items():
        if subj in subjects:
            sid = subjects.index(subj)
            for r, room in enumerate(rooms):
                if room in forbidden:
                    for d in range(n_d):
                        for s in range(n_s):
                            model.Add(x[d, s, r, sid] == 0)

    # Slot restrictions: subject allowed only in certain slots
    for subj, allowed in constraints.get("slot_restrictions", {}).items():
        if subj in subjects:
            sid = subjects.index(subj)
            for s, slot in enumerate(slots):
                if slot not in allowed:
                    for d in range(n_d):
                        for r in range(n_r):
                            model.Add(x[d, s, r, sid] == 0)

    # Max slots per day per professor
    for prof, max_s in constraints.get("max_slots_per_day", {}).items():
        ids = [subjects.index(s) for s in teachers.get(prof, []) if s in subjects]
        for d in range(n_d):
            model.Add(
                sum(x[d, s, r, sid]
                    for s in range(n_s) for r in range(n_r) for sid in ids) <= max_s
            )

    # Every subject must appear at least once per week across all days/rooms.
    # Per-day would conflict with professor unavailability (e.g. Vaibhav blocked Wednesday
    # → AI/ML can't appear that day, making per-day minimum impossible).
    for sub in range(n_sub):
        model.Add(
            sum(x[d, s, r, sub] for d in range(n_d) for s in range(n_s) for r in range(n_r)) >= 1
        )

    # Fixed assignments: pin specific cells
    for fa in constraints.get("fixed_assignments", []):
        try:
            d = days.index(fa["day"])
            s = slots.index(fa["slot"])
            r = rooms.index(fa["room"])
            sid = subjects.index(fa["subject"])
            model.Add(x[d, s, r, sid] == 1)
        except (ValueError, KeyError):
            pass

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    solver.parameters.num_search_workers = 4
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        tt = {}
        for d, day in enumerate(days):
            tt[day] = {}
            for s, slot in enumerate(slots):
                tt[day][slot] = {}
                for r, room in enumerate(rooms):
                    for sid, subject in enumerate(subjects):
                        if solver.Value(x[d, s, r, sid]) == 1:
                            tt[day][slot][room] = {
                                "subject": subject,
                                "professor": sp[subject],
                            }
        return {
            "status": "success",
            "timetable": tt,
            "solver_status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
        }
    if status == cp_model.INFEASIBLE:
        return {
            "status": "infeasible",
            "reason": (
                "No valid timetable exists with the current constraints. "
                "Check for conflicting rules (e.g. a professor unavailable on all days, "
                "or a subject restricted from all rooms)."
            ),
        }
    return {"status": "timeout", "reason": "Solver timed out. Try fewer constraints."}

# ── Constraint Merge ─────────────────────────────────────────────────────────────
def _merge(current: dict, parsed: dict) -> dict:
    c = copy.deepcopy(current)
    for k, v in parsed.items():
        if k in ("teachers", "room_restrictions", "professor_unavailability", "slot_restrictions"):
            if isinstance(v, dict):
                d = c.setdefault(k, {})
                for kk, vv in v.items():
                    if isinstance(vv, list):
                        e = d.setdefault(kk, [])
                        for i in vv:
                            if i not in e:
                                e.append(i)
                    else:
                        d[kk] = vv
        elif k == "max_slots_per_day" and isinstance(v, dict):
            c.setdefault(k, {}).update(v)
        elif k in ("custom_rules", "fixed_assignments") and isinstance(v, list):
            e = c.setdefault(k, [])
            for r in v:
                if r not in e:
                    e.append(r)
        elif v:
            # Protect established core fields from partial-update hallucinations
            if k in ("rooms", "slots", "days") and c.get(k):
                pass
            else:
                c[k] = v
    return c

# ── Tools ────────────────────────────────────────────────────────────────────────
@tool
def extract_constraints(
    teachers: Dict[str, List[str]],
    rooms: List[str],
    slots: List[str],
    days: List[str],
    room_restrictions: Optional[Dict[str, List[str]]] = None,
    slot_restrictions: Optional[Dict[str, List[str]]] = None,
    professor_unavailability: Optional[Dict[str, List[str]]] = None,
    max_slots_per_day: Optional[Dict[str, int]] = None,
    custom_rules: Optional[List[str]] = None,
) -> str:
    """Store timetable scheduling constraints. YOU (the agent) extract these from the
    user's message and pass them as typed arguments — no inner LLM call is made here.

    teachers: professor-name → list of subjects. Strip titles (Dr./Prof.).
      e.g. {"Vaibhav": ["AI", "ML"], "Simha": ["DSA", "AA"], "Bob": ["DBMS"]}
    rooms: list of room names.  e.g. ["A", "B", "C", "D"]
    slots: list of time slots.  e.g. ["09:00-09:50", "09:50-10:40", "10:55-11:45"]
    days:  list of working days. e.g. ["Monday", "Tuesday", "Wednesday"]
    room_restrictions: subjects forbidden from certain rooms.
      e.g. {"CN": ["D"]}  means CN must never be in Room D
    slot_restrictions: subjects restricted to specific slots only.
      e.g. {"AI": ["09:00-09:50", "09:50-10:40"], "ML": ["09:00-09:50", "09:50-10:40"]}
    professor_unavailability: days a professor cannot teach.
      e.g. {"Alice": ["Friday"]}
    max_slots_per_day: max classes a professor can have in one day.
      e.g. {"Vaibhav": 2}
    custom_rules: any other rules as plain-text strings.
    """
    sess    = _s()
    current = sess.get("constraints", {})
    parsed: dict = {"teachers": teachers, "rooms": rooms, "slots": slots, "days": days}
    if room_restrictions:         parsed["room_restrictions"]        = room_restrictions
    if slot_restrictions:         parsed["slot_restrictions"]         = slot_restrictions
    if professor_unavailability:  parsed["professor_unavailability"]  = professor_unavailability
    if max_slots_per_day:         parsed["max_slots_per_day"]         = max_slots_per_day
    if custom_rules:              parsed["custom_rules"]              = custom_rules

    merged = _merge(current, parsed)
    sess["constraints"] = merged
    miss = _missing(merged)
    return json.dumps({
        "status":      "ready" if not miss else "incomplete",
        "missing":     miss,
        "constraints": merged,
        "summary": {
            "professors": list(merged.get("teachers", {}).keys()),
            "subjects":   [s for ss in merged.get("teachers", {}).values() for s in ss],
            "rooms":      merged.get("rooms", []),
            "slots":      merged.get("slots", []),
            "days":       merged.get("days", []),
            "active_restrictions": (
                len(merged.get("room_restrictions", {}))
                + len(merged.get("slot_restrictions", {}))
                + len(merged.get("professor_unavailability", {}))
                + len(merged.get("max_slots_per_day", {}))
            ),
        },
    })


@tool
def solve_timetable() -> str:
    """Generate the timetable using OR-Tools CP-SAT constraint solver.
    Call this when all constraints are collected and the user wants a timetable,
    or after constraints are updated to regenerate."""
    sess = _s()
    c    = sess.get("constraints", {})
    miss = _missing(c)
    if miss:
        return json.dumps({"status": "incomplete", "missing": miss})

    result = _ortools_solve(c)

    if result["status"] == "success":
        existing = sess.get("timetable", {})
        if existing:
            sess.setdefault("timetable_history", []).append({
                "version":     len(sess.get("timetable_history", [])) + 1,
                "timetable":   copy.deepcopy(existing),
                "constraints": copy.deepcopy(c),
                "timestamp":   datetime.now().isoformat(),
            })
        sess["timetable"]   = result["timetable"]
        sess["tt_updated"]  = True
        return json.dumps({
            "status":        "success",
            "solver_status": result["solver_status"],
            "timetable":     result["timetable"],
            "days":          list(result["timetable"].keys()),
            "total_cells":   (
                len(c.get("slots", [])) *
                len(c.get("days",  [])) *
                len(c.get("rooms", []))
            ),
        })

    return json.dumps(result)


@tool
def edit_timetable(
    room_restrictions: Optional[Dict[str, List[str]]] = None,
    slot_restrictions: Optional[Dict[str, List[str]]] = None,
    professor_unavailability: Optional[Dict[str, List[str]]] = None,
    max_slots_per_day: Optional[Dict[str, int]] = None,
    fixed_assignments: Optional[List[Dict[str, str]]] = None,
    custom_rules: Optional[List[str]] = None,
    clear_fixed_assignments: bool = False,
) -> str:
    """Update constraints and regenerate the timetable using OR-Tools.
    Pass ONLY the constraints that changed. The solver re-runs with the merged constraints.

    room_restrictions: subjects forbidden from rooms. e.g. {"CN": ["D"]}
    slot_restrictions: subjects limited to certain slots. e.g. {"AI": ["09:00-09:50"]}
    professor_unavailability: professors unavailable on certain days. e.g. {"Bob": ["Friday"]}
    max_slots_per_day: teaching load limit per professor. e.g. {"Vaibhav": 2}
    fixed_assignments: pin a specific subject to a cell.
      e.g. [{"day":"Monday","slot":"09:00-09:50","room":"A","subject":"AI"}]
    custom_rules: additional rule text.
    clear_fixed_assignments: set True when replacing all pinned cells (e.g. a swap).
    """
    sess = _s()
    if not sess.get("timetable"):
        return json.dumps({"status": "error", "message": "No timetable exists yet. Generate one first."})

    current = sess.get("constraints", {})
    if clear_fixed_assignments:
        current.pop("fixed_assignments", None)

    delta: dict = {}
    if room_restrictions:         delta["room_restrictions"]        = room_restrictions
    if slot_restrictions:         delta["slot_restrictions"]         = slot_restrictions
    if professor_unavailability:  delta["professor_unavailability"]  = professor_unavailability
    if max_slots_per_day:         delta["max_slots_per_day"]         = max_slots_per_day
    if fixed_assignments:         delta["fixed_assignments"]         = fixed_assignments
    if custom_rules:              delta["custom_rules"]              = custom_rules

    merged = _merge(current, delta)
    sess["constraints"] = merged
    result = _ortools_solve(merged)

    if result["status"] == "success":
        existing = sess.get("timetable", {})
        if existing:
            sess.setdefault("timetable_history", []).append({
                "version":     len(sess.get("timetable_history", [])) + 1,
                "timetable":   copy.deepcopy(existing),
                "constraints": copy.deepcopy(merged),
                "timestamp":   datetime.now().isoformat(),
            })
        sess["timetable"]   = result["timetable"]
        sess["tt_updated"]  = True
        return json.dumps({
            "status":        "success",
            "solver_status": result["solver_status"],
            "timetable":     result["timetable"],
            "constraints":   merged,
            "days":          list(result["timetable"].keys()),
            "message":       "Timetable updated successfully.",
        })

    return json.dumps(result)


@tool
def validate_timetable() -> str:
    """Validate the current timetable against all active constraints.
    Returns a detailed list of any violations found (conflicts, banned rooms, overloads, etc.)."""
    sess = _s()
    tt = sess.get("timetable", {})
    c  = sess.get("constraints", {})
    if not tt:
        return json.dumps({"status": "error", "message": "No timetable to validate."})

    issues = []

    for day, slots in tt.items():
        for slot, rooms in slots.items():
            seen_profs: dict = {}
            for room, cell in rooms.items():
                if not isinstance(cell, dict) or not cell.get("subject"):
                    continue
                prof = cell.get("professor", "")
                subj = cell.get("subject",   "")

                if prof in seen_profs:
                    issues.append(
                        f"DOUBLE-BOOKING: {prof} in rooms {seen_profs[prof]} and {room} "
                        f"— {day} {slot}"
                    )
                seen_profs[prof] = room

                if room in c.get("room_restrictions", {}).get(subj, []):
                    issues.append(f"ROOM VIOLATION: {subj} in restricted room {room} — {day} {slot}")

                allowed = c.get("slot_restrictions", {}).get(subj)
                if allowed and slot not in allowed:
                    issues.append(f"SLOT VIOLATION: {subj} in non-allowed slot {slot} — {day}")

        for prof, ud in c.get("professor_unavailability", {}).items():
            if day in ud:
                for slot, rooms in slots.items():
                    for cell in rooms.values():
                        if isinstance(cell, dict) and cell.get("professor") == prof:
                            issues.append(
                                f"UNAVAILABILITY: {prof} scheduled on {day} at {slot}"
                            )

    for prof, max_s in c.get("max_slots_per_day", {}).items():
        for day, slots in tt.items():
            count = sum(
                1 for s_data in slots.values()
                for cell in s_data.values()
                if isinstance(cell, dict) and cell.get("professor") == prof
            )
            if count > max_s:
                issues.append(
                    f"OVERLOAD: {prof} teaches {count} slots on {day} (max allowed: {max_s})"
                )

    return json.dumps({
        "status":      "valid" if not issues else "violations_found",
        "issue_count": len(issues),
        "issues":      issues[:20],
        "message":     "Timetable is valid!" if not issues else f"{len(issues)} violation(s) found.",
    })


@tool
def get_timetable_history() -> str:
    """Return a summary of all previously generated timetable versions with timestamps.
    Call when user asks about past timetables or wants to compare or roll back."""
    history = _s().get("timetable_history", [])
    if not history:
        return json.dumps({"status": "empty", "message": "No timetable history yet."})

    return json.dumps({
        "status":  "ok",
        "count":   len(history),
        "versions": [
            {
                "version":    e["version"],
                "timestamp":  e["timestamp"],
                "professors": list(e["constraints"].get("teachers", {}).keys()),
                "days":       list(e["timetable"].keys()),
                "rooms":      e["constraints"].get("rooms", []),
            }
            for e in history
        ],
    })


@tool
def compare_timetables(version_a: int, version_b: int) -> str:
    """Compare two timetable versions cell by cell and show what changed.
    Use version number 0 to refer to the current (latest) timetable.
    Other version numbers come from get_timetable_history."""
    sess    = _s()
    history = sess.get("timetable_history", [])
    current = sess.get("timetable", {})

    def _get(v: int):
        if v == 0:
            return current
        for e in history:
            if e["version"] == v:
                return e["timetable"]
        return None

    t1, t2 = _get(version_a), _get(version_b)
    if t1 is None or t2 is None:
        return json.dumps({"status": "error", "message": f"Version {version_a} or {version_b} not found."})

    changes = []
    for day in sorted(set(list(t1) + list(t2))):
        for slot in sorted(set(list(t1.get(day, {})) + list(t2.get(day, {})))):
            r1 = t1.get(day, {}).get(slot, {})
            r2 = t2.get(day, {}).get(slot, {})
            for room in sorted(set(list(r1) + list(r2))):
                s1 = r1.get(room, {}).get("subject", "—") if isinstance(r1.get(room), dict) else "—"
                s2 = r2.get(room, {}).get("subject", "—") if isinstance(r2.get(room), dict) else "—"
                if s1 != s2:
                    changes.append(f"{day}  |  {slot}  |  Room {room}:  {s1} → {s2}")

    return json.dumps({
        "status":  "ok",
        "changes": len(changes),
        "diff":    changes[:50],
        "message": f"{len(changes)} change(s) between v{version_a} and v{version_b}.",
    })


@tool
def show_timetable() -> str:
    """Display the current timetable in the UI.
    Call whenever the user asks to see, show, view, or display the timetable."""
    sess = _s()
    tt = sess.get("timetable", {})
    if not tt:
        return json.dumps({"status": "empty", "message": "No timetable has been generated yet. Please provide scheduling details first."})
    sess["tt_updated"] = True
    return json.dumps({"status": "ok", "show_timetable": True, "message": "Timetable is now displayed in the UI."})


# ── LangGraph Agent ──────────────────────────────────────────────────────────────
TOOLS = [
    extract_constraints,
    solve_timetable,
    edit_timetable,
    validate_timetable,
    show_timetable,
    get_timetable_history,
    compare_timetables,
]

_SYSTEM = SystemMessage(content="""\
You are SLOT AI — a general-purpose scheduling and timetable assistant with a warm, natural personality.
You can schedule ANYTHING: school classes, university lectures, gym workouts, work shifts, meetings, sports sessions, personal routines — any domain where things need to be assigned to time slots.
You have access to an OR-Tools CP-SAT constraint solver via tools. You are a conversational agent first.

PERSONALITY:
- Respond naturally to any message. Greetings, small talk, questions — handle them like a smart, friendly colleague.
- Never repeat the same phrasing twice. Vary your tone and wording naturally every time.
- Never sound robotic or scripted.
- Only call tools when the user is clearly asking you to schedule something. For everything else, just converse.

DOMAIN MAPPING — the solver uses generic field names. Map ANY domain onto them:
  "teachers"   → whoever is doing the activity: trainer, instructor, employee, machine, person
  "subjects"   → what is being scheduled: workout, task, class, meeting, exercise, shift
  "rooms"      → where it happens: gym, room, court, zone, equipment, location
  "slots"      → time blocks: "09:00-10:00", "Morning", "Round 1", anything
  "days"       → days or sessions: Monday, Day 1, Week 1, etc.
  "professor_unavailability" → when an assignee is unavailable
  "max_slots_per_day"        → max sessions per day for an assignee
  "room_restrictions"        → activity forbidden from a location
  "slot_restrictions"        → activity restricted to certain time blocks

TOOLS AVAILABLE:
  extract_constraints  — parse and store all scheduling info from the user's message
  solve_timetable      — run the solver to generate the schedule
  edit_timetable       — update specific constraints and regenerate
  show_timetable       — display the current schedule in the UI
  validate_timetable   — check for violations
  get_timetable_history / compare_timetables — version history

HOW TO CALL extract_constraints — YOU read the user's message and fill every arg:
- teachers: {"assignee_name": ["activity1", "activity2"]}
- rooms: ["location1", "location2"]
- slots: ["time_block1", "time_block2"]  — exclude breaks/rest periods
- days: ["Day1", "Day2"]
- room_restrictions, slot_restrictions, professor_unavailability, max_slots_per_day, custom_rules as applicable

HOW TO CALL edit_timetable:
Pass ONLY what changed. Use clear_fixed_assignments=True for swaps.

SCHEDULING WORKFLOW:
1. Full info given → extract_constraints → solve_timetable
2. Partial info → extract_constraints with what's available, ask for what's missing
3. Change requested → edit_timetable with only the delta
4. INFEASIBLE → explain which constraints conflict and suggest how to resolve
5. User wants to view the schedule → call show_timetable

TOOL CALL DISCIPLINE — CRITICAL:
- Never call solve_timetable unless extract_constraints has already been called successfully earlier in THIS conversation, or the user's current message contains all the scheduling data needed.
- If the user says something vague like "generate a timetable" or "make a schedule" WITHOUT providing details in their current message, respond naturally and ask what they want to schedule — do NOT silently reuse old constraints.
- Never call any scheduling tool based solely on context carried over from old messages when the user's current intent is ambiguous. When in doubt, ask.\
""")

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def _build_graph(memory: MemorySaver) -> Any:
    def agent_node(state: AgentState):
        llm      = _get_llm().bind_tools(TOOLS)
        response = llm.invoke([_SYSTEM] + state["messages"])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=memory)

# ── Exports ──────────────────────────────────────────────────────────────────────
def _export_pdf():
    try:
        from fpdf import FPDF
        _enc = lambda t: str(t).encode("latin-1", "replace").decode("latin-1")
        pdf = FPDF()
        for day, df in _to_dfs(st.session_state.timetable).items():
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, _enc(day))
            pdf.ln()
            pdf.set_font("Helvetica", size=7)
            cols = ["Time"] + list(df.columns)
            w = 190 / len(cols)
            pdf.set_fill_color(220, 220, 220)
            for col in cols:
                pdf.cell(w, 7, _enc(col), border=1, fill=True)
            pdf.ln()
            for idx, row in df.iterrows():
                pdf.cell(w, 7, _enc(idx), border=1)
                for v in row:
                    pdf.cell(w, 7, _enc(v)[:20], border=1)
                pdf.ln()
        return bytes(pdf.output())
    except ImportError:
        return None


def _export_excel():
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for day, df in _to_dfs(st.session_state.timetable).items():
            df.to_excel(w, sheet_name=day[:31])
    return buf.getvalue()

# ── Context-menu action handler (right-click → query params → rerun) ─────────────
_ctx        = st.query_params.get("ctx_action", "")
_ctx_tid    = st.query_params.get("ctx_tid",    "")
_ctx_title  = st.query_params.get("ctx_title",  "")
if _ctx:
    st.query_params.clear()
    if _ctx == "switch" and _ctx_tid:
        if _ctx_tid != st.session_state.active_thread_id:
            _save_current_chat()
            _load_chat(_ctx_tid)
    elif _ctx == "rename" and _ctx_tid:
        st.session_state.renaming_chat_tid = _ctx_tid
    elif _ctx == "save_rename" and _ctx_tid:
        for _s_ in st.session_state.chat_sessions:
            if _s_["thread_id"] == _ctx_tid:
                _s_["title"] = _ctx_title.strip() or _s_["title"]
                break
        st.session_state.renaming_chat_tid = None
    elif _ctx == "cancel_rename":
        st.session_state.renaming_chat_tid = None
    elif _ctx == "delete" and _ctx_tid:
        _delete_chat(_ctx_tid)
    st.rerun()

# ── Sidebar ───────────────────────────────────────────────────────────────────────
st.title("🤖 SLOT AI")

# Inject CSS: remove red/orange from buttons; keep everything dark-theme neutral
st.markdown("""
<style>
/* ── All sidebar buttons: dark-theme neutral ── */
section[data-testid="stSidebar"] button {
    background-color: rgba(255,255,255,0.06) !important;
    color: rgba(255,255,255,0.88) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 6px !important;
}
section[data-testid="stSidebar"] button:hover {
    background-color: rgba(255,255,255,0.13) !important;
    border-color: rgba(255,255,255,0.28) !important;
}
section[data-testid="stSidebar"] button[kind="primary"] {
    background-color: rgba(255,255,255,0.16) !important;
    border-color: rgba(255,255,255,0.35) !important;
    font-weight: 600 !important;
}

/* ── Password field: flush eye icon to right edge ── */
section[data-testid="stSidebar"] [data-testid="stTextInput"] [data-baseweb="base-input"] {
    padding-right: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stTextInput"] [data-baseweb="base-input"] > div {
    padding-right: 0 !important;
}

</style>
""", unsafe_allow_html=True)

with st.sidebar:
    # ── Credentials ──
    st.header("🔑 Credentials")
    key_in = st.text_input("Groq API Key", type="password",
                            value=st.session_state.api_key, placeholder="gsk_...",
                            help="Get a free key at console.groq.com")
    st.session_state.api_key = key_in

    st.divider()

    # ── New Chat ──
    if st.button("New Chat +", use_container_width=True, type="secondary"):
        _new_chat()
        st.rerun()

    st.caption("Chats")

    # ── Chat list (custom HTML — right-click for Rename / Delete) ──
    _active_tid    = st.session_state.active_thread_id
    _renaming_tid  = st.session_state.renaming_chat_tid or ""
    _chat_items    = ""
    for _sess in st.session_state.chat_sessions:
        _tid  = _sess["thread_id"]
        _safe = (_sess["title"]
                 .replace("&", "&amp;").replace("<", "&lt;")
                 .replace(">", "&gt;").replace('"', "&quot;"))
        if _renaming_tid == _tid:
            _chat_items += (
                f'<div class="sai-rename-row">'
                f'<input id="sai-ri-{_tid}" class="sai-rename-input" type="text" value="{_safe}">'
                f'<div class="sai-rename-actions">'
                f'<button onclick="saiSaveRename(\'{_tid}\')">Save</button>'
                f'<button onclick="saiNav({{ctx_action:\'cancel_rename\'}})">Cancel</button>'
                f'</div></div>'
            )
        else:
            _ac  = " sai-active" if _tid == _active_tid else ""
            _arr = "› " if _tid == _active_tid else ""
            _chat_items += (
                f'<button class="sai-chat-btn{_ac}" data-tid="{_tid}" '
                f'onclick="saiSwitch(\'{_tid}\')">{_arr}{_safe}</button>'
            )
    st.markdown(f"""
<div id="sai-chat-list">{_chat_items}</div>
<div id="sai-ctx-menu">
  <button id="sai-ctx-rename-btn">Rename</button>
  <button id="sai-ctx-delete-btn">Delete</button>
</div>
<script>
(function(){{
  window.saiNav = function(p) {{
    var u = new URL(window.location.href);
    Object.keys(p).forEach(function(k){{ u.searchParams.set(k, p[k]); }});
    window.location.href = u.toString();
  }};
  window.saiSwitch = function(tid) {{ saiNav({{ctx_action:'switch',ctx_tid:tid}}); }};
  window.saiSaveRename = function(tid) {{
    var el = document.getElementById('sai-ri-'+tid);
    saiNav({{ctx_action:'save_rename',ctx_tid:tid,ctx_title:el?el.value:''}});
  }};

  var _ctxTid = null;

  if (window._saiCtxH) document.removeEventListener('contextmenu', window._saiCtxH);
  window._saiCtxH = function(e) {{
    var btn = e.target && e.target.closest ? e.target.closest('.sai-chat-btn') : null;
    var menu = document.getElementById('sai-ctx-menu');
    if (!menu) return;
    if (!btn) {{ menu.style.display='none'; return; }}
    e.preventDefault();
    _ctxTid = btn.getAttribute('data-tid');
    menu.style.left = e.clientX+'px';
    menu.style.top  = e.clientY+'px';
    menu.style.display = 'block';
  }};
  document.addEventListener('contextmenu', window._saiCtxH);

  if (window._saiClickH) document.removeEventListener('click', window._saiClickH);
  window._saiClickH = function(e) {{
    var menu = document.getElementById('sai-ctx-menu');
    if (!menu) return;
    if (e.target && e.target.id === 'sai-ctx-rename-btn') {{
      e.stopPropagation(); menu.style.display='none';
      if (_ctxTid) saiNav({{ctx_action:'rename',ctx_tid:_ctxTid}});
      return;
    }}
    if (e.target && e.target.id === 'sai-ctx-delete-btn') {{
      e.stopPropagation(); menu.style.display='none';
      if (_ctxTid) saiNav({{ctx_action:'delete',ctx_tid:_ctxTid}});
      return;
    }}
    menu.style.display='none';
  }};
  document.addEventListener('click', window._saiClickH);
}})();
</script>
<style>
#sai-chat-list {{ margin-bottom:2px; }}
.sai-chat-btn {{
  display:block !important; width:100% !important; text-align:left !important;
  padding:7px 12px !important; margin-bottom:3px !important;
  background:rgba(255,255,255,0.06) !important; color:rgba(255,255,255,0.88) !important;
  border:1px solid rgba(255,255,255,0.12) !important; border-radius:6px !important;
  font-size:14px !important; cursor:pointer !important;
  white-space:nowrap !important; overflow:hidden !important; text-overflow:ellipsis !important;
  font-family:inherit !important; box-sizing:border-box !important;
}}
.sai-chat-btn:hover {{
  background:rgba(255,255,255,0.13) !important;
  border-color:rgba(255,255,255,0.28) !important;
}}
.sai-active {{
  background:rgba(255,255,255,0.16) !important;
  border-color:rgba(255,255,255,0.35) !important; font-weight:600 !important;
}}
.sai-rename-row {{ margin-bottom:3px; }}
.sai-rename-input {{
  width:100%; padding:6px 10px; box-sizing:border-box; margin-bottom:4px;
  background:rgba(255,255,255,0.08); color:rgba(255,255,255,0.9);
  border:1px solid rgba(255,255,255,0.25); border-radius:6px;
  font-size:14px; font-family:inherit; outline:none;
}}
.sai-rename-actions {{ display:flex; gap:4px; }}
.sai-rename-actions button {{
  flex:1; padding:4px 0; font-size:12px; font-family:inherit;
  border:1px solid rgba(255,255,255,0.2) !important; border-radius:4px !important;
  background:rgba(255,255,255,0.07) !important; color:rgba(255,255,255,0.8) !important;
  cursor:pointer !important;
}}
.sai-rename-actions button:hover {{ background:rgba(255,255,255,0.15) !important; }}
#sai-ctx-menu {{
  display:none; position:fixed; z-index:99999;
  background:rgb(22,22,26); border:1px solid rgba(255,255,255,0.18);
  border-radius:6px; padding:3px; min-width:110px;
  box-shadow:0 4px 16px rgba(0,0,0,0.7);
}}
#sai-ctx-menu button {{
  display:block !important; width:100% !important; text-align:left !important;
  padding:5px 10px !important; background:transparent !important;
  border:none !important; color:rgba(255,255,255,0.82) !important;
  font-size:12px !important; cursor:pointer !important;
  border-radius:4px !important; font-family:inherit;
}}
#sai-ctx-menu button:hover {{ background:rgba(255,255,255,0.1) !important; }}
</style>
""", unsafe_allow_html=True)

    st.divider()

    # ── View Timetable ──
    has_tt = bool(st.session_state.timetable)
    if st.button("📊 View Timetable", use_container_width=True,
                  type="primary" if st.session_state.page == "timetable" else "secondary",
                  disabled=not has_tt):
        st.session_state.page = "timetable"
        st.rerun()
    if not has_tt:
        st.caption("_Generate a timetable first._")
    elif st.session_state.get("timetable_history"):
        st.caption(f"📚 {len(st.session_state.timetable_history)} version(s) saved")

# ── Page: Timetable ──────────────────────────────────────────────────────────────
if st.session_state.page == "timetable":
    col_back, col_title = st.columns([1, 6])
    if col_back.button("← Back"):
        st.session_state.page = "chat"
        st.rerun()
    col_title.subheader("📊 Timetable")

    if st.session_state.timetable:
        pdf_data  = _export_pdf()
        xlsx_data = _export_excel()
        dl1, dl2, _ = st.columns([1, 1, 4])
        if pdf_data:
            dl1.download_button("⬇️ All (PDF)", pdf_data, "timetable.pdf",
                                "application/pdf", use_container_width=True)
        dl2.download_button("⬇️ All (Excel)", xlsx_data, "timetable.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True)
        st.divider()
        _show_tt(st.session_state.timetable, key_prefix="view")
    else:
        st.info("No timetable generated yet. Go to Chat and describe your schedule.")
    st.stop()

# ── Page: Chat ───────────────────────────────────────────────────────────────────
if st.session_state.timetable:
    st.success("Timetable ready — click **📊 View Timetable** in the sidebar to see it.")

for msg in st.session_state.get("display_messages", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.api_key:
    st.info("👈 Enter your Groq API key in the sidebar to get started.")
    st.stop()

if prompt := st.chat_input("What would you like to schedule today?"):
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    os.environ["GROQ_API_KEY"] = st.session_state.api_key

    tid = st.session_state.thread_id
    _tl.session_id = tid
    _STORE[tid] = {
        "api_key":           st.session_state.api_key,
        "model":             MODEL,
        "constraints":       copy.deepcopy(st.session_state.constraints),
        "timetable":         copy.deepcopy(st.session_state.timetable),
        "timetable_history": list(st.session_state.timetable_history),
        "tt_updated":        False,
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                config = {"configurable": {"thread_id": tid}}
                result = _get_agent(tid).invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config,
                )
                reply = result["messages"][-1].content
            except Exception as e:
                err = str(e)
                result = None
                if "429" in err or "rate_limit" in err.lower():
                    reply = "⚠️ Rate limit hit. Please wait ~1 minute and try again."
                elif "tool_use_failed" in err or "failed_generation" in err:
                    reply = "⚠️ The model failed to call a tool correctly. Try rephrasing your request more simply."
                else:
                    reply = f"⚠️ Error ({type(e).__name__}): {err[:300]}"

        # Parse tool outputs from the LangGraph message chain — runs on the main
        # thread so it always works regardless of how tools were dispatched.
        if result:
            for m in result.get("messages", []):
                raw = getattr(m, "content", "")
                if not isinstance(raw, str):
                    continue
                try:
                    data = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    continue
                if "constraints" in data and isinstance(data["constraints"], dict):
                    st.session_state.constraints = data["constraints"]
                if (data.get("status") == "success"
                        and "timetable" in data
                        and isinstance(data["timetable"], dict)
                        and data["timetable"]):
                    st.session_state.timetable  = data["timetable"]
                    st.session_state.tt_updated = True
                if data.get("show_timetable") and st.session_state.timetable:
                    st.session_state.tt_updated = True

        # Also sync history from _STORE (written inside tools, not returned in messages)
        sess = _STORE.get(tid, _STORE.get("_default", {}))
        if sess.get("timetable_history"):
            st.session_state.timetable_history = sess["timetable_history"]

        st.markdown(reply)

    st.session_state.display_messages.append({"role": "assistant", "content": reply})
    st.session_state.tt_updated = False
    _save_current_chat()   # updates title in sidebar + persists state
    st.rerun()             # refresh sidebar (title, View Timetable button state)
