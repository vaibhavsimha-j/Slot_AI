"""SLOT AI — General Purpose Scheduling Agent
pip install streamlit langchain-google-genai langgraph langchain-core ortools pydantic pandas openpyxl fpdf2
streamlit run app.py
"""
import os, json, copy, uuid, threading
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, TypedDict
import streamlit as st
import pandas as pd
from io import BytesIO

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from ortools.sat.python import cp_model

# ── Config ──────────────────────────────────────────────────────────────────────
MODEL = "gemini-flash-latest"
st.set_page_config(page_title="SLOT AI", layout="wide")

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
if "thread_memories" not in st.session_state:
    st.session_state.thread_memories = {}
if "thread_agents" not in st.session_state:
    st.session_state.thread_agents = {}

# Bootstrap the first chat session on a fresh load
if not st.session_state.chat_sessions:
    _init_tid = st.session_state.thread_id
    st.session_state.chat_sessions = [{
        "thread_id": _init_tid, "title": "New Chat", "custom_title": False,
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
            if not s.get("custom_title"):
                s["title"] = _chat_title(s["display_messages"])
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
        "thread_id": new_tid, "title": "New Chat", "custom_title": False,
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

# ── Session store ────────────────────────────────────────────────────────────────
_tl    = threading.local()
_STORE: dict[str, dict] = {}

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
def _get_llm(t: float = 0.0) -> ChatGoogleGenerativeAI:
    # api_key is injected into os.environ["GOOGLE_API_KEY"] before every agent invocation
    return ChatGoogleGenerativeAI(
        model=MODEL,
        temperature=t,
    )

def _cell(e):
    if isinstance(e, dict):
        t = e.get("task") or e.get("subject", "")
        a = e.get("assignee") or e.get("professor", "")
        return f"{t} ({a})" if t not in ("FREE", "", None) else "—"
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
    def _has(new_k, old_k=None):
        return bool(c.get(new_k) or (old_k and c.get(old_k)))
    out = []
    if not _has("assignees", "teachers"):  out.append("assignees (who/what is being scheduled)")
    if not _has("time_slots", "slots"):    out.append("time slots")
    if not _has("locations", "rooms"):     out.append("locations")
    if not _has("periods", "days"):        out.append("periods / days")
    return out

# ── OR-Tools CP-SAT Solver ──────────────────────────────────────────────────────
def _ortools_solve(constraints: dict) -> dict:
    assignees  = constraints.get("assignees")  or constraints.get("teachers", {})
    locations  = constraints.get("locations")  or constraints.get("rooms", [])
    time_slots = constraints.get("time_slots") or constraints.get("slots", [])
    periods    = constraints.get("periods")    or constraints.get("days", [])

    task_to_assignee = {t: a for a, ts in assignees.items() for t in ts}
    tasks = list(task_to_assignee.keys())
    n_p, n_ts, n_l, n_t = len(periods), len(time_slots), len(locations), len(tasks)

    if not all([n_p, n_ts, n_l, n_t]):
        return {"status": "error", "reason": "Missing core fields: assignees, locations, time_slots, periods."}

    model = cp_model.CpModel()
    x = {
        (p, ts, l, t): model.NewBoolVar(f"x{p}{ts}{l}{t}")
        for p in range(n_p) for ts in range(n_ts)
        for l in range(n_l) for t in range(n_t)
    }

    for p in range(n_p):
        for ts in range(n_ts):
            for l in range(n_l):
                model.AddExactlyOne(x[p, ts, l, t] for t in range(n_t))

    for p in range(n_p):
        for ts in range(n_ts):
            for asgn, asgn_tasks in assignees.items():
                ids = [tasks.index(t) for t in asgn_tasks if t in tasks]
                if ids:
                    model.Add(
                        sum(x[p, ts, l, tid] for l in range(n_l) for tid in ids) <= 1
                    )

    unavail = {**constraints.get("professor_unavailability", {}),
               **constraints.get("assignee_unavailability", {})}
    for asgn, unavail_periods in unavail.items():
        ids = [tasks.index(t) for t in assignees.get(asgn, []) if t in tasks]
        for p, period in enumerate(periods):
            if period in unavail_periods:
                for ts in range(n_ts):
                    for l in range(n_l):
                        for tid in ids:
                            model.Add(x[p, ts, l, tid] == 0)

    loc_restr = {**constraints.get("room_restrictions", {}),
                 **constraints.get("location_restrictions", {})}
    for task, forbidden in loc_restr.items():
        if task in tasks:
            tid = tasks.index(task)
            for l, loc in enumerate(locations):
                if loc in forbidden:
                    for p in range(n_p):
                        for ts in range(n_ts):
                            model.Add(x[p, ts, l, tid] == 0)

    ts_restr = {**constraints.get("slot_restrictions", {}),
                **constraints.get("time_slot_restrictions", {})}
    for task, allowed in ts_restr.items():
        if task in tasks:
            tid = tasks.index(task)
            for ts, slot in enumerate(time_slots):
                if slot not in allowed:
                    for p in range(n_p):
                        for l in range(n_l):
                            model.Add(x[p, ts, l, tid] == 0)

    max_pp = {**constraints.get("max_slots_per_day", {}),
              **constraints.get("max_per_period", {})}
    for asgn, max_v in max_pp.items():
        ids = [tasks.index(t) for t in assignees.get(asgn, []) if t in tasks]
        for p in range(n_p):
            model.Add(
                sum(x[p, ts, l, tid]
                    for ts in range(n_ts) for l in range(n_l) for tid in ids) <= max_v
            )

    for t in range(n_t):
        model.Add(
            sum(x[p, ts, l, t]
                for p in range(n_p) for ts in range(n_ts) for l in range(n_l)) >= 1
        )

    for fa in constraints.get("fixed_assignments", []):
        try:
            p   = periods.index(fa.get("period") or fa.get("day"))
            ts  = time_slots.index(fa.get("time_slot") or fa.get("slot"))
            l   = locations.index(fa.get("location") or fa.get("room"))
            tid = tasks.index(fa.get("task") or fa.get("subject"))
            model.Add(x[p, ts, l, tid] == 1)
        except (ValueError, KeyError):
            pass

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    solver.parameters.num_search_workers = 4
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        tt = {}
        for p, period in enumerate(periods):
            tt[period] = {}
            for ts, slot in enumerate(time_slots):
                tt[period][slot] = {}
                for l, loc in enumerate(locations):
                    for tid, task in enumerate(tasks):
                        if solver.Value(x[p, ts, l, tid]) == 1:
                            tt[period][slot][loc] = {
                                "task":     task,
                                "assignee": task_to_assignee[task],
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
                "No valid schedule exists with the current constraints. "
                "Check for conflicting rules (e.g. an assignee unavailable on all periods, "
                "or a task restricted from all locations)."
            ),
        }
    return {"status": "timeout", "reason": "Solver timed out. Try fewer constraints."}

# ── Constraint Merge ─────────────────────────────────────────────────────────────
def _merge(current: dict, parsed: dict) -> dict:
    c = copy.deepcopy(current)
    _DICT_MERGE = {
        "assignees", "location_restrictions", "assignee_unavailability", "time_slot_restrictions",
        "teachers", "room_restrictions", "professor_unavailability", "slot_restrictions",
    }
    _CORE_FIELDS = {"locations", "time_slots", "periods", "rooms", "slots", "days"}
    for k, v in parsed.items():
        if k in _DICT_MERGE:
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
        elif k in ("max_per_period", "max_slots_per_day") and isinstance(v, dict):
            c.setdefault(k, {}).update(v)
        elif k in ("custom_rules", "fixed_assignments") and isinstance(v, list):
            e = c.setdefault(k, [])
            for r in v:
                if r not in e:
                    e.append(r)
        elif v:
            if k in _CORE_FIELDS and c.get(k):
                pass
            else:
                c[k] = v
    return c

# ── Tools ────────────────────────────────────────────────────────────────────────
@tool
def extract_constraints(
    assignees: Dict[str, List[str]],
    locations: List[str],
    time_slots: List[str],
    periods: List[str],
    location_restrictions: Optional[Dict[str, List[str]]] = None,
    time_slot_restrictions: Optional[Dict[str, List[str]]] = None,
    assignee_unavailability: Optional[Dict[str, List[str]]] = None,
    max_per_period: Optional[Dict[str, int]] = None,
    custom_rules: Optional[List[str]] = None,
) -> str:
    """Store scheduling constraints. Extract these from the user's message and pass as typed args.
    Works for ANY domain — school, gym, shifts, meetings, sports, etc.

    assignees: who/what does each task. e.g.
      School  → {"Alice": ["Math", "Physics"], "Bob": ["History"]}
      Gym     → {"trainer": ["squat", "bench", "deadlift"]}
      Shifts  → {"Alice": ["morning_shift"], "Bob": ["evening_shift"]}
    locations: where it happens. e.g. ["Room A", "Room B"] / ["gym_floor"] / ["Ward A", "ICU"]
    time_slots: sub-period time blocks. e.g. ["09:00-10:00", "10:00-11:00"] / ["Morning", "Afternoon"]
    periods: top-level repeating units. e.g. ["Monday", "Tuesday"] / ["Day 1", "Day 2"]
    location_restrictions: task forbidden from certain locations. e.g. {"Math": ["Room C"]}
    time_slot_restrictions: task allowed only in certain time_slots. e.g. {"Math": ["09:00-10:00"]}
    assignee_unavailability: periods an assignee cannot work. e.g. {"Alice": ["Friday"]}
    max_per_period: max tasks an assignee can have in one period. e.g. {"Alice": 2}
    custom_rules: any other rules as plain-text strings.
    """
    sess    = _s()
    current = sess.get("constraints", {})
    parsed: dict = {
        "assignees": assignees, "locations": locations,
        "time_slots": time_slots, "periods": periods,
    }
    if location_restrictions:   parsed["location_restrictions"]   = location_restrictions
    if time_slot_restrictions:  parsed["time_slot_restrictions"]  = time_slot_restrictions
    if assignee_unavailability: parsed["assignee_unavailability"] = assignee_unavailability
    if max_per_period:          parsed["max_per_period"]          = max_per_period
    if custom_rules:            parsed["custom_rules"]            = custom_rules

    merged = _merge(current, parsed)
    sess["constraints"] = merged
    miss = _missing(merged)
    _asgn = merged.get("assignees", {})
    return json.dumps({
        "status":      "ready" if not miss else "incomplete",
        "missing":     miss,
        "constraints": merged,
        "summary": {
            "assignees":  list(_asgn.keys()),
            "tasks":      [t for ts in _asgn.values() for t in ts],
            "locations":  merged.get("locations", []),
            "time_slots": merged.get("time_slots", []),
            "periods":    merged.get("periods", []),
            "active_restrictions": (
                len(merged.get("location_restrictions", {}))
                + len(merged.get("time_slot_restrictions", {}))
                + len(merged.get("assignee_unavailability", {}))
                + len(merged.get("max_per_period", {}))
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
    location_restrictions: Optional[Dict[str, List[str]]] = None,
    time_slot_restrictions: Optional[Dict[str, List[str]]] = None,
    assignee_unavailability: Optional[Dict[str, List[str]]] = None,
    max_per_period: Optional[Dict[str, int]] = None,
    fixed_assignments: Optional[List[Dict[str, str]]] = None,
    custom_rules: Optional[List[str]] = None,
    clear_fixed_assignments: bool = False,
) -> str:
    """Update constraints and regenerate the schedule using OR-Tools.
    Pass ONLY the constraints that changed. The solver re-runs with merged constraints.

    location_restrictions: task forbidden from locations. e.g. {"Math": ["Room C"]}
    time_slot_restrictions: task limited to certain time_slots. e.g. {"Math": ["Morning"]}
    assignee_unavailability: assignee unavailable on certain periods. e.g. {"Alice": ["Friday"]}
    max_per_period: max tasks an assignee can have per period. e.g. {"Alice": 2}
    fixed_assignments: pin a specific task to a cell.
      e.g. [{"period":"Monday","time_slot":"09:00-10:00","location":"Room A","task":"Math"}]
    custom_rules: additional rule text.
    clear_fixed_assignments: set True when replacing all pinned cells (e.g. a swap).
    """
    sess = _s()
    if not sess.get("timetable"):
        return json.dumps({"status": "error", "message": "No schedule exists yet. Generate one first."})

    current = sess.get("constraints", {})
    if clear_fixed_assignments:
        current.pop("fixed_assignments", None)

    delta: dict = {}
    if location_restrictions:   delta["location_restrictions"]   = location_restrictions
    if time_slot_restrictions:  delta["time_slot_restrictions"]  = time_slot_restrictions
    if assignee_unavailability: delta["assignee_unavailability"] = assignee_unavailability
    if max_per_period:          delta["max_per_period"]          = max_per_period
    if fixed_assignments:       delta["fixed_assignments"]       = fixed_assignments
    if custom_rules:            delta["custom_rules"]            = custom_rules

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
    """Validate the current schedule against all active constraints.
    Returns a detailed list of any violations found (conflicts, banned locations, overloads, etc.)."""
    sess = _s()
    tt = sess.get("timetable", {})
    c  = sess.get("constraints", {})
    if not tt:
        return json.dumps({"status": "error", "message": "No schedule to validate."})

    loc_restr  = {**c.get("room_restrictions", {}),     **c.get("location_restrictions", {})}
    ts_restr   = {**c.get("slot_restrictions", {}),     **c.get("time_slot_restrictions", {})}
    unavail    = {**c.get("professor_unavailability", {}), **c.get("assignee_unavailability", {})}
    max_pp     = {**c.get("max_slots_per_day", {}),     **c.get("max_per_period", {})}

    issues = []

    for period, time_slots in tt.items():
        for ts, locs in time_slots.items():
            seen_assignees: dict = {}
            for loc, cell in locs.items():
                if not isinstance(cell, dict):
                    continue
                task  = cell.get("task") or cell.get("subject", "")
                asgn  = cell.get("assignee") or cell.get("professor", "")
                if not task:
                    continue

                if asgn and asgn in seen_assignees:
                    issues.append(
                        f"DOUBLE-BOOKING: {asgn} in {seen_assignees[asgn]} and {loc} "
                        f"— {period} {ts}"
                    )
                if asgn:
                    seen_assignees[asgn] = loc

                if loc in loc_restr.get(task, []):
                    issues.append(f"LOCATION VIOLATION: {task} in restricted location {loc} — {period} {ts}")

                allowed_ts = ts_restr.get(task)
                if allowed_ts and ts not in allowed_ts:
                    issues.append(f"TIME-SLOT VIOLATION: {task} in non-allowed slot {ts} — {period}")

        for asgn, ud in unavail.items():
            if period in ud:
                for ts, locs in time_slots.items():
                    for cell in locs.values():
                        a = cell.get("assignee") or cell.get("professor", "") if isinstance(cell, dict) else ""
                        if a == asgn:
                            issues.append(f"UNAVAILABILITY: {asgn} scheduled on {period} at {ts}")

    for asgn, max_v in max_pp.items():
        for period, time_slots in tt.items():
            count = sum(
                1 for ts_data in time_slots.values()
                for cell in ts_data.values()
                if isinstance(cell, dict)
                and (cell.get("assignee") or cell.get("professor", "")) == asgn
            )
            if count > max_v:
                issues.append(
                    f"OVERLOAD: {asgn} has {count} tasks on {period} (max allowed: {max_v})"
                )

    return json.dumps({
        "status":      "valid" if not issues else "violations_found",
        "issue_count": len(issues),
        "issues":      issues[:20],
        "message":     "Schedule is valid!" if not issues else f"{len(issues)} violation(s) found.",
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
You are SLOT AI — a smart, friendly scheduling assistant. You operate in two modes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE ONLY RULE THAT DECIDES YOUR MODE:

  Ask yourself: "Did the user already provide all the details,
                 or do I need to come up with them myself?"

  USER PROVIDED ALL DETAILS  →  Solver mode  (use tools)
  AGENT MUST GENERATE DETAILS  →  Generate directly  (no tools)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODE 1 — GENERATE DIRECTLY (no tools)
The user has NOT given all the details. You need to come up with them.
→ Just write the answer. Be specific, practical, and helpful.

Examples:
  "Plan a 3-day trip to Paris for me"         — you decide the places, timings, activities
  "Give me a gym schedule for Mon/Wed/Fri"    — you decide the exercises, sets, reps
  "Help me organize my study week for exams"  — you decide the subjects and time blocks
  "Schedule my client meetings this week"     — you decide the slots and order
  "Suggest a morning routine"                 — you decide everything

MODE 2 — SOLVER MODE (use tools, NEVER write the table yourself)
The user HAS provided all the details — who, what, where, when, and the rules.
Your job is only to assign them correctly, not to invent anything.
IMPORTANT: Never write a schedule or timetable table yourself in this mode.
Your text generation cannot verify hard constraints. Only the solver can.
Always call extract_constraints → solve_timetable and let the solver produce the result.

Examples:
  "Here are 6 professors and their subjects, 4 rooms, 6 time slots, Mon–Fri.
   AI/ML only in morning. CN never in Room D. No professor in two rooms at once.
   Generate the timetable."                   — ALL details given → solver mode

  "Schedule these 15 nurses across 3 wards, morning/evening/night shifts,
   max 5 shifts/week each, Alice unavailable Fridays."  — ALL details given → solver mode

  "I have 10 teams, 3 courts, matches from 9am-5pm Sat & Sun,
   each team plays every other team once."   — ALL details given → solver mode

━━ TOOLS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  extract_constraints  → store all user-provided scheduling data
  solve_timetable      → run OR-Tools CP-SAT solver to find valid assignment
  edit_timetable       → update constraints and re-solve (pass only what changed)
  show_timetable       → display the schedule in the UI
  validate_timetable   → check for constraint violations
  get_timetable_history / compare_timetables → version history

TOOL PARAMETER MAPPING (domain-agnostic):
  assignees  → {"Dr. Vaibhav": ["AI","ML"]} / {"Alice": ["morning_shift"]}
  locations  → ["Room A","Room B"] / ["Ward A","ICU"] / ["Court 1"]
  time_slots → ["09:00-09:50","09:50-10:40"] / ["Morning","Afternoon"]
               Exclude breaks — only schedulable slots.
  periods    → ["Monday","Tuesday","Wednesday"] / ["Day 1","Day 2"]

SOLVER WORKFLOW:
1. All details given → extract_constraints → solve_timetable
2. Partial details → extract what exists, ask for the rest before solving
3. Edit/change → edit_timetable (delta only)
4. View → show_timetable
5. Infeasible → explain which rules conflict, suggest how to fix

DISCIPLINE:
- Never call solve_timetable without extract_constraints first in this conversation
- Never write a timetable/grid yourself when the user has given all the details — use the solver
- If truly ambiguous, ask one clarifying question: "Have you decided all the details, or should I suggest them?"

PERSONALITY: Warm, natural, never robotic. Vary phrasing every response.\
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

# ── Sidebar ───────────────────────────────────────────────────────────────────────
st.title("🤖 SLOT AI")

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
section[data-testid="stSidebar"] [data-testid="stTextInput"] [data-baseweb="base-input"] {
    padding-right: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stTextInput"] [data-baseweb="base-input"] > div {
    padding-right: 0 !important;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
    gap: 2px !important;
    align-items: center !important;
}
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]
    div[data-testid="stColumn"]:last-child button {
    padding: 2px 4px !important;
    min-height: 0 !important;
    font-size: 16px !important;
    line-height: 1.2 !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    # ── Credentials ──
    st.header("🔑 Credentials")
    key_in = st.text_input("Google AI API Key", type="password",
                            value=st.session_state.api_key, placeholder="AIza...",
                            help="Get a free key at aistudio.google.com")
    st.session_state.api_key = key_in

    st.divider()

    # ── New Chat ──
    if st.button("New Chat +", use_container_width=True, type="secondary"):
        _new_chat()
        st.rerun()

    st.caption("Chats")

    # ── Chat list ──
    for _sess in st.session_state.chat_sessions:
        _tid = _sess["thread_id"]
        _is_active = _tid == st.session_state.active_thread_id
        _label = ("› " if _is_active else "") + _sess["title"]

        _col_chat, _col_menu = st.columns([8, 1], gap="small")

        if _col_chat.button(_label, key=f"_sw_{_tid}", use_container_width=True,
                            type="primary" if _is_active else "secondary"):
            _save_current_chat()
            _load_chat(_tid)
            st.rerun()

        with _col_menu:
            with st.popover("⋮", use_container_width=True):
                st.caption(f"**{_sess['title']}**")
                _new_name = st.text_input("Chat name", value=_sess["title"],
                                          key=f"_pn_{_tid}")
                if st.button("💾 Save name", key=f"_ps_{_tid}", use_container_width=True):
                    _sess["title"] = _new_name.strip() or _sess["title"]
                    _sess["custom_title"] = True
                    st.rerun()
                st.divider()
                if st.button("🗑️ Delete chat", key=f"_pd_{_tid}",
                             use_container_width=True):
                    _delete_chat(_tid)
                    st.rerun()

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
    st.info("👈 Enter your Google AI API key in the sidebar to get started.")
    st.stop()

if prompt := st.chat_input("What would you like to schedule today?"):
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    os.environ["GOOGLE_API_KEY"] = st.session_state.api_key

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
                last_msg = result["messages"][-1]
                content = last_msg.content
                if isinstance(content, list):
                    reply = " ".join(
                        block.get("text", "") for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                else:
                    reply = content
            except Exception as e:
                err = str(e)
                result = None
                if "429" in err or "rate_limit" in err.lower():
                    reply = "⚠️ Rate limit hit. Please wait ~1 minute and try again."
                elif "tool_use_failed" in err or "failed_generation" in err:
                    reply = "⚠️ The model failed to call a tool correctly. Try rephrasing your request more simply."
                else:
                    reply = f"⚠️ Error ({type(e).__name__}): {err[:300]}"

        if result:
            for m in result.get("messages", []):
                raw = getattr(m, "content", "")
                if isinstance(raw, list):
                    raw = " ".join(
                        block.get("text", "") for block in raw
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
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

        sess = _STORE.get(tid, _STORE.get("_default", {}))
        if sess.get("timetable_history"):
            st.session_state.timetable_history = sess["timetable_history"]

        st.markdown(reply)

    st.session_state.display_messages.append({"role": "assistant", "content": reply})
    st.session_state.tt_updated = False
    _save_current_chat()
    st.rerun()