"""SLOT AI — Production Timetable Scheduling Agent
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
MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
st.set_page_config(page_title="SLOT AI", page_icon="📅", layout="wide")

# ── Session Init ────────────────────────────────────────────────────────────────
_DEFAULTS = dict(
    api_key="", model=MODELS[0],
    constraints={}, timetable={}, timetable_history=[],
    display_messages=[], tt_updated=False,
    thread_id=str(uuid.uuid4())
)
for k, v in _DEFAULTS.items():
    st.session_state.setdefault(k, v)
if not st.session_state.api_key and os.getenv("GROQ_API_KEY"):
    st.session_state.api_key = os.getenv("GROQ_API_KEY", "")
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

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
        "api_key": "", "model": MODELS[0],
        "constraints": {}, "timetable": {}, "timetable_history": [], "tt_updated": False,
    })

# ── Utilities ───────────────────────────────────────────────────────────────────
def _get_llm(t: float = 0.0) -> ChatGroq:
    # api_key is injected into os.environ["GROQ_API_KEY"] before every agent invocation
    # so ChatGroq picks it up automatically — avoids asyncio context propagation issues
    return ChatGroq(
        model=_s().get("model", MODELS[0]),
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
        "status":  "ready" if not miss else "incomplete",
        "missing": miss,
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
    return json.dumps({"status": "ok", "message": "Timetable is now displayed in the UI."})


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
You are SLOT AI, a school timetable assistant powered by Google OR-Tools CP-SAT solver.

You are the orchestrator. You NEVER output timetable data yourself.
The solver generates the timetable; you call tools and relay results.

HOW TO CALL extract_constraints:
You must read the user's message and populate every argument yourself from the text.
- teachers: strip Dr./Prof. titles. Map each professor to their subjects.
  "AI-Vaibhav, ML-Vaibhav, DSA-Simha" -> {"Vaibhav": ["AI","ML"], "Simha": ["DSA"]}
  Parallel lists "Subjects: A,B Professors: X,Y" -> positional match -> {"X":["A"],"Y":["B"]}
- rooms: list of room names exactly as written.
- slots: list of time slot strings exactly as written (exclude break times).
- days: list of day names.
- room_restrictions: {"Subject": ["ForbiddenRoom"]}. "CN not in Room D" -> {"CN":["D"]}
- slot_restrictions: {"Subject": ["allowed","slots"]}. "AI/ML only morning 09:00-10:40" ->
  {"AI": ["09:00-09:50","09:50-10:40"], "ML": ["09:00-09:50","09:50-10:40"]}
- professor_unavailability: {"Prof": ["Day"]}
- max_slots_per_day: {"Prof": N}
- custom_rules: other rules as strings.

HOW TO CALL edit_timetable:
Pass only the constraints that changed. Set clear_fixed_assignments=True for swaps.

Workflow:
1. User gives full info -> call extract_constraints (populate all args), then solve_timetable
2. Info incomplete -> call extract_constraints with what you have, tell user what is missing
3. User wants a change -> call edit_timetable with only the changed constraints
4. After successful solve: confirm status (OPTIMAL/FEASIBLE). UI shows the timetable automatically.
5. INFEASIBLE: diagnose which constraints conflict.
6. User asks to see/show/display/view the timetable -> call show_timetable immediately.\
""")

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def _build_graph() -> Any:
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
    return graph.compile(checkpointer=st.session_state.memory)


if "agent" not in st.session_state:
    st.session_state.agent = _build_graph()

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
st.title("📅 SLOT AI")

with st.sidebar:
    st.header("⚙️ Setup")
    key_in   = st.text_input("Groq API Key", type="password",
                              value=st.session_state.api_key, placeholder="gsk_...",
                              help="Get a free key at console.groq.com")
    model_in = st.selectbox("Model", MODELS, index=MODELS.index(st.session_state.model))
    st.session_state.api_key = key_in
    st.session_state.model   = model_in

    # Rebuild agent if model/key changed
    prev_key   = st.session_state.get("_prev_key")
    prev_model = st.session_state.get("_prev_model")
    if (key_in and model_in) and (key_in != prev_key or model_in != prev_model):
        st.session_state.agent      = _build_graph()
        st.session_state._prev_key  = key_in
        st.session_state._prev_model = model_in

    st.divider()
    if st.button("🗑️ Reset Everything", type="secondary", use_container_width=True):
        for k in ["constraints", "timetable", "timetable_history",
                  "display_messages", "tt_updated", "agent", "memory",
                  "_prev_key", "_prev_model"]:
            st.session_state.pop(k, None)
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.subheader("📤 Export")
    if st.session_state.timetable:
        pdf_data  = _export_pdf()
        xlsx_data = _export_excel()
        c1, c2 = st.columns(2)
        if pdf_data:
            c1.download_button("⬇️ PDF", pdf_data, "timetable.pdf",
                               "application/pdf", use_container_width=True)
        else:
            c1.caption("pip install fpdf2")
        c2.download_button("⬇️ Excel", xlsx_data, "timetable.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
        history = st.session_state.get("timetable_history", [])
        if history:
            st.divider()
            st.caption(f"📚 {len(history)} version(s) saved")
    else:
        st.caption("_Generate a timetable first._")

# ── Timetable Display ─────────────────────────────────────────────────────────────
if st.session_state.timetable:
    st.subheader("📊 Current Timetable")
    _show_tt(st.session_state.timetable, key_prefix="main")
    st.divider()

# ── Chat ──────────────────────────────────────────────────────────────────────────
st.subheader("💬 Chat with SLOT AI")

for msg in st.session_state.get("display_messages", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.api_key:
    st.info("👈 Enter your Groq API key in the sidebar to get started.")
    st.stop()

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
if "agent" not in st.session_state:
    st.session_state.agent = _build_graph()

if prompt := st.chat_input("Describe your timetable — professors, rooms, slots, days, and any rules…"):
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Inject API key into env so ChatGroq finds it regardless of async context
    os.environ["GROQ_API_KEY"] = st.session_state.api_key

    # Point the thread-local session id at this session's store dict.
    # LangGraph invoke() is synchronous — tools run on the same OS thread,
    # so _tl.session_id is visible inside every tool call without any context copying.
    tid = st.session_state.thread_id
    _tl.session_id = tid
    _STORE[tid] = {
        "api_key":           st.session_state.api_key,
        "model":             st.session_state.model,
        "constraints":       copy.deepcopy(st.session_state.constraints),
        "timetable":         copy.deepcopy(st.session_state.timetable),
        "timetable_history": list(st.session_state.timetable_history),
        "tt_updated":        False,
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                config = {"configurable": {"thread_id": tid}}
                result = st.session_state.agent.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config,
                )
                reply = result["messages"][-1].content
            except Exception as e:
                err = str(e)
                if "429" in err or "rate_limit" in err.lower():
                    model_now = st.session_state.get("model", MODELS[0])
                    if "8b" in model_now:
                        reply = "⚠️ Rate limit hit on 8B model too. Wait ~1 minute and try again."
                    else:
                        reply = "⚠️ Rate limit hit. Switch to llama-3.1-8b-instant in the sidebar and try again."
                elif "tool_use_failed" in err or "failed_generation" in err:
                    reply = "⚠️ The model failed to call a tool correctly. Try rephrasing your request more simply."
                else:
                    reply = f"⚠️ Error ({type(e).__name__}): {err[:300]}"

        # Sync tool-updated state back into session_state
        sess = _STORE.get(tid, {})
        st.session_state.constraints       = sess.get("constraints",       st.session_state.constraints)
        st.session_state.timetable         = sess.get("timetable",         st.session_state.timetable)
        st.session_state.timetable_history = sess.get("timetable_history", st.session_state.timetable_history)
        st.session_state.tt_updated        = sess.get("tt_updated", False)

        st.markdown(reply)

    st.session_state.display_messages.append({"role": "assistant", "content": reply})
    if st.session_state.tt_updated:
        st.rerun()
