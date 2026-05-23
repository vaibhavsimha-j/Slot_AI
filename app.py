"""SLOT AI — Production Timetable Scheduling Agent
pip install streamlit langchain-groq langgraph langchain-core ortools pydantic pandas openpyxl fpdf2
streamlit run app.py
"""
import os, json, re, copy, uuid, contextvars
from datetime import datetime
from typing import Annotated, Any, TypedDict
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

# ── Thread-safe session store (tools run in async context, can't use st.session_state) ──
# ContextVar holds the current session ID; _STORE holds per-session state dicts.
_SESSION_ID: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="")
_STORE: dict[str, dict] = {}

def _s() -> dict:
    """Return the current session's state dict (safe from any async/thread context)."""
    sid = _SESSION_ID.get()
    return _STORE.setdefault(sid, {
        "api_key": "", "model": MODELS[0],
        "constraints": {}, "timetable": {}, "timetable_history": [], "tt_updated": False,
    })

# ── Utilities ───────────────────────────────────────────────────────────────────
def _get_llm(t: float = 0.0) -> ChatGroq:
    s = _s()
    return ChatGroq(
        api_key=s.get("api_key", ""),
        model=s.get("model", MODELS[0]),
        temperature=t,
    )

def _extract_json(s: str):
    s = re.sub(r"^```[a-z]*\n?", "", s.strip())
    s = re.sub(r"\n?```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None

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

def _show_tt(tt: dict):
    for day, df in _to_dfs(tt).items():
        st.markdown(f"**{day}**")
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
                if len(ids) > 1:
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
_EXTRACTION_PROMPT = """\
Extract scheduling constraints from the user text as JSON and merge with existing.

Schema (include ONLY keys that appear or change):
{{
  "teachers": {{"Prof": ["Subj"]}},
  "rooms": [],
  "slots": [],
  "days": [],
  "room_restrictions":        {{"Subj": ["ForbiddenRoom"]}},
  "professor_unavailability": {{"Prof": ["Day"]}},
  "slot_restrictions":        {{"Subj": ["AllowedSlot"]}},
  "max_slots_per_day":        {{"Prof": 3}},
  "fixed_assignments": [{{"day":"","slot":"","room":"","subject":""}}],
  "custom_rules": ["verbatim rule text"]
}}

Rules:
- Invert Subject-Prof notation → {{Prof: [Subject]}}
- slot_restrictions: convert "X only morning/afternoon" to exact slots from {slots}
- If message only updates availability/restrictions, omit rooms/slots/days/teachers
- Copy all names EXACTLY as written — no Dr./Prof. prefix
- fixed_assignments: only when user pins a specific cell (e.g. "swap A and B at Monday 09:00")

Existing constraints:
{existing}

User text: {text}

JSON:"""

def _llm_extract(text: str, current: dict) -> dict:
    raw = _get_llm(0.0).invoke(
        _EXTRACTION_PROMPT.format(
            slots=current.get("slots", []),
            existing=json.dumps(current, indent=2),
            text=text,
        )
    ).content
    return _extract_json(raw) or {}

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
def extract_constraints(user_text: str) -> str:
    """Extract and store scheduling constraints from user text.
    Call whenever the user provides professors, subjects, rooms, time slots,
    days, unavailability, room restrictions, or any other scheduling rules."""
    sess     = _s()
    current  = sess.get("constraints", {})
    parsed   = _llm_extract(user_text, current)
    if not parsed:
        return json.dumps({"status": "error", "message": "Could not parse constraints from input."})

    merged = _merge(current, parsed)
    sess["constraints"] = merged
    miss   = _missing(merged)
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
def edit_timetable(instruction: str) -> str:
    """Apply an edit or update to the current timetable.
    Use for: updating constraints and regenerating, swapping subjects between rooms,
    or any modification to an existing timetable."""
    sess = _s()
    if not sess.get("timetable"):
        return json.dumps({"status": "error", "message": "No timetable exists yet. Generate one first."})

    current = sess.get("constraints", {})
    parsed  = _llm_extract(instruction, current)

    if parsed:
        if "fixed_assignments" in parsed:
            current.pop("fixed_assignments", None)
        merged = _merge(current, parsed)
        sess["constraints"] = merged

    result = _ortools_solve(sess["constraints"])

    if result["status"] == "success":
        existing = sess.get("timetable", {})
        if existing:
            sess.setdefault("timetable_history", []).append({
                "version":     len(sess.get("timetable_history", [])) + 1,
                "timetable":   copy.deepcopy(existing),
                "constraints": copy.deepcopy(sess["constraints"]),
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


# ── LangGraph Agent ──────────────────────────────────────────────────────────────
TOOLS = [
    extract_constraints,
    solve_timetable,
    edit_timetable,
    validate_timetable,
    get_timetable_history,
    compare_timetables,
]

_SYSTEM = SystemMessage(content="""\
You are SLOT AI — a professional school timetable scheduling assistant.

You are the orchestrator. You NEVER generate or modify timetables yourself.
The actual solving is done by OR-Tools CP-SAT (a mathematical constraint solver).
Your job: understand the user, call the right tools, and communicate results clearly.

Tools available:
• extract_constraints  — call when user provides professors, subjects, rooms, slots, days, or any rule
• solve_timetable      — call when constraints are complete and user wants a timetable generated
• edit_timetable       — call when user wants changes to an existing timetable
• validate_timetable   — call when user wants to check their timetable for violations
• get_timetable_history — call when user asks about previous versions
• compare_timetables   — call when user wants to diff two versions (use version 0 for current)

Workflow:
1. If user gives full info in one message → call extract_constraints, then solve_timetable.
2. If info is incomplete → call extract_constraints, then tell user exactly what is missing.
3. If user wants an edit → call edit_timetable with the instruction.
4. After solving, always report the solver status (OPTIMAL / FEASIBLE) in plain language.
5. If solver returns INFEASIBLE, explain which constraints are likely conflicting.

Be concise, professional, and helpful. Do not invent timetable data.\
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
        _s = lambda t: str(t).encode("latin-1", "replace").decode("latin-1")
        pdf = FPDF()
        for day, df in _to_dfs(st.session_state.timetable).items():
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, _s(day))
            pdf.ln()
            pdf.set_font("Helvetica", size=7)
            cols = ["Time"] + list(df.columns)
            w = 190 / len(cols)
            pdf.set_fill_color(220, 220, 220)
            for col in cols:
                pdf.cell(w, 7, _s(col), border=1, fill=True)
            pdf.ln()
            for idx, row in df.iterrows():
                pdf.cell(w, 7, _s(idx), border=1)
                for v in row:
                    pdf.cell(w, 7, _s(v)[:20], border=1)
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

# ── Chat ──────────────────────────────────────────────────────────────────────────
st.subheader("💬 Chat with SLOT AI")

for msg in st.session_state.get("display_messages", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("timetable"):
            _show_tt(msg["timetable"])

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

    # Sync session_state → thread-safe store before entering async agent context
    tid = st.session_state.thread_id
    _SESSION_ID.set(tid)
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
                # copy_context() propagates the ContextVar into LangGraph's async tasks
                ctx    = contextvars.copy_context()
                result = ctx.run(
                    st.session_state.agent.invoke,
                    {"messages": [HumanMessage(content=prompt)]},
                    config,
                )
                reply = result["messages"][-1].content
            except Exception as e:
                err = str(e)
                if "429" in err or "rate_limit" in err.lower():
                    reply = (
                        "⚠️ Rate limit hit. Wait ~1 minute and try again, "
                        "or switch to llama-3.1-8b-instant in the sidebar."
                    )
                else:
                    reply = f"Something went wrong: {e}"

        # Sync tool-updated state back to session_state
        sess = _STORE.get(tid, {})
        st.session_state.constraints       = sess.get("constraints",       st.session_state.constraints)
        st.session_state.timetable         = sess.get("timetable",         st.session_state.timetable)
        st.session_state.timetable_history = sess.get("timetable_history", st.session_state.timetable_history)
        st.session_state.tt_updated        = sess.get("tt_updated", False)

        st.markdown(reply)
        tt_snap = copy.deepcopy(st.session_state.timetable) if st.session_state.tt_updated else None
        if tt_snap:
            _show_tt(tt_snap)

    st.session_state.display_messages.append(
        {"role": "assistant", "content": reply, "timetable": tt_snap}
    )
    if tt_snap:
        st.rerun()
