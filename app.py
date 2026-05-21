"""SLOT AI — School Timetable Scheduler
pip install streamlit langchain==0.2.16 langchain-groq==0.1.9 langchain-community==0.2.16 pandas openpyxl fpdf2
streamlit run app.py
"""
import os, json, re, random, copy
import streamlit as st
import pandas as pd
from io import BytesIO
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

MODELS = {
    "llama-3.3-70b-versatile": "Llama 3.3 70B — best quality, 100K tokens/day (default)",
    "llama-3.1-8b-instant":    "Llama 3.1 8B — if 70B hits limit, switch here (~500K/day)",
}
st.set_page_config(page_title="SLOT AI", page_icon="📅", layout="wide")

for k, v in dict(messages=[], constraints={}, timetable={}, agent=None, api_key="",
                 tt_updated=False, key_changed=False, model="llama-3.3-70b-versatile").items():
    st.session_state.setdefault(k, v)
if not st.session_state.api_key and os.getenv("GROQ_API_KEY"):
    st.session_state.api_key = os.getenv("GROQ_API_KEY", "")

def _llm(t: float = 0.0): return ChatGroq(api_key=st.session_state.api_key, model=st.session_state.model, temperature=t)
def _strip(s): return re.sub(r"\n?```$", "", re.sub(r"^```[a-z]*\n?", "", s.strip())).strip()

def _missing(c):
    return [m for k, m in [("teachers", "professor→subject mapping"), ("slots", "time slots"),
                            ("rooms", "room names"), ("days", "working days")] if not c.get(k)]

def _cell(e):
    if isinstance(e, dict):
        s = e.get("subject", "")
        return f"{s} ({e.get('professor', '')})" if s not in ("FREE", "", None) else "—"
    return str(e).strip() or "—"

def _to_dfs(tt):
    out = {}
    for day, data in tt.items():
        if not data: continue
        rs = list(next(iter(data.values()))); ss = list(data)
        out[day] = pd.DataFrame({r: [_cell(data[s].get(r, "—")) for s in ss] for r in rs},
                                 index=pd.Index(ss, name="Time"))
    return out

def _show_tt(tt):
    for day, df in _to_dfs(tt).items():
        st.markdown(f"**{day}**"); st.dataframe(df, use_container_width=True)

# ── Solver ─────────────────────────────────────────────────────────────────────
def _solve(c):
    sp = {s: p for p, ss in c.get("teachers", {}).items() for s in ss}
    days, rooms, slots = c["days"], c["rooms"], c["slots"]
    morning = set(c.get("morning_slots", [])); mo = set(c.get("morning_only_subjects", []))
    restr = c.get("room_restrictions", {}); unavail = c.get("professor_unavailability", {})
    custom = c.get("custom_rules", []); tt = {}

    for day in days:
        tt[day] = {}
        off = {p for p, ds in unavail.items() if day in ds}; prev = set()
        for slot in slots:
            avail = [s for s in sp if sp[s] not in off and (s not in mo or slot in morning)]

            def bt(rl, used, out, relax=False, _a=avail, _p=prev):
                if not rl: return True
                r, *rest = rl; cands = _a[:]; random.shuffle(cands)
                for s in cands:
                    p = sp[s]
                    if p in used or r in restr.get(s, []): continue
                    if not relax and p in _p: continue
                    out[r] = {"subject": s, "professor": p}
                    if bt(rest, used | {p}, out, relax, _a, _p): return True
                    del out[r]
                return False

            asgn = {}
            if not bt(list(rooms), set(), asgn): asgn = {}; bt(list(rooms), set(), asgn, True)
            tt[day][slot] = {r: asgn.get(r, {"subject": "FREE", "professor": ""}) for r in rooms}
            prev = {v["professor"] for v in asgn.values()}

    if custom:
        raw = _strip(_llm().invoke(
            "Fix this timetable JSON to respect:\n" + "\n".join(f"- {r}" for r in custom) +
            f"\nJSON:{json.dumps(tt,separators=(',',':'))}\nReturn ONLY valid JSON:"
        ).content)
        try: tt = json.loads(raw)
        except: pass
    return tt

# ── Exports ────────────────────────────────────────────────────────────────────
def _export_pdf():
    try:
        from fpdf import FPDF
        pdf = FPDF()
        for day, df in _to_dfs(st.session_state.timetable).items():
            pdf.add_page(); pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, day); pdf.ln()
            pdf.set_font("Helvetica", size=7)
            cols = ["Time"] + list(df.columns); w = 190 / len(cols)
            pdf.set_fill_color(220, 220, 220)
            for col in cols: pdf.cell(w, 7, str(col), border=1, fill=True)
            pdf.ln()
            for idx, row in df.iterrows():
                pdf.cell(w, 7, str(idx), border=1)
                for v in row: pdf.cell(w, 7, str(v)[:20], border=1)
                pdf.ln()
        return bytes(pdf.output())
    except ImportError: return None

def _export_excel():
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for day, df in _to_dfs(st.session_state.timetable).items():
            df.to_excel(w, sheet_name=day[:31])
    return buf.getvalue()

# ── Agent tools ────────────────────────────────────────────────────────────────
def t_parse(text):
    raw = _strip(_llm(0.1).invoke(
        "Extract scheduling constraints as JSON. Allowed keys only:\n"
        '{"teachers":{"Prof":["Subj"]},"rooms":[],"slots":[],"days":[],"breaks":[],'
        '"morning_slots":[],"morning_only_subjects":[],"room_restrictions":{"Subj":["Room"]},'
        '"professor_unavailability":{"Prof":["Day"]},'
        '"custom_rules":["any other constraint verbatim as a plain string"]}\n\n'
        "CRITICAL rules:\n"
        "- teachers: ProfessorName → list of subjects they teach. The input often uses\n"
        "  'Subject-Professor' or 'Subject:Professor' notation — INVERT it.\n"
        "  Example: 'AI-Vaibhav, ML-Vaibhav, DSA-Simha' → {\"Vaibhav\":[\"AI\",\"ML\"],\"Simha\":[\"DSA\"]}\n"
        "- rooms: shared spaces (e.g. A, B, C, D) — NOT subject-specific, just a list\n"
        "- Copy professor names EXACTLY as written — do NOT add 'Dr.', 'Prof.', or any title prefix\n"
        "- morning_slots: slots before the first break\n"
        "- room_restrictions: subject→rooms it CANNOT use (only if user says so)\n"
        "- professor_unavailability: prof→days absent/unavailable\n"
        "- custom_rules: ALL other constraints verbatim (frequency limits, consecutive slots, etc.)\n"
        "- Exclude break ranges from slots; omit keys not mentioned\n\n"
        f"Text: {text}\n\nJSON:"
    ).content)
    try:
        p = json.loads(raw); c = st.session_state.constraints
        for k, v in p.items():
            if k in ("teachers", "room_restrictions", "professor_unavailability") and isinstance(v, dict):
                d = c.setdefault(k, {})
                for kk, vv in v.items():
                    if isinstance(vv, list):
                        e = d.setdefault(kk, [])
                        for i in vv:
                            if i not in e: e.append(i)
                    else: d[kk] = vv
            elif k == "custom_rules" and isinstance(v, list):
                e = c.setdefault("custom_rules", [])
                for r in v:
                    if r not in e: e.append(r)
            else: c[k] = v
        miss = _missing(c)
        if miss:
            return "Stored. Still missing: " + ", ".join(miss) + ". Ask the user for the missing items."
        return "All required info collected (teachers, slots, rooms, days). Call generate_timetable now."
    except json.JSONDecodeError: return f"Parse failed: {raw}"

def t_generate(_):
    c = st.session_state.constraints; miss = _missing(c)
    if miss: return "Still need:\n" + "\n".join(f"- {m}" for m in miss)
    tt = _solve(c)
    if not tt: return "Could not generate. Check constraints."
    st.session_state.timetable = tt; st.session_state.tt_updated = True
    return "Timetable generated!"

def t_display(_):
    if not st.session_state.timetable: return "No timetable yet. Generate one first!"
    st.session_state.tt_updated = True; return "Displaying timetable."

def t_edit(instr):
    if not st.session_state.timetable: return "No timetable to edit yet."
    raw = _strip(_llm().invoke(
        f"Apply this edit and return ONLY valid JSON (no spaces):\n"
        f"Timetable:{json.dumps(st.session_state.timetable,separators=(',',':'))}\nEdit:{instr}\nJSON:"
    ).content)
    try:
        st.session_state.timetable = json.loads(raw); st.session_state.tt_updated = True
        return "Timetable updated!"
    except: return "Edit failed — invalid JSON returned."

TOOLS = [
    Tool("parse_constraints", t_parse,
         "Extract and store ALL scheduling info from user text: professors, subjects, rooms, slots, "
         "days, breaks, unavailability, room restrictions, and any custom rules. "
         "Call this for ANY scheduling detail the user mentions."),
    Tool("generate_timetable", t_generate,
         "Build the full weekly timetable from stored constraints. Call immediately once all "
         "required info (teachers, slots, rooms, days) is present, or when user asks to generate."),
    Tool("display_timetable", t_display,
         "Show the current timetable. Call when user says show/display/view/print/see."),
    Tool("edit_timetable", t_edit,
         "Apply a natural-language edit to the existing timetable. Input: the edit instruction."),
]

_PREFIX = """\
You are SLOT AI — a school timetable scheduling assistant. You adapt to whatever the teacher provides.

Rules:
1. ANY scheduling detail in the user message → call parse_constraints immediately.
2. parse_constraints tells you exactly what is still missing. Trust it — do NOT guess what is missing.
3. If parse_constraints says "All required info collected" → call generate_timetable immediately.
4. If info is missing → ask the user ONE question for ONE missing field. Nothing else.
5. Rooms (e.g. A, B, C, D) are SHARED spaces — any subject can go in any room. Never ask which
   room is "assigned to" a subject unless the user explicitly said so.
6. "show"/"display"/"view"/"print"/"see" → call display_timetable.
7. "swap"/"move"/"change"/"fix"/"remove"/"update" → call edit_timetable.
8. New constraint or unavailability → parse_constraints, then regenerate.
9. Never invent or assume scheduling details. After a tool call, reply in one sentence.

Tools:\
"""

def _build_agent(key):
    return initialize_agent(
        TOOLS, ChatGroq(api_key=key, model=st.session_state.model, temperature=0),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        verbose=False, handle_parsing_errors=True, max_iterations=8,
        agent_kwargs={"prefix": _PREFIX},
    )

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("📅 SLOT AI")

with st.sidebar:
    st.header("⚙️ Setup")
    key_in = st.text_input("Groq API Key", type="password", value=st.session_state.api_key,
                           placeholder="gsk_...", help="Get a free key at console.groq.com")
    model_in = st.selectbox("Model", list(MODELS.keys()),
                            index=list(MODELS.keys()).index(st.session_state.model),
                            format_func=lambda m: MODELS[m])
    if key_in != st.session_state.api_key or model_in != st.session_state.model:
        st.session_state.api_key = key_in
        st.session_state.model = model_in
        st.session_state.agent = None
        if st.session_state.timetable:
            st.session_state.key_changed = True
    if st.session_state.api_key and st.session_state.agent is None:
        with st.spinner("Refreshing agent…"):
            try: st.session_state.agent = _build_agent(st.session_state.api_key); st.success("✅ Agent ready")
            except Exception as e: st.error(f"Agent init failed: {e}")

    st.divider()
    if st.button("🗑️ Reset Everything", type="secondary", use_container_width=True):
        st.session_state.update(messages=[], constraints={}, timetable={}, agent=None, tt_updated=False)
        st.rerun()

    st.divider()
    st.subheader("📤 Export Timetable")
    if st.session_state.timetable:
        pdf_data = _export_pdf(); xlsx_data = _export_excel()
        c1, c2 = st.columns(2)
        if pdf_data:
            c1.download_button("⬇️ PDF", pdf_data, "timetable.pdf", "application/pdf", use_container_width=True)
        else:
            c1.caption("PDF: run `pip install fpdf2`")
        c2.download_button("⬇️ Excel", xlsx_data, "timetable.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    else:
        st.caption("_Generate a timetable first._")

# ── Chat ───────────────────────────────────────────────────────────────────────
st.subheader("💬 Chat with SLOT AI")
if st.session_state.key_changed:
    st.session_state.messages.append({
        "role": "assistant",
        "content": "🔑 API key updated — agent refreshed with new key. Your timetable and history are preserved. Continue where you left off!",
        "timetable": None
    })
    st.session_state.key_changed = False
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("timetable"):
            _show_tt(msg["timetable"])

if not st.session_state.api_key:
    st.info("👈 Enter your Groq API key in the sidebar to get started.")
    st.stop()

if prompt := st.chat_input("Describe your timetable — professors, subjects, rooms, slots, days, and any rules…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    st.session_state.tt_updated = False
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            if st.session_state.agent is None:
                reply = "Agent not ready — check your Groq API key in the sidebar."
            else:
                try:
                    result = st.session_state.agent.invoke({"input": prompt})
                    reply = result.get("output", str(result))
                except Exception as e:
                    err = str(e)
                    if "429" in err or "rate_limit" in err.lower():
                        reply = "⚠️ Groq API daily token limit reached. Please wait a few minutes and try again, or upgrade at console.groq.com/settings/billing."
                    else:
                        reply = f"Something went wrong: {e}"
        st.markdown(reply)
        tt_snap = copy.deepcopy(st.session_state.timetable) if st.session_state.tt_updated else None
        if tt_snap: _show_tt(tt_snap)

    st.session_state.messages.append({"role": "assistant", "content": reply, "timetable": tt_snap})
    if tt_snap:
        st.rerun()
