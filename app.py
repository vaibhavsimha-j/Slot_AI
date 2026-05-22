"""SLOT AI — School Timetable Scheduler
pip install streamlit langchain-groq==0.1.9 pandas openpyxl fpdf2
streamlit run app.py
"""
import os, json, re, random, copy
import streamlit as st
import pandas as pd
from io import BytesIO
from langchain_groq import ChatGroq

MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
st.set_page_config(page_title="SLOT AI", page_icon="📅", layout="wide")

for k, v in dict(messages=[], constraints={}, timetable={}, api_key="",
                 model="llama-3.3-70b-versatile", tt_updated=False).items():
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

    _trivial = {"one subject", "respect break", "each cell", "per room per time", "per slot"}
    real_custom = [r for r in custom if not any(t in r.lower() for t in _trivial)]
    if real_custom:
        raw = _strip(_llm().invoke(
            "Fix this timetable JSON to respect:\n" + "\n".join(f"- {r}" for r in real_custom) +
            f"\nJSON:{json.dumps(tt,separators=(',',':'))}\nReturn ONLY valid JSON:"
        ).content)
        try: tt = json.loads(raw)
        except: pass
    return tt

# ── Exports ────────────────────────────────────────────────────────────────────
def _export_pdf():
    try:
        from fpdf import FPDF
        def _s(t): return str(t).encode("latin-1", "replace").decode("latin-1")
        pdf = FPDF()
        for day, df in _to_dfs(st.session_state.timetable).items():
            pdf.add_page(); pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, _s(day)); pdf.ln()
            pdf.set_font("Helvetica", size=7)
            cols = ["Time"] + list(df.columns); w = 190 / len(cols)
            pdf.set_fill_color(220, 220, 220)
            for col in cols: pdf.cell(w, 7, _s(col), border=1, fill=True)
            pdf.ln()
            for idx, row in df.iterrows():
                pdf.cell(w, 7, _s(idx), border=1)
                for v in row: pdf.cell(w, 7, _s(v)[:20], border=1)
                pdf.ln()
        return bytes(pdf.output())
    except ImportError: return None

def _export_excel():
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for day, df in _to_dfs(st.session_state.timetable).items():
            df.to_excel(w, sheet_name=day[:31])
    return buf.getvalue()

# ── Core ───────────────────────────────────────────────────────────────────────
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
            elif v:
                c[k] = v
        miss = _missing(c)
        return ("missing:" + ", ".join(miss)) if miss else "ok"
    except json.JSONDecodeError:
        return "error"

def t_generate():
    c = st.session_state.constraints; miss = _missing(c)
    if miss: return "missing:" + ", ".join(miss)
    tt = _solve(c)
    if not tt: return "Could not generate — check your constraints."
    st.session_state.timetable = tt; st.session_state.tt_updated = True
    return "generated"

def t_edit(instr):
    raw = _strip(_llm().invoke(
        "Apply this edit and return ONLY valid JSON (no spaces):\n"
        f"Timetable:{json.dumps(st.session_state.timetable,separators=(',',':'))}\nEdit:{instr}\nJSON:"
    ).content)
    try:
        st.session_state.timetable = json.loads(raw); st.session_state.tt_updated = True
        return "updated"
    except: return "Edit failed — invalid JSON returned."

def _handle(prompt):
    p = prompt.lower(); pw = set(p.split()); tt = st.session_state.timetable

    # Pure display request
    if tt and pw & {"show", "display", "view", "print"} and \
       not any(w in p for w in ["unavail", "cannot", "restrict", "professor", "teacher", "subject", "slot", "room"]):
        st.session_state.tt_updated = True
        return "Here's your current timetable."

    # Cell-level swap / move
    if tt and pw & {"swap", "move", "replace", "switch"}:
        r = t_edit(prompt)
        return "Done — timetable updated!" if r == "updated" else r

    # Parse constraints, then auto-generate if ready
    c_snap = copy.deepcopy(st.session_state.constraints)
    result = t_parse(prompt)

    if result == "ok":
        changed = st.session_state.constraints != c_snap
        wants_gen = bool(pw & {"generate", "create", "make", "build", "regenerate", "redo"})
        if changed or wants_gen or not tt:
            gen = t_generate()
            return "Your timetable is ready!" if gen == "generated" else gen
        st.session_state.tt_updated = True
        return "Here's your current timetable."

    if result.startswith("missing:"):
        return "I still need: **" + result[8:] + "**. Please share that."
    return "I couldn't parse that — please try rephrasing."

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("📅 SLOT AI")

with st.sidebar:
    st.header("⚙️ Setup")
    key_in = st.text_input("Groq API Key", type="password", value=st.session_state.api_key,
                           placeholder="gsk_...", help="Get a free key at console.groq.com")
    model_in = st.selectbox("Model", MODELS, index=MODELS.index(st.session_state.model))
    st.session_state.api_key = key_in
    st.session_state.model = model_in

    st.divider()
    if st.button("🗑️ Reset Everything", type="secondary", use_container_width=True):
        st.session_state.update(messages=[], constraints={}, timetable={}, tt_updated=False)
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
            try:
                reply = _handle(prompt)
            except Exception as e:
                err = str(e)
                if "429" in err or "rate_limit" in err.lower():
                    reply = "⚠️ Rate limit hit. Wait ~1 minute and try again, or switch to llama-3.1-8b-instant in the sidebar."
                else:
                    reply = f"Something went wrong: {e}"
        st.markdown(reply)
        tt_snap = copy.deepcopy(st.session_state.timetable) if st.session_state.tt_updated else None
        if tt_snap: _show_tt(tt_snap)

    st.session_state.messages.append({"role": "assistant", "content": reply, "timetable": tt_snap})
    if tt_snap:
        st.rerun()
