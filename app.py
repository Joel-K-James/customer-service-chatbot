import streamlit as st
import sqlite3
import json
import sys
import os
import re
from datetime import datetime

# ==================================================================
# Page config
# ==================================================================
st.set_page_config(
    page_title="Customer Service AI",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================================
# Styles
# ==================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.stApp { background-color: #0f0f0f; color: #e5e5e5; }

[data-testid="stSidebar"] { background-color: #111111 !important; border-right: 1px solid #222 !important; }
[data-testid="stSidebar"] * { color: #e5e5e5 !important; }
[data-testid="stSidebar"] .stButton button { background: transparent !important; border: 1px solid #222 !important; text-align: left !important; padding: 9px 12px !important; border-radius: 8px !important; font-size: 13px !important; color: #b0b0b0 !important; width: 100% !important; }
[data-testid="stSidebar"] .stButton button:hover { background: #1e1e1e !important; color: #fff !important; border-color: #10a37f !important; }

[data-testid="stChatInput"] textarea { background-color: #1a1a1a !important; border: 1px solid #2a2a2a !important; border-radius: 24px !important; color: #e5e5e5 !important; font-size: 14px !important; }
[data-testid="stChatInput"] textarea:focus { border-color: #10a37f !important; box-shadow: 0 0 0 2px rgba(16,163,127,0.15) !important; }
[data-testid="stExpander"] { background: #1a1a1a !important; border: 1px solid #2a2a2a !important; border-radius: 8px !important; }

.section-label { font-size: 11px; color: #555; text-transform: uppercase; letter-spacing: .06em; margin: 14px 4px 6px; }
.status-dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:6px; }
.dot-green { background:#10a37f; } .dot-red { background:#ef4444; }

.bubble-row-user { display:flex; justify-content:flex-end; margin:10px 0; }
.bubble-row-bot  { display:flex; justify-content:flex-start; margin:10px 0; }
.bubble-user { background:#1e293b; border:1px solid #334155; color:#e5e5e5; border-radius:16px 16px 4px 16px; padding:11px 16px; max-width:74%; font-size:14px; line-height:1.6; }
.bubble-bot  { background:linear-gradient(135deg,rgba(16,163,127,.14),rgba(14,165,233,.12)); border:1px solid rgba(16,163,127,.3); color:#e5e5e5; border-radius:16px 16px 16px 4px; padding:11px 16px; max-width:74%; font-size:14px; line-height:1.6; white-space:pre-wrap; }
.bubble-err  { background:rgba(239,68,68,.1); border:1px solid rgba(239,68,68,.35); color:#fca5a5; border-radius:12px; padding:11px 16px; max-width:90%; font-size:13px; white-space:pre-wrap; font-family:monospace; }
.meta-time { font-size:10px; color:#444; margin:2px 6px; }

.info-row { display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid #1e1e1e; font-size:12px; }
.info-label { color:#666; } .info-value { color:#e5e5e5; font-weight:500; }
.tag { display:inline-block; padding:3px 10px; border-radius:99px; font-size:11px; font-weight:500; margin:2px 3px 2px 0; }
.tag-green { background:rgba(16,163,127,.15); color:#10a37f; border:1px solid rgba(16,163,127,.25); }
.tag-red   { background:rgba(239,68,68,.12);  color:#f87171; border:1px solid rgba(239,68,68,.25); }
.tag-yellow{ background:rgba(234,179,8,.12);  color:#eab308; border:1px solid rgba(234,179,8,.25); }
.tag-blue  { background:rgba(96,165,250,.12); color:#60a5fa; border:1px solid rgba(96,165,250,.25); }
.tag-gray  { background:rgba(120,120,120,.15);color:#999;    border:1px solid rgba(120,120,120,.25); }
</style>
""", unsafe_allow_html=True)

# ==================================================================
# Paths  (must match the P5 notebook)
# ==================================================================
REPO_DIR = "."
DB_PATH  = "conversation_history.db"
P4_PATH  = os.path.join(REPO_DIR, "P4_RAG_Agent_Engineer.ipynb")
P3_PATH  = os.path.join(REPO_DIR, "P3_Qwen_LoRA.ipynb")
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


# ==================================================================
# Notebook loader — execs code cells from another notebook into a globals dict
# ==================================================================
def load_notebook_functions(notebook_path, g):
    import nbformat
    if not os.path.exists(notebook_path):
        return False
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        # drop shell (!) lines and notebook magics (%)
        lines = [l for l in cell.source.splitlines()
                 if not l.strip().startswith("!") and not l.strip().startswith("%")]
        src = "\n".join(lines).strip()
        if not src:
            continue
        try:
            exec(compile(src, notebook_path, "exec"), g)
        except Exception:
            # training / NameError cells are expected to fail when loaded in isolation
            pass
    return True


def fix_lora_config():
    """Make the LoRA adapter_config compatible with the installed PEFT version."""
    import shutil
    if os.path.exists("lora-weight") and not os.path.exists("lora-weights"):
        try:
            shutil.copytree("lora-weight", "lora-weights")
        except Exception:
            pass
    cfg_path = "./lora-weights/adapter_config.json"
    if not os.path.exists(cfg_path):
        return
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        bad_keys = [
            "lora_ga_config", "alora_invocation_tokens", "arrow_config", "corda_config",
            "ensure_weight_tying", "eva_config", "exclude_modules", "qalora_group_size",
            "target_parameters", "trainable_token_indices", "use_bdlora", "use_qalora", "lora_bias",
        ]
        for k in bad_keys:
            cfg.pop(k, None)
        cfg["peft_version"] = "0.13.2"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


# ==================================================================
# Load the whole pipeline ONCE (cached for the session)
# ==================================================================
@st.cache_resource(show_spinner="Loading AI models (P2 \u2192 P4 \u2192 P3)\u2026 first run can take a few minutes.")
def load_pipeline():
    g = {}

    # ---- P2: load pkl models + spaCy, define analyze() ----
    analyze = None
    try:
        import joblib, spacy
        sentiment_model = joblib.load("models/sentiment_model.pkl")
        category_model  = joblib.load("models/category_model.pkl")
        nlp = spacy.load("en_core_web_sm")

        def analyze(text):
            clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower()).strip()
            sen_result = sentiment_model.predict([clean])[0]
            sen_proba  = sentiment_model.predict_proba([clean]).max()
            cat_result = category_model.predict([clean])[0]
            doc = nlp(text)
            entities = {"customer_name": [], "product": [], "shop": [],
                        "date": [], "money": [], "issue": [], "other": []}
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    entities["customer_name"].append(ent.text)
                elif ent.label_ in ("PRODUCT", "ORG"):
                    entities["product"].append(ent.text.lower())
                elif ent.label_ in ("GPE", "FAC"):
                    entities["shop"].append(ent.text)
                elif ent.label_ in ("DATE", "TIME"):
                    entities["date"].append(ent.text)
                elif ent.label_ == "MONEY":
                    entities["money"].append(ent.text)
            pos_tags = [(t.text, t.pos_) for t in doc]
            return {
                "clean_text": clean,
                "tokens": clean.split(),
                "pos_tags": pos_tags,
                "entities": entities,
                "sentiment": {"sentiment": sen_result, "confidence": f"{sen_proba*100:.1f}%"},
                "category": {"category": cat_result, "confidence": "model"},
            }
    except Exception as e:
        print("P2 load failed:", e)

    # ---- P4: load notebook, then apply embedding compatibility fix ----
    load_notebook_functions(P4_PATH, g)
    try:
        from sentence_transformers import SentenceTransformer
        if "embedding_info" not in g or not isinstance(g.get("embedding_info"), dict):
            g["embedding_info"] = {}
        if "embedding_model" not in g:
            g["embedding_model"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        ei = g["embedding_info"]
        ei.setdefault("method", "sentence-transformers")
        ei.setdefault("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        ei.setdefault("embedding_dim", 384)
        ei.setdefault("index_type", "FAISS")
        ei.setdefault("status", "loaded")
        ei.setdefault("vectorizer", g["embedding_model"])
    except Exception as e:
        print("P4 embedding fix failed:", e)

    # ---- P3: fix LoRA config, then load notebook ----
    fix_lora_config()
    load_notebook_functions(P3_PATH, g)

    return {
        "analyze": analyze,
        "run_part4_rag_agent": g.get("run_part4_rag_agent"),
        "final_response_for_UI": g.get("final_response_for_UI"),
    }


PIPE = load_pipeline()
_analyze = PIPE["analyze"]
_rag     = PIPE["run_part4_rag_agent"]
_reply   = PIPE["final_response_for_UI"]


# ==================================================================
# Database
# ==================================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, user_message TEXT, bot_reply TEXT,
        category TEXT, sentiment TEXT, escalated INTEGER,
        agent_action TEXT, entities TEXT)""")
    conn.commit(); conn.close()


def save_conversation(msg, res):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO conversations VALUES (NULL,?,?,?,?,?,?,?,?)",
                 (datetime.now().isoformat(timespec="seconds"), msg,
                  res.get("reply", ""), res.get("category", ""), res.get("sentiment", ""),
                  int(res.get("escalate", False)), res.get("agent_action", ""),
                  json.dumps(res.get("entities", {}))))
    conn.commit(); conn.close()


def load_history(limit=6):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT timestamp, user_message, category, sentiment, escalated "
        "FROM conversations ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return rows


init_db()


# ==================================================================
# Pipeline  -  P2 -> P4 -> P3.  Bot reply = P3 output ONLY. No fallback.
# ==================================================================
def run_pipeline(msg):
    res = {"reply": "", "category": "UNKNOWN", "sentiment": "neutral",
           "escalate": False, "agent_action": "UNKNOWN", "entities": {},
           "pos_tags": [], "retrieved_context_text": "", "error": None}

    # P2
    p2 = _analyze(msg)
    cat = p2.get("category", {})
    sen = p2.get("sentiment", {})
    res["category"]  = cat.get("category", "UNKNOWN") if isinstance(cat, dict) else str(cat)
    res["sentiment"] = sen.get("sentiment", "neutral") if isinstance(sen, dict) else str(sen)
    res["entities"]  = p2.get("entities", {}) or {}
    res["pos_tags"]  = p2.get("pos_tags", [])

    # P4
    p4 = _rag(msg, {"category": res["category"], "sentiment": res["sentiment"],
                    "entities": res["entities"]})
    res["agent_action"] = p4.get("agent_output", {}).get("action", "UNKNOWN")
    res["escalate"]     = bool(p4.get("escalation_output", {}).get("escalate", False))
    res["retrieved_context_text"] = p4.get("retrieved_context_text", "")

    # P3  - take the model's reply exactly as produced
    p3 = _reply(msg, p2, p4)
    res["reply"] = (p3.get("reply", "") if isinstance(p3, dict) else str(p3)).strip()
    return res


# ==================================================================
# Session state
# ==================================================================
st.session_state.setdefault("messages", [])
st.session_state.setdefault("last_res", None)
st.session_state.setdefault("customer_name", "Customer")
st.session_state.setdefault("quick_msg", None)
st.session_state.setdefault("conv_id", f"CONV-{datetime.now().strftime('%H%M%S')}")


def reset_chat():
    st.session_state.messages = []
    st.session_state.last_res = None
    st.session_state.conv_id = f"CONV-{datetime.now().strftime('%H%M%S')}"


# ==================================================================
# Left panel (replaces Streamlit sidebar so it is always visible)
# ==================================================================
def render_left_panel():
    st.markdown("""
    <div style="padding:18px 14px 14px;border-bottom:1px solid #222;">
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:32px;height:32px;background:linear-gradient(135deg,#10a37f,#0ea5e9);
                        border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px;">💬</div>
            <div>
                <div style="font-size:14px;font-weight:600;">ServiceAI</div>
                <div style="font-size:11px;color:#555;">COMP8420 · Use Case 1</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    if st.button("\uff0b  New conversation", key="new_conv", use_container_width=True):
        reset_chat(); st.rerun()

    st.markdown('<div class="section-label">Customer</div>', unsafe_allow_html=True)
    name = st.text_input("name", value=st.session_state.customer_name,
                         placeholder="Customer name\u2026", label_visibility="collapsed")
    if name != st.session_state.customer_name:
        st.session_state.customer_name = name

    st.markdown('<div class="section-label">Quick Tests</div>', unsafe_allow_html=True)
    quick = [("🛍️", "I want to cancel my order"),
             ("📦", "My package has not arrived yet"),
             ("💳", "I was charged extra on my invoice"),
             ("😡", "I want to speak to a manager, this is unacceptable"),
             ("📦", "Hi, my name is John. I ordered a MacBook Pro from Apple last week. It arrived damaged and I was charged an extra $200.")]
    for icon, label in quick:
        if st.button(f"{icon}  {label}", key=f"q_{label}", use_container_width=True):
            st.session_state.quick_msg = label
            st.rerun()

    st.markdown('<div class="section-label">Pipeline Status</div>', unsafe_allow_html=True)
    for lbl, ok in [("P2 NLP Analysis", _analyze is not None),
                    ("P4 RAG + Agent", _rag is not None),
                    ("P3 LLM Response", _reply is not None)]:
        dot = "dot-green" if ok else "dot-red"
        st.markdown(f'<div style="font-size:12px;padding:4px;color:#aaa;">'
                    f'<span class="status-dot {dot}"></span>{lbl}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Recent</div>', unsafe_allow_html=True)
    try:
        for ts, m, c, s, e in load_history(5):
            icon = "🔴" if e else "🟢"
            st.markdown(f'<div style="font-size:12px;color:#999;padding:4px 2px;">'
                        f'{icon} {c or "—"} · <span style="color:#555;">{(m or "")[:24]}\u2026</span></div>',
                        unsafe_allow_html=True)
    except Exception:
        st.markdown('<div style="font-size:12px;color:#444;padding:4px;">No history yet.</div>',
                    unsafe_allow_html=True)



# ==================================================================
# Main layout
# ==================================================================
col_left, col_chat, col_info = st.columns([1.15, 3, 2], gap="medium")

with col_left:
    render_left_panel()

with col_chat:
    top_l, _top_r = st.columns([1, 4])
    with top_l:
        if st.button("\uff0b New Chat", key="new_chat_top", use_container_width=True):
            reset_chat(); st.rerun()

    r = st.session_state.last_res
    cat_val = r.get("category", "—") if r else "—"
    sen_val = (r.get("sentiment", "neutral") if r else "neutral")
    esc_val = r.get("escalate", False) if r else False
    sen_map = {"positive": ("tag-green", "POSITIVE"),
               "negative": ("tag-red", "NEGATIVE"),
               "neutral":  ("tag-yellow", "NEUTRAL")}
    sen_cls, sen_lbl = sen_map.get(str(sen_val).lower(), ("tag-gray", str(sen_val).upper()))
    esc_cls, esc_lbl = ("tag-red", "Escalated") if esc_val else ("tag-green", "Automated")
    init = (st.session_state.customer_name or "C")[0].upper()

    st.markdown(f"""
    <div style="padding:14px 4px 12px;border-bottom:1px solid #1e1e1e;
                display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:36px;height:36px;background:linear-gradient(135deg,#6366f1,#8b5cf6);
                        border-radius:50%;display:flex;align-items:center;justify-content:center;
                        font-weight:600;color:#fff;">{init}</div>
            <div>
                <div style="font-size:15px;font-weight:600;">{st.session_state.customer_name}</div>
                <div style="font-size:11px;color:#555;">{st.session_state.conv_id} · Web chat</div>
            </div>
        </div>
        <div>
            <span class="tag tag-blue">{cat_val}</span>
            <span class="tag {sen_cls}">{sen_lbl}</span>
            <span class="tag {esc_cls}">{esc_lbl}</span>
        </div>
    </div>""", unsafe_allow_html=True)

    # scrollable chat area (fixed height, scrolls internally)
    chat_box = st.container(height=440, border=False)
    with chat_box:
        if not st.session_state.messages:
            st.markdown(f"""
            <div style="text-align:center;padding:60px 20px;color:#444;">
                <div style="font-size:40px;">👋</div>
                <div style="font-size:15px;color:#666;margin-top:8px;">New conversation with {st.session_state.customer_name}</div>
                <div style="font-size:12px;color:#444;margin-top:4px;">Type a message or use a quick test from the left panel</div>
            </div>""", unsafe_allow_html=True)
        else:
            for m in st.session_state.messages:
                content = (m["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
                t = m.get("time", "")
                if m["role"] == "user":
                    st.markdown(f'<div class="bubble-row-user"><div class="bubble-user">{content}</div></div>'
                                f'<div class="meta-time" style="text-align:right;">{t}</div>',
                                unsafe_allow_html=True)
                elif m.get("is_error"):
                    st.markdown(f'<div class="bubble-row-bot"><div class="bubble-err">{content}</div></div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bubble-row-bot"><div class="bubble-bot">{content}</div></div>'
                                f'<div class="meta-time">{t}</div>', unsafe_allow_html=True)
        # auto-scroll to the latest message
        st.markdown('<div id="chat-end"></div>'
                    '<script>var e=document.getElementById("chat-end");'
                    'if(e){e.scrollIntoView();}</script>', unsafe_allow_html=True)

    # input
    user_input = st.chat_input(f"Message {st.session_state.customer_name}\u2026")
    if st.session_state.quick_msg:
        user_input = st.session_state.quick_msg
        st.session_state.quick_msg = None

    if user_input:
        st.session_state.messages.append(
            {"role": "user", "content": user_input, "time": datetime.now().strftime("%H:%M")})
        with st.spinner("Analysing\u2026"):
            try:
                res = run_pipeline(user_input)
                st.session_state.last_res = res
                save_conversation(user_input, res)
                st.session_state.messages.append(
                    {"role": "assistant", "content": res["reply"] or "(empty response from P3)",
                     "time": datetime.now().strftime("%H:%M")})
            except Exception as e:
                # surface the real error instead of a canned fallback
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"PIPELINE ERROR:\n{e}",
                     "time": datetime.now().strftime("%H:%M"), "is_error": True})
        st.rerun()


# ==================================================================
# Right panel - live NLP insights (metadata only; reply stays pure P3)
# ==================================================================
with col_info:
    st.markdown('<div style="padding:14px 4px 10px;border-bottom:1px solid #1e1e1e;">'
                '<div style="font-size:14px;font-weight:600;">Insights</div>'
                '<div style="font-size:11px;color:#555;">Real-time NLP analysis</div></div>',
                unsafe_allow_html=True)

    r = st.session_state.last_res
    if r:
        for label, value in [
            ("Conversation ID", st.session_state.conv_id),
            ("Channel", "Web chat"),
            ("Status", "🔴 Escalated" if r.get("escalate") else "🟢 Open"),
            ("Category", r.get("category", "—")),
            ("Sentiment", str(r.get("sentiment", "—")).title()),
            ("Agent Action", str(r.get("agent_action", "—")).replace("_", " ").title()),
        ]:
            st.markdown(f'<div class="info-row"><span class="info-label">{label}</span>'
                        f'<span class="info-value">{value}</span></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Extracted Entities (NER)</div>', unsafe_allow_html=True)
        ents = r.get("entities", {})
        has = False
        for label, key in [("Customer", "customer_name"), ("Product", "product"),
                           ("Shop", "shop"), ("Date", "date"), ("Money", "money"), ("Issue", "issue")]:
            vals = ents.get(key, [])
            if vals:
                has = True
                st.markdown(f'<div class="info-row"><span class="info-label">{label}</span>'
                            f'<span class="info-value" style="color:#10a37f;">'
                            f'{", ".join(str(v) for v in vals)}</span></div>', unsafe_allow_html=True)
        if not has:
            st.markdown('<div style="font-size:12px;color:#444;padding:6px 0;">No entities extracted.</div>',
                        unsafe_allow_html=True)

        pos = r.get("pos_tags", [])
        if pos:
            import pandas as pd
            with st.expander("🏷️ POS Tags — P2"):
                st.dataframe(pd.DataFrame([{"Word": w, "POS": p} for w, p in pos[:14]]),
                             hide_index=True, use_container_width=True)

        rag = r.get("retrieved_context_text", "")
        if rag:
            with st.expander("🔍 RAG Context — P4"):
                st.markdown(f'<div style="font-size:11px;color:#aaa;line-height:1.6;white-space:pre-wrap;">'
                            f'{rag[:600]}{"…" if len(rag) > 600 else ""}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:center;padding:48px 16px;color:#333;">'
                    '<div style="font-size:28px;">📊</div>'
                    '<div style="font-size:13px;color:#444;margin-top:8px;">'
                    'Send a message to see real-time NLP insights.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">History</div>', unsafe_allow_html=True)
    try:
        import pandas as pd
        rows = load_history(6)
        if rows:
            df = pd.DataFrame(rows, columns=["Time", "Message", "Category", "Sentiment", "Escalated"])
            df["Message"]   = df["Message"].str[:26] + "\u2026"
            df["Time"]      = df["Time"].str[11:16]
            df["Escalated"] = df["Escalated"].apply(lambda x: "🔴" if x else "🟢")
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.markdown('<div style="font-size:12px;color:#333;">No conversations yet.</div>',
                        unsafe_allow_html=True)
    except Exception:
        st.markdown('<div style="font-size:12px;color:#333;">History unavailable.</div>',
                    unsafe_allow_html=True)
