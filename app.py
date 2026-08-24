"""
app.py — Production-style Streamlit storefront UI for agent.py.
Run: streamlit run app.py
"""

import hashlib
import hmac
import os
import sqlite3
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

BASE_DIR   = Path(__file__).resolve().parent
DB_PATH    = BASE_DIR / "store.db"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp"]
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

st.set_page_config(
    page_title="Noir Pantry",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dark luxury theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root { --bg:#090b10; --panel:#10131a; --panel2:#151923; --border:#282d3a;
        --text:#f4f2ec; --muted:#999fab; --gold:#d6b463; --mint:#7de2bf; --red:#ff7b89; }
html,body,[class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:radial-gradient(circle at 12% 0%,#1b1830 0,transparent 29%),
         radial-gradient(circle at 95% 2%,#142920 0,transparent 24%),var(--bg); color:var(--text); }
h1,h2,h3 { font-family:'Manrope',sans-serif!important; letter-spacing:-.035em!important; }
[data-testid="stSidebar"] { background:rgba(14,17,23,.97); border-right:1px solid var(--border); }
[data-testid="stSidebar"] * { color:var(--text); }
[data-testid="stHeader"] { background:transparent; }
.hero { padding:2.1rem 2.2rem; border:1px solid var(--border); border-radius:24px;
        background:linear-gradient(135deg,rgba(31,34,48,.92),rgba(13,16,22,.95));
        box-shadow:0 18px 70px rgba(0,0,0,.32); margin-bottom:1.1rem; }
.hero .eyebrow { color:var(--mint); font-weight:700; font-size:.72rem; letter-spacing:.18em; text-transform:uppercase; }
.hero h1 { font-size:clamp(2rem,4vw,4rem); margin:.35rem 0 .5rem; line-height:1.02; }
.hero p { color:var(--muted); max-width:680px; font-size:1.02rem; margin:0; }
.pill { display:inline-block; margin:.9rem .35rem 0 0; padding:.36rem .68rem; border-radius:999px;
        border:1px solid #343947; color:#c9cdd6; background:#171b24; font-size:.75rem; }
.notice { padding:.8rem 1rem; border-radius:14px; border:1px solid #373d4c;
          color:#b7bdc9; background:rgba(20,24,33,.8); font-size:.86rem; margin:.4rem 0 1rem; }
.chat-user { margin:12px 0 12px auto; max-width:78%; padding:13px 16px; border-radius:18px 18px 4px 18px;
             background:linear-gradient(135deg,#7556d8,#5a3eb4); color:#fff; box-shadow:0 7px 30px rgba(87,55,180,.22); }
.chat-ai { margin:12px auto 12px 0; max-width:86%; padding:14px 17px; border-radius:18px 18px 18px 4px;
           background:linear-gradient(145deg,#171b24,#12151c); color:#e8e8e8; border:1px solid var(--border); }
.chat-label { font-size:.67rem; text-transform:uppercase; letter-spacing:.13em; opacity:.62; margin-bottom:5px; font-weight:700; }
div[data-testid="stChatInput"] { border:1px solid #353b49; border-radius:18px; background:#11151d; }
.stButton>button, .stDownloadButton>button { border-radius:12px!important; border:1px solid #3b4150!important;
  background:#181d27!important; color:#f4f2ec!important; font-weight:700!important; transition:.18s ease!important; }
.stButton>button:hover, .stDownloadButton>button:hover { border-color:var(--gold)!important; color:var(--gold)!important; transform:translateY(-1px); }
.stTextInput input,.stNumberInput input { background:#11151d!important; color:#f5f3ed!important; border-color:#343a48!important; }
[data-testid="stFileUploader"] { background:#11151d; border:1px dashed #3a4151; padding:.45rem; border-radius:16px; }
hr { border-color:#282d3a!important; }
.footer { color:#656c7a; text-align:center; font-size:.75rem; padding:2rem 0 1rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session / utility helpers
# ---------------------------------------------------------------------------
def _session_user_id() -> str:
    """Pseudonymous per-browser-session id. Replace with verified auth in production."""
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"guest-{uuid.uuid4().hex}"
    return st.session_state.user_id


def _verify_admin_password(password: str) -> bool:
    expected = os.getenv("ADMIN_PASSWORD", "")
    return bool(expected) and hmac.compare_digest(password, expected)


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _save_upload(uploaded) -> str:
    raw = uploaded.getvalue()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Image is larger than 5 MB.")
    ext = Path(uploaded.name).suffix.lower()
    if ext.lstrip(".") not in ALLOWED_UPLOAD_TYPES:
        raise ValueError("Only JPG, PNG and WEBP images are supported.")
    digest = hashlib.sha256(raw).hexdigest()[:18]
    name = f"{digest}{ext}"
    destination = (UPLOAD_DIR / name).resolve()
    if not destination.is_relative_to(UPLOAD_DIR.resolve()):
        raise ValueError("Invalid upload path.")
    destination.write_bytes(raw)
    return name


@st.cache_resource
def get_agent(user_id: str):
    # Delayed import lets Streamlit render a clear setup error instead of a blank page.
    from agent import build_agent
    return build_agent(user_id)


def run_agent(text: str) -> str:
    history = []
    for msg in st.session_state.messages[-16:]:
        history.append(HumanMessage(content=msg["content"]) if msg["role"] == "user"
                       else AIMessage(content=msg["content"]))
    # Current text is already at the end after append; build from stored history only.
    result = get_agent(_session_user_id()).invoke({"messages": history})
    return _content_to_text(result["messages"][-1].content)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Welcome. Tell me what you want to buy, your budget, organic preference, and minimum rating. You can also upload a product photo.",
    }]
_session_user_id()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## NOIR PANTRY")
    st.caption("AI-assisted grocery discovery")
    st.divider()

    st.markdown("### Quick search")
    q = st.text_input("Product", placeholder="e.g. honey, green tea")
    c1, c2 = st.columns(2)
    with c1:
        max_price = st.number_input("Max $", min_value=0.0, value=20.0, step=1.0)
    with c2:
        min_rating = st.number_input("Min ★", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
    organic = st.checkbox("Organic only", value=False)
    if st.button("Find products", use_container_width=True, type="primary"):
        query = f"Find {q or 'products'} under ${max_price:.2f} with at least {min_rating:.1f} rating"
        if organic:
            query += " and organic only"
        st.session_state.pending_prompt = query

    st.divider()
    st.markdown("### Search with a photo")
    uploaded = st.file_uploader("Upload product image", type=ALLOWED_UPLOAD_TYPES,
                                help="JPG, PNG or WEBP. Maximum 5 MB.")
    if uploaded and st.button("Analyze photo", use_container_width=True):
        try:
            filename = _save_upload(uploaded)
            st.session_state.pending_prompt = f"I uploaded a product image named {filename}. Find matching products."
        except ValueError as exc:
            st.error(str(exc))

    st.divider()
    if st.button("Start new conversation", use_container_width=True):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()

    with st.expander("Order administration"):
        st.caption("Requires ADMIN_PASSWORD in your .env file.")
        admin_password = st.text_input("Admin password", type="password")
        if _verify_admin_password(admin_password):
            try:
                conn = sqlite3.connect(DB_PATH)
                rows = conn.execute("""
                    SELECT id, product_name, price, user_id, created_at
                    FROM orders ORDER BY id DESC LIMIT 50
                """).fetchall()
                conn.close()
                if rows:
                    st.dataframe(
                        [{"Order": r[0], "Product": r[1], "Price": r[2], "User": r[3], "Created": r[4]} for r in rows],
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.info("No orders yet.")
            except sqlite3.Error:
                st.error("Could not load orders.")
        elif admin_password:
            st.error("Invalid password.")


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">Curated pantry · intelligent shopping</div>
  <h1>Better staples,<br>found in seconds.</h1>
  <p>Search your local catalogue by product, price, organic status and customer rating—then confirm before anything is ordered.</p>
  <span class="pill">No automatic checkout</span>
  <span class="pill">Photo search</span>
  <span class="pill">Verified catalogue only</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="notice">Orders are created only after an explicit confirmation. Product availability and pricing come directly from your local store.db.</div>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    label = "You" if msg["role"] == "user" else "Shopping assistant"
    cls = "chat-user" if msg["role"] == "user" else "chat-ai"
    safe = (str(msg["content"]).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace("\n", "<br>"))
    st.markdown(f'<div class="{cls}"><div class="chat-label">{label}</div>{safe}</div>', unsafe_allow_html=True)

prompt = st.chat_input("Ask for a product, set a budget, or confirm an order…")
if not prompt and "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Searching the catalogue…"):
        try:
            answer = run_agent(prompt)
        except EnvironmentError:
            answer = "Setup error: MISTRAL_API_KEY is missing from the .env file."
        except sqlite3.Error:
            answer = "The product database could not be read. Please verify store.db and try again."
        except Exception:
            answer = "Something went wrong while processing your request. Please try again."
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

st.markdown('<div class="footer">NOIR PANTRY · Built for deliberate, human-confirmed shopping</div>', unsafe_allow_html=True)
