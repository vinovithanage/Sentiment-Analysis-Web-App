import streamlit as st
import joblib
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentIQ",
    page_icon="🧠",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0A0A0F !important;
    color: #E8E6FF !important;
    font-family: 'Syne', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(99,60,255,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 90%, rgba(255,60,120,0.12) 0%, transparent 60%),
        #0A0A0F !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
.block-container { max-width: 720px !important; padding: 2rem 1.5rem 4rem !important; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #9B8FFF;
    border: 1px solid rgba(155,143,255,0.35);
    border-radius: 100px;
    padding: 5px 16px;
    margin-bottom: 1.2rem;
    background: rgba(155,143,255,0.07);
}
.hero-title {
    font-size: clamp(2.8rem, 8vw, 4.5rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.03em;
    margin: 0 0 0.5rem;
    background: linear-gradient(135deg, #FFFFFF 30%, #9B8FFF 70%, #FF6090 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1.05rem;
    color: rgba(232,230,255,0.5);
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.01em;
}

/* ── Input card ── */
.input-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 1.8rem;
    margin: 2rem 0 1.2rem;
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
}
.input-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(155,143,255,0.6), transparent);
}
.input-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(155,143,255,0.8);
    margin-bottom: 0.8rem;
    display: block;
}

/* ── Streamlit text area override ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1.5px solid rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
    color: #E8E6FF !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    padding: 14px 16px !important;
    transition: border-color 0.25s !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: rgba(155,143,255,0.6) !important;
    box-shadow: 0 0 0 3px rgba(155,143,255,0.12) !important;
    outline: none !important;
}
.stTextArea label { display: none !important; }

/* ── Buttons ── */
.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.4rem !important;
    transition: all 0.2s ease !important;
    border: none !important;
    width: 100% !important;
}
/* Primary analyze button */
div[data-testid="column"]:first-child .stButton > button {
    background: linear-gradient(135deg, #7C5CFC, #9B3FD0) !important;
    color: #fff !important;
    box-shadow: 0 4px 20px rgba(124,92,252,0.4) !important;
}
div[data-testid="column"]:first-child .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px rgba(124,92,252,0.55) !important;
}
/* Secondary clear button */
div[data-testid="column"]:nth-child(2) .stButton > button {
    background: rgba(255,255,255,0.06) !important;
    color: rgba(232,230,255,0.7) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
}
div[data-testid="column"]:nth-child(2) .stButton > button:hover {
    background: rgba(255,255,255,0.1) !important;
    color: #E8E6FF !important;
}
/* Example buttons */
div[data-testid="column"]:nth-child(n+3) .stButton > button,
.example-row .stButton > button {
    background: rgba(255,255,255,0.04) !important;
    color: rgba(232,230,255,0.6) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    font-size: 0.82rem !important;
    padding: 0.45rem 0.9rem !important;
}
div[data-testid="column"]:nth-child(n+3) .stButton > button:hover {
    background: rgba(155,143,255,0.1) !important;
    border-color: rgba(155,143,255,0.3) !important;
    color: #E8E6FF !important;
}

/* ── Result cards ── */
.result-positive {
    background: linear-gradient(135deg, rgba(52,211,90,0.12), rgba(52,211,90,0.05));
    border: 1px solid rgba(52,211,90,0.35);
    border-radius: 18px;
    padding: 1.8rem 2rem;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
    animation: slideUp 0.4s cubic-bezier(0.16,1,0.3,1);
}
.result-negative {
    background: linear-gradient(135deg, rgba(255,60,100,0.12), rgba(255,60,100,0.05));
    border: 1px solid rgba(255,60,100,0.35);
    border-radius: 18px;
    padding: 1.8rem 2rem;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
    animation: slideUp 0.4s cubic-bezier(0.16,1,0.3,1);
}
@keyframes slideUp {
    from { opacity:0; transform: translateY(20px); }
    to   { opacity:1; transform: translateY(0); }
}
.result-emoji { font-size: 2.8rem; display: block; margin-bottom: 0.4rem; }
.result-sentiment-label {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.result-positive .result-sentiment-label { color: #4ADE80; }
.result-negative .result-sentiment-label { color: #FF6090; }
.result-confidence {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: rgba(232,230,255,0.5);
    letter-spacing: 0.05em;
}
.conf-track {
    height: 5px;
    background: rgba(255,255,255,0.08);
    border-radius: 100px;
    margin-top: 14px;
    overflow: hidden;
}
.conf-fill-pos {
    height: 100%;
    background: linear-gradient(90deg, #22C55E, #4ADE80);
    border-radius: 100px;
    transition: width 1s cubic-bezier(0.16,1,0.3,1);
}
.conf-fill-neg {
    height: 100%;
    background: linear-gradient(90deg, #EF4444, #FF6090);
    border-radius: 100px;
    transition: width 1s cubic-bezier(0.16,1,0.3,1);
}
.result-glow-pos {
    position: absolute;
    top: -40px; right: -40px;
    width: 120px; height: 120px;
    background: rgba(74,222,128,0.15);
    border-radius: 50%;
    filter: blur(30px);
}
.result-glow-neg {
    position: absolute;
    top: -40px; right: -40px;
    width: 120px; height: 120px;
    background: rgba(255,96,144,0.15);
    border-radius: 50%;
    filter: blur(30px);
}

/* ── Section labels ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(155,143,255,0.6);
    margin: 2rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.07);
}

/* ── Cleaned text expander ── */
.stExpander {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}
.stExpander summary { color: rgba(232,230,255,0.5) !important; font-family: 'DM Mono', monospace !important; font-size: 0.8rem !important; }
.stExpander pre { background: rgba(0,0,0,0.3) !important; border-radius: 8px !important; font-family: 'DM Mono', monospace !important; font-size: 0.82rem !important; color: #9B8FFF !important; }

/* ── Warning / info ── */
.stAlert { border-radius: 12px !important; border: none !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: rgba(232,230,255,0.2);
    letter-spacing: 0.08em;
    padding-top: 3rem;
}
.footer span { color: rgba(155,143,255,0.4); }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 1.5rem 0 !important; }

/* ── Hide Streamlit default elements ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    return model, tfidf

# ── Text cleaning ─────────────────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = [ps.stem(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

def predict(text):
    model, tfidf = load_model()
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    score = model.decision_function(vec)[0]
    conf = min(99, int(50 + abs(score) * 15))
    return pred, conf, cleaned

def show_result(pred, conf):
    if pred == 1:
        st.markdown(f"""
        <div class="result-positive">
            <div class="result-glow-pos"></div>
            <span class="result-emoji">😊</span>
            <div class="result-sentiment-label">Positive</div>
            <div class="result-confidence">CONFIDENCE — {conf}%</div>
            <div class="conf-track"><div class="conf-fill-pos" style="width:{conf}%"></div></div>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f"""
        <div class="result-negative">
            <div class="result-glow-neg"></div>
            <span class="result-emoji">😞</span>
            <div class="result-sentiment-label">Negative</div>
            <div class="result-confidence">CONFIDENCE — {conf}%</div>
            <div class="conf-track"><div class="conf-fill-neg" style="width:{conf}%"></div></div>
        </div>
        """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">NLP · Machine Learning · LinearSVC</div>
    <h1 class="hero-title">SentimentIQ</h1>
    <p class="hero-sub">Drop any text. Instantly know if it's positive or negative.</p>
</div>
""", unsafe_allow_html=True)

# ── Input card ────────────────────────────────────────────────────────────────
st.markdown('<div class="input-card"><span class="input-label">Your text input</span>', unsafe_allow_html=True)
user_input = st.text_area("", height=130, placeholder="E.g. This movie was absolutely incredible. A masterpiece.")
st.markdown('</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)
with col2:
    clear_btn = st.button("Clear", use_container_width=True)

# ── Analyze ───────────────────────────────────────────────────────────────────
if analyze_btn:
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            pred, conf, cleaned = predict(user_input)
        show_result(pred, conf)
        with st.expander("🔬 Preprocessed text"):
            st.code(cleaned, language=None)

# ── Examples ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Try an example</div>', unsafe_allow_html=True)

examples = [
    ("😊 Great movie",        "This movie was absolutely incredible. The performances were outstanding and the story gripped me from start to finish."),
    ("😞 Terrible service",   "Worst customer service I have ever experienced. Waited 3 hours, rude staff, and still got my order wrong."),
    ("😊 Loved the book",     "One of the best books I have read in years. Beautifully written and impossible to put down."),
    ("😞 Broken product",     "Complete waste of money. Stopped working after one day. Extremely disappointed with this purchase."),
]

col_a, col_b = st.columns(2)
for i, (label, text) in enumerate(examples):
    col = col_a if i % 2 == 0 else col_b
    with col:
        if st.button(label, key=f"ex_{i}", use_container_width=True):
            pred, conf, cleaned = predict(text)
            st.markdown(f'<div style="font-size:0.85rem; color:rgba(232,230,255,0.45); font-family:\'DM Mono\',monospace; margin:8px 0 4px; line-height:1.5;">"{text[:80]}..."</div>', unsafe_allow_html=True)
            show_result(pred, conf)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Model <span>LinearSVC</span> &nbsp;·&nbsp; Dataset <span>IMDB 50K</span> &nbsp;·&nbsp; Accuracy <span>~90%</span> &nbsp;·&nbsp; Built with <span>Python & Streamlit</span>
</div>
""", unsafe_allow_html=True)