import streamlit as st
import pickle
import string
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spam Detection Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.15rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1.8rem;
    }

    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        font-size: 0.95rem;
    }

    div.stButton > button {
        background-color: #0F172A;
        color: white;
        border-radius: 8px;
        padding: 0.55rem 1.6rem;
        font-weight: 600;
        border: none;
        transition: background-color 0.15s ease;
    }
    div.stButton > button:hover {
        background-color: #1E293B;
        color: white;
    }

    .result-card {
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-top: 1rem;
        border: 1px solid;
    }
    .result-spam {
        background-color: #FEF2F2;
        border-color: #FCA5A5;
    }
    .result-ham {
        background-color: #F0FDF4;
        border-color: #86EFAC;
    }
    .result-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .result-spam .result-title { color: #B91C1C; }
    .result-ham .result-title { color: #15803D; }

    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .footer-note {
        color: #94A3B8;
        font-size: 0.8rem;
        margin-top: 3rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return tfidf, model

try:
    tfidf, model = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
STOPWORDS = set([
    'a', 'an', 'the', 'is', 'it', 'this', 'that', 'in', 'on', 'for', 'to',
    'of', 'and', 'or', 'if', 'are', 'as', 'at', 'be', 'by', 'from', 'has',
    'he', 'she', 'i', 'you', 'we', 'they', 'them', 'with', 'was', 'but'
])

SUFFIXES = ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']


def simple_stem(word: str) -> str:
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word


def transform_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]
    words = [simple_stem(w) for w in words]
    return ' '.join(words)


# ---------------------------------------------------------------------------
# Session state for history
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ Spam Detection Engine")
    st.caption("TF-IDF + Machine Learning classifier")

    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown(
        "1. Lowercase & strip punctuation\n"
        "2. Tokenize\n"
        "3. Remove stopwords\n"
        "4. Suffix-based stemming\n"
        "5. TF-IDF vectorization\n"
        "6. Model inference"
    )

    st.markdown("---")
    st.markdown("**Status**")
    if artifacts_loaded:
        st.success("Model artifacts loaded")
    else:
        st.error("vectorizer.pkl / model.pkl not found")

    st.markdown("---")
    st.markdown("**Session stats**")
    total = len(st.session_state.history)
    spam_count = sum(1 for h in st.session_state.history if h["label"] == "Spam")
    st.metric("Messages checked", total)
    st.metric("Flagged as spam", spam_count)

    if st.session_state.history:
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">Spam Detection Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Classify email or SMS content as spam or legitimate using a '
    'TF-IDF vectorizer and trained ML model.</div>',
    unsafe_allow_html=True,
)

col_input, col_info = st.columns([2.2, 1])

with col_input:
    input_sms = st.text_area(
        "Message content",
        height=160,
        placeholder="Paste an email or SMS message here...",
        label_visibility="collapsed",
    )

    c1, c2 = st.columns([1, 3])
    with c1:
        predict_clicked = st.button("Analyze Message", type="primary", use_container_width=True)
    with c2:
        st.caption(f"{len(input_sms)} characters · {len(input_sms.split())} words")

with col_info:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.markdown('<div class="metric-value">TF-IDF</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-label">Feature Extraction</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if predict_clicked:
    if not artifacts_loaded:
        st.error("Model files are missing. Place vectorizer.pkl and model.pkl in the app directory.")
    elif input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        with st.spinner("Analyzing message..."):
            time.sleep(0.3)
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

            confidence = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vector_input)[0]
                confidence = max(proba) * 100

        label = "Spam" if result == 1 else "Not Spam"
        css_class = "result-spam" if result == 1 else "result-ham"
        icon = "🚫" if result == 1 else "✅"

        st.markdown(f"""
        <div class="result-card {css_class}">
            <div class="result-title">{icon} {label}</div>
            <div style="color:#475569; font-size:0.9rem;">
                {"This message shows patterns consistent with spam content." if result == 1
                 else "This message appears to be legitimate."}
            </div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{label}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Prediction</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with m2:
            conf_display = f"{confidence:.1f}%" if confidence is not None else "N/A"
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{conf_display}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Confidence</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{len(input_sms.split())}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Word Count</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("View preprocessed text"):
            st.code(transformed_sms if transformed_sms else "(empty after preprocessing)")

        st.session_state.history.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": input_sms[:60] + ("..." if len(input_sms) > 60 else ""),
            "label": label,
            "confidence": f"{confidence:.1f}%" if confidence is not None else "N/A",
        })

# ---------------------------------------------------------------------------
# History table
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.markdown("### Recent Predictions")
    st.dataframe(
        st.session_state.history,
        use_container_width=True,
        hide_index=True,
        column_config={
            "time": "Time",
            "message": "Message Preview",
            "label": "Prediction",
            "confidence": "Confidence",
        },
    )

st.markdown(
    '<div class="footer-note">Spam Detection Engine · TF-IDF + ML Classifier · '
    'For demonstration purposes only</div>',
    unsafe_allow_html=True,
)
