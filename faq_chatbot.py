# ============================================================
#  FAQ Chatbot — CodeAlpha AI Internship Task 2
#  Platform : Google Colab + Streamlit
#  NLP      : sklearn TF-IDF + Cosine Similarity
#
#  ── COLAB CELL 1 (Install) ──────────────────────────────
#  !pip install streamlit scikit-learn pyngrok -q
#
#  ── COLAB CELL 2 (Run) ──────────────────────────────────
#  import subprocess, threading, time
#  from pyngrok import ngrok
#  subprocess.run(["pkill","-f","streamlit"], capture_output=True)
#  time.sleep(1)
#  def run():
#      subprocess.run(["streamlit","run","faq_chatbot.py",
#          "--server.port","8501","--server.headless","true"])
#  threading.Thread(target=run, daemon=True).start()
#  time.sleep(5)
#  url = ngrok.connect(8501)
#  print("👉 Open:", url)
# ============================================================

import streamlit as st
import string
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="CodeAlpha FAQ Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
}

.stApp {
    background: linear-gradient(145deg, #f0f7ff 0%, #e8f1fd 50%, #f5f9ff 100%);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.8rem !important; padding-bottom: 2rem !important; }

/* ── TITLE ── */
.app-title {
    text-align: center;
    font-size: 2.6em;
    font-weight: 800;
    background: linear-gradient(90deg, #1e40af, #2563eb, #1d4ed8, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 30px rgba(37,99,235,0.5));
    margin-bottom: 4px;
}
.app-subtitle {
    text-align: center;
    color: #2563eb;
    font-size: 0.88em;
    margin-bottom: 24px;
}

/* ── CHAT CONTAINER ── */
.chat-wrapper {
    background: rgba(255,255,255,0.9);
    border: 1.5px solid rgba(37,99,235,0.25);
    border-radius: 20px;
    padding: 20px;
    min-height: 420px;
    max-height: 460px;
    overflow-y: auto;
    margin-bottom: 16px;
}

/* ── BOT MESSAGE ── */
.msg-bot {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 18px;
}
.bot-avatar {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1em;
    flex-shrink: 0;
    box-shadow: 0 3px 12px rgba(37,99,235,0.4);
}
.bot-bubble {
    background: #ffffff;
    border-radius: 4px 18px 18px 18px;
    padding: 12px 18px;
    color: #000000;
    font-size: 0.95em;
    font-weight: 500;
    line-height: 1.7;
    max-width: 75%;
    box-shadow: 0 2px 12px rgba(0,0,0,0.15);
}
.bot-confidence {
    font-size: 0.72em;
    color: #2563eb;
    margin-top: 6px;
    font-weight: 600;
}

/* ── USER MESSAGE ── */
.msg-user {
    display: flex;
    align-items: flex-start;
    justify-content: flex-end;
    gap: 12px;
    margin-bottom: 18px;
}
.user-avatar {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #1e3a8a, #0c4a6e);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1em;
    flex-shrink: 0;
    box-shadow: 0 3px 12px rgba(3,105,161,0.4);
}
.user-bubble {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    border-radius: 18px 4px 18px 18px;
    padding: 12px 18px;
    color: #1e3a8a;
    font-size: 0.95em;
    font-weight: 500;
    line-height: 1.7;
    max-width: 75%;
    box-shadow: 0 2px 12px rgba(37,99,235,0.3);
}

/* ── TYPING INDICATOR ── */
.typing {
    display: flex; align-items: center; gap: 5px;
    padding: 10px 16px;
    background: #ffffff;
    border-radius: 4px 18px 18px 18px;
    width: fit-content;
    color: #64748b;
    font-size: 0.85em;
    font-style: italic;
}

/* ── INPUT BOX ── */
.stTextInput > div > div > input {
    background: #ffffff !important;
    border: 2px solid rgba(37,99,235,0.5) !important;
    border-radius: 14px !important;
    color: #000000 !important;
    font-size: 1em !important;
    font-weight: 500 !important;
    padding: 12px 18px !important;
    caret-color: #2563eb;
}
.stTextInput > div > div > input::placeholder {
    color: #94a3b8 !important;
    font-style: italic;
}
.stTextInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.25) !important;
}

/* ── SEND BUTTON ── */
button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: #1e3a8a !important;
    font-weight: 700 !important;
    font-size: 1em !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 12px 0 !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.4) !important;
}
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e3a8a) !important;
    transform: translateY(-1px) !important;
}

/* ── SECONDARY BUTTON ── */
button[kind="secondary"] {
    background: rgba(219,234,254,0.8) !important;
    color: #3b82f6 !important;
    border: 1.5px solid rgba(37,99,235,0.4) !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
}
button[kind="secondary"]:hover {
    background: rgba(37,99,235,0.25) !important;
    color: #1e3a8a !important;
}

/* ── OTHER SMALL BUTTONS ── */
.stButton > button:not([kind="primary"]):not([kind="secondary"]) {
    background: rgba(219,234,254,0.8) !important;
    color: #3b82f6 !important;
    border: 1.5px solid rgba(37,99,235,0.3) !important;
    border-radius: 10px !important;
    font-size: 0.82em !important;
    font-weight: 600 !important;
    padding: 6px 0 !important;
}
.stButton > button:not([kind="primary"]):not([kind="secondary"]):hover {
    background: rgba(37,99,235,0.25) !important;
    color: #1e3a8a !important;
}

/* ── QUICK QUESTION LABEL ── */
.sec-label {
    font-size: 0.73em;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #2563eb;
    margin-bottom: 8px;
}

/* ── SIDEBAR / INFO CARD ── */
.info-card {
    background: rgba(255,255,255,0.9);
    border: 1.5px solid rgba(37,99,235,0.25);
    border-radius: 16px;
    padding: 18px 20px;
    margin-bottom: 16px;
}
.info-title {
    color: #2563eb;
    font-size: 0.8em;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.info-item {
    color: #1e3a8a;
    font-size: 0.85em;
    padding: 5px 0;
    border-bottom: 1px solid rgba(59,130,246,0.1);
    font-weight: 400;
}
.info-item:last-child { border-bottom: none; }

/* ── DIVIDER ── */
.divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(37,99,235,0.4), transparent);
    margin: 16px 0;
}

/* ── METRICS ── */
[data-testid="metric-container"] {
    background: rgba(219,234,254,0.9);
    border: 1.5px solid rgba(37,99,235,0.25);
    border-radius: 14px;
    padding: 14px 18px;
    text-align: center;
}
[data-testid="metric-container"] label {
    color: #3b82f6 !important;
    font-size: 0.75em !important;
    font-weight: 700 !important;
    letter-spacing: 1px;
}
[data-testid="stMetricValue"] {
    color: #1e3a8a !important;
    font-size: 1.1em !important;
    font-weight: 700 !important;
}

/* ── FOOTER ── */
.footer {
    text-align: center;
    color: #2563eb;
    font-size: 0.78em;
    padding: 20px 0 6px;
    border-top: 1px solid rgba(59,130,246,0.15);
    margin-top: 14px;
}

/* scrollbar */
.chat-wrapper::-webkit-scrollbar { width: 5px; }
.chat-wrapper::-webkit-scrollbar-track { background: transparent; }
.chat-wrapper::-webkit-scrollbar-thumb { background: rgba(37,99,235,0.3); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── FAQ Database ─────────────────────────────────────────────
FAQ_DATA = [
    ("What is CodeAlpha?",
     "CodeAlpha is a leading software development company dedicated to driving innovation across emerging technologies including AI, web, and mobile development."),
    ("How do I apply for the internship?",
     "You can apply for the CodeAlpha internship by visiting our website at www.codealpha.tech and filling out the application form."),
    ("What are the internship tasks?",
     "The AI internship includes 4 tasks: Language Translation Tool, FAQ Chatbot, Music Generation with AI, and Object Detection & Tracking. You must complete at least 2 tasks."),
    ("Will I get a certificate?",
     "Yes! Upon completing at least 2 tasks, you will receive a QR-verified Completion Certificate along with a unique ID Certificate."),
    ("How do I submit my tasks?",
     "Upload your code to GitHub in a repository named CodeAlpha_ProjectName, post a LinkedIn video explanation, and submit via the form shared in your WhatsApp group."),
    ("How many tasks must I complete?",
     "You must complete a minimum of 2 tasks out of the 4 listed. Submitting only 1 task will not qualify for a certificate."),
    ("What is the contact email?",
     "You can reach CodeAlpha at services@codealpha.tech or services.codealpha@gmail.com."),
    ("What is the WhatsApp number?",
     "The CodeAlpha WhatsApp contact number is +91 9336576683."),
    ("Will I get a letter of recommendation?",
     "Yes! A Letter of Recommendation is provided based on your performance during the internship."),
    ("What skills will I learn?",
     "You will gain hands-on experience in AI model development, machine learning, NLP, computer vision, and real-time data processing."),
    ("Is the internship paid?",
     "The internship is skill-based. Benefits include certificates, recommendation letters, resume support, and job placement opportunities."),
    ("How long is the internship?",
     "The internship duration varies by cohort. Please check your offer letter or WhatsApp group for your specific timeline."),
    ("What technologies are used?",
     "The internship uses Python, TensorFlow, OpenCV, NLTK, scikit-learn, YOLO, LSTM/RNN, and various AI/ML libraries."),
    ("How do I post on LinkedIn?",
     "Share your internship progress on LinkedIn by tagging @CodeAlpha and posting a video explanation with your GitHub repository link."),
    ("What is object detection task?",
     "Task 4 involves real-time video input using OpenCV, YOLO or Faster R-CNN for object detection, and SORT or Deep SORT for tracking."),
    ("What is music generation task?",
     "Task 3 involves collecting MIDI data, training an LSTM model, and generating new music sequences saved as MIDI files."),
    ("What is the website?",
     "The official CodeAlpha website is www.codealpha.tech where you can find all internship details."),
    ("Can I get job placement support?",
     "Yes! CodeAlpha provides job opportunities and placement support to interns who perform well during the internship."),
    ("What is the GitHub repository name?",
     "You must name your GitHub repository as CodeAlpha_ProjectName, for example: CodeAlpha_LanguageTranslator."),
    ("Hello",
     "Hello! 👋 I am the CodeAlpha FAQ Assistant. Ask me anything about the internship program!"),
    ("Hi",
     "Hi there! 👋 Welcome to CodeAlpha. How can I assist you with the internship today?"),
    ("Thank you",
     "You're welcome! 😊 Feel free to ask if you have any more questions about the CodeAlpha internship."),
    ("Bye",
     "Goodbye! 👋 Best of luck with your CodeAlpha internship. Feel free to return anytime!"),
]

QUESTIONS = [q for q, _ in FAQ_DATA]
ANSWERS   = [a for _, a in FAQ_DATA]

# ── NLP Setup ────────────────────────────────────────────────
@st.cache_resource
def load_vectorizer():
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    mat = vec.fit_transform(QUESTIONS)
    return vec, mat

vectorizer, tfidf_matrix = load_vectorizer()

def get_answer(user_input: str):
    cleaned = user_input.lower().translate(str.maketrans("", "", string.punctuation)).strip()
    if not cleaned:
        return "Please type a question so I can help you! 😊", 0.0
    vec  = vectorizer.transform([cleaned])
    sims = cosine_similarity(vec, tfidf_matrix).flatten()
    idx  = sims.argmax()
    score = float(sims[idx])
    if score < 0.08:
        return (
            "I'm sorry, I couldn't find a relevant answer. 🤔\n"
            "Please contact CodeAlpha directly:\n"
            "📧 services@codealpha.tech\n"
            "📞 +91 9336576683\n"
            "🌐 www.codealpha.tech",
            0.0,
        )
    return ANSWERS[idx], score

# ── Session State ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "bot",
            "text": "👋 Hello! I'm the **CodeAlpha FAQ Assistant**.\n\nAsk me anything about the internship — tasks, certificates, submission, contact info, and more!",
            "confidence": None,
        }
    ]

# ════════════════════════════════════════════════════════════
#  UI
# ════════════════════════════════════════════════════════════

# Header
st.markdown("<div class='app-title'>🤖 CodeAlpha FAQ Chatbot</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='app-subtitle'>"
    "CodeAlpha AI Internship — Task 2 &nbsp;·&nbsp; "
    "Powered by TF-IDF + Cosine Similarity NLP"
    "</div>",
    unsafe_allow_html=True,
)

# ── Two column layout ────────────────────────────────────────
chat_col, info_col = st.columns([3, 1], gap="large")

# ── RIGHT: Info Panel ────────────────────────────────────────
with info_col:
    st.markdown("<div class='info-card'>"
                "<div class='info-title'>📌 About This Bot</div>"
                "<div class='info-item'>🧠 TF-IDF Vectorization</div>"
                "<div class='info-item'>📐 Cosine Similarity</div>"
                "<div class='info-item'>📚 20 FAQ Topics</div>"
                "<div class='info-item'>⚡ Real-time NLP</div>"
                "<div class='info-item'>🔒 No API Key Needed</div>"
                "</div>", unsafe_allow_html=True)

    st.markdown("<div class='info-card'>"
                "<div class='info-title'>📞 Contact</div>"
                "<div class='info-item'>🌐 www.codealpha.tech</div>"
                "<div class='info-item'>📧 services@codealpha.tech</div>"
                "<div class='info-item'>📱 +91 9336576683</div>"
                "</div>", unsafe_allow_html=True)

    # Stats
    st.metric("📚 FAQs", "20+")
    st.metric("🎯 Method", "TF-IDF")

# ── LEFT: Chat Panel ─────────────────────────────────────────
with chat_col:

    # Build chat HTML
    chat_html = "<div class='chat-wrapper' id='chat-box'>"
    for msg in st.session_state.messages:
        if msg["role"] == "bot":
            conf_html = ""
            if msg.get("confidence") and msg["confidence"] > 0:
                conf_html = f"<div class='bot-confidence'>🎯 Confidence: {msg['confidence']:.0%}</div>"
            text = msg["text"].replace("\n", "<br>").replace("**", "")
            chat_html += f"""
            <div class='msg-bot'>
                <div class='bot-avatar'>🤖</div>
                <div>
                    <div class='bot-bubble'>{text}{conf_html}</div>
                </div>
            </div>"""
        else:
            chat_html += f"""
            <div class='msg-user'>
                <div class='user-bubble'>{msg['text']}</div>
                <div class='user-avatar'>👤</div>
            </div>"""

    chat_html += "</div>"
    chat_html += "<script>var c=document.getElementById('chat-box');if(c)c.scrollTop=c.scrollHeight;</script>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # ── Quick Questions ──────────────────────────────────────
    st.markdown("<div class='sec-label'>💡 Quick Questions</div>", unsafe_allow_html=True)
    quick_qs = [
        "How to get certificate?",
        "How to submit tasks?",
        "How many tasks needed?",
        "Contact information?",
        "What skills will I learn?",
        "Is internship paid?",
    ]
    cols = st.columns(3)
    for i, q in enumerate(quick_qs):
        with cols[i % 3]:
            if st.button(q, use_container_width=True, key=f"quick_{i}"):
                st.session_state.messages.append({"role": "user", "text": q})
                answer, score = get_answer(q)
                st.session_state.messages.append({
                    "role": "bot", "text": answer, "confidence": score
                })
                st.rerun()

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Input Row ────────────────────────────────────────────
    inp_col, btn_col, clr_col = st.columns([6, 1, 1], gap="small")

    with inp_col:
        user_input = st.text_input(
            "msg", placeholder="Type your question here and press Enter or click Send...",
            label_visibility="collapsed", key="user_msg",
        )
    with btn_col:
        send = st.button("➤ Send", type="primary", use_container_width=True)
    with clr_col:
        clear = st.button("🗑️", type="secondary", use_container_width=True, help="Clear chat")

    # ── Send Logic ───────────────────────────────────────────
    if (send or user_input) and user_input.strip():
        q = user_input.strip()
        st.session_state.messages.append({"role": "user", "text": q})
        answer, score = get_answer(q)
        st.session_state.messages.append({
            "role": "bot", "text": answer, "confidence": score
        })
        st.rerun()

    # ── Clear Logic ──────────────────────────────────────────
    if clear:
        st.session_state.messages = [
            {
                "role": "bot",
                "text": "👋 Chat cleared! I'm the CodeAlpha FAQ Assistant. How can I help you?",
                "confidence": None,
            }
        ]
        st.rerun()

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    🤖 <b>CodeAlpha AI Internship</b> &nbsp;|&nbsp;
    Task 2: FAQ Chatbot &nbsp;|&nbsp;
    📧 services@codealpha.tech &nbsp;|&nbsp;
    🌐 www.codealpha.tech
</div>
""", unsafe_allow_html=True)
