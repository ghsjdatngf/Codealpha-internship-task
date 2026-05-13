# ============================================================
#  AI Smart Translator — CodeAlpha AI Internship Task 1
#  Platform : Google Colab + Streamlit
#
#  ── COLAB CELL 1 (Install) ──────────────────────────────
#  !pip install streamlit deep_translator gtts pyngrok -q
#
#  ── COLAB CELL 2 (Run) ──────────────────────────────────
#  import subprocess, threading, time
#  from pyngrok import ngrok
#  subprocess.run(["pkill","-f","streamlit"], capture_output=True)
#  time.sleep(1)
#  def run():
#      subprocess.run(["streamlit","run","translator_app.py",
#          "--server.port","8501","--server.headless","true"])
#  threading.Thread(target=run, daemon=True).start()
#  time.sleep(5)
#  url = ngrok.connect(8501)
#  print("Open:", url)
# ============================================================

import streamlit as st
from deep_translator import GoogleTranslator
from gtts import gTTS
import base64, io

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Smart Translator",
    page_icon="🌐",
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

/* Sky blue background */
.stApp {
    background: linear-gradient(145deg, #0c1e3d 0%, #0a3060 50%, #0d4a8a 100%);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; }

/* ── TITLE ── */
.app-title {
    text-align: center;
    font-size: 2.8em;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
    background: linear-gradient(90deg, #38bdf8, #7dd3fc, #bae6fd, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 30px rgba(56,189,248,0.5));
}
.app-subtitle {
    text-align: center;
    color: #7dd3fc;
    font-size: 0.9em;
    font-weight: 400;
    margin-bottom: 30px;
    letter-spacing: 0.3px;
}

/* ── SECTION LABEL ── */
.sec-label {
    font-size: 0.75em;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 7px;
}

/* ── SELECTBOX ── */
.stSelectbox > div > div {
    background: rgba(14,116,144,0.25) !important;
    border: 1.5px solid rgba(56,189,248,0.5) !important;
    border-radius: 14px !important;
    color: #000000 !important;
    font-size: 0.97em !important;
    font-weight: 600 !important;
}
.stSelectbox > div > div:hover {
    border-color: #38bdf8 !important;
}
/* Dropdown selected text black */
.stSelectbox > div > div > div {
    color: #000000 !important;
    font-weight: 600 !important;
}
.stSelectbox svg { fill: #38bdf8 !important; }

/* ── TEXT AREA ── */
.stTextArea > div > div > textarea {
    background: #ffffff !important;
    border: 2px solid rgba(56,189,248,0.5) !important;
    border-radius: 14px !important;
    color: #000000 !important;
    font-size: 1.05em !important;
    font-weight: 500 !important;
    line-height: 1.75 !important;
    padding: 14px 16px !important;
    caret-color: #0ea5e9;
}
.stTextArea > div > div > textarea::placeholder {
    color: #94a3b8 !important;
    font-style: italic;
}
.stTextArea > div > div > textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.2) !important;
}

/* ── OUTPUT BOX ── */
.output-box {
    background: #ffffff;
    border: 2px solid rgba(56,189,248,0.5);
    border-radius: 14px;
    padding: 16px 20px;
    min-height: 152px;
    color: #000000;
    font-size: 1.08em;
    font-weight: 500;
    line-height: 1.8;
    word-break: break-word;
    white-space: pre-wrap;
    font-family: 'Poppins', sans-serif;
}
.output-box.filled {
    border-color: #0ea5e9;
    box-shadow: 0 0 20px rgba(56,189,248,0.15);
}
.output-placeholder {
    color: #94a3b8;
    font-style: italic;
    font-size: 0.9em;
}

/* ── PRIMARY BUTTON (Translate) ── */
button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.05em !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 13px 0 !important;
    letter-spacing: 0.4px;
    box-shadow: 0 4px 24px rgba(14,165,233,0.45) !important;
    transition: all 0.25s ease !important;
}
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0284c7, #0369a1) !important;
    box-shadow: 0 6px 32px rgba(14,165,233,0.65) !important;
    transform: translateY(-2px) !important;
}

/* ── SECONDARY BUTTON (Clear) ── */
button[kind="secondary"] {
    background: rgba(14,116,144,0.2) !important;
    color: #38bdf8 !important;
    border: 1.5px solid rgba(56,189,248,0.4) !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    padding: 11px 0 !important;
}
button[kind="secondary"]:hover {
    background: rgba(56,189,248,0.2) !important;
    border-color: #38bdf8 !important;
    color: #bae6fd !important;
}

/* ── SMALL BUTTONS (TTS, etc) ── */
.stButton > button:not([kind="primary"]):not([kind="secondary"]) {
    background: rgba(14,116,144,0.2) !important;
    color: #38bdf8 !important;
    border: 1.5px solid rgba(56,189,248,0.35) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.9em !important;
    padding: 8px 0 !important;
}
.stButton > button:not([kind="primary"]):not([kind="secondary"]):hover {
    background: rgba(56,189,248,0.2) !important;
    border-color: #38bdf8 !important;
    color: #bae6fd !important;
}

/* ── COPY HTML BUTTON ── */
.copy-btn {
    margin-top: 10px;
    width: 100%;
    padding: 10px;
    border-radius: 12px;
    background: rgba(14,116,144,0.2);
    border: 1.5px solid rgba(56,189,248,0.35);
    color: #38bdf8;
    font-size: 0.92em;
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    cursor: pointer;
    letter-spacing: 0.3px;
    transition: all 0.2s;
}
.copy-btn:hover {
    background: rgba(56,189,248,0.2);
    border-color: #38bdf8;
    color: #bae6fd;
}

/* ── AUDIO PLAYER ── */
audio {
    width: 100%;
    height: 42px;
    border-radius: 12px;
    margin-top: 10px;
}

/* ── DIVIDER ── */
.divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.4), transparent);
    margin: 22px 0;
}

/* ── CHAR COUNT ── */
.char-count {
    text-align: right;
    color: #7dd3fc;
    font-size: 0.76em;
    margin-top: 5px;
}

/* ── METRICS ── */
[data-testid="metric-container"] {
    background: rgba(14,116,144,0.2);
    border: 1.5px solid rgba(56,189,248,0.3);
    border-radius: 14px;
    padding: 16px 20px;
    text-align: center;
}
[data-testid="metric-container"] label {
    color: #38bdf8 !important;
    font-size: 0.78em !important;
    font-weight: 700 !important;
    letter-spacing: 1px;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.1em !important;
    font-weight: 700 !important;
}

/* ── ALERTS ── */
.stAlert { border-radius: 12px !important; }
.stSpinner > div { border-top-color: #38bdf8 !important; }

/* ── FOOTER ── */
.footer {
    text-align: center;
    color: #7dd3fc;
    font-size: 0.8em;
    padding: 22px 0 8px;
    border-top: 1px solid rgba(56,189,248,0.2);
    margin-top: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── Language Data ────────────────────────────────────────────
LANGUAGES = {
    "Auto Detect"          : "auto",
    "English"              : "en",
    "Urdu"                 : "ur",
    "Arabic"               : "ar",
    "Hindi"                : "hi",
    "French"               : "fr",
    "German"               : "de",
    "Spanish"              : "es",
    "Turkish"              : "tr",
    "Russian"              : "ru",
    "Chinese (Simplified)" : "zh-CN",
    "Japanese"             : "ja",
    "Korean"               : "ko",
    "Italian"              : "it",
    "Portuguese"           : "pt",
    "Dutch"                : "nl",
    "Persian"              : "fa",
    "Bengali"              : "bn",
    "Punjabi"              : "pa",
    "Indonesian"           : "id",
    "Malay"                : "ms",
    "Swedish"              : "sv",
    "Norwegian"            : "no",
    "Danish"               : "da",
    "Greek"                : "el",
    "Polish"               : "pl",
    "Thai"                 : "th",
    "Vietnamese"           : "vi",
}

LANG_NAMES = list(LANGUAGES.keys())
GTTS_LANGS = {
    "en","ur","ar","hi","fr","de","es","tr","ru",
    "ja","ko","it","pt","nl","bn","id","sv","no",
    "da","el","pl","th","vi",
}

# ── Session State ────────────────────────────────────────────
for k, v in {
    "translated": "", "audio_in": "", "audio_out": "",
    "src_index": LANG_NAMES.index("English"),
    "tgt_index": LANG_NAMES.index("Urdu"),
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── TTS Helper ───────────────────────────────────────────────
def make_audio(text: str, lang_code: str) -> str:
    try:
        if lang_code in ("auto", "") or lang_code not in GTTS_LANGS:
            lang_code = "en"
        tts = gTTS(text=text, lang=lang_code, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception:
        return ""

def audio_player(b64: str) -> str:
    return (
        '<audio controls style="width:100%;border-radius:12px;margin-top:10px;">'
        f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    )

# ════════════════════════════════════════════════════════════
#  PAGE UI
# ════════════════════════════════════════════════════════════

# ── Header ──────────────────────────────────────────────────
st.markdown("<div class='app-title'> AI Smart Translator</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='app-subtitle'>"
    "CodeAlpha AI Internship — Task 1 &nbsp;·&nbsp; "
    "Google Translate Engine &nbsp;·&nbsp; 28 Languages &nbsp;·&nbsp; 🔊 Text-to-Speech"
    "</div>",
    unsafe_allow_html=True,
)

# ── Language Row ─────────────────────────────────────────────
c1, c2, c3 = st.columns([5, 1, 5])

with c1:
    st.markdown("<div class='sec-label'> Source Language</div>", unsafe_allow_html=True)
    src_lang = st.selectbox(
        "src", LANG_NAMES,
        index=st.session_state.src_index,
        label_visibility="collapsed", key="src_sel",
    )

with c2:
    st.markdown("<div style='height:29px'></div>", unsafe_allow_html=True)
    if st.button("⇄", use_container_width=True, help="Swap languages"):
        if src_lang != "Auto Detect":
            a, b = st.session_state.src_index, st.session_state.tgt_index
            st.session_state.src_index  = b
            st.session_state.tgt_index  = a
            st.session_state.translated = ""
            st.session_state.audio_in   = ""
            st.session_state.audio_out  = ""
            st.rerun()

with c3:
    st.markdown("<div class='sec-label'>🌍 Target Language</div>", unsafe_allow_html=True)
    tgt_lang = st.selectbox(
        "tgt", LANG_NAMES,
        index=st.session_state.tgt_index,
        label_visibility="collapsed", key="tgt_sel",
    )

st.session_state.src_index = LANG_NAMES.index(src_lang)
st.session_state.tgt_index = LANG_NAMES.index(tgt_lang)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Text Panels ──────────────────────────────────────────────
left, right = st.columns(2, gap="large")

# LEFT — Input
with left:
    st.markdown("<div class='sec-label'>✏️ Enter Text</div>", unsafe_allow_html=True)
    input_text = st.text_area(
        "inp", placeholder="Type or paste your text here...",
        height=200, label_visibility="collapsed", key="inp_box",
    )
    char_len = len(input_text) if input_text else 0
    c_color  = "#ef4444" if char_len > 5000 else "#7dd3fc"
    st.markdown(
        f"<div class='char-count' style='color:{c_color}'>{char_len} / 5000 characters</div>",
        unsafe_allow_html=True,
    )
    if input_text:
        if st.button("🔊  Listen to Input", use_container_width=True, key="tts_in"):
            with st.spinner("Generating audio..."):
                b64 = make_audio(input_text, LANGUAGES[src_lang])
            if b64:
                st.session_state.audio_in = b64
            else:
                st.warning(" Audio not supported for this language.")
    if st.session_state.audio_in and input_text:
        st.markdown(audio_player(st.session_state.audio_in), unsafe_allow_html=True)

# RIGHT — Output
with right:
    st.markdown("<div class='sec-label'> Translated Text</div>", unsafe_allow_html=True)
    translated = st.session_state.translated

    if translated:
        st.markdown(
            f"<div class='output-box filled'>{translated}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='char-count'>{len(translated)} characters</div>",
            unsafe_allow_html=True,
        )
        safe = translated.replace("`", "'").replace("\\", "\\\\")
        st.markdown(f"""
        <button class='copy-btn'
            onclick="navigator.clipboard.writeText(`{safe}`)
                .then(()=>{{ this.textContent=' Copied!';
                             setTimeout(()=>this.textContent='  Copy to Clipboard',2000); }})
                .catch(()=>this.textContent=' Failed')">
              Copy to Clipboard
        </button>""", unsafe_allow_html=True)

        if st.button("🔊  Listen to Translation", use_container_width=True, key="tts_out"):
            with st.spinner("Generating audio..."):
                b64 = make_audio(translated, LANGUAGES[tgt_lang])
            if b64:
                st.session_state.audio_out = b64
            else:
                st.warning(" Audio not supported for this language.")
        if st.session_state.audio_out:
            st.markdown(audio_player(st.session_state.audio_out), unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='output-box'>"
            "<span class='output-placeholder'>Translation will appear here...</span>"
            "</div>",
            unsafe_allow_html=True,
        )

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Translate / Clear Buttons ────────────────────────────────
b1, b2 = st.columns([4, 1], gap="medium")
with b1:
    translate_btn = st.button("  Translate Now", use_container_width=True, type="primary")
with b2:
    clear_btn = st.button("  Clear", use_container_width=True, type="secondary")

if clear_btn:
    st.session_state.translated = ""
    st.session_state.audio_in   = ""
    st.session_state.audio_out  = ""
    st.rerun()

if translate_btn:
    if not input_text or not input_text.strip():
        st.error("  Please enter some text before translating.")
    elif char_len > 5000:
        st.error("  Text too long. Keep it under 5000 characters.")
    elif src_lang == tgt_lang and src_lang != "Auto Detect":
        st.warning("  Source and Target languages cannot be the same.")
    else:
        with st.spinner("Translating... ✨"):
            try:
                result = GoogleTranslator(
                    source=LANGUAGES[src_lang],
                    target=LANGUAGES[tgt_lang],
                ).translate(input_text.strip())
                if result:
                    st.session_state.translated = result
                    st.session_state.audio_out  = ""
                    st.success(f" Translated: **{src_lang}** → **{tgt_lang}**")
                    st.rerun()
                else:
                    st.error(" Empty result. Please try again.")
            except Exception as e:
                st.error(f" Error: {str(e)}")

# ── Quick Examples ───────────────────────────────────────────
st.markdown(
    "<div class='sec-label' style='margin-top:8px;'> Quick Examples</div>",
    unsafe_allow_html=True,
)
examples = [
    ("Hello! How are you?",       "English",      "Urdu"),
    ("AI is future of tech.",     "English",      "Arabic"),
    ("Bonjour! Comment ça va?",   "French",       "English"),
    ("میں ٹھیک ہوں، شکریہ۔",    "Auto Detect",  "English"),
]
for col, (txt, src, tgt) in zip(st.columns(4), examples):
    with col:
        if st.button(f" {txt[:18]}…", use_container_width=True, key=f"ex_{txt[:5]}"):
            st.session_state.src_index  = LANG_NAMES.index(src)
            st.session_state.tgt_index  = LANG_NAMES.index(tgt)
            st.session_state.translated = ""
            st.session_state.audio_in   = ""
            st.session_state.audio_out  = ""
            st.rerun()

# ── Stats ────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
for col, (label, val) in zip(
    st.columns(4),
    [(" Languages","28+"), (" Engine","Google Translate"),
     (" Text-to-Speech","Audio Playback"), (" Copy","One Click")],
):
    col.metric(label, val)

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
     <b>CodeAlpha AI Internship</b> &nbsp;|&nbsp;
    Task 1: AI Smart Translator &nbsp;|&nbsp;
     services@codealpha.tech &nbsp;|&nbsp;
     www.codealpha.tech
</div>
""", unsafe_allow_html=True)
