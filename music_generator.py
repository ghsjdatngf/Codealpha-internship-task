# ============================================================
#  Music Generation with AI — CodeAlpha AI Internship Task 3
#  Platform : Google Colab + Streamlit
#  Model    : LSTM (TensorFlow/Keras) + music21
#
#  ── COLAB CELL 1 (Install) ──────────────────────────────
#  !pip install streamlit music21 tensorflow pyngrok -q
#
#  ── COLAB CELL 2 (Run) ──────────────────────────────────
#  import subprocess, threading, time
#  from pyngrok import ngrok
#  subprocess.run(["pkill","-f","streamlit"], capture_output=True)
#  time.sleep(1)
#  def run():
#      subprocess.run(["streamlit","run","music_generator.py",
#          "--server.port","8501","--server.headless","true"])
#  threading.Thread(target=run, daemon=True).start()
#  time.sleep(5)
#  url = ngrok.connect(8501)
#  print("👉 Open:", url)
# ============================================================

import streamlit as st
import numpy as np
import random
import base64
import io
import os
import time

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Music Generator",
    page_icon="🎵",
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
    margin-bottom: 26px;
}

/* ── SECTION LABEL ── */
.sec-label {
    font-size: 0.73em;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #2563eb;
    margin-bottom: 8px;
}

/* ── CARDS ── */
.card {
    background: rgba(255,255,255,0.9);
    border: 1.5px solid rgba(37,99,235,0.25);
    border-radius: 18px;
    padding: 22px 24px;
    margin-bottom: 16px;
}

/* ── NOTE DISPLAY ── */
.note-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    padding: 16px;
    background: #ffffff;
    border-radius: 14px;
    border: 2px solid rgba(37,99,235,0.4);
    min-height: 80px;
}
.note-pill {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: #1e3a8a;
    font-size: 0.75em;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
    font-family: 'Poppins', sans-serif;
}
.note-pill.rest {
    background: linear-gradient(135deg, #64748b, #475569);
}
.note-pill.long {
    background: linear-gradient(135deg, #1e3a8a, #075985);
}

/* ── PIANO ROLL ── */
.piano-roll {
    background: #000000;
    border-radius: 14px;
    border: 2px solid rgba(37,99,235,0.3);
    padding: 14px;
    overflow-x: auto;
}
.piano-bar {
    display: inline-block;
    background: linear-gradient(180deg, #3b82f6, #1d4ed8);
    border-radius: 3px;
    margin: 1px;
    vertical-align: bottom;
    min-width: 12px;
}

/* ── SLIDERS ── */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #2563eb, #3b82f6) !important;
}
.stSlider > div > div > div {
    background: rgba(37,99,235,0.25) !important;
}

/* ── SELECTBOX ── */
.stSelectbox > div > div {
    background: rgba(191,219,254,0.9) !important;
    border: 1.5px solid rgba(37,99,235,0.5) !important;
    border-radius: 14px !important;
    color: #000000 !important;
    font-size: 0.95em !important;
    font-weight: 600 !important;
}
.stSelectbox > div > div > div { color: #000000 !important; font-weight: 600 !important; }
.stSelectbox svg { fill: #3b82f6 !important; }

/* ── PRIMARY BUTTON ── */
button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: #1e3a8a !important;
    font-weight: 700 !important;
    font-size: 1.05em !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 13px 0 !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.4) !important;
}
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e3a8a) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px rgba(37,99,235,0.6) !important;
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

/* ── OTHER BUTTONS ── */
.stButton > button:not([kind="primary"]):not([kind="secondary"]) {
    background: rgba(219,234,254,0.8) !important;
    color: #3b82f6 !important;
    border: 1.5px solid rgba(37,99,235,0.3) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 0.9em !important;
}
.stButton > button:not([kind="primary"]):not([kind="secondary"]):hover {
    background: rgba(37,99,235,0.25) !important;
    color: #1e3a8a !important;
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

/* ── PROGRESS ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #2563eb, #3b82f6) !important;
    border-radius: 10px !important;
}

/* ── AUDIO ── */
audio {
    width: 100%;
    border-radius: 12px;
    margin-top: 8px;
}

/* ── DIVIDER ── */
.divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(37,99,235,0.4), transparent);
    margin: 18px 0;
}

/* ── DOWNLOAD BUTTON ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    border: none !important;
    width: 100% !important;
    padding: 12px !important;
    font-size: 1em !important;
    box-shadow: 0 4px 18px rgba(37,99,235,0.35) !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e3a8a) !important;
    transform: translateY(-1px) !important;
}

/* ── ALERT ── */
.stAlert { border-radius: 12px !important; }
.stSpinner > div { border-top-color: #2563eb !important; }

/* ── FOOTER ── */
.footer {
    text-align: center;
    color: #2563eb;
    font-size: 0.78em;
    padding: 20px 0 6px;
    border-top: 1px solid rgba(59,130,246,0.15);
    margin-top: 14px;
}

/* ── INFO BOX ── */
.info-box {
    background: rgba(219,234,254,0.8);
    border: 1.5px solid rgba(37,99,235,0.25);
    border-radius: 14px;
    padding: 16px 20px;
    color: #1e3a8a;
    font-size: 0.85em;
    line-height: 1.7;
}
.info-box b { color: #2563eb; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  MUSIC GENERATION ENGINE
# ════════════════════════════════════════════════════════════

# ── Music Theory Data ────────────────────────────────────────
SCALES = {
    "C Major"      : ["C4","D4","E4","F4","G4","A4","B4","C5","D5","E5","G5"],
    "G Major"      : ["G3","A3","B3","C4","D4","E4","F#4","G4","A4","B4","D5"],
    "D Minor"      : ["D4","E4","F4","G4","A4","Bb4","C5","D5","E5","F5","A5"],
    "A Minor"      : ["A3","B3","C4","D4","E4","F4","G4","A4","B4","C5","E5"],
    "F Major"      : ["F3","G3","A3","Bb3","C4","D4","E4","F4","G4","A4","C5"],
    "Pentatonic"   : ["C4","D4","E4","G4","A4","C5","D5","E5","G5","A5","C6"],
    "Blues"        : ["C4","Eb4","F4","F#4","G4","Bb4","C5","Eb5","F5","G5","Bb5"],
    "Chromatic"    : ["C4","C#4","D4","D#4","E4","F4","F#4","G4","G#4","A4","A#4","B4"],
}

STYLES = {
    "Classical"  : {"dur_weights": [0.05,0.15,0.35,0.25,0.15,0.05], "rest_prob": 0.05, "repeat": 0.3},
    "Jazz"       : {"dur_weights": [0.15,0.25,0.25,0.20,0.10,0.05], "rest_prob": 0.10, "repeat": 0.2},
    "Pop"        : {"dur_weights": [0.10,0.20,0.40,0.20,0.08,0.02], "rest_prob": 0.05, "repeat": 0.4},
    "Ambient"    : {"dur_weights": [0.02,0.05,0.15,0.25,0.30,0.23], "rest_prob": 0.15, "repeat": 0.25},
    "Waltz"      : {"dur_weights": [0.05,0.10,0.50,0.20,0.10,0.05], "rest_prob": 0.05, "repeat": 0.35},
    "Electronic" : {"dur_weights": [0.20,0.35,0.25,0.15,0.04,0.01], "rest_prob": 0.08, "repeat": 0.3},
}

DURATIONS = [0.25, 0.5, 1.0, 1.5, 2.0, 4.0]

# ── LSTM Model (TensorFlow) ──────────────────────────────────
def try_lstm_generate(scale_notes, style_cfg, n_notes, temperature, seq_len=12):
    """Try LSTM-based generation. Falls back gracefully if TF unavailable."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        N = len(scale_notes)
        # Build tiny training sequences from scale patterns
        seqs, targets = [], []
        pattern = scale_notes * 6   # repeat scale 6x for training data
        for i in range(len(pattern) - seq_len):
            seqs.append([scale_notes.index(n) if n in scale_notes else 0
                         for n in pattern[i:i+seq_len]])
            targets.append(scale_notes.index(pattern[i+seq_len])
                           if pattern[i+seq_len] in scale_notes else 0)

        if len(seqs) < 5:
            return None

        X = np.array(seqs, dtype="float32") / N
        y = tf.keras.utils.to_categorical(targets, num_classes=N)
        X = X.reshape(X.shape[0], seq_len, 1)

        model = Sequential([
            LSTM(64, input_shape=(seq_len, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(N, activation="softmax"),
        ])
        model.compile(loss="categorical_crossentropy", optimizer="adam")
        model.fit(X, y, epochs=30, batch_size=8, verbose=0)

        # Generate
        seed = list(range(min(seq_len, N)))
        result = []
        for _ in range(n_notes):
            inp = np.array(seed[-seq_len:], dtype="float32") / N
            inp = inp.reshape(1, seq_len, 1)
            pred = model.predict(inp, verbose=0)[0]
            # Temperature sampling
            pred = np.log(pred + 1e-8) / temperature
            pred = np.exp(pred) / np.sum(np.exp(pred))
            idx  = np.random.choice(N, p=pred)
            result.append(scale_notes[idx])
            seed.append(idx)

        return result

    except Exception:
        return None


def markov_generate(scale_notes, style_cfg, n_notes, temperature):
    """Markov chain music generator — always works."""
    N = len(scale_notes)

    # Build transition matrix with temperature influence
    trans = {}
    for i, note in enumerate(scale_notes):
        neighbors = []
        for j in range(-2, 3):
            idx = (i + j) % N
            weight = max(0.1, 1.0 - abs(j) * 0.25 * temperature)
            neighbors.extend([scale_notes[idx]] * int(weight * 10))
        trans[note] = neighbors

    current = random.choice(scale_notes)
    notes   = [current]
    repeat_prob = style_cfg["repeat"]

    for _ in range(n_notes - 1):
        # Occasional repetition for musical coherence
        if random.random() < repeat_prob and len(notes) >= 2:
            current = notes[-2]
        else:
            current = random.choice(trans[current])
        notes.append(current)

    return notes


def generate_durations(n, style_cfg):
    dur_weights = style_cfg["dur_weights"]
    rest_prob   = style_cfg["rest_prob"]
    result = []
    for _ in range(n):
        if random.random() < rest_prob:
            result.append(("REST", random.choice([0.5, 1.0])))
        else:
            d = random.choices(DURATIONS, weights=dur_weights)[0]
            result.append(("NOTE", d))
    return result


def generate_music_sequence(scale_name, style_name, n_notes, bpm, temperature, use_lstm):
    """Main generation function — returns (notes, dur_types, durations)."""
    scale_notes = SCALES[scale_name]
    style_cfg   = STYLES[style_name]

    notes = None
    method_used = "Markov Chain"

    if use_lstm:
        with st.spinner("🧠 Training LSTM model... (may take 30-60 seconds)"):
            notes = try_lstm_generate(scale_notes, style_cfg, n_notes, temperature)
            if notes:
                method_used = "LSTM Neural Network"

    if notes is None:
        notes = markov_generate(scale_notes, style_cfg, n_notes, temperature)

    dur_info  = generate_durations(n_notes, style_cfg)
    dur_types = [d[0] for d in dur_info]
    durations  = [d[1] for d in dur_info]

    # Merge rests into notes list
    final_notes = []
    for n, (dtype, _) in zip(notes, dur_info):
        final_notes.append("REST" if dtype == "REST" else n)

    return final_notes, durations, method_used


def build_midi_bytes(notes, durations, bpm=120):
    """Build MIDI file bytes using music21."""
    try:
        from music21 import stream, note, tempo, instrument, chord

        s = stream.Stream()
        s.append(tempo.MetronomeMark(number=bpm))
        s.append(instrument.Piano())

        for n_str, dur in zip(notes, durations):
            if n_str == "REST":
                r = note.Rest()
                r.quarterLength = dur
                s.append(r)
            else:
                try:
                    n = note.Note(n_str)
                    n.quarterLength = dur
                    s.append(n)
                except Exception:
                    pass

        # Write to temp file then read bytes
        tmp_path = "/tmp/generated_music.mid"
        s.write("midi", fp=tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()

    except Exception as e:
        return None


def piano_roll_html(notes, durations):
    """Generate visual piano roll HTML."""
    NOTE_HEIGHTS = {
        "C": 1, "D": 2, "E": 3, "F": 4,
        "G": 5, "A": 6, "B": 7,
    }
    bars = ""
    for n, d in zip(notes[:60], durations[:60]):
        if n == "REST":
            h = 4
            color = "rgba(100,116,139,0.5)"
        else:
            root = n[0] if n else "C"
            octave = int(n[-1]) if n and n[-1].isdigit() else 4
            h = NOTE_HEIGHTS.get(root, 4) * 6 + (octave - 3) * 10
            h = max(8, min(h, 70))
            color = f"linear-gradient(180deg,#3b82f6,#1d4ed8)"
        w = max(10, int(d * 18))
        bars += (
            f'<div style="display:inline-block;width:{w}px;height:{h}px;'
            f'background:{color};border-radius:3px;margin:1px;'
            f'vertical-align:bottom;"></div>'
        )
    return f"""
    <div style="background:#000;border-radius:14px;border:2px solid rgba(37,99,235,0.3);
                padding:14px 12px 8px;overflow-x:auto;white-space:nowrap;">
        <div style="font-size:0.68em;color:#3b82f6;font-weight:700;
                    letter-spacing:2px;margin-bottom:10px;">
            🎹 PIANO ROLL VISUALIZATION
        </div>
        <div style="display:flex;align-items:flex-end;gap:2px;min-height:80px;">
            {bars}
        </div>
        <div style="font-size:0.65em;color:#475569;margin-top:8px;text-align:right;">
            Showing first 60 notes
        </div>
    </div>"""


# ════════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════════
for k, v in {
    "gen_notes": [], "gen_durs": [], "midi_bytes": None,
    "method_used": "", "generated": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ════════════════════════════════════════════════════════════
#  UI
# ════════════════════════════════════════════════════════════

st.markdown("<div class='app-title'>🎵 AI Music Generator</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='app-subtitle'>"
    "CodeAlpha AI Internship — Task 3 &nbsp;·&nbsp; "
    "LSTM Neural Network + Markov Chain &nbsp;·&nbsp; MIDI Generation"
    "</div>",
    unsafe_allow_html=True,
)

# ── Two column layout ────────────────────────────────────────
ctrl_col, out_col = st.columns([1, 2], gap="large")

# ── LEFT: Controls ───────────────────────────────────────────
with ctrl_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>🎼 Music Settings</div>", unsafe_allow_html=True)

    scale_name = st.selectbox(
        "🎵 Scale / Key",
        list(SCALES.keys()),
        index=0,
    )

    style_name = st.selectbox(
        "🎨 Music Style",
        list(STYLES.keys()),
        index=0,
    )

    bpm = st.slider("⏱️ Tempo (BPM)", 60, 200, 120, step=5)

    n_notes = st.slider("🎹 Number of Notes", 20, 150, 60, step=10)

    temperature = st.slider(
        "🌡️ Creativity (Temperature)",
        0.1, 2.0, 0.8, step=0.1,
        help="Low = predictable melody | High = more random/creative"
    )

    use_lstm = st.toggle(
        "🧠 Use LSTM Model",
        value=False,
        help="ON = LSTM Neural Network (slower, needs TensorFlow)\nOFF = Markov Chain (fast, always works)"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # Info box
    st.markdown(f"""
    <div class='info-box'>
        <b>Scale:</b> {scale_name}<br>
        <b>Style:</b> {style_name}<br>
        <b>BPM:</b> {bpm} &nbsp;|&nbsp; <b>Notes:</b> {n_notes}<br>
        <b>Creativity:</b> {temperature}<br>
        <b>Model:</b> {"🧠 LSTM" if use_lstm else "🔗 Markov Chain"}
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    generate_btn = st.button(
        "🎵  Generate Music", type="primary", use_container_width=True
    )

# ── RIGHT: Output ─────────────────────────────────────────────
with out_col:

    if generate_btn:
        with st.spinner("🎶 Composing your music..."):
            notes, durs, method = generate_music_sequence(
                scale_name, style_name, n_notes, bpm, temperature, use_lstm
            )
            midi_bytes = build_midi_bytes(notes, durs, bpm)

        st.session_state.gen_notes   = notes
        st.session_state.gen_durs    = durs
        st.session_state.midi_bytes  = midi_bytes
        st.session_state.method_used = method
        st.session_state.generated   = True
        st.success(f"✅ Music generated using **{method}** — {len(notes)} notes composed!")

    if st.session_state.generated:
        notes  = st.session_state.gen_notes
        durs   = st.session_state.gen_durs

        # ── Stats row ────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎵 Notes",    len(notes))
        m2.metric("⏱️ BPM",      bpm)
        m3.metric("⏳ Duration",  f"{sum(durs)/2:.0f}s")
        m4.metric("🧠 Model",    st.session_state.method_used.split()[0])

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ── Piano Roll ───────────────────────────────────────
        st.markdown("<div class='sec-label'>🎹 Piano Roll</div>", unsafe_allow_html=True)
        st.markdown(piano_roll_html(notes, durs), unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ── Generated Notes ──────────────────────────────────
        st.markdown("<div class='sec-label'>🎼 Generated Note Sequence</div>", unsafe_allow_html=True)

        note_pills = ""
        for n, d in zip(notes[:40], durs[:40]):
            if n == "REST":
                note_pills += f"<span class='note-pill rest'>REST</span>"
            elif d >= 2.0:
                note_pills += f"<span class='note-pill long'>{n}</span>"
            else:
                note_pills += f"<span class='note-pill'>{n}</span>"
        if len(notes) > 40:
            note_pills += f"<span class='note-pill' style='background:#475569'>+{len(notes)-40} more</span>"

        st.markdown(
            f"<div class='note-grid'>{note_pills}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ── Download / Audio ─────────────────────────────────
        dl_col, info_col2 = st.columns([1, 1], gap="medium")

        with dl_col:
            st.markdown("<div class='sec-label'>💾 Download MIDI</div>", unsafe_allow_html=True)
            if st.session_state.midi_bytes:
                st.download_button(
                    label="⬇️  Download generated_music.mid",
                    data=st.session_state.midi_bytes,
                    file_name=f"codealpha_{style_name.lower()}_{scale_name.replace(' ','_')}.mid",
                    mime="audio/midi",
                    use_container_width=True,
                )
                st.markdown("""
                <div style='color:#93c5fd;font-size:0.78em;margin-top:8px;line-height:1.6;'>
                    🎧 Open the MIDI file with:<br>
                    &nbsp;&nbsp;• <b>VLC Media Player</b><br>
                    &nbsp;&nbsp;• <b>MuseScore</b> (view sheet music)<br>
                    &nbsp;&nbsp;• <b>GarageBand</b> (Mac/iOS)<br>
                    &nbsp;&nbsp;• <b>FL Studio / LMMS</b> (DAW)
                </div>""", unsafe_allow_html=True)
            else:
                st.warning("⚠️ MIDI generation requires music21.\nInstall: `pip install music21`")

        with info_col2:
            st.markdown("<div class='sec-label'>📊 Music Info</div>", unsafe_allow_html=True)
            rest_count = notes.count("REST")
            unique_notes = len(set(n for n in notes if n != "REST"))
            avg_dur = np.mean(durs)

            st.markdown(f"""
            <div class='info-box'>
                <b>Total Notes:</b> {len(notes)}<br>
                <b>Rests:</b> {rest_count}<br>
                <b>Unique Pitches:</b> {unique_notes}<br>
                <b>Avg Duration:</b> {avg_dur:.2f} beats<br>
                <b>Approx Duration:</b> {sum(durs)*60/bpm:.1f} seconds<br>
                <b>Scale:</b> {scale_name}<br>
                <b>Style:</b> {style_name}<br>
                <b>Generation:</b> {st.session_state.method_used}
            </div>""", unsafe_allow_html=True)

    else:
        # Empty state
        st.markdown("""
        <div style='text-align:center;padding:80px 20px;'>
            <div style='font-size:4em;margin-bottom:16px;'>🎵</div>
            <div style='color:#3b82f6;font-size:1.2em;font-weight:700;margin-bottom:8px;'>
                Ready to Compose!
            </div>
            <div style='color:#93c5fd;font-size:0.9em;line-height:1.7;'>
                Select your scale, style, and settings<br>
                then click <b>Generate Music</b> to create<br>
                an original AI-composed melody 🎶
            </div>
        </div>""", unsafe_allow_html=True)

# ── How It Works ─────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("<div class='sec-label'>⚙️ How It Works</div>", unsafe_allow_html=True)

h1, h2, h3, h4 = st.columns(4)
for col, icon, title, desc in zip(
    [h1, h2, h3, h4],
    ["🎼", "🧠", "🎹", "💾"],
    ["Music Data", "AI Model", "Generate", "Export"],
    [
        "Scale notes & music theory patterns feed the model",
        "LSTM Neural Net or Markov Chain learns note patterns",
        "Temperature sampling creates creative note sequences",
        "Output saved as MIDI — play in any music software",
    ],
):
    with col:
        st.markdown(f"""
        <div class='card' style='text-align:center;'>
            <div style='font-size:2em;margin-bottom:8px;'>{icon}</div>
            <div style='color:#3b82f6;font-weight:700;font-size:0.85em;
                        margin-bottom:6px;'>{title}</div>
            <div style='color:#ffffff;font-size:0.78em;line-height:1.6;'>{desc}</div>
        </div>""", unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    🤖 <b>CodeAlpha AI Internship</b> &nbsp;|&nbsp;
    Task 3: AI Music Generator &nbsp;|&nbsp;
    📧 services@codealpha.tech &nbsp;|&nbsp;
    🌐 www.codealpha.tech
</div>
""", unsafe_allow_html=True)
