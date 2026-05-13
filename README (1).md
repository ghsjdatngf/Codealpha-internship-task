# CodeAlpha AI Internship — Task Documentation

![CodeAlpha](https://img.shields.io/badge/CodeAlpha-AI%20Internship-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red?style=for-the-badge&logo=streamlit)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange?style=for-the-badge&logo=googlecolab)

---

## Intern Information

| Field        | Details                 |
| ------------ | ----------------------- |
| **Name**     | [Your Name]             |
| **Domain**   | Artificial Intelligence |
| **Company**  | CodeAlpha               |
| **Website**  | www.codealpha.tech      |
| **Email**    | services@codealpha.tech |
| **WhatsApp** | +91 9336576683          |

---

## Project Structure

```
CodeAlpha_AI_Tasks/
│
├── README.md
│
├──  Task1_Language_Translation/
│   ├── translator_app.py
│   └── requirements.txt
│
├──  Task2_FAQ_Chatbot/
│   ├── faq_chatbot.py
│   └── requirements.txt
│
└──  Task3_Music_Generation/
    ├── music_generator.py
    └── requirements.txt
```

---

## Task 1 — Language Translation Tool

### Overview

A professional AI-powered language translation web application built with Streamlit. It uses Google Translate API via `deep_translator` to translate text between 28 languages in real time.

### Technologies Used

| Library           | Purpose                         |
| ----------------- | ------------------------------- |
| `streamlit`       | Web UI framework                |
| `deep_translator` | Google Translate API wrapper    |
| `gtts`            | Text-to-Speech audio generation |
| `pyngrok`         | Public URL for Google Colab     |

### Features

**28 Languages** — Urdu, Arabic, Hindi, French, German, Spanish and more
**Auto Detect** source language automatically
**Swap Button** to switch languages instantly
**Text-to-Speech** for both input and translated text
**Copy to Clipboard** one-click copy button
**Error Handling** for empty input, API errors, character limit
**Quick Examples** to test the app instantly
**Character Counter** with 5000 character limit

### How to Run on Google Colab

**Cell 1 — Install Libraries:**

```python
!pip install streamlit deep_translator gtts pyngrok -q
```

**Cell 2 — Upload File:**

> Upload `translator_app.py` using the Files panel (left sidebar) in Colab

**Cell 3 — Launch App:**

```python
import subprocess, threading, time
from pyngrok import ngrok

subprocess.run(["pkill", "-f", "streamlit"], capture_output=True)
time.sleep(1)

def run():
    subprocess.run(["streamlit", "run", "translator_app.py",
        "--server.port", "8501", "--server.headless", "true"])

threading.Thread(target=run, daemon=True).start()
time.sleep(5)

url = ngrok.connect(8501)
print(" Open this link:", url)
```

### How It Works

```
User enters text
        ↓
Select Source & Target Language
        ↓
GoogleTranslator API processes text
        ↓
Translated text displayed on screen
        ↓
Optional: Listen (TTS) or Copy to Clipboard
```

---

## Task 2 — FAQ Chatbot

### Overview

An intelligent FAQ chatbot powered by NLP techniques. It uses TF-IDF vectorization and Cosine Similarity to match user questions with the most relevant FAQ answer from a database of 20+ CodeAlpha internship questions.

### Technologies Used

| Library        | Purpose                               |
| -------------- | ------------------------------------- |
| `streamlit`    | Web UI / Chat interface               |
| `scikit-learn` | TF-IDF Vectorizer + Cosine Similarity |
| `pyngrok`      | Public URL for Google Colab           |

### Features

**Real Chat UI** with bot and user message bubbles
**NLP Matching** using TF-IDF + Cosine Similarity
**Confidence Score** shown for every answer
**6 Quick Question** shortcut buttons
**Clear Chat** button to reset conversation
**Info Panel** showing bot details and contact info
**20+ FAQ topics** about CodeAlpha internship
**No API Key** required — runs fully offline

### How to Run on Google Colab

**Cell 1 — Install Libraries:**

```python
!pip install streamlit scikit-learn pyngrok -q
```

**Cell 2 — Upload File:**

> Upload `faq_chatbot.py` using the Files panel in Colab

**Cell 3 — Launch App:**

```python
import subprocess, threading, time
from pyngrok import ngrok

subprocess.run(["pkill", "-f", "streamlit"], capture_output=True)
time.sleep(1)

def run():
    subprocess.run(["streamlit", "run", "faq_chatbot.py",
        "--server.port", "8501", "--server.headless", "true"])

threading.Thread(target=run, daemon=True).start()
time.sleep(5)

url = ngrok.connect(8501)
print(" Open this link:", url)
```

### How It Works

```
User types a question
        ↓
Text preprocessed (lowercase + punctuation removed)
        ↓
TF-IDF Vectorization applied
        ↓
Cosine Similarity calculated vs all FAQs
        ↓
Best matching answer returned with confidence %
```

### NLP Pipeline

| Step              | Description                             |
| ----------------- | --------------------------------------- |
| **Preprocessing** | Lowercase + punctuation removal         |
| **Vectorization** | TF-IDF with bigrams (1,2)               |
| **Matching**      | Cosine Similarity scoring               |
| **Threshold**     | Score < 0.08 triggers fallback response |

---

## Task 3 — Music Generation with AI

### Overview

An AI-powered music composition tool that uses LSTM Neural Networks (TensorFlow) or Markov Chain generation to compose original melodies. Generated music is exported as a downloadable MIDI file with a visual piano roll display.

### Technologies Used

| Library                | Purpose                        |
| ---------------------- | ------------------------------ |
| `streamlit`            | Web UI framework               |
| `tensorflow` / `keras` | LSTM Neural Network model      |
| `music21`              | MIDI music creation and export |
| `numpy`                | Numerical computations         |
| `pyngrok`              | Public URL for Google Colab    |

### Features

- **LSTM Model** using TensorFlow/Keras for AI generation
- **Markov Chain** fallback — works without TensorFlow too
- **8 Musical Scales** — C Major, G Major, D Minor, Blues, Pentatonic and more
- **6 Music Styles** — Classical, Jazz, Pop, Ambient, Waltz, Electronic
- **BPM Slider** control from 60 to 200 tempo
- **Note Count** slider from 20 to 150 notes
- **Temperature Slider** for creativity and randomness control
- **Piano Roll** visual note display
- **MIDI Download** — play in VLC, MuseScore, GarageBand, FL Studio
- **Music Stats** — notes count, rests, duration, unique pitches

### How to Run on Google Colab

**Cell 1 — Install Libraries:**

```python
!pip install streamlit music21 tensorflow pyngrok -q
```

**Cell 2 — Upload File:**

> Upload `music_generator.py` using the Files panel in Colab

**Cell 3 — Launch App:**

```python
import subprocess, threading, time
from pyngrok import ngrok

subprocess.run(["pkill", "-f", "streamlit"], capture_output=True)
time.sleep(1)

def run():
    subprocess.run(["streamlit", "run", "music_generator.py",
        "--server.port", "8501", "--server.headless", "true"])

threading.Thread(target=run, daemon=True).start()
time.sleep(5)

url = ngrok.connect(8501)
print(" Open this link:", url)
```

###  How It Works

```
Select Scale + Style + BPM + Notes + Temperature
        ↓
LSTM Model trains on scale patterns (30 epochs)
        ↓
Temperature sampling generates creative note sequence
        ↓
Duration values assigned per music style
        ↓
music21 converts sequence to MIDI file
        ↓
Piano Roll displayed + MIDI download ready
```

### LSTM Model Architecture

```
Input Layer  →  Note sequence (normalized)
        ↓
LSTM Layer   →  64 units, return_sequences=True
        ↓
Dropout      →  0.2 (prevents overfitting)
        ↓
LSTM Layer   →  32 units
        ↓
Dense Layer  →  N notes output (softmax activation)
        ↓
Output       →  Next note prediction
```

---

## Submission Checklist

- [ ] Task 1 — Language Translation Tool
- [ ] Task 2 — FAQ Chatbot
- [ ] Task 3 — Music Generation with AI
- [ ] Upload all code to GitHub → `CodeAlpha_AI_Tasks`
- [ ] Post LinkedIn video tagging **@CodeAlpha**
- [ ] Add GitHub repo link in LinkedIn post
- [ ] Submit via WhatsApp group form

---

## All Requirements

### Task 1 — `requirements.txt`

```
streamlit
deep_translator
gtts
pyngrok
```

### Task 2 — `requirements.txt`

```
streamlit
scikit-learn
pyngrok
```

### Task 3 — `requirements.txt`

```
streamlit
music21
tensorflow
numpy
pyngrok
```

---

## Contact CodeAlpha

| Channel   | Details                      |
| --------- | ---------------------------- |
| Website   | www.codealpha.tech           |
| Email     | services@codealpha.tech      |
| Alt Email | services.codealpha@gmail.com |
| WhatsApp  | +91 9336576683               |

---

<div align="center">

** CodeAlpha AI Internship**

_Built with using Python, Streamlit, and AI_

</div>
