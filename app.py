# Directory Structure:
# neuro_app/
# ├── app.py
# ├── core/
#     ├── analyzer.py
#     ├── audio.py
#     ├── mmse_tasks.py
#     ├── pdf_utils.py
#     └── ui_helpers.py

# ---------- core/analyzer.py ----------
class MobileCognitiveAnalyzer:
    def __init__(self):
        self.reset_session()

    def reset_session(self):
        self.mmse_score = 0
        self.stress_history = []
        self.anxiety_signals = []
        self.depression_signals = []
        self.confusion_count = 0
        self.emotion_timeline = []
        self.task_performance = {}


# ---------- core/audio.py ----------
import numpy as np
import librosa
import whisper

model = whisper.load_model("base")


def analyze_voice(audio, sr):
    if audio is None:
        return {"stress": 0, "anxiety": 0, "depression": 0}

    if len(audio.shape) > 1:
        audio = audio[:, 0]

    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr))
    pitch = librosa.yin(audio, fmin=100, fmax=400)
    jitter = np.mean(np.abs(np.diff(pitch))) if len(pitch) > 1 else 0

    return {
        "stress": min(jitter * 8, 1.0),
        "anxiety": len(librosa.effects.split(y=audio, top_db=25)) / 10,
        "depression": 1 - (np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)) / 500)
    }


def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]


# ---------- core/mmse_tasks.py ----------
MMSE_TASKS = {
    "orientation": [
        {"question": "What year is it?", "answer": "2024", "points": 1},
        {"question": "Current season?", "answer": "summer", "points": 1}
    ],
    "memory": [
        {"question": "Remember: Apple, Table, Penny", "answer": ["apple", "table", "penny"], "points": 3}
    ],
    "attention": [
        {"question": "Count down from 20 by 3", "answer": [20, 17, 14, 11, 8, 5, 2], "points": 5}
    ],
    "language": [
        {"question": "Repeat: 'No ifs, ands, or buts'", "answer": "no ifs ands or buts", "points": 1}
    ]
}


# ---------- core/pdf_utils.py ----------
from fpdf import FPDF
from datetime import datetime
import numpy as np

def generate_pdf_report(results, filename=None):
    pdf = FPDF('P', 'mm', 'A5')
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "NeuroMind Mobile Report", 0, 1, 'C')
    pdf.ln(5)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"MMSE Score: {results['mmse_score']}/30", 0, 1)
    pdf.cell(0, 10, f"Stress Level: {np.mean(results['stress_history']):.2f}/1.0", 0, 1)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Task Performance:", 0, 1)
    for task, perf in results['task_performance'].items():
        pdf.cell(0, 10, f"- {task}: {perf['score']}/{perf['max']}", 0, 1)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Suggestions:", 0, 1)
    pdf.set_font("Arial", '', 10)
    if results['mmse_score'] < 24:
        pdf.multi_cell(0, 8, "Consider cognitive screening with a specialist")
    if np.mean(results['stress_history']) > 0.7:
        pdf.multi_cell(0, 8, "Try breathing exercises for stress")

    if not filename:
        filename = f"NeuroMind_Mobile_Report_{datetime.now().strftime('%Y%m%d')}.pdf"

    pdf.output(filename)
    return filename


# ---------- core/ui_helpers.py ----------
import streamlit as st

def set_mobile_styles():
    st.markdown("""
    <style>
        button {
            min-height: 3em !important;
            padding: 1em !important;
        }
        .stTextInput, .stNumberInput, .stSelectbox {
            width: 100% !important;
        }
        .stImage > img {
            max-width: 100% !important;
            height: auto !important;
        }
        @media (max-width: 768px) {
            .hide-mobile {
                display: none;
            }
        }
    </style>
    """, unsafe_allow_html=True)


# ---------- app.py ----------
import streamlit as st
import numpy as np
import tempfile
import json
from PIL import Image
from datetime import datetime
from streamlit_webrtc import webrtc_streamer
import av
import soundfile as sf
import firebase_admin
from firebase_admin import credentials, firestore, storage

from core.analyzer import MobileCognitiveAnalyzer
from core.audio import analyze_voice, transcribe_audio
from core.mmse_tasks import MMSE_TASKS
from core.pdf_utils import generate_pdf_report
from core.ui_helpers import set_mobile_styles

# Initialize Firebase (expects serviceAccountKey.json in repo root)
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': '<YOUR_FIREBASE_STORAGE_BUCKET>'
})
db = firestore.client()
bucket = storage.bucket()

st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_title="NeuroMind Mobile",
    page_icon="🧠"
)

# Initialize session state
if "analyzer" not in st.session_state:
    st.session_state.analyzer = MobileCognitiveAnalyzer()
if "stage" not in st.session_state:
    st.session_state.stage = "consent"
if "audio_recording" not in st.session_state:
    st.session_state.audio_recording = None
if "recording" not in st.session_state:
    st.session_state.recording = False
if "task_idx" not in st.session_state:
    st.session_state.task_idx = 0
if "tasks_remaining" not in st.session_state:
    st.session_state.tasks_remaining = []
if "current_q" not in st.session_state:
    st.session_state.current_q = ""

# Helper to save session data & report to Firebase
def save_session_and_report(user_id, results, pdf_path):
    # Save JSON of session
    data = {
        'results': results,
        'timestamp': datetime.utcnow().isoformat()
    }
    doc_ref = db.collection('users').document(user_id).collection('sessions').document()
    doc_ref.set(data)
    # Upload PDF
    blob = bucket.blob(f"{user_id}/reports/{pdf_path}")
    blob.upload_from_filename(pdf_path)

# Audio recorder using streamlit-webrtc
def record_audio():
    fs = 16000
    frames = []
    def callback(frame: av.AudioFrame):
        frames.append(frame.to_ndarray(format='flt32'))
        return frame
    ctx = webrtc_streamer(
        key='recorder',
        mode='sendrecv',
        media_stream_constraints={'audio': True, 'video': False},
        audio_frame_callback=callback,
        async_processing=True
    )
    if ctx.state.playing:
        st.info("Recording... click Stop to finish.")
    else:
        st.info("Recorder stopped.")
    if not ctx.state.playing and frames:
        audio = np.concatenate(frames, axis=1)[0]
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(tmp.name, audio, fs)
        return tmp.name, audio, fs
    return None, None, fs

# Main app flow
def mobile_main():
    set_mobile_styles()
    analyzer = st.session_state.analyzer

    # Consent
    if st.session_state.stage == 'consent':
        st.title("🧠 NeuroMind EMotion Assessment")
        if st.button("I Consent →"):
            st.session_state.stage = 'user_info'
            st.experimental_rerun()
        return

    # User Info
    if st.session_state.stage == 'user_info':
        st.title("Your Information")
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=18, max_value=120)
        if st.button("Continue →") and name:
            st.session_state.user_id = name.replace(' ','_')
            st.session_state.stage = 'baseline'
            st.experimental_rerun()
        return

    # Baseline
    if st.session_state.stage == 'baseline':
        st.header("Baseline Check")
        # Face
        img_buf = st.camera_input("Baseline Face Recording")
        if img_buf:
            st.session_state.baseline_face = np.array(Image.open(img_buf))
        # Voice
        if not st.session_state.recording:
            if st.button("Start Baseline Voice Recording"):
                st.session_state.recording = True
                st.experimental_rerun()
        else:
            path, audio_np, sr = record_audio()
            if path:
                st.session_state.baseline_audio = (path, audio_np, sr)
                st.session_state.recording = False
                st.experimental_rerun()
        if 'baseline_face' in st.session_state and 'baseline_audio' in st.session_state:
            if st.button("Begin Assessment →"):
                st.session_state.stage = 'mmse'
                st.experimental_rerun()
        return

    # MMSE Tasks
    if st.session_state.stage == 'mmse':
        # Initialize tasks
        if not st.session_state.tasks_remaining:
            key = list(MMSE_TASKS.keys())[st.session_state.task_idx]
            st.session_state.tasks_remaining = MMSE_TASKS[key].copy()
            st.session_state.current_q = st.session_state.tasks_remaining[0]['question']
        st.header(f"MMSE Task {st.session_state.task_idx+1}")
        st.write(st.session_state.current_q)
        # Record response
        if not st.session_state.recording:
            if st.button("Record Response"):
                st.session_state.recording = True
                st.experimental_rerun()
        else:
            path, audio_np, sr = record_audio()
            if path:
                st.session_state.audio_response = (path, audio_np, sr)
                st.session_state.recording = False
                st.experimental_rerun()
        if 'audio_response' in st.session_state:
            path, audio_np, sr = st.session_state.audio_response
            st.audio(audio_np, sr)
            metrics = analyze_voice(audio_np, sr)
            st.write(metrics)
            # Transcribe
            text = transcribe_audio(path)
            st.write("Transcription:", text)
            # Scoring
            task = st.session_state.tasks_remaining.pop(0)
            points = task['points'] if text.strip().lower().startswith(str(task['answer'])) else 0
            analyzer.mmse_score += points
            analyzer.stress_history.append(metrics['stress'])
            analyzer.anxiety_signals.append(metrics['anxiety'])
            analyzer.depression_signals.append(metrics['depression'])
            analyzer.emotion_timeline.append(text)
            analyzer.task_performance[task['question']] = {'score': points, 'max': task['points']}
            # Next or finish
            if not st.session_state.tasks_remaining:
                st.session_state.stage = 'results'
            else:
                st.session_state.current_q = st.session_state.tasks_remaining[0]['question']
            st.experimental_rerun()
        return

    # Results
    if st.session_state.stage == 'results':
        res = analyzer
        st.subheader("Results")
        st.write(f"MMSE Score: {res.mmse_score}/30")
        st.write(f"Avg Stress: {np.mean(res.stress_history):.2f}")
        st.write(f"Emotions Timeline: {res.emotion_timeline}")
        if st.button("Generate & Save Report"):
            results = {
                'mmse_score': res.mmse_score,
                'stress_history': res.stress_history,
                'task_performance': res.task_performance,
            }
            pdf_path = generate_pdf_report(results)
            save_session_and_report(st.session_state.user_id, results, pdf_path)
            st.success("Report generated and saved!")
        if st.button("Restart"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()

if __name__ == '__main__':
    mobile_main()
