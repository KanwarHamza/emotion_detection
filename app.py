# app.py
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

# --- Firebase initialization from Streamlit secrets ---
firebase_creds_json = st.secrets["FIREBASE_CREDENTIALS"]
firebase_creds_dict = json.loads(firebase_creds_json)

cred = credentials.Certificate(firebase_creds_dict)
firebase_admin.initialize_app(cred, {'storageBucket': st.secrets["STORAGE_BUCKET"]})

db = firestore.client()
bucket = storage.bucket()

# --- Streamlit page config ---
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_title="NeuroMind Mobile",
    page_icon="🧠"
)

# --- Initialize session state variables ---
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

# --- Helper to save session data and PDF report to Firebase ---
def save_session_and_report(user_id, results, pdf_path):
    data = {
        'results': results,
        'timestamp': datetime.utcnow().isoformat()
    }
    doc_ref = db.collection('users').document(user_id).collection('sessions').document()
    doc_ref.set(data)
    blob = bucket.blob(f"{user_id}/reports/{pdf_path}")
    blob.upload_from_filename(pdf_path)

# --- Audio recorder using streamlit-webrtc ---
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

# --- Main app flow ---
def mobile_main():
    set_mobile_styles()
    analyzer = st.session_state.analyzer

    # Consent stage
    if st.session_state.stage == 'consent':
        st.title("🧠 NeuroMind EMotion Assessment")
        if st.button("I Consent →"):
            st.session_state.stage = 'user_info'
            st.experimental_rerun()
        return

    # User information stage
    if st.session_state.stage == 'user_info':
        st.title("Your Information")
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=18, max_value=120)
        if st.button("Continue →") and name:
            st.session_state.user_id = name.replace(' ', '_')
            st.session_state.stage = 'baseline'
            st.experimental_rerun()
        return

    # Baseline recording stage
    if st.session_state.stage == 'baseline':
        st.header("Baseline Check")
        img_buf = st.camera_input("Baseline Face Recording")
        if img_buf:
            st.session_state.baseline_face = np.array(Image.open(img_buf))

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

    # MMSE tasks stage
    if st.session_state.stage == 'mmse':
        if not st.session_state.tasks_remaining:
            key = list(MMSE_TASKS.keys())[st.session_state.task_idx]
            st.session_state.tasks_remaining = MMSE_TASKS[key].copy()
            st.session_state.current_q = st.session_state.tasks_remaining[0]['question']

        st.header(f"MMSE Task {st.session_state.task_idx + 1}")
        st.write(st.session_state.current_q)

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

            text = transcribe_audio(path)
            st.write("Transcription:", text)

            task = st.session_state.tasks_remaining.pop(0)
            # Simple check if transcription starts with answer (case-insensitive)
            correct_answer = False
            if isinstance(task['answer'], list):
                # For list answers, check if any answer word in transcription
                correct_answer = any(ans.lower() in text.lower() for ans in task['answer'])
            else:
                correct_answer = text.strip().lower().startswith(str(task['answer']).lower())

            points = task['points'] if correct_answer else 0

            analyzer.mmse_score += points
            analyzer.stress_history.append(metrics['stress'])
            analyzer.anxiety_signals.append(metrics['anxiety'])
            analyzer.depression_signals.append(metrics['depression'])
            analyzer.emotion_timeline.append(text)
            analyzer.task_performance[task['question']] = {'score': points, 'max': task['points']}

            if not st.session_state.tasks_remaining:
                st.session_state.task_idx += 1
                # If all MMSE tasks done, move to results
                if st.session_state.task_idx >= len(MMSE_TASKS):
                    st.session_state.stage = 'results'
                else:
                    st.session_state.tasks_remaining = []
                st.experimental_rerun()
            else:
                st.session_state.current_q = st.session_state.tasks_remaining[0]['question']
                st.experimental_rerun()
        return

    # Results stage
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