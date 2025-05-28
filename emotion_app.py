import streamlit as st
import numpy as np
import librosa
from streamlit_webrtc import webrtc_streamer
import av
from PIL import Image
import tensorflow as tf
from fpdf import FPDF
from datetime import datetime
from deepface import DeepFace
import time

# Constants
FACE_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
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

st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_title="NeuroMind Mobile",
    page_icon="🧠"
)

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

def mobile_record_audio():
    """
    Records audio via browser mic using streamlit-webrtc.
    User manually starts/stops recording.
    Returns (audio_np, sample_rate) when recording stopped, else (None, fs).
    """
    fs = 16000  # Sample rate

    audio_frames = []

    def audio_frame_callback(frame: av.AudioFrame):
        audio_frames.append(frame.to_ndarray(format="flt32"))
        return frame

    webrtc_ctx = webrtc_streamer(
        key="audio-recorder",
        mode="sendrecv",
        media_stream_constraints={"audio": True, "video": False},
        audio_frame_callback=audio_frame_callback,
        async_processing=True
    )

    if webrtc_ctx.state.playing:
        st.write("Recording... Speak now!")
    else:
        st.write("Recorder stopped.")

    if not webrtc_ctx.state.playing and len(audio_frames) > 0:
        # Concatenate all frames to one 1D numpy array
        audio_np = np.concatenate(audio_frames, axis=1)[0]
        return audio_np, fs
    else:
        return None, fs

def mobile_analyze_voice(audio, sr):
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

def mobile_camera_capture():
    img_file_buffer = st.camera_input("Look at the camera")
    if img_file_buffer is not None:
        return Image.open(img_file_buffer)
    return None

def generate_mobile_pdf(results):
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
    filename = f"NeuroMind_Mobile_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
    pdf.output(filename)
    return filename

def mobile_main():
    set_mobile_styles()

    if "analyzer" not in st.session_state:
        st.session_state.analyzer = MobileCognitiveAnalyzer()
        st.session_state.stage = "consent"
        st.session_state.user = {}
        st.session_state.recording_in_progress = False
        st.session_state.audio_recording = None
        st.session_state.task_idx = 0
        st.session_state.tasks_remaining = []
        st.session_state.current_q = ""

    # Consent
    if st.session_state.stage == "consent":
        st.title("🧠 NeuroMind EMotion Assessment")
        st.markdown ("### Developed by Kanwar Hamza Shuja")
        st.markdown("""
        <div style='font-size: 1.2em;'>
            This assessment analyzes:<br>
            - Cognitive function<br>
            - Stress levels<br>
            - Emotional state
        </div>
        """, unsafe_allow_html=True)

        if st.button("I Consent →", type="primary"):
            st.session_state.stage = "user_info"
            st.experimental_rerun()

    # User Info
    elif st.session_state.stage == "user_info":
        st.title("Your Information")
        with st.form("user_form"):
            st.session_state.user['name'] = st.text_input("Name")
            st.session_state.user['age'] = st.number_input("Age", min_value=18, max_value=120)
            if st.form_submit_button("Continue →"):
                st.session_state.stage = "baseline"
                st.experimental_rerun()

    # Baseline Recording
    elif st.session_state.stage == "baseline":
        st.title("Baseline Check")
        st.write("We'll measure your neutral state")

        # Face capture
        st.subheader("1. Face Recording")
        img = mobile_camera_capture()
        if img:
            st.session_state.baseline_face = np.array(img)
            st.success("Face captured!")

        # Voice recording
        st.subheader("2. Voice Recording")

        if not st.session_state.recording_in_progress:
            if st.button("Start Recording Neutral Voice"):
                st.session_state.recording_in_progress = True
                st.experimental_rerun()
        else:
            audio, sr = mobile_record_audio()
            if audio is not None:
                st.session_state.audio_recording = (audio, sr)
                st.session_state.recording_in_progress = False
                st.experimental_rerun()

        if st.session_state.audio_recording:
            audio, sr = st.session_state.audio_recording
            st.audio(audio, sample_rate=sr)
            st.success("Voice recorded!")

        if ('baseline_face' in st.session_state) and st.session_state.audio_recording:
            if st.button("Begin Assessment →", type="primary"):
                st.session_state.stage = "mmse"
                st.experimental_rerun()

    # MMSE Tasks
    elif st.session_state.stage == "mmse":
        if not st.session_state.tasks_remaining:
            task_type = list(MMSE_TASKS.keys())[st.session_state.task_idx]
            st.session_state.tasks_remaining = MMSE_TASKS[task_type].copy()
            st.session_state.current_q = st.session_state.tasks_remaining[0]["question"]

        task_type = list(MMSE_TASKS.keys())[st.session_state.task_idx]
        st.title(f"Task {st.session_state.task_idx + 1}")
        st.markdown(f"<div style='font-size: 1.5em; margin-bottom: 1em;'>{st.session_state.current_q}</div>", unsafe_allow_html=True)

        if "recording_in_progress" not in st.session_state or not st.session_state.recording_in_progress:
            if st.button("Record Response"):
                st.session_state.recording_in_progress = True
                st.experimental_rerun()
        else:
            audio, sr = mobile_record_audio()
            if audio is not None:
                st.session_state.audio_response = (audio, sr)
                st.session_state.recording_in_progress = False
                st.experimental_rerun()

        if "audio_response" in st.session_state:
            audio, sr = st.session_state.audio_response
            st.audio(audio, sample_rate=sr)
            voice_metrics = mobile_analyze_voice(audio, sr)
            st.write("Voice metrics:", voice_metrics)

            img = mobile_camera_capture()
            if img:
                try:
                    face_result = DeepFace.analyze(np.array(img), actions=['emotion'], enforce_detection=False)
                    emotion = face_result[0]['dominant_emotion']
                except:
                    emotion = "neutral"
            else:
                emotion = "neutral"

            current_task = next(t for t in MMSE_TASKS[task_type] if t["question"] == st.session_state.current_q)
            score = 0
            user_resp = st.text_input("Verify response (simplified)")

            if user_resp.strip():
                score = current_task["points"]

            st.session_state.analyzer.mmse_score += score
            st.session_state.analyzer.stress_history.append(voice_metrics['stress'])
            st.session_state.analyzer.anxiety_signals.append(voice_metrics['anxiety'])
            st.session_state.analyzer.depression_signals.append(voice_metrics['depression'])
            st.session_state.analyzer.emotion_timeline.append(emotion)
            st.session_state.analyzer.task_performance[st.session_state.current_q] = {
                "score": score, "max": current_task["points"], "emotion": emotion
            }

            st.session_state.tasks_remaining.pop(0)
            st.session_state.current_q = st.session_state.tasks_remaining[0]["question"] if st.session_state.tasks_remaining else ""

            del st.session_state.audio_response

            if not st.session_state.tasks_remaining:
                st.session_state.task_idx += 1
                if st.session_state.task_idx >= len(MMSE_TASKS):
                    st.session_state.stage = "result"
                else:
                    st.session_state.tasks_remaining = []
                st.experimental_rerun()

    # Results
    elif st.session_state.stage == "result":
        st.title("Assessment Results")
        res = st.session_state.analyzer
        st.markdown(f"**MMSE Score:** {res.mmse_score}/30")
        st.markdown(f"**Average Stress Level:** {np.mean(res.stress_history):.2f}/1.0")
        st.markdown(f"**Detected Emotions:** {set(res.emotion_timeline)}")
        st.markdown(f"**Task Performance:**")
        for task, perf in res.task_performance.items():
            st.write(f"- {task}: {perf['score']}/{perf['max']} (Emotion: {perf['emotion']})")

        if st.button("Generate PDF Report"):
            filename = generate_mobile_pdf({
                "mmse_score": res.mmse_score,
                "stress_history": res.stress_history,
                "task_performance": res.task_performance
            })
            with open(filename, "rb") as f:
                st.download_button("Download Report", f, file_name=filename, mime="application/pdf")

        if st.button("Restart"):
            st.session_state.clear()
            st.experimental_rerun()

if __name__ == "__main__":
    mobile_main()
