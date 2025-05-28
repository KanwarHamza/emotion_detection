import streamlit as st
import cv2
import numpy as np
import librosa
import tensorflow as tf
import tempfile
import sounddevice as sd
from scipy.io.wavfile import write
import pandas as pd
import pyarrow as pa
import matplotlib.pyplot as plt
from deepface import DeepFace
import mediapipe as mp
from collections import deque
import time
from fpdf import FPDF
import plotly.graph_objects as go
import os
from datetime import datetime
from PIL import Image

# Constants - Adjusted for mobile
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

# Mobile-optimized UI config
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_title="NeuroMind Mobile",
    page_icon="🧠"
)

def set_mobile_styles():
    """Inject CSS for mobile responsiveness"""
    st.markdown("""
    <style>
        /* Larger touch targets */
        button {
            min-height: 3em !important;
            padding: 1em !important;
        }
        
        /* Full-width elements */
        .stTextInput, .stNumberInput, .stSelectbox {
            width: 100% !important;
        }
        
        /* Optimize camera feed */
        .stImage > img {
            max-width: 100% !important;
            height: auto !important;
        }
        
        /* Hide desktop elements */
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

def mobile_record_audio(duration=5, fs=16000):  # Reduced sample rate for mobile
    st.info("Speak now...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return recording, fs

def mobile_analyze_voice(audio, sr):
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    
    # Simplified mobile-friendly analysis
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr))
    pitch = librosa.yin(audio, fmin=100, fmax=400)  # Adjusted for mobile mics
    jitter = np.mean(np.abs(np.diff(pitch)))
    
    return {
        "stress": min(jitter * 8, 1.0),  # Capped at 1.0
        "anxiety": len(librosa.effects.split(y=audio, top_db=25)) / 10,
        "depression": 1 - (np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)) / 500)
    }

def mobile_camera_capture():
    """Mobile-optimized camera capture"""
    img_file_buffer = st.camera_input("Look at the camera")
    if img_file_buffer is not None:
        return Image.open(img_file_buffer)
    return None

def generate_mobile_pdf(results):
    pdf = FPDF('P', 'mm', 'A5')  # Smaller page size for mobile
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "NeuroMind Mobile Report", 0, 1, 'C')
    pdf.ln(5)
    
    # Summary
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"MMSE Score: {results['mmse_score']}/30", 0, 1)
    pdf.cell(0, 10, f"Stress Level: {np.mean(results['stress_history']):.2f}/1.0", 0, 1)
    pdf.ln(5)
    
    # Simple bar chart (mobile-friendly)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Task Performance:", 0, 1)
    for task, perf in results['task_performance'].items():
        pdf.cell(0, 10, f"- {task}: {perf['score']}/{perf['max']}", 0, 1)
    
    # Recommendations
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
    set_mobile_styles()  # Apply mobile CSS
    
    # Initialize session
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = MobileCognitiveAnalyzer()
        st.session_state.stage = "consent"
        st.session_state.user = {}
    
    # Consent Screen (Mobile-optimized)
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
            st.rerun()
    
    # User Info (Simplified for mobile)
    elif st.session_state.stage == "user_info":
        st.title("Your Information")
        with st.form("user_form"):
            st.session_state.user['name'] = st.text_input("Name")
            st.session_state.user['age'] = st.number_input("Age", min_value=18, max_value=120)
            
            if st.form_submit_button("Continue →"):
                st.session_state.stage = "baseline"
                st.rerun()
    
    # Baseline Recording (Mobile camera/mic)
    elif st.session_state.stage == "baseline":
        st.title("Baseline Check")
        st.write("We'll measure your neutral state")
        
        # Mobile camera capture
        st.subheader("1. Face Recording")
        img = mobile_camera_capture()
        if img:
            st.session_state.baseline_face = np.array(img)
            st.success("Face captured!")
        
        # Voice recording
        st.subheader("2. Voice Recording")
        if st.button("Record Neutral Voice"):
            audio, sr = mobile_record_audio()
            st.session_state.baseline_audio = (audio, sr)
            st.audio(audio, sample_rate=sr)
            st.success("Voice recorded!")
        
        if 'baseline_face' in st.session_state and 'baseline_audio' in st.session_state:
            if st.button("Begin Assessment →", type="primary"):
                st.session_state.stage = "mmse"
                st.rerun()
    
    # MMSE Tasks (Mobile-optimized flow)
    elif st.session_state.stage == "mmse":
        if "task_idx" not in st.session_state:
            st.session_state.task_idx = 0
            st.session_state.current_q = ""
            st.session_state.tasks_remaining = []
        
        task_type = list(MMSE_TASKS.keys())[st.session_state.task_idx]
        
        if not st.session_state.tasks_remaining:
            st.session_state.tasks_remaining = MMSE_TASKS[task_type].copy()
            st.session_state.current_q = st.session_state.tasks_remaining[0]["question"]
        
        # Mobile task display
        st.title(f"Task {st.session_state.task_idx + 1}")
        st.markdown(f"<div style='font-size: 1.5em; margin-bottom: 1em;'>{st.session_state.current_q}</div>", 
                   unsafe_allow_html=True)
        
        # Response capture
        if st.button("Record Response", type="primary"):
            st.session_state.audio_response = mobile_record_audio()
            st.audio(st.session_state.audio_response[0], sample_rate=st.session_state.audio_response[1])
        
        if 'audio_response' in st.session_state:
            if st.button("Submit Response"):
                # Simplified analysis for mobile
                voice_metrics = mobile_analyze_voice(*st.session_state.audio_response)
                img = mobile_camera_capture()
                
                if img:
                    try:
                        face_result = DeepFace.analyze(np.array(img), actions=['emotion'], enforce_detection=False)
                        emotion = face_result[0]['dominant_emotion']
                    except:
                        emotion = "neutral"
                
                # Score calculation
                current_task = next(t for t in MMSE_TASKS[task_type] if t["question"] == st.session_state.current_q)
                score = 0
                
                # (Actual implementation would use speech-to-text)
                if st.text_input("Verify response (simplified)"):
                    score = current_task["points"]  # Simplified scoring
                
                # Update session
                st.session_state.analyzer.mmse_score += score
                st.session_state.analyzer.stress_history.append(voice_metrics['stress'])
                st.session_state.analyzer.anxiety_signals.append(voice_metrics['anxiety'])
                
                # Next task
                st.session_state.tasks_remaining.pop(0)
                if st.session_state.tasks_remaining:
                    st.session_state.current_q = st.session_state.tasks_remaining[0]["question"]
                else:
                    st.session_state.task_idx += 1
                    if st.session_state.task_idx < len(MMSE_TASKS):
                        st.session_state.current_q = MMSE_TASKS[list(MMSE_TASKS.keys())[st.session_state.task_idx]][0]["question"]
                    else:
                        st.session_state.stage = "results"
                
                del st.session_state.audio_response
                st.rerun()
    
    # Results (Mobile-friendly)
    elif st.session_state.stage == "results":
        st.title("Assessment Complete")
        
        # Prepare results
        results = {
            "mmse_score": st.session_state.analyzer.mmse_score,
            "stress_history": st.session_state.analyzer.stress_history,
            "task_performance": {k: {"score": 1, "max": 1} for k in MMSE_TASKS.keys()}  # Simplified
        }
        
        # Mobile summary
        st.metric("Cognitive Score", f"{results['mmse_score']}/30")
        st.metric("Avg Stress", f"{np.mean(results['stress_history']):.2f}/1.0")
        
        # Generate PDF
        if st.button("Generate PDF Report", type="primary"):
            report_path = generate_mobile_pdf(results)
            with open(report_path, "rb") as f:
                st.download_button(
                    "Download Report",
                    f,
                    file_name="NeuroMind_Report.pdf",
                    mime="application/pdf"
                )
            
            st.success("Report ready!")

if __name__ == "__main__":
    mobile_main()