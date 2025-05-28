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