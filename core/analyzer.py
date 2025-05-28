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