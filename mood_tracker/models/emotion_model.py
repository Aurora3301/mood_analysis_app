# models/emotion_model.py
class Emotion:
    def __init__(self, date, emotion):
        self.date = date
        self.emotion = emotion

    def __repr__(self):
        return f"Emotion(date={self.date}, emotion={self.emotion})"