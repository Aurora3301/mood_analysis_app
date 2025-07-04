# mood_tracker/admin.py
from django.contrib import admin
from .models import MoodLog, FaceRecognitionLog, Symptom
from .models import SpeechEmotionLog

admin.site.register(SpeechEmotionLog)
admin.site.register(MoodLog) # Register MoodLog model
admin.site.register(FaceRecognitionLog) # Register FaceRecognitionLog model
admin.site.register(Symptom) # Register Symptom model (add this line to register Symptom model)