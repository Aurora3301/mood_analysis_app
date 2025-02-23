# mood_tracker/serializers.py
from rest_framework import serializers
from .models import FaceRecognitionLog, MoodLog, Symptom  # Add Symptom to imports

class FaceRecognitionLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = FaceRecognitionLog
        fields = '__all__'

class MoodLogSerializer(serializers.ModelSerializer):
    
    symptoms = serializers.PrimaryKeyRelatedField(
        many=True,
        queryset=Symptom.objects.all(),  # This now works because Symptom is imported
        required=False
    )

    class Meta:
        model = MoodLog
        fields = '__all__'

