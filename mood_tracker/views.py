# mood_tracker/views.py
from django.shortcuts import render
from django.db.models import Avg, Count
from django.utils import timezone
from datetime import timedelta
from .models import FaceRecognitionLog, MoodLog, Symptom # Removed Medication
from .serializers import FaceRecognitionLogSerializer, MoodLogSerializer # Removed MedicationSerializer
from rest_framework import viewsets


class FaceRecognitionLogViewSet(viewsets.ModelViewSet):
    queryset = FaceRecognitionLog.objects.all().order_by('-timestamp')
    serializer_class = FaceRecognitionLogSerializer

class MoodLogViewSet(viewsets.ModelViewSet):
    queryset = MoodLog.objects.all().order_by('-timestamp')
    serializer_class = MoodLogSerializer

    def perform_create(self, serializer):
        instance = serializer.save()
        # Handle symptoms
        symptoms = self.request.data.get('symptoms', [])
        instance.symptoms.set(symptoms)


# DELETE the MedicationViewSet class entirely:
# class MedicationViewSet(viewsets.ModelViewSet):
#     queryset = Medication.objects.all()
#     serializer_class = MedicationSerializer


def home_page(request):
    return render(request, 'mood_tracker/home.html')

def face_recognition_page(request):
    return render(request, 'mood_tracker/face_recognition_page.html')

def mood_logging_page(request):
    return render(request, 'mood_tracker/mood_logging_page.html', {
        'symptoms': Symptom.objects.all(),
        'current_date': timezone.now().date()
    })

def analysis_page(request):
    # Date calculations
    today = timezone.now().date()
    past_week = today - timedelta(days=7)
    past_month = today - timedelta(days=30)

    # Mood Logs
    mood_logs_week = MoodLog.objects.filter(
        timestamp__date__gte=past_week
    )

    mood_logs_month = MoodLog.objects.filter(
        timestamp__date__gte=past_month
    )

    # Mood data aggregation
    daily_mood_avg_week = mood_logs_week.values('timestamp__date').annotate(
        avg_depressed_mood=Avg('depressed_mood'),
        avg_elevated_mood=Avg('elevated_mood'),
        avg_irritability=Avg('irritability'),
        avg_anxiety=Avg('anxiety'),
        avg_energy_level=Avg('energy_level'),
        avg_hours_of_sleep=Avg('hours_of_sleep')
    )

    daily_mood_avg_month = mood_logs_month.values('timestamp__date').annotate(
        avg_depressed_mood=Avg('depressed_mood'),
        avg_elevated_mood=Avg('elevated_mood'),
        avg_irritability=Avg('irritability'),
        avg_anxiety=Avg('anxiety'),
        avg_energy_level=Avg('energy_level'),
        avg_hours_of_sleep=Avg('hours_of_sleep')
    )

    # Facial expression data
    daily_emotion_counts_week = FaceRecognitionLog.objects.filter(
        timestamp__date__gte=past_week
    ).values('emotion').annotate(count=Count('emotion'))

    daily_emotion_counts_month = FaceRecognitionLog.objects.filter(
        timestamp__date__gte=past_month
    ).values('emotion').annotate(count=Count('emotion'))

    context = {
        'daily_mood_avg_week': list(daily_mood_avg_week),
        'daily_mood_avg_month': list(daily_mood_avg_month),
        'daily_emotion_counts_week': list(daily_emotion_counts_week),
        'daily_emotion_counts_month': list(daily_emotion_counts_month),
    }
    return render(request, 'mood_tracker/analysis_page.html', context)