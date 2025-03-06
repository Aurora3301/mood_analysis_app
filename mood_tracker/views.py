from django.shortcuts import render
from django.db.models import Avg, Count
from django.utils import timezone
from datetime import timedelta
from .models import FaceRecognitionLog, MoodLog, Symptom, Article
from .serializers import FaceRecognitionLogSerializer, MoodLogSerializer, ArticleSerializer
from rest_framework import viewsets
from django.http import JsonResponse
import tempfile
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from .models import SpeechEmotionLog

# Additional imports for the new view
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64

class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer

class FaceRecognitionLogViewSet(viewsets.ModelViewSet):
    queryset = FaceRecognitionLog.objects.all().order_by('-timestamp')
    serializer_class = FaceRecognitionLogSerializer

class MoodLogViewSet(viewsets.ModelViewSet):
    queryset = MoodLog.objects.all().order_by('-timestamp')
    serializer_class = MoodLogSerializer

    def perform_create(self, serializer):
        instance = serializer.save()
        symptoms = self.request.data.get('symptoms', [])
        instance.symptoms.set(symptoms)

def home_page(request):
    return render(request, 'mood_tracker/home.html')

def face_recognition_page(request):
    return render(request, 'mood_tracker/face_recognition_page.html')

def mood_logging_page(request):
    return render(request, 'mood_tracker/mood_logging_page.html', {
        'symptoms': Symptom.objects.all(),
        'current_date': timezone.now().date()
    })

ser_model = load_model("mood_tracker/ser_model.h5")
emotion_labels = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "ps", 5: "sad", 6: "neutral"}
max_frames = 200

def predict_emotion(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    audio = audio / np.max(np.abs(audio))
    
    segment_samples = 2 * sr
    segments = [audio[i*segment_samples:(i+1)*segment_samples] 
               for i in range(len(audio)//segment_samples)]
    
    predictions = []
    for segment in segments:
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
        if mfcc.shape[1] > max_frames:
            mfcc = mfcc[:, :max_frames]
        else:
            padding = max_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode='constant')
        mfcc_input = mfcc[np.newaxis, ..., np.newaxis]
        pred = ser_model.predict(mfcc_input, verbose=0)
        predictions.append(pred[0])
    
    avg_pred = np.mean(predictions, axis=0)
    return emotion_labels[np.argmax(avg_pred)]

def speech_emotion_page(request):
    if request.method == 'POST' and request.FILES.get('audio'):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            for chunk in request.FILES['audio'].chunks():
                tmp_file.write(chunk)
            emotion = predict_emotion(tmp_file.name)
            SpeechEmotionLog.objects.create(emotion=emotion)
            os.unlink(tmp_file.name)
            return JsonResponse({'emotion': emotion})
    return render(request, 'mood_tracker/speech_emotion_page.html')


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

    # Symptom data aggregation
    symptom_counts_week = [
        {'name': symptom.get_name_display(), 'count': symptom.count, 'id': symptom.id } # Include ID
        for symptom in Symptom.objects.filter(
            moodlog__timestamp__date__gte=past_week
        ).annotate(count=Count('moodlog'))
    ]

    symptom_counts_month = [
        {'name': symptom.get_name_display(), 'count': symptom.count, 'id': symptom.id} # Include ID
        for symptom in Symptom.objects.filter(
            moodlog__timestamp__date__gte=past_month
        ).annotate(count=Count('moodlog'))
    ]
    # Add to context
    daily_speech_emotion_week = SpeechEmotionLog.objects.filter(
        timestamp__date__gte=past_week
    ).values('emotion').annotate(count=Count('emotion'))

    daily_speech_emotion_month = SpeechEmotionLog.objects.filter(
        timestamp__date__gte=past_month
    ).values('emotion').annotate(count=Count('emotion'))

    

    context = {
        'daily_mood_avg_week': list(daily_mood_avg_week),
        'daily_mood_avg_month': list(daily_mood_avg_month),
        'daily_emotion_counts_week': list(daily_emotion_counts_week),
        'daily_emotion_counts_month': list(daily_emotion_counts_month),
        'symptom_counts_week': symptom_counts_week,
        'symptom_counts_month': symptom_counts_month,
        'daily_speech_emotion_week': list(daily_speech_emotion_week),
        'daily_speech_emotion_month': list(daily_speech_emotion_month),
   
    }
    return render(request, 'mood_tracker/analysis_page.html', context)



@csrf_exempt
def process_face_frame(request):
    """
    A simple view to process a face frame.
    Expects a POST request with a parameter 'image' containing a base64 JPEG image.
    Returns a JSON response with dummy processed image and emotion data.
    Replace the dummy code with your actual processing logic.
    """
    if request.method == 'POST':
        image_data = request.POST.get('image')
        if not image_data:
            return JsonResponse({'error': 'No image data provided'}, status=400)
        
        # Here you would normally decode and process the image data
        # For example:
        # header, encoded = image_data.split(",", 1)
        # image_bytes = base64.b64decode(encoded)
        # Process the image_bytes...
         # Dummy response values for demonstration purposes
        processed_image_base64 = "dummy_base64_image_string"
        detected_emotions = ["happy", "neutral"]
        top_emotions = {"happy": 80.0, "neutral": 20.0}
      
        return JsonResponse({
            'processed_image': processed_image_base64,
            'detected_emotions': detected_emotions,
            'top_emotions': top_emotions,
        })
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)