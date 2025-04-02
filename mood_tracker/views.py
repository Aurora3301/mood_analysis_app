from django.shortcuts import render
from django.db.models import Avg, Count
from django.utils import timezone
from datetime import timedelta
from .models import FaceRecognitionLog, MoodLog, Symptom, Article, SpeechEmotionLog
from .serializers import FaceRecognitionLogSerializer, MoodLogSerializer, ArticleSerializer
from rest_framework import viewsets
from django.http import JsonResponse
import tempfile
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Import pydub
from pydub import AudioSegment

# Additional imports for the new view
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
import logging # Import logging

logger = logging.getLogger(__name__) # Get a logger instance

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

ser_model = load_model("mood_tracker/tc_ser_model.h5")
emotion_labels = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "ps", 5: "sad", 6: "neutral"}
max_frames = 200

def predict_emotion(audio_path):
    logger.info(f"predict_emotion: Processing audio file: {audio_path}") # Log input path
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        logger.info(f"librosa.load successful. Sample rate: {sr}, Audio shape: {audio.shape}, Audio dtype: {audio.dtype}") # Log after load
        logger.info(f"First 10 audio samples: {audio[:10]}") # Log first few samples
        if not np.isfinite(audio).all(): # Check for non-finite values immediately after loading
            logger.error("Error: Audio buffer contains non-finite values (NaN or Inf) after librosa.load")
            raise ValueError("Audio buffer is not finite everywhere after loading")


        max_amplitude = np.max(np.abs(audio))
        logger.info(f"Max amplitude: {max_amplitude}") # Log max amplitude
        if max_amplitude == 0: # Check if max amplitude is zero
            logger.error("Error: Max amplitude is zero, audio might be silent")
            raise ValueError("Audio data appears to be silent or empty")


        audio = audio / max_amplitude # Normalization - potential division by zero issue
        logger.info("Audio normalization successful.")


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

            # Transpose the MFCC matrix to match the expected input shape
            mfcc_transposed = np.transpose(mfcc, (1, 0))
            mfcc_input = mfcc_transposed[np.newaxis, :]

            pred = ser_model.predict(mfcc_input, verbose=0)
            predictions.append(pred[0])

        avg_pred = np.mean(predictions, axis=0)
        predicted_emotion = emotion_labels[np.argmax(avg_pred)]
        logger.info(f"Predicted emotion: {predicted_emotion}") # Log predicted emotion
        return predicted_emotion

    except Exception as e:
        logger.error(f"Exception in predict_emotion: {e}", exc_info=True) # Log full exception with traceback
        raise ValueError(f"Emotion prediction failed: {e}") # Re-raise for speech_emotion_page to catch


def speech_emotion_page(request):
    if request.method == 'POST' and request.FILES.get('audio'):
        audio_file = request.FILES['audio']
        temp_audio_path = "" # Define outside try block for cleanup in except block
        converted_audio_path = "" # Define outside try block
        try: # Added try-except block for error handling
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file: # Ensure saving as WAV
                for chunk in audio_file.chunks():
                    tmp_file.write(chunk)
                temp_audio_path = tmp_file.name # Capture temp file name
                logger.info(f"speech_emotion_page: Temporary audio file saved to: {temp_audio_path}") # Log temp file path

            # --- Audio Format Conversion using pydub ---
            try:
                audio = AudioSegment.from_file(temp_audio_path) # pydub auto-detects format
                converted_audio_path = temp_audio_path.replace(".wav", "_converted.wav") # Create new path for converted file
                audio.export(converted_audio_path, format="wav", parameters=["-ac", "1", "-ar", "16000"]) # Convert to PCM WAV, mono, 16kHz
                logger.info(f"speech_emotion_page: Audio conversion successful. Converted file saved to: {converted_audio_path}") # Log converted file path
            except Exception as e_convert:
                logger.error(f"speech_emotion_page: Audio conversion failed: {str(e_convert)}", exc_info=True) # Log conversion error
                os.unlink(temp_audio_path) # Clean up original temp file
                return JsonResponse({'error': f"Audio conversion failed: {str(e_convert)}"}, status=400) # Return error if conversion fails
            # --- End Conversion ---

            try:
                emotion = predict_emotion(converted_audio_path) # Process the *converted* file
                SpeechEmotionLog.objects.create(emotion=emotion)
                os.unlink(temp_audio_path) # Clean up both temp files
                os.unlink(converted_audio_path)
                return JsonResponse({'emotion': emotion})
            except ValueError as e_predict: # Catch ValueError from predict_emotion (and log in predict_emotion)
                os.unlink(temp_audio_path) # Clean up both temp files even if prediction fails
                os.unlink(converted_audio_path)
                return JsonResponse({'error': str(e_predict)}, status=500) # Indicate server error during prediction
            except Exception as e_predict_other: # Catch other prediction errors
                logger.error(f"speech_emotion_page: Unexpected error during emotion prediction: {e_predict_other}", exc_info=True)
                os.unlink(temp_audio_path) # Clean up both temp files
                os.unlink(converted_audio_path)
                return JsonResponse({'error': f"Emotion prediction failed unexpectedly."}, status=500)


        except Exception as e_file: # Catch file handling errors
            logger.error(f"speech_emotion_page: File handling error: {str(e_file)}", exc_info=True) # Log file error
            return JsonResponse({'error': f"File error: {str(e_file)}"}, status=400) # Indicate bad request if file issue

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