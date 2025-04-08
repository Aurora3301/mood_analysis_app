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
import os
import librosa
import numpy as np
from scipy.signal import butter, filtfilt
from tensorflow.keras.models import load_model
from django.http import JsonResponse
from django.shortcuts import render
import tempfile
from pydub import AudioSegment
import logging
from tensorflow.keras.models import load_model
from scipy.signal import butter, filtfilt
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


logger = logging.getLogger(__name__)

# Load the new model
ser_model = load_model("mood_tracker/enhanced_ser_model.h5")
# Updated emotion labels to match tc_ser_model.h5
emotion_labels = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "neutral", 6: "surprise"}
max_frames = 200


def butter_bandpass(lowcut, highcut, sr, order=5):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(audio, sr, lowcut=80.0, highcut=8000.0, order=5):
    nyquist = 0.5 * sr
    if highcut >= nyquist:
        highcut = nyquist * 0.999
    if lowcut <= 0:
        lowcut = 0.001 * nyquist
    b, a = butter_bandpass(lowcut, highcut, sr, order=order)
    filtered_audio = filtfilt(b, a, audio)
    return filtered_audio

def predict_emotion(audio_path):
    logger.info(f"predict_emotion: Processing audio file: {audio_path}")
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        logger.info(f"librosa.load successful. Sample rate: {sr}, Audio shape: {audio.shape}, Audio dtype: {audio.dtype}")
        logger.info(f"First 10 audio samples: {audio[:10]}")
        if not np.isfinite(audio).all():
            logger.error("Error: Audio buffer contains non-finite values (NaN or Inf) after librosa.load")
            raise ValueError("Audio buffer is not finite everywhere after loading")

        # Apply bandpass filter
        audio = apply_bandpass_filter(audio, sr)
        logger.info("Bandpass filter applied.")
        if not np.isfinite(audio).all():
            logger.error("Error: Audio buffer contains non-finite values (NaN or Inf) after filtering")
            raise ValueError("Audio buffer is not finite everywhere after filtering")

        segment_samples = 2 * sr
        segments = [audio[i*segment_samples:(i+1)*segment_samples]
                   for i in range(len(audio)//segment_samples)]

        # Handle short audio
        if not segments:
            segments = [audio]

        predictions = []
        for segment in segments:
            # Extract MFCC and Mel-spectrogram
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
            mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=40, n_fft=2048, hop_length=512)
            
            # Adjust to max_frames (200)
            if mfcc.shape[1] > max_frames:
                mfcc = mfcc[:, :max_frames]
                mel_spec = mel_spec[:, :max_frames]
            else:
                padding = max_frames - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode='constant')
                mel_spec = np.pad(mel_spec, ((0, 0), (0, padding)), mode='constant')

            # Concatenate MFCC and Mel-spectrogram to get 80 features
            features = np.concatenate((mfcc, mel_spec), axis=0)  # Shape: (80, max_frames)

            # Transpose to (max_frames, 80) for Conv1D input
            features_transposed = np.transpose(features, (1, 0))  # Shape: (200, 80)
            features_input = features_transposed[np.newaxis, :]   # Shape: (1, 200, 80)

            pred = ser_model.predict(features_input, verbose=0)
            predictions.append(pred[0])

        if predictions:
            avg_pred = np.mean(predictions, axis=0)
            predicted_emotion = emotion_labels[np.argmax(avg_pred)]
        else:
            predicted_emotion = "unknown"
        logger.info(f"Predicted emotion: {predicted_emotion}")
        return predicted_emotion

    except Exception as e:
        logger.error(f"Exception in predict_emotion: {e}", exc_info=True)
        raise ValueError(f"Emotion prediction failed: {e}")

def speech_emotion_page(request):
    if request.method == 'POST' and request.FILES.get('audio'):
        audio_file = request.FILES['audio']
        temp_audio_path = ""
        converted_audio_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                for chunk in audio_file.chunks():
                    tmp_file.write(chunk)
                temp_audio_path = tmp_file.name
                logger.info(f"speech_emotion_page: Temporary audio file saved to: {temp_audio_path}")

            # Audio format conversion using pydub
            try:
                audio = AudioSegment.from_file(temp_audio_path)
                converted_audio_path = temp_audio_path.replace(".wav", "_converted.wav")
                audio.export(converted_audio_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
                logger.info(f"speech_emotion_page: Audio conversion successful. Converted file saved to: {converted_audio_path}")
            except Exception as e_convert:
                logger.error(f"speech_emotion_page: Audio conversion failed: {str(e_convert)}", exc_info=True)
                os.unlink(temp_audio_path)
                return JsonResponse({'error': f"Audio conversion failed: {str(e_convert)}"}, status=400)

            try:
                emotion = predict_emotion(converted_audio_path)
                SpeechEmotionLog.objects.create(emotion=emotion)
                os.unlink(temp_audio_path)
                os.unlink(converted_audio_path)
                return JsonResponse({'emotion': emotion})
            except ValueError as e_predict:
                os.unlink(temp_audio_path)
                os.unlink(converted_audio_path)
                return JsonResponse({'error': str(e_predict)}, status=500)
            except Exception as e_predict_other:
                logger.error(f"speech_emotion_page: Unexpected error during emotion prediction: {e_predict_other}", exc_info=True)
                os.unlink(temp_audio_path)
                os.unlink(converted_audio_path)
                return JsonResponse({'error': "Emotion prediction failed unexpectedly."}, status=500)

        except Exception as e_file:
            logger.error(f"speech_emotion_page: File handling error: {str(e_file)}", exc_info=True)
            return JsonResponse({'error': f"File error: {str(e_file)}"}, status=400)

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