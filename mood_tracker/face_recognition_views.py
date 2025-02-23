# mood_tracker/face_recognition_views.py

import sys
import os

# Get the directory containing your mood_tracker app
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Add the app directory to Python's sys.path
sys.path.insert(0, APP_DIR)

# Now your imports should work (hopefully)
from utils.datasets import get_labels
import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from collections import Counter
from utils.datasets import get_labels # Make sure utils is in mood_tracker or adjust path
from utils.inference import detect_faces # Make sure utils is in mood_tracker or adjust path
from utils.inference import draw_text # Make sure utils is in mood_tracker or adjust path
from utils.inference import draw_bounding_box # Make sure utils is in mood_tracker or adjust path
from utils.inference import apply_offsets # Make sure utils is in mood_tracker or adjust path
from utils.inference import load_detection_model # Make sure utils is in mood_tracker or adjust path
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
from django.conf import settings # Import settings to use static paths if needed
from .models import FaceRecognitionLog # Import your FaceRecognitionLog model


# --- Configuration and Model Loading ---
# **Important:** Adjust paths to be relative to your Django project structure.
# Place 'models' and 'utils' folders inside your mood_tracker app, or adjust paths accordingly.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project base directory

emotion_model_path = os.path.join(BASE_DIR, 'mood_tracker', 'models', 'emotion_model.hdf5') # Correct path to model
face_cascade_path = os.path.join(BASE_DIR, 'mood_tracker', 'models', 'haarcascade_frontalface_default.xml') # Correct path to haarcascade
emotion_labels_path = os.path.join(BASE_DIR, 'mood_tracker', 'utils', 'datasets', 'datasets.py') # Assuming labels file is needed
emotion_labels = get_labels('fer2013')  # Assuming labels file is in utils/datasets
frame_window = 10
emotion_offsets = (20, 40)

face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    print("Error loading face cascade!")

emotion_classifier = load_model(emotion_model_path, compile=False)
from keras.optimizers import Adam # type: ignore
emotion_classifier.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
emotion_target_size = emotion_classifier.input_shape[1:3]
emotion_window = []
all_emotions = []

def preprocess_input(image, v2=True):
    image = Image.fromarray(image)
    image = image.resize((64, 64), Image.Resampling.LANCZOS)
    image = np.array(image)
    if v2:
        image = image / 127.5 - 1.
    return image

def detect_emotions_in_frame(frame):
    global emotion_window, all_emotions
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5,
        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    )
    emotions = []
    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)
        all_emotions.append(emotion_text)
        emotions.append(emotion_text)
        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue
        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))
        color = color.astype(int)
        color = color.tolist()
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)
    return rgb_image, emotions

def get_top_emotions():
    emotion_counts = Counter(all_emotions)
    total_emotions = sum(emotion_counts.values())
    top_3_emotions = emotion_counts.most_common(3) # changed to top 3 for richer analysis
    return {emotion: (count / total_emotions) * 100 for emotion, count in top_3_emotions}


@csrf_exempt  # For simplicity in this example, disable CSRF (for production, handle CSRF properly)
def process_face_recognition_frame(request):
    if request.method == 'POST':
        try:
            image_data = request.POST.get('image') # or request.body if sending raw data
            if not image_data:
                return JsonResponse({'error': 'No image data received'}, status=400)

            # Decode base64 image
            base64_data = image_data.split(',')[1] # remove header if present
            image_bytes = base64.b64decode(base64_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return JsonResponse({'error': 'Failed to decode image'}, status=400)

            processed_frame, detected_emotions = detect_emotions_in_frame(frame)

            # Get top emotions for analysis
            top_emotions = get_top_emotions()

            # Convert processed frame to base64 for web display
            is_success, im_buf_arr = cv2.imencode(".jpg", processed_frame)
            if not is_success:
                return JsonResponse({'error': 'Failed to encode processed image'}, status=500)
            processed_image_base64 = base64.b64encode(im_buf_arr.tobytes()).decode('utf-8')

            # Save detected emotion to database (using the most frequent emotion for simplicity)
            if detected_emotions:
                dominant_emotion = mode(detected_emotions) # Get most frequent emotion
                FaceRecognitionLog.objects.create(emotion=dominant_emotion) # Save to model

            return JsonResponse({
                'processed_image': processed_image_base64,
                'detected_emotions': detected_emotions,
                'top_emotions': top_emotions
            })

        except Exception as e:
            print(f"Error processing image: {e}") # Log detailed error on server side
            return JsonResponse({'error': 'Error processing image'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)