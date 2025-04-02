import sys
import os
import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from collections import Counter
from .utils.datasets import get_labels
from .utils.inference import draw_text
from .utils.inference import draw_bounding_box
from .utils.inference import apply_offsets
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
from django.conf import settings
from .models import FaceRecognitionLog
import dlib  # Added Dlib import

# Get the directory containing your mood_tracker app
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

# --- Configuration and Model Loading ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project base directory

emotion_model_path = os.path.join(BASE_DIR, 'mood_tracker', 'models', 'emotion_model.hdf5')
emotion_labels = get_labels('fer2013')
frame_window = 10
emotion_offsets = (20, 40)

# Initialize Dlib's HOG face detector instead of Haar cascade
face_detector = dlib.get_frontal_face_detector()

emotion_classifier = load_model(emotion_model_path, compile=False)
from keras.optimizers import Adam  # type: ignore
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
    
    # Use Dlib's HOG detector for face detection
    faces = face_detector(gray_image, 1)  # 1 upsampling layer
    
    emotions = []
    for face in faces:
        # Extract coordinates from Dlib rectangle
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        face_coordinates = (x, y, w, h)
        
        # Apply offsets for emotion detection
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        
        # Clip coordinates to image bounds
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, gray_image.shape[1]), min(y2, gray_image.shape[0])
        if x2 <= x1 or y2 <= y1:
            continue  # Skip invalid regions
        
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, emotion_target_size)
        except Exception as e:
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
        
        # Define color based on emotion
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
        color = color.astype(int).tolist()
        
        # Draw using original face coordinates
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 1, 1)
    
    return rgb_image, emotions

def get_top_emotions():
    emotion_counts = Counter(all_emotions)
    total_emotions = sum(emotion_counts.values())
    if total_emotions == 0:
        return {}
    top_3_emotions = emotion_counts.most_common(3)
    return {emotion: (count / total_emotions) * 100 for emotion, count in top_3_emotions}

@csrf_exempt
def process_face_recognition_frame(request):
    if request.method == 'POST':
        try:
            image_data = request.POST.get('image')
            if not image_data:
                return JsonResponse({'error': 'No image data received'}, status=400)

            if ',' in image_data:
                base64_data = image_data.split(',')[1]
            else:
                base64_data = image_data
            image_bytes = base64.b64decode(base64_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return JsonResponse({'error': 'Failed to decode image'}, status=400)

            processed_frame, detected_emotions = detect_emotions_in_frame(frame)
            top_emotions = get_top_emotions()

            is_success, im_buf_arr = cv2.imencode(".jpg", processed_frame)
            if not is_success:
                return JsonResponse({'error': 'Failed to encode processed image'}, status=500)
            processed_image_base64 = base64.b64encode(im_buf_arr.tobytes()).decode('utf-8')

            if detected_emotions:
                try:
                    dominant_emotion = mode(detected_emotions)
                    FaceRecognitionLog.objects.create(emotion=dominant_emotion)
                except Exception as e:
                    print(f"Error saving emotion log: {e}")

            return JsonResponse({
                'processed_image': processed_image_base64,
                'detected_emotions': detected_emotions,
                'top_emotions': top_emotions,
            })
        except Exception as e:
            print(f"Error processing image: {e}")
            return JsonResponse({'error': 'Error processing image'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)