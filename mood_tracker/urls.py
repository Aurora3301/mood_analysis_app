from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    FaceRecognitionLogViewSet,
    MoodLogViewSet,
    home_page,
    face_recognition_page,
    mood_logging_page,
    analysis_page
)
from .face_recognition_views import process_face_recognition_frame  # Import the new view

router = DefaultRouter()
router.register(r'face-recognition-logs', FaceRecognitionLogViewSet)
router.register(r'mood-logs', MoodLogViewSet)

urlpatterns = [
    # New endpoint for processing face recognition frames
    path('api/process-face-recognition-frame/', process_face_recognition_frame, name='process_face_recognition_frame'),
    path('api/', include(router.urls)),
    path('', home_page, name='home'),
    path('face-recognition/', face_recognition_page, name='face_recognition_page'),
    path('mood-logging/', mood_logging_page, name='mood_logging_page'),
    path('analysis/', analysis_page, name='analysis_page'),
]
