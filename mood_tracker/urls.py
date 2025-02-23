# mood_tracker/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import FaceRecognitionLogViewSet, MoodLogViewSet, home_page, face_recognition_page, mood_logging_page, analysis_page #  ... possibly missing analysis_page
router = DefaultRouter()
router.register(r'face-recognition-logs', FaceRecognitionLogViewSet)
router.register(r'mood-logs', MoodLogViewSet)
# DELETE or comment out the MedicationViewSet registration:
# router.register(r'medications', views.MedicationViewSet) # Removed MedicationViewSet


urlpatterns = [
    path('api/', include(router.urls)),
    path('', home_page, name='home'),
    path('face-recognition/', face_recognition_page, name='face_recognition_page'),
    path('mood-logging/', mood_logging_page, name='mood_logging_page'),
    path('analysis/', analysis_page, name='analysis_page'),
]