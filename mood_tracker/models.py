from django.db import models
from django.utils.translation import gettext_lazy as _
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone

class Symptom(models.Model):
    """
    Model for symptoms. This makes symptoms manageable as separate entities.
    """
    SYMPTOM_CHOICES = [
        ('sadness_despair', _('Sadness/Despair')),
        ('helplessness_hopelessness', _('Helplessness/Hopelessness')),
        ('agitation_irritability', _('Agitation/Irritability')),
        ('social_withdrawal', _('Social Withdrawal')),
        ('low_motivation', _('Low Motivation')),
        ('low_self_esteem', _('Low Self-Esteem')),
        ('high_anxiety_excessive_worry', _('High Anxiety or Excessive Worry')),
        ('sleep_problems', _('Sleep Problems')),
        ('headache', _('Headache')),
        ('body_ache_pain', _('Body Ache/Pain')),
        ('decreased_increased_appetite', _('Decreased or Increased Appetite')),
        ('feelings_of_guilt_self_blame', _('Feelings of Guilt or Self-Blame')),
        ('thoughts_of_death_suicide', _('Thoughts of Death or Suicide')),
    ]
    name = models.CharField(
        max_length=50,
        choices=SYMPTOM_CHOICES,
        unique=True,
        verbose_name=_("Symptom Name")
    )

    def __str__(self):
        return self.get_name_display()

    class Meta:
        verbose_name = _("Symptom")
        verbose_name_plural = _("Symptoms")


class MoodLog(models.Model):
    """
    Data model for user's daily mood logging.
    """
    models.DateField(verbose_name=_("Date of Mood Log"))
    timestamp = models.DateTimeField(auto_now_add=True)
    # Today's Depressed Mood
    DEPRESSED_MOOD_CHOICES = [
        (0, _('None')),
        (1, _('Mild Depression')),
        (2, _('Somewhat Depressed')),
        (3, _('Very Depressed')),
        (4, _('Extremely Depressed')),
    ]
    depressed_mood = models.IntegerField(
        choices=DEPRESSED_MOOD_CHOICES,
        default=0,
        verbose_name=_("Today's Depressed Mood")
    )

    # Today's Elevated Mood
    ELEVATED_MOOD_CHOICES = [
        (0, _('None')),
        (1, _('Mild Mood Elevation')),
        (2, _('Somewhat Elevated Mood')),
        (3, _('Very Elevated Mood')),
        (4, _('Extremely Elevated Mood')),
    ]
    elevated_mood = models.IntegerField(
        choices=ELEVATED_MOOD_CHOICES,
        default=0,
        verbose_name=_("Today's Elevated Mood")
    )

    # Today's Irritability
    IRRITABILITY_CHOICES = [
        (0, _('None')),
        (1, _('Mild')),
        (2, _('Moderate')),
        (3, _('Severe')),
    ]
    irritability = models.IntegerField(
        choices=IRRITABILITY_CHOICES,
        default=0,
        verbose_name=_("Today's Irritability")
    )

    # Today's Anxiety
    ANXIETY_CHOICES = [
        (0, _('None')),
        (1, _('Mild')),
        (2, _('Moderate')),
        (3, _('Severe')),
    ]
    anxiety = models.IntegerField(
        choices=ANXIETY_CHOICES,
        default=0,
        verbose_name=_("Today's Anxiety")
    )

    # Hours of Sleep
    hours_of_sleep = models.IntegerField(
        verbose_name=_("Hours of Sleep"),
        default=0,
        help_text=_("Enter the number of hours you slept.")
    )

    # Energy Level
    energy_level = models.IntegerField(
        verbose_name=_("Energy Level"),
        default=5,
        help_text=_("Energy level from 0 (lowest) to 10 (highest)."),
        validators=[MinValueValidator(0), MaxValueValidator(10)]
    )

    # Symptoms (ManyToManyField to Symptom model)
    symptoms = models.ManyToManyField(
        Symptom,
        blank=True,
        verbose_name=_("Symptoms")
    )

    def __str__(self):
        return f"{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - Mood Log"

    class Meta:
        verbose_name = _("Mood Log")
        verbose_name_plural = _("Mood Logs")
        ordering = ['-timestamp']


class FaceRecognitionLog(models.Model):
    """
    Data model for face recognition emotion logs.
    """
    timestamp = models.DateTimeField(auto_now_add=True, verbose_name=_("Time of Recording"))
    emotion = models.CharField(
        max_length=20,
        choices=[
            ('surprised', _('Surprised')),
            ('sad', _('Sad')),
            ('neutral', _('Neutral')),
            ('happy', _('Happy')),
            ('fearful', _('Fearful')),
            ('disgusted', _('Disgusted')),
            ('angry', _('Angry')),
        ],
        verbose_name=_("Detected Emotion from Face")
    )

    def __str__(self):
        return f"{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {self.emotion}"

    class Meta:
        verbose_name = _("Face Recognition Log")
        verbose_name_plural = _("Face Recognition Logs")
        ordering = ['-timestamp']