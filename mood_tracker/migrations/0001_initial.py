# Generated by Django 4.2.18 on 2025-02-22 14:48

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='FaceRecognitionLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.DateTimeField(auto_now_add=True, verbose_name='Time of Recording')),
                ('emotion', models.CharField(choices=[('surprised', 'Surprised'), ('sad', 'Sad'), ('neutral', 'Neutral'), ('happy', 'Happy'), ('fearful', 'Fearful'), ('disgusted', 'Disgusted'), ('angry', 'Angry')], max_length=20, verbose_name='Detected Emotion from Face')),
            ],
            options={
                'verbose_name': 'Face Recognition Log',
                'verbose_name_plural': 'Face Recognition Logs',
                'ordering': ['-timestamp'],
            },
        ),
        migrations.CreateModel(
            name='Medication',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100, verbose_name='Medication Name')),
                ('dosage', models.IntegerField(verbose_name='Dosage')),
                ('units', models.CharField(default='mg', help_text='Units of dosage (e.g., mg, ml)', max_length=10, verbose_name='Units')),
                ('time_of_day', models.TimeField(verbose_name='Time of Day')),
                ('frequency_days', models.IntegerField(verbose_name='Frequency (Days)')),
                ('start_date', models.DateField(auto_now_add=True, verbose_name='Start Date')),
            ],
            options={
                'verbose_name': 'Medication',
                'verbose_name_plural': 'Medications',
            },
        ),
        migrations.CreateModel(
            name='MoodLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.DateTimeField(auto_now_add=True, verbose_name='Time of Logging')),
                ('depressed_mood', models.IntegerField(choices=[(0, 'None'), (1, 'Mild Depression'), (2, 'Somewhat Depressed'), (3, 'Very Depressed'), (4, 'Extremely Depressed')], default=0, verbose_name="Today's Depressed Mood")),
                ('elevated_mood', models.IntegerField(choices=[(0, 'None'), (1, 'Mild Mood Elevation'), (2, 'Somewhat Elevated Mood'), (3, 'Very Elevated Mood'), (4, 'Extremely Elevated Mood')], default=0, verbose_name="Today's Elevated Mood")),
                ('irritability', models.IntegerField(choices=[(0, 'None'), (1, 'Mild'), (2, 'Moderate'), (3, 'Severe')], default=0, verbose_name="Today's Irritability")),
                ('anxiety', models.IntegerField(choices=[(0, 'None'), (1, 'Mild'), (2, 'Moderate'), (3, 'Severe')], default=0, verbose_name="Today's Anxiety")),
                ('hours_of_sleep', models.IntegerField(default=0, help_text='Enter the number of hours you slept.', verbose_name='Hours of Sleep')),
                ('energy_level', models.IntegerField(default=5, help_text='Energy level from 0 (lowest) to 10 (highest).', validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(10)], verbose_name='Energy Level')),
                ('symptoms', models.CharField(blank=True, choices=[('sadness_despair', 'Sadness/Despair'), ('helplessness_hopelessness', 'Helplessness/Hopelessness'), ('agitation_irritability', 'Agitation/Irritability'), ('social_withdrawal', 'Social Withdrawal'), ('low_motivation', 'Low Motivation'), ('low_self_esteem', 'Low Self-Esteem'), ('high_anxiety_excessive_worry', 'High Anxiety or Excessive Worry'), ('sleep_problems', 'Sleep Problems'), ('headache', 'Headache'), ('body_ache_pain', 'Body Ache/Pain'), ('decreased_increased_appetite', 'Decreased or Increased Appetite'), ('feelings_of_guilt_self_blame', 'Feelings of Guilt or Self-Blame'), ('thoughts_of_death_suicide', 'Thoughts of Death or Suicide')], max_length=255, verbose_name='Symptoms')),
            ],
            options={
                'verbose_name': 'Mood Log',
                'verbose_name_plural': 'Mood Logs',
                'ordering': ['-timestamp'],
            },
        ),
    ]
