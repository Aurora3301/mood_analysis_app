# Generated by Django 4.2.18 on 2025-02-22 16:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mood_tracker', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Symptom',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(choices=[('sadness_despair', 'Sadness/Despair'), ('helplessness_hopelessness', 'Helplessness/Hopelessness'), ('agitation_irritability', 'Agitation/Irritability'), ('social_withdrawal', 'Social Withdrawal'), ('low_motivation', 'Low Motivation'), ('low_self_esteem', 'Low Self-Esteem'), ('high_anxiety_excessive_worry', 'High Anxiety or Excessive Worry'), ('sleep_problems', 'Sleep Problems'), ('headache', 'Headache'), ('body_ache_pain', 'Body Ache/Pain'), ('decreased_increased_appetite', 'Decreased or Increased Appetite'), ('feelings_of_guilt_self_blame', 'Feelings of Guilt or Self-Blame'), ('thoughts_of_death_suicide', 'Thoughts of Death or Suicide')], max_length=50, unique=True, verbose_name='Symptom Name')),
            ],
            options={
                'verbose_name': 'Symptom',
                'verbose_name_plural': 'Symptoms',
            },
        ),
        migrations.RemoveField(
            model_name='moodlog',
            name='symptoms',
        ),
        migrations.AddField(
            model_name='moodlog',
            name='symptoms',
            field=models.ManyToManyField(blank=True, to='mood_tracker.symptom', verbose_name='Symptoms'),
        ),
    ]
