import sqlite3
from models.emotion_model import Emotion

DATABASE = 'data/emotions.db'

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        
        # Drop old tables if they exist
        c.execute('DROP TABLE IF EXISTS emotions')
        c.execute('DROP TABLE IF EXISTS mood_entries')
        c.execute('DROP TABLE IF EXISTS symptoms')
        c.execute('DROP TABLE IF EXISTS mood_symptoms')
        
        # Create new tables
        c.execute('''
            CREATE TABLE IF NOT EXISTS emotions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                emotion TEXT CHECK(emotion IN (
                    'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'
                )) NOT NULL,
                percentage REAL NOT NULL
            )
        ''')
        conn.commit()

        c.execute('''CREATE TABLE mood_entries
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      date TEXT,
                      mood TEXT,
                      energy INTEGER)''')
        
        c.execute('''CREATE TABLE symptoms
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT)''')
        
        c.execute('''CREATE TABLE mood_symptoms
                     (mood_id INTEGER,
                      symptom_id INTEGER,
                      FOREIGN KEY(mood_id) REFERENCES mood_entries(id),
                      FOREIGN KEY(symptom_id) REFERENCES symptoms(id))''')
        
        # Pre-populate symptoms
        c.executemany('INSERT INTO symptoms (name) VALUES (?)',
                     [('Anxiety',), ('Fatigue',), ('Headache',)])
        
        conn.commit()

def insert_mood_entry(date, mood, energy, symptom_ids=None):
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        # Insert mood/energy
        c.execute('INSERT INTO mood_entries (date, mood, energy) VALUES (?, ?, ?)',
                  (date, mood, energy))
        mood_id = c.lastrowid
        
        # Link symptoms (convert to integers)
        if symptom_ids:
            for symptom_id in symptom_ids:
                c.execute('INSERT INTO mood_symptoms (mood_id, symptom_id) VALUES (?, ?)',
                          (mood_id, int(symptom_id)))
        conn.commit()

# Existing functions for emotion detection

def insert_emotion(date: str, emotion_data):
    """
    Insert emotion data into the database.
    Args:
        date (str): The timestamp of the data.
        emotion_data: Can be a single emotion (str) or a dictionary of emotions and their percentages.
    """
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        if isinstance(emotion_data, dict):
            # Handle dictionary of emotions and percentages
            for emotion, percentage in emotion_data.items():
                c.execute(
                    "INSERT INTO emotions (date, emotion, percentage) VALUES (?, ?, ?)",
                    (date, emotion, percentage)
                )
        else:
            # Handle single emotion string
            c.execute(
                "INSERT INTO emotions (date, emotion, percentage) VALUES (?, ?, ?)",
                (date, emotion_data, 100.0)  # Assume 100% for single emotions
            )
        conn.commit()

def get_frequent_emotions():
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute('''SELECT emotion, COUNT(emotion) as frequency
                     FROM emotions
                     GROUP BY emotion
                     ORDER BY frequency DESC''')
        return c.fetchall()

def get_daily_records():
    with sqlite3.connect(DATABASE) as conn:
        c = conn.cursor()
        c.execute('''SELECT date, emotion
                     FROM emotions
                     ORDER BY date DESC''')
        return [Emotion(date=row[0], emotion=row[1]) for row in c.fetchall()]