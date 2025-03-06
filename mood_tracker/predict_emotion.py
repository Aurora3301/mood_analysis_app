import librosa
import numpy as np
from tensorflow.keras.models import load_model
import sounddevice as sd

# Parameters
sr = 16000
duration = 20  # 20-second recordings
channels = 1
max_frames = 200
emotion_labels = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "ps", 5: "sad", 6: "neutral"}

def record_audio(duration, sr):
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=channels, dtype='float32')
    sd.wait()
    print("Recording finished.")
    return audio.flatten()

def real_time_prediction():
    model = load_model("ser_model.h5")
    segment_duration = 2
    segment_samples = int(segment_duration * sr)

    while True:
        audio = record_audio(duration, sr)
        audio = audio / np.max(np.abs(audio))

        num_segments = int(duration / segment_duration)
        segments = [audio[i * segment_samples:(i + 1) * segment_samples] 
                   for i in range(num_segments)]

        predictions = []
        for segment in segments:
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
            if mfcc.shape[1] > max_frames:
                mfcc = mfcc[:, :max_frames]
            else:
                padding = max_frames - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode='constant')

            mfcc_input = mfcc[np.newaxis, ..., np.newaxis]
            pred = model.predict(mfcc_input, verbose=0)
            predictions.append(pred[0])

        avg_pred = np.mean(predictions, axis=0)
        emotion_idx = np.argmax(avg_pred)
        emotion = emotion_labels[emotion_idx]
        confidence = avg_pred[emotion_idx]

        print(f"Predicted Emotion: {emotion} (Confidence: {confidence:.2f})")
        cont = input("Press Enter to record again, or 'q' to quit: ")
        if cont.lower() == 'q':
            break

if __name__ == "__main__":
    real_time_prediction()