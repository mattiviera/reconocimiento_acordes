import pyaudio
import numpy as np
import librosa
import tensorflow as tf
from .utils import load_labels
import os
import soundfile as sf  

def record_audio(seconds=3, rate=22050):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=rate,
                    input=True,
                    frames_per_buffer=1024)
    print("Grabando...")
    frames = []
    for i in range(0, int(rate / 1024 * seconds)):
        data = stream.read(1024)
        frames.append(data)
    print("Fin de grabación.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    audio = np.frombuffer(b''.join(frames), dtype=np.float32)
    return audio

def preprocess_live_audio(signal, sr=22050, duration=3, n_mels=128):
    # Aplicar el mismo proceso que en preprocess.py
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Garantizar la misma forma fija que en preprocess_audio (129 bins)
    target_shape = (n_mels, 129) # Fijo en 129

    if mel_spec_db.shape[1] < target_shape[1]:
        pad_width = target_shape[1] - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0,0), (0, pad_width)), mode='constant')
    elif mel_spec_db.shape[1] > target_shape[1]:
        mel_spec_db = mel_spec_db[:, :target_shape[1]]

    # Agregar dimensión del batch
    return np.expand_dims(mel_spec_db, axis=0)

def save_new_audio(audio, chord_name, sr=22050):
    folder = f"data/raw/{chord_name}/"
    os.makedirs(folder, exist_ok=True)
    filename = f"{chord_name}_{len(os.listdir(folder)) + 1}.wav"
    filepath = os.path.join(folder, filename)
    sf.write(filepath, audio, samplerate=sr)  # Corregido: 'samplerate' en lugar de 'sr'
    print(f"Audio guardado en: {filepath}")

def predict_chord(model_path='models/modelo_entrenado.h5'):
    if not os.path.exists(model_path):
        print("❌ No se encontró un modelo entrenado. Entrena uno primero.")
        return

    model = tf.keras.models.load_model(model_path)
    labels = load_labels()
    inv_labels = {v: k for k, v in labels.items()}

    audio = record_audio()
    spec = preprocess_live_audio(audio) # Ahora debería tener forma (1, 128, 129)
    pred = model.predict(spec)
    chord_idx = np.argmax(pred)
    confidence = pred[0][chord_idx]

    if confidence < 0.7:  # Umbral de confianza
        print(f"⚠️ Confianza baja: {confidence:.2f}. ¿Quieres etiquetar este audio como nuevo acorde?")
        save = input("¿Guardar nuevo acorde? (s/n): ")
        if save.lower() == 's':
            chord_name = input("Nombre del acorde (por ejemplo: Em, F, etc.): ")
            save_new_audio(audio, chord_name)
    else:
        chord = inv_labels[chord_idx]
        print(f"Predicción: {chord} (Confianza: {confidence:.2f})")