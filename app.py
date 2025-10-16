# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import librosa
import tensorflow as tf
import soundfile as sf
import io
import os


from src.utils import load_labels

# --- Inicializar Flask ---
app = Flask(__name__)

# --- Cargar modelo y etiquetas al iniciar la app ---
MODEL_PATH = 'models/modelo_entrenado.h5' 
LABELS_PATH = 'data/labels.json'

print("Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo cargado.")

print("Cargando etiquetas...")
labels = load_labels(LABELS_PATH)
inv_labels = {v: k for k, v in labels.items()} # Invertir para predicción
print("Etiquetas cargadas:", inv_labels)

# --- Función para preprocesar audio desde bytes (lo que recibiremos desde el frontend) ---
def preprocess_audio_from_bytes(audio_bytes, sr=22050, duration=3, n_mels=128):
    # Cargar audio desde bytes
    # Usamos soundfile para leer desde un buffer en memoria (más robusto para WAV)
    # Si soundfile falla, usamos librosa.load que puede manejar más formatos desde BytesIO
    audio_io = io.BytesIO(audio_bytes)

    # Opción 1: Intentar con soundfile (funciona bien con WAV y algunos otros)
    try:
        signal, file_sr = sf.read(audio_io)
        # Si el audio es multicanal, convertir a mono
        if signal.ndim > 1:
            signal = signal.mean(axis=1)
        # Re-muestrear si es necesario
        if file_sr != sr:
            signal = librosa.resample(y=signal, orig_sr=file_sr, target_sr=sr)
    except Exception as e:
        # Si soundfile falla, intentar con librosa (más formatos)
        # Reiniciar el puntero del BytesIO
        print(f"SoundFile falló ({e}), intentando con librosa...")
        audio_io.seek(0)
        signal, file_sr = librosa.load(audio_io, sr=sr, duration=duration)
        # librosa ya convierte a mono y resamplea internamente si sr es distinto

    # Ajustar la duración si librosa no lo hizo (por ejemplo, si usamos soundfile)
    if len(signal) > duration * sr:
        signal = signal[: int(duration * sr)]
    elif len(signal) < duration * sr:
        # Opcional: rellenar con ceros si es más corto
        target_len = int(duration * sr)
        pad_len = target_len - len(signal)
        signal = np.pad(signal, (0, pad_len), mode='constant')

    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Garantizar la misma forma fija que en entrenamiento (129 bins)
    target_shape = (n_mels, 129) # Fijo en 129

    if mel_spec_db.shape[1] < target_shape[1]:
        pad_width = target_shape[1] - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0,0), (0, pad_width)), mode='constant')
    elif mel_spec_db.shape[1] > target_shape[1]:
        mel_spec_db = mel_spec_db[:, :target_shape[1]]

    # Agregar dimensión del batch
    return np.expand_dims(mel_spec_db, axis=0)


# --- Ruta para servir la página principal ---
@app.route('/')
def index():
    return render_template('index.html')


# --- Ruta para manejar la predicción ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Recibir archivo de audio
        if 'audio' not in request.files:
            return jsonify({'error': 'No se envió ningún archivo de audio'}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'Nombre de archivo vacío'}), 400

        # Leer el contenido binario del archivo
        audio_bytes = audio_file.read()

        # 2. Preprocesar el audio
        spec = preprocess_audio_from_bytes(audio_bytes)

        # 3. Hacer la predicción
        pred = model.predict(spec, verbose=0) # verbose=0 para no mostrar progreso
        chord_idx = np.argmax(pred)
        confidence = float(pred[0][chord_idx]) # Convertir a tipo Python para JSON
        chord = inv_labels[chord_idx]

        # 4. Devolver la predicción como JSON
        return jsonify({'chord': chord, 'confidence': confidence})

    except Exception as e:
        print(f"Error en la predicción: {e}") # Imprime el error en la consola del servidor
        return jsonify({'error': f'Error interno del servidor: {str(e)}'}), 500


# --- Punto de entrada ---
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)