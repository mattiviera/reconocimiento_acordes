import librosa
import numpy as np
import os
from .utils import ensure_dir

def preprocess_audio(file_path, duration=3, sr=22050, n_mels=128):
    signal, sr = librosa.load(file_path, sr=sr, duration=duration)
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Asegurar forma fija
    # Forzamos el número de bins a 129 para consistencia con predict.py
    target_shape = (n_mels, 129) # Fijo en 129
    if mel_spec_db.shape[1] < target_shape[1]:
        # Rellenar con ceros
        pad_width = target_shape[1] - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0,0), (0, pad_width)), mode='constant')
    elif mel_spec_db.shape[1] > target_shape[1]:
        # Recortar
        mel_spec_db = mel_spec_db[:, :target_shape[1]]
    return mel_spec_db

def load_data(data_dir="data/raw/"):
    X = []
    y = []
    labels = {}

    idx = 0
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        labels[folder] = idx
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                file_path = os.path.join(folder_path, file)
                spec = preprocess_audio(file_path)
                X.append(spec)
                y.append(idx)
        idx += 1
    return np.array(X), np.array(y), labels