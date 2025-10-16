import numpy as np
from .preprocess import load_data
from .model import create_model
from .utils import save_labels
from sklearn.model_selection import train_test_split
import tensorflow as tf

def train_model():
    X, y, labels = load_data()
    save_labels(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = X_train.shape[1:]
    num_classes = len(labels)

    model = create_model(input_shape, num_classes)

    # Compilamos SIN label_smoothing por compatibilidad
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', # Quitamos label_smoothing
                  metrics=['accuracy'])

    # Callback para detener si no mejora (Punto 2)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Ajustamos el número de épocas a un valor alto, ya que EarlyStopping lo detendrá
    model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stop])

    model.save('models/modelo_entrenado.h5')
    print("Modelo guardado en models/modelo_entrenado.h5")

if __name__ == "__main__":
    train_model()