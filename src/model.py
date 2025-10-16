import tensorflow as tf
from keras import layers, models

def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),  # Nueva capa
        layers.MaxPooling2D((2, 2)),                   # Nueva capa
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),          # Más neuronas
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model