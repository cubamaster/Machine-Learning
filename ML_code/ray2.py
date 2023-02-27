import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import ray

ray.init()

(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()

chunk_size = len(x_train_full) // 3
x_train_chunks = [x_train_full[i:i+chunk_size] for i in range(0, len(x_train_full), chunk_size)]
y_train_chunks = [y_train_full[i:i+chunk_size] for i in range(0, len(y_train_full), chunk_size)]

def create_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


@ray.remote
def train_chunk(x_chunk, y_chunk):
    model = create_model()
    model.fit(x_chunk, y_chunk, epochs=10, validation_split=0.2)
    return model


models = ray.get([train_chunk.remote(x_train_chunks[i], y_train_chunks[i]) for i in range(3)])

combined_model = create_model()
for model in models:
    combined_model.set_weights(model.get_weights())

loss, accuracy = combined_model.evaluate(x_test, y_test)
print('Test accuracy:', accuracy)