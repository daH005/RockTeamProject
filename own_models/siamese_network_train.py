from pathlib import Path
from random import choice
import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from ..config import DB_PATH


def train_siamese_network_train(data_dir):
    img_size = (180, 180)
    input_shape = (*img_size, 3)

    batch_size = 32
    num_epochs = 20

    input_layer = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), activation='relu')(input_layer)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    base_network = Model(inputs=input_layer, outputs=x)

    left_input = tf.keras.Input(shape=(*img_size, 3), name='left_input')
    right_input = tf.keras.Input(shape=(*img_size, 3), name='right_input')

    processed_left = base_network(left_input)
    processed_right = base_network(right_input)

    def euclidean_distance(vectors):
        x, y = vectors
        return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True))

    distance_layer = layers.Lambda(euclidean_distance)([processed_left, processed_right])
    outputs = layers.Dense(1, activation='sigmoid')(distance_layer)

    model = Model(inputs=[left_input, right_input], outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    train_generator, train_steps, validation_generator, validation_steps = _make_train_and_validation_generators(
        data_dir,
        batch_size=batch_size,
        validation_split=0.2,
        target_size=img_size,
    )
    history = model.fit(
        train_generator,
        epochs=num_epochs,
        steps_per_epoch=train_steps,
        validation_data=validation_generator,
        validation_steps=validation_steps,
    )
    model.save('siamese_model.h5')


def _make_train_and_validation_generators(data_dir,
                                          batch_size: int,
                                          validation_split: float,
                                          target_size,
                                          ):
    data_dir = Path(data_dir)

    data = []
    for person_dir in data_dir.iterdir():
        if not person_dir.is_dir():
            continue

        for face_path in person_dir.iterdir():
            for face_path2 in person_dir.iterdir():
                data.append(((face_path, face_path2), 1))

            for face_path2 in choice([d for d in data_dir.iterdir() if d != person_dir and d.is_dir()]).iterdir():
                data.append(((face_path, face_path2), 0))

    print('Количество пар:', len(data))
    validation_start_index: int = int((len(data) - 1) * (1 - validation_split))

    def data_generator(start_index, end_index):
        while True:
            batch_input1 = []
            batch_input2 = []
            batch_labels = []
            for i, record in enumerate(data[start_index:end_index]):
                batch_input1.append(_load_image(record[0][0]))
                batch_input2.append(_load_image(record[0][1]))
                batch_labels.append(record[1])

                if len(batch_labels) % batch_size == 0 or i == end_index - 1:
                    yield (np.array(batch_input1), np.array(batch_input2)), np.array(batch_labels)

                    batch_input1.clear()
                    batch_input2.clear()
                    batch_labels.clear()

    def _load_image(path):
        img = load_img(path, target_size=target_size)
        return img_to_array(img) / 255.0

    train_steps = math.ceil(validation_start_index / batch_size)
    validation_steps = math.ceil((len(data) - validation_start_index) / batch_size)
    return data_generator(0, validation_start_index), train_steps, data_generator(validation_start_index, len(data)), validation_steps


if __name__ == '__main__':
    train_siamese_network_train(DB_PATH)
