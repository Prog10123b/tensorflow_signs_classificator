# tf.keras и numpy
from tensorflow import keras
import numpy as np

class classificator:
    def __init__(self):
        self.dataset = None
        self.labels = ['stop', 'only-right', 'only-left', 'only-straight', 'straight-right', 'straight-left']
        self.use_aliases = True

        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(6, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',   #sparse_categorical_crossentropy
                      metrics=['accuracy'])


    def make_prediction(self, image):
        predictions = self.model.predict((image,))
        prediction = np.argmax(predictions[0])

        if self.use_aliases and self.labels is not None:
            prediction = self.labels[np.argmax(predictions[0])]

        return prediction

    def load_model(self, filepath):
        self.model.load_weights(filepath=filepath)

    def save_model(self, filepath):
        self.model.save_weights(filepath=filepath)

    def train_model(self, epochs):
        if self.dataset is not None:
            self.model.fit(self.dataset[0][0], self.dataset[0][1], epochs=epochs)
        else:
            raise ValueError('Error! Empty dataset!')

    def set_labels(self, labels):
        self.labels = labels
