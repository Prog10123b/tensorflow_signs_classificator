# v1.0.0
# tf.keras и numpy
from tensorflow import keras
# for cnn
from tensorflow.keras import datasets, layers, models
import numpy as np

class classificator:
    def __init__(self):
        self.dataset = None
        self.labels = ['stop', 'only-right', 'only-left', 'only-straight', 'straight-right', 'straight-left']
        self.use_aliases = True
        self.model = None
        print('[INFO] Image classificator prepairing done.')
        print('[WARN] No model! Create the module to use classificator!')

    def create_perceptron(self, input_shape, hidden):
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=input_shape),    # input shape is a shape of image. Ex. (28, 28)
            keras.layers.Dense(hidden, activation='relu'),    # hidden is a number of neurons in the hidden layer
            keras.layers.Dense(6, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',  # sparse_categorical_crossentropy
                           metrics=['accuracy'])
        print(f'[INFO] Perceptron is created! Input shape is {input_shape}')

    def create_cnn(self, epochs):
        model = models.Sequential()                                                          # creating standart tensorflow cnn
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(6))

        model.compile(optimizer='adam',                                                        # compilation
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        history = model.fit(self.dataset[0][0], self.dataset[0][1], epochs=epochs,             # training
                            validation_data=(self.dataset[1][0], self.dataset[1][1]))
        self.model = model

        return history


    def waste_neural_network(self, agree):
        if not agree:
            prompt = input('[WARN] Your model will be wasted! Are you sure? [y/n]')
            if prompt.lower() == 'y':
                self.model = None
            else:
                return
        else:
            self.model = None

    def make_prediction(self, image):
        predictions = self.model.predict((image,))
        prediction = np.argmax(predictions[0])

        if self.use_aliases and self.labels is not None:
            prediction = self.labels[np.argmax(predictions[0])]

        return prediction

    def load_model_weights(self, filepath):
        self.model.load_weights(filepath=filepath)
        print('[INFO] Model weights loaded.')

    def save_model_weights(self, filepath):
        self.model.save_weights(filepath=filepath)
        print('[INFO] Model weights saved.')

    def load_entire_model(self, filepath):
        self.model = models.load_model(filepath)
        print('[INFO] Entire model loaded.')

    def save_entire_model(self, filepath):
        self.model.save(filepath)
        print('[INFO] Entire model saved.')

    def train_model(self, epochs):
        if self.dataset is not None and self.model is not None:
            self.model.fit(self.dataset[0][0], self.dataset[0][1], epochs=epochs)
        else:
            print('[ERROR] Dataset is empty or no model.')
            raise ValueError('Error! Empty dataset or model!')

    def set_labels(self, labels):
        self.labels = labels
