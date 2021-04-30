import os

import numpy as np
import keras_metrics  # for recall and precision metrics
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, Dropout, Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.saving.model_config import model_from_json
import time
import pandas as pd
import pickle

from .Classifier import Classifier


class LSTMClassifier(Classifier):
    tensorboard: TensorBoard
    _model = Sequential

    def __init__(self, navec_path, batch_size=64, embedding_size=300, train_size=0.8, epochs=20):
        super().__init__(navec_path, embedding_size)
        self.model_file = "model.json"
        self.weights_file = "checkpoint.h5"
        self.sequence_length = self.embedding_size
        self.batch_size = batch_size
        self.train_size = train_size
        self.epochs = epochs

    def set_up(self,**kwargs):
        if os.path.exists(self.model_file) and os.path.exists(self.weights_file):
            self.load_from_file()
        else:
            self._model=self._build_model(kwargs["embedding_data"])
            self._train(kwargs['x'], kwargs['y'], self.weights_file)

    def _build_model(self, embedding_data, lstm_units=128):
        """
        Constructs the model,
        Embedding vectors => LSTM => 2 output Fully-Connected neurons with softmax activation
        """

        self._tokenizer.fit_tokenizer(embedding_data)
        embedding_matrix = self._tokenizer.get_embedding_matrix()
        model: Sequential = Sequential()
        model.add(Embedding(len(self._tokenizer.word_index) + 1,
                            self.embedding_size,
                            weights=[embedding_matrix],
                            trainable=False,
                            input_length=self.sequence_length))

        model.add(LSTM(lstm_units, recurrent_dropout=0.2))
        model.add(Dropout(0.3))
        model.add(Dense(2, activation="softmax"))
        # compile as rmsprop optimizer
        # aswell as with recall metric
        model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                      metrics=["accuracy", keras_metrics.precision(), keras_metrics.recall()])
        # model.summary()
        return model

    def load_from_file(self):
        model_filename=self.model_file
        weights_filename=self.weights_file
        if not os.path.exists(model_filename):
            raise Exception("Model file does not exist")
        if not os.path.exists(weights_filename):
            raise Exception("Weights file does not exist")
        try:
            with open(model_filename, 'r') as f:
                self._model = model_from_json(f.read())
            self._model.load_weights(weights_filename)
        except:
            raise Exception("Can not load model or weights from file")

    def _train(self, x: pd.Series, y: pd.Series, weights_file="checkpoint.h5"):
        self.tensorboard = TensorBoard(f"logs/spam_classifier_{time.time()}", histogram_freq=0,
                                       write_graph=True, write_images=False)
        y=[self.label2int(i) for i in y]
        x_train, x_test, y_train, y_test = self._split_data(x, y)
        model_checkpoint = ModelCheckpoint(weights_file,
                                           monitor='acc',
                                           mode='max',
                                           save_best_only=True)

        # train the model
        self._model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        batch_size=self.batch_size, epochs=self.epochs,
                        callbacks=[self.tensorboard, model_checkpoint],
                        # callbacks=[self.tensorboard],
                        verbose=1)
        return self._get_statistics(x_test, y_test)

    def _split_data(self, x, y, train_size=80):
        return train_test_split(x, y, train_size=train_size, random_state=7)

    def _get_statistics(self, x_test, y_test):
        result = self._model.evaluate(x_test, y_test)
        # extract those
        loss = result[0]
        accuracy = result[1]
        precision = result[2]
        recall = result[3]

        return loss, accuracy, precision, recall

    def get_predictions(self, text):
        sequence = self._tokenizer.get_sequences([text])
        # get the prediction
        prediction = self._model.predict(sequence)[0]
        # one-hot encoded vector, revert using np.argmax
        return self.int2label[np.argmax(prediction)]
