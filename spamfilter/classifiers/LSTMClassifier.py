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

from spamfilter.classifiers.utils import label2int, int2label
from spamfilter.Tokenizer import Tokenizer as MyTokenizer


class LSTMClassifier(object):
    tensorboard: TensorBoard
    _model = Sequential

    def __init__(self, batch_size=64, embedding_size=300,sequence_length=100, train_size=0.8, epochs=20,model_file = "model.json",weights_file = "checkpoint.h5"):
        self.model_file = model_file
        self.weights_file = weights_file
        # self.navec_path = navec_path
        self.model_file = "model.json"
        self.weights_file = "checkpoint.h5"
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.train_size = train_size
        self.epochs = epochs

    def build_model(self, embedding_matrix, lstm_units=128,save_model=True):
        """
        Constructs the model,
        Embedding vectors => LSTM => 2 output Fully-Connected neurons with softmax activation

        :type lstm_units: int Количество слоев
        :type embedding_data: pd.Series Данные для Embedding слоя
        """

        model: Sequential = Sequential()
        model.add(Embedding(len(embedding_matrix),
                            self.embedding_size,
                            weights=[embedding_matrix],
                            trainable=False,
                            input_length=self.sequence_length))

        model.add(LSTM(lstm_units, recurrent_dropout=0.2))
        model.add(Dropout(0.3))
        model.add(Dense(2, activation="softmax"))
        # compile as rmsprop optimizer
        # aswell as with recall metric

        # model.summary()
        self._model=model
        self._compile_model()
        if save_model:
            self.save_model()

    def _compile_model(self):
        self._model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                      metrics=["accuracy", keras_metrics.precision(), keras_metrics.recall()])

    def load_model(self):
        filename=self.model_file
        if not os.path.exists(filename):
            raise Exception("Указанный файл не существует")
        with open(filename, 'r') as f:
            self._model = model_from_json(f.read())
        self._compile_model()

    def load_weights(self):
        filename = self.weights_file
        if not os.path.exists(filename):
            raise Exception("Указанный файл не существует")
        self._model.load_weights(filename)

    def save_model(self,filename:str=None):
        if filename:
            model_file=filename
        else:
            model_file = self.model_file
        model_json = self._model.to_json()

        with open(model_file, 'w') as f:
            f.write(model_json)

    def train(self, x: pd.Series, y: pd.Series):
        """

        
        :rtype:
        :param x: Обучающие данные
        :param y: Выходные данные
        :return: tuple [loss,accuracy,precision,recall]
        """
        self.tensorboard = TensorBoard(f"logs/spam_classifier_{time.time()}", histogram_freq=0,
                                       write_graph=True, write_images=False)
        y = to_categorical(y)
        x_train, x_test, y_train, y_test = self._split_data(x, y)
        model_checkpoint = ModelCheckpoint(self.weights_file,
                                           monitor='accuracy',
                                           mode='max',
                                           save_best_only=True)

        # train the model
        self._model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        batch_size=self.batch_size, epochs=self.epochs,
                        callbacks=[self.tensorboard, model_checkpoint],
                        # callbacks=[self.tensorboard],
                        verbose=1)
        return self._get_statistics(x_test, y_test)

    @staticmethod
    def _split_data(x, y, train_size=0.8):
        return train_test_split(x, y, train_size=train_size, random_state=7)

    def _get_statistics(self, x_test, y_test):
        result = self._model.evaluate(x_test, y_test)
        # extract those
        loss = result[0]
        accuracy = result[1]
        precision = result[2]
        recall = result[3]

        return loss, accuracy, precision, recall

    def get_predictions(self, sequence):
        # get the prediction
        prediction = self._model.predict(sequence)[0]
        # one-hot encoded vector, revert using np.argmax
        return int2label[np.argmax(prediction)]
