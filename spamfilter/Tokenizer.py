from collections import defaultdict, Iterable
from typing import Union, List

import keras
from keras.preprocessing.text import Tokenizer as Tknzer
import keras.preprocessing.sequence
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import nltk
# nltk.download("stopwords")
import re

from navec import Navec
from pandas import Series
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords


# patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"


class Tokenizer(object):
    navec_dimensions = 300
    pattern = r"[^А-я-]+"

    def __init__(self,
                 navec_path: str):
        # self.sequence_length = sequence_length
        self.navec = Navec.load(navec_path)
        self.stopwords = stopwords.words('russian')
        self.morph = MorphAnalyzer()

        self._tokenizer: Tknzer = Tknzer()

    @property
    def word_index(self):
        return self._tokenizer.word_index

    def lemmatize(self, doc):
        doc = re.sub(self.pattern, " ", str(doc))
        tokens = []
        for token in doc.split():
            if token and token not in self.stopwords:
                token = token.strip()
                token = self.morph.normal_forms(token)[0]
                if token not in self.stopwords:
                    tokens.append(token)
        # if len(tokens) >= 1:
        return " ".join(tokens)
        # return None

    def fit_tokenizer(self, data: Union[Series, List[str]]):
        self._tokenizer.fit_on_texts(data)

    def get_embedding_matrix(self):
        word_index = self._tokenizer.word_index
        embedding_matrix = np.zeros((len(word_index) + 1, self.navec_dimensions))
        for word, index in word_index.items():
            if word in self.navec:
                embedding_matrix[index] = self.navec[word]
        return embedding_matrix

    def lemmatize_texts(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.applymap(self.lemmatize)
        elif isinstance(data, pd.Series):
            data = data.apply(self.lemmatize)
        elif isinstance(data, Iterable):
            data = [self.lemmatize(elem) for elem in data]
        else:
            raise Exception("Неожиданный тип data")
        return data

    def texts_to_sequences(self, data, pad_length=100):
        """

        :type data: Union[pd.Series,pd.DataFrame,Iterable]
        """
        seq = self._tokenizer.texts_to_sequences(data)
        return keras.preprocessing.sequence.pad_sequences(seq, maxlen=pad_length)

