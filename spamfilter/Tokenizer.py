from collections import defaultdict, Iterable
from typing import Union

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
morph = MorphAnalyzer()


class Tokenizer(object):

    def __init__(self,
                 navec_path:str,  #='../../../../ExtractSpamMails/navec_hudlit_v1_12B_500K_300d_100q.tar',
                 dimensions:int):
        self.dimensions = dimensions
        self.navec = Navec.load(navec_path)
        self.pattern=r"[^А-я-]+"
        self.stopwords=stopwords.words('russian')
        self._tokenizer:Tknzer=Tknzer()

    @property
    def word_index(self):
        return self._tokenizer.word_index

    def lemmatize(self,doc):
        doc = re.sub(self.pattern, " ", doc)
        tokens = []
        for token in doc.split():
            if token and token not in self.stopwords:
                token = token.strip()
                token = morph.normal_forms(token)[0]
                if token not in self.stopwords:
                    tokens.append(token)
        # if len(tokens) >= 1:
        return " ".join(tokens)
        # return None

    def fit_tokenizer(self,data:Series):
        data=data.apply(self.lemmatize)
        self._tokenizer.fit_on_texts(data)

    def get_embedding_matrix(self, dim=300):
        word_index=self._tokenizer.word_index
        embedding_matrix=np.zeros((len(word_index)+1,dim))
        for word,index in word_index.items():
            if word in self.navec:
                embedding_matrix[index]=self.navec[word]
        return embedding_matrix

    def get_sequences(self,data):
        """

        :type data: Union[pd.Series,pd.DataFrame,Iterable]
        """
        if isinstance(data,(pd.Series,pd.DataFrame)):
            data=data.apply(self.lemmatize)
        elif isinstance(data,Iterable):
            data=[self.lemmatize(elem) for elem in data]
        else:
            raise Exception("Неожиданный тип data")
        seq=self._tokenizer.texts_to_sequences(data)
        return keras.preprocessing.sequence.pad_sequences(seq)

    # def main():
    #     df = pd.read_csv("../../../../ExtractSpamMails/myspam.csv")
    #     print(df.iloc[0:3, :])
    #     # patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
    #     df['subject'] = df['subject'].apply(lemmatize)
    #     df['text'] = df['text'].apply(lemmatize)
    #
    #     # path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
    #     # navec = Navec.load(path)
    #     tokenizer=keras.preprocessing.text.Tokenizer()
    #     tokenizer.fit_on_texts(df['subject']+df['text'])
    #     df['subject']=tokenizer.texts_to_sequences(df['subject'])
    #     subject=keras.preprocessing.sequence.pad_sequences(df['subject'])
    #     df['text']=tokenizer.texts_to_sequences(df['text'])
    #     text=keras.preprocessing.sequence.pad_sequences(df['text'])

        # w2v_model = Word2Vec(
        #     min_count=10,
        #     window=2,
        #     vector_size=300,
        #     negative=10,
        #     alpha=0.03,
        #     min_alpha=0.0007,
        #     sample=6e-5,
        #     sg=1)
        # w2v_model.build_vocab(df['text'])
        # w2v_model.build_vocab(df['subject'],update=True)
        # w2v_model.init_sims(replace=True)
        # print(w2v_model.wv.most_similar(positive=["скидка"]))


        # word_freq = defaultdict(int)
        # for tokens in df['subject']:
        #     if tokens:
        #         for token in tokens:
        #             word_freq[token] += 1
        #
        # print(len(word_freq))
        #
        # print(sorted(word_freq, key=word_freq.get, reverse=True)[:10])
        #
        # word_freq = defaultdict(int)
        # for tokens in df['text']:
        #     if tokens:
        #         for token in tokens:
        #             word_freq[token] += 1
        #
        # print(len(word_freq))
        #
        # print(sorted(word_freq, key=word_freq.get, reverse=True)[:10])

