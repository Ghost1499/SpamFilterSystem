import os
from typing import List

import pandas as pd
import numpy as np
import sklearn
from matplotlib import pyplot
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from spamfilter.Tokenizer import Tokenizer
from spamfilter.classifiers.LSTMClassifier import LSTMClassifier
from spamfilter.classifiers.NaiveBayesClassifier import NaiveBayesClassifier
from spamfilter.classifiers.utils import label2int, int2label
from spamfilter.Extractor import Extractor
from email_system.NecessaryEmail import NecessaryEmail
from spamfilter.utils.utils import *

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


class DataPrepareEngine:

    def __init__(self, recieve_mail_system, spam_folder="emails/spam/", ham_folder="emails/ham/",
                 navec_path="navec_hudlit_v1_12B_500K_300d_100q.tar", lemmatized_path="lemmatized.csv",data_path='data'):
        self.recieve_mail_system = recieve_mail_system
        self.spam_folder = spam_folder
        self.ham_folder = ham_folder
        self.navec_path = navec_path
        self.lemmatized_path = lemmatized_path
        # self.test_size = test_size
        self._extractor = Extractor(recieve_mail_system)
        self._tokenizer = Tokenizer(navec_path)
        self.data_path = data_path

    def _get_data_path(self, filename: str):
        return make_path([self.data_path],filename)

    def lemmatize(self, is_load_lemmatize=False):
        if is_load_lemmatize and os.path.exists(self.lemmatized_path):
            lemmatized = pd.read_csv(self.lemmatized_path)
        else:
            df = self._extract()
            df.to_csv("dataset.csv")
            lemmatized = self._lemmatize(df)
            lemmatized.to_csv(self.lemmatized_path)
        return lemmatized

    def _extract(self):
        my_spam = list(self._extractor.from_mbox(self.spam_folder + "Myspam.mbox", True))
        my_spam_df = self._extractor.get_dataframe(my_spam)

        spam_df = pd.read_excel(self.spam_folder + "Pisma_spam.xlsx", index_col=None)
        spam_df['label'] = 'spam'

        ham_df = pd.read_csv(self.ham_folder + "ham.CSV")
        ham_df['label'] = "ham"

        df = pd.concat([my_spam_df, spam_df, ham_df], join="inner", ignore_index=True)
        df = df.sample(frac=1)
        return df

    def _lemmatize(self, df):
        df['subject'] = self._tokenizer.lemmatize_texts(df['subject'])
        df['text'] = self._tokenizer.lemmatize_texts(df['text'])
        return df

    def prepare_data(self, df: pd.DataFrame):
        subject = self._split_df(df, 'subject')
        subject = self._drop_empty(subject)
        text = self._split_df(df, 'text')
        text = self._drop_empty(text)
        subject: pd.DataFrame
        text: pd.DataFrame
        subject = subject.drop_duplicates()
        text = text.drop_duplicates()
        return subject, text

    def fit_tokenizer(self, subject, text):
        self._tokenizer.fit_tokenizer(subject['subject'])
        self._tokenizer.fit_tokenizer(text['text'])
        emb_matrix = self._tokenizer.get_embedding_matrix()
        return emb_matrix
        # x = self._tokenizer.text_to_sequences(subject['subject'],dim)
        # y = [label2int[label] for label in subject['label']]
        # return x, y

    def texts_to_sequences(self, dim, texts, name=None):
        if isinstance(texts, pd.DataFrame):
            x = self._tokenizer.texts_to_sequences(texts[name], dim)
            y = [label2int[label] for label in texts['label']]
            return x, y
        else:
            x = self._tokenizer.texts_to_sequences(texts,dim)
            return x

    def _split_df(self, df, name):
        return df[[name, 'label']]

    def _drop_empty(self, df: pd.DataFrame):
        """

        @rtype: pd.Dataframe
        """
        df = df.replace("", np.nan)
        return df.dropna()

    def dataframe_statistics(self, df: pd.DataFrame, colname4hist: str):
        print(df.shape)
        df.info()
        print(df.describe())
        print(df.nunique())
        hist_data = df[colname4hist].str.count(" ") + 1
        label = f"Hist of words count in {colname4hist}"
        if colname4hist != 'text':
            pyplot.hist(hist_data, label=label)
        else:
            pyplot.hist(hist_data, bins=30, range=(0, 600), label=label)
        # pyplot.show()
        pyplot.savefig(self._get_data_path(label + '.png'))
        pyplot.clf()

        value_counts = df['label'].value_counts()
        label = f'Spam Ham bar in {colname4hist}'
        pyplot.bar(value_counts.index.tolist(), value_counts.values.tolist(), label=label)
        # pyplot.show()
        pyplot.savefig(self._get_data_path(label + '.png'))
        pyplot.clf()

    @staticmethod
    def train_test_split(x, y, test_size=0.2):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
        return x_train, x_test, y_train, y_test
