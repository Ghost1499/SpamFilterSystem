import json
import os
import time
from collections import Iterable
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
from spamfilter.DataPrepareEngine import DataPrepareEngine
from spamfilter.TrainEngine import *
from spamfilter.TestEngine import *
from email_system.NecessaryEmail import NecessaryEmail
from spamfilter.utils.utils import *

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


class SpamFilter(object):

    def __init__(self, recieve_mail_system):
        self.recieve_mail_system = recieve_mail_system
        self._subject_seq_length = 20
        self._text_seq_length = 100
        self.data_path = "data"
        self._data_prepare = DataPrepareEngine(None, data_path=self.data_path)

        data_prepare = self._data_prepare
        lemmatized = data_prepare.lemmatize(True)
        subject, text = data_prepare.prepare_data(lemmatized)
        emb_matrix = data_prepare.fit_tokenizer(subject, text)

        subject=self._equalize_classes(subject)
        text=self._equalize_classes(text)

        data_prepare.dataframe_statistics(subject, 'subject')
        data_prepare.dataframe_statistics(text, 'text')

        x_subject, y_subject = data_prepare.texts_to_sequences(self._subject_seq_length, subject, 'subject')
        x_text, y_text = data_prepare.texts_to_sequences(self._text_seq_length, text, 'text')

        x_subject_train, x_subject_test, y_subject_train, y_subject_test = train_test_split(x_subject, y_subject,
                                                                                            test_size=0.1)
        x_text_train, x_text_test, y_text_train, y_text_test = train_test_split(x_text, y_text, test_size=0.1)

        self._subject_LSTM = LSTMClassifier(sequence_length=self._subject_seq_length, model_name='Subject LSTM')
        self._body_LSTM = LSTMClassifier(sequence_length=self._text_seq_length, model_name='Body LSTM')

        self._subject_classifiers = [(NaiveBayesClassifier(), 'Subject Naive Bayes'),
                                     (SVC(), 'Subject SVC'),
                                     (KNeighborsClassifier(n_neighbors=18), 'Subject KNN'),
                                     (DecisionTreeClassifier(), "Subject DecTree")]
        self._body_classifiers = [(NaiveBayesClassifier(), 'Body Naive Bayes'),
                                  (SVC(), 'Body SVC'),
                                  (KNeighborsClassifier(n_neighbors=18), 'Body KNN'),
                                  (DecisionTreeClassifier(), 'Body DecTree')]

        self._fit_clsfr(self._subject_classifiers, x_subject_train, y_subject_train)
        print(f'Training {self._subject_LSTM.model_name}')
        train_lstm(self._subject_LSTM, x_subject_train, y_subject_train, emb_matrix, validation_size=0.1,
                   is_load_model=False,
                   is_load_weights=False)
        self._fit_clsfr(self._body_classifiers, x_text_train, y_text_train)
        print(f'Training {self._body_LSTM.model_name}')
        train_lstm(self._body_LSTM, x_text_train, y_text_train, emb_matrix, validation_size=0.1,
                   is_load_model=False,
                   is_load_weights=False)

        self._test(x_subject_test, y_subject_test,x_text_test, y_text_test)

    @staticmethod
    def _equalize_classes(df:pd.DataFrame):
        spam = df[df['label'] == 'spam']
        spam_count = spam.shape[0]
        ham = df[df['label'] == 'ham'].sample(n=spam_count)
        df = pd.concat([spam, ham], ignore_index=True).sample(frac=1)
        return df

    def _test(self, x_subject_test, y_subject_test, x_text_test, y_text_test):
        if not os.path.isdir(self.data_path):
            os.mkdir(self.data_path)
        os.chdir(self.data_path)
        self.statistics_path = f'statistics_{time.ctime(time.time())}'.replace(' ', '_').replace(':', '-')
        os.mkdir(self.statistics_path)
        os.chdir('..')
        self._test_clsfr(self._subject_classifiers, x_subject_test, y_subject_test)
        self._test_clsfr([(self._subject_LSTM, self._subject_LSTM.model_name)], x_subject_test, y_subject_test,
                         method=test_lstm)
        self._test_clsfr(self._body_classifiers, x_text_test, y_text_test)
        self._test_clsfr([(self._body_LSTM, self._body_LSTM.model_name)], x_text_test, y_text_test, method=test_lstm)

    def _fit_clsfr(self, classifiers: Iterable, x, y):
        for classifier, name in classifiers:
            print(f'Fitting {name}')
            fit_classifier(classifier, x, y)

    def _test_clsfr(self, classifiers: Iterable, x, y, method=None):
        if method is None:
            method = test_classifier
        for classifier, name in classifiers:
            result = method(classifier, x, y, name)
            print(result)
            with open(make_path([self.data_path, self.statistics_path], name + '.txt'), 'a') as file:
                file.write(result)

    # def classify(self, emails: List[NecessaryEmail]):
    #     classified = []
    #     self.subject_classifier.fit_tokenizer([email.prepared_subejct for email in emails])
    #     for email in emails:
    #         classified.append([email, self.subject_classifier.get_predictions(
    #             email.prepared_subejct)])  # добавить другие результаты классификации
    #     return classified

    def classify(self, subjects: Iterable, texts: Iterable):
        subj_s = self._data_prepare.texts_to_sequences(self._subject_seq_length, subjects)
        txt_s = self._data_prepare.texts_to_sequences(self._text_seq_length, texts)
        preds = []
        for clsfr, name in [*self._subject_classifiers, (self._subject_LSTM, self._subject_LSTM.model_name)]:
            preds.append((clsfr.predict(subj_s), name))
        for clsfr, name in [*self._body_classifiers, (self._body_LSTM, self._body_LSTM.model_name)]:
            preds.append((clsfr.predict(txt_s), name))
        return preds
        # return int2label[self._subject_NB.predict(subj_s[0])], int2label[self._subject_LSTM.predict(subj_s)], \
        #        int2label[
        #            self._text_NB.predict(txt_s[0])], int2label[self._body_LSTM.predict(txt_s)]


def print_results(mess):
    subj,body=mess
    print(f'Тема: {subj[0]}')
    print(f'Текст: {body[0]}')
    m_res = spam_filter.classify(*mess)
    for res, name in m_res:
        if isinstance(res, Iterable):
            res = res[0]
        print(f'{res} - {name}')


if __name__ == "__main__":
    spam_filter = SpamFilter(None)
    m1 = (["Встреча"],['Встреча для предзащиты дипломной работы состоится 23 июня']) # ham
    m2=(["Ограниченное предложение"],['Сегодня последний день, когда вы можете купить универсальный камнедробитель по '
                                      'рекордно низкой цене!']) # spam
    print_results(m1)
    print_results(m2)


