import os
from typing import List

import pandas as pd
import numpy as np
import sklearn
from matplotlib import pyplot
from sklearn.metrics import accuracy_score, precision_score,recall_score,classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from spamfilter.Tokenizer import Tokenizer
from spamfilter.classifiers.LSTMClassifier import LSTMClassifier
from spamfilter.classifiers.NaiveBayesClassifier import NaiveBayesClassifier
from spamfilter.classifiers.utils import label2int, int2label
from spamfilter.Extractor import Extractor
from email_system.NecessaryEmail import NecessaryEmail

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


class SpamFilter(object):

    def __init__(self, recieve_mail_system, spam_folder="emails/spam/", ham_folder="emails/ham/",
                 navec_path="navec_hudlit_v1_12B_500K_300d_100q.tar", is_load_model=True, is_load_weigth=True, dimensions=300,
                 sequence_length=100, lemmatized_path="lemmatized.csv", test_size=0.2):
        # self.root= "spamfilter/"
        self.is_load_model = is_load_model
        self.is_load_weigth = is_load_weigth
        self.spam_folder = spam_folder
        self.ham_folder = ham_folder
        self.lemmatized_path = lemmatized_path
        self.test_size = test_size
        self._extractor = Extractor(recieve_mail_system)
        self._tokenizer = Tokenizer(navec_path, dimensions=dimensions, sequence_length=sequence_length)

        x_train, x_test, y_train, y_test = self._prepare_data()

        self._naive_bayes_classifier = NaiveBayesClassifier()
        self._lstm_classifier = LSTMClassifier(embedding_size=dimensions, sequence_length=sequence_length)
        self._neighbors=KNeighborsClassifier()
        self._svm=SVC()
        self._tree=DecisionTreeClassifier()
        self._train_methods=[self._train_naive_bayes,self._train_lstm]
        self._fit(x_train, y_train)
        self._test(x_test,y_test)
        # spam,spam_uids=self.extractor.from_email_folder(spam_folder,self.extractor.spam_uids)
        # ham,ham_uids=self.extractor.from_email_folder(ham_folder,self.extractor.ham_uids)
        # # self.extractor.extract(spam_folder, ham_folder)
        # self.extractor.save_to_csv(spam,ham,"spam.csv")
        # self.df=self.extractor.get_dataframe(spam,ham)
        # if  self.df is not None :
        #     self.df=self.df.sample(frac=1)

    def _prepare_data(self):
        if os.path.exists(self.lemmatized_path):
            lemmatized = pd.read_csv(self.lemmatized_path)
        else:
            df = self._extract()
            df.to_csv("dataset.csv")
            lemmatized = self._lemmatize(df)
            lemmatized.to_csv(self.lemmatized_path)
        self.dataframe_statistics(lemmatized)
        x, y = self._tokenize(lemmatized)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, random_state=42)
        return x_train, x_test, y_train, y_test

    def dataframe_statistics(self, df):
        print(df.shape)
        df.info()
        print(df.describe())
        print(df.nunique())
        new_subj=df['subject'].str.count(" ")+1
        pyplot.hist(new_subj)
        pyplot.show()

    def _fit(self, x, y):
        # for method in self._train_methods:
        #     method(x,y)
        self._train_naive_bayes(x, y)
        self._train_lstm(x, y)
        # self._train_neigbors(x,y)
        self._lstm_classifier.plot_model()

    def _test(self,x_test,y_test):
        self._test_naive_bayes(x_test,y_test)
        self._test_lstm(x_test,y_test)
        # self._test_neigbors(x_test,y_test)

    # def _relative_path(self,path:str):
    #     return self.root+path

    def _extract(self):
        my_spam = list(self._extractor.from_mbox(self.spam_folder + "Myspam.mbox", True))
        my_spam_df = self._extractor.get_dataframe(my_spam)

        spam_df = pd.read_excel(self.spam_folder + "Pisma_spam.xlsx", index_col=None)
        spam_df['label'] = 'spam'

        ham_df = pd.read_csv(self.ham_folder + "ham.CSV")
        ham_df['label'] = "ham"

        df = pd.concat([my_spam_df, spam_df, ham_df], join="inner", ignore_index=True)
        df=df.sample(frac=1)
        return df

    def _lemmatize(self, df):
        df['subject'] = self._tokenizer.lemmatize_texts(df['subject'])
        df['text'] = self._tokenizer.lemmatize_texts(df['text'])
        # сохранить лемматизированный датасет
        return df

    def drop_empty(self,df):
        """

        @rtype: pd.Dataframe
        """
        df = df.replace("", np.nan)
        subject = df[['subject','label']].dropna()
        text = df[['text','label']].dropna()
        return subject,text

    def _tokenize(self, df:pd.DataFrame):
        subject,text=self.drop_empty(df)
        subject:pd.DataFrame
        text:pd.DataFrame
        subject=subject.drop_duplicates()
        text=text.drop_duplicates()
        self._tokenizer.fit_tokenizer(subject['subject'])
        self._tokenizer.fit_tokenizer(text['text'])
        x = self._tokenizer.text_to_sequences(subject['subject'])
        y = [label2int[label] for label in subject['label']]
        return x, y

    def _train_naive_bayes(self, x, y):
        self._naive_bayes_classifier.fit(x, y)

    def _train_lstm(self, x, y):
        if self.is_load_model and os.path.exists(self._lstm_classifier.model_file):
            self._lstm_classifier.load_model()
        else:
            self._lstm_classifier.build_model(self._tokenizer.get_embedding_matrix())

        if self.is_load_weigth and os.path.exists(self._lstm_classifier.weights_file):
            self._lstm_classifier.load_weights()
        else:
            self._lstm_classifier.train(x, y)

    def _train_neigbors(self, x, y):
        x = x.reshape(-1, 1)
        # y = y.reshape(-1, 1)
        self._neighbors.fit(x,y)

    def _print_statistics(self, predictions, y_test):
        print(f"accuracy score - {accuracy_score(predictions, y_test)} precision score {precision_score(predictions,y_test)} recall score - {recall_score(predictions,y_test)}")
        report=classification_report(y_test,predictions,target_names=["Not Spam","Spam"])
        print(report)

    def _test_naive_bayes(self, x_test, y_test):
        # print("Naive Bayes")
        predictions = list(tqdm((self._naive_bayes_classifier.predict(x) for x in x_test),"Naive Bayes"))
        self._print_statistics(y_test,predictions)

    def _test_lstm(self, x_test, y_test):
        # print("Lstm")
        predictions =[]
        for x in tqdm(x_test,"Lstm"):
            x=np.reshape(x,(1,x.shape[0]))
            predictions.append(self._lstm_classifier.get_predictions(x))
        self._print_statistics(y_test,predictions)

    def _test_neigbors(self, x_test, y_test):
        predictions = list(tqdm((self._neighbors.predict(x) for x in x_test), "Neighbors"))
        self._print_statistics(y_test,predictions)

    def classify(self, emails: List[NecessaryEmail]):
        classified = []
        self.subject_classifier.fit_tokenizer([email.prepared_subejct for email in emails])
        for email in emails:
            classified.append([email, self.subject_classifier.get_predictions(
                email.prepared_subejct)])  # добавить другие результаты классификации
        return classified


if __name__ == "__main__":
    spam_filter = SpamFilter(None,is_load_model=False,is_load_weigth=False,sequence_length=20)
