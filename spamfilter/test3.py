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
from spamfilter.DataPrepareEngine import DataPrepareEngine
from spamfilter.TrainEngine import train_lstm,fit_naive_bayes
from spamfilter.TestEngine import test_naive_bayes,test_lstm
from email_system.NecessaryEmail import NecessaryEmail

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

subject_seq_length=20
text_seq_length=100

data_prepare=DataPrepareEngine(None)
lemmatized=data_prepare.lemmatize(True)
subject,text=data_prepare.prepare_data(lemmatized)
emb_matrix=data_prepare.fit_tokenizer(subject,text)
data_prepare.dataframe_statistics(subject,'subject')
data_prepare.dataframe_statistics(text,'text')
x_subject,y_subject=data_prepare.texts_to_sequences(subject_seq_length,subject,'subject')
x_text,y_text=data_prepare.texts_to_sequences(text_seq_length,text,'text')

x_subject_train,x_subject_test,y_subject_train,y_subject_test=train_test_split(x_subject,y_subject,test_size=0.1)
x_text_train,x_text_test,y_text_train,y_text_test=train_test_split(x_text,y_text,test_size=0.1)

subject_NB=NaiveBayesClassifier()
text_NB=NaiveBayesClassifier()
subject_LSTM=LSTMClassifier(sequence_length=subject_seq_length,model_name='subject')
text_LSTM=LSTMClassifier(sequence_length=text_seq_length,model_name='text')

fit_naive_bayes(subject_NB,x_subject_train,y_subject_train)
fit_naive_bayes(text_NB,x_text_train,y_text_train)
train_lstm(subject_LSTM,x_subject_train,y_subject_train,emb_matrix,validation_size=0.1,is_load_model=True,is_load_weights=True)
train_lstm(text_LSTM,x_text_train,y_text_train,emb_matrix,validation_size=0.1,is_load_model=True,is_load_weights=True)

print(test_naive_bayes(subject_NB,x_subject_test,y_subject_test))
print(test_naive_bayes(text_NB,x_text_test,y_text_test))
print(test_lstm(subject_LSTM,x_subject_test,y_subject_test))
print(test_lstm(text_LSTM,x_text_test,y_text_test))
