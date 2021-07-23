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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def test_lstm(classifier:LSTMClassifier, x_test, y_test,name:str):
    # print("Lstm")
    predictions = []
    for x in tqdm(x_test, name):
        x = np.reshape(x, (1, x.shape[0]))
        predictions.append(classifier.predict(x))
    return _get_statistics(predictions,y_test)


def test_classifier(classifier,x_test,y_test,name):
    print(f"Testing {name}")
    predictions =classifier.predict(x_test)
    return _get_statistics(predictions,y_test)


def _get_statistics(predictions, y_test):
    # print(
        # f"accuracy score - {accuracy_score(predictions, y_test)} precision score {precision_score(predictions, y_test)} recall score - {recall_score(predictions, y_test)}")
    report = classification_report(y_test, predictions, target_names=["Not Spam", "Spam"])
    return report
