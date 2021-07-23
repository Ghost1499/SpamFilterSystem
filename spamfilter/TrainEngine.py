import os

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from spamfilter.classifiers.LSTMClassifier import LSTMClassifier
from spamfilter.classifiers.NaiveBayesClassifier import NaiveBayesClassifier


def fit_classifier(classifier,x,y):
    classifier.fit(x, y)


def train_lstm(classifier:LSTMClassifier,x,y, embedding_matrix,validation_size=0.2, is_load_model=False, is_load_weights=False):
    if is_load_model and os.path.exists(classifier.model_file):
        classifier.load_model()
    else:
        classifier.build_model(embedding_matrix=embedding_matrix)

    if is_load_weights and os.path.exists(classifier.weights_file):
        classifier.load_weights()
    else:
        classifier.train(x, y, validation_size=validation_size)
