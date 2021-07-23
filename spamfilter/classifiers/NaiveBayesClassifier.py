from collections import defaultdict
from typing import List, Iterable

import numpy as np

class NaiveBayesClassifier(object):
    def __init__(self):
        self.__class_freq = defaultdict(lambda: 0)  # априорная вероятность классов
        self.__feat_freq = defaultdict(lambda: 0)  # функция вероятности признаков

    def fit(self, X, y):
        # calculate classes and features frequencies
        for feature, label in zip(X, y):
            self.__class_freq[label] += 1
            for value in feature:
                self.__feat_freq[(value, label)] += 1

        # normalizate values
        num_samples = len(X)
        for k in self.__class_freq:
            self.__class_freq[k] /= num_samples

        for value, label in self.__feat_freq:
            self.__feat_freq[(value, label)] /= self.__class_freq[label]

        return self

    def predict(self, X:Iterable):
        # return argmin of classes
        preds=[]
        for x in X:
            preds.append(min(self.__class_freq.keys(),
                   key=lambda c: self.__calculate_class_freq(x, c)))
        return preds

    def __calculate_class_freq(self, X, clss):
        # calculate frequence for current class
        freq = - np.log(self.__class_freq[clss])
        for feat in X:
            freq += - np.log(self.__feat_freq.get((feat, clss), 10 ** (-7)))
        return freq
