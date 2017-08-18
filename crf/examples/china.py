#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 14/08/2017 3:43 PM
# @Author  : zhangzhen
# @Site    : 
# @File    : china.py
# @Software: PyCharm
from util.rmrb import load_train_data_to_tuple
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from itertools import chain
import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
# from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from sklearn.grid_search import RandomizedSearchCV
from sklearn.externals import joblib
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter


def word2features(sent, i):
    """将单个词转变成特征"""
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word': word,
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word': word1
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word': word1
        })
    else:
        features['EOS'] = True
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, label in sent]


def sent2tokens(sent):
    return [token for token, label in sent]


def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


if __name__ == '__main__':
    originPath = "/Users/zhangzhen/gitRepository/hmmlearn/util/train.txt"
    dataSet = load_train_data_to_tuple(originPath)
    # print train

    # pprint(sent2features(dataSet[0])[0])
    X_train = [sent2features(s) for s in dataSet]
    y_train = [sent2labels(s) for s in dataSet]

    X_test = [sent2features(s) for s in dataSet]
    y_test = [sent2labels(s) for s in dataSet]

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    labels = list(crf.classes_)
    print labels
    y_pred = crf.predict(X_test)
    print metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
    joblib.dump(crf, "crf.m")

    y_pred = crf.predict(X_test)
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=labels, digits=3
    ))

    print("Top likely transitions:")
    print_transitions(Counter(crf.transition_features_).most_common(20))

    print("\nTop unlikely transitions:")
    print_transitions(Counter(crf.transition_features_).most_common()[-20:])
