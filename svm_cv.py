import re, glob, string, csv, sys, random, math, time
from tabulate import tabulate
import argparse

import numpy as np
import nibabel as nb

from sklearn import cross_validation
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn import feature_selection
from sklearn.metrics import confusion_matrix

from multiprocessing import cpu_count
from multiprocessing import Pool
import matplotlib.pyplot as plt

def svm(X_train, y_train, X_test, y_test):
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    return list(precision_recall_fscore_support(y_test, y_hat, average='binary')[0:3]), confusion_matrix(y_test, y_hat)

if __name__ == "__main__":

    np.seterr(all='ignore')

    parser = argparse.ArgumentParser(description="SVM CV train and test")
    parser.add_argument('train_filename', help='Specifies the csv file with the values to train ( %(fold)s for fold replacement )')
    parser.add_argument('test_filename', help='Specifies the csv file with the values to predict ( %(fold)s for fold replacement )')
    parser.add_argument('cv_folds', help='CV Folds', type=int)
    args = parser.parse_args()

    metrics = []
    for fold in xrange(1, cv_folds + 1):

        data_train = np.loadtxt(args.train_filename % { 'fold': str(fold) }, delimiter=',')
        y_train = data_train[:,0]
        X_train = data_train[:,1:]

        data_test = np.loadtxt(args.test_filename % { 'fold': str(fold) }, delimiter=',')
        y_test = data_test[:,0]
        X_test = data_test[:,1:]

        result = svm(X_train, y_train, X_test, y_test)

        metrics.append(result[0])

    metrics.append(np.mean(metrics, axis=0))
    metrics = np.insert(np.array(metrics, dtype=str), 0, range(1, cv_folds + 1) + ['Mean'], 1)

    print tabulate(metrics, headers=['Fold', 'Precision', 'Recall', 'F-score'], tablefmt='grid')