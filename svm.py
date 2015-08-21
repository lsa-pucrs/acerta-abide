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
    return np.mean(precision_recall_fscore_support(y_test, y_hat), axis=1)[0:3], confusion_matrix(y_test, y_hat)

if __name__ == "__main__":

    np.seterr(all='ignore')

    parser = argparse.ArgumentParser(description="Launch a prediction from a pkl file")
    parser.add_argument('train_filename', help='Specifies the csv file with the values to train')
    parser.add_argument('test_filename', help='Specifies the csv file with the values to test')
    args = parser.parse_args()

    results = []

    data = np.loadtxt(args.train_filename, delimiter=',')
    y_train = data[:,0]
    X_train = data[:,1:]

    data = np.loadtxt(args.test_filename, delimiter=',')
    y_test = data[:,0]
    X_test = data[:,1:]

    result = svm(X_train, y_train, X_test, y_test)
    print tabulate(result[0]], headers=['Precision', 'Recall', 'F-score'], tablefmt='grid')