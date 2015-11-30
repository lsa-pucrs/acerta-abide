import re, glob, string, csv, sys, random, math, time, os
from tabulate import tabulate
import argparse

import numpy as np
import nibabel as nb

from sklearn import cross_validation
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from multiprocessing import cpu_count
from multiprocessing import Pool
import matplotlib.pyplot as plt

from utils import *

def svm(X_train, y_train, X_test, y_test):

    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    [[TN, FP], [FN, TP]] = confusion_matrix(y_test, y_pred).astype(float)
    accuracy = ( TP + TN ) / ( TP + TN + FP + FN )
    specificity = TN / ( FP + TN )
    precision = TP / ( TP + FP )
    sensivity = recall = TP / ( TP + FN )
    fscore = 2 * TP / ( 2 * TP + FP + FN )

    return [accuracy, precision, recall, fscore, sensivity, specificity]

if __name__ == "__main__":

    np.seterr(all="ignore")

    parser = argparse.ArgumentParser(description="SVM CV train and test")
    parser.add_argument("cv_folds", type=nrangearg, help="CV Folds")
    parser.add_argument("filename", help="Specifies the csv file with the values to train and test")
    parser.add_argument("--mean", action="store_true")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    metrics = []
    for fold in args.cv_folds:

        name, extension = os.path.splitext(args.filename)
        data_fold_train_path = ( name + "_cv_%(fold)s_train" + extension ) % {"fold": fold}

        data_train = np.loadtxt(data_fold_train_path, delimiter=",")
        y_train = data_train[:,0]
        X_train = data_train[:,1:]

        name, extension = os.path.splitext(args.filename)
        data_fold_test_path = ( name + "_cv_%(fold)s_test" + extension ) % {"fold": fold}

        data_test = np.loadtxt(data_fold_test_path, delimiter=",")
        y_test = data_test[:,0]
        X_test = data_test[:,1:]

        result = svm(X_train, y_train, X_test, y_test)

        metrics.append(result)

    if args.mean:
        metrics = [np.mean(metrics, axis=0)]
        metrics = np.insert(np.array(metrics, dtype=str), 0, ["Mean"], 1)
    else:
        metrics.append(np.mean(metrics, axis=0))
        metrics = np.insert(np.array(metrics, dtype=str), 0, args.cv_folds + ["Mean"], 1)

    tablefmt = "grid"
    if args.latex:
        tablefmt = "latex"
    print tabulate(metrics, headers=["Fold", "Accuracy", "Precision", "Recall", "F-score", "Sensivity", "Specificity"], tablefmt=tablefmt)