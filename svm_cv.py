import os
from tabulate import tabulate
import argparse
import numpy as np
from sklearn.svm import (SVC, LinearSVC)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from utils import *


def svm(X_train, y_train, X_test, y_test):

    clf = SVC(C=1.0, kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    [[TN, FP], [FN, TP]] = confusion_matrix(y_test, y_pred).astype(float)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    specificity = TN / (FP + TN)
    precision = TP / (TP + FP)
    sensivity = recall = TP / (TP + FN)
    fscore = 2 * TP / (2 * TP + FP + FN)

    auc = roc_auc_score(y_test, y_proba[:, 1])

    return [accuracy, precision, recall, fscore, sensivity, specificity, auc], clf.coef_.flatten()

if __name__ == "__main__":

    np.seterr(all="ignore")
    np.set_printoptions(threshold=np.nan)

    parser = argparse.ArgumentParser(description="SVM CV train and test")
    parser.add_argument("cv_folds", type=nrangearg, help="CV Folds")
    parser.add_argument("filename", help="Specifies the csv file with the values to train and test")
    parser.add_argument("--mean", action="store_true")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    folder = './experiments/svm/analysis'
    if not os.path.isdir(folder):
        os.makedirs(folder)

    all_coefs = []
    metrics = []
    for fold in args.cv_folds:

        name, extension = os.path.splitext(args.filename)
        data_fold_train_path = (name + "_cv_%(fold)s_train" + extension) % {"fold": fold}

        data_train = np.loadtxt(data_fold_train_path, delimiter=",")
        y_train = data_train[:, 0]
        X_train = data_train[:, 1:]

        data_fold_valid_path = (name + "_cv_%(fold)s_valid" + extension) % {"fold": fold}

        data_valid = np.loadtxt(data_fold_valid_path, delimiter=",")
        y_valid = data_valid[:, 0]
        X_valid = data_valid[:, 1:]

        y_train = np.concatenate((y_valid, y_train))
        X_train = np.concatenate((X_valid, X_train))

        name, extension = os.path.splitext(args.filename)
        data_fold_test_path = (name + "_cv_%(fold)s_test" + extension) % {"fold": fold}

        data_test = np.loadtxt(data_fold_test_path, delimiter=",")
        y_test = data_test[:, 0]
        X_test = data_test[:, 1:]

        print X_train.shape, y_train.shape, X_test.shape, y_test.shape

        result, coefs = svm(X_train, y_train, X_test, y_test)
        np.savetxt(folder + '/coeffs_%s' % fold, coefs, delimiter=',')

        all_coefs.append(coefs)
        metrics.append(result)

        print 'Done fold %s' % fold

    all_coefs = np.array(all_coefs)
    mean_coefs = np.mean(all_coefs, axis=0)
    np.savetxt(folder + '/coeffs_mean', mean_coefs, delimiter=',')

    if args.mean:
        metrics = [np.mean(metrics, axis=0)]
        metrics = np.insert(np.array(metrics, dtype=str), 0, ["Mean"], 1)
    else:
        metrics.append(np.mean(metrics, axis=0))
        metrics = np.insert(np.array(metrics, dtype=str), 0, args.cv_folds + ["Mean"], 1)

    tablefmt = "grid"
    if args.latex:
        tablefmt = "latex"
    print tabulate(metrics, headers=["Fold", "Accuracy", "Precision", "Recall", "F-score", "Sensivity", "Specificity", "AUC"], tablefmt=tablefmt)
