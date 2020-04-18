#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

SVM evaluation.

Usage:
  svm.py [--whole] [--male] [--threshold] [--leave-site-out] [<derivative> ...]
  svm.py (-h | --help)

Options:
  -h --help           Show this screen
  --whole             Run model for the whole dataset
  --male              Run model for male subjects
  --threshold         Run model for thresholded subjects
  --leave-site-out    Prepare data using leave-site-out method
  derivative          Derivatives to process

"""

import random
import numpy as np
import tabulate
from docopt import docopt
from utils import (load_phenotypes, format_config, hdf5_handler, load_fold, reset)
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


def run(X_train, y_train, X_test, y_test):

    clf = SVC()
    clf.fit(X_train, y_train)
    pred_y = clf.predict(X_test)

    [[TN, FP], [FN, TP]] = confusion_matrix(y_test, pred_y).astype(float)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    specificity = TN / (FP + TN)
    precision = TP / (TP + FP)
    sensivity = recall = TP / (TP + FN)
    fscore = 2 * TP / (2 * TP + FP + FN)

    return [accuracy, precision, recall, fscore, sensivity, specificity]


def run_svm(hdf5, experiment):

    exp_storage = hdf5["experiments"][experiment]

    folds = []
    for fold in exp_storage:

        X_train, y_train, \
        X_valid, y_valid, \
        X_test, y_test = load_fold(hdf5["patients"], exp_storage, fold)

        X_train = np.concatenate([X_train, X_valid])
        y_train = np.concatenate([y_train, y_valid])

        folds.append(run(X_train, y_train, X_test, y_test))

    return np.mean(folds, axis=0).tolist()


if __name__ == "__main__":

    reset()

    arguments = docopt(__doc__)

    pheno_path = "./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv"
    pheno = load_phenotypes(pheno_path)

    hdf5 = hdf5_handler("./data/abide.hdf5", "a")

    valid_derivatives = ["cc200", "aal", "ez", "ho", "tt", "dosenbach160"]
    derivatives = [derivative for derivative
                   in arguments["<derivative>"]
                   if derivative in valid_derivatives]

    experiments = []

    for derivative in derivatives:

        config = {"derivative": derivative}

        if arguments["--whole"]:
            experiments += [format_config("{derivative}_whole", config)],

        if arguments["--male"]:
            experiments += [format_config("{derivative}_male", config)]

        if arguments["--threshold"]:
            experiments += [format_config("{derivative}_threshold", config)]

        if arguments["--leave-site-out"]:
            for site in pheno["SITE_ID"].unique():
                site_config = {"site": site}
                experiments += [
                    format_config("{derivative}_leavesiteout-{site}",
                                  config, site_config)
                ]

    experiments = sorted(experiments)
    experiment_results = []
    for experiment in experiments:
        results = run_svm(hdf5, experiment)
        experiment_results += [[experiment] + results]

    print tabulate.tabulate(experiment_results,
                   headers=["exp", "acc", "prec", "recall",
                            "fscore", "sens", "spec"])
