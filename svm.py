#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

SVM evaluation.

Usage:
  svm.py [--folds=N] [--whole] [--male] [--threshold] [<derivative> ...]
  svm.py (-h | --help)

Options:
  -h --help     Show this screen
  --folds=N     Number of folds [default: 10]
  --whole       Run model for the whole dataset
  --male        Run model for male subjects
  --threshold   Run model for thresholded subjects
  derivative    Derivatives to process

"""

import numpy as np
from docopt import docopt
from utils import format_config
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from nn import nn, to_softmax, reset
import tensorflow as tf


def run(train_X, train_y, test_X, test_y):

    clf = SVC()
    clf.fit(train_X, train_y)
    pred_y = clf.predict(test_X)

    [[TN, FP], [FN, TP]] = confusion_matrix(test_y, pred_y).astype(float)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    specificity = TN / (FP + TN)
    precision = TP / (TP + FP)
    sensivity = recall = TP / (TP + FN)
    fscore = 2 * TP / (2 * TP + FP + FN)

    return [accuracy, precision, recall, fscore, sensivity, specificity]


def svm(config, model_path, data_path):

    n_classes = 2

    train_path = format_config(data_path, config, {"datatype": "train"})
    valid_path = format_config(data_path, config, {"datatype": "valid"})
    test_path = format_config(data_path, config, {"datatype": "test"})
    model_path = format_config(model_path, config)

    train_data = np.loadtxt(train_path, delimiter=",")
    train_X, train_y = train_data[:, 1:], train_data[:, 0]

    valid_data = np.loadtxt(valid_path, delimiter=",")
    valid_X, valid_y = valid_data[:, 1:], valid_data[:, 0]

    train_X = np.concatenate([train_X, valid_X])
    train_y = np.concatenate([train_y, valid_y])

    test_data = np.loadtxt(test_path, delimiter=",")
    test_X, test_y = test_data[:, 1:], test_data[:, 0]

    try:

        model = nn(test_X.shape[1], n_classes, [
            {"size": 1000, "actv": tf.nn.tanh},
            {"size": 600, "actv": tf.nn.tanh},
        ])

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            saver = tf.train.Saver(model["params"])
            saver.restore(sess, model_path)

            train_X = sess.run(
                model["actvs"][1],
                feed_dict={
                    model["input"]: train_X,
                    model["dropouts"][0]: 1.0,
                    model["dropouts"][1]: 1.0,
                }
            )

            test_X = sess.run(
                model["actvs"][1],
                feed_dict={
                    model["input"]: test_X,
                    model["dropouts"][0]: 1.0,
                    model["dropouts"][1]: 1.0,
                }
            )

            print run(train_X, train_y, test_X, test_y)

    finally:
        reset()


if __name__ == "__main__":

    arguments = docopt(__doc__)

    experiments = []
    if arguments["--whole"]:
        experiments.append("whole")
    if arguments["--male"]:
        experiments.append("male")
    if arguments["--threshold"]:
        experiments.append("threshold")

    maxfolds = int(arguments["--folds"]) + 1

    valid_derivatives = ["cc200", "aal", "ez", "ho", "tt", "dosenbach160"]
    derivatives = [
        derivative
        for derivative in arguments["<derivative>"]
        if derivative in valid_derivatives
    ]
    maxfolds = int(arguments["--folds"]) + 1

    for derivative in derivatives:
        for exp in experiments:
            for fold in range(1, maxfolds):

                config = {
                    "derivative": derivative,
                    "exp": exp,
                    "fold": fold,
                }

                svm(config,
                    "./data/models/{derivative}_{exp}_{fold}_mlp.ckpt",
                    "./data/corr/{derivative}_{exp}_{fold}_{datatype}.csv")
