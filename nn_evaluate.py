#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Autoencoders evaluation.

Usage:
  nn_evaluate.py [--folds=N] [--whole] [--male] [--threshold] [<derivative> ...]
  nn_evaluate.py (-h | --help)

Options:
  -h --help     Show this screen
  --folds=N     Number of folds [default: 10]
  --whole       Run model for the whole dataset
  --male        Run model for male subjects
  --threshold   Run model for thresholded subjects
  derivative    Derivatives to process

"""

import numpy as np
import pandas as pd
import tensorflow as tf
from docopt import docopt
from nn import nn, to_softmax, reset
from utils import format_config
from sklearn.metrics import confusion_matrix


def fold_results(config, model_path, data_path):

    n_classes = 2

    model_path = format_config(model_path, config)
    test_path = format_config(data_path, config, {"datatype": "test"})

    test_data = np.loadtxt(test_path, delimiter=",")
    test_X, test_y = test_data[:, 1:], test_data[:, 0]
    test_y = np.array([to_softmax(n_classes, y) for y in test_y])

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

            output = sess.run(
                model["output"],
                feed_dict={
                    model["input"]: test_X,
                    model["dropouts"][0]: 1.0,
                    model["dropouts"][1]: 1.0,
                }
            )

            y_pred = np.argmax(output, axis=1)
            y_true = np.argmax(test_y, axis=1)

            [[TN, FP], [FN, TP]] = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
            accuracy = (TP+TN)/(TP+TN+FP+FN)
            specificity = TN/(FP+TN)
            precision = TP/(TP+FP)
            sensivity = recall = TP/(TP+FN)
            fscore = 2*TP/(2*TP+FP+FN)

            return [accuracy, precision, recall, fscore, sensivity, specificity]
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

    pd.set_option("display.expand_frame_repr", False)

    valid_derivatives = ["cc200", "aal", "ez", "ho", "tt", "dosenbach160"]
    derivatives = [derivative for derivative in arguments["<derivative>"] if derivative in valid_derivatives]

    results = []
    for derivative in derivatives:
        for exp in experiments:
            for fold in range(1, maxfolds):

                config = {
                    "derivative": derivative,
                    "fold": fold,
                    "exp": exp
                }

                results.append([derivative, exp, fold] +
                               fold_results(config,
                                            "./data/models/{derivative}_{exp}_{fold}_mlp.ckpt",
                                            "./data/corr/{derivative}_{exp}_{fold}_{datatype}.csv"))

    cols = ["D", "Exp", "Fold", "Accuracy", "Precision", "Recall", "F-score", "Sensivity", "Specificity"]
    df = pd.DataFrame(results, columns=cols)
    grouped = df.groupby(["Exp"])
    mean = grouped.agg(np.mean).reset_index()
    mean["Fold"] = "Mean"
    df = df.append(mean)
    print df[cols].sort_values(["D", "Exp", "Fold"])
