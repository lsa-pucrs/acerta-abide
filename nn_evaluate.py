#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
from nn import nn, to_softmax, reset
from utils import format_config
from sklearn.metrics import confusion_matrix


def fold_results(fold, model_path, data_path):

    n_classes = 2

    model_path = format_config(model_path, {'fold': str(fold)})
    test_path = format_config(data_path, {'fold': str(fold), "datatype": "test"})

    test_data = np.loadtxt(test_path, delimiter=",")
    test_X, test_y = test_data[:, 1:], test_data[:, 0]
    test_y = np.array([to_softmax(n_classes, y) for y in test_y])

    try:
        model = nn(19900, n_classes, [1000, 600])
        init = tf.global_variables_initializer()
        with tf.Session() as sess:

            sess.run(init)

            saver = tf.train.Saver(model["params"])
            saver.restore(sess, model_path)

            output = sess.run(
                model['output'],
                feed_dict={
                    model['input']: test_X,
                    model['dropouts'][0]: 1.0,
                    model['dropouts'][1]: 1.0,
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

    results = []

    for fold in range(1, 11):
        results.append(["Whole", fold] + fold_results(fold, "./data/models/mlp-{fold}.ckpt", "./data/corr/corr_1D_cv_{fold}_{datatype}.csv"))

    for fold in range(1, 11):
        results.append(["Male", fold] + fold_results(fold, "./data/models/mlp-{fold}_male.ckpt", "./data/corr/corr_1D_cv_{fold}_{datatype}_male.csv"))

    cols = ['Exp', 'Fold', 'Accuracy', 'Precision', 'Recall', 'F-score', 'Sensivity', 'Specificity']
    df = pd.DataFrame(results, columns=cols)
    grouped = df.groupby(['Exp'])
    mean = grouped.agg(np.mean).reset_index()
    mean['Fold'] = 'Mean'
    df = df.append(mean)
    print df[cols]