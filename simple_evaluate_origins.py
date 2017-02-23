import argparse
import numpy as np
import os
import sys
from functools import partial
import pandas as pd
from utils import *


def compute_metrics(model_path, test_path):

    from sklearn.metrics import confusion_matrix
    from theano import function
    from theano import tensor as T
    from pylearn2.utils import serial

    try:
        model = serial.load(model_path)
    except Exception as e:
        print "Error loading {}:".format(model_path)
        print e
        return False

    name, ext = os.path.splitext(test_path)

    x = np.loadtxt(test_path, delimiter=',')

    y_true = x[:, 0]
    x = x[:, 1:]

    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)
    Y_max = T.argmax(Y, axis=1)

    f = function([X], Y_max, allow_input_downcast=True)
    # prob = function([X], Y, allow_input_downcast=True)

    y_pred = f(x)

    [[TN, FP], [FN, TP]] = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    specificity = TN/(FP+TN)
    precision = TP/(TP+FP)
    sensivity = recall = TP/(TP+FN)
    fscore = 2*TP/(2*TP+FP+FN)

    return [len(x), accuracy, precision, recall, fscore, sensivity, specificity]


if __name__ == "__main__":

    pd.set_option('display.width', 500)

    results = []
    for i in range(1,11):
        print [str(i)]
        [str(i)] + compute_metrics(
            "/home/anibal.heinsfeld/repos/acerta-abide/experiments/first.valid/final/models/first.valid.mlp-valid_cv_" + str(i) + ".pkl",
            "/home/anibal.heinsfeld/repos/acerta-abide/data/corr/corr_1D_cv_" + str(i) + "_test.csv"
        )

    results.append(['Mean'] + np.mean(np.array(results, dtype=np.float64)[:,1:], axis=0).tolist())        

    cols = ['Site', 'Size', 'Accuracy', 'Precision', 'Recall', 'F-score', 'Sensivity', 'Specificity']
    df = pd.DataFrame(results, columns=cols)
    print df[cols]
