import sys
import os
import argparse
import numpy as np

from pylearn2.utils import serial
from theano import tensor as T
from theano import function

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

def predict(model_path, test_path, cv_folds):

    metrics = []
    for fold in xrange(1, cv_folds + 1):

        try:
            model = serial.load(model_path % { 'fold': str(fold) })
        except Exception as e:
            print "error loading {}:".format(model_path)
            print e
            return False

        x = np.loadtxt(test_path % { 'fold': str(fold) }, delimiter=',')

        y_true = x[:,0]
        x = x[:,1:]

        X = model.get_input_space().make_theano_batch()
        Y = model.fprop(X)
        Y_max = T.argmax(Y, axis=1)

        f = function([X], Y_max, allow_input_downcast=True)
        f_prob = function([X], Y, allow_input_downcast=True)

        y_pred = f(x)
        y_prob = f_prob(x)

        metrics.append(list(precision_recall_fscore_support(y_true, y_pred, average='binary')[0:3]))


    metrics.append(np.mean(metrics, axis=0))
    metrics = np.insert(np.array(metrics, dtype=str), 0, range(1, cv_folds + 1) + ['Mean'], 1)

    print tabulate(metrics, headers=['Fold', 'Precision', 'Recall', 'F-score'], tablefmt='grid')

    return True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launch a prediction from a pkl file")
    parser.add_argument('model_filename', help='Specifies the pkl model file ( %(fold)s for fold replacement )')
    parser.add_argument('test_filename', help='Specifies the csv file with the values to predict ( %(fold)s for fold replacement )')
    parser.add_argument('cv_folds', help='CV Folds', type=int)
    args = parser.parse_args()

    ret = predict(args.model_filename, args.test_filename, args.cv_folds)
    if not ret:
        sys.exit(-1)

