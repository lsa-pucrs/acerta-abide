import sys, os, argparse
import numpy as np
from functools import partial
import multiprocessing as mp

from pylearn2.utils import serial
from theano import tensor as T
from theano import function

from sklearn.metrics import confusion_matrix
from tabulate import tabulate

from utils import config_dict, compute_config, format_config, elapsed_time, nrangearg

def compute_metrics(fold, model_fold_path, test_fold_path):
    try:
        model = serial.load(model_fold_path % { 'fold': str(fold) })
    except Exception as e:
        print "Error loading {}:".format(model_fold_path % { 'fold': str(fold) })
        print e
        return False

    x = np.loadtxt(test_fold_path % { 'fold': str(fold) }, delimiter=',')

    y_true = x[:,0]
    x = x[:,1:]

    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)
    Y_max = T.argmax(Y, axis=1)

    f = function([X], Y_max, allow_input_downcast=True)
    f_prob = function([X], Y, allow_input_downcast=True)

    y_pred = f(x)
    y_prob = f_prob(x)

    [[TN, FP], [FN, TP]] = confusion_matrix(y_true, y_pred).astype(float)
    accuracy = ( TP + TN ) / ( TP + TN + FP + FN )
    specificity = TN / ( FP + TN )
    precision = TP / ( TP + FP )
    sensivity = recall = TP / ( TP + FN )
    fscore = 2 * TP / ( 2 * TP + FP + FN )

    return [accuracy, precision, recall, fscore, sensivity, specificity]

def predict(model_path, test_path, cv_folds, jobs=1):

    name, extension = os.path.splitext(model_path)
    model_fold_path = name + '_cv_%(fold)s' + extension

    name, extension = os.path.splitext(test_path)
    test_fold_path = name + '_cv_%(fold)s_test' + extension

    partial_compute_metrics = partial(compute_metrics, model_fold_path=model_fold_path, test_fold_path=test_fold_path)

    if jobs == 1:
        metrics = []
        for fold in cv_folds:
            metrics.append(partial_compute_metrics(fold))
    else:
        pool = mp.Pool(processes=jobs)
        metrics = pool.map(partial_compute_metrics, cv_folds)
    return metrics

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launch a prediction from a pkl file")
    parser.add_argument('model_filename', help='Specifies the pkl model file ( %(fold)s for fold replacement )')
    parser.add_argument('test_filename', help='Specifies the csv file with the values to predict ( %(fold)s for fold replacement )')
    parser.add_argument('cv_folds', type=nrangearg, help='CV Folds')
    args = parser.parse_args()

    metrics = predict(args.model_filename, args.test_filename, args.cv_folds)
    metrics.append(np.mean(metrics, axis=0))
    metrics = np.insert(np.array(metrics, dtype=str), 0, args.cv_folds + ['Mean'], 1)

    print tabulate(metrics, headers=['Fold', 'Accuracy', 'Precision', 'Recall', 'F-score', 'Sensivity', 'Specificity'], tablefmt='grid', floatfmt=".4f")