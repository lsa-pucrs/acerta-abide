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

    print model_path

    try:
        model = serial.load(model_path)
    except Exception as e:
        print "Error loading {}:".format(model_path)
        print e
        return False

    name, ext = os.path.splitext(test_path)

    x = np.loadtxt(test_path, delimiter=',')
    ids = np.loadtxt(name + '.ids' + ext, delimiter=',', dtype=str)

    y_true = x[:, 0]
    x = x[:, 1:]

    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)
    Y_max = T.argmax(Y, axis=1)

    f = function([X], Y_max, allow_input_downcast=True)
    # prob = function([X], Y, allow_input_downcast=True)

    y_pred = f(x)

    right = [ pid for i, pid in enumerate(ids) if y_true[i] == y_pred[i]]
    wrong = [ pid for i, pid in enumerate(ids) if y_true[i] != y_pred[i]]

    [[TN, FP], [FN, TP]] = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    specificity = TN/(FP+TN)
    precision = TP/(TP+FP)
    sensivity = recall = TP/(TP+FN)
    fscore = 2*TP/(2*TP+FP+FN)

    return [accuracy, precision, recall, fscore, sensivity, specificity]


def compute_metrics_config(config):

    model_path = cvize(config['model_path'], config['fold'])
    test_path = config['test_path']

    return [config['experiment'], config['fold']] + compute_metrics(model_path, test_path)


def evaluate(config, gpus, threads):

    t0 = time.time()

    results = []
    for exp in config['experiments']:

        model_name = '.'.join([config['pipeline'], config['config'], config['model'], 'pkl'])
        folder = '.'.join([config['pipeline'], config['config']])

        exp_config = compute_config(config, {
            'experiment': exp,
            'model_path': os.path.join(root(), 'experiments', folder, exp, 'models', model_name),
        }, replace=False)

        theano_run = partial(parallel_theano, execute=compute_metrics_config, concurr_key='fold')
        results = results + run_parallel(theano_run, exp_config, gpus=gpus, threads=threads, concurr_key='fold')

    print "%s: %s" % ("Total", elapsed_time(t0))

    return results


if __name__ == "__main__":

    pd.set_option('display.width', 500)

    parser = argparse.ArgumentParser(description='Evaluate cv deep-learning pipeline.')
    parser.add_argument('pipeline', help='Pipeline')
    parser.add_argument('config', help='Config')
    parser.add_argument('model', help='Model')
    parser.add_argument('cv_folds', type=nrangearg, help='CV Folds')
    parser.add_argument('testdata', type=str, help='Test Dataset')
    parser.add_argument('--experiment', type=str, default=None, help='Experiment ID')
    parser.add_argument('--gpus', type=gpurangearg, default=[], help='Number of gpus')
    parser.add_argument('--threads', type=int, help='Number of threads')
    args = parser.parse_args()

    experiments = executed_experiments(args.pipeline, args.config)
    if args.experiment is not None:
        experiments = [args.experiment]

    config = {
        'pipeline': args.pipeline,
        'config': args.config,
        'model': args.model,
        'experiments': experiments,
        'fold': args.cv_folds,
        'test_path': args.testdata,
    }

    results = evaluate(config, gpus=args.gpus, threads=args.threads)

    cols = ['Exp', 'Fold', 'Accuracy', 'Precision', 'Recall', 'F-score', 'Sensivity', 'Specificity']
    df = pd.DataFrame(results, columns=cols)
    grouped = df.groupby(['Exp'])
    mean = grouped.agg(np.mean).reset_index()
    mean['Fold'] = 'Mean'
    df = df.append(mean)

    print df[cols]
