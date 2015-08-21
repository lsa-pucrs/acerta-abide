import sys
import os
import argparse
import numpy as np

from pylearn2.utils import serial
from theano import tensor as T
from theano import function

from sklearn.metrics import classification_report

def predict(model_path, test_path, headers=False, first_col_label=False, probabilities=False):

    try:
        model = serial.load(model_path)
    except Exception as e:
        print "error loading {}:".format(model_path)
        print e
        return False

    skiprows = 1 if headers else 0
    x = np.loadtxt(test_path, delimiter=',', skiprows=skiprows)

    y_true = None
    if first_col_label:
        y_true = x[:,0]
        x = x[:,1:]

    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)
    Y_max = T.argmax(Y, axis=1)

    f = function([X], Y_max, allow_input_downcast=True)
    f_prob = function([X], Y, allow_input_downcast=True)

    y = f(x)
    y_prob = f_prob(x)

    print confusion_matrix(y_true, y)
    target_names = ['ASD', 'Control']
    print classification_report(y_true, y, target_names=target_names)
    if probabilities:
        print zip(y_true.astype(int), y_prob[:,0], y_prob[:,1])

    return True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launch a prediction from a pkl file")
    parser.add_argument('model_filename', help='Specifies the pkl model file')
    parser.add_argument('test_filename', help='Specifies the csv file with the values to predict')
    parser.add_argument('--has-headers', '-H', dest='has_headers', action='store_true', help='Indicates the first row in the input file is feature labels')
    parser.add_argument('--has-row-label', '-L', dest='has_row_label', action='store_true', help='Indicates the first column in the input file is row labels')
    parser.add_argument('--probabilities', '-P', dest='probabilities', action='store_true', help='Compute probabilities of a softmax for each example')
    args = parser.parse_args()

    ret = predict(args.model_filename, args.test_filename, args.has_headers, args.has_row_label, args.probabilities)
    if not ret:
        sys.exit(-1)

