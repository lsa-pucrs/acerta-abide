import sys
import os
import argparse
import numpy as np

from pylearn2.utils import serial
from theano import tensor as T
from theano import function

from utils import *

def save(X, y, filename):
    data = np.insert(X, 0, y, 1)
    np.savetxt(filename, data, delimiter=',')
    print 'Saved', filename, ':', X.shape, y.shape

def load(filename):
    data = np.loadtxt(filename, delimiter=',')
    y = data[:,0]
    X = data[:,1:]
    return X, y

def transform(model, X, y):
    return model.perform(X), y

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Transform data using a trained model")
    parser.add_argument('cv_folds', type=nrangearg, help='CV Folds')
    parser.add_argument('data_filename', help='Specifies the csv file with the values to transform')
    parser.add_argument('data_destination_filename', help='Specifies the csv file to store transformed data')
    parser.add_argument('model_filename', metavar='M', type=str, nargs='+', help='Specifies the pkl model file')
    parser.add_argument('--last', action='store_true', help='Extract the last model only')
    args = parser.parse_args()

    try:

        for fold in args.cv_folds:

            name, extension = os.path.splitext(args.data_filename)
            data_fold_train_path = ( name + '_cv_%(fold)s_train' + extension ) % {'fold': fold}
            data_fold_test_path = ( name + '_cv_%(fold)s_test' + extension ) % {'fold': fold}

            dest_name, dest_extension = os.path.splitext(args.data_destination_filename)
            data_destination_fold_train_path = dest_name + '.%(model)s' + ( '_cv_%(fold)s_train' + dest_extension ) % {'fold': fold}
            data_destination_fold_test_path = dest_name + '.%(model)s' + ( '_cv_%(fold)s_test' + dest_extension ) % {'fold': fold}

            print "Loading data", data_fold_train_path
            X_train, y_train = load(data_fold_train_path)
            print "Loading data", data_fold_test_path
            X_test, y_test = load(data_fold_test_path)

            X_train = X_train.astype(np.float32)
            X_test = X_test.astype(np.float32)

            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

            for model_i, model_filename in enumerate(args.model_filename):

                name, extension = os.path.splitext(model_filename)
                model_fold_path = ( name + '_cv_%(fold)s' + extension ) % {'fold': fold}

                model_name = os.path.splitext(os.path.basename(model_filename))[0]

                print "Loading model", model_fold_path
                model = serial.load(model_fold_path)
                model_class = model.__class__.__name__

                if model_class == 'MLP':
                    for i, l in enumerate(model.layers):
                        if not hasattr(l, 'layer_content'):
                            break
                        X_train, y_train = transform(l.layer_content, X_train, y_train)
                        X_test, y_test = transform(l.layer_content, X_test, y_test)
                        save(X_train, y_train, data_destination_fold_train_path % { 'model': model_name + '.' + l.layer_name })
                        save(X_test, y_test, data_destination_fold_test_path  % { 'model': model_name + '.' + l.layer_name })
                else:
                    X_train, y_train = transform(model, X_train, y_train)
                    X_test, y_test = transform(model, X_test, y_test)
                    save(X_train, y_train, data_destination_fold_train_path % { 'model': model_name })
                    save(X_test, y_test, data_destination_fold_test_path % { 'model': model_name })

                print 'Performed', model_filename

    except Exception as e:
        print e

# THEANO_FLAGS="device=gpu1,floatX=float32" python export_data_cv.py 1-10 data/corr/corr.csv data/trans/corr.csv  \
# result-external/models/first.original-valid.pre-autoencoder-1-valid.pkl \
# result-external/models/first.original-valid.pre-autoencoder-2-valid.pkl

# THEANO_FLAGS="device=gpu1,floatX=float32" python export_data_cv.py --last 1-10 \
# data/corr/corr.csv data/trans/corr.csv \
# result-external/models/secondo.deep-4.MLP.pkl
