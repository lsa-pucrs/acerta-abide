import sys
import os
import argparse
import numpy as np

from pylearn2.utils import serial
from theano import tensor as T
from theano import function

def save(X, y, filename):
    data = np.insert(X, 0, y, 1)
    np.savetxt(filename + '.csv', data, delimiter=',')
    print 'Performed', filename + '.csv', ':', X.shape, y.shape

def load(filename):
    data = np.loadtxt(filename, delimiter=',')
    y = data[:,0]
    X = data[:,1:]
    return X, y

def transform(model, X, y):
    return model.perform(X), y

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Transform data using a trained model")
    parser.add_argument('data_filename', help='Specifies the csv file with the values to transform')
    parser.add_argument('data_destination_filename', help='Specifies the csv file to store transformed data')
    parser.add_argument('model_filename', metavar='M', type=str, nargs='+', help='Specifies the pkl model file')
    args = parser.parse_args()

    try:

        X, y = load(args.filename)

        for model_filename in args.model_filename:

            model_name = os.path.splitext(os.path.basename(model_filename))[0]
            model = serial.load(model_filename)
            model_class = model.__class__.__name__
            if model_class == 'MLP':
                for l in model.layers:
                    X, y = transform(l.layer_content, X, y)
                    save(X, y, args.data_final_filename + '.' + model_name + '.' + l.layer_name )
            else:
                X, y = transform(model, X, y)
                save(X, y, args.data_final_filename + '.' + model_name)

            print 'Performed', model_filename, ':', X.shape

    except Exception as e:
        print e