import multiprocessing as mp
import numpy as np
import numpy.ma as ma
import nibabel as nb
import csv
from tabulate import tabulate
from functools import partial
import sys, time
import re
from pylearn2.utils import serial
from theano import tensor as T
from theano import function
import pickle

def compute_parcellations(mask):
    mask_vals = np.unique( mask )
    mask_vals = np.array([i for i in mask_vals if i > 0])
    avg_ts = np.zeros( (mask_vals.shape[0]) )
    for i, r in enumerate(mask_vals):
        avg_ts[ i ] = r
    return avg_ts

def compute_connectivity(functional):
    with np.errstate(invalid='ignore'):
        corr = np.zeros((functional.shape[0], functional.shape[0]), dtype=object)
        for i, f in enumerate(functional):
            for j, g in enumerate(functional):
                corr[i, j] = '%s,%s' % (int(f), int(g))
        mask = np.invert(np.tri(corr.shape[0], k = -1, dtype=bool))
        m = ma.masked_where(mask == 1, mask)
        return ma.masked_where(m, corr).compressed()

def prepare_mask_input(mask):
    mask = nb.load(mask).get_data()
    functional = compute_parcellations(mask)
    functional = compute_connectivity(functional)
    return functional

if __name__ == "__main__":

    np.set_printoptions(threshold=np.nan)

    inputs = prepare_mask_input('./data/masks/cc200.nii.gz')

    folds = [[]] * 10

    for fold in range(10):

        print 'Fold %s' % fold
        folds[fold] = [[]] * inputs.shape[0]

        print 'Loading "result/first.config-0.pre-autoencoder-1_cv_%s.pkl"' % (fold+1)
        model = serial.load('result/first.config-0.pre-autoencoder-1_cv_%s.pkl' % (fold+1))
        X = model.get_input_space().make_theano_batch()
        Y = model.encode(X)
        # Y_max = T.argmax(Y, axis=1)
        # f = function([X], Y_max, allow_input_downcast=True)
        encode = function([X], Y, allow_input_downcast=True)
        print 'Loaded'

        xs = []
        for i in range(inputs.shape[0]):
            x = np.empty(inputs.shape[0])
            x.fill(0)
            x[i] = 1
            xs.append(x)
        for i in range(inputs.shape[0]):
            x = np.empty(inputs.shape[0])
            x.fill(0)
            x[i] = -1
            xs.append(x)

        output = encode(xs)
        print np.max(output)
        sys.exit(1)


        # for i in range(inputs.shape[0]):
    #         folds[fold][i] = []
    #         for this in [-1, 1]:
    #             xs = []
    #             for other in np.linspace( -0.5 , 0.5 , 3 ):
    #                 x = np.empty(inputs.shape[0])
    #                 x.fill(other)
    #                 x[i] = this
    #                 xs.append(x)
    #             y_prob = f_prob(xs)

    #             mean = np.mean(y_prob, axis=0).flatten()
    #             folds[fold][i].append(mean)

    # output = open('data.pkl', 'wb')
    # pickle.dump(inputs, output)
    # pickle.dump(folds, output)
    # output.close()