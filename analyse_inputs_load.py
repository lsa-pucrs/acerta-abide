import sys
import numpy as np
import numpy.ma as ma
import nibabel as nb
import sys, time
import re
from pylearn2.utils import serial
from theano import tensor as T
from theano import function
import itertools
import json
from collections import defaultdict
from sklearn.preprocessing import scale


model = serial.load('/home/anibal.heinsfeld/repos/acerta-abide/experiments/first.valid/final/models/first.valid.mlp-valid_cv_2.pkl')
train_data = np.loadtxt("/home/anibal.heinsfeld/repos/acerta-abide/data/corr/corr_1D_cv_2_train.csv", delimiter=",")
test_data = np.loadtxt("/home/anibal.heinsfeld/repos/acerta-abide/data/corr/corr_1D_cv_2_test.csv", delimiter=",")

# execfile('analyse_inputs_load.py')

X = model.get_input_space().make_theano_batch()
Y = model.fprop(X)
Y_max = T.argmax(Y, axis=1)
f = function([X], Y_max, allow_input_downcast=True)
f_prob = function([X], Y, allow_input_downcast=True)

ASD = 0
TC = 1


train_data_X = scale(train_data[:, 1:])
train_data_y = train_data[:, 0]
train_asd_mean = np.mean(train_data_X[train_data_y == ASD], axis=0)
train_tc_mean = np.mean(train_data_X[train_data_y == TC], axis=0)

test_data_X = test_data[:, 1:]
test_data_y = test_data[:, 0]
test_asd_mean = np.mean(test_data_X[test_data_y == ASD], axis=0)
test_tc_mean = np.mean(test_data_X[test_data_y == TC], axis=0)