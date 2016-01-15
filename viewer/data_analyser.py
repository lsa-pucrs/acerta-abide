import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import pandas as pd
import numpy as np
from os.path import isdir, join
from os import listdir
from utils import *

def extract_pheno(fold):
    f = root() + '/data/corr/corr_1D_cv_%d_%s.ids.csv'
    counts = {}
    institutes = []
    for t in ['train', 'valid', 'test']:
        pheno_file = f % (fold, t)
        ids = np.loadtxt(pheno_file, delimiter=',', dtype=str).flatten().tolist()
        ints = [ i.split('_')[0] for i in ids ]
        counts[t] = { ins: ints.count(ins) for ins in set(ints) }
        institutes = institutes + ints
    return counts, list(set(institutes))
