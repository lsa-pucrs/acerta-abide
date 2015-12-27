import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import pandas as pd
import numpy as np
from os.path import isdir, join
from os import listdir
from utils import *


def extract_pheno():
    f = root() + '/data/corr/corr_1D_cv_%d_%s.pheno.csv'
    dfs = []
    for fold in range(1, 11):
        for t in ['train', 'test']:
            pheno_file = f % (fold, t)
            pheno = np.loadtxt(pheno_file, delimiter=',', dtype=str)
            df = pd.DataFrame(pheno, columns=['pid', 'class', 'sex', 'institute'])
            df['fold'] = fold
            df['type'] = t
            dfs.append(df)
    return pd.concat(dfs)
