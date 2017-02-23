#!/usr/bin/env python
import os
import random
import numpy as np
import numpy.ma as ma
import pandas as pd
import urllib
import urllib2
import hashlib
from functools import partial
from sklearn import preprocessing
from utils import *

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold

def download_derivative(url_path, download_file):

    class HeadRequest(urllib2.Request):
        def get_method(self):
            return "HEAD"

    print time.strftime("%H:%M:%S"), download_file, ':',

    if os.path.exists(download_file):
        fhash = hashlib.md5(open(download_file, 'rb').read()).hexdigest()
        try:
            request = HeadRequest(url_path)
            response = urllib2.urlopen(request)
            response_headers = response.info()
            etag = re.sub(r'^"|"$', '', response_headers['etag'])
            if etag != fhash:
                os.remove(download_file)
            else:
                print 'Match'
        except urllib2.HTTPError, e:
            print ("Error code: %s" % e.code)

    if not os.path.exists(download_file):
        print 'Downloading from', url_path, 'to', download_file
        try:
            urllib.urlretrieve(url_path, download_file)
        except:
            pass
        download_data(url_path, download_file)
    return download_file


def compute_connectivity(functional):
    with np.errstate(invalid='ignore'):
        corr = np.nan_to_num(np.corrcoef(functional))
        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
        m = ma.masked_where(mask == 1, mask)
        return ma.masked_where(m, corr).compressed()


def load_patient(subj, tmpl):
    df = pd.read_csv(format_config(tmpl, {
        'subject': subj,
    }), sep="\t")
    ROIs = ['#' + str(y) for y in sorted([int(x[1:]) for x in df.keys().tolist()])]
    functional = df[ROIs].as_matrix().T
    functional = preprocessing.scale(functional, axis=1)
    functional = compute_connectivity(functional)
    functional = functional.astype(np.float32)
    return subj, functional.tolist()


def load_patients(subjs, tmpl, jobs=10):
    partial_load_patient = partial(load_patient, tmpl=tmpl)
    msg = 'Done {current} of {total}'
    return dict(run_progress(partial_load_patient, subjs, message=msg, jobs=jobs))


def store_data(pheno, ids, derivatives_data, output):

    name, ext = os.path.splitext(output)

    features = []
    classes = []

    for pid in ids:
        features.append(derivatives_data[pid])
        classes.append(pheno[pheno['FILE_ID'] == pid]['DX_GROUP'][0])

    features = np.array(features).astype(np.float32)
    classes = np.array(classes).astype(int)
    final = np.insert(features, 0, classes, axis=1)

    np.savetxt(name + ext, final, delimiter=',')
    np.savetxt(name + '.ids' + ext, ids, delimiter=',', fmt="%s")


if __name__ == "__main__":

    SEED = 19
    FOLDS = 10
    download_it = False

    pheno_path = './data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv'
    pheno = load_phenotypes(pheno_path)

    random.seed(SEED)

    data = pheno[['FILE_ID', 'DX_GROUP']].as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(data[:, 0], data[:, 1], test_size=0.188405797101, random_state=SEED, stratify=data[:, 1])

    s3_prefix = 'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/'
    download_root = './data/functionals'
    derivatives = ['cpac/filt_global/rois_cc200/{subject}_rois_cc200.1D']

    file_ids = pheno['FILE_ID'].tolist()
    for derivative in derivatives:
        if download_it:
            for file_id in file_ids:
                file_path = format_config(derivative, {'subject': file_id})
                url_path = s3_prefix + file_path
                output_file = os.path.join(download_root, file_path)
                output_dir = os.path.dirname(output_file)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                download_derivative(url_path, output_file)
        file_template = os.path.join(download_root, derivative)
        derivatives_data = load_patients(file_ids, tmpl=file_template)

        for fold, (train_index, valid_index) in enumerate(StratifiedKFold(y_train, n_folds=FOLDS, shuffle=True, random_state=SEED)):
            store_data(pheno, X_train[train_index], derivatives_data, output='./data/newcorr/corr_1D_cv_%d_train.csv' % (fold+1))
            store_data(pheno, X_train[valid_index], derivatives_data, output='./data/newcorr/corr_1D_cv_%d_valid.csv' % (fold+1))
        store_data(pheno, X_test, derivatives_data, output='./data/newcorr/corr_1D_test.csv')
        store_data(pheno, file_ids, derivatives_data, output='./data/newcorr/corr_1D.csv')
