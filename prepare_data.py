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


def prepare_data(pheno, fold_idxs, derivatives_data, output):

    name, ext = os.path.splitext(output)

    for fold, data in enumerate(fold_idxs):

        print 'fold ' + str(fold+1) + ':',

        for datatype in data:

            features = []
            classes = []
            ids = []

            for pid in data[datatype]:
                features.append(derivatives_data[pid])
                classes.append(pheno[pheno['FILE_ID'] == pid]['DX_GROUP'][0])
                ids.append(pid)

            features = np.array(features).astype(np.float32)
            classes = np.array(classes).astype(int)
            final = np.insert(features, 0, classes, axis=1)

            typename = format_config(name, {
                'fold': str(fold+1),
                'datatype': datatype,
            })

            np.savetxt(typename + ext, final, delimiter=',')
            np.savetxt(typename + '.ids' + ext, ids, delimiter=',', fmt="%s")

            print datatype,

        print


def full_data(pheno, derivatives_data, output):

    name, ext = os.path.splitext(output)

    features = []
    classes = []
    ids = []

    for pid in derivatives_data:
        features.append(derivatives_data[pid])
        classes.append(pheno[pheno['FILE_ID'] == pid]['DX_GROUP'][0])
        ids.append(pid)

    features = np.array(features).astype(np.float32)
    classes = np.array(classes).astype(int)
    final = np.insert(features, 0, classes, axis=1)

    np.savetxt(output, final, delimiter=',')
    np.savetxt(name + '.ids' + ext, ids, delimiter=',', fmt="%s")


if __name__ == "__main__":

    SEED = 19
    FOLDS = 10
    download_it = False

    pheno_path = './data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv'
    pheno = load_phenotypes(pheno_path)

    random.seed(SEED)

    fold_idxs = [{'train': [], 'valid': [], 'test': []} for i in range(FOLDS)]
    groups = pheno.groupby(('SITE_ID', 'DX_GROUP'))
    for group, data in groups:

        n = len(data)
        fold_sizes = (n // FOLDS) * np.ones(FOLDS, dtype=np.int)
        fold_sizes[:n % FOLDS] += 1

        random.shuffle(fold_sizes)

        idxs = data['FILE_ID'].tolist()

        current = 0
        for fold, fold_size in enumerate(fold_sizes):
            test = idxs[current:current+fold_size]
            if current-fold_size < 0:
                valid = idxs[current-fold_size:]
            else:
                valid = idxs[current-fold_size:current]
            train = [idx for idx in idxs if idx not in test and idx not in valid]
            fold_idxs[fold]['train'] = fold_idxs[fold]['train'] + train
            fold_idxs[fold]['valid'] = fold_idxs[fold]['valid'] + valid
            fold_idxs[fold]['test'] = fold_idxs[fold]['test'] + test
            current = current + fold_size

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

        means = []
        means_perclass = [[], []]
        derivatives_data = load_patients(file_ids, tmpl=file_template)
        for pid in derivatives_data:
            m = np.mean(derivatives_data[pid])
            means.append(m)
            means_perclass[pheno[pheno['FILE_ID'] == pid]['DX_GROUP'][0]].append(m)
        print np.mean(means), np.std(means)
        print np.mean(means_perclass[0]), np.mean(means_perclass[1])
        print np.std(means_perclass[0]), np.std(means_perclass[1])

        prepare_data(pheno, fold_idxs, derivatives_data, output='./data/corr/corr_1D_cv_{fold}_{datatype}.csv')
        full_data(pheno, derivatives_data, output='./data/corr/corr_1D.csv')
