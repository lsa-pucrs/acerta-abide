#!/usr/bin/env python
import os
import numpy as np
import numpy.ma as ma
import nibabel as nb
import pandas as pd
import urllib
import urllib2
import hashlib
from functools import partial
from sklearn import preprocessing
from sklearn import cross_validation
from utils import *


class HeadRequest(urllib2.Request):
    def get_method(self):
        return "HEAD"


def download_data(url_path, download_file):
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


def compute_parcellations(functional, mask):
    mask_vals = np.unique(mask)
    mask_vals = np.array([i for i in mask_vals if i > 0])
    avg_ts = np.zeros((mask_vals.shape[0], functional.shape[1]))
    for i, r in enumerate(mask_vals):
        avg_ts[i, :] = np.mean(functional[np.where(mask == r)[0], :], 0)
    return avg_ts


def compute_connectivity(functional):
    with np.errstate(invalid='ignore'):
        corr = np.nan_to_num(np.corrcoef(functional))
        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
        m = ma.masked_where(mask == 1, mask)
        return ma.masked_where(m, corr).compressed()


def load_patient(subj, tmpl, mask):
    if tmpl[-2:] == '1D':
        df = pd.DataFrame.from_csv(tmpl % subj, sep="\t")
        functional = df.as_matrix().T
        functional = preprocessing.scale(functional, axis=1)
    else:
        functional = nb.load(tmpl % subj).get_data()
        voxels = np.prod(functional.shape[0:3])
        flatten_shape = (voxels, np.prod(functional.shape[3:]))
        functional = np.reshape(functional, flatten_shape)
        if mask is not None:
            functional = compute_parcellations(functional, mask)
    functional = compute_connectivity(functional)
    return subj, functional.tolist()


def load_patients(subjs, mask, tmpl, jobs=10):

    if mask is not None:
        mask = nb.load(mask)
        mask = np.reshape(mask.get_data(), np.prod(mask.shape[0:3]))

    partial_load_patient = partial(load_patient, tmpl=tmpl, mask=mask)
    msg = 'Done {current} of {total}'
    return run_progress(partial_load_patient, subjs, message=msg, jobs=jobs)

def prepare_data(phenotypes_file, tmpl, destination, mask=None):
    """TODO Fill in with a description"""

    phenotypes = load_phenotypes(phenotypes_file)
    keys = phenotypes.index.get_values()
    results = load_patients(keys, mask=mask, tmpl=tmpl)

    name, ext = os.path.splitext(destination)

    features = []
    ids = []
    classes = []
    phenos = []

    for pid, feats in results:
        pdata = phenotypes[phenotypes['FILE_ID'] == pid].iloc[0,:].tolist()
        features.append(feats)
        ids.append(pid)
        classes.append(phenotypes[phenotypes['FILE_ID'] == pid]['DX_GROUP'])
        phenos.append(pdata)

    features = np.array(features).astype(np.float32)
    classes = np.array(classes)

    final = np.insert(features, 0, classes, axis=1)
    np.savetxt(name + ext, final, delimiter=',')
    np.savetxt(name + '.pheno' + ext, phenos, delimiter=',', fmt="%s")

    scaled = preprocessing.scale(features)

    final = np.insert(scaled, 0, classes, axis=1)
    np.savetxt(name + '-norm' + ext, final, delimiter=',')
    np.savetxt(name + '-norm.pheno' + ext, phenos, delimiter=',', fmt="%s")


def prepare_data_cv(filename, folds, seed=42, pheno_file=True):
    name, ext = os.path.splitext(filename)
    data = np.loadtxt(filename, delimiter=',')

    if pheno_file:
        pfile = name + '.pheno' + ext
        pheno = np.genfromtxt(pfile, delimiter=',', dtype=np.str)

    cv = cross_validation.StratifiedKFold(y=data[:, 0], n_folds=folds,
                                          shuffle=True, random_state=seed)

    for fold, (train_index, test_index) in enumerate(cv):
        fold = str(fold + 1)

        np.savetxt(name + '_cv_' + fold + '_train' + ext,
                   data[train_index], delimiter=',')

        np.savetxt(name + '_cv_' + fold + '_test' + ext,
                   data[test_index], delimiter=',')

        if pheno_file:
            np.savetxt(name + '_cv_' + fold + '_train.pheno' +
                       ext, pheno[train_index], delimiter=',', fmt="%s")
            np.savetxt(name + '_cv_' + fold + '_test.pheno' +
                       ext, pheno[test_index], delimiter=',', fmt="%s")

        print filename, fold

if __name__ == "__main__":

    seed = 42
    folds = 10
    download_it = True

    pheno_path = './data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv'
    pheno = load_phenotypes(pheno_path)

    s3_prefix = 'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/'

    if download_it:

        download_root = './data/functionals'

        derivatives = [
            'cpac/filt_global/func_preproc/{}_func_preproc.nii.gz',
            'cpac/filt_global/rois_cc200/{}_rois_cc200.1D',
        ]

        file_ids = pheno['FILE_ID'].tolist()

        for derivative in derivatives:
            for file_id in file_ids:

                file_path = derivative.format(file_id)

                url_path = s3_prefix + file_path
                download_file = os.path.join(download_root, file_path)

                download_dir = os.path.dirname(download_file)
                if not os.path.exists(download_dir):
                    os.makedirs(download_dir)

                download_data(url_path, download_file)

    # Our own pipeline (create our version of CC200)
    tmpl = './data/functionals/cpac/filt_global/func_preproc/%s_func_preproc.nii.gz'
    prepare_data(pheno_path, tmpl, './data/corr/corr.csv', mask='./data/masks/cc200.nii.gz') # Apply CC200 and create the connectivity matrix
    prepare_data_cv('./data/corr/corr.csv', folds, seed=seed) # Create folds for training and cv
    prepare_data_cv('./data/corr/corr-norm.csv', folds, seed=seed) # Scale data to mean 0 sd 1

    # ABIDE's pipeline (their own CC200 parcellations)
    tmpl = './data/functionals/cpac/filt_global/rois_cc200/%s_rois_cc200.1D'
    prepare_data(pheno_path, tmpl, './data/corr/corr_1D.csv')
    prepare_data_cv('./data/corr/corr_1D.csv', folds, seed=seed)
    prepare_data_cv('./data/corr/corr_1D-norm.csv', folds, seed=seed)
