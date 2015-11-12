import os
import multiprocessing as mp
import numpy as np
import numpy.ma as ma
import nibabel as nb
import csv
from tabulate import tabulate
from functools import partial
import sys, time
import re
from sklearn import preprocessing
from sklearn import cross_validation

def compute_parcellations(functional, mask):
    mask_vals = np.unique( mask )
    mask_vals = np.array([i for i in mask_vals if i > 0])
    avg_ts = np.zeros( ( mask_vals.shape[0] , functional.shape[1] ) )
    for i, r in enumerate(mask_vals):
        avg_ts[ i , : ] = np.mean( functional[ np.where( mask == r )[0] , : ] , 0 )
    return avg_ts

def compute_connectivity(functional):
    with np.errstate(invalid='ignore'):
        corr = np.nan_to_num(np.corrcoef(functional))
        mask = np.invert(np.tri(corr.shape[0], k = -1, dtype=bool))
        m = ma.masked_where(mask == 1, mask)
        return ma.masked_where(m, corr).compressed()

def load_patient(subj, mask=None, patient_template=None):
    functional = nb.load(patient_template % subj).get_data()
    flatten_shape = ( np.prod(functional.shape[0:3]) , np.prod(functional.shape[3:]) )
    functional = np.reshape( functional , flatten_shape )
    if mask is not None:
        functional = compute_parcellations(functional, mask)
    functional = compute_connectivity(functional)
    return subj, functional.tolist()

def load_patients(subjs, mask, patient_template, jobs=10):

    mask = nb.load(mask)
    mask = np.reshape( mask.get_data() , np.prod(mask.shape[0:3]) )

    results = []
    partial_load_patient = partial(load_patient, mask=mask, patient_template=patient_template)

    pool = mp.Pool(processes=jobs)
    [pool.apply_async(partial_load_patient, (subj,), callback=results.append) for subj in subjs]

    total = len(subjs)
    complete = 0
    while complete < total:
        complete = len(results)
        sys.stdout.write("\rDone {} of {}".format(complete, total))
        sys.stdout.flush()
        time.sleep(0.5)

    pool.close()
    pool.join()

    print ''

    return results;

def load_phenotypes(file):
    phenotypes = {}
    with open(file, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            subj = row['FILE_ID']
            if subj == 'no_filename':
                continue
            if row['DX_GROUP'] not in ['1', '2']:
                continue
            # if row['SEX'] not in ['1', '2']:
            #     continue
            phenotypes[subj] = [ int(row['DX_GROUP'])-1, row['SEX'], re.sub('_[0-9]', '', row['SITE_ID']) ]
    return phenotypes

def prepare_data(phenotypes_file, mask, patient_template):

    phenotypes = load_phenotypes(phenotypes_file)
    results = load_patients(phenotypes.keys(), mask=mask, patient_template=patient_template)

    features = []
    ids = []
    classes = []
    for pid, feats in results:
        features.append(feats)
        ids.append(pid)
        classes.append(phenotypes[pid][0])

    features = np.array(features).astype(np.float32)
    classes = np.array(classes)

    # institutesset = set([ re.sub('_[0-9]', '', x) for x in set(phenotypes[:,3]) ])

    # male = ( phenotypes[:,2] == '1' )
    # female = ( phenotypes[:,2] == '2' )
    # results = { x: [
    #     len(np.where( ( phenotypes[:,3] == x ) )[0]),
    #     len(np.where( ( phenotypes[:,3] == x ) & male )[0]),
    #     len(np.where( ( phenotypes[:,3] == x ) & female )[0]),
    # ] for x in institutesset }

    # results = [ [x]+results[x] for x in results ]

    # results.append(['',
    #     str(len(phenotypes)),
    #     str(len(np.where(male)[0])),
    #     str(len(np.where(female)[0]))
    # ])

    # print tabulate(results, headers=['Total', 'Male', 'Female'], tablefmt='grid')

    # for institute in institutesset:
    #     patients_ids = phenotypes[np.where((phenotypes[:,3] != institute))][:,0]
    #     results = np.array(list([ data[p] for p in patients_ids ]))
    #     np.savetxt('./data/corr/corr-looeft-'+ institute +'.csv', results, delimiter=',')
        # results[:,1:] = results[:,1:] + 1
        # results[:,1:] = results[:,1:] / 2
        # np.savetxt('./data/corr/corr-norm-looeft-'+ institute +'.csv', results, delimiter=',')

    print np.unique(classes)

    print features.shape, features.dtype
    print np.min(features), np.max(features)
    np.savetxt('./data/corr/corr.csv', np.insert(features, 0, classes, axis=1), delimiter=',')

    # scaled = preprocessing.scale(features, axis=0)
    # print scaled.shape, scaled.dtype
    # print np.min(scaled), np.max(scaled)
    # np.savetxt('./data/corr/corr-norm.csv', np.insert(scaled, 0, classes, axis=1), delimiter=',')

def prepare_data_cv(filename, folds, seed=42):
    name, extension = os.path.splitext(filename)
    data = np.loadtxt(filename, delimiter=',')
    X = data[:,1:]
    y = data[:,0]
    cv = cross_validation.StratifiedKFold(y=y, n_folds=folds, shuffle=True, random_state=seed)
    for fold, (train_index, test_index) in enumerate(cv):
        fold = str(fold + 1)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train = np.insert(X_train, 0, y_train, 1)
        test = np.insert(X_test, 0, y_test, 1)
        np.savetxt(name + '_cv_' + fold + '_train' + extension, train, delimiter=',')
        np.savetxt(name + '_cv_' + fold + '_test' + extension, test, delimiter=',')

if __name__ == "__main__":

    prepare_data('./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv',
                 './data/masks/cc200.nii.gz',
                 './data/functionals/cpac/filt_global/func_preproc/%s_func_preproc.nii.gz')
    prepare_data_cv('./data/corr/corr.csv', 10, seed=42)
    # prepare_data_cv('./data/corr/corr-norm.csv', 10, seed=19)
