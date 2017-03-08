#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Data preparation

Usage:
  prepare_data.py [--folds=N] [--whole] [--male] [--threshold] [<derivative> ...]
  prepare_data.py (-h | --help)

Options:
  -h --help           Show this screen
  --folds=N           Number of folds [default: 10]
  --whole             Run model for the whole dataset
  --male              Run model for male subjects
  --threshold         Run model for thresholded subjects
  derivative          Derivatives to process

"""

import os
import random
import pandas as pd
import numpy as np
import numpy.ma as ma
from docopt import docopt
from functools import partial
from sklearn import preprocessing
from utils import (load_phenotypes, format_config, run_progress)


def compute_connectivity(functional):
    with np.errstate(invalid="ignore"):
        corr = np.nan_to_num(np.corrcoef(functional))
        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
        m = ma.masked_where(mask == 1, mask)
        return ma.masked_where(m, corr).compressed()


def load_patient(subj, tmpl):
    df = pd.read_csv(format_config(tmpl, {
        "subject": subj,
    }), sep="\t", header=0)
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    ROIs = ["#" + str(y) for y in sorted([int(x[1:]) for x in df.keys().tolist()])]

    functional = np.nan_to_num(df[ROIs].as_matrix().T)
    functional = preprocessing.scale(functional, axis=1)
    functional = compute_connectivity(functional)
    functional = functional.astype(np.float32)

    return subj, functional.tolist()


def load_patients(subjs, tmpl, jobs=10):
    partial_load_patient = partial(load_patient, tmpl=tmpl)
    msg = "Processing {current} of {total}"
    return dict(run_progress(partial_load_patient, subjs, message=msg, jobs=jobs))


def prepare_data(pheno, fold_idxs, derivative, func_data, output):

    name, ext = os.path.splitext(output)

    for fold, data in enumerate(fold_idxs):

        for datatype in data:

            features = []
            classes = []
            ids = []

            for pid in data[datatype]:
                features.append(func_data[pid])
                classes.append(pheno[pheno["FILE_ID"] == pid]["DX_GROUP"][0])
                ids.append(pid)

            features = np.array(features).astype(np.float32)
            classes = np.array(classes).astype(int)
            final = np.insert(features, 0, classes, axis=1)

            typename = format_config(name, {
                "fold": str(fold+1),
                "datatype": datatype,
                "derivative": derivative,
            })

            np.savetxt(typename + ext, final, delimiter=",")
            np.savetxt(typename + ".ids" + ext, ids, delimiter=",", fmt="%s")


def prepare_folds(folds, pheno, derivatives, output):

    fold_idxs = [{"train": [], "valid": [], "test": []} for i in range(folds)]
    groups = pheno.groupby(("SITE_ID", "DX_GROUP"))
    for group, data in groups:

        n = len(data)
        fold_sizes = (n // folds) * np.ones(folds, dtype=np.int)
        fold_sizes[:n % folds] += 1

        random.shuffle(fold_sizes)

        idxs = data["FILE_ID"].tolist()

        current = 0
        for fold, fold_size in enumerate(fold_sizes):
            test = idxs[current:current+fold_size]
            if current-fold_size < 0:
                valid = idxs[current-fold_size:]
            else:
                valid = idxs[current-fold_size:current]
            train = [idx for idx in idxs if idx not in test and idx not in valid]
            fold_idxs[fold]["train"] = fold_idxs[fold]["train"] + train
            fold_idxs[fold]["valid"] = fold_idxs[fold]["valid"] + valid
            fold_idxs[fold]["test"] = fold_idxs[fold]["test"] + test
            current = current + fold_size

    download_root = "./data/functionals"
    derivatives_path = {
        "aal": "cpac/filt_global/rois_aal/{subject}_rois_aal.1D",
        "cc200": "cpac/filt_global/rois_cc200/{subject}_rois_cc200.1D",
        "dosenbach160": "cpac/filt_global/rois_dosenbach160/{subject}_rois_dosenbach160.1D",
        "ez": "cpac/filt_global/rois_ez/{subject}_rois_ez.1D",
        "ho": "cpac/filt_global/rois_ho/{subject}_rois_ho.1D",
        "tt": "cpac/filt_global/rois_tt/{subject}_rois_tt.1D",
    }

    file_ids = pheno["FILE_ID"].tolist()
    for derivative in derivatives:
        file_template = os.path.join(download_root, derivatives_path[derivative])

        means = []
        means_perclass = [[], []]
        func_data = load_patients(file_ids, tmpl=file_template)
        for pid in func_data:
            m = np.mean(func_data[pid])
            means.append(m)
            means_perclass[pheno[pheno["FILE_ID"] == pid]["DX_GROUP"][0]].append(m)

        print "Class=All, Mean={:.6f}, Std={:.6f}".format(np.mean(means), np.std(means))
        print "Class=ASD, Mean={:.6f}, Std={:.6f}".format(np.mean(means_perclass[0]), np.std(means_perclass[0]))
        print "Class= TC, Mean={:.6f}, Std={:.6f}".format(np.mean(means_perclass[1]), np.std(means_perclass[1]))

        prepare_data(pheno, fold_idxs, derivative, func_data, output=output)


if __name__ == "__main__":

    random.seed(19)
    np.random.seed(19)

    arguments = docopt(__doc__)

    folds = int(arguments["--folds"])
    pheno_path = "./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv"
    pheno = load_phenotypes(pheno_path)

    valid_derivatives = ["cc200", "aal", "ez", "ho", "tt", "dosenbach160"]
    derivatives = [derivative for derivative in arguments["<derivative>"] if derivative in valid_derivatives]

    if arguments["--whole"]:
        print
        print "Preparing whole dataset"
        prepare_folds(folds, pheno, derivatives, output="./data/corr/{derivative}_whole_{fold}_{datatype}.csv")

    if arguments["--male"]:
        print
        print "Preparing male dataset"
        pheno_male = pheno[pheno["SEX"] == "M"]
        prepare_folds(folds, pheno_male, derivatives, output="./data/corr/{derivative}_male_{fold}_{datatype}.csv")

    if arguments["--threshold"]:
        print
        print "Preparing thresholded dataset"
        pheno_thresh = pheno[pheno["MEAN_FD"] <= 0.2]
        prepare_folds(folds, pheno_thresh, derivatives, output="./data/corr/{derivative}_threshold_{fold}_{datatype}.csv")
