#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Data preparation

Usage:
  prepare_data.py [--folds=N] [--whole] [--male] [--threshold] [--leave-site-out] [<derivative> ...]
  prepare_data.py (-h | --help)

Options:
  -h --help           Show this screen
  --folds=N           Number of folds [default: 10]
  --whole             Prepare data of the whole dataset
  --male              Prepare data of male subjects
  --threshold         Prepare data of thresholded subjects
  --leave-site-out    Prepare data using leave-site-out method
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from utils import (load_phenotypes, format_config, run_progress, hdf5_handler)


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

    functional = np.nan_to_num(df[ROIs].to_numpy().T).tolist()
    functional = preprocessing.scale(functional, axis=1)
    functional = compute_connectivity(functional)
    functional = functional.astype(np.float32)

    return subj, functional.tolist()


def load_patients(subjs, tmpl, jobs=1):
    partial_load_patient = partial(load_patient, tmpl=tmpl)
    msg = "Processing {current} of {total}"
    return dict(run_progress(partial_load_patient, subjs, message=msg, jobs=jobs))


def prepare_folds(hdf5, folds, pheno, derivatives, experiment):
    exps = hdf5.require_group("experiments")
    ids = pheno["FILE_ID"]

    for derivative in derivatives:
        exp = exps.require_group(format_config(
            experiment,
            {
                "derivative": derivative,
            }
        ))

        exp.attrs["derivative"] = derivative

        skf = StratifiedKFold(n_splits=folds, shuffle=True)
        for i, (train_index, test_index) in enumerate(skf.split(ids, pheno["STRAT"])):
            train_index, valid_index = train_test_split(train_index, test_size=0.33)
            fold = exp.require_group(str(i))

            fold['train'] = [ind.encode('utf8') for ind in ids[train_index]] 

            fold['valid'] = [indv.encode('utf8') for indv in ids[valid_index]]
 
            fold["test"] = [indt.encode('utf8') for indt in ids[test_index]]

            # fold["train"] = ids[train_index].tolist()
            # fold["valid"] = ids[valid_index].tolist()
            # fold["test"] = ids[test_index].tolist()


def load_patients_to_file(hdf5, pheno, derivatives):

    download_root = "./data/functionals"
    derivatives_path = {
        "aal": "cpac/filt_global/rois_aal/{subject}_rois_aal.1D",
        "cc200": "cpac/filt_global/rois_cc200/{subject}_rois_cc200.1D",
        "dosenbach160": "cpac/filt_global/rois_dosenbach160/{subject}_rois_dosenbach160.1D",
        "ez": "cpac/filt_global/rois_ez/{subject}_rois_ez.1D",
        "ho": "cpac/filt_global/rois_ho/{subject}_rois_ho.1D",
        "tt": "cpac/filt_global/rois_tt/{subject}_rois_tt.1D",
    }
    #print('storing_patients')
    storage = hdf5.require_group("patients")
    file_ids = pheno["FILE_ID"].tolist()
    #print('storing_finished')

    for derivative in derivatives:

        #print('derivative_loop')
        file_template = os.path.join(download_root, derivatives_path[derivative])
        # print('one_over')
        func_data = load_patients(file_ids, tmpl=file_template)
        #print('two_over')

        for pid in func_data:
            print('func_data_filling')
            record = pheno[pheno["FILE_ID"] == pid].iloc[0]
            patient_storage = storage.require_group(pid)
            patient_storage.attrs["id"] = record["FILE_ID"]
            patient_storage.attrs["y"] = record["DX_GROUP"]
            patient_storage.attrs["site"] = record["SITE_ID"]
            patient_storage.attrs["sex"] = record["SEX"]
            patient_storage.create_dataset(derivative, data=func_data[pid])

if __name__ == "__main__":

    random.seed(19)
    np.random.seed(19)

    arguments = docopt(__doc__)

    folds = int(arguments["--folds"])
    pheno_path = "./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv"
    pheno = load_phenotypes(pheno_path)

    hdf5 = hdf5_handler(bytes("./data/abide.hdf5",encoding="utf8"), 'a')

    valid_derivatives = ["cc200", "aal", "ez", "ho", "tt", "dosenbach160"]
    derivatives = [derivative for derivative in arguments["<derivative>"] if derivative in valid_derivatives]

    if "patients" not in hdf5:
        load_patients_to_file(hdf5, pheno, derivatives)

    if arguments["--whole"]:
        
        print ("Preparing whole dataset")
        prepare_folds(hdf5, folds, pheno, derivatives, experiment="{derivative}_whole")

    if arguments["--male"]:
        
        print ("Preparing male dataset")
        pheno_male = pheno[pheno["SEX"] == "M"]
        prepare_folds(hdf5, folds, pheno_male, derivatives, experiment="{derivative}_male")

    if arguments["--threshold"]:
        
        print ("Preparing thresholded dataset")
        pheno_thresh = pheno[pheno["MEAN_FD"] <= 0.2]
        prepare_folds(hdf5, folds, pheno_thresh, derivatives, experiment="{derivative}_threshold")

    if arguments["--leave-site-out"]:
        
        # print('Hi')
        print ("Preparing leave-site-out dataset")
        for site in pheno["SITE_ID"].unique():
            pheno_without_site = pheno[pheno["SITE_ID"] != site]
            prepare_folds(hdf5, folds, pheno_without_site, derivatives, experiment=format_config(
                "{derivative}_leavesiteout-{site}",
                {
                    "site": site,
                })
            )
