import numpy as np
import csv
import sys
import argparse
from tabulate import tabulate
from utils import *

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
            phenotypes[subj] = [ subj, str(int(row['DX_GROUP'])-1), row['SEX'], re.sub('_[0-9]', '', row['SITE_ID']) ]
    return phenotypes

def analyse_data(data_filename, pheno_filename, phenotypes, print_metrics=True, print_classes=True):

    # data = np.loadtxt(data_filename, dtype=float, delimiter=',')
    # classes =  data[:,0]
    # features = data[:,1:]

    pheno = np.loadtxt(pheno_filename, delimiter=',', dtype=str)

    sex = pheno[:,2].tolist()
    print ','.join([ x + ':' + str(sex.count(x)) for x in set(sex)])

    # inst = pheno[:,3].tolist()
    # print ','.join([ x + ':' + str(inst.count(x)) for x in sorted(set(inst))])



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Check .csv file")
    parser.add_argument('cv_folds', type=nrangearg, help='CV Folds')
    parser.add_argument('data_filename', help='Specifies the csv file with the values to analyse')
    args = parser.parse_args()

    phenotypes = load_phenotypes('./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv')

    name, extension = os.path.splitext(args.data_filename)

    for fold in args.cv_folds:
        fold = str(fold)
        analyse_data(name + '_cv_' + fold + '_train' + extension, name + '_cv_' + fold + '_train.pheno' + extension, phenotypes)
    for fold in args.cv_folds:
        fold = str(fold)
        analyse_data(name + '_cv_' + fold + '_test' + extension, name + '_cv_' + fold + '_test.pheno' + extension, phenotypes)