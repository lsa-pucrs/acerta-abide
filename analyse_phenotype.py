import numpy as np
import pandas as pd
import sys
import argparse
import re
from tabulate import tabulate

def analyse_phenotype(phenotype_filename):

    pheno = pd.read_csv(phenotype_filename)
    pheno = pheno[pheno['FILE_ID'] != 'no_filename']
    pheno['SITE_ID'] = pheno['SITE_ID'].apply(lambda v: re.sub('_[0-9]', '', v))

    sex_fmt = '%d (M %d, F %d)'

    data = []
    for group, examples in pheno.groupby('SITE_ID'):
        asd = examples['DX_GROUP'] == 1
        tc = examples['DX_GROUP'] == 2
        asd_age = examples[asd]['AGE_AT_SCAN'].mean()
        asd_sex = examples[asd]['SEX'].value_counts().to_dict() # int index
        tc_age = examples[tc]['AGE_AT_SCAN'].mean()
        tc_sex = examples[tc]['SEX'].value_counts().to_dict() # int index

        if 1 not in asd_sex:
            asd_sex[1] = 0
        if 2 not in asd_sex:
            asd_sex[2] = 0
        if 1 not in tc_sex:
            tc_sex[1] = 0
        if 2 not in tc_sex:
            tc_sex[2] = 0

        data.append([group, asd_age, sex_fmt % (asd_sex[1] + asd_sex[2], asd_sex[1], asd_sex[2]), tc_age, sex_fmt % (tc_sex[1] + tc_sex[2], tc_sex[1], tc_sex[2])])

    asd = pheno['DX_GROUP'] == 1
    tc = pheno['DX_GROUP'] == 2
    asd_age = pheno[asd]['AGE_AT_SCAN'].mean()
    asd_sex = pheno[asd]['SEX'].value_counts().to_dict()
    tc_age = pheno[tc]['AGE_AT_SCAN'].mean()
    tc_sex = pheno[tc]['SEX'].value_counts().to_dict()

    data.append(["All", asd_age, sex_fmt % (asd_sex[1] + asd_sex[2], asd_sex[1], asd_sex[2]), tc_age, sex_fmt % (tc_sex[1] + tc_sex[2], tc_sex[1], tc_sex[2])])

    print tabulate(data, headers=['Institute', 'ASD Age', 'ASD Count', 'TC Age', 'TC Count'], tablefmt='plain')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Check .csv file")
    parser.add_argument('data_filename', help='Specifies the csv file with the values to analyse.')
    args = parser.parse_args()

    analyse_phenotype(args.data_filename)
