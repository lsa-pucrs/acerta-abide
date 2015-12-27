import numpy as np
import csv
import sys
import argparse
import re
from tabulate import tabulate

def analyse_phenotype(phenotype_filename):

    phenotypes = []
    with open(phenotype_filename, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            subj = row['FILE_ID']
            if subj == 'no_filename':
                continue
            if row['DX_GROUP'] not in ['1', '2']:
                continue
            if row['SEX'] not in ['1', '2']:
                continue
            phenotypes.append([ subj, row['DX_GROUP'], row['SEX'], re.sub('_[0-9]', '', row['SITE_ID']) ])
    phenotypes = np.array(phenotypes)

    institutesset = set([ re.sub('_[0-9]', '', x) for x in set(phenotypes[:,3]) ])

    male = ( phenotypes[:,2] == '1' )
    female = ( phenotypes[:,2] == '2' )
    results = { x: [
        len(np.where( ( phenotypes[:,3] == x ) )[0]),
        len(np.where( ( phenotypes[:,3] == x ) & male )[0]),
        len(np.where( ( phenotypes[:,3] == x ) & female )[0]),
    ] for x in institutesset }

    results = [ [x]+results[x] for x in results ]
    results.append(['Mean',
        str(int(np.mean( [ len(np.where( ( phenotypes[:,3] == x ) )[0]) for x in institutesset ] ))),
        str(int(np.mean( [ len(np.where( ( phenotypes[:,3] == x ) & male )[0]) for x in institutesset ] ))),
        str(int(np.mean( [ len(np.where( ( phenotypes[:,3] == x ) & female )[0]) for x in institutesset ] )))
    ])
    results.append([len(institutesset),
        str(len(phenotypes)),
        str(len(np.where(male)[0])),
        str(len(np.where(female)[0]))
    ])

    print tabulate(results, headers=['Total', 'Male', 'Female'], tablefmt='grid')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Check .csv file")
    parser.add_argument('data_filename', help='Specifies the csv file with the values to analyse.')
    args = parser.parse_args()

    analyse_phenotype(args.data_filename)