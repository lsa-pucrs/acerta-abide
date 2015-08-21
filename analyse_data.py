import numpy as np
import csv
import sys
import argparse
from tabulate import tabulate

def analyse_data(data_filename):

    data = np.loadtxt(data_filename, dtype=float, delimiter=',')
    classes =  data[:,0]
    features = data[:,1:]
    class_names, counts = np.unique(classes, return_counts=True)
    metrics = [[features.shape[0], features.shape[1], np.min(features), np.max(features)]]

    results = np.array([
        class_names,
        counts
    ])

    print tabulate(metrics, headers=['Examples','Features', 'Min', 'Max'], tablefmt='grid')
    print tabulate(results.T, headers=['Classes', 'Count'], tablefmt='grid')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Check .csv file")
    parser.add_argument('data_filename', help='Specifies the csv file with the values to transform')
    args = parser.parse_args()

    analyse_data(args.data_filename)