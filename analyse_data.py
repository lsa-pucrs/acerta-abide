import numpy as np
import csv
import sys
import argparse
from tabulate import tabulate

def rawcount(filename):
    f = open(filename, 'rb')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)

    return lines

def analyse_data(data_filename, print_metrics=True, print_classes=True):

    data = np.loadtxt(data_filename, dtype=float, delimiter=',')
    classes =  data[:,0]
    features = data[:,1:]

    print tabulate([[data_filename]], tablefmt='grid')

    if print_metrics:
        metrics = [[features.shape[0], features.shape[1], np.min(features), np.max(features)]]
        print tabulate(metrics, headers=['Examples','Features', 'Min', 'Max'], tablefmt='grid')

    if print_classes:
        class_names, counts = np.unique(classes, return_counts=True)
        results = np.array([
            class_names,
            counts
        ])
        print tabulate(results.T, headers=['Classes', 'Count'], tablefmt='grid')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Check .csv file")
    parser.add_argument('data_filenames', nargs='+', help='Specifies the csv file with the values to transform')
    parser.add_argument('--print-metrics', action='store_true')
    parser.add_argument('--print-classes', action='store_true')
    args = parser.parse_args()

    print_metrics = args.print_metrics or not args.print_classes
    print_classes = args.print_classes or not args.print_metrics

    for filename in args.data_filenames:
        analyse_data(filename, print_metrics=print_metrics, print_classes=print_classes)