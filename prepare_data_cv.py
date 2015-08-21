import os, csv, argparse
import numpy as np
from sklearn import cross_validation

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

    parser = argparse.ArgumentParser(description='Turn data into cv-folds.')
    parser.add_argument('n', help='Number of folds', type=int)
    parser.add_argument('seed', help='Random seed', type=int)
    parser.add_argument('files', nargs='+', help='Files to be shuffled')
    args = parser.parse_args()

    for file in args.files:
        prepare_data_cv(file, args.n, seed=args.seed)