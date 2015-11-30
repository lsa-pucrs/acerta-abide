import json
import numpy as np
import numpy.ma as ma
import nibabel as nb
import sys, time
import re
import pickle

if __name__ == "__main__":

    pkl_file = open('data.pkl', 'rb')
    inputs = pickle.load(pkl_file)
    data = pickle.load(pkl_file)
    pkl_file.close()

    folds = 10

    ASD = 0
    TC = 1

    CORR = 1
    ANTICORR = 0

    correlations = np.empty((inputs.shape[0], folds, 2))

    for fold in range(folds):
        for feature in range(inputs.shape[0]):
            for klass in [ASD, TC]:
                maximum = np.max([ data[fold][feature][CORR][klass] , data[fold][feature][ANTICORR][klass] ])
                modifier =  -1 if data[fold][feature][ANTICORR][klass] > data[fold][feature][CORR][klass] else 1
                correlations[feature][fold][klass] = maximum * modifier
                # correlations[feature][fold][klass] = data[fold][feature][ANTICORR][klass] - data[fold][feature][CORR][klass]

    meancorr = np.mean(correlations, axis=1)
    print np.min(meancorr), np.max(meancorr)
    print meancorr.shape

    elems = {}

    for i in range(inputs.shape[0]):
        f, t = inputs[i].split(',')
        if f not in elems:
            elems[f] = {}
        if t not in elems:
            elems[t] = {}
        elems[f][t] = meancorr[i].tolist()
        elems[t][f] = meancorr[i].tolist()

    d3data = []
    for e in elems:
        piece = {
            'region': e,
            'connections': []
        }
        for k in elems[e]:
            if abs(elems[e][k]) > 0.55:
                piece['connections'].append({
                    'region': k,
                    'value': elems[e][k]
                })
        if len(piece['connections']) > 0:
            d3data.append(piece)

    with open('data.json', 'w') as outfile:
        json.dump(d3data, outfile)