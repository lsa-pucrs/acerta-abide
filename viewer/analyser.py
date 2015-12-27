import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import pandas as pd
from os.path import isdir, join
from os import listdir
import csv
import glob
from utils import *


def filter(data, query):
    for key in query:
        val = query[key]
        if isinstance(val, list):
            data = data[data[key].isin(val)]
        else:
            data = data[data[key] == val]
    return data


def extract(pipeline, config):
    logs = glob.glob(join(root(), 'experiments', pipeline + '.' + config, '*', 'logs', '*'))
    logs_table = []
    for log in logs:
        experiment, _ , name = log.split('/')[-3:]
        model, cv = name.split('.')[2:4]
        cv = cv[3:]
        with open(log, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                logs_table.append([pipeline, config, model, experiment, cv] + row)
    df = pd.DataFrame(logs_table, columns=['pipeline', 'config', 'model', 'experiment', 'cv', 'channel', 'epoch', 'value'])

    df['value'] = df['value'].astype(float)
    df['epoch'] = df['epoch'].astype(int)

    return df