import argparse
import os
from utils import *
import shutil


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Clear empty experiments.')
    parser.add_argument('pipeline', help='Pipeline')
    parser.add_argument('config', help='Config')
    args = parser.parse_args()

    experiments = executed_experiments(args.pipeline, args.config)
    folder = '.'.join([args.pipeline, args.config])
    for exp in experiments:
        if len(os.listdir(os.path.join(root(), 'experiments', folder, exp, 'models'))) == 0:
            print 'Removing', os.path.join(root(), 'experiments', folder, exp)
            shutil.rmtree(os.path.join(root(), 'experiments', folder, exp), ignore_errors=True)
