import os, re, csv, sys, time, argparse
import multiprocessing
from utils import nrangearg
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import seaborn as sns
from functools import partial

def files(model, config, step, cv_folds):
    here = os.path.abspath('.')
    log_path = here + '/logs/' + model + '.' + config + '.' + step
    for fold in cv_folds:
        csv_path = log_path + '.cv_' + str(fold) + '.csv'
        if not os.path.exists(csv_path):
            logger.critical('Path %s does not exists.', csv_path)
            continue
        try:
            df = pd.read_csv(csv_path, sep=',', header=None)
            df.columns = ['channel', 'epoch', 'value']
            yield df
        except Exception, e:
            logger.critical('Path %s: %s', csv_path, e)
            yield None

def update_plot(frame, ax, model, config, step, channels, cv_folds, epochs, mean=False):

    values = []
    for i, df in enumerate(files(model, config, step, cv_folds)):

        values.append([])

        # File not found or not ready
        if df is None:
            continue

        for j, channel in enumerate(channels):
            channel_values = np.array(df[df['channel'] == channel]['value'].tolist()).flatten().tolist()
            if len(channel_values) < epochs:
                channel_values = channel_values + [np.nan] * (epochs - len(channel_values) + 1)
            if channel_values is None:
                channel_values = [np.nan] * epochs
            values[i].append(channel_values)

    values = np.array(values)
    if mean:
        values = np.array([np.mean(values, axis=0)])

    if args.epochs is not None:
        ax.set_xlim([0, args.epochs])

    colors = sns.color_palette(n_colors=values.shape[0] if values.shape[0] > 1 else len(channels))
    for f, fold in enumerate(values):
        for c, v in enumerate(fold):
            ax.plot(v, c=colors[f if values.shape[0] > 1 else c])

if __name__ == "__main__":

    import logging
    logging.basicConfig()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='Run cv deep-learning pipeline.')
    parser.add_argument('model', help='Model')
    parser.add_argument('config', help='Config')
    parser.add_argument('step', help='Step')
    parser.add_argument('cv_folds', type=nrangearg, help='CV Folds')
    parser.add_argument('channel', help='Channel', nargs='?', default=None)
    parser.add_argument('--epochs', type=int, help='Epochs')
    parser.add_argument('--mean', action='store_true', help='Mean of folds')
    args = parser.parse_args()

    if args.channel is None:
        uniq = []
        for df in files(args.model, args.config, args.step, args.cv_folds):
            uniq = uniq + df['channel'].tolist()
        print np.unique(uniq)
        sys.exit(1)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    update_func = partial(update_plot, ax=ax, model=args.model, config=args.config,
                            step=args.step, channels=args.channel.split(','),
                            cv_folds=args.cv_folds, epochs=args.epochs, mean=args.mean)

    ani = animation.FuncAnimation(fig, update_func, interval=1000)

    plt.show()