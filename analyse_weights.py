import numpy as np
from pylearn2.utils import serial
from sklearn.preprocessing import scale
from utils import *
import argparse


def extract_weights(pipeline, config, model, folds, experiments):

    for experiment in experiments:

        gs = []

        folder = format_config('experiments/{pipeline}.{config}/{experiment}', {
            'pipeline': pipeline,
            'config': config,
            'experiment': experiment,
        })

        if not os.path.isdir(folder + '/analysis'):
            os.makedirs(folder + '/analysis')

        file_templ = folder + '/analysis/{model}_weights_{fold}'

        for fold in folds:

            f = format_config(folder + '/models/{pipeline}.{config}.{model}_cv_{fold}.pkl', {
                'pipeline': pipeline,
                'config': config,
                'model': model,
                'fold': fold,
            })

            model_pkl = serial.load(f)

            ae1wft = model_pkl.layers[0].layer_content.get_param_values()[2].astype(np.float64)
            ae1bft = model_pkl.layers[0].layer_content.get_param_values()[1].astype(np.float64)

            ae2wft = model_pkl.layers[1].layer_content.get_param_values()[2].astype(np.float64)
            ae2bft = model_pkl.layers[1].layer_content.get_param_values()[1].astype(np.float64)

            somw = model_pkl.layers[2].get_param_values()[1].astype(np.float64)
            somb = model_pkl.layers[2].get_param_values()[0].astype(np.float64)

            del model_pkl

            ae1 = ae1wft + ae1bft
            ae2 = ae2wft + ae2bft
            som = somw + somb

            g = scale(np.dot(np.dot(ae1, ae2), som).T, axis=1)

            out = format_config(file_templ, {
                'model': model,
                'fold': fold,
            })

            np.savetxt(out, g, delimiter=',')
            gs.append(g)

            print format_config('Fold {fold} done', {
                'fold': fold,
            })

        gs = np.mean(np.array(gs), axis=0)
        out = format_config(file_templ, {
            'model': model,
            'fold': 'mean',
        })
        np.savetxt(out, gs, delimiter=',')

        print 'Mean done'

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate cv deep-learning pipeline.')
    parser.add_argument('pipeline', help='Pipeline')
    parser.add_argument('config', help='Config')
    parser.add_argument('model', help='Model')
    parser.add_argument('cv_folds', type=nrangearg, help='CV Folds')
    parser.add_argument('--experiment', type=str, default=None, help='Experiment ID')
    args = parser.parse_args()

    experiments = executed_experiments(args.pipeline, args.config)
    if args.experiment is not None:
        experiments = [args.experiment]
    extract_weights(args.pipeline, args.config, args.model, args.cv_folds, experiments)
