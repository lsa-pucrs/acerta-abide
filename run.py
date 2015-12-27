import os
import sys
import time
import yaml
import argparse
import traceback
from functools import partial
from utils import *


def run_model(fold_config):

    from pylearn2.config import yaml_parse
    from pylearn2.utils import serial
    from pylearn2.utils.logger import restore_defaults
    from best_params import MonitorBasedSaveBest

    restore_defaults()

    content = open(fold_config['run_config_path'] + '/' + fold_config['step'] + '.yaml').read()
    yaml = format_config(content, fold_config)

    with open(fold_config['experiment_config_path'] + '/' + fold_config['step'] + '.yaml', "w") as yaml_file:
        yaml_file.write(yaml)

    start = time.time()

    try:
        step_model = yaml_parse.load(yaml)

        save_path = None
        for e in step_model.extensions:
            if not isinstance(e, MonitorBasedSaveBest):
                continue
            if e.save_path is not None:
                save_path = e.save_path
                e.save_path = None
                e.store_best_model = True

        step_model.main_loop()

        fold_config['elapsed'] = elapsed_time(start)

        for e in step_model.extensions:
            if isinstance(e, MonitorBasedSaveBest) and save_path is not None:
                print 'Saving %s' % save_path
                serial.save(save_path, e.best_model, on_overwrite='ignore')

    except Exception, e:
        traceback.print_exc(file=sys.stdout)
        raise e


def run_config_cv(config, gpus, threads):

    t0 = time.time()

    with open(config['run_config_file']) as f:
        pipeline = yaml.load(f)
    yaml_config = compute_config(config, pipeline['config']['common'], replace=False)

    for step in pipeline['pipeline']:
        step_config = {}
        if step in pipeline['config']:
            step_config = pipeline['config'][step]
        step_config['step'] = step
        step_config = compute_config(yaml_config, step_config, replace=False)

        theano_run = partial(parallel_theano, execute=run_model, concurr_key='fold')
        run_parallel(theano_run, step_config, gpus=gpus, threads=threads, concurr_key='fold')

    print "%s: %s" % ("Total", elapsed_time(t0))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run cv deep-learning pipeline.')
    parser.add_argument('pipeline', help='Pipeline')
    parser.add_argument('config', help='Config')
    parser.add_argument('cv_folds', type=nrangearg, help='CV Folds')
    parser.add_argument('--experiment', type=str, default=None, help='Experiment ID')
    parser.add_argument('--gpus', type=gpurangearg, default=[], help='Number of gpus')
    parser.add_argument('--threads', type=int, help='Number of threads')
    args = parser.parse_args()

    here = os.path.abspath('.')

    config = {
        'data_path': here + '/data',
        'config_path': here + '/config',
        'experiment_path': here + '/experiments/{namespace}/{run}',
        'experiment_config_path': '{experiment_path}/config',
        'log_path': '{experiment_path}/logs',
        'analysis_path': '{experiment_path}/analysis',
        'result_path': '{experiment_path}/models',
    }

    config['run_config_path'] = config['config_path'] + '/' + args.pipeline
    config['run_config_file'] = config['config_path'] + '/' + args.pipeline + '/' + args.config + '.yaml'

    config['namespace'] = args.pipeline + '.' + args.config
    config['pipeline'] = args.pipeline
    config['config'] = args.config
    config['fold'] = args.cv_folds

    if args.experiment is not None:
        config['run'] = args.experiment
    else:
        config['run'] = str(int(time.time()))

    if not os.path.isdir(config['run_config_path']):
        print 'Path %s does not exists.' % config['run_config_path']
        sys.exit(1)

    if not os.path.exists(config['run_config_file']):
        print 'Path %s does not exists.' % config['run_config_file']
        sys.exit(1)

    # TODO review
    config = compute_config(config)
    config = compute_config(config)

    if not os.path.isdir(config['experiment_path']):
        os.makedirs(config['experiment_path'])
        os.makedirs(config['log_path'])
        os.makedirs(config['result_path'])
        os.makedirs(config['experiment_config_path'])

    run_config_cv(config, gpus=args.gpus, threads=args.threads)
