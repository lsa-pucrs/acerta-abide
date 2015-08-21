import os, sys, time, argparse
from pylearn2.config import yaml_parse
from multiprocessing import Pool
import logging
import multiprocessing
from pylearn2.utils.logger import (
    CustomStreamHandler, CustomFormatter, restore_defaults
)
from threading import Thread, current_thread

def elapsed_time(tstart):
    tnow = time.time()
    total = tnow - tstart
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

class format_dict(dict):
    def __missing__(self, key):
        return '%(' + key + ')s'

def compute_config(parent, config):
    final = format_dict(parent)
    parent = format_dict(parent)
    for key in config:
        if isinstance(config[key], str):
            try:
                final[key] = config[key] % parent
            except Exception, e:
                print config[key], parent
        else:
            final[key] = config[key]
    return final

def run_config(foldconfig):
    current = multiprocessing.current_process()
    foldconfig['thread'] = current.name
    logging.info('Fold %(fold)s: %(thread)s' % foldconfig)
    yaml = open(foldconfig['run_config_path'] + '/' + step + '.yaml').read() % foldconfig
    tstart = time.time()
    stepmodel = yaml_parse.load(yaml)
    stepmodel.main_loop()
    foldconfig['elapsed'] = elapsed_time(tstart)
    logging.info('Fold %(fold)s: ended with %(elapsed)s' % foldconfig)

def run_model_cv(model, step, stepconfig, jobs=6):
    configs = []
    for fold in range(1, int(stepconfig['folds'])+1):
        foldconfig = compute_config({'fold': str(fold)}, stepconfig)
        configs.append(foldconfig)
    pool = Pool(processes=jobs)
    pool.map(run_config, configs)

def run_config_cv(config):

    t0 = time.time()

    restore_defaults()
    root_logger = logging.getLogger()
    formatter = ThreadFormatter()
    handler = logging.FileHandler('logs/' + config['namespace'] + '.log')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)

    yaml = open(config['run_config_file']).read()
    pipeline = yaml_parse.load(yaml)
    yaml_config = compute_config(config, pipeline['config']['common'])

    for step in pipeline['pipeline']:
        pipelineconfig = {}
        if step in pipeline['config']:
            pipelineconfig = pipeline['config'][step]
        yaml_config['step'] = step
        stepconfig = compute_config(yaml_config, pipelineconfig)
        run_model(yaml_config['model'], step, stepconfig)

    print "%s: %s" % ("Total", elapsed_time(t0))

class ThreadFormatter(logging.Formatter):
    def __init__(self):
        super(ThreadFormatter, self).__init__("#process_name#,%(asctime)s,%(name)s,%(levelname)s,%(message)s")
    def format(self, record):
        s = super(ThreadFormatter, self).format(record)
        current = multiprocessing.current_process()
        return s.replace('#process_name#', current.name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run cv deep-learning pipeline.')
    parser.add_argument('model', help='Model')
    parser.add_argument('config', help='Config')
    parser.add_argument('cv_folds', type=int, help='CV Folds')
    args = parser.parse_args()

    config['run_config_path'] = config['config_path'] + '/' + args.model
    config['run_config_file'] = config['config_path'] + '/' + args.model + '/' + args.config + '.yaml'
    config['namespace'] = args.model + '.' + args.config
    config['model'] = args.model
    config['config'] = args.config
    config['folds'] = str(args.cv_folds)

    if not os.path.isdir(config['run_config_path']):
        sys.exit(1)

    if not os.path.exists(config['run_config_file']):
        sys.exit(1)

    run_config_cv(config)