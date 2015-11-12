import os, re, sys, time, yaml, argparse, traceback
import multiprocessing
from utils import config_dict, compute_config, format_config, elapsed_time, nrangearg

def run_config(fold_config):

    if 'queue' in fold_config:
        q = fold_config['queue']
        gpu = q.get(True)
        os.environ['THEANO_FLAGS'] = "device=" + gpu + ",floatX=float32,nvcc.fastmath=True"
    else:
        if 'gpu' in fold_config:
            os.environ['THEANO_FLAGS'] = "device=" + fold_config['gpu'] + ",floatX=float32,nvcc.fastmath=True"
        else:
            os.environ['THEANO_FLAGS'] = "device=cpu,floatX=float32,nvcc.fastmath=True"

    os.environ['THEANO_FLAGS'] = os.environ['THEANO_FLAGS'] + ",base_compiledir=/tmp/theano/" + str(os.getpid())

    from pylearn2.config import yaml_parse
    from pylearn2.utils import serial
    from pylearn2.utils.logger import restore_defaults
    from best_params import MonitorBasedSaveBest

    restore_defaults()

    content = open(fold_config['run_config_path'] + '/' + fold_config['step'] + '.yaml').read()
    yaml = format_config(content, fold_config)
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

    if 'queue' in fold_config:
        q.put(gpu)

def run_model_cv(model, step, step_config, gpus, threads):

    # Do not pool if you don't have enough folds
    # Keep it low
    threads = min(threads, len(step_config['folds']))

    # If you are using GPUs, allocate a resource queue
    if len(gpus) > 1:
        manager = multiprocessing.Manager()
        q = manager.Queue()
        for i in xrange(0, threads/len(gpus)):
            for g in gpus:
                q.put('gpu' + str(g))

    configs = []
    for fold in step_config['folds']:
        fold_config = compute_config({'fold': str(fold)}, step_config)
        if len(gpus) > 1:
            fold_config['queue'] = q
        elif len(gpus) > 0:
            fold_config['gpu'] = 'gpu' + str(gpus[0])
        configs.append(fold_config)

    # If you are not multithreading, run it as our ancestors did
    if threads == 1:
        for config in configs:
            run_config(config)


    # Or allocate a pool for multithreading
    else:
        pool = multiprocessing.Pool(processes=threads)
        pool.map(run_config, configs)
        pool.close()
        pool.join()

def run_config_cv(config, gpus, threads):

    t0 = time.time()

    with open(config['run_config_file']) as f:
        pipeline = yaml.load(f)
    yaml_config = compute_config(config, pipeline['config']['common'])

    for step in pipeline['pipeline']:

        step_config = {}

        if step in pipeline['config']:
            step_config = pipeline['config'][step]

        step_config['step'] = step
        step_config = compute_config(yaml_config, step_config)

        run_model_cv(yaml_config['model'], step, step_config, gpus=gpus, threads=threads)

    print "%s: %s" % ("Total", elapsed_time(t0))

if __name__ == "__main__":

    # import logging

    # logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='Run cv deep-learning pipeline.')
    parser.add_argument('model', help='Model')
    parser.add_argument('config', help='Config')
    parser.add_argument('cv_folds', type=nrangearg, help='CV Folds')
    parser.add_argument('--gpus', type=nrangearg, help='Number of gpus')
    parser.add_argument('--threads', type=int, help='Number of threads')
    args = parser.parse_args()

    here = os.path.abspath('.')

    config = {
        'result_path': here + '/result',
        'data_path': here + '/data',
        'log_path': here + '/logs',
        'config_path': here + '/config',
    }

    config['run_config_path'] = config['config_path'] + '/' + args.model
    config['run_config_file'] = config['config_path'] + '/' + args.model + '/' + args.config + '.yaml'
    config['namespace'] = args.model + '.' + args.config
    config['model'] = args.model
    config['config'] = args.config
    config['folds'] = args.cv_folds

    if not os.path.isdir(config['run_config_path']):
        print 'Path %s does not exists.' % config['run_config_path']
        sys.exit(1)

    if not os.path.exists(config['run_config_file']):
        print 'Path %s does not exists.' % config['run_config_file']
        sys.exit(1)

    run_config_cv(config, gpus=args.gpus, threads=args.threads)
