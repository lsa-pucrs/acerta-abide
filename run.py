import os, sys, time, argparse
from pylearn2.config import yaml_parse

def elapsed_time(tstart):
    tnow = time.time()
    total = tnow - tstart
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

def compute_config(parent, config):
    final = {}
    final.update(parent)
    for key in config:
        if isinstance(config[key], str):
            final[key] = config[key] % parent
        else:
            final[key] = config[key]
    return final

def run_model(model, step, stepconfig):
    tstart = time.time()
    yaml = open(config['run_config_path'] + '/' + step + '.yaml').read() % stepconfig
    stepmodel = yaml_parse.load(yaml)
    stepmodel.main_loop()
    elapsed = elapsed_time(tstart)
    print "%s %s: %s" % (model, step, elapsed)

def run_config(config):
    t0 = time.time()

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run deep-learning pipeline.')
    parser.add_argument('model', help='Model')
    parser.add_argument('config', help='Config')
    args = parser.parse_args()

    here = os.path.abspath('.')

    config = {
        'result_path': here + '/result',
        'data_path': here + '/data',
        'config_path': here + '/config',
    }

    config['run_config_path'] = config['config_path'] + '/' + args.model
    config['run_config_file'] = config['config_path'] + '/' + args.model + '/' + args.config + '.yaml'
    config['namespace'] = args.model + '.' + args.config
    config['model'] = args.model
    config['config'] = args.config

    if not os.path.isdir(config['run_config_path']):
        sys.exit(1)

    if not os.path.exists(config['run_config_file']):
        sys.exit(1)

    run_config(config)