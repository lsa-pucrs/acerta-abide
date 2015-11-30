#!/usr/bin/env python
import sys
import re
import time
import multiprocessing
import pandas as pd

identifier = '(([a-zA-Z]_)?([a-zA-Z0-9_]*))'
replacement_field = '{' + identifier + '}'

def load_phenotypes(pheno_path):
    pheno = pd.read_csv(pheno_path)
    pheno = pheno[pheno['FILE_ID'] != 'no_filename']
    pheno['DX_GROUP'] = pheno['DX_GROUP'].apply(lambda v: str(int(v)-1))
    pheno['SITE_ID'] = pheno['SITE_ID'].apply(lambda v: re.sub('_[0-9]', '', v))
    pheno.index = pheno['FILE_ID']
    return pheno[['FILE_ID', 'DX_GROUP', 'SEX', 'SITE_ID']]

class config_dict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


def _count_replacements(d):
    count = 0
    for key in d:
        if isinstance(d[key], str):
            count = count + len(re.findall(replacement_field, d[key]))
    return count


def format_config(s, d):
    for k in d:
        s = s.replace('{' + k + '}', str(d[k]))
    return s


def nrangearg(string):
    m = re.match(r'(\d+)(?:-(\d+))?$', string)
    if not m:
        raise Exception("'" + string + "' is not a range of number.")
    start = m.group(1)
    end = m.group(2) or start
    return list(range(int(start, 10), int(end, 10) + 1))


def gpurangearg(string):
    m = re.match(r'(gpu\d+)(\s*,\s*(gpu\d+))*$', string)
    if m:
        return [x.strip() for x in m.group(0).split(',')]
    else:
        return []


def compute_config(config1, config2=None, replace=True):
    final = config_dict(config1)
    if config2 is not None:
        final.update(config2)
        final = compute_config(final, replace=replace)
    if replace:
        for key in final:
            if isinstance(final[key], str):
                try:
                    final[key] = format_config(final[key], final)
                except Exception:
                    pass
    return final


def elapsed_time(tstart):
    tnow = time.time()
    total = tnow - tstart
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def run_parallel(args, call, gpus=None, threads=1, concurr_key='concurr'):

    # Do not pool if you don't have enough concurr
    # Keep it low
    threads = max(1, min(threads, args[concurr_key]))

    # If you are using GPUs, allocate a resource queue
    if len(gpus) > 0:
        manager = multiprocessing.Manager()
        q = manager.Queue()
        for i in xrange(0, threads/len(gpus)):
            for g in gpus:
                q.put(g)

    configs = []
    for concurr in args[concurr_key]:

        args_config = compute_config(args, {concurr_key: str(concurr)})

        if q.qsize() > 0:
            args_config[concurr_key + '_queue'] = q
        configs.append(args_config)

    # If you are not multithreading, run it as our ancestors did
    if threads == 1:
        for config in configs:
            call(config)

    # Or allocate a pool for multithreading
    else:
        pool = multiprocessing.Pool(processes=threads)
        pool.map(call, configs)
        pool.close()
        pool.join()


def run_progress(callable_func, items, message=None, jobs=1):

    results = []

    pool = multiprocessing.Pool(processes=jobs)
    for item in items:
        pool.apply_async(callable_func, args=(item,), callback=results.append)

    pool.close()
    pool.join()

    current = 0
    total = len(items)

    while current < total:
        current = len(results)
        if message is not None:
            args = {'current': current, 'total': total}
            sys.stdout.write("\r" + message.format(**args))
            sys.stdout.flush()
        time.sleep(0.5)

    return results


if __name__ == "__main__":

    # print gpurangearg('gpu0,gpu1,gpu2')
    # print gpurangearg('1-7')
    # print load_phenotypes('./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv')


    config = { 'run': '1448904237', 'data_transform': '/home/anibal.heinsfeld/repos/acerta-abide/experiments/grbm.original/{run}/models/grbm.original.grbm-3_cv_1.pkl'}

    print compute_config(config)['data_transform']