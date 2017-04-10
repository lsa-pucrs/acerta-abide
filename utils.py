#!/usr/bin/env python
import os
import sys
import re
import time
import multiprocessing
import pandas as pd
import random
import numpy as np
import tensorflow as tf

from model import ae

identifier = '(([a-zA-Z]_)?([a-zA-Z0-9_]*))'
replacement_field = '{' + identifier + '}'


def load_phenotypes(pheno_path):
    pheno = pd.read_csv(pheno_path)
    pheno = pheno[pheno['FILE_ID'] != 'no_filename']

    pheno['DX_GROUP'] = pheno['DX_GROUP'].apply(lambda v: int(v)-1)
    pheno['SITE_ID'] = pheno['SITE_ID'].apply(lambda v: re.sub('_[0-9]', '', v))
    pheno['SEX'] = pheno['SEX'].apply(lambda v: {1: "M", 2: "F"}[v])
    pheno['MEAN_FD'] = pheno['func_mean_fd']
    pheno['SUB_IN_SMP'] = pheno['SUB_IN_SMP'].apply(lambda v: v == 1)

    pheno.index = pheno['FILE_ID']

    return pheno[['FILE_ID', 'DX_GROUP', 'SEX', 'SITE_ID', 'MEAN_FD', 'SUB_IN_SMP']]


def merge_dicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def format_config(s, *d):
    dd = merge_dicts(*d)
    return s.format(**dd)


def elapsed_time(tstart):
    tnow = time.time()
    total = tnow - tstart
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def run_progress(callable_func, items, message=None, jobs=1):

    results = []

    print 'Starting pool of %d jobs' % jobs

    current = 0
    total = len(items)

    if jobs == 1:
        results = []
        for item in items:
            results.append(callable_func(item))
            current = len(results)
            if message is not None:
                args = {'current': current, 'total': total}
                sys.stdout.write("\r" + message.format(**args))
                sys.stdout.flush()

    # Or allocate a pool for multithreading
    else:
        pool = multiprocessing.Pool(processes=jobs)
        for item in items:
            pool.apply_async(callable_func, args=(item,), callback=results.append)

        while current < total:
            current = len(results)
            if message is not None:
                args = {'current': current, 'total': total}
                sys.stdout.write("\r" + message.format(**args))
                sys.stdout.flush()
            time.sleep(0.5)

        pool.close()
        pool.join()

    print
    return results


def root():
    return os.path.dirname(os.path.realpath(__file__))


def to_softmax(n_classes, classe):
    sm = [0.0] * n_classes
    sm[int(classe)] = 1.0
    return sm


def load_ae_encoder(input_size, code_size, model_path):
    model = ae(input_size, code_size)
    init = tf.global_variables_initializer()
    try:
        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.Saver(model["params"], write_version=tf.train.SaverDef.V2)
            if os.path.isfile(model_path):
                print "Restoring", model_path
                saver.restore(sess, model_path)
            params = sess.run(model["params"])
            return {"W_enc": params["W_enc"], "b_enc": params["b_enc"]}
    finally:
        reset()


def sparsity_penalty(x, p, coeff):
    p_hat = tf.reduce_mean(tf.abs(x), 0)
    kl = p * tf.log(p / p_hat) + \
        (1 - p) * tf.log((1 - p) / (1 - p_hat))
    return coeff * tf.reduce_sum(kl)


def reset():
    tf.reset_default_graph()
    random.seed(19)
    np.random.seed(19)
    tf.set_random_seed(19)
