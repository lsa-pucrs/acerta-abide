#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import random
import tensorflow as tf
import numpy as np
from utils import format_config


def ae(input_size, code_size, corruption=0.0):

    x = tf.placeholder(tf.float32, [None, input_size])

    if corruption > 0.0:
        _x = tf.mul(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                                 minval=0,
                                                 maxval=1 - corruption,
                                                 dtype=tf.float32), tf.float32))
    else:
        _x = x

    b_enc = tf.Variable(tf.zeros([code_size]))
    W_enc = tf.Variable(tf.random_uniform(
                [input_size, code_size],
                -6.0 / math.sqrt(input_size + code_size),
                6.0 / math.sqrt(input_size + code_size))
            )

    encode = tf.nn.tanh(tf.matmul(_x, W_enc) + b_enc)

    b_dec = tf.Variable(tf.zeros([input_size]))
    # W_dec = tf.transpose(W_enc)
    W_dec = tf.Variable(tf.random_uniform(
                [code_size, input_size],
                -6.0 / math.sqrt(code_size + input_size),
                6.0 / math.sqrt(code_size + input_size))
            )

    # decode = tf.nn.tanh(tf.matmul(encode, W_dec) + b_dec)
    decode = tf.matmul(encode, W_dec) + b_dec

    return {
        'input': x,
        'encode': encode,
        'decode': decode,
        'cost': tf.sqrt(tf.reduce_mean(tf.square(x - decode))),
        'params': {
            'W_enc': W_enc,
            'b_enc': b_enc,
            # 'W_dec': W_dec,
            'b_dec': b_dec,
        }
    }


def nn(input_size, n_classes, layers, init):

    input = x = tf.placeholder(tf.float32, [None, input_size])
    y = tf.placeholder("float", [None, n_classes])

    actvs = []
    dropouts = []
    params = {}
    for i, layer in enumerate(layers):
        # W = tf.Variable(tf.zeros([input_size, layer]))
        # b = tf.Variable(tf.zeros([layer]))

        dropout = tf.placeholder(tf.float32)

        W = tf.Variable(init[i]["W"])
        b = tf.Variable(init[i]["b"])

        x = tf.nn.tanh(tf.matmul(x, W) + b)
        x = tf.nn.dropout(x, dropout)

        input_size = layer
        params.update({
            "W_" + str(i+1): W,
            "b_" + str(i+1): b,
        })
        actvs.append(x)
        dropouts.append(dropout)

    W = tf.Variable(tf.random_uniform(
            [input_size, n_classes],
            -3.0 / math.sqrt(input_size + n_classes),
            3.0 / math.sqrt(input_size + n_classes)))
    b = tf.Variable(tf.zeros([n_classes]))
    y_hat = tf.matmul(x, W) + b

    params.update({"W_out": W, "b_out": b})
    actvs.append(y_hat)

    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y)
    )

    return {
        'input': input,
        'expected': y,
        'output': y_hat,
        'dropouts': dropouts,
        'cost': cost,
        'actvs': actvs,
        'params': params,
    }


def run_ae1(fold):

    learning_rate = 0.0001
    training_iters = 4000
    sparse_p = 0.2
    sparse_coeff = 0.5
    batch_size = 100
    n_classes = 2

    model_path = format_config("./data/models/autoencoder-1-{fold}.ckpt", {'fold': str(fold)})
    train_path = format_config("./data/corr/corr_1D_cv_{fold}_train.csv", {'fold': str(fold)})
    valid_path = format_config("./data/corr/corr_1D_cv_{fold}_valid.csv", {'fold': str(fold)})
    test_path = format_config("./data/corr/corr_1D_cv_{fold}_test.csv", {'fold': str(fold)})

    model = ae(19900, 1000, corruption=0.7)

    # Sparsity penalty
    p_hat = tf.reduce_mean(tf.abs(model['encode']), 0)
    kl = sparse_p * tf.log(sparse_p / p_hat) + \
        (1 - sparse_p) * tf.log((1 - sparse_p) / (1 - p_hat))
    penalty = sparse_coeff * tf.reduce_sum(kl)
    model['cost'] += penalty

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model['cost'])

    train_data = np.loadtxt(train_path, delimiter=",")
    train_X, train_y = train_data[:, 1:], train_data[:, 0]

    valid_data = np.loadtxt(valid_path, delimiter=",")
    valid_X, valid_y = valid_data[:, 1:], valid_data[:, 0]

    test_data = np.loadtxt(test_path, delimiter=",")
    test_X, test_y = test_data[:, 1:], test_data[:, 0]

    init = tf.initialize_all_variables()
    with tf.Session() as sess:

        sess.run(init)

        saver = tf.train.Saver(model["params"])
        # if os.path.isfile(model_path):
        #     print "Restoring", model_path
        #     saver.restore(sess, model_path)

        prev_costs = np.array([9999999999] * 3)

        for epoch in range(training_iters):

            batches = range(len(train_X) / batch_size)
            costs = np.zeros((len(batches), 3))

            for ib in batches:

                from_i = ib * batch_size
                to_i = (ib + 1) * batch_size

                batch_xs, batch_ys = train_X[from_i:to_i], train_y[from_i:to_i]
                _, cost_train = sess.run(
                    [optimizer, model['cost']],
                    feed_dict={
                        model['input']: batch_xs
                    }
                )

                cost_valid = sess.run(
                    model['cost'],
                    feed_dict={
                        model['input']: valid_X
                    }
                )

                cost_test = sess.run(
                    model['cost'],
                    feed_dict={
                        model['input']: test_X
                    }
                )

                costs[ib] = [cost_train, cost_valid, cost_test]

            costs = costs.mean(axis=0)
            cost_train, cost_valid, cost_test = costs
            print "Iter {:5d}, Cost= {:.6f} {:.6f} {:.6f}".format(epoch, cost_train, cost_valid, cost_test),

            if cost_valid < prev_costs[1]:
                print "Saving better model"
                saver.save(sess, model_path)
                prev_costs = costs
            else:
                print


def run_ae2(fold):

    train_path = format_config("./data/corr/corr_1D_cv_{fold}_train.csv", {'fold': str(fold)})
    valid_path = format_config("./data/corr/corr_1D_cv_{fold}_valid.csv", {'fold': str(fold)})
    test_path = format_config("./data/corr/corr_1D_cv_{fold}_test.csv", {'fold': str(fold)})

    train_data = np.loadtxt(train_path, delimiter=",")
    train_X, train_y = train_data[:, 1:], train_data[:, 0]

    valid_data = np.loadtxt(valid_path, delimiter=",")
    valid_X, valid_y = valid_data[:, 1:], valid_data[:, 0]

    test_data = np.loadtxt(test_path, delimiter=",")
    test_X, test_y = test_data[:, 1:], test_data[:, 0]

    model_path = format_config("./data/models/autoencoder-1-{fold}.ckpt", {'fold': str(fold)})
    model = ae(19900, 1000, corruption=0.0)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver(model["params"])
        if os.path.isfile(model_path):
            print "Restoring", model_path
            saver.restore(sess, model_path)
        train_X = sess.run(model['encode'], feed_dict={model['input']: train_X})
        valid_X = sess.run(model['encode'], feed_dict={model['input']: valid_X})
        test_X = sess.run(model['encode'], feed_dict={model['input']: test_X})

    tf.reset_default_graph()

    learning_rate = 0.0001
    training_iters = 2000
    sparse_p = 0.2
    sparse_coeff = 0.5
    batch_size = 10
    n_classes = 2

    model_path = format_config("./data/models/autoencoder-2-{fold}.ckpt", {'fold': str(fold)})
    train_path = format_config("./data/corr/corr_1D_cv_{fold}_train.csv", {'fold': str(fold)})
    valid_path = format_config("./data/corr/corr_1D_cv_{fold}_valid.csv", {'fold': str(fold)})
    test_path = format_config("./data/corr/corr_1D_cv_{fold}_test.csv", {'fold': str(fold)})

    model = ae(1000, 600, corruption=0.9)

    # Sparsity penalty
    p_hat = tf.reduce_mean(tf.abs(model['encode']), 0)
    kl = sparse_p * tf.log(sparse_p / p_hat) + \
        (1 - sparse_p) * tf.log((1 - sparse_p) / (1 - p_hat))
    penalty = sparse_coeff * tf.reduce_sum(kl)
    model['cost'] += penalty

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model['cost'])

    init = tf.initialize_all_variables()
    with tf.Session() as sess:

        sess.run(init)

        saver = tf.train.Saver(model["params"])
        # if os.path.isfile(model_path):
        #     print "Restoring", model_path
        #     saver.restore(sess, model_path)

        prev_costs = np.array([9999999999] * 3)

        for epoch in range(training_iters):

            batches = range(len(train_X) / batch_size)
            costs = np.zeros((len(batches), 3))

            for ib in batches:

                from_i = ib * batch_size
                to_i = (ib + 1) * batch_size

                batch_xs, batch_ys = train_X[from_i:to_i], train_y[from_i:to_i]
                _, cost_train = sess.run(
                    [optimizer, model['cost']],
                    feed_dict={
                        model['input']: batch_xs
                    }
                )

                cost_valid = sess.run(
                    model['cost'],
                    feed_dict={
                        model['input']: valid_X
                    }
                )

                cost_test = sess.run(
                    model['cost'],
                    feed_dict={
                        model['input']: test_X
                    }
                )

                costs[ib] = [cost_train, cost_valid, cost_test]

            costs = costs.mean(axis=0)
            cost_train, cost_valid, cost_test = costs
            print "Iter {:5d}, Cost= {:.6f} {:.6f} {:.6f}".format(epoch, cost_train, cost_valid, cost_test),

            if cost_valid < prev_costs[1]:
                print "Saving better model"
                saver.save(sess, model_path)
                prev_costs = costs
            else:
                print


def to_softmax(n_classes, classe):
    sm = [0.0] * n_classes
    sm[int(classe)] = 1.0
    return sm


def load_ae_encoder(input_size, code_size, model_path):
    model = ae(input_size, code_size)
    init = tf.initialize_all_variables()
    try:
        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.Saver(model["params"])
            if os.path.isfile(model_path):
                print "Restoring", model_path
                saver.restore(sess, model_path)
            params = sess.run(model["params"])
            return {"W_enc": params["W_enc"], "b_enc": params["b_enc"]}
    finally:
        tf.reset_default_graph()


def run_nn(fold):

    learning_rate = 0.0005
    training_iters = 100
    batch_size = 10
    n_classes = 2

    dropout_1 = 0.6
    dropout_2 = 0.8

    model_path = format_config("./data/models/autoencoder-1-{fold}.ckpt", {'fold': str(fold)})
    ae1 = load_ae_encoder(19900, 1000, model_path)
    model_path = format_config("./data/models/autoencoder-2-{fold}.ckpt", {'fold': str(fold)})
    ae2 = load_ae_encoder(1000, 600, model_path)

    model_path = format_config("./data/models/mlp-{fold}.ckpt", {'fold': str(fold)})
    train_path = format_config("./data/corr/corr_1D_cv_{fold}_train.csv", {'fold': str(fold)})
    valid_path = format_config("./data/corr/corr_1D_cv_{fold}_valid.csv", {'fold': str(fold)})
    test_path = format_config("./data/corr/corr_1D_cv_{fold}_test.csv", {'fold': str(fold)})

    model = nn(19900, n_classes, [1000, 600], [
        {"W": ae1["W_enc"], "b": ae1["b_enc"]},
        {"W": ae2["W_enc"], "b": ae2["b_enc"]},
    ])

    # momentum = tf.placeholder("float32")
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model['cost'])

    correct_prediction = tf.equal(
        tf.argmax(model['output'], 1),
        tf.argmax(model['expected'], 1)
    )
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    train_data = np.loadtxt(train_path, delimiter=",")
    train_X, train_y = train_data[:, 1:], train_data[:, 0]
    train_y = np.array([to_softmax(n_classes, y) for y in train_y])

    valid_data = np.loadtxt(valid_path, delimiter=",")
    valid_X, valid_y = valid_data[:, 1:], valid_data[:, 0]
    valid_y = np.array([to_softmax(n_classes, y) for y in valid_y])

    test_data = np.loadtxt(test_path, delimiter=",")
    test_X, test_y = test_data[:, 1:], test_data[:, 0]
    test_y = np.array([to_softmax(n_classes, y) for y in test_y])

    init = tf.initialize_all_variables()
    with tf.Session() as sess:

        sess.run(init)

        saver = tf.train.Saver(model["params"])
        # if os.path.isfile(model_path):
        #     print "Restoring", model_path
        #     saver.restore(sess, model_path)

        prev_costs = np.array([9999999999] * 3)

        for epoch in range(training_iters):

            batches = range(len(train_X) / batch_size)
            costs = np.zeros((len(batches), 3))
            accs = np.zeros((len(batches), 3))

            for ib in batches:

                from_i = ib * batch_size
                to_i = (ib + 1) * batch_size

                batch_xs, batch_ys = train_X[from_i:to_i], train_y[from_i:to_i]

                _, cost_train, acc_train = sess.run(
                    [optimizer, model['cost'], accuracy],
                    feed_dict={
                        model['input']: batch_xs,
                        model['expected']: batch_ys,
                        model['dropouts'][0]: dropout_1,
                        model['dropouts'][1]: dropout_2,
                    }
                )

                cost_valid, acc_valid = sess.run(
                    [model['cost'], accuracy],
                    feed_dict={
                        model['input']: valid_X,
                        model['expected']: valid_y,
                        model['dropouts'][0]: 1.0,
                        model['dropouts'][1]: 1.0,
                    }
                )

                cost_test, acc_test = sess.run(
                    [model['cost'], accuracy],
                    feed_dict={
                        model['input']: test_X,
                        model['expected']: test_y,
                        model['dropouts'][0]: 1.0,
                        model['dropouts'][1]: 1.0,
                    }
                )

                costs[ib] = [cost_train, cost_valid, cost_test]
                accs[ib] = [acc_train, acc_valid, acc_test]

            costs = costs.mean(axis=0)
            cost_train, cost_valid, cost_test = costs

            accs = accs.mean(axis=0)
            acc_train, acc_valid, acc_test = accs

            print "Iter {:5d}, Acc= {:.6f} {:.6f} {:.6f}".format(epoch, acc_train, acc_valid, acc_test),

            if cost_valid < prev_costs[1]:
                print "Saving better model"
                saver.save(sess, model_path)
                prev_costs = costs
            else:
                print


random.seed(19)
np.random.seed(19)
tf.set_random_seed(19)

for fold in range(10):
    run_ae1(fold+1)
    run_ae2(fold+1)
    run_nn(fold+1)
