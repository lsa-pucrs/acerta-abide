#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Autoencoders training and fine-tuning.

Usage:
  nn.py [--folds=N] [--whole] [--male] [--threshold]
  nn.py (-h | --help)

Options:
  -h --help     Show this screen
  --folds=N     Number of folds [default: 10]
  --whole       Run model for the whole dataset
  --male        Run model for male subjects
  --threshold   Run model for thresholded subjects

"""

import os
import math
import random
import numpy as np
import tensorflow as tf
from docopt import docopt
from utils import format_config


def ae(input_size, code_size, corruption=0.0):

    x = tf.placeholder(tf.float32, [None, input_size])

    if corruption > 0.0:
        _x = tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
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
        "input": x,
        "encode": encode,
        "decode": decode,
        "cost": tf.sqrt(tf.reduce_mean(tf.square(x - decode))),
        "params": {
            "W_enc": W_enc,
            "b_enc": b_enc,
            "W_dec": W_dec,
            "b_dec": b_dec,
        }
    }


def nn(input_size, n_classes, layers, init=None):

    input = x = tf.placeholder(tf.float32, [None, input_size])
    y = tf.placeholder("float", [None, n_classes])

    actvs = []
    dropouts = []
    params = {}
    for i, layer in enumerate(layers):

        dropout = tf.placeholder(tf.float32)

        if init is None:
            W = tf.Variable(tf.zeros([input_size, layer]))
            b = tf.Variable(tf.zeros([layer]))
        else:
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
        "input": input,
        "expected": y,
        "output": tf.nn.softmax(y_hat),
        "dropouts": dropouts,
        "cost": cost,
        "actvs": actvs,
        "params": params,
    }


def run_ae1(exp, fold, model_path, data_path, code_size=1000):

    learning_rate = 0.0001
    training_iters = 700
    sparse_p = 0.2
    sparse_coeff = 0.5
    batch_size = 100
    n_classes = 2

    model_path = format_config(model_path, {"fold": str(fold), "exp": exp})
    train_path = format_config(data_path, {"fold": str(fold), "exp": exp, "datatype": "train"})
    valid_path = format_config(data_path, {"fold": str(fold), "exp": exp, "datatype": "valid"})
    test_path = format_config(data_path, {"fold": str(fold), "exp": exp, "datatype": "test"})

    train_data = np.loadtxt(train_path, delimiter=",")
    train_X, train_y = train_data[:, 1:], train_data[:, 0]

    valid_data = np.loadtxt(valid_path, delimiter=",")
    valid_X, valid_y = valid_data[:, 1:], valid_data[:, 0]

    test_data = np.loadtxt(test_path, delimiter=",")
    test_X, test_y = test_data[:, 1:], test_data[:, 0]

    model = ae(train_X.shape[1], code_size, corruption=0.7)

    # Sparsity penalty
    p_hat = tf.reduce_mean(tf.abs(model["encode"]), 0)
    kl = sparse_p * tf.log(sparse_p / p_hat) + \
        (1 - sparse_p) * tf.log((1 - sparse_p) / (1 - p_hat))
    penalty = sparse_coeff * tf.reduce_sum(kl)
    model["cost"] += penalty

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model["cost"])

    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)

        saver = tf.train.Saver(model["params"], write_version=tf.train.SaverDef.V2)

        prev_costs = np.array([9999999999] * 3)

        for epoch in range(training_iters):

            batches = range(len(train_X) / batch_size)
            costs = np.zeros((len(batches), 3))

            for ib in batches:

                from_i = ib * batch_size
                to_i = (ib + 1) * batch_size

                batch_xs, batch_ys = train_X[from_i:to_i], train_y[from_i:to_i]
                _, cost_train = sess.run(
                    [optimizer, model["cost"]],
                    feed_dict={
                        model["input"]: batch_xs
                    }
                )

                cost_valid = sess.run(
                    model["cost"],
                    feed_dict={
                        model["input"]: valid_X
                    }
                )

                cost_test = sess.run(
                    model["cost"],
                    feed_dict={
                        model["input"]: test_X
                    }
                )

                costs[ib] = [cost_train, cost_valid, cost_test]

            costs = costs.mean(axis=0)
            cost_train, cost_valid, cost_test = costs
            print "Exp={:s}, Fold={:2d}, Model=ae1, Iter {:5d}, Cost= {:.6f} {:.6f} {:.6f}".format(exp, fold, epoch, cost_train, cost_valid, cost_test),

            if cost_valid < prev_costs[1]:
                print "Saving better model"
                saver.save(sess, model_path)
                prev_costs = costs
            else:
                print


def run_ae2(exp, fold, model_path, data_path, prev_model_path, code_size=600, prev_code_size=1000):

    train_path = format_config(data_path, {"fold": str(fold), "exp": exp, "datatype": "train"})
    valid_path = format_config(data_path, {"fold": str(fold), "exp": exp, "datatype": "valid"})
    test_path = format_config(data_path, {"fold": str(fold), "exp": exp, "datatype": "test"})

    train_data = np.loadtxt(train_path, delimiter=",")
    train_X, train_y = train_data[:, 1:], train_data[:, 0]

    valid_data = np.loadtxt(valid_path, delimiter=",")
    valid_X, valid_y = valid_data[:, 1:], valid_data[:, 0]

    test_data = np.loadtxt(test_path, delimiter=",")
    test_X, test_y = test_data[:, 1:], test_data[:, 0]

    prev_model_path = format_config(prev_model_path, {"fold": str(fold), "exp": exp})
    prev_model = ae(train_X.shape[1], prev_code_size, corruption=0.0)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver(prev_model["params"], write_version=tf.train.SaverDef.V2)
        if os.path.isfile(prev_model_path):
            print "Restoring", prev_model_path
            saver.restore(sess, prev_model_path)
        train_X = sess.run(prev_model["encode"], feed_dict={prev_model["input"]: train_X})
        valid_X = sess.run(prev_model["encode"], feed_dict={prev_model["input"]: valid_X})
        test_X = sess.run(prev_model["encode"], feed_dict={prev_model["input"]: test_X})
    del prev_model
    tf.reset_default_graph()

    learning_rate = 0.0001
    training_iters = 2000
    sparse_p = 0.2
    sparse_coeff = 0.5
    batch_size = 10
    n_classes = 2

    model_path = format_config(model_path, {"fold": str(fold), "exp": exp})
    model = ae(prev_code_size, code_size, corruption=0.9)

    # Sparsity penalty
    """
    p_hat = tf.reduce_mean(tf.abs(model["encode"]), 0)
    kl = sparse_p * tf.log(sparse_p / p_hat) + \
        (1 - sparse_p) * tf.log((1 - sparse_p) / (1 - p_hat))
    penalty = sparse_coeff * tf.reduce_sum(kl)
    model["cost"] += penalty
    """

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model["cost"])

    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)

        saver = tf.train.Saver(model["params"], write_version=tf.train.SaverDef.V2)

        prev_costs = np.array([9999999999] * 3)

        for epoch in range(training_iters):

            batches = range(len(train_X) / batch_size)
            costs = np.zeros((len(batches), 3))

            for ib in batches:

                from_i = ib * batch_size
                to_i = (ib + 1) * batch_size

                batch_xs, batch_ys = train_X[from_i:to_i], train_y[from_i:to_i]
                _, cost_train = sess.run(
                    [optimizer, model["cost"]],
                    feed_dict={
                        model["input"]: batch_xs
                    }
                )

                cost_valid = sess.run(
                    model["cost"],
                    feed_dict={
                        model["input"]: valid_X
                    }
                )

                cost_test = sess.run(
                    model["cost"],
                    feed_dict={
                        model["input"]: test_X
                    }
                )

                costs[ib] = [cost_train, cost_valid, cost_test]

            costs = costs.mean(axis=0)
            cost_train, cost_valid, cost_test = costs
            print "Exp={:s}, Fold={:2d}, Model=ae2, Iter {:5d}, Cost= {:.6f} {:.6f} {:.6f}".format(exp, fold, epoch, cost_train, cost_valid, cost_test),

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


def run_nn(exp, fold, model_path, data_path, prev_model_1_path, prev_model_2_path, code_size_1=1000, code_size_2=600):

    learning_rate = 0.0005
    training_iters = 100
    batch_size = 10
    n_classes = 2

    dropout_1 = 0.6
    dropout_2 = 0.8

    initial_momentum = 0.1
    final_momentum = .9
    saturate = 100

    model_path = format_config(model_path, {"fold": str(fold), "exp": exp})
    train_path = format_config(data_path, {"fold": str(fold), "exp": exp, "datatype": "train"})
    valid_path = format_config(data_path, {"fold": str(fold), "exp": exp, "datatype": "valid"})
    test_path = format_config(data_path, {"fold": str(fold), "exp": exp, "datatype": "test"})

    train_data = np.loadtxt(train_path, delimiter=",")
    train_X, train_y = train_data[:, 1:], train_data[:, 0]
    train_y = np.array([to_softmax(n_classes, y) for y in train_y])

    valid_data = np.loadtxt(valid_path, delimiter=",")
    valid_X, valid_y = valid_data[:, 1:], valid_data[:, 0]
    valid_y = np.array([to_softmax(n_classes, y) for y in valid_y])

    test_data = np.loadtxt(test_path, delimiter=",")
    test_X, test_y = test_data[:, 1:], test_data[:, 0]
    test_y = np.array([to_softmax(n_classes, y) for y in test_y])

    ae1 = load_ae_encoder(train_X.shape[1], code_size_1, format_config(prev_model_1_path, {"fold": str(fold), "exp": exp}))
    ae2 = load_ae_encoder(code_size_1, code_size_2, format_config(prev_model_2_path, {"fold": str(fold), "exp": exp}))

    model = nn(train_X.shape[1], n_classes, [code_size_1, code_size_2], [
        {"W": ae1["W_enc"], "b": ae1["b_enc"]},
        {"W": ae2["W_enc"], "b": ae2["b_enc"]},
    ])
    model["momentum"] = tf.placeholder("float32")
    optimizer = tf.train.MomentumOptimizer(learning_rate, model["momentum"]).minimize(model["cost"])

    correct_prediction = tf.equal(
        tf.argmax(model["output"], 1),
        tf.argmax(model["expected"], 1)
    )
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)

        saver = tf.train.Saver(model["params"], write_version=tf.train.SaverDef.V2)

        prev_costs = np.array([9999999999] * 3)
        prev_accs = np.array([0.0] * 3)

        for epoch in range(training_iters):

            batches = range(len(train_X) / batch_size)
            costs = np.zeros((len(batches), 3))
            accs = np.zeros((len(batches), 3))

            alpha = float(epoch) / float(saturate)
            if alpha < 0.:
                alpha = 0.
            if alpha > 1.:
                alpha = 1.

            momentum = initial_momentum * (1 - alpha) + alpha * final_momentum

            for ib in batches:

                from_i = ib * batch_size
                to_i = (ib + 1) * batch_size

                batch_xs, batch_ys = train_X[from_i:to_i], train_y[from_i:to_i]

                _, cost_train, acc_train = sess.run(
                    [optimizer, model["cost"], accuracy],
                    feed_dict={
                        model["input"]: batch_xs,
                        model["expected"]: batch_ys,
                        model["dropouts"][0]: dropout_1,
                        model["dropouts"][1]: dropout_2,
                        model["momentum"]: momentum,
                    }
                )

                cost_valid, acc_valid = sess.run(
                    [model["cost"], accuracy],
                    feed_dict={
                        model["input"]: valid_X,
                        model["expected"]: valid_y,
                        model["dropouts"][0]: 1.0,
                        model["dropouts"][1]: 1.0,
                    }
                )

                cost_test, acc_test = sess.run(
                    [model["cost"], accuracy],
                    feed_dict={
                        model["input"]: test_X,
                        model["expected"]: test_y,
                        model["dropouts"][0]: 1.0,
                        model["dropouts"][1]: 1.0,
                    }
                )

                costs[ib] = [cost_train, cost_valid, cost_test]
                accs[ib] = [acc_train, acc_valid, acc_test]

            costs = costs.mean(axis=0)
            cost_train, cost_valid, cost_test = costs

            accs = accs.mean(axis=0)
            acc_train, acc_valid, acc_test = accs

            print "Exp={:s}, Fold={:2d}, Model=nn, Iter={:5d}, Acc={:.6f} {:.6f} {:.6f}, Momentum={:.6f}".format(exp, fold, epoch, acc_train, acc_valid, acc_test, momentum),

            if acc_valid > prev_accs[1] and epoch > 20:
                print "Saving better model"
                saver.save(sess, model_path)
                prev_accs = accs
            else:
                print


def reset():
    tf.reset_default_graph()
    random.seed(19)
    np.random.seed(19)
    tf.set_random_seed(19)


if __name__ == "__main__":

    code_size_1 = 1000
    code_size_2 = 600

    arguments = docopt(__doc__)

    experiments = []
    if arguments["--whole"]:
        experiments.append("whole")
    if arguments["--male"]:
        experiments.append("male")
    if arguments["--threshold"]:
        experiments.append("threshold")

    maxfolds = int(arguments["--folds"]) + 1

    for exp in experiments:

        for fold in range(1, maxfolds):

            reset()

            run_ae1(exp, fold, model_path="./data/models/{exp}_{fold}_autoencoder-1_.ckpt",
                               data_path="./data/corr/{exp}_{fold}_{datatype}.csv",
                               code_size=code_size_1)

            reset()

            run_ae2(exp, fold, model_path="./data/models/{exp}_{fold}_autoencoder-2.ckpt",
                               data_path="./data/corr/{exp}_{fold}_{datatype}.csv",
                               prev_model_path="./data/models/{exp}_{fold}_autoencoder-1.ckpt",
                               prev_code_size=code_size_1,
                               code_size=code_size_2)

            reset()

            run_nn(exp, fold, model_path="./data/models/{exp}_{fold}_mlp.ckpt",
                              data_path="./data/corr/{exp}_{fold}_{datatype}.csv",
                              prev_model_1_path="./data/models/{exp}_{fold}_autoencoder-1.ckpt",
                              prev_model_2_path="./data/models/{exp}_{fold}_autoencoder-2.ckpt",
                              code_size_1=code_size_1,
                              code_size_2=code_size_2)
