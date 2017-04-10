#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Autoencoders evaluation.

Usage:
  nn_evaluate.py [--folds=N] [--whole] [--male] [--threshold] [<derivative> ...]
  nn_evaluate.py (-h | --help)

Options:
  -h --help     Show this screen
  --folds=N     Number of folds [default: 10]
  --whole       Run model for the whole dataset
  --male        Run model for male subjects
  --threshold   Run model for thresholded subjects
  derivative    Derivatives to process

"""
import time
import os
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import pandas as pd
import tensorflow as tf
from docopt import docopt
from nn import nn, to_softmax, reset
from utils import format_config
from sklearn.manifold import TSNE


def reduce(config, folds, model_path, data_path, id_path):

    n_classes = 2
    fold_data = []

    for fold in range(1, folds + 1):

        config = config.copy()
        config["fold"] = fold

        fold_model_path = format_config(model_path, config)

        train_path = format_config(data_path, config, {"datatype": "train"})
        valid_path = format_config(data_path, config, {"datatype": "valid"})
        test_path = format_config(data_path, config, {"datatype": "test"})

        train_id_path = format_config(id_path, config, {"datatype": "train"})
        valid_id_path = format_config(id_path, config, {"datatype": "valid"})
        test_id_path = format_config(id_path, config, {"datatype": "test"})

        train_data = np.loadtxt(train_path, delimiter=",")
        train_X, train_y = train_data[:, 1:], train_data[:, 0]

        valid_data = np.loadtxt(valid_path, delimiter=",")
        valid_X, valid_y = valid_data[:, 1:], valid_data[:, 0]

        train_ids = np.genfromtxt(train_id_path, dtype="str")
        valid_ids = np.genfromtxt(valid_id_path, dtype="str")

        train_X = np.concatenate([train_X, valid_X])
        train_y = np.concatenate([train_y, valid_y])
        train_ids = np.concatenate([train_ids, valid_ids])

        test_data = np.loadtxt(test_path, delimiter=",")
        test_X, test_y = test_data[:, 1:], test_data[:, 0]
        test_ids = np.genfromtxt(test_id_path, dtype="str")

        model = nn(test_X.shape[1], n_classes, [
            {"size": 1000, "actv": tf.nn.tanh},
            {"size": 600, "actv": tf.nn.tanh},
        ])

        init = tf.global_variables_initializer()
        with tf.Session() as sess:

            sess.run(init)

            saver = tf.train.Saver(model["params"])
            saver.restore(sess, fold_model_path)

            train_X = sess.run(
                model["actvs"][1],
                feed_dict={
                    model["input"]: train_X,
                    model["dropouts"][0]: 1.0,
                    model["dropouts"][1]: 1.0,
                }
            )

            test_X = sess.run(
                model["actvs"][1],
                feed_dict={
                    model["input"]: test_X,
                    model["dropouts"][0]: 1.0,
                    model["dropouts"][1]: 1.0,
                }
            )

        X = np.concatenate([train_X, test_X])
        y = np.concatenate([train_y, test_y]).astype(int)
        ids = np.concatenate([train_ids, test_ids])
        datatype = np.concatenate([np.ones(train_y.shape), np.zeros(test_y.shape)]).astype(int)

        fold_data.append({
            "X": X,
            "y": y,
            "ids": ids,
            "datatype": datatype,
            "config": config
        })

    LOG_DIR = './tensorboard/'

    glob_sess = tf.InteractiveSession()
    summary_writer = tf.summary.FileWriter(LOG_DIR)
    projector_config = projector.ProjectorConfig()

    embeddings = []

    for data in fold_data:

        X = data["X"]
        y = data["y"]
        ids = data["ids"]
        datatype = data["datatype"]
        config = data["config"]

        embedding_tensor = format_config("embedding_{fold}", config)
        embedding_var = tf.Variable(X, trainable=False, name=embedding_tensor)
        embedding_var.initializer.run()
        embeddings.append(embedding_var)

        embedding = projector_config.embeddings.add()
        embedding.tensor_name = embedding_tensor
        embedding.metadata_path = os.path.join(LOG_DIR, format_config("metadata_{fold}.tsv", config))

        names = ["ASD", "TC"]
        dt = ["Test", "Train"]
        with open(embedding.metadata_path, "w") as metadata_file:
            metadata_file.write("ID\tSite\tClass\tDatatype\n")
            for i, subject in enumerate(ids):
                site = "_".join(subject.split("_")[:-1])
                metadata_file.write("%s\t%s\t%s\t%s\n" % (subject, site, names[y[i]], dt[datatype[i]]))

    saver = tf.train.Saver(embeddings)
    projector.visualize_embeddings(summary_writer, projector_config)
    saver.save(glob_sess, os.path.join(LOG_DIR, "embeddings.ckpt"))
    glob_sess.close()


if __name__ == "__main__":

    arguments = docopt(__doc__)

    experiments = []
    if arguments["--whole"]:
        experiments.append("whole")
    if arguments["--male"]:
        experiments.append("male")
    if arguments["--threshold"]:
        experiments.append("threshold")

    maxfolds = int(arguments["--folds"]) + 1

    valid_derivatives = ["cc200", "aal", "ez", "ho", "tt", "dosenbach160"]
    derivatives = [derivative for derivative in arguments["<derivative>"] if derivative in valid_derivatives]

    results = []
    for derivative in derivatives:
        for exp in experiments:

            config = {
                "derivative": derivative,
                "exp": exp
            }

            reduce(config, int(arguments["--folds"]),
                   "./data/models/{derivative}_{exp}_{fold}_mlp.ckpt",
                   "./data/corr/{derivative}_{exp}_{fold}_{datatype}.csv",
                   "./data/corr/{derivative}_{exp}_{fold}_{datatype}.ids.csv")
