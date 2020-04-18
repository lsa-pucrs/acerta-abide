#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import tensorflow as tf


def ae(input_size, code_size,
       corruption=0.0, tight=False,
       enc=tf.nn.tanh, dec=tf.nn.tanh):
    """

    Autoencoder model: input_size -> code_size -> input_size
    Supports tight weights and corruption.

    """

    # Define data input placeholder
    x = tf.placeholder(tf.float32, [None, input_size])

    if corruption > 0.0:

        # Corrupt data based on random sampling
        _x = tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                                      minval=0,
                                                      maxval=1 - corruption,
                                                      dtype=tf.float32), tf.float32))

    else:
        _x = x

    # Initialize encoder bias
    b_enc = tf.Variable(tf.zeros([code_size]))

    # Initialize encoder weights using Glorot method
    W_enc = tf.Variable(tf.random_uniform(
                [input_size, code_size],
                -6.0 / math.sqrt(input_size + code_size),
                6.0 / math.sqrt(input_size + code_size))
            )

    # Compute activation for encoding
    encode = tf.matmul(_x, W_enc) + b_enc
    if enc is not None:
        encode = enc(encode)

    # Initialize decoder bias
    b_dec = tf.Variable(tf.zeros([input_size]))
    if tight:

        # Tightening using encoder weights
        W_dec = tf.transpose(W_enc)

    else:

        # Initialize decoder weights using Glorot method
        W_dec = tf.Variable(tf.random_uniform(
                    [code_size, input_size],
                    -6.0 / math.sqrt(code_size + input_size),
                    6.0 / math.sqrt(code_size + input_size))
                )

    # Compute activation for decoding
    decode = tf.matmul(encode, W_dec) + b_dec
    if dec is not None:
        decode = enc(decode)

    model = {

        # Input placeholder
        "input": x,

        # Encode function
        "encode": encode,

        # Decode function
        "decode": decode,

        # Cost function: mean squared error
        "cost": tf.sqrt(tf.reduce_mean(tf.square(x - decode))),

        # Model parameters
        "params": {
            "W_enc": W_enc,
            "b_enc": b_enc,
            "b_dec": b_dec,
        }

    }

    # Add weight decoder parameters
    if not tight:
        model["params"]["W_dec"] = W_dec

    return model


def nn(input_size, n_classes, layers, init=None):
    """

    Multi-layer model
    Supports tight weights and corruption.

    """

    # Define data input placeholder
    input = x = tf.placeholder(tf.float32, [None, input_size])

    # Define expected output placeholder
    y = tf.placeholder("float", [None, n_classes])

    actvs = []
    dropouts = []
    params = {}
    for i, layer in enumerate(layers):

        # Define dropout placeholder
        dropout = tf.placeholder(tf.float32)

        if init is None:

            # Initialize empty weights
            W = tf.Variable(tf.zeros([input_size, layer["size"]]))
            b = tf.Variable(tf.zeros([layer["size"]]))

        else:

            # Initialize weights with pre-training
            W = tf.Variable(init[i]["W"])
            b = tf.Variable(init[i]["b"])

        # Compute layer activation
        x = tf.matmul(x, W) + b
        if "actv" in layer and layer["actv"] is not None:
            x = layer["actv"](x)

        # Compute layer dropout
        x = tf.nn.dropout(x, dropout)

        # Store parameters
        params.update({
            "W_" + str(i+1): W,
            "b_" + str(i+1): b,
        })
        actvs.append(x)
        dropouts.append(dropout)

        input_size = layer["size"]

    # Initialize output weights
    W = tf.Variable(tf.random_uniform(
            [input_size, n_classes],
            -3.0 / math.sqrt(input_size + n_classes),
            3.0 / math.sqrt(input_size + n_classes)))
    b = tf.Variable(tf.zeros([n_classes]))

    # Compute logits output
    y_hat = tf.matmul(x, W) + b

    # Add layer parameters
    params.update({"W_out": W, "b_out": b})
    actvs.append(y_hat)

    return {

        # Input placeholder
        "input": input,

        # Expected output placeholder
        "expected": y,

        # NN output function
        "output": tf.nn.softmax(y_hat),

        # Cost function: cross-entropy
        "cost": tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=y)),

        # Droupout placeholders
        "dropouts": dropouts,

        # Layer activations
        "actvs": actvs,

        # Model parameters
        "params": params,
    }
