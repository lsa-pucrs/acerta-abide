import tensorflow as tf
import numpy as np
import math

def ae(input_size, code_size):

    x = tf.placeholder("float", [None, input_size])
    W_enc = tf.Variable(tf.random_uniform([input_size, code_size], -1.0 / math.sqrt(input_size), 1.0 / math.sqrt(input_size)))
    b_enc = tf.Variable(tf.zeros([code_size]))
    encode = tf.nn.tanh(tf.matmul(x, W_enc) + b_enc)

    W_dec = tf.transpose(W_enc)
    b_dec = tf.Variable(tf.zeros([input_size]))
    decode = tf.nn.tanh(tf.matmul(encode, W_dec) + b_dec)

    return {
        'input': x,
        'encode': encode,
        'decode': decode,
        'cost': tf.sqrt(tf.reduce_mean(tf.square(x - decode))),
        'params': {
            'W_enc': W_enc,
            'b_enc': b_enc,
            'W_dec': W_dec,
            'b_dec': b_dec,
        }
    }


def onehot(y):
    y = y.astype(int)
    y_hot = np.zeros((y.size, y.max() + 1)).astype(float)
    y_hot[np.arange(y.size), y] = 1.0
    return y_hot

def load():
    print 'Loading',
    train_data = np.loadtxt("/home/anibal.heinsfeld/repos/acerta-abide/data/corr/corr_1D_cv_2_train.csv", delimiter=",")
    print 'train',
    valid_data = np.loadtxt("/home/anibal.heinsfeld/repos/acerta-abide/data/corr/corr_1D_cv_2_test.csv", delimiter=",")
    print 'valid',
    test_data = np.loadtxt("/home/anibal.heinsfeld/repos/acerta-abide/data/corr/corr_1D_cv_2_test.csv", delimiter=",")
    print 'test',
    print
    return train_data, valid_data, test_data

if 'train_data' not in vars() or 'train_data' not in globals():
    train_data, valid_data, test_data = load()

# train_data_X = train_data[:, 1:]
# train_data_y = onehot(train_data[:, 0])
# valid_data_X = valid_data[:, 1:]
# valid_data_y = onehot(valid_data[:, 0])
# test_data_X = test_data[:, 1:]
# test_data_y = onehot(test_data[:, 0])

def run():

    train_data_X = train_data[:, 1:]
    train_data_y = onehot(train_data[:, 0])
    valid_data_X = valid_data[:, 1:]
    valid_data_y = onehot(valid_data[:, 0])
    test_data_X = test_data[:, 1:]
    test_data_y = onehot(test_data[:, 0])

    params = []

    train_set, valid_set, test_set = train_data_X, valid_data_X, test_data_X

    input_size = 19900
    for code_size in [1000, 600]:
        autoencoder = ae(input_size, code_size)
        train = tf.train.AdagradOptimizer(0.5).minimize(autoencoder['cost'])

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        for i in range(400):
            sess.run(train, feed_dict={autoencoder['input']: train_set})
            train_cost = sess.run(autoencoder['cost'], feed_dict={autoencoder['input']: train_set})
            valid_cost = sess.run(autoencoder['cost'], feed_dict={autoencoder['input']: valid_set})
            test_cost = sess.run(autoencoder['cost'], feed_dict={autoencoder['input']: test_set})
            print i, "Cost:", train_cost, valid_cost, test_cost

        train_set = sess.run(autoencoder['encode'], feed_dict={autoencoder['input']: train_set})
        valid_set = sess.run(autoencoder['encode'], feed_dict={autoencoder['input']: valid_set})
        test_set = sess.run(autoencoder['encode'], feed_dict={autoencoder['input']: test_set})
        params.append(autoencoder['params'])
        input_size = code_size

    print params

    x = tf.placeholder(tf.float32, shape=[None, 19900])
    y = tf.placeholder(tf.float32, shape=[None, 2])

    ae1 = tf.nn.tanh(tf.matmul(x, params[0]['W_enc']) + params[0]['b_enc'])
    ae2 = tf.nn.tanh(tf.matmul(ae1, params[1]['W_enc']) + params[1]['b_enc'])

    b = tf.Variable(tf.random_normal([2]))
    W = tf.Variable(tf.random_normal([600, 2]))
    y_hat = tf.nn.softmax(tf.matmul(ae2, W) + b)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_hat, y)
    loss = tf.reduce_mean(cross_entropy)

    train = tf.train.AdagradOptimizer(0.05).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for epoch in range(500):
        _, train_loss = sess.run([train, loss], feed_dict={x: train_data_X, y: train_data_y})
        print epoch, train_loss, sess.run(accuracy, feed_dict={x: train_data_X, y: train_data_y})
