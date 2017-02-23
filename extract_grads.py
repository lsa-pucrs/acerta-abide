import numpy as np
import theano
from theano import config
import theano.tensor as T
from pylearn2.utils import serial
import pickle

from sklearn.metrics import confusion_matrix


def one_hot(Y, outputs=None):
    if outputs is None:
        outputs = np.unique(Y).shape[0]
    Y_hot = np.zeros((Y.shape[0], outputs)).astype(int)
    for i, v in enumerate(Y):
        Y_hot[i][v] = 1
    return Y_hot


def init_weights(name, n_in, n_out, weights=None, bias=None):
    if weights is None:
        irange = np.sqrt(float(6) / (float(n_in) + float(n_out)))
        weights = np.asarray(np.random.uniform(-irange, irange, (n_in, n_out)), dtype=theano.config.floatX)
    if bias is None:
        bias = np.zeros((n_out,), dtype=theano.config.floatX)
    return (
        theano.shared(value=weights, name=name + '_W', borrow=True),
        theano.shared(value=bias, name=name + '_b', borrow=True)
    )


def feed_forward(activation, weights, bias, X):
    return activation(T.dot(X, weights) + bias)


def L1(L1_reg, weights):
    return L1_reg * sum([abs(w).sum() for w in weights])


def L2(L2_reg, weights):
    return L2_reg * sum([(w ** 2).sum() for w in weights])


def GD(cost, params, learning_rate=0.001):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for param, grad in zip(params, grads):
        updates.append((param, param - learning_rate * grad))
    return updates

def loglikelihood(y, p_y_given_x):
    return (-T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y]))

x = T.matrix('x')
y = T.ivector('y')

ae1_W, ae1_b = init_weights('ae1', 17955, 1000)
ae2_W, ae2_b = init_weights('ae2', 1000, 1000)
som_W, som_b = init_weights('som', 1000, 2)

ff = feed_forward(T.tanh, ae1_W, ae1_b, x)
ff = feed_forward(T.tanh, ae2_W, ae2_b, ff)
p_y_given_x = feed_forward(T.nnet.softmax, som_W, som_b, ff)

cost = loglikelihood(y, p_y_given_x)
# cost = T.mean(T.nnet.binary_crossentropy(y, p_y_given_x))
# cost = (-T.mean(T.log(p_y_given_x[0, y])))
# cost = (-T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y]))

# weights = [ae1_W, ae2_W, som_W]
# cost = cost + L1(0.001, weights) + L2(0.0001, weights)

params = [ae1_W, ae1_b, ae2_W, ae2_b, som_W, som_b]
train_updates = GD(cost, params, learning_rate=0.1)

# train_model = theano.function(inputs=[x, y], outputs=cost, updates=train_updates, allow_input_downcast=True)
# evaluate_model = theano.function(inputs=[x, y], outputs=T.neq(y, T.argmax(p_y_given_x)), allow_input_downcast=True)
# run_model = theano.function(inputs=[x], outputs=p_y_given_x, allow_input_downcast=True)
get_grads = theano.function(inputs=[x, y], outputs=T.grad(cost, params), allow_input_downcast=True)

for fold in range(1,11):

    print 'Loading fold %d' % fold

    train_data = np.loadtxt('data/corr/corr_1D_cv_%d_train.csv' % fold, delimiter=',')
    X_train_data = train_data[:, 1:]
    y_train_data = train_data[:, 0].astype(int)

    # test_data = np.loadtxt('data/corr/corr_cv_%d_test.csv' % fold, delimiter=',')
    # X_test_data = test_data[:, 1:]
    # y_test_data = test_data[:, 0].astype(int)

    model = serial.load('experiments/first.valid/final/models/first.valid.mlp-valid_cv_%d.pkl' % fold)

    ae1w = model.layers[0].layer_content.get_param_values()
    ae2w = model.layers[1].layer_content.get_param_values()
    somw = model.layers[2].get_param_values()

    ae1_W.set_value(ae1w[2])
    ae1_b.set_value(ae1w[1])

    ae2_W.set_value(ae2w[2])
    ae2_b.set_value(ae2w[1])

    som_W.set_value(somw[1])
    som_b.set_value(somw[0])

    for klass in [0, 1]:

        print 'Grading class %d' % klass

        X_data = X_train_data[y_train_data == klass]
        y_data = y_train_data[y_train_data == klass]

        storage = {}
        grads = get_grads(X_data, y_data)
        for i, p in enumerate(params):
            storage[p.name] = grads[i]
        f = open('grads/grads_%d_%d' % (fold, klass), 'wb')
        pickle.dump(storage, f)
        f.close()
