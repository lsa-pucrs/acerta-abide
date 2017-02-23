import numpy as np
import theano
from theano import config
import theano.tensor as T
from pylearn2.utils import serial
import pickle
from collections import OrderedDict

from sklearn.metrics import confusion_matrix

def evaluate(run_model, X_data, y_data):

    y_desi = y_data
    y_pred = run_model(X_data)

    [[TN, FP], [FN, TP]] = confusion_matrix(y_desi, y_pred, labels=[0, 1]).astype(float)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    specificity = TN/(FP+TN)
    precision = TP/(TP+FP)
    sensivity = recall = TP/(TP+FN)
    fscore = 2*TP/(2*TP+FP+FN)

    return accuracy


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


def AdaDelta(parameters, cost, lr, decay=0.95):
    grads = T.grad(cost, parameters)
    updates = OrderedDict()

    for param, grad in zip(parameters, grads):
        mean_square_grad = theano.shared(value=param.get_value() * 0., borrow=True)
        mean_square_dx = theano.shared(value=param.get_value() * 0., borrow=True)

        if param.name is not None:
            mean_square_grad.name = 'mean_square_grad_' + param.name
            mean_square_dx.name = 'mean_square_dx_' + param.name

        new_mean_squared_grad = (decay * mean_square_grad + (1 - decay) * T.sqr(grad))

        epsilon = lr
        rms_dx_tm1 = T.sqrt(mean_square_dx + epsilon)
        rms_grad_t = T.sqrt(new_mean_squared_grad + epsilon)
        delta_x_t = - rms_dx_tm1 / rms_grad_t * grad

        new_mean_square_dx = (decay * mean_square_dx + (1 - decay) * T.sqr(delta_x_t))

        updates[mean_square_grad] = new_mean_squared_grad
        updates[mean_square_dx] = new_mean_square_dx
        updates[param] = param + delta_x_t

    return updates


def GD(parameters, cost, lr=2e-2, momentum=None):
    grads = T.grad(cost, parameters)
    updates = OrderedDict()

    for param, grad in zip(parameters, grads):
        if momentum is not None:
            mparam = theano.shared(param.get_value() * 0.)
            updates[param] = param - lr * mparam
            updates[mparam] = mparam * momentum + (1. - momentum) * grad
        else:
            updates[param] = param - lr * grad

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

# weights = [ae1_W, ae2_W, som_W]
# cost = cost + L1(0.000, weights) + L2(0.0001, weights)

params = [ae1_W, ae1_b, ae2_W, ae2_b, som_W, som_b]
train_updates = AdaDelta(params, cost, lr=0.0025)
train_updates = GD(params, cost, momentum=0.5)

train_model = theano.function(inputs=[x, y], outputs=cost, updates=train_updates, allow_input_downcast=True)
evaluate_model = theano.function(inputs=[x], outputs=T.argmax(p_y_given_x, axis=1), allow_input_downcast=True)

fold = 1

ae1w = serial.load('experiments/first.valid/1449165276/models/first.valid.pre-autoencoder-1-valid_cv_%d.pkl' % fold).get_param_values()
ae1_W.set_value(ae1w[2])
ae1_b.set_value(ae1w[1])

ae2w = serial.load('experiments/first.valid/1449165276/models/first.valid.pre-autoencoder-2-valid_cv_%d.pkl' % fold).get_param_values()
ae2_W.set_value(ae2w[2])
ae2_b.set_value(ae2w[1])

train_data = np.loadtxt('data/corr/corr_cv_%d_train.csv' % fold, delimiter=',')
X_train_data = train_data[:, 1:]
y_train_data = train_data[:, 0].astype(int)

test_data = np.loadtxt('data/corr/corr_cv_%d_test.csv' % fold, delimiter=',')
X_test_data = test_data[:, 1:]
y_test_data = test_data[:, 0].astype(int)

# X_train_data = np.zeros((100, 17955))
# y_train_data = np.zeros((100,))
# X_test_data = np.ones((100, 17955))
# y_test_data = np.ones((100,))

epochs = 100
batch_size = 10
for i in range(epochs):
    cost_mean = []
    for batch in range(X_train_data.shape[0] / batch_size):
        f, t = batch * batch_size, (batch + 1) * batch_size
        cost = train_model(X_train_data[f:t], y_train_data[f:t])
        if np.isnan(cost):
            break
        cost_mean.append(cost)
    print np.mean(cost_mean), evaluate(evaluate_model, X_train_data, y_train_data), evaluate(evaluate_model, X_test_data, y_test_data)
    if np.isnan(cost):
        break
