import os

os.environ['THEANO_FLAGS'] = "device=gpu0" + "," + \
                            "floatX=float32," + \
                            "nvcc.fastmath=True," + \
                            "base_compiledir=/tmp/theano/" + str(os.getpid())

import theano
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder
from keras.activations import (tanh, linear, softmax)
from keras.utils import np_utils

class AutoEncoder(Layer):
    '''
        A customizable autoencoder model.
          - Supports deep architectures by passing appropriate encoders/decoders list
          - If output_reconstruction then dim(input) = dim(output) else dim(output) = dim(hidden)
    '''
    def __init__(self, encoders=[], decoders=[], output_reconstruction=True, tie_weights=False, weights=None):

        super(AutoEncoder,self).__init__()
        if not encoders or not decoders:
            raise Exception("Please specify the encoder/decoder layers")

        if not len(encoders) == len(decoders):
            raise Exception("There need to be an equal number of encoders and decoders")

        # connect all encoders & decoders to their previous (respectively)
        for i in xrange(len(encoders)-1, 0, -1):
            encoders[i].connect(encoders[i-1])
            decoders[i].connect(decoders[i-1])
        decoders[0].connect(encoders[-1])  # connect the first to the last

        self.input_dim = encoders[0].input_dim
        self.hidden_dim = reversed([d.input_dim for d in decoders])
        self.output_reconstruction = output_reconstruction
        self.tie_weights = tie_weights
        self.encoders = encoders
        self.decoders = decoders

        self.params = []
        self.regularizers = []
        self.constraints = []
        for m in encoders + decoders:
            self.params += m.params
            if hasattr(m, 'regularizers'):
                self.regularizers += m.regularizers
            if hasattr(m, 'constraints'):
                self.constraints += m.constraints

        if weights is not None:
            self.set_weights(weights)

    def connect(self, node):
        self.encoders[0].previous = node

    def get_weights(self):
        weights = []
        for m in self.encoders + self.decoders:
            weights += m.get_weights()
        return weights

    def set_weights(self, weights):
        models = self.encoders + self.decoders
        for i in range(len(models)):
            nb_param = len(models[i].params)
            models[i].set_weights(weights[:nb_param])
            weights = weights[nb_param:]

    def get_input(self, train=False):
        if hasattr(self.encoders[0], 'previous'):
            return self.encoders[0].previous.get_output(train=train)
        else:
            return self.encoders[0].input

    @property
    def input(self):
        return self.get_input()

    def _get_hidden(self, train):
        return self.encoders[-1].get_output(train)

    def _tranpose_weights(self, src, dest):
        if len(dest.shape) > 1 and len(src.shape) > 1:
            dest = src.T

    def get_output(self, train):
        if not train and not self.output_reconstruction:
            return self._get_hidden(train)

        if self.tie_weights:
            for e, d in zip(self.encoders, self.decoders):
                map(self._tranpose_weights, e.get_weights(), d.get_weights())

        return self.decoders[-1].get_output(train)

    def get_config(self):
        return {"name":self.__class__.__name__,
                "encoder_config":[e.get_config() for e in self.encoders],
                "decoder_config":[d.get_config() for d in self.decoders],
                "output_reconstruction":self.output_reconstruction,
                "tie_weights":self.tie_weights}


class DenoisingAutoEncoder(AutoEncoder):
    '''
        A denoising autoencoder model that inherits the base features from autoencoder
    '''
    def __init__(self, encoders=None, decoders=None, output_reconstruction=True, tie_weights=False, weights=None, corruption_level=0.3):
        super(DenoisingAutoEncoder, self).__init__(encoders, decoders, output_reconstruction, tie_weights, weights)
        self.corruption_level = corruption_level

    def _get_corrupted_input(self, input):
        """
            http://deeplearning.net/tutorial/dA.html
        """
        return srng.binomial(size=(self.input_dim, 1), n=1,
                             p=1-self.corruption_level,
                             dtype=theano.config.floatX) * input

    def get_input(self, train=False):
        uncorrupted_input = super(DenoisingAutoEncoder, self).get_input(train)
        return self._get_corrupted_input(uncorrupted_input)

    def get_config(self):
        return {"name":self.__class__.__name__,
                "encoder_config":[e.get_config() for e in self.encoders],
                "decoder_config":[d.get_config() for d in self.decoders],
                "corruption_level":self.corruption_level,
                "output_reconstruction":self.output_reconstruction,
                "tie_weights":self.tie_weights}

fold = 6

train_data = np.loadtxt('data/corr/corr_1D_cv_%d_train.csv' % fold, delimiter=',')
test_data = np.loadtxt('data/corr/corr_1D_cv_%d_test.csv' % fold, delimiter=',')
X_train = train_data[:, 1:]
Y_train = train_data[:, 0].astype(int)
X_test = test_data[:, 1:]
Y_test = test_data[:, 0].astype(int)

# X_train = np.zeros((400, 19701))
# Y_train = np.zeros((400, 1))
# X_test = np.zeros((400, 19701))
# Y_test = np.zeros((400, 1))

batch_size = 10
nb_epoch = 30
nb_hidden_layers = [19701, 1000, 600]

encoders = []
X_train_tmp = np.copy(X_train)
for i, (n_in, n_out) in enumerate(zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]), start=1):
    print('Training the layer {}: Input {} -> Output {}'.format(i, n_in, n_out))
    ae = Sequential()
    encoder = containers.Sequential([Dense(n_out, input_dim=n_in, activation=tanh)])
    decoder = containers.Sequential([Dense(n_in, input_dim=n_out, activation=linear)])
    ae.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=False))
    ae.compile(loss='mean_squared_error', optimizer='rmsprop')
    ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, nb_epoch=nb_epoch)
    encoders.append(ae.layers[0].encoder)
    X_train_tmp = ae.predict(X_train_tmp)

nb_epoch = 20
nb_classes = 2

Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

# Fine-turning
model = Sequential()
for encoder in encoders:
    model.add(encoder)
# for i, (n_in, n_out) in enumerate(zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]), start=1):
#     model.add(Dense(n_out, input_dim=n_in, activation='softmax'))
model.add(Dense(nb_classes, input_dim=nb_hidden_layers[-1], activation=softmax))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score before fine turning:', score[0])
print('Test accuracy after fine turning:', score[1])
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score after fine turning:', score[0])
print('Test accuracy after fine turning:', score[1])
