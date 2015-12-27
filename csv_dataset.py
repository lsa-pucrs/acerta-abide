import numpy as np
import theano
from pylearn2.utils.string_utils import preprocess
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix


class CSVDataset(DenseDesignMatrix):

    def __init__(self, path='train.csv',
                 expect_labels=True, expect_headers=True,
                 delimiter=',', start=None, stop=None,
                 start_fraction=None, end_fraction=None,
                 batch_size=None, num_outputs=1, transformers=[], **kwargs):

        self.path = path
        self.expect_labels = expect_labels
        self.expect_headers = expect_headers
        self.delimiter = delimiter
        self.start = start
        self.stop = stop
        self.start_fraction = start_fraction
        self.batch_size = batch_size
        self.end_fraction = end_fraction
        self.num_outputs = num_outputs

        if start_fraction is not None:
            if end_fraction is not None:
                raise ValueError("Use start_fraction or end_fraction, "
                                 " not both.")

            if start_fraction <= 0:
                raise ValueError("start_fraction should be > 0")

            if start_fraction >= 1:
                raise ValueError("start_fraction should be < 1")

        if end_fraction is not None:
            if end_fraction <= 0:
                raise ValueError("end_fraction should be > 0")

            if end_fraction >= 1:
                raise ValueError("end_fraction should be < 1")

        if start is not None:
            if start_fraction is not None or end_fraction is not None:
                raise ValueError("Use start, start_fraction, or end_fraction,"
                                 " just not together.")

        if stop is not None:
            if start_fraction is not None or end_fraction is not None:
                raise ValueError("Use stop, start_fraction, or end_fraction,"
                                 " just not together.")

        self.path = preprocess(self.path)
        X, y = self._load_data()

        for trans in transformers:
            X = trans.perform(X)

        super(CSVDataset, self).__init__(X=X, y=y, y_labels=np.max(y) + 1, **kwargs)

    def _load_data(self):

        assert self.path.endswith('.csv')
        if self.expect_headers:
            data = np.loadtxt(self.path, delimiter=self.delimiter, skiprows=1)
        else:
            data = np.loadtxt(self.path, delimiter=self.delimiter)

        def take_subset(X, y):
            if self.start_fraction is not None:
                n = X.shape[0]
                subset_end = int(self.start_fraction * n)
                if self.batch_size is not None:
                    subset_end = subset_end - (subset_end % self.batch_size)
                X = X[0:subset_end, :]
                y = y[0:subset_end]
            elif self.end_fraction is not None:
                n = X.shape[0]
                subset_start = int((1 - self.end_fraction) * n)
                if self.batch_size is not None:
                    subset_start = subset_start + ((n - subset_start) % self.batch_size)
                X = X[subset_start:, ]
                y = y[subset_start:]
            elif self.start is not None:
                X = X[self.start:self.stop, ]
                if y is not None:
                    y = y[self.start:self.stop]

            return X, y

        if self.expect_labels:
            y = data[:, 0:self.num_outputs]
            X = data[:, self.num_outputs:]
            y = y.reshape((y.shape[0], self.num_outputs))
        else:
            X = data
            y = None

        X, y = take_subset(X, y)

        return X.astype(theano.config.floatX), y.astype(int)
