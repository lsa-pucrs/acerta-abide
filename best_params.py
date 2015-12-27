from copy import deepcopy
import gc
import os.path
import socket
import numpy as np
from pylearn2.train_extensions import TrainExtension


class MonitorBasedSaveBest(TrainExtension):
    def __init__(self, channel_name, save_path=None, store_best_model=False, start_epoch=0, higher_is_better=False, tag_key=None):

        self.channel_name = channel_name

        self.save_path = save_path
        self.store_best_model = store_best_model
        self.start_epoch = start_epoch
        self.higher_is_better = higher_is_better

        if higher_is_better:
            self.coeff = -1.
        else:
            self.coeff = 1.

        if tag_key is None:
            tag_key = self.__class__.__name__
        self._tag_key = tag_key

        self.best_cost = self.coeff * np.inf
        self.best_model = None

    def setup(self, model, dataset, algorithm):
        model.tag[self._tag_key]['channel_name'] = self.channel_name
        model.tag[self._tag_key]['hostname'] = socket.gethostname()
        if self.save_path is not None:
            model.tag[self._tag_key]['save_path'] = os.path.abspath(self.save_path)

        self._update_tag(model)

    def on_monitor(self, model, dataset, algorithm):
        monitor = model.monitor
        channels = monitor.channels
        channel = channels[self.channel_name]
        val_record = channel.val_record
        new_cost = val_record[-1]

        if self.coeff * new_cost < self.coeff * self.best_cost and monitor._epochs_seen >= self.start_epoch:
            self.best_cost = new_cost
            self.best_epoch = monitor._epochs_seen
            self._update_tag(model)
            if self.store_best_model:
                del self.best_model
                gc.collect()
                gc.collect()
                gc.collect()
                self.best_model = deepcopy(model)

    def _update_tag(self, model):
        model.tag[self._tag_key]['best_cost'] = self.best_cost
