import sys, csv, logging
from functools import wraps
from pylearn2.monitor import Monitor
from pylearn2.train_extensions import TrainExtension

log = logging.getLogger(__name__)

class CSVMonitoring(TrainExtension):

    def __init__(self, name=None, file=None):
        self.file_handler = open(file, 'wb')
        self.name = name
        self.csv_writer = csv.writer(self.file_handler, delimiter=',')

    @wraps(TrainExtension.on_monitor)
    def on_monitor(self, model, dataset, algorithm):
        try:
            monitor = Monitor.get_monitor(model)
            if self.name is not None:
                print "%s: epoch %d" % (self.name, monitor._epochs_seen)
            else:
                print "Epoch %d" % (monitor._epochs_seen)
            for channel_name in monitor.channels.keys():
                if channel_name == 'total_seconds_last_epoch':
                    pass
                channel = monitor.channels[channel_name]
                row = [channel_name, channel.epoch_record[-1], channel.val_record[-1]]
                self.csv_writer.writerow(row)
                self.file_handler.flush()
        except Exception, e:
            pass