import os
import sys
import traceback


def run_model(model, site):

    from pylearn2.config import yaml_parse
    from pylearn2.utils import serial
    from pylearn2.utils.logger import restore_defaults
    from best_params import MonitorBasedSaveBest

    restore_defaults()

    yaml = open(model).read()
    yaml = yaml.replace('{site}', site)

    try:
        step_model = yaml_parse.load(yaml)

        save_path = None
        for e in step_model.extensions:
            if not isinstance(e, MonitorBasedSaveBest):
                continue
            if e.save_path is not None:
                save_path = e.save_path
                e.save_path = None
                e.store_best_model = True
        
        if os.path.isfile(save_path):
            print save_path, "exists"
            return

        step_model.main_loop()

        for e in step_model.extensions:
            if isinstance(e, MonitorBasedSaveBest) and save_path is not None:
                print 'Saving %s' % save_path
                serial.save(save_path, e.best_model, on_overwrite='ignore')

    except Exception, e:
        traceback.print_exc(file=sys.stdout)
        raise e

sites = [
    "CALTECH", "CMU", "KKI", "LEUVEN", "MAX_MUN", "NYU", "OHSU",
    "OLIN", "PITT", "SBL", "SDSU", "STANFORD",
    "TRINITY", "UCLA", "UM", "USM", "YALE"
]

# print sys.argv

# for site in sys.argv[1:]:
if __name__ == '__main__':
    #for i in range(1,5):
    for i in sys.argv[1:]:
#        sitename = site + '-' + str(i)
        sitename = str(i)
        run_model('./configs/pre-autoencoder-1-valid.yaml', sitename)
        run_model('./configs/pre-autoencoder-2-valid.yaml', sitename)
        run_model('./configs/mlp-valid.yaml', sitename)
