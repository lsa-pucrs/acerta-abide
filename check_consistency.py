import os, sys, argparse
from pylearn2.utils import serial

def load(model_paths):

    for model_path in model_paths:
        try:
            print "Loading model: {}".format(model_path)
            serial.load(model_path)
        except Exception as e:
            print "Error loading model {}:".format(model_path)
            print e
            return False

    return True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Try to load a pkl file")
    parser.add_argument("model_filename", type=str, nargs="+", help="Specifies the pkl model file")
    args = parser.parse_args()

    ret = load(args.model_filename)
    if not ret:
        sys.exit(-1)

