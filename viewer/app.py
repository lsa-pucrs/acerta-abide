import flask
from flask import request
from analyser import (extract, filter)
from data_analyser import extract_pheno
import numpy as np
import numpy.ma as ma
import json
import nibabel as nb
import pandas as pd

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__)))+'/..')
from utils import *

app = flask.Flask(__name__)
app.corr_data_X = None
app.corr_data_y = None

ASD = 0
TC = 1

def compute_connectivity():
    ROIs = range(1, 201)
    with np.errstate(invalid='ignore'):
        corr = np.zeros((len(ROIs), len(ROIs)), dtype=object)
        for i, f in enumerate(ROIs):
            for j, g in enumerate(ROIs):
                if f < g:
                    corr[i, j] = '%d,%d' % (f, g)
                else:
                    corr[i, j] = '%d,%d' % (g, f)
        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
        m = ma.masked_where(mask == 1, mask)
        return ma.masked_where(m, corr).compressed()


@app.route("/experiments")
def index():
    return flask.render_template("index.html")


@app.route("/experiments/<pipeline>.<config>/summary/<query>")
def summary(pipeline, config, query):

    data = extract(pipeline, config)
    query = json.loads(query)
    data = filter(data, query)

    fields = ["model", "experiment", "cv", "channel"]
    fields = [f for f in fields if f not in query.keys()]

    response = {}
    for f in fields:
        response[f] = np.unique(data[f]).tolist()
    return flask.jsonify(**response)


@app.route("/experiments/<pipeline>.<config>/<query>")
def experiments(pipeline, config, query):

    data = extract(pipeline, config)
    query = json.loads(query)
    data = filter(data, query)

    avg = request.args.get('average') is not None

    if avg:
        grouped = data.groupby(["pipeline", "config", "experiment", "model", "channel"])
    else:
        grouped = data.groupby(["pipeline", "config", "experiment", "model", "cv", "channel"])
    response = {"query": query, "data": []}

    for name, group in grouped:

        expdata = group.sort_values(by="epoch")
        if avg:

            name = list(name)
            channel = name[4]
            name[4] = "mean"
            name.append(channel)
            name = tuple(name)

            df = pd.DataFrame({
                'epoch': expdata.groupby(['epoch']).groups.keys(),
                'min': expdata.groupby(['epoch']).min()["value"].tolist(),
                'max': expdata.groupby(['epoch']).max()["value"].tolist(),
                'mean': expdata.groupby(['epoch']).mean()["value"].tolist(),
            })

            expvalues = {
                "name": ".".join(name),
                "range": df[["epoch", "min", "max"]].to_dict(orient="records"),
                "mean": df[["epoch", "mean"]].to_dict(orient="records"),
            }
        else:
            expvalues = {
                "name": ".".join(name),
                "values": expdata[["epoch", "value"]].to_dict(orient="records"),
            }

        response["data"].append(expvalues)

    return flask.jsonify(**response)


@app.route("/data")
def data():
    return flask.render_template("data.html")


@app.route("/data/<query>")
def dataquery(query):
    query = json.loads(query)
    counts, institutes = extract_pheno(int(query['fold']))
    series = [{
        'name': i,
        'data': [counts[ty][i] for ty in ['train', 'valid', 'test']]
    } for i in institutes]
    return flask.jsonify(**{
        "data": series,
    })


@app.route("/analysis")
def analysis():
    return flask.render_template("analysis.html")


@app.route("/analysis/mask")
def analysis_mask():
    inputs = compute_connectivity()
    return flask.jsonify({
        "voxels": nb.load("../data/masks/CC200.nii.gz").get_data().tolist(),
        "connections": inputs.tolist(),
    })


@app.route("/analysis/svm/<fold>")
def analysis_svm(fold):
    folder = '../experiments/svm/analysis'
    features = np.loadtxt(folder + '/coeffs_' + fold, delimiter=',')
    return flask.jsonify({
        "svm": list(reversed(np.argsort(np.abs(features))))
    })


@app.route("/analysis/weights/<pipeline>.<config>.<experiment>.<model>.<fold>")
def analysis_weights(pipeline, config, experiment, model, fold):
    f = format_config('../experiments/{pipeline}.{config}/{experiment}/analysis/{model}_weights_{fold}', {
        'pipeline': pipeline,
        'config': config,
        'experiment': experiment,
        'model': model,
        'fold': fold,
    })
    data = np.loadtxt(f, delimiter=",").tolist()
    asd = np.abs(data[ASD])
    asd = asd / np.sum(asd) * 100
    tc = np.abs(data[TC])
    tc = tc / np.sum(tc) * 100
    return flask.jsonify({
        "asd": [[x, asd[x]] for x in reversed(np.argsort(asd))],
        "tc": [[x, tc[x]] for x in reversed(np.argsort(tc))]
    })


def percentiles(data):
    return [np.min(data)] + [np.percentile(data, v, interpolation="nearest") for v in [25, 50, 75]] + [np.max(data)]


@app.route("/analysis/distribution/<connection>")
def analysis_distribution(connection):
    if app.corr_data_X is None:

        print 'Loading data...'
        data = np.loadtxt("../data/corr/corr_1D.csv", delimiter=",")
        print 'Loaded!'

        app.corr_data_X = data[:, 1:]
        app.corr_data_y = data[:, 0]

    feature = compute_connectivity().tolist().index(connection)

    asd = app.corr_data_X[app.corr_data_y == ASD][:, feature]
    tc = app.corr_data_X[app.corr_data_y == TC][:, feature]

    return flask.jsonify({
        "asd": percentiles(asd),
        "tc": percentiles(tc),
    })


@app.route("/analysis/atlas")
def analysis_atlas():

    with open("../data/masks/CC200.json", 'r') as f:
        data = json.load(f)

    return flask.jsonify(
        {parcel: {atlas: [x[0] for x in data[parcel]["labels"][atlas] if x[0] != "None"] for atlas in data[parcel]["labels"]} for parcel in data}
    )


if __name__ == "__main__":
    app.run(debug=True)
