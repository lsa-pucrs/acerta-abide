import flask
from flask import request
from analyser import (extract, filter)
from data_analyser import extract_pheno
import numpy as np
import numpy.ma as ma
import json
import nibabel as nb
import pandas as pd
from collections import defaultdict

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

    query = json.loads(query)

    experiment = []
    model = []
    if "experiment" in query:
        experiment = query["experiment"]
    if "model" in query:
        model = query["model"]

    data = extract(pipeline, config, experiment, model)
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


@app.route("/analysis/connections")
def analysis_connections():
    return flask.render_template("analysis_connections.html")


def atlasregions(atlas, overlap=False):
    maskdata = json.load(open("../data/masks/CC200.json", 'r'))
    regions = defaultdict(list)
    parcels = defaultdict(list)
    for r in maskdata:
        if overlap:
            for label, _ in maskdata[r]['labels'][atlas]:
                if label == 'None': # or label.lower().find('cerebellum') >= 0:
                    continue
                regions[label].append(r)
                parcels[r].append(label)
        else:
            label = max(maskdata[r]['labels'][atlas], key=lambda item: item[1])[0]
            if label == 'None': # or label.lower().find('cerebellum') >= 0:
                continue
            regions[label].append(r)
            parcels[r].append(label)
    return regions, parcels


@app.route("/analysis/connections/data/<query>")
def analysis_connections_data(query):
    query = json.loads(query)

    # atlas = 'HarvardOxford Cortical'
    # atlas = 'HarvardOxford Subcortical'
    # atlas = 'MNI Structural'

    atlas = query["atlas"]

    refregions, refparcels = atlasregions(atlas)

    features = json.load(open("../experiments/first.valid/final/analysis/features", 'r'))
    inputs = compute_connectivity()

    feats = int(query["features"])
    from_feat = inputs.size - feats

    subjclass = query["class"]

    features_sort = np.argsort(features[subjclass])
    most_relevant = features_sort[from_feat:]
    most_relevant_name = inputs[most_relevant]
    most_relevant_conns = [x.split(',') for x in most_relevant_name]
    most_relevant_value = np.array(features[subjclass])[most_relevant]

    meantype = query["mean"]
    if meantype == "diff":
        most_relevant_mean = np.array(features["asd_mean"])[most_relevant] - np.array(features["tc_mean"])[most_relevant]
    elif meantype == "tc":
        most_relevant_mean = np.array(features["tc_mean"])[most_relevant]
    elif meantype == "asd":
        most_relevant_mean = np.array(features["asd_mean"])[most_relevant]

    parcels = np.unique(','.join(most_relevant_name).split(',')).tolist()
    parcels = [parcel for parcel in parcels if parcel in refparcels and len(refparcels[parcel]) > 0]

    regions = []

    conn_matrix = {parcel: {parcel2: 0.0 for parcel2 in parcels} for parcel in parcels}
    corr_matrix = {parcel: {parcel2: 0.0 for parcel2 in parcels} for parcel in parcels}
    for corr, value, mean in zip(most_relevant_name, most_relevant_value, most_relevant_mean):
        corr_parcels = corr.split(',')
        if corr_parcels[0] in parcels and corr_parcels[1] in parcels:
            conn_matrix[corr_parcels[0]][corr_parcels[1]] = value
            conn_matrix[corr_parcels[1]][corr_parcels[0]] = value
            corr_matrix[corr_parcels[0]][corr_parcels[1]] = mean
            corr_matrix[corr_parcels[1]][corr_parcels[0]] = mean

    valid_parcels = defaultdict(list)
    for parcel in parcels:
        region = refparcels[parcel][0]
        for corr in most_relevant_conns:
            otherparcel = list(set(corr) - set([parcel]))[0]
            if parcel in corr and otherparcel in refparcels and len(refparcels[otherparcel]) > 0 and region != refparcels[otherparcel][0]:
                valid_parcels[parcel].append(otherparcel)

    for parcel in valid_parcels:
        new_valid = []
        for otherparcel in valid_parcels[parcel]:
            if otherparcel in valid_parcels:
                new_valid.append(otherparcel)
        valid_parcels[parcel] = new_valid

    parcel_data = []
    for parcel in valid_parcels:

        region = refparcels[parcel][0]

        parcel_obj = {
            "id": parcel,
            "region": region,
            "correlations": []
        }

        for otherparcel in valid_parcels[parcel]:
            parcel_obj["correlations"].append({
                "destination": otherparcel,
                "relevance": conn_matrix[parcel][otherparcel],
                "mean": corr_matrix[parcel][otherparcel],
            })

        if len(parcel_obj["correlations"]) > 0:
            regions.append(region)
            parcel_data.append(parcel_obj)

    regions = sorted([{"id": region, "region": region} for region in np.unique(regions).tolist()], key=lambda k: k["id"])
    parcel_data = sorted(parcel_data, key=lambda k: k["region"])

    return flask.jsonify({
        "regions": regions,
        "parcels": parcel_data,
        "values": [np.min(most_relevant_value), np.max(most_relevant_value)],
        "means": [np.min(most_relevant_mean), np.max(most_relevant_mean)],
    })


# @app.route("/analysis/connections/data/<query>")
# def analysis_connections_data(query):
#     query = json.loads(query)

#     # atlas = 'HarvardOxford Cortical'
#     # atlas = 'HarvardOxford Subcortical'
#     # atlas = 'MNI Structural'

#     atlas = query["atlas"]

#     refregions, refparcels = atlasregions(atlas)

#     features = json.load(open("../experiments/first.valid/final/analysis/features", 'r'))
#     inputs = compute_connectivity()

#     feats = int(query["features"])
#     from_feat = inputs.size - feats

#     subjclass = query["class"]

#     features_sort = np.argsort(features[subjclass])
#     most_relevant = features_sort[from_feat:]
#     most_relevant_name = inputs[most_relevant]
#     most_relevant_conns = [x.split(',') for x in most_relevant_name]
#     most_relevant_value = np.array(features[subjclass])[most_relevant]

#     meantype = query["mean"]
#     if meantype == "diff":
#         most_relevant_mean = np.array(features["asd_mean"])[most_relevant] - np.array(features["tc_mean"])[most_relevant]
#     elif meantype == "tc":
#         most_relevant_mean = np.array(features["tc_mean"])[most_relevant]
#     elif meantype == "asd":
#         most_relevant_mean = np.array(features["asd_mean"])[most_relevant]

#     parcels = np.unique(','.join(most_relevant_name).split(',')).tolist()
#     parcels = [parcel for parcel in parcels if parcel in refparcels and len(refparcels[parcel]) > 0]

#     regions = []

#     conn_matrix = {parcel: {parcel2: 0.0 for parcel2 in parcels} for parcel in parcels}
#     corr_matrix = {parcel: {parcel2: 0.0 for parcel2 in parcels} for parcel in parcels}
#     for corr, value, mean in zip(most_relevant_name, most_relevant_value, most_relevant_mean):
#         corr_parcels = corr.split(',')
#         if corr_parcels[0] in parcels and corr_parcels[1] in parcels:
#             conn_matrix[corr_parcels[0]][corr_parcels[1]] = value
#             conn_matrix[corr_parcels[1]][corr_parcels[0]] = value
#             corr_matrix[corr_parcels[0]][corr_parcels[1]] = mean
#             corr_matrix[corr_parcels[1]][corr_parcels[0]] = mean

#     valid_parcels = defaultdict(list)
#     for parcel in parcels:
#         region = refparcels[parcel][0]
#         for corr in most_relevant_conns:
#             otherparcel = list(set(corr) - set([parcel]))[0]
#             if parcel in corr and otherparcel in refparcels and len(refparcels[otherparcel]) > 0 and region != refparcels[otherparcel][0]:
#                 valid_parcels[parcel].append(otherparcel)

#     for parcel in valid_parcels:
#         new_valid = []
#         for otherparcel in valid_parcels[parcel]:
#             if otherparcel in valid_parcels:
#                 new_valid.append(otherparcel)
#         valid_parcels[parcel] = new_valid

#     parcel_data = {}
#     for parcel in valid_parcels:

#         region = refparcels[parcel][0]
#         name = region + "." + str(parcel)

#         if parcel == '117':
#             print 'p', parcel, region


#         parcel_data[name] = {
#             "name": name,
#             "size": 0.0,
#             "imports": []
#         }

#         for otherparcel in valid_parcels[parcel]:

#             otherregion = refparcels[otherparcel][0]
#             othername = otherregion + "." + str(otherparcel)

#             if otherparcel == '117':
#                 print 'o', otherparcel, otherregion

#             parcel_data[name]["imports"].append(othername)
#             parcel_data[name]["size"] = parcel_data[name]["size"] + int(conn_matrix[parcel][otherparcel]*100)

#             if othername not in parcel_data:
#                 parcel_data[othername] = {
#                     "name": othername,
#                     "size": 0.0,
#                     "imports": []
#                 }
#             parcel_data[othername]["imports"].append(region + "." + str(parcel))
#             parcel_data[othername]["size"] = parcel_data[othername]["size"] + int(conn_matrix[parcel][otherparcel]*100)

#         # if len(parcel_obj["imports"]) > 0:
#         # parcel_data[name] = parcel_obj

#     parcel_data = sorted(parcel_data.values(), key=lambda k: k["name"])

#     return flask.jsonify({
#         "parcels": parcel_data,
#     })


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
        data = np.loadtxt("../data/corr/corr_1D_cv_1_train.csv", delimiter=",")
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
