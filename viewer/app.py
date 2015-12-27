import flask
from analyser import (extract, filter)
from data_analyser import extract_pheno
import numpy as np
import numpy.ma as ma
import json
import nibabel as nb

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

    grouped = data.groupby(["pipeline", "config", "experiment", "model", "cv", "channel"])
    response = {"query": query, "data": []}

    for name, group in grouped:

        expdata = group.sort_values(by="epoch")
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
    data = extract_pheno()
    institutes = data["institute"].unique()
    folds = data["fold"].unique()
    t = data["type"].unique()

    return flask.jsonify(**{
        "institutes": institutes.tolist(),
        "folds": folds.tolist(),
        "types": t.tolist(),
        "data": data.to_dict(orient="records"),
    })


@app.route("/analysis")
def analysis():
    return flask.render_template("analysis.html")


@app.route("/analysis/mask")
def analysis_mask():
    inputs = compute_connectivity()
    return flask.jsonify({
        "voxels": nb.load("../data/masks/CC200.nii").get_data().tolist(),
        "connections": inputs.tolist(),
    })


@app.route("/analysis/weights/<fold>")
def analysis_weights(fold):
    data = np.loadtxt("../experiments/first.valid/final/analysis/weights_%s" % fold, delimiter=",").tolist()
    return flask.jsonify({
        "asd": [[x, data[ASD][x]] for x in reversed(np.argsort(np.abs(data[ASD])))],
        "tc": [[x, data[TC][x]] for x in reversed(np.argsort(np.abs(data[TC])))]
    })


@app.route("/analysis/distribution/<feature>")
def analysis_distribution(feature):
    if app.corr_data_X is None:

        print 'Loading data...'
        data = np.loadtxt("../data/corr/corr_1D.csv", delimiter=",")
        print 'Loaded!'

        app.corr_data_X = data[:, 1:]
        app.corr_data_y = data[:, 0]

    return flask.jsonify({
        "asd": sorted(app.corr_data_X[app.corr_data_y == ASD][:, feature].tolist()),
        "tc": sorted(app.corr_data_X[app.corr_data_y == TC][:, feature].tolist()),
    })


if __name__ == "__main__":
    app.run(debug=True)
