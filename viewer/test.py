import flask
from flask import request
from analyser import (extract, filter)
from data_analyser import extract_pheno
import numpy as np
import numpy.ma as ma
import json
import nibabel as nb
import pandas as pd

data = extract('first', 'valid')
query = {"experiment":["final-2"],"model":["mlp-valid"],"channel":["valid_y_misclass"], "cv":["2","3","4"]}
data = filter(data, query)

avg = True
if avg:
    grouped = data.groupby(["pipeline", "config", "experiment", "model", "channel"])
else:
    grouped = data.groupby(["pipeline", "config", "experiment", "model", "cv", "channel"])
response = {"query": query, "data": []}

for name, group in grouped:

    expdata = group.sort_values(by="epoch")
    # expdata = expdata[expdata["epoch"] > 19]
    # expdata = expdata[expdata["epoch"] < 22]
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


        df = df[df["epoch"] > 19]
        df = df[df["epoch"] < 22]

        expvalues = {
            "name": ".".join(name),
            "range": df[["epoch", "min", "max"]].to_dict(orient="records"),
            "mean": df[["epoch", "mean"]].to_dict(orient="records"),
        }

        print expvalues

    else:
        expvalues = {
            "name": ".".join(name),
            "values": expdata[["epoch", "value"]].to_dict(orient="records"),
        }

    # response["data"].append(expvalues)

# print response