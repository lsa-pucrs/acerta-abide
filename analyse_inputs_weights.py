import sys
import json
from collections import defaultdict
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


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

def regions(atlas, overlap=False):
    with open("data/masks/CC200.json", 'r') as f:
        maskdata = json.load(f)
    regions = defaultdict(list)
    for r in maskdata:
        if overlap:
            for label, _ in maskdata[r]['labels'][atlas]:
                regions[label].append(r)
        else:
            label = max(maskdata[r]['labels'][atlas], key=lambda item: item[1])[0]
            regions[label].append(r)
    del regions['None']
    return regions


atlas = 'HarvardOxford Cortical'
# atlas = 'HarvardOxford Subcortical'
atlas = 'MNI Structural'

ae1wft = model.layers[0].layer_content.get_param_values()[2].astype(np.float64)
ae1bft = model.layers[0].layer_content.get_param_values()[1].astype(np.float64)
ae2wft = model.layers[1].layer_content.get_param_values()[2].astype(np.float64)
ae2bft = model.layers[1].layer_content.get_param_values()[1].astype(np.float64)
somw = model.layers[2].get_param_values()[1].astype(np.float64)
somb = model.layers[2].get_param_values()[0].astype(np.float64)

ae1 = ae1wft # + ae1bft
ae2 = ae2wft # + ae2bft
som = somw # + somb

g = scale(np.abs(np.dot(np.dot(ae1, ae2), som).T), axis=1)

# json.dump({
#     'asd': g[ASD].tolist(),
#     'tc': g[TC].tolist(),
#     'asd_mean': (train_asd_mean * g[ASD]).tolist(),
#     'tc_mean': (train_tc_mean * g[TC]).tolist(),
# }, open('/home/anibal.heinsfeld/repos/acerta-abide/experiments/first.valid/final/analysis/features', 'wb'))

# sys.exit()

features_asd = list(np.argsort(g[ASD]))
features_tc = list(np.argsort(g[TC]))

# feature_parcels = compute_connectivity()
# most_relevant = feature_parcels[features_tc[17402:]]

def replace(original, against, features):
    feature_set = original.copy()
    # for i in range(0, feature_set.shape[0]):
    for i in features:
        feature_set[i] = against[i]
    return feature_set

ns = []
conditions = defaultdict(list)

from random import shuffle
features_random = list(range(0, train_data_X.shape[1]))
shuffle(features_random)
features_random = np.array(features_random)

asd_from = tc_from = 0

for n in range(asd_from, train_data_X.shape[1], 1):

    ns.append(n)

    train_asd_feature_set = replace(train_asd_mean, train_tc_mean, features_asd[0:n])
    train_asd_feature_set_tc = replace(train_asd_mean, train_tc_mean, features_tc[0:n])

    train_tc_feature_set = replace(train_tc_mean, train_asd_mean, features_tc[0:n])
    train_tc_feature_set_asd = replace(train_tc_mean, train_asd_mean, features_asd[0:n])

    train_random_asd_feature_set = replace(train_asd_mean, train_tc_mean, features_random[0:n])
    train_random_tc_feature_set = replace(train_tc_mean, train_asd_mean, features_random[0:n])

    test_asd_feature_set = replace(test_asd_mean, test_tc_mean, features_asd[0:n])
    test_tc_feature_set = replace(test_tc_mean, test_asd_mean, features_tc[0:n])

    test_random_asd_feature_set = replace(test_asd_mean, test_tc_mean, features_random[0:n])
    test_random_tc_feature_set = replace(test_tc_mean, test_asd_mean, features_random[0:n])

    prob_asd = f_prob([train_asd_feature_set])[0][ASD]
    prob_tc = f_prob([train_tc_feature_set])[0][TC]

    if prob_asd >= 0.7:
        asd_from = n
    if prob_tc >= 0.7:
        tc_from = n

    conditions['train_asd_feature_set'].append(prob_asd)
    conditions['train_tc_feature_set'].append(prob_tc)
    conditions['train_asd_feature_set_tc'].append(f_prob([train_asd_feature_set_tc])[0][ASD])
    conditions['train_tc_feature_set_asd'].append(f_prob([train_tc_feature_set_asd])[0][TC])
    conditions['train_random_asd_feature_set'].append(f_prob([train_random_asd_feature_set])[0][ASD])
    conditions['train_random_tc_feature_set'].append(f_prob([train_random_tc_feature_set])[0][TC])
    conditions['test_asd_feature_set'].append(f_prob([test_asd_feature_set])[0][ASD])
    conditions['test_tc_feature_set'].append(f_prob([test_tc_feature_set])[0][TC])
    conditions['test_random_asd_feature_set'].append(f_prob([test_random_asd_feature_set])[0][ASD])
    conditions['test_random_tc_feature_set'].append(f_prob([test_random_tc_feature_set])[0][TC])

    print n

print 'From:', asd_from, tc_from

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

json.dump({
    'asd': asd_from,
    'tc': tc_from,
    'conditions': dict(conditions),
    'ns': ns,
}, open('/home/anibal.heinsfeld/repos/acerta-abide/experiments/first.valid/final/analysis/conditions.reversed.test.json', 'wb'), cls=NumpyEncoder)

sys.exit()

def plot(fig, ns, conditions, setname):

    half = np.ones(len(ns)) * 0.5

    plt.figure(fig)
    for cond in conditions:
        if not cond.startswith(setname):
            continue
        if cond.find('_asd_') > -1:
            style = 'r'
        if cond.find('_tc_') > -1:
            style = 'b'
        if cond.find('_random_') > -1:
            style = style + ':'
        plt.plot(ns, conditions[cond], style)
    plt.plot(ns, half, 'k')

plot(1, ns, conditions, 'train')
# plot(2, ns, conditions, 'test')

plt.show()

# def scatter(x, y, colors):

#     nullfmt = NullFormatter()

#     left, width = 0.1, 0.65
#     bottom, height = 0.1, 0.65
#     bottom_h = left_h = left + width + 0.02

#     rect_scatter = [left, bottom, width, height]
#     rect_histx = [left, bottom_h, width, 0.2]
#     rect_histy = [left_h, bottom, 0.2, height]

#     plt.figure(3, figsize=(8, 8))

#     axScatter = plt.axes(rect_scatter)
#     axHistx = plt.axes(rect_histx)
#     axHisty = plt.axes(rect_histy)

#     axHistx.xaxis.set_major_formatter(nullfmt)
#     axHisty.yaxis.set_major_formatter(nullfmt)

#     axScatter.scatter(x, y, color=colors)

#     binwidth = 0.05
#     xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
#     lim = (int(xymax/binwidth) + 1) * binwidth

#     axScatter.set_xlim((-lim, lim))
#     axScatter.set_ylim((-lim, lim))

#     bins = np.arange(-lim, lim + binwidth, binwidth)
#     axHistx.hist(x, bins=bins)
#     axHisty.hist(y, bins=bins, orientation='horizontal')

#     axHistx.set_xlim(axScatter.get_xlim())
#     axHisty.set_ylim(axScatter.get_ylim())

# def svm(X_train, y_train, X_test, y_test):
#     clf = SVC(C=1.0, kernel='linear', probability=True)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     [[TN, FP], [FN, TP]] = confusion_matrix(y_test, y_pred).astype(float)
#     accuracy = (TP + TN) / (TP + TN + FP + FN)
#     print accuracy
#     return clf.coef_.flatten()

# colors = np.array(['r'] * train_data_X.shape[0])
# colors[train_data_y == TC] = 'b'
# scatter(train_data_X[:, features_asd[0]], train_data_X[:, features_asd[1]], colors)

# from sklearn.svm import (SVC, LinearSVC)
# from sklearn.metrics import confusion_matrix

# svm(train_data_X, train_data_y, test_data_X, test_data_y)
# coefs = svm(train_data_X[:, features_asd[asd_from:]], train_data_y, test_data_X[:, features_asd[asd_from:]], test_data_y)
# coefs = np.array(list(np.argsort(np.abs(coefs))))

# coefs = svm(train_data_X[:, features_asd[tc_from:]], train_data_y, test_data_X[:, features_asd[tc_from:]], test_data_y)
# coefs = np.array(list(np.argsort(np.abs(coefs))))

# n = 4000
# ns = []
# conditions = {
# 'train_asd_feature_set': [],
# # 'train_tc_feature_set': [],
# # 'train_random_asd_feature_set': [],
# # 'train_random_tc_feature_set': [],

# 'test_asd_feature_set': [],
# # 'test_tc_feature_set': [],
# # 'test_random_asd_feature_set': [],
# # 'test_random_tc_feature_set': [],
# }
# for i in range(1, len(coefs)):

#     ns.append(i)

#     remove = np.concatenate([features_asd[0:asd_from], features_asd[asd_from:][coefs[0:i]]])
#     train_asd_feature_set = replace(train_asd_mean, train_tc_mean, remove)
#     test_asd_feature_set = replace(test_asd_mean, test_tc_mean, remove)

#     # remove = np.concatenate([features_tc[0:tc_from], features_tc[tc_from:][coefs[0:i]]])
#     # train_tc_feature_set = replace(train_tc_mean, train_asd_mean, remove)
#     # test_tc_feature_set = replace(test_tc_mean, test_asd_mean, remove)

#     conditions['train_asd_feature_set'].append(f_prob([train_asd_feature_set])[0][ASD])
#     # conditions['train_tc_feature_set'].append(f_prob([train_tc_feature_set])[0][TC])
#     conditions['test_asd_feature_set'].append(f_prob([test_asd_feature_set])[0][ASD])
#     # conditions['test_tc_feature_set'].append(f_prob([test_tc_feature_set])[0][TC])

#     print i, remove.shape

# plot(1, ns, conditions, 'train')
# plot(2, ns, conditions, 'test')

# plt.show()