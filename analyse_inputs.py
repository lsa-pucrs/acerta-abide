import sys, time
import re
import itertools
import json
from collections import defaultdict
from sklearn.preprocessing import scale


# asd_probs = np.zeros(data_X.shape[1])
# for i in range(data_X.shape[1]):
#     feature_set = asd_mean.copy()
#     feature_set[i] = tc_mean[i]
#     asd_probs[i] = f_prob([feature_set])[0][ASD]
# print asd_probs[list(reversed(np.argsort(asd_probs)))]

# tc_probs = np.zeros(data_X.shape[1])
# for i in range(data_X.shape[1]):
#     feature_set = tc_mean.copy()
#     feature_set[i] = asd_mean[i]
#     tc_probs[i] = f_prob([feature_set])[0][TC]
# print tc_probs[list(reversed(np.argsort(tc_probs)))]

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


def features_from(conns, regions, r1, r2):
    conns = conns.tolist()
    features = []
    for i in regions[r1]:
        for j in regions[r2]:
            if i == j:
                continue
            fr = min(int(i), int(j))
            to = max(int(i), int(j))
            conn = str(fr) + ',' + str(to)
            features.append(conns.index(conn))
    return features

asd_probs = []
tc_probs = []

with open("data/masks/CC200.json", 'r') as f:
    maskdata = json.load(f)

atlas = 'HarvardOxford Cortical'
# atlas = 'HarvardOxford Subcortical'
atlas = 'MNI Structural'
regions = defaultdict(list)
for r in maskdata:
    label = max(maskdata[r]['labels'][atlas], key=lambda item: item[1])[0]
    regions[label].append(r)
    # for region, _ in maskdata[r]['labels'][atlas]:
    #     regions[label].append(r)
del regions['None']

conns = compute_connectivity()

print f_prob([asd_mean, tc_mean])
#     feature_set[i] = tc_mean[i]
#     asd_probs[i] = f_prob([feature_set])[0][ASD]

for r1, r2 in itertools.combinations(regions.keys(), 2):

    connection_features = features_from(conns, regions, r1, r2)

    feature_set = asd_mean.copy()
    for i in range(data_X.shape[1]):
        if i in connection_features:
            feature_set[i] = tc_mean[i]
    p = f_prob([feature_set])[0][ASD]
    asd_probs.append(([r1, r2], p, [np.mean(asd_mean[connection_features]), np.mean(tc_mean[connection_features])]))

    # feature_set = tc_mean.copy()
    # for i in range(data_X.shape[1]):
    #     if i not in connection_features:
    #         feature_set[i] = asd_mean[i]
    # p = f_prob([feature_set])[0][TC]
    # tc_probs.append(([r1, r2], p))

asd_probs = sorted(asd_probs, key=lambda item: item[1])
# conn_probs = []
# for best_conns in itertools.combinations([x[0] for x in asd_probs[-10:]], 3):
#     print best_conns
#     connection_features = np.array([features_from(conns, regions, conn[0], conn[1]) for conn in best_conns]).flatten().tolist()
#     feature_set = asd_mean.copy()
#     for i in range(data_X.shape[1]):
#         if i in connection_features:
#             feature_set[i] = tc_mean[i]
#     p = f_prob([feature_set])[0][ASD]
#     print p
    # conn_probs.append((best_conns, p))
# conn_probs = sorted(conn_probs, key=lambda item: item[1])


# asd_winner = sorted(asd_probs, key=lambda item: item[1])[0]
# print regions[asd_winner[0][0]]
# print regions[asd_winner[0][1]]

# tc_winner = sorted(tc_probs, key=lambda item: item[1])[0]
# print tc_winner[0][0], regions[tc_winner[0][0]]
# print tc_winner[0][1], regions[tc_winner[0][1]]

# plt.figure(1)
# plt.subplot(211)
# plt.bar(range(200), asd_probs)
# plt.subplot(212)
# plt.bar(range(200), tc_probs)
# plt.show()

# ns = []
# asd_ps = []
# tc_ps = []
# for n in range(1, 18000, 1):
#     ns.append(n)
#     feature_set = asd_mean.copy()
#     for i in features_asd[0:n]:
#         feature_set[i] = tc_mean[i]
#     prob = f_prob([feature_set])
#     asd_ps.append(prob[0][ASD])
#     feature_set = tc_mean.copy()
#     for i in features_tc[0:n]:
#         feature_set[i] = asd_mean[i]
#     prob = f_prob([feature_set])
#     tc_ps.append(prob[0][TC])
# half = np.ones(len(ns)) * 0.5
# plt.figure(1)
# plt.plot(ns, asd_ps, 'r', ns, tc_ps, 'b', ns, half, 'k')
# plt.show()
