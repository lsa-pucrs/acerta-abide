import sys
import numpy as np
import numpy.ma as ma

a = np.ones((4,3))
b = np.ones((3,2))

print np.dot(a,b)

sys.exit()

# from sklearn.preprocessing import MinMaxScaler

from pylearn2.utils import serial
model_pkl = serial.load('/home/anibal.heinsfeld/repos/acerta-abide/experiments/first.valid/final/models/first.valid.mlp-valid_cv_1.pkl')

ae1wft = model_pkl.layers[0].layer_content.get_param_values()[2].astype(np.float64)
ae1bft = model_pkl.layers[0].layer_content.get_param_values()[1].astype(np.float64)

ae2wft = model_pkl.layers[1].layer_content.get_param_values()[2].astype(np.float64)
ae2bft = model_pkl.layers[1].layer_content.get_param_values()[1].astype(np.float64)

somw = model_pkl.layers[2].get_param_values()[1].astype(np.float64)
somb = model_pkl.layers[2].get_param_values()[0].astype(np.float64)

del model_pkl

ae1 = ae1wft + ae1bft
ae2 = ae2wft + ae2bft
som = somw + somb

print ae1.shape, ae2.shape, som.shape

# v1 = serial.load('/home/anibal.heinsfeld/repos/acerta-abide/experiments/first.valid/1449165276/models/first.valid.pre-autoencoder-1-valid_cv_6.pkl').get_param_values()[2]
# # v2 = serial.load('/home/anibal.heinsfeld/repos/acerta-abide/experiments/first.valid/1449165276/models/first.valid.pre-autoencoder-2-valid_cv_6.pkl').get_param_values()[2]

# ftm = serial.load('/home/anibal.heinsfeld/repos/acerta-abide/experiments/first.valid/1449165276/models/first.valid.mlp-valid_cv_6.pkl')
# ftv1 = ftm.layers[0].layer_content.get_param_values()[2]
# ftv2 = ftm.layers[1].layer_content.get_param_values()[2]
# ftsm = ftm.layers[2].get_param_values()[1]
# del ftm

# print np.mean(ftv1), np.std(ftv1), np.min(ftv1), np.max(ftv1)
# feats = MinMaxScaler(feature_range=(-1, 1)).fit_transform(ftv1-v1)
# print np.mean(feats), np.std(feats), np.min(feats), np.max(feats)

# print np.mean(ftv2), np.std(ftv2), np.min(ftv2), np.max(ftv2)
# feats = MinMaxScaler(feature_range=(-1, 1)).fit_transform(ftv2)
# print np.mean(feats), np.std(feats), np.min(feats), np.max(feats)

# print np.mean(ftsm), np.std(ftsm), np.min(ftsm), np.max(ftsm)
# feats = MinMaxScaler(feature_range=(-1, 1)).fit_transform(ftsm)
# print np.mean(feats), np.std(feats), np.min(feats), np.max(feats)

# feats = MinMaxScaler(feature_range=(-255, 255)).fit_transform(ftv1 - v1).astype(int)
# print np.mean(feats), np.std(feats)
# feats = MinMaxScaler(feature_range=(-255, 255)).fit_transform(ftv2 - v2).astype(int)
# print np.mean(feats), np.std(feats)



# from PIL import Image
# img = Image.new('RGB', feats.shape, "black")
# pixels = img.load()
# for i in range(img.size[0]):
#     for j in range(img.size[1]):
#         if feats[i][j] < 0:
#             pixels[i,j] = (0, 0, abs(feats[i][j]))
#         else:
#             pixels[i,j] = (abs(feats[i][j]), 0, 0)
# img.resize((1795,1000)).show()

# code = {}
# for i in range(1000):
#     code[i] = {
#         'max': np.max(values[:, i])*100,
#         'argmax': np.argmax(values[:, i]),
#         'mean': np.mean(values[:, i])*100,
#     }
#     print code[i]


# feats = feats[:,0]

# size = 190
# old = np.zeros((size, size))
# per_row = 1
# pos = 0
# for i in range(1,size):
#     old[i][0:per_row] = feats[pos:pos+per_row]
#     pos = pos + per_row
#     per_row = per_row + 1


# from matplotlib import pyplot as plt
# plt.figure(figsize=(10, 10))
# plt.imshow(old, interpolation="nearest", cmap="RdBu_r")
# plt.colorbar()
# plt.show()



# def compute_connectivity():
#     ROIs = range(1, 201)
#     with np.errstate(invalid='ignore'):
#         corr = np.zeros((len(ROIs), len(ROIs)), dtype=object)
#         for i, f in enumerate(ROIs):
#             for j, g in enumerate(ROIs):
#                 if f < g:
#                     corr[i, j] = '%d,%d' % (f, g)
#                 else:
#                     corr[i, j] = '%d,%d' % (g, f)
#         mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
#         m = ma.masked_where(mask == 1, mask)
#         return ma.masked_where(m, corr).compressed()

# conn = compute_connectivity()
# print conn.shape
# print conn[1061], conn[377]

# a = np.ones((4, 19701))
# print np.mean(a, axis=0).shape

# g0 = gs[0]
# g1 = gs[1]

# # print g.shape
# # print np.min(g[0]), np.max(g[0]), np.mean(g[0])
# # print np.min(g[1]), np.max(g[1]), np.mean(g[1])

# train_data = np.loadtxt('data/corr/corr_1D.csv', delimiter=',')
# X_train_data = train_data[:, 1:]
# y_train_data = train_data[:, 0].astype(int)

# g0sort = np.argsort(np.abs(g0))[-10:]
# print inputs[g0sort]
# print np.mean(X_train_data[y_train_data == 0][:, g0sort], axis=0)
# print np.mean(X_train_data[y_train_data == 1][:, g0sort], axis=0)
# # g0 = (g0 - np.mean(g0)) / np.std(g0, ddof=1)

# g1sort = np.argsort(np.abs(g1))[-10:]
# print inputs[g1sort]
# print np.mean(X_train_data[y_train_data == 0][:, g1sort], axis=0)
# print np.mean(X_train_data[y_train_data == 1][:, g1sort], axis=0)
# # g1 = (g1 - np.mean(g1)) / np.std(g1, ddof=1)


# def triangulize(data, size, old=None, top=False):
#     if old is None:
#         old = np.zeros((size, size))
#     elif top:
#         old = old.T

#     per_row = 1
#     pos = 0
#     for i in range(1, size):
#         old[i][0:per_row] = data[pos:pos+per_row]
#         pos = pos + per_row
#         per_row = per_row + 1
#     if top:
#         return old.T
#     return old

# triangle = triangulize(g1 * 1000, 190, triangulize(g0 * 1000, 190), True)

# from matplotlib import pyplot as plt
# from matplotlib.widgets import Slider

# fig, ax = plt.subplots()
# plt.subplots_adjust(bottom=0.1)
# conn = plt.imshow(triangle, interpolation="nearest", cmap="RdBu_r")
# axfreq = plt.axes([0.1, 0.01, 0.8, 0.03])
# sfreq = Slider(axfreq, 'Freq', valmin=0.0, valmax=np.max(np.abs(triangle)), valinit=0.0)

# def update(val):
#     data = triangle.copy()
#     data[np.abs(data) < sfreq.val] = 0.0
#     conn.set_data(data)
#     fig.canvas.draw_idle()

# sfreq.on_changed(update)

# plt.show()
