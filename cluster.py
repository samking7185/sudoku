import numpy as np
import load_mnist
import cv2
from features import *
import matplotlib.pyplot as plt
from copy import deepcopy
import time
from skimage.morphology import skeletonize
from skimage.util import invert

tic = time.perf_counter()

training_data, validation_data, test_data = load_mnist.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)
validation_data = list(validation_data)

image = training_data[52][0]
# image_size = [20, 20]
# new_image = extraction(image, image_size)
new_image = image.reshape((28,28))
BW = cv2.threshold(new_image, 0.6, 1, cv2.THRESH_BINARY)
BW = np.array(BW[1])
bigdata = np.nonzero(BW)
bigdata = np.stack(bigdata, axis=1)

# M = cv2.moments(BW)
# cX = int(M["m10"] / M["m00"])
# cY = int(M["m01"] / M["m00"])
# plt.imshow(BW)
# plt.scatter(cX,cY,color='red')
# plt.show()
data_str = {}
data_full = {}

n = len(bigdata)
maxiter = 100
max_cluster = 10
min_cluster = 2
fpc = 0

for coord in bigdata:
    coord = tuple(coord)
    data_str[coord] = {"cluster_id": 0, "mu": 0}

data_full = {"pts": data_str, "fpc": 0, "center_xy": [], "loop_id": 0}

k = max_cluster
q = 2
error = True
error_val = 0.00001
power = -2 / (q - 1)

Mu_last = np.zeros((n, k))
Dmat = np.zeros((len(bigdata), k))

Mu = np.random.rand(k, n)
Mu /= np.ones((k, 1)).dot(np.atleast_2d(Mu.sum(axis=0))).astype(np.float64)
Mu = np.fmax(Mu, np.finfo(np.float64).eps)
Mu = Mu.T

index_vals = np.argmax(Mu, axis=1)
cluster_assign = np.empty(k, dtype=object)

for iteration in range(maxiter):
    Vhsq = np.power(Mu, q)
    Vh = np.zeros((k, 2))

    for idx in range(k):
        Vh[idx, :] = np.sum(np.transpose(np.tile(Vhsq[:, idx], (2, 1))) * bigdata, axis=0) / np.sum(Vhsq[:, idx])
    for idx in range(k):
        Dmat[:, idx] = np.linalg.norm(bigdata - Vh[idx, :], axis=1)

    Dmat = np.fmax(Dmat, np.finfo(np.float64).eps)
    Mu = np.power(Dmat, power).T
    Mu /= np.ones((k, 1)).dot(np.atleast_2d(Mu.sum(axis=0))).astype(np.float64)

    if np.absolute(np.linalg.norm(Mu.T - Mu_last)) < error_val:
        Mu = Mu.T
        fpc = np.trace(np.transpose(Mu).dot(Mu)) / Mu.shape[0]
        break
    Mu = Mu.T
    Mu_last = Mu

for keys in data_full["pts"]:
    i = bigdata.tolist().index(list(keys))
    cid = np.argmax(Mu[i]) + 1
    data_full["pts"][keys].update(mu=Mu[i], cluster_id=cid)

data_full.update(fpc=fpc, center_xy=list(Vh))

t1 = data_full["fpc"]
cldata = data_full
climage = np.zeros_like(BW)

for keys in cldata['pts']:
    climage[keys] = data_full["pts"][keys]["cluster_id"]
connect = []
for i in range(0, climage.shape[0], 1):
    for j in range(0, climage.shape[1], 1):
        imbox = climage[i:i+2, j:j+2]
        if np.sum(imbox):
            if not np.any(imbox == 0):
                if len(np.unique(imbox)) != 1:
                    connect.append(np.sort(np.unique(imbox)))
tconnect = np.asarray(connect, dtype=object)

connect = []
for idx, x in enumerate(tconnect):
    if len(x) > 2:
        sconnect = np.delete(tconnect, idx)
        data_full['loop_id'] = 1
    else:
        if x.tolist() not in connect:
            connect.append(x.tolist())

skeleton = skeletonize(BW)
fig, axes = plt.subplots(nrows=1, ncols=3)

ax = axes.ravel()

ax[0].imshow(BW, aspect="auto")
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)

ax[1].imshow(skeleton, aspect="auto")
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

x11 = []
x22 = []
for val in connect:
    num1 = int(val[0] - 1)
    num2 = int(val[1] - 1)

    x1 = data_full['center_xy'][num1]
    x2 = data_full['center_xy'][num2]
    x1p = [round(num1, 0) for num1 in x1]
    x2p = [round(num2, 0) for num2 in x2]
    ax[2].plot([x1p[1], x2p[1]], [x1p[0], x2p[0]], color='red')
ax[2].set_title('FCM skeleton', fontsize=20)
plt.gca().invert_yaxis()
fig.tight_layout()
plt.show()

