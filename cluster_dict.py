import numpy as np
import load_mnist
from features import *
import matplotlib.pyplot as plt
from copy import deepcopy
import time

tic = time.perf_counter()

training_data, validation_data, test_data = load_mnist.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)
validation_data = list(validation_data)

image = training_data[1][0]
image_size = [20, 20]
new_image = extraction(image, image_size)
BW = cv2.threshold(new_image, 0.6, 1, cv2.THRESH_BINARY)
BW = np.array(BW[1])
bigdata = np.nonzero(BW)
bigdata = np.stack(bigdata, axis=1)

data_str = {}
data_full = {}

n = len(bigdata)
maxiter = 100
max_cluster = 10
min_cluster = 2

for coord in bigdata:
    coord = tuple(coord)
    data_str[coord] = {"cluster": 0, "mu": 0, "shsc": 0}

for j in range(max_cluster - min_cluster):
    data_full[j] = deepcopy(data_str)

for k in range(min_cluster, max_cluster, 1):
    didx = k - min_cluster
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
    cluster_dist_bar = np.empty(k, dtype=object)
    cluster_score = np.empty(k, dtype=object)

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
            break
        Mu = Mu.T
        Mu_last = Mu

    for idx in range(k):
        cluster_assign[idx] = bigdata[index_vals == idx]
        cluster_dist = np.array([np.linalg.norm(cluster_assign[idx] - x, axis=1) for x in cluster_assign[idx]])
        cluster_dist_bar[idx] = np.mean(cluster_dist, axis=1)

    for j, (cls, bar) in enumerate(zip(cluster_assign, cluster_dist_bar)):
        cluster_score[j] = np.zeros(len(cls))
        for i, (x, m) in enumerate(zip(cls, bar)):
            cls_dist = [np.mean(np.linalg.norm(y - x)) for y in cluster_assign if not np.array_equal(y, cls)]
            cls_dbar = np.mean(cls_dist)
            cluster_score[j][i] = 1 - m / cls_dbar

    for keys in data_full[didx]:
        i = bigdata.tolist().index(list(keys))
        ii = cluster_assign[index_vals[i]].tolist().index(list(keys))
        data_full[didx][keys].update(mu=Mu[i], cluster=index_vals[i], shsc=cluster_score[index_vals[i]][ii])

toc = time.perf_counter()
print(toc-tic)