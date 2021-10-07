import numpy as np
import load_mnist
import cv2
from features import *
import matplotlib.pyplot as plt


training_data, validation_data, test_data = load_mnist.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)
validation_data = list(validation_data)

image = training_data[1][0]
image_size = [20,20]
new_image = extraction(image, image_size)
BW = cv2.threshold(new_image, 0.6, 1, cv2.THRESH_BINARY)
BW = np.array(BW[1])
bigdata = np.nonzero(BW)
bigdata = np.stack(bigdata, axis=1)
n = len(bigdata)
J = []
H = []
maxiter = 100

for k in range(3, 10, 1):
    # k = 4
    q = 2
    error = True
    error_val = 0.00001
    power = -2 / (q - 1)

    Mu_last = np.zeros((n, k))
    Dmat = np.zeros((bigdata.shape[0], k))

    Mu = np.random.rand(k, n)
    Mu /= np.ones((k, 1)).dot(np.atleast_2d(Mu.sum(axis=0))).astype(np.float64)
    Mu = np.fmax(Mu, np.finfo(np.float64).eps)
    Mu = Mu.T

    for iteration in range(maxiter):
        Vhsq = np.power(Mu, q)
        Vh = np.zeros((k, 2))

        for idx in range(k):
            Vh[idx,:] = np.sum(np.transpose(np.tile(Vhsq[:,idx], (2,1))) * bigdata, axis=0) / np.sum(Vhsq[:,idx])
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

index_vals = np.argmax(Mu, axis=1)
cluster_assign = np.empty(k, dtype=object)
cluster_dist_bar = np.empty(k, dtype=object)
for idx in range(k):
    cluster_assign[idx] = bigdata[index_vals == idx]
    cluster_dist = np.array([np.linalg.norm(cluster_assign[idx] - x, axis=1) for x in cluster_assign[idx]])
    cluster_dist_bar[idx] = np.mean(cluster_dist, axis=1)
cluster_score = np.empty(k, dtype=object)

for j,(cls,bar) in enumerate(zip(cluster_assign, cluster_dist_bar)):
    cluster_score[j] = np.zeros(len(cls))
    for i,(x,m) in enumerate(zip(cls, bar)):
        cls_dist = [np.mean(np.linalg.norm(y - x)) for y in cluster_assign if not np.array_equal(y,cls)]
        cls_dbar = np.mean(cls_dist)
        cluster_score[j][i] = 1 - m / cls_dbar

s = 1

# print(C)
# plt.scatter(bigdata[:,0], bigdata[:,1])
# plt.scatter(Vh[:,0], Vh[:,1])
# plt.show()


