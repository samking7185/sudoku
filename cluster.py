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

k = 8
q = 2
error = True
error_val = 0.00001
power = -2 / (q - 1)

Mu_last = np.zeros((len(bigdata), k))
Dmat = np.zeros((bigdata.shape[0], k))

Mu = np.random.rand(k, len(bigdata))
Mu /= np.ones((k, 1)).dot(np.atleast_2d(Mu.sum(axis=0))).astype(np.float64)
Mu = np.fmax(Mu, np.finfo(np.float64).eps)
Mu = Mu.T

while error:
    Vhsq = np.power(Mu, q)
    Vh = np.zeros((k, 2))

    for idx in range(k):
        Vh[idx,:] = np.sum(np.transpose(np.tile(Vhsq[:,idx], (2,1))) * bigdata, axis=0) / np.sum(Vhsq[:,idx])
    for idx in range(k):
        Dmat[:, idx] = np.linalg.norm(bigdata - Vh[idx, :], axis=1)

    Dmat = np.fmax(Dmat, np.finfo(np.float64).eps)
    J = np.sum(np.power(np.multiply(Dmat, Mu), 2))
    Mu = np.power(Dmat, power).T
    Mu /= np.ones((k, 1)).dot(np.atleast_2d(Mu.sum(axis=0))).astype(np.float64)

    if np.absolute(np.linalg.norm(Mu.T - Mu_last)) < error_val:
        error = False
    Mu = Mu.T
    Mu_last = Mu

# print(C)
plt.scatter(bigdata[:,0], bigdata[:,1])
plt.scatter(Vh[:,0], Vh[:,1])
plt.show()


