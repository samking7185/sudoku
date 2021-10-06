import numpy as np
import load_mnist
from features import *
import matplotlib.pyplot as plt

training_data, validation_data, test_data = load_mnist.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)
validation_data = list(validation_data)

image = training_data[0][0]
image_size = [20,20]
new_image = extraction(image, image_size)
BW = cv2.threshold(new_image, 0.6, 1, cv2.THRESH_BINARY)
BW = np.array(BW[1])
bigdata = np.nonzero(BW)
bigdata = np.stack(bigdata, axis=1)
k = 3
q = 3
n = 0.5
# random_idx = np.sort(np.random.choice(n, size=k, replace=False))
# centers = bigdata[random_idx, :]

data_bar = np.mean(bigdata, axis=0)
data_std = np.std(bigdata, axis=0)
C = np.zeros((k,2))
Mu_last = np.zeros((len(bigdata), k))

if k % 2 == 0:
    for idx in range(0, k, 2):
        C[idx,:] = data_bar - n*data_std
        C[idx+1,:] = data_bar + n*data_std
        n += 0.5
else:
    C[0, :] = data_bar
    for idx in range(1,k,2):
        C[idx,:] = data_bar - n*data_std
        C[idx+1,:] = data_bar + n*data_std
        n += 0.5

C = C[np.argsort(C[:,1])]
D = np.zeros((bigdata.shape[0], k))
error = True
error_val = 0.51
while error:
    for idx in range(k):
        power = 1 / (q-1)
        dsquared = np.array([np.linalg.norm(v) for v in bigdata - C[idx]])
        Dtemp = np.power(1 / np.power(dsquared, 2), power)
        D[:,idx] = Dtemp

    Mu = np.divide(D, np.reshape(np.repeat(np.sum(D, axis=1), k), (len(bigdata), k)))
    if np.amax(np.amax(np.absolute((Mu - Mu_last)))) < error_val:
        error = False
    Vhsq = np.power(Mu, q)
    Vh = np.zeros((k, 2))
    for idx in range(k):
        Vtemp = np.multiply(bigdata, np.reshape(np.repeat(Vhsq[:,idx], 2), (len(bigdata), 2)))
        Vh[idx,:] = np.sum(Vtemp, axis=0) / np.sum(Vhsq[:,idx])

    C = Vh
    Mu_last = Mu

plt.scatter(bigdata[:,0], bigdata[:,1])
plt.scatter(C[:,0], C[:,1])
plt.gca().invert_yaxis()
plt.show()
# val = np.power(MUi, 2)
# bottom = np.sum(val, axis=1)
#
# topX = np.multiply(np.reshape(np.repeat(bigdata[:,0], k), (97,k)), val)
# topY = np.multiply(np.reshape(np.repeat(bigdata[:,1], k), (97,k)), val)
#
# Cx, Cy = np.divide(topX, bottom), np.divide(topY, bottom)
#
# Dtx = np.reshape(np.repeat(bigdata[:,0], k), (97,k)) - Cx
# Dty = np.reshape(np.repeat(bigdata[:,1], k), (97,k)) - Cy
#
# DSQx = np.array([np.dot(np.transpose(Dtx[:,idx]), Dtx[:,idx]) for idx in range(k)])
# DSQy = np.array([np.dot(np.transpose(Dty[:,idx]), Dty[:,idx]) for idx in range(k)])
#
# power = 1 / (q - 1)
#
# Dpx = np.power(1 / DSQx, power)
# Dpy = np.power(1 / DSQy, power)

# for idx in range(k):
#     val = np.power(MUi[:,idx], 2)
#     bottom = np.sum(val)
#     top = np.sum(np.multiply(np.stack((val, val), axis=1), bigdata), axis=0)
#     Ctemp = np.divide(top, bottom)
#     C[idx,:] = Ctemp
#
#     Dtemp = bigdata - Ctemp
#     Dsq = np.array([np.dot(np.transpose(x), x) for x in Dtemp])
#     power = 1 / (q - 1)
#
#     Dp = np.power(1/Dsq, power)
#     D[:,idx] = Dp

# sMU = np.sum(D, axis=1)
# MU = np.divide(D, np.reshape(np.repeat(sMU, 3), (97,k)))
# V = np.power(MU, q)
#
# Vhx = np.multiply(V, np.reshape(np.repeat(bigdata[:,0], 3), (97,k)))
# Vhy = np.multiply(V, np.reshape(np.repeat(bigdata[:,1], 3), (97,k)))
#
# Vsum, Vhxsum, Vhysum = np.sum(V, axis=0), np.sum(Vhx, axis=0), np.sum(Vhy, axis=0)
# Vh = np.array([Vhxsum / Vsum, Vhysum / Vsum])
s = 1

# d1 = bigdata
# d2 = bigdata
#
# dt1 = np.array([np.dot(np.transpose(x), x) for x in d1])
# dt2 = np.array([np.dot(np.transpose(x), x) for x in d2])
#
# power = 1 / (q - 1)
#
# v1 = np.power(1/dt1, power)
# v2 = np.power(1/dt2, power)
#
# u1 = np.divide(v1, v1 + v2)
# u2 = np.divide(v2, v1 + v2)
#
# V1 = np.power(u1, q)
# V2 = np.power(u2, q)
#
# Vh1t = np.multiply(np.stack((V1, V1), axis=1), bigdata)
# Vh2t = np.multiply(np.stack((V2, V2), axis=1), bigdata)
#
# Vh1 = np.sum(Vh1t, axis=0) / np.sum(V1)
# Vh2 = np.sum(Vh2t, axis=0) / np.sum(V2)

