import numpy as np
import load_mnist
from features import *
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn import metrics
import time

tic = time.perf_counter()

training_data, validation_data, test_data = load_mnist.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)
validation_data = list(validation_data)

image = training_data[10][0]
image_size = [20, 20]
new_image = extraction(image, image_size)
BW = cv2.threshold(new_image, 0.6, 1, cv2.THRESH_BINARY)
BW = np.array(BW[1])
bigdata = np.nonzero(BW)
bigdata = np.stack(bigdata, axis=1)
xpts = bigdata[:,0]
ypts = bigdata[:,1]
alldata = np.vstack((xpts, ypts))

fpcs = []
res = []
resavg = []
colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
fig2, axes2 = plt.subplots(3, 3, figsize=(8, 8))

for ncenters, ax1 in enumerate(axes1.reshape(-1), 2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)
    fpcs.append(fpc)
    cluster_membership = np.argmax(u, axis=0)
    results = metrics.silhouette_samples(alldata.T, cluster_membership)
    results_avg = metrics.silhouette_score(alldata.T, cluster_membership)
    res.append(results)
    resavg.append(results_avg)
    y_lower = 10
    for j in range(ncenters):
        ax1.plot(xpts[cluster_membership == j], ypts[cluster_membership == j], '.', color=colors[j])
        ith_cluster_silhouette_values = results[cluster_membership == j]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        axes2.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, alpha=0.7)
        axes2.axvline(x=results_avg)
        # Label the silhouette plots with their cluster numbers at the middle
        axes2.text(-0.05, y_lower + 0.5 * size_cluster_i, str(j))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    # Mark the center of each fuzzy cluster
    # for pt in cntr:
    #     ax1.plot(pt[0], pt[1], 'rs')

    ax1.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    axes2.set_title('Centers = {0}'.format(ncenters))

    # ax1.axis('off')
    # ax2.axis('off')
fig1.tight_layout()
fig2.tight_layout()
#
fig3, ax3 = plt.subplots()
ax3.plot(np.r_[2:11], fpcs)
ax3.set_xlabel("Number of centers")
ax3.set_ylabel("Fuzzy partition coefficient")

fig4, ax4 = plt.subplots()
ax4.plot(np.r_[2:11], resavg)
ax4.set_xlabel("Number of centers")
ax4.set_ylabel("Silhouette Average")
#
plt.show()
toc = time.perf_counter()
print(f'With Library {toc - tic} seconds')
s = 1
