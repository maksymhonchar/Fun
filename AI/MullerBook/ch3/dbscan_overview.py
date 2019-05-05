# DBSCAN - density-based spatial clustering of applications with noise.

# DBSCAN works by identifying points that are in “crowded” regions of the feature
# space, where many data points are close together.
# These regions are referred to as dense regions in feature space.

# The idea behind DBSCAN is that clusters form dense regions of data, separated by regions that are relatively empty.

# Points that are within a dense region are called core samples (or core points).
# If there are at least min_samples many data points within a distance of eps to a given
# data point, that data point is classified as a core sample.

import matplotlib.pyplot as plt

import mglearn

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler


X, y = make_blobs(n_samples=20, random_state=0)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X)

print("Cluster membershipd:\n{0}".format(clusters)) # [-1, -1, ..., -1, -1] - all points were assigned to noise

# Increasing min_samples (going from top to bottom in the figure) means that fewer points will be core points, and 
# more points will be labeled as noise.

# The parameter eps is somewhat more important, as it determines what it means for
# points to be “close.” Setting eps to be very small will mean that no points are core
# samples, and may lead to all points being labeled as noise. Setting eps to be very large
# will result in all points forming a single cluster.

# While DBSCAN doesn’t require setting the number of clusters explicitly, setting eps
# implicitly controls how many clusters will be found.

mglearn.plots.plot_dbscan()
plt.show()


def dbscan_moon():
    X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

    # rescale the data to zero mean and unit variance
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    # create & apply DBSCAN
    dbscan = DBSCAN()
    clusters = dbscan.fit_predict(X_scaled)

    # Plot cluster assignmets
    plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap=mglearn.cm2, s=60)
    plt.xlabel('f0')
    plt.ylabel('f1')
    plt.show()

dbscan_moon()
