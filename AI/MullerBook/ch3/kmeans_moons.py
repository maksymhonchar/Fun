# Reducing it to one dimension with PCA or NMF would completely destroy the structure of the data.
# But we can find a more expressive representation with k-means, by using more cluster centers

# Using this 10-dimensional representation, it would now be possible to separate the two 
# half-moon shapes using a linear model, which would not have been possible using the original two features.

import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.cluster import KMeans



X, y = make_moons(
    n_samples=200, noise=0.05,
    random_state=50
)

kmeans = KMeans(
    n_clusters=10,  # data being represented using 10 components (== we have 10 new features)
    random_state=0
)
kmeans.fit(X)

y_pred = kmeans.predict(X)

print("Cluster memberships:\n{}".format(y_pred))

plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=60, cmap='Paired')
plt.scatter(
    kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
    s=60, marker='^', c=range(kmeans.n_clusters), linewidth=2, cmap='Paired'
)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

distance_features = kmeans.transform(X)
print("Distance feature shape: {0}".format(distance_features.shape))
print("Distance features:\n{0}".format(distance_features))
