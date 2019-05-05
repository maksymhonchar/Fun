# k-means tries to find cluster centers that are representative of certain regions of the data.

import matplotlib.pyplot as plt

import mglearn

import numpy as np

from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans


def kmeans_blobs():
    X, y = make_blobs(random_state=42, n_samples=10000, centers=10)

    kmeans_model = KMeans(n_clusters=5, n_jobs=-1)
    kmeans_model.fit(X)

    print("Cluster memberships:\n{}".format(kmeans_model.labels_))
    # print("prediction on X: {0}".format(kmeans_model.predict(X)))  # Outputs the same result as labels_

    # Plot #1

    mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans_model.labels_, markers='o')
    mglearn.discrete_scatter(
        kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], [0, 1, 2, 3, 4],
        markers='^', markeredgewidth=2
    )
    plt.show()

    # Plot #2

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # using two cluster centers:
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    assignments = kmeans.labels_
    mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[0])

    # using five cluster centers:
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X)
    assignments = kmeans.labels_
    mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[1])

    plt.show()

# kmeans_blobs()


def kmeans_failure():
    # Each cluster is defined solely by its center, which means that each cluster is a convex shape.
    # As a result of this, k-means can only capture relatively simple shapes.
    # k-means also assumes that all clusters have the same “diameter” in some sense; it always draws the boundary between clusters to be exactly in the middle between the cluster centers.
    X_varied, y_varied = make_blobs(
        n_samples=150, cluster_std=[1., 2.5, .5],
        random_state=170
    )
    y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(X_varied)
    mglearn.discrete_scatter(X_varied[:, 0], X_varied[:, 1], y_pred)
    plt.legend(["cluster 0", "cluster 1", "cluster 2"], loc='best')
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()

# kmeans_failure()


def kmeans_failure_2():
    X, y = make_blobs(random_state=170, n_samples=600)
    rng = np.random.RandomState(74)

    # Transform the data to be stretched
    transformation = rng.normal(size=(2, 2))
    X = np.dot(X, transformation)

    # Cluster the data into 3 clusters
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)

    # plot the cluster assignments and cluster centers
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.scatter(
        kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
        marker='^', c=[0, 1, 2], s=100, linewidth=2
    )
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()

# kmeans_failure_2()


def kmeans_failure_3():  # moon
    # k-means also performs poorly if the clusters have more complex shapes, like the two_moons data
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)
    # plot the cluster assignments and cluster centers
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=60)
    plt.scatter(
        kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
        c=[mglearn.cm2(0), mglearn.cm2(1)],
        marker='^', s=100, linewidth=2
    )
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()

kmeans_failure_3()
