# Agglomerative clustering refers to a collection of clustering algorithms that all build
# upon the same principles: the algorithm starts by declaring each point its own cluster,
# and then merges the two most similar clusters until some stopping criterion is satisfied.

# The stopping criterion implemented in scikit-learn is the number of clusters,
# so similar clusters are merged until only the specified number of clusters are left.

# There are several linkage criteria that specify how exactly the “most similar cluster” is measured.
# This measure is always defined between two existing clusters.

# Linkage options in scikit-learn:
# ward - pick the two clusters to merge such that the variance within all clusters increases the least.
# average - merge the two clusters that have the smallest average distance between all their points.
# complete (maximum linkage) - merge the two clusters that have the smallest maximum distance between their points.

import matplotlib.pyplot as plt

import mglearn

from scipy.cluster.hierarchy import dendrogram, ward

from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs


def blobs_example():
    X, y = make_blobs(random_state=40, n_samples=250)

    agg = AgglomerativeClustering(n_clusters=3)
    # Because of the way the algorithm works, agglomerativ clustering cannot make predictions for new data points.
    # Therefore, Agglomerative Clustering has no predict method - use fit_predict instead
    assignment = agg.fit_predict(X)

    mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
    plt.xlabel("f0")
    plt.ylabel("f1")
    plt.show()

# blobs_example()


def visualize_hierarchical_clustering_dendrogram():
    # Dendrograms - use SciPy instead of sklearn
    X, y = make_blobs(n_samples=12, random_state=40)
    
    # Apply the ward clustering to X adta array.
    # SciPy ward functioon returns an array that specifies the distances bridged when performing agglomerative clustering.
    linkage_array = ward(X)  # distances between clusters

    # Plot dendrogram
    dendrogram(linkage_array)
    ax = plt.gca()
    bounds = ax.get_xbound()
    ax.plot(bounds, [7.25, 7.25], '--', c='k')
    ax.plot(bounds, [4, 4], '--', c='k')
    ax.text(bounds[1], 7.25, ' two clusters', va='center', fontdict={'size': 15})
    ax.text(bounds[1], 4, ' three clusters', va='center', fontdict={'size': 15})
    plt.xlabel("Sample index")
    plt.ylabel("Cluster distance")
    plt.show()

visualize_hierarchical_clustering_dendrogram()
