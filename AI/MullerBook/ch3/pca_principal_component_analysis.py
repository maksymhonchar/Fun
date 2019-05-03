# Principal component analysis is a method that rotates the dataset in a way such that
# the rotated features are statistically uncorrelated. This rotation is often followed by
# selecting only a subset of the new features, according to how important they are for
# explaining the data.

import matplotlib.pyplot as plt

import mglearn

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


cancer = load_breast_cancer()


def visualize_cancer_dataset_histograms():
    # One of the most common applications of PCA is visualizing high-dimensional datasets.
    fig, axes = plt.subplots(15, 2, figsize=(10, 20))
    malignant = cancer.data[cancer.target == 0]
    benign = cancer.data[cancer.target == 1]

    ax = axes.ravel()

    # Create a histogram for each of the features, counting how often a data point appears with a feature in a certain range (called a bin).
    for i in range(30):
        _, bins = np.histogram(cancer.data[:, i], bins=50)
        ax[i].hist(malignant[:, i], bins=bins, alpha=.5)
        ax[i].hist(benign[:, i], bins=bins, alpha=.5)
        ax[i].set_title(cancer.feature_names[i])
        ax[i].set_yticks(())
    ax[0].set_xlabel("Feature magnitude")
    ax[0].set_ylabel("Frequency")
    ax[0].legend(["malignant", "benign"], loc="best")
    plt.show()

# these plots dont show us anything about the interactions between variables and how these relate to the classes.
# visualize_cancer_dataset_histograms()


def visualize_cancer_dataset_pca():
    # Before applying PCA, scale the data so that each feature has unit variance.
    scaler = StandardScaler()
    scaler.fit(cancer.data)
    X_scaled = scaler.transform(cancer.data)

    pca = PCA(n_components=2)  # keep the first 2 principal components of the data
    pca.fit(X_scaled)

    X_pca = pca.transform(X_scaled)  # transform data onto the first two principal components.

    print("Original shape: {}".format(str(X_scaled.shape)))  # (569, 30)
    print("Reduced shape: {}".format(str(X_pca.shape)))  # (569, 2)

    print("PCA component shape: {}".format(pca.components_.shape))  # (2, 30)

    # plot first vs second principal component
    plt.figure(figsize=(8, 8))
    mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
    plt.legend(cancer.target_names, loc="best")
    plt.gca().set_aspect("equal")
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.show()

    # Visualize PCA coefficients
    plt.matshow(pca.components_, cmap='viridis')
    plt.yticks([0, 1], ["First component", "Second component"])
    plt.colorbar()
    plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
    plt.xlabel("Feature")
    plt.ylabel("Principal components")
    plt.show()

visualize_cancer_dataset_pca()
