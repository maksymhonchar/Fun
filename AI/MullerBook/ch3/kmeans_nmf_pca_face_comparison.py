# Even though k-means is a clustering algorithm, there are interesting parallels between k-means and the decomposition methods like PCA and NMF
# k-means tries to represent each data point using a cluster center.
# You can think of that as each point being represented using only a single component, which is given by the cluster center.
# This view of k-means as a decomposition method, where each point is represented using a single component, is called vector quantization.

import matplotlib.pyplot as plt

import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import train_test_split


people = fetch_lfw_people(min_faces_per_person=20, resize=0.5)

image_shape = people.images[0].shape

mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

# scale the grayscale values to be in (0;1) instead of (0;255) for better numeric stability
X_people = X_people / 255.

X_train, X_test, y_train, y_test = train_test_split(
    X_people, y_people,
    stratify=y_people,
    random_state=42
)

# Compare 3 algorithms showing the components extracted on subplots and reconstructions of faces from the test set.

nmf = NMF(n_components=100, random_state=0)
nmf.fit(X_train)

pca = PCA(n_components=100, random_state=0)
pca.fit(X_train)

kmeans = KMeans(n_clusters=100, random_state=0)
kmeans.fit(X_train)

X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
X_reconstructed_nmf = np.dot(nmf.transform(X_test), nmf.components_)
X_reconstructed_kmeans = kmeans.cluster_centers_[kmeans.predict(X_test)]

# Plotting 1
def comparing_cluster_centers_to_components():
    fig, axes = plt.subplots(
        3, 5, figsize=(8, 8),
        subplot_kw={'xticks': (), 'yticks': ()}
    )
    fig.suptitle('extracted components')
    for ax, comp_kmeans, comp_pca, comp_nmf in zip(axes.T, kmeans.cluster_centers_, pca.components_, nmf.components_):
        ax[0].imshow(comp_kmeans.reshape(image_shape))
        ax[1].imshow(comp_pca.reshape(image_shape), cmap='viridis')
        ax[2].imshow(comp_nmf.reshape(image_shape))
    axes[0, 0].set_ylabel("kmeans")
    axes[1, 0].set_ylabel("pca")
    axes[2, 0].set_ylabel("nmf")

# Plotting 2
def comparing_img_reconstructions():
    fig, axes = plt.subplots(
        4, 5, figsize=(8, 8),
        subplot_kw={'xticks': (), 'yticks': ()}
    )
    fig.suptitle('reconstructions')
    for ax, orig, rec_kmeans, rec_pca, rec_nmf in zip(axes.T, X_test, X_reconstructed_kmeans, X_reconstructed_pca,X_reconstructed_nmf):
        ax[0].imshow(orig.reshape(image_shape))
        ax[1].imshow(rec_kmeans.reshape(image_shape))
        ax[2].imshow(rec_pca.reshape(image_shape))
        ax[3].imshow(rec_nmf.reshape(image_shape))
        axes[0, 0].set_ylabel("original")
        axes[1, 0].set_ylabel("kmeans")
        axes[2, 0].set_ylabel("pca")
        axes[3, 0].set_ylabel("nmf")

# comparing_cluster_centers_to_components()
comparing_img_reconstructions()
plt.show()
