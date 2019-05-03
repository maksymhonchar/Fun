# Manifold Learning Algorithms - they allow for much more complex mappings, and often provide better visualizations than PCA.

# Manifold learning algorithms are mainly aimed at visualization, and so are rarely used to generate more than two new features

# t-SNE computes a new representation of the training data, but doesn't allow transformations of new data.

# The idea behind t-SNE is to find a two-dimensional representation of the data that preserves the distances between points as best as possible.

import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


digits = load_digits()

fig, axes = plt.subplots(
    2, 5, figsize=(10, 5),
    subplot_kw={'xticks':(), 'yticks': ()}
)
for ax, img in zip(axes.ravel(), digits.images):
    ax.imshow(img)
plt.show()

# Use PCA to visualize the data reduced to two dimensions.
pca = PCA(n_components=3)
pca.fit(digits.data)
# transform the digits data onto the first two principal components
digits_pca = pca.transform(digits.data)
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525", "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]
plt.figure(figsize=(10, 10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
for i in range(len(digits.data)):
    # Plot the digits as text instead of using scatter
    plt.text(
        digits_pca[i, 0], digits_pca[i, 1],
        str(digits.target[i]),
        color = colors[digits.target[i]], fontdict={'weight': 'bold', 'size': 9}
    )
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.show()  # problems: 0,6,4 are well separated but the others are not - significant overlap

# Use t-SNE to the same dataset & visualize digits
# t-SNE doesn't support transforming new data, the TSNE class has no transform method.
# Instead, call the fit_transform method, which will build the model and immediately return the transformed data.
tsne = TSNE(random_state=42)
digits_tsne = tsne.fit_transform(digits.data)
plt.figure(figsize=(10, 10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
for i in range(len(digits.data)):
    # actually plot the digits as text instead of using scatter
    plt.text(
        digits_tsne[i, 0], digits_tsne[i, 1],
        str(digits.target[i]),
        color = colors[digits.target[i]], fontdict={'weight': 'bold', 'size': 9}
    )
plt.xlabel("t-SNE feature 0")
plt.xlabel("t-SNE feature 1")
plt.show()
