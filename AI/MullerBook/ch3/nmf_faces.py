import matplotlib.pyplot as plt

import numpy as np

from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import NMF
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

nmf = NMF(n_components=15, random_state=42)
nmf.fit(X_train)

X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)

# extract a few components (15) using NMF
fix, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape))
    ax.set_title("{}. component".format(i))
plt.show()

# show images for which components shown above are particularly strong
compn = 3
# sort by 3rd component, plot first 10 images. 3rd component == looking right
inds = np.argsort(X_train_nmf[:, compn])[::-1]
fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
    ax.imshow(X_train[ind].reshape(image_shape))
plt.show()

compn = 7
# sort by 7th component, plot first 10 images. 7th component == looking left.
inds = np.argsort(X_train_nmf[:, compn])[::-1]
fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
    ax.imshow(X_train[ind].reshape(image_shape))
plt.show()
