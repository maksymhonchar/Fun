# The idea behind feature extraction is that it is possible to find a representation 
# of your data that is better suited to analysis than the raw representation you were given.
# A great example of an application where feature extraction is helpful is with images.

import matplotlib.pyplot as plt

import mglearn

import numpy as np

from sklearn.datasets import fetch_lfw_people
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


people = fetch_lfw_people(min_faces_per_person=20, resize=0.5)
image_shape = people.images[0].shape

print("people.images.shape: {}".format(people.images.shape))  # (3023, 62, 47) 3023 images, 62x47 pixels large
print("Number of classes: {}".format(len(people.target_names)))  # 62 - belonging to 62 different people


def view_person_appearance_rate():
    counts = np.bincount(people.target)
    for i, (count, name) in enumerate(zip(counts, people.target_names)):
        print("{0:25} {1:3}\t".format(name, count), end='')
        print()

# view_person_appearance_rate()


def view_images():
    fix, axes = plt.subplots(
        2, 5, figsize=(15, 8),
        subplot_kw={ 'xticks': (), 'yticks': () }
    )
    for target, image, ax in zip(people.target, people.images, axes.ravel()):
        ax.imshow(image)
        ax.set_title(people.target_names[target])
    plt.show()

# view_images()

# make the data less skewed: take up to 50 images of each person
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

knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
knn.fit(X_train, y_train)

print("Test set score of 1-nn: {:.2f}".format(knn.score(X_test, y_test)))  # 0.26

pca = PCA(
    n_components=100, whiten=True, random_state=42
)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
    
print("X_train_pca.shape: {}".format(X_train_pca.shape))  # (1547, 100)

knn_pca = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
knn.fit(X_train_pca, y_train)

print("Test set accuracy: {:.2f}".format(knn.score(X_test_pca, y_test)))  # 0.34

print("pca.components_.shape: {}".format(pca.components_.shape))  # (100, 2914)

fix, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape), cmap='viridis')
    ax.set_title("{}. component".format((i + 1)))
plt.show()

mglearn.plots.plot_pca_faces(X_train, X_test, image_shape)

plt.show()
