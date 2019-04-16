import mglearn

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


X, y = mglearn.datasets.make_forge()

# Split our data into a training set.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Load classifier.
clf = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier using the training set.
clf.fit(X_train, y_train)

print("test set predictions: {0}".format(clf.predict(X_test)))

print("test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))  # 0.86

# Draw decision boundary
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 5, 10], axes):
    # the fit method returns the object self, so we can instantiate and fit in one line
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=0.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
plt.show()
