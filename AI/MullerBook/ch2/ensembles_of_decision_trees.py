# Ensembles are methods that combine multiple machine learning models to create more powerful models.
# Two ensemble models that have proven to be effective on a wide range of datasets for classification and regression - random forests and gradient boosted decision trees.

# max_features=sqrt(n_features) for classification
# max_features=log2(n_features) for regression

import matplotlib.pyplot as plt

import mglearn

from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

# In any real application, we would use many more tres (often hundreds or thousands), leading to really smooth boundaries in the end.

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)

# print(forest.estimators_)

# Visualize the decision boundaries learned by each tree
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title("Tree {0}".format(i))
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, 1], alpha=0.4)
axes[-1, 1].set_title("Random forest")
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
plt.show()