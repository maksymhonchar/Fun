# DT models are widely used for classification & regression tasks.
# Essentially, they learn a hierarchy of if/else questions, leading to a decision.
# if/else questions are called "tests"
# tests for continuous data are: "Is feature X1 larger than value M?"

import matplotlib.pyplot as plt

import mglearn

import numpy as np

from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# Moon dataset parameters.
moon_ds_size = 100
moon_ds_noise = 0.15

X_moons, y_moons = make_moons(n_samples=moon_ds_size, noise=moon_ds_noise, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_moons, y_moons, random_state=42)

# print(X_moons.shape)  # (100, 2)
# print(y_moons.shape)  # (100,)

moons_feature_1 = X_moons[:, 0]
moons_feature_2 = X_moons[:, 1]
# plt.scatter(moons_feature_1, moons_feature_2, c=y_moons)

# plt.title("sklearn.datasets.make_moons")
# plt.show()

# Implement decision tree to classification problem.

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
y_predict = tree_model.predict(X_test)
print(accuracy_score(y_test, y_predict))
print(tree_model.score(X_train, y_train))
print(tree_model.score(X_test, y_test))

n_features = X_moons.shape[1]
features_names = ("f1", "f2")
for i, color in zip(range(n_features), "ry"):
    idx = np.where(y_moons == i)
    plt.scatter(
        X_moons[idx, 0], X_moons[idx, 1],
        c=color, label=features_names[i], edgecolor='black', s=15
    )
plt.show()  # ???

mglearn.plots.plot_tree_progressive()
plt.show()
