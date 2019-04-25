import graphviz

import matplotlib.pyplot as plt

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz


cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target,
    stratify=cancer.target, random_state=42
)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

print("Training set accuracy: {:.2f}".format(tree.score(X_train, y_train)))  # 1.000
print("Test set accuracy: {:.2f}".format(tree.score(X_test, y_test)))  # 0.94

# Limiting the depth of the tree decreases overfitting.
tree_prepruning = DecisionTreeClassifier(max_depth=3, random_state=0)
tree_prepruning.fit(X_train, y_train)

print("Training set accuracy: {:.2f}".format(tree_prepruning.score(X_train, y_train)))  # 0.98
print("Test set accuracy: {:.2f}".format(tree_prepruning.score(X_test, y_test)))  # 0.94

# Analyze decision trees.
export_graphviz(tree, class_names=["malignant", "benign"], feature_names=cancer.feature_names,
    out_file='tree.dot', impurity=False, filled=True
)
with open('tree.dot') as f:
    dot_graph = f.read()
print(graphviz.Source(dot_graph))

print(tree.feature_importances_)

# visualize feature importances:
n_features = cancer.data.shape[1]
plt.barh(range(n_features), tree.feature_importances_, align='center')
plt.yticks(np.arange(n_features), cancer.feature_names)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()
