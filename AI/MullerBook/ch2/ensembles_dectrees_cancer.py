import matplotlib.pyplot as plt

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

cancer = load_breast_cancer()

n_features = cancer.data.shape[1]
tree_max_features = int(np.sqrt(n_features))

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target,
    random_state=42
)

forest = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1, max_features=tree_max_features
)
forest.fit(X_train, y_train)

print("Forest accuracy on training set: {:.5f}".format(forest.score(X_train, y_train)))  # 1.00
print("Forest accuracy on test set: {:.5f}".format(forest.score(X_test, y_test)))  # 0.96503

# Typically, feature importances provided by the random forest are more reliable than the ones provided by a single tree.

plt.barh(range(n_features), forest.feature_importances_, align='center')
plt.yticks(np.arange(n_features), cancer.feature_names)
plt.xlabel('feature importance')
plt.ylabel('feature')
plt.show()
