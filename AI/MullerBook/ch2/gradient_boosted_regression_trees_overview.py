# GB regression treee is ensemble method that combines multiple decision trees to create a more powerful model.
# These models can be used for both regression and classification.
# GB approach: each tree tries to correct the mistakes of the previous one.

import matplotlib.pyplot as plt

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target,
    random_state=42
)

gbregrtree = GradientBoostingClassifier(random_state=42)
gbregrtree.fit(X_train, y_train)

print("Accuracy on train set: {:.5f}".format(gbregrtree.score(X_train, y_train)))  # 1.00
print("Accuracy on test set: {:.5f}".format(gbregrtree.score(X_test, y_test)))  # 0.95804

gbrt_depth1 = GradientBoostingClassifier(random_state=42, max_depth=1)
gbrt_depth1.fit(X_train, y_train)

print("Accuracy on train set: {:.5f}".format(gbrt_depth1.score(X_train, y_train)))  # 0.99061
print("Accuracy on test set: {:.5f}".format(gbrt_depth1.score(X_test, y_test)))  # 0.96503

gbrt_lrate001 = GradientBoostingClassifier(random_state=42, learning_rate=0.01)
gbrt_lrate001.fit(X_train, y_train)

print("Accuracy on train set: {:.5f}".format(gbrt_lrate001.score(X_train, y_train)))  # 0.99296
print("Accuracy on test set: {:.5f}".format(gbrt_lrate001.score(X_test, y_test)))  # 0.95804

# Visualize feature importances
n_features = cancer.data.shape[1]
plt.barh(range(n_features), gbregrtree.feature_importances_, align='center')
plt.yticks(np.arange(n_features), cancer.feature_names)
plt.xlabel('feature importance')
plt.ylabel('feature')
plt.show()
