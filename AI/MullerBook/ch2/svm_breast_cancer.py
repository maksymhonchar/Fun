import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target,
    random_state=42
)

svc = SVC()
svc.fit(X_train, y_train)

print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))  # 1.00
print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))  # 0.62 -> overfitting

# visualize min-max values for each featuress, lotted in log-space
plt.plot(X_train.min(axis=0), 'o', label='min')
plt.plot(X_train.max(axis=0), '^', label='max')
plt.yscale('log')
plt.xlabel('feature index')
plt.ylabel('feature magnitude')
plt.legend()
# plt.show()  # features are of completely different orders of magnitude == issue for svm kernel

# Rescale each feature so that they are all approximately on the same scale.

# do for the train set
# common method - scale the data that all features are between 0 and 1.
min_on_training = X_train.min(axis=0)  # get min value per feature on the training set.
range_on_training = (X_train - min_on_training).max(axis=0)
# subtract the min, and divide by range. afterward, min=0 and max=1 for each feature.
X_train_scaled = (X_train - min_on_training) / range_on_training
# print("Minimum for each feature\n{}".format(X_train_scaled.min(axis=0)))
# print("Maximum for each feature\n {}".format(X_train_scaled.max(axis=0)))

# do for the test set
min_on_test = X_test.min(axis=0)
range_on_testing = (X_test - min_on_test).max(axis=0)
X_test_scaled = (X_test - min_on_test) / range_on_testing
# print("Minimum for each feature\n{}".format(X_test_scaled.min(axis=0)))
# print("Maximum for each feature\n {}".format(X_test_scaled.max(axis=0)))

svc_scaled_features = SVC()
svc_scaled_features.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(svc_scaled_features.score(X_train_scaled, y_train)))  # 0.946 
print("Accuracy on test set: {:.3f}".format(svc_scaled_features.score(X_test_scaled, y_test)))  # 0.979 - a huge difference! now model underfits

svc_scaled_features_C1000 = SVC(C=5, gamma=0.10)
svc_scaled_features_C1000.fit(X_train_scaled, y_train)
print("Accuracy on training set: {:.3f}".format(
    svc_scaled_features_C1000.score(X_train_scaled, y_train))
)  # 0.979
print("Accuracy on test set: {:.3f}".format(
    svc_scaled_features_C1000.score(X_test_scaled, y_test))
)  # 0.944 (!)
