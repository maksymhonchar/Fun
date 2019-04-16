import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer


cancer = load_breast_cancer()

print("cancer.keys(): \n{0}".format(cancer.keys()))
# print("cancer.values(): \n{0}".format(cancer.values()))

print("shape: {0}".format(cancer.data.shape))  # (569, 30): 569 data points, 30 features

print("Sampe counts per class: \n{0}".format(
    {
        n: v
        for n, v in zip(cancer.target_names, np.bincount(cancer.target))
    }
))  # malignant: 212, benign: 357

print("Feature names are: \n{0}".format(cancer.feature_names))

print("Dataset description: \n{0}".format(cancer.DESCR))

plt.plot(cancer.data[:, 0], cancer.target, 'bo')
plt.plot(cancer.data[:, 1], cancer.target, 'ro')
plt.plot(cancer.data[:, 2], cancer.target, 'co')
plt.plot(cancer.data[:, 3], cancer.target, 'yo')
plt.plot(cancer.data[:, 4], cancer.target, 'ko')
plt.plot(cancer.data[:, 5], cancer.target, 'go')
plt.xlabel("First feature")
plt.ylabel("Cancer status")
plt.show()
