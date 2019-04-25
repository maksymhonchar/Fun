# iris problem
#   supervised learning problem.
#   three-class classification problem.
# features: the length and width of the petals and the length and width of the sepals, cm
# labels: setosa, versicolor, or virginica.

import matplotlib.pyplot as plt

import mglearn

import numpy as np

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import pandas as pd


iris_dataset = load_iris()
print('keys of iris_dataset Bunch obj are {0}'.format(iris_dataset.keys()))
# print('descr is {0}'.format(iris_dataset.DESCR))
print('target names are {0}'.format(iris_dataset.target_names))
print('feature names are {0}'.format(iris_dataset.feature_names))

# Data itself is contained in the [target] and [data] fields.
# print(iris_dataset.target)  # 0 setosa; 1 versicolor; 2 virginica
# print(iris_dataset.data)  # numpy.ndarray
print(type(iris_dataset.target), iris_dataset.target.shape)  # shape: (150,)
print(type(iris_dataset.data), iris_dataset.data.shape)  # shape: (150,4)

print("First 5 samples of iris_dataset.data:\n{0}".format(iris_dataset.data[:5]))

# train_test_split: 75%:25%
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset.data, iris_dataset.target, random_state=50
)
print(X_train.shape, y_train.shape)  # (112,4) (112,)
print(X_test.shape, y_test.shape)  # (38,4) (38,)

# visualize data
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(
    iris_dataframe, c=y_train, 
    figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3
)
plt.show()
print(type(grr))

# try to predict new data using kNN algorithm
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

X_new = np.array( [ [5, 2.9, 1, 0.2] ] )

prediction = knn.predict(X_new)
print("Prediction: {0}".format(prediction))
print("Predicted label is {0}".format(iris_dataset.target_names[prediction]))  # setosa

# compute the accuracy of the model
y_pred = knn.predict(X_test)
print("Test set score: {0}".format(np.mean(y_pred == y_test)))  # 0.9736842105263158
print("Test set score v2: {0}".format(knn.score(X_test, y_test)))  # 0.9736842105263158
