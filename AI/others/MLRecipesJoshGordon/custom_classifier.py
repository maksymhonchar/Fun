# Writing classifier

import random

from scipy.spatial import distance
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class RandomCLF(object):
    # accuracy (sorted): 0.26 0.28 0.3 0.32 0.33 0.36 0.44.
    # ~32%

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        # X is a 2D array - list of lists.
        predictions = []
        for row in X:  # each row contains features for testing example
            label = random.choice(self.y)
            predictions.append(label)
        return predictions


class CustomKNN(object):
    # k-nearest neighbors - find the training point that is closest to the testing point & consider k - # of neighbors to consider.
    # 2d distance == euclidean distance: d(a,b)=sqrt((x2-x1)^2+(y2-y1)^2)
    # nd distance: d(a,b) = sqrt((x2-x1)^2+...+(n2-n1)^2)
    # euclidean distance: scipy.spatial.distance.euclidean
    # NOTE: k=1 in the CustomKNN

    # Accuracy: >90%

    # PROS: relatively simple
    # CONS: computationally intensive - needs to iterate over every training point
    # CONS: hard to represent relationships between features - we cannot identify bad features easily.

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        predictions = []
        for row in X:
            label = self.closest(row)  # only this row has been changed
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = self.euc(row, self.X[0])
        best_index = 0
        for i in range(len(self.X)):
            dist = self.euc(row, self.X[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y[best_index]

    def euc(self, a, b):
        return distance.euclidean(a, b)


# classifier = RandomCLF()
classifier = CustomKNN()


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    clf = classifier
    clf.fit(X_train, y_train)
    clf_predictions = clf.predict(X_test)
    print(accuracy_score(y_test, clf_predictions))
