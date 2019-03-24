from sklearn import svm

X = [ [0, 0], [1, 1] ]
y = [0, 1]

clf = svm.SVC(gamma='scale')

clf.fit(X, y)

# print(clf.fit(X, y))
"""
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
"""

prediction = clf.predict( [ [2.0, 2.0] ] )

# print(prediction)
"""
array([1])
"""

print(clf.support_vectors_)  # get support vectors
"""
[[0. 0.]
 [1. 1.]]
"""

print(clf.support_)  # get indices of support vectors
"""
[0 1]
"""

print(clf.n_support_)  # get number of support vectors for each class
"""
[1 1]
"""