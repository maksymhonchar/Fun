# SVMs are an extension to linearsvc that allows for more complex models that are not defined sumply by hyperplanes in the input space.

import matplotlib.pyplot as plt

import mglearn

import numpy as np

from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC

from mpl_toolkits.mplot3d import Axes3D, axes3d


# Example of two-class classification dataset in which classes are not linearly separable.

X, y = make_blobs(centers=4, random_state=42)
y = y % 2

# linear_svm = LinearSVC().fit(X, y)

# mglearn.plots.plot_2d_separator(linear_svm, X)

# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.xlabel("f0")
# plt.ylabel("f1")
# plt.show()

##############################################

# Expand the set of input features: add "feature1**2"
X_new = np.hstack( [X, X[:, 1:] ** 2] )

linear_svm_3d = LinearSVC(max_iter=10000000)
linear_svm_3d.fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

# Visualize data.
figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim=-26)

# Show linear decision boundary
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)

# Visualize figure in 3d
mask = y == 0  # plot first all the points with y==0, then all with y==1
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], 
    c='b', cmap=mglearn.cm2, s=60
)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2],
    c='r', marker='^', cmap=mglearn.cm2, s=60
)

ax.set_xlabel("f0")
ax.set_ylabel("f1")
ax.set_zlabel("f1 ** 2")
plt.show()

ZZ = YY ** 2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape),
    levels=[dec.min(), 0, dec.max()], cmap=mglearn.cm2, alpha=0.5
)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("f0")
plt.ylabel("f1")
plt.show()
