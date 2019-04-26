import matplotlib.pyplot as plt

import mglearn

from sklearn.svm import SVC


X, y = mglearn.tools.make_handcrafted_dataset()

svm_gamma = 0.1  # controls the width of the Gaussian kernel - scale of points to be close together.
svm_C = 10  # regularization parameter - limits the importance of each point (their dual_coef_)
svm = SVC(kernel='rbf', C=svm_C, gamma=svm_gamma)
svm.fit(X, y)

mglearn.plots.plot_2d_separator(svm, X, eps=0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

sv = svm.support_vectors_  # get support vectors
sv_labels = svm.dual_coef_.ravel() > 0  # class labels of support vectors are given by the sign of the dual coefficients.
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)

plt.xlabel('f0')
plt.ylabel('f1')
plt.show()

###############################

# try out different parameters

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for ax, C in zip(axes, [-1, 0, 3]):
    for a, gamma in zip(ax, range(-1, 2)):
        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
axes[0, 0].legend(
    ['class 0', 'class 1', 'sv class 0', 'sv class 1'],
    ncol=4, loc=(1, 1.2)
)
plt.show()
