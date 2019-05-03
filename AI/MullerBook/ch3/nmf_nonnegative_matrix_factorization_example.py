# Non-negative matrix factorization is another unsupervised learning algorithm that
# aims to extract useful features. It works similarly to PCA and can also be used for
# dimensionality reduction.

# In NMF, we want the components and the coefficients to be nonnegative

import matplotlib.pyplot as plt

import mglearn

# 1 component: component toward the mean, as pointing there best explains the data
# 2 components: show the algorithm choosing directions that point toward the extremes of the data
mglearn.plots.plot_nmf_illustration()
plt.show()
