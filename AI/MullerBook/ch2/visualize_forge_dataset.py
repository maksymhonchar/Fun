# Create a scatter plot and visualize all of the data points in "forge" dataset.
# This dataset could be used to solve classification problem.

import mglearn

import matplotlib.pyplot as plt

# generate dataset
X, y = mglearn.datasets.make_forge()

# plot dataset
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=1)
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()

print("X.shape: {0}".format(X.shape))  # (26, 2): 26 data points, 2 features.
