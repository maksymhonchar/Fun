# This dataset could be used to illustrate regression algorithms

import mglearn

import matplotlib.pyplot as plt

X, y = mglearn.datasets.make_wave(n_samples=100)

plt.plot(X, y, 'o')
plt.ylim(-30, 30)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()