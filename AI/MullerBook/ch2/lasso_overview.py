# An alternative to Ridge for regularizing linear regression.
import mglearn

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split


X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lasso = Lasso()
lasso.fit(X_train, y_train)

print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {0}".format(np.sum(lasso.coef_ != 0)))

print('---\n{0}'.format(lasso.coef_))

#####################

lasso001 = Lasso(alpha=0.01, max_iter=100000)
lasso001.fit(X_train, y_train)

print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("# of features used in Lasso: {0}".format(np.sum(lasso001.coef_ != 0)))

#####################

lasso0001 = Lasso(alpha=0.0001, max_iter=100000)
lasso0001.fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso0001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso0001.score(X_test, y_test)))
print("# of features used in Lasso: {0}".format(np.sum(lasso0001.coef_ != 0)))

### Plotting ###
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)

plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
plt.plot(lasso0001.coef_, 'v', label="Lasso alpha=0.0001")
plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.show()
