# Ridge regression is one of the most commonly used alternatives to standard linear regression.

import mglearn

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression


X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

ridge = Ridge()
ridge.fit(X_train, y_train)

print('Training set score: {:.2f}'.format(ridge.score(X_train, y_train)))
print('Testing set score: {:.2f}'.format(ridge.score(X_test, y_test)))

# Change alpha - increasing alpha forces coefficients to move more towards zero, which decreases training set performance but might help generalization.
def ridge_set_alpha(alpha_value=1.0):
    ridge_alpha = Ridge(alpha=alpha_value)
    ridge_alpha.fit(X_train, y_train)
    print("-\nAlpha:{0}".format(alpha_value))
    print("Training set score: {:.2f}".format(ridge_alpha.score(X_train, y_train)))
    print("Testing set score: {:.2f}".format(ridge_alpha.score(X_test, y_test)))
    print("-")
ridge_set_alpha(20)
ridge_set_alpha(10)
ridge_set_alpha(0.5)
ridge_set_alpha(0.1)
ridge_set_alpha(0.01)
ridge_set_alpha(0.0001)

# plot coef_-s for different alpha & LR algorithm
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
lr = LinearRegression().fit(X_train, y_train)

plt.plot(ridge.coef_, 's', label='Ridge alpha=1')
plt.plot(ridge10.coef_, '^', label='Ridge alpha=10')
plt.plot(ridge01.coef_, 'v', label='Ridge alpha=0.1')

plt.plot(lr.coef_, 'o', label='LR')

plt.xlabel('coefficient index')
plt.ylabel('coefficient magnitude')

plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)

plt.legend()

plt.show()

mglearn.plots.plot_ridge_n_samples()

plt.show()
