# MLPs can be viewed as generalizations of linear models that perform multiple stages of processing to come to a decision.
# y=weigh_sum(X)+b

import matplotlib.pyplot as plt

import mglearn

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


# Display nonlinear functions tanh and relu
# line = np.linspace(-3, 3, 100)
# plt.plot(line, np.tanh(line), label='tanh')
# plt.plot(line, np.maximum(line, 0), label='relu')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('relu(x), tanh(x)')
# plt.show()

# Tuning neural networks
X, y = make_moons(n_samples=1000, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)


def mlp_n_layers(n_layers=100):
    mlp = MLPClassifier(solver='lbfgs', random_state=42, hidden_layer_sizes=[n_layers])

    mlp = MLPClassifier(solver='lbfgs', random_state=42, hidden_layer_sizes=[n_layers, 10], activation='tanh')

    mlp.fit(X_train, y_train)

    print(mlp.score(X_train, y_train))
    print(mlp.score(X_test, y_test))

    mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.show()

mlp_n_layers()
mlp_n_layers(10)  # more ragged decision boundary.

def view_complexity_l2_alpha():
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax_x, n_hidden_nodes in zip(axes, [10, 100]):
        for ax, alpha in zip(ax_x, [0.0001, 0.01, 0.1, 1]):
            mlp = MLPClassifier(
                solver='lbfgs', random_state=42, 
                hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes], 
                alpha=alpha
            )
            mlp.fit(X_train, y_train)
            mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3, ax=ax)
            mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
            ax.set_title("n_hidden=[{}, {}]\nalpha={:.4f}".format(
            n_hidden_nodes, n_hidden_nodes, alpha))


view_complexity_l2_alpha()
plt.show()
