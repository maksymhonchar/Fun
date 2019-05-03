import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=42
)

print(X_train.shape, X_test.shape)  # (426, 30) (143, 30)

minmaxsclr = MinMaxScaler()
minmaxsclr.fit(X_train)  # Compute min and max value of each feature on the training set.

print(minmaxsclr)  # feature_range=(0,1)

X_train_scaled = minmaxsclr.transform(X_train)  # Apply the transformation - scale the training data

print("transformed shape: {}".format(X_train_scaled.shape))  # (426, 30)
print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))
print("per-feature minimum after scaling:\n {}".format(X_train_scaled.min(axis=0)))  # zeros
print("per-feature maximum after scaling:\n {}".format(X_train_scaled.max(axis=0)))  # ones

X_test_scaled = minmaxsclr.transform(X_test)

# Not only zeros and ones: subtraction min values and divide by the training set range.
print("per-feature minimum after scaling:\n {}".format(X_test_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n {}".format(X_test_scaled.max(axis=0)))

def scale_train_test_sets_same_way():
    # It is important to apply exactly the same transformation to the training set and teh test set the supervised model to work on the test set.
    X, _ = make_blobs(n_samples=100, centers=5, random_state=42, cluster_std=2)
    X_train, X_test = train_test_split(X, random_state=42, test_size=.1)
    
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].scatter(X_train[:, 0], X_train[:, 1], label="Training set", s=60)
    axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^', label="Test set", s=60)
    axes[0].legend(loc='upper left')
    axes[0].set_title("Original Data")

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    # X_train_scaled = scaler.fit_transform(X_train)  # shortcut
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], label="Training set", s=60)
    axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker='^', label="Test set", s=60)
    axes[1].set_title("Scaled Data")

    # Rescale the test set separately so teste set min is 0 and test set max is 1 - mistake
    # DON'T DO THIS! The next pic is for illustration purposes only
    test_scaler = MinMaxScaler()
    test_scaler.fit(X_test)
    X_test_scaled_badly = test_scaler.transform(X_test)

    axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], label="training set", s=60)
    axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1], marker='^', label="test set", s=60)
    axes[2].set_title("Improperly Scaled Data")

    for ax in axes:
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")

    plt.show()

scale_train_test_sets_same_way()
