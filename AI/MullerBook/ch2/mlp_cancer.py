import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


cancer = load_breast_cancer()

print("Cancer data per-feature maxima:\n{0}".format(cancer.data.max(axis=0)))

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=42
)

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

print(mlp.score(X_train, y_train))  # 0.903
print(mlp.score(X_test, y_test))  # 0.93

mean_on_train = X_train.mean(axis=0)  # mean value per feature on the training set.
std_on_train = X_train.std(axis=0)  # standard deviation of each feature on the training set.

# Subtract the mean, and scale by inverse std; afterward, mean=0 and std=1
# Scale data for train and test sets.
X_train_scaled = (X_train - mean_on_train) / std_on_train
X_test_scaled = (X_test - mean_on_train) / std_on_train

mlp_scaled_dataset = MLPClassifier(random_state=42, max_iter=10000)
mlp_scaled_dataset.fit(X_train_scaled, y_train)

print(mlp_scaled_dataset.score(X_train_scaled, y_train))  # 1.0
print(mlp_scaled_dataset.score(X_test_scaled, y_test))  # 0.972

# Analyze what a neural network has learned: weights in the model
plt.figure(figsize=(20, 5))
plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()
plt.show()
