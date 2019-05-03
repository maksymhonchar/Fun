from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=42
)

svm = SVC(C=100, gamma='auto')
svm.fit(X_train, y_train)

print("Training set accuracy: {:.3f}".format(svm.score(X_train, y_train)))  # 1.000
print("Test set accuracy: {:.3f}".format(svm.score(X_test, y_test)))  # 0.622

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)

print("Training set accuracy: {:.3f}".format(svm.score(X_train_scaled, y_train)))  # 0.981
print("Test set accuracy: {:.3f}".format(svm.score(X_test_scaled, y_test)))  # 0.979

svm_gamma_scaled = SVC(C=100, gamma='scale')
svm_gamma_scaled.fit(X_train, y_train)

print("Training set accuracy: {:.3f}".format(svm_gamma_scaled.score(X_train, y_train)))  # 0.998
print("Test set accuracy: {:.3f}".format(svm_gamma_scaled.score(X_test, y_test)))  # 0.937

# Preprocessing using zero mean and unit variance scaling
std_scaler = StandardScaler()
std_scaler.fit(X_train)

X_train_scaled = std_scaler.transform(X_train)
X_test_scaled = std_scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)

print("Training set accuracy: {:.3f}".format(svm.score(X_train_scaled, y_train)))  # 1.000
print("Test set accuracy: {:.3f}".format(svm.score(X_test_scaled, y_test)))  # 0.944
