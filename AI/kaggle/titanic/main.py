"""
Kaggle "Titanic" competition.
overview: https://www.kaggle.com/c/titanic/overview

Plan:
1. Load dataset from CSV
2. Create X and y vectors
    2.1. Remove excessive columns from train_dataset.
    2.2. Remove raws with containing NaN values.
        train_dataset.loc[:, 'Age'].isnull().values.any()
        train_dataset.loc[:, 'Age'].isnull().sum().sum()
    2.3. Replace string values with numerical ones.
    2.4. Separate X and y vectors from train_dataset CSV.
3. Create & fit classifier.
    3.1. Create train/test sets from train_dataset CSV.
    3.2. Create classifier.
    3.3. Fit classifier
4. Calculate score.
5. Plot some compelling data.


6. Load prediction to CSV ???

Results:
    todo: add results for different features/models/etc
    todo: linearSVC
    todo: GradientBoostingClassifier
    todo: some pictures maybe?
"""
# todo: feature engineering: add **2, multiplications etc
# todo: try all of the known algorithms, no deep learning
# todo: 'cabin' - maybe it is useful?

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. Load dataset from CSV
train_dataset = pd.read_csv('/home/max/Documents/learn/learnai/kaggle/titanic/data/train.csv')
test_dataset = pd.read_csv('/home/max/Documents/learn/learnai/kaggle/titanic/data/test.csv')

# 2. Create X and y vectors
# 2.1. Remove excessive columns from train_dataset
excessive_columns_names = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked']
train_dataset = train_dataset.drop(labels=excessive_columns_names, axis='columns')
# 2.2. Remove raws with containing NaN values.
train_dataset = train_dataset.dropna()
# 2.3. Replace string values with numerical ones.
train_dataset = train_dataset.replace('male', 0)
train_dataset = train_dataset.replace('female', 1)
# 2.4. Separate X and y vectors from train_dataset CSV.
target_column_name = 'Survived'
train_dataset_X = train_dataset.loc[:, train_dataset.columns != target_column_name]
train_dataset_y = train_dataset.loc[:, target_column_name]

# 3. Create & fit classifier.
# 3.1. Create train/test sets from train_dataset CSV.
X_train, X_test, y_train, y_test = train_test_split(
    train_dataset_X, train_dataset_y,
    random_state=42
)


def logreg():
    # 3.2. Create classifier.
    # clf_C = 5
    clf = LogisticRegression(solver='lbfgs', C=0.1)
    clf1 = LogisticRegression(solver='lbfgs', C=1)
    clf100 = LogisticRegression(solver='lbfgs', C=100)
    # 3.3. Fit classifier.
    clf.fit(X_train, y_train)
    clf1.fit(X_train, y_train)
    clf100.fit(X_train, y_train)

    # 4. Calculate score.
    print("Train set score: {0}".format(clf.score(X_train, y_train)))  # 0.822429906542056
    print("Test set score: {0}".format(clf.score(X_test, y_test)))  # 0.7877094972067039

    # 5. Plot some compelling data.
    # Plot coefficients learned by the models with the 3 different settings of C: 1 100 0.01
    plt.plot(clf.coef_.T, 'o', label="C=0.1")
    plt.plot(clf1.coef_.T, '^', label="C=1")
    plt.plot(clf100.coef_.T, 'v', label="C=100")
    plt.xticks(range(train_dataset_X.shape[1]), train_dataset_X.columns.values, rotation=90)
    plt.hlines(0, 0, train_dataset_X.shape[1])
    plt.ylim(-5, 5)
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.legend()
    plt.show()


def linsvc():
    pass


if __name__ == '__main__':
    # logreg()
    print('end of the program')
