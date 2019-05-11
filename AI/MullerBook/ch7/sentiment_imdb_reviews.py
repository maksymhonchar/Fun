# dataset src: http://ai.stanford.edu/~amaas/data/sentiment/.

import numpy as np

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression


# load_files returns a bunch, containing training texts and training labels.
reviews_train = load_files('muller_book/ch7_text_data/aclImdb/train/')

text_train, y_train = reviews_train.data, reviews_train.target

# Remove HTML formatting before proceeding
text_train = [doc.replace(b"<br />", b" ") for doc in text_train]

# print(type(text_train), type(y_train))  # list, numpy.ndarray
# print(len(text_train), len(y_train))  # 75000, 75000
# print(text_train[1], y_train[1])  # "Amount of disappointment..." 2

# The dataset was collected such that the positive class and the negative class balanced, so that there are as many positive as negative strings
# print('Samples per class (training): {0}'.format(np.bincount(y_train)))  # [12500 12500 50000]

reviews_test = load_files('muller_book/ch7_text_data/aclImdb/test/')

text_test, y_test = reviews_test.data, reviews_test.target

text_test = [doc.replace(b"<br />", b" ") for doc in text_test]

# print(len(text_test), len(y_test))  # 25000, 25000
# print(text_test[1], y_test[1])  # "I don't know how this movie has..." 0

# print('Samples per class (test): {0}'.format(np.bincount(y_test)))  # [12500 12500]

# Task: given a review, assign the label 'positive' or 'negative' based on the text content in the review - binary classification task.

# Solution 1: bag of words
def bagofwords_solution():
    vect = CountVectorizer()
    vect.fit(text_train)
    
    X_train = vect.transform(text_train)
    print('X_train:\n{0}'.format(repr(X_train)))
    # X_train: <75000x124255 sparse matrix of type '<class 'numpy.int64'>'
    #   with 10315542 stored elements in Compressed Sparse Row format>

    # Another way to access the vocabulary is using the get_feature_name method of the
    # vectorizer, which returns a convenient list where each entry corresponds to one feature:
    feature_names = vect.get_feature_names()
    print("Number of features: {}".format(len(feature_names)))  # 124255
    print("First 20 features:\n{}".format(feature_names[:20]))  # all are numbers
    print("Features 20010 to 20030:\n{}".format(feature_names[20010:20030]))
    print("Every 2000th feature:\n{}".format(feature_names[::2000]))

    # Measure performance by building a classifier - LogisticRegression
    print('start estimating cross validation score')
    scores = cross_val_score(
        LogisticRegression(),
        X_train, y_train,
        cv=5
    )
    print('Mean cross-validation accuracy: {:.3f}'.format(np.mean(scores)))  # 

    print('start searching for optimal C parameter for LR')
    # Try to search optimal C parameter
    param_grid = { 'C': [0.001, 0.01, 0.1, 1, 10] }
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print('Best cross-validation score: {:.3f}'.format(grid.best_score_))
    print('Best parameters: {:.3f}'.format(grid.best_params_))  # c:0.1 accuracy 0.89

    X_test = vect.transform(text_test)
    print("{:.2f}".format(grid.score(X_test, y_test)))  # 0.88

    # Improve extraction of words
    vect = CountVectorizer(min_df=5).fit(text_train)
    X_train = vect.transform(text_train)
    print('X_train with min_df: {0}'.format(repr(X_train)))

    feature_names = vect.get_feature_names()

    print("First 50 features:\n{}".format(feature_names[:50]))
    print("Features 20010 to 20030:\n{}".format(feature_names[20010:20030]))
    print("Every 700th feature:\n{}".format(feature_names[::700]))

    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("Best cross-validation score: {:.2f}".format(grid.best_score_))  # 0.89


bagofwords_solution()
