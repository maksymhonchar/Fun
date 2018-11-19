import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def first_ml_problem():
    # First 2 columns.  "Input".
    # [0]:weight, g.  [1]:0 bumpy, 1 smooth.
    features = [[140,1], [130,1], [150,0], [170,0]]
    # Last column.  Output.
    # [0]:apple.  [1]:orange.
    labels = [0, 0, 1, 1]
    # Use the examples to train a classifier.
    # Here we use a decision tree.
    # Classifier == "box of rules".
    clf = tree.DecisionTreeClassifier()
    # "fit" == "find patterns in data".
    clf = clf.fit(features, labels)
    print(clf.predict([[150, 0]]))
    print(clf.predict([[120, 0]]))

#first_ml_problem() # [1](orange) [0](apple)

def visualize_decision_tree():
    iris = load_iris()

    # Metadata of dataset:
    #print(iris.feature_names) # [sl sw pl pw]: length and width of sepal and petal.
    #print(iris.target_names) # [setosa, versicolor, virginica]
    
    # Real values of dataset:
    #print(iris.data[0]) # Measurements from [0] flower: [5.1 3.5 1.4 0.2]
    #print(iris.target[0]) # Labels.  Ret: 0 -> satosa
    
    # Create testing dataset to check the trained model on them later.
    test_idx = [0, 50, 100]  # first satosa, first versicolor, first virginica
    # Training data
    train_target = np.delete(iris.target, test_idx)
    train_data = np.delete(iris.data, test_idx, axis=0)
    # Testing data - keep them separate from training data.
    test_target = iris.target[test_idx]
    test_data = iris.data[test_idx]

    # Create a classifier
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data, train_target)

    print(test_target)  # should be the same
    print(clf.predict(test_data))  # should be the same

    # Visualize the tree
    # viz code
    dot_data = StringIO()
    tree.export_graphviz(clf,
        out_file=dot_data,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True,
        impurity=False)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('iris.pdf')

    print(test_data[2], test_target[2])
    print(iris.feature_names, iris.target_names)
    
#visualize_decision_tree()

def dogs_height_distribution():
    # Create a population of 1000 dogs: 50%/50% greyhounds and labradors.
    greyhounds = 500
    labradors = 500 
    # Give each of them a height.
    grey_height = 28 + 4 + np.random.randn(greyhounds)
    lab_height = 28 + 4 + np.random.randn(labradors)

    # Display histogram of these heights.
    plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
    plt.show()

    print('Choice of the feature "dog height" is pretty bad!')
    print('Independent features are best!')
    print('Remove highly correlated features in your dataset!')
    print('Features have to be easy to understand!')
    print('Ideal features are: Informative, Independent, Simple!')

#dogs_height_distribution()


def supervised_learning_pipeline():
    # Spam classifier: spam/not spam.
    # Before production, we have a question: how accurate will it be after
    #   algorithm will encounter data, which is not in training set?
    # What we really want - verify our model is working well before we deploy them.
    
    # 1 approach: partition dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    # Partition dataset into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)
    # Create a classifier.
    # Classifier 1 - decision tree
    clf_tree = tree.DecisionTreeClassifier()
    clf_tree.fit(X_train, y_train)
    clf_tree_predictions = clf_tree.predict(X_test)
    print(clf_tree_predictions)
    print(accuracy_score(y_test, clf_tree_predictions))
    # Try a different classifier to accomplish the same task.
    clf_knb = KNeighborsClassifier()
    clf_knb.fit(X_train, y_train)
    clf_knb_predictions = clf_knb.predict(X_test)
    print(clf_knb_predictions)
    print(accuracy_score(y_test, clf_knb_predictions))


supervised_learning_pipeline()
