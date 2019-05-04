src: https://towardsdatascience.com/https-medium-com-pupalerushikesh-svm-f4b42800e989

# Introduction
- SVM - Support Vector Machine
- Support Vector Machine is a linear model for classification and regression problems. It can solve linear and non-linear problems and work well for many practical problems. The idea of SVM is simple: The algorithm creates a line or a hyperplane which separates the data into classes.
- At first approximation what SVMs do is to find a separating line(or hyperplane) between data of two classes.
- SVM is an algorithm that takes the data as an input and outputs a line that separates those classes if possible.

# SVM’s way to find the best line
- According to the SVM algorithm we find the points closest to the line from both the classes. These points are called **support vectors**.
Now, we compute the distance between the line and the support vectors. This distance is called the margin. Our goal is to maximize the margin.
The hyperplane for which the margin is maximum is the optimal hyperplane.
- Thus SVM tries to make a decision boundary in such a way that the separation between the two classes(that street) is as wide as possible.

# Hyperplane
- A hyperplane in an n-dimensional Euclidean space is a flat, n-1 dimensional subset of that space that divides the space into two disconnected parts.
- For example let’s assume a line to be our one dimensional Euclidean space(i.e. let’s say our datasets lie on a line). Now pick a point on the line, this point divides the line into two parts. The line has 1 dimension, while the point has 0 dimensions. So a point is a hyperplane of the line.
- For two dimensions we saw that the separating line was the hyperplane. Similarly, for three dimensions a plane with two dimensions divides the 3d space into two parts and thus act as a hyperplane. Thus for a space of n dimensions we have a hyperplane of n-1 dimensions separating it into two parts.

# Tuning parameters
- Parameter C
It controls the trade off between smooth decision boundary and classifying training points correctly. A large value of c means you will get more training points correctly.
Large value of c means you will get more intricate decision curves trying to fit in all the points. Figuring out how much you want to have a smooth decision boundary vs one that gets things correct is part of artistry of machine learning. So try different values of c for your dataset to get the perfectly balanced curve and avoid over fitting.
- Parameter Gamma:
It defines how far the influence of a single training example reaches. If it has a low value it means that every point has a far reach and conversely high value of gamma means that every point has close reach.
If gamma has a very high value, then the decision boundary is just going to be dependent upon the points that are very close to the line which effectively results in ignoring some of the points that are very far from the decision boundary.
On the other hand, if the gamma value is low even the far away points get considerable weight and we get a more linear curve.
