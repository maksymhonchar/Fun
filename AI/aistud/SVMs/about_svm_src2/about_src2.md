# src:
# International Journal for Research in Applied Science & Engineering Technology (IJRASET)
# https://www.ijraset.com/fileserve.php?FID=11852

# Comparison: kNN & SVM Algorithm

# kNN
- kNN is one of the simplest of classification algorithms available for supervised learning.
The idea is to search for closest match of the test data in feature space. We will look into it with below image (knn_pic1.png).
- Meaning of "K" - classification depends on k nearest neighbours.

# SVM
- In machine learning, support vector machines (SVMs, also support vector networks) analyse data used for classification and regression analysis.
Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other.
-  Due to the optimal nature of SVM, it is guaranteed that the separated data would be optimally separated.
- The image below (svm_pic1.png) which has two types of data, red and blue. In kNN, for a test data, we used to measure its distance to all the training samples and take the one with minimum distance. It takes plenty of time to measure all the distances and plenty of memory to store all the training-samples.
- Our main goal is to find a line that uniquely divides the data into two regions.
Such data which can be divided into two with a straight line (or hyperplanes in higher dimensions) is called **Linear Separable**.
In above image (svm_pic1.png), plenty of such lines are possible. Intuitively, the line should be passing as far as possible from all the points because there can be noise in the incoming data.
This data should not affect the classification accuracy. Hence, a farthest line will provide more immunity against noise. Hence, SVM finds a straight line (or hyperplane) with largest minimum distance to the training samples.
- We need training data to find this decision boundary. In the image above (svm_pic2.png), the training data are the shapes filled up with colour. This training data is support vector and the lines passing through them Support Planes.

# Comparison between kNN and SVM
- kNN classifies data based on the distance metric whereas SVM need a proper phase of training.
- Generally, kNN is used as multi-class classifiers whereas standard SVM separate binary data belonging to either of one class.
- For a multiclass SVM, One-vs-One and One-vs-All approach is used. In One-vs-one approach, we have to train n*(n-1)/2 SVMs: for each pair of classes, one SVM.  When it comes to One-vs-All approach, we have to train as many SVMs as there are classes of unlabelled data. As in the other approach, we give the unknown pattern to the system and the final result if given to the SVM with largest decision value. 
- Although, SVMs look more computationally intensive, once training of data is done, that model can be used to predict classes even when we come across new unlabelled data. However, in KNN, the distance metric is calculated each time we come across a set of new unlabelled data. Hence, in KNN we always have to define the distance metric.
-  SVMs have two major cases in which classes might be linearly separable or non -linearly separable. When the classes are non-linearly separable, we use kernel function such as Gaussian basis function or polynomials.
Hence, we only have to set the K parameter and select the distance metric suitable for classification in KNN whereas in SVMs we have to select the R parameter (Regularization term) and also the parameters for kernel if the classes are not linearly separable.
- When we talk about accuracy of both of the classifiers, SVMs usually have higher accuracy than KNN as shown. (knn_svm_accuracy.png).
While conducting the tests in visual studio after incorporating the libraries of OpenCV, the accuracy percentage for SVM was found to be 94% and for KNN it was 93%.

# Conclusion
- The results and observations show that SVMs are a more reliable more of classifiers.
- However, KNN is less computationally intensive than SVM.
-  Since, KNN is easy to implement, the classification of Multi-class data should be done with kNN.
- The algorithm that guarantees reliable detection in unpredictable situations depends upon the data. If the data points are heterogeneously distributed, both should work well. If data is homogenous to look at, one might be able to classify better by putting in a kernel into the SVM.
- For most practical problems, KNN is a bad choice because it scales badly - if there are a million labelled examples, it would take a long time (linear to the number of examples) to find K nearest neighbours.
