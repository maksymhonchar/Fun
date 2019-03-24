src: https://medium.com/data-science-group-iitr/support-vector-machines-svm-unraveled-e0e7e3ccd49b

# Introduction
- Support Vector Machine is a supervised machine learning algorithm which can be used for both classification and regression problems.
- SVM is a generalization of a simple classifier called maximum margin classifier.
- Best thing I like about SVM is that it’s completely based on Mathematical Optimization problem.

# Mathematics
- The goal of SVM is to identify an optimal separating hyperplane which maximizes the margin between different classes of the training data.
- Hyperplane: It is basically a generalization of plane.
    - in one dimension, an hyperplane is called a point
    - in two dimensions, it is a line
    - in three dimensions, it is a plane
    - in more dimensions you can call it an hyperplane
- Margin could be defined in two ways:
    - Functional Margin: Let’s focus on binary classification {-1, 1} problem for now. We’ll write our classifier as func_margin_pic1.png
    Now functional margin of an ith observation is defined by func_margin_pic2.png
    Now functional margin of classifier is smallest of the functional margins of individuals training examples, func_margin_pic3.png
    - Geometric margin
    Defined by geometric_margin_pic1.png
- Optimal Separating Hyperplane: Idea is to choose the hyperplane which is at maximize margin from both the classes of training data. Maximizing the distance between the nearest points of each class (minimum of functional margins of all the examples) and the hyperplane would result in an optimal separating hyperplane, optimal_separating_hyperplane.png.
- Support Vectors: Support Vectors are simply the co-ordinates of data points which are nearest to the optimal separating hyperplane.

# How to find this optimal separating Hyperplane?
- So, basically what we want to do is finding_hyperplane_pic1.png
- But now, for us we can set functional margin to 1. Because it’s nothing but a scaling which should not affect optimization problem. So, this maximization problem converges to minimization problem. Which is, as shown on finding_hyperplane_pic2.png

# What if Data is not Linearly separable?
- In the case of non-linearly separable data points SVM uses the kernel trick. The idea stems from the fact that if the data cannot be partitioned by a linear boundary in its current dimension, then projecting the data into a higher dimensional space may make it linearly separable.
Shown in kernel_trick.png file.
- Type of kernels are shown in kernels_types_pic1.png
- From my experience, RBF is the most popular kernel choice in SVM.

# SVM for multiclass Classification
- How to extend it for multiclass classification problem is still an ongoing research.
- Mainly there are two methods for Multiclass classification in SVM.
    1. One-against-one method: This method constructs k(k − 1)/2 classifiers where each one is trained on data from two classes.
    2. One-against-all method: It constructs k SVM models where k is the number of classes. The mth SVM is trained with all of the examples in the mth class with positive labels, and all other examples with negative labels.

# Pros
- It is really effective in higher dimension. If you have more features than training examples, most of the algorithms perform very bad, but SVM is the only algorithm which can saves you in this situation.
- Best algorithm if you data are separable. That two classes are not mixed.
- Only support vectors affect the optimally spaced hyperplane. So, it is less affected by outliers.

# Cons
- On large dataset it takes too much time. Mainly because of kernel function calculations and finding optimal hyperplane in higher dimensions.
- Can not perform well in case of overlapping classes.
- Can only give you 0–1 classification. Probably estimates computation are really expensive.

# Algorithm Parameters
- **kernel**: ‘linear’, ‘polynomial’ or ‘rbf’. Type of kernel you want to use
- **C**: penalty on data point on the opposite side of separating Hyperplane.
- **gamma**: Kernel parameter. Not applicable for linear kernel though.
- **degree**: In case of polynomial kernel you can specify degree of polynomial you want to use.

- SVM has very less parameter to tune on. And generally it works best with default values.