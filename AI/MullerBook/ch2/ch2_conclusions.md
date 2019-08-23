# Some useful terms
1. generalization - learning a model that is able to perform well on new, previously unseen data
2. underfitting - describes a model that cannot capture the variations present in the training data
3. overfitting - describes a model that focuses too much on the training data and is not able to generalize to new data very well
4. regularization - explicitly restricting the model to avoid overfitting

# Quick summary of when to use each model from Ch1:
- Nearest neighbors - small datasets, good as a baseline, easy to explain
- Linear models - go-to as a first algorithm to try, good for very large datasets, good for very high-dimensional data
- Naive Bayes - only for classification; even faster than linear models, good for every large datasets and highdimensional data; often less accurate than linear models.
- Decision trees - very fast, don't need scaling of the data; can be visualized and easily explained.
- random forests - nearly always perform better than a single decision tree, very robust and powerful; don't need scaling of data; not good for very high-dimensional sparse data;
- gradient boosted decision trees - often slightly more accurate than random forests; slower to train but faster to predict than random forests; smaller in memory; need more parameter tuning than random forests
- SVMs - powerful for medium-sized datasets of features with similar meaning; require scaling of data, sensitive to parameters
- Neural networks - can build very complex models, particularly for large datasets; sensitive to scaling of the data and to the choice of parameters; large models need a long time to train.

# Some tips
1. When working with a new dataset, it is in general a good idea to start with a simple
model, such as a linear model or a naive Bayes or nearest neighbors classifier, and see
how far you can get.
2. score on training set is lower for new algo; score on test set is higher for new algo => overfitting
3. With enough training data, regularization becomes LESS IMPORTANT, and given enough data, ridge and linear regression will have the same performance.
4. Logistic Regression applies L2 regularization by default. In sklearn, use parameter "penalty='l1' to switch regularization for LogReg.
5. Parameters C and alpha for LogReg/LinReg are searched for on a logarithmic scale.

# To note
1. Ridge - L2 regularization, Lasso - L1 regularization.

# SWP - strenghts, weaknesses, parameters
1. kNN
        - Params:
                - # of neighbors
                        - 3-5 neighbors should be fine
                - how we measure distance between data points
                        - Euclidean
        - Strengths:
                - Very easy to understand
                - Often gives reasonable performance without a lot of adjustments       
                - a good baseline method
        - Weaknesses
                - for very large datasets (either for # of features or # of samples),
                prediction could be slow
                - Doesn't perform well with many feateres (~ >=100)
                - works bad with sparse datasets (most features = 0)
2. Linear models
        - Strengths:
                - are very fast to train
                - are very fast to predict
                - Scale well to very large datasets
                        - solver='sag'
                        - stochastic - SGDClassifier/SGDRegressor
                - Work well with sparse data
                - Linear models perform well when the **number of features is large compared to the number of samples**.
        - Weaknesses
                - it is often not entirely clear why coefficients are the way they are
                - In lower-dimensional spaces, other models might yield better generalization performance.
        2.1 Ridge regression
                - Params:
                        -Alpha - regularization parameter
                                - increasing alpha - force coeffs toward zero
                                - decreasing alpha - coeffs are less restricted
                - Strengths:
                        - In practice, Ridge Regression is the first choice for ridge vs lasso
        2.2. Lasso regression
                - Params:
                        - Alpha - regularization parameter
                                - increasing alpha - remove the effect of regularization, LOTS of
                                params will be 0
                                - decreasing alpha - more complex model, less params will be close
                                to 0 => unregularized model.
                - Strengths:
                        - We have a model that is easy to interpret -> we select only a subset of
                        the input featres.
        2.3. ElasticNet - combines the penalties of Lasso an dRidge.
                - Weaknesses:
                        - we have to setup parameters for L1 and L2 regularization 
        2.4. LogisticRegression
                - Params:
                        - C - regularization parameter
                                - C=1 default
                                - Increasing C - more complex model (should increase training set
                                accuracy and slightly increase test set accuracy) (params are less
                                towards 0)
                                - Decreasing C - more regularized model (params are more towards 0)
3. Naive Bayes classifiers
        - GaussianNB - **VERY high-dimensional data**
                - MultinomialNB , BernoulliNB - sparse count data (i.e. text)
        - Strenghts:
                - Are even faster in training than linear models
                - Fast to train and to predict
                - Work with high-dimensional data
                - Might worth to use on very large datasets, where training even a linear model might take too long.
        - Parameters:
                - alpha - controls model complexity
                        - alpha "smoothes" the statistics - the algorithm adds to the data "alpha" many virtual data points that have positive values for all the features.
                - however, setting alpha is not critical for good performance.
4. Decision trees
        - Parameters:
                - pre-pruning parameters that stop the building of the tree before it is fully developed
                - pre-pruning strategies:
                        1. setting max_depth and max_leaf_nodes
                        2. setting min_samples_leaf
        - Strengths:
                - the resulting model can easily be visualized and understood by nonexperts (for smaller trees)
                - algorithms are completely invariant to scaling of the data
                        - == possible splits of the data don't depend on scaling
                        - == no preprocessing (normalization / standardization of features) needed
                - Decision Trees work WELL when you have features that are on **completely different scales** or a **mix of binary and continuous features**
        - Weaknesses:
                - even with the use of pre-pruning they tend to overfit and provide poor generalization performance.
                        - Therefore, ensemble methods are used in place of a single decision tree.
5. Random forests
        - Parameters:
                - n_estimators - lagrer is always better
                        - averaging more trees -> more robust ensemble by reducing overfitting
                - max_features
                        - determines how random each tree is.
                        - smaller max_features REDUCES overfitting
                        - good rule of thumb:
                                - classification: max_features=sqrt(n_features)
                                - regression: max_features=log_2(n_features)
                - max_depth
                - max_leaf_nodes
        - Strengths:
                - Often work well without heavy tuning of the parameters
                - Don't require scaling of the data.
                - work well even on very large datasets.
        - Weaknesses:
                - impossible to interpret tens or hundreds of trees in detail
                - time/cpu/memory consuming algorithm
                - don't perform well on very high dimensional, sparse data (i.e. text data)
6. Gradient boosted regression trees
        - note: could be used for both classification and regression
        - Strengths:
                - works well without scaling
                - Works well on a mixture of binary and continuous features.
        - Weaknesses
                - main drawback - they require careful tuning of the parameters
                - may take a long time to train
                - often doesn't work well on high-dimensional sparse data
        - Parameters:
                - learning_rate - how strongly each tree tries to correct the mistakes of the previous trees.
                        - higher learning_rate - increases model complexity -> model has more chances to correct mistakes on the training set.
                        - lower learning_rate - reduce overfitting
                - n_estimators
                        - more estimators -> more complex model
                - max_depth / max_leaf_nodes
                        - limiting max_depth - reduce overfitting
                        - usually max_depth is set very low for gradient boosted models, often **not deeper** than 5 splits.
                - NOTE: lowering learning_rate -> increase n_estimators
                - NOTE: good practice - fit n_estimators depending on the time and memory budget => search over different learning_rate-s parameters
7. Kernelized SVMs
        - Parameters
                - C - regularization parameter
                        - limits the importance of each point
                        - Small C - a very restricted model
                        - large C - more complex model
                - Choice of the kernel
                        - gamma (gaussian kernel - rbf kernel)
                                - small gamma - a large radius for the Gaussian kernel -> many points are considered close by -> low complexity
                                - large gamme - complex model
                - NOTE: C and gamma should be adjusted together! these are correlated with each other.
        - Strengths:
                - allow for complex decision boundarise, even if the data has only a few featuress.
                - work well on low-dimensional and high-dimensional data
                - If all of your features represent measurements in similar units (i.e. pixel intensities) and they are on similar scales - worth to try SVMs!
        - Weaknesses:
                - don't scale very wewll with the number of samples
                - 10k samples - might work well; 100k samples - challenging in terms of runtime and memory usage.
                - require careful preprocessing of the data and tuning of the parameters
                - hard to inspect - it can be difficult to understand why a particular prediction was made and it might be tricky to explain the model to a nonexpert
8. MLPs (from sklearn)
        - Strengths:
                - main advantage - they are able to capture information contained in large amounts of data and build incredibly complex models.
                - work best with "homogeneous data" == all the features have similar meanings.
        - Weaknesses:
                - large NNs take a long time to train
                - require careful preprorcessing of the data