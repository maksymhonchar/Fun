# Q1
- Show by simple mathematical argument that finding the optimal decision trees for a classification problem among all the possible tree structures, can be an exponentially hard problem. (Hint: How many trees are there in the jungle anyway?)

## general Q1
- Decision tree is a type of supervised learning algorithm that can be used in both regression and classification problems. It works for both categorical and continuous input and output variables.
- The depth of a decision tree is the length of the longest path from a root to a leaf.
    - length of binary tree: 2^(d+1) - 1, d=depth
- To control tree complexity - work on these parameters:
    - MaxNumSplits
    - MinLeafSize
    - MinParentSize

## answer Q1
- Because the length of CART is 2^(d+1)-1 this actually is exponentially hard problem

# Q2 
- Both decision trees and deep neural networks are non-linear classifier i.e. they separates the space by complicated decision boundary. Why, then, it is so much easier for us to intuitively follow a decision tree model vs. a deep neural network?

## general Q2
- btw, if data is structured well, well-tuned CARTs will in most cases outperform NNs.

## answer Q2
- Because it is good in visualization
- There are still a lot of problems about how to visualize NN results / intermediate results.
- Hyperparameters tuning is well understandable (unlike layers in most DNNs)

# Q3
- Back-propagation is the workhorse of deep learning. Name a few possible alternative techniques to train a neural network without using back-propagation. (Hint: Random search…)

## general Q3
- First of - You can use pretty much any numerical optimization algorithm to optimize weights of a neural network.
- there are genetic algorithms that also solve optimization problems
- there is random-start hill climbing algorithm to find an OK solution quickle (but it wouldn't be feasible to find a near optimal solution)
- For problems where finding the precise global optimum is less important than finding an acceptable local optimum in a fixed amount of time, simulated annealing may be preferable to alternatives such as gradient descent.
- Recent work shows that with random initialization, gradient descent converges to a local minimum (rather than a saddle point).
- There is also Random Sampling Back Propagation
    https://towardsdatascience.com/two-weird-ways-to-regularize-your-neural-network-manual-back-propagation-in-tensorflow-f4e63c41bd95
- Actually injecting noise during the training process of deep neural network results in better generalization power

## answer Q3
- ? todo: dive into optimization algorithms for NNs

# Q4
- Let’s say you have two problems — a linear regression and a logistic regression (classification). Which one of them is more likely to be benefited from a newly discovered super-fast large matrix multiplication algorithm? Why? (Hint: Which one is more likely to use a matrix manipulation?)

## general Q4
- In linear regression:
    - basic form: no matrix multiplications, only work with mean of X or y values (b=Y^-mX^)
- In linear regression the Maximize Likelihood Estimation (MLE) solution for estimating x has the following closed form solution (assuming that A is a matrix with full column rank):
    > x^ = argmin||Ax-b||22 = (A^T * A)^-1 * A^T * b
    - this is read as "find the x that minimizes the objective function, ||Ax-b||22
- Alternatively, the MLE solution for estimating the coefficients in logistic regression is:
    > x^ = argmin( sum_i=1_N(log(1 + e^(-x^T*a(i)))) ) + (1-y(i)) * log(1+e^(x^T*a(i)))
- Conclusion: we can represent that notation in matrix form, but you don't gain anything from doing this. Logistic regression does not have a closed form solution and does not gain the same benefits as linear regression does by representing it in matrix notation.
- To solve for x^log estimation techniques such as gradient descent and the Newton-Raphson method are used

## answer:
- logistic regression

# Q5
- What is the impact of correlation among predictors on principal component analysis? How can you tackle it?

## general / answer Q5
- High correlation among predictors means you ca predict one variable using second predictor variable. This is called the problem of multicollinearity
- This results in unstable parameter estimates of regression which makes it very difficult to assess the effect of independent variables on dependent variables
    - It may not be an issue. The main impact of collinearity (the real issue here - not merely simple correlations between predictors) is loss of statistical power to detect effects or precisely estimate individual predictors.

- How to deal:
    - penalized likelihood function (Ridge Regression, Lasso).
    - Principal Component Analysis is also a common technique but if your predictor variables are measured in different scale you have to standardize the data matrix prior to the PCA.
- How to deal[2]:
    - Simple is to drop one such high correlated independent variable.
    - Else covert your data into Principal Component Scores (PCA-scores) and the perform Multiple regression.
- How to deal[3]:
    - The best remedy may be to increase sample size, but it could also be sensible to re-parameterize the model, transform data or use data reduction methods (depending on your goal).

- Adjoining nearly correlated variables increases the contribution of their common underlying factor to the PCA.
    - https://stats.stackexchange.com/questions/50537/should-one-remove-highly-correlated-variables-before-doing-pca
