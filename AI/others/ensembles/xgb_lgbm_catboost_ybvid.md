# src
# https://www.youtube.com/watch?v=V5158Oug4W8

# xgboost
- xgboost is basically the same gradient boosting, but with some nice heuristics
- objective: include partial derivatives of loss function (1st and 2nd)
- decision trees in xgboost don't minimize criterios: information gain or entropy
- so xgboost provides some nice heuristic for a loss function and it also provides some heuristic for regularization
- model complexity is controlled by some more hyperparameters: gamma (number of leaves) (reducing model complexity - high gamma), l2 regularization (each leaf has instances with it's weights and we sum up all squared weights) has lambda parameter.
- gamma arises in splitting criterion (in decision trees: GAIN)
- trees in xgboost are sort of regularized during training (regularize trees while process of growing).

# lightgmb
- another implementation of gradient boosting
- microsoft library
- while in xgboost we have a mixture of a lot of programming languages and it is really hard to adapt code for own purposes, Microsoft wrote all the implementation in c++ and therefore it is on average faster than xgboost (depends on the task of course)
- in light gmb there is a new way to calculate the instance weight
    - there is no instance weight in xgboost
- in lgbm there is gradient-based 1-side sampling
    - idea - define instances as important ones in terms of their gradients
    - if a gradient for an instance is large == the instance is important (instance influences loss function very much)
- in lgbm, they propose to actually calculate loss function based on some proportion of weights (which influences loss function very much)
- crucial point - we DON'T estimate gradient in respect to whole dataset, which is going to be slow. Instead, we subsample instances with high values of gradient = fast

# catboost
- if we have a lot of categorical variables, we go to catboost - it is specifically designed to work nicely with categorical variables
- catboost - library by Yandex
- technique - mean-target encoding
    - calculate average values of your target for some values of categorical variable
- step1: how many unique values for this categorical features we have: 4
    - if it <7 - we go for one-hot encoding
    - if number of unique values is large: (10,000) - use catboost!