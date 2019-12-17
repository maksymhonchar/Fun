# week4 overview
- data in financial field are usually non-stationary and have a high level of noises
    - many successful statistical models do not work with financial data because of that
    - model evaluation is much more important than model building
        - A statistically successful model is not necessarily successful in terms of profits
- main goal week4 - explore multiple linear regression model
- plan:
    1. introduce association of a random variable & introduce linear regression
    2. present assumptions for linear regression models & demonstrate how to validate these assumptions
    3. extend simple linear regression model to multiple linear regression model by allowing more than one predictors.
    4. apply the model we achieved in trading to ETF-SPY.
        - focus on the consistency of performance of models, and how to evaluate them correctly.

# Association of ranadom variables
- In real life, we may be interested in the association of 2 or multiple random variables.
    - Example: is there any association or pattern between the stock price change of tomorrow and the number of full days in the last five days?

- Covariance: use it to measure the association between two variables.
- Sample covariance (similar to sample covariance) is also divided by the degrees of freedom.

- From the values of .cov() method we cannot tell which pair has a stronger association
    - Indeed, the covariance is also affected by variance of two random variables.
    - We need to factorize it out in order to get a measure only for the strength of association, which is the coefficient of correlation.
    - Now, the correlation will only take values in between negative one and one, no matter what are the variation of the two variables. 

- We also can apply method correlation of data frame. You can find that diagonal elements are one. This is because correlation with itself must be perfectly correlated.

- Covariance and correlation can only address linear pattern.
    - There are quite a lot of quantitative measure for non-linear association.
        - scatter_matrix

- Correlation only measures strength of association between two variables.
    - But in practice, we need to evaluate associations between one variable and a set of multiple variables.
    - Secondly, in many applications, we not only need to evaluate the strengths of association.

# Simple linear regression model
- In conclusion, if we apply linear regression model, we assume there exists such a real pattern in population.
    - More specifically, linearity: The mean of y is linearly determined by predictors.
    - Independence: With different X, responses are independent.
    - Normality: The random noise and y follow normal distributions.
    - Equal variance: The variance y are all equal even if the values of predictors are different.

- These assumptions need to be validated if you want to make inference using linear regression models. But in most cases, we do not need to be strict about these assumptions if you just use the model to make a prediction. In real application, we do not know beta_0 which is called intercept, beta_1 which is called coefficient of slope, and sigma neither. So, we cannot identify exact position of a line of a mean equation.
    - However, the model can still be applied to make a prediction. The accuracy and the consistency of your model, do not rely on these four assumptions. 
    
- conclusion chapters 4.1 - 4.3. So far we've discussed simple linear regression model, and it's idea of ordinary least square estimation and the diagnostics of model assumptions.

# Multiple linear regression model
- The reason to choose SPY as a target to view the regression model is because it is very suitable for trading frequently.
    - It is cheap.
    - Each unit of SPY is always approximately one over 10 of S&P 500 index level.
    - To earn SPY, it requires very low fee ratios.
    - Volatility of SPY is very high. Two digits loss and gains race appears often.

- Different from Simple Linear Regression, Multiple Linear Regression will have multiple predictors.

- Multicollinearity refers to a situation in which two or more predictors in the multiple regression model are highly, linearly related.
    - One predictor can be predicted from the others with a substantial degree of accuracy and it is typical for our model since all indices of different markets are correlated.
    - Multicollinearity does not reduce predictive power

- Next, we will evaluate our models by comparing two statistics in train and test.
    - First statistic is RMSE, which is the square root of sum of squared errors averaged by degrees of freedom, where k is number of predictors.
        - This statistic is to measure the prediction error.
        - The reason to use the degrees of freedom is that, square of RMSE is unbiased estimator of variance of the noise.
    - The second is adjusted R-square.
        - In Simple Linear Regression, we use R-square to get the percentage of variation that can be explained by a model.
        - We found that by adding more predictors, the skew is always increasing, but the accuracy is even worse.
        - To compensate the effects of numbers predictors, we have adjusted R-square, which measures percentage of variation of a response that is explained by the model. 

# Evaluate the strategy
- Sharpe ratio measures excess return per unit of deviation in an investment asset or trading strategy.
    - For example, daily Sharpe ratio is equal to the mean of excess return divided by standard deviation of excess return.
- Maximum drawdown is a maximum percentage decline in the strategy from the historical peak profit at each point in time.
    - Maximum drawdown is that risk of mirror for extreme loss of a strategy. 
- From the mirror of a Sharpe ratio and maximum drawdown, we can tell that the performance of strategy is quite consistent in place of extreme loss. But the return per unit risk is not very consistent.