# week 5 theory

# week 5 overview
- Regression is an important part of an analytics toolkit that allows us to understand how 2 variables are related
    - regeression is a tool that allows you to understand customer behavior and connect it to business and marketing decisions
- Some questions that could be answered using regression analysis:
    - how brand managers of brand_name could know if their promotions work?
    - does having a lot of promotions like X and Y have an effect on consumers?
    - do consumers react to these promotions?
    - what happens if we increase the number of promotions from 3 to 4?
        or we reduce the # of promotions from 3 to 2?
    - what is the effect on consumers?
    - how do we know how many promotions to run in a week?
- Plan:
    - discuss how to interpret regression outputs - how to connect it to marketing process?
    - explore confounding effects and the biases introduced by missing variables
    - distinguish between economic and statistical significance
- Statistical significance - what the regression output gives you
    - really does it translate into some kind of economic and business sense? distinguish that

# Regression
- y=kx+b
    - k - slope == coefficient of promotion
    - b - intercept
- Coefficient of promotion:
    - What we're trying to extrapolate on here is the relationship between the money a customer will spend based on the amount a company spends on promotions.

- Interpreting regression outputs:
    - Regression statistics:
        - multiple R
        - r**2
        - adjusted r**2
        - standard error
        - observations
    - ANOVA (Analysis of variance)
        - regression (df, SS, MS, F, Sig F)
        - residual (df, SS, MS, F, Sig F)
        - total (df, SS, MS, F, Sig F)
    - Intercept (coefficients, standard error, t stat, p-value)
        - == b when x=0
    - # of promotions (coefficients, standard error, t stat, p-value)
        - == k coefficient

- r**2 - percent of variation in the dataset
- r**2 tells you THE ACCURACY OF A FUNCTION
- low r**2 == Not much can be accounted for in that relationship based on the data points.
    - low r**2 - normal typical behavior because of complex human nature
    - low r**2 indicates unexplained variation in the model

- p-value - confidence in a regression
    - low value == high confidence
    - 0.00 value - it is very unlikely the regression will change
    - high value == You cannot trust the accuracy from the regression model.
- p-value:
    - The p-value lets you know the significance of findings given the sample size. It indicates the percent chance--the probability--that the coefficients will change beyond the standard error given the addition of more data points or different samples.
    - The p-value shows the probability that the coefficients are NOT statistically significant. The lower the p-value, the more likely your findings are significant.

# Multivariable regression
- It is really important to know WHAT TO PUT and WHAT TO NOT PUT in regression function
- this is about omitted variable bias
    - What makes our regression more accurate - as many inputs as we can - proper constants, variables and coefficients.
- Issue if a regression model has an omitted variable bias:
    - The model might predict larger coefficients, for example, larger impact for the rest of the variables than what they actually have.
    - Since the model does not include the omitted variable, the impact of the omitted variable is dispersed among the rest of the variables. Such a model might lose out on identifying the real driver influencing the dependent variable.

# Elasticity
- Price elasticity - % change in sales for a % change in price.
    - == ratio of change in quantity demanded (% deltaQ) and percentage change in price (% deltaP)
    - == PED = [Change in sales / change in price] * [price / sales] = (deltaQ / deltaP) * p/q
        - [change in sales / change in price] == coefficient
- Reasons to use elasticity - it has no units == we can track this metric and compare these values to identify improvements or declines in the effectiveness of your market

# log-log models
- The coefficient from a log-log model is equal to the elasticity, which shows how price effects sales.

- log allows to compute CHANGES == rate at which something changes

- log transform is particularly useful, when the rate of change of a variable X (such as sales) is relative to other variables Y (such as price)

- First difference of natural log == percentage change
    - loggin converts absolute differences into relative (== percentage) differences.
    - the series DIFF(log(Y)) represents the percentage change in Y from period to period.

- ekasticity can be obtained from log/log models
    - log(# of promotions + 1) = 0.317
        - 0.317 == change in log ($spent) when log (promos #) increases by 1 unit
        - == elasticity of promotion
    - same thing:
        - log($ spent) when log(# promo) is 0 = 2.236 == (1)
        - log($ spent) when log(# promo) is 1 = 2.553 == (2)
        - (1) - (2) = 0.317

# Marketing mix models
- Factors:
    - product quality
    - distribution
    - brand life cycle - early
    - Price
    - Promotion
    - Include carryover

- Statistical and Economic Significance
    - Statistical significance:
        - is the relationship observed in the sample likely to be observed in the population as well
        - p-value: <10% - for the coefficient of interest
    - Economic significance:
        - does the benefit from a marketing intervention (i.e., the size of the coefficient) justify the expense?
            - looks at "effect size" - "is it worth to invest into it?"

- What would indicate that you should continue with the promotions?
    - answer: both statistical and economical significance

- Economic significance: y=1.42x + 9.90
    - a unit increase in number of promotions increases units purchased by 1.42 == coefficient
    - Assume gross profit per unit is $5
    - cost of promotion is 0.50
    - profit = (units purchased * gross profit) - (cost of promotion * number of promotions)
    - profit = (1.42 * 5 - 0.50 * 1) = (7.1 - 0.5) = 6.6

- 6.6 means promotions are worth to conduct

# Main conclusion
- Just conducting the regression is not enough. You need to understand whether the outputs from this regression and the coefficients from these regressions actually have any economic significance for your business decisions.