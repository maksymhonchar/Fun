# Q1
- You are asked to build a classification model about meteorites impact with Earth (important project for human civilization). After preliminary analysis, you get 99% accuracy. Should you be happy? Why not? What can you do about it? (Hint: Rare event…)

## general q1
- An overfitted model is a statistical model that contains more parameters than can be justified by the data.
- An underfitted model is a model where some parameters or terms that would appear in a correctly specified model are missing.

- As a general rule: Start by overfitting the model, then take measures against overfitting.

## answer q1
- Should you be happy - NO
- Why not - you couldn't generalize to new examplse that were not in the training set - meteorites are such a rare event so there is not much of a test set.
- What can you do about it:
        - collect more data
                - HOWEVER - in our case, we just are not able to do that
        - use data augmentation - cropping, padding, horizontal flipping (NN)
                - randomly rotating the image, zooming in, adding a color filter (NN)
                - data augmentation only happens to the training set and not on the
                validation/test set.
        - Use architectures that generalize well
                - 
        - Add regularization (mostly dropout, L1/L2 regularization are also possible)
                - deep learning: dropout. Dropout deletes a random sample of the
                activations (makes them zero) in training
        - Reduce architecture complexity
- How to deal with unbalanced classes [2]:
        - https://towardsdatascience.com/deep-learning-unbalanced-training-data-solve-it-like-this-6c528e9efea6

# Q2
- Is it possible capture the correlation between continuous and categorical variable? If yes, how?

## general q2
- Correlation is a measure of the linear relationship between two variables. That makes no sense with a categorical variable.
        - by definition, a measure of linear association between two conitnuous variables

- Correlation between categorical / continuous variables:
        - 2 categorical: Chi-square
        - >2 quantitative (continuous/discrete): Pearson correlation
        - categorical and quantitative variable: ANOVA

- There are ways to measure the relationship between a continuous and categorical variable; probably the closest to correlation is a log linear model. 

- for ANOVA and Kruskall-Wallis, the null hypothesis is that the two variables are independant (ANOVA: Y is gaussian and has the same variance and mean for each X value; KW: Y has the same distribution function for each X value -- not forgetting the tests assumptions!).
- Hence, a significant result prooves that Y and X are dependant.

## answer q2
- Answer - you can: ANOVA, point biserial correlation.
        - test will not estimate anything, juste give a kind of yes/or no answer, here «
        there is/there is not association/correlation between X and Y ».
        - since correlation is defined by means and categorical variables do not have
        mean. Speaking of association is better.
- ANOVA for comparison of means


# Q3
- If you are working with gene expression data, there are often millions of predictor variables and only hundreds of sample. Give simple mathematical argument why ordinary-least-square is not a good choice for such situation if you to build a regression model. (Hint: Some matrix algebra…)

## general q3
- Ordinary least squares (OLS) is a type of linear least squares method for estimating the unknown parameters in a linear regression model.

- Assumptions of OLS regression:
        - The regression model is linear in the coefficients and the error term
                -- == "linear in parameters"
        - The error term has a population mean of zero
        - All independent variables are uncorrelated with the error term
        - Observations of the error term are uncorrelated with each other
        - The error term has a constant variance (no heteroscedasticity)
        - No independent variable is a perfect linear function of other explanatory
        variables
        - The error term is normally distributed (optional)

        - There is a random sampling of observations
                - The sample taken for the linear regression model must be drawn randomly
                from the population

        - The number of observations taken in the sample for making the linear regression
        model should be greater than the number of parameters to be estimated
                - mathematical sense: If a number of parameters to be estimated
                (unknowns) are more than the number of observations, then estimation is
                not possible. If a number of parameters to be estimated (unknowns) equal
                the number of observations, then OLS is not required. You can simply use
                algebra.
                
- formulas:
        - linear regression mode: y_i = (x_i)T * beta + eps_i
                - beta: p x 1 vector of unknown parameters
                - eps_i: unobserved scalar random variables (errors)
        - linear regression, matrix notation: y = X * beta + eps
                - y: n x 1 vector
                - X: n x p matrix of regressors - design matrix
        
## answer q3
- If number of observation < number of variables:
        - There is no longer a unique least squares coefficient estimate: the variance
        is infinite so the method cannot be used at all.
- Let's say we have: y = X * beta + eps
        - X - n x k
        - y - n x 1
        - beta - k x 1
        - eps - n x 1
- Let’s say there are 5 features, for each observation we will
have (x, x², x³ , x⁴, x⁵ , y).
For each (x, x², x³ , x⁴, x⁵ ) we will have a parameter θi.
h(x) =θ0 + θ1*x + θ2*x² + θ3*x³ + θ4*x⁴ + θ5*x⁵
- Say we only have 2 data points (1, 1, 0, 0, 0, 1) and (0, 1, 1, 1, 1, 3).
For J(θ) to be zero, there are numerous solutions.
After entering the two data points:
        J(θ) = 1/2 *(θ0+θ1+θ2–1)² + 1/2 * (θ0+θ1+θ2+θ3+θ4+θ5–3)²
As you can see, as long as we have θ0+θ1+θ2=1 and θ0+θ1+θ2+θ3+θ4+θ5=3, J(θ) would be zero and there are infinite number of solutions.

# Q4
- Explain why k-fold cross-validation does not work well with time-series model. What can you do about it? (Hint: Immediate past is a close indicator of future…)

## general q4
- Cross-validation (CV) is a popular technique for tuning hyperparameters and producing robust measurements of model performance

- CV procedure:
        - split the dataset into a subset called the training set, and another subset called the test set
        - The model is trained on the training subset and the parameters that minimize error on the validation set are chosen
        - Finally, the model is trained on the full training set using the chosen parameters, and the error on the test set is recorded.
        
- When dealing with time series data, traditional cross-validation (like k-fold) should not be used for two reasons:
        - Temporal Dependencies: With time series data, particular care must be taken in splitting the data in order to prevent data leakage.
        - Arbitrary Choice of Test Set: choice may mean that our test set error is a poor estimate of error on an independent test set.

## answer q4
-  If some pattern emerges in year 3 and stays for years 4-6, then your model can pick up on it, even though it wasn't part of years 1 & 2.
- What to do about it: apply "canonical" way to do time-series cross validation - "ROLL" through the dataset:
        - Basically, your training set should not contain information that occurs after
        the test set.
                fold 1 : training [1], test [2]
                fold 2 : training [1 2], test [3]
                fold 3 : training [1 2 3], test [4]
                fold 4 : training [1 2 3 4], test [5]
                fold 5 : training [1 2 3 4 5], test [6]

# Q5
- Simple random sampling of training data set into training and validation set works well for the regression problem. But what can go wrong with this approach for a classification problem? What can be done about it? (Hint: Are all classes prevalent to the same degree?)

## general q5
- Random sampling is a process that involves choosing one or more values from an underlying probability distribution
        - probability distribution defines how frequently a number (or range of numbers)
        will be chosen. == CONTINUOUS VARIABLE

- Simple random sampling is the most straightforward approach for getting a random sample. It involves picking a desired sample size and selecting observations from a population in such a way that each observation has an equal chance of selection until the desired sample size is achieved

- SRS ensures that each subgroup of the population of size n has an equal probability of being chosen as the sample.

## answer q5
?
- What you an do about it?
        - value = rand()
        if value < p(A) 
            return A
        if value < p(A)+p(B) 
            return B
        if value < p(A)+p(B)+P(C) 
            return C
        else            
            return D
        - randsample()

