# Question 1. What is a linear machine learning model? Simple example of linear models in real world.
- Линейные модели (классификации и регрессии) - это клас моделей, которые определены линейной комбинацией features - то есть модели определены линейной функцией. Основываясь на тренировочных данных, процесс тренировки вычисляет один за другим веса для каждой features для формирования модели, которая может предсказать или оценить нужное target value.
- Example of linear model: amount of insurance a customer will purchase; variables are income and age. estimated target = 0.2 + 5 * age + 0.0003 * income
- Примеры таких моделей:
    - Regression: linear regression (ordinary least squares), ridge regression, lasso, ElasticNet.

# Question 2. Linear models for regression - please write & explain mathematical ground for a linear model. Examples of models.
- Для регрессии, формула предсказания для линейной модели выглядит следующим образом:
y=w0*x0 + w1*x1 + ... = wp*xp + b, где x0 и xp - features одной точки, одного экземпляра в датасете, w и b - параметры модели, которую мы обучаем, а y - предикшн, который делает модель.
- Предсказание будет являться взвешенной суммой input features, с весами даными в векторе w.
- Предсказания будут такими: прямая для одной фичи, площина для 2ух фичей, гиперпространства для больше фичей.
- Существует множество линейный моделей для регрессии. Разница между ними в том, как параметры модели w и b изменяются в процесе обучения от тренировочных данных И способами контроля сложности модели.
- Examples of models: linear regression (Ordinary least squares), ridge regression, lasso regression, elasticnet.

# Question 3. Linear models for classification - please write & explain mathematical ground for a linear model. Examples of models.
- Возьмем пример бинарной классификации.
В этом случае, предикшн будет сделан по формуле: y=w0*x0 + w1*x1 + ... = wp*xp + b > 0. 
Формула похожа на формулу линейной регрессии, но в нашем случае мы не просто возвращаем суму взвешеных фич, а сравниваем эту сумму с 0. Если сумма < 0 - тогда предикшн - "класс -1". Если больше 0 - предикшн бует "класс 1".
- Предсказание для classification linear models - это decision boundary. То есть именно decision boundary будет функцией инпута.
Другими словами - бинарный линейный классификатор это классификатор, который отделяет 2 класса используя линию, плоскость или киперплоскость.
- Алгоритмы для linear classification - их много, они оличаются по:
    - The way in which they measure how well a particular combination of coefficients and intercept fits the training data - choosing Loss Function J.
    - If and what kind of regularization they use
- Примеры: logistic regression, linear support vector machines (linearSVC, C - for classifier).

# Question 4. Differences between KNN and linear models.
- Во-первых, линейная регрессия - это параметрический подход, потому что в нем есть линейная функциональная форма для f(X). А knn - непараметрический метод.
- Во-вторых, предикшны, сделаные прямой по сравнению с теми предикшнами сделаные по kNN ОЧЕНЬ ОГРАНИЧЕНЫ. Кажется, что детали утрачены. В 2д пространстве данные могут быть и есть сильно искажены, а вот в более высоких пространствах любой y может быть идеально смоделлирован как линейная функция.
- Параметрические модели:
    + Easy to fit. One needs to estimate a small number of coefficients.
    + often easy to interpret (easier than kNN)
    - They make strong assumptions about the form of f (X).
    - Suppose we assume a linear relationship between X and Y but the true relationship is far from linear, then the resulting model will provide a poor fit to the data, and any conclusions drawn from it will be suspect.
- Непараметрические модели:
    + They do not assume an explicit form for f (X), providing a more flexible approach.
    - They can be often more complex to understand and interpret
    - I If there is a small number of observations per predictor, then parametric methods then to work better

# Question 5. Linear regression (ordinary least squares) - explain how it works (mathematical POV).
- Linear regression, or ordinary least squares (OLS), is the simplest and most classic linear method for regression.
- Linear regression finds the parameters w and b that minimize the mean squared error between predictions and the true regression targets, y, on the training set.
- The mean squared error is the sum of the squared differences between the predictions and the true values.
- Linear regression has no parameters, which is a benefit, but it also has no way to control model complexity.
- The choice of quantity to minimize when finding a best-fit line is by no means unique. The sum of the errors, or the sum of the absolute values of the errors, often seems more natural. Why is least squares the standard?
One reason is that the equations involved in solving for the best-fit line are straightforward, as can be seen in the above example. Equations involving absolute value functions are more difficult to work with than polynomial equations. Another qualitative reason is that it is generally preferred to penalize a single large error rather than many "medium-sized" errors. But this does not necessarily explain why the exponent  is preferred to, say, 1.5 or 3.
The most convincing justification of least squares is the following result due to Gauss: ...
That is, the least-squares line gives the model that is most likely to be correct, under natural assumptions about sampling errors.

# Question 6. Correlation between training set score & test set score (R^2) - when are we underfitting/overfitting?
- Scores on the both sets are very close together - underfitting.
- Training set prediction is more accurate than R2 - overfitting. We should try to find a model that allows us to control complexity - ridge regression, lasso. 

# Question 7. Ridge regression - explain how it works, similarities with OLS.
- Ridge regression is also a linear model for regression, so the formula it uses to make predictions is the same one used for ordinary least squares.
- In ridge regression, though, the coefficients (w) are chosen not only so that they predict well on the training data, but also to fit an additional constraint. We also want the magnitude of coefficients to be as small as possible; in other words, all entries of w should be close to zero
- Intuitively, this means each feature should have as little effect on the outcome as possible (which translates to having a small slope), while still predicting well. This constraint is an example of what is called regularization. Regularization means explicitly restricting a model to avoid overfitting.
- The Ridge model makes a trade-off between the simplicity of the model (near-zero coefficients) and its performance on the training set. How much importance the model places on simplicity versus training set performance can be specified by the user, using the alpha parameter.
Increasing alpha forces coefficients to move more toward zero, which decreases training set performance but might help generalization. Decreasing alpha allows the coefficients to be less restricted. For very small values of alpha , coefficients are barely restricted at all. alpha=0 - is a LinearRegression.
A higher alpha means a more restricted model, so we expect the entries of coef_ to have smaller magnitude for a high value of alpha than for a low value of alpha.

# Question 8. Lasso - explain how it works, when is it alternative to Ridge.
- An alternative to Ridge for regularizing linear regression is Lasso . As with ridge regression, using the lasso also restricts coefficients to be close to zero, but in a slightly different way, called L1 regularization.
The consequence of L1 regularization is that when using the lasso, some coefficients are exactly zero. This means some features are entirely ignored by the model. This can reveal the most important features of your model & make the model potentially easier to understand.
- Similarly to Ridge , the Lasso also has a regularization parameter, alpha , that controls how strongly coefficients are pushed toward zero.
- A lower alpha allowed us to fit a more complex model. If we set alpha too low - we get a result similar to LinearRegression.
- In practice, ridge regression is usually the first choice between these two models. However, if you have a large amount of features and expect only a few of them to be important, Lasso might be a better choice. Similarly, if you would like to have a model that is easy to interpret, Lasso will provide a model that is easier to understand, as it will select only a subset of the input features

# Question 9. L1 and L2 regularization. Penalty term in these regularizations. Mathematical writings. Why to use these regularizations, penalties?
- Why to use thesee: In order to create less complex (parsimonious) model when you have a large number of features in your dataset.
- A regression model that uses L1 regularization technique is called Lasso Regression and model which uses L2 is called Ridge Regression.
- The key difference between these two is the penalty term.
    - Ridge regression adds “squared magnitude” of coefficient as penalty term to the loss function.
    - Lasso Regression adds “absolute value of magnitude” of coefficient as penalty term to the loss function.
- The key difference between these techniques is that Lasso shrinks the less important feature’s coefficient to zero thus, removing some feature altogether. So, this works well for feature selection in case we have a huge number of features.

# Question 10. Meaning of high and low parameter of "C" parameter in LinearSVC or LogisticRegression. Real world examples of how to work with this value - lower/increase value.
- For LogisticRegression and LinearSVC the trade-off parameter that determines the
strength of the regularization is called C.
- higher values of C correspond to less regularization. In other words, when you use a high value for the parameter C , LogisticRegression and LinearSVC try to fit the training set as best as possible, while with low values of the parameter C , the models put more emphasis on finding a coefficient vector (w) that is close to zero.
- lower values of C- underfitting, high values of C - overfitting.

# Questsion 11. Linear models for multiclass classification - mathematical ground for them. What is one-vs-rest approach they use?
- In the one-vs.-rest approach, a binary model is learned for each class that tries to separate that class from all of the other classes, resulting in as many binary models as there are classes.

# Question 12. Tell what strength and weaknesses of linear models do you know?
+ Linear models are very fast to train, and also fast to predict.
+ They scale to very large datasets and work well with sparse (разбросанный) data.
+ Another strength of linear models is that they make it relatively easy to understand how a prediction is made, using the formulas we saw earlier for regression and classification.
+ Linear models often perform well when the number of features is large compared to the number of samples.
+ They are also often used on very large datasets, simply because it’s not feasible to train other models.
- Unfortunately, it is often not entirely clear why coefficients are the way they are. This is particularly true if your dataset has highly correlated features;
- However, in lower-dimensional spaces, other models might yield better generalization performance.


---

https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c

https://stackoverflow.com/questions/12283184/how-is-elastic-net-used

----

# Additional Question 3. scikit - what is coef_, what is intercept_ ?
- coef_ (w) - estimated coefficients for the LR problem.
- intercept (b) - Independent term in the linear model. where the prediction line should cross the y-axis or any axis.

# Additional Question 4. Lasso - Acronym?
Lasso - (Least Absolute Shrinkage and Selection Operator)

# Additional Question 5. *** scale for  C and alpha search
logarithmic
