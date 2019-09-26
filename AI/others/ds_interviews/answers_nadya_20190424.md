# Q1. Naive Bayes Classification Algorithm - define it. What is it used primarily for now - give some real world examples
- Overview: The NB algorithm is a ML algorithm for classification problems. It is primarily used for text classification, which involves high-dimensional training data sets.
- NB is a probabilistic classifier - основан на рассчете вероятности.
- NB algorithm is called "naive" because it makes the assumption that the occurence of a certain feature is independent of the occurence of other features.
- The price paid for this efficiency is that NB models often provide generalization performance that is slightly worse than that of linear classifiers like LR and LinearSVC.
- Real world examples:
    - realtime classification
    - multiclass classification
    - text classification, spam filtration, sentiment analysis (тональность текста)
    - recommendation system - NB with collaborative filtering

# Q3. Naive Bayes Classifiers: compare them to the linear models. +/- (adv/disadv).
- NB models share many of the strengths and weaknessses of the linear models
    + are very fast to train.
    + are vert fast to predict.
    + training procedure is easy to understand.
    + work very well with high-dimensional sparse data.
    + are relatively robust to the parameters.
    + are often used on very large datasets (where training even on a linear model might take too long).

# Q4. GaussianNB. +/- (adv/disadv). Parameters. Use cases irl.
- Can be applied to any continuous data, normally distributed.
- Stores the average value as well as the standard deviation of each feature for each class.
- Is mostly used on very high-dimensional data.

# Q5. BernoulliNB. +/- (adv/disadv). Parameters. Use cases irl.
- This classifier assumes binary discrete data {0, 1}.
- Counts if a word is in text.
- Has a single parameter - alpha, which controls model complexity - "smoothing" of the statistics.

# Q6. MultinomialNB. +/- (adv/disadv). Parameters. Use cases irl.
- Assumes count data (each feature represents an integer count of something, like often a word appears in a sentence).
- Has a single parameter - alpha, which controls model complexity - "smoothing" of the statistics.
- Takes into account the average value of each feature for each class.
- Is mostly used in text data classification.

# Q7. NV algorithm: +/-
+ Классификация, в том числе многоклассовая, выполняется легко и быстро.
+ Когда допущение о независимости выполняется, НБА превосходит другие алгоритмы, такие как логистическая регрессия (logistic regression), и при этом требует меньший объем обучающих данных.
+ НБА лучше работает с категорийными признаками, чем с непрерывными. Для непрерывных признаков предполагается нормальное распределение, что является достаточно сильным допущением.
- Если в тестовом наборе данных присутствует некоторое значение категорийного признака, которое не встречалось в обучающем наборе данных, тогда модель присвоит нулевую вероятность этому значению и не сможет сделать прогноз. Это явление известно под названием «нулевая частота» (zero frequency). Данную проблему можно решить с помощью сглаживания. Одним из самых простых методов является сглаживание по Лапласу (Laplace smoothing).
- Хотя НБА является хорошим классификатором, значения спрогнозированных вероятностей не всегда являются достаточно точными. Поэтому не следует слишком полагаться на результаты, возвращенные методом predict_proba.
- Еще одним ограничением НБА является допущение о независимости признаков. В реальности наборы полностью независимых признаков встречаются крайне редко.
