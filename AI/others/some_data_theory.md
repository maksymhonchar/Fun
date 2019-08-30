# zero-mean preprocessing
- It is a linear transformation of the data that shifts the data so it's centered at the origin.
    - In other words: you shifted your data with your new mean set to the zero.
- Usually done by subtracting the mean vector from every data point.
- It's done to find the natural coordinate system for the data.
- Extensions could be:
    - Scaling so the data is not just centered at the origin but the standard deviation is normalized to one by scaling.
    - Principal component analysis, which aims to transform the coordinate system not just so that it is origin centered, but so that the primary components of independent variation lie on the different axis.
- Reasons to do this:
    - It makes the data much easier to work with.
    - Nonlinear operations and analysis generally do much better when the data is moved to the data's natural coordinate system (example: PCA assumes the data is zero-centered).
    - It's much easier to comprehend variation when you are centered at the origin.
- Cases where centering the data on its mean is useful:
    - Visual detection of whether a distribution is "the same" as another distribution, only, it has been shifted on the real line
    - Simplify calculations of higher moments: if the variables are de-meaned, you save a lot of useless calculations.
    - Random variables centered on their mean are the subject matter of the Central Limit Theorem
    - Deviations from the "average value" are in many cases the issue of interest, and whether they tend to be "above or below average", rather than the actual values of the random variables. "Translating" (visually and/or computationally) deviations below the mean as negative values and deviations above the mean as positive values, makes the message clearer and stronger.
    - It is advantageous to center the data when training neural networks.
    - Centering your data at zero is extremely important when using Bayesian statistics or regularization, since otherwise the data can be correlated with the intercept, which makes regularization not do what you usually want
    - Making the data zero mean can diminish many off-diagonal terms of the covariance matrix, so it makes the data more easily interpretable, and the coefficients more directly meaningful, since each coefficient is applying more primarily to that factor, and acting less through correlation with other factors.
- Make zero-mean in Matlab:
    X=X-mean(X(:));
- Make zero-mean in sklearn:
    from sklearn import preprocessing
    X_scaled = preprocessing.scale(X_train)
    X_scaled.mean(axis=0)  # array([0., 0., 0., ...])
    X_scaled.std(axis=0)  # array([1., 1., 1., ...])

# unit-variance
- "Variables have unit variance" means for example they are standardized
- Стандартизованная случайная величина имеет нулевое математическое ожидание и единичную дисперсию.
- Get unit variance by dividing by the standard deviation:
    X=X/std(X(:));

# SVM (support vector machine)
- Purpose: split the data in the best possible way
    - SVM finds a hyperplane, that best splits the data, because this hyperplance is as far as possible from the support vectors
    - SVM maximizes the margin
- How to maximize the margin:
    - this is a constrained optimization problem
        - optimization: we try to maximize the margin
        - contraint: the support vectors points CANT be on the road, the have to be away from it
- Way to solve this optimization problem - use Lagrange Multipliers technique
- Higher dimensions:
    - reason we want more dimensions - it is an opportunity to maximize margin better, therefore creating a better model.
- C parameter
    - C parameter allows to decide how much you want to penalize missclassified points
    - default=1.0
    - Low C: prioritize simplicity (soft margin)
    - High C: prioritize making few mistakes (probably overfitting)
- Multiple Classes
    - SVM are able to classify multiple classes (Multiclass SVC)
    - how to do that: in sklearn, add decision_function_shape='ovr' (or 'ovo')
        - ovr: one vs rest
            - pro: fewer classifications
            - cons: classes may be imbalanced
        - ovo: one vs one
            - pro: less sensitive to imbalance
            - more classifications
- Kernel-trick:
    - how does it work (informal): "it adds another dimension to the data"
    - Kernel options:
        - linear
        - rbf (radial basis function)
            - set 'C' and 'gamma'
                - 'gamma' parameter: small gamma = less complexity; large gamma: more complexity.
        - polynomial
        - sigmoid
- Pros and Cons of SVM
    - pros:
        - good at dealing with high dimensional data
        - works well on small data sets
    - cons:
        - picking the right kernel and parameters can be computationally intensive

# AUC-ROC curve
- todo: https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5
- todo: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc

# Normalize the data
- Amount-based -> Percent based (of total volume)
    - flour 2 cups, sugar 1/2 cup, ... -> flour 47%, sugar 24%, ..., total 100%

# Jupyter notebook steps
- Import libraries
- Import data
- Prepare the data
    - step 1: visualize the data
- Fit the model
- Visualize results
- Predict new case
    - add new additional point(s) & print and visualize the results

# Data cleaning methods

## src 1: https://towardsdatascience.com/data-handling-using-pandas-cleaning-and-processing-3aa657dc9418
1. load the data and check it's shape, head-tail values
2. check column names and dtypes
3. handle missing data: isna() fillna() dropna()
    - search by columns: isna().sum() / isnull().sum()
    - all: ....sum().sum()
    - fill the missing data:
        - ffill (last valid observation), bfill (next observation)
        - fill with mean/median/mode - 3 kinds of "averages"
4. Fix duplicated data:
    - .duplicated()
    - .unique()
    - easy way: drop_duplicated()
5. Binning data
    - pd.cut, pd.qcut
6. Detect outliers
    - box plot
    - z score - number that signifies how much standard deviation a data point is, from the mean.
    Z = (X - mean) / sigma
    - stats.zscore

## src2: https://towardsdatascience.com/3-steps-to-a-clean-dataset-with-pandas-2b80ef0c81ae
1. Dropping features
    - Perform a correlation analysis of the feature variables
    - Check how many rows each feature variable missing. If a variable is missing 90% of its data points then it’s probably wise to just drop it all together
    - Consider the nature of the variable itself. Is it actual useful, from a practical point of view, to be using this feature variable? Only drop it if you’re quite sure it won’t be helpful
2. Handling missing values
    - Fill in the missing rows with an arbitrary value
    - Fill in the missing rows with a value computed from the data’s statistics
    - Ignore missing rows - drop them
        - The last option can be taken if we have a large enough dataset to afford throwing away some of the rows.
        - However, before you do this, be sure to take a quick look at the data to be sure that those data points aren’t critically important.
3. Formatting the data (example: California: CA, C.A, california, Cali)
    - Standardising data format including acronyms, capitalisation, and style
    - Discretising continuous data, or vice versa

## src3: https://elitedatascience.com/data-cleaning
1. removing unwanted observations from your dataset
    - duplicates
    - irrelevant observations
        - For example, if you were building a model for Single-Family homes only, you wouldn't want observations for Apartments in there.
        - This is also a great time to review your charts from Exploratory Analysis. You can look at the distribution charts for categorical features to see if there are any classes that shouldn’t be there.
        - Checking for irrelevant observations before engineering features can save you many headaches down the road.
    - invalid observations
2. Fix Structural Errors
    - Structural errors are those that arise during measurement, data transfer, or other types of "poor housekeeping."
    - example:
        - typos or inconsistent capitalization
        - 'composition' is the same as 'Composition'
        - 'asphalt' should be 'Asphalt'
        - 'shake-shingle' should be 'Shake Shingle'
        - 'asphalt,shake-shingle' could probably just be 'Shake Shingle' as well
    - Check for mislabeled classes, 
        - i.e. separate classes that should really be the same.
            - e.g. If ’N/A’ and ’Not Applicable’ appear as two separate classes, you should combine them.
            - e.g. ’IT’ and ’information_technology’ should be a single class.
3. Filter Unwanted Outliers
    - We can’t stress this enough: you must have a good reason for removing an outlier, such as suspicious measurements that are unlikely to be real data.
4. Handle missing data
    - Missing categorical data: simply label them as ’Missing’
        - You’re essentially adding a new class for the feature.
        - This tells the algorithm that the value was missing.
        - This also gets around the technical requirement for no missing values.
    - Missing numeric data - flag and fill the values.
        - Flag the observation with an indicator variable of missingness.
        - Then, fill the original missing value with 0 just to meet the technical requirement of no missing values.
        - By using this technique of flagging and filling, you are essentially allowing the algorithm to estimate the optimal constant for missingness, instead of just filling it in with the mean.

# Dealing with outliers
- Reason to work on them:
    - The outliers may negatively bias the entire result of an analysis;
    - The behavior of outliers may be precisely what is being sought.
- Identify which record is outlier:
    - The simplest way to find outliers in your data is to look directly at the dataset.
        - When the number of observations goes into the thousands or millions, it becomes impossible.
    - Using charts - drawing and looking at the graphs
    - Using statistical methods
        - find the statistical distribution that most closely approximates the distribution of the data and to use statistical methods to detect discrepant points.
        - histograms!
            - result example: By normal distribution, data that is less than twice the standard deviation corresponds to 95% of all data; the outliers represent, in this analysis, 5%.

- Exclude the discrepant observations from the data sample: when the discrepant data is the result of an input error of the data, then it needs to be removed from the sample;
- Perform a separate analysis with only the outliers: this approach is useful when you want to investigate extreme cases, such as students who only get good grades, companies that make a profit even in times of crisis, fraud cases, among others.
- Use clustering methods to find an approximation that corrects and gives a new value to the outliers data.
- In cases of data input errors, instead of deleting and losing an entire row of records due to a single outlier observation, one solution is to use clustering algorithms that find the behavior of the observations closest to the given outlier and make inferences of which would be the best approximate value.

- methods src: https://towardsdatascience.com/a-brief-overview-of-outlier-detection-techniques-1e0b2c19e561
- Z-score
    - Z-Score pros:
        - It is a very effective method if you can describe the values in the feature space with a gaussian distribution. (Parametric)
        - The implementation is very easy using pandas and scipy.stats libraries.
    - Z-Score cons:
        - It is only convenient to use in a low dimensional feature space, in a small to medium sized dataset.
        - Is not recommended when distributions can not be assumed to be parametric.
- DBscan
    - Dbscan pros:
        - It is a super effective method when the distribution of values in the feature space can not be assumed.
        - Works well if the feature space for searching outliers is multidimensional (ie. 3 or more dimensions)
        - Scikit learn’s implementation is easy to use and the documentation is superb.
        - Visualizing the results is easy and the method itself is very intuitive.
    - Dbscan cons:
        - The values in the feature space need to be scaled accordingly.
        - Selecting the optimal parameters eps, MinPts and metric can be difficult since it is very sensitive to any of the three params.
        - It is an unsupervised model and needs to be re-calibrated each time a new batch of data is analyzed.
        - It can predict once calibrated but is strongly not recommended.
- Isolation Forest
    - Isolation Forest pros:
        - There is no need of scaling the values in the feature space.
        - It is an effective method when value distributions can not be assumed.
        - It has few parameters, this makes this method fairly robust and easy to optimize.
        - Scikit-Learn’s implementation is easy to use and the documentation is superb.
    - Isolation Forest cons:
        - The Python implementation exists only in the development version of Sklearn.
        - Visualizing results is complicated.
        - If not correctly optimized, training time can be very long and computationally expensive.

# Как бы вы могли наиболее эффективно представить данные с пятью измерениями?
    - Нам надо работать с 1д, или 2д, или 3д визуализациями
    - Можно выбрать лучшие измерения - которые лучше всего объясняют наши данные
    - Если данные находятся в одном рендже - можно рисовать их разными формами - кружками или квадратами
    - Если существуют категор и непрер данные - можно цветами обозначить различные категории 
    - PCA для визуализации. Правда, тогда мы не будем знать что точно означают наши координаты
    - Параллельные координаты
    - Glyphplot
    - Andrew's plot
    - Arc diagram
    - distplot - как соотносятся на гистограмме или точечном графике данные друг с другом

# Sparse data
- Sparse matrices come up in encoding schemes used in the preparation of data.
- Three common examples include:
    - One-hot encoding, used to represent categorical data as sparse binary vectors.
    - Count encoding, used to represent the frequency of words in a vocabulary for a document
    - TF-IDF encoding, used to represent normalized word frequency scores in a vocabulary.

- use numpy sparse matrices

- There are multiple data structures that can be used to efficiently construct a sparse matrix
- Three common examples are listed below.
    - Dictionary of Keys. A dictionary is used where a row and column index is mapped to a value.
    - List of Lists. Each row of the matrix is stored as a list, with each sublist containing the column index and the value.
    - Coordinate List. A list of tuples is stored with each tuple containing the row index, column index, and the value.

- There are also data structures that are more suitable for performing efficient operations:
    - Compressed Sparse Row. The sparse matrix is represented using three one-dimensional arrays for the non-zero values, the extents of the rows, and the column indexes.
    - Compressed Sparse Column. The same as the Compressed Sparse Row method except the column indices are compressed and read first before the row indices.

# Bernoulli & Multinomial Naive Bayes for Text Classification problems - Comparison
- src: http://www.inf.ed.ac.uk/teaching/courses/inf2b/learnnotes/inf2b-learn-note07-2up.pdf

- Short 'about':
    - Bernoulli document model: a document is represented by a binary feature vector, whose elements indicate absence or presence of corresponding word in the document.
    - Multinomial document model: a document is represented by an integer feature vector, whose elements indicate frequency of corresponding word in the document.

- Similarities:
    - The Bernoulli and the multinomial document models are both based on a bag of words
- Differences:
    - Underlying model of text:
        - **Bernoulli**: a document can be thought of as being generated from a multidimensional Bernoulli distribution: the probability of a word being present can be though of as a (weighted) coin flip with probability P(w_t | C)
        - **Multinomial**: a document is formed by drawing words from a multinomial distribution: you can think of obtaining the next word in the document by rolling a (weighted) |V|-sided dice with probabilities P(w_t | C)
    - Document representation:
        - **Bernoulli**: binary vector, elements indicating presence or absence of a word
        - **Multinomial**: integer vector, elements indicating frequency of occurance of a word in sentence
    - Multiple occurences of words:
        - **Bernoulli**: ignored
        - **Multinomial**: taken into account
    - Behaviour with document length:
        - **Bernoulli**: BEST for SHORT documents
        - **Multinomial**: longer documents are OK
    - Behaviour with "the":
        - **Bernoulli**: since "the" is present in almost every document, P("the" | C) ~ 1.0
        - **Multinomial**: since probabilities are based on relative frequencies of word occurence in a class, P("the" | C) ~ 0.05

# CART vs C4.5 vs ID3
- CART supports numerical target variables (regression)
- CART doesn't compute rule sets
- CART constructs binary trees using the feature and threshold that yield the largest information gain at each node.

- src: https://www.quora.com/What-are-the-differences-between-ID3-C4-5-and-CART

- ID3, or Iternative Dichotomizer
    - the first of three Decision Tree implementations developed by Ross Quinlan
    - Shannon Entropy to pick features with the greatest information gain as nodes
    - It builds a decision tree for the given data in a top-down fashion, starting from a set of objects and a specification of properties Resources and Information.
    - Each node of the tree, one property is tested based on **maximizing information gain and minimizing entropy**, and the results are used to split the object set
    - This process is recursively done until the set in a given sub-tree is homogeneous
    - The ID3 algorithm uses a greedy search.
    - It selects a test using the information gain criterion, and then never explores the 
    possibility of alternate choices.
- **Disadvantages**:
    - Data may be over-fitted or over-classified, if a small sample is tested.
    - Only one attribute at a time is tested for making a decision.
    - Does not handle numeric attributes and missing values.

- CART
    - Classification and Regression Trees
    - Constructs binary trees
    - CART uses Gini Impurity instead
    - The splits are selected using the twoing criteria and the obtained tree is pruned by cost–complexity Pruning
    - CART can handle both numeric and categorical variables and it can easily handle outliers.
- **Disadvantages**:
    - It can split on only one variable
    - Trees formed may be unstable

- C4.5
    - Improved version on ID 3 by Quinlan's
    - Shannon Entropy to pick features with the greatest information gain as nodes
    - The new features (vs ID3) are:
        - (i) accepts both continuous and discrete features;
        - (ii) handles incomplete data points;
        - (iii) solves over-fitting problem by (very clever) bottom-up technique usually known as "pruning";
        - (iv) different weights can be applied the features that comprise the training data.
- **Disadvantages**:
    - C4.5 constructs empty branches with zero values
    - Over fitting happens when algorithm model picks up data with uncommon characteristics , especially when data is noisy.






