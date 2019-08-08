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
