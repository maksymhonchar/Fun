# Q1 I built a linear regression model showing 95% confidence interval. Does it mean that there is a 95% chance that my model coefficients are the true estimate of the function I am trying to approximate? (Hint: It actually means 95% of the time…)

## general Q1
- Answering the question: what is the reasonable ranges for k and b in y=kx+b ? 
    - answer - calculate confidence interval for both k and b.

- confidence interval - интервал, который покрывает неизвестный параметр с заданной надёжностью
    - CI is a RANGE OF VALUES we are fairly sure our true value lies in
- Example: Average Height
    - We measure the heights of 40 randomly chosen men, and get a mean height of 175cm,
    - std of men's heights is 20cm
    - The 95% confidence interval is: 175cm +- 6.2cm
    - This says the true mean of ALL men (if we could measure all their heights) is likely to be between 168.8cm and 181.2cm.
        - But it might not be!
    - The "95%" says that 95% of experiments like we just did will include the true mean, but 5% won't.
    - So there is a 1-in-20 chance (5%) that our Confidence Interval does NOT include the true mean.
- width of confidence interval is a measure of the overall quality of the regression 

- Regression slope : confidence interval
    - equation of simple linear regression: y=b0+b1*x
        - b0 is constant, b1 is slope, x is the value of independent variable, y is predicted variable of the dependent variable
    - estimation requirements:
        - The dependent variable Y has a linear relationship to the independent variable X.
        - For each value of X, the probability distribution of Y has the same standard deviation σ.
        - For any given value of X:
            - The Y values are independent.
            - The Y values are roughly normally distributed (i.e., symmetric and unimodal). A little skewness is ok if the sample size is large.
        - distribution are like normal
    - use OLS analysis to get SE Coef
        - from there, we get constant coefficient (b0) and slope (b1)
        - SE Coef - standard error of the coefficient
            - what we should care of - slope 

- What we've done - simulate you and my friends
    - since me and my friends are calculating and reporting 95% confidence intervals, it means ON AVERAGE, 95\100 of us will calculate intervals, that contain true slope and contain true intercept
    - BUT: we never get to know which of 5/100 will miss the true slope
    - SO: 5/100 will on average miss the true slope and the true intercept

- Meaning of 95% confidence interval - if you and 999 of your friends run the exact same experiment, about 50 of you would miss either the true intercept or the true slope, BUT we'd never actually know which of you miss, because in real world we DON'T KNOW real k and b

# answer Q1
- 95% confidence interval - 95% of the time, that you calculate the 95% confidence interval, it is going to overlap with the true value of the parameter that we are estimating

# Q2 What is a similarity between Hadoop file system and k-nearest neighbor algorithm? (Hint: ‘lazy’)

# general Q2
- K-NN is a lazy learner because it doesn’t learn a discriminative function from the training data but “memorizes” the training dataset instead.
- For example, the logistic regression algorithm learns its model weights (parameters) during training time. In contrast, there is no training time in K-NN. Although this may sound very convenient, this property doesn’t come without a cost: The “prediction” step in K-NN is relatively expensive! Each time we want to make a prediction, K-NN is searching for the nearest neighbor(s) in the entire training set! (Note that there are certain tricks such as BallTrees and KDtrees to speed this up a bit.)

- The distributed cache doesn't support lazy loading, they are copied down to each task node prior to the first map / reduce task for your job being executed on that node (note the files are only copied to nodes where a map / reduce task will take place)
-  If you want lazy loading, just open the files directly in HDFS, although this doesn't scale too well for you namenode / datanodes if you have 1000's of concurrent tasks trying to read from the same file
- You can use symbolic linking to give files in the distributed cache friendly names, and they will appear in the local working directory (symlinked) for each map / reduce task.
- For example, with the generic options parser options -files, you can upload a file to HDFS, add it to the DistributedCache and assign a friendly name as follows:
hadoop jar myjar.jar MainClass -files ref-map.txt#map1.txt ...
- Now you should be able to open the ref-map.txt file just by calling the following in your map / reducer:
File map1 = new File("map1.txt");
- If you're files are already in HDFS, then just add then as normal, and call the createSymlink(Configuration) method. You can also assign friendly nanes by using fragment URIs when you add files to the distibuted cache:
DistributedCache.addCacheFile(new URI("/path/to/file.txt#file1", conf);

- Lazy persist writes
    - HDFS supports writing to off-heap memory managed by the Data Nodes
    - The Data Nodes will flush in-memory data to disk asynchronously thus removing expensive disk IO and checksum computations from the performance-sensitive IO path
    -  Applications can choose to use Lazy Persist Writes to trade off some durability guarantees in favor of reduced latency.

# Q3 Which structure is more powerful in terms of expressiveness (i.e. it can represent a given Boolean function, accurately) — a single-layer perceptron or a 2-layer decision tree? (Hint: XOR)

# general Q3
- XOR: 
    - A*notB + notA*B
    - (A+B)*(notA+notB)
- single perceptron can’t represent the boolean XOR function
    - [XOR] is not linearly separable so the perceptron cannot learn it
    - Single layer perceptrons are only capable of learning linearly separable patterns

- Quora answer:
    - Assuming the decision tree branches on the value of a single bit, and that there are at least three bits of input, these are incommensurate.
    The decision tree can represent XOR on the first two bits of the input, which the single-layer perceptron cannot, while the single-layer perceptron can represent n-bit majority (i.e., true if at least n/2 input bits are true) which a decision tree of depth less than n cannot.

# answer Q3
- ?

# Q4 And, which one is more powerful — a 2 layer decision tree or a 2-layer neural network without any activation function? (Hint: non-linearity?)

# answer Q4
- ?

# Q5 Can a neural network be used as a tool for dimensionality reduction? Explain how.

## general Q5
- none

## answer Q5
- yes, it can be used as a tool for DR
- Autoencoders!
    - But you will obviously have to train the network and choose an appropriate cost-function (unlike the unsupervised methods such as PCA and t-SNE).
- neural net architecture:
    - w1(d=4) -> w2(n=8) -> w2^(n=2) -> w1^(n=4)
    - At the bottom is input layer with 4 units where the sample is inserted. The one above is hidden layer with 12 units and its task is to learn non-linear features.
    - Layer in the middle has only 2 units and its task is to learn data features represented in 2 dimensions
        - This layer will provide us with the reduced dimensionality representation.
    - Output layer should match the input layer and is real-valued, hence the cost function is mean squared error.
    The requirement on the input/output equality also makes the learning unsupervised
    

# Q6 Everybody maligns and belittles the intercept term in a linear regression model. Tell me one of its utilities. (Hint: noise/garbage collector)

## general Q6
- the model indicatse that teams with coaches who had a salary of 0 million dollars will average a winning percentage of approximately 39%

- Paradoxically, while the value is generally meaningless, it is crucial to include the constant term in most regression models!

- Mathematically correct version: constant is the mean response value when all predictor variables are set to zero.
- However, a zero setting for all predictors in a model is often an impossible/nonsensical combination, as it is in the following example.
- Example: weight=-114.3+106.5*height_m
    - If height is zero, the regression equation predicts that weight is -114.3 kilograms!
    - Clearly this constant is meaningless and you shouldn’t even try to give it meaning. No human can have zero height or a negative weight!

- Even if it’s possible for all of the predictor variables to equal zero, that data point might be outside the range of the observed data.

- The Constant Is the Garbage Collector for the Regression Model
    - The constant term is in part estimated by the omission of predictors from a regression analysis
    - In essence, it serves as a garbage bin for any bias that is not accounted for by the terms in the model. 
        - You can picture this by imagining that the regression line floats up and down (by adjusting the constant) to a point where the mean of the residuals is zero, which is a key assumption for residual analysis. This floating is not based on what makes sense for the constant, but rather what works mathematically to produce that zero mean.
        - The constant guarantees that the residuals don’t have an overall positive or negative bias, but also makes it harder to interpret the value of the constant because it absorbs the bias.

## answer Q6
- ...

# Q7 LASSO regularization reduces coefficients to exact zero. Ridge regression reduces them to very small but non-zero value. Can you explain the difference intuitively from the plots of two simple function|x| and x²? (Hint: Those sharp corners in the |x| plot)

## general Q7
- https://xavierbourretsicotte.github.io/ridge_lasso_visual.html
- The Lasso L1 term provides a much more aggressive regularization because the intersection between the constraint function and the cost function happens at the “vertices” of the diamond where either / or each variable is equal to zero. The effect of such regularization is to “cancel” out each variable, and by varying λ we can test the impact of each variable on the model. In effect this is automatic variable selection.
- On the other hand, Ridge regression provides a less aggressive form of regularization where the coefficients tend to zero in the limit only.

- https://stats.stackexchange.com/questions/30456/geometric-interpretation-of-penalized-linear-regression

- https://medium.com/@alexfharlan/ridge-vs-lasso-regression-how-to-keep-them-straight-5ee4a2d7f606
- note: cost functions:
    - ridge: alpha*|beta|^2
    - lasso: alpha*|beta|
- Lasso:
    - as λ gets significantly large β is forced to 0. But what is important to note about the lasso cost function, is that because we are using the absolute value, instead of the square of our coefficients, less important features’ coefficients can be assigned a value of 0. 
- Ridge:
    - we’re ‘adding all the coefficients squared’
    - λ is significantly large the sum of squared errors term in our cost function becomes insignificant. Since we are looking to minimize the above function, this forces β to 0.
    - So we see that as λ increases, the values in β decrease.
    - This is the behavior we wanted because it reduces the variance. 

- The reason why the penalty term results in coefficients being zero has to do with the constraint region, which I won’t go into here, but I recommend reading into because it will be helpful in order to cement the idea that the L1 penalty allows for some coefficients to be zero.


## answer Q7
- ...

# Q8 Let’s say that you don’t know anything about the distribution from which a data set (continuous valued numbers) came and you are forbidden to assume that it is Normal Gaussian. Show by simplest possible arguments that no matter what the true distribution is, you can guarantee that ~89% of the data will lie within +/- 3 standard deviations away from the mean (Hint: Markov’s Ph.D. adviser)

## general Q8
- Chebyshev’s inequality says that at least 1-1/K2 of data from a sample must fall within K standard deviations from the mean (here K is any positive real number greater than one).
- 1-1/9~=0.89

- Any data set that is normally distributed, or in the shape of a bell curve, has several features. One of them deals with the spread of the data relative to the number of standard deviations from the mean. In a normal distribution, we know that 68% of the data is one standard deviation from the mean, 95% is two standard deviations from the mean, and approximately 99% is within three standard deviations from the mean.
- But if the data set is not distributed in the shape of a bell curve, then a different amount could be within one standard deviation. Chebyshev’s inequality provides a way to know what fraction of data falls within K standard deviations from the mean for any data set.

- Facts About the Inequality
We can also state the inequality above by replacing the phrase “data from a sample” with probability distribution. This is because Chebyshev’s inequality is a result from probability, which can then be applied to statistics.
- It is important to note that this inequality is a result that has been proven mathematically. It is not like the empirical relationship between the mean and mode, or the rule of thumb that connects the range and standard deviation.

- Example
Suppose we have sampled the weights of dogs in the local animal shelter and found that our sample has a mean of 20 pounds with a standard deviation of 3 pounds. With the use of Chebyshev’s inequality, we know that at least 75% of the dogs that we sampled have weights that are two standard deviations from the mean. Two times the standard deviation gives us 2 x 3 = 6. Subtract and add this from the mean of 20. This tells us that 75% of the dogs have weight from 14 pounds to 26 pounds.

- Use of the Inequality
If we know more about the distribution that we’re working with, then we can usually guarantee that more data is a certain number of standard deviations away from the mean. For example, if we know that we have a normal distribution, then 95% of the data is two standard deviations from the mean. Chebyshev’s inequality says that in this situation we know that at least 75% of the data is two standard deviations from the mean. As we can see in this case, it could be much more than this 75%.
- The value of the inequality is that it gives us a “worse case” scenario in which the only things we know about our sample data (or probability distribution) is the mean and standard deviation. When we know nothing else about our data, Chebyshev’s inequality provides some additional insight into how spread out the data set is.

# Q9 Majority of machine learning algorithms involve some kind of matrix manipulation like multiplication or inversion. Give a simple mathematical argument why a mini-batch version of such ML algorithm might be computationally more efficient than a training with full data set. (Hint: Time complexity of matrix multiplication…)

# Q10 Don’t you think that a time series is a really simple linear regression problem with only one response variable and a single predictor — time? What’s the problem with a linear regression fit (not necessarily with a single linear term but even with polynomial degree terms) approach in case of a time series data? (Hint: Past is an indicator of future…)

## general Q10
- The main argument against using linear regression for time series data is that we're usually interested in predicting the future, which would be extrapolation (prediction outside the range of the data) for linear regression. Extrapolating linear regression is seldom reliable. However, many popular time series models do not extrapolate reliably either.

- The main problem is that it is unlikely to hold in the long run, since that would imply that sales keep growing at a steady pace, no matter what happens. You are extrapolating from limited data, which is rarely a good thing to do.

- It gets even worse when the correlation is negative, because that implies that at some point people will come in and sell their stuff at your store (which you then need to sell back to your original provider), which is probably not part of the business model.

- Another reason that linear regression for time series is a bad ‘fit’ has to do with sampling. In a non time series framework it is reasonable to assume that each variable is a mere sample from a larger population that can take on many values according to some distribution. And regression methods take that into account.
- With time series regression, you are just seeing just one realization of a sampling sequence which has been generated from some unknown stochastic process. In a different sample, you could see a slightly different result. However people usually never consider alternate time series sequences, but rely on that one fixed sequence that history has played out.

- The problem arises from the fact that ordinary least squares does not work for Time Series Data.
The principal reason for this is implicit in your question: Autocorrelation.
You can’t assume that the errors are i.i.d. or your model will be inaccurate.
There are ways of getting around that and using regression, but they are limited.
For example you could compute the difference between two different times:
Yit=β0+β1Xit+β2Zi+μit
where Z is a factor which does not change overtime (but changes over entities).
Then, you could build a linear model of the difference:
(Yt−Yt−1)=β1(Xt−Xt−1)+(μt−μt−1)

- "Time series data" can cover a lot of things.  But the problem isn't so much randomness as independence.  In time series data, the value for the previous time period is (almost always) a good predictor of the value for the current period.