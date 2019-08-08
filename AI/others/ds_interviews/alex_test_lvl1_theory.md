# Duality principle in boolean algebra
- "1 + 1 = 1" is a boolean statement, TRUE boolean statement
- Duality principle: "If we exchange **every** symbol by its dual in a formula, we get the dual result"
	- everywhere we see 1, change to 0; everywhere we see 0, change to 1
	- change + to *; change * to +

# Gradient function
- Gradient of a function shows a direction of it's fastest ascent (NOT descent)
- Gradient is a multi-variable generalization of derivative.
        - Gradient is a way of packing together all the partial derivative information of
        a function
- Length of the gradient vector tells us the steepness of that direction of steepest ascent.
- Whereas the ordinary derivative of a function of a single variable is a scalar-valued function, the gradient of a function of several variables is a vector-valued function.
- Calculate gradient:
        - compute the partial derivatives (df/dx, df/dy for f(x,y))
        - put both derivatives in a fector
        - gradient: nabla f(x,y) = vector(df/dx; df/dy) 
                - gradient is a vector-valued function.
                - nabla == vector of full of partial derivative operators
                        - nable = vector(d/dx, d/d..., ...) - something where we can give
                        a function and it gives us another function.
        - gradient of any function == vector of its partial derivatives
- Gradient in the context of graph representation:
        - let's say we have f(x,y)=x^2 + y^2
        - nabla f(x,y) = vector(df/dx, df/dy) = vector(2x + 2y)
        - output - bunch of vectors facing away from the origin

# Combinatorics
- Three persons entering a railway carriage where there are 5 vacant seats. In how many ways can they seat themselves?
        - Choose one person: they have 5 places in which they can sit
        - Choose second person: they have 4 places
        - Thus, 5x4x3 possible arrangements
        - The answer: 60.
- There are 8 vacant chairs in a room. In how many ways can 5 persons take their seats?
        - If chairs are arranged in a line, first person will have 8 choices, second will
        have 7 choices third will have 6 and so on
        - The answer: 8×7×6×5×4= 6720 ways
- How many ways 12 persons may be divided into three groups of 4 persons each?
        - From 12 persons , 4 persons can be chosen in vector(12 4) ways.
        - From rest (12−4)=8 persons 4 persons can be chosen in vector(8 4) ways
        - Remaining 4 will form the third group.
        - Thus vector(12 4) x vector(84) are the ways for form three groups of 4 persons
        out of 12 persons.
        - But we have introduced order in group formation.
        - Now three groups can be permuted 3! ways and they are all same
        - Hence correct number of ways = vector(12 4) x vector(8 4) / 3! = 5775
- (v2) How many ways 12 persons may be divided into three groups of 4 persons each?
        - Main formula: C_n_k = n!/(k!(n-k)!)
        - The first team: C_12_4 = 495
        - The second team: C_8_4 = 70
        - The third team: C_4_4 = 1
        - Types of forming 3 groups together: 495*70*1 = 34650
        - This sorting assumes that position of each team is important == there are no
        "team numbers"
        - Hence, we need to divide by the ways of assigning team numbers - by 3!
        - 34650/3! = 5775 - correct answer
- How can you get a fair coin toss if someone hands you a coin that is weighted to come up heads more often than tails?
        - Answer: flip it twice
        - If you get heads first and then tails, call it "Heads". 
        - If you get tails first and then heads, call it "Tails". 
        - If you get the same result twice, start over. 
        - You may need to do this a few times, even lots of times if you're unlucky, but
        eventually you'll get either HT or TH and the likelihood of either is the same. 
        - There's a much more efficient algorithm scheme given in Tree Algorithms for
        Unbiased Coin Tossing with a Biased Coin that can be generalized to random
        variables with more than two values.

# Probability desnity function
- Introduction:
        - A continuous random variable takes on an uncountably infinite number of
        possible values. 
        - For a discrete random variable X that takes on a finite or countably infinite
        number of possible values, we determined P(X = x) for all of the possible values
        of X, and called it the probability mass function ("p.m.f.").
        - For continuous random variables, as we shall soon see, the probability that X
        takes on any particular value x is 0. 
                - That is, finding P(X = x) for a continuous random variable X is not
                going to work. Instead, we'll need to find the probability that X falls
                in some interval (a, b), that is, we'll need to find P(a < X < b). 
                - We'll do that using a probability density function ("p.d.f.").
- Example:
        - Imagine hamburger weights around 0.25 pounds - one randomly selected hamburger
        might weight 0.23 pounds while another might weigh 0.27 pounds. What is the
        probability that a randomly selected hamburger weighs between 0.20 and 0.30
        pounds?
        - X - weight of a randomly selected hamburger, in pounds.
        - Main question: P(0.20 < X < 0.30) = ?
- Solution of an example:
        - Imagine weights of 100 hamburgers, plotted on a density histogram
        - The histogram illustrates that most of the sampled hamburgers weigh close to X
        pounds, lets say to X=0.25 pounds.
        - now, decrease the length of the class interval - we get more bars
        - next, push the previous step further and decrease the intervals even more - we
        get not a histogram, but rather a curve
        - such a curve is a continuous probability density function
        - Finding a probability that a continious random variable X falls in some
        interval of values involves finding THE AREA UNDER the curve f(x) by the
        endpoitns of the interval.
- Definition: the probability density function (p.d.f.) of a continuous random variable X with support S is an integrable function f(x) satisfying the following:
        1) f(x) is positive everywhere: f(x)>0 for all x in S
        2) The area under the curve f(x)=1 for the whole interval S (-inf;+inf):
        integral_-inf_+inf(f(x)dx) = 1 (1 == 100%)
        3) If f(x) is the p.d.f. of x, then the PROBABILITY that x belongs to A, where A
        is some interval, is given by the integral of f(x) over that interval:
        P(X є A) = integral_over_A(f(x)dx)

# Mixture of Gaussians
- Let's assume we have pictures of Forest, Clouds and Sunsets:
        - For any of our image categories, and for any dimension of observed vector
        (like blue intensity in an image), we can assume Gaussian distribution model
        that random variable.
        - For forests, blue distribution might have mean=0.42, for cloud: mean=0.8
        - Note: different gaussians for every color channel (r g b) might (and will)
        have correlation between their structures, because rgb is dependant.
        - Unfortunately, we don't have SEPARATE distribution for each type of pictures
        (forests, clouds, sunsets). Instead, we have space of intensities, mixed with
        each other.
- Question: how do we model such distribution?
        - Answer: take each one of categories specific distribution and simply average
        them together. Resulting density: average of each gaussians. 
        - Simple gaussians - NOT ok. Instead: do weighted average.
        - Introduce cluster weights: Pi_k with each Gaussian component
                - 0<= Pi_k <= 1
                - sum(Pi_k) over k = 1
        - Next: add cluster specific mean and variance terms (location and spread of
        each of the distributions).
        Result: each mixture component represents a unique
        cluster (Pi_k, mu_k, sigma^2_k)
- According to the model, without observing the image content, what's the probability it's from cluster k?
        - Answer: p(z_i = k) = Pi_k
                - z_i - cluster assignment for observation x_i
- Example of Pi_k meaning: how prevalent "cloud images" are in our dataset
        - == how many "cloud images" we have relative to sunset images
        - == probability of seeing "clouds" image
- Question: given observation x_i is from cluster k, what's the likelihood of seeing x_i?
        - example: just look at distribution for "clouds"
        - p(x_i | z_i=k, mu_k, sum_k) = N(x_i | mu_k, sum_k)

# Kurtosis
- Kurtosis is a measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution.
        - That is, data sets with high kurtosis tend to have heavy tails, or outliers.
        - Data sets with low kurtosis tend to have light tails, or lack of outliers.
- A uniform distribution would be the extreme case.
- The histogram is an effective graphical technique for showing kurtosis of data set.
- Kurtosis of normal distribution = 3.
- Example: double exponential distribution
        - Compared to the normal, this distribution has a stronger peak, more rapid
        decay, heavier tails. =>
        => we eould expect kurtosis higher than 3
- Definition 2:
        - Kurtosis is a statistical measure that defines how heavily the tails of a
        distribution differ from the tails of a normal distribution
        - In other words, kurtosis identifies whether the tails of a given distribution
        contain extreme values.
        
# gradient descent algorithm - system of non-linear equations
- Wiki: Gradient descent can also be used to solve a system of nonlinear equations. Below is an example that shows how to use the gradient descent to solve for three unknown variables, x1, x2, and x3. This example shows one iteration of the gradient descent. == todo: view an example
        - https://en.wikipedia.org/wiki/Gradient_descent#Solution_of_a_non-linear_system

# Hamming distance
- Wiki: the Hamming distance between two strings of equal length is the number of positions at which the corresponding symbols are different
        - In other words, it measures the minimum number of substitutions required to
        change one string into the other, or the minimum number of errors that could
        have transformed one string into the other.
- In kNN, Hamming Distance == Calculate the distance between binary vectors
- Examples:
        - "karolin" and "kathrin". (kaROLin kaTHRin, HD=3)
        - "karolin" and "kerstin". (kArOLin kErSTin, HD=3)
        - 1011101 and 1001001.     (1 0 *1* 1 *1* 0 1 and 1 0 *0* 1 *1* 0 1)

# Log loss
- Usage: loss function in binary classifier.
- Logarithmic loss (related to cross-entropy) measures the performance of a classification model where the prediction input is a probability value between 0 and 1.
- The goal of our machine learning models is to minimize this value.
- Log Loss vs Accuracy:
        - Accuracy is the count of predictions where your predicted value equals the
        actual value. Accuracy is not always a good indicator because of its yes or no
        nature.
        - Log Loss takes into account the uncertainty of your prediction based on how
        much it varies from the actual label. This gives us a more nuanced view into the
        performance of our model.

# Micro average vs Macro average performance in multiclass classification
- A macro-average will compute the metric independently for each class and then take the average (hence treating all classes equally)
        - Just take the average of the precision and recall of the system on different
        sets.
        - Macro-average method can be used when you want to know how the system performs
        overall across the sets of data.
        - You should not come up with any specificdecision with this average.
- A micro-average will aggregate the contributions of all classes to compute the average metric
        - In Micro-average method, you sum up the individual true positives, false
        positives, and false negatives of the system for different sets and the apply
        them to get the statistics.
        - In a multi-class classification setup, micro-average is preferable if you
        suspect there might be class imbalance (i.e you may have many more examples of
        one class than of other classes).
                - micro-average can be a useful measure when your dataset varies in size.

# PCA for dimensionality reduction
- Curse of dimensionality - at a certain point, more features or dimensions can decrease a model’s accuracy since there is more data that needs to be generalized.
- Dimensionality reduction is way to reduce the complexity of a model and avoid overfitting. There are two main categories of dimensionality reduction: feature selection and feature extraction.
        - Feature selection - we select a subset of the original features
        - Feature extraction -  we derive information from the feature set to construct
        a new feature subspace.
- PCA - algorithm used to compress a dataset onto a lower-dimensional feature subspace with the goal of maintaining most of the relevant information.
- PCA - an unsupervised linear transformation technique that is widely used across different fields, most prominently for feature extraction and dimensionality reduction.
        - Other popular applications of PCA include exploratory data analyses and de
        noising of signals in stock market trading, and the analysis of genome data and
        gene expression levels in the field of bioinformatics.
- PCA helps us to identify patterns in data based on the correlation between features
- In a nutshell, PCA aims to find the directions of maximum variance in high-dimensional data and projects it onto a new subspace with equal or fewer dimensions than the original one.

# Transfer learning
- Wiki: Transfer learning is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.
        - Example: knowledge gained while learning to recognize cars could apply when
        trying to recognize trucks.
