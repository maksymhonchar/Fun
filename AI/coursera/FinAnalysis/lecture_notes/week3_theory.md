# week3 overview
- main: explain statistical inference.
- in fin analysis, we are concerned about characteristics of some targets called POPULATION.

- Example:
    - we want to make use of historical data to estimate the real mean return of some private equity funds.
    - we also want to testify some claims (someone's claim that their investment strategy can generate 30% yearly return)
    - we have to validate claims using data over the last N years.

- Plan:
    - talk about population, samples and random sampling
    - find out the distribution of sample mean
    - use of confidence interval to estimate population mean (ie average data return)
    - learn about hypothesis testing: apply it in validating some claims about the mean return

# Population and sample
- Population - group of individuals who have something in common
    - we may be interested in properties about a certain group, that cannot be observed completely
        - Example: all registered voters in Thailand
        - Example: all HK citizens who played golf at least once past year
        - Example: all neurons from your brain
- Since we cannot get information for every individual of these populations, we have to take a SAMPLE, which is a PART OF TARGET POPULATION.
    - Because sample is a small group of population, it has to be randomly selected.
- RANDOM SAMPLING - Process of selecting a sample
    - 2 kinds of random samling: 
        1. without replacement (putting back == sample with duplicates)
        2. with replacement (not putting back == unique items in sample)
- Parameter and statistics:
    - parameter - characteristic of summary number of POPULATIONS
    - statistic - chararcteristic of summary number of SAMPLE
        - characteristic == mean, variance, std
- Once target populations are fixed, these summary numbers will not change
- Sample statistic changes in different samples, even if they are drawn from the same population.

- DDOF - "0" for population
    - It means the denominator of the population variance is N, which is the number of the population. 
- DDOF - "1" for sample
    - It means the denominator of a sample variance has to be n-1 instead of n, which is the sample size.
    - n-1, in fact, is called degrees of freedom. 

- Why denominator of the sample variance is n-1?
    - Degrees of freedom is a number of values used in calculation that are free to variate. When we use the sample mean to compute the sample variance, we will lose one degree of freedom.
        - For example, sample size equal to 3, sample mean equal to 100 is used in computing sample variance. We can choose two individuals independently, for example, 90 and 80 in the table, but the third is not free to variate because of constraint of sample mean.
    - As you can see, the average of sample variances using n-1 is closer to population variance than using n.
    - The average of sample variance using n as denominator is always smaller than population variance, which is some mathematical reasons.

# Variation of Sample - Distribution of Sample Mean
- Knowing the variation and its rule is important to have to correctly evaluate the estimation and validate assertion about population based on the samples.
    - Example: we have historical data of 100 days. We can compute sample mean and variance of stock return. Mean distribution answers question:
        - inference of parameters?
        - how close are the statistics to population parameters?
        - can we make a claim that this stock is in upward trend? (== mean for return > 0)

- Central Limit theory - if the sample size is large, the distribution of sample means looks like normal one. 
    - Even if the population is not normal, the sample is approximately normal if the sample size is large enough
    - Intuitionally, if a sample is a good representative of the population, the population mean should be close to sample mean.

- It is plausible to say that the population mean is in a range with sample mean centered.

# Confidence interval
- Task: estimate population mean using interval with lower and upper bound. 

- To start with, we need to standardize sample mean because different sample has different mean and a standard deviation.
    - We can standardize sample mean by minus it's mean, which is identical to population mean and then divided by its standard deviation, which is the standard deviation of population divided by square root of sample size.
    - After standardization, it'll become standard normal, and follows Z-distribution.

- (Talking about confidence interval formula) Notice that Sigma is the population standard deviation, which is usually unknown. In practice, we can replace it using the sample standard deviation if sample size is large enough.
    - The interval here, for Mu is called confidence interval at the level of one minus Alpha.

- The 80 percent of confidence interval is printed out. Average return of Apple stocks falls in this interval with 80 percent chance.
    - Notice, this interval is on the positive side.
        - It implies that the average return is very likely to be positive.

# Hypothesis testing
- Main task of hypothesis testing: make a judgement whether the condition is satisfied. 
    - In statistics, hypothesis testing can use sample information to test the validity of conjectures about these parameters.

- The first step is to set hypothesis.
    - We have null hypothesis and alternative hypothesis.
        - The null hypothesis is assertion we are against.
            - In our example, NH: the null is population mean, the average daily return is 0.
        - Alternative hypothesis is a conclusion we accept whenever we reject the null.
            - In our example, AH: average daily return is not equal to 0.
    - Whether to reject or not reject null hypothesis is a sample based decision. Intuitionally, given that the null is correct, the difference between sample statistic, x-bar, and the population parameter mu cannot be very large. If it's significantly large, the null should be incorrect, and we should accept alternative.

- As the sample size increases, the degree of freedom increases, and the t is more and more like z-distribution. So with a large sample, we can treat t as if it is a z-distribution.

- rejection regions
- two-tailed test & one-tail test

- If statistic z-hat falls into rejection region, we can tell that statistics is far away from 0, significantly and then we can reject the null. Here, we should notice that z-hat is also possible to take values in rejection region even if the null is correct, and the mu equal to 0. 