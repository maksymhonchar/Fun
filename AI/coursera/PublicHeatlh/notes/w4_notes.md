# Sampling errors for proportions and central limit theorem
- Does going from sample to population for proportions variables work? why?
- Example: what proportion of my country's population eat the recommended 5 portions of fruit and vegetables a day
    - sample 1: 100 people, 20 answer yes, 80 no
    - sample 2: ...
    - ...
    - sample N: ...
    - Resulting frequency is a binomial distribution and will look like a normal one
        - Binomial distribution - describes the probability of having a given number of events in a given number of trials
            - events: number of people eating 5 portions a day; trials: number of people
        - The bigger the sample, the more the binomial distribution resembles the normal
    - Why: central limit theorem
        - Theorem: if you take any distribution, then as you increase the sample size, the distribution increasingly resembles the normal
            - Note: there are exceptions!
- How many people do we need to proceed further?
    - Rule of thumb: for np to be greater than 5 (n - number of people, p - proportion eating five a day) and for n*(1-p)>5. If not - get the table for binomial distribution / info from the software

- With large enough samples, many important distributions in medicine resemble the normal distribution
- The great thing about this is it makes the important task of calculating 95 percent confidence intervals much easier.

# Hypothesis Testing
- Example: suppose we want to see whether people who got cancer are more/less/equally likely to eat 5 portions of fruit/vegs daily
    - we got 50 people with cancer, 10 of them ate 5 portions a day
    - we got 100 people without cancer, 30 of them ate 5 portions a day
    - all people were randomly sampled
    - our proportions: 20% with cancer, 30% without cancer
    - The main question: is a difference really in 10% points? Answer - we don't know yet. We need to test whether it's just due to random variation
    - Solution: chi-squared test - statistical test for comparing two or more proportions
- Our hypothesis:
    - null hypothesis: no difference between the two proportions
- p-value: the proportion of the distribution that is equal to or greater than the test statistic
    - p-value: the probability that the difference in the proportions of people getting their five a day with and without cancer is 10 percentage points or more if in fact, their proportions are the same.
    - p-value: It's the probability of getting the result you got or a more extreme result, that is of getting 10 percentage points or more given that the null hypothesis is true.
    - Convention: in order to reject the null hypothesis you need a p value of 0.05 or less
- p-value = 0.2 means there's no strong evidence of an association between fruit and vegetable eating, and risk of cancer
    - note: there might be an association, it is just not so strong
    - maybe much more bigger sample could find it
- The smaller the p value, the stronger the evidence for an association and the less likely that result is random

- Hypothesis testing steps:
    1. set up the null hypothesis
        - Proportion of people eating 5 portions daily is the same in people with or without cancer
    2. choose the measure to be tested
        - The proportion getting 5 portions a day
    3. Decide the appropriate distribution
        - Chi-squared distribution for proportions getting 5 portions a day
    4. Choose an appropriate statistical test
        - Chi-squared test for comparing proportions
    5. Run the test and interpret the p value

- Hypothesis testing steps, for coin tossing example:
    1) Set up your null hypothesis to test whether the coin is fair.
    2) Choose and calculate the measure to be tested, e.g. the proportion of successes (say, heads)
    3) In the previous lecture 'Sampling errors for proportions and central limit theorem', we covered the central limit theorem and how this allows you to use the normal approximation to get a confidence interval for a proportion if n*p ≥ 5 and n*(1-p) ≥ 5. Check this assumption. If it's valid, you can proceed with the normal approximation; if not, you’ll need to use the tables of the binomial distribution.
    4) Using the appropriate distribution determined in step (3), calculate the confidence interval for the proportion of successes.
    5) Now calculate the corresponding p-value. To get this, you need to calculate a test statistic and compare it with what's expected under the null hypothesis. If using the normal approximation, the appropriate test statistic is the z statistic.
    6) You can now look up the p-value from the appropriate distribution table.
    7) Interpret your p-value, recalling what your null and alternative hypotheses are from step (1)

# Degrees of freedom
- The sample used to calculate the estimate may comprise 4, 10 or 100 individuals, and therefore some estimates are based on more information than others
    - The degrees of freedom (df) is a way of incorporating this information into your analysis.
- DDoF - an estimate of the number of independent pieces of information that were used to calculate your estimate.
    - For example, if a sample of BMI values from 4 people has a mean of 24, once you have the first 3 values, you can work out the fourth value
    - For example if the first 3 values are 23, 25 and 26 then for the mean to equal 24 the fourth value has to equal 22. Only the first 3 numbers are able to vary freely and so the degrees of freedom is 3.
    - If there were 10 or 100 values in the sample the degrees of freedom would equal 9 and 99 respectively.
- The degrees of freedom is often one minus the number of items in the sample
    - More generally: it's the number of values minus the number of parameters estimated to get the estimate you are interested in.
        - If you want to estimate the mean BMI for the population, then that’s only one parameter (the mean).

# Chi2 test by example
- The null hypothesis of the chi-squared test is that there is no association between the row and column variables
- The chi-squared test looks at how different the observed values are compared with the values you would expect if there really was no association.
- The test statistic has a χ2 distribution with (r-1)*(c-1) degrees of freedom, where r is the number of rows and c is the number of columns in the table.
- For the chi-squared test, the degrees of freedom is equal to the number of categories minus 1.

# Chi2 test: different proportions results:
- Doubling sample size - smaller p-value
- Halving sample size - larger p-value
- Similar sample sizes - chi2 is very small, p value -> 1
- Different proportions - chi2 is large, p value -> 0

# Comparing two means
- The t-test works well for variables that are roughly normally distributed (it can in fact handle some skew)
- The t-distribution exists in its own right and looks like a normal distribution but with wider tails (extremes)
    - These wider tails become less and less important as the sample size increases, so that the t-distribution gets ever closer to the normal
- Using the t-distribution rather than the normal ensures you do not underestimate the test statistic and therefore overestimate your p-value when your sample size is small
- It’s also useful when the standard deviation of the population is unknown, which is a fairly common occurrence.
- Therefore in practice the t-distribution and its test statistic (called the t-statistic) are more commonly used when comparing means than the normal distribution and its test statistic (called the z-statistic).

- Student’s t-test for comparing two continuous variables
    - set up a null hypothesis of no difference
    - calculate an appropriate test statistic
    - compare it against a table of the relevant distribution
    - get a p value (and confidence interval if the test allows) for interpretation
- Our example:
     - Null hypothesis: H0: μc = μnc or H0: μc - μnc = 0 (difference between means = 0)
     - Alternative hypothesis: Ha: μc != μnc or μc - μnc != 0
     - calculate the measure to be tested: sample_mean_bmi_c - sample_mean_bmi_nc

- How many degrees of freedom should this t-test have?
    - You have two groups of patients – those with and those without cancer – and therefore are estimating two values – the mean BMI for each group.
    - The degrees of freedom is therefore calculated as (n1 + n2 – 2), where n1 and n2 represent the number of observations in each of your two samples.

- The bigger the test statistic (either more positive or more negative), the more likely that it’s come from a distribution with a different mean.
- P-value of t test: under the null hypothesis, we are expecting it to come from a t-distribution with mean zero, so how likely is that given our test statistic? Answer - p-value
