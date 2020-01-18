# Introduction to variables, distribution and sampling
- 3 statistical building blocks:
    - types of variables
    - distributions
    - sampling
- Types of variables:
    - Variable is a characteristics of people and their environment
        - changes over time; changes from person to person
        - number and types of fruit and vegetables that some eats - a variable
        - getting cancer - outcome of interest
- Frequency distributions:
    - Knowing the distribution of a variable is vital to make sense of what is typical and what is unusual
    - Normal distribution - Gaussian distribution
    - Distributions describe the range of variables and how common each of values are
        - Number of kidneys the person was born with: 0, 1, 2, 3+. Vast majority has 2
    - Knowing a distribution is important for 2 reasons:
        - helps to understand how common or unusual a given value is
        - helps make predictions
    - Example: many blood test results are interpreted by comparing a patient's results with the distribution of results across a huge number of people
- Sampling:
    - Everything we know in medicine we've learned from a sample of people
    - Knowing results from the sample, we extrapolate those to the whole population
    - This is only valid if a sample is representative of a population for age/gender/...
    - Assumptions are often invalid in real life
- To sum up: we cant do a good public health research ourselves, or correctly interpret other people's work without understanding the fundamentals of variables, distributions and sampling

# Overview of types of variables
- There are discrete and continuous types of variables
- Discrete:
    - Categorical values
    - Autism could be described as "autism" or "not autism"
    - No sense in calculating averages/sums/...
    - The number of migraine days is a discrete variable and so can only take a finite number of values between 0 and 365. On any particular day, you either had a migraine or didn't have one.
- Continuous:
    - Take infinite number of different values between the extremes
    - Example: height and weight, if measured precisely (such precision isnt that useful)
    - Millilitres of blood transfused in an operation can take on any value in a range subject to the precision of the measuring instrument and so is a continuous variable.
    - The weights have been measured on two different scales (kg and lbs), but measurements can take on any value in a range subject to the precision of the measuring instrument, and so both are continuous variables.
- Integers (whole numbers)
    - These are NOT categorical variables!
    - Example: number of beds in hospitals, number of patients in the waiting list
    - We can calculate averages or standard deviations
- Discrete -> Nominal and Ordinal variables:
    - nominal: no order in the categories (gender and ethnic groups: male is not higher or lower than female)
    - ordinal: tehre is a ranking (severity of a disease or of a symptom: mild, moderate, severe)
- Binary data:
    - Dead/Alive, Pregnant/Not

# Well-behaved distributions
- Examples of such: normal and uniform
- Normal: the probability of getting a number between a and b is the same
    - example: lottery balls 
- Normal: could describe height/weight
    - distribution is symmetrical
    - SD (standard deviation) - a measure of spread
    - To build a density function, all you need is SD and mean - so you cana get every point on the curve
    - 1 SD: 68%
    - 2 SD: 95% of all data; 2.5% are taller than 2 SD above the mean, 2.5% of people are shorter than 2 SD below the mean
- Poisson: for count variables; describes the probability of a given number of events occuring in a certain amount of time or space
    - example: number of patients arriving in an hour
    - Underlying assumptions for poisson: rate should be constant over time; patients arriving have to be independent of each other
- Binomial: number of outcomes in a given number of trials
    - how many of sample patients have cancer

# Real-world distributions problems
- Distributions for single variable might and will differ from country to country
    - true even if distributions are well-behaved
- Distrtributions might be skewed - one cause could be that the distribution has a low average and can't take negative values (we cant eat <0 apples per day)
- If we just want to tell what proportion of people in our sample eat a certain number of fruit and vegetables - jsut show the figures!
- Problem: when we want to estimate some variable for the population
- Invalid distribution assumption makes the transition from sample to population absolutely incorrect
- How to analyze the numbers?
    - If we just describing data - we don't need to do anything else apart from showing all the numbers we got
    - We could aggregate some of the numbers to produce fewer categories
        - example: transform distribution into binomial distribution - this one is used a lot in medicine in general
        - simplification loses some details, losing some information value
- Dichotomising the data at a point that represents a clinically meaningful improvement for patients is the best option presented here. It is easy for patients to understand and for clinicians to communicate. With such skewed data, the mean change scores are likely to be small and could mask potentially important clinical differences.
- Means are sensitive to skewed distributions so would not be an accurate reflection of the data in this situation.

# The Role of Sampling in Public Health Research
- Sampling == measure a sample of people and then scale up to the population level
- How to use BMI of sample to estimate BMI of the whole population:
    - Suppose we measure 50 randomly chosen people
    - Mathematically: if BMI is distributed normally; if people from sample are chosen truly randomly => mean BMI is a unbiased estimate => is a population mean
    - Question: how certain could be this estimate?
- By keeping taking samples - what we get is a normal distribution
- Measure of uncertainty - uncertainty interval, most commonly a 95% confidence interval
- A 95% confidence interval has a 95% chance that the true population mean lies within the interval
    - Though this is not strictly true!
        - First, let's say something that is strictly true. If you take 100 samples, calculate their means and generate a 95% CI around each sample's mean, then 95 of those CIs will contain the population mean. The problem is that of course you don't take 100 samples. You only take one - the one that comes from your data set. Your 95% CI either includes the population mean or it doesn't. It's a yes/no thing, so nothing to do with probability. This is because the bounds (the two values) that make up the 95% CI are fixed values because they're just properties of your data set, like the number of patients in your sample. There's nothing random about the number of patients in your sample because it was something that you set when you went out to recruit them. So it is with the confidence interval. 
    - There is a 5% chance we are unlucky and our hypothesis is wrong!
- standard error of a mean ~= SD == width of the confidence interval
    - standard error of a mean != standard error!!!
    - SEoaM - You're trying to estimate the population mean, but random variation in your sample often results in some difference between your sample mean and the population mean. This difference is what the SEM describes.
- Standard error depends on:
    - amount of variation in the sample
    - number of people in our sample
- Do large samples have wider or narrower confidence intervals than smaller samples?
    - Answer: No
    - The bigger the sample = the narrower the interval
    - The bigger the sample = the closer population is described = the less uncertainty there is about population mean
- Would the confidence interval be narrower or wider if BMIs vary a lot than if they don't vary much?
    - Answer: wider
    - The more variation - the wider the interval
    - The more variation - the more uncertainty about what the real population mean is

# Bias
- When you're thinking about how you might select a sample, it’s important to remember the wider population you are intending to make inferences about. You want your sample to be representative of this population, and therefore you need to be careful to avoid introducing bias through your selection process.
- Bias in research is a huge topic, but it can be thought of as any deviation from the truth in data collection, data analysis, interpretation and publication that can cause false conclusions.

# How to choose a Sample
- For instance, if you want to evaluate the quality of care of people with migraine, then you could choose either community clinics or hospital clinics, or indeed both.
    - One issue is that the characteristics of the people seen in a community will differ from those seen in hospital.
    - Another consideration is that the measures of quality of care you use will also need to reflect the differing roles of the two sectors.
- You may choose to restrict your sample to a particular: age group.
    - Reasons:
        - practicality and cost
        - clinical or epidemiological reasons
    - If so, then the results will likely only apply to people of that age group.
- Sampling methods:
    - simple
    - stratified
    - cluster
- Simple random sample:
    - every person has an equal chance of being selected to the sample
- Stratified random sample:
    - Stratified sampling is used when there are important characteristics that you can split into subgroups or strata that you want to ensure are adequately represented in your sample
        - And it's a way of ensuring that you get, for instance, equal numbers of people with and without the risk factor.
    - First divide the population into important subgroups, or "strata" (sex, smoking status)
    - Then using the simple random sample method, you take a sample from each subgroup or stratum
    - Strata could be defined by age, gender, ...
- Cluster sample:
    - First divide the population into clusters, then randomly select a subset of clusters, lastly take a simple random sample from people in each subset
    - Clusters - commonly geographical areas, might also be clinics, schools
    - Example: school - cluster, randomly choose 5 schools, recruit all children from the school and they will be our sample
- People often struggle to distinguish between stratified and cluster samples. Stratified sampling produces homogenous groups in terms of the stratification factor, whereas cluster sampling produces heterogeneous groups within the cluster.

- Methods to avoid:
    - Non-probabilistic methods of sampling are likely to introduce bias into our samples and reduce the sample’s representativeness
        - convenience sampling and volunteer samples
- Convenience sampling uses a readily available sample to select participants in a non-random manner e.g. selecting all patients that walk into a healthcare clinic to answer a questionnaire. With convenience sampling there is no way of knowing if the time, location etc. effected the type of potential participant available and how the sample might differ if it had been conducted at a different time or location.
- Volunteer sampling is a non-random method where participants themselves opt into the sample. For example, an advert goes out on the radio requesting that listeners go online to complete a questionnaire. With volunteer sampling, you have no way to assess if and how those that choose to participate differ to those that did not. Think about what might motivate someone to volunteer into a sample and how this might affect your data.

