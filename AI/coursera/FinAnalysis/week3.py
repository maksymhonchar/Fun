#%% Load libraries
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import numpy as np
import pandas as pd
from scipy.stats import norm

#%% Population and samples
data = pd.DataFrame()
data['population'] = [47, 48, 49, 50, 19, 20, 21, 22, 25, 30, 60]

sample_noreplacement = data['population'].sample(10, replace=False)
sample_withreplacement = data['population'].sample(10, replace=True)

display(sample_noreplacement)  # no duplicates in index or values column
display(sample_withreplacement)  # duplicates could be observed

# parameters - population
# statistics - sample

# Print parameters of population:
print('Population mean, var, std, size are: {0}'.format([
    data['population'].mean(), data['population'].var(ddof=0),
    data['population'].std(ddof=0), data['population'].shape[0]
])) 

# Print statistics os sample
a_sample = data['population'].sample(10, replace=True)
print('Sample mean, var, std, size are: {0}'.format([
    a_sample.mean(), a_sample.var(ddof=1), a_sample.std(ddof=1), a_sample.shape[0]
]))

#%% Explain why ddof=1
sample_length = 500
sample_variance_collection_0 = [
    data['population'].sample(50, replace=True).var(ddof=0)
    for i in range(sample_length)
]
sample_variance_collection_1 = [
    data['population'].sample(50, replace=True).var(ddof=1)
    for i in range(sample_length)
]

print('population variance is {0}'.format(data['population'].var(ddof=0)))
print('average of sample variance with n is {0}'.format(
    pd.DataFrame(sample_variance_collection_0).mean()
))
print('average of sample variance with n-1 is {0}'.format(
    pd.DataFrame(sample_variance_collection_1).mean()  # closer than using n to real population variance
))

# on average, the average of estimator using division n-1 equals population variance

# Degrees of Freedom == the number of values in calculation that are free to variate

#%% Variation and distribution of the sample mean

# Sampling fromm normal distribution
norm_dist_sample = pd.DataFrame(np.random.normal(10, 5, size=100))
print('Sample mean, std are {0}'.format([
    norm_dist_sample[0].mean(), norm_dist_sample[0].std(ddof=1)
]))

# Empirical distribution of sample mean and variance
mean_lst = []
var_lst = []
for t in range(1000):
    sample = pd.DataFrame(np.random.normal(10, 5, size=10))
    mean_lst.append(sample[0].mean())
    var_lst.append(sample[0].var(ddof=1))

collection = pd.DataFrame()
collection['mean_lst'] = mean_lst
collection['var_lst'] = var_lst

collection['mean_lst'].hist(bins=500, normed=1)  # kind of normally distributed
plt.show()

collection['var_lst'].hist(bins=500, normed=1)  # non-normal, right-skewed
plt.show()

#%% Why variance of sample mean is smaller than variance of population?
population = pd.DataFrame(np.random.normal(10, 5, size=100000))
population[0].hist(bins=100, color='cyan', normed=1)
collection['mean_lst'].hist(bins=100, normed=1, color='red')
plt.show()

#%% Sampling from general distribution | sampling vs big sampling
# ЦПТ - if sample size is large enough, the distribution of sample mean is approximately NORMAL
sample_mean_lst = []
population = pd.DataFrame([1, 0, 1, 0, 1])
for t in range(10000):
    sample_100 = population[0].sample(100, replace=True)  # pretty big sample size - 100
    sample_mean_lst.append(sample_100.mean())

collected_means = pd.DataFrame()
collected_means['mean_lst'] = sample_mean_lst
collected_means.hist(bins=100, normed=1, figsize=(15, 10))

#%% Try the same stuff but with lower sample size
sample_mean_lst = []
for t in range(100000):
    sample_10 = population[0].sample(10, replace=True)  # pretty big sample size - 100
    sample_mean_lst.append(sample_10.mean())

collected_means = pd.DataFrame()
collected_means['mean_lst'] = sample_mean_lst
collected_means.hist(bins=100, normed=1, figsize=(15, 10))

#%% Confidence intervals
aapl = pd.DataFrame.from_csv('data/apple.csv')
aapl['log_return'] = np.log(
    aapl['Close'].shift(-1) / aapl['Close']
)

# sigma - std; we can replace std with sample standard deviation IF sample size is large enough.

# Calculate 80% confidence interval
z_left = norm.ppf(0.1)
z_right = norm.ppf(0.9)
sample_mean = aapl['log_return'].mean()
sample_std = aapl['log_return'].std(ddof=1) / (aapl.shape[0])**0.5  # replace std with sample std

# Calculate left and right quantiles
interval_left = sample_mean + z_left * sample_std
interval_right = sample_mean + z_right * sample_std

print('Sample mean is {0}'.format(sample_mean))
print('80%% interval is {0}'.format([interval_left, interval_right]))

# 80%% interval is [0.0004927367254936761, 0.0014581987928065012]
# Because interval is on the positive side, it implies that the average return is very likely to be POSITIVE.

#%% 90% confidence interval
# Lets build 90% confidence interval for log return
sample_size = aapl['log_return'].shape[0]
sample_mean = aapl['log_return'].mean()
sample_std = aapl['log_return'].std(ddof=1) / sample_size**0.5

# left and right quantile
z_left = norm.ppf(0.05)
z_right = norm.ppf(0.95)

# upper and lower bound
interval_left = sample_mean + z_left * sample_std
interval_right = sample_mean + z_right * sample_std
print('90%% interval is {0}'.format([interval_left, interval_right]))
# 90%% interval is [0.0003558891850999989, 0.001595046333200178]

#%% Hypothesis testing
# Making a judgement, whether the condition (some kind of forecast, like "should you invest in this project with 36 months data at hand?") is satisfied.

# Plot AAPL close price & return
fig, [axis_0, axis_1] = plt.subplots(1, 2, figsize=(15, 2))
aapl.loc[:, 'Close'].plot(ax=axis_0)
aapl.loc[:, 'log_return'].plot(ax=axis_1)
axis_0.set_title('AAPL 2017-2018 Close price')
axis_1.set_title('AAPL 2017-2018 log Return')
plt.show()

# 1st step: set hypothesis
# we have NULL HYPOTHESIS and ALTERNATIVE HYPOTHESIS
# Null hypothesis is assertion we are against
# Alternative hypothesis is a conclusion we accept whenever we reject the null hypothesis.

# In our example, the Null H is "population mean, the average daily return is 0". (mu=0)
#                 the alternative hypothesis is "average daily return does not equal to 0". (mu!=0)

# In our example, treat t as if it is z distribution when N is large

# calculate z_hat
xbar = aapl['log_return'].mean()
s = aapl['log_return'].std(ddof=1)
n = aapl['log_return'].shape[0]
z_hat = (xbar - 0) / (s / (n**0.5))
print('z_hat is {0}'.format(z_hat))  # z_hat is 2.58966618410296

#%% Set decision criteria
alpha = 0.05
z_left = norm.ppf(alpha / 2, 0, 1)
z_right = -z_left

print('z_left and z_right are {0} at the significance of {1}'.format([z_left, z_right], alpha))
# z_left and z_right are [-1.9599639845400545, 1.9599639845400545] at the significance of 0.05

print('Shall we reject? {0}'.format(z_hat>z_right or z_hat<z_left))
# Shall we reject? True
