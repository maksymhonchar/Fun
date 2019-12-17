#%% Load libraries
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')

import pandas as pd
import numpy as np
from scipy.stats import norm

#%% Mimic 'dice rolling' game to describe what a random variable is

# Random variable - X: The sum of faces (we don't know the outcome)
# Observed outcomes - [2; 12]: The results of the game (we know the outcomes)

cube_die = pd.DataFrame([1, 2, 3, 4, 5, 6])
sum_of_die = cube_die.sample(
    n=2,  # number of samples to return
    replace=True
).sum()
print('Sum of dice is {0}'.format(sum_of_die.loc[0]))

def get_experiment_results(rolls_cnt):
    """Roll 2 die rolls_cnt times; Return sum of faces of 2 die."""

    results = [cube_die.sample(n=2, replace=True).sum().loc[0] for i in range(rolls_cnt)]
    return results

# 50 ovservations
results_50_rolls = get_experiment_results(50)
print(np.mean(results_50_rolls))

#%% Frequency (Relative frequency of observed outcomes) and Distributions

# If we want to compare FREQUENCIES OF DIFFERENT TRIALS - use relative frequency.

# Calculate frequency in collection of outcomes
def get_frequency(experiment_results):
    """Get frequencies and relative frequencies out of experiment results."""

    frequencies = pd.DataFrame(experiment_results)[0].value_counts()
    sorted_frequencies = frequencies.sort_index()
    # relative_frequencies = freq/numb_of_trials
    relative_frequencies = sorted_frequencies / len(experiment_results)  # / sorted_frequencies.sum()
    return sorted_frequencies, relative_frequencies

def plot_freqs(freqs, rel_freqs):
    """Plot frequencies and relative frequencies as bar charts."""
    fig, [axis_0, axis_1] = plt.subplots(1, 2, figsize=(15, 4))
    axis_0.set_title('frequencies')
    freqs.plot(kind='bar', ax=axis_0)
    axis_1.set_title('relative frequencies')
    rel_freqs.plot(kind='bar', ax=axis_1)

freqs, rel_freqs = get_frequency(results_50_rolls)
plot_freqs(freqs, rel_freqs)

results_100_rolls = get_experiment_results(100)
freqs, rel_freqs = get_frequency(results_100_rolls)
plot_freqs(freqs, rel_freqs)

# As we increase num of trials, RELATIVE FREQUENCIES BECOME MORE AND MORE STABLE.

# In this example, distribution is close to a normal distribution
results_1000_rolls = get_experiment_results(1000)
freqs, rel_freqs = get_frequency(results_1000_rolls)
plot_freqs(freqs, rel_freqs)


#%% Evaluate distribution

# If we had a infinite number of trials, the limit would be DISTRIBUTION OF SUM OF FACE X

# Build a distribution table.
X_distribution = pd.DataFrame(
    index=[2,3,4,5,6,7,8,9,10,11,12]  # die faces
)
X_distribution['prob'] = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
X_distribution['prob'] = X_distribution['prob'] / 36  # 36 means (1/6)^2
display(X_distribution)

# Mean == Expectation
# Use 'Mean' and 'Expectation' terms TO DESCRIBE THE DISTRIBUTION OF PROBABILITY

# mean (expectation) = sum(p_i * x_i)
mean = (X_distribution.index * X_distribution['prob']).sum()
display(mean)

# variance = sum( (x_i - mean)^2 * p_i )
variance = ( ((X_distribution.index - mean)**2) * X_distribution['prob'] ).sum()
display(variance)

#%% Distribution for continuous random variable
# Probability == Area under density function

#%% Calculate empirical  mean and variance
results_2000_rolls = get_experiment_results(2000)
results_df = pd.DataFrame(results_2000_rolls)
print(results_df.mean())
print(results_df.var())
print(results_df.std())

#%% Model stock return using normal random variable

aapl = pd.DataFrame.from_csv('data/apple.csv')
aapl.loc['2012-08-01':'2013-08-01', 'Close'].plot()
plt.show()  # QUESTION: probability of dropping over 40% in a year - ?

# Log daily return of Apple
aapl['return_log'] = np.log(
    aapl['Close'].shift(-1) / aapl['Close']
)
aapl['return_log'].hist(bins=50)
plt.show()  # Really similar to normal distribution

# Approximate mean and variance of the log daily return
mu = aapl['return_log'].mean()
sigma = aapl['return_log'].std(ddof=1)
print(mu, sigma)

#%% Get PDF and CDF
# PDF - probability density function
# CDF - cumulative distribution function
density = pd.DataFrame()
density['x'] = np.arange(-4, 4, 0.001)  # could be from -inf to +inf
density['pdf'] = norm.pdf(density['x'], 0, 1)  # mean - 0; std - 1.
density['cdf'] = norm.cdf(density['x'], 0, 1)  # mean - 0; std - 1.

plt.plot(density['x'], density['pdf'])
plt.show()

plt.plot(density['x'], density['cdf'])
plt.show()

#%% What is the chance of losing over 5% in a day?
den_aapl = pd.DataFrame()
den_aapl['x'] = np.arange(-0.1, 0.01, 0.001)
den_aapl['pdf'] = norm.pdf(den_aapl['x'], mu, sigma)

plt.plot(den_aapl['x'], den_aapl['pdf'])
plt.show()

plt.ylim(0, 25)
plt.plot(den_aapl['x'], den_aapl['pdf'])
plt.fill_between(
    x=np.arange(-0.1, -0.01, 0.0001),
    y2=0,
    y1=norm.pdf(np.arange(-0.1, -0.01, 0.0001), mu, sigma)
)
plt.show()  # Blue shape == probability of LOSING 5% IN MORE THAN 1 DAY

prob_return1 = norm.cdf(-0.05, mu, sigma)
print('The probability is {0}'.format(prob_return1))  # The probability is 0.005495344250959501
# 0.005495344250959501 MEANS we have ~0.55% to lose more than 5% in a day

#%% How about probability of dropping over 40% in 1 year (220 trading days?)
# P(Drop over 40% in 220 days) - ?

# Let's make an assumption (for simplification): daily returns are independent (which in the reality is absolutely wrong)

# Then we can get mean and standard deviation of the yearly return.

mu_220 = 220 * mu
sigma_220 = 220**0.5 * sigma  # ?!
print(mu_220, sigma_220)  # 0.2146029070130195 0.2973220365637178

print('The probability of dropping over 40%% in 220 days is: {0}'.format(
    norm.cdf(-0.4, mu_220, sigma_220)
))  # 0.019361015454142615

print('The probability of dropping over 5%% in 220 days is: {0}'.format(
    norm.cdf(-0.05, mu_220, sigma_220)
))  # 0.18674531918074777

#%% Get quantiles
print(norm.ppf(0.05, mu, sigma))  # VaR (Value of Risk) at the level 95%
# -0.03199635945565469 => with 5% chance, the daily return is worse than -3%
 
#%% Conclusion:
# CDF - "Calculate the probability of the stock price will drop over a certain percentage in a year"
# PPF - "VaR (Value of Risk) - with X% chance, the daily return is worse than ppf_result"
# PPF - Quantile