# week2 overview
- main question: how to compute the chance of bankruptcy if applying strategy of comparison of moving averages?
- Hence MA10 and MA50 are random variables, what is the distribution of these variables (what is a probability rule?)
- Plan
    - what random variable is?
    - describe the distribution of random variables
    - after knowing the distribution of random variables, apply this concept to measure the risk of investing money in AAPL stock.

# Outcomes and Random Variables
- Main example: rolling two dice
    - possible faces: 1 to 6, all with equal chance. THIS is a random variable.
    - process: mimic the game (make N ovservation (N trials)) -> calculate and record the sum for face values of 2 dice for each trial.
    - In this game, sum (we denote as X) is a RANDOM VARIABLE
        - RANDOMNESS OF VARIABLE means that before we roll a die, we aren't sure which outcome we can get.
    - BUT we know the collection of outcomes, which ranged from 2 to 12 (sum of 2 dice)
        - The collection of outcomes IS NOT a random variable.
        - Instead, they are REALIZED or OBSERVED OUTCOMES of values of X.
            - Outcome == result of the game.
    - Possible outcomes include integers from [2; 12] == Discrete random variable
        - Daily return of a stock price == Continuous random variable

# Frequency, Relative Frequency of observed outcomes, Distribution
- Frequency:
    - DataFrame.value_counts() - frequency of a DF. Index is a list of different outcomes; Value column holds frequency values.
    - Don't forget to sort index to get observable results
- To compare the frequency of different trials, convert frequency into RELATIVE FREQUENCY
    - rel_freq = freq / num_of_trials
    - shape of barchart doesn't change, but OY axis changes
- As we increase number of trials, the bar chart goes toward a limit.
    - == the relative frequency becomes more and more stable.
- Distribution of a random variable is a table, which consists of 2 sets of values:
    1. different values of outcome
    2. probability for each value
    - Note: from distribution table alone, we don't know the shape of outcome.
- Distribution
    - Distribution helps identify extreme values of events
    - Mean and variance are two characteristics of the distribution of random variables.
        - Mean is also called "Expectation"
        - Mean is average of all outcomes weighted by probabilities.
- Distribution of continuous random variables - density function.
- Distribution of discrete random variable - table of distribution.
- PDF (probability density function)
    - NOTE: PDF is NOT probability
        - probability is the area below the PDF
    - PDF characterises by mean and standard deviation

# Models of distribution
- Example: stock price fo AAPL drops over 40% from AUG 2012 to MAY 2013
    - WE NEED TO compute what's the chance that the yearly return can be less than -40%.
- z-distribution: mean=0, std=1

# CDF (cumulative distribution function)
- Outputs the probability for the area, and LOWER side of each possible value
- Answers the question "What is the chance to lose more than 5% in certain period of time?"

# PPF (percent point function)
- Use it to obtain quantiles of a normal distribution.

# VaR (Value of Risk)
- It estimates how much a set of investments might lose with a given probability.
- Is used by firms and the regulators in the financial industry to estimate the amount of assess needed to cover possible loss.
- Example:
    - 5% of quantile of daily return is called a 95% VaR (or VaR at the level of 95%)
        - 5% of quantile = -0.03 => with the 5% chance that daily return is worse than 3%.

# Fama and French about using normal distribution to model stock return
- Distribution of a daily and monthly stock return are rather symmetric about their means, but the tails are fatter.
    - Which means there are more outliers that would be expected with normal distributions.
    - It means that, if tail returns negative, as well as positive, may occur more often than we expect.
- To model a fat tail, people proposed model return using t-distributions with low degree of freedom