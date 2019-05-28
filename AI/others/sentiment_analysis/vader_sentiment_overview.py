#%% [markdown]
# src: https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f

# VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media.

# VADER not only tells about the Positivity and Negativity score but also tells us about how positive or negative a sentiment is.

# pypi: vaderSentiment

#%% Load libraries.
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#%% Try out VADER
analyzer = SentimentIntensityAnalyzer()

# Use the polarity_scores() method to obtain the polarity indices for the given sentence
def sentiment_analyzer_scores(sentence):
    score = analyzer.polarity_scores(sentence) 
    print('{0}\n{1}'.format(sentence, str(score)))

sentiment_analyzer_scores('The phone is super cool')  # {'neg': 0.0, 'neu': 0.326, 'pos': 0.674, 'compound': 0.7351}

# The Positive, Negative and Neutral scores represent the proportion of text that falls in these categories. This means our sentence was rated as 67% Positive, 33% Neutral and 0% Negative. Hence all these should add up to 1.

# The Compound score is a metric that calculates the sum of all the lexicon ratings which have been normalized between -1(most extreme negative) and +1 (most extreme positive). In the case above, lexicon ratings for andsupercool are 2.9and respectively1.3. The compound score turns out to be 0.75 , denoting a very high positive sentiment.

# positive sentiment: compound score >= 0.05
# neutral sentiment: compound score is (-0.05; 0.05)
# negative sentiment: compound score is <= -0.05

# VADER scoring methodology:
# https://github.com/cjhutto/vaderSentiment#about-the-scoring

sentiment_analyzer_scores('I hate this product, Apple simply sucks')

#%% Try out several tweets from twitter.com/bitcointrader
sentiment_analyzer_scores("""
Current #Bitcoin run very similar to Jan-Mar 2013 run ($7->$266->$90; $700-$20k->$9k) with backlogged new accounts, etc. then Silk Road catalyst (salable supply shock) for $100->$1,200. Mountain of #WallStreet money lining up. What are potential 2018 $BTC scenarios? @ToneVays? 😎
""")
sentiment_analyzer_scores("""
Reality check: can't get BCH on to either Bittrex or Kraken.  Price isn't real if sellers can't sell.
""")
sentiment_analyzer_scores("""
Reminder: Bitcoin's biggest value is in ability to own money securely.
""")
sentiment_analyzer_scores("""
Folks who feel they missed the bitcoin boat are going to get a lesson in how small of a number 21,000,000 is relative to global population.
""")
sentiment_analyzer_scores("""
more empty blocks than statistically expected from SPY mining, likely indicates covert asicBoost. and SPY mining is bad for security itself.
""")

#%% Punctuation examples
# The use of an exclamation mark(!), increases the magnitude of the intensity without modifying the semantic orientation.

sentiment_analyzer_scores('The food here is good')
sentiment_analyzer_scores('The food here is good!')
sentiment_analyzer_scores('The food here is good!!')
sentiment_analyzer_scores('The food here is good!!!')

#%% Capitalization example
sentiment_analyzer_scores('The food here is great')
sentiment_analyzer_scores('The food here is GREAT!')

#%% Degree modifiers
# Also called intensifiers, they impact the sentiment intensity by either increasing or decreasing the intensity.
sentiment_analyzer_scores('The service here is good')
sentiment_analyzer_scores('The service here is extremely good')
sentiment_analyzer_scores('The service here is marginally good')

#%% Conjunctions
# Use of conjunctions like “but” signals a shift in sentiment polarity, with the sentiment of the text following the conjunction being dominant.
sentiment_analyzer_scores('The food here is great, but the service is horrible')

#%% Preceding Tri-gram
# By examining the tri-gram preceding a sentiment-laden lexical feature, we catch nearly 90% of cases where negation flips the polarity of the text
sentiment_analyzer_scores('The food here is not really that great')

#%% Emojis
print(sentiment_analyzer_scores('I am 😄 today'))
print(sentiment_analyzer_scores('😊'))
print(sentiment_analyzer_scores('😥'))
print(sentiment_analyzer_scores('☹️'))

#%% Slangs
print(sentiment_analyzer_scores("Today SUX!"))
print(sentiment_analyzer_scores("Today only kinda sux! But I'll get by, lol"))

#%% Emoticons
print(sentiment_analyzer_scores("Make sure you :) or :D today!"))
