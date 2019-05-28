#%% [markdown]
# https://pythonprogramming.net/sentiment-analysis-python-textblob-vader/

#%% Load libraries
from textblob import TextBlob

#%% Check if TextBlob performs well
polarity_threshold = [0, 0.1, 0.2, 0.5]
def try_polarity_threshold(threshold_value):
    pos_count = 0
    pos_correct = 0
    with open("/home/max/Documents/learn/learnai/short_movies_reviews_sentiment/positive.txt", "r", encoding='ISO-8859-1') as f:
        for line in f.read().split('\n'):
            analysis = TextBlob(line)
            if analysis.sentiment.polarity > threshold_value:
                pos_correct += 1
            pos_count += 1
    neg_count = 0
    neg_correct = 0
    with open("/home/max/Documents/learn/learnai/short_movies_reviews_sentiment/negative.txt", "r", encoding='ISO-8859-1') as f:
        for line in f.read().split('\n'):
            analysis = TextBlob(line)
            if analysis.sentiment.polarity <= 0:
                neg_correct += 1
            neg_count += 1
    print('Polarity threshold is {0}'.format(threshold_value))
    print("Positive accuracy = {}% via {} samples".format(pos_correct/pos_count*100.0, pos_count))
    print("Negative accuracy = {}% via {} samples".format(neg_correct/neg_count*100.0, neg_count))

for pt in polarity_threshold:
    try_polarity_threshold(pt)
# 0 - 71.1 and 55.9
# 0.1 - 60.5 and 55.9
# 0.2 - 46.27 and 55.9
# 0.5 - 9.22 and 55.9

#%% Add subjectivity to analysis
def try_subjectivity_polarity(subj_thr, pol_thr):
    pos_count = 0
    pos_correct = 0
    with open("/home/max/Documents/learn/learnai/short_movies_reviews_sentiment/positive.txt", "r", encoding='ISO-8859-1') as f:
        for line in f.read().split('\n'):
            analysis = TextBlob(line)
            if analysis.sentiment.subjectivity < subj_thr:
                if analysis.sentiment.polarity > pol_thr:
                    pos_correct += 1
                pos_count +=1
    neg_count = 0
    neg_correct = 0
    with open("/home/max/Documents/learn/learnai/short_movies_reviews_sentiment/negative.txt", "r", encoding='ISO-8859-1') as f:
        for line in f.read().split('\n'):
            analysis = TextBlob(line)
            if analysis.sentiment.subjectivity < subj_thr:
                if analysis.sentiment.polarity <= pol_thr:
                    neg_correct += 1
                neg_count +=1
    print('subj_thr is {0}; pol_thr is {1}'.format(subj_thr, pol_thr))
    print("Positive accuracy = {}% via {} samples".format(pos_correct/pos_count*100.0, pos_count))
    print("Negative accuracy = {}% via {} samples".format(neg_correct/neg_count*100.0, neg_count))

try_subjectivity_polarity(0.1, 0)  # 2.90, 98.1
try_subjectivity_polarity(0.3, 0)  # 29.08, 76.03
try_subjectivity_polarity(0.9, 0)  # 70.5, 55.1

#%% Try out VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
vs = analyzer.polarity_scores("VADER Sentiment looks interesting, I have high hopes!")  # {'neg': 0.0, 'neu': 0.463, 'pos': 0.537, 'compound': 0.6996}
# compound is "computed by summing the valence scores of each word in the lexicon, adjusted according to the rules, then normalized... [it] is the most useful metric if you want a single unidimensional measure of sentiment."

def try_vader(compound_threshold):
    pos_count = 0
    pos_correct = 0
    with open("/home/max/Documents/learn/learnai/short_movies_reviews_sentiment/positive.txt", "r", encoding='ISO-8859-1') as f:
        for line in f.read().split('\n'):
            vs = analyzer.polarity_scores(line)
            if vs['compound'] >= compound_threshold or vs['compound'] <= -compound_threshold:
                pos_correct += 1
            pos_count +=1
    neg_count = 0
    neg_correct = 0
    with open("/home/max/Documents/learn/learnai/short_movies_reviews_sentiment/negative.txt", "r", encoding='ISO-8859-1') as f:
        for line in f.read().split('\n'):
            vs = analyzer.polarity_scores(line)
            if vs['compound'] >= compound_threshold or vs['compound'] <= -compound_threshold:
                neg_correct += 1
            neg_count +=1
    print('VADER compound threshold is {0}'.format(compound_threshold))
    print("Positive accuracy = {}% via {} samples".format(pos_correct/pos_count*100.0, pos_count))
    print("Negative accuracy = {}% via {} samples".format(neg_correct/neg_count*100.0, neg_count))

try_vader(0)  # 69.5; 57.8
try_vader(0.5)
