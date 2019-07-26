#!/usr/bin/env python
# coding: utf-8

# In[38]:


# Load libraries

import re
import csv

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from wordcloud import WordCloud, STOPWORDS

import pandas as pd

import numpy as np

from scipy import stats

from bs4 import BeautifulSoup

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier, NLTKClassifier, DecisionTreeClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


# In[2]:


# Load data

def load_tsv_data(file_path):
    return pd.read_csv(file_path, delimiter='\t', quoting=csv.QUOTE_NONE, header=0)

train_df = load_tsv_data("data/labeledTrainData.tsv")

test_df = load_tsv_data("data/testData.tsv")

unlabeled_df = load_tsv_data("data/unlabeledTrainData.tsv")


# In[3]:


# Overview loaded data

def overview_dataset(dataset_df):
    # Data inside
    display(dataset_df.head(3))
    display(dataset_df.tail(3))
    # Dimensions and size
    display(dataset_df.shape)
    # Columns names
    display(dataset_df.columns.values)
    # Duplicated values
    display(dataset_df[dataset_df.duplicated(keep=False)])
    # .describe()
    display(dataset_df.describe(include='all').T)


# In[4]:


overview_dataset(train_df)

overview_dataset(test_df)

overview_dataset(unlabeled_df)


# In[5]:


# Explore the data

# Explore sentiments of the reviews

display(
    train_df['sentiment'].value_counts()
)

display(
    train_df.groupby('sentiment')['review'].describe()
)


# In[6]:


# Explore length of the reviews

train_df['rev_len'] = train_df['review'].apply(len)


# In[7]:


display(train_df['rev_len'].describe())

# Display distribution of the reviews by review length
train_df['rev_len'].hist(bins=100)
plt.show()

# Display distribution of the reviews by review length and by sentiment score
train_df.hist(column='rev_len', by='sentiment', bins=100, figsize=(15, 5))
plt.show()

# Display Kolmogorov-Smirnov statistic
# From scipy docs:
    # If the K-S statistic is small or the p-value is high,
    # then we cannot reject the hypothesis that
    # the distributions of the two samples are the same.
grouped_by_sentiment = train_df.groupby('sentiment')['rev_len']
display(
    stats.ks_2samp(
        grouped_by_sentiment.get_group(0),
        grouped_by_sentiment.get_group(1)
    )
)  # statistic=0.027760000000000007, pvalue=0.0001310970303242206

# Conclusion: reject the hypothesis that the distributions are the same.


# In[8]:


# Clean the data

wnlemmatizer = WordNetLemmatizer()

def clean_review(raw_text, to_lower, lemmatize, remove_numbers, remove_stopwords):
    # 1
    text_nohtml = BeautifulSoup(raw_text).get_text()
    # 2
    if remove_numbers:
        re_clean_pattern = "[^a-zA-Z]"
    else:
        re_clean_pattern = "[^a-zA-Z0-9]"
    text_regexclean = re.sub(re_clean_pattern, " ", text_nohtml)
    # 3
    if to_lower:
        text_tokens = text_regexclean.lower().split(" ")
    else:
        text_tokens = text_regexclean.split(" ")
    # 4
    if remove_stopwords:
        nltk_stopwords = set(stopwords.words("english"))
        text_tokens_nostopwords = [
            token for token in text_tokens
            if token not in nltk_stopwords
        ]
        text_tokens = text_tokens_nostopwords
    # 5
    if lemmatize:
        text_lemmatized_tokens = [wnlemmatizer.lemmatize(token) for token in text_tokens]
        text_lemmatized_tokens = [wnlemmatizer.lemmatize(token, "v") for token in text_lemmatized_tokens]
        text_tokens = text_lemmatized_tokens
    # 6
    text_cleaned = " ".join(text_tokens)
    return text_cleaned    


# In[9]:


# Apply data cleaning to datasets;
# Create columns to describe amount of tokens and length of cleaned review

to_lower = True
lemmatize = True
remove_numbers = True
remove_stopwords = True

train_df['cleaned_review'] = train_df['review'].apply(
    lambda x: clean_review(
        x,
        to_lower=to_lower,
        lemmatize=lemmatize,
        remove_numbers=remove_numbers,
        remove_stopwords=remove_stopwords
    )
)

train_df['cln_rev_len'] = train_df['cleaned_review'].apply(
    lambda x: len(' '.join(x))
)

train_df['cln_rev_tokens_len'] = train_df['cleaned_review'].apply(len)


# In[10]:


display(train_df.describe())

# Explore created clb_rev_len feature

# Display distribution of the reviews by cleaned review length
train_df['cln_rev_len'].hist(bins=100)
plt.show()
# Display distribution of the reviews by cleaned review length and by sentiment score
train_df.hist(column='cln_rev_len', by='sentiment', bins=100, figsize=(15, 5))
plt.show()
# Display Kolmogorov-Smirnov statistic
grouped_by_sentiment = train_df.groupby('sentiment')['cln_rev_len']
display(
    stats.ks_2samp(
        grouped_by_sentiment.get_group(0),
        grouped_by_sentiment.get_group(1)
    )
)  # statistic=0.030240000000000045, pvalue=2.171357711776904e-05

# Conclusion: reject the hypothesis that the distributions are the same.


# In[11]:


# Explore created cln_rev_tokens_len feature

# Display distribution of the reviews by tokens cnt from cleaned review length
train_df['cln_rev_tokens_len'].hist(bins=100)
plt.show()
# Display distribution of the reviews by tokens cnt and by sentiment score
train_df.hist(column='cln_rev_tokens_len', by='sentiment', bins=100, figsize=(15, 5))
plt.show()
# Display Kolmogorov-Smirnov statistic
grouped_by_sentiment = train_df.groupby('sentiment')['cln_rev_tokens_len']
display(
    stats.ks_2samp(
        grouped_by_sentiment.get_group(0),
        grouped_by_sentiment.get_group(1)
    )
)  # statistic=0.030240000000000045, pvalue=2.171357711776904e-05

# Conclusion: reject the hypothesis that the distributions are the same.


# In[12]:


plt.figure(figsize=(5, 5))
sns.heatmap(train_df.corr(), annot=True)
plt.show()


# In[13]:


# Display cloud of words for the datasets

def display_word_cloud(dataset_df, col_name):
    wordcloud_obj = WordCloud(width=1920, height=1080)
    wordcloud_img = wordcloud_obj.generate(
        ' '.join(dataset_df.loc[:, col_name])
    )
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud_img)
    plt.axis('off')
    plt.show()


# In[14]:


display_word_cloud(train_df, 'review')

display_word_cloud(train_df, 'cleaned_review')


# In[15]:


# Function to display statistics on predicted values

def display_y_pred_stats(y_true, y_pred):
    # Display confusion matrix
    cm = pd.crosstab(y_true, y_pred)
    TN = cm.iloc[0, 0]
    FN = cm.iloc[1, 0]
    TP = cm.iloc[1, 1]
    FP = cm.iloc[0 ,1]
    display("Confusion matrix", cm)
    # Display accuracy metrics
    display("Accuracy (custom) is {0}".format(round(((TP+TN)*100)/(TP+TN+FP+FN), 2)))
    display("Accuracy (sklearn) is {0}".format(accuracy_score(y_true, y_pred)))
    display("FN rate: {0}".format(round((FN*100)/(FN+TP), 2)))
    display("FP rate: {0}".format(round((FP*100)/(FP+TN), 2)))
    display("F1 score is {0}".format(f1_score(y_true, y_pred)))    
    # Display classification report
    print(classification_report(y_true, y_pred))


# In[16]:


# Functions to try out BOW and TfIdf

def vectorize_df_col(dataset_df, col_name, perform_tfidf=False, cv_max_features=None):
    """Vectorize dataset_df[col_name] using BOW algorithm.
    Apply Tf-idf transofrmation after that
    """
    vectorized_words = CountVectorizer(max_features=cv_max_features).fit_transform(dataset_df[col_name])
    if perform_tfidf:
        normalized_words = TfidfTransformer().fit_transform(vectorized_words)
        return normalized_words
    return vectorized_words

def apply_model(X_train_full, y_train_full, model, display_stats=False, return_validat=False):
    X_train, X_validat, y_train, y_validat = train_test_split(
        X_train_full, y_train_full,
        test_size=0.35, random_state=42
    )
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_validat)
    if display_stats:
        display_y_pred_stats(y_validat, y_pred)  
    if return_validat:
        return (y_validat, y_pred)
    return y_pred


# In[17]:


# Approach 1: use raw reviews (don't clean them) to predict sentiment
# NOTE: try out both BOW+tfidf and BOW (== no tf-idf) approaches

models = [
    LogisticRegression(solver='saga', max_iter=10000),
#     MultinomialNB(),
#     RandomForestClassifier(n_estimators=500, n_jobs=-1, verbose=2)
]
X_train_full = vectorize_df_col(train_df, 'review', perform_tfidf=False, cv_max_features=10000)
y_train_full = train_df['sentiment']
for model in models:
    y_pred = apply_model(X_train_full, y_train_full, model, display_stats=True)
    
# Results: accuracy with tf-idf
# LogReg, solver=saga: 89.03
# MultinomialNB: 85.82
# RandomForestClf: n_est=100: 84.02; n_est=500: 85.04

# Results: accuracy without tf-idf
# LogReg: solver=saga: 88.64; solver=liblinear: 87.82; 
# MultinomialNB: 84.43
# RandomForestClf: n_estimators=100: 84.65; n_estimators=500: 86.06


# In[18]:


# Approach 2: apply BOW + tf-idf transformation to the cleaned text 
# NOTE: cleaned data == all params True

models = [
    LogisticRegression(solver='saga', max_iter=10000),
#     MultinomialNB(),
#     RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=2)
]
X_train_full = vectorize_df_col(train_df, 'cleaned_review', perform_tfidf=True, cv_max_features=8000)
y_train_full = train_df['sentiment']
for model in models:
    y_pred = apply_model(X_train_full, y_train_full, model, display_stats=True)
    
# Results: accuracy with 0.35 of train_set size for validation set
# LogisticRegression lbfgs, newton-cg, liblinear, sag, saga: 89.01
# MultinomialNB: 86.95
# RandomForestClassifier: 86.77


# In[19]:


# Approach 3: use length to predict text sentiment
# NOTE: cleaned data == all params True

X_train_full = train_df.loc[:, ['rev_len', 'cln_rev_len', 'cln_rev_tokens_len']]
y_train_full = train_df['sentiment']
model = LogisticRegression(solver='saga')
y_pred = apply_model(X_train_full, y_train_full, model, display_stats=True)

# Results: accuracy with 0.35 of train_set size for validation set
# LogisticRegression: ~[56.30; 56.4]
# MultinomialNB: 56.43
# RandomForestClassifier: 52.18


# In[20]:


# Approach 4: play with BOW hyperparameters, no TfIdf
# NOTE: cleaned data == all params True
# NOTE: tuning TfidfTransformer didn't have any positive outcome

n_max_features = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 5000, 7500, 10000, 15000, 20000, 50000]
accuracy_values = []
f1_score_values = []

for n_max_features_value in n_max_features:
    X_train_full = vectorize_df_col(train_df, 'cleaned_review',
                                    perform_tfidf=True,
                                    cv_max_features=n_max_features_value)
    y_train_full = train_df['sentiment']
    model = LogisticRegression(solver='liblinear')
    y_true, y_pred = apply_model(X_train_full, y_train_full, model,
                                 display_stats=False, return_validat=True)
    accuracy_values.append(accuracy_score(y_true, y_pred))
    f1_score_values.append(f1_score(y_true, y_pred))
    print("dbg: solved for {0} param".format(n_max_features_value))
    
plt.plot(n_max_features, accuracy_values)
plt.title("LogisticRegression(solver='liblinear')")
plt.xlabel("n_max_features for CountVectorizer")
plt.ylabel("accuracy")
plt.show()

plt.plot(n_max_features, f1_score_values)
plt.title("LogisticRegression(solver='liblinear')")
plt.xlabel("n_max_features for CountVectorizer")
plt.ylabel("f1 score")
plt.show()

display(max(accuracy_values), max(f1_score_values))

# Conclusion: n_max_features=10000 & turned on tf-idf transformation is fine


# In[21]:


# Approach 5: use VADER

def get_discrete_sentiment_score_vader(text):
    """Return 0 or 1, depending on compound score.
    Note: positive sentiment: score>=0.05;
          negative sentiment: score<=-0.05;
          use value random from {0, 1} for neutral sentiment: -0.05<=score<=0.05
    """
    compound_score = vader_analyzer.polarity_scores(text).get('compound')
    if compound_score >= 0.05:
        return 1
    elif compound_score <= -0.05:
        return 0
    else:
        return np.random.randint(0, 2)

def try_vader():
    # Use VADER to predict sentiment for data in train set
    vader_analyzer = SentimentIntensityAnalyzer()

    train_df['vader_sentiment_raw'] = train_df['review'].apply(
        lambda x: get_discrete_sentiment_score_vader(x)
    )
    train_df['vader_sentiment_cln'] = train_df['cleaned_review'].apply(
        lambda x: get_discrete_sentiment_score_vader(x)
    )

    # Estimate VADER accuracy

    display_y_pred_stats(train_df['sentiment'], train_df['vader_sentiment_raw'])  # acc: 69.25

    display_y_pred_stats(train_df['sentiment'], train_df['vader_sentiment_cln'])  # acc: 67.33

# Conclusion: VADER doesn't perform well for train set - don't use it in final submission


# In[22]:


# Approach 6: use default version of TextBlob

def get_discrete_sentiment_score_textblob(text):
    """Return 0 or 1, depending on sentiment score.
    Note: positive sentiment: score>=0;
          negative sentiment: score<0;
    """
    sentiment_score = TextBlob(text).sentiment.polarity
    return 1 if sentiment_score >= 0 else 0

def try_textblob():
    # Use TextBlob to predict sentiment for data in train set
    train_df['textblob_sentiment_raw'] = train_df['review'].apply(
        lambda x: get_discrete_sentiment_score_textblob(x)
    )
    train_df['textblob_sentiment_cln'] = train_df['cleaned_review'].apply(
        lambda x: get_discrete_sentiment_score_textblob(x)
    )

    display_y_pred_stats(train_df['sentiment'], train_df['textblob_sentiment_raw'])  # acc: 68.5

    display_y_pred_stats(train_df['sentiment'], train_df['textblob_sentiment_cln'])  # acc: 68.59

# Conclusion: default TextBlob doesn't perform well for train set - don't use it in final submission


# In[23]:


# Approach 6: use customized TextBlob

def try_customized_textblob():
    
    train_df['sentiment_posneg'] = train_df['sentiment'].apply(
        lambda x: "pos" if x == 1 else "neg"
    )

    textblob_train_data_rawreview = [
        tuple(row)
        for row in train_df.loc[:, ['review', 'sentiment_posneg']].values
    ]

    textblob_train_data_clnreview = [
        tuple(row)
        for row in train_df.loc[:, ['cleaned_review', 'sentiment_posneg']].values
    ]
    
    # textblob_nb_clf_rawreview = NaiveBayesClassifier(textblob_train_data_rawreview[:1000])  # 38% MEM
    # del textblob_nb_clf_rawreview

    # textblob_nltk_clf_rawreview = NLTKClassifier(textblob_train_data_rawreview[:1000])  # 34% MEM
    # textblob_dtree_clf_rawreview = DecisionTreeClassifier(textblob_train_data_rawreview[:1000])  # 56% MEM
    
    # train_df['textblob_nb_raw'] = train_df['review'].apply(
    #     lambda x: 1 if textblob_nb_clf_rawreview.classify(x) == "pos" else 0
    # )

    # train_df['textblob_nb_raw'] = train_df['review'].apply(
    #     lambda x: 1 if textblob_nb_clf_rawreview.classify(x) == "pos" else 0
    # )
    
    # textblob_nb_clf_clnreview = NaiveBayesClassifier(textblob_train_data_clnreview[:1000])  # 71% MEM
    # textblob_nltk_clf_clnreview = NLTKClassifier(textblob_train_data_clnreview[:1000])  # 75% MEM
    # textblob_dtree_clf_clnreview = DecisionTreeClassifier(textblob_train_data_clnreview[:1000])  # 85% MEM

    # after that: use .prob_classify OR .classify

# Conclusion: because memory usage is too high for only 1000 rows (out of 25000) - skip this approach


# In[33]:


# Approach 7: try to clean data differently: with/without lowering/lemmatizing/stopwords_removal/
ч
n_max_features = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 5000, 7500, 10000, 15000]
accuracy_values = []
f1_score_values = []

train_df['cleaned_review'] = train_df['review'].apply(
    lambda x: clean_review(
        x,
        to_lower=True, lemmatize=False, remove_numbers=False, remove_stopwords=False
    )
)

for n_max_features_value in n_max_features:
    X_train_full = vectorize_df_col(train_df, 'cleaned_review', perform_tfidf=True, cv_max_features=n_max_features_value)
    y_train_full = train_df['sentiment']
    model = LogisticRegression(solver='liblinear')
    y_true, y_pred = apply_model(X_train_full, y_train_full, model, display_stats=False, return_validat=True)
    accuracy_values.append(accuracy_score(y_true, y_pred))
    f1_score_values.append(f1_score(y_true, y_pred))
    print("dbg: solved for {0} param".format(n_max_features_value))
    
plt.plot(n_max_features, accuracy_values)
plt.title("LogisticRegression(solver='liblinear')")
plt.xlabel("n_max_features for CountVectorizer")
plt.ylabel("accuracy")
plt.show()

plt.plot(n_max_features, f1_score_values)
plt.title("LogisticRegression(solver='liblinear')")
plt.xlabel("n_max_features for CountVectorizer")
plt.ylabel("f1 score")
plt.show()

display(max(accuracy_values), max(f1_score_values))


# In[46]:


# Approach 7: try out tfidf vectorizer

train_df = load_tsv_data("data/labeledTrainData.tsv")

test_df = load_tsv_data("data/testData.tsv")

tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),
    use_idf=1,
    smooth_idf=1,
    sublinear_tf=1,
    stop_words='english'
)

train_df['cleaned_review'] = train_df['review'].apply(
    lambda x: clean_review(
        x,
        to_lower=True, lemmatize=False, remove_numbers=True, remove_stopwords=False
    )
)

test_df['cleaned_review'] = test_df['review'].apply(
    lambda x: clean_review(
        x,
        to_lower=True, lemmatize=False, remove_numbers=True, remove_stopwords=False
    )
)

train_vectorized_reviews = tfidf_vectorizer.fit_transform(train_df['cleaned_review'])
test_vectorized_reviews = tfidf_vectorizer.transform(test_df['cleaned_review'])

clf = MultinomialNB()
clf.fit(train_vectorized_reviews, train_df['sentiment'])
pred = clf.predict(test_vectorized_reviews)

display(pred)

df = pd.DataFrame({"id": test_df['id'],"sentiment": pred})

df.to_csv('submission.csv', index=False, header=True)


# In[45]:


# Apply transformations to test set and create a prediction

test_df['cleaned_review'] = test_df['review'].apply(
    lambda x: clean_review(
        x,
        to_lower=True, lemmatize=False, remove_numbers=True, remove_stopwords=False
    )
)

model = LogisticRegression(solver='liblinear')
model.fit(
    vectorize_df_col(train_df, 'cleaned_review', perform_tfidf=True, cv_max_features=10000),
    train_df['sentiment']
)

y_pred = model.predict(
    vectorize_df_col(test_df, 'review', perform_tfidf=True, cv_max_features=10000)
)


# In[35]:


# Submit predictions

output = pd.DataFrame(
    {'id': test_df['id'], 'sentiment': y_pred}
)

output.to_csv('submission.csv', index=False, quoting=3)


# In[26]:


# todo:

# try out keras NN approaches from other kernels
# https://www.kaggle.com/amreshtech/imdb-reviews-with-keras-95-58-accuracy
# https://www.kaggle.com/abhijeet0101/imdb-review-deep-model-93-51-accuracy
# https://www.kaggle.com/sumitdua10/imdb-text-classification-dropout-comparison
# https://www.kaggle.com/sumitdua10/imdb-text-classification-ensemble
# https://www.kaggle.com/shwetabh123/assignementproblem

# check how to use unlabeled data
# check if unlabeled_set contains values for train_set - opportunity to extend training set!!!

