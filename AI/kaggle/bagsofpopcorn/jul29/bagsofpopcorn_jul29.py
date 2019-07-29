#!/usr/bin/env python
# coding: utf-8

# In[74]:


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

import nltk
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
from sklearn.svm import LinearSVC


# In[86]:


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


# In[40]:


# Clean the data

wnlemmatizer = WordNetLemmatizer()

def clean_review(raw_text, to_lower, lemmatize, remove_numbers, remove_stopwords, return_tokens=False):
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
    if return_tokens:
        return text_tokens
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


# In[24]:


# Approach 7: try to clean data differently: with/without lowering/lemmatizing/stopwords_removal/

n_max_features = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 5000, 7500, 10000, 15000]
accuracy_values = []
f1_score_values = []

train_df['cleaned_review'] = train_df['review'].apply(
    lambda x: clean_review(
        x,
        to_lower=True, lemmatize=False, remove_numbers=False, remove_stopwords=False  # best combination
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


# In[25]:


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


# In[27]:


# Apply transformations to test set and create a prediction

# test_df['cleaned_review'] = test_df['review'].apply(
#     lambda x: clean_review(
#         x,
#         to_lower=True, lemmatize=False, remove_numbers=True, remove_stopwords=False
#     )
# )

# model = LogisticRegression(solver='liblinear')
# model.fit(
#     vectorize_df_col(train_df, 'cleaned_review', perform_tfidf=True, cv_max_features=10000),
#     train_df['sentiment']
# )

# y_pred = model.predict(
#     vectorize_df_col(test_df, 'review', perform_tfidf=True, cv_max_features=10000)
# )

# # Submit predictions

# output = pd.DataFrame(
#     {'id': test_df['id'], 'sentiment': y_pred}
# )

# output.to_csv('submission.csv', index=False, quoting=3)


# In[28]:


# src: https://www.kaggle.com/varun08/sentiment-analysis-using-word2vec

# NOTE: performance of word2vec is much better when applying to big datasets.
    # In this example, because we are considering only 25,000 training examples, the
    # performance is similiar to the BOW approach


# In[48]:


# Create list of lists for word2vec

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def split_clean_review(raw_text, tokenizer, to_lower, lemmatize, remove_numbers, remove_stopwords):
    raw_sentences = tokenizer.tokenize(raw_text.strip())
    cleaned_sentences = list()
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            cleaned_sentences.append(
                clean_review(
                    raw_sentence, return_tokens=True,
                    to_lower=to_lower, lemmatize=lemmatize, remove_numbers=remove_numbers, remove_stopwords=remove_stopwords
                )
            )
    return cleaned_sentences


# In[49]:


# Create list of lists for word2vec: list of sentences

sentences = list()
for review in train_df['review']:
    sentences += split_clean_review(
        review, tokenizer,
        True, False, True, False
    )


# In[50]:


display(len(sentences[0]))

display(sentences[0])


# In[51]:


# Creating the model and setting values for the various parameters
num_features = 300  # Word vector dimensionality
min_word_count = 40 # Minimum word count
num_workers = 4     # Number of parallel threads
context = 10        # Context window size
downsampling = 1e-3 # (0.001) Downsample setting for frequent words

# Initializing the train model
from gensim.models import word2vec
print("Training model....")
model = word2vec.Word2Vec(sentences,                          workers=num_workers,                          size=num_features,                          min_count=min_word_count,                          window=context,
                          sample=downsampling)

# To make the model memory efficient
model.init_sims(replace=True)

# Saving the model for later use. Can be loaded using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)


# In[70]:


def featureVecMethod(words, model, num_features):
    """Average all word vectors in a paragraph"""
    featureVec = np.zeros(num_features, dtype="float32")
    nwords = 0    
    index2word_set = set(model.wv.index2word)  # set() for speed purposes
    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    """Calculating the average feature vector"""
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for idx, review in enumerate(reviews):
        if idx%1000 == 0:
            print(idx)
        reviewFeatureVecs[idx] = featureVecMethod(review, model, num_features)
    return reviewFeatureVecs


# In[71]:


# Get average feature vector for training set

clean_train_reviews = []
for review in train_df['review']:
    clean_train_reviews.append(
        clean_review(review, True, False, True, True)
    )
    
trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)


# In[72]:


# Get average feature vector for test set

clean_test_reviews = []
for review in test_df['review']:
    clean_test_reviews.append(
        clean_review(review, True, False, True, True)
    )
    
testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)


# In[73]:


model_rndforest = RandomForestClassifier(n_estimators=100)

model_rndforest = model_rndforest.fit(trainDataVecs, train_df['sentiment'])

y_pred = model_rndforest.predict(testDataVecs)


# In[75]:


# Submit predictions

output = pd.DataFrame(
    {'id': test_df['id'], 'sentiment': y_pred}
)

output.to_csv('submission.csv', index=False)


# In[78]:


display(test_df.shape)

display(y_pred.shape)

display(train_df.head(5))
display(test_df.head(5))


# In[82]:


# Try out LinearSVC

stop_words = ['in', 'of', 'at', 'a', 'the']

ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)

ngram_vectorizer.fit(train_df['cleaned_review'])

X = ngram_vectorizer.transform(train_df['cleaned_review'])

X_test = ngram_vectorizer.transform(test_df['cleaned_review'])


# In[83]:


X_train, X_val, y_train, y_val = train_test_split(
    X, train_df['sentiment'], train_size = 0.75
)

for c in [0.001, 0.005, 0.01, 0.05, 0.1]:    
    svm = LinearSVC(C=c)
    svm.fit(X_train, y_train)
    print("Accuracy for C={0}: {1}".format(c, accuracy_score(y_val, svm.predict(X_val))))
    
# Accuracy for C=0.001: 0.88544
# Accuracy for C=0.005: 0.89088
# Accuracy for C=0.01: 0.88992
# Accuracy for C=0.05: 0.8896
# Accuracy for C=0.1: 0.88944


# In[84]:


# src: https://www.kaggle.com/drscarlat/imdb-sentiment-analysis-keras-and-tensorflow

# Import keras and tensorflow libraries

import tensorflow as tf

from keras import models, regularizers, layers, optimizers, losses, metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils, to_categorical
 
from keras.datasets import imdb


# In[93]:


# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# restore np.load for future normal usage
np.load = np_load_old


# In[99]:


# Vectorize inputs.
# Encoding the integer sequences into a binary matrix - one hot encoder basically
# From integers representing words, at various lengths - to a normalized one hot encoded tensor (matrix) of 10k columns

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# In[106]:


X_train = vectorize_sequences(train_data)
X_test = vectorize_sequences(test_data)

print("x_train ", X_train.shape, X_train.dtype)
print("x_test ", X_test.shape, X_train.dtype)


# In[107]:


y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

print("y_train", y_train.shape, y_train.dtype)
print("y_test ", y_test.shape, y_train.dtype)


# In[110]:


X_val = X_train[:10000]
partial_X_train = X_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

print("x_val ", X_val.shape)
print("partial_x_train ", partial_X_train.shape)
print("y_val ", y_val.shape)
print("partial_y_train ", partial_y_train.shape)


# In[111]:


# NN model

model = models.Sequential()
model.add(layers.Dense(
    16, kernel_regularizer=regularizers.l1(0.001), activation='relu', input_shape=(10000,))
)
model.add(layers.Dropout(0.5))
model.add(layers.Dense(
    16, kernel_regularizer=regularizers.l1(0.001),activation='relu')
)
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))


# In[113]:


NumEpochs = 10
BatchSize = 512

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(
    partial_X_train, partial_y_train,
    epochs=NumEpochs, batch_size=BatchSize, validation_data=(X_val, y_val)
)

results = model.evaluate(X_test, y_test)

print("Test Loss and Accuracy")
print("results ", results)

history_dict = history.history
display(history_dict.keys())


# In[114]:


# Loss curve

plt.clf()
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, (len(history_dict['loss']) + 1))
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[115]:


# Validation accuracy curve

plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
epochs = range(1, (len(history_dict['acc']) + 1))
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[117]:


model.predict(X_test)

