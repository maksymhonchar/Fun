#!/usr/bin/env python
# coding: utf-8

# In[52]:


# Load libraries
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import csv

import pandas as pd

from bs4 import BeautifulSoup

import re

import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier


# In[53]:


# Load data

train_df = pd.read_csv(
    'data/labeledTrainData.tsv', delimiter='\t',
    quoting=csv.QUOTE_NONE, header=0
)

test_df = pd.read_csv(
    'data/testData.tsv', delimiter='\t',
    quoting=csv.QUOTE_NONE, header=0
)


# In[54]:


# Overview loaded data

def overview_dataset(dataset_df):
    # Data inside
    display(dataset_df.head(3))
    display(dataset_df.tail(3))
    # Dimensions and size
    display(dataset_df.shape)
    # Columns names
    display(dataset_df.columns.values)
    # .describe()
    display(dataset_df.describe(include='all').T)
    
def overview_review(dataset_df, overview_idx=0):
    display(dataset_df.loc[overview_idx, 'review'])
    display(len(dataset_df.loc[overview_idx, 'review']))    


# In[55]:


overview_dataset(train_df)
overview_review(train_df)

overview_dataset(test_df)
overview_review(test_df)


# In[56]:


def clean_review(raw_review):
    """Function to convert raw review from df to words"""
    text_nothtml = BeautifulSoup(raw_review).get_text()
    text_letters = re.sub("[^a-zA-Z]", " ", text_nothtml)
    text_lower = text_letters.lower()
    text_tokens = text_lower.split()
    all_stopwords = set(stopwords.words("english"))
    text_nostopwords = [
        token for token in text_tokens
        if token not in all_stopwords
    ]
    text_cleaned = " ".join(text_nostopwords)
    return text_cleaned    


# In[57]:


# Clean all training set reviews

cleaned_train_reviews = []

for row_idx in range(train_df.shape[0]):
    cleaned_review = clean_review(train_df.loc[row_idx, 'review'])
    cleaned_train_reviews.append(cleaned_review)
    
cleaned_test_reviews = []

for row_idx in range(test_df.shape[0]):
    cleaned_review = clean_review(test_df.loc[row_idx, 'review'])
    cleaned_test_reviews.append(cleaned_review)


# In[58]:


# Create features from bag-of-words

vectorizer = CountVectorizer(max_features=5000)

train_data_features = vectorizer.fit_transform(cleaned_train_reviews)
train_data_features = train_data_features.toarray()

test_data_features = vectorizer.transform(cleaned_test_reviews)
test_data_features = test_data_features.toarray()

display(train_data_features.shape)  # 25000 words and 5000 features - one for each vocabulary word
display(test_data_features.shape)  # (25000, 5000)

# display(vectorizer.get_feature_names())

# display(vectorizer.get_params())


# In[59]:


# Apply random forest to vectorized data

model_forest = RandomForestClassifier(n_estimators=100, verbose=2, n_jobs=7)

model_forest = model_forest.fit(
    train_data_features, train_df.loc[:, 'sentiment']
)

result = model_forest.predict(test_data_features)


# In[60]:


# Create a submission

output = pd.DataFrame(
    {'id': test_df['id'], 'sentiment': result}
)

output.to_csv('p1forbeginners_submission.csv', index=False, quoting=3)

