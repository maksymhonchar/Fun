#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load libraries
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import csv

import pandas as pd

import numpy as np

from bs4 import BeautifulSoup

import re

import nltk
from nltk.corpus import stopwords

from gensim.models import word2vec

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


# In[2]:


# Load data

train_df = pd.read_csv(
    'data/labeledTrainData.tsv', delimiter='\t',
    quoting=csv.QUOTE_NONE, header=0
)

test_df = pd.read_csv(
    'data/testData.tsv', delimiter='\t',
    quoting=csv.QUOTE_NONE, header=0
)

unlabeled_train_df = pd.read_csv(
    'data/unlabeledTrainData.tsv', delimiter='\t',
    quoting=csv.QUOTE_NONE, header=0
)


# In[3]:


# Overview loaded data

display(train_df.shape, test_df.shape, unlabeled_train_df.shape)  # sum: 100k rows

display(train_df.columns.values, test_df.columns.values)
display(unlabeled_train_df.columns.values)


# In[4]:


# Clean the data

def clean_review(review, remove_numbers=False, remove_stopwords=False):
    text_nohtml = BeautifulSoup(review).get_text()
    if remove_numbers:
        re_pattern = "[^a-zA-Z]"
    else:
        re_pattern = "[^a-zA-Z0-9]"
    text_cleanedchars = re.sub(re_pattern, " ", text_nohtml)
    text_tokens = text_cleanedchars.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text_nostopwords = [
            token for token in text_tokens
            if token not in stops
        ]
        return text_nostopwords
    return text_tokens


# In[5]:


# Convert to specific input format.
# Word2Vec expects single sentences, each one as a list of words.

def review_sentences(review, tokenizer, remove_numbers=False, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(clean_review(review, remove_numbers, remove_stopwords))
    return sentences


# In[6]:


# Prepare our data for input to Word2Vec
# use "+=" in order to join all of the lists at once

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sentences = []

for idx, review in enumerate(train_df.loc[:, 'review']):
    sentences += review_sentences(review, tokenizer, True, True)
    if idx % 1000 == 0:
        print("train {0}".format(idx))

# Skip: out of memory
# print("Parsing sentences from unlabeled set")
# for idx, review in enumerate(unlabeled_train_df.loc[:, 'review']):
#     sentences += review_sentences(review, tokenizer)
#     if idx % 1000 == 0:
#         print("train {0}".format(idx))


# In[ ]:


display(len(sentences))
print(sentences[0])


# In[ ]:


# Word2Vec parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Init and train the word2vec model
model = word2vec.Word2Vec(
    sentences,
    size=num_features, min_count=min_word_count,
    sample=downsampling, workers=7, window=context
)


# In[ ]:


model.init_sims(replace=True)

# Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)


# In[ ]:


# Explore model results

display(
    model.wv.doesnt_match("man woman child kitchen toy".split())
)

display(
    model.wv.doesnt_match("france england germany moscow".split())
)


# In[ ]:


display(
    model.wv.most_similar("man")
)

display(
    model.wv.most_similar("woman")
)

display(
    model.wv.most_similar("russia")
)

display(
    model.wv.most_similar("loving")
)


# In[ ]:


# Load the model again

model = word2vec.Word2Vec.load("300features_40minwords_10context")


# In[ ]:


display(
    type(model.wv.vectors)  # numpy.ndarray
)

display(
    model.wv.vectors.shape  # (37858, 300)
)

display(
    model.wv['flower'].shape  # (300,)
)

display(
    len(model.wv.index2word)  # 37858
)


# In[ ]:


# Vector Averaging
# Find a way to take individual word vectors and transform them
    # into a feature set that is the same length for every review.
    
def get_feature_vec(words, model, n_features):
    feature_vec = np.zeros((n_features), dtype=np.float32)
    n_words = 0
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            n_words += 1.0
            feature_vec = np.add(feature_vec, model.wv[word])
    feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

def get_avg_feature_vecs(reviews, model, n_features):
    review_feature_vecs = np.zeros(
        (len(reviews), n_features), dtype=np.float32
    )
    for idx, review in enumerate(reviews):
        if idx % 1000 == 0:
            print("{0} of {1}".format(idx, len(reviews)))
        review_feature_vecs[idx] = get_feature_vec(
            review, model, n_features
        )
    return review_feature_vecs


# In[ ]:


# Calculate average feature vectors for training and testing sets

clean_train_reviews = []
for review in train_df['review']:
    clean_train_reviews.append(
        clean_review(review, remove_numbers=True, remove_stopwords=True)
    )    


# In[ ]:


train_data_vecs = get_avg_feature_vecs(
    clean_train_reviews, model, 300
)


# In[ ]:


clean_test_reviews = []
for review in test_df['review']:
    clean_test_reviews.append(
        clean_review(review, remove_numbers=True, remove_stopwords=True)
    )


# In[ ]:


test_data_vecs = get_avg_feature_vecs(
    clean_test_reviews, model, 300
)


# In[ ]:


# Use the average paragraph vectors to train a random forest

forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data_vecs, train_df["sentiment"])


# In[ ]:


result = forest.predict(test_data_vecs)

output = pd.DataFrame(
    data={"id": test_df["id"], "sentiment": result}
)

output.to_csv(
    "Word2Vec_AverageVectors.csv", index=False, quoting=3
)


# In[ ]:


# Clustering
# Word2Vec creates clusters of semantically related words, so
    # another possible approach is to exploit the similarity of
    # words within a cluster.
# Grouping vectors in this way is known as "vector quantization."
# To accomplish this, we first need to find the centers of
    # the word clusters, which we can do by using a
    # clustering algorithm such as K-Means.

def perform_kmeans_clustering(model):
    word_vectors = model.wv.syn0
    n_clusters = word_vectors.shape[0] / 5
    kmeans_model = KMeans(n_clusters=n_clusters)
    centroids = kmeans_model.fit_predict(word_vectors)
    
    word_centroid_map = dict(zip(model.wv.index2word, centroids))


# In[ ]:


# perform_kmeans_clustering(model)

# Overview of the results: https://www.kaggle.com/c/word2vec-nlp-tutorial/overview/part-3-more-fun-with-word-vectors

