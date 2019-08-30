#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math

import re

from bs4 import BeautifulSoup

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, SpatialDropout1D, Dropout
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

import gensim


# In[2]:


labeled_data = pd.read_csv("data/labeledTrainData.tsv", sep="\t", quoting=3)

test_data = pd.read_csv("data/testData.tsv", sep="\t", quoting=3)


# In[3]:


display(
    labeled_data.shape, test_data.shape
)


# In[4]:


train_reviews = labeled_data['review']
train_sentiments = labeled_data['sentiment']


# In[5]:


all_reviews = train_reviews.append( test_data['review'] )


# In[6]:


nltk_stopwords_set = set( stopwords.words('english') )

def preprocess_text(text):
    text = BeautifulSoup(text).get_text()
    text = text.lower()
    text = re.sub(r'[^\w\s]','', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in nltk_stopwords_set]
    
    return tokens


# In[7]:


display('preprocessing all_reviews...')
all_reviews = all_reviews.apply(
    lambda x: preprocess_text(x)
)


# In[8]:


# Build Word2Vec model to get embedding layer

embedding_vector_size = 152  # faster: "% 4 = 0"

display('training Word2Vec model...')
word2vec_model = gensim.models.Word2Vec(
    sentences=all_reviews,
    size=embedding_vector_size, min_count=1, window=5,
    workers=8
)


# In[9]:


# Tokenize all reviews
# Note: tokenize train AND test data in one go

max_features = 5000

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(all_reviews)

all_reviews_seq = tokenizer.texts_to_sequences(all_reviews)


# In[10]:


print(
    len(all_reviews_seq),
    all_reviews_seq[0]
)


# In[11]:


# Pad

# Calculate maxlen for a document => AVG
# Note: reduce() could be used to summarize length of array

len_sum = 0
for doc in all_reviews_seq:
    len_sum += len(doc)

avg_doc_len = math.ceil(len_sum / len(all_reviews_seq))

display(avg_doc_len)

# Apply padding
all_reviews_pad = pad_sequences(
    all_reviews_seq,
    maxlen=avg_doc_len
)

display(all_reviews_pad.shape)


# In[13]:


# Weights for embedded layer

embedding_l_weights = np.zeros(shape=(
    len(tokenizer.word_index) + 1,  # i starts from 1 in the next for-loop
    embedding_vector_size    
))

for word, idx in tokenizer.word_index.items():
    vector_i = word2vec_model.wv[word]
    if vector_i is not None:
        embedding_l_weights[idx] = vector_i


# In[55]:


# BiLSTM RNN

model = Sequential()

model.add(Embedding(
    input_dim=len(tokenizer.word_index) + 1,
    output_dim = embedding_vector_size, 
    
    input_length=avg_doc_len,
    
    weights=[embedding_l_weights]
))
model.add(Bidirectional(LSTM(128, dropout=0.25, recurrent_dropout=0.1)))
model.add(Dense(10))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))


# In[56]:


# Keras: Reduce learning rate when a metric has stopped improving.

learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_acc', 
    patience=2, factor=0.5, min_lr=0.0001,
    verbose=1
)


# In[57]:


# Get data to train the model
X = all_reviews_pad[:25000, :]
X = X.reshape(-1, avg_doc_len)
y = train_sentiments

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y,
    test_size=0.15, shuffle=True
)


# In[58]:


# Compile Keras model

model.compile(
    optimizer='RMSprop',
    loss='binary_crossentropy',
    metrics=['acc']
)


# In[59]:


# Fit Keras model

history = model.fit(
    X_tr, y_tr,
    epochs=10,
    batch_size=500,
    callbacks=[learning_rate_reduction],
    validation_data=(X_val, y_val)
)


# In[63]:


import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[64]:


y_test_pred = model.predict(X_val)


# In[65]:


from sklearn.metrics import roc_auc_score
roc_auc_score(y_val, y_test_pred, average = 'weighted')


# In[66]:


#predicting test_data
y_pred = model.predict(
    all_reviews_pad[25000:, :]
)


# In[67]:


display(y_pred.shape, test_data.shape)


# In[69]:


plt.hist(y_pred)


# In[72]:


pred_median = np.median(y_pred)

submission_predictions = [
    1 if v > pred_median
    else 0
    for v in y_pred
]


# In[76]:


corrected_ids = test_data['id'].str.replace('"', '')

submission = pd.DataFrame({
    'id': corrected_ids,
    'sentiment': submission_predictions
})
submission.to_csv('submission.csv', index=False)

