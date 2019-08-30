#!/usr/bin/env python
# coding: utf-8

# In[12]:


# Import libraries

import re

from bs4 import BeautifulSoup

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import pandas as pd

from keras import constraints, initializers, layers, optimizers, regularizers
from keras.layers import (GRU, LSTM, Activation, Bidirectional, Convolution1D,
                          Dense, Dropout, Embedding, Flatten, GlobalMaxPool1D,
                          Input)
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, f1_score


# In[21]:


# Load the data

train_df = pd.read_csv('data/labeledTrainData.tsv', sep='\t')

test_df = pd.read_csv('data/testData.tsv', sep='\t')


# In[22]:


# Overview the data

display(train_df.shape, test_df.shape)

display(train_df.sample(3))

display(test_df.sample(3))


# In[23]:


# Data preprocessing

no_punctuation_tokenizer = RegexpTokenizer(r'\w+')

lemmatizer = WordNetLemmatizer()

nltk_stopwords_set = set( stopwords.words('english') )  # set to speed up the algo

def preprocess_text_inplace(df, text_col_name='review'):
    # Remove HTML tags
    df[text_col_name] = df[text_col_name].apply(
        lambda x: BeautifulSoup(x).get_text()
    )
    # Convert to lowercase
    df[text_col_name] = df[text_col_name].str.lower()
    # Remove punctuation AND convert to tokens
    df[text_col_name] = df[text_col_name].apply(
        lambda x: no_punctuation_tokenizer.tokenize(x)
    )
    # Lemmatize tokens
    # nouns
    df[text_col_name] = df[text_col_name].apply(
        lambda x: [lemmatizer.lemmatize(token) for token in x]
    )
    # verbs
    df[text_col_name] = df[text_col_name].apply(
        lambda x: [lemmatizer.lemmatize(token, 'v') for token in x]
    )
    # Remove stopwords
    df[text_col_name] = df[text_col_name].apply(
        lambda x: [word for word in x if not word in nltk_stopwords_set]
    )
    # Join tokens into whole sentence
    df[text_col_name] = df[text_col_name].apply(
        lambda x: ' '.join(x)
    )    


# In[24]:


display('processing train_df reviews...')
preprocess_text_inplace(train_df)

display('processing test_df reviews...')
preprocess_text_inplace(test_df)


# In[25]:


display(train_df.shape, test_df.shape)

display(train_df.sample(3))

display(test_df.sample(3))


# In[46]:


# Build a bidirectional lstm rnn

max_features = 5000

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts( train_df['review'] )

train_tokens = tokenizer.texts_to_sequences( train_df['review'] )  # lst; len()=25000


# In[47]:


# Pad tokenized data

X_t = pad_sequences( train_tokens, maxlen=150 )  # shape = (25000, 150)

y = train_df['sentiment']  # shape = (25000, )


# In[48]:


# NN

embed_size = 128

model = Sequential()
model.add( Embedding(max_features, embed_size) )  # L1
model.add( Bidirectional( LSTM(32, return_sequences = True) ) )  # L2: LSTM
model.add( GlobalMaxPool1D() )  # Fully connected layer
model.add( Dense(20, activation="relu") )  # RELU
model.add( Dropout(0.05) )
model.add( Dense(1, activation="sigmoid") )  # Sigmoid

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[49]:


# Start NN training

batch_size = 100
epochs = 5

model.fit(
    X_t, y,
    validation_split=0.2,
    batch_size=batch_size, epochs=epochs,
)


# In[52]:


# Predict on test set

test_df['sentiment'] = test_df["id"].map(
    lambda x: 1 if int(x.strip('"').split("_")[1]) >= 5 else 0
)

y_test = test_df["sentiment"]


# In[53]:


test_tokens = tokenizer.texts_to_sequences( test_df['review'] )  # lst; len()=25000


# In[56]:


X_t_test = pad_sequences( test_tokens, maxlen=150 )  # (25000, 150)


# In[59]:


# Create a prediction

prediction = model.predict(X_t_test)

y_pred = (prediction > 0.5)


# In[60]:


display(
    confusion_matrix(y_pred, y_test)
)

display(
    f1_score(y_pred, y_test)  # 0.84
)


# In[96]:


# Make a submission

submission = pd.DataFrame({
    'id': test_df['id'],
    'sentiment': [1 if pred >= .5 else 0 for pred in prediction]
})

display(submission.head())

submission.to_csv("aug29_submission.csv", index=False)

# display(
#     y_pred.astype(int).shape,
#     test_df.shape,
#     test_df['id'].shape
# )


# In[ ]:


# 0.84932

