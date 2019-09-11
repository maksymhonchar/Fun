#!/usr/bin/env python
# coding: utf-8

# In[1]:


# src: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

# A word embedding is a class of approaches for representing words and documents, 
    # using a dense vector representation.

# Word embeddings can be learned from text data and reused among projects.
# They can also be learned as part of fitting a neural network on text data.
    
# Word embeddings provide a dense representation of words and their relative meanings.
# They are IMPROVED representations compared to used in BOW model

# Instead, in an embedding, words are represented by dense vectors where a
    # vector represents the projection of the word into a continuous vector space.
    
# The position of a word within the vector space is learned from text and
    # is based on the words that surround the word when it is used.

# The position of a word in the learned vector space is referred to as its embedding.

# Methods to obtain word embeddings:
# Word2Vec
# GloVe


# In[2]:


# Keras Embedding layer arguments:
# 1. input_dim:
    # This is the size of the vocabulary in the text data.
    # For example, if your data is integer encoded to values between 0-10,
        # then the size of the vocabulary would be 11 words.
# 2. output_dim:
    # This is the size of the vector space in which words will be embedded.
    # It defines the size of the output vectors from this layer for each word.
    # For example, it could be 32 or 100 or even larger.
    # Test different values for your problem.
# 3. input_length:
    # This is the length of input sequences, as you would define for any
        # input layer of a Keras model.
    # For example, if all of your input documents are comprised of 1000 words,
        # this would be 1000.
        
# The Embedding layer has weights that are learned.

# The output of the Embedding layer is a 2D vector with one embedding
# for each word in the input sequence of words (input document).

# If you wish to connect a Dense layer directly to an Embedding layer, you
# must first flatten the 2D output matrix to a 1D vector using the Flatten layer.


# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences


# In[4]:


# Documents
docs = [
    # GOOD ONES
    'Well done!',
    'Good work',
    'Great effort',
    'nice work',
    'Excellent!',
    
    # BAD ONES
    'Weak',
    'Poor effort!',
    'not good',
    'poor work',
    'Could have done better.'
]

# Class labels
labels = np.array( [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] )


# In[7]:


# Encode each document

# Estimate the vocabulary size of 50 (which is much larger than needed to reduce the probability of collisions from the hash function)

vocab_size = 50

enc_docs = [ one_hot(doc, vocab_size) for doc in docs ]

display(enc_docs)


# In[10]:


# Pad every sequence to have the length of 4 => Keras prefers inputs to be vectorized and all inputs to have the same length

max_length = max( [len(seq) for seq in enc_docs] )

padded_docs = pad_sequences( enc_docs, maxlen=max_length, padding='post' )

display(padded_docs)


# In[11]:


# Define Embedding layer as part of NN model

def nn():
    model = Sequential()
    
    model.add( Embedding(vocab_size, 8, input_length=max_length) )
    model.add( Flatten() )
    model.add( Dense(1, activation='sigmoid') )
    
    return model

model = nn()
model.compile( optimizer='adam', loss='binary_crossentropy', metrics=['acc'] )


# In[14]:


model.summary()


# In[16]:


# Fit the model

history = model.fit(
    padded_docs, labels,
    epochs=1000,
    verbose=1
)


# In[19]:


# Overview fit history

fig, [ax_0, ax_1] = plt.subplots( 1, 2, figsize=(12, 6) )

ax_0.plot(history.history['loss'])
ax_0.set_title('loss')
ax_0.set_xlabel('epoch')
ax_0.set_ylabel('loss value')

ax_1.plot(history.history['acc'])
ax_1.set_title('accuracy')
ax_1.set_xlabel('epoch')
ax_1.set_ylabel('accuracy value')

plt.show()


# In[20]:


loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

