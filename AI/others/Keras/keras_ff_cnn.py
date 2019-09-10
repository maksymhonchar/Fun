#!/usr/bin/env python
# coding: utf-8

# In[59]:


import gc
gc.collect()


# In[33]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

import numpy as np

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential  # Sequential is a linear stack of neural network layers
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D


# In[34]:


(X_tr, y_tr), (X_test, y_test) = mnist.load_data()

display(X_tr.shape, y_tr.shape, X_test.shape, y_test.shape)


# In[35]:


plt.imshow( X_tr[0] ); plt.show()
plt.imshow( X_test[0] ); plt.show()


# In[36]:


# Data preprocessing 2

# MNIST images have depth=1
# Transform the dataset from (n, width, height) to (n, width, height, depth)

X_tr = X_tr.reshape( X_tr.shape[0], 28, 28, 1)
X_test = X_test.reshape( X_test.shape[0], 28, 28, 1 )

display(X_tr.shape, X_test.shape)


# In[37]:


# Data preprocessing 2

# Convert all data to float32 type
# Normalize data to the range [0, 1]

X_tr = X_tr.astype(np.float32)
X_test = X_test.astype(np.float32)

X_tr /= 255
X_test /= 255


# In[38]:


# Data preprocessing 3

# Preprocessing train labels: encode them to categorical type

y_tr = np_utils.to_categorical(y_tr, 10)
y_test = np_utils.to_categorical(y_test, 10)


# In[39]:


# Define CNN architecture

def nn():
    model = Sequential()
    
    # 32 - number of convolution filters to use
    # 3 - number of rows in each convolution kernel
    # 3 - number of columns in each convolution kernel
    # Input: (depth, width, height) => (1, 28, 28)
    model.add( Convolution2D(32, 3, 3, activation='relu', input_shape=(28, 28, 1)) )
    
    model.add( Convolution2D(32, 3, 3, activation='relu') )
    
    # MaxPooling2D is a way to reduce the number of parameters in our model by sliding a 2x2 pooling filter across the prev layer and taking the max of the 4 values in the 2x2 filter.
    model.add( MaxPooling2D(pool_size=(2, 2)) )
    
    # Dropout layer: this is a method for regluarizing the model in order to prevent overfitting
    model.add( Dropout(0.25) )
    
    # Weights from the Conv layers should be flattened (made 1-dimensional) before passing them to the fully connected Dense layer
    model.add( Flatten() )
    
    # 128 - output size of the layer
    model.add( Dense(128, activation='relu') )
    
    model.add( Dropout(0.5) )
    
    # Final output layer; size=10 == 10 classes of digits
    model.add( Dense(10, activation='softmax') )
    
    return model


# In[40]:


model = nn()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# In[52]:


history = model.fit(
    X_tr, y_tr,
    batch_size=32, epochs=10,
    validation_split=0.33,
    verbose=1
)


# In[54]:


score = model.evaluate( X_test, y_test )

display(score)


# In[53]:


display(history.history.keys())


# In[58]:


# Overview fit history

fig, [ax_0, ax_1] = plt.subplots( 1, 2, figsize=(12, 6) )

ax_0.plot(history.history['loss'])
ax_0.plot(history.history['val_loss'])
ax_0.set_title('loss')
ax_0.set_xlabel('epoch')
ax_0.set_ylabel('loss value')
ax_0.legend(['loss', 'val_loss'])

ax_1.plot(history.history['acc'])
ax_1.plot(history.history['val_acc'])
ax_1.set_title('accuracy')
ax_1.set_xlabel('epoch')
ax_1.set_ylabel('accuracy value')
ax_1.legend(['acc', 'val_acc'])

plt.show()

