#!/usr/bin/env python

#System tools 
import pickle as pkl
import os,sys,csv,re

from tqdm import tqdm_notebook as tqdm
import pylab as pl
import numpy as np

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, Flatten
from keras.models import Sequential, load_model
from keras.regularizers import l2, l1
#from Modeling.gen_features import *


# Define useful functions
def mse(x, y):
    return ((x-y)**2).mean()

def corr(x, y):
    return np.corrcoef(x, y)[0, 1] ** 2

def onehotencoder(seq):
    nt= ['A','T','C','G']
    head = []
    l = len(seq)
    for k in range(l):
        for i in range(4):
            head.append(nt[i]+str(k))

    for k in range(l-1):
        for i in range(4):
            for j in range(4):
                head.append(nt[i]+nt[j]+str(k))
    head_idx = {}
    for idx,key in enumerate(head):
        head_idx[key] = idx
    encode = np.zeros(len(head_idx))
    for j in range(l):
        encode[head_idx[seq[j]+str(j)]] =1.
    for k in range(l-1):
        encode[head_idx[seq[k:k+2]+str(k)]] =1.
    return encode

# Load data

# Arguments: working directory (workdir) and file name (fname)
workdir = sys.argv[1]
fname   = sys.argv[2]

# Load 1) labels: dictionary mapping 557 classes to indices 
#      2) rev_index: reverse indexing of labels (keys are indices and classes are values)
#      3) features: microhomology features (total of 2649; keys are feature names and values are indices)
label,rev_index,features = pkl.load(open(workdir+'feature_index_all.pkl','rb'))
# Total feature size: add additional 384 sequence one-hot encoded features
feature_size = len(features) + 384
# Load training data
data     = np.loadtxt(workdir+fname, delimiter="\t", dtype=str)
# Sequences are on column 0
Seqs = data[:,0]
# Features and labels: columns 1 to 3590 (3033 [mh features + sequence features]; the last 557 columns are the classes (probability distribution))
data = data[:,1:].astype('float32')

# X = features; y = labels
X = data[:,:feature_size]
y = data[:, feature_size:]

np.random.seed(121)

# Indices of the classes, from 0 to 556
idx = np.arange(len(y))

# Shuffle data
np.random.shuffle(idx)
X, y = X[idx], y[idx]

# Adjust sizes of training and validation sets
train_size = round(len(data) * 0.9) if 'ForeCasT' in fname else 3900
valid_size = round(len(data) * 0.1) if 'ForeCasT' in fname else 450 

# Select sequences corresponding to shuffled indices
Seq_train = Seqs[idx]
x_train,x_valid = [],[]
y_train,y_valid = [],[]

# Create training set with 3900 samples
for i in range(train_size):
    # Check that labels for each sample sum up to at most 1
    # We only check for the first 536 classes, which are predicted by the deletion model
    # Bug? -> Final training set will not be 3900 samples
    if 1> sum(y[i,:536])> 0 :
        # Normalize (probability distribution)
        y_train.append(y[i,:536]/sum(y[i,:536]))
        x_train.append(X[i])

# Add remaining samples from the dataset to the validation set
for i in range(train_size,len(Seq_train)):
    if 1> sum(y[i,:536])>0 : 
        y_valid.append(y[i,:536]/sum(y[i,:536]))
        x_valid.append(X[i])

# Convert to numpy arrays
x_train,x_valid = np.array(x_train),np.array(x_valid)
y_train,y_valid = np.array(y_train),np.array(y_valid)

# Number of features of the input
size_input = x_train.shape[1]

# Train model
# Regularization strengths, from 10^(-10) to 10^(-1)
lambdas = 10 ** np.arange(-10, -1, 0.1)
errors_l1, errors_l2 = [], []

# tqgm is for progress bars

# L2 regularization
for l in tqdm(lambdas):
    np.random.seed(0)

    # Initialize keras Sequential() model
    model = Sequential()

    # Replicate logistic regression model, add a densely connected layer
    # This calculates the polynomial with coefficients being the model weights, and maps it to the range [0, 1] with softmax
    model.add(Dense(536,  activation='softmax', input_shape=(size_input,), kernel_regularizer=l2(l)))
    
    # Train with cross-entropy loss, 100 epochs and patience 1
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
    model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid), 
              callbacks=[EarlyStopping(patience=1)], verbose=0)

    # Evaluate MSE on the validation set, for each lambda
    y_hat = model.predict(x_valid)
    errors_l2.append(mse(y_hat, y_valid))

# Similar as before, L1 regularization
for l in tqdm(lambdas):
    np.random.seed(0)
    model = Sequential()
    model.add(Dense(536,  activation='softmax', input_shape=(size_input,), kernel_regularizer=l1(l)))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
    model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid), 
              callbacks=[EarlyStopping(patience=1)], verbose=0)
    y_hat = model.predict(x_valid)
    errors_l1.append(mse(y_hat, y_valid))

# Save mean squared errors
np.save(workdir+'mse_l1_del.npy',errors_l1)
np.save(workdir+'mse_l2_del.npy',errors_l2)

# Final model
# Find best lambda for L1 regularization
l = lambdas[np.argmin(errors_l1)]
np.random.seed(0)

# Re-create model and train again with chosen lambda
model = Sequential()
model.add(Dense(536, activation='softmax', input_shape=(size_input,), kernel_regularizer=l1(l)))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
# Store intermediate training steps
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid), 
          callbacks=[EarlyStopping(patience=1)], verbose=0)

model.save(workdir+'L1_del.h5')

# As before, but with L2 regularization
l = lambdas[np.argmin(errors_l2)]
np.random.seed(0)
model = Sequential()
model.add(Dense(536, activation='softmax', input_shape=(size_input,), kernel_regularizer=l2(l)))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mse'])
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid), 
          callbacks=[EarlyStopping(patience=1)], verbose=0)

model.save(workdir+'L2_del.h5')
