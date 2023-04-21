#!/usr/bin/env python

#System tools 
import pickle as pkl
import os,sys,csv,re

from tqdm import tqdm_notebook as tqdm
import pylab as pl
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
import seaborn as sns
import matplotlib.pyplot as plt


# Define useful functions
def mse(x, y):
    return ((x-y)**2).mean()

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



workdir  = "data/"
fname    = "Lindel_training.txt"

label,rev_index,features = pkl.load(open(workdir+'feature_index_all.pkl','rb'))
feature_size = len(features) + 384 
data     = np.loadtxt(workdir+fname, delimiter="\t", dtype=str)
Seqs = data[:,0]
data = data[:,1:].astype('float32')

# Sum up deletions and insertions to 
X = data[:,:feature_size]
y = data[:, feature_size:]

x_train = []
y_train = []

Seq_train = Seqs
train_size = int(len(Seq_train))

# train_size = int(len(Seq_train) * 0.9)

for i in range(train_size):
    if 1> sum(y[i,-21:])> 0 :
        y_train.append(y[i,-21:]/sum(y[i,-21:]))
        x_train.append(onehotencoder(Seq_train[i][-6:]))

x_train = np.array(x_train)
y_train = np.array(y_train)
size_input = x_train.shape[1]

# create random forest model MultiOutputRegressor 
clf = MultiOutputRegressor(RandomForestRegressor(n_estimators=20, max_depth=10, random_state=0, n_jobs=-1, bootstrap=True, oob_score=True), n_jobs=-1)

# Train model
clf.fit(x_train, y_train)

# save model
pkl.dump(clf, open('data/RF_insertion.pkl', 'wb'))
