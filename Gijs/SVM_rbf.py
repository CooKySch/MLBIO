#!/usr/bin/env python

#System tools
import pickle as pkl
import os,sys,csv,re
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from tqdm import tqdm_notebook as tqdm
import pylab as pl
import numpy as np

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, Flatten
from keras.models import Sequential, load_model
from keras.regularizers import l2, l1


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

def train_svm(x_train, y_train, y_valid, x_valid, model_name):
    lambdas = 10 ** np.arange(-1, 1, 0.1)
    errors_auto, errors_scale = [], []

    # Gamma hyperparameter is set auto (1 / n_features)
    for l in tqdm(lambdas):
        np.random.seed(0)

        # Initialize SVR model
        clf = MultiOutputRegressor(SVR(kernel='rbf', C=l, gamma='auto'), n_jobs=-1).fit(x_train, y_train)

        # Evaluate MSE on the validation set, for each lambda
        y_hat = clf.predict(x_valid)
        errors_auto.append(mse(y_hat, y_valid))


    # Gamma hyperparameter is set scaled (1 / (n_features * X.var()))
    for l in tqdm(lambdas):
        np.random.seed(0)
        # Initialize SVR model
        clf = MultiOutputRegressor(SVR(kernel='rbf', C=l, gamma='scale'), n_jobs=-1).fit(x_train, y_train)

        # Evaluate MSE on the validation set, for each lambda
        y_hat = clf.predict(x_valid)
        errors_scale.append(mse(y_hat, y_valid))

    # Find the best value for C for auto
    l = lambdas[np.argmin(errors_auto)]
    np.random.seed(0)
    clf = MultiOutputRegressor(SVR(kernel='rbf', C=l, gamma='auto'), n_jobs=-1).fit(x_valid, y_valid)

    pkl_file = f"Gijs/Data/SVR_{model_name}_rbf_gamma_auto_reg.pkl"
    with open(pkl_file, 'wb') as file:
        pkl.dump(clf, file)

    # Find the best value for C for scale
    l = lambdas[np.argmin(errors_scale)]
    np.random.seed(0)
    clf = MultiOutputRegressor(SVR(kernel='rbf', C=l, gamma='scale'), n_jobs=-1).fit(x_valid, y_valid)

    pkl_file = f"Gijs/Data/SVR_{model_name}_rbf_gamma_scale_reg.pkl"
    with open(pkl_file, 'wb') as file:
        pkl.dump(clf, file)

def train_insertions(Seqs, y, train_size, idx):
    Seq_train = Seqs[idx]
    x_train,x_valid = [],[]
    y_train,y_valid = [],[]
    for i in range(train_size):
        if 1> sum(y[i,-21:])> 0 :# 5 is a random number i picked if i use pred_size here it will be -21:0 it will just generate empty array
            y_train.append(y[i,-21:]/sum(y[i,-21:]))
            x_train.append(onehotencoder(Seq_train[i][-6:]))
    for i in range(train_size,len(Seq_train)):
        if 1> sum(y[i,-21:])>0 :
            y_valid.append(y[i,-21:]/sum(y[i,-21:]))
            x_valid.append(onehotencoder(Seq_train[i][-6:]))

    x_train,x_valid = np.array(x_train),np.array(x_valid)
    y_train,y_valid = np.array(y_train),np.array(y_valid)

    train_svm(x_train, y_train, y_valid, x_valid, "ins")

def train_deletions(Seqs, X, y, train_size, idx):
    Seq_train = Seqs[idx]
    x_train,x_valid = [],[]
    y_train,y_valid = [],[]
    # Create training set with 3900 samples
    for i in range(train_size):
        # Check that labels for each sample sum up to at most 1
        if 1> sum(y[i,:536])> 0 :
            # Normalize (probability distribution)
            y_train.append(y[i,:536]/sum(y[i,:536]))
            x_train.append(X[i])

    # Add remaining samples from the dataset to the validation set
    for i in range(train_size,len(Seq_train)):
        if 1> sum(y[i,:536])>0 : 
            y_valid.append(y[i,:536]/sum(y[i,:536]))
            x_valid.append(X[i])

    train_svm(x_train, y_train, y_valid, x_valid, "del")

def train_indels(Seqs, y, train_size, idx):
    Seq_train = Seqs[idx]
    x_train,x_valid = [],[]
    y_train,y_valid = [],[]
    for i in range(train_size):
        x_train.append(onehotencoder(Seq_train[i]))
        y_train.append((sum(y[i][:-21]),sum(y[i][-21:])))
    for i in range(train_size,len(Seq_train)):
        x_valid.append(onehotencoder(Seq_train[i]))
        y_valid.append((sum(y[i][:-21]),sum(y[i][-21:])))

    x_train,x_valid = np.array(x_train),np.array(x_valid)
    y_train,y_valid = np.array(y_train),np.array(y_valid)

    train_svm(x_train, y_train, y_valid, x_valid, "indel")

def SVM_rbf():
    # Load data
    workdir = 'data/'
    fname   = 'Lindel_training.txt'

    label,rev_index,features = pkl.load(open(workdir+'feature_index_all.pkl','rb'))
    feature_size = len(features) + 384
    data     = np.loadtxt(workdir+fname, delimiter="\t", dtype=str)
    Seqs = data[:,0]
    data = data[:,1:].astype('float32')

    # Sum up deletions and insertions to
    X = data[:,:feature_size]
    y = data[:, feature_size:]

    np.random.seed(121)
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]
    train_size = round(len(data) * 0.9) if 'ForeCasT' in fname else 3900

    # Train the model
    train_insertions(Seqs, y, train_size, idx)
    train_deletions(Seqs, X, y, train_size, idx)
    train_indels(Seqs, y, train_size, idx)

SVM_rbf()