
import sys
import os
import argparse
import pandas as pd
import glob
import datetime as dt
import pickle as pkl
from tensorflow import keras 
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import re
import json
import sklearn.dummy as sk
import seaborn as sns

def mse(x, y):
        return ((x-y)**2).mean()

def gen_indel(sequence,cut_site):
    '''This is the function that used to generate all possible unique indels and 
    list the redundant classes which will be combined after'''
    nt = ['A','T','C','G']
    up = sequence[0:cut_site]
    down = sequence[cut_site:]
    dmax = min(len(up),len(down))
    uniqe_seq ={}
    for dstart in range(1,cut_site+3):
        for dlen in range(1,dmax):
            if len(sequence) > dlen+dstart > cut_site-2:
                seq = sequence[0:dstart]+sequence[dstart+dlen:]
                indel = sequence[0:dstart] + '-'*dlen + sequence[dstart+dlen:]
                array = [indel,sequence,13,'del',dstart-30,dlen,None,None,None]
                try: 
                    uniqe_seq[seq]
                    if dstart-30 <1:
                        uniqe_seq[seq] = array
                except KeyError: uniqe_seq[seq] = array
    for base in nt:
        seq = sequence[0:cut_site]+base+sequence[cut_site:]
        indel = sequence[0:cut_site]+'-'+sequence[cut_site:]
        array = [sequence,indel,13,'ins',0,1,base,None,None]
        try: uniqe_seq[seq] = array
        except KeyError: uniqe_seq[seq] = array
        for base2 in nt:
            seq = sequence[0:cut_site] + base + base2 + sequence[cut_site:]
            indel = sequence[0:cut_site]+'--'+sequence[cut_site:]
            array = [sequence,indel,13,'ins',0,2,base+base2,None,None]
            try: uniqe_seq[seq] = array
            except KeyError:uniqe_seq[seq] = array
    uniq_align = label_mh(list(uniqe_seq.values()),4)
    for read in uniq_align:
        if read[-2]=='mh':
            merged=[]
            for i in range(0,read[-1]+1):
                merged.append((read[4]-i,read[5]))
            read[-3] = merged
    return uniq_align


def gen_cmatrix(indels,label): 
    ''' Combine redundant classes based on microhomology, matrix operation'''
    combine = []
    for s in indels:
        if s[-2] == 'mh':
            tmp = []
            for k in s[-3]:
                try:
                    tmp.append(label['+'.join(list(map(str,k)))])
                except KeyError:
                    pass
            if len(tmp)>1:
                combine.append(tmp)
    temp = np.diag(np.ones(557), 0)
    for key in combine:
        for i in key[1:]:
            temp[i,key[0]] = 1
            temp[i,i]=0    
    return (sparse.csr_matrix(temp))

def label_mh(sample,mh_len):
    '''Function to label microhomology in deletion events'''
    for k in range(len(sample)):
        read = sample[k]
        if read[3] == 'del':
            idx = read[2] + read[4] + 17
            idx2 = idx + read[5]
            x = mh_len if read[5] > mh_len else read[5]
            for i in range(x,0,-1):
                if read[1][idx-i:idx] == read[1][idx2-i:idx2] and i <= read[5]:
                    sample[k][-2] = 'mh'
                    sample[k][-1] = i
                    break
            if sample[k][-2]!='mh':
                sample[k][-1]=0
    return sample

def create_feature_array(ft,uniq_indels):
    '''Used to create microhomology feature array 
       require the features and label 
    '''
    ft_array = np.zeros(len(ft))
    for read in uniq_indels:
        if read[-2] == 'mh':
            mh = str(read[4]) + '+' + str(read[5]) + '+' + str(read[-1])
            try:
                ft_array[ft[mh]] = 1
            except KeyError:
                pass
        else:
            pt = str(read[4]) + '+' + str(read[5]) + '+' + str(0)
            try:
                ft_array[ft[pt]]=1
            except KeyError:
                pass
    return ft_array

def softmax(weights):
    return (np.exp(weights)/sum(np.exp(weights)))


def main():
    parser = argparse.ArgumentParser(description="MSEs for all prediction tasks")
    args = parser.parse_args()

    # load predictions
    predictions_indels = np.load("data/predictions_indels.npy")
    predictions_del = np.load("data/predictions_del.npy")
    predictions_ins = np.load("data/predictions_ins.npy")


    # load labels
    label, rev_index, features = pkl.load(open("data/feature_index_all.pkl", "rb"))
    feature_size = len(features) + 384
    test_data  = np.loadtxt("data/Lindel_test_with_full_seqs.txt", delimiter='\t', dtype='str')
    data = test_data[:, 1:].astype('float32')
    y = data[:, feature_size:]

    # concatenate predictions
    predictions = np.concatenate(((predictions_ins.T * predictions_indels[:, 1]).T, (predictions_del.T * predictions_indels[:, 0]).T), axis=1)
    # iterate over test data
    for i, row in enumerate(test_data):

        # get sequence
        sequence = row[0]
        indels = gen_indel(sequence, 30)
        cmax = gen_cmatrix(indels, label)

        # get the predictions for the sequence
        predictions[i] =  predictions[i] * cmax

        y[i] = y[i] * cmax


    mses = []
    # calculate MSEs
    for i, prediction in enumerate(predictions):
        mses.append(np.mean((prediction - y[i])**2))
    
    # print the mean MSE
    print("Mean MSE: {}".format(np.mean(mses)))

    # save the MSEs
    np.save("data/mses", mses)

    # Most of the code was taken from LR_deletion.py
    
    # Load aggregate model predictions
    aggregate_test_predictions = pkl.load(open("data/aggregate_model_test_predictions.pkl", "rb"))

    # Load true predictions
    test_data = np.loadtxt("data/Lindel_test.txt", delimiter="\t", dtype=str)
    test_data = test_data[:,1:].astype('float32')

    _, _, features = pkl.load(open('data/feature_index_all.pkl','rb'))
    feature_size = len(features) + 384
    y_test = np.array(test_data[:, feature_size:])

    dummy_mses = []
    for i in range(len(y)):
        dummy_mses.append(mse(y_test[i], aggregate_test_predictions[i]))
  
   
    # # load the training data
    # Lindel_training = pd.read_csv("data/Lindel_training.txt", sep='\t')
    # # load the test data
    # Lindel_test = pd.read_csv("data/Lindel_test.txt", sep='\t')

    # x_test = Lindel_test.iloc[ :, 1:3034] # 3033 binary features [2649 MH binary features + 384 one hot encoded features]
    # y_test = Lindel_test.iloc[ :, 3034:] # 557 observed outcome frequencies


    # X_train = Lindel_training.iloc[:, 1:3034] # 3033 binary features [2649 MH binary features + 384 one hot encoded features]
    # y_train = Lindel_training.iloc[:, 3034:] # 557 observed outcome frequencies
    
    # # dummy classifier based on prior distribution of the training data
    # dummy_clf = sk.DummyClassifier(strategy="prior", random_state=0)
    # dummy_clf.fit(X_train, y_train)

    # dummy_predictions = dummy_clf.predict_proba(x_test)
    # print(len(dummy_predictions[0]), len(y_test))
    # dummy_mses = []

    # for i, row in enumerate(dummy_predictions):
    #     dummy_mse = (dummy_predictions[i] - y_test.iloc[i])**2
    #     dummy_mses.append(np.mean(dummy_mse))

    
    # # devide mses and dummt_mses by 10^-3 to make the plot more readable
    dummy_mses = np.array(dummy_mses)/10**-3
    mses = np.array(mses)/10**-3

    # plot histogram with sns 
    sns.histplot(dummy_mses, bins=30, color='pink', label='Dummy', alpha=0.8, edgecolor=None)
    sns.histplot(mses, bins=30, color='#ADD8E6', label='Lindel', alpha=0.8, edgecolor=None)
    plt.xticks(np.arange(0, 3, 0.2))
    # set to log scale
    plt.legend()
    plt.xlabel("MSEs (10^-3)")

    plt.show()
    


    return

if __name__ == "__main__":
    sys.exit(main())



