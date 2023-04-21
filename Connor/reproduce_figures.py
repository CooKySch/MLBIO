
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
import scipy.sparse as sparse
import re
import json
from tensorflow import keras
import seaborn as sns
from scipy import stats

def onehotencoder(seq):
    '''convert to single and di-nucleotide hotencode'''
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

def gen_prediction(seq,wb,prereq, selected_features):
    # '''generate the prediction for all classes, redundant classes will be combined'''
    if len(seq)!= 60:
        return ('Error: The input sequence is not 60bp long.')
    pam = {'AGG':0,'TGG':0,'CGG':0,'GGG':0}
    guide = seq[13:33]
    if seq[33:36] not in pam:
        return ('Error: No PAM sequence is identified.')
    w1,b1,w2,b2,w3,b3 = wb
    label,rev_index,features,frame_shift = prereq
    indels = gen_indel(seq,30)
    input_indel = onehotencoder(guide)
    input_ins   = onehotencoder(guide[-6:])
    input_del   = np.concatenate((create_feature_array(features,indels),input_indel),axis=None)

    input_del = input_del[selected_features]

    cmax = gen_cmatrix(indels,label) # combine redundant classes
    dratio, insratio = softmax(np.dot(input_indel,w1)+b1)
    ds  = softmax(np.dot(input_del,w2)+b2)
    ins = softmax(np.dot(input_ins,w3)+b3)
    y_hat = np.concatenate((ds*dratio,ins*insratio),axis=None) * cmax
    return (y_hat,np.dot(y_hat,frame_shift))

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

    #names = ["_Connor1.2", "_Connor1.4", "_Connor1.6", "_Connor1.8", "_Connor2", "_Connor3", ""]
    names=["_Connornotzero", ""]
    selected_features = pkl.load(open("../Connor/selected_features_notzero.pkl", "rb"))


    # load labels
    label, rev_index, features = pkl.load(open("../data/feature_index_all.pkl", "rb"))
    feature_size = len(features) + 384
    test_data = np.loadtxt("../data/Lindel_test_with_full_seqs.txt", delimiter='\t', dtype='str')
    data = test_data[:, 1:].astype('float32')
    y = data[:, feature_size:]
    model_preq = pkl.load(open("../data/model_prereq.pkl", 'rb'))
    label, rev_index, features, frame_shift = model_preq
    all_predictions = list()
    all_frameshifts = list()

    all_features = list(range(feature_size))
    selected_features.append(all_features)
    for i in range(len(names)):
        # load L1_del.h5
        model_del = keras.models.load_model("../data/L1_del"+names[i]+".h5")
        model_ins = keras.models.load_model("../data/L1_ins_Connor.h5")
        model_indels = keras.models.load_model("../data/L2_indel_Connor.h5")
        weights = [model_indels.get_weights()[0], model_indels.get_weights()[1], model_del.get_weights()[0],
                   model_del.get_weights()[1], model_ins.get_weights()[0], model_ins.get_weights()[1]]

        # initialize an empty array to store predicted frameshifts and one for the predictions
        predicted_frameshift = []
        predictions = []
        for sample in test_data:
            # get the sequence
            sequence = sample[0]
            pred = gen_prediction(sequence, weights, model_preq, selected_features[i])
            predictions.append(pred[0])
            predicted_frameshift.append(pred[1])

        all_predictions.append(predictions)
        all_frameshifts.append(predicted_frameshift)

    mses = list()
    pearsons=list()
    for pred in range(len(all_predictions)):
        actual_frameshift = np.dot(y, frame_shift).tolist()
        predicted_frameshift = all_frameshifts[pred]
        predictions = all_predictions[pred]

        for i, pfs in enumerate(predicted_frameshift):
            if pfs == 'r':
                predicted_frameshift.pop(i)
                actual_frameshift.pop(i)

        # calculate the MSE
        mse = np.mean((np.array(actual_frameshift) - np.array(predicted_frameshift)) ** 2)
        # keep 3 decimal places
        mse = round(mse, 3)
        # calculate pearson correlation
        pearson = stats.pearsonr(actual_frameshift, predicted_frameshift)
        # keep 3 decimal places
        pearson = round(pearson[0], 3)
        mses.append(mse)
        pearsons.append(pearson)


    # include in the plot the mse and pearson correlation
    title = ""
    for i in range(len(names)):
        title += "MSE" + names[i] + ": " + str(mses[i]) + ", Pearson correlation" + names[i] + ": " + str(pearsons[i]) + "\n"

    plt.title(title)
    plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, color="red")
    for pfs in all_frameshifts:
        plt.scatter(actual_frameshift, pfs)
    plt.xlabel("Measured frameshift ratio")
    plt.ylabel("Predicted frameshift ratio")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig("../data/frameshift.png")
    plt.tight_layout()
    plt.show()
    plt.close()

    all_mses = []
    for predictions in all_predictions:
        # calculate MSEs
        mses = []
        for i, prediction in enumerate(predictions):
            if prediction == "Error: No PAM sequence is identified." or prediction == "'Error: The input sequence is not 60bp long." or type(
                    prediction) == str:
                continue
            mses.append(np.mean((prediction - y[i]) ** 2))
        all_mses.append(mses)

    all_mses = np.array(all_mses)
    # print the mean MSE
    print("Mean MSE: {}".format(np.mean(all_mses, axis=1)))

    # save the MSEs
    np.save("../data/mses", mses)

    # Load aggregate model predictions
    aggregate_test_predictions = pkl.load(open("../data/aggregate_model_test_predictions.pkl", "rb"))

    # Load true predictions
    test_data = np.loadtxt("../data/Lindel_test.txt", delimiter="\t", dtype=str)
    test_data = test_data[:, 1:].astype('float32')

    _, _, features = pkl.load(open('../data/feature_index_all.pkl', 'rb'))
    feature_size = len(features) + 384
    y_test = np.array(test_data[:, feature_size:])


    all_mses = np.array(all_mses) / 10 ** -3

    # plot histogram of MSEs
    #sns.histplot(dummy_mses, bins=40, color='pink', label='Aggregate model', alpha=0.8, edgecolor=None)
    names = ["NotZero", 'Lindel']
    for i in range(len(names)-1):
        sns.histplot(all_mses[i], bins=40, label=names[i], alpha=0.2, edgecolor=None)
    sns.histplot(all_mses[-1], bins=40, label='Lindel', alpha=0.2, edgecolor=None)
    plt.xticks(np.arange(0, 3, 0.2))
    # set to log scale
    plt.legend()
    plt.xlabel("MSEs (10^-3)")
    plt.xlim(0.0, 2.0)
    plt.title('Model performance on test set')
    plt.savefig("../data/hist.png")
    plt.show()

    '''
    names = ["res=1.8", "res=2", "res=3", 'Lindel']
    for i in range(len(names)):
        sns.histplot(all_mses[i+3], bins=40, label=names[i], alpha=0.2, edgecolor=None)
    plt.xticks(np.arange(0, 3, 0.2))
    # set to log scale
    plt.legend()
    plt.xlabel("MSEs (10^-3)")
    plt.xlim(0.0, 2.0)
    plt.title('Model performance on test set')
    plt.savefig("../data/hist.png")
    plt.show()
    '''




if __name__ == "__main__":
    sys.exit(main())