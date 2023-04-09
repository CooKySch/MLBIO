import numpy as np
from scipy import stats
import scipy.sparse as sparse
import argparse
import sys
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm





def gen_prediction(seq,prereq, model_del, model_ins, model_indel):
    '''generate the prediction for all classes, redundant classes will be combined'''
    pam = {'AGG':0,'TGG':0,'CGG':0,'GGG':0}
    guide = seq[13:33]
    if seq[33:36] not in pam:
        return (None, None)
    
    label,rev_index,features,frame_shift = prereq
    indels = gen_indel(seq,30) 
    input_indel = onehotencoder(guide).reshape(1, -1)
    input_ins   = onehotencoder(guide[-6:]).reshape(1, -1)
    input_del   = np.concatenate((create_feature_array(features,indels),input_indel),axis=None).reshape(1, -1)
    cmax = gen_cmatrix(indels,label) # combine redundant classes
    dratio, insratio = model_indel.predict(input_indel)[0]
    ds  = model_del.predict(input_del)
    ins = model_ins.predict(input_ins)
    y_hat = np.concatenate((ds*dratio,ins*insratio),axis=None) * cmax
    return (y_hat,np.dot(y_hat,frame_shift))

def softmax(weights):
    return (np.exp(weights)/sum(np.exp(weights)))

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

def label_mh(sample,mh_len):
    '''Function to label microhomology in deletion events'''
    for k in range(len(sample)):
        read = sample[k]
        if read[3] == 'del':
            idx = read[2] + read[4] +17
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

def main():
    parser = argparse.ArgumentParser(description="MSEs for all prediction tasks")
    args = parser.parse_args()

    # load models
    model_del = pkl.load(open("data/RF_deletion_0.9_training.pkl", "rb"))
    model_ins =  pkl.load(open("data/RF_insertion_0.9_training.pkl", "rb"))
    model_indels = pkl.load(open("data/RF_indel_0.9_training.pkl", "rb"))
   
    # load labels
    label, rev_index, features = pkl.load(open("data/feature_index_all.pkl", "rb"))
    feature_size = len(features) + 384
    test_data  = np.loadtxt("data/Lindel_test_with_full_seqs.txt", delimiter='\t', dtype='str')
    data = test_data[:, 1:].astype('float32')
    y = data[:, feature_size:]

    model_preq = pkl.load(open("data/model_prereq.pkl", 'rb'))
    label,rev_index,features,frame_shift = model_preq

    # initialize an empty array to store predicted frameshifts
    predicted_frameshift = []

    # initialize an empty predictions array
    predictions = []
    for sample in tqdm(test_data):
        # get the sequence
        sequence = sample[0]
        pred, p_frame_shift = gen_prediction(sequence, model_preq, model_del, model_ins, model_indels)
        predictions.append(pred)
        predicted_frameshift.append(p_frame_shift)
    
    # save the predictions  and predicted frameshifts
    pkl.dump(predictions, open("data/predictions_0.9_training.pkl", "wb"))
    pkl.dump(predicted_frameshift, open("data/predicted_frameshift_0.9_training.pkl", "wb"))

    return

if __name__ == "__main__":
    sys.exit(main())

