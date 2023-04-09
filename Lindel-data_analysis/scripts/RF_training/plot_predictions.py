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
import datetime as dt
import pickle as pkl
from tensorflow import keras 
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse

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

def gen_prediction_lindel(seq,wb,prereq):
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
    cmax = gen_cmatrix(indels,label) # combine redundant classes
    dratio, insratio = softmax(np.dot(input_indel,w1)+b1)
    ds  = softmax(np.dot(input_del,w2)+b2)
    ins = softmax(np.dot(input_ins,w3)+b3)
    y_hat = np.concatenate((ds*dratio,ins*insratio),axis=None) * cmax
    return (y_hat,np.dot(y_hat,frame_shift))

def main():
    parser = argparse.ArgumentParser(description="plot the predicted frameshifts vs. the actual frameshifts and mse histogram")
    args = parser.parse_args()

    # load labels
    label, rev_index, features = pkl.load(open("data/feature_index_all.pkl", "rb"))
    feature_size = len(features) + 384
    test_data  = np.loadtxt("data/Lindel_test_with_full_seqs.txt", delimiter='\t', dtype='str')
    data = test_data[:, 1:].astype('float32')
    y = data[:, feature_size:]

    model_preq = pkl.load(open("data/model_prereq.pkl", 'rb'))
    label,rev_index,features,frame_shift = model_preq
    # load pkle file with the predicted frameshifts and predictions for Random Forest
    predicted_frameshift = pkl.load(open('data/predicted_frameshift_0.9_training.pkl','rb'))
    predictions = pkl.load(open('data/predictions_0.9_training.pkl','rb'))
    # normalize predictions
    for i, p in enumerate(predictions):
        try:
            predictions[i] = p / np.sum(p)
        except:
            continue

    # get the actual frameshifts
    actual_frameshift = np.dot(y,frame_shift).tolist()

    # plot the actual vs. predicted frameshifts    
    for i, pfs in enumerate(predicted_frameshift): 
        if type(pfs) != np.float64:
            # remove the predicted frameshifts that are not floats
            predicted_frameshift.pop(i)
            actual_frameshift.pop(i)

    mse = np.mean((np.array(actual_frameshift) - np.array(predicted_frameshift))**2)
    mse = round(mse, 3)

    # calculate pearson correlation
    pearson = stats.pearsonr(actual_frameshift, predicted_frameshift)
    # keep 3 decimal places
    pearson = round(pearson[0], 3)
    # include in the plot the mse and pearson correlation
    plt.title("MSE: {}, Pearson correlation: {}".format(mse, pearson))
    plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, color = "red")
    plt.scatter(actual_frameshift, predicted_frameshift)
    plt.xlabel("Measured frameshift ratio")
    plt.ylabel("Predicted frameshift ratio")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig("Lindel-data_analysis/scripts/RF_training/frameshifts_0.9_training.png")
    plt.close()


    # load lindel models
    model_del_lindel = keras.models.load_model("data/L1_del.h5")
    model_ins_lindel = keras.models.load_model("data/L1_ins.h5")
    model_indels_lindel = keras.models.load_model("data/L2_indel.h5")
    weights_lindel = [model_indels_lindel.get_weights()[0], model_indels_lindel.get_weights()[1] , model_del_lindel.get_weights()[0], model_del_lindel.get_weights()[1],  model_ins_lindel.get_weights()[0], model_ins_lindel.get_weights()[1]]

    predictions_lindel = []
    for sample in test_data:
        # get the sequence
        sequence = sample[0]
        pred_lindel = gen_prediction_lindel(sequence, weights_lindel, model_preq)[0]
        predictions_lindel.append(pred_lindel)
    

    mses_lindel = []
    # calculate MSEs
    for i, prediction in enumerate(predictions_lindel):
        if type(prediction) == str  or type(prediction) != np.ndarray:
            continue
        mses_lindel.append(np.mean((prediction - y[i])**2))
    
    # plot the mse histogram
    mses = []
    # calculate MSEs
    for i, prediction in enumerate(predictions):
        if type(prediction) == str  or type(prediction) != np.ndarray:
            continue
        mses.append(np.mean((prediction - y[i])**2))

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
        try: 
            dmse = np.mean((aggregate_test_predictions[i] - y_test[i])**2)
        except:
            continue
        dummy_mses.append(dmse)


    # # devide mses and dummt_mses by 10^-3 to make the plot more readable
    dummy_mses = np.array(dummy_mses)/10**-3
    mses = np.array(mses)/10**-3
    mses_lindel = np.array(mses_lindel)/10**-3

    # plot histogram of MSEs
    sns.histplot(dummy_mses, bins=40, color='pink', label='Aggregate model', alpha=0.8, edgecolor=None)
    sns.histplot(mses_lindel, bins=40, color='#ADD8E6', label='Lindel', alpha=1.0, edgecolor=None)
    sns.histplot(mses, bins=40, color='#F88379', label='Random Forest', alpha=0.7, edgecolor=None)

    plt.xticks(np.arange(0, 3, 0.2))
    # set to log scale
    plt.legend()
    plt.xlabel("MSEs (10^-3)")
    plt.xlim(0.0, 2.0)
    plt.title('Model performance on test set')
    plt.savefig("Lindel-data_analysis/scripts/RF_training/hist_0.9_training.png")

    return

if __name__ == "__main__":
    sys.exit(main())

