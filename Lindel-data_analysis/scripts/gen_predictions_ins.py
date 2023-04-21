
import sys
import os
import argparse
import pandas as pd
import glob
import datetime as dt
import pickle as pkl
from tensorflow import keras 
import numpy as np

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


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for some encoded feautures")
    parser.add_argument('-f', '--file_encoded_features', dest='file_name', default= "data/Lindel_test.txt", type=str, help="File path of the encoded features")
    parser.add_argument('-m', '--model', dest='model', default= "data/L1_ins.h5", type=str, help="File path of the model")
    parser.add_argument('-o, --output_file_predictions', dest='output_file', default="data/predictions_ins",type=str, help="File path of the output file")
    args = parser.parse_args()

    # Read the encoded features
    label, rev_index, features = pkl.load(open("data/feature_index_all.pkl", "rb"))
    df = pd.read_csv(args.file_name, sep='\t', header=None)

    feature_size = len(features) + 384
    data = df.iloc[:, 1:].astype('float32')

    data6bp_encoded = df.iloc[:, 0].apply(lambda x: onehotencoder(x[-6:]))

    # pandas series to dataframe
    data6bp_encoded = pd.DataFrame(data6bp_encoded.tolist())    

    #X = features, y = labels
    X = data6bp_encoded
    y = data.iloc[:, feature_size:]

    # Load the model h5 file with keras
    model = keras.models.load_model(args.model)

    # Generate predictions
    predictions = model.predict(X)


    # save the predictions with numpy

    np.save(args.output_file, predictions)





    return

if __name__ == "__main__":
    sys.exit(main())