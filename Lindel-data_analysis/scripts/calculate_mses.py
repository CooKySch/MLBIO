
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
    data  = np.loadtxt("data/Lindel_test.txt", delimiter='\t', dtype='str')
    data = data[:, 1:].astype('float32')
    y = data[:, feature_size:]

    # concatenate predictions
    predictions = np.concatenate(((predictions_ins.T * predictions_indels[:, 1]).T, (predictions_del.T * predictions_indels[:, 0]).T), axis=1)

    
    mses = []
    # calculate MSEs
    for i, prediction in enumerate(predictions):
        mses.append(np.mean((prediction - y[i])**2))

    # save the MSEs
    np.save("data/mses", mses)

    # pltot histogram with MSEs with 30 bins and x scale from 0.0 to 1.4  step 0.2 and all multiplied by 10**-3
    plt.hist(mses, bins=25, range=(0.0*10**-3, 1.4*10**-3))
    plt.xlabel("MSE")
    plt.ylabel("Frequency")
    plt.show()

    return

if __name__ == "__main__":
    sys.exit(main())