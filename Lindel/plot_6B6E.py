import sys, os, argparse

# Do not print tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas as pd
import pickle as pkl
import numpy as np

import tensorflow as tf
from tensorflow import keras 

import matplotlib.pyplot as plt
import scipy.sparse as sparse
import seaborn as sns
from scipy import stats

from Predictor import gen_prediction, gen_indel, gen_cmatrix

def mse(x, y):
    return ((x-y)**2).mean()

def main():
    parser = argparse.ArgumentParser(description="MSEs for all prediction tasks")
    args = parser.parse_args()

    # Load model files
    model_del = keras.models.load_model("data/L1_del.h5")
    model_ins = keras.models.load_model("data/L1_ins.h5")
    model_indels = keras.models.load_model("data/L2_indel.h5")
    weights = [model_indels.get_weights()[0], model_indels.get_weights()[1],
               model_del.get_weights()[0], model_del.get_weights()[1],
               model_ins.get_weights()[0], model_ins.get_weights()[1]]

   
    # Load labels and features
    label, rev_index, features = pkl.load(open("data/feature_index_all.pkl", "rb"))
    feature_size = len(features) + 384

    # Load test data with full 60 bp sequences
    test_data  = np.loadtxt("data/Lindel_test_with_full_seqs.txt", delimiter='\t', dtype='str')
    data = test_data[:, 1:].astype('float32')
    y = data[:, feature_size:]

    model_preq = pkl.load(open("data/model_prereq.pkl", 'rb'))
    label,rev_index,features,frame_shift = model_preq

    # Initialize an empty array to store predicted frameshifts
    predicted_frameshift = []
    # Initialize an empty predictions array
    predictions = []
    
    for i, sample in enumerate(test_data):
        # Get the sequence
        sequence = sample[0]      

        # Generate prediction
        pred, fs = gen_prediction(sequence, weights, model_preq)

        # Store prediction and predicted frameshift
        predictions.append(pred)
        predicted_frameshift.append(fs)
        
    # Get the actual frameshifts
    actual_frameshift = np.dot(y,frame_shift).tolist()

    # PLOT FRAMESHIFT
    # Plot the actual vs. predicted frameshifts
    for i, pfs in enumerate(predicted_frameshift): 
        if pfs is None:
            predicted_frameshift.pop(i)
            actual_frameshift.pop(i)
    # Calculate the frameshift MSE
    mse = np.mean((np.array(actual_frameshift) - np.array(predicted_frameshift))**2)
    # Keep 3 decimal places
    mse = round(mse, 3)
    # Calculate pearson correlation
    pearson = stats.pearsonr(actual_frameshift, predicted_frameshift)
    # Keep 3 decimal places
    pearson = round(pearson[0], 3)
    
    # Include in the plot the mse and pearson correlation
    plt.title("MSE: {}, Pearson correlation: {}".format(mse, pearson))
    plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, color = "red")
    plt.scatter(actual_frameshift, predicted_frameshift)
    plt.xlabel("Measured frameshift ratio")
    plt.ylabel("Predicted frameshift ratio")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig ("data/frameshift.png")
    plt.close()

    mses = []
    count_invalid = 0
    invalid_seqs = set()
    # Calculate MSEs
    for i, prediction in enumerate(predictions):
        if prediction is None:
            count_invalid += 1
            invalid_seqs.add(test_data[i, 0])
            continue
    
        mses.append(np.mean((prediction - y[i])**2))

    print("Invalid guide sequences:", count_invalid)
    print("Invalid sequences:, not found under 70k seq design", invalid_seqs)
    
    # Print the mean MSE
    print("Mean MSE: {}".format(np.mean(mses)))

    # Save the MSEs
    np.save("data/mses", mses)
    
    # Load aggregate model predictions
    aggregate_test_predictions = pkl.load(open("data/aggregate_model_test_predictions.pkl", "rb"))

    dummy_mses = []
    for i in range(len(y)):
        dmse = np.mean((aggregate_test_predictions[i] - y[i])**2)
        dummy_mses.append(dmse)
    
    # Divide mses and dummy_mses by 10^-3 to make the plot more readable
    dummy_mses = np.array(dummy_mses)/10**-3
    mses = np.array(mses)/10**-3

    # PLOT MSE HISTOGRAM
    sns.histplot(dummy_mses, bins=40, color='pink', label='Aggregate model', alpha=0.8, edgecolor=None)
    sns.histplot(mses, bins=40, color='#ADD8E6', label='Lindel', alpha=0.8, edgecolor=None)
    plt.xticks(np.arange(0, 3, 0.2))
    # Set to log scale
    plt.legend()
    plt.xlabel("MSEs (10^-3)")
    plt.xlim(0.0, 2.0)
    plt.title('Model performance on test set')
    plt.savefig("data/hist.png")
    plt.show()
    
    return

if __name__ == "__main__":
    sys.exit(main())
