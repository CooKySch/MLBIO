import argparse
import pickle as pkl
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def main():
    # Load in everything
    label, rev_index, features = pkl.load(open("../data/feature_index_all.pkl", "rb"))
    feature_size = len(features) + 384
    #all_data = np.loadtxt("../data/Lindel_training.txt", delimiter="\t", dtype=str)
    df = pd.read_table("../data/Lindel_training.txt", delimiter="\t", header=None)

    # Data description: 4350 x 3591, Training on :3900(samples) x 1:3033(nodes/features)
    columns = list(range(1, 3034))
    df = df.iloc[:4350, columns].copy()

    vars = df.var()
    vars = np.array(vars)

    sorted_vars = dict()
    for idx, var in enumerate(vars):
        sorted_vars[idx] = var

    sorted_vars = dict(sorted(sorted_vars.items(), key=lambda x: x[1]))
    '''plt.title("Variance of features")
    plt.xlabel("Feature")
    plt.ylabel("Variance")
    plt.yscale('log')
    plt.plot(vars)
    plt.show()'''
    model_del = keras.models.load_model("../data/L1_del.h5")
    w_del, b_del = model_del.get_weights()

    mean_weights = dict()
    for feature, mean in enumerate(np.mean(np.abs(w_del), axis=1)):
        mean_weights[feature] = mean

    sorted_weights = dict(sorted(mean_weights.items(), key=lambda x: x[1]))

    print(list(sorted_vars.items())[:10])
    print(list(sorted_weights.items())[:10])
    fractions = list(np.arange(0.2, 1.0, 0.1))
    recognition_rate_vars_weights = list()
    for f in fractions:
        top_N = int(len(sorted_vars)* f)

        top_N_vars = list(sorted_vars.keys())[:top_N]
        top_N_weights = list(sorted_weights.keys())[:top_N]

        recognition_rate_vars_weights.append(len(intersection(top_N_vars, top_N_weights)) / len(top_N_weights))

    plt.plot(fractions, recognition_rate_vars_weights)
    plt.xlabel("Fraction of weights and labels")
    plt.ylabel("Recognition rate")
    #plt.scatter(list(mean_weights.keys()), list(sorted_weights.values()))
    plt.show()

main()