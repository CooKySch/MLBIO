import numpy as np
import pandas as pd
import pickle as pkl
from Connor.Graph import Graph

def main():
    # Load in everything
    label, rev_index, features = pkl.load(open("../data/feature_index_all.pkl", "rb"))
    feature_size = len(features) + 384
    all_data = np.loadtxt("../data/Lindel_training.txt", delimiter="\t", dtype=str)
    # Data description: 4350 x 3591, Training on :3900(samples) x 1:3033(nodes/features)
    data = all_data[0:10, 1:100].astype('float32')
    # print(len(data[:, 0]))
    graph = Graph(data)
    print(len(graph.nodes))
    print(len(graph.edges))



if __name__ == "__main__":
    main()