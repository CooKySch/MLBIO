import time
import networkx as nx
import pandas as pd
import pickle as pkl
import numpy as np
from Connor.Graph import Graph

def main():
    # Load in everything
    label, rev_index, features = pkl.load(open("../data/feature_index_all.pkl", "rb"))
    feature_size = len(features) + 384
    #all_data = np.loadtxt("../data/Lindel_training.txt", delimiter="\t", dtype=str)
    df = pd.read_table("../data/Lindel_training.txt", delimiter="\t", header=None)

    # Data description: 4350 x 3591, Training on :3900(samples) x 1:3033(nodes/features)
    columns = list(range(1, 3034))
    df = df.iloc[:3900, columns].copy()
    #data = all_data[0:3900, 1:3033].astype('float32')
    # print(len(data[:, 0]))
    graph = Graph(df)

    '''
    graph.create_edges()
    
    f = open("edges_phi.pkl", "wb")
    pkl.dump(graph.edges, f)
    f.close()
    '''

    edges = pkl.load(open("edges_phi.pkl", "rb"))
    graph.set_edges(edges)
    edges_louvain_graph = list()
    for key in edges.keys():
        edges_louvain_graph.append((key[0], key[1], edges[key]))
    graph.find_node_weights()

    louvain_graph = nx.Graph()

    louvain_graph.add_nodes_from(graph.node_weights)
    louvain_graph.add_weighted_edges_from(edges_louvain_graph)

    communities = nx.algorithms.community.louvain_communities(louvain_graph)

    '''for i, community in enumerate(communities):
        print(str(i) + ": " + str(len(community)))
        print(community)

    partitions = nx.algorithms.community.louvain_partitions(louvain_graph)
    for i, partition in enumerate(partitions):
        print(len(partition))
        for j in partition:
            print(str(i) + ": " + str(len(j)))
            #print(j)
    '''

    # Select features from the communities
    all_features = list()
    expected = 0
    for community in communities:
        print(min(community))
        feature_set_size = int(np.sqrt(len(community)))
        expected += feature_set_size
        features = dict()
        for feature in community:
            std = df.iloc[:3900, feature].std()
            features[feature] = std
        features = dict(sorted(features.items(), key=lambda item: item[1]))
        community_features = list(features.keys())[:feature_set_size]
        all_features.extend(community_features)

    f = open("selected_features.pkl", "wb")
    pkl.dump(all_features, f)
    f.close()


if __name__ == "__main__":
    main()