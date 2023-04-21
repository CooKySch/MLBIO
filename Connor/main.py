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
    df = pd.read_table("../data/Lindel_training.txt", delimiter="\t", header=None)

    # Data description: 4350 x 3591, Training on :3900(samples) x 1:3033(nodes/features)
    columns = list(range(1, 3034))
    df = df.iloc[:3900, columns].copy()
    graph = Graph(df)

    #Comment the following lines if you already have the edge weights to save time
    graph.create_edges()
    f = open("edges_phi.pkl", "wb")
    pkl.dump(graph.edges, f)
    f.close()

    # Load the edge weights, get the node weights
    edges = pkl.load(open("edges_phi.pkl", "rb"))
    graph.set_edges(edges)
    edges_louvain_graph = list()
    for key in edges.keys():
        edges_louvain_graph.append((key[0], key[1], edges[key]))
    graph.find_node_weights()

    # Apply Louvain clustering
    louvain_graph = nx.Graph()
    louvain_graph.add_nodes_from(graph.node_weights)
    louvain_graph.add_weighted_edges_from(edges_louvain_graph)

    # Find features with zero variance.
    zero_features = -1
    all_communities = list(nx.algorithms.community.louvain_communities(louvain_graph, weight='weight'))
    for i in range(len(all_communities)):
        if 0 in all_communities[i]:
            zero_features = i
    del all_communities[zero_features]

    for community in all_communities:
        all_features.extend(list(community))
    all_features = [sorted(all_features)]
    print(len(all_features[0]))
    f = open("selected_features_notzero.pkl", "wb")
    pkl.dump(all_features, f)
    f.close()

    #Uncomment the lines below to select features after clustering with different resolutions
    '''
    all_communities = list()
    for res in [1.2, 1.4, 1.6, 1.8, 2, 3]:
        communities = list(nx.algorithms.community.louvain_communities(louvain_graph, weight='weight', resolution=res))
        all_communities.append(communities)
            
    for communities in all_communities:
        current_features = list()
        for community in communities:
            size = int(np.sqrt(len(community)))
            features = dict()
            for feature in community:
                std = df.iloc[:4350, feature].std()
                features[feature] = std
            features = dict(sorted(features.items(), key=lambda item: item[1], reverse=True))
            current_features.extend(list(features.keys())[:size])

        all_features.append(current_features)
    '''

    # Select features from the communities
    all_features = list()

    #Uncomment the lines below to select the sqrt(C), 2sqrt(C) and C/2 features from the clusters
    '''
    features_p = list()
    features_2p = list()
    features_half = list()
    for community in communities:
        size = int(np.sqrt(len(community)))
        features = dict()
        for feature in community:
            std = df.iloc[:4350, feature].std()
            features[feature] = std
        features = dict(sorted(features.items(), key=lambda item: item[1], reverse=True))
        features_p.extend(list(features.keys())[:size])
        features_2p.extend(list(features.keys())[:2*size])
        features_half.extend(list(features.keys())[:int(0.5*len(community))])
    features_p = sorted(features_p)
    features_2p = sorted(features_2p)
    features_half = sorted(features_half)    
    all_features.append(features_p)
    all_features.append(features_2p)
    all_features.append(features_half)
    '''


if __name__ == "__main__":
    main()