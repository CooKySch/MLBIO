import time
import numpy as np

class Graph:

    def __init__(self, data):
        # Nodes is a list the size of all the features
        # Edges is a dict where node pairs are the keys and PCC is the key
        self.nodes = list()
        self.node_weights = dict()
        self.edges = dict()
        self.data = data
        self.total_edge_weights = 0
        for i in range(len(data.columns)):
            self.nodes.append(i)

    def set_edges(self, edges):
        self.edges = edges

    def tetrachoric(self, node1, node2):
        data_node1 = self.data[:, node1]
        data_node2 = self.data[:, node2]
        count_a = count_b = count_c = count_d = 0
        for i in range(len(data_node1)):
            if data_node1[i] == 0 and data_node2[i] == 0:
                count_a += 1
            elif data_node1[i] == 0:
                count_b += 1
            elif data_node2[i] == 0:
                count_c += 1
            else:
                count_d += 1
        r = np.cos((np.pi / 180) * 180/(1 + np.sqrt(count_b * count_c / (count_a * count_d + 1e-6))))
        return r

    def create_edges(self):
        # For all features, find the correlation between one another
        start = time.time()
        for i in range(0, len(self.data.columns)):
            for j in range(i, len(self.data.columns)):
                if i != j:
                    self.edges[(i, j)] = self.phi_coefficient(i, j)
            if i % 10 == 0:
                print(str(i), ": ", str(time.time() - start))
                start = time.time()
        return

    def tetrachoric_pandas(self, i, j):
        counts = self.data.iloc[:, [i, j]].value_counts().reset_index(name='count')
        count_a = count_b = count_c = count_d = 1e-6
        for idx, row in counts.iterrows():
            if row[i+1] == 0 and row[j+1] == 0:
                count_a += row['count']
            elif row[i+1] == 0:
                count_b += row['count']
            elif row[j+1] == 0:
                count_c += row['count']
            else:
                count_d += row['count']
        r = np.cos(np.pi / (1 + np.sqrt((count_a * count_d) / (count_b * count_c))))
        return r

    def phi_coefficient(self, i, j):
        counts = self.data.iloc[:, [i, j]].value_counts().reset_index(name='count')
        count_a = count_b = count_c = count_d = 1e-6
        for idx, row in counts.iterrows():
            if row[i+1] == 0 and row[j+1] == 0:
                count_a += row['count']
            elif row[i+1] == 0:
                count_b += row['count']
            elif row[j+1] == 0:
                count_c += row['count']
            else:
                count_d += row['count']
        phi = count_a * count_d - count_b * count_c / np.sqrt(
            (count_a + count_b) * (count_a + count_c) * (count_b + count_d) * (count_c + count_d))
        return phi

    def find_node_weights(self):
        # Gets the sum of all weights of the nodes and puts them in node_weights
        total_edge_weights = 0
        for node_pair in self.edges.keys():
            if node_pair[0] not in self.node_weights.keys():
                self.node_weights[node_pair[0]] = dict()
                self.node_weights[node_pair[0]]['weight'] = 0
            if node_pair[1] not in self.node_weights.keys():
                self.node_weights[node_pair[1]] = dict()
                self.node_weights[node_pair[1]]['weight'] = 0
            self.node_weights[node_pair[0]]['weight'] += self.edges[node_pair]
            self.node_weights[node_pair[1]]['weight'] += self.edges[node_pair]
            total_edge_weights += self.edges[node_pair]
        self.total_edge_weights = total_edge_weights


