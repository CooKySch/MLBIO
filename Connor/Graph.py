import time
import numpy as np

class Graph:

    def __init__(self, data):
        # Nodes is a list the size of all the features
        # Edges is a dict where node pairs are the keys and PCC is the key
        self.nodes = list()
        self.edges = dict()
        self.data = data
        for i in range(len(data[0])):
            self.nodes.append(i)
        self.create_edges()
        return

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

    def pcc(self, node1, node2):
        ''' Calculates the pearson correlation coefficient between two nodes/features'''
        data_node1 = self.data[:, node1]
        data_node2 = self.data[:, node2]
        mean_node1 = np.mean(data_node1)
        mean_node2 = np.mean(data_node2)
        data_node1 = data_node1 - mean_node1
        data_node2 = data_node2 - mean_node2
        r = np.sum(np.multiply(data_node1, data_node2))/np.sqrt(np.sum(data_node1)**2 * np.sum(data_node2)**2)
        return r


    def create_edges(self):
        # For all features, find the correlation between one another
        start = time.time()
        for i in range(0, len(self.data[0])):
            for j in range(i, len(self.data[0])):
                if i != j:
                    self.edges[(i, j)] = self.tetrachoric(i, j)
            if i % 10 == 0:
                print(time.time() - start)
                start = time.time()
        return