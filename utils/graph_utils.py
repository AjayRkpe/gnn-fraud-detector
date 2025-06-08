# graph_utils.py

import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(edge_index, labels, title="Transaction Graph"):
    G = nx.Graph()
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)

    color_map = ['red' if labels[n] == 1 else 'green' for n in G.nodes()]
    plt.figure(figsize=(12, 8))
    nx.draw(G, node_color=color_map, with_labels=False, node_size=20)
    plt.title(title)
    plt.show()
