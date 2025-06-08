# make_dataset.py

import torch
import random
import numpy as np
from torch_geometric.data import Data
from utils.config import *

def simulate_transaction_graph():
    random.seed(SEED)
    np.random.seed(SEED)

    user_nodes = list(range(NUM_USERS))
    merchant_nodes = list(range(NUM_USERS, NUM_USERS + NUM_MERCHANTS))
    total_nodes = NUM_USERS + NUM_MERCHANTS

    # Assign fraud labels to users
    fraud_users = set(random.sample(user_nodes, int(FRAUD_RATIO * NUM_USERS)))
    labels = torch.zeros(total_nodes, dtype=torch.long)
    for idx in fraud_users:
        labels[idx] = 1

    # Generate transactions
    edge_index = []
    edge_attr = []
    for _ in range(NUM_TRANSACTIONS):
        u = random.choice(user_nodes)
        m = random.choice(merchant_nodes)
        edge_index.append([u, m])
        edge_index.append([m, u])  # undirected

        amount = np.random.exponential(scale=50.0 if u not in fraud_users else 10.0)
        transaction_type = random.choice([0, 1, 2])
        time = np.random.randint(0, 100000)

        edge_attr.append([amount, transaction_type, time])
        edge_attr.append([amount, transaction_type, time])

    edge_index = torch.tensor(edge_index).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    x = torch.randn((total_nodes, 16))  # dummy node features
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=labels)

    torch.save(data, GRAPH_OUTPUT_PATH)
    print(f"Graph data saved to {GRAPH_OUTPUT_PATH}")

if __name__ == "__main__":
    simulate_transaction_graph()
