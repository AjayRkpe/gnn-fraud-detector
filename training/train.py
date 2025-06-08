# train.py

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from models.gnn_model import GCNNet
from utils.config import GRAPH_OUTPUT_PATH, SEED
import random
import numpy as np

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data):
    model.eval()
    pred = model(data).argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask] == data.y[mask]
        accs.append(int(correct.sum()) / int(mask.sum()))
    return accs

def split_data(data, train_ratio=0.7, val_ratio=0.1):
    num_nodes = data.num_nodes
    indices = torch.randperm(num_nodes)
    train_end = int(train_ratio * num_nodes)
    val_end = int((train_ratio + val_ratio) * num_nodes)

    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    data.train_mask[indices[:train_end]] = True
    data.val_mask[indices[train_end:val_end]] = True
    data.test_mask[indices[val_end:]] = True
    return data

if __name__ == "__main__":
    from torch_geometric.data import Data
    data = torch.load(GRAPH_OUTPUT_PATH)
    data = split_data(data)

    model = GCNNet(in_channels=data.num_node_features, hidden_channels=32, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 201):
        loss = train(model, data, optimizer)
        train_acc, val_acc, test_acc = test(model, data)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
