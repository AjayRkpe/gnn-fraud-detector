# 🕵️‍♂️ GNN Fraud Detector

A modular pipeline to detect fraudulent transactions using Graph Neural Networks (GNNs) built with PyTorch Geometric. The project simulates realistic transaction graphs and classifies nodes (accounts) as fraudulent or not.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.4.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📁 Project Structure

```
gnn-fraud-detector/
├── data/
│   └── make_dataset.py         # Generates synthetic transaction graph
├── models/
│   └── gnn_model.py            # GCN model definition
├── training/
│   └── train.py                # Training and evaluation pipeline
├── utils/
│   ├── config.py               # Global settings and paths
│   └── visualize.py            # Optional: Graph visualization tools
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Graph Data

```bash
python data/make_dataset.py
```

This creates a graph with nodes as bank accounts and edges as transactions, with 5% of the nodes marked as fraudulent.

### 3. Train GCN Model

```bash
python training/train.py
```

This will:
- Split the graph into train/val/test
- Train a 2-layer GCN
- Output epoch-wise performance metrics

---

## 📊 Model Details

- **Architecture:** 2-layer GCN (`GCNConv`)
- **Loss:** Negative log likelihood
- **Optimizer:** Adam
- **Evaluation Metrics:** Accuracy on train/val/test sets

---

## 🔍 Future Work

- Add Graph Attention Networks (GAT) and GraphSAGE
- Incorporate temporal features and edge attributes
- Deploy model using FastAPI or Streamlit
- Add unit tests and CI workflow

---

## 🧠 Why Graphs for Fraud Detection?

Fraudulent behavior often exhibits structural patterns (e.g., rings of accounts), making GNNs an ideal fit. This project simulates such structures to explore GNN effectiveness.

---

## 📄 License

MIT License

---

## 👤 Author

Ajay Rathikumar  
🔗 [LinkedIn](https://www.linkedin.com/in/ajay-a-iiserpune)  
📧 ajay.rathikumar18@gmail.com
