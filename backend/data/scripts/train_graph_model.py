"""
Training Script — Train Graph Attention Network for Fraud Node Classification.

Trains the GAT model on the fraud network graph:
 - Nodes: phone numbers, bank accounts
 - Labels: scammer/mule/victim/legitimate
 - Task: Binary classification (fraudulent vs. legitimate)

Usage:
 cd backend
 python data/scripts/train_graph_model.py

The trained model is saved to data/trained_models/fraud_gat/
"""

import json
import sys
from pathlib import Path

# Fix Windows encoding
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from agents.graph_agent import FraudGAT, GraphAgent

# ─── Configuration ───────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "trained_models" / "fraud_gat"
EPOCHS = 200
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 5e-4
HIDDEN_DIM = 32
NUM_HEADS = 4
DROPOUT = 0.3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def train():
    """Train GAT on fraud network graph."""
    print("=" * 60)
    print(" Training Graph Attention Network (GAT)")
    print(" Fraud Node Classification")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ─── Build Graph ──────────────────────────────────────────
    print("\nBuilding fraud graph...")
    import asyncio

    graph_agent = GraphAgent()
    asyncio.run(graph_agent.initialize())

    features, adj, labels = graph_agent._extract_node_features()
    nodes = list(graph_agent._graph.nodes())

    print(f" Nodes: {len(nodes)}")
    print(f" Features per node: {features.shape[1]}")
    print(f" Edges: {int(adj.sum() - len(nodes))}")  # minus self-loops
    print(f" Fraud nodes: {int((labels > 0.5).sum())}")
    print(f" Legit nodes: {int((labels < 0.5).sum())}")

    # Convert to tensors
    x = torch.tensor(features, dtype=torch.float32).to(device)
    a = torch.tensor(adj, dtype=torch.float32).to(device)
    y = torch.tensor((labels > 0.5).astype(np.float32), dtype=torch.float32).to(device)

    # ─── Split: semi-supervised (train on 60%, validate on 40%) ──
    n = len(nodes)
    indices = np.random.permutation(n)
    train_idx = indices[: int(0.6 * n)]
    val_idx = indices[int(0.6 * n) :]

    train_mask = torch.zeros(n, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    val_mask = torch.zeros(n, dtype=torch.bool, device=device)
    val_mask[val_idx] = True

    print(
        f" Train nodes: {train_mask.sum().item()}, Val nodes: {val_mask.sum().item()}"
    )

    # ─── Model ────────────────────────────────────────────────
    model = FraudGAT(
        in_features=features.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nGAT Model: {total_params:,} parameters")
    print(
        f" Architecture: {features.shape[1]}D → GAT(4-head) → {HIDDEN_DIM * NUM_HEADS}D → GAT → 1D"
    )

    # ─── Training ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Handle class imbalance with weighted loss
    pos_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)]).to(device)
    criterion = nn.BCELoss(weight=None)

    best_val_f1 = 0.0
    best_state = None
    patience = 30
    no_improve = 0

    print(f"\n Training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        model.train()
        pred = model(x, a)

        # Weighted loss for class imbalance
        weight = torch.where(y == 1, pos_weight, torch.ones_like(pos_weight))
        loss = F.binary_cross_entropy(
            pred[train_mask], y[train_mask], weight=weight.expand_as(pred)[train_mask]
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validate
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(x, a)
                val_pred_binary = (val_pred[val_mask] > 0.5).cpu().numpy().astype(int)
                val_true = y[val_mask].cpu().numpy().astype(int)

                val_acc = accuracy_score(val_true, val_pred_binary)
                val_f1 = f1_score(
                    val_true, val_pred_binary, average="binary", zero_division=0
                )
                val_prec = precision_score(
                    val_true, val_pred_binary, average="binary", zero_division=0
                )
                val_rec = recall_score(
                    val_true, val_pred_binary, average="binary", zero_division=0
                )

                print(
                    f" Epoch {epoch + 1:3d}/{EPOCHS} — Loss: {loss.item():.4f} | "
                    f"Val Acc: {val_acc:.3f} | F1: {val_f1:.3f} | Prec: {val_prec:.3f} | Rec: {val_rec:.3f}"
                )

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state = model.state_dict().copy()
                    no_improve = 0
                else:
                    no_improve += 1

                    if no_improve >= patience // 10:
                        print(f"\n Early stopping at epoch {epoch + 1}")
                        break

                        # ─── Final Evaluation ─────────────────────────────────────
                        if best_state:
                            model.load_state_dict(best_state)

                            model.eval()
                            with torch.no_grad():
                                final_pred = model(x, a).cpu().numpy()

                                print(f"\n{'=' * 50}")
                                print("Final Node Predictions:")
                                print(f"{'=' * 50}")
                                for i, node in enumerate(nodes):
                                    attrs = graph_agent._graph.nodes[node]
                                    true_label = "FRAUD" if labels[i] > 0.5 else "LEGIT"
                                    pred_label = (
                                        "FRAUD" if final_pred[i] > 0.5 else "LEGIT"
                                    )
                                    status = "" if true_label == pred_label else ""
                                    print(
                                        f" {status} {node:20s} | True: {true_label} | Pred: {pred_label} ({final_pred[i]:.3f}) | Type: {attrs.get('type')}"
                                    )

                                    # ─── Save ─────────────────────────────────────────────────
                                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                                    save_path = OUTPUT_DIR / "gat_model.pth"
                                    torch.save(
                                        best_state or model.state_dict(), str(save_path)
                                    )

                                    metadata = {
                                        "architecture": "Graph Attention Network (GAT)",
                                        "layers": 2,
                                        "attention_heads": NUM_HEADS,
                                        "hidden_dim": HIDDEN_DIM,
                                        "input_features": features.shape[1],
                                        "total_parameters": total_params,
                                        "epochs_trained": EPOCHS,
                                        "best_val_f1": best_val_f1,
                                        "graph_nodes": len(nodes),
                                        "graph_edges": int(adj.sum() - len(nodes)),
                                    }
                                    with open(
                                        OUTPUT_DIR / "training_metadata.json", "w"
                                    ) as f:
                                        json.dump(metadata, f, indent=2)

                                        print(f"\n GAT model saved to: {save_path}")
                                        print(f" Best Val F1: {best_val_f1:.4f}")
                                        print("=" * 60)


if __name__ == "__main__":
    train()
