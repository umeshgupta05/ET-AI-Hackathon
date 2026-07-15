"""
Training Script — Fine-tune DistilBERT for Scam Detection.

Uses pure PyTorch training loop (no HF Trainer — avoids TF/Keras conflicts).

Usage:
    cd backend
    python data/scripts/train_text_classifier.py
"""

import json
import os
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ─── Configuration ───────────────────────────────────────────────────────
BASE_MODEL = "cross-encoder/nli-distilroberta-base"
DATASET_PATH = Path(__file__).resolve().parent.parent / "training" / "scam_detection_dataset.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "trained_models" / "scam_classifier"
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 2e-5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


class ScamDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def train(smoke: bool = False):
    print("=" * 60)
    print("🎓 Fine-tuning DistilBERT for Scam Detection")
    print("   Pure PyTorch training loop")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ─── Load Dataset ─────────────────────────────────────────
    if not DATASET_PATH.exists():
        from generate_text_dataset import save_dataset
        save_dataset()

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [s["text"] for s in data]
    labels = [s["label"] for s in data]
    groups = [s.get("template_group", f"row:{index}") for index, s in enumerate(data)]
    print(f"\nDataset: {len(texts)} samples (Scam: {sum(labels)}, Legit: {len(labels) - sum(labels)})")

    # Split
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_indices, val_indices = next(splitter.split(texts, labels, groups=groups))
    train_texts = [texts[index] for index in train_indices]
    val_texts = [texts[index] for index in val_indices]
    train_labels = [labels[index] for index in train_indices]
    val_labels = [labels[index] for index in val_indices]
    if set(np.asarray(groups)[train_indices]) & set(np.asarray(groups)[val_indices]):
        raise RuntimeError("Template-group leakage detected")
    print(f"  Train: {len(train_texts)}, Val: {len(val_texts)}")

    # ─── Tokenize ─────────────────────────────────────────────
    print(f"\nLoading model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, ignore_mismatched_sizes=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable:,} trainable")

    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")
    val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt")

    train_dataset = ScamDataset(train_enc, train_labels)
    val_dataset = ScamDataset(val_enc, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # ─── Optimizer ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    training_epochs = 1 if smoke else EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_epochs)

    best_f1 = 0.0
    best_state = None
    best_metrics = None
    patience = 3
    no_improve = 0

    # ─── Training Loop ────────────────────────────────────────
    print(f"\n🚀 Training for {training_epochs} epochs...")

    for epoch in range(training_epochs):
        # Train
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validate
        model.eval()
        all_preds = []
        all_true = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(batch["labels"].numpy())

        val_acc = accuracy_score(all_true, all_preds)
        val_f1 = f1_score(all_true, all_preds, average="binary", zero_division=0)
        val_prec = precision_score(all_true, all_preds, average="binary", zero_division=0)
        val_rec = recall_score(all_true, all_preds, average="binary", zero_division=0)

        print(f"  Epoch {epoch+1:2d}/{training_epochs} — Loss: {avg_loss:.4f} | "
              f"Val Acc: {val_acc:.3f} | F1: {val_f1:.3f} | Prec: {val_prec:.3f} | Rec: {val_rec:.3f}")

        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "accuracy": float(val_acc),
                "f1": float(val_f1),
                "precision": float(val_prec),
                "recall": float(val_rec),
            }
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break

        scheduler.step()

    # ─── Save ─────────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)

    final_path = OUTPUT_DIR / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    metadata = {
        "base_model": BASE_MODEL,
        "dataset_size": len(texts),
        "scam_count": int(sum(labels)),
        "legitimate_count": int(len(labels) - sum(labels)),
        "best_val_f1": best_f1,
        "validation_metrics": best_metrics,
        "epochs_trained": epoch + 1,
        "training_mode": "smoke" if smoke else "full",
        "split_method": "StratifiedGroupKFold(first fold, n_splits=5)",
        "train_count": len(train_texts),
        "validation_count": len(val_texts),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(final_path / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Model saved to: {final_path}")
    print(f"   Best Val F1: {best_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    train(smoke=arguments.smoke)
