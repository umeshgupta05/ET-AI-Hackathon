"""
Training Script — Fine-tune Hybrid CNN-Transformer for Forgery Detection.

Fine-tunes EfficientNet-B0 backbone + Transformer attention head on
currency note images. Uses transfer learning from ImageNet with
contrastive SimCLR pre-training followed by supervised fine-tuning.

Usage:
    cd backend
    python data/scripts/train_vision_classifier.py

The trained model is saved to data/trained_models/forgery_classifier/
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ─── Configuration ───────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).resolve().parent.parent / "training" / "currency"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "trained_models" / "forgery_classifier"
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS_CONTRASTIVE = 5   # SimCLR pre-training epochs
EPOCHS_SUPERVISED = 10   # Supervised fine-tuning epochs
LEARNING_RATE = 1e-4
TEMPERATURE = 0.07       # SimCLR temperature
SEED = 42
SYNTHETIC_PER_CLASS = int(os.getenv("SYNTHETIC_PER_CLASS", "250"))

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── Augmentation Transforms ─────────────────────────────────────────────

# Contrastive augmentation (aggressive — SimCLR style)
contrastive_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Supervised training (moderate augmentation)
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Validation (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─── Datasets ─────────────────────────────────────────────────────────────

class CurrencyDataset(Dataset):
    """Standard supervised dataset for currency classification."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or val_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class ContrastiveDataset(Dataset):
    """SimCLR contrastive dataset — returns two augmented views of each image."""

    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform or contrastive_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2


# ─── NT-Xent Loss (SimCLR) ───────────────────────────────────────────────

class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss for SimCLR."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # [2B, D]

        sim = torch.mm(z, z.T) / self.temperature  # [2B, 2B]

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim = sim.masked_fill(mask, float("-inf"))

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size),
        ]).to(z.device)

        loss = F.cross_entropy(sim, labels)
        return loss


# ─── Synthetic Data Generation ────────────────────────────────────────────

def generate_synthetic_data():
    """
    Generate synthetic training images if no real dataset is available.
    Creates fake 'currency' images with distinct patterns for genuine vs. counterfeit.
    """
    print("📸 No real currency images found. Generating synthetic training data...")

    genuine_dir = DATASET_DIR / "genuine"
    counterfeit_dir = DATASET_DIR / "counterfeit"
    genuine_dir.mkdir(parents=True, exist_ok=True)
    counterfeit_dir.mkdir(parents=True, exist_ok=True)

    for folder in (genuine_dir, counterfeit_dir):
        for image_path in folder.glob("*.png"):
            image_path.unlink()

    for i in range(SYNTHETIC_PER_CLASS):
        # Genuine: clean, structured patterns
        img = np.random.randint(200, 240, (224, 224, 3), dtype=np.uint8)
        # Add "watermark" pattern
        img[50:70, 30:100, :] = [180, 200, 220]
        # Add "serial number" region
        img[180:195, 20:180, :] = [100, 100, 100]
        # Add fine grid (security feature)
        img[80:160:4, 30:190, :] = [190, 190, 190]
        img[32:190, 112:116, :] = [110, 130, 145]
        img[120:136, 140:190, :] = [145, 172, 120]
        img = np.clip(
            img.astype(np.int16) + np.random.randint(-5, 6, img.shape),
            0,
            255,
        ).astype(np.uint8)

        pil_img = Image.fromarray(img)
        pil_img.save(genuine_dir / f"genuine_{i:03d}.png")

    for i in range(SYNTHETIC_PER_CLASS):
        # Counterfeit: noisy, inconsistent patterns
        img = np.random.randint(180, 250, (224, 224, 3), dtype=np.uint8)
        # Blurred "watermark"
        img[48:74, 28:104, :] = np.random.randint(160, 230, (26, 76, 3), dtype=np.uint8)
        # Misaligned serial
        offset = np.random.randint(-5, 5)
        img[180 + offset:195 + offset, 20:180, :] = np.random.randint(80, 120, (15, 160, 3), dtype=np.uint8)
        # No security grid (or random noise)
        img[80:160, 30:190, :] += np.random.randint(0, 30, (80, 160, 3), dtype=np.uint8).clip(0, 255)
        if i % 3 == 0:
            img = np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=1.2)))
        if i % 4 == 0:
            img[:, :, 1] = np.clip(img[:, :, 1].astype(np.int16) + 24, 0, 255)

        pil_img = Image.fromarray(img.clip(0, 255).astype(np.uint8))
        pil_img.save(counterfeit_dir / f"counterfeit_{i:03d}.png")

    print(f"  Generated {SYNTHETIC_PER_CLASS} genuine + {SYNTHETIC_PER_CLASS} counterfeit synthetic images")
    print("  Replace synthetic images with real notes for production accuracy")


# ─── Training Functions ──────────────────────────────────────────────────

def load_dataset():
    """Load images from directory structure."""
    genuine_dir = DATASET_DIR / "genuine"
    counterfeit_dir = DATASET_DIR / "counterfeit"

    if not genuine_dir.exists() or not counterfeit_dir.exists():
        generate_synthetic_data()

    genuine_images = list(genuine_dir.glob("*.png")) + list(genuine_dir.glob("*.jpg")) + list(genuine_dir.glob("*.jpeg"))
    counterfeit_images = list(counterfeit_dir.glob("*.png")) + list(counterfeit_dir.glob("*.jpg")) + list(counterfeit_dir.glob("*.jpeg"))

    if len(genuine_images) == 0 and len(counterfeit_images) == 0:
        generate_synthetic_data()
        genuine_images = list(genuine_dir.glob("*.png"))
        counterfeit_images = list(counterfeit_dir.glob("*.png"))

    paths = [str(p) for p in genuine_images] + [str(p) for p in counterfeit_images]
    labels = [0] * len(genuine_images) + [1] * len(counterfeit_images)

    print(f"\nDataset: {len(genuine_images)} genuine + {len(counterfeit_images)} counterfeit")
    return paths, labels


def train_contrastive(model, backbone, train_paths, device):
    """Phase 1: SimCLR contrastive pre-training."""
    print("\n" + "=" * 50)
    print("Phase 1: SimCLR Contrastive Pre-training")
    print("=" * 50)

    dataset = ContrastiveDataset(train_paths, contrastive_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Projection head for contrastive learning
    feature_dim = backbone.num_features
    projection_head = nn.Sequential(
        nn.Linear(feature_dim, feature_dim),
        nn.GELU(),
        nn.Linear(feature_dim, 128),
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(projection_head.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    criterion = NTXentLoss(temperature=TEMPERATURE)

    backbone.train()
    projection_head.train()

    for epoch in range(EPOCHS_CONTRASTIVE):
        total_loss = 0
        for batch_idx, (view1, view2) in enumerate(loader):
            view1 = view1.to(device)
            view2 = view2.to(device)

            # Extract features
            z1 = projection_head(backbone(view1))
            z2 = projection_head(backbone(view2))

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch + 1}/{EPOCHS_CONTRASTIVE} — Contrastive Loss: {avg_loss:.4f}")

    print("✅ Contrastive pre-training complete")
    return backbone


def train_supervised(backbone, attention_head, classifier_head, train_paths, train_labels, val_paths, val_labels, device):
    """Phase 2: Supervised fine-tuning."""
    print("\n" + "=" * 50)
    print("Phase 2: Supervised Fine-tuning")
    print("=" * 50)

    train_dataset = CurrencyDataset(train_paths, train_labels, train_transform)
    val_dataset = CurrencyDataset(val_paths, val_labels, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(attention_head.parameters()) + list(classifier_head.parameters()),
        lr=LEARNING_RATE * 0.5,  # Lower LR for fine-tuning
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_SUPERVISED)

    best_f1 = 0.0
    best_state = None

    for epoch in range(EPOCHS_SUPERVISED):
        # Train
        backbone.train()
        attention_head.train()
        classifier_head.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward: CNN → Transformer → Classifier
            features = backbone(images)
            attended = attention_head(features)
            logits = classifier_head(attended)

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)

        # Validate
        backbone.eval()
        attention_head.eval()
        classifier_head.eval()
        val_preds = []
        val_true = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                features = backbone(images)
                attended = attention_head(features)
                logits = classifier_head(attended)
                _, predicted = logits.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(labels.numpy())

        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average="binary", zero_division=0)

        print(
            f"  Epoch {epoch + 1}/{EPOCHS_SUPERVISED} — "
            f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
            f"Val Acc: {val_acc:.3f} | Val F1: {val_f1:.3f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {
                "backbone": backbone.state_dict(),
                "attention_head": attention_head.state_dict(),
                "classifier_head": classifier_head.state_dict(),
            }

        scheduler.step()

    return best_state, best_f1


def train():
    """Full training pipeline."""
    print("=" * 60)
    print("🔍 Training Hybrid CNN-Transformer Forgery Classifier")
    print("   EfficientNet-B0 + Transformer Attention + SimCLR")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ─── Load Data ────────────────────────────────────────────
    paths, labels = load_dataset()

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    # ─── Build Model ──────────────────────────────────────────
    import timm
    from models.vision.classifier import TransformerAttentionHead

    backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
    backbone.to(device)
    feature_dim = backbone.num_features  # 1280

    attention_head = TransformerAttentionHead(feature_dim=feature_dim, num_heads=8, num_layers=2)
    attention_head.to(device)
    proj_dim = attention_head.proj_dim

    classifier_head = nn.Sequential(
        nn.Linear(proj_dim, 512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, 128),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(128, 2),
    ).to(device)

    total_params = (
        sum(p.numel() for p in backbone.parameters())
        + sum(p.numel() for p in attention_head.parameters())
        + sum(p.numel() for p in classifier_head.parameters())
    )
    print(f"Total parameters: {total_params:,}")

    # ─── Phase 1: Contrastive Pre-training ────────────────────
    backbone = train_contrastive(None, backbone, train_paths, device)

    # ─── Phase 2: Supervised Fine-tuning ──────────────────────
    best_state, best_f1 = train_supervised(
        backbone, attention_head, classifier_head,
        train_paths, train_labels, val_paths, val_labels, device,
    )

    # ─── Save Model ───────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / "model.pth"
    torch.save(best_state, str(save_path))

    metadata = {
        "architecture": "EfficientNet-B0 + Transformer Attention (Hybrid)",
        "training_phases": ["SimCLR contrastive", "Supervised fine-tuning"],
        "contrastive_epochs": EPOCHS_CONTRASTIVE,
        "supervised_epochs": EPOCHS_SUPERVISED,
        "best_val_f1": best_f1,
        "dataset_size": len(paths),
        "genuine_count": int(sum(1 for label in labels if label == 0)),
        "counterfeit_count": int(sum(1 for label in labels if label == 1)),
        "image_size": IMAGE_SIZE,
        "feature_dim": feature_dim,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(OUTPUT_DIR / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Model saved to: {save_path}")
    print(f"   Best Val F1: {best_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    train()
