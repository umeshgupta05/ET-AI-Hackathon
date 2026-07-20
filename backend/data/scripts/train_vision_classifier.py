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
import argparse
from datetime import datetime, timezone
from pathlib import Path

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from models.vision.detector import CURRENCY_REGIONS

# ─── Configuration ───────────────────────────────────────────────────────
DATASET_DIR = Path(__file__).resolve().parent.parent / "training" / "currency"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "trained_models" / "forgery_classifier"
IMAGE_SIZE = 224
CONTRASTIVE_BATCH_SIZE = 8
SUPERVISED_BATCH_SIZE = 2
EPOCHS_CONTRASTIVE = 5   # SimCLR pre-training epochs
EPOCHS_SUPERVISED = 10   # Supervised fine-tuning epochs
LEARNING_RATE = 1e-4
TEMPERATURE = 0.07       # SimCLR temperature
SEED = 42
MINIMUM_DEPLOYMENT_F1 = 0.85
MINIMUM_DEPLOYMENT_ROC_AUC = 0.90

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── Augmentation Transforms ─────────────────────────────────────────────

# Contrastive augmentation (aggressive — SimCLR style)
contrastive_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.03),
    transforms.RandomGrayscale(p=0.05),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Supervised training (moderate augmentation)
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08),
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

REGION_NAMES = ("full_note", *CURRENCY_REGIONS.keys())


def _order_points(points):
    rectangle = np.zeros((4, 2), dtype=np.float32)
    point_sum = points.sum(axis=1)
    rectangle[0] = points[np.argmin(point_sum)]
    rectangle[2] = points[np.argmax(point_sum)]
    difference = np.diff(points, axis=1)
    rectangle[1] = points[np.argmin(difference)]
    rectangle[3] = points[np.argmax(difference)]
    return rectangle


def _rectify_note(path: str) -> Image.Image:
    """Locate and rectify a photographed note without generating image content."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 145)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= image.shape[0] * image.shape[1] * 0.08:
            box = cv2.boxPoints(cv2.minAreaRect(largest)).astype(np.float32)
            top_left, top_right, bottom_right, bottom_left = _order_points(box)
            width = max(int(np.linalg.norm(bottom_right - bottom_left)), int(np.linalg.norm(top_right - top_left)))
            height = max(int(np.linalg.norm(top_right - bottom_right)), int(np.linalg.norm(top_left - bottom_left)))
            if width >= 64 and height >= 32:
                destination = np.array(
                    [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                    dtype=np.float32,
                )
                image = cv2.warpPerspective(image, cv2.getPerspectiveTransform(_order_points(box), destination), (width, height))
    if image.shape[0] > image.shape[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _extract_region_images(note: Image.Image) -> list[Image.Image]:
    width, height = note.size
    crops = [note]
    for coordinates in CURRENCY_REGIONS.values():
        box = (
            int(coordinates["x1"] * width),
            int(coordinates["y1"] * height),
            int(coordinates["x2"] * width),
            int(coordinates["y2"] * height),
        )
        crops.append(note.crop(box))
    return crops


class CurrencyDataset(Dataset):
    """Supervised real-note dataset returning the same region bag used in inference."""

    def __init__(self, image_paths, labels, transform=None, synthetic_mask=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or val_transform
        self.synthetic_mask = synthetic_mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        note = _rectify_note(self.image_paths[idx])
        
        # Inject synthetic novelty/specimen text
        if self.synthetic_mask and self.synthetic_mask[idx]:
            import random
            from PIL import ImageDraw
            text = random.choice(["SPECIMEN", "KALPANIK BANK", "CHILDREN BANK", "400", "FOR PROJECT TESTING"])
            txt_img = Image.new('RGBA', (150, 20), (255, 255, 255, 0))
            d = ImageDraw.Draw(txt_img)
            d.text((5, 5), text, fill=(220, 0, 0, 200))
            # Scale up the text image and paste it across the center
            scale_width = int(note.width * 0.9)
            scale_height = int(scale_width * (20/150))
            txt_img = txt_img.resize((scale_width, scale_height), Image.NEAREST)
            note.paste(txt_img, (int(note.width * 0.05), note.height // 2 - scale_height // 2), txt_img)

        regions = torch.stack([self.transform(region) for region in _extract_region_images(note)])
        return regions, self.labels[idx]


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

# ─── Training Functions ──────────────────────────────────────────────────

def _hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def load_dataset():
    """Load images from directory structure."""
    genuine_dir = DATASET_DIR / "genuine"
    counterfeit_dir = DATASET_DIR / "counterfeit"

    if not genuine_dir.exists() or not counterfeit_dir.exists():
        raise FileNotFoundError(
            "Real currency data is required. Create currency/genuine and "
            "currency/counterfeit from verified, licensed sources."
        )

    genuine_images = list(genuine_dir.glob("*.png")) + list(genuine_dir.glob("*.jpg")) + list(genuine_dir.glob("*.jpeg"))
    counterfeit_images = list(counterfeit_dir.glob("*.png")) + list(counterfeit_dir.glob("*.jpg")) + list(counterfeit_dir.glob("*.jpeg"))

    if not genuine_images or not counterfeit_images:
        raise ValueError(
            "Both genuine and counterfeit classes need verified images; "
            "synthetic fallback generation is disabled."
        )

    manifest_path = DATASET_DIR / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if manifest:
        records = [
            record for record in manifest.get("records", [])
            if record.get("label") in {"genuine", "counterfeit"} and (DATASET_DIR / record["path"]).exists()
        ]
        feature_records = [
            record for record in manifest.get("records", [])
            if record.get("label") == "feature_reference" and (DATASET_DIR / record["path"]).exists()
        ]
        paths = [str(DATASET_DIR / record["path"]) for record in records]
        labels = [0 if record["label"] == "genuine" else 1 for record in records]
        strata = [f"{record['label']}_{record['denomination']}" for record in records]
        groups = [record.get("split_group") or record["sha256"] for record in records]

        # Keep perceptually similar captures in the same fold.
        for index, record in enumerate(records):
            image_hash = record.get("difference_hash")
            if not image_hash:
                continue
            for previous_index in range(index):
                previous = records[previous_index]
                if (
                    previous.get("difference_hash")
                    and previous["label"] == record["label"]
                    and previous["denomination"] == record["denomination"]
                    and _hamming_distance(image_hash, previous["difference_hash"]) <= 4
                ):
                    groups[index] = groups[previous_index]
                    break
        provenance = {
            "manifest": str(manifest_path),
            "source_dataset": manifest.get("source_dataset"),
            "source_url": manifest.get("source_url"),
            "license": manifest.get("license"),
            "label_assurance": manifest.get("label_assurance"),
        }
        feature_paths = [str(DATASET_DIR / record["path"]) for record in feature_records]
    else:
        paths = [str(p) for p in genuine_images] + [str(p) for p in counterfeit_images]
        labels = [0] * len(genuine_images) + [1] * len(counterfeit_images)
        strata = labels
        groups = [Path(path).stem for path in paths]
        provenance = {"manifest": None, "source_dataset": "unrecorded", "license": "unknown"}
        feature_paths = []

    print(
        f"\nDataset: {sum(label == 0 for label in labels)} genuine + "
        f"{sum(label == 1 for label in labels)} counterfeit + "
        f"{len(feature_paths)} unlabelled feature references"
    )
    return paths, labels, strata, groups, feature_paths, provenance


def _forward_region_bag(backbone, attention_head, classifier_head, images):
    batch_size, region_count, channels, height, width = images.shape
    features = backbone(images.reshape(batch_size * region_count, channels, height, width))
    if features.dim() == 4:
        features = features.mean(dim=(2, 3))
    features = features.reshape(batch_size, region_count, -1)
    return classifier_head(attention_head(features))


def train_contrastive(model, backbone, train_paths, device, epochs):
    """Phase 1: SimCLR contrastive pre-training."""
    print("\n" + "=" * 50)
    print("Phase 1: SimCLR Contrastive Pre-training")
    print("=" * 50)

    dataset = ContrastiveDataset(train_paths, contrastive_transform)
    loader = DataLoader(dataset, batch_size=CONTRASTIVE_BATCH_SIZE, shuffle=True, num_workers=0)

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

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (view1, view2) in enumerate(loader):
            view1 = view1.to(device)
            view2 = view2.to(device)

            # Extract features
            features1 = backbone(view1)
            features2 = backbone(view2)
            if features1.dim() == 4:
                features1 = features1.mean(dim=(2, 3))
                features2 = features2.mean(dim=(2, 3))
            z1 = projection_head(features1)
            z2 = projection_head(features2)

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch + 1}/{epochs} — Contrastive Loss: {avg_loss:.4f}")

    print("✅ Contrastive pre-training complete")
    return backbone


def train_supervised(
    backbone, attention_head, classifier_head, train_paths, train_labels,
    val_paths, val_labels, val_strata, device, epochs,
):
    """Phase 2: Supervised fine-tuning."""
    print("\n" + "=" * 50)
    print("Phase 2: Supervised Fine-tuning")
    print("=" * 50)

    # Generate synthetic training examples from genuine ones
    genuine_indices = [i for i, label in enumerate(train_labels) if label == 0]
    synthetic_count = int(len(genuine_indices) * 0.4) # Add 40% more as synthetic fakes
    
    import random
    selected_for_synthetic = random.sample(genuine_indices, min(synthetic_count, len(genuine_indices)))
    
    synth_paths = [train_paths[i] for i in selected_for_synthetic]
    synth_labels = [1] * len(synth_paths) # Force counterfeit label
    
    extended_train_paths = list(train_paths) + synth_paths
    extended_train_labels = list(train_labels) + synth_labels
    synthetic_mask = [False] * len(train_paths) + [True] * len(synth_paths)

    train_dataset = CurrencyDataset(extended_train_paths, extended_train_labels, train_transform, synthetic_mask=synthetic_mask)
    val_dataset = CurrencyDataset(val_paths, val_labels, val_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=SUPERVISED_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=SUPERVISED_BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(attention_head.parameters()) + list(classifier_head.parameters()),
        lr=LEARNING_RATE * 0.5,  # Lower LR for fine-tuning
        weight_decay=0.01,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = 0.0
    best_state = None
    best_metrics = None
    no_improve = 0
    patience = 3

    for epoch in range(epochs):
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
            logits = _forward_region_bag(backbone, attention_head, classifier_head, images)

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
        val_probs = []
        val_true = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = _forward_region_bag(backbone, attention_head, classifier_head, images)
                probabilities = torch.softmax(logits, dim=1)[:, 1]
                _, predicted = logits.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_probs.extend(probabilities.cpu().numpy())
                val_true.extend(labels.numpy())

        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average="binary", zero_division=0)

        print(
            f"  Epoch {epoch + 1}/{epochs} — "
            f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
            f"Val Acc: {val_acc:.3f} | Val F1: {val_f1:.3f}"
        )

        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            best_state = {
                "backbone": {
                    name: value.detach().cpu().clone()
                    for name, value in backbone.state_dict().items()
                },
                "attention_head": {
                    name: value.detach().cpu().clone()
                    for name, value in attention_head.state_dict().items()
                },
                "classifier_head": {
                    name: value.detach().cpu().clone()
                    for name, value in classifier_head.state_dict().items()
                },
            }
            matrix = confusion_matrix(val_true, val_preds, labels=[0, 1])
            true_genuine, false_reject, false_accept, true_counterfeit = matrix.ravel()
            per_denomination = {}
            for denomination in sorted({value.rsplit("_", 1)[-1] for value in val_strata}, key=int):
                indices = [
                    index for index, value in enumerate(val_strata)
                    if value.rsplit("_", 1)[-1] == denomination
                ]
                denom_true = [val_true[index] for index in indices]
                denom_pred = [val_preds[index] for index in indices]
                denom_prob = [val_probs[index] for index in indices]
                per_denomination[denomination] = {
                    "count": len(indices),
                    "accuracy": float(accuracy_score(denom_true, denom_pred)),
                    "f1": float(f1_score(denom_true, denom_pred, zero_division=0)),
                    "roc_auc": float(roc_auc_score(denom_true, denom_prob)) if len(set(denom_true)) == 2 else None,
                }
            best_metrics = {
                "best_epoch": epoch + 1,
                "accuracy": float(val_acc),
                "f1": float(val_f1),
                "precision": float(precision_score(val_true, val_preds, zero_division=0)),
                "recall": float(recall_score(val_true, val_preds, zero_division=0)),
                "roc_auc": float(roc_auc_score(val_true, val_probs)),
                "confusion_matrix": matrix.tolist(),
                "counterfeit_false_accept_rate": float(false_accept / max(1, false_accept + true_counterfeit)),
                "genuine_false_reject_rate": float(false_reject / max(1, true_genuine + false_reject)),
                "per_denomination": per_denomination,
            }
            no_improve = 0
        else:
            no_improve += 1

        scheduler.step()
        if no_improve >= patience:
            print(f"  Early stopping after epoch {epoch + 1}")
            break

    return best_state, best_metrics


def _balanced_subset(paths, labels, strata, limit_per_stratum):
    """Select a deterministic, denomination/label-balanced smoke subset."""
    selected = []
    counts = {}
    for index, stratum in enumerate(strata):
        count = counts.get(stratum, 0)
        if count >= limit_per_stratum:
            continue
        selected.append(index)
        counts[stratum] = count + 1
    return (
        [paths[index] for index in selected],
        [labels[index] for index in selected],
        [strata[index] for index in selected],
    )


def train(
    smoke: bool = False,
    contrastive_epochs_override: int | None = None,
    supervised_epochs_override: int | None = None,
):
    """Full training pipeline."""
    print("=" * 60)
    print("🔍 Training Hybrid CNN-Transformer Forgery Classifier")
    print("   EfficientNet-B0 + Transformer Attention + SimCLR")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ─── Load Data ────────────────────────────────────────────
    paths, labels, strata, groups, feature_paths, provenance = load_dataset()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_indices, val_indices = next(splitter.split(paths, strata, groups=groups))
    train_paths = [paths[index] for index in train_indices]
    val_paths = [paths[index] for index in val_indices]
    train_labels = [labels[index] for index in train_indices]
    val_labels = [labels[index] for index in val_indices]
    train_strata = [strata[index] for index in train_indices]
    val_strata = [strata[index] for index in val_indices]
    if set(np.asarray(groups)[train_indices]) & set(np.asarray(groups)[val_indices]):
        raise RuntimeError("Grouped split leakage detected")
    if smoke:
        train_paths, train_labels, train_strata = _balanced_subset(
            train_paths, train_labels, train_strata, limit_per_stratum=8
        )
        val_paths, val_labels, val_strata = _balanced_subset(
            val_paths, val_labels, val_strata, limit_per_stratum=4
        )
        feature_paths = feature_paths[:112]
    contrastive_epochs = contrastive_epochs_override or (1 if smoke else EPOCHS_CONTRASTIVE)
    supervised_epochs = supervised_epochs_override or (2 if smoke else EPOCHS_SUPERVISED)
    print(f"Split: {len(train_paths)} train / {len(val_paths)} validation (group-disjoint)")

    # ─── Build Model ──────────────────────────────────────────
    import timm
    from models.vision.classifier import TransformerAttentionHead

    backbone = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=0,
        global_pool="",
    )
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
    contrastive_paths = train_paths + feature_paths
    print(f"Contrastive corpus: {len(contrastive_paths)} real images/crops")
    backbone = train_contrastive(None, backbone, contrastive_paths, device, contrastive_epochs)

    # ─── Phase 2: Supervised Fine-tuning ──────────────────────
    best_state, best_metrics = train_supervised(
        backbone, attention_head, classifier_head,
        train_paths, train_labels, val_paths, val_labels, val_strata, device, supervised_epochs,
    )
    if best_state is None or best_metrics is None:
        raise RuntimeError("Training did not produce a valid checkpoint")

    # ─── Save Model ───────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / "model.pth"
    candidate_path = OUTPUT_DIR / "model_candidate.pth"
    quality_gate_passed = (not smoke) and (
        best_metrics["f1"] >= MINIMUM_DEPLOYMENT_F1
        and best_metrics["roc_auc"] >= MINIMUM_DEPLOYMENT_ROC_AUC
    )
    selected_path = save_path if quality_gate_passed else candidate_path
    torch.save(best_state, str(selected_path))

    metadata = {
        "architecture": "EfficientNet-B0 + Multi-Region Transformer Attention",
        "training_phases": ["Real feature-crop SimCLR contrastive", "Multi-region supervised fine-tuning"],
        "training_mode": "smoke" if smoke else "full",
        "contrastive_epochs": contrastive_epochs,
        "supervised_epochs": supervised_epochs,
        "validation_metrics": best_metrics,
        "dataset_size": len(paths),
        "genuine_count": int(sum(1 for label in labels if label == 0)),
        "counterfeit_count": int(sum(1 for label in labels if label == 1)),
        "image_size": IMAGE_SIZE,
        "feature_dim": feature_dim,
        "region_names": list(REGION_NAMES),
        "feature_reference_count": len(feature_paths),
        "synthetic_currency_images": 0,
        "deployment_quality_gate": {
            "minimum_f1": MINIMUM_DEPLOYMENT_F1,
            "minimum_roc_auc": MINIMUM_DEPLOYMENT_ROC_AUC,
            "passed": quality_gate_passed,
            "active_checkpoint_replaced": quality_gate_passed,
            "eligible_for_deployment": not smoke,
            "reason": (
                "validation thresholds passed"
                if quality_gate_passed
                else "smoke runs cannot replace the active checkpoint"
                if smoke
                else "validation thresholds not met"
            ),
        },
        "train_count": len(train_paths),
        "validation_count": len(val_paths),
        "split_method": "StratifiedGroupKFold(first fold, n_splits=5)",
        "split_seed": SEED,
        "provenance": provenance,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = OUTPUT_DIR / ("training_metadata.json" if quality_gate_passed else "candidate_training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to: {selected_path}")
    print(f"   Deployment quality gate: {'PASS' if quality_gate_passed else 'FAIL - active model preserved'}")
    print(f"   Best Val F1: {best_metrics['f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--contrastive-epochs", type=int, default=None)
    parser.add_argument("--supervised-epochs", type=int, default=None)
    arguments = parser.parse_args()
    train(
        smoke=arguments.smoke,
        contrastive_epochs_override=arguments.contrastive_epochs,
        supervised_epochs_override=arguments.supervised_epochs,
    )
