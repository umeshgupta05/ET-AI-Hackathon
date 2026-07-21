"""
Training Script — Fine-tune Hybrid CNN-Transformer for Forgery Detection.

Features:
- SimCLR Contrastive Pre-training (NT-Xent Loss) for robust unsupervised representations.
- Staged Fine-tuning (Warmup -> Backbone Unfreeze) on supervised data.
- 5-Fold Cross Validation.
- Detailed metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
- Advanced region transformations (forensic-safe).
- False Reject Rate Threshold Calibration.
"""

import json
import os
import sys
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

# Fix Windows encoding
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

import timm
from models.vision.detector import CURRENCY_REGIONS
from models.vision.backbone_registry import get_backbone_spec
from models.vision.classifier import PreNormMILAggregator, ContrastiveHead, REGION_NAMES, TRANSFORM, TRAIN_TRANSFORM

DATASET_DIR = Path(__file__).resolve().parent.parent / "training" / "currency"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "trained_models" / "forgery_classifier"
CONTRASTIVE_BATCH_SIZE = 8
SUPERVISED_BATCH_SIZE = 2
EPOCHS_CONTRASTIVE = 5
EPOCHS_SUPERVISED = 10
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# Contrastive augmentation (aggressive — SimCLR style)
contrastive_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

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
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None: raise ValueError(f"Unable to decode {path}")
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
                destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
                image = cv2.warpPerspective(image, cv2.getPerspectiveTransform(_order_points(box), destination), (width, height))
    if image.shape[0] > image.shape[1]: image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def _extract_region_images(note: Image.Image) -> list[Image.Image]:
    width, height = note.size
    crops = [note]
    for coords in CURRENCY_REGIONS.values():
        box = (int(coords["x1"] * width), int(coords["y1"] * height), int(coords["x2"] * width), int(coords["y2"] * height))
        crops.append(note.crop(box))
    return crops


class CurrencyDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, synthetic_mask=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or TRANSFORM
        self.synthetic_mask = synthetic_mask

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        note = _rectify_note(self.image_paths[idx])
        if self.synthetic_mask and self.synthetic_mask[idx]:
            import random
            text = random.choice(["SPECIMEN", "KALPANIK BANK", "CHILDREN BANK", "400", "FOR PROJECT TESTING"])
            txt_img = Image.new('RGBA', (150, 20), (255, 255, 255, 0))
            d = ImageDraw.Draw(txt_img)
            d.text((5, 5), text, fill=(220, 0, 0, 200))
            scale_width = int(note.width * 0.9)
            scale_height = int(scale_width * (20/150))
            txt_img = txt_img.resize((scale_width, scale_height), Image.NEAREST)
            note.paste(txt_img, (int(note.width * 0.05), note.height // 2 - scale_height // 2), txt_img)
            
        regions = torch.stack([self.transform(np.array(region)) for region in _extract_region_images(note)])
        mask = torch.zeros(len(REGION_NAMES), dtype=torch.bool)
        return regions, mask, self.labels[idx]


class ContrastiveDataset(Dataset):
    """SimCLR contrastive dataset — returns two augmented views of each image."""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform or contrastive_transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.transform(img)


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss for SimCLR."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / self.temperature
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim = sim.masked_fill(mask, float("-inf"))
        labels = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]).to(z.device)
        return F.cross_entropy(sim, labels)


def train_contrastive(backbone, train_paths, device, epochs, spec):
    print(f"\n{'='*50}\nPhase 1: SimCLR Contrastive Pre-training\n{'='*50}")
    dataset = ContrastiveDataset(train_paths)
    loader = DataLoader(dataset, batch_size=CONTRASTIVE_BATCH_SIZE, shuffle=True, num_workers=0)

    feature_dim = backbone.num_features
    projection_head = nn.Sequential(
        nn.Linear(feature_dim, feature_dim),
        nn.GELU(),
        nn.Linear(feature_dim, 128),
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(projection_head.parameters()),
        lr=spec.default_backbone_lr,
        weight_decay=spec.default_weight_decay,
    )
    criterion = NTXentLoss(temperature=0.07)
    
    backbone.train()
    projection_head.train()

    for epoch in range(epochs):
        total_loss = 0
        for view1, view2 in loader:
            view1, view2 = view1.to(device), view2.to(device)
            f1, f2 = backbone(view1), backbone(view2)
            if f1.dim() == 4:
                f1, f2 = f1.mean(dim=(2, 3)), f2.mean(dim=(2, 3))
            loss = criterion(projection_head(f1), projection_head(f2))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"  Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(loader):.4f}")
    
    print("✅ Contrastive Pre-training Complete.")
    return backbone


def _hamming_distance(left: str, right: str) -> int: return (int(left, 16) ^ int(right, 16)).bit_count()


def load_dataset():
    genuine_dir = DATASET_DIR / "genuine"
    counterfeit_dir = DATASET_DIR / "counterfeit"
    genuine_images = list(genuine_dir.glob("*.png")) + list(genuine_dir.glob("*.jpg")) + list(genuine_dir.glob("*.jpeg"))
    counterfeit_images = list(counterfeit_dir.glob("*.png")) + list(counterfeit_dir.glob("*.jpg")) + list(counterfeit_dir.glob("*.jpeg"))
    manifest_path = DATASET_DIR / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    
    if manifest:
        records = [r for r in manifest.get("records", []) if r.get("label") in {"genuine", "counterfeit"} and (DATASET_DIR / r["path"]).exists()]
        feature_records = [r for r in manifest.get("records", []) if r.get("label") == "feature_reference" and (DATASET_DIR / r["path"]).exists()]
        paths = [str(DATASET_DIR / r["path"]) for r in records]
        labels = [0 if r["label"] == "genuine" else 1 for r in records]
        strata = [f"{r['label']}_{r['denomination']}" for r in records]
        groups = [r.get("split_group") or r["sha256"] for r in records]

        for index, record in enumerate(records):
            image_hash = record.get("difference_hash")
            if not image_hash: continue
            for previous_index in range(index):
                previous = records[previous_index]
                if (previous.get("difference_hash") and previous["label"] == record["label"]
                    and previous["denomination"] == record["denomination"]
                    and _hamming_distance(image_hash, previous["difference_hash"]) <= 4):
                    groups[index] = groups[previous_index]
                    break
        provenance = {"manifest": str(manifest_path), "source_dataset": manifest.get("source_dataset"), "license": manifest.get("license")}
        feature_paths = [str(DATASET_DIR / r["path"]) for r in feature_records]
    else:
        paths = [str(p) for p in genuine_images] + [str(p) for p in counterfeit_images]
        labels = [0] * len(genuine_images) + [1] * len(counterfeit_images)
        strata = labels
        groups = [Path(path).stem for path in paths]
        provenance = {"manifest": None, "license": "unknown"}
        feature_paths = []
    return paths, labels, strata, groups, feature_paths, provenance


def train_supervised_fold(fold_idx, backbone, aggregator, classifier_head, train_paths, train_labels, val_paths, val_labels, device, epochs, spec, args, allow_synthetic=False):
    print(f"\n{'='*50}\nPhase 2: Supervised Fine-tuning (Staged)\n{'='*50}")
    
    # Synthetic injection gating
    synthetic_mask = None
    if allow_synthetic:
        synthetic_mask = [True if i % 2 != 0 else False for i in range(len(train_paths))]

    train_dataset = CurrencyDataset(train_paths, train_labels, TRAIN_TRANSFORM, synthetic_mask=synthetic_mask)
    val_dataset = CurrencyDataset(val_paths, val_labels, TRANSFORM)
    train_loader = DataLoader(dataset=train_dataset, batch_size=SUPERVISED_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=SUPERVISED_BATCH_SIZE, shuffle=False, num_workers=0)

    for param in backbone.parameters(): param.requires_grad = False
    
    optimizer = torch.optim.AdamW(
        list(aggregator.parameters()) + list(classifier_head.parameters()),
        lr=spec.default_head_lr, weight_decay=spec.default_weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler() if args.amp else None

    best_f1 = 0.0
    best_state = None
    best_metrics = None
    all_val_probs, all_val_true = [], []
    
    for epoch in range(epochs):
        if epoch == int(epochs * 0.3):  # Stage B: Unfreeze
            print("  -> Unfreezing backbone for full end-to-end fine-tuning")
            for param in backbone.parameters(): param.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": backbone.parameters(), "lr": spec.default_backbone_lr},
                {"params": aggregator.parameters(), "lr": spec.default_head_lr},
                {"params": classifier_head.parameters(), "lr": spec.default_head_lr},
            ], weight_decay=spec.default_weight_decay)

        backbone.train() if epoch >= int(epochs * 0.3) else backbone.eval()
        aggregator.train(); classifier_head.train()
        
        optimizer.zero_grad()
        for batch_idx, (images, mask, labels) in enumerate(train_loader):
            images, mask, labels = images.to(device), mask.to(device), labels.to(device)
            B, R, C, H, W = images.shape
            
            with torch.amp.autocast(device_type=device.type) if args.amp else torch.enable_grad():
                features = backbone(images.reshape(B*R, C, H, W))
                features = features.mean(dim=(2, 3)).reshape(B, R, -1)
                attended = aggregator(features, mask)
                logits = classifier_head(attended)
                loss = criterion(logits, labels) / args.gradient_accumulation
            
            if args.amp:
                scaler.scale(loss).backward()
                if (batch_idx + 1) % args.gradient_accumulation == 0:
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            else:
                loss.backward()
                if (batch_idx + 1) % args.gradient_accumulation == 0:
                    optimizer.step(); optimizer.zero_grad()

        # Robust Validation & Advanced Metrics
        backbone.eval(); aggregator.eval(); classifier_head.eval()
        val_preds, val_probs, val_true = [], [], []
        with torch.no_grad():
            for images, mask, labels in val_loader:
                images, mask, labels = images.to(device), mask.to(device), labels.to(device)
                B, R, C, H, W = images.shape
                features = backbone(images.reshape(B*R, C, H, W))
                features = features.mean(dim=(2, 3)).reshape(B, R, -1)
                attended = aggregator(features, mask)
                probs = F.softmax(classifier_head(attended), dim=1)[:, 1]
                val_preds.extend((probs > 0.5).long().cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        v_acc = accuracy_score(val_true, val_preds)
        v_f1 = f1_score(val_true, val_preds, average="binary", zero_division=0)
        v_prec = precision_score(val_true, val_preds, zero_division=0)
        v_rec = recall_score(val_true, val_preds, zero_division=0)
        v_auc = roc_auc_score(val_true, val_probs) if len(set(val_true)) > 1 else 0.5
        
        matrix = confusion_matrix(val_true, val_preds, labels=[0, 1])
        true_gen, false_rej, false_acc, true_cft = matrix.ravel()
        
        print(f"  Epoch {epoch + 1}/{epochs} — Acc: {v_acc:.3f} | F1: {v_f1:.3f} | Prec: {v_prec:.3f} | Rec: {v_rec:.3f} | AUC: {v_auc:.3f}")
        
        if v_f1 >= best_f1:
            best_f1 = v_f1
            best_state = {
                "backbone": {k: v.cpu().clone() for k, v in backbone.state_dict().items()},
                "aggregator": {k: v.cpu().clone() for k, v in aggregator.state_dict().items()},
                "classifier_head": {k: v.cpu().clone() for k, v in classifier_head.state_dict().items()},
            }
            best_metrics = {
                "f1": v_f1, "accuracy": v_acc, "precision": v_prec, "recall": v_rec, "roc_auc": v_auc,
                "genuine_false_reject_rate": float(false_rej / max(1, true_gen + false_rej)),
                "counterfeit_false_accept_rate": float(false_acc / max(1, true_cft + false_acc)),
                "true_negative": int(true_gen),
                "true_positive": int(true_cft)
            }
            all_val_probs, all_val_true = val_probs, val_true
            
    return best_state, best_metrics, all_val_probs, all_val_true

def train(args=None, **kwargs):
    if args is None:
        args = argparse.Namespace(
            backbone=kwargs.get("backbone", "efficientnet_b0"),
            smoke=kwargs.get("smoke", False),
            allow_synthetic_debug_data=kwargs.get("allow_synthetic_debug_data", False),
            ssl_mode=kwargs.get("ssl_mode", "simclr" if kwargs.get("backbone", "efficientnet_b0") == "efficientnet_b0" else "none"),
            folds=2 if kwargs.get("smoke", False) else 5,
            fold_index=None, loss="cross_entropy", amp=False, gradient_accumulation=1
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = get_backbone_spec(args.backbone)
    paths, labels, strata, groups, feature_paths, provenance = load_dataset()
    
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    fold_results = []
    best_overall_state = None
    best_overall_f1 = 0.0
    all_val_probs, all_val_true = [], []

    print(f"\nRobust Pipeline Execution - Backbone: {args.backbone}")
    
    for fold, (train_idx, val_idx) in enumerate(splitter.split(paths, strata, groups=groups)):
        if args.fold_index is not None and fold != args.fold_index: continue
        print(f"\n{'='*60}\nFOLD {fold+1}/{args.folds}\n{'='*60}")
        
        train_paths = [paths[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_paths = [paths[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        if args.smoke:
            train_paths, train_labels = train_paths[:8], train_labels[:8]
            val_paths, val_labels = val_paths[:4], val_labels[:4]
            epochs = 3
        else:
            epochs = EPOCHS_SUPERVISED
            
        backbone = timm.create_model(spec.timm_name, pretrained=True, num_classes=0, global_pool="").to(device)
        aggregator = PreNormMILAggregator(feature_dim=backbone.num_features, num_regions=len(REGION_NAMES)).to(device)
        classifier_head = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 2)).to(device)
        
        if args.ssl_mode == "simclr":
            contrastive_paths = train_paths + feature_paths
            if args.smoke: contrastive_paths = contrastive_paths[:8]
            c_epochs = 1 if args.smoke else EPOCHS_CONTRASTIVE
            backbone = train_contrastive(backbone, contrastive_paths, device, c_epochs, spec)
            
        best_state, metrics, val_probs, val_true = train_supervised_fold(
            fold, backbone, aggregator, classifier_head, train_paths, train_labels, val_paths, val_labels, 
            device, epochs, spec, args, allow_synthetic=args.allow_synthetic_debug_data
        )
        
        if metrics is not None:
            fold_results.append(metrics)
            all_val_probs.extend(val_probs)
            all_val_true.extend(val_true)
            if metrics["f1"] >= best_overall_f1:
                best_overall_f1 = metrics["f1"]
                best_overall_state = best_state

    threshold = 0.5
    if all_val_probs:
        for t in np.arange(0.1, 0.9, 0.05):
            preds = [1 if p >= t else 0 for p in all_val_probs]
            matrix = confusion_matrix(all_val_true, preds, labels=[0, 1])
            true_gen, false_rej, false_acc, true_cft = matrix.ravel()
            frr = false_rej / max(1, true_gen + false_rej)
            if frr <= 0.05:  
                threshold = float(t)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = OUTPUT_DIR / args.backbone
    out_dir.mkdir(exist_ok=True)
    
    if best_overall_state:
        best_overall_state["backbone_key"] = args.backbone
        torch.save(best_overall_state, out_dir / "model.pth")
        
    metadata = {
        "backbone_key": args.backbone, "timm_name": spec.timm_name, "region_names": list(REGION_NAMES),
        "calibration": {"threshold": threshold, "policy": "max 5% false reject rate"},
        "fold_results": fold_results,
        "mean_f1": np.mean([m["f1"] for m in fold_results]) if fold_results else 0.0,
        "mean_roc_auc": np.mean([m["roc_auc"] for m in fold_results]) if fold_results else 0.5,
        "mean_precision": np.mean([m["precision"] for m in fold_results]) if fold_results else 0.0,
        "mean_recall": np.mean([m["recall"] for m in fold_results]) if fold_results else 0.0,
    }
    with open(out_dir / "training_metadata.json", "w") as f: json.dump(metadata, f, indent=2)
    print(f"Finished {args.backbone}. Mean F1: {metadata['mean_f1']:.3f}, Threshold: {threshold:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="efficientnet_b0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-synthetic-debug-data", action="store_true")
    parser.add_argument("--ssl-mode", choices=["none", "simclr"], default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int, default=None)
    parser.add_argument("--loss", choices=["cross_entropy", "focal"], default="cross_entropy")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    args = parser.parse_args()
    if args.ssl_mode is None: args.ssl_mode = "simclr" if args.backbone == "efficientnet_b0" else "none"
    train(args)
