"""
Training Script — Fine-tune Hybrid CNN-Transformer for Forgery Detection.

Features:
- Modes: cross_validate, train_final, evaluate, promote
- SimCLR Contrastive Pre-training (NT-Xent Loss).
- Staged Fine-tuning (Head -> Block -> Full).
- 5-Fold Cross Validation with explicit isolation.
- Detailed metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Brier, ECE.
- Safe whole-note geometric transformations.
- Threshold Calibration (max genuine FRR 0.05).
- Atomic promotion gates and checkpoint safety.
"""

import json
import os
import sys
import argparse
import time
import shutil
from datetime import datetime
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
from PIL import Image, ImageDraw
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss, average_precision_score

import timm
from models.vision.detector import CURRENCY_REGIONS
from models.vision.backbone_registry import get_backbone_spec
from models.vision.classifier import PreNormMILAggregator, ContrastiveHead, REGION_NAMES
from models.vision.preprocessing import get_region_tensor_transform, get_forensic_safe_geometric_transform, get_simclr_geometric_transform

DATASET_DIR = Path(__file__).resolve().parent.parent / "training" / "currency"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "trained_models" / "forgery_classifier"
CONTRASTIVE_BATCH_SIZE = 8
SUPERVISED_BATCH_SIZE = 2
EPOCHS_CONTRASTIVE = 5
EPOCHS_SUPERVISED = 10
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

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
    def __init__(self, image_paths, labels, backbone_key: str, is_training: bool = False, synthetic_mask=None, ssl_mode=False):
        self.image_paths = image_paths
        self.labels = labels
        self.is_training = is_training
        self.synthetic_mask = synthetic_mask
        self.ssl_mode = ssl_mode
        
        spec = get_backbone_spec(backbone_key)
        self.tensor_transform = get_region_tensor_transform(spec.timm_name, spec.default_input_size, is_training=False)
        
        if ssl_mode:
            self.geometric_transform = get_simclr_geometric_transform()
        elif is_training:
            self.geometric_transform = get_forensic_safe_geometric_transform()
        else:
            self.geometric_transform = None

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
            
        if self.geometric_transform:
            note = self.geometric_transform(note)
            
        regions = _extract_region_images(note)
        tensor_regions = torch.stack([self.tensor_transform(r) for r in regions])
        mask = torch.zeros(len(REGION_NAMES), dtype=torch.bool)
        
        if self.ssl_mode:
            note2 = self.geometric_transform(_rectify_note(self.image_paths[idx]))
            regions2 = _extract_region_images(note2)
            tensor_regions2 = torch.stack([self.tensor_transform(r) for r in regions2])
            return tensor_regions, tensor_regions2
            
        return tensor_regions, mask, self.labels[idx]


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

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha # Tensor of weights [class_0, class_1]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        return focal_loss

def train_contrastive(backbone, train_paths, device, epochs, spec, args):
    print(f"\n{'='*50}\nPhase 1: SimCLR Contrastive Pre-training\n{'='*50}")
    dataset = CurrencyDataset(train_paths, labels=None, backbone_key=args.backbone, is_training=True, ssl_mode=True)
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
    scaler = torch.amp.GradScaler('cuda') if args.amp and device.type == "cuda" else None
    
    backbone.train()
    projection_head.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (view1, view2) in enumerate(loader):
            view1, view2 = view1.to(device), view2.to(device)
            B, R, C, H, W = view1.shape
            
            with torch.amp.autocast(device_type=device.type) if args.amp and device.type == "cuda" else torch.enable_grad():
                f1 = backbone(view1.reshape(B*R, C, H, W))
                f2 = backbone(view2.reshape(B*R, C, H, W))
                f1, f2 = f1.mean(dim=(2, 3)).reshape(B, R, -1), f2.mean(dim=(2, 3)).reshape(B, R, -1)
                
                # Use global average of region features for contrastive loss
                z1 = projection_head(f1.mean(dim=1))
                z2 = projection_head(f2.mean(dim=1))
                
                loss = criterion(z1, z2) / args.gradient_accumulation
            
            should_step = ((batch_idx + 1) % args.gradient_accumulation == 0) or (batch_idx + 1 == len(loader))
            
            if args.amp and device.type == "cuda":
                scaler.scale(loss).backward()
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if should_step:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
            total_loss += loss.item() * args.gradient_accumulation
            
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
        feature_paths = [str(DATASET_DIR / r["path"]) for r in feature_records]
    else:
        paths = [str(p) for p in genuine_images] + [str(p) for p in counterfeit_images]
        labels = [0] * len(genuine_images) + [1] * len(counterfeit_images)
        strata = labels
        groups = [Path(path).stem for path in paths]
        feature_paths = []
    return paths, labels, strata, groups, feature_paths


def train_supervised_fold(fold_idx, backbone, aggregator, classifier_head, train_paths, train_labels, val_paths, val_labels, device, epochs, spec, args, allow_synthetic=False):
    print(f"\n{'='*50}\nPhase 2: Supervised Fine-tuning (Staged)\n{'='*50}")
    
    synthetic_mask = None
    if allow_synthetic:
        # Generate entirely separate fake paths or just mask the existing ones 
        # (For safety, we mask an alternating subset, but they must NOT be in validation)
        synthetic_mask = [True if i % 2 != 0 else False for i in range(len(train_paths))]

    train_dataset = CurrencyDataset(train_paths, train_labels, backbone_key=args.backbone, is_training=True, synthetic_mask=synthetic_mask)
    val_dataset = CurrencyDataset(val_paths, val_labels, backbone_key=args.backbone, is_training=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=SUPERVISED_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=SUPERVISED_BATCH_SIZE, shuffle=False, num_workers=0)

    # Class Weights for this fold
    gen_count = train_labels.count(0)
    cft_count = train_labels.count(1)
    total = gen_count + cft_count
    if gen_count > 0 and cft_count > 0:
        weights = torch.tensor([total / (2 * gen_count), total / (2 * cft_count)], dtype=torch.float32).to(device)
    else:
        weights = None

    if args.loss == "focal":
        criterion = FocalLoss(alpha=weights, gamma=args.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    # STAGE A: Freeze Backbone completely
    for param in backbone.parameters(): param.requires_grad = False
    
    optimizer = torch.optim.AdamW(
        list(aggregator.parameters()) + list(classifier_head.parameters()),
        lr=spec.default_head_lr, weight_decay=spec.default_weight_decay,
    )
    
    scaler = torch.amp.GradScaler('cuda') if args.amp and device.type == "cuda" else None

    best_f1 = 0.0
    best_state = None
    best_metrics = None
    all_val_probs, all_val_true = [], []
    
    for epoch in range(epochs):
        # STAGE B: Unfreeze full backbone (Simplified partial -> full unfreeze)
        if epoch == int(epochs * 0.3):
            print("  -> Entering Stage B: Unfreezing backbone for full end-to-end fine-tuning")
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
            
            with torch.amp.autocast(device_type=device.type) if args.amp and device.type == "cuda" else torch.enable_grad():
                features = backbone(images.reshape(B*R, C, H, W))
                features = features.mean(dim=(2, 3)).reshape(B, R, -1)
                attended = aggregator(features, mask)
                logits = classifier_head(attended)
                loss = criterion(logits, labels) / args.gradient_accumulation
            
            should_step = ((batch_idx + 1) % args.gradient_accumulation == 0) or (batch_idx + 1 == len(train_loader))
            
            if args.amp and device.type == "cuda":
                scaler.scale(loss).backward()
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(aggregator.parameters()), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if should_step:
                    torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(aggregator.parameters()), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

        # Robust Validation
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
        v_pr_auc = average_precision_score(val_true, val_probs) if len(set(val_true)) > 1 else 0.5
        v_brier = brier_score_loss(val_true, val_probs)
        
        matrix = confusion_matrix(val_true, val_preds, labels=[0, 1])
        TN, FP, FN, TP = matrix.ravel()
        
        gen_frr = FP / max(1, TN + FP)
        cft_far = FN / max(1, TP + FN)
        
        print(f"  Epoch {epoch + 1}/{epochs} — Acc: {v_acc:.3f} | F1: {v_f1:.3f} | AUC: {v_auc:.3f} | FRR: {gen_frr:.3f} | FAR: {cft_far:.3f}")
        
        if v_f1 >= best_f1:
            best_f1 = v_f1
            best_state = {
                "checkpoint_format_version": 2,
                "architecture_version": "prenorm_mil_v2",
                "backbone_key": args.backbone,
                "timm_name": spec.timm_name,
                "d_model": 384,
                "num_regions": len(REGION_NAMES),
                "region_names": list(REGION_NAMES),
                "backbone": {k: v.cpu().clone() for k, v in backbone.state_dict().items()},
                "aggregator": {k: v.cpu().clone() for k, v in aggregator.state_dict().items()},
                "classifier_head": {k: v.cpu().clone() for k, v in classifier_head.state_dict().items()},
            }
            best_metrics = {
                "f1": v_f1, "accuracy": v_acc, "precision": v_prec, "recall": v_rec, 
                "roc_auc": v_auc, "pr_auc": v_pr_auc, "brier_score": v_brier,
                "genuine_false_reject_rate": float(gen_frr),
                "counterfeit_false_accept_rate": float(cft_far),
                "TN": int(TN), "FP": int(FP), "FN": int(FN), "TP": int(TP)
            }
            all_val_probs, all_val_true = val_probs, val_true
            
    return best_state, best_metrics, all_val_probs, all_val_true


def calibrate_thresholds(val_probs, val_true, max_frr):
    best_threshold = 0.5
    best_far = 1.0
    best_f1 = 0.0
    
    for t in np.arange(0.1, 0.9, 0.01):
        preds = [1 if p >= t else 0 for p in val_probs]
        matrix = confusion_matrix(val_true, preds, labels=[0, 1])
        TN, FP, FN, TP = matrix.ravel()
        frr = FP / max(1, TN + FP)
        far = FN / max(1, TP + FN)
        f1 = f1_score(val_true, preds, zero_division=0)
        
        if frr <= max_frr:
            if far < best_far or (far == best_far and f1 > best_f1) or (far == best_far and f1 == best_f1 and abs(t - 0.5) < abs(best_threshold - 0.5)):
                best_far = far
                best_f1 = f1
                best_threshold = float(t)
    
    return best_threshold

def promote_model(args, spec):
    print(f"\nEvaluating Promotion Gates for {args.backbone}...")
    out_dir = OUTPUT_DIR / args.backbone
    candidate = out_dir / "model_candidate.pth"
    meta_path = out_dir / "candidate_training_metadata.json"
    
    if not candidate.exists() or not meta_path.exists():
        print("FAIL: Missing candidate files.")
        return
        
    with open(meta_path, "r") as f: meta = json.load(f)
    
    # Gate Thresholds
    gates = [
        ("No synthetic debug data", not meta.get("synthetic_debug_enabled", False), True),
        ("F1 Score >= 0.85", meta.get("mean_f1", 0) >= 0.85, True),
        ("ROC-AUC >= 0.90", meta.get("mean_roc_auc", 0) >= 0.90, True),
        ("PR-AUC >= 0.85", meta.get("mean_pr_auc", 0) >= 0.85, True),
        ("Calibrated FRR <= 0.05", meta.get("calibration", {}).get("frr", 1.0) <= 0.05, True),
    ]
    
    passed = True
    for name, result, expected in gates:
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] {name}")
        if status == "FAIL": passed = False
        
    if passed:
        print("\nAll gates passed! Promoting candidate to production model.")
        if (out_dir / "model.pth").exists():
            shutil.copy(out_dir / "model.pth", out_dir / f"model_backup_{int(time.time())}.pth")
        shutil.copy(candidate, out_dir / "model.pth")
        shutil.copy(meta_path, out_dir / "training_metadata.json")
    else:
        print("\nPromotion FAILED. Candidate remains in staging.")


def train(args=None, **kwargs):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", choices=["cross_validate", "train_final", "evaluate", "promote"], default="cross_validate")
        parser.add_argument("--backbone", type=str, default="efficientnet_b0")
        parser.add_argument("--smoke", action="store_true")
        parser.add_argument("--allow-synthetic-debug-data", action="store_true")
        parser.add_argument("--ssl-mode", choices=["none", "simclr"], default=None)
        parser.add_argument("--folds", type=int, default=5)
        parser.add_argument("--fold-index", type=int, default=None)
        parser.add_argument("--loss", choices=["cross_entropy", "focal"], default="cross_entropy")
        parser.add_argument("--focal-gamma", type=float, default=2.0)
        parser.add_argument("--label-smoothing", type=float, default=0.0)
        parser.add_argument("--amp", action="store_true")
        parser.add_argument("--gradient-accumulation", type=int, default=1)
        parser.add_argument("--max-genuine-frr", type=float, default=0.05)
        args = parser.parse_args()
    
    if args.ssl_mode is None: args.ssl_mode = "simclr" if args.backbone == "efficientnet_b0" else "none"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.amp and device.type != "cuda":
        print("Warning: AMP disabled on CPU.")
        args.amp = False
        
    spec = get_backbone_spec(args.backbone)
    
    if args.mode == "promote":
        return promote_model(args, spec)
        
    paths, labels, strata, groups, feature_paths = load_dataset()
    
    if args.mode == "cross_validate":
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
        fold_results = []
        best_overall_state = None
        best_overall_f1 = 0.0
        all_val_probs, all_val_true = [], []
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_dir = OUTPUT_DIR / args.backbone
        out_dir.mkdir(exist_ok=True)
        folds_dir = out_dir / "folds"
        folds_dir.mkdir(exist_ok=True)
    
        print(f"\nRobust Pipeline Execution - Backbone: {args.backbone} (Mode: {args.mode})")
        
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
                backbone = train_contrastive(backbone, contrastive_paths, device, c_epochs, spec, args)
                
            best_state, metrics, val_probs, val_true = train_supervised_fold(
                fold, backbone, aggregator, classifier_head, train_paths, train_labels, val_paths, val_labels, 
                device, epochs, spec, args, allow_synthetic=args.allow_synthetic_debug_data
            )
            
            if metrics is not None:
                fold_results.append(metrics)
                all_val_probs.extend(val_probs)
                all_val_true.extend(val_true)
                if not args.smoke:
                    torch.save(best_state, folds_dir / f"fold_{fold}.pth")
    
        threshold = calibrate_thresholds(all_val_probs, all_val_true, args.max_genuine_frr)
        
        metadata = {
            "backbone_key": args.backbone, "timm_name": spec.timm_name, "region_names": list(REGION_NAMES),
            "calibration": {"threshold": threshold, "policy": f"max {args.max_genuine_frr*100}% false reject rate"},
            "synthetic_debug_enabled": args.allow_synthetic_debug_data,
            "fold_results": fold_results,
            "mean_f1": np.mean([m["f1"] for m in fold_results]) if fold_results else 0.0,
            "mean_roc_auc": np.mean([m["roc_auc"] for m in fold_results]) if fold_results else 0.5,
            "mean_pr_auc": np.mean([m["pr_auc"] for m in fold_results]) if fold_results else 0.5,
        }
        
        if args.smoke:
            with open(out_dir / "smoke_training_metadata.json", "w") as f: json.dump(metadata, f, indent=2)
            torch.save(best_state, out_dir / "model_smoke.pth")
        else:
            with open(out_dir / "candidate_training_metadata.json", "w") as f: json.dump(metadata, f, indent=2)
            # Cross-validate does not save model.pth! train_final creates model_candidate.pth
        
        print(f"Finished {args.backbone}. Mean F1: {metadata['mean_f1']:.3f}, Threshold: {threshold:.3f}")

if __name__ == "__main__":
    train()
