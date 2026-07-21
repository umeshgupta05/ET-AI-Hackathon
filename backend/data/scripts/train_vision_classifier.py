"""Train, evaluate, calibrate, and promote the currency-forgery vision model."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
from PIL import Image, ImageDraw
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from models.vision.backbone_registry import get_backbone_spec
from models.vision.classifier import (
    ARCHITECTURE_VERSION,
    CHECKPOINT_FORMAT_VERSION,
    D_MODEL,
    NUM_CLASSES,
    NUM_HEADS,
    NUM_LAYERS,
    REGION_NAMES,
    ContrastiveHead,
    HybridForgeryClassifier,
    PreNormMILAggregator,
    _pool_backbone_output,
    build_classifier_head,
)
from models.vision.detector import CURRENCY_REGIONS
from models.vision.preprocessing import (
    get_base_data_config,
    get_forensic_safe_geometric_transform,
    get_region_tensor_transform,
    get_simclr_geometric_transform,
)

DATASET_DIR = Path(__file__).resolve().parent.parent / "training" / "currency"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "trained_models" / "forgery_classifier"
SEED = 42
SUPERVISED_BATCH_SIZE = 2
CONTRASTIVE_BATCH_SIZE = 8
DEFAULT_CV_EPOCHS = 10
DEFAULT_SSL_EPOCHS = 5


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


@dataclass(frozen=True)
class DatasetBundle:
    paths: list[str]
    labels: list[int]
    strata: list[str]
    groups: list[str]
    denominations: list[str]
    feature_paths: list[str]
    manifest_path: Optional[str]


def _order_points(points: np.ndarray) -> np.ndarray:
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
    if image is None:
        raise ValueError(f"Unable to decode {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 145)
    edges = cv2.dilate(
        edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= image.shape[0] * image.shape[1] * 0.08:
            box = cv2.boxPoints(cv2.minAreaRect(largest)).astype(np.float32)
            top_left, top_right, bottom_right, bottom_left = _order_points(box)
            width = max(
                int(np.linalg.norm(bottom_right - bottom_left)),
                int(np.linalg.norm(top_right - top_left)),
            )
            height = max(
                int(np.linalg.norm(top_right - bottom_right)),
                int(np.linalg.norm(top_left - bottom_left)),
            )
            if width >= 64 and height >= 32:
                destination = np.array(
                    [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                    dtype=np.float32,
                )
                transform = cv2.getPerspectiveTransform(_order_points(box), destination)
                image = cv2.warpPerspective(image, transform, (width, height))
    if image.shape[0] > image.shape[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _extract_region_images(note: Image.Image) -> list[Image.Image]:
    width, height = note.size
    regions = [note]
    for coordinates in CURRENCY_REGIONS.values():
        box = (
            int(coordinates["x1"] * width),
            int(coordinates["y1"] * height),
            int(coordinates["x2"] * width),
            int(coordinates["y2"] * height),
        )
        regions.append(note.crop(box))
    return regions


class CurrencyDataset(Dataset):
    def __init__(
        self,
        paths: list[str],
        labels: Optional[list[int]],
        backbone_key: str,
        is_training: bool,
        ssl_mode: bool = False,
    ) -> None:
        self.paths = paths
        self.labels = labels
        self.ssl_mode = ssl_mode
        spec = get_backbone_spec(backbone_key)
        self.tensor_transform = get_region_tensor_transform(
            spec.timm_name, spec.default_input_size
        )
        if ssl_mode:
            self.whole_note_transform = get_simclr_geometric_transform()
        elif is_training:
            self.whole_note_transform = get_forensic_safe_geometric_transform()
        else:
            self.whole_note_transform = None

    def __len__(self) -> int:
        return len(self.paths)

    def _view(self, path: str) -> torch.Tensor:
        note = _rectify_note(path)
        if self.whole_note_transform is not None:
            note = self.whole_note_transform(note)
        return torch.stack(
            [self.tensor_transform(region) for region in _extract_region_images(note)]
        )

    def __getitem__(self, index: int):
        first = self._view(self.paths[index])
        if self.ssl_mode:
            return first, self._view(self.paths[index])
        if self.labels is None:
            raise RuntimeError("Supervised dataset requires labels")
        missing_mask = torch.zeros(len(REGION_NAMES), dtype=torch.bool)
        return first, missing_mask, int(self.labels[index])


class SyntheticSpecimenDebugDataset(Dataset):
    """Separate appended debug-only examples; never used for validation or promotion."""

    def __init__(self, source_paths: list[str], backbone_key: str, maximum: int = 16) -> None:
        self.source_paths = source_paths[:maximum]
        spec = get_backbone_spec(backbone_key)
        self.tensor_transform = get_region_tensor_transform(
            spec.timm_name, spec.default_input_size
        )

    def __len__(self) -> int:
        return len(self.source_paths)

    def __getitem__(self, index: int):
        note = _rectify_note(self.source_paths[index]).convert("RGBA")
        overlay = Image.new("RGBA", note.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        label = random.choice(("SPECIMEN", "DEMO COPY", "NOT LEGAL TENDER"))
        draw.text((max(4, note.width // 6), max(4, note.height // 2)), label, fill=(220, 0, 0, 210))
        note = Image.alpha_composite(note, overlay).convert("RGB")
        regions = torch.stack(
            [self.tensor_transform(region) for region in _extract_region_images(note)]
        )
        return regions, torch.zeros(len(REGION_NAMES), dtype=torch.bool), 1


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first = F.normalize(first, dim=1)
        second = F.normalize(second, dim=1)
        batch_size = first.shape[0]
        embeddings = torch.cat([first, second], dim=0)
        similarity = embeddings @ embeddings.T / self.temperature
        similarity = similarity.masked_fill(
            torch.eye(2 * batch_size, device=embeddings.device, dtype=torch.bool),
            float("-inf"),
        )
        targets = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size),
                torch.arange(0, batch_size),
            ]
        ).to(embeddings.device)
        return F.cross_entropy(similarity, targets)


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probabilities = F.log_softmax(logits, dim=1)
        probabilities = log_probabilities.exp()
        target_log_probability = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probability = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        losses = -((1.0 - target_probability) ** self.gamma) * target_log_probability
        if self.alpha is not None:
            losses = losses * self.alpha.gather(0, targets)
        if self.reduction == "sum":
            return losses.sum()
        if self.reduction == "none":
            return losses
        return losses.mean()


def supervised_embedding_loss(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Small pairwise objective that makes exported embeddings meaningful."""
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0
    similarities = embeddings @ embeddings.T
    identity = torch.eye(embeddings.shape[0], device=embeddings.device, dtype=torch.bool)
    same_class = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~identity
    different_class = ~labels.unsqueeze(0).eq(labels.unsqueeze(1))
    positive_loss = (
        (1.0 - similarities[same_class]).clamp_min(0).mean()
        if same_class.any()
        else similarities.sum() * 0.0
    )
    negative_loss = (
        (similarities[different_class] - 0.2).clamp_min(0).mean()
        if different_class.any()
        else similarities.sum() * 0.0
    )
    return positive_loss + negative_loss


def _hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def load_dataset(manifest_path: Optional[str] = None) -> DatasetBundle:
    root = DATASET_DIR
    selected_manifest = Path(manifest_path) if manifest_path else root / "source_manifest.json"
    if selected_manifest.exists():
        payload = json.loads(selected_manifest.read_text(encoding="utf-8"))
        records = [
            record
            for record in payload.get("records", [])
            if record.get("label") in {"genuine", "counterfeit"}
            and (root / record["path"]).exists()
        ]
        if not records:
            raise ValueError(f"No usable labelled records in {selected_manifest}")
        feature_records = [
            record
            for record in payload.get("records", [])
            if record.get("label") == "feature_reference"
            and (root / record["path"]).exists()
        ]
        paths = [str(root / record["path"]) for record in records]
        labels = [0 if record["label"] == "genuine" else 1 for record in records]
        denominations = [str(record.get("denomination", "unknown")) for record in records]
        strata = [f"{record['label']}_{denominations[index]}" for index, record in enumerate(records)]
        groups = [str(record.get("split_group") or record.get("sha256") or index) for index, record in enumerate(records)]
        for index, record in enumerate(records):
            difference_hash = record.get("difference_hash")
            if not difference_hash:
                continue
            for previous_index in range(index):
                previous = records[previous_index]
                previous_hash = previous.get("difference_hash")
                if (
                    previous_hash
                    and previous.get("label") == record.get("label")
                    and previous.get("denomination") == record.get("denomination")
                    and _hamming_distance(difference_hash, previous_hash) <= 4
                ):
                    groups[index] = groups[previous_index]
                    break
        return DatasetBundle(
            paths=paths,
            labels=labels,
            strata=strata,
            groups=groups,
            denominations=denominations,
            feature_paths=[str(root / record["path"]) for record in feature_records],
            manifest_path=str(selected_manifest),
        )

    genuine = sorted((root / "genuine").glob("*.*"))
    counterfeit = sorted((root / "counterfeit").glob("*.*"))
    valid_suffixes = {".png", ".jpg", ".jpeg"}
    genuine = [path for path in genuine if path.suffix.lower() in valid_suffixes]
    counterfeit = [path for path in counterfeit if path.suffix.lower() in valid_suffixes]
    paths = [str(path) for path in genuine + counterfeit]
    labels = [0] * len(genuine) + [1] * len(counterfeit)
    if not paths:
        raise ValueError(f"No images found under {root}")
    return DatasetBundle(
        paths=paths,
        labels=labels,
        strata=[str(label) for label in labels],
        groups=[Path(path).stem for path in paths],
        denominations=["unknown"] * len(paths),
        feature_paths=[],
        manifest_path=None,
    )


def build_components(backbone_key: str, device: torch.device, pretrained: bool = True):
    spec = get_backbone_spec(backbone_key)
    backbone = timm.create_model(
        spec.timm_name, pretrained=pretrained, num_classes=0, global_pool=""
    ).to(device)
    aggregator = PreNormMILAggregator(
        feature_dim=int(backbone.num_features), num_regions=len(REGION_NAMES)
    ).to(device)
    classifier_head = build_classifier_head().to(device)
    contrastive_head = ContrastiveHead(D_MODEL).to(device)
    return spec, backbone, aggregator, classifier_head, contrastive_head


def _partial_backbone_modules(backbone: nn.Module) -> list[nn.Module]:
    modules: list[nn.Module] = []
    if hasattr(backbone, "stages"):
        stages = getattr(backbone, "stages")
        modules.append(stages[-1])
    elif hasattr(backbone, "blocks"):
        blocks = getattr(backbone, "blocks")
        modules.append(blocks[-1])
    for name in ("conv_head", "bn2", "norm_pre", "head"):
        module = getattr(backbone, name, None)
        if isinstance(module, nn.Module):
            modules.append(module)
    if not modules:
        children = list(backbone.children())
        if children:
            modules.append(children[-1])
    return modules


def set_backbone_stage(backbone: nn.Module, stage: str) -> None:
    for parameter in backbone.parameters():
        parameter.requires_grad = stage == "full"
    if stage == "partial":
        for module in _partial_backbone_modules(backbone):
            for parameter in module.parameters():
                parameter.requires_grad = True
    elif stage not in {"frozen", "full"}:
        raise ValueError(f"Unknown fine-tuning stage: {stage}")


def build_optimizer(
    spec,
    backbone: nn.Module,
    aggregator: nn.Module,
    classifier_head: nn.Module,
    contrastive_head: nn.Module,
):
    return torch.optim.AdamW(
        [
            {"params": backbone.parameters(), "lr": spec.default_backbone_lr},
            {"params": aggregator.parameters(), "lr": spec.default_head_lr},
            {"params": classifier_head.parameters(), "lr": spec.default_head_lr},
            {"params": contrastive_head.parameters(), "lr": spec.default_head_lr},
        ],
        weight_decay=spec.default_weight_decay,
    )


def build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    total_steps = max(1, total_steps)
    warmup_steps = min(max(0, warmup_steps), total_steps - 1)

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _class_weights(labels: Iterable[int], device: torch.device) -> Optional[torch.Tensor]:
    labels = list(labels)
    counts = [labels.count(0), labels.count(1)]
    if min(counts) == 0:
        return None
    total = sum(counts)
    return torch.tensor(
        [total / (NUM_CLASSES * count) for count in counts],
        dtype=torch.float32,
        device=device,
    )


def _criterion(args, weights: Optional[torch.Tensor]):
    if args.loss == "focal":
        return FocalLoss(alpha=weights, gamma=args.focal_gamma)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)



def should_optimizer_step(batch_index: int, loader_length: int, accumulation: int) -> bool:
    if accumulation < 1:
        raise ValueError("accumulation must be at least 1")
    return (batch_index + 1) % accumulation == 0 or batch_index + 1 == loader_length


def train_contrastive(
    backbone: nn.Module,
    paths: list[str],
    device: torch.device,
    epochs: int,
    spec,
    args,
) -> nn.Module:
    if CONTRASTIVE_BATCH_SIZE < 32:
        print(
            "Warning: SimCLR is running with a small in-batch negative set; "
            "treat SSL as an experiment, not a guaranteed improvement."
        )
    dataset = CurrencyDataset(
        paths, labels=None, backbone_key=args.backbone, is_training=True, ssl_mode=True
    )
    loader = DataLoader(dataset, batch_size=CONTRASTIVE_BATCH_SIZE, shuffle=True, num_workers=0)
    projection = nn.Sequential(
        nn.Linear(int(backbone.num_features), int(backbone.num_features)),
        nn.GELU(),
        nn.Linear(int(backbone.num_features), 128),
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(projection.parameters()),
        lr=spec.default_backbone_lr,
        weight_decay=spec.default_weight_decay,
    )
    criterion = NTXentLoss(args.ssl_temperature)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    backbone.train()
    projection.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_index, (first, second) in enumerate(loader):
            first, second = first.to(device), second.to(device)
            batch, regions, channels, height, width = first.shape
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                first_features = _pool_backbone_output(
                    backbone(first.reshape(batch * regions, channels, height, width))
                ).reshape(batch, regions, -1)
                second_features = _pool_backbone_output(
                    backbone(second.reshape(batch * regions, channels, height, width))
                ).reshape(batch, regions, -1)
                loss = criterion(
                    projection(first_features.mean(dim=1)),
                    projection(second_features.mean(dim=1)),
                ) / args.gradient_accumulation
            should_step = should_optimizer_step(
                batch_index, len(loader), args.gradient_accumulation
            )
            scaler.scale(loss).backward()
            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(projection.parameters()), 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.item()) * args.gradient_accumulation
        print(f"SSL epoch {epoch + 1}/{epochs}: loss={total_loss / max(1, len(loader)):.4f}")
    return backbone


def expected_calibration_error(
    labels: list[int], probabilities: list[float], bins: int = 10
) -> float:
    labels_array = np.asarray(labels)
    probabilities_array = np.asarray(probabilities)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (probabilities_array > lower) & (probabilities_array <= upper)
        if not selected.any():
            continue
        confidence = probabilities_array[selected].mean()
        accuracy = labels_array[selected].mean()
        value += selected.mean() * abs(float(confidence - accuracy))
    return float(value)


def compute_metrics(
    labels: list[int],
    probabilities: list[float],
    threshold: float = 0.5,
    denominations: Optional[list[str]] = None,
) -> dict:
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    genuine_frr = false_positive / max(1, true_negative + false_positive)
    counterfeit_far = false_negative / max(1, true_positive + false_negative)
    specificity = true_negative / max(1, true_negative + false_positive)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(set(labels)) > 1 else 0.5,
        "pr_auc": float(average_precision_score(labels, probabilities)) if len(set(labels)) > 1 else 0.5,
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "expected_calibration_error": expected_calibration_error(labels, probabilities),
        "genuine_false_reject_rate": float(genuine_frr),
        "counterfeit_false_accept_rate": float(counterfeit_far),
        "confusion_matrix": {
            "TN": int(true_negative),
            "FP": int(false_positive),
            "FN": int(false_negative),
            "TP": int(true_positive),
        },
    }
    if denominations:
        per_denomination = {}
        for denomination in sorted(set(denominations)):
            indices = [index for index, value in enumerate(denominations) if value == denomination]
            sub_labels = [labels[index] for index in indices]
            sub_probabilities = [probabilities[index] for index in indices]
            if indices:
                per_denomination[denomination] = compute_metrics(
                    sub_labels, sub_probabilities, threshold, denominations=None
                )
                per_denomination[denomination]["sample_count"] = len(indices)
        metrics["per_denomination"] = per_denomination
    return metrics


def fit_temperature(logits: list[list[float]], labels: list[int]) -> float:
    """Fit scalar temperature by deterministic NLL search on held-out logits."""
    if not logits:
        return 1.0
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    candidates = torch.logspace(math.log10(0.05), math.log10(20.0), steps=160)
    best_temperature = 1.0
    best_loss = float("inf")
    with torch.no_grad():
        for candidate in candidates:
            loss = float(F.cross_entropy(logits_tensor / candidate, labels_tensor).item())
            if loss < best_loss:
                best_loss = loss
                best_temperature = float(candidate.item())
    return best_temperature


def calibrate_thresholds(
    val_probs: list[float],
    val_true: list[int],
    max_frr: float,
    uncertainty_margin: float = 0.05,
) -> dict:
    if not val_probs or not val_true:
        return {
            "counterfeit_threshold": 0.5,
            "threshold": 0.5,
            "uncertainty_margin": uncertainty_margin,
            "frr": 1.0,
            "far": 1.0,
            "f1": 0.0,
            "balanced_accuracy": 0.0,
            "policy_satisfied": False,
        }

    feasible: list[dict] = []
    fallback: list[dict] = []
    for threshold in np.arange(0.05, 0.951, 0.005):
        metrics = compute_metrics(val_true, val_probs, float(threshold))
        candidate = {
            "counterfeit_threshold": float(threshold),
            "threshold": float(threshold),
            "uncertainty_margin": uncertainty_margin,
            "frr": metrics["genuine_false_reject_rate"],
            "far": metrics["counterfeit_false_accept_rate"],
            "f1": metrics["f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
        }
        fallback.append(candidate)
        if candidate["frr"] <= max_frr:
            feasible.append(candidate)

    candidates = feasible or fallback
    if feasible:
        candidates.sort(
            key=lambda item: (
                item["far"],
                -item["balanced_accuracy"],
                -item["f1"],
                abs(item["threshold"] - 0.5),
            )
        )
    else:
        candidates.sort(
            key=lambda item: (
                item["frr"],
                item["far"],
                -item["balanced_accuracy"],
                abs(item["threshold"] - 0.5),
            )
        )
    selected = dict(candidates[0])
    selected["policy_satisfied"] = bool(feasible)
    selected["max_genuine_frr"] = float(max_frr)
    return selected


def apply_temperature(logits: list[list[float]], temperature: float) -> list[float]:
    tensor = torch.tensor(logits, dtype=torch.float32)
    return F.softmax(tensor / temperature, dim=1)[:, 1].tolist()


def make_checkpoint(
    args,
    spec,
    backbone,
    aggregator,
    classifier_head,
    contrastive_head,
    calibration: dict,
    feature_dim: int,
    training_metadata_reference: str,
) -> dict:
    data_config = get_base_data_config(spec.timm_name, spec.default_input_size)
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "backbone_key": args.backbone,
        "timm_name": spec.timm_name,
        "pretrained_tag": spec.pretrained_tag,
        "pretrained_weight_license": spec.license_name,
        "feature_dim": int(feature_dim),
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "num_regions": len(REGION_NAMES),
        "region_names": list(REGION_NAMES),
        "num_classes": NUM_CLASSES,
        "input_size": spec.default_input_size,
        "preprocessing": data_config.to_dict(),
        "backbone": {key: value.detach().cpu() for key, value in backbone.state_dict().items()},
        "aggregator": {key: value.detach().cpu() for key, value in aggregator.state_dict().items()},
        "classifier_head": {key: value.detach().cpu() for key, value in classifier_head.state_dict().items()},
        "contrastive_head": {key: value.detach().cpu() for key, value in contrastive_head.state_dict().items()},
        "calibration": calibration,
        "training_metadata_reference": training_metadata_reference,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _stage_for_epoch(epoch: int, epochs: int, warmup_epochs: int, full_finetune: bool) -> str:
    if epoch < warmup_epochs:
        return "frozen"
    if full_finetune and epoch >= max(warmup_epochs + 1, int(epochs * 0.70)):
        return "full"
    return "partial"


def train_supervised(
    args,
    spec,
    backbone,
    aggregator,
    classifier_head,
    contrastive_head,
    train_dataset: Dataset,
    train_labels: list[int],
    device: torch.device,
    epochs: int,
    validation_dataset: Optional[Dataset] = None,
    validation_denominations: Optional[list[str]] = None,
):
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = (
        DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        if validation_dataset is not None
        else None
    )
    weights = _class_weights(train_labels, device)
    criterion = _criterion(args, weights)
    optimizer = build_optimizer(
        spec, backbone, aggregator, classifier_head, contrastive_head
    )
    optimizer_steps_per_epoch = max(1, math.ceil(len(loader) / args.gradient_accumulation))
    scheduler = build_scheduler(
        optimizer,
        total_steps=max(1, epochs * optimizer_steps_per_epoch),
        warmup_steps=min(
            max(1, args.scheduler_warmup_epochs * optimizer_steps_per_epoch),
            max(1, epochs * optimizer_steps_per_epoch - 1),
        ),
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_state = None
    best_metrics = None
    best_logits: list[list[float]] = []
    best_labels: list[int] = []
    best_epoch = 0
    patience = 0
    current_stage = None

    for epoch in range(epochs):
        stage = _stage_for_epoch(epoch, epochs, args.head_warmup_epochs, args.full_finetune)
        if stage != current_stage:
            set_backbone_stage(backbone, stage)
            current_stage = stage
            print(f"Fine-tuning stage: {stage}")

        backbone.train(stage != "frozen")
        aggregator.train()
        classifier_head.train()
        contrastive_head.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for batch_index, (images, missing_mask, labels) in enumerate(loader):
            images = images.to(device)
            missing_mask = missing_mask.to(device)
            labels = labels.to(device)
            batch, regions, channels, height, width = images.shape
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                pooled = _pool_backbone_output(
                    backbone(images.reshape(batch * regions, channels, height, width))
                ).reshape(batch, regions, -1)
                attended = aggregator(pooled, missing_mask)
                logits = classifier_head(attended)
                embeddings = contrastive_head(attended)
                classification_loss = criterion(logits, labels)
                embedding_loss = supervised_embedding_loss(embeddings, labels)
                loss = (classification_loss + args.embedding_loss_weight * embedding_loss) / args.gradient_accumulation
            should_step = should_optimizer_step(
                batch_index, len(loader), args.gradient_accumulation
            )
            scaler.scale(loss).backward()
            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    list(backbone.parameters())
                    + list(aggregator.parameters())
                    + list(classifier_head.parameters())
                    + list(contrastive_head.parameters()),
                    args.gradient_clip,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            total_loss += float(loss.item()) * args.gradient_accumulation

        print(f"Epoch {epoch + 1}/{epochs}: train_loss={total_loss / max(1, len(loader)):.4f}")
        if validation_loader is None:
            continue

        labels_out, logits_out = predict_loader(
            backbone, aggregator, classifier_head, validation_loader, device
        )
        probabilities = apply_temperature(logits_out, 1.0)
        metrics = compute_metrics(
            labels_out, probabilities, 0.5, validation_denominations
        )
        print(
            f"Validation F1={metrics['f1']:.3f} ROC-AUC={metrics['roc_auc']:.3f} "
            f"FAR={metrics['counterfeit_false_accept_rate']:.3f} "
            f"FRR={metrics['genuine_false_reject_rate']:.3f}"
        )
        if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
            best_metrics = metrics
            best_logits = logits_out
            best_labels = labels_out
            best_epoch = epoch + 1
            best_state = {
                "backbone": {key: value.detach().cpu().clone() for key, value in backbone.state_dict().items()},
                "aggregator": {key: value.detach().cpu().clone() for key, value in aggregator.state_dict().items()},
                "classifier_head": {key: value.detach().cpu().clone() for key, value in classifier_head.state_dict().items()},
                "contrastive_head": {key: value.detach().cpu().clone() for key, value in contrastive_head.state_dict().items()},
            }
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stopping_patience:
                print("Early stopping triggered")
                break

    if best_state is not None:
        backbone.load_state_dict(best_state["backbone"], strict=True)
        aggregator.load_state_dict(best_state["aggregator"], strict=True)
        classifier_head.load_state_dict(best_state["classifier_head"], strict=True)
        contrastive_head.load_state_dict(best_state["contrastive_head"], strict=True)
    return best_metrics, best_logits, best_labels, best_epoch


@torch.no_grad()
def predict_loader(backbone, aggregator, classifier_head, loader, device):
    backbone.eval()
    aggregator.eval()
    classifier_head.eval()
    all_labels: list[int] = []
    all_logits: list[list[float]] = []
    for images, missing_mask, labels in loader:
        images = images.to(device)
        missing_mask = missing_mask.to(device)
        batch, regions, channels, height, width = images.shape
        pooled = _pool_backbone_output(
            backbone(images.reshape(batch * regions, channels, height, width))
        ).reshape(batch, regions, -1)
        logits = classifier_head(aggregator(pooled, missing_mask))
        all_labels.extend(int(value) for value in labels.tolist())
        all_logits.extend(logits.detach().cpu().tolist())
    return all_labels, all_logits


def aggregate_fold_metrics(fold_results: list[dict]) -> dict:
    keys = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "expected_calibration_error",
        "genuine_false_reject_rate",
        "counterfeit_false_accept_rate",
    )
    aggregate = {}
    for key in keys:
        values = [float(result[key]) for result in fold_results if key in result]
        if not values:
            continue
        mean = float(np.mean(values))
        standard_deviation = float(np.std(values))
        confidence = float(1.96 * standard_deviation / math.sqrt(len(values))) if len(values) > 1 else 0.0
        aggregate[key] = {
            "mean": mean,
            "std": standard_deviation,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "ci95": [mean - confidence, mean + confidence],
        }
    return aggregate


def run_cross_validation(args, bundle: DatasetBundle, device: torch.device) -> dict:
    if args.folds < 2:
        raise ValueError("cross_validate requires at least two folds")
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    output = OUTPUT_DIR / args.backbone
    folds_dir = output / "folds"
    output.mkdir(parents=True, exist_ok=True)
    folds_dir.mkdir(parents=True, exist_ok=True)

    fold_results: list[dict] = []
    out_of_fold_logits: list[list[float]] = []
    out_of_fold_labels: list[int] = []
    out_of_fold_denominations: list[str] = []
    completed_folds: list[int] = []
    smoke_checkpoint = None

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(bundle.paths, bundle.strata, groups=bundle.groups)
    ):
        if args.fold_index is not None and fold != args.fold_index:
            continue
        seed_everything(args.seed + fold)
        train_paths = [bundle.paths[index] for index in train_indices]
        train_labels = [bundle.labels[index] for index in train_indices]
        validation_paths = [bundle.paths[index] for index in validation_indices]
        validation_labels = [bundle.labels[index] for index in validation_indices]
        validation_denominations = [bundle.denominations[index] for index in validation_indices]
        if set(bundle.groups[index] for index in train_indices).intersection(
            bundle.groups[index] for index in validation_indices
        ):
            raise RuntimeError("Group leakage detected between training and validation")

        epochs = args.smoke_epochs if args.smoke else args.epochs
        if args.smoke:
            train_paths, train_labels = train_paths[: max(4, args.batch_size * 2)], train_labels[: max(4, args.batch_size * 2)]
            validation_paths, validation_labels = validation_paths[: max(4, args.batch_size * 2)], validation_labels[: max(4, args.batch_size * 2)]
            validation_denominations = validation_denominations[: len(validation_paths)]

        spec, backbone, aggregator, classifier_head, contrastive_head = build_components(
            args.backbone, device, pretrained=not args.no_pretrained
        )
        if args.ssl_mode == "simclr":
            ssl_paths = train_paths + bundle.feature_paths
            backbone = train_contrastive(
                backbone,
                ssl_paths[: max(8, len(ssl_paths))] if args.smoke else ssl_paths,
                device,
                1 if args.smoke else args.ssl_epochs,
                spec,
                args,
            )

        real_training = CurrencyDataset(
            train_paths, train_labels, args.backbone, is_training=True
        )
        training_dataset: Dataset = real_training
        synthetic_count = 0
        if args.allow_synthetic_debug_data:
            synthetic_dataset = SyntheticSpecimenDebugDataset(train_paths, args.backbone)
            training_dataset = ConcatDataset([real_training, synthetic_dataset])
            train_labels = train_labels + [1] * len(synthetic_dataset)
            synthetic_count = len(synthetic_dataset)
        validation_dataset = CurrencyDataset(
            validation_paths, validation_labels, args.backbone, is_training=False
        )
        metrics, logits, labels, best_epoch = train_supervised(
            args,
            spec,
            backbone,
            aggregator,
            classifier_head,
            contrastive_head,
            training_dataset,
            train_labels,
            device,
            epochs,
            validation_dataset,
            validation_denominations,
        )
        if metrics is None:
            continue
        metrics["fold"] = fold
        metrics["best_epoch"] = best_epoch
        metrics["synthetic_debug_count"] = synthetic_count
        fold_results.append(metrics)
        completed_folds.append(fold)
        out_of_fold_logits.extend(logits)
        out_of_fold_labels.extend(labels)
        out_of_fold_denominations.extend(validation_denominations[: len(labels)])

        temporary_calibration = {
            "temperature": 1.0,
            "counterfeit_threshold": 0.5,
            "threshold": 0.5,
            "uncertainty_margin": args.uncertainty_margin,
            "policy_satisfied": False,
        }
        checkpoint = make_checkpoint(
            args,
            spec,
            backbone,
            aggregator,
            classifier_head,
            contrastive_head,
            temporary_calibration,
            int(backbone.num_features),
            "cross_validation_metadata.json",
        )
        if args.smoke:
            smoke_checkpoint = checkpoint
        else:
            torch.save(checkpoint, folds_dir / f"fold_{fold}.pth")

    temperature = fit_temperature(out_of_fold_logits, out_of_fold_labels)
    calibrated_probabilities = apply_temperature(out_of_fold_logits, temperature)
    calibration = calibrate_thresholds(
        calibrated_probabilities,
        out_of_fold_labels,
        args.max_genuine_frr,
        args.uncertainty_margin,
    )
    calibration.update(
        {
            "temperature": temperature,
            "selection_policy": "minimize counterfeit FAR subject to maximum genuine FRR",
            "sample_count": len(out_of_fold_labels),
            "metrics_before": compute_metrics(
                out_of_fold_labels,
                apply_temperature(out_of_fold_logits, 1.0),
                0.5,
            ) if out_of_fold_labels else {},
            "metrics_after": compute_metrics(
                out_of_fold_labels,
                calibrated_probabilities,
                calibration["counterfeit_threshold"],
            ) if out_of_fold_labels else {},
        }
    )
    aggregate = aggregate_fold_metrics(fold_results)
    metadata = {
        "mode": "cross_validate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backbone_key": args.backbone,
        "timm_name": get_backbone_spec(args.backbone).timm_name,
        "dataset_manifest": bundle.manifest_path,
        "region_names": list(REGION_NAMES),
        "folds_requested": args.folds,
        "folds_completed": completed_folds,
        "all_requested_folds_completed": len(completed_folds) == args.folds and args.fold_index is None,
        "fold_results": fold_results,
        "aggregate": aggregate,
        "mean_f1": aggregate.get("f1", {}).get("mean", 0.0),
        "mean_roc_auc": aggregate.get("roc_auc", {}).get("mean", 0.5),
        "mean_pr_auc": aggregate.get("pr_auc", {}).get("mean", 0.5),
        "calibration": calibration,
        "synthetic_debug_enabled": bool(args.allow_synthetic_debug_data),
        "eligible_for_promotion": bool(
            not args.smoke
            and not args.allow_synthetic_debug_data
            and len(completed_folds) == args.folds
            and args.fold_index is None
        ),
        "ssl_mode": args.ssl_mode,
        "ssl_batch_size": CONTRASTIVE_BATCH_SIZE,
        "ssl_temperature": args.ssl_temperature,
        "seed": args.seed,
    }
    if args.smoke:
        (output / "smoke_training_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        if smoke_checkpoint is not None:
            smoke_checkpoint["calibration"] = calibration
            torch.save(smoke_checkpoint, output / "model_smoke.pth")
    else:
        (output / "cross_validation_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    return metadata


def run_final_training(args, bundle: DatasetBundle, device: torch.device) -> dict:
    if args.allow_synthetic_debug_data:
        raise ValueError("train_final refuses synthetic debug data")
    output = OUTPUT_DIR / args.backbone
    cv_path = output / "cross_validation_metadata.json"
    if not cv_path.exists():
        raise FileNotFoundError("Run full cross_validate before train_final")
    cv_metadata = json.loads(cv_path.read_text(encoding="utf-8"))
    if not cv_metadata.get("eligible_for_promotion"):
        raise ValueError("Cross-validation metadata is not eligible for final training")

    epochs = args.final_epochs or max(
        1,
        int(round(np.mean([result.get("best_epoch", args.epochs) for result in cv_metadata.get("fold_results", [])]))),
    )
    spec, backbone, aggregator, classifier_head, contrastive_head = build_components(
        args.backbone, device, pretrained=not args.no_pretrained
    )
    if args.ssl_mode == "simclr":
        backbone = train_contrastive(
            backbone,
            bundle.paths + bundle.feature_paths,
            device,
            args.ssl_epochs,
            spec,
            args,
        )
    training_dataset = CurrencyDataset(
        bundle.paths, bundle.labels, args.backbone, is_training=True
    )
    train_supervised(
        args,
        spec,
        backbone,
        aggregator,
        classifier_head,
        contrastive_head,
        training_dataset,
        bundle.labels,
        device,
        epochs,
        validation_dataset=None,
    )
    calibration = dict(cv_metadata["calibration"])
    metadata = {
        "mode": "train_final",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backbone_key": args.backbone,
        "timm_name": spec.timm_name,
        "dataset_manifest": bundle.manifest_path,
        "training_sample_count": len(bundle.paths),
        "epochs": epochs,
        "cross_validation_reference": str(cv_path),
        "folds_completed": cv_metadata.get("folds_completed", []),
        "mean_f1": cv_metadata.get("mean_f1", 0.0),
        "mean_roc_auc": cv_metadata.get("mean_roc_auc", 0.5),
        "mean_pr_auc": cv_metadata.get("mean_pr_auc", 0.5),
        "aggregate": cv_metadata.get("aggregate", {}),
        "calibration": calibration,
        "synthetic_debug_enabled": False,
        "eligible_for_promotion": True,
        "ssl_mode": args.ssl_mode,
    }
    checkpoint = make_checkpoint(
        args,
        spec,
        backbone,
        aggregator,
        classifier_head,
        contrastive_head,
        calibration,
        int(backbone.num_features),
        "candidate_training_metadata.json",
    )
    output.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output / "model_candidate.pth")
    (output / "candidate_training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def run_evaluation(args, bundle: DatasetBundle) -> dict:
    if not args.evaluation_manifest:
        raise ValueError("evaluate requires --evaluation-manifest")
    classifier = HybridForgeryClassifier(
        backbone_key=args.backbone, checkpoint_name="model_candidate.pth"
    )
    asyncio.run(classifier.initialize())
    if not classifier._trained_weights_loaded:
        raise RuntimeError("Candidate checkpoint could not be loaded")

    probabilities: list[float] = []
    labels: list[int] = []
    denominations: list[str] = []
    for path, label, denomination in zip(
        bundle.paths, bundle.labels, bundle.denominations
    ):
        note = _rectify_note(path)
        region_images = _extract_region_images(note)
        regions = {
            name: cv2.cvtColor(np.asarray(region.convert("RGB")), cv2.COLOR_RGB2BGR)
            for name, region in zip(REGION_NAMES, region_images)
        }
        result = classifier.classify_all_regions(regions)
        if not result.get("model_available"):
            raise RuntimeError("Candidate inference became unavailable")
        probabilities.append(float(result["fused_counterfeit_score"]))
        labels.append(label)
        denominations.append(denomination)

    metrics = compute_metrics(
        labels,
        probabilities,
        classifier._counterfeit_threshold,
        denominations,
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backbone_key": args.backbone,
        "evaluation_manifest": args.evaluation_manifest,
        "independent_evaluation": True,
        "sample_count": len(labels),
        "metrics": metrics,
        "calibration_loaded": classifier._calibration_loaded,
    }
    output = OUTPUT_DIR / args.backbone
    (output / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def validate_candidate_checkpoint(backbone_key: str) -> tuple[bool, str]:
    try:
        classifier = HybridForgeryClassifier(
            backbone_key=backbone_key, checkpoint_name="model_candidate.pth"
        )
        asyncio.run(classifier.initialize())
        if not classifier._trained_weights_loaded:
            return False, classifier.get_stats().get("model_status", "unavailable")
        return True, "strict checkpoint load passed"
    except Exception as exc:
        return False, str(exc)


def run_promotion(args) -> dict:
    output = OUTPUT_DIR / args.backbone
    candidate_path = output / "model_candidate.pth"
    metadata_path = output / "candidate_training_metadata.json"
    evaluation_path = output / "evaluation_report.json"
    if not candidate_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Candidate model and metadata are required")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    calibration = metadata.get("calibration", {})
    checkpoint_valid, checkpoint_reason = validate_candidate_checkpoint(args.backbone)
    checks = [
        {
            "name": "strict_checkpoint_load",
            "passed": checkpoint_valid,
            "observed": checkpoint_reason,
            "required": True,
        },
        {
            "name": "no_synthetic_debug_data",
            "passed": not metadata.get("synthetic_debug_enabled", True),
            "observed": metadata.get("synthetic_debug_enabled"),
            "required": False,
        },
        {
            "name": "mean_f1",
            "passed": metadata.get("mean_f1", 0.0) >= args.minimum_mean_f1,
            "observed": metadata.get("mean_f1", 0.0),
            "required": args.minimum_mean_f1,
        },
        {
            "name": "mean_roc_auc",
            "passed": metadata.get("mean_roc_auc", 0.0) >= args.minimum_mean_roc_auc,
            "observed": metadata.get("mean_roc_auc", 0.0),
            "required": args.minimum_mean_roc_auc,
        },
        {
            "name": "mean_pr_auc",
            "passed": metadata.get("mean_pr_auc", 0.0) >= args.minimum_mean_pr_auc,
            "observed": metadata.get("mean_pr_auc", 0.0),
            "required": args.minimum_mean_pr_auc,
        },
        {
            "name": "calibration_policy",
            "passed": bool(calibration.get("policy_satisfied", False)),
            "observed": calibration.get("policy_satisfied"),
            "required": True,
        },
        {
            "name": "counterfeit_false_accept_rate",
            "passed": calibration.get("far", 1.0) <= args.maximum_counterfeit_far,
            "observed": calibration.get("far", 1.0),
            "required": args.maximum_counterfeit_far,
        },
        {
            "name": "genuine_false_reject_rate",
            "passed": calibration.get("frr", 1.0) <= args.max_genuine_frr,
            "observed": calibration.get("frr", 1.0),
            "required": args.max_genuine_frr,
        },
    ]
    if args.require_independent_evaluation:
        checks.append(
            {
                "name": "independent_evaluation",
                "passed": evaluation_path.exists(),
                "observed": str(evaluation_path) if evaluation_path.exists() else "missing",
                "required": True,
            }
        )
    passed = all(check["passed"] for check in checks)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backbone_key": args.backbone,
        "passed": passed,
        "checks": checks,
    }
    (output / "promotion_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    markdown = ["# Vision Model Promotion Report", ""]
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        markdown.append(
            f"- **{status}** `{check['name']}`: observed `{check['observed']}`, required `{check['required']}`"
        )
    (output / "promotion_report.md").write_text("\n".join(markdown), encoding="utf-8")
    if not passed:
        return report

    active_path = output / "model.pth"
    active_metadata = output / "training_metadata.json"
    if active_path.exists():
        shutil.copy2(active_path, output / f"model_backup_{int(time.time())}.pth")
    temporary_model = output / ".model.pth.tmp"
    temporary_metadata = output / ".training_metadata.json.tmp"
    shutil.copy2(candidate_path, temporary_model)
    shutil.copy2(metadata_path, temporary_metadata)
    os.replace(temporary_model, active_path)
    os.replace(temporary_metadata, active_metadata)
    return report


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Research-grade currency-forgery vision training lifecycle"
    )
    parser.add_argument(
        "--mode",
        choices=["cross_validate", "train_final", "evaluate", "promote"],
        default="cross_validate",
    )
    parser.add_argument(
        "--backbone",
        choices=["efficientnet_b0", "convnextv2_tiny_fcmae"],
        default="efficientnet_b0",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CV_EPOCHS)
    parser.add_argument("--final-epochs", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=SUPERVISED_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--loss", choices=["cross_entropy", "focal"], default="cross_entropy")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--embedding-loss-weight", type=float, default=0.05)
    parser.add_argument("--ssl-mode", choices=["none", "simclr"])
    parser.add_argument("--ssl-epochs", type=int, default=DEFAULT_SSL_EPOCHS)
    parser.add_argument("--ssl-temperature", type=float, default=0.07)
    parser.add_argument("--allow-synthetic-debug-data", action="store_true")
    parser.add_argument("--head-warmup-epochs", type=int, default=2)
    parser.add_argument("--scheduler-warmup-epochs", type=int, default=1)
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--max-genuine-frr", type=float, default=0.05)
    parser.add_argument("--maximum-counterfeit-far", type=float, default=0.10)
    parser.add_argument("--uncertainty-margin", type=float, default=0.05)
    parser.add_argument("--minimum-mean-f1", type=float, default=0.85)
    parser.add_argument("--minimum-mean-roc-auc", type=float, default=0.90)
    parser.add_argument("--minimum-mean-pr-auc", type=float, default=0.85)
    parser.add_argument("--require-independent-evaluation", action="store_true")
    parser.add_argument("--evaluation-manifest")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser


def validate_args(args) -> None:
    if args.gradient_accumulation < 1:
        raise ValueError("gradient_accumulation must be at least 1")
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not 0.0 <= args.label_smoothing <= 0.1:
        raise ValueError("label_smoothing must be between 0 and 0.1")
    if not 0.0 <= args.max_genuine_frr <= 1.0:
        raise ValueError("max_genuine_frr must be between 0 and 1")
    if args.ssl_mode is None:
        args.ssl_mode = get_backbone_spec(args.backbone).default_ssl_mode


def train(args=None, **kwargs):
    if args is None:
        args = create_parser().parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.amp and device.type != "cuda":
        print("Warning: AMP requested but CUDA is unavailable; AMP disabled.")
        args.amp = False

    if args.mode == "promote":
        return run_promotion(args)
    bundle = load_dataset(args.evaluation_manifest if args.mode == "evaluate" else None)
    if args.mode == "cross_validate":
        return run_cross_validation(args, bundle, device)
    if args.mode == "train_final":
        return run_final_training(args, bundle, device)
    if args.mode == "evaluate":
        return run_evaluation(args, bundle)
    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    train()