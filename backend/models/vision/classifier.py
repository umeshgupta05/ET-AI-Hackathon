"""Configurable CNN + Pre-Norm MIL forgery-classification runtime."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vision.backbone_registry import get_backbone_spec
from models.vision.detector import CURRENCY_REGIONS
from models.vision.preprocessing import get_region_tensor_transform

logger = logging.getLogger(__name__)

REGION_NAMES = ("full_note", *CURRENCY_REGIONS.keys())
CHECKPOINT_FORMAT_VERSION = 2
ARCHITECTURE_VERSION = "prenorm_mil_v2"
D_MODEL = 384
NUM_HEADS = 6
NUM_LAYERS = 3
NUM_CLASSES = 2


def _pool_backbone_output(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 4:
        return features.mean(dim=(2, 3))
    if features.ndim == 2:
        return features
    raise ValueError(f"Unsupported backbone output shape: {tuple(features.shape)}")


class PreNormMILAggregator(nn.Module):
    """Aggregate a fixed semantic bag of currency-region embeddings."""

    def __init__(
        self,
        feature_dim: int,
        num_regions: int,
        d_model: int = D_MODEL,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if num_regions < 1:
            raise ValueError("num_regions must be positive")

        self.feature_dim = feature_dim
        self.num_regions = num_regions
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.region_embeddings = nn.Parameter(torch.randn(1, num_regions, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cls_pos_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        features: torch.Tensor,
        missing_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [B, R, feature_dim]")
        batch_size, region_count, feature_dim = features.shape
        if feature_dim != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, received {feature_dim}"
            )
        if region_count > self.num_regions:
            raise ValueError(
                f"Received {region_count} regions, configured maximum is {self.num_regions}"
            )
        if missing_mask is not None and missing_mask.shape != (batch_size, region_count):
            raise ValueError(
                f"missing_mask must be {(batch_size, region_count)}, got {tuple(missing_mask.shape)}"
            )

        x = self.input_proj(features)
        if missing_mask is not None:
            missing = self.missing_token.expand(batch_size, region_count, -1)
            x = torch.where(missing_mask.unsqueeze(-1), missing, x)

        # Semantic identity is retained even when a region is represented by a missing token.
        x = x + self.region_embeddings[:, :region_count, :]
        cls = self.cls_token.expand(batch_size, -1, -1) + self.cls_pos_embedding
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        return self.norm(x)[:, 0, :]


class ContrastiveHead(nn.Module):
    def __init__(self, in_features: int, projection_dim: int = 128) -> None:
        super().__init__()
        self.projection_dim = projection_dim
        self.projection = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.GELU(),
            nn.Linear(in_features, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(x), dim=1)


def build_classifier_head(d_model: int = D_MODEL) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, 256),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(256, NUM_CLASSES),
    )


class HybridForgeryClassifier:
    """Inference runtime with strict checkpoints and calibrated decisions."""

    def __init__(
        self,
        backbone_key: Optional[str] = None,
        checkpoint_name: str = "model.pth",
    ) -> None:
        self._backbone = None
        self._aggregator = None
        self._contrastive_head = None
        self._classifier_head = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialized = False
        self._trained_weights_loaded = False
        self._backbone_key = backbone_key or os.getenv("VISION_BACKBONE", "efficientnet_b0")
        self._checkpoint_name = checkpoint_name
        self._checkpoint_path: Optional[Path] = None
        self._transform = None

        self._checkpoint_format_version = None
        self._architecture_version = None
        self._model_status = "not_initialized"
        self._promotion_status = "unknown"

        self._calibration_loaded = False
        self._temperature = 1.0
        self._counterfeit_threshold = 0.5
        self._uncertainty_margin = 0.05
        self._threshold_source = "safe_default"

    async def initialize(self) -> None:
        if self._initialized:
            return

        import timm

        spec = get_backbone_spec(self._backbone_key)
        trained_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "trained_models"
            / "forgery_classifier"
            / self._backbone_key
        )
        checkpoint_path = trained_dir / self._checkpoint_name
        metadata_path = trained_dir / (
            "training_metadata.json"
            if self._checkpoint_name == "model.pth"
            else "candidate_training_metadata.json"
        )

        legacy_path = trained_dir.parent / "model.pth"
        if (
            self._checkpoint_name == "model.pth"
            and not checkpoint_path.exists()
            and self._backbone_key == "efficientnet_b0"
            and legacy_path.exists()
        ):
            checkpoint_path = legacy_path
            logger.warning("Legacy vision checkpoint found; strict MIL compatibility is required")

        self._checkpoint_path = checkpoint_path
        use_pretrained = not checkpoint_path.exists()
        self._backbone = timm.create_model(
            spec.timm_name,
            pretrained=use_pretrained,
            num_classes=0,
            global_pool="",
        ).to(self._device)
        feature_dim = int(self._backbone.num_features)
        self._aggregator = PreNormMILAggregator(
            feature_dim=feature_dim,
            num_regions=len(REGION_NAMES),
        ).to(self._device)
        self._contrastive_head = ContrastiveHead(D_MODEL).to(self._device)
        self._classifier_head = build_classifier_head().to(self._device)
        self._transform = get_region_tensor_transform(spec.timm_name, spec.default_input_size)

        if checkpoint_path.exists():
            try:
                state = torch.load(
                    str(checkpoint_path), map_location=self._device, weights_only=True
                )
                self._validate_checkpoint(state, spec.timm_name, feature_dim)
                self._checkpoint_format_version = state["checkpoint_format_version"]
                self._architecture_version = state["architecture_version"]
                self._backbone.load_state_dict(state["backbone"], strict=True)
                self._aggregator.load_state_dict(state["aggregator"], strict=True)
                self._classifier_head.load_state_dict(state["classifier_head"], strict=True)
                self._contrastive_head.load_state_dict(state["contrastive_head"], strict=True)
                self._load_calibration(state, metadata_path)
                self._trained_weights_loaded = True
                self._model_status = (
                    "verified" if self._checkpoint_name == "model.pth" else "candidate"
                )
                logger.info("Loaded verified vision checkpoint from %s", checkpoint_path)
            except Exception as exc:
                self._model_status = "incompatible_checkpoint"
                logger.error("Vision checkpoint rejected: %s", exc)
        else:
            self._model_status = "weights_required"
            logger.warning("No trained checkpoint found for %s", self._backbone_key)

        for module in (
            self._backbone,
            self._aggregator,
            self._contrastive_head,
            self._classifier_head,
        ):
            module.eval()
        self._initialized = True

    def _validate_checkpoint(self, state: dict, timm_name: str, feature_dim: int) -> None:
        required = {
            "checkpoint_format_version",
            "architecture_version",
            "backbone_key",
            "timm_name",
            "feature_dim",
            "d_model",
            "num_heads",
            "num_layers",
            "num_regions",
            "region_names",
            "num_classes",
            "backbone",
            "aggregator",
            "classifier_head",
            "contrastive_head",
            "calibration",
        }
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(f"Checkpoint is incomplete; missing keys: {missing}")

        expected = {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "architecture_version": ARCHITECTURE_VERSION,
            "backbone_key": self._backbone_key,
            "timm_name": timm_name,
            "feature_dim": feature_dim,
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "num_regions": len(REGION_NAMES),
            "region_names": list(REGION_NAMES),
            "num_classes": NUM_CLASSES,
        }
        mismatches = {
            key: (expected_value, state.get(key))
            for key, expected_value in expected.items()
            if state.get(key) != expected_value
        }
        if mismatches:
            raise ValueError(f"Checkpoint architecture mismatch: {mismatches}")

    def _load_calibration(self, state: dict, metadata_path: Path) -> None:
        calibration = dict(state.get("calibration") or {})
        if not calibration and metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                calibration = dict(json.load(handle).get("calibration") or {})
                self._threshold_source = "metadata"
        else:
            self._threshold_source = "checkpoint"

        temperature = float(calibration.get("temperature", 1.0))
        threshold = float(
            calibration.get("counterfeit_threshold", calibration.get("threshold", 0.5))
        )
        margin = float(calibration.get("uncertainty_margin", 0.05))
        if temperature <= 0:
            raise ValueError("Calibration temperature must be positive")
        if not 0.0 < threshold < 1.0:
            raise ValueError("Counterfeit threshold must be between 0 and 1")
        if not 0.0 <= margin < 0.5:
            raise ValueError("Uncertainty margin must be in [0, 0.5)")

        self._temperature = temperature
        self._counterfeit_threshold = threshold
        self._uncertainty_margin = margin
        self._calibration_loaded = True

    def _probabilities(self, logits: torch.Tensor) -> tuple[float, float]:
        probs = F.softmax(logits / self._temperature, dim=1).detach().cpu().numpy()[0]
        return float(probs[0]), float(probs[1])

    def _decision(self, counterfeit_probability: float) -> str:
        upper = self._counterfeit_threshold + self._uncertainty_margin
        lower = self._counterfeit_threshold - self._uncertainty_margin
        if counterfeit_probability >= upper:
            return "counterfeit"
        if counterfeit_probability <= lower:
            return "genuine"
        return "uncertain"

    def _unavailable_region_result(self) -> dict:
        return {
            "genuine_score": 0.5,
            "counterfeit_score": 0.5,
            "verdict": "unavailable",
            "embedding": None,
            "feature_dim": 0,
            "model_available": False,
            "backbone_key": self._backbone_key,
        }

    @torch.no_grad()
    def classify_region(self, region: np.ndarray) -> dict:
        if not self._initialized:
            raise RuntimeError("Classifier not initialized")
        if not self._trained_weights_loaded:
            return self._unavailable_region_result()

        rgb = self._to_rgb(region)
        tensor = self._transform(Image.fromarray(rgb)).unsqueeze(0).to(self._device)
        pooled = _pool_backbone_output(self._backbone(tensor)).unsqueeze(1)
        missing_mask = torch.zeros((1, 1), dtype=torch.bool, device=self._device)
        attended = self._aggregator(pooled, missing_mask)
        logits = self._classifier_head(attended)
        genuine, counterfeit = self._probabilities(logits)
        embedding = self._contrastive_head(attended).cpu().numpy()[0]
        return {
            "genuine_score": round(genuine, 4),
            "counterfeit_score": round(counterfeit, 4),
            "verdict": self._decision(counterfeit),
            "embedding": embedding,
            "feature_dim": int(attended.shape[-1]),
            "model_available": True,
            "backbone_key": self._backbone_key,
            "calibration_loaded": self._calibration_loaded,
        }

    @torch.no_grad()
    def classify_all_regions(self, regions: dict[str, np.ndarray]) -> dict:
        if not self._initialized:
            raise RuntimeError("Classifier not initialized")
        if not self._trained_weights_loaded:
            return {
                "region_scores": {},
                "fused_counterfeit_score": 0.5,
                "fused_genuine_score": 0.5,
                "verdict": "unavailable",
                "suspicious_regions": [],
                "missing_regions": [],
                "cross_region_attention": False,
                "cross_region_aggregation": True,
                "region_importance_method": "single_region_ablation",
                "attention_weights_available": False,
                "model_available": False,
                "backbone_key": self._backbone_key,
            }

        spec = get_backbone_spec(self._backbone_key)
        blank = torch.zeros((3, spec.default_input_size, spec.default_input_size))
        inputs: list[torch.Tensor] = []
        missing: list[bool] = []
        missing_names: list[str] = []

        for name in REGION_NAMES:
            image = regions.get(name)
            if image is None or image.size == 0:
                inputs.append(blank)
                missing.append(True)
                missing_names.append(name)
                continue
            try:
                inputs.append(self._transform(Image.fromarray(self._to_rgb(image))))
                missing.append(False)
            except Exception as exc:
                logger.warning("Region %s preprocessing failed: %s", name, exc)
                inputs.append(blank)
                missing.append(True)
                missing_names.append(name)

        batch = torch.stack(inputs).to(self._device)
        pooled = _pool_backbone_output(self._backbone(batch)).unsqueeze(0)
        mask = torch.tensor([missing], dtype=torch.bool, device=self._device)
        attended = self._aggregator(pooled, mask)
        logits = self._classifier_head(attended)
        genuine, counterfeit = self._probabilities(logits)
        overall = self._decision(counterfeit)

        region_scores: dict[str, dict] = {}
        suspicious: list[str] = []
        for index, name in enumerate(REGION_NAMES):
            if missing[index]:
                continue
            ablated_mask = mask.clone()
            ablated_mask[0, index] = True
            ablated = self._aggregator(pooled, ablated_mask)
            _, ablated_counterfeit = self._probabilities(self._classifier_head(ablated))
            delta = counterfeit - ablated_counterfeit
            region_scores[name] = {
                "importance_delta": round(delta, 4),
                "importance_magnitude": round(abs(delta), 4),
            }
            if delta >= 0.05:
                suspicious.append(name)

        return {
            "region_scores": region_scores,
            "fused_counterfeit_score": round(counterfeit, 4),
            "fused_genuine_score": round(genuine, 4),
            "verdict": overall,
            "suspicious_regions": suspicious,
            "missing_regions": missing_names,
            "region_order": list(REGION_NAMES),
            "cross_region_attention": False,
            "cross_region_aggregation": True,
            "region_importance_method": "single_region_ablation",
            "attention_weights_available": False,
            "model_available": True,
            "backbone_key": self._backbone_key,
            "calibration_loaded": self._calibration_loaded,
            "counterfeit_threshold": self._counterfeit_threshold,
            "uncertainty_margin": self._uncertainty_margin,
            "threshold_source": self._threshold_source,
        }

    @staticmethod
    def _to_rgb(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @torch.no_grad()
    def compute_similarity(self, region1: np.ndarray, region2: np.ndarray) -> float:
        first = self.classify_region(region1).get("embedding")
        second = self.classify_region(region2).get("embedding")
        if first is None or second is None:
            raise RuntimeError("Similarity unavailable because trained embeddings are not loaded")
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator == 0.0:
            return 0.0
        return round(float(np.dot(first, second) / denominator), 4)

    def get_stats(self) -> dict:
        status = self._model_status if self._initialized else "not_initialized"
        return {
            "status": status,
            "model_status": status,
            "trained_weights_loaded": self._trained_weights_loaded,
            "architecture": "Pre-Norm MIL Aggregator",
            "backbone_key": self._backbone_key,
            "checkpoint_path": str(self._checkpoint_path) if self._checkpoint_path else None,
            "checkpoint_format_version": self._checkpoint_format_version,
            "architecture_version": self._architecture_version,
            "calibration_loaded": self._calibration_loaded,
            "promotion_status": self._promotion_status,
            "region_count": len(REGION_NAMES),
            "preprocessing_source": "timm_pretrained_cfg",
            "device": str(self._device),
        }


_classifier: Optional[HybridForgeryClassifier] = None


def get_forgery_classifier() -> HybridForgeryClassifier:
    global _classifier
    if _classifier is None:
        _classifier = HybridForgeryClassifier()
    return _classifier