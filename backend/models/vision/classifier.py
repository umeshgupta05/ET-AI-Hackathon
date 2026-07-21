"""
Hybrid Vision Classifier — CNN + Pre-Norm MIL Transformer Architecture.

This module provides the robust inference runtime for document forgery detection, 
featuring missing region masking, explicit identity embeddings, and robust legacy checkpoint compatibility.
"""

import json
import logging
import os
from typing import Optional
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
from models.vision.detector import CURRENCY_REGIONS
from models.vision.backbone_registry import get_backbone_spec
from models.vision.preprocessing import get_region_tensor_transform

logger = logging.getLogger(__name__)

REGION_NAMES = ("full_note", *CURRENCY_REGIONS.keys())


class PreNormMILAggregator(nn.Module):
    """
    Pre-Norm Multiple Instance Learning Transformer Aggregator.
    Takes a bag of region embeddings [B, R, backbone_feature_dim],
    projects them, applies learned region identity embeddings, and runs them
    through a Pre-Norm Transformer.
    """
    def __init__(self, feature_dim: int, num_regions: int, d_model: int = 384, num_heads: int = 6, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_regions <= 20, f"Region count {num_regions} exceeds limits"
        
        self.d_model = d_model
        
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        self.region_embeddings = nn.Parameter(torch.randn(1, num_regions, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cls_pos_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, features: torch.Tensor, missing_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        features: [B, R, feature_dim]
        missing_mask: [B, R] boolean tensor where True means region is missing
        Returns: [B, d_model] normalized CLS representation
        """
        B, R, _ = features.shape
        x = self.input_proj(features)
        
        if missing_mask is not None:
            expanded_mask = missing_mask.unsqueeze(-1).expand(-1, -1, self.d_model)
            missing_tokens = self.missing_token.expand(B, R, -1)
            x = torch.where(expanded_mask, missing_tokens, x)
            
        x = x + self.region_embeddings[:, :R, :]
        cls_tokens = self.cls_token.expand(B, -1, -1) + self.cls_pos_embedding
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = self.transformer(x)
        x = self.norm(x)
        return x[:, 0, :]


class ContrastiveHead(nn.Module):
    """SimCLR-style contrastive projection head for forensic embeddings."""
    def __init__(self, in_features: int, proj_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.GELU(),
            nn.Linear(in_features, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.projection(x)
        return F.normalize(z, dim=1)


class HybridForgeryClassifier:
    """
    Robust Hybrid CNN-Transformer forgery classifier with Pre-Norm MIL Aggregator.
    Supports detailed explainability and cross-region forensic matching.
    """
    def __init__(self):
        self._backbone = None
        self._aggregator = None
        self._contrastive_head = None
        self._classifier_head = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialized = False
        self._trained_weights_loaded = False
        self._backbone_key = os.getenv("VISION_BACKBONE", "efficientnet_b0")
        
        self._transform = None
        self._checkpoint_format_version = None
        self._architecture_version = None
        
        # Calibration state
        self._calibration_loaded = False
        self._temperature = 1.0
        self._counterfeit_threshold = 0.5
        self._uncertainty_margin = 0.1
        self._threshold_source = "safe_default"

    async def initialize(self) -> None:
        """Load the robust hybrid model."""
        if self._initialized:
            return

        logger.info(f"🔍 Initializing Robust Forgery Classifier with backbone {self._backbone_key}...")

        try:
            import timm

            spec = get_backbone_spec(self._backbone_key)
            self._transform = get_region_tensor_transform(spec.timm_name, spec.default_input_size)
            
            trained_dir = Path(__file__).resolve().parent.parent.parent / "data" / "trained_models" / "forgery_classifier"
            trained_path = trained_dir / self._backbone_key / "model.pth"
            metadata_path = trained_dir / self._backbone_key / "training_metadata.json"
            
            # Robust Legacy fallback support for older checkpoints
            legacy_path = trained_dir / "model.pth"
            if not trained_path.exists() and self._backbone_key == "efficientnet_b0" and legacy_path.exists():
                logger.warning(f"Using legacy checkpoint path: {legacy_path}")
                trained_path = legacy_path

            use_pretrained = not trained_path.exists()

            self._backbone = timm.create_model(
                spec.timm_name,
                pretrained=use_pretrained,
                num_classes=0,
                global_pool="",
            )
            self._backbone.eval()
            self._backbone.to(self._device)

            feature_dim = self._backbone.num_features
            d_model = 384
            self._aggregator = PreNormMILAggregator(
                feature_dim=feature_dim,
                num_regions=len(REGION_NAMES),
                d_model=d_model,
                num_heads=6,
                num_layers=3,
                dropout=0.1
            )
            self._aggregator.to(self._device)

            self._contrastive_head = ContrastiveHead(d_model, proj_dim=128)
            self._contrastive_head.to(self._device)

            self._classifier_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 2),
            )
            self._classifier_head.to(self._device)

            # Load checkpoint strictly
            if trained_path.exists():
                state = torch.load(str(trained_path), map_location=self._device, weights_only=True)
                
                # Checkpoint format version check
                self._checkpoint_format_version = state.get("checkpoint_format_version", 1)
                self._architecture_version = state.get("architecture_version", "prenorm_mil_v1")
                
                if "backbone_key" in state and state["backbone_key"] != self._backbone_key:
                    if self._backbone_key != "efficientnet_b0":
                        raise ValueError(f"Checkpoint backbone ({state.get('backbone_key')}) mismatches runtime ({self._backbone_key}). Cross-backbone loading rejected.")

                # Ensure dimensions match
                if "d_model" in state and state["d_model"] != d_model:
                    raise ValueError(f"Checkpoint d_model mismatch.")
                if "region_names" in state and state["region_names"] != list(REGION_NAMES):
                    raise ValueError(f"Checkpoint region_names mismatch.")

                load_errors = False
                if "backbone" in state:
                    self._backbone.load_state_dict(state["backbone"], strict=True)
                else:
                    load_errors = True
                    
                if "aggregator" in state:
                    self._aggregator.load_state_dict(state["aggregator"], strict=True)
                elif "attention_head" in state and self._backbone_key == "efficientnet_b0":
                    logger.warning("Legacy attention_head detected. Partial load only.")
                    load_errors = True
                else:
                    load_errors = True
                    
                if "classifier_head" in state:
                    self._classifier_head.load_state_dict(state["classifier_head"], strict=True)
                else:
                    load_errors = True
                    
                if "contrastive_head" in state:
                    self._contrastive_head.load_state_dict(state["contrastive_head"], strict=True)
                
                if not load_errors:
                    self._trained_weights_loaded = True
                    logger.info(f"✅ Loaded fine-tuned weights successfully from {trained_path}")
                else:
                    logger.warning(f"Incomplete weights loaded; classifier signal degraded.")

                # Load Calibration
                if "calibration" in state:
                    cal = state["calibration"]
                    self._temperature = cal.get("temperature", 1.0)
                    self._counterfeit_threshold = cal.get("counterfeit_threshold", 0.5)
                    self._uncertainty_margin = cal.get("uncertainty_margin", 0.1)
                    self._calibration_loaded = True
                    self._threshold_source = "checkpoint"
                elif metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            meta = json.load(f)
                        if "calibration" in meta:
                            cal = meta["calibration"]
                            self._temperature = cal.get("temperature", 1.0)
                            self._counterfeit_threshold = cal.get("threshold", cal.get("counterfeit_threshold", 0.5))
                            self._uncertainty_margin = cal.get("uncertainty_margin", 0.1)
                            self._calibration_loaded = True
                            self._threshold_source = "metadata"
                    except Exception as e:
                        logger.warning(f"Failed to read calibration metadata: {e}")

            else:
                logger.warning(f"No verified weights found for {self._backbone_key}")

            self._backbone.eval()
            self._aggregator.eval()
            self._contrastive_head.eval()
            self._classifier_head.eval()
            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize robust hybrid classifier: {e}")
            raise

    def _apply_calibration(self, logits: torch.Tensor) -> tuple[float, float, str]:
        """Applies temperature scaling and calibrated thresholds."""
        logits = logits / self._temperature
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        gs, cs = float(probs[0]), float(probs[1])
        
        if cs >= self._counterfeit_threshold + self._uncertainty_margin:
            verdict = "counterfeit"
        elif cs <= self._counterfeit_threshold - self._uncertainty_margin:
            verdict = "genuine"
        else:
            verdict = "uncertain"
            
        return gs, cs, verdict

    @torch.no_grad()
    def classify_region(self, region: np.ndarray) -> dict:
        """Classify a single currency region."""
        if not self._initialized:
            raise RuntimeError("Classifier not initialized")
        if not self._trained_weights_loaded:
            return {
                "genuine_score": 0.5,
                "counterfeit_score": 0.5,
                "verdict": "unavailable",
                "embedding": None,
                "feature_dim": 0,
                "model_available": False,
            }

        if len(region.shape) == 2:
            region = cv2.cvtColor(region, cv2.COLOR_GRAY2RGB)
        elif region.shape[2] == 4:
            region = cv2.cvtColor(region, cv2.COLOR_BGRA2RGB)
        else:
            region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)

        tensor = self._transform(Image.fromarray(region)).unsqueeze(0).to(self._device)
        cnn_features = self._backbone(tensor) 
        cnn_features = cnn_features.mean(dim=(2, 3)).unsqueeze(1)
        
        mask = torch.zeros((1, 1), dtype=torch.bool, device=self._device)
        attended_features = self._aggregator(cnn_features, mask)
        embedding = self._contrastive_head(attended_features).cpu().numpy()[0]
        logits = self._classifier_head(attended_features)
        
        gs, cs, verdict = self._apply_calibration(logits)

        return {
            "genuine_score": round(gs, 4),
            "counterfeit_score": round(cs, 4),
            "verdict": verdict,
            "embedding": embedding,
            "feature_dim": attended_features.shape[-1],
        }

    @torch.no_grad()
    def classify_all_regions(self, regions: dict[str, np.ndarray]) -> dict:
        """
        Robustly classify all extracted regions using the Pre-Norm MIL Aggregator.
        Provides detailed explainability on region importance and masking logic.
        """
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
                "cross_region_aggregation": True,
                "region_importance_method": "single_region_ablation",
                "attention_weights_available": False,
                "model_available": False,
            }

        region_inputs = []
        missing_regions = []
        missing_mask = []
        
        # Determine tensor shape from transform
        spec = get_backbone_spec(self._backbone_key)
        h, w = spec.default_input_size, spec.default_input_size
        dummy_tensor = torch.zeros((3, h, w))

        for name in REGION_NAMES:
            region_img = regions.get(name)
            if region_img is None or region_img.size == 0:
                missing_regions.append(name)
                region_inputs.append(dummy_tensor)
                missing_mask.append(True)
                continue
                
            try:
                if len(region_img.shape) == 2:
                    region_img = cv2.cvtColor(region_img, cv2.COLOR_GRAY2RGB)
                elif region_img.shape[2] == 4:
                    region_img = cv2.cvtColor(region_img, cv2.COLOR_BGRA2RGB)
                else:
                    region_img = cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)

                region_inputs.append(self._transform(Image.fromarray(region_img)))
                missing_mask.append(False)
            except Exception as e:
                logger.warning(f"Failed to extract features for '{name}': {e}")
                missing_regions.append(name)
                region_inputs.append(dummy_tensor)
                missing_mask.append(True)

        batch = torch.stack(region_inputs, dim=0).to(self._device)
        features = self._backbone(batch)
        features = features.mean(dim=(2, 3)).unsqueeze(0)
        mask_tensor = torch.tensor([missing_mask], dtype=torch.bool, device=self._device)

        attended = self._aggregator(features, mask_tensor)
        global_logits = self._classifier_head(attended)
        fused_genuine, fused_counterfeit, overall = self._apply_calibration(global_logits)
        
        region_scores = {}
        suspicious = []
        # Robust explainability: single region isolation scoring
        for i, name in enumerate(REGION_NAMES):
            if missing_mask[i]:
                continue
            
            single_mask = mask_tensor.clone()
            single_mask[0, i] = True # Ablate this specific region
            
            single_attended = self._aggregator(features, single_mask)
            single_logits = self._classifier_head(single_attended)
            
            # Use raw uncalibrated probs for delta to avoid margin distortion
            sp = F.softmax(single_logits, dim=1).cpu().numpy()[0]
            sp_cs = float(sp[1])
            gp = F.softmax(global_logits, dim=1).cpu().numpy()[0]
            gp_cs = float(gp[1])
            
            delta = gp_cs - sp_cs
            
            _, _, single_verdict = self._apply_calibration(single_logits)
            
            region_scores[name] = {
                "importance_delta": round(delta, 4),
                "importance_magnitude": round(abs(delta), 4)
            }
            if delta > 0.05: # Removing it caused counterfeit score to drop significantly
                suspicious.append(name)

        return {
            "region_scores": region_scores,
            "fused_counterfeit_score": round(fused_counterfeit, 4),
            "fused_genuine_score": round(fused_genuine, 4),
            "verdict": overall,
            "suspicious_regions": suspicious,
            "missing_regions": missing_regions,
            "region_order": list(REGION_NAMES),
            "cross_region_aggregation": True,
            "region_importance_method": "single_region_ablation",
            "attention_weights_available": False,
            "model_available": True,
            "backbone_key": self._backbone_key,
            "calibration_loaded": self._calibration_loaded,
            "counterfeit_threshold": self._counterfeit_threshold,
            "uncertainty_margin": self._uncertainty_margin,
            "threshold_source": self._threshold_source
        }

    @torch.no_grad()
    def compute_similarity(self, region1: np.ndarray, region2: np.ndarray) -> float:
        """Compute forensic SimCLR contrastive similarity between two regions."""
        emb1 = self.classify_region(region1)["embedding"]
        emb2 = self.classify_region(region2)["embedding"]
        similarity = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        return round(similarity, 4)

    def get_stats(self) -> dict:
        status = "not_initialized"
        if self._initialized:
            if self._trained_weights_loaded:
                status = "verified" if self._backbone_key == "efficientnet_b0" else "candidate"
            else:
                status = "weights_required"
                
        return {
            "status": status,
            "trained_weights_loaded": self._trained_weights_loaded,
            "architecture": "Pre-Norm MIL Aggregator",
            "backbone_key": self._backbone_key,
            "checkpoint_format_version": self._checkpoint_format_version,
            "architecture_version": self._architecture_version,
            "calibration_loaded": self._calibration_loaded,
            "region_count": len(REGION_NAMES),
            "preprocessing_source": "timm_resolve_model_data_config",
            "device": str(self._device),
        }

_classifier: Optional[HybridForgeryClassifier] = None

def get_forgery_classifier() -> HybridForgeryClassifier:
    global _classifier
    if _classifier is None:
        _classifier = HybridForgeryClassifier()
    return _classifier
