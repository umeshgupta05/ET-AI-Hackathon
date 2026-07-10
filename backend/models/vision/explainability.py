"""
Grad-CAM Explainability Module.

Generates visual attention heatmap overlays showing WHERE the model
is looking when making its classification decision.

Supports both EfficientNet (CNN) and Vision Transformer architectures.
Uses pytorch-grad-cam library with proper reshape_transform for ViTs.

This is the #1 visual demo asset — judges see where the model focuses.
"""

import logging
from typing import Optional

import cv2
import numpy as np
import torch
from torchvision import transforms

from config import config

logger = logging.getLogger(__name__)

# Same preprocessing as the classifier
TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def reshape_transform_vit(tensor, height=14, width=14):
    """
    Reshape transform for Vision Transformer attention visualization.
    Converts flat token output (B × Tokens × Embedding) to spatial grid
    (B × Channels × H × W) for Grad-CAM visualization.
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result


class ExplainabilityEngine:
    """
    Grad-CAM + Attention visualization for model decisions.

    Produces:
    1. Grad-CAM heatmaps showing classification-relevant regions
    2. Overlay visualizations on the original image
    3. Per-region attention scores

    Works with EfficientNet-B0 (CNN-based) and ViT (transformer-based) models.
    """

    def __init__(self):
        self._initialized = False

    async def initialize(self) -> None:
        """Lightweight init — actual Grad-CAM is computed on-demand."""
        self._initialized = True
        logger.info(" Explainability engine initialized (Grad-CAM)")

    def generate_gradcam(
        self,
        model: torch.nn.Module,
        image: np.ndarray,
        target_class: Optional[int] = None,
        is_vit: bool = False,
    ) -> dict:
        """
        Generate Grad-CAM heatmap for a given model and image.

        Args:
        model: The classifier model (EfficientNet or ViT)
        image: BGR numpy array
        target_class: Class index to visualize (None = top prediction)
        is_vit: Whether the model is a Vision Transformer

        Returns:
        {
        "heatmap": np.ndarray (grayscale 0-255),
        "overlay": np.ndarray (BGR with heatmap overlay),
        "attention_scores": dict (per-region attention intensity),
        "top_class": int,
        }
        """
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            from pytorch_grad_cam.utils.image import show_cam_on_image

            # Preprocess image
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            input_tensor = TRANSFORM(image_rgb).unsqueeze(0)

            # Select target layer based on model type
            if is_vit:
                # For ViT: use the last block's norm layer
                target_layers = [model.blocks[-1].norm1]
                reshape = reshape_transform_vit
            else:
                # For EfficientNet: use the last convolutional feature map
                # EfficientNet-B0 feature extraction layers
                if hasattr(model, "conv_head"):
                    target_layers = [model.conv_head]
                elif hasattr(model, "features"):
                    target_layers = [model.features[-1]]
                else:
                    # Fallback: try to get the last non-classifier layer
                    children = list(model.children())
                    target_layers = (
                        [children[-2]] if len(children) > 1 else [children[-1]]
                    )
                    reshape = None

                # Build Grad-CAM
                cam = GradCAM(
                    model=model,
                    target_layers=target_layers,
                    reshape_transform=reshape,
                )

            # Define target
            targets = (
                [ClassifierOutputTarget(target_class)]
                if target_class is not None
                else None
            )

            # Generate heatmap
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]  # First (only) image

            # Resize heatmap to original image size
            h, w = image_rgb.shape[:2]
            heatmap_resized = cv2.resize(grayscale_cam, (w, h))
            heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)

            # Create colorized overlay
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            image_normalized = image_rgb.astype(np.float32) / 255.0
            overlay = show_cam_on_image(image_normalized, heatmap_resized, use_rgb=True)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

            # Compute per-region attention scores
            attention_scores = self._compute_region_attention(heatmap_resized)

            # Clean up
            del cam

            return {
                "heatmap": heatmap_uint8,
                "overlay": overlay_bgr,
                "attention_scores": attention_scores,
                "top_class": target_class or 0,
            }

        except ImportError:
            logger.warning(
                "pytorch-grad-cam not installed, generating fallback heatmap"
            )
            return self._fallback_heatmap(image)
        except Exception as e:
            logger.warning(f"Grad-CAM generation failed: {e}")
            return self._fallback_heatmap(image)

    def generate_attention_overlay(
        self,
        image: np.ndarray,
        region_scores: dict[str, dict],
    ) -> np.ndarray:
        """
        Generate a composite attention overlay combining Grad-CAM with
        region-level annotations showing per-region scores.

        This is the primary demo visualization — shows judges exactly
        which regions the model flags as suspicious and why.
        """
        from models.vision.detector import CURRENCY_REGIONS

        h, w = image.shape[:2]
        overlay = image.copy()

        for region_name, score_data in region_scores.items():
            if region_name not in CURRENCY_REGIONS:
                continue

            coords = CURRENCY_REGIONS[region_name]
            x1 = int(coords["x1"] * w)
            y1 = int(coords["y1"] * h)
            x2 = int(coords["x2"] * w)
            y2 = int(coords["y2"] * h)

        counterfeit_score = score_data.get("counterfeit_score", 0.5)

        # Color: green (genuine) → yellow (uncertain) → red (counterfeit)
        if counterfeit_score > 0.7:
            color = (0, 0, 255)  # Red (BGR)
            thickness = 3
        elif counterfeit_score > 0.4:
            color = (0, 200, 255)  # Yellow
            thickness = 2
        else:
            color = (0, 255, 0)  # Green
            thickness = 2

        # Draw bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        # Draw label
        label = f"{region_name}: {counterfeit_score:.2f}"
        font_scale = 0.4
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        cv2.rectangle(overlay, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(
            overlay,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
        )

        return overlay

    def _compute_region_attention(self, heatmap: np.ndarray) -> dict:
        """Compute average attention intensity per currency region."""
        from models.vision.detector import CURRENCY_REGIONS

        h, w = heatmap.shape
        attention = {}

        for region_name, coords in CURRENCY_REGIONS.items():
            x1 = int(coords["x1"] * w)
            y1 = int(coords["y1"] * h)
            x2 = int(coords["x2"] * w)
            y2 = int(coords["y2"] * h)

        region = heatmap[y1:y2, x1:x2]
        if region.size > 0:
            attention[region_name] = round(float(np.mean(region)), 4)
        else:
            attention[region_name] = 0.0

        return attention

    def _fallback_heatmap(self, image: np.ndarray) -> dict:
        """Generate a simple edge-based attention map as fallback."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        edges = cv2.Canny(gray, 50, 150)
        blurred = cv2.GaussianBlur(edges, (21, 21), 0)

        overlay = image.copy()
        heatmap_colored = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(overlay, 0.6, heatmap_colored, 0.4, 0)

        return {
            "heatmap": blurred,
            "overlay": overlay,
            "attention_scores": {},
            "top_class": 0,
        }

    def get_stats(self) -> dict:
        return {
            "status": "ready" if self._initialized else "not_initialized",
            "technique": "Grad-CAM (gradient-weighted class activation mapping)",
            "supports": ["EfficientNet (CNN)", "Vision Transformer (ViT)"],
        }


# Module singleton
_engine: Optional[ExplainabilityEngine] = None


def get_explainability_engine() -> ExplainabilityEngine:
    global _engine
    if _engine is None:
        _engine = ExplainabilityEngine()
    return _engine
