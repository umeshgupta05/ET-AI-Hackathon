"""Lazy CLIP zero-shot scoring for currency/document authenticity signals."""

import logging
from typing import Optional

import cv2
import numpy as np

from config import config

logger = logging.getLogger(__name__)


class CLIPVisionScorer:
    """Zero-shot CLIP prompt scorer with graceful fallback."""

    def __init__(self):
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = "cpu"
        self._initialized = False
        self._available = False
        self._error: Optional[str] = None
        self._prompts = {
            "genuine_currency": "a clear genuine Indian currency note with intact security features",
            "counterfeit_currency": "a counterfeit fake currency note with printing defects",
            "tampered_document": "a tampered forged document or manipulated banknote image",
            "poor_quality_scan": "a blurry low quality scan of a currency note",
            "synthetic_print": "a synthetic printed fake note with unnatural texture",
            "invalid_currency": "a novelty fake note, toy money, or invalid imaginary denomination",
        }

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        try:
            import torch
            import open_clip

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model_name = "ViT-B-32"
            pretrained = "openai"
            if "clip-vit-base-patch32" not in config.local_models.clip_model:
                logger.info(f"Using configured CLIP model hint: {config.local_models.clip_model}")
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self._device
            )
            self._tokenizer = open_clip.get_tokenizer(model_name)
            self._model.eval()
            self._available = True
            logger.info(f"CLIP scorer ready ({model_name}/{pretrained}, device={self._device})")
        except Exception as exc:
            self._error = str(exc)
            self._available = False
            logger.warning(f"CLIP scorer unavailable: {exc}")

    def score(self, image_bgr: np.ndarray) -> dict:
        """Return zero-shot prompt probabilities and a counterfeit-oriented risk score."""
        if not self._available:
            return {
                "available": False,
                "risk_score": 0.5,
                "top_label": "unavailable",
                "prompt_scores": {},
                "error": self._error,
            }

        import torch
        from PIL import Image

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        image_tensor = self._preprocess(pil_image).unsqueeze(0).to(self._device)
        labels = list(self._prompts.keys())
        text = self._tokenizer([self._prompts[label] for label in labels]).to(self._device)

        with torch.no_grad():
            image_features = self._model.encode_image(image_tensor)
            text_features = self._model.encode_text(text)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ text_features.T
            probs = logits.softmax(dim=-1).cpu().numpy()[0]

        prompt_scores = {label: round(float(score), 4) for label, score in zip(labels, probs)}
        risk_score = (
            prompt_scores["counterfeit_currency"] * 0.35
            + prompt_scores["invalid_currency"] * 0.35
            + prompt_scores["tampered_document"] * 0.15
            + prompt_scores["synthetic_print"] * 0.10
            + prompt_scores["poor_quality_scan"] * 0.05
        )
        top_label = max(prompt_scores, key=prompt_scores.get)
        return {
            "available": True,
            "risk_score": round(float(risk_score), 4),
            "top_label": top_label,
            "prompt_scores": prompt_scores,
            "model": "open_clip ViT-B-32/openai",
        }

    def get_stats(self) -> dict:
        return {
            "status": "ready" if self._available else "unavailable" if self._initialized else "not_initialized",
            "model": "open_clip ViT-B-32/openai",
            "error": self._error,
        }


_scorer: Optional[CLIPVisionScorer] = None


def get_clip_scorer() -> CLIPVisionScorer:
    global _scorer
    if _scorer is None:
        _scorer = CLIPVisionScorer()
    return _scorer
