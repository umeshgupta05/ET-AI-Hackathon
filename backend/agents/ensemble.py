"""XGBoost ensemble fusion helper with weighted-fusion fallback."""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


FEATURE_NAMES = [
    "vision_score",
    "vision_forensic_score",
    "vision_clip_score",
    "speech_spoof_score",
    "nlp_score",
    "graph_score",
    "has_vision",
    "has_speech",
    "has_nlp",
    "has_graph",
    "modality_image",
    "modality_audio",
    "modality_text",
]


class XGBoostFusion:
    """Loads an optional XGBoost meta-learner and exposes a safe predict API."""

    def __init__(self):
        self._model = None
        self._available = False
        self._error: Optional[str] = None
        self._model_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "trained_models"
            / "xgboost_fusion"
            / "model.json"
        )
        self._metadata_path = self._model_path.parent / "training_metadata.json"

    def initialize(self) -> None:
        if self._available:
            return
        try:
            if not self._model_path.exists():
                self._error = "model_not_found"
                return
            from xgboost import XGBClassifier

            model = XGBClassifier()
            model.load_model(str(self._model_path))
            self._model = model
            self._available = True
            logger.info(f"XGBoost fusion model loaded from {self._model_path}")
        except Exception as exc:
            self._error = str(exc)
            self._available = False
            logger.warning(f"XGBoost fusion unavailable: {exc}")

    def extract_features(self, state: dict) -> dict[str, float]:
        vision = state.get("vision_result") or {}
        speech = state.get("speech_result") or {}
        nlp = state.get("nlp_result") or {}
        graph = state.get("graph_result") or {}
        modality = state.get("modality", "")
        forensics = vision.get("forensics") or {}
        clip = vision.get("clip") or {}
        spoof = speech.get("spoof_detection") or {}

        features = {
            "vision_score": float(vision.get("model_confidence", 0.5) or 0.5),
            "vision_forensic_score": float(forensics.get("fused_forensic_score", 0.5) or 0.5),
            "vision_clip_score": float(clip.get("risk_score", 0.5) or 0.5),
            "speech_spoof_score": float(spoof.get("spoof_score", 0.5) or 0.5),
            "nlp_score": float(nlp.get("fused_confidence", 0.5) or 0.5),
            "graph_score": float(graph.get("network_risk_score", 0.0) or 0.0),
            "has_vision": 1.0 if vision else 0.0,
            "has_speech": 1.0 if speech else 0.0,
            "has_nlp": 1.0 if nlp else 0.0,
            "has_graph": 1.0 if graph else 0.0,
            "modality_image": 1.0 if "image" in modality or modality == "multimodal" else 0.0,
            "modality_audio": 1.0 if "audio" in modality or modality == "multimodal" else 0.0,
            "modality_text": 1.0 if "text" in modality or modality == "multimodal" else 0.0,
        }
        return {name: round(max(0.0, min(1.0, value)), 4) for name, value in features.items()}

    def predict(self, state: dict, fallback_score: float) -> dict:
        self.initialize()
        features = self.extract_features(state)
        if not self._available or self._model is None:
            return {
                "score": round(float(fallback_score), 4),
                "method": "weighted_fallback",
                "features": features,
                "model_available": False,
                "error": self._error,
            }

        vector = np.array([[features[name] for name in FEATURE_NAMES]], dtype=np.float32)
        probability = float(self._model.predict_proba(vector)[0][1])
        # Guard against an overconfident tiny meta-model by blending with the base score.
        blended = probability * 0.70 + float(fallback_score) * 0.30
        return {
            "score": round(blended, 4),
            "raw_xgboost_score": round(probability, 4),
            "method": "xgboost_meta_learner",
            "features": features,
            "model_available": True,
        }

    def get_stats(self) -> dict:
        metadata = None
        if self._metadata_path.exists():
            try:
                metadata = json.loads(self._metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = None
        return {
            "status": "ready" if self._available else "fallback",
            "model_path": str(self._model_path),
            "feature_names": FEATURE_NAMES,
            "error": self._error,
            "metadata": metadata,
        }


_fusion: Optional[XGBoostFusion] = None


def get_xgboost_fusion() -> XGBoostFusion:
    global _fusion
    if _fusion is None:
        _fusion = XGBoostFusion()
    return _fusion
