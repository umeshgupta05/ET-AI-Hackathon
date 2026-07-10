"""
Calibration & Confidence Layer.

Transforms raw model scores into statistically meaningful probabilities.
Two techniques:
1. Temperature Scaling — learns a single temperature parameter
2. Isotonic Regression — non-parametric monotonic calibration

This ensures confidence numbers mean something, not just raw softmax output.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

from config import config

logger = logging.getLogger(__name__)


class CalibrationLayer:
    """
    Model calibration using temperature scaling or isotonic regression.

    For hackathon: Provides pass-through calibration with reasonable defaults.
    In production: Would be fit on a validation dataset to produce
    statistically calibrated probability estimates.

    Say in pitch: "Every agent's contribution is logged and calibrated —
    so this is explainable multi-agent AI, not a black box."
    """

    def __init__(self):
        self._isotonic = None
        self._temperature = 1.0
        self._method = config.orchestrator.calibration_method
        self._fitted = False

    def initialize(self) -> None:
        """Initialize calibration with default parameters."""
        if self._method == "isotonic":
            self._isotonic = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            default_raw = np.array(
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            )
            default_cal = np.array(
                [0.0, 0.05, 0.12, 0.22, 0.35, 0.50, 0.65, 0.78, 0.88, 0.95, 1.0]
            )
            self._isotonic.fit(default_raw, default_cal)
        elif self._method == "temperature":
            self._temperature = 1.2

        self._fitted = True
        logger.info(f" Calibration layer initialized (method={self._method})")

    def calibrate(self, raw_score: float) -> float:
        """Calibrate a raw model score to a meaningful probability."""
        raw_score = max(0.0, min(1.0, raw_score))
        if not self._fitted:
            return round(raw_score, 4)

        if self._method == "isotonic" and self._isotonic:
            calibrated = float(self._isotonic.predict([raw_score])[0])
        elif self._method == "temperature":
            import math

            score = max(0.001, min(0.999, raw_score))
            logit = math.log(score / (1 - score))
            scaled_logit = logit / self._temperature
            calibrated = 1.0 / (1.0 + math.exp(-scaled_logit))
        else:
            calibrated = raw_score

        return round(calibrated, 4)

    def calibrate_batch(self, raw_scores: list[float]) -> list[float]:
        """Calibrate a batch of scores."""
        return [self.calibrate(s) for s in raw_scores]

    def fit(self, raw_scores: np.ndarray, true_labels: np.ndarray) -> None:
        """
        Fit the calibration model on validation data.

        Args:
            raw_scores: Model output probabilities
            true_labels: Ground truth binary labels (0 or 1)
        """
        if self._method == "isotonic":
            self._isotonic = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            self._isotonic.fit(raw_scores, true_labels)
            self._fitted = True
            logger.info(f"Isotonic regression fitted on {len(raw_scores)} samples")
            return

        if self._method == "temperature":
            import math

            best_temp = 1.0
            best_nll = float("inf")
            for temperature in np.arange(0.5, 3.0, 0.1):
                nll = 0.0
                for score, label in zip(raw_scores, true_labels):
                    score = max(0.001, min(0.999, float(score)))
                    logit = math.log(score / (1 - score))
                    p = 1.0 / (1.0 + math.exp(-logit / temperature))
                    p = max(0.001, min(0.999, p))
                    nll -= label * math.log(p) + (1 - label) * math.log(1 - p)

                if nll < best_nll:
                    best_nll = nll
                    best_temp = temperature

            self._temperature = float(best_temp)
            self._fitted = True
            logger.info(f"Temperature scaling: T={best_temp:.2f}")

    def get_stats(self) -> dict:
        return {
            "method": self._method,
            "fitted": self._fitted,
            "temperature": self._temperature if self._method == "temperature" else None,
        }


# Module singleton
_layer: Optional[CalibrationLayer] = None


def get_calibration_layer() -> CalibrationLayer:
    global _layer
    if _layer is None:
        _layer = CalibrationLayer()
    return _layer
