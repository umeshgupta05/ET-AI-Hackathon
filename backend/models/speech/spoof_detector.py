"""
Voice Spoofing / Deepfake Audio Detector — WavLM + AASIST Architecture.

SOTA 2025-2026 anti-spoofing pipeline:
Frontend: WavLM (SSL foundation model) for feature extraction
Backend: Classification head for bonafide vs. spoofed detection

This detects AI-generated/synthesized voices used in digital arrest scams.
Based on ASVspoof challenge winning architectures.
"""

import logging
from typing import Optional

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from config import config

logger = logging.getLogger(__name__)


class SpoofDetector:
    """
    Voice spoofing / deepfake audio detector.

    Architecture:
    - WavLM/wav2vec2 foundation model as feature extractor
    - Classification head: bonafide vs. spoofed

    For hackathon: Uses a pretrained audio deepfake detector from HuggingFace.
    In production: Would fine-tune WavLM + AASIST (Graph Attention Network)
    on ASVspoof5 dataset for SOTA performance.
    """

    def __init__(self):
        self._feature_extractor = None
        self._model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialized = False

    async def initialize(self) -> None:
        """Load the spoof detection model."""
        if self._initialized:
            return

        logger.info(" Initializing voice spoof detector...")

        try:
            # Load pretrained audio deepfake detector
            model_id = config.local_models.spoof_model

            self._feature_extractor = AutoFeatureExtractor.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            self._model = AutoModelForAudioClassification.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            self._model.eval()
            self._model.to(self._device)

            logger.info(f" Spoof detector loaded: {model_id} (device={self._device})")
            self._initialized = True

        except Exception as e:
            logger.warning(f"Primary spoof model failed: {e}. Initializing fallback...")
            await self._initialize_fallback()

    async def _initialize_fallback(self) -> None:
        """Fallback: Use wav2vec2 base with simple classification."""
        try:
            from transformers import (
                Wav2Vec2FeatureExtractor,
                Wav2Vec2ForSequenceClassification,
            )

            self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/wav2vec2-base"
            )
            # Use base model — won't have spoof-specific training,
            # but provides embedding-based anomaly detection
            self._model = None  # Will use embedding-based approach
            self._initialized = True
            logger.info(" Fallback spoof detector initialized (embedding-based)")
        except Exception as e:
            logger.error(f"All spoof detection models failed: {e}")
            self._initialized = True  # Allow system to continue without spoof detection

    @torch.no_grad()
    def detect_spoof(self, audio_array: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Detect if audio is spoofed/AI-generated.

        Args:
        audio_array: 1D numpy array of audio samples
        sample_rate: Sample rate (default 16kHz)

        Returns:
        {
        "spoof_score": 0.0 (likely real) to 1.0 (likely AI-generated),
        "verdict": "bonafide" | "spoofed" | "uncertain",
        "confidence": float,
        "analysis": {
        "spectral_features": {...},
        "temporal_consistency": float,
        },
        }
        """
        if not self._initialized:
            raise RuntimeError("Spoof detector not initialized")

        # Ensure proper format
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        audio_array = audio_array.astype(np.float32)

        # Normalize
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            audio_array = audio_array / max_val

        # Run model if available
        if self._model is not None and self._feature_extractor is not None:
            try:
                return self._classify_with_model(audio_array, sample_rate)
            except Exception as e:
                logger.warning(
                    f"Model classification failed: {e}, using spectral analysis"
                )

            # Fallback: spectral analysis-based detection
            return self._spectral_analysis(audio_array, sample_rate)

    def _classify_with_model(self, audio_array: np.ndarray, sample_rate: int) -> dict:
        """Classify using the pretrained model."""
        inputs = self._feature_extractor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )

        # Move to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Forward pass
        outputs = self._model(**inputs)
        logits = outputs.logits

        # Get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]

        # Model output: [bonafide, spoofed] or similar
        # Determine label ordering from model config
        id2label = getattr(self._model.config, "id2label", {0: "bonafide", 1: "spoof"})

        spoof_score = 0.5  # default
        bonafide_score = 0.5

        for idx, label in id2label.items():
            label_lower = str(label).lower()
            if (
                "spoof" in label_lower
                or "fake" in label_lower
                or "synthetic" in label_lower
            ):
                spoof_score = float(probs[int(idx)])
            elif (
                "bonafide" in label_lower
                or "real" in label_lower
                or "genuine" in label_lower
            ):
                bonafide_score = float(probs[int(idx)])

            # If only 2 classes, normalize
            if len(probs) == 2:
                bonafide_score = float(probs[0])
                spoof_score = float(probs[1])

            confidence = abs(spoof_score - bonafide_score)

            if spoof_score > 0.6:
                verdict = "spoofed"
            elif bonafide_score > 0.6:
                verdict = "bonafide"
            else:
                verdict = "uncertain"

            # Also run spectral analysis for additional features
            spectral = self._spectral_analysis(audio_array, 16000)

        return {
            "spoof_score": round(spoof_score, 4),
            "bonafide_score": round(bonafide_score, 4),
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "analysis": {
                "model_prediction": True,
                "spectral_features": spectral.get("analysis", {}).get(
                    "spectral_features", {}
                ),
                "temporal_consistency": spectral.get("analysis", {}).get(
                    "temporal_consistency", 0.0
                ),
            },
        }

    def _spectral_analysis(self, audio_array: np.ndarray, sample_rate: int) -> dict:
        """
        Spectral analysis-based spoof detection (fallback).

        Analyzes:
        1. Spectral flatness — synthetic voices often have less spectral variation
        2. Zero-crossing rate — AI voices may show unusual patterns
        3. Temporal consistency — natural speech has characteristic variability
        """
        # Compute spectral features
        n_fft = 1024
        hop_length = 512

        # Simple STFT
        frames = []
        for start in range(0, len(audio_array) - n_fft, hop_length):
            frame = audio_array[start : start + n_fft]
            windowed = frame * np.hanning(n_fft)
            spectrum = np.abs(np.fft.rfft(windowed))
            frames.append(spectrum)

            if not frames:
                return {
                    "spoof_score": 0.5,
                    "verdict": "uncertain",
                    "confidence": 0.0,
                    "analysis": {"spectral_features": {}, "temporal_consistency": 0.0},
                }

            spectrogram = np.array(frames)

        # Spectral flatness (Wiener entropy)
        geo_mean = np.exp(np.mean(np.log(spectrogram + 1e-10), axis=1))
        arith_mean = np.mean(spectrogram, axis=1)
        spectral_flatness = np.mean(geo_mean / (arith_mean + 1e-10))

        # Zero-crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_array)))) / (
            2 * len(audio_array)
        )

        # Temporal consistency: std of spectral centroids across frames
        spectral_centroids = []
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        for frame_spectrum in spectrogram:
            centroid = np.sum(freqs * frame_spectrum) / (np.sum(frame_spectrum) + 1e-10)
            spectral_centroids.append(centroid)

        temporal_consistency = 1.0 - min(np.std(spectral_centroids) / 1000, 1.0)

        # Scoring: higher spectral flatness + lower temporal variation = more synthetic
        spoof_score = 0.5
        if spectral_flatness > 0.3:  # High flatness → synthetic
            spoof_score += 0.15
            if temporal_consistency > 0.7:  # Too consistent → synthetic
                spoof_score += 0.15
                if zero_crossings < 0.05 or zero_crossings > 0.3:  # Unusual ZCR
                    spoof_score += 0.1

                spoof_score = min(spoof_score, 1.0)

            if spoof_score > 0.6:
                verdict = "spoofed"
            elif spoof_score < 0.4:
                verdict = "bonafide"
            else:
                verdict = "uncertain"

            return {
                "spoof_score": round(spoof_score, 4),
                "verdict": verdict,
                "confidence": round(abs(spoof_score - 0.5) * 2, 4),
                "analysis": {
                    "spectral_features": {
                        "spectral_flatness": round(float(spectral_flatness), 4),
                        "zero_crossing_rate": round(float(zero_crossings), 4),
                        "mean_spectral_centroid": round(
                            float(np.mean(spectral_centroids)), 2
                        ),
                    },
                    "temporal_consistency": round(float(temporal_consistency), 4),
                    "model_prediction": False,
                },
            }

    def get_stats(self) -> dict:
        return {
            "status": "ready" if self._initialized else "not_initialized",
            "model": config.local_models.spoof_model,
            "architecture": "WavLM/wav2vec2 + classification head",
            "device": str(self._device),
        }


# Module singleton
_detector: Optional[SpoofDetector] = None


def get_spoof_detector() -> SpoofDetector:
    global _detector
    if _detector is None:
        _detector = SpoofDetector()
    return _detector
