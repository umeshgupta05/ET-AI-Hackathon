"""
Forensic Analysis Module — ELA + FFT + NPR.

Three forensic signal detection techniques:
1. Error Level Analysis (ELA) — detects edited/composited regions
2. FFT Frequency Analysis — detects printing artifacts (intaglio vs. inkjet)
3. Neighboring Pixel Relationship (NPR) — SOTA 2025 technique for AI-generated detection

These are SUPPORTING signals fed into the fusion layer, not primary classifiers.
All three are genuinely used in forensic document/deepfake detection research.
"""

import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from io import BytesIO

from config import config

logger = logging.getLogger(__name__)


class ForensicAnalyzer:
    """
    Multi-technique forensic analysis for currency note authenticity.

    Each technique detects different forgery artifacts:
    - ELA: Detects digital manipulation (copy-paste, content-aware fill)
    - FFT: Detects printing method differences (genuine intaglio vs. fake inkjet/laser)
    - NPR: Detects AI-generated content through pixel-level dependency analysis
    """

    def __init__(self):
        self._initialized = False

    async def initialize(self) -> None:
        """No heavy initialization needed — all CPU-based analysis."""
        self._initialized = True
        logger.info(" Forensic analyzer initialized (ELA + FFT + NPR)")

    def analyze(self, image: np.ndarray) -> dict:
        """
        Run all forensic analyses on the image.

        Returns:
        {
        "ela": {"anomaly_score": float, "heatmap": np.ndarray},
        "fft": {"print_artifact_score": float, "spectrum": np.ndarray},
        "npr": {"synthetic_score": float, "relationship_map": np.ndarray},
        "fused_forensic_score": float,
        }
        """
        ela_result = self.error_level_analysis(image)
        fft_result = self.frequency_analysis(image)
        npr_result = self.neighboring_pixel_relationship(image)

        # Fuse forensic scores (weighted average)
        fused = (
            ela_result["anomaly_score"] * 0.35
            + fft_result["print_artifact_score"] * 0.30
            + npr_result["synthetic_score"] * 0.35
        )

        return {
            "ela": ela_result,
            "fft": fft_result,
            "npr": npr_result,
            "fused_forensic_score": round(float(fused), 4),
        }

    def error_level_analysis(
        self,
        image: np.ndarray,
        quality: int = 90,
        scale: float = 10.0,
    ) -> dict:
        """
        Error Level Analysis (ELA).

        Technique: Re-compress the image at a known quality level, then
        compute pixel-level differences. Manipulated regions show higher
        error levels because they were compressed at a different quality.

        This is a standard forensic technique used in real DFDC research.
        """
        try:
            # Convert to PIL for JPEG re-compression
            if len(image.shape) == 2:
                pil_img = Image.fromarray(image)
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

            # Re-compress at known quality
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            recompressed = Image.open(buffer)

            # Compute pixel-level difference
            original_arr = np.array(pil_img).astype(np.float32)
            recompressed_arr = np.array(recompressed).astype(np.float32)

            ela_map = np.abs(original_arr - recompressed_arr) * scale

            # Clip to valid range
            ela_map = np.clip(ela_map, 0, 255).astype(np.uint8)

            # Compute anomaly score
            if len(ela_map.shape) == 3:
                ela_gray = cv2.cvtColor(ela_map, cv2.COLOR_RGB2GRAY)
            else:
                ela_gray = ela_map

            # Anomaly = mean of high-error regions (above threshold)
            threshold = np.percentile(ela_gray, 90)
            high_error_mask = ela_gray > threshold
            anomaly_score = (
                float(np.mean(ela_gray[high_error_mask]) / 255.0)
                if high_error_mask.any()
                else 0.0
            )

            return {
                "anomaly_score": round(min(anomaly_score, 1.0), 4),
                "heatmap": ela_map,
                "mean_error": round(float(np.mean(ela_gray)), 2),
                "max_error": round(float(np.max(ela_gray)), 2),
            }

        except Exception as e:
            logger.warning(f"ELA analysis failed: {e}")
            return {
                "anomaly_score": 0.5,
                "heatmap": None,
                "mean_error": 0,
                "max_error": 0,
            }

    def frequency_analysis(self, image: np.ndarray) -> dict:
        """
        FFT-Based Frequency Domain Analysis.

        Technique: Genuine currency printed with intaglio produces specific
        frequency patterns. Inkjet/laser counterfeits show different spectral
        signatures — periodic banding artifacts from printer mechanisms.

        We compute the 2D FFT and analyze the power spectrum distribution.
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Compute 2D DFT
            f_transform = np.fft.fft2(gray.astype(np.float32))
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.log1p(np.abs(f_shift))

            # Normalize magnitude spectrum
            magnitude_norm = (magnitude - magnitude.min()) / (
                magnitude.max() - magnitude.min() + 1e-8
            )

            # Analyze frequency distribution
            h, w = magnitude_norm.shape
            center = (h // 2, w // 2)

            # Radial frequency bins
            y_coords, x_coords = np.ogrid[:h, :w]
            distances = np.sqrt(
                (y_coords - center[0]) ** 2 + (x_coords - center[1]) ** 2
            )

            # Low frequency (center), mid frequency, high frequency
            max_dist = np.sqrt(center[0] ** 2 + center[1] ** 2)
            low_mask = distances < max_dist * 0.2
            mid_mask = (distances >= max_dist * 0.2) & (distances < max_dist * 0.6)
            high_mask = distances >= max_dist * 0.6

            low_energy = float(np.mean(magnitude_norm[low_mask]))
            mid_energy = float(np.mean(magnitude_norm[mid_mask]))
            high_energy = float(np.mean(magnitude_norm[high_mask]))

            # Print artifact detection:
            # Genuine intaglio: strong mid-frequency content (fine texture)
            # Fake inkjet/laser: weaker mid, possible periodic peaks
            # Score: higher = more likely counterfeit (lacking genuine frequency patterns)
            mid_to_high_ratio = mid_energy / (high_energy + 1e-8)

            # Detect periodic artifacts (peaks in frequency domain)
            # These indicate printer banding/screening patterns
            periodic_score = self._detect_periodic_artifacts(magnitude_norm)

            # Combined print artifact score
            print_artifact_score = 0.5  # default neutral
            if mid_to_high_ratio < 2.0:  # Low mid-frequency content → suspect
                print_artifact_score += 0.2
                if periodic_score > 0.3:  # Periodic artifacts detected → suspect
                    print_artifact_score += 0.2

                print_artifact_score = min(print_artifact_score, 1.0)

            # Create visualization spectrum
            spectrum_vis = (magnitude_norm * 255).astype(np.uint8)

            return {
                "print_artifact_score": round(print_artifact_score, 4),
                "spectrum": spectrum_vis,
                "frequency_distribution": {
                    "low_energy": round(low_energy, 4),
                    "mid_energy": round(mid_energy, 4),
                    "high_energy": round(high_energy, 4),
                    "mid_to_high_ratio": round(mid_to_high_ratio, 4),
                },
                "periodic_artifact_score": round(periodic_score, 4),
            }

        except Exception as e:
            logger.warning(f"FFT analysis failed: {e}")
            return {
                "print_artifact_score": 0.5,
                "spectrum": None,
                "frequency_distribution": {},
                "periodic_artifact_score": 0.0,
            }

    def _detect_periodic_artifacts(self, magnitude: np.ndarray) -> float:
        """Detect periodic peaks in frequency domain (printer banding artifacts)."""
        h, w = magnitude.shape
        center = (h // 2, w // 2)

        # Analyze horizontal and vertical lines through center
        h_line = magnitude[center[0], :]
        v_line = magnitude[:, center[1]]

        # Look for repeated peaks (indicates periodic printing artifacts)
        def count_peaks(signal: np.ndarray) -> int:
            if len(signal) < 3:
                return 0
            # Simple peak detection
            peaks = 0
            mean_val = np.mean(signal)
            threshold = mean_val + 2 * np.std(signal)
            for i in range(1, len(signal) - 1):
                if (
                    signal[i] > signal[i - 1]
                    and signal[i] > signal[i + 1]
                    and signal[i] > threshold
                ):
                    peaks += 1
            return peaks

        h_peaks = count_peaks(h_line)
        v_peaks = count_peaks(v_line)

        # More peaks → more likely periodic artifacts from a printer
        total_peaks = h_peaks + v_peaks
        normalized_score = min(total_peaks / 20.0, 1.0)

        return normalized_score

    def neighboring_pixel_relationship(self, image: np.ndarray) -> dict:
        """
        Neighboring Pixel Relationship (NPR) Analysis.

        SOTA 2025 technique: Models pixel-level dependencies to detect
        artifacts from AI-generated/printed images. Real photos and genuine
        prints have characteristic NPR patterns that are disrupted in
        AI-generated or digitally manipulated content.

        Based on: "Rethinking the Up-Sampling Operations in CNN-based
        Generative Network for Generalizable Deepfake Detection" (CVPR 2024).
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                gray = image.astype(np.float32)

            h, w = gray.shape

            # Compute NPR in 4 directions
            # NPR = pixel[i,j] - pixel[i+d, j+d] for various d
            npr_maps = []
            directions = [
                (0, 1),
                (1, 0),
                (1, 1),
                (1, -1),
            ]  # right, down, diagonal, anti-diagonal

            for dy, dx in directions:
                shifted = np.roll(np.roll(gray, -dy, axis=0), -dx, axis=1)
                npr_map = gray - shifted

            # Trim edges affected by rolling
            if dy > 0:
                npr_map = npr_map[:-1, :]
                if dx > 0:
                    npr_map = npr_map[:, :-1]
                elif dx < 0:
                    npr_map = npr_map[:, 1:]

                npr_maps.append(npr_map)

            # Analyze NPR statistics
            # Genuine/real content: NPR follows a Laplacian-like distribution
            # AI-generated content: NPR shows checkerboard artifacts, unusual kurtosis
            npr_combined = np.concatenate([m.flatten() for m in npr_maps])

            mean_npr = float(np.mean(np.abs(npr_combined)))
            std_npr = float(np.std(npr_combined))

            # Kurtosis: genuine images have higher kurtosis (sharper NPR distribution)
            n = len(npr_combined)
            if n > 0 and std_npr > 0:
                kurtosis = (
                    float(
                        np.mean(((npr_combined - np.mean(npr_combined)) / std_npr) ** 4)
                    )
                    - 3
                )
            else:
                kurtosis = 0.0

            # Checkerboard artifact detection (common in upsampled/generated images)
            checkerboard_score = self._detect_checkerboard(gray)

            # Synthetic score: lower kurtosis + checkerboard artifacts = more synthetic
            synthetic_score = 0.5  # baseline
            if kurtosis < 3.0:  # Low kurtosis = smoother NPR = possibly synthetic
                synthetic_score += 0.15
                if checkerboard_score > 0.3:
                    synthetic_score += 0.2
                    if (
                        mean_npr < 5.0
                    ):  # Very low NPR differences = suspicious smoothness
                        synthetic_score += 0.1

                    synthetic_score = min(synthetic_score, 1.0)

                # Create NPR visualization
                if len(npr_maps) > 0:
                    vis = np.abs(npr_maps[0])
                    vis = (vis / (vis.max() + 1e-8) * 255).astype(np.uint8)
                else:
                    vis = None

                return {
                    "synthetic_score": round(synthetic_score, 4),
                    "relationship_map": vis,
                    "statistics": {
                        "mean_npr": round(mean_npr, 4),
                        "std_npr": round(std_npr, 4),
                        "kurtosis": round(kurtosis, 4),
                        "checkerboard_score": round(checkerboard_score, 4),
                    },
                }

        except Exception as e:
            logger.warning(f"NPR analysis failed: {e}")
            return {
                "synthetic_score": 0.5,
                "relationship_map": None,
                "statistics": {},
            }

    def _detect_checkerboard(self, gray: np.ndarray) -> float:
        """Detect checkerboard artifacts common in AI-generated images."""
        # Compute Laplacian (second derivative)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)

        # Checkerboard: alternating positive/negative in Laplacian
        # Compute autocorrelation at step 2 in both directions
        h, w = laplacian.shape
        if h < 4 or w < 4:
            return 0.0

        # Check for 2-pixel periodicity
        shift_2h = np.roll(laplacian, 2, axis=0)[2:, :]
        shift_2w = np.roll(laplacian, 2, axis=1)[:, 2:]
        original_h = laplacian[2:, :]
        original_w = laplacian[:, 2:]

        corr_h = float(np.corrcoef(original_h.flatten(), shift_2h.flatten())[0, 1])
        corr_w = float(np.corrcoef(original_w.flatten(), shift_2w.flatten())[0, 1])

        # High positive correlation at step 2 = checkerboard pattern
        checkerboard = max(corr_h, corr_w, 0.0)
        return round(checkerboard, 4)


# Module singleton
_analyzer: Optional[ForensicAnalyzer] = None


def get_forensic_analyzer() -> ForensicAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = ForensicAnalyzer()
    return _analyzer
