"""Capture validation and explainable Indian currency security-feature checks."""

from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np


SUPPORTED_DENOMINATIONS = ("10", "20", "50", "100", "200", "500", "2000")


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _clip_currency_probability(clip_result: dict[str, Any]) -> float | None:
    scores = clip_result.get("prompt_scores") or clip_result.get("scores") or {}
    if not scores:
        return None
    currency = sum(float(scores.get(key, 0)) for key in (
        "genuine_currency", "counterfeit_currency", "synthetic_print", "poor_quality_scan"
    ))
    return max(0.0, min(1.0, currency))


def validate_currency_candidate(
    image: np.ndarray,
    note_crop: np.ndarray,
    detection: dict[str, Any],
    clip_result: dict[str, Any],
) -> dict[str, Any]:
    h, w = note_crop.shape[:2]
    long_short_ratio = max(w, h) / max(1, min(w, h))
    area_ratio = (w * h) / max(1, image.shape[0] * image.shape[1])
    clip_probability = _clip_currency_probability(clip_result)
    reasons: list[str] = []
    localized = bool(detection.get("geometric_candidate", detection.get("detected")))
    strong_currency_semantics = clip_probability is not None and clip_probability >= 0.75
    checks = {
        "minimum_resolution": min(w, h) >= 160,
        "plausible_note_aspect_ratio": 1.25 <= long_short_ratio <= 3.25,
        "sufficient_note_area": area_ratio >= 0.10,
        "currency_semantics": clip_probability is None or clip_probability >= 0.30,
        "localized_or_strong_currency_semantics": localized or strong_currency_semantics,
    }
    if not checks["minimum_resolution"]:
        reasons.append("Capture is too small for security-feature inspection")
    if not checks["plausible_note_aspect_ratio"]:
        reasons.append("Detected object does not have a plausible banknote aspect ratio")
    if not checks["sufficient_note_area"]:
        reasons.append("Banknote occupies too little of the frame")
    if not checks["currency_semantics"]:
        reasons.append("Vision-language model does not identify the object as currency")
    if not checks["localized_or_strong_currency_semantics"]:
        reasons.append("A banknote boundary could not be localized with sufficient semantic confidence")
    # Generic COCO YOLO detections are not accepted as proof of currency.
    confidence = sum(1.0 for passed in checks.values() if passed) / len(checks)
    valid = all(checks.values())
    return {
        "is_currency_candidate": valid,
        "confidence": round(confidence, 4),
        "checks": checks,
        "rejection_reasons": reasons,
        "requires_recapture": not valid,
        "aspect_ratio": round(long_short_ratio, 4),
        "frame_coverage": round(area_ratio, 4),
        "clip_currency_probability": None if clip_probability is None else round(clip_probability, 4),
    }


def inspect_security_features(
    regions: dict[str, np.ndarray],
    *,
    capture_mode: str = "rgb",
    expected_denomination: str | None = None,
    supplied_serial: str | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}

    micro = regions.get("micro_lettering")
    if micro is not None:
        sharpness = float(cv2.Laplacian(_gray(micro), cv2.CV_64F).var())
        edge_density = float(np.mean(cv2.Canny(_gray(micro), 70, 160) > 0))
        results["microprint"] = {
            "status": "pass" if sharpness >= 45 and edge_density >= 0.035 else "review",
            "sharpness": round(sharpness, 2),
            "edge_density": round(edge_density, 4),
        }

    thread = regions.get("security_thread")
    if thread is not None:
        gray = _gray(thread)
        vertical_profile = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0)), axis=0)
        prominence = float(np.max(vertical_profile) / max(np.mean(vertical_profile), 1e-6))
        continuity = float(np.mean(gray < np.percentile(gray, 35)))
        results["security_thread"] = {
            "status": "pass" if prominence >= 1.45 and continuity >= 0.20 else "review",
            "line_prominence": round(prominence, 3),
            "continuity": round(continuity, 3),
        }

    watermark = regions.get("watermark")
    if watermark is not None:
        gray = _gray(watermark)
        contrast = float(np.std(gray) / 64.0)
        structure = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        results["watermark"] = {
            "status": "pass" if contrast >= 0.18 and structure >= 12 else "review",
            "contrast": round(contrast, 3),
            "structure": round(structure, 2),
            "limitation": "RGB screening; transmitted-light capture improves reliability",
        }

    serial = regions.get("serial_number")
    if serial is not None:
        gray = cv2.resize(_gray(serial), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        character_candidates = 0
        for contour in contours:
            _, _, cw, ch = cv2.boundingRect(contour)
            if ch >= binary.shape[0] * 0.20 and 0.12 <= cw / max(ch, 1) <= 1.25:
                character_candidates += 1
        normalized_serial = re.sub(r"[^A-Z0-9]", "", str(supplied_serial or "").upper())
        pattern_valid = bool(re.fullmatch(r"[A-Z0-9]{6,12}", normalized_serial)) if normalized_serial else None
        visual_pass = 5 <= character_candidates <= 14
        results["serial_number"] = {
            "status": "pass" if visual_pass and pattern_valid is not False else "review",
            "character_candidates": character_candidates,
            "ocr_or_scanner_value": normalized_serial or None,
            "pattern_valid": pattern_valid,
            "source": "client_ocr_or_scanner" if normalized_serial else "visual_consistency_only",
            "limitation": None if normalized_serial else "No OCR/scanner value supplied; visual character consistency checked",
        }

    capture_mode = capture_mode.lower().strip()
    if capture_mode == "uv":
        values = cv2.cvtColor(next(iter(regions.values())), cv2.COLOR_BGR2HSV) if regions else None
        fluorescence = float(np.mean(values[:, :, 1] > 80)) if values is not None else 0.0
        results["uv_fluorescence"] = {
            "status": "pass" if 0.01 <= fluorescence <= 0.65 else "review",
            "fluorescent_pixel_ratio": round(fluorescence, 4),
        }
    else:
        results["uv_fluorescence"] = {
            "status": "not_captured",
            "required_capture_mode": "uv",
            "limitation": "UV properties cannot be inferred from an RGB image",
        }

    denomination = str(expected_denomination or "").replace("INR", "").replace("Rs", "").strip()
    results["denomination"] = {
        "expected": denomination or None,
        "supported": denomination in SUPPORTED_DENOMINATIONS if denomination else None,
        "supported_denominations": list(SUPPORTED_DENOMINATIONS),
        "recognized": None,
        "status": "review" if denomination and denomination not in SUPPORTED_DENOMINATIONS else "not_verified",
    }
    passed = sum(item.get("status") == "pass" for item in results.values())
    reviewed = sum(item.get("status") == "review" for item in results.values())
    return {
        "capture_mode": capture_mode,
        "features": results,
        "pass_count": passed,
        "review_count": reviewed,
        "screening_complete": reviewed == 0 and capture_mode == "uv",
    }
