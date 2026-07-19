"""Capture validation and explainable Indian currency security-feature checks."""

from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np

from models.vision.currency_specifications import get_rbi_specification


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


def compare_tilt_captures(front_bytes: bytes, tilt_bytes: bytes, denomination: str | None) -> dict[str, Any]:
    """Measure an optically-variable colour signal from a registered image pair."""
    front = cv2.imdecode(np.frombuffer(front_bytes, np.uint8), cv2.IMREAD_COLOR)
    tilt = cv2.imdecode(np.frombuffer(tilt_bytes, np.uint8), cv2.IMREAD_COLOR)
    specification = get_rbi_specification(denomination)
    required = bool(specification and specification.get("thread_colour_shift"))
    if front is None or tilt is None:
        return {"status": "invalid_capture", "required": required}
    tilt = cv2.resize(tilt, (front.shape[1], front.shape[0]), interpolation=cv2.INTER_AREA)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        correlation, warp = cv2.findTransformECC(
            cv2.cvtColor(front, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(tilt, cv2.COLOR_BGR2GRAY),
            warp,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-5),
        )
        registered = cv2.warpAffine(
            tilt,
            warp,
            (front.shape[1], front.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        )
    except cv2.error:
        correlation = 0.0
        registered = tilt
    height, width = front.shape[:2]
    y1, y2 = int(height * 0.65), int(height * 0.96)
    x1, x2 = int(width * 0.70), int(width * 0.97)
    flat_lab = cv2.cvtColor(front[y1:y2, x1:x2], cv2.COLOR_BGR2LAB).reshape(-1, 3)
    tilt_lab = cv2.cvtColor(registered[y1:y2, x1:x2], cv2.COLOR_BGR2LAB).reshape(-1, 3)
    colour_delta = float(np.linalg.norm(np.median(flat_lab, axis=0) - np.median(tilt_lab, axis=0)))
    registered_pair = float(correlation) >= 0.45
    signal_present = registered_pair and 4.0 <= colour_delta <= 45.0
    return {
        "status": "pass" if signal_present else "review" if required else "not_required",
        "required": required,
        "registration_correlation": round(float(correlation), 4),
        "lab_colour_delta": round(colour_delta, 3),
        "signal_present": signal_present,
        "method": "ECC-registered paired flat/tilt LAB delta",
        "limitation": "Optical screening signal; calibrated illumination and spectral sensors are required for certification",
    }


def validate_currency_candidate(
    image: np.ndarray,
    note_crop: np.ndarray,
    detection: dict[str, Any],
    clip_result: dict[str, Any],
    capture_mode: str = "rgb",
) -> dict[str, Any]:
    h, w = note_crop.shape[:2]
    long_short_ratio = max(w, h) / max(1, min(w, h))
    area_ratio = (w * h) / max(1, image.shape[0] * image.shape[1])
    clip_probability = _clip_currency_probability(clip_result)
    reasons: list[str] = []
    localized = bool(detection.get("geometric_candidate", detection.get("detected")))
    strong_currency_semantics = clip_probability is not None and clip_probability >= 0.75
    sensor_capture = capture_mode.lower().strip() in {"uv", "ir", "transmitted"}
    checks = {
        "minimum_resolution": min(w, h) >= 160,
        "plausible_note_aspect_ratio": 1.25 <= long_short_ratio <= 3.25,
        "sufficient_note_area": area_ratio >= 0.10,
        "currency_semantics": sensor_capture or clip_probability is None or clip_probability >= 0.30,
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
        "capture_mode": capture_mode,
        "sensor_semantics_bypassed": sensor_capture,
    }


def inspect_security_features(
    regions: dict[str, np.ndarray],
    *,
    capture_mode: str = "rgb",
    expected_denomination: str | None = None,
    supplied_serial: str | None = None,
    supplied_microtext: str | None = None,
    machine_signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    machine_signals = machine_signals or {}
    denomination = str(expected_denomination or "").replace("INR", "").replace("Rs", "").strip()
    official_specification = get_rbi_specification(denomination)
    full_note = regions.get("full_note")

    if full_note is not None:
        height, width = full_note.shape[:2]
        observed_aspect_ratio = max(width, height) / max(1, min(width, height))
        expected_aspect_ratio = (
            float(official_specification["expected_aspect_ratio"])
            if official_specification else None
        )
        aspect_error = (
            abs(observed_aspect_ratio - expected_aspect_ratio) / expected_aspect_ratio
            if expected_aspect_ratio else None
        )
        measured_width = machine_signals.get("physical_width_mm")
        measured_height = machine_signals.get("physical_height_mm")
        dimensional_pass = None
        if official_specification and measured_width is not None and measured_height is not None:
            expected_width, expected_height = official_specification["dimensions_mm"]
            dimensional_pass = (
                abs(float(measured_width) - expected_width) <= 2.0
                and abs(float(measured_height) - expected_height) <= 2.0
            )
        results["physical_geometry"] = {
            "status": (
                "pass" if dimensional_pass is True
                else "review" if dimensional_pass is False or (aspect_error is not None and aspect_error > 0.18)
                else "not_captured"
            ),
            "observed_image_aspect_ratio": round(observed_aspect_ratio, 4),
            "expected_aspect_ratio": expected_aspect_ratio,
            "aspect_ratio_error": None if aspect_error is None else round(aspect_error, 4),
            "measured_dimensions_mm": [measured_width, measured_height] if measured_width is not None else None,
            "required_sensor": "calibrated_physical_dimension_sensor",
        }

    micro = regions.get("micro_lettering")
    if micro is not None:
        sharpness = float(cv2.Laplacian(_gray(micro), cv2.CV_64F).var())
        edge_density = float(np.mean(cv2.Canny(_gray(micro), 70, 160) > 0))
        normalized_microtext = re.sub(r"[^A-Z0-9]", "", str(supplied_microtext or "").upper())
        expected_microtext = official_specification.get("microtext", []) if official_specification else []
        matched_microtext = [
            token for token in expected_microtext
            if re.sub(r"[^A-Z0-9]", "", token.upper()) in normalized_microtext
        ]
        text_verified = len(matched_microtext) >= min(2, len(expected_microtext)) if supplied_microtext else False
        results["microprint"] = {
            "status": "pass" if sharpness >= 45 and edge_density >= 0.035 and text_verified else "review",
            "sharpness": round(sharpness, 2),
            "edge_density": round(edge_density, 4),
            "expected_text": expected_microtext,
            "ocr_or_scanner_value": supplied_microtext,
            "matched_tokens": matched_microtext,
            "text_verified": text_verified,
            "limitation": None if supplied_microtext else "Visual detail measured, but microtext requires macro OCR/scanner evidence",
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
        transmitted = capture_mode.lower().strip() == "transmitted"
        results["watermark"] = {
            "status": "pass" if transmitted and contrast >= 0.18 and structure >= 12 else "review" if transmitted else "not_captured",
            "contrast": round(contrast, 3),
            "structure": round(structure, 2),
            "limitation": None if transmitted else "Watermark and electrotype require transmitted-light capture",
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
        secondary = regions.get("serial_number_secondary")
        secondary_candidates = None
        if secondary is not None:
            secondary_gray = cv2.resize(_gray(secondary), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            secondary_binary = cv2.threshold(
                secondary_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )[1]
            secondary_contours, _ = cv2.findContours(
                secondary_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            secondary_candidates = sum(
                1 for contour in secondary_contours
                for _, _, candidate_width, candidate_height in [cv2.boundingRect(contour)]
                if candidate_height >= secondary_binary.shape[0] * 0.20
                and 0.12 <= candidate_width / max(candidate_height, 1) <= 1.25
            )
        panels_consistent = secondary_candidates is None or abs(character_candidates - secondary_candidates) <= 3
        results["serial_number"] = {
            "status": "pass" if visual_pass and panels_consistent and pattern_valid is True else "review",
            "character_candidates": character_candidates,
            "secondary_panel_character_candidates": secondary_candidates,
            "panels_visually_consistent": panels_consistent,
            "ocr_or_scanner_value": normalized_serial or None,
            "pattern_valid": pattern_valid,
            "source": "client_ocr_or_scanner" if normalized_serial else "visual_consistency_only",
            "limitation": None if normalized_serial else "No OCR/scanner value supplied; visual character consistency checked",
        }

    register = regions.get("see_through_register")
    if register is not None:
        register_edges = float(np.mean(cv2.Canny(_gray(register), 60, 150) > 0))
        results["see_through_register"] = {
            "status": "pass" if capture_mode == "transmitted" and register_edges >= 0.03 else "review" if capture_mode == "transmitted" else "not_captured",
            "edge_density": round(register_edges, 4),
            "required_capture_mode": "transmitted",
        }

    expected_bleed_lines = int(official_specification.get("angular_bleed_lines", 0)) if official_specification else 0
    observed_bleed_lines = {}
    for side in ("left", "right"):
        region = regions.get(f"bleed_lines_{side}")
        count = 0
        if region is not None:
            lines = cv2.HoughLinesP(cv2.Canny(_gray(region), 60, 150), 1, np.pi / 180, 18, minLineLength=12, maxLineGap=4)
            if lines is not None:
                count = min(12, len(lines))
        observed_bleed_lines[side] = count
    bleed_match = (
        expected_bleed_lines == 0
        or all(abs(count - expected_bleed_lines) <= 2 for count in observed_bleed_lines.values())
    )
    results["angular_bleed_lines"] = {
        "status": "pass" if bleed_match else "review",
        "expected_per_side": expected_bleed_lines,
        "observed_line_candidates": observed_bleed_lines,
        "limitation": "Optical line candidates do not verify raised intaglio",
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

    magnetic_thread = machine_signals.get("magnetic_thread_detected")
    results["magnetic_thread"] = {
        "status": "pass" if magnetic_thread is True else "review" if magnetic_thread is False else "not_captured",
        "detected": magnetic_thread,
        "required_sensor": "magnetic_ink_or_thread_sensor",
    }
    results["double_feed"] = {
        "status": "review" if machine_signals.get("double_feed_detected") is True else "pass" if machine_signals.get("double_feed_detected") is False else "not_captured",
        "detected": machine_signals.get("double_feed_detected"),
        "thickness_mm": machine_signals.get("thickness_mm"),
        "required_sensor": "ultrasonic_or_capacitive_thickness_sensor",
    }
    colour_shift_required = bool(official_specification and official_specification.get("thread_colour_shift"))
    results["colour_shift"] = {
        "status": "review" if capture_mode == "tilt_rgb" else "not_required" if not colour_shift_required else "not_captured",
        "required": colour_shift_required,
        "limitation": "Requires paired flat and tilted captures; single-image colour cannot prove optically variable ink",
    }

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
        "screening_complete": reviewed == 0 and capture_mode in {"uv", "transmitted"},
        "official_rbi_specification": official_specification,
        "claim_boundary": "Machine-assisted screening; certification requires calibrated sensors and governed specimens",
    }
