import cv2
import numpy as np

from models.vision.currency_features import compare_tilt_captures, inspect_security_features, validate_currency_candidate


def test_non_currency_clip_semantics_reject_document():
    image = np.full((800, 600, 3), 255, dtype=np.uint8)
    result = validate_currency_candidate(
        image,
        image,
        {"detected": True, "geometric_candidate": True},
        {
            "available": True,
            "prompt_scores": {
                "genuine_currency": 0.02,
                "counterfeit_currency": 0.03,
                "synthetic_print": 0.01,
                "tampered_document": 0.94,
            },
        },
    )
    assert result["is_currency_candidate"] is False
    assert result["checks"]["currency_semantics"] is False


def test_uv_capture_does_not_require_rgb_clip_semantics():
    image = np.full((300, 700, 3), 50, dtype=np.uint8)
    result = validate_currency_candidate(
        image,
        image,
        {"detected": True, "geometric_candidate": True},
        {"prompt_scores": {"tampered_document": 0.99}},
        capture_mode="uv",
    )
    assert result["checks"]["currency_semantics"] is True
    assert result["sensor_semantics_bypassed"] is True


def test_strong_currency_semantics_can_recover_failed_contour_localization():
    image = np.full((480, 640, 3), 180, dtype=np.uint8)
    result = validate_currency_candidate(
        image,
        image,
        {"detected": False, "geometric_candidate": False},
        {
            "prompt_scores": {
                "genuine_currency": 0.80,
                "counterfeit_currency": 0.10,
                "poor_quality_scan": 0.05,
                "synthetic_print": 0.02,
            }
        },
    )

    assert result["is_currency_candidate"] is True
    assert result["checks"]["localized_or_strong_currency_semantics"] is True


def test_uv_is_never_claimed_from_rgb_capture():
    note = np.random.default_rng(3).integers(0, 255, (300, 700, 3), dtype=np.uint8)
    regions = {
        "micro_lettering": note[80:150, 100:600],
        "security_thread": note[:, 260:300],
        "serial_number": note[20:80, 20:300],
        "watermark": note[30:210, 400:680],
    }
    result = inspect_security_features(
        regions,
        capture_mode="rgb",
        expected_denomination="200",
        supplied_serial="9AB 123456",
    )
    assert result["features"]["uv_fluorescence"]["status"] == "not_captured"
    assert result["features"]["denomination"]["supported"] is True
    assert result["features"]["serial_number"]["pattern_valid"] is True
    assert result["official_rbi_specification"]["dimensions_mm"] == [146, 66]
    assert result["features"]["watermark"]["status"] == "not_captured"


def test_tilt_pair_reports_registered_colour_signal_without_certification_claim():
    base = np.full((240, 520, 3), (110, 120, 120), dtype=np.uint8)
    tilted = base.copy()
    tilted[156:230, 364:504] = (150, 105, 105)
    ok_front, front_bytes = cv2.imencode(".jpg", base)
    ok_tilt, tilt_bytes = cv2.imencode(".jpg", tilted)
    assert ok_front and ok_tilt
    result = compare_tilt_captures(front_bytes.tobytes(), tilt_bytes.tobytes(), "500")
    assert result["required"] is True
    assert result["method"].startswith("ECC-registered")
    assert "certification" in result["limitation"]
