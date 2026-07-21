import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.vision.backbone_registry import get_backbone_spec
from models.vision.classifier import (
    ARCHITECTURE_VERSION,
    CHECKPOINT_FORMAT_VERSION,
    D_MODEL,
    NUM_CLASSES,
    NUM_HEADS,
    NUM_LAYERS,
    REGION_NAMES,
    HybridForgeryClassifier,
    PreNormMILAggregator,
)
from models.vision.preprocessing import (
    get_forensic_safe_geometric_transform,
    get_simclr_geometric_transform,
)


def test_backbone_registry_keeps_efficientnet_default_candidate_mapping():
    assert get_backbone_spec("efficientnet_b0").timm_name == "efficientnet_b0"
    assert get_backbone_spec("convnextv2_tiny_fcmae").timm_name == "convnextv2_tiny.fcmae"
    assert get_backbone_spec("convnextv2_tiny_fcmae").default_ssl_mode == "none"
    with pytest.raises(ValueError):
        get_backbone_spec("unknown")


def test_forensic_transforms_never_flip_and_limit_rotation():
    for pipeline in (get_forensic_safe_geometric_transform(), get_simclr_geometric_transform()):
        names = [type(transform).__name__ for transform in pipeline.transforms]
        assert "RandomHorizontalFlip" not in names
        assert "RandomVerticalFlip" not in names
        rotations = [transform for transform in pipeline.transforms if type(transform).__name__ == "RandomRotation"]
        assert rotations
        assert rotations[0].degrees == [-3.0, 3.0]


def test_prenorm_mil_shapes_and_missing_token():
    feature_dim = 32
    aggregator = PreNormMILAggregator(feature_dim, len(REGION_NAMES), d_model=24, num_heads=6, num_layers=1)
    aggregator.eval()
    features = torch.randn(2, len(REGION_NAMES), feature_dim)
    mask = torch.zeros(2, len(REGION_NAMES), dtype=torch.bool)
    with torch.no_grad():
        baseline = aggregator(features, mask)
        mask[0, 1] = True
        missing = aggregator(features, mask)
    assert baseline.shape == (2, 24)
    assert torch.isfinite(baseline).all()
    assert not torch.allclose(baseline[0], missing[0])
    assert torch.allclose(baseline[1], missing[1])


def test_prenorm_mil_validates_dimensions():
    with pytest.raises(ValueError):
        PreNormMILAggregator(768, len(REGION_NAMES), d_model=384, num_heads=7)
    aggregator = PreNormMILAggregator(32, len(REGION_NAMES), d_model=24, num_heads=6, num_layers=1)
    with pytest.raises(ValueError):
        aggregator(torch.randn(2, len(REGION_NAMES), 31))


def test_checkpoint_schema_rejects_cross_architecture():
    classifier = HybridForgeryClassifier(backbone_key="efficientnet_b0")
    state = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "backbone_key": "convnextv2_tiny_fcmae",
        "timm_name": "convnextv2_tiny.fcmae",
        "feature_dim": 1280,
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "num_regions": len(REGION_NAMES),
        "region_names": list(REGION_NAMES),
        "num_classes": NUM_CLASSES,
        "backbone": {},
        "aggregator": {},
        "classifier_head": {},
        "contrastive_head": {},
        "calibration": {},
    }
    with pytest.raises(ValueError):
        classifier._validate_checkpoint(state, "efficientnet_b0", 1280)