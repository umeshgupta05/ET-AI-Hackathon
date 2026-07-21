import sys
from pathlib import Path
import pytest
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.vision.backbone_registry import get_backbone_spec, BACKBONE_REGISTRY
from models.vision.classifier import PreNormMILAggregator, REGION_NAMES
from models.vision.preprocessing import get_region_tensor_transform, get_forensic_safe_geometric_transform

def test_backbone_registry():
    spec = get_backbone_spec("efficientnet_b0")
    assert spec.timm_name == "efficientnet_b0"
    assert spec.default_input_size == 224
    
    spec_cx = get_backbone_spec("convnextv2_tiny_fcmae")
    assert spec_cx.timm_name == "convnextv2_tiny.fcmae"
    
    with pytest.raises(ValueError):
        get_backbone_spec("unknown_backbone")

def test_preprocessing():
    # Test deterministic validation transform
    transform = get_region_tensor_transform("efficientnet_b0", 224, is_training=False)
    assert transform is not None
    
    # Test forensic geometric transform has no RandomHorizontalFlip
    geo_transform = get_forensic_safe_geometric_transform()
    transform_names = [type(t).__name__ for t in geo_transform.transforms]
    assert "RandomHorizontalFlip" not in transform_names

def test_mil_aggregator():
    B = 2
    R = len(REGION_NAMES)
    feature_dim = 1280
    d_model = 384
    
    aggregator = PreNormMILAggregator(feature_dim=feature_dim, num_regions=R, d_model=d_model, num_heads=6, num_layers=2)
    aggregator.eval()
    
    features = torch.randn(B, R, feature_dim)
    mask = torch.zeros(B, R, dtype=torch.bool)
    
    with torch.no_grad():
        # Test valid forward pass
        output = aggregator(features, mask)
        assert output.shape == (B, d_model)
        
        # Test missing token behavior
        mask[0, 1] = True # Region 1 missing in batch 0
        output_with_missing = aggregator(features, mask)
        
    assert output_with_missing.shape == (B, d_model)
    
    # Ensure missing region changed the output for batch 0 but not batch 1
    assert not torch.allclose(output[0], output_with_missing[0])
    assert torch.allclose(output[1], output_with_missing[1])

def test_mil_aggregator_validation():
    with pytest.raises(AssertionError):
        # d_model % num_heads != 0 (384 % 7 != 0)
        PreNormMILAggregator(feature_dim=1280, num_regions=14, d_model=384, num_heads=7)
