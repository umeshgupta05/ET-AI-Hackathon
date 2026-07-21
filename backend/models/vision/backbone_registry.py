from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BackboneSpec:
    key: str
    timm_name: str
    default_input_size: int
    default_backbone_lr: float
    default_head_lr: float
    default_weight_decay: float
    license_name: str


BACKBONE_REGISTRY = {
    "efficientnet_b0": BackboneSpec(
        key="efficientnet_b0",
        timm_name="efficientnet_b0",
        default_input_size=224,
        default_backbone_lr=3e-5,
        default_head_lr=3e-4,
        default_weight_decay=0.01,
        license_name="Apache-2.0"
    ),
    "convnextv2_tiny_fcmae": BackboneSpec(
        key="convnextv2_tiny_fcmae",
        timm_name="convnextv2_tiny.fcmae",
        default_input_size=224,
        default_backbone_lr=1e-5,
        default_head_lr=2e-4,
        default_weight_decay=0.05,
        license_name="CC-BY-NC-4.0"
    )
}

def get_backbone_spec(key: str) -> BackboneSpec:
    if key not in BACKBONE_REGISTRY:
        supported = ", ".join(BACKBONE_REGISTRY.keys())
        raise ValueError(f"Unknown backbone '{key}'. Supported options: {supported}")
    return BACKBONE_REGISTRY[key]
