"""Export and benchmark the trained multi-region currency model for edge CPUs."""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from models.vision.classifier import PreNormMILAggregator
from models.vision.detector import CURRENCY_REGIONS


MODEL_DIR = ROOT / "data" / "trained_models" / "forgery_classifier"
REGION_COUNT = 1 + len(CURRENCY_REGIONS)


class EdgeCurrencyModel(nn.Module):
    def __init__(self, backbone, attention_head, classifier_head):
        super().__init__()
        self.backbone = backbone
        self.attention_head = attention_head
        self.classifier_head = classifier_head

    def forward(self, images):
        batch_size, region_count, channels, height, width = images.shape
        features = self.backbone(images.reshape(batch_size * region_count, channels, height, width))
        features = features.mean(dim=(2, 3)).reshape(batch_size, region_count, -1)
        return torch.softmax(self.classifier_head(self.attention_head(features)), dim=1)


def main() -> None:
    import timm

    checkpoint_path = MODEL_DIR / "model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0, global_pool="")
    attention = PreNormMILAggregator(backbone.num_features, num_regions=REGION_COUNT)
    classifier = nn.Sequential(
        nn.LayerNorm(attention.d_model),
        nn.Linear(attention.d_model, 256),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(128, 2),
    )
    backbone.load_state_dict(state["backbone"])
    attention.load_state_dict(state["aggregator"])
    classifier.load_state_dict(state["classifier_head"])
    model = EdgeCurrencyModel(backbone, attention, classifier).eval()
    example = torch.zeros(1, REGION_COUNT, 3, 224, 224)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example)
        traced = torch.jit.optimize_for_inference(traced)
        output_path = MODEL_DIR / "currency_edge_model.pt"
        torch.jit.save(traced, output_path)
        for _ in range(5):
            traced(example)
        latencies = []
        for _ in range(30):
            started = time.perf_counter()
            traced(example)
            latencies.append((time.perf_counter() - started) * 1000)
    metadata = {
        "artifact": str(output_path),
        "format": "TorchScript",
        "input_shape": [1, REGION_COUNT, 3, 224, 224],
        "output": ["genuine_probability", "counterfeit_probability"],
        "cpu_threads": torch.get_num_threads(),
        "latency_ms": {
            "median": round(statistics.median(latencies), 2),
            "p95": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 2),
        },
        "sensor_contract": [
            "rgb_front", "rgb_back", "uv", "transmitted", "tilt_rgb",
            "ir_optional", "magnetic", "thickness",
        ],
        "claim_boundary": "Screening artifact; hardware calibration and certified specimens required before bank deployment",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (MODEL_DIR / "edge_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
