"""Compare supported vision backbones without misreporting unavailable models."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
import torch

from data.scripts.train_vision_classifier import OUTPUT_DIR, create_parser, train
from models.vision.backbone_registry import BACKBONE_REGISTRY
from models.vision.classifier import HybridForgeryClassifier, REGION_NAMES


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return float(ordered[index])


def _checkpoint_name(backbone_key: str) -> str | None:
    directory = OUTPUT_DIR / backbone_key
    if (directory / "model_candidate.pth").exists():
        return "model_candidate.pth"
    if (directory / "model.pth").exists():
        return "model.pth"
    return None


def measure_runtime(backbone_key: str, iterations: int) -> dict:
    checkpoint_name = _checkpoint_name(backbone_key)
    if checkpoint_name is None:
        return {
            "status": "NOT_RUN",
            "reason": "No candidate or active checkpoint",
        }

    classifier = HybridForgeryClassifier(
        backbone_key=backbone_key, checkpoint_name=checkpoint_name
    )
    try:
        asyncio.run(classifier.initialize())
    except Exception as exc:
        return {"status": "FAILED", "reason": str(exc)}
    if not classifier._trained_weights_loaded:
        return {
            "status": "FAILED",
            "reason": classifier.get_stats().get("model_status", "checkpoint unavailable"),
        }

    size = BACKBONE_REGISTRY[backbone_key].default_input_size
    dummy_bgr = np.zeros((size, size, 3), dtype=np.uint8)
    dummy_regions = {name: dummy_bgr.copy() for name in REGION_NAMES}

    for _ in range(3):
        classifier.classify_region(dummy_bgr)
        classifier.classify_all_regions(dummy_regions)
    if classifier._device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    single_latencies: list[float] = []
    bag_latencies: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        classifier.classify_region(dummy_bgr)
        if classifier._device.type == "cuda":
            torch.cuda.synchronize()
        single_latencies.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        classifier.classify_all_regions(dummy_regions)
        if classifier._device.type == "cuda":
            torch.cuda.synchronize()
        bag_latencies.append((time.perf_counter() - start) * 1000.0)

    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if classifier._device.type == "cuda"
        else 0.0
    )
    parameter_count = sum(
        parameter.numel()
        for module in (
            classifier._backbone,
            classifier._aggregator,
            classifier._classifier_head,
            classifier._contrastive_head,
        )
        for parameter in module.parameters()
    )
    checkpoint_path = OUTPUT_DIR / backbone_key / checkpoint_name
    return {
        "status": "PASS",
        "checkpoint": checkpoint_name,
        "device": str(classifier._device),
        "parameter_count": int(parameter_count),
        "checkpoint_size_mb": checkpoint_path.stat().st_size / (1024**2),
        "single_region_latency_ms_median": statistics.median(single_latencies),
        "single_region_latency_ms_p95": _percentile(single_latencies, 0.95),
        "region_bag_latency_ms_median": statistics.median(bag_latencies),
        "region_bag_latency_ms_p95": _percentile(bag_latencies, 0.95),
        "notes_per_second": 1000.0 / max(1e-9, statistics.median(bag_latencies)),
        "peak_cuda_memory_mb": peak_memory,
    }


def read_training_metrics(backbone_key: str) -> dict:
    directory = OUTPUT_DIR / backbone_key
    for name in (
        "cross_validation_metadata.json",
        "candidate_training_metadata.json",
        "smoke_training_metadata.json",
    ):
        path = directory / name
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            calibration = payload.get("calibration", {})
            return {
                "status": "PASS",
                "source": name,
                "mean_f1": payload.get("mean_f1", 0.0),
                "mean_roc_auc": payload.get("mean_roc_auc", 0.5),
                "mean_pr_auc": payload.get("mean_pr_auc", 0.5),
                "counterfeit_false_accept_rate": calibration.get("far"),
                "genuine_false_reject_rate": calibration.get("frr"),
                "calibration_policy_satisfied": calibration.get("policy_satisfied"),
            }
    return {"status": "NOT_RUN", "reason": "No training metadata"}


def optionally_run_training(backbone_key: str, smoke: bool) -> dict:
    parser = create_parser()
    args = parser.parse_args(
        [
            "--mode",
            "cross_validate",
            "--backbone",
            backbone_key,
            "--folds",
            "2" if smoke else "5",
            *( ["--smoke"] if smoke else [] ),
        ]
    )
    start = time.perf_counter()
    try:
        train(args=args)
        return {"status": "PASS", "seconds": time.perf_counter() - start}
    except Exception as exc:
        return {
            "status": "FAILED",
            "seconds": time.perf_counter() - start,
            "reason": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "benchmarks",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for backbone_key in BACKBONE_REGISTRY:
        training_run = (
            optionally_run_training(backbone_key, args.smoke)
            if args.run_training
            else {"status": "NOT_RUN", "reason": "--run-training not supplied"}
        )
        results[backbone_key] = {
            "training_run": training_run,
            "metrics": read_training_metrics(backbone_key),
            "runtime": measure_runtime(backbone_key, args.iterations),
        }

    json_path = args.output_dir / "vision_backbone_benchmark.json"
    csv_path = args.output_dir / "vision_backbone_benchmark.csv"
    markdown_path = args.output_dir / "vision_backbone_benchmark.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    rows = []
    for backbone_key, result in results.items():
        metrics = result["metrics"]
        runtime = result["runtime"]
        rows.append(
            {
                "backbone": backbone_key,
                "metrics_status": metrics.get("status"),
                "runtime_status": runtime.get("status"),
                "mean_f1": metrics.get("mean_f1"),
                "mean_roc_auc": metrics.get("mean_roc_auc"),
                "mean_pr_auc": metrics.get("mean_pr_auc"),
                "counterfeit_far": metrics.get("counterfeit_false_accept_rate"),
                "genuine_frr": metrics.get("genuine_false_reject_rate"),
                "parameter_count": runtime.get("parameter_count"),
                "checkpoint_size_mb": runtime.get("checkpoint_size_mb"),
                "bag_latency_median_ms": runtime.get("region_bag_latency_ms_median"),
                "bag_latency_p95_ms": runtime.get("region_bag_latency_ms_p95"),
                "peak_cuda_memory_mb": runtime.get("peak_cuda_memory_mb"),
            }
        )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# Vision Backbone Benchmark",
        "",
        "No winner is declared unless both accuracy and runtime evidence are available.",
        "",
        "| Backbone | Metrics | Runtime | Mean F1 | ROC-AUC | FAR | FRR | Bag median ms | Bag p95 ms |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            "| {backbone} | {metrics_status} | {runtime_status} | {mean_f1} | "
            "{mean_roc_auc} | {counterfeit_far} | {genuine_frr} | "
            "{bag_latency_median_ms} | {bag_latency_p95_ms} |".format(**row)
        )
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    print(f"Wrote benchmark artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()