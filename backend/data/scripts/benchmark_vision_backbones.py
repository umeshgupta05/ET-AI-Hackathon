"""
Benchmark Vision Backbones
Runs comprehensive evaluation across supported architectures.
Compares Accuracy, Safety, Latency, Memory, and Deployability.
"""
import sys
import os
import argparse
import time
import json
import csv
from pathlib import Path

# Fix Windows encoding
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import numpy as np
from data.scripts.train_vision_classifier import train, OUTPUT_DIR
from models.vision.backbone_registry import BACKBONE_REGISTRY
from models.vision.classifier import get_forgery_classifier
from models.vision.detector import REGION_NAMES

def measure_latency_and_memory(backbone_key, device):
    classifier = get_forgery_classifier()
    classifier._backbone_key = backbone_key
    
    try:
        import asyncio
        asyncio.run(classifier.initialize())
    except Exception as e:
        return None, None
        
    if not classifier._trained_weights_loaded and not classifier._initialized:
        return None, None
        
    dummy_region = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_regions = {name: dummy_region for name in REGION_NAMES}
    
    # Warmup
    for _ in range(3):
        classifier.classify_all_regions(dummy_regions)
        
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
    latencies = []
    for _ in range(10):
        t0 = time.time()
        classifier.classify_all_regions(dummy_regions)
        if device.type == "cuda": torch.cuda.synchronize()
        t1 = time.time()
        latencies.append((t1 - t0) * 1000)
        
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2) if device.type == "cuda" else 0.0
    
    return np.median(latencies), peak_mem

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test")
    args = parser.parse_args()

    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class TrainArgs:
        def __init__(self, backbone, smoke):
            self.mode = "cross_validate"
            self.backbone = backbone
            self.smoke = smoke
            self.allow_synthetic_debug_data = False
            self.ssl_mode = "none"
            self.folds = 2 if smoke else 5
            self.fold_index = None
            self.loss = "cross_entropy"
            self.focal_gamma = 2.0
            self.label_smoothing = 0.0
            self.amp = True
            self.gradient_accumulation = 1
            self.max_genuine_frr = 0.05

    for backbone_key in BACKBONE_REGISTRY.keys():
        print(f"\n{'='*80}\nBenchmarking Backbone: {backbone_key}\n{'='*80}")
        t_args = TrainArgs(backbone=backbone_key, smoke=args.smoke)
        start = time.time()
        
        try:
            train(args=t_args)
            success = True
        except Exception as e:
            print(f"Failed to benchmark {backbone_key}: {e}")
            success = False
            
        elapsed = time.time() - start
        
        # Read generated metadata
        meta_file = "smoke_training_metadata.json" if args.smoke else "candidate_training_metadata.json"
        meta_path = OUTPUT_DIR / backbone_key / meta_file
        
        meta = {}
        if meta_path.exists():
            with open(meta_path, "r") as f: meta = json.load(f)
            
        latency_ms, peak_mem_mb = measure_latency_and_memory(backbone_key, device)
        
        # Parameter count
        param_count = 0
        model_size_mb = 0
        model_path = OUTPUT_DIR / backbone_key / ("model_smoke.pth" if args.smoke else "model_candidate.pth")
        if model_path.exists():
            model_size_mb = model_path.stat().st_size / (1024**2)
            try:
                state = torch.load(model_path, weights_only=True, map_location="cpu")
                param_count = sum(p.numel() for k in ["backbone", "aggregator", "classifier_head"] if k in state for p in state[k].values())
            except: pass
            
        results[backbone_key] = {
            "success": success,
            "training_time_seconds": elapsed,
            "mean_f1": meta.get("mean_f1", 0.0),
            "mean_roc_auc": meta.get("mean_roc_auc", 0.5),
            "mean_pr_auc": meta.get("mean_pr_auc", 0.5),
            "calibrated_frr": meta.get("calibration", {}).get("frr", 1.0),
            "latency_ms": latency_ms or 0.0,
            "peak_memory_mb": peak_mem_mb or 0.0,
            "param_count": param_count,
            "model_size_mb": model_size_mb
        }
        
    print("\nBenchmark Complete!")
    
    # Save outputs
    with open("vision_backbone_benchmark.json", "w") as f: json.dump(results, f, indent=2)
    
    with open("vision_backbone_benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["backbone", "success", "mean_f1", "mean_roc_auc", "mean_pr_auc", "latency_ms", "peak_memory_mb", "param_count", "model_size_mb", "training_time_seconds"])
        writer.writeheader()
        for k, v in results.items():
            row = {"backbone": k}
            row.update(v)
            writer.writerow(row)
            
    with open("vision_backbone_benchmark.md", "w") as f:
        f.write("# Vision Backbone Benchmark Results\n\n")
        f.write("| Backbone | Success | Mean F1 | ROC-AUC | Latency (ms) | Peak Mem (MB) | Params | Size (MB) |\n")
        f.write("|----------|---------|---------|---------|--------------|---------------|--------|-----------|\n")
        for k, v in results.items():
            f.write(f"| {k} | {v['success']} | {v['mean_f1']:.3f} | {v['mean_roc_auc']:.3f} | {v['latency_ms']:.1f} | {v['peak_memory_mb']:.1f} | {v['param_count']:,} | {v['model_size_mb']:.1f} |\n")
            print(f"{k}: Success={v['success']}, F1={v['mean_f1']:.3f}, Latency={v['latency_ms']:.1f}ms")
            
if __name__ == "__main__":
    main()
