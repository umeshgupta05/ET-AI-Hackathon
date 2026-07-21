"""
Benchmark Vision Backbones
Runs evaluation across supported architectures to select the promotion candidate.
"""
import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.scripts.train_vision_classifier import train
from models.vision.backbone_registry import BACKBONE_REGISTRY

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test")
    args = parser.parse_args()

    results = {}
    
    class TrainArgs:
        def __init__(self, backbone, smoke):
            self.backbone = backbone
            self.smoke = smoke
            self.allow_synthetic_debug_data = False
            self.ssl_mode = "none"
            self.folds = 2 if smoke else 5
            self.fold_index = None
            self.loss = "cross_entropy"
            self.amp = True
            self.gradient_accumulation = 1

    for backbone_key in BACKBONE_REGISTRY.keys():
        print(f"\n{'='*80}\nBenchmarking Backbone: {backbone_key}\n{'='*80}")
        t_args = TrainArgs(backbone=backbone_key, smoke=args.smoke)
        start = time.time()
        
        try:
            train(smoke=t_args.smoke, args=t_args)
            success = True
        except Exception as e:
            print(f"Failed to benchmark {backbone_key}: {e}")
            success = False
            
        elapsed = time.time() - start
        
        results[backbone_key] = {
            "success": success,
            "training_time_seconds": elapsed,
        }
        
    print("\nBenchmark Complete!")
    for k, v in results.items():
        print(f"{k}: Success={v['success']}, Time={v['training_time_seconds']:.2f}s")
        
if __name__ == "__main__":
    main()
