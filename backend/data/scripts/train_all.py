"""
Master Training Pipeline — Run All Training Steps.

Trains models from available datasets and saves them.

Usage:
    cd backend
    python data/scripts/train_all.py

This will:
1. Generate the scam text dataset (240+ samples)
2. Fine-tune DistilBERT for scam detection
3. Train vision only when verified real currency data is available
4. Train GAT for fraud network classification
5. Train XGBoost only from labelled out-of-fold agent predictions
"""

import sys
import time
import argparse
import asyncio
from pathlib import Path

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(smoke: bool = False):
    start = time.time()
    print("=" * 70)
    print("🛡️  DIGITAL PUBLIC SAFETY SHIELD — Master Training Pipeline")
    print("=" * 70)

    # ─── Step 1: Generate Datasets ────────────────────────────
    print("\n\n" + "▶" * 35)
    print("STEP 1: Generating Datasets")
    print("▶" * 35)

    from generate_text_dataset import save_dataset
    save_dataset()

    # ─── Step 2: Train Text Classifier ────────────────────────
    print("\n\n" + "▶" * 35)
    print("STEP 2: Training DistilBERT Scam Classifier")
    print("▶" * 35)

    try:
        from train_text_classifier import train as train_text
        train_text(smoke=smoke)
    except Exception as e:
        print(f"⚠️ Text classifier training failed: {e}")
        print("   This might need: pip install transformers torch")

    # ─── Step 3: Train Vision Classifier ──────────────────────
    print("\n\n" + "▶" * 35)
    print("STEP 3: Training Hybrid CNN-Transformer Forgery Classifier")
    print("▶" * 35)

    try:
        from train_vision_classifier import train as train_vision
        train_vision(smoke=smoke)
    except Exception as e:
        print(f"⚠️ Vision classifier training failed: {e}")
        print("   Add verified images under data/training/currency/genuine and counterfeit")

    # ─── Step 4: Train Graph Model ────────────────────────────
    print("\n\n" + "▶" * 35)
    print("STEP 4: Training Graph Attention Network")
    print("▶" * 35)

    try:
        from train_graph_model import train as train_graph
        train_graph(smoke=smoke)
    except Exception as e:
        print(f"⚠️ GAT training failed: {e}")
        print("   This might need: pip install networkx torch")

    # ─── Summary ──────────────────────────────────────────────
    print("\n\n" + "â–¶" * 35)
    print("STEP 5: Training XGBoost Fusion Meta-Learner")
    print("â–¶" * 35)

    try:
        from prepare_fusion_validation import main as prepare_fusion_validation
        from train_xgboost_fusion import train as train_xgboost
        asyncio.run(prepare_fusion_validation())
        train_xgboost(smoke=smoke)
    except Exception as e:
        print(f"âš ï¸ XGBoost fusion training failed: {e}")
        print("   Add labelled data/training/fusion_validation.jsonl")

    elapsed = time.time() - start
    print("\n\n" + "=" * 70)
    print(f"✅ ALL TRAINING COMPLETE — {elapsed:.1f}s total")
    print("=" * 70)

    models_dir = Path(__file__).resolve().parent.parent / "trained_models"
    print(f"\nTrained models saved to: {models_dir}")
    if models_dir.exists():
        for model_dir in sorted(models_dir.iterdir()):
            if model_dir.is_dir():
                meta = model_dir / "training_metadata.json"
                if meta.exists():
                    import json
                    with open(meta) as f:
                        m = json.load(f)
                    print(f"  📦 {model_dir.name}/")
                    for k, v in m.items():
                        print(f"     {k}: {v}")

    print("\n🚀 Next: Run the server with: python main.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    main(smoke=arguments.smoke)
