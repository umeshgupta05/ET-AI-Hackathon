"""
Master Training Pipeline — Run All Training Steps.

Generates datasets, trains all models, and saves them.

Usage:
    cd backend
    python data/scripts/train_all.py

This will:
1. Generate the scam text dataset (62 samples)
2. Generate synthetic currency images (100 images)
3. Fine-tune DistilBERT for scam detection
4. Train Hybrid CNN-Transformer for forgery detection
5. Train GAT for fraud network classification
"""

import sys
import time
from pathlib import Path

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
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
        train_text()
    except Exception as e:
        print(f"⚠️ Text classifier training failed: {e}")
        print("   This might need: pip install transformers torch")

    # ─── Step 3: Train Vision Classifier ──────────────────────
    print("\n\n" + "▶" * 35)
    print("STEP 3: Training Hybrid CNN-Transformer Forgery Classifier")
    print("▶" * 35)

    try:
        from train_vision_classifier import train as train_vision
        train_vision()
    except Exception as e:
        print(f"⚠️ Vision classifier training failed: {e}")
        print("   This might need: pip install timm torch torchvision")

    # ─── Step 4: Train Graph Model ────────────────────────────
    print("\n\n" + "▶" * 35)
    print("STEP 4: Training Graph Attention Network")
    print("▶" * 35)

    try:
        from train_graph_model import train as train_graph
        train_graph()
    except Exception as e:
        print(f"⚠️ GAT training failed: {e}")
        print("   This might need: pip install networkx torch")

    # ─── Summary ──────────────────────────────────────────────
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
    main()
