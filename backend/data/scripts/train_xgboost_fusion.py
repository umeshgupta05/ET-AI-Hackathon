"""Train a lightweight XGBoost fusion meta-learner on synthetic validation features."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from agents.ensemble import FEATURE_NAMES

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "trained_models" / "xgboost_fusion"
SEED = 42


def build_synthetic_features(n: int = 800) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    rows = []
    labels = []
    for _ in range(n):
        has_vision = rng.random() < 0.45
        has_speech = rng.random() < 0.35
        has_nlp = rng.random() < 0.85
        has_graph = True

        scam = rng.random() < 0.50
        center = 0.72 if scam else 0.24
        vision = np.clip(rng.normal(center, 0.16), 0, 1) if has_vision else 0.5
        forensic = np.clip(rng.normal(center, 0.18), 0, 1) if has_vision else 0.5
        clip = np.clip(rng.normal(center, 0.20), 0, 1) if has_vision else 0.5
        speech = np.clip(rng.normal(center, 0.18), 0, 1) if has_speech else 0.5
        nlp = np.clip(rng.normal(center, 0.14), 0, 1) if has_nlp else 0.5
        graph = np.clip(rng.normal(0.55 if scam else 0.22, 0.16), 0, 1)

        row = {
            "vision_score": vision,
            "vision_forensic_score": forensic,
            "vision_clip_score": clip,
            "speech_spoof_score": speech,
            "nlp_score": nlp,
            "graph_score": graph,
            "has_vision": float(has_vision),
            "has_speech": float(has_speech),
            "has_nlp": float(has_nlp),
            "has_graph": float(has_graph),
            "modality_image": float(has_vision),
            "modality_audio": float(has_speech),
            "modality_text": float(has_nlp),
        }
        rows.append([row[name] for name in FEATURE_NAMES])
        labels.append(int(scam))
    return np.array(rows, dtype=np.float32), np.array(labels, dtype=np.int64)


def train(smoke: bool = False) -> dict:
    sample_count = 240 if smoke else 800
    x, y = build_synthetic_features(sample_count)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.25, random_state=SEED, stratify=y
    )
    model = XGBClassifier(
        n_estimators=30 if smoke else 120,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=SEED,
    )
    model.fit(x_train, y_train)
    probs = model.predict_proba(x_val)[:, 1]
    preds = (probs >= 0.5).astype(int)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "model.json"
    model.save_model(str(model_path))

    metadata = {
        "model": "XGBClassifier",
        "training_mode": "smoke" if smoke else "full_synthetic",
        "sample_count": sample_count,
        "feature_names": FEATURE_NAMES,
        "validation_accuracy": float(accuracy_score(y_val, preds)),
        "validation_f1": float(f1_score(y_val, preds, zero_division=0)),
        "validation_roc_auc": float(roc_auc_score(y_val, probs)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT_DIR / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"XGBoost fusion model saved to: {model_path}")
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    train(smoke="--smoke" in sys.argv)
