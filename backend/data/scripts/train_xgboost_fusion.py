"""Train XGBoost fusion from labelled out-of-fold agent predictions."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from agents.ensemble import FEATURE_NAMES

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "trained_models" / "xgboost_fusion"
DATASET_PATH = Path(__file__).resolve().parent.parent / "training" / "fusion_validation.jsonl"
SEED = 42
MIN_DEPLOYMENT_ACCURACY = 0.75
MIN_DEPLOYMENT_ROC_AUC = 0.80


def load_validation_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATASET_PATH}. Export labelled out-of-fold agent "
            "predictions before training the fusion model."
        )
    rows = []
    labels = []
    modality_groups = []
    with DATASET_PATH.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                rows.append([float(record[name]) for name in FEATURE_NAMES])
                labels.append(int(record["label"]))
                modality_groups.append(str(record.get("modality_group", "unknown")))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid fusion row {line_number}: {exc}") from exc
    if len(rows) < 100 or len(set(labels)) != 2:
        raise ValueError("Fusion training requires at least 100 labelled rows across both classes")
    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(modality_groups),
    )


def train(smoke: bool = False) -> dict:
    x, y, modality_groups = load_validation_features()
    sample_count = len(y)
    feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    base_scores = np.where(
        x[:, feature_index["has_vision"]] >= 0.5,
        x[:, feature_index["vision_score"]],
        np.where(
            x[:, feature_index["has_nlp"]] >= 0.5,
            x[:, feature_index["nlp_score"]],
            x[:, feature_index["speech_spoof_score"]],
        ),
    )
    (
        x_train,
        x_val,
        y_train,
        y_val,
        _,
        base_val,
        _,
        modality_val,
    ) = train_test_split(
        x,
        y,
        base_scores,
        modality_groups,
        test_size=0.25,
        random_state=SEED,
        stratify=np.asarray([f"{label}:{group}" for label, group in zip(y, modality_groups)]),
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
    base_preds = (base_val >= 0.5).astype(int)
    candidate_weights = (0.50, 0.70, 0.85, 1.00)
    xgboost_weight = min(
        candidate_weights,
        key=lambda weight: brier_score_loss(
            y_val, probs * weight + base_val * (1.0 - weight)
        ),
    )
    blended_probs = probs * xgboost_weight + base_val * (1.0 - xgboost_weight)
    blended_preds = (blended_probs >= 0.5).astype(int)
    per_modality = {}
    for modality in sorted(set(modality_val)):
        mask = modality_val == modality
        modality_y = y_val[mask]
        modality_probs = probs[mask]
        modality_preds = preds[mask]
        per_modality[modality] = {
            "count": int(mask.sum()),
            "accuracy": float(accuracy_score(modality_y, modality_preds)),
            "f1": float(f1_score(modality_y, modality_preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(modality_y, modality_probs))
            if len(set(modality_y)) == 2
            else None,
        }

    supported_modality_groups = [
        modality
        for modality, metrics in per_modality.items()
        if metrics["accuracy"] >= MIN_DEPLOYMENT_ACCURACY
        and metrics["roc_auc"] is not None
        and metrics["roc_auc"] >= MIN_DEPLOYMENT_ROC_AUC
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "model.json"
    model.save_model(str(model_path))

    metadata = {
        "model": "XGBClassifier",
        "training_mode": "labelled_out_of_fold",
        "run_mode": "smoke" if smoke else "full",
        "dataset_path": str(DATASET_PATH),
        "sample_count": sample_count,
        "feature_names": FEATURE_NAMES,
        "trained_modality_groups": sorted(set(modality_groups.tolist())),
        "supported_modality_groups": supported_modality_groups,
        "deployment_quality_gate": {
            "minimum_accuracy": MIN_DEPLOYMENT_ACCURACY,
            "minimum_roc_auc": MIN_DEPLOYMENT_ROC_AUC,
        },
        "validation_accuracy": float(accuracy_score(y_val, preds)),
        "validation_f1": float(f1_score(y_val, preds, zero_division=0)),
        "validation_roc_auc": float(roc_auc_score(y_val, probs)),
        "validation_brier_score": float(brier_score_loss(y_val, probs)),
        "validation_log_loss": float(log_loss(y_val, probs)),
        "deployment_blend_metrics": {
            "xgboost_weight": xgboost_weight,
            "base_weight": 1.0 - xgboost_weight,
            "accuracy": float(accuracy_score(y_val, blended_preds)),
            "f1": float(f1_score(y_val, blended_preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_val, blended_probs)),
            "brier_score": float(brier_score_loss(y_val, blended_probs)),
            "log_loss": float(log_loss(y_val, blended_probs)),
        },
        "base_signal_metrics": {
            "accuracy": float(accuracy_score(y_val, base_preds)),
            "f1": float(f1_score(y_val, base_preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_val, base_val)),
            "brier_score": float(brier_score_loss(y_val, base_val)),
        },
        "per_modality_metrics": per_modality,
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
