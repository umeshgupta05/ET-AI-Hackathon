"""Evaluate the local text classifier on the test-only Chakravyuh benchmark."""

import asyncio
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.nlp.text_classifier import TextClassifier


DATASET_ID = "ujjwalpardeshi/chakravyuh-bench-v0"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}"
VIEWER_URL = "https://datasets-server.huggingface.co/rows"
DATASET_REVISION = "143c8fe73aefca018adf642ce1b54be0f53bd11d"
RAW_URL = f"{DATASET_URL}/resolve/{DATASET_REVISION}/scenarios.jsonl"
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "trained_models"
    / "scam_classifier"
    / "benchmark_metadata.json"
)


def fetch_benchmark() -> list[dict]:
    rows = []
    try:
        offset = 0
        while True:
            response = requests.get(
                VIEWER_URL,
                params={
                    "dataset": DATASET_ID,
                    "config": "default",
                    "split": "test",
                    "offset": offset,
                    "length": 100,
                },
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            page = [item["row"] for item in payload.get("rows", [])]
            rows.extend(page)
            total = int(payload.get("num_rows_total", len(rows)))
            if not page or len(rows) >= total:
                break
            offset += len(page)
    except (requests.RequestException, ValueError, KeyError):
        response = requests.get(RAW_URL, timeout=60)
        response.raise_for_status()
        rows = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    if len(rows) < 100:
        raise RuntimeError(f"Benchmark download was unexpectedly small: {len(rows)} rows")
    return rows


def grouped_metrics(labels: np.ndarray, predictions: np.ndarray, groups: list[str]) -> dict:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        grouped_indices[str(group or "unknown")].append(index)
    results = {}
    for group, indices in sorted(grouped_indices.items()):
        truth = labels[indices]
        predicted = predictions[indices]
        results[group] = {
            "count": len(indices),
            "accuracy": float(accuracy_score(truth, predicted)),
            "f1": float(f1_score(truth, predicted, zero_division=0)),
        }
    return results


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> tuple[np.ndarray, dict]:
    """Measure an explicit binary action threshold on held-out benchmark scores."""
    predictions = (probabilities >= threshold).astype(np.int64)
    benign_mask = labels == 0
    return predictions, {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "false_positive_rate": float(predictions[benign_mask].mean()),
        "false_positive_count": int(predictions[benign_mask].sum()),
        "benign_count": int(benign_mask.sum()),
    }


async def evaluate(
    action_threshold: float,
    review_threshold: float,
    sweep_thresholds: list[float] | None = None,
) -> dict:
    if not 0.0 <= review_threshold < action_threshold <= 1.0:
        raise ValueError("review threshold must be lower than the action threshold, both between 0 and 1")
    benchmark = fetch_benchmark()
    texts = [
        "\n".join(turn.get("text", "") for turn in row.get("attack_sequence", []))
        for row in benchmark
    ]
    labels = np.asarray(
        [int(row.get("ground_truth", {}).get("is_scam", False)) for row in benchmark],
        dtype=np.int64,
    )

    classifier = TextClassifier()
    await classifier.initialize()
    outputs = classifier._binary_classifier(
        texts,
        batch_size=32,
        top_k=None,
        truncation=True,
    )
    probabilities = np.asarray(
        [
            classifier._extract_fraud_probability(
                {item["label"]: float(item["score"]) for item in output}
            )
            for output in outputs
        ],
        dtype=np.float64,
    )
    baseline_predictions, baseline_metrics = classification_metrics(labels, probabilities, 0.5)
    action_predictions, action_metrics = classification_metrics(labels, probabilities, action_threshold)
    benign_mask = labels == 0
    review_mask = (probabilities >= review_threshold) & (probabilities < action_threshold)
    review_benign_count = int((review_mask & benign_mask).sum())
    review_scam_count = int((review_mask & ~benign_mask).sum())

    metadata = {
        "benchmark": "Chakravyuh-Bench-v0",
        "dataset_id": DATASET_ID,
        "dataset_url": DATASET_URL,
        "dataset_version": "0.2.0",
        "dataset_revision": DATASET_REVISION,
        "license": "CC BY 4.0",
        "usage": "test_only_not_used_for_training",
        "sample_count": len(labels),
        "scam_count": int(labels.sum()),
        "benign_and_borderline_count": int((labels == 0).sum()),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "classifier_default_threshold_metrics": baseline_metrics,
        "operating_policy": {
            "action_threshold": action_threshold,
            "needs_review_threshold": review_threshold,
            "action_metrics": action_metrics,
            "needs_review_count": int(review_mask.sum()),
            "needs_review_benign_count": review_benign_count,
            "needs_review_scam_count": review_scam_count,
            "needs_review_benign_rate": float(review_benign_count / max(int(benign_mask.sum()), 1)),
            "disclosure": "Needs-review cases are withheld from automated action; they are not counted as cleared-safe cases.",
        },
        "per_difficulty": grouped_metrics(
            labels,
            action_predictions,
            [row.get("ground_truth", {}).get("difficulty", "unknown") for row in benchmark],
        ),
        "per_category": grouped_metrics(
            labels,
            action_predictions,
            [row.get("ground_truth", {}).get("category", "unknown") for row in benchmark],
        ),
        "per_language": grouped_metrics(
            labels,
            action_predictions,
            [row.get("metadata", {}).get("language", "unknown") for row in benchmark],
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Single-curator reconstructed scenarios rather than verbatim victim transcripts",
            "English-dominant and scam-heavy test distribution",
            "Metrics evaluate the local text classifier, not the full Kimi/RAG orchestrator",
        ],
    }
    if sweep_thresholds:
        metadata["threshold_sweep"] = [
            classification_metrics(labels, probabilities, threshold)[1]
            for threshold in sweep_thresholds
        ]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-threshold", type=float, default=0.60)
    parser.add_argument("--review-threshold", type=float, default=0.45)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Record standard operating-point candidates in benchmark metadata.",
    )
    arguments = parser.parse_args()
    thresholds = [0.07, 0.15, 0.30, 0.45, 0.60] if arguments.sweep else None
    asyncio.run(evaluate(arguments.action_threshold, arguments.review_threshold, thresholds))
