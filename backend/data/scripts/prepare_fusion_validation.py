"""Export labelled, held-out base-model predictions for fusion training."""

import asyncio
import hashlib
import io
import json
import random
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import requests
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agents.ensemble import FEATURE_NAMES
from models.nlp.text_classifier import TextClassifier
from models.vision.classifier import HybridForgeryClassifier
from models.vision.clip_scorer import CLIPVisionScorer
from models.vision.detector import CurrencyDetector
from models.vision.forensics import ForensicAnalyzer

from train_vision_classifier import SEED, load_dataset


UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
UCI_DOI = "10.24432/C5CC84"
UCI_LICENSE = "CC BY 4.0"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "training" / "fusion_validation.jsonl"


def neutral_features() -> dict[str, float]:
    return {
        "vision_score": 0.5,
        "vision_forensic_score": 0.5,
        "vision_clip_score": 0.5,
        "speech_spoof_score": 0.5,
        "nlp_score": 0.5,
        "graph_score": 0.0,
        "has_vision": 0.0,
        "has_speech": 0.0,
        "has_nlp": 0.0,
        "has_graph": 0.0,
        "modality_image": 0.0,
        "modality_audio": 0.0,
        "modality_text": 0.0,
    }


def load_uci_sms() -> tuple[list[tuple[str, int]], str]:
    response = requests.get(UCI_URL, timeout=60)
    response.raise_for_status()
    source_sha256 = hashlib.sha256(response.content).hexdigest()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        content = archive.read("SMSSpamCollection").decode("utf-8", errors="replace")
    deduplicated = {}
    for line in content.splitlines():
        label, separator, text = line.partition("\t")
        if separator and label in {"ham", "spam"} and text.strip():
            deduplicated.setdefault(text.strip(), 1 if label == "spam" else 0)
    rows = list(deduplicated.items())
    spam = [row for row in rows if row[1] == 1]
    ham = [row for row in rows if row[1] == 0]
    random.Random(SEED).shuffle(ham)
    return spam + ham[: len(spam)], source_sha256


async def text_rows() -> list[dict]:
    samples, source_sha256 = load_uci_sms()
    classifier = TextClassifier()
    await classifier.initialize()
    texts = [text for text, _ in samples]
    outputs = classifier._binary_classifier(texts, batch_size=32, top_k=None, truncation=True)
    rows = []
    for index, ((text, label), output) in enumerate(zip(samples, outputs)):
        scores = {item["label"]: float(item["score"]) for item in output}
        fraud_probability = classifier._extract_fraud_probability(scores)
        features = neutral_features()
        features.update({"nlp_score": fraud_probability, "has_nlp": 1.0, "modality_text": 1.0})
        rows.append(
            {
                **{name: round(float(features[name]), 6) for name in FEATURE_NAMES},
                "label": label,
                "source": "UCI SMS Spam Collection",
                "source_reference": UCI_DOI,
                "source_license": UCI_LICENSE,
                "source_sha256": source_sha256,
                "sample_id": f"uci_sms_{index:04d}",
                "modality_group": "text",
            }
        )
    return rows


async def image_rows() -> list[dict]:
    paths, labels, groups, provenance = load_dataset()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    _, validation_indices = next(splitter.split(paths, labels, groups=groups))

    classifier = HybridForgeryClassifier()
    detector = CurrencyDetector()
    forensics = ForensicAnalyzer()
    clip = CLIPVisionScorer()
    await classifier.initialize()
    await detector.initialize()
    await forensics.initialize()
    await clip.initialize()
    if not classifier._trained_weights_loaded:
        raise RuntimeError("Train the verified currency classifier before exporting fusion predictions")

    rows = []
    for completed, index in enumerate(validation_indices, start=1):
        image = cv2.imread(paths[index])
        if image is None:
            raise RuntimeError(f"Could not read {paths[index]}")
        detection = detector.detect_currency(image)
        note_crop = detection.get("note_crop", image)
        classification = classifier.classify_all_regions(detection.get("regions", {}))
        forensic_result = forensics.analyze(note_crop)
        clip_result = clip.score(note_crop)
        classifier_score = classification["fused_counterfeit_score"]
        if clip_result.get("available"):
            vision_score = (
                classifier_score * 0.50
                + forensic_result["fused_forensic_score"] * 0.35
                + clip_result["risk_score"] * 0.15
            )
        else:
            vision_score = (
                classifier_score * 0.60
                + forensic_result["fused_forensic_score"] * 0.40
            )
        features = neutral_features()
        features.update(
            {
                "vision_score": vision_score,
                "vision_forensic_score": forensic_result["fused_forensic_score"],
                "vision_clip_score": clip_result["risk_score"],
                "has_vision": 1.0,
                "modality_image": 1.0,
            }
        )
        rows.append(
            {
                **{name: round(float(features[name]), 6) for name in FEATURE_NAMES},
                "label": int(labels[index]),
                "source": provenance["source_dataset"],
                "source_reference": provenance["source_url"],
                "source_license": provenance["license"],
                "sample_id": Path(paths[index]).name,
                "modality_group": "image",
            }
        )
        if completed % 20 == 0 or completed == len(validation_indices):
            print(f"Exported image features {completed}/{len(validation_indices)}")
    return rows


async def main() -> None:
    print("Exporting independent UCI SMS predictions...")
    rows = await text_rows()
    print("Exporting held-out currency predictions...")
    rows.extend(await image_rows())
    OUTPUT_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    labels = [row["label"] for row in rows]
    print(
        f"Wrote {len(rows)} rows to {OUTPUT_PATH}: "
        f"{labels.count(0)} legitimate/genuine, {labels.count(1)} scam/counterfeit"
    )


if __name__ == "__main__":
    asyncio.run(main())
