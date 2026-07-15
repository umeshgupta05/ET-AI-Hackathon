"""Fast integrity checks for datasets and ensemble feature contracts."""

import asyncio
import json
from pathlib import Path

from agents.ensemble import XGBoostFusion
from agents.graph_agent import GraphAgent


ROOT = Path(__file__).parent


def test_currency_manifest() -> None:
    manifest_path = ROOT / "data" / "training" / "currency" / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["records"]
    assert len(records) >= 500
    assert len({record["sha256"] for record in records}) == len(records)
    assert {record["label"] for record in records} == {"genuine", "counterfeit"}
    assert all((manifest_path.parent / record["path"]).is_file() for record in records)
    assert "not RBI" in manifest["label_assurance"]


def test_text_dataset_groups() -> None:
    dataset_path = ROOT / "data" / "training" / "scam_detection_dataset.json"
    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert len(records) >= 200
    assert sum(record["label"] == 1 for record in records) == sum(
        record["label"] == 0 for record in records
    )
    assert all(record.get("template_group") for record in records)
    assert len({record["text"] for record in records}) == len(records)


def test_graph_scale() -> None:
    agent = GraphAgent()
    asyncio.run(agent.initialize())
    assert agent._graph.number_of_nodes() >= 50
    assert agent._graph.number_of_edges() >= 50


def test_ensemble_feature_semantics() -> None:
    ensemble = XGBoostFusion()
    features = ensemble.extract_features(
        {
            "modality": "text",
            "nlp_result": {
                "fused_confidence": 0.91,
                "text_classifier_score": 0.72,
                "text_binary_score": 0.63,
            },
        }
    )
    assert features["nlp_score"] == 0.63
    assert features["modality_text"] == 1.0
    assert features["has_nlp"] == 1.0
    assert features["has_vision"] == 0.0

    prediction = ensemble.predict(
        {
            "modality": "text",
            "nlp_result": {
                "fused_confidence": 0.91,
                "text_classifier_score": 0.72,
                "text_binary_score": 0.63,
            },
        },
        fallback_score=0.91,
    )
    assert prediction["method"] == "weighted_fallback"
    assert prediction["error"] == "unsupported_modality_signature"


def main() -> None:
    test_currency_manifest()
    test_text_dataset_groups()
    test_graph_scale()
    test_ensemble_feature_semantics()
    print("ML contract checks passed")


if __name__ == "__main__":
    main()
