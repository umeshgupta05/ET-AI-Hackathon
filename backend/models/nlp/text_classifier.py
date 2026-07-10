"""
DistilBERT Text Classifier -- Scam transcript classification.

Independent signal from the LLM reasoning agent.
Uses zero-shot classification with a pretrained NLI model as a proxy for
a fine-tuned classifier, plus linguistic scam-marker features.
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from config import config

logger = logging.getLogger(__name__)


SCAM_LABELS = [
    "digital arrest scam",
    "financial fraud",
    "government impersonation",
    "phishing attempt",
    "legitimate conversation",
    "customer service call",
]

BINARY_LABELS = [
    "fraudulent scam communication",
    "legitimate normal conversation",
]


class TextClassifier:
    """
    DistilBERT/NLI-based scam text classifier.

    Strategy: use zero-shot NLI classification as the primary approach and
    combine it with simple linguistic markers used as a supporting signal.
    """

    def __init__(self):
        self._classifier = None
        self._binary_classifier = None
        self._using_finetuned_binary = False
        self._initialized = False

    async def initialize(self) -> None:
        """Load the text classification models."""
        if self._initialized:
            return

        logger.info(" Initializing text classifier...")
        try:
            self._classifier = pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-distilroberta-base",
                device=-1,
            )
            logger.info(
                "✅ Zero-shot classifier loaded (cross-encoder/nli-distilroberta-base)"
            )

            # Try loading our fine-tuned scam classifier first
            from pathlib import Path

            finetuned_path = (
                Path(__file__).resolve().parent.parent.parent
                / "data"
                / "trained_models"
                / "scam_classifier"
                / "final"
            )
            if (
                finetuned_path.exists()
                and (finetuned_path / "model.safetensors").exists()
            ):
                try:
                    self._binary_classifier = pipeline(
                        "text-classification",
                        model=str(finetuned_path),
                        device=-1,
                    )
                    logger.info(
                        f"✅ Fine-tuned scam classifier loaded from {finetuned_path}"
                    )
                    self._using_finetuned_binary = True
                except Exception as e:
                    logger.warning(
                        f"Could not load fine-tuned model: {e}, using pretrained"
                    )
                    self._binary_classifier = pipeline(
                        "text-classification",
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        device=-1,
                    )
            else:
                self._binary_classifier = pipeline(
                    "text-classification",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1,
                )
                logger.info("✅ Sentiment classifier loaded (distilbert-base-uncased)")

            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize text classifier: {e}")
            raise

    def classify_scam(self, text: str) -> dict:
        """Classify text against known scam categories."""
        if not self._initialized or not self._classifier:
            raise RuntimeError("Text classifier not initialized")

        result = self._classifier(text, SCAM_LABELS, multi_label=True)
        all_scores = dict(zip(result["labels"], result["scores"]))

        scam_categories = [
            label
            for label in SCAM_LABELS
            if label not in {"legitimate conversation", "customer service call"}
        ]
        scam_scores = [all_scores.get(category, 0.0) for category in scam_categories]
        scam_score = max(scam_scores) if scam_scores else 0.0

        top_category = result["labels"][0]
        top_confidence = result["scores"][0]

        if self._using_finetuned_binary and self._binary_classifier is not None:
            binary_result = self._binary_classifier(text, top_k=None)
            if binary_result and isinstance(binary_result[0], list):
                binary_result = binary_result[0]
            binary_scores = {
                item["label"]: float(item["score"])
                for item in binary_result
                if isinstance(item, dict)
            }
            fraud_prob = self._extract_fraud_probability(binary_scores)
        else:
            binary_result = self._classifier(text, BINARY_LABELS)
            binary_scores = dict(zip(binary_result["labels"], binary_result["scores"]))
            fraud_prob = binary_scores.get("fraudulent scam communication", 0.0)

        features = self._extract_linguistic_features(text)
        combined_score = (
            scam_score * 0.5 + fraud_prob * 0.3 + features["threat_density"] * 0.2
        )

        return {
            "scam_score": round(combined_score, 4),
            "category": top_category.replace(" ", "_"),
            "category_confidence": round(top_confidence, 4),
            "binary_fraud_score": round(fraud_prob, 4),
            "all_scores": {k: round(v, 4) for k, v in all_scores.items()},
            "binary_scores": {k: round(v, 4) for k, v in binary_scores.items()},
            "features": features,
        }

    def _extract_fraud_probability(self, binary_scores: dict[str, float]) -> float:
        """Map model labels to the training convention: label 1 = scam."""
        for label, score in binary_scores.items():
            normalized = label.lower()
            if (
                normalized in {"label_1", "1"}
                or "scam" in normalized
                or "fraud" in normalized
            ):
                return float(score)

        label_zero = next(
            (
                score
                for label, score in binary_scores.items()
                if label.lower() in {"label_0", "0"}
            ),
            None,
        )
        if label_zero is not None:
            return 1.0 - float(label_zero)

        return 0.0

    def _extract_linguistic_features(self, text: str) -> dict:
        """Extract scam-indicative linguistic features."""
        text_lower = text.lower()

        urgency_words = [
            "immediately",
            "right now",
            "urgent",
            "hurry",
            "quickly",
            "within the hour",
            "time is running out",
            "last chance",
            "act now",
            "don't delay",
        ]
        authority_words = [
            "cbi",
            "police",
            "officer",
            "enforcement directorate",
            "customs",
            "trai",
            "rbi",
            "court",
            "warrant",
            "arrest",
            "investigation",
            "case number",
            "fir",
            "section",
            "magistrate",
            "judge",
            "superintendent",
        ]
        secrecy_words = [
            "don't tell",
            "do not tell",
            "confidential",
            "secret",
            "sub-judice",
            "classified",
            "not to discuss",
            "keep quiet",
            "stay on the call",
            "don't disconnect",
        ]
        financial_words = [
            "transfer",
            "account",
            "payment",
            "deposit",
            "safe custody",
            "verification amount",
            "clearance fee",
            "upi",
            "neft",
            "bank details",
            "send money",
            "pay",
        ]
        threat_words = [
            "arrested",
            "jail",
            "prison",
            "penalty",
            "fine",
            "suspended",
            "cancelled",
            "blacklisted",
            "legal action",
            "consequences",
            "family",
            "affected",
        ]

        urgency_count = sum(1 for word in urgency_words if word in text_lower)
        authority_count = sum(1 for word in authority_words if word in text_lower)
        secrecy_count = sum(1 for word in secrecy_words if word in text_lower)
        financial_count = sum(1 for word in financial_words if word in text_lower)
        threat_count = sum(1 for word in threat_words if word in text_lower)

        total_markers = (
            urgency_count
            + authority_count
            + secrecy_count
            + financial_count
            + threat_count
        )
        word_count = max(len(text.split()), 1)
        threat_density = min(total_markers / (word_count / 50), 1.0)

        return {
            "urgency_count": urgency_count,
            "authority_count": authority_count,
            "secrecy_count": secrecy_count,
            "financial_pressure_count": financial_count,
            "threat_count": threat_count,
            "total_markers": total_markers,
            "threat_density": round(threat_density, 4),
        }

    def classify_batch(self, texts: list[str]) -> list[dict]:
        """Classify multiple texts efficiently."""
        return [self.classify_scam(text) for text in texts]

    def get_stats(self) -> dict:
        """Return classifier statistics."""
        return {
            "status": "ready" if self._initialized else "not_initialized",
            "model": "cross-encoder/nli-distilroberta-base",
            "technique": "zero-shot NLI classification + linguistic feature extraction",
            "binary_classifier": "fine-tuned scam classifier"
            if self._using_finetuned_binary
            else "zero-shot binary fallback",
            "labels": SCAM_LABELS,
        }


_classifier: Optional[TextClassifier] = None


def get_text_classifier() -> TextClassifier:
    """Get or create the singleton text classifier."""
    global _classifier
    if _classifier is None:
        _classifier = TextClassifier()
    return _classifier
