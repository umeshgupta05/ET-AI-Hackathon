"""
Vision Agent — Counterfeit & Deepfake Currency Detection.

Orchestrates the full vision pipeline:
YOLOv8 detect → crop regions → EfficientNet classify →
ELA+FFT+NPR forensics → Grad-CAM explain → Kimi multimodal reasoning

8 AI techniques in one agent:
1. YOLOv8 (object detection/segmentation)
2. EfficientNet-B0 + contrastive learning (classification)
3. NPR (Neighboring Pixel Relationship) analysis
4. ELA (Error Level Analysis)
5. FFT (frequency-domain print artifact detection)
6. Grad-CAM (attention visualization)
7. Kimi K2.5 (multimodal vision-language reasoning)
8. CLIP-based deepfake document detection (when applicable)
"""

import base64
import logging
from typing import Optional

from core.localization import model_language_instruction, normalize_language

import cv2
import numpy as np

from models.vision.detector import get_currency_detector
from models.vision.classifier import get_forgery_classifier
from models.vision.forensics import get_forensic_analyzer
from models.vision.explainability import get_explainability_engine
from models.vision.clip_scorer import get_clip_scorer
from models.vision.currency_features import (
    inspect_security_features,
    validate_currency_candidate,
)
from models.nlp.llm_client import get_llm_client

logger = logging.getLogger(__name__)


class VisionAgent:
    """
    Full counterfeit & deepfake currency detection agent.

    Pipeline:
    1. YOLOv8 detects the note and crops regions of interest
    2. EfficientNet-B0 classifies each region (genuine vs. counterfeit)
    3. Forensic analysis (ELA + FFT + NPR) provides supporting signals
    4. Grad-CAM generates attention heatmap showing model focus areas
    5. Kimi K2.5 reasons about the image in natural language
    6. All signals fused into a single verdict with per-region attribution
    """

    def __init__(self):
        self._detector = get_currency_detector()
        self._classifier = get_forgery_classifier()
        self._forensics = get_forensic_analyzer()
        self._explainability = get_explainability_engine()
        self._clip = get_clip_scorer()
        self._llm = get_llm_client()
        self._ocr_reader = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all vision sub-models."""
        if self._initialized:
            return
        logger.info(" Initializing Vision Agent...")
        await self._detector.initialize()
        await self._classifier.initialize()
        await self._forensics.initialize()
        await self._explainability.initialize()
        await self._clip.initialize()
        
        # Initialize lightweight local OCR
        import easyocr
        import torch
        gpu = torch.cuda.is_available()
        self._ocr_reader = easyocr.Reader(['en'], gpu=gpu)
        
        self._initialized = True
        logger.info(" Vision Agent ready (9 AI techniques incl. CLIP and local EasyOCR)")

    async def analyze(
        self,
        image_bytes: bytes,
        language: str = "en",
        context: Optional[dict] = None,
    ) -> dict:
        """
        Full analysis pipeline for a currency note image.

        Args:
        image_bytes: Raw image bytes (JPEG, PNG, etc.)

        Returns structured verdict with per-region scores, forensics,
        attention maps, and natural language explanation.
        """
        if not self._initialized:
            await self.initialize()

        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")

        logger.info(
            f" Vision Agent analyzing image ({image.shape[1]}x{image.shape[0]})"
        )

        # Step 1: YOLOv8 detection + region extraction
        detection = self._detector.detect_currency(image)
        note_crop = detection.get("note_crop", image)

        # CLIP and geometry jointly establish that the input is plausibly a
        # banknote. Generic object detections are never accepted as currency.
        capture_context = context or {}
        capture_mode = str(capture_context.get("capture_mode", "rgb")).lower().strip()
        clip_result = self._clip.score(note_crop)
        
        # Fast local OCR check
        try:
            text_results = self._ocr_reader.readtext(note_crop, detail=0, paragraph=True)
            ocr_text = " ".join(text_results).upper()
            ocr_is_fake = any(word in ocr_text for word in ["400", "KALPANIK", "SPECIMEN", "CHURAN", "CHILDREN", "TESTING"])
        except Exception as e:
            logger.warning(f"Local OCR failed: {e}")
            ocr_is_fake = False
            
        candidate = validate_currency_candidate(
            image, note_crop, detection, clip_result, capture_mode=capture_mode
        )
        if not candidate["is_currency_candidate"]:
            return {
                "agent": "vision",
                "verdict": "not_currency",
                "model_confidence": 0.0,
                "input_rejected": True,
                "rejection_code": "currency_candidate_not_found",
                "candidate_validation": candidate,
                "detection": {
                    "note_detected": False,
                    "detection_confidence": detection["confidence"],
                    "detector_type": detection.get("detector_type"),
                    "note_dimensions": detection.get("note_dimensions"),
                },
                "clip": clip_result,
                "response_language": normalize_language(language),
                "explanation": "The uploaded image was rejected because a plausible currency note could not be verified.",
                "techniques_used": [
                    "Currency candidate validation",
                    "CLIP zero-shot vision-language scoring",
                    "Geometric document validation",
                ],
            }

        regions = detection.get("regions", {})
        security_features = inspect_security_features(
            regions,
            capture_mode=capture_mode,
            expected_denomination=capture_context.get("denomination"),
            supplied_serial=capture_context.get("serial_number"),
            supplied_microtext=capture_context.get("microtext_ocr"),
            machine_signals=capture_context,
        )
        if capture_mode in {"uv", "ir", "transmitted"}:
            return {
                "agent": "vision",
                "verdict": "sensor_evidence_only",
                "model_confidence": 0.0,
                "sensor_capture_only": True,
                "detection": {
                    "note_detected": detection["detected"],
                    "detection_confidence": detection["confidence"],
                    "detector_type": detection.get("detector_type"),
                    "note_dimensions": detection.get("note_dimensions"),
                },
                "candidate_validation": candidate,
                "security_features": security_features,
                "classification": {"model_available": False, "skipped_reason": f"{capture_mode}_is_not_rgb_model_input"},
                "clip": clip_result,
                "response_language": normalize_language(language),
                "explanation": f"{capture_mode.upper()} capture recorded as sensor evidence; RGB authenticity classification was intentionally skipped.",
                "techniques_used": [
                    "Currency candidate validation",
                    "Controlled-lane sensor evidence",
                    "RBI feature specification checks",
                ],
            }

        # Step 2: EfficientNet-B0 classification per region
        classification = self._classifier.classify_all_regions(regions)

        # Step 3: Forensic analysis (ELA + FFT + NPR)
        forensics = self._forensics.analyze(note_crop)

        # Step 4: Grad-CAM explainability
        gradcam_result = None
        attention_overlay = None
        try:
            if self._classifier._backbone is not None:
                gradcam_result = self._explainability.generate_gradcam(
                    model=self._classifier._backbone,
                    image=note_crop,
                    target_class=1,  # counterfeit class
                    is_vit=False,
                )
                # Generate annotated overlay with region scores
                if classification.get("region_scores"):
                    attention_overlay = self._explainability.generate_attention_overlay(
                        note_crop, classification["region_scores"]
                    )
        except Exception as e:
            logger.warning(f"Grad-CAM generation failed: {e}")

        # Step 5: Kimi K2.5 multimodal reasoning (if API available)
        multimodal_reasoning = await self._get_multimodal_reasoning(
            image_bytes, classification, forensics, language=language
        )

        # Encode visualizations as base64 for frontend
        attention_map_b64 = None
        overlay_b64 = None

        if gradcam_result and gradcam_result.get("overlay") is not None:
            _, buffer = cv2.imencode(".png", gradcam_result["overlay"])
            attention_map_b64 = base64.b64encode(buffer).decode("utf-8")

        if attention_overlay is not None:
            _, buffer = cv2.imencode(".png", attention_overlay)
            overlay_b64 = base64.b64encode(buffer).decode("utf-8")

        # Fuse all scores into final verdict
        classifier_score = classification.get("fused_counterfeit_score", 0.5)
        forensic_score = forensics.get("fused_forensic_score", 0.5)
        clip_score = clip_result.get("risk_score", 0.5)

        # Conservative weighted fusion: CLIP is useful, but not allowed to dominate.
        if clip_result.get("available"):
            fused_score = classifier_score * 0.50 + forensic_score * 0.35 + clip_score * 0.15
        else:
            fused_score = classifier_score * 0.60 + forensic_score * 0.40

        # Local OCR strict denominator check
        if locals().get("ocr_is_fake", False):
            fused_score = 0.99
            verdict = "likely_counterfeit"
            logger.warning("Local EasyOCR triggered strict denomination veto. Forcing counterfeit verdict.")
        elif fused_score > 0.65:
            verdict = "likely_counterfeit"
        elif fused_score < 0.35:
            verdict = "likely_genuine"
        else:
            verdict = "uncertain"

        return {
            "agent": "vision",
            "verdict": verdict,
            "model_confidence": round(fused_score, 4),
            "detection": {
                "note_detected": detection["detected"],
                "detection_confidence": detection["confidence"],
                "detector_type": detection.get("detector_type"),
                "note_dimensions": detection.get("note_dimensions"),
            },
            "candidate_validation": candidate,
            "security_features": security_features,
            "classification": {
                "model_available": classification.get("model_available", False),
                "genuine_score": classification.get("fused_genuine_score", 0.5),
                "counterfeit_score": classification.get("fused_counterfeit_score", 0.5),
                "verdict": classification.get("verdict", "unavailable"),
                "cross_region_attention": classification.get("cross_region_attention", False),
            },
            "region_scores": classification.get("region_scores", {}),
            "suspicious_regions": classification.get("suspicious_regions", []),
            "forensics": {
                "ela_score": forensics["ela"]["anomaly_score"],
                "fft_score": forensics["fft"]["print_artifact_score"],
                "npr_score": forensics["npr"]["synthetic_score"],
                "fused_forensic_score": forensics["fused_forensic_score"],
            },
            "clip": clip_result,
            "attention_map_base64": attention_map_b64,
            "annotated_overlay_base64": overlay_b64,
            "multimodal_reasoning": multimodal_reasoning,
            "groq_multimodal_reasoning": multimodal_reasoning,
            "maverick_reasoning": multimodal_reasoning,  # backward-compatible response alias
            "response_language": normalize_language(language),
            "explanation": self._generate_explanation(
                classification, forensics, multimodal_reasoning
            ),
            "techniques_used": [
                "YOLOv8 (object detection)",
                "Currency candidate validation",
                "Security feature inspection (microprint/thread/serial/watermark/UV)",
                *(["EfficientNet-B0 (forgery classification)", "Contrastive Learning (SimCLR-style)"] if classification.get("model_available") else []),
                "Error Level Analysis (ELA)",
                "FFT Frequency Analysis",
                "Neighboring Pixel Relationship (NPR)",
                "CLIP zero-shot vision-language scoring",
                "Grad-CAM (explainability)",
                "Kimi K2.5 (multimodal reasoning)",
            ],
        }

    async def _get_multimodal_reasoning(
        self,
        image_bytes: bytes,
        classification: dict,
        forensics: dict,
        language: str = "en",
    ) -> Optional[str]:
        """
        Use Kimi K2.5 to reason about the currency image.
        Kimi natively processes images, providing multimodal AI reasoning.
        """
        try:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            prompt = (
                f"Analyze this currency note image for authenticity. "
                f"Our vision model reports:\n"
                f"- Classifier verdict: {classification.get('verdict', 'unknown')}\n"
                f"- Suspicious regions: {classification.get('suspicious_regions', [])}\n"
                f"- Forensic ELA score: {forensics.get('ela', {}).get('anomaly_score', 'N/A')}\n"
                f"- Forensic NPR score: {forensics.get('npr', {}).get('synthetic_score', 'N/A')}\n\n"
                f"Based on what you can see in the image and these model outputs, "
                f"provide a brief expert analysis of the note's authenticity. "
                f"Focus on visible security features, print quality, and any anomalies.\n\n"
                f"{model_language_instruction(language)}"
            )

            result = await self._llm.analyze_image(
                image_base64=image_b64,
                prompt=prompt,
                temperature=0.2,
                max_tokens=512,
            )
            return result.get("content", "")

        except Exception as e:
            logger.warning(f"Kimi multimodal reasoning failed: {e}")
            return None

    def _generate_explanation(
        self, classification: dict, forensics: dict, multimodal_reasoning: Optional[str]
    ) -> str:
        """Generate human-readable explanation of the verdict."""
        parts = []

        # Suspicious regions
        suspicious = classification.get("suspicious_regions", [])
        if suspicious:
            parts.append(
                f"Model flagged {len(suspicious)} suspicious region(s): "
                f"{', '.join(suspicious)}."
            )

        # Forensic findings
        ela_score = forensics.get("ela", {}).get("anomaly_score", 0)
        npr_score = forensics.get("npr", {}).get("synthetic_score", 0)

        if ela_score > 0.6:
            parts.append(
                "Error Level Analysis detected potential digital manipulation."
            )
        if npr_score > 0.6:
            parts.append("NPR analysis suggests possible synthetic/printed origin.")

        # Multimodal reasoning
        if multimodal_reasoning:
            parts.append(f"AI vision analysis: {multimodal_reasoning[:200]}")

        if not parts:
            parts.append("Analysis complete. No strong indicators found.")

        return " ".join(parts)

    def get_stats(self) -> dict:
        return {
            "agent": "vision",
            "status": "ready" if self._initialized else "not_initialized",
            "techniques": 8,
            "sub_models": {
                "detector": self._detector.get_stats(),
                "classifier": self._classifier.get_stats(),
                "forensics": "ELA + FFT + NPR",
                "clip": self._clip.get_stats(),
                "explainability": self._explainability.get_stats(),
            },
        }
