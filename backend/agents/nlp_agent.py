"""
NLP/LLM Agent — Scam Detection with Agentic Reasoning.

Uses Kimi K2.5 (via OpenRouter) for multi-step agentic reasoning:
- Hierarchical multi-role CoT prompting (Investigator → Policy Checker → Risk Assessor)
- RAG-grounded verdicts (retrieval from known scam pattern corpus)
- DistilBERT text classifier as independent voting signal

This is the core AI reasoning engine — not a rules engine, not a simple classifier.
"""

import json
import logging
from typing import Optional

from core.localization import localized_fallback, model_language_instruction, normalize_language

from models.nlp.llm_client import get_llm_client
from models.nlp.rag_engine import get_rag_engine
from models.nlp.text_classifier import get_text_classifier
from config import config

logger = logging.getLogger(__name__)

# ─── Hierarchical Multi-Role CoT System Prompts ─────────────────────────

INVESTIGATOR_PROMPT = """You are the INVESTIGATOR agent in a multi-agent scam detection system.
Your role: Analyze the conversation/transcript for suspicious patterns.

Return at most four of the strongest suspicious elements. Keep every string concise.
For each suspicious element, output:
1. What pattern you found (e.g., "authority impersonation", "urgency creation")
2. The specific text evidence
3. Your confidence that this is a scam indicator (0.0-1.0)

Think step by step. Be specific about which scam tactics are being used.

Output ONLY valid JSON:
{
"findings": [
{
"pattern": "string",
"evidence": "string (quote from text)",
"confidence": float,
"severity": "low|medium|high|critical"
}
],
"preliminary_assessment": "string",
"scam_likelihood": float
}"""

POLICY_CHECKER_PROMPT = """You are the POLICY CHECKER agent in a multi-agent scam detection system.
Your role: Verify the Investigator's findings against known scam patterns from our database.

You will receive:
1. The Investigator's findings
2. Retrieved known scam patterns from our RAG database

Evaluate at most four findings and keep every string concise. For each finding, determine:
- Does it match a known scam pattern? (cite which one)
- Is the evidence strong enough to support the finding?
- Could this be a false positive? (legitimate scenario explanation)

Output ONLY valid JSON:
{
"verified_findings": [
{
"pattern": "string",
"matched_known_pattern": "string or null",
"match_confidence": float,
"false_positive_risk": "low|medium|high",
"reasoning": "string"
}
],
"pattern_match_summary": "string",
"adjusted_scam_likelihood": float
}"""

RISK_ASSESSOR_PROMPT = """You are the RISK ASSESSOR agent in a multi-agent scam detection system.
Your role: Produce the final calibrated verdict based on all evidence.

You will receive:
1. Investigator's findings
2. Policy Checker's verification
3. Text classifier's independent score
4. RAG semantic deviation score

Keep the reasoning to two short sentences and return at most four indicators.
Produce a final verdict considering:
- Strength of evidence (multiple independent signals agreeing = higher confidence)
- False positive risk (citizen-facing tools MUST have very low false positive rate)
- Severity assessment

Output ONLY valid JSON:
{
"verdict": "active_scam_high_confidence|likely_scam|suspicious|likely_legitimate|legitimate",
"confidence": float,
"risk_level": "critical|high|medium|low|none",
"reasoning": "string (2-3 sentence explanation)",
"key_indicators": ["string"],
"recommended_action": "string"
}"""


class NLPAgent:
    """
    Multi-model NLP agent for scam detection.

    Uses 3 independent AI models voting:
    1. Kimi K2.5 (agentic LLM reasoning via OpenRouter)
    2. DistilBERT (text classification)
    3. RAG (semantic similarity + deviation scoring)

    Plus hierarchical multi-role CoT:
    - Investigator: finds suspicious patterns
    - Policy Checker: verifies against known patterns (RAG)
    - Risk Assessor: produces final calibrated verdict
    """

    def __init__(self):
        self._llm = get_llm_client()
        self._rag = get_rag_engine()
        self._text_classifier = get_text_classifier()
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        logger.info(" Initializing NLP Agent...")
        await self._rag.initialize()
        await self._text_classifier.initialize()
        self._initialized = True
        logger.info(" NLP Agent ready (Kimi K2.5 + DistilBERT + Hybrid RAG)")

    async def analyze(self, text: str, context: Optional[dict] = None) -> dict:
        """
        Full NLP analysis pipeline for scam detection.

        Args:
        text: The transcript or message to analyze
        context: Optional context (e.g., audio spoof score)

        Returns comprehensive verdict with multi-agent reasoning trace.
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f" NLP Agent analyzing text ({len(text)} chars)...")

        # ─── Independent Signal 1: DistilBERT Text Classifier ────────
        text_classification = self._text_classifier.classify_scam(text)
        logger.info(f" Text classifier score: {text_classification['scam_score']:.3f}")

        # ─── Independent Signal 2: RAG Retrieval + Deviation ─────────
        rag_matches = self._rag.query(text, top_k=3)
        semantic_deviation = self._rag.compute_semantic_deviation(text)
        logger.info(
            f" RAG matches: {len(rag_matches)}, deviation: {semantic_deviation:.3f}"
        )

        # ─── Independent Signal 3: Kimi K2.5 Multi-Role CoT ─────────
        audio_spoof_score = context.get("audio_spoof_score", None) if context else None
        response_language = normalize_language(
            context.get("response_language") if context else "en"
        )
        multi_role = await self._run_multi_role_analysis(
            text,
            rag_matches,
            text_classification,
            semantic_deviation,
            audio_spoof_score,
            response_language,
        )
        investigator_result = multi_role["investigator"]
        policy_result = multi_role["policy_checker"]
        risk_result = multi_role["risk_assessor"]
        agent_trace = [
            {"role": "investigator", "result": investigator_result},
            {"role": "policy_checker", "result": policy_result},
            {"role": "risk_assessor", "result": risk_result},
        ]

        # ─── Fuse all signals ────────────────────────────────────────
        llm_scam_score = self._llm_scam_score(risk_result)
        fused_confidence = self._fuse_signals(
            llm_score=llm_scam_score,
            text_classifier_score=text_classification["scam_score"],
            semantic_deviation=semantic_deviation,
        )

        # Format RAG matches for output
        retrieved_patterns = [
            {
                "pattern": m["document"][:200],
                "similarity": m["similarity"],
                "category": m["category"],
                "source": m["source"],
            }
            for m in rag_matches[:3]
        ]

        return {
            "agent": "nlp",
            "verdict": risk_result.get("verdict", "uncertain"),
            "fused_confidence": round(fused_confidence, 4),
            "risk_level": risk_result.get("risk_level", "medium"),
            "agent_trace": agent_trace,
            "text_classifier_score": text_classification["scam_score"],
            "text_binary_score": text_classification.get(
                "binary_fraud_score", text_classification["scam_score"]
            ),
            "llm_scam_score": round(llm_scam_score, 4),
            "semantic_deviation": semantic_deviation,
            "retrieved_pattern_matches": retrieved_patterns,
            "linguistic_features": text_classification.get("features", {}),
            "reasoning": risk_result.get("reasoning", ""),
            "key_indicators": risk_result.get("key_indicators", []),
            "recommended_action": risk_result.get("recommended_action", ""),
            "response_language": response_language,
            "techniques_used": [
                "Kimi K2.5 (agentic LLM reasoning)",
                "Hierarchical Multi-Role CoT Prompting",
                "DistilBERT (zero-shot NLI classification)",
                "RAG (ChromaDB + sentence-transformers)",
                "Semantic Deviation Scoring",
                "Linguistic Feature Extraction",
            ],
        }

    async def _run_multi_role_analysis(
        self,
        text: str,
        rag_matches: list,
        text_classification: dict,
        semantic_deviation: float,
        audio_spoof_score: Optional[float],
        response_language: str = "en",
    ) -> dict:
        """Run the three review roles in one structured Kimi inference."""
        rag_context = [
            {
                "category": match["category"],
                "pattern": match["document"][:180],
                "similarity": round(match["similarity"], 3),
            }
            for match in rag_matches[:3]
        ]
        evidence = {
            "conversation": text,
            "known_patterns": rag_context,
            "text_classifier_score": round(text_classification["scam_score"], 3),
            "semantic_deviation": round(semantic_deviation, 3),
            "audio_spoof_score": audio_spoof_score,
        }
        system = """You are a three-role fraud review panel. Analyze the evidence once,
then return ONLY compact valid JSON with exactly these top-level keys:
{
  "investigator": {
    "findings": [{"pattern": "short string", "evidence": "short quote", "confidence": 0.0, "severity": "low|medium|high|critical"}],
    "preliminary_assessment": "short string",
    "scam_likelihood": 0.0
  },
  "policy_checker": {
    "verified_findings": [{"pattern": "short string", "matched_known_pattern": "string or null", "match_confidence": 0.0, "false_positive_risk": "low|medium|high", "reasoning": "short string"}],
    "pattern_match_summary": "short string",
    "adjusted_scam_likelihood": 0.0
  },
  "risk_assessor": {
    "verdict": "active_scam_high_confidence|likely_scam|suspicious|likely_legitimate|legitimate",
    "confidence": 0.0,
    "risk_level": "critical|high|medium|low|none",
    "reasoning": "two short sentences",
    "key_indicators": ["at most four short strings"],
    "recommended_action": "one short sentence"
  }
}
Use at most four findings. Give extra weight when independent signals agree, and avoid false positives.

""" + model_language_instruction(response_language)
        try:
            result = await self._llm.reason(
                system=system,
                user=json.dumps(evidence, ensure_ascii=False),
                json_mode=True,
                temperature=0.1,
                max_tokens=3000,
            )
            parsed = json.loads(result["content"])
            return {
                "investigator": parsed["investigator"],
                "policy_checker": parsed["policy_checker"],
                "risk_assessor": parsed["risk_assessor"],
            }
        except Exception as exc:
            logger.warning(f"Multi-role Kimi analysis failed: {exc}")
            score = min(
                max(
                    text_classification["scam_score"] * 0.6
                    + semantic_deviation * 0.4,
                    0.0,
                ),
                1.0,
            )
            if score >= 0.7:
                verdict, risk_level = "likely_scam", "high"
            elif score >= 0.45:
                verdict, risk_level = "suspicious", "medium"
            else:
                verdict, risk_level = "likely_legitimate", "low"
            return {
                "investigator": {
                    "findings": [],
                    "preliminary_assessment": "LLM review unavailable; independent signals used",
                    "scam_likelihood": score,
                },
                "policy_checker": {
                    "verified_findings": [],
                    "pattern_match_summary": "Hybrid retrieval evidence retained",
                    "adjusted_scam_likelihood": score,
                },
                "risk_assessor": {
                    "verdict": verdict,
                    "confidence": score,
                    "risk_level": risk_level,
                    "reasoning": localized_fallback(response_language, "reasoning"),
                    "key_indicators": [],
                    "recommended_action": localized_fallback(response_language, "action"),
                },
            }

    async def _run_investigator(self, text: str) -> dict:
        """Run the Investigator role — find suspicious patterns."""
        try:
            result = await self._llm.reason(
                system=INVESTIGATOR_PROMPT,
                user=f"Analyze this conversation/message for scam patterns:\n\n{text}",
                json_mode=True,
                temperature=0.2,
                max_tokens=512,
            )
            return json.loads(result["content"])
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Investigator JSON parse failed: {e}")
            return {
                "findings": [],
                "preliminary_assessment": "Analysis inconclusive",
                "scam_likelihood": 0.5,
            }
        except Exception as e:
            logger.warning(f"Investigator failed: {e}")
            return {
                "findings": [],
                "preliminary_assessment": str(e),
                "scam_likelihood": 0.5,
            }

    async def _run_policy_checker(
        self, investigator_result: dict, rag_matches: list
    ) -> dict:
        """Run the Policy Checker role — verify against known patterns."""
        try:
            rag_context = "\n".join(
                [
                    f"- [{m['category']}] {m['document'][:150]} (similarity: {m['similarity']:.2f})"
                    for m in rag_matches[:5]
                ]
            )

            user_msg = (
                f"Investigator findings:\n{json.dumps(investigator_result, indent=2)}\n\n"
                f"Known scam patterns from database:\n{rag_context}"
            )

            result = await self._llm.reason(
                system=POLICY_CHECKER_PROMPT,
                user=user_msg,
                json_mode=True,
                temperature=0.2,
                max_tokens=512,
            )
            return json.loads(result["content"])
        except (json.JSONDecodeError, KeyError):
            return {
                "verified_findings": [],
                "pattern_match_summary": "Verification inconclusive",
                "adjusted_scam_likelihood": 0.5,
            }
        except Exception as e:
            logger.warning(f"Policy checker failed: {e}")
            return {
                "verified_findings": [],
                "pattern_match_summary": str(e),
                "adjusted_scam_likelihood": 0.5,
            }

    async def _run_risk_assessor(
        self,
        investigator_result: dict,
        policy_result: dict,
        text_classification: dict,
        semantic_deviation: float,
        audio_spoof_score: Optional[float],
    ) -> dict:
        """Run the Risk Assessor role — produce final verdict."""
        try:
            evidence_summary = (
                f"Investigator findings:\n{json.dumps(investigator_result, indent=2)}\n\n"
                f"Policy verification:\n{json.dumps(policy_result, indent=2)}\n\n"
                f"Independent text classifier scam score: {text_classification['scam_score']:.3f}\n"
                f"RAG semantic deviation score: {semantic_deviation:.3f}\n"
            )
            if audio_spoof_score is not None:
                evidence_summary += (
                    f"Audio spoof detection score: {audio_spoof_score:.3f}\n"
                )

            result = await self._llm.reason(
                system=RISK_ASSESSOR_PROMPT,
                user=evidence_summary,
                json_mode=True,
                temperature=0.1,
                max_tokens=384,
            )
            return json.loads(result["content"])
        except (json.JSONDecodeError, KeyError):
            return {
                "verdict": "uncertain",
                "confidence": 0.5,
                "risk_level": "medium",
                "reasoning": "Risk assessment inconclusive",
                "key_indicators": [],
                "recommended_action": "Exercise caution",
            }
        except Exception as e:
            logger.warning(f"Risk assessor failed: {e}")
            return {
                "verdict": "uncertain",
                "confidence": 0.5,
                "risk_level": "medium",
                "reasoning": str(e),
                "key_indicators": [],
                "recommended_action": "Exercise caution",
            }

    def _fuse_signals(
        self,
        llm_score: float,
        text_classifier_score: float,
        semantic_deviation: float,
    ) -> float:
        """Fuse independent model signals into a single score."""
        # Three independent model families voting
        fused = (
            llm_score * 0.45 + text_classifier_score * 0.30 + semantic_deviation * 0.25
        )
        return min(max(fused, 0.0), 1.0)

    @staticmethod
    def _llm_scam_score(risk_result: dict) -> float:
        """Normalize confidence into probability of scam, respecting verdict direction."""
        confidence = min(max(float(risk_result.get("confidence", 0.5)), 0.0), 1.0)
        verdict = str(risk_result.get("verdict", "suspicious")).lower()
        if verdict == "legitimate":
            return min(1.0 - confidence, 0.10)
        if verdict == "likely_legitimate":
            return min(1.0 - confidence, 0.30)
        if verdict == "suspicious":
            return 0.5 + (confidence - 0.5) * 0.25
        if verdict == "active_scam_high_confidence":
            return max(confidence, 0.90)
        return max(confidence, 0.70)

    async def analyze_turn_by_turn(
        self, turns: list[str], language: str = "en"
    ) -> list[dict]:
        """
        Analyze a conversation turn by turn, building confidence trajectory.
        For live-demo: shows confidence climbing as scam patterns accumulate.
        """
        trajectory = []
        accumulated_text = ""

        for i, turn in enumerate(turns):
            accumulated_text += f"\n{turn}"
            current_text = accumulated_text.strip()
            is_final_turn = i == len(turns) - 1
            if is_final_turn:
                result = await self.analyze(
                    current_text,
                    context={"response_language": normalize_language(language)},
                )
                confidence = result["fused_confidence"]
                verdict = result["verdict"]
                reasoning = result.get("reasoning", "")[:200]
                analysis_mode = "full_multi_role"
            else:
                classification = self._text_classifier.classify_scam(current_text)
                deviation = self._rag.compute_semantic_deviation(current_text)
                confidence = min(
                    max(classification["scam_score"] * 0.70 + deviation * 0.30, 0.0),
                    1.0,
                )
                if confidence >= 0.70:
                    verdict = "likely_scam"
                elif confidence >= 0.45:
                    verdict = "suspicious"
                else:
                    verdict = "likely_legitimate"
                reasoning = localized_fallback(language, "incremental")
                analysis_mode = "incremental_local"
            trajectory.append(
                {
                    "turn": i + 1,
                    "turn_text": turn[:100],
                    "fused_confidence": round(confidence, 4),
                    "verdict": verdict,
                    "reasoning": reasoning,
                    "analysis_mode": analysis_mode,
                    "confidence_delta": (
                        confidence - trajectory[-1]["fused_confidence"]
                        if trajectory
                        else confidence
                    ),
                }
            )

        return trajectory

    def get_stats(self) -> dict:
        return {
            "agent": "nlp",
            "status": "ready" if self._initialized else "not_initialized",
            "models": 3,
            "techniques": [
                "Kimi K2.5 agentic reasoning",
                "Hierarchical Multi-Role CoT",
                "DistilBERT zero-shot NLI",
                "RAG with semantic deviation",
            ],
        }
