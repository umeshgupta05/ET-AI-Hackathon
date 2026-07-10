"""
Agentic Fusion Orchestrator — LangGraph StateGraph Multi-Agent System.

A production-grade multi-agent orchestrator built on LangGraph:
- StateGraph with conditional edges for dynamic routing
- Cyclic self-correction: evaluator node loops back for ambiguous cases
- Full agent trace with timing for observability
- XGBoost ensemble calibration as final node

Graph Structure:
  input_node → route_node → [vision | speech | nlp] → graph_node → fusion_node → evaluator_node → calibration_node
                                                                        ↑                    |
                                                                        └────── (re-analyze) ←┘
"""

import json
import logging
import time
from typing import Any, Optional, TypedDict, Annotated
from enum import Enum

from langgraph.graph import StateGraph, END

from models.nlp.llm_client import get_llm_client
from agents.vision_agent import VisionAgent
from agents.speech_agent import SpeechAgent
from agents.nlp_agent import NLPAgent
from agents.graph_agent import GraphAgent, get_graph_agent
from agents.calibration import CalibrationLayer, get_calibration_layer
from agents.ensemble import get_xgboost_fusion
from config import config

logger = logging.getLogger(__name__)


# ─── State Schema ────────────────────────────────────────────────────────


class InputModality(str, Enum):
    IMAGE = "image"
    AUDIO = "audio"
    TEXT = "text"
    IMAGE_TEXT = "image_text"
    AUDIO_TEXT = "audio_text"
    MULTIMODAL = "multimodal"


class FraudAnalysisState(TypedDict, total=False):
    """Shared state across all agents in the LangGraph pipeline."""

    # ─── Input ───
    text: Optional[str]
    image_bytes: Optional[bytes]
    audio_bytes: Optional[bytes]
    modality: str

    # ─── Routing ───
    agents_to_invoke: list[str]
    routing_reasoning: str

    # ─── Agent Results ───
    vision_result: Optional[dict]
    speech_result: Optional[dict]
    nlp_result: Optional[dict]
    graph_result: Optional[dict]
    transcript_text: str

    # ─── Fusion ───
    fused_score: float
    base_weighted_score: float
    verdict: str
    per_agent_scores: dict
    per_agent_weights: dict
    fusion_method: str
    xgboost_features: dict
    raw_xgboost_score: Optional[float]

    # ─── Calibration ───
    calibrated_score: float
    risk_level: str

    # ─── Trace ───
    trace: list[dict]
    start_time: float
    iteration: int


# ─── Routing Prompt ──────────────────────────────────────────────────────

ROUTING_PROMPT = """You are the ROUTING AGENT in a multi-agent fraud detection system.
Given citizen input, determine which specialist agents should be invoked.

Available agents:
1. VISION: Counterfeit currency detection, fake document detection, deepfake image detection
2. SPEECH: Voice spoofing detection, audio transcription
3. NLP: Scam script analysis, fraud pattern matching, threat assessment

Rules:
- Image of currency/document → VISION + NLP (for multimodal reasoning)
- Audio recording → SPEECH + NLP (transcribe then analyze)
- Text message/transcript → NLP only
- Screenshot of govt portal → VISION + NLP
- Mixed input → invoke ALL relevant agents

Output ONLY valid JSON:
{
"agents_to_invoke": ["vision", "speech", "nlp"],
"reasoning": "Brief explanation of routing decision",
"primary_agent": "vision|speech|nlp",
"input_type": "currency_image|document_screenshot|audio_call|text_message|mixed"
}"""


# ─── Node Functions ──────────────────────────────────────────────────────


def input_node(state: FraudAnalysisState) -> dict:
    """Detect input modality and initialize trace."""
    text = state.get("text")
    image_bytes = state.get("image_bytes")
    audio_bytes = state.get("audio_bytes")

    has_text = bool(text and str(text).strip())
    has_image = bool(image_bytes)
    has_audio = bool(audio_bytes)

    if has_image and has_audio and has_text:
        modality = InputModality.MULTIMODAL
    elif has_image and has_text:
        modality = InputModality.IMAGE_TEXT
    elif has_audio and has_text:
        modality = InputModality.AUDIO_TEXT
    elif has_image:
        modality = InputModality.IMAGE
    elif has_audio:
        modality = InputModality.AUDIO
    else:
        modality = InputModality.TEXT

    start_time = time.time()
    return {
        "modality": modality.value,
        "start_time": start_time,
        "iteration": 0,
        "transcript_text": text or "",
        "trace": [{
            "step": "input_detection",
            "modality": modality.value,
            "timestamp": 0.0,
        }],
    }


# ─── Orchestrator Class ─────────────────────────────────────────────────


class FusionOrchestrator:
    """
    LangGraph-powered multi-agent fraud detection orchestrator.

    Key features:
    - StateGraph with conditional edges for dynamic routing
    - Cyclic self-correction for ambiguous cases (0.4–0.6 confidence)
    - Per-agent isolation: each agent reads/writes only its state slice
    - Full observability via trace log
    """

    def __init__(self):
        self._llm = get_llm_client()
        self._vision_agent = VisionAgent()
        self._speech_agent = SpeechAgent()
        self._nlp_agent = NLPAgent()
        self._graph_agent = get_graph_agent()
        self._calibration = get_calibration_layer()
        self._ensemble = get_xgboost_fusion()
        self._initialized = False
        self._graph = None

    async def initialize(self) -> None:
        """Initialize all sub-agents and build the LangGraph."""
        if self._initialized:
            return
        logger.info("🧠 Initializing LangGraph Fusion Orchestrator...")

        # Initialize NLP agent first (lightweight text classifier + RAG)
        await self._nlp_agent.initialize()
        # Initialize Graph Agent (NetworkX + PyTorch GAT)
        await self._graph_agent.initialize()
        # Calibration layer
        self._calibration.initialize()
        self._ensemble.initialize()
        # Build the LangGraph state machine
        self._graph = self._build_graph()

        self._initialized = True
        logger.info("✅ LangGraph Fusion Orchestrator ready")

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph StateGraph.

        Graph topology:
          input → route → [vision | speech | nlp] → graph → fusion → evaluator
                                                                         ↓
                                                              calibration → END
                                                              (or loop back)
        """
        graph = StateGraph(FraudAnalysisState)

        # ─── Add Nodes ───
        graph.add_node("input_node", input_node)
        graph.add_node("route_node", self._route_node)
        graph.add_node("vision_node", self._vision_node)
        graph.add_node("speech_node", self._speech_node)
        graph.add_node("nlp_node", self._nlp_node)
        graph.add_node("graph_node", self._graph_node)
        graph.add_node("fusion_node", self._fusion_node)
        graph.add_node("evaluator_node", self._evaluator_node)
        graph.add_node("calibration_node", self._calibration_node)

        # ─── Set Entry Point ───
        graph.set_entry_point("input_node")

        # ─── Add Edges ───
        graph.add_edge("input_node", "route_node")

        # Conditional routing: route_node → agents based on modality
        graph.add_conditional_edges(
            "route_node",
            self._decide_next_agent,
            {
                "vision_node": "vision_node",
                "speech_node": "speech_node",
                "nlp_node": "nlp_node",
            },
        )

        # Agent → next agent or graph
        graph.add_conditional_edges(
            "vision_node",
            self._after_vision,
            {
                "speech_node": "speech_node",
                "nlp_node": "nlp_node",
                "graph_node": "graph_node",
            },
        )

        graph.add_conditional_edges(
            "speech_node",
            self._after_speech,
            {
                "nlp_node": "nlp_node",
                "graph_node": "graph_node",
            },
        )

        graph.add_edge("nlp_node", "graph_node")
        graph.add_edge("graph_node", "fusion_node")
        graph.add_edge("fusion_node", "evaluator_node")

        # Evaluator: loop back for ambiguous cases or proceed to calibration
        graph.add_conditional_edges(
            "evaluator_node",
            self._should_recalibrate,
            {
                "nlp_node": "nlp_node",
                "calibration_node": "calibration_node",
            },
        )

        graph.add_edge("calibration_node", END)

        return graph.compile()

    # ─── Node Implementations ────────────────────────────────────────

    async def _route_node(self, state: FraudAnalysisState) -> dict:
        """LLM-powered routing decision using Kimi K2."""
        modality = state.get("modality", "text")
        text = state.get("text", "")
        image_bytes = state.get("image_bytes")
        audio_bytes = state.get("audio_bytes")
        start_time = state.get("start_time", time.time())

        # Fast path for single modality
        if modality == "image":
            routing = {
                "agents_to_invoke": ["vision", "nlp"],
                "reasoning": "Image input — vision for detection + NLP for multimodal reasoning",
            }
        elif modality == "audio":
            routing = {
                "agents_to_invoke": ["speech", "nlp"],
                "reasoning": "Audio input — speech for transcription/spoof + NLP for content",
            }
        elif modality == "text":
            routing = {
                "agents_to_invoke": ["nlp"],
                "reasoning": "Text input — NLP for scam pattern analysis",
            }
        else:
            # Complex multimodal: use LLM routing
            try:
                input_desc = f"Modality: {modality}\n"
                if text:
                    input_desc += f"Text (first 200 chars): {text[:200]}\n"
                if image_bytes:
                    input_desc += f"Image: {len(image_bytes)} bytes\n"
                if audio_bytes:
                    input_desc += f"Audio: {len(audio_bytes)} bytes\n"

                result = await self._llm.classify_fast(
                    system=ROUTING_PROMPT,
                    user=input_desc,
                    json_mode=True,
                )
                routing = json.loads(result["content"])
            except Exception as e:
                logger.warning(f"LLM routing failed ({e}), using fallback")
                agents = []
                if image_bytes:
                    agents.append("vision")
                if audio_bytes:
                    agents.append("speech")
                agents.append("nlp")
                routing = {
                    "agents_to_invoke": agents,
                    "reasoning": "Fallback routing — invoking all relevant agents",
                }

        trace_entry = {
            "step": "routing",
            "agents": routing.get("agents_to_invoke", []),
            "reasoning": routing.get("reasoning", ""),
            "timestamp": time.time() - start_time,
        }

        return {
            "agents_to_invoke": routing.get("agents_to_invoke", ["nlp"]),
            "routing_reasoning": routing.get("reasoning", ""),
            "trace": state.get("trace", []) + [trace_entry],
        }

    async def _vision_node(self, state: FraudAnalysisState) -> dict:
        """Vision agent: counterfeit detection + forensics + Grad-CAM."""
        image_bytes = state.get("image_bytes")
        start_time = state.get("start_time", time.time())

        if not image_bytes:
            return {}

        try:
            logger.info("  → [LangGraph] Vision Agent")
            await self._vision_agent.initialize()
            vision_result = await self._vision_agent.analyze(image_bytes)

            trace_entry = {
                "step": "vision_agent",
                "verdict": vision_result.get("verdict"),
                "confidence": vision_result.get("model_confidence"),
                "techniques": ["YOLOv8", "EfficientNet+Transformer", "ELA", "FFT", "NPR", "CLIP", "Grad-CAM"],
                "timestamp": time.time() - start_time,
            }
            return {
                "vision_result": vision_result,
                "trace": state.get("trace", []) + [trace_entry],
            }
        except Exception as e:
            logger.error(f"Vision agent failed: {e}")
            return {
                "trace": state.get("trace", []) + [{"step": "vision_agent", "error": str(e)}],
            }

    async def _speech_node(self, state: FraudAnalysisState) -> dict:
        """Speech agent: transcription + voice spoofing detection."""
        audio_bytes = state.get("audio_bytes")
        start_time = state.get("start_time", time.time())

        if not audio_bytes:
            return {}

        try:
            logger.info("  → [LangGraph] Speech Agent")
            await self._speech_agent.initialize()
            speech_result = await self._speech_agent.analyze(audio_bytes)
            transcript = speech_result.get("transcript", {}).get("text", "")

            trace_entry = {
                "step": "speech_agent",
                "spoof_score": speech_result.get("spoof_detection", {}).get("spoof_score"),
                "transcript_length": len(transcript),
                "techniques": ["Whisper", "WavLM/AASIST"],
                "timestamp": time.time() - start_time,
            }
            return {
                "speech_result": speech_result,
                "transcript_text": transcript or state.get("transcript_text", ""),
                "trace": state.get("trace", []) + [trace_entry],
            }
        except Exception as e:
            logger.error(f"Speech agent failed: {e}")
            return {
                "trace": state.get("trace", []) + [{"step": "speech_agent", "error": str(e)}],
            }

    async def _nlp_node(self, state: FraudAnalysisState) -> dict:
        """NLP agent: Kimi K2 reasoning + DistilBERT + RAG."""
        transcript = state.get("transcript_text", state.get("text", ""))
        start_time = state.get("start_time", time.time())

        if not transcript:
            return {}

        try:
            logger.info("  → [LangGraph] NLP Agent")
            nlp_context = {}
            speech_result = state.get("speech_result")
            if speech_result:
                spoof_data = speech_result.get("spoof_detection", {})
                nlp_context["audio_spoof_score"] = spoof_data.get("spoof_score", 0.5)

            nlp_result = await self._nlp_agent.analyze(transcript, context=nlp_context)

            trace_entry = {
                "step": "nlp_agent",
                "verdict": nlp_result.get("verdict"),
                "confidence": nlp_result.get("fused_confidence"),
                "techniques": ["Groq GPT-OSS CoT", "DistilBERT NLI", "Hybrid RAG (BM25 + Dense + Rerank)", "Linguistic Features"],
                "timestamp": time.time() - start_time,
            }
            return {
                "nlp_result": nlp_result,
                "trace": state.get("trace", []) + [trace_entry],
            }
        except Exception as e:
            logger.error(f"NLP agent failed: {e}")
            return {
                "trace": state.get("trace", []) + [{"step": "nlp_agent", "error": str(e)}],
            }

    async def _graph_node(self, state: FraudAnalysisState) -> dict:
        """Graph agent: GAT node classification + community detection."""
        start_time = state.get("start_time", time.time())

        try:
            logger.info("  → [LangGraph] Graph Agent")
            graph_result = self._graph_agent.analyze_network()

            trace_entry = {
                "step": "graph_agent",
                "confidence": graph_result.get("network_risk_score", 0.0),
                "high_risk_nodes": len(graph_result.get("high_risk_nodes", [])),
                "communities": len(graph_result.get("communities", [])),
                "techniques": ["GAT (4-head)", "Community Detection", "Network Centrality"],
                "timestamp": time.time() - start_time,
            }
            return {
                "graph_result": graph_result,
                "trace": state.get("trace", []) + [trace_entry],
            }
        except Exception as e:
            logger.warning(f"Graph agent failed: {e}")
            return {
                "trace": state.get("trace", []) + [{"step": "graph_agent", "error": str(e)}],
            }

    async def _fusion_node(self, state: FraudAnalysisState) -> dict:
        """Fuse all agent outputs via weighted scoring."""
        start_time = state.get("start_time", time.time())

        scores = {}
        weights = {}

        vision_result = state.get("vision_result")
        if vision_result:
            scores["vision"] = vision_result.get("model_confidence", 0.5)
            weights["vision"] = config.orchestrator.vision_weight

        speech_result = state.get("speech_result")
        if speech_result:
            spoof = speech_result.get("spoof_detection", {})
            scores["speech"] = spoof.get("spoof_score", 0.5)
            weights["speech"] = config.orchestrator.speech_weight

        nlp_result = state.get("nlp_result")
        if nlp_result:
            scores["nlp"] = nlp_result.get("fused_confidence", 0.5)
            weights["nlp"] = config.orchestrator.nlp_weight

        graph_result = state.get("graph_result")
        if graph_result:
            scores["graph"] = graph_result.get("network_risk_score", 0.0)
            weights["graph"] = 0.10

        if not scores:
            base_score = 0.5
            verdict = "no_analysis"
        else:
            verdict = "pending"
            total_weight = sum(weights.values())
            base_score = sum(
                scores[agent] * weights[agent] for agent in scores
            ) / total_weight

        ensemble_result = self._ensemble.predict(state, base_score)
        fused_score = ensemble_result["score"]
        fusion_method = ensemble_result["method"]

        if verdict != "no_analysis":
            if fused_score > config.orchestrator.high_risk_threshold:
                verdict = "high_risk"
            elif fused_score > config.orchestrator.medium_risk_threshold:
                verdict = "medium_risk"
            elif fused_score > config.orchestrator.low_risk_threshold:
                verdict = "low_risk"
            else:
                verdict = "safe"

        trace_entry = {
            "step": "fusion",
            "fused_score": round(fused_score, 4),
            "base_weighted_score": round(base_score, 4),
            "fusion_method": fusion_method,
            "verdict": verdict,
            "per_agent_scores": {k: round(v, 4) for k, v in scores.items()},
            "xgboost_available": ensemble_result.get("model_available", False),
            "timestamp": time.time() - start_time,
        }

        return {
            "fused_score": round(float(fused_score), 4),
            "base_weighted_score": round(float(base_score), 4),
            "verdict": verdict,
            "per_agent_scores": {k: round(v, 4) for k, v in scores.items()},
            "per_agent_weights": weights,
            "fusion_method": fusion_method,
            "xgboost_features": ensemble_result["features"],
            "raw_xgboost_score": ensemble_result.get("raw_xgboost_score"),
            "trace": state.get("trace", []) + [trace_entry],
        }

    async def _evaluator_node(self, state: FraudAnalysisState) -> dict:
        """
        Evaluator: checks if the verdict is ambiguous and triggers re-analysis.
        This is the cyclic self-correction feature of the LangGraph.
        """
        iteration = state.get("iteration", 0)
        fused_score = state.get("fused_score", 0.5)
        start_time = state.get("start_time", time.time())

        is_ambiguous = 0.40 <= fused_score <= 0.60
        should_retry = is_ambiguous and iteration < 1  # Max 1 retry

        trace_entry = {
            "step": "evaluator",
            "fused_score": fused_score,
            "is_ambiguous": is_ambiguous,
            "will_retry": should_retry,
            "iteration": iteration,
            "timestamp": time.time() - start_time,
        }

        return {
            "iteration": iteration + 1,
            "trace": state.get("trace", []) + [trace_entry],
        }

    async def _calibration_node(self, state: FraudAnalysisState) -> dict:
        """Final calibration: isotonic/Platt scaling for reliable probabilities."""
        fused_score = state.get("fused_score", 0.5)
        start_time = state.get("start_time", time.time())

        calibrated_score = self._calibration.calibrate(fused_score)

        if calibrated_score > 0.80:
            risk_level = "critical"
        elif calibrated_score > 0.60:
            risk_level = "high"
        elif calibrated_score > 0.40:
            risk_level = "medium"
        elif calibrated_score > 0.20:
            risk_level = "low"
        else:
            risk_level = "safe"

        trace_entry = {
            "step": "calibration",
            "raw_score": fused_score,
            "calibrated_score": calibrated_score,
            "risk_level": risk_level,
            "timestamp": time.time() - start_time,
        }

        return {
            "calibrated_score": calibrated_score,
            "risk_level": risk_level,
            "trace": state.get("trace", []) + [trace_entry],
        }

    # ─── Conditional Edge Functions ──────────────────────────────────

    def _decide_next_agent(self, state: FraudAnalysisState) -> str:
        """Route to the first agent in the invocation list."""
        agents = state.get("agents_to_invoke", ["nlp"])
        if "vision" in agents and state.get("image_bytes"):
            return "vision_node"
        elif "speech" in agents and state.get("audio_bytes"):
            return "speech_node"
        else:
            return "nlp_node"

    def _after_vision(self, state: FraudAnalysisState) -> str:
        """After vision, route to speech or NLP."""
        agents = state.get("agents_to_invoke", [])
        if "speech" in agents and state.get("audio_bytes"):
            return "speech_node"
        elif "nlp" in agents:
            return "nlp_node"
        else:
            return "graph_node"

    def _after_speech(self, state: FraudAnalysisState) -> str:
        """After speech, route to NLP or graph."""
        agents = state.get("agents_to_invoke", [])
        if "nlp" in agents:
            return "nlp_node"
        else:
            return "graph_node"

    def _should_recalibrate(self, state: FraudAnalysisState) -> str:
        """Evaluator decision: retry or finalize."""
        iteration = state.get("iteration", 0)
        fused_score = state.get("fused_score", 0.5)

        is_ambiguous = 0.40 <= fused_score <= 0.60
        if is_ambiguous and iteration <= 1:
            logger.info(f"  ↻ [LangGraph] Evaluator: ambiguous ({fused_score:.3f}), re-analyzing...")
            return "nlp_node"
        return "calibration_node"

    # ─── Public API ──────────────────────────────────────────────────

    async def process(
        self,
        text: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        audio_bytes: Optional[bytes] = None,
    ) -> dict:
        """
        Main entry point — process citizen input through the LangGraph pipeline.
        Returns fused verdict with full agent trace.
        """
        if not self._initialized:
            await self.initialize()

        # Build initial state
        initial_state: FraudAnalysisState = {
            "text": text,
            "image_bytes": image_bytes,
            "audio_bytes": audio_bytes,
            "trace": [],
        }

        # Execute the LangGraph
        start_time = time.time()
        final_state = await self._graph.ainvoke(initial_state)
        processing_time = time.time() - start_time

        # Build response from final state
        agent_results = {}
        if final_state.get("vision_result"):
            agent_results["vision"] = {
                k: v for k, v in final_state["vision_result"].items()
                if k not in ["attention_map_base64", "annotated_overlay_base64", "note_crop"]
            }
        if final_state.get("speech_result"):
            agent_results["speech"] = final_state["speech_result"]
        if final_state.get("nlp_result"):
            agent_results["nlp"] = final_state["nlp_result"]
        if final_state.get("graph_result"):
            agent_results["graph"] = final_state["graph_result"]

        return {
            "verdict": final_state.get("verdict", "unknown"),
            "confidence": round(final_state.get("calibrated_score", 0.5), 4),
            "risk_level": final_state.get("risk_level", "unknown"),
            "agent_results": agent_results,
            "agent_visualizations": {
                "attention_map": (final_state.get("vision_result") or {}).get("attention_map_base64"),
                "annotated_overlay": (final_state.get("vision_result") or {}).get("annotated_overlay_base64"),
            },
            "fusion_details": {
                "fused_score": final_state.get("fused_score", 0.5),
                "base_weighted_score": final_state.get("base_weighted_score", 0.5),
                "verdict": final_state.get("verdict", "unknown"),
                "per_agent_scores": final_state.get("per_agent_scores", {}),
                "per_agent_weights": final_state.get("per_agent_weights", {}),
                "fusion_method": final_state.get("fusion_method", "weighted_fallback"),
                "xgboost_features": final_state.get("xgboost_features", {}),
                "raw_xgboost_score": final_state.get("raw_xgboost_score"),
            },
            "trace": final_state.get("trace", []),
            "processing_time_seconds": round(processing_time, 2),
            "agents_invoked": list(agent_results.keys()),
            "input_modality": final_state.get("modality", "text"),
            "langgraph": {
                "graph_type": "StateGraph",
                "nodes": ["input", "route", "vision", "speech", "nlp", "graph", "fusion", "evaluator", "calibration"],
                "iterations": final_state.get("iteration", 1),
                "self_correction": final_state.get("iteration", 1) > 1,
            },
        }

    def get_stats(self) -> dict:
        return {
            "orchestrator": "LangGraph StateGraph (cyclic self-correction)",
            "status": "ready" if self._initialized else "not_initialized",
            "agents": {
                "vision": self._vision_agent.get_stats(),
                "speech": self._speech_agent.get_stats(),
                "nlp": self._nlp_agent.get_stats(),
                "graph": self._graph_agent.get_stats(),
            },
            "ensemble": self._ensemble.get_stats(),
            "graph_topology": {
                "type": "StateGraph with conditional edges",
                "nodes": 9,
                "edges": 12,
                "cyclic": True,
                "self_correction": "Evaluator loops back for ambiguous verdicts",
            },
            "total_ai_techniques": 21,
        }
