"""
Configuration for Digital Public Safety Shield.
All model IDs, API endpoints, and tunable parameters in one place.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ─── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SCAM_PATTERNS_DIR = DATA_DIR / "scam_patterns"
SAMPLE_AUDIO_DIR = DATA_DIR / "sample_audio"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"
MODELS_CACHE_DIR = BASE_DIR / "model_cache"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# Ensure directories exist
for d in [
    DATA_DIR,
    SCAM_PATTERNS_DIR,
    SAMPLE_AUDIO_DIR,
    SAMPLE_IMAGES_DIR,
    MODELS_CACHE_DIR,
    CHROMA_DB_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


# ─── LLM Provider Configuration ─────────────────────────────────────────
@dataclass
class GroqConfig:
    """Groq free tier — primary LLM provider."""

    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    base_url: str = "https://api.groq.com/openai/v1"
    # Primary: Kimi K2 — best-in-class agentic reasoning, 200-300 tool calls
    primary_model: str = "openai/gpt-oss-20b"
    # Multimodal: Llama 4 Maverick — natively sees images
    multimodal_model: str = "meta-llama/llama-4-maverick-17b-128e-instruct"
    # Reasoning fallback: DeepSeek R1 distill
    reasoning_model: str = "openai/gpt-oss-120b"
    # Fast lightweight: Llama 4 Scout
    fast_model: str = "llama-3.1-8b-instant"
    # Whisper for speech-to-text
    whisper_model: str = "whisper-large-v3"
    # Rate limits (free tier)
    max_rpm: int = 30
    max_tpm: int = 15000
    timeout: int = 60


@dataclass
class OpenRouterConfig:
    """OpenRouter free tier — fallback LLM provider."""

    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    base_url: str = "https://openrouter.ai/api/v1"
    # Free auto-router
    free_router: str = "openrouter/free"
    # DeepSeek V4 Flash (free)
    reasoning_model: str = "openrouter/free"
    max_rpm: int = 20
    timeout: int = 90


@dataclass
class LocalModelConfig:
    """Local HuggingFace models — offline emergency fallback."""

    # Emergency offline LLM
    offline_llm: str = "microsoft/Phi-4-mini-instruct"
    # Vision models (always local)
    yolo_model: str = "yolov8n.pt"
    efficientnet_model: str = "efficientnet_b0"
    clip_model: str = "openai/clip-vit-base-patch32"
    # Speech models (always local)
    whisper_model: str = "openai/whisper-large-v3"
    wavlm_model: str = "microsoft/wavlm-base"
    # NLP models (always local)
    distilbert_model: str = "distilbert-base-uncased"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Spoof detection
    spoof_model: str = "Vansh180/deepfake-audio-wav2vec2"


# ─── Agent Configuration ─────────────────────────────────────────────────
@dataclass
class VisionAgentConfig:
    """Vision Agent parameters."""

    confidence_threshold: float = 0.5
    yolo_confidence: float = 0.25
    yolo_iou: float = 0.45
    # ELA parameters
    ela_quality: int = 90
    ela_scale: float = 10.0
    # Grad-CAM target layer
    gradcam_target_layer: str = "blocks[-1].norm1"
    # Regions of interest for currency notes
    regions_of_interest: list = field(
        default_factory=lambda: [
            "security_thread",
            "micro_lettering",
            "serial_number",
            "watermark",
            "latent_image",
            "color_shifting_ink",
        ]
    )


@dataclass
class SpeechAgentConfig:
    """Speech Agent parameters."""

    # Whisper settings
    whisper_language: str = "en"
    whisper_task: str = "transcribe"
    chunk_duration_sec: float = 5.0
    # Spoof detection threshold
    spoof_threshold: float = 0.5
    sample_rate: int = 16000


@dataclass
class NLPAgentConfig:
    """NLP/LLM Agent parameters."""

    # RAG settings
    rag_top_k: int = 3
    rag_similarity_threshold: float = 0.4
    # DistilBERT classification
    text_classifier_threshold: float = 0.5
    # LLM reasoning
    max_reasoning_turns: int = 10
    temperature: float = 0.3
    max_tokens: int = 2048


@dataclass
class OrchestratorConfig:
    """Fusion Orchestrator parameters."""

    # Ensemble stacking weights (initial — XGBoost will learn optimal)
    vision_weight: float = 0.30
    speech_weight: float = 0.25
    nlp_weight: float = 0.30
    text_classifier_weight: float = 0.15
    # Calibration
    calibration_method: str = "isotonic"  # "isotonic" or "temperature"
    # Verdict thresholds
    high_risk_threshold: float = 0.75
    medium_risk_threshold: float = 0.45
    low_risk_threshold: float = 0.25


# ─── Master Config ────────────────────────────────────────────────────────
@dataclass
class AppConfig:
    """Master application configuration."""

    app_name: str = "Digital Public Safety Shield"
    app_version: str = "1.0.0"
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list = field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"]
    )

    groq: GroqConfig = field(default_factory=GroqConfig)
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    local_models: LocalModelConfig = field(default_factory=LocalModelConfig)
    vision_agent: VisionAgentConfig = field(default_factory=VisionAgentConfig)
    speech_agent: SpeechAgentConfig = field(default_factory=SpeechAgentConfig)
    nlp_agent: NLPAgentConfig = field(default_factory=NLPAgentConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)


# Singleton config instance
config = AppConfig()
