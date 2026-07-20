"""
Digital Public Safety Shield — FastAPI Backend.

Main API server exposing multi-agent AI system via REST + WebSocket.
Accepts image/audio/text input, routes through the Agentic Fusion Orchestrator,
returns structured verdicts with full agent trace.

AI inference uses configured hosted APIs and local open-weight models.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # suppress TF warnings

import asyncio
import base64
import html
import hmac
import io
import json
import logging
import sys
import time
import uuid
from urllib.parse import urlparse
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Query,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from agents.orchestrator import FusionOrchestrator
from models.vision.currency_features import compare_tilt_captures
from auth_store import (
    SUPPORTED_LANGUAGES,
    authenticate_user,
    create_access_token,
    create_user,
    decode_token,
    get_case,
    get_history,
    init_db,
    save_case,
    revoke_tokens,
    update_user,
)
from intelligence import (
    build_evidence_package,
    build_reporting_draft,
    command_center_plan,
    geospatial_overview,
    reporting_guidance,
)
from localization import normalize_language
from analytics import get_analytics_tracker
from broker import broker_status, close_broker, jobs_enabled, publish_job
from evidence_store import evidence_store_status, mirror_case_evidence
from feed_connectors import feed_status, poll_once, start_feed_pollers, stop_feed_pollers
from job_store import create_job, fail_job, get_job, init_job_db
from production_readiness import readiness_report, require_operational_integration
from operational_store import (
    certified_currency_count,
    init_operational_db,
    operational_counts,
    operational_provenance_counts,
    list_model_feedback,
    save_model_feedback,
    upsert_currency_specimen,
    upsert_geospatial_incident,
    upsert_graph_edge,
    upsert_graph_entity,
)
from realtime_safety import (
    append_event as append_realtime_event,
    close_session as close_realtime_session,
    create_session as create_realtime_session,
    dispatch_alerts,
    ensure_alerts,
    get_session as get_realtime_session,
    get_transcript as get_realtime_transcript,
    init_realtime_db,
    list_alerts,
)
from redis_service import (
    close_redis,
    consume_rate_limit,
    initialize_redis,
    redis_status,
    reset_rate_limit,
)
from telemetry import configure_telemetry

# ─── Logging Setup ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ─── Global Orchestrator ─────────────────────────────────────────────────
orchestrator: Optional[FusionOrchestrator] = None
telemetry_status: dict[str, Any] = {"enabled": False, "status": "disabled"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize AI models on startup."""
    global orchestrator
    init_db()
    init_job_db()
    init_operational_db()
    init_realtime_db()
    await initialize_redis()
    start_feed_pollers()
    jwt_secret = os.getenv("JWT_SECRET", "")
    if len(jwt_secret) < 32:
        logger.warning(" JWT_SECRET is not production-ready; set a random value of at least 32 characters")
    logger.info("=" * 60)
    logger.info(" DIGITAL PUBLIC SAFETY SHIELD")
    logger.info(" Multi-Agent AI System — 17 AI Techniques")
    logger.info(" Providers: Groq + OpenRouter + HuggingFace/local models")
    logger.info("=" * 60)

    orchestrator = FusionOrchestrator()
    try:
        await orchestrator.initialize()
        logger.info(" All systems initialized successfully")
    except Exception as e:
        logger.warning(f" Partial initialization: {e}")
        logger.info("System will initialize remaining models on first use")

    yield

    await close_broker()
    await stop_feed_pollers()
    await close_redis()
    logger.info(" Shutting down...")


# ─── FastAPI App ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Digital Public Safety Shield",
    description=(
        "AI-powered fraud detection platform using 17 AI techniques: "
        "YOLOv8, EfficientNet, Contrastive Learning, ELA, FFT, NPR, CLIP, Grad-CAM, "
        "Whisper, WavLM/AASIST, Kimi K2.5, Qwen 3.6, DistilBERT, "
        "Hybrid RAG, Multi-Role CoT, Ensemble Fusion, Calibration."
    ),
    version=config.app_version,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
telemetry_status = configure_telemetry(app)


@app.middleware("http")
async def operational_headers(request: Request, call_next):
    """Attach correlation and baseline security headers to every response."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    started = time.perf_counter()
    analysis_limit = None
    if request.url.path.startswith("/api/analyze"):
        client_id = request.client.host if request.client else "unknown"
        analysis_limit = await consume_rate_limit(
            "analysis",
            client_id,
            limit=int(os.getenv("ANALYSIS_RATE_LIMIT", "30")),
            window_seconds=int(os.getenv("ANALYSIS_RATE_WINDOW_SECONDS", "60")),
        )
        if analysis_limit and not analysis_limit["allowed"]:
            return JSONResponse(
                status_code=429,
                content={"detail": "Analysis rate limit exceeded. Try again shortly."},
                headers={
                    "Retry-After": str(analysis_limit["retry_after"]),
                    "X-Request-ID": request_id,
                    "X-RateLimit-Limit": str(analysis_limit["limit"]),
                    "X-RateLimit-Remaining": str(analysis_limit["remaining"]),
                },
            )
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(self), microphone=(self), geolocation=(self)"
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
    if analysis_limit:
        response.headers["X-RateLimit-Limit"] = str(analysis_limit["limit"])
        response.headers["X-RateLimit-Remaining"] = str(analysis_limit["remaining"])
    return response


# ─── Request / Response Models ───────────────────────────────────────────


class TextAnalysisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50_000)
    context: Optional[dict] = None
    language: str = "en"


class TurnByTurnRequest(BaseModel):
    turns: list[str] = Field(min_length=1, max_length=100)
    language: str = "en"


class AsyncTextAnalysisRequest(TextAnalysisRequest):
    pass


class RegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=8, max_length=128)
    preferred_language: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    preferred_language: Optional[str] = None


class VoiceSynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5_000)
    language: str = "en"


class RealtimeSessionCreate(BaseModel):
    channel: str = Field(default="mobile_app", pattern="^(mobile_app|whatsapp|ivr|telecom|bank_pos|web)$")
    language: str = "en"
    caller_id: Optional[str] = Field(default=None, max_length=160)
    participant_id: Optional[str] = Field(default=None, max_length=160)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RealtimeEventRequest(BaseModel):
    transcript: str = Field(default="", max_length=20_000)
    caller_verification: str = Field(default="unknown", pattern="^(verified|failed|unknown)$")
    stir_shaken_attestation: str = Field(default="unavailable", pattern="^(A|B|C|failed|unavailable)$")
    spoof_risk: float = Field(default=0.0, ge=0, le=1)
    claimed_authority: Optional[str] = Field(default=None, max_length=120)
    video_present: bool = False
    video_identity_mismatch: bool = False
    face_swap_score: float = Field(default=0.0, ge=0, le=1)
    virtual_background: bool = False
    payment_requested: bool = False
    amount: Optional[float] = Field(default=None, ge=0)
    destination_account: Optional[str] = Field(default=None, max_length=180)
    secrecy_requested: bool = False
    screen_share_requested: bool = False
    remote_app_requested: bool = False
    urgency_seconds: Optional[int] = Field(default=None, ge=0)
    occurred_at: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeospatialIncidentIngest(BaseModel):
    id: str = Field(min_length=3, max_length=160)
    district: str = Field(min_length=1, max_length=120)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    type: str = Field(min_length=1, max_length=80)
    reports: int = Field(default=1, ge=1)
    severity: float = Field(default=0.5, ge=0, le=1)
    source: str = Field(default="authorized_feed", max_length=120)
    source_reference: Optional[str] = Field(default=None, max_length=240)
    occurred_at: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance_tier: str = Field(default="authorized", pattern="^(authorized|public_research|synthetic_sandbox)$")


class GraphEntityIngest(BaseModel):
    id: str = Field(min_length=2, max_length=180)
    type: str = Field(min_length=1, max_length=80)
    label: str = Field(default="unknown", max_length=80)
    source: str = Field(default="authorized_feed", max_length=120)
    source_reference: Optional[str] = Field(default=None, max_length=240)
    attrs: dict[str, Any] = Field(default_factory=dict)
    provenance_tier: str = Field(default="authorized", pattern="^(authorized|public_research|synthetic_sandbox)$")


class GraphEdgeIngest(BaseModel):
    source_id: str = Field(min_length=2, max_length=180)
    target_id: str = Field(min_length=2, max_length=180)
    type: str = Field(default="connected", max_length=80)
    weight: float = Field(default=1.0, ge=0)
    source: str = Field(default="authorized_feed", max_length=120)
    source_reference: Optional[str] = Field(default=None, max_length=240)
    observed_at: Optional[str] = None
    attrs: dict[str, Any] = Field(default_factory=dict)
    provenance_tier: str = Field(default="authorized", pattern="^(authorized|public_research|synthetic_sandbox)$")


class FraudNetworkEventIngest(BaseModel):
    id: str = Field(min_length=3, max_length=180)
    event_type: str = Field(min_length=1, max_length=80)
    source_entity: GraphEntityIngest
    target_entity: GraphEntityIngest
    relationship: str = Field(default="connected", max_length=80)
    weight: float = Field(default=1.0, ge=0)
    source: str = Field(default="authorized_feed", max_length=120)
    source_reference: Optional[str] = Field(default=None, max_length=240)
    observed_at: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CurrencySpecimenIngest(BaseModel):
    id: str = Field(min_length=3, max_length=180)
    denomination: str = Field(min_length=1, max_length=40)
    label: str = Field(pattern="^(genuine|counterfeit|tampered|unknown)$")
    issuer: str = Field(min_length=2, max_length=160)
    image_sha256: str = Field(min_length=64, max_length=64)
    certification_reference: str = Field(min_length=3, max_length=240)
    captured_device: Optional[str] = Field(default=None, max_length=160)
    captured_at: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance_tier: str = Field(default="authorized", pattern="^(authorized|public_research|synthetic_sandbox)$")


class ReportingDraftRequest(BaseModel):
    destination: str = Field(default="NCRP", max_length=80)


class CaseFeedbackRequest(BaseModel):
    observed_outcome: str = Field(
        pattern="^(confirmed_fraud|confirmed_legitimate|confirmed_counterfeit|confirmed_genuine|inconclusive)$"
    )
    reporting_reference: Optional[str] = Field(default=None, max_length=240)
    notes: Optional[str] = Field(default=None, max_length=2_000)


class HealthResponse(BaseModel):
    status: str
    version: str
    agents: dict


login_attempts: dict[str, list[float]] = {}
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_AUDIO_BYTES = 30 * 1024 * 1024


async def _verify_ingest_token(request: Request) -> None:
    expected = os.getenv("SHIELD_INGEST_TOKEN", "").strip()
    supplied = (
        request.headers.get("X-Shield-Ingest-Token")
        or request.headers.get("X-API-Key")
        or request.query_params.get("token")
        or ""
    ).strip()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="SHIELD_INGEST_TOKEN must be configured before operational feed ingestion is enabled",
        )
    if not hmac.compare_digest(expected, supplied):
        raise HTTPException(status_code=401, detail="Invalid operational ingest token")


async def _read_upload(upload: UploadFile, *, kind: str) -> bytes:
    allowed = {
        "image": {"image/jpeg", "image/png", "image/webp"},
        "audio": {
            "audio/wav",
            "audio/x-wav",
            "audio/mpeg",
            "audio/mp4",
            "audio/x-m4a",
            "audio/aac",
            "audio/flac",
            "audio/ogg",
            "audio/webm",
            "video/webm",
        },
    }
    limit = MAX_IMAGE_BYTES if kind == "image" else MAX_AUDIO_BYTES
    if upload.content_type and upload.content_type not in allowed[kind]:
        raise HTTPException(status_code=415, detail=f"Unsupported {kind} content type")
    data = await upload.read(limit + 1)
    if not data:
        raise HTTPException(status_code=400, detail=f"Empty {kind} upload")
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"{kind.title()} exceeds {limit // (1024 * 1024)} MB limit")
    return data


async def _rate_limit_login(email: str) -> None:
    distributed = await consume_rate_limit(
        "login",
        email,
        limit=int(os.getenv("LOGIN_RATE_LIMIT", "8")),
        window_seconds=int(os.getenv("LOGIN_RATE_WINDOW_SECONDS", "300")),
    )
    if distributed:
        if not distributed["allowed"]:
            raise HTTPException(
                status_code=429,
                detail="Too many login attempts. Try again shortly.",
                headers={"Retry-After": str(distributed["retry_after"])},
            )
        return

    now = time.time()
    key = email.lower().strip()
    recent = [ts for ts in login_attempts.get(key, []) if now - ts < 300]
    if len(recent) >= 8:
        raise HTTPException(status_code=429, detail="Too many login attempts. Try again shortly.")
    recent.append(now)
    login_attempts[key] = recent


def _auth_payload(user: dict) -> dict:
    return {"access_token": create_access_token(user), "token_type": "bearer", "user": user}


async def get_optional_user(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    return decode_token(authorization.split(" ", 1)[1].strip())


async def get_current_user(user: Optional[dict] = Depends(get_optional_user)) -> dict:
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def _case_type(text: Optional[str], image_bytes: Optional[bytes], audio_bytes: Optional[bytes]) -> str:
    if image_bytes and audio_bytes:
        return "multimodal"
    if image_bytes:
        return "image"
    if audio_bytes:
        return "audio"
    return "text"


def _persist_if_user(user: Optional[dict], case_type: str, result: dict) -> dict:
    if user:
        result = save_case(user["id"], case_type, result)
        result["external_evidence_store"] = mirror_case_evidence(
            user["id"],
            result.get("case_id", "unknown"),
            result,
        )
        logger.info(
            json.dumps(
                {
                    "event": "verdict_created",
                    "user_id": user["id"],
                    "case_id": result.get("case_id"),
                    "case_type": case_type,
                    "verdict": result.get("verdict"),
                    "risk_level": result.get("risk_level"),
                    "confidence": result.get("confidence"),
                    "agents_invoked": result.get("agents_invoked", []),
                    "latency_seconds": result.get("processing_time_seconds"),
                }
            )
        )
    return result


def _log_to_analytics(result: dict, modality: str = "text") -> None:
    """Log analysis result to real-time analytics tracker."""
    try:
        tracker = get_analytics_tracker()

        # Extract stable pattern categories from the structured NLP output.
        # The raw LLM prose remains in the case result rather than analytics.
        scam_types: list[str] = []
        nlp_result = result.get("agent_results", {}).get("nlp", {})
        if nlp_result:
            for match in nlp_result.get("retrieved_pattern_matches", []):
                stype = match.get("scam_type") or match.get("category", "")
                if stype and stype not in scam_types:
                    scam_types.append(stype)
            for step in nlp_result.get("agent_trace", []):
                for finding in step.get("result", {}).get("findings", []):
                    pattern = finding.get("pattern", "")
                    if pattern and pattern not in scam_types:
                        scam_types.append(pattern)

        # From fusion details
        fusion = result.get("fusion_details", {})
        if not scam_types and fusion:
            verdict = result.get("verdict", "")
            if verdict in ("high_risk", "medium_risk", "needs_review"):
                scam_types.append(verdict)

        tracker.log_analysis(
            verdict=result.get("verdict", "unknown"),
            confidence=result.get("confidence", 0.0),
            risk_level=result.get("risk_level", "unknown"),
            scam_types=scam_types,
            modality=modality,
            agents_invoked=result.get("agents_invoked", []),
            processing_time=result.get("processing_time_seconds", 0.0),
        )
    except Exception as e:
        logger.warning(f"Analytics logging failed: {e}")


def _annotate_currency_certification(result: dict) -> dict:
    """Attach certification-readiness status to currency/image verdicts."""
    count = certified_currency_count()
    certified_manifest = bool(config.deployment.currency_certified_manifest.strip())
    result["currency_certification"] = {
        "mode": "certified_reference_available" if count or certified_manifest else "screening_only",
        "certified_specimens": count,
        "configured_manifest": config.deployment.currency_certified_manifest or None,
        "claim": (
            "Screening verdict cross-referenced with configured certified specimen metadata."
            if count or certified_manifest
            else "AI screening only; not RBI, bank, or forensic-lab certification."
        ),
    }
    return result


async def _verify_channel_webhook(request: Request) -> None:
    """Validate Twilio signatures or the explicit sandbox shared token."""
    provider = os.getenv("CHANNEL_WEBHOOK_PROVIDER", "shared").strip().lower()
    if provider == "twilio":
        auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
        supplied = request.headers.get("X-Twilio-Signature", "").strip()
        if not auth_token:
            raise HTTPException(status_code=503, detail="TWILIO_AUTH_TOKEN is required")
        public_base = os.getenv("TWILIO_WEBHOOK_BASE_URL", "").rstrip("/")
        url = f"{public_base}{request.url.path}"
        if request.url.query:
            url += f"?{request.url.query}"
        if not public_base:
            url = str(request.url)
        form = await request.form() if request.method == "POST" else {}
        signed = url + "".join(
            f"{key}{value}"
            for key in sorted(form.keys())
            for value in (form.getlist(key) if hasattr(form, "getlist") else [form[key]])
        )
        expected = base64.b64encode(
            hmac.new(auth_token.encode(), signed.encode(), digestmod="sha1").digest()
        ).decode()
        if not supplied or not hmac.compare_digest(expected, supplied):
            raise HTTPException(status_code=401, detail="Invalid Twilio webhook signature")
        return

    expected = os.getenv("MULTICHANNEL_WEBHOOK_TOKEN", "").strip()
    if not expected and not config.deployment.is_production:
        return
    if not expected:
        raise HTTPException(status_code=503, detail="A signed channel webhook mode is required in production")
    supplied = (
        request.headers.get("X-Shield-Channel-Token")
        or request.query_params.get("token")
        or ""
    ).strip()
    if not supplied or not hmac.compare_digest(expected, supplied):
        raise HTTPException(status_code=401, detail="Invalid channel webhook token")


def _twiml(*parts: str) -> Response:
    return Response(
        content=f'<?xml version="1.0" encoding="UTF-8"?><Response>{"".join(parts)}</Response>',
        media_type="application/xml",
    )


def _say(text: str, *, language: str = "en-IN") -> str:
    return f'<Say language="{html.escape(language)}">{html.escape(text)}</Say>'


def _message(text: str) -> Response:
    return _twiml(f"<Message>{html.escape(text)}</Message>")


def _compact_channel_guidance(result: dict, *, channel: str, language: str) -> str:
    verdict = str(result.get("verdict") or result.get("final_verdict") or "unknown").replace("_", " ")
    risk_level = str(result.get("risk_level") or "unknown").replace("_", " ")
    confidence = round(float(result.get("confidence") or 0) * 100)
    nlp = result.get("agent_results", {}).get("nlp", {})
    action = (
        nlp.get("recommended_action")
        or nlp.get("reasoning")
        or reporting_guidance(risk_level).get("immediate_actions", ["Use caution."])[0]
    )
    action = " ".join(str(action).split())
    if len(action) > 260:
        action = action[:257].rstrip() + "..."
    return (
        f"Shield verdict ({channel}): {risk_level.upper()} risk, {confidence}% confidence. "
        f"Assessment: {verdict}. Next step: {action} "
        "If money was transferred, call 1930 and preserve screenshots, numbers, audio, and transaction IDs."
    )


async def _analyze_channel_text(
    *,
    text: str,
    language: str,
    channel: str,
    sender: str = "",
) -> dict:
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    clean_text = " ".join(text.split())
    if not clean_text:
        raise HTTPException(status_code=400, detail="Message text is required")
    result = await orchestrator.process(
        text=f"{channel} citizen report: {clean_text}",
        context={"channel": channel, "sender": sender},
        language=normalize_language(language),
    )
    result["channel"] = channel
    _log_to_analytics(result, modality=channel)
    return result


async def _download_channel_media(media_url: str, content_type: str) -> bytes:
    """Download provider-hosted media when explicitly configured."""
    if not config.deployment.whatsapp_media_integration:
        raise HTTPException(status_code=503, detail="WhatsApp media ingestion is not enabled")
    parsed = urlparse(media_url)
    if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1"}:
        raise HTTPException(status_code=422, detail="Media URL must be HTTPS")
    allowlist = {
        host.strip().lower()
        for host in os.getenv(
            "WHATSAPP_MEDIA_ALLOWED_HOSTS",
            "api.twilio.com,media.twiliocdn.com",
        ).split(",")
        if host.strip()
    }
    hostname = (parsed.hostname or "").lower()
    if hostname not in {"localhost", "127.0.0.1"} and not any(
        hostname == allowed or hostname.endswith(f".{allowed}") for allowed in allowlist
    ):
        raise HTTPException(status_code=422, detail="Media host is not allowlisted")

    headers = {}
    auth = None
    bearer = os.getenv("WHATSAPP_MEDIA_BEARER_TOKEN", "").strip()
    twilio_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    twilio_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    elif twilio_sid and twilio_token:
        auth = (twilio_sid, twilio_token)

    import httpx

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        response = await client.get(media_url, headers=headers, auth=auth)
    response.raise_for_status()
    media = response.content
    max_bytes = MAX_IMAGE_BYTES if content_type.startswith("image/") else MAX_AUDIO_BYTES
    if len(media) > max_bytes:
        raise HTTPException(status_code=413, detail="Provider media exceeds allowed size")
    return media


@app.post("/api/auth/register")
async def register(request: RegisterRequest):
    if request.preferred_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")
    if len(request.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    try:
        user = create_user(request.name, request.email, request.password, request.preferred_language)
        return _auth_payload(user)
    except Exception as e:
        if "UNIQUE" in str(e).upper():
            raise HTTPException(status_code=409, detail="Email already registered")
        raise


@app.post("/api/auth/login")
async def login(request: LoginRequest):
    await _rate_limit_login(request.email)
    user = authenticate_user(request.email, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    await reset_rate_limit("login", request.email)
    return _auth_payload(user)


@app.get("/api/auth/me")
async def me(user: dict = Depends(get_current_user)):
    return user


@app.patch("/api/auth/me")
async def update_me(request: ProfileUpdateRequest, user: dict = Depends(get_current_user)):
    updated = update_user(user["id"], name=request.name, preferred_language=request.preferred_language)
    return updated


@app.post("/api/auth/logout")
async def logout(user: dict = Depends(get_current_user)):
    revoke_tokens(user["id"])
    return {"ok": True}


@app.get("/api/languages")
async def languages():
    return [{"code": code, "name": name} for code, name in SUPPORTED_LANGUAGES.items()]


@app.get("/api/history")
async def history(user: dict = Depends(get_current_user)):
    return {"items": get_history(user["id"])}


@app.get("/api/cases/{case_id}/evidence")
async def evidence_package(case_id: str, user: dict = Depends(get_current_user)):
    case = get_case(user["id"], case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return build_evidence_package(case, user)


@app.post("/api/cases/{case_id}/feedback", status_code=201)
async def case_outcome_feedback(
    case_id: str,
    request: CaseFeedbackRequest,
    user: dict = Depends(get_current_user),
):
    """Capture outcome labels for monitored, human-reviewed future retraining."""
    case = get_case(user["id"], case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    result = case["result"]
    record = {
        "id": str(uuid.uuid4()),
        "case_id": case_id,
        "user_id": user["id"],
        "predicted_verdict": str(result.get("verdict", "unknown")),
        "predicted_confidence": float(result.get("confidence", 0.0)),
        "observed_outcome": request.observed_outcome,
        "reporting_reference": request.reporting_reference,
        "notes": request.notes,
        "review_status": "pending_human_validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return save_model_feedback(record)


@app.get("/api/model-monitoring/feedback")
async def model_feedback_history(user: dict = Depends(get_current_user)):
    records = list_model_feedback(user["id"])
    return {
        "records": records,
        "count": len(records),
        "training_policy": "Feedback is quarantined until human validation; it is never auto-trained.",
    }


@app.get("/api/intelligence/hotspots")
async def intelligence_hotspots(
    latitude: Optional[float] = Query(None, ge=-90, le=90),
    longitude: Optional[float] = Query(None, ge=-180, le=180),
):
    require_operational_integration("geospatial")
    if (latitude is None) != (longitude is None):
        raise HTTPException(status_code=400, detail="Latitude and longitude must be supplied together")
    return geospatial_overview(latitude, longitude)


@app.get("/api/intelligence/command-center")
async def intelligence_command_center(
    available_units: int = Query(default=10, ge=1, le=10_000),
):
    require_operational_integration("geospatial")
    return command_center_plan(available_units)


@app.get("/api/reporting/guidance")
async def report_guidance(risk_level: str = Query("medium")):
    return reporting_guidance(risk_level)


@app.post("/api/reporting/cases/{case_id}/draft")
async def reporting_draft(
    case_id: str,
    request: ReportingDraftRequest,
    user: dict = Depends(get_current_user),
):
    case = get_case(user["id"], case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return build_reporting_draft(case, user, destination=request.destination)


@app.post("/api/reporting/cases/{case_id}/submit")
async def reporting_submit(
    case_id: str,
    request: ReportingDraftRequest,
    user: dict = Depends(get_current_user),
):
    case = get_case(user["id"], case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    draft = build_reporting_draft(case, user, destination=request.destination)
    if not config.deployment.official_reporting_api_url:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "official_reporting_bridge_not_configured",
                "message": "Draft created, but OFFICIAL_REPORTING_API_URL is not configured.",
                "draft": draft,
            },
        )

    import httpx

    headers = {
        "Authorization": f"Bearer {config.deployment.official_reporting_api_token}",
        "Content-Type": "application/json",
        "X-Shield-Case-ID": case_id,
        "X-Shield-Evidence-Hash": draft["integrity_hash"],
    }
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(
                config.deployment.official_reporting_api_url,
                headers=headers,
                json=draft["payload"],
            )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "official_reporting_bridge_rejected",
                "status_code": exc.response.status_code,
                "response": exc.response.text[:500],
                "draft": draft,
            },
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "official_reporting_bridge_unavailable",
                "message": str(exc),
                "draft": draft,
            },
        ) from exc

    return {
        "status": "submitted_to_configured_bridge",
        "case_id": case_id,
        "destination": request.destination,
        "evidence_hash": draft["integrity_hash"],
        "bridge_status_code": response.status_code,
        "bridge_response": response.text[:1000],
        "disclosure": "Submitted only to the configured authorized bridge URL.",
    }


@app.get("/api/integrations/status")
async def integrations_status():
    queue_readiness = await broker_status()
    redis_readiness = await redis_status()
    return {
        "operational_counts": operational_counts(),
        "provenance_counts": operational_provenance_counts(),
        "readiness": readiness_report(
            queue_status=queue_readiness,
            redis_status=redis_readiness,
            currency_manifest_installed=(Path(__file__).parent / "data" / "training" / "currency" / "source_manifest.json").exists(),
            certified_currency_specimens=certified_currency_count(),
        ),
        "ingest": {
            "token_configured": bool(os.getenv("SHIELD_INGEST_TOKEN", "").strip()),
            "geospatial_endpoint": "/api/integrations/geospatial/incidents",
            "graph_entities_endpoint": "/api/integrations/graph/entities",
            "graph_edges_endpoint": "/api/integrations/graph/edges",
        },
    }


@app.post("/api/integrations/geospatial/incidents")
async def ingest_geospatial_incidents(
    records: list[GeospatialIncidentIngest],
    request: Request,
):
    await _verify_ingest_token(request)
    if len(records) > 500:
        raise HTTPException(status_code=422, detail="A single ingest request supports up to 500 incidents")
    saved = [upsert_geospatial_incident(record.model_dump()) for record in records]
    return {"accepted": len(saved), "records": saved[:20], "counts": operational_counts()}


@app.post("/api/integrations/graph/entities")
async def ingest_graph_entities(
    records: list[GraphEntityIngest],
    request: Request,
):
    await _verify_ingest_token(request)
    if len(records) > 1000:
        raise HTTPException(status_code=422, detail="A single ingest request supports up to 1000 entities")
    saved = [upsert_graph_entity(record.model_dump()) for record in records]
    if orchestrator and orchestrator._graph_agent:
        orchestrator._graph_agent._initialized = False
    return {"accepted": len(saved), "records": saved[:20], "counts": operational_counts()}


@app.post("/api/integrations/graph/edges")
async def ingest_graph_edges(
    records: list[GraphEdgeIngest],
    request: Request,
):
    await _verify_ingest_token(request)
    if len(records) > 2000:
        raise HTTPException(status_code=422, detail="A single ingest request supports up to 2000 edges")
    saved = [upsert_graph_edge(record.model_dump()) for record in records]
    if orchestrator and orchestrator._graph_agent:
        orchestrator._graph_agent._initialized = False
    return {"accepted": len(saved), "records": saved[:20], "counts": operational_counts()}


@app.post("/api/integrations/fraud-network/events")
async def ingest_fraud_network_events(
    records: list[FraudNetworkEventIngest],
    request: Request,
):
    await _verify_ingest_token(request)
    if len(records) > 1000:
        raise HTTPException(status_code=422, detail="A single ingest request supports up to 1000 events")

    saved_events = []
    for record in records:
        payload = record.model_dump()
        source_entity = upsert_graph_entity(payload["source_entity"])
        target_entity = upsert_graph_entity(payload["target_entity"])
        edge = upsert_graph_edge({
            "id": payload["id"],
            "source_id": source_entity["id"],
            "target_id": target_entity["id"],
            "type": payload["relationship"],
            "weight": payload["weight"],
            "source": payload["source"],
            "source_reference": payload["source_reference"],
            "observed_at": payload["observed_at"],
            "attrs": {
                "event_type": payload["event_type"],
                **payload["metadata"],
            },
        })
        saved_events.append({"source_entity": source_entity, "target_entity": target_entity, "edge": edge})

    if orchestrator and orchestrator._graph_agent:
        orchestrator._graph_agent._initialized = False
    return {"accepted": len(saved_events), "events": saved_events[:20], "counts": operational_counts()}


@app.post("/api/integrations/currency/certified-specimens")
async def ingest_currency_specimens(
    records: list[CurrencySpecimenIngest],
    request: Request,
):
    await _verify_ingest_token(request)
    if len(records) > 1000:
        raise HTTPException(status_code=422, detail="A single ingest request supports up to 1000 specimens")
    saved = [upsert_currency_specimen(record.model_dump()) for record in records]
    return {"accepted": len(saved), "records": saved[:20], "counts": operational_counts()}


@app.post("/api/voice/transcribe")
async def voice_transcribe(
    audio: UploadFile = File(...),
    language: str = Form("en"),
):
    """Transcribe citizen voice input using the existing Whisper stack."""
    try:
        from models.speech.transcriber import get_transcriber

        transcriber = get_transcriber()
        if not transcriber._initialized:
            await transcriber.initialize()
        audio_bytes = await _read_upload(audio, kind="audio")
        result = await transcriber.transcribe_and_translate(audio_bytes, language=language, use_groq=True)
        return {
            "transcript": result.get("original_text", result.get("text", "")),
            "english_transcript": result.get("english_text", result.get("text", "")),
            "detected_language": result.get("language", language),
            "translated_to_english": result.get("translated_to_english", False),
            "provider": result.get("provider", "unknown"),
            "translation_provider": result.get("translation_provider"),
            "segments": result.get("segments", []),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice/synthesize")
async def voice_synthesize(request: VoiceSynthesisRequest):
    """Synthesize short English safety guidance with Groq Orpheus."""
    if request.language.lower().split("-")[0] != "en":
        raise HTTPException(
            status_code=422,
            detail="Server TTS currently supports English; use browser speech synthesis for other languages",
        )
    if len(request.text) > 200:
        raise HTTPException(status_code=422, detail="Server TTS supports up to 200 characters per request")
    if not config.groq.api_key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured")

    import httpx

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/audio/speech",
                headers={"Authorization": f"Bearer {config.groq.api_key}"},
                json={
                    "model": "canopylabs/orpheus-v1-english",
                    "input": request.text,
                    "voice": "hannah",
                    "response_format": "wav",
                },
            )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error("Groq TTS rejected the request: %s", exc.response.text[:300])
        raise HTTPException(status_code=502, detail="Speech provider rejected the synthesis request") from exc
    except httpx.HTTPError as exc:
        logger.error("Groq TTS request failed: %s", exc)
        raise HTTPException(status_code=502, detail="Speech provider is unavailable") from exc

    return StreamingResponse(
        io.BytesIO(response.content),
        media_type="audio/wav",
        headers={
            "X-TTS-Mode": "groq-orpheus",
            "Cache-Control": "private, max-age=3600",
        },
    )


# ─── API Endpoints ───────────────────────────────────────────────────────


@app.get("/api/channels")
async def channel_capabilities():
    """Advertise citizen-facing app, WhatsApp, and IVR readiness."""
    protected = bool(os.getenv("MULTICHANNEL_WEBHOOK_TOKEN", "").strip())
    production = config.deployment.is_production
    return {
        "channels": {
            "app": {
                "status": "ready",
                "surface": "React responsive web app / mobile-web app",
                "capabilities": ["text", "image", "audio", "voice_recording", "case_history"],
            },
            "whatsapp": {
                "status": "ready" if not production or protected else "configuration_required",
                "webhook": "/api/channels/whatsapp",
                "provider_contract": "Twilio/Exotel-style x-www-form-urlencoded webhook",
                "capabilities": ["text_conversation", "localized_reply", "official_reporting_guidance"],
                "protected_by_token": protected,
                "media_analysis_ready": config.deployment.whatsapp_media_integration,
            },
            "ivr": {
                "status": "ready" if not production or config.deployment.ivr_provider_configured else "configuration_required",
                "start_webhook": "/api/channels/ivr/start",
                "analysis_webhook": "/api/channels/ivr/analyze",
                "provider_contract": "TwiML-compatible speech and DTMF flow",
                "capabilities": ["speech_capture", "dtmf_menu", "spoken_verdict", "official_reporting_guidance"],
                "protected_by_token": protected,
                "provider_configured": config.deployment.ivr_provider_configured,
            },
        },
        "language_count": len(SUPPORTED_LANGUAGES),
        "deployment_mode": config.deployment.mode,
        "disclosure": "Channel webhooks provide guidance and risk assessment; they do not file official complaints automatically.",
    }


@app.post("/api/channels/whatsapp")
async def whatsapp_channel(
    request: Request,
    Body: str = Form(""),
    From: str = Form(""),
    ProfileName: str = Form(""),
    WaId: str = Form(""),
    NumMedia: int = Form(0),
    MediaUrl0: str = Form(""),
    MediaContentType0: str = Form(""),
    language: str = Form("en"),
):
    """WhatsApp conversational webhook returning provider-compatible XML."""
    await _verify_channel_webhook(request)
    sender = WaId or From or ProfileName
    if NumMedia and MediaUrl0 and config.deployment.whatsapp_media_integration:
        media = await _download_channel_media(MediaUrl0, MediaContentType0)
        if not orchestrator:
            raise HTTPException(status_code=503, detail="System not initialized")
        if MediaContentType0.startswith("image/"):
            result = await orchestrator.process(
                text=Body or "WhatsApp citizen media report",
                image_bytes=media,
                context={"channel": "whatsapp", "sender": sender, "media_type": MediaContentType0},
                language=normalize_language(language),
            )
            result = _annotate_currency_certification(result)
            _log_to_analytics(result, modality="whatsapp_image")
            return _message(_compact_channel_guidance(result, channel="WhatsApp image", language=language))
        if MediaContentType0.startswith("audio/"):
            result = await orchestrator.process(
                text=Body or "WhatsApp citizen voice-note report",
                audio_bytes=media,
                context={"channel": "whatsapp", "sender": sender, "media_type": MediaContentType0},
                language=normalize_language(language),
            )
            _log_to_analytics(result, modality="whatsapp_audio")
            return _message(_compact_channel_guidance(result, channel="WhatsApp audio", language=language))
        return _message("Media type received but not supported for automated analysis. Preserve the original file and describe the issue in text.")
    if NumMedia and not Body.strip():
        return _message(
            "Media received. WhatsApp media analysis is not enabled for this deployment. Please send a short text description or use the app upload flow."
        )

    try:
        result = await _analyze_channel_text(
            text=Body,
            language=language,
            channel="whatsapp",
            sender=sender,
        )
    except HTTPException as exc:
        if exc.status_code == 400:
            return _message("Send the suspicious WhatsApp/SMS text, call transcript, or payment request here for a fraud-risk check.")
        raise

    reply = _compact_channel_guidance(result, channel="WhatsApp", language=language)
    if NumMedia:
        reply += f" Media noted ({MediaContentType0 or 'attachment'}); preserve the original file as evidence."
    return _message(reply)


@app.api_route("/api/channels/ivr/start", methods=["GET", "POST"])
async def ivr_start(request: Request, language: str = Query("en")):
    """Start an IVR flow for citizens calling a fraud-help number."""
    await _verify_channel_webhook(request)
    lang = normalize_language(language)
    prompt = (
        "Welcome to Digital Public Safety Shield. Briefly describe the suspicious call, message, "
        "payment request, or currency note after the beep. Press 2 for official reporting guidance, "
        "or 9 to repeat this menu."
    )
    gather = (
        f'<Gather input="speech dtmf" timeout="6" speechTimeout="auto" numDigits="1" '
        f'action="/api/channels/ivr/analyze?language={html.escape(lang)}" method="POST">'
        f'{_say(prompt)}</Gather>'
    )
    return _twiml(gather, _say("We did not receive input. Please call again or use the mobile app."))


@app.post("/api/channels/ivr/analyze")
async def ivr_analyze(
    request: Request,
    SpeechResult: str = Form(""),
    Digits: str = Form(""),
    From: str = Form(""),
    language: str = Query("en"),
):
    """Analyze an IVR speech capture and return a spoken verdict."""
    await _verify_channel_webhook(request)
    lang = normalize_language(language)

    if Digits == "9":
        return await ivr_start(request, language=lang)
    if Digits == "2":
        guidance = reporting_guidance("high")
        text = " ".join(guidance["immediate_actions"][:3])
        return _twiml(
            _say(text),
            _say("For cyber fraud, call 1930 or visit cybercrime dot gov dot in. This system does not file a complaint automatically."),
        )
    if not SpeechResult.strip():
        gather = (
            f'<Gather input="speech dtmf" timeout="6" speechTimeout="auto" numDigits="1" '
            f'action="/api/channels/ivr/analyze?language={html.escape(lang)}" method="POST">'
            f'{_say("Please describe the suspicious incident now. Press 2 for reporting guidance.")}</Gather>'
        )
        return _twiml(gather)

    result = await _analyze_channel_text(
        text=SpeechResult,
        language=lang,
        channel="ivr",
        sender=From,
    )
    reply = _compact_channel_guidance(result, channel="IVR", language=lang)
    return _twiml(
        _say(reply),
        _say("To report financial cyber fraud, call 1930 immediately. Preserve evidence and do not share OTP or banking credentials."),
    )


@app.get("/api/health")
async def health_check():
    """System health check with agent status."""
    training_dir = Path(__file__).parent / "data" / "training"
    text_dataset = training_dir / "scam_detection_dataset.json"
    try:
        text_records = len(json.loads(text_dataset.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        text_records = 0
    currency_dir = training_dir / "currency"
    currency_manifest_path = currency_dir / "source_manifest.json"
    currency_images = (
        sum(
            1
            for suffix in ("*.jpg", "*.jpeg", "*.png", "*.webp")
            for _ in currency_dir.rglob(suffix)
        )
        if currency_dir.exists()
        else 0
    )
    try:
        currency_manifest = json.loads(currency_manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        currency_manifest = {}
    trained_models_dir = Path(__file__).parent / "data" / "trained_models"
    vision_model_path = trained_models_dir / "forgery_classifier" / "model.pth"
    vision_metadata_path = vision_model_path.parent / "training_metadata.json"
    xgboost_model_path = trained_models_dir / "xgboost_fusion" / "model.json"
    xgboost_metadata_path = xgboost_model_path.parent / "training_metadata.json"
    text_metadata_path = trained_models_dir / "scam_classifier" / "final" / "training_metadata.json"
    text_benchmark_path = trained_models_dir / "scam_classifier" / "benchmark_metadata.json"

    def load_metadata(path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}

    vision_metadata = load_metadata(vision_metadata_path)
    xgboost_metadata = load_metadata(xgboost_metadata_path)
    text_metadata = load_metadata(text_metadata_path)
    text_benchmark = load_metadata(text_benchmark_path)

    queue_readiness = await broker_status()
    redis_readiness = await redis_status()
    production_readiness = readiness_report(
        queue_status=queue_readiness,
        redis_status=redis_readiness,
        currency_manifest_installed=bool(currency_manifest),
        certified_currency_specimens=certified_currency_count(),
    )
    return {
        "status": "operational" if production_readiness["production_ready"] or not config.deployment.is_production else "not_production_ready",
        "version": config.app_version,
        "app_name": config.app_name,
        "deployment": production_readiness,
        "agents": orchestrator.get_stats()
        if orchestrator
        else {"status": "not_initialized"},
        "capabilities": {
            "languages": len(SUPPORTED_LANGUAGES),
            "geospatial_intelligence": {
                "implemented": True,
                "operational": production_readiness["demo_intelligence_allowed"]
                or all(
                    check["status"] == "ready"
                    for check in production_readiness["checks"]
                    if check["id"] == "geospatial_feeds"
                ),
                "source": "authorized_feeds" if config.deployment.is_production else "demo_or_authorized_feeds",
            },
            "fraud_network_graph": {
                "implemented": True,
                "operational": production_readiness["demo_intelligence_allowed"]
                or all(
                    check["status"] == "ready"
                    for check in production_readiness["checks"]
                    if check["id"] == "graph_feeds"
                ),
                "source": "authorized_feeds" if config.deployment.is_production else "demo_or_authorized_feeds",
            },
            "integrity_hashed_evidence": True,
            "guided_official_reporting": True,
            "realtime_websocket": True,
            "realtime_call_intervention": {
                "session_api": "/api/realtime/sessions",
                "websocket": "/ws/session/{session_id}",
                "signals": [
                    "call_flow", "caller_verification", "stir_shaken_attestation",
                    "spoof_risk", "video_identity", "payment_pressure",
                ],
                "signed_alert_outbox": True,
                "configured_destinations": [
                    destination
                    for destination, value in {
                        "citizen": os.getenv("CITIZEN_ALERT_WEBHOOK_URL", ""),
                        "telecom": os.getenv("TELECOM_ALERT_WEBHOOK_URL", ""),
                        "mha": os.getenv("MHA_ALERT_WEBHOOK_URL", ""),
                    }.items()
                    if value.strip()
                ],
            },
            "currency_field_inspection": {
                "endpoint": "/api/currency/inspect",
                "capture_modes": ["rgb", "uv", "ir", "transmitted"],
                "capture_sides": ["front", "back"],
                "supported_denominations": [10, 20, 50, 100, 200, 500, 2000],
                "non_currency_rejection": True,
                "certification_mode": "screening_only" if certified_currency_count() == 0 else "certified_reference_available",
            },
            "authorized_feed_polling": feed_status(),
            "localized_ai_explanations": True,
            "multi_channel_citizen_ai": {
                "app": True,
                "whatsapp_webhook": "/api/channels/whatsapp",
                "ivr_start_webhook": "/api/channels/ivr/start",
                "ivr_analysis_webhook": "/api/channels/ivr/analyze",
                "webhook_token_required": bool(os.getenv("MULTICHANNEL_WEBHOOK_TOKEN", "").strip()),
            },
            "rabbitmq_jobs": queue_readiness,
            "redis_coordination": redis_readiness,
            "evidence_store": evidence_store_status(),
            "mcp_adapter": {
                "available": True,
                "status": "external_process",
                "transport": "stdio",
                "authenticated_tools": True,
            },
            "opentelemetry": telemetry_status,
        },
        "training_readiness": {
            "text_records": text_records,
            "text_dataset_type": "template_generated",
            "text_training_metadata": text_metadata,
            "text_external_benchmark": text_benchmark,
            "currency_images": currency_images,
            "currency_dataset_ready": currency_images >= 500 and bool(currency_manifest),
            "currency_dataset_verified": False,
            "currency_certified_specimens": certified_currency_count(),
            "currency_label_assurance": currency_manifest.get(
                "label_assurance",
                "No research dataset manifest is installed",
            ),
            "currency_dataset_source": {
                "dataset": currency_manifest.get("source_dataset"),
                "url": currency_manifest.get("source_url"),
                "license": currency_manifest.get("license"),
            }
            if currency_manifest
            else None,
            "supervised_currency_model_ready": vision_model_path.exists(),
            "vision_training_metadata": vision_metadata,
            "xgboost_model_ready": xgboost_model_path.exists(),
            "xgboost_training_metadata": xgboost_metadata,
            "production_accuracy_claimed": False,
        },
        "security_readiness": {
            "jwt_secret_configured": len(os.getenv("JWT_SECRET", "")) >= 32,
            "password_kdf": "PBKDF2-HMAC-SHA256",
            "password_kdf_iterations": 600_000,
            "login_rate_limiting": True,
            "distributed_rate_limiting": redis_readiness["status"] == "ready",
            "security_headers": True,
        },
    }


@app.get("/api/readiness")
async def production_readiness_check():
    """Production readiness gate with explicit blockers."""
    training_dir = Path(__file__).parent / "data" / "training"
    currency_manifest = training_dir / "currency" / "source_manifest.json"
    queue_readiness = await broker_status()
    redis_readiness = await redis_status()
    report = readiness_report(
        queue_status=queue_readiness,
        redis_status=redis_readiness,
        currency_manifest_installed=currency_manifest.exists(),
        certified_currency_specimens=certified_currency_count(),
    )
    status_code = 200 if report["production_ready"] or not config.deployment.is_production else 503
    return JSONResponse(content=report, status_code=status_code)


@app.get("/api/feeds/status")
async def authorized_feed_status():
    return feed_status()


@app.post("/api/feeds/poll")
async def poll_authorized_feeds(
    request: Request,
    user: dict = Depends(get_current_user),
):
    """Run an on-demand authorized feed sync for an authenticated operator."""
    await _verify_ingest_token(request)
    result = await poll_once()
    result["requested_by"] = user["id"]
    return result


@app.post("/api/jobs/analyze/text", status_code=202)
async def submit_text_job(
    request: AsyncTextAnalysisRequest,
    http_request: Request,
    user: dict = Depends(get_current_user),
):
    """Queue a durable text analysis; RabbitMQ receives only the opaque job ID."""
    if not jobs_enabled():
        raise HTTPException(status_code=503, detail="Asynchronous jobs are disabled")
    job = create_job(
        user["id"],
        {
            "text": request.text,
            "context": request.context,
            "language": normalize_language(request.language),
        },
    )
    try:
        await publish_job(job["job_id"], http_request.headers.get("X-Request-ID"))
    except Exception as exc:
        logger.error("Unable to publish analysis job %s: %s", job.get("job_id"), exc)
        fail_job(job["job_id"], "Queue publication failed")
        raise HTTPException(status_code=503, detail="Analysis queue is unavailable") from exc
    return job


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str, user: dict = Depends(get_current_user)):
    job = get_job(user["id"], job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _realtime_metadata(request: RealtimeEventRequest) -> dict[str, Any]:
    metadata = request.model_dump(exclude={"transcript", "metadata"})
    metadata.update(request.metadata)
    return {key: value for key, value in metadata.items() if value is not None}


async def _process_realtime_event(
    session_id: str,
    request: RealtimeEventRequest,
    *,
    audio_bytes: bytes | None = None,
) -> dict[str, Any]:
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    try:
        session = get_realtime_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    previous = get_realtime_transcript(session_id)
    transcript = request.transcript.strip()
    context = _realtime_metadata(request)
    context["channel"] = session["channel"]
    analysis_text = "\n".join(part for part in (previous, transcript) if part).strip()
    if not analysis_text and not audio_bytes:
        raise HTTPException(status_code=400, detail="Transcript or audio chunk is required")

    result = await orchestrator.process(
        text=analysis_text or None,
        audio_bytes=audio_bytes,
        context=context,
        language=normalize_language(session["language"]),
    )
    if audio_bytes and not transcript:
        transcript = (
            result.get("agent_results", {})
            .get("speech", {})
            .get("transcript", {})
            .get("text", "")
        )
    event = append_realtime_event(
        session_id,
        transcript=transcript,
        metadata=context,
        model_score=float(result.get("confidence", 0.0)),
        model_verdict=str(result.get("verdict", "unknown")),
    )
    alerts = ensure_alerts(session_id, event)
    delivered = await dispatch_alerts(alerts) if alerts else []
    return {
        "type": "realtime_verdict",
        "session_id": session_id,
        "event": event,
        "analysis": result,
        "alerts": delivered,
        "pre_transfer_intervention": event["combined_score"] >= float(
            os.getenv("REALTIME_ALERT_THRESHOLD", "0.65")
        ),
    }


@app.post("/api/realtime/sessions", status_code=201)
async def start_realtime_session(request: RealtimeSessionCreate):
    """Start a persistent call/payment risk session for app, IVR, or provider events."""
    return create_realtime_session(
        channel=request.channel,
        language=normalize_language(request.language),
        caller_id=request.caller_id,
        participant_id=request.participant_id,
        metadata=request.metadata,
    )


@app.get("/api/realtime/sessions/{session_id}")
async def realtime_session_status(session_id: str):
    try:
        session = get_realtime_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session["alerts"] = list_alerts(session_id=session_id)
    return session


@app.post("/api/realtime/sessions/{session_id}/events")
async def ingest_realtime_event(session_id: str, request: RealtimeEventRequest):
    """Score the accumulated call flow plus telecom, video, and payment signals."""
    try:
        return await _process_realtime_event(session_id, request)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/realtime/sessions/{session_id}/audio")
async def ingest_realtime_audio(
    session_id: str,
    audio: UploadFile = File(...),
    metadata: str = Form("{}"),
):
    """Transcribe and score a live-call audio chunk with spoof detection."""
    audio_bytes = await _read_upload(audio, kind="audio")
    try:
        fields = json.loads(metadata)
        request = RealtimeEventRequest(**fields)
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="metadata must be valid RealtimeEventRequest JSON") from exc
    return await _process_realtime_event(session_id, request, audio_bytes=audio_bytes)


@app.post("/api/realtime/sessions/{session_id}/close")
async def end_realtime_session(session_id: str):
    try:
        session = close_realtime_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session["alerts"] = list_alerts(session_id=session_id)
    return session


@app.get("/api/realtime/alerts")
async def realtime_alert_outbox(
    session_id: Optional[str] = Query(default=None),
    user: dict = Depends(get_current_user),
):
    """Authorized operator view of signed alert delivery and retry states."""
    return {"alerts": list_alerts(session_id=session_id), "requested_by": user["id"]}


@app.post("/api/analyze")
async def analyze_multimodal(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    language: str = Form("en"),
    capture_mode: str = Form("rgb", pattern="^(rgb|uv|ir|transmitted|tilt_rgb)$"),
    denomination: Optional[str] = Form(None),
    serial_number: Optional[str] = Form(None, max_length=32),
    microtext_ocr: Optional[str] = Form(None, max_length=128),
    note_side: str = Form("unknown", pattern="^(front|back|unknown)$"),
    user: Optional[dict] = Depends(get_optional_user),
):
    """
    Main analysis endpoint — accepts any combination of text, image, audio.
    Routes through the Agentic Fusion Orchestrator for multi-agent analysis.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")

    # Read uploaded files
    image_bytes = None
    audio_bytes = None

    if image:
        image_bytes = await _read_upload(image, kind="image")
        logger.info(f" Received image: {image.filename} ({len(image_bytes)} bytes)")

    if audio:
        audio_bytes = await _read_upload(audio, kind="audio")
        logger.info(f" Received audio: {audio.filename} ({len(audio_bytes)} bytes)")

    if not text and not image_bytes and not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="At least one input (text, image, or audio) is required",
        )

    logger.info(
        f" Processing multimodal input: text={bool(text)}, image={bool(image_bytes)}, audio={bool(audio_bytes)}"
    )

    try:
        result = await orchestrator.process(
            text=text,
            image_bytes=image_bytes,
            audio_bytes=audio_bytes,
            context={
                "capture_mode": capture_mode,
                "denomination": denomination,
                "serial_number": serial_number,
                "microtext_ocr": microtext_ocr,
                "note_side": note_side,
            } if image_bytes else None,
            language=normalize_language(language),
        )
        if image_bytes:
            result = _annotate_currency_certification(result)
        result = _persist_if_user(user, _case_type(text, image_bytes, audio_bytes), result)
        _log_to_analytics(result, modality=_case_type(text, image_bytes, audio_bytes))
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/analyze/text")
async def analyze_text(request: TextAnalysisRequest, user: Optional[dict] = Depends(get_optional_user)):
    """Text-only analysis — scam message/transcript detection."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        result = await orchestrator.process(
            text=request.text,
            context=request.context,
            language=normalize_language(request.language),
        )
        result = _persist_if_user(user, "text", result)
        _log_to_analytics(result, modality="text")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Text analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/image")
async def analyze_image(
    image: UploadFile = File(...),
    language: str = Form("en"),
    capture_mode: str = Form("rgb", pattern="^(rgb|uv|ir|transmitted|tilt_rgb)$"),
    denomination: Optional[str] = Form(None),
    serial_number: Optional[str] = Form(None, max_length=32),
    microtext_ocr: Optional[str] = Form(None, max_length=128),
    note_side: str = Form("unknown", pattern="^(front|back|unknown)$"),
    user: Optional[dict] = Depends(get_optional_user),
):
    """Image-only analysis — counterfeit currency detection."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")

    image_bytes = await _read_upload(image, kind="image")

    try:
        result = await orchestrator.process(
            image_bytes=image_bytes,
            context={
                "capture_mode": capture_mode,
                "denomination": denomination,
                "serial_number": serial_number,
                "microtext_ocr": microtext_ocr,
                "note_side": note_side,
            },
            language=normalize_language(language),
        )
        result = _annotate_currency_certification(result)
        result = _persist_if_user(user, "image", result)
        _log_to_analytics(result, modality="image")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Image analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/currency/inspect")
async def inspect_currency_note(
    front: UploadFile = File(...),
    back: Optional[UploadFile] = File(None),
    uv: Optional[UploadFile] = File(None),
    transmitted: Optional[UploadFile] = File(None),
    tilt: Optional[UploadFile] = File(None),
    ir: Optional[UploadFile] = File(None),
    denomination: Optional[str] = Form(None),
    serial_number: Optional[str] = Form(None, max_length=32),
    microtext_ocr: Optional[str] = Form(None, max_length=128),
    physical_width_mm: Optional[float] = Form(None, ge=40, le=250),
    physical_height_mm: Optional[float] = Form(None, ge=30, le=120),
    thickness_mm: Optional[float] = Form(None, ge=0.01, le=1.0),
    magnetic_thread_detected: Optional[bool] = Form(None),
    double_feed_detected: Optional[bool] = Form(None),
    client_type: str = Form("mobile_app", pattern="^(mobile_app|bank_counting_machine|point_of_sale)$"),
    language: str = Form("en"),
    user: Optional[dict] = Depends(get_optional_user),
):
    """Inspect front/back/UV captures as one field-screening transaction."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    captures: list[tuple[str, bytes, str]] = [
        ("front", await _read_upload(front, kind="image"), "rgb")
    ]
    if back:
        captures.append(("back", await _read_upload(back, kind="image"), "rgb"))
    if uv:
        captures.append(("uv", await _read_upload(uv, kind="image"), "uv"))
    if transmitted:
        captures.append(("transmitted", await _read_upload(transmitted, kind="image"), "transmitted"))
    if tilt:
        captures.append(("tilt", await _read_upload(tilt, kind="image"), "tilt_rgb"))
    if ir:
        captures.append(("ir", await _read_upload(ir, kind="image"), "ir"))

    results: dict[str, dict] = {}
    capture_bytes = {side: image_bytes for side, image_bytes, _ in captures}
    for side, image_bytes, mode in captures:
        result = await orchestrator.process(
            image_bytes=image_bytes,
            context={
                "capture_mode": mode,
                "denomination": denomination,
                "serial_number": serial_number,
                "microtext_ocr": microtext_ocr,
                "note_side": side,
                "physical_width_mm": physical_width_mm,
                "physical_height_mm": physical_height_mm,
                "thickness_mm": thickness_mm,
                "magnetic_thread_detected": magnetic_thread_detected,
                "double_feed_detected": double_feed_detected,
            },
            language=normalize_language(language),
        )
        results[side] = _annotate_currency_certification(result)

    rejected = [side for side, result in results.items() if result.get("verdict") == "invalid_input"]
    valid_results = [
        result for side, result in results.items()
        if side in {"front", "back", "tilt"} and result.get("verdict") != "invalid_input"
    ]
    confidence = max((float(result.get("confidence", 0)) for result in valid_results), default=0.0)
    required_captures = {"front", "back", "uv", "transmitted"}
    if denomination and str(denomination).replace("INR", "").replace("Rs", "").strip() in {"100", "200", "500", "2000"}:
        required_captures.add("tilt")
    complete = required_captures.issubset(results)
    paired_feature_checks = {}
    if "front" in capture_bytes and "tilt" in capture_bytes:
        paired_feature_checks["colour_shift"] = compare_tilt_captures(
            capture_bytes["front"], capture_bytes["tilt"], denomination
        )
    machine_signal_values = {
        "physical_width_mm": physical_width_mm,
        "physical_height_mm": physical_height_mm,
        "thickness_mm": thickness_mm,
        "magnetic_thread_detected": magnetic_thread_detected,
        "double_feed_detected": double_feed_detected,
        "serial_number": serial_number,
        "microtext_ocr": microtext_ocr,
    }
    missing_machine_signals = (
        [name for name, value in machine_signal_values.items() if value is None]
        if client_type == "bank_counting_machine" else []
    )
    colour_shift_failed = any(
        check.get("required") and check.get("status") != "pass"
        for check in paired_feature_checks.values()
    )
    screening_complete = complete and not rejected and not missing_machine_signals and not colour_shift_failed
    if rejected:
        verdict, risk_level = "invalid_input", "invalid_input"
    elif confidence >= 0.65:
        verdict, risk_level = "likely_counterfeit", "high"
    elif not screening_complete or confidence >= 0.35:
        verdict, risk_level = "manual_review", "review"
    else:
        verdict, risk_level = "likely_genuine", "low"
    response = {
        "inspection_id": str(uuid.uuid4()),
        "verdict": verdict,
        "risk_level": risk_level,
        "confidence": round(confidence, 4),
        "denomination": denomination,
        "client_type": client_type,
        "captures_received": list(results),
        "captures_rejected": rejected,
        "screening_complete": screening_complete,
        "required_recaptures": [name for name in sorted(required_captures) if name not in results] + rejected,
        "missing_machine_signals": missing_machine_signals,
        "capture_results": results,
        "paired_feature_checks": paired_feature_checks,
        "deployment_contract": {
            "clients": ["mobile_app", "bank_counting_machine", "point_of_sale"],
            "transport": "multipart HTTPS API",
            "controlled_lane_channels": ["rgb_front", "rgb_back", "uv", "transmitted", "tilt_rgb", "ir_optional"],
            "hardware_signals": ["physical_dimensions", "thickness", "magnetic_thread", "double_feed"],
            "decision_policy": "reject_or_manual_review_on_missing_required_channel",
            "certification": "screening_only unless certified specimen integration is configured",
        },
    }
    response = _persist_if_user(user, "currency_inspection", response)
    _log_to_analytics(response, modality="currency_inspection")
    return response


@app.post("/api/analyze/audio")
async def analyze_audio(
    audio: UploadFile = File(...),
    language: str = Form("en"),
    user: Optional[dict] = Depends(get_optional_user),
):
    """Audio-only analysis — scam call detection + voice spoofing."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")

    audio_bytes = await _read_upload(audio, kind="audio")

    try:
        result = await orchestrator.process(
            audio_bytes=audio_bytes,
            language=normalize_language(language),
        )
        result = _persist_if_user(user, "audio", result)
        _log_to_analytics(result, modality="audio")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Audio analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/turns")
async def analyze_turn_by_turn(request: TurnByTurnRequest):
    """
    Turn-by-turn analysis — shows confidence trajectory climbing over a call.
    Best live-demo feature: paste a scam script turn by turn, watch confidence rise.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        nlp = orchestrator._nlp_agent
        if not nlp._initialized:
            await nlp.initialize()
        trajectory = await nlp.analyze_turn_by_turn(
            request.turns,
            language=normalize_language(request.language),
        )
        return JSONResponse(content={"trajectory": trajectory})
    except Exception as e:
        logger.error(f"Turn-by-turn analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Graph AI Endpoint ──────────────────────────────────────────────────


@app.get("/api/graph/analyze")
async def graph_analysis():
    require_operational_integration("graph")
    """
    Graph AI — Fraud network analysis using Graph Attention Networks.
    Returns GAT node classification, community detection, and network stats.
    """
    try:
        from agents.graph_agent import get_graph_agent

        graph = get_graph_agent()
        await graph.initialize()
        result = graph.analyze_network()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Graph analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/visualization")
async def graph_visualization():
    """Return graph data for frontend network visualization."""
    require_operational_integration("graph")
    try:
        from agents.graph_agent import get_graph_agent

        graph = get_graph_agent()
        await graph.initialize()
        return JSONResponse(content=graph.get_graph_visualization_data())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── WebSocket for Real-time Agent Trace ─────────────────────────────────


class ConnectionManager:
    """WebSocket connection manager for live agent trace streaming."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


ws_manager = ConnectionManager()


@app.websocket("/ws/trace")
async def websocket_trace(websocket: WebSocket):
    """WebSocket endpoint for real-time agent trace updates."""
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Client can send analysis requests via WebSocket too
            try:
                request = json.loads(data)
                text = request.get("text")
                if text and orchestrator:
                    result = await orchestrator.process(text=text)
                    await websocket.send_json(result)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@app.websocket("/ws/session/{session_id}")
async def websocket_session(websocket: WebSocket, session_id: str):
    """Realtime call-flow channel accepting transcripts or base64 audio chunks."""
    await websocket.accept()
    try:
        session = get_realtime_session(session_id)
    except KeyError:
        await websocket.send_json({"type": "error", "message": "Realtime session not found"})
        await websocket.close(code=4404)
        return
    await websocket.send_json({
        "type": "status",
        "session_id": session_id,
        "status": "live",
        "channel": session["channel"],
    })
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong", "session_id": session_id})
                continue

            if message.get("type") not in {"event", "audio_chunk"}:
                await websocket.send_json({"type": "error", "message": "Expected event or audio_chunk"})
                continue
            try:
                request = RealtimeEventRequest(
                    transcript=str(message.get("transcript") or message.get("text") or ""),
                    **(message.get("metadata") or {}),
                )
                audio_bytes = None
                if message.get("type") == "audio_chunk":
                    encoded = str(message.get("audio_base64") or "")
                    audio_bytes = base64.b64decode(encoded, validate=True)
                    if not audio_bytes or len(audio_bytes) > MAX_AUDIO_BYTES:
                        raise ValueError("Invalid audio chunk size")
                await websocket.send_json({"type": "trace", "step": "received"})
                result = await _process_realtime_event(
                    session_id,
                    request,
                    audio_bytes=audio_bytes,
                )
                await websocket.send_json(result)
            except Exception as exc:
                await websocket.send_json({"type": "error", "message": str(exc)[:500]})
    except WebSocketDisconnect:
        logger.info(f"WebSocket session disconnected: {session_id}")


# ─── Demo Data Endpoints ────────────────────────────────────────────────


@app.get("/api/demo/scam-transcript")
async def get_demo_transcript():
    """Return a sample scam transcript for demo purposes."""
    require_operational_integration("demo")
    return {
        "title": "Digital Arrest Scam — Sample Transcript",
        "turns": [
            "Hello, this is Inspector Sharma from CBI Cyber Cell. Am I speaking to the account holder of State Bank account ending 4521?",
            "Sir, we have detected suspicious transactions from your account linked to a money laundering case under PMLA Act.",
            "I need you to stay on this video call for verification. Do NOT disconnect or contact anyone else.",
            "Your Aadhaar has been misused to open 17 fraudulent bank accounts. An arrest warrant has been issued.",
            "To avoid immediate arrest, transfer your funds to an RBI safe custody account. I will share the details now.",
            "The amount needs to be transferred within 30 minutes or I will dispatch officers to your location.",
        ],
        "expected_result": "High confidence scam detection with escalating urgency and authority impersonation.",
    }


@app.get("/api/demo/benign-transcript")
async def get_demo_benign():
    """Return a benign conversation transcript for comparison."""
    require_operational_integration("demo")
    return {
        "title": "Legitimate Customer Service Call — Sample",
        "turns": [
            "Good afternoon, thank you for calling State Bank customer service. My name is Priya, employee ID 45221.",
            "I can see your account details. You mentioned a pending transaction — let me look that up.",
            "I can see the transaction of Rs 5,000 from yesterday. It was a regular UPI transfer.",
            "If you have any concerns, I'd recommend visiting your nearest branch with your ID for a review.",
            "Is there anything else I can help you with today? Thank you for banking with us.",
        ],
        "expected_result": "Low confidence — legitimate customer service interaction.",
    }


# ─── Predictive Threat Intelligence ─────────────────────────────────────


@app.get("/api/intelligence/threat-feed")
async def threat_feed():
    """Real-time threat intelligence feed — driven by actual system analyses."""
    tracker = get_analytics_tracker()
    return tracker.get_live_stats()


@app.get("/api/intelligence/command-centre")
async def command_centre():
    """Unified command centre data."""
    require_operational_integration("geospatial")
    require_operational_integration("graph")
    geo = geospatial_overview()
    graph_stats: dict = {}
    graph_nodes: list[dict] = []
    if orchestrator and orchestrator._graph_agent:
        graph_stats = orchestrator._graph_agent.get_stats()
        graph_nodes = orchestrator._graph_agent.get_graph_visualization_data().get("nodes", [])
    graph_size = graph_stats.get("graph_size", {})
    return {
        "geospatial": geo,
        "resource_plan": command_center_plan(10),
        "network": {
            "total_nodes": graph_size.get("nodes", 0),
            "total_edges": graph_size.get("edges", 0),
            "high_risk_entities": sum(
                node.get("label") in {"scammer", "mule"} for node in graph_nodes
            ),
        },
        "system": {
            "orchestrator": "LangGraph StateGraph (cyclic self-correction)",
            "agents": ["Vision", "Speech", "NLP", "Graph"],
            "languages_supported": len(SUPPORTED_LANGUAGES),
            "uptime_status": "operational" if orchestrator else "initializing",
        },
    }


@app.get("/api/benchmarks")
async def get_benchmarks():
    """Return only metrics recorded by the current local training/evaluation runs."""
    models_dir = Path(__file__).parent / "data" / "trained_models"

    def load_metadata(relative_path: str) -> dict:
        try:
            return json.loads((models_dir / relative_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    text_benchmark = load_metadata("scam_classifier/benchmark_metadata.json")
    vision_training = load_metadata("forgery_classifier/training_metadata.json")
    graph_training = load_metadata("fraud_gat/training_metadata.json")
    fusion_training = load_metadata("xgboost_fusion/training_metadata.json")

    models = []
    if text_benchmark:
        action_metrics = text_benchmark.get("operating_policy", {}).get("action_metrics", {})
        benchmark_metrics = action_metrics or {
            key: text_benchmark[key]
            for key in ("precision", "recall", "f1", "accuracy", "roc_auc", "false_positive_rate")
            if key in text_benchmark
        }
        if "roc_auc" in text_benchmark:
            benchmark_metrics.setdefault("roc_auc", text_benchmark["roc_auc"])
        models.append({
            "name": "Scam Text Classifier",
            "type": "DistilRoBERTa NLI",
            "metrics": benchmark_metrics,
            "evaluation_set": f"{text_benchmark.get('benchmark', 'held-out benchmark')} ({text_benchmark.get('sample_count', 0)} samples; test-only)",
            "limitations": text_benchmark.get("limitations", []),
        })
    if vision_training:
        metrics = vision_training.get("validation_metrics", {})
        models.append({
            "name": "Currency Vision Classifier",
            "type": vision_training.get("architecture", "vision classifier"),
            "metrics": {key: metrics[key] for key in ("accuracy", "precision", "recall", "f1", "roc_auc") if key in metrics},
            "evaluation_set": f"Grouped validation ({vision_training.get('validation_count', 0)} of {vision_training.get('dataset_size', 0)} research-labelled images)",
            "limitations": ["Research labels only; not RBI or forensic certification."],
        })
    if graph_training:
        metrics = graph_training.get("best_metrics", {})
        models.append({
            "name": "Fraud Network GAT",
            "type": graph_training.get("architecture", "Graph Attention Network"),
            "metrics": {key.removeprefix("val_"): value for key, value in metrics.items() if key.startswith("val_")},
            "evaluation_set": f"Validation split of {graph_training.get('graph_nodes', 0)}-node demonstration graph",
            "limitations": ["Demonstration graph; not live law-enforcement intelligence."],
        })
    if fusion_training:
        models.append({
            "name": "XGBoost Fusion",
            "type": fusion_training.get("model", "XGBoost"),
            "metrics": {
                key.removeprefix("validation_"): value
                for key, value in fusion_training.items()
                if key.startswith("validation_")
            },
            "evaluation_set": f"Held-out fusion validation ({fusion_training.get('sample_count', 0)} rows)",
            "limitations": [
                "Deployment quality gate currently enables this model only for image signatures.",
            ],
        })

    return {
        "models": models,
        "system_totals": {"total_models": len(models), "modalities": ["text", "image", "audio", "graph"]},
        "disclosure": "Metrics are loaded from local metadata files and may change after retraining.",
    }


# ─── Run ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info",
    )
