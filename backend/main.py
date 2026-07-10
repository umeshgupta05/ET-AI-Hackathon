"""
Digital Public Safety Shield — FastAPI Backend.

Main API server exposing multi-agent AI system via REST + WebSocket.
Accepts image/audio/text input, routes through the Agentic Fusion Orchestrator,
returns structured verdicts with full agent trace.

All AI inference: $0.00 (free-tier APIs + open-weight models).
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # suppress TF warnings

import asyncio
import base64
import io
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

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
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from agents.orchestrator import FusionOrchestrator
from auth_store import (
    SUPPORTED_LANGUAGES,
    authenticate_user,
    create_access_token,
    create_user,
    decode_token,
    get_history,
    init_db,
    save_case,
    update_user,
)

# ─── Logging Setup ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ─── Global Orchestrator ─────────────────────────────────────────────────
orchestrator: Optional[FusionOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize AI models on startup."""
    global orchestrator
    init_db()
    logger.info("=" * 60)
    logger.info(" DIGITAL PUBLIC SAFETY SHIELD")
    logger.info(" Multi-Agent AI System — 17 AI Techniques")
    logger.info(" All Free: Groq + OpenRouter + HuggingFace")
    logger.info("=" * 60)

    orchestrator = FusionOrchestrator()
    try:
        await orchestrator.initialize()
        logger.info(" All systems initialized successfully")
    except Exception as e:
        logger.warning(f" Partial initialization: {e}")
        logger.info("System will initialize remaining models on first use")

    yield

    logger.info(" Shutting down...")


# ─── FastAPI App ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Digital Public Safety Shield",
    description=(
        "AI-powered fraud detection platform using 17 AI techniques: "
        "YOLOv8, EfficientNet, Contrastive Learning, ELA, FFT, NPR, CLIP, Grad-CAM, "
        "Whisper, WavLM/AASIST, Groq GPT-OSS, Llama 4 Scout, DistilBERT, "
        "Hybrid RAG, Multi-Role CoT, Ensemble Fusion, Calibration."
    ),
    version=config.app_version,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ───────────────────────────────────────────


class TextAnalysisRequest(BaseModel):
    text: str
    context: Optional[dict] = None


class TurnByTurnRequest(BaseModel):
    turns: list[str]


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    preferred_language: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    preferred_language: Optional[str] = None


class VoiceSynthesisRequest(BaseModel):
    text: str
    language: str = "en"


class HealthResponse(BaseModel):
    status: str
    version: str
    agents: dict


login_attempts: dict[str, list[float]] = {}


def _rate_limit_login(email: str) -> None:
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
    _rate_limit_login(request.email)
    user = authenticate_user(request.email, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return _auth_payload(user)


@app.get("/api/auth/me")
async def me(user: dict = Depends(get_current_user)):
    return user


@app.patch("/api/auth/me")
async def update_me(request: ProfileUpdateRequest, user: dict = Depends(get_current_user)):
    updated = update_user(user["id"], name=request.name, preferred_language=request.preferred_language)
    return updated


@app.post("/api/auth/logout")
async def logout():
    return {"ok": True}


@app.get("/api/languages")
async def languages():
    return [{"code": code, "name": name} for code, name in SUPPORTED_LANGUAGES.items()]


@app.get("/api/history")
async def history(user: dict = Depends(get_current_user)):
    return {"items": get_history(user["id"])}


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
        audio_bytes = await audio.read()
        result = await transcriber.transcribe(audio_bytes, language=language, use_groq=True)
        return {
            "transcript": result.get("text", ""),
            "detected_language": result.get("language", language),
            "provider": result.get("provider", "unknown"),
            "segments": result.get("segments", []),
        }
    except Exception as e:
        logger.error(f"Voice transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice/synthesize")
async def voice_synthesize(request: VoiceSynthesisRequest):
    """
    Lightweight speech endpoint for hackathon demos.
    Browser speech synthesis is the primary multilingual TTS path; this endpoint
    confirms server readiness and returns a tiny WAV placeholder when requested.
    """
    import wave

    sample_rate = 16000
    duration = 0.12
    frames = int(sample_rate * duration)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"X-TTS-Mode": "browser-speech-synthesis-fallback"},
    )


# ─── API Endpoints ───────────────────────────────────────────────────────


@app.get("/api/health")
async def health_check():
    """System health check with agent status."""
    return {
        "status": "operational",
        "version": config.app_version,
        "app_name": config.app_name,
        "agents": orchestrator.get_stats()
        if orchestrator
        else {"status": "not_initialized"},
    }


@app.post("/api/analyze")
async def analyze_multimodal(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
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
        image_bytes = await image.read()
        logger.info(f" Received image: {image.filename} ({len(image_bytes)} bytes)")

    if audio:
        audio_bytes = await audio.read()
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
        )
        result = _persist_if_user(user, _case_type(text, image_bytes, audio_bytes), result)
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
        result = await orchestrator.process(text=request.text)
        result = _persist_if_user(user, "text", result)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Text analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/image")
async def analyze_image(image: UploadFile = File(...), user: Optional[dict] = Depends(get_optional_user)):
    """Image-only analysis — counterfeit currency detection."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")

    image_bytes = await image.read()

    try:
        result = await orchestrator.process(image_bytes=image_bytes)
        result = _persist_if_user(user, "image", result)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Image analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/audio")
async def analyze_audio(audio: UploadFile = File(...), user: Optional[dict] = Depends(get_optional_user)):
    """Audio-only analysis — scam call detection + voice spoofing."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")

    audio_bytes = await audio.read()

    try:
        result = await orchestrator.process(audio_bytes=audio_bytes)
        result = _persist_if_user(user, "audio", result)
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
        trajectory = await nlp.analyze_turn_by_turn(request.turns)
        return JSONResponse(content={"trajectory": trajectory})
    except Exception as e:
        logger.error(f"Turn-by-turn analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Graph AI Endpoint ──────────────────────────────────────────────────


@app.get("/api/graph/analyze")
async def graph_analysis():
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
    """Session-scoped realtime channel with keepalive and optional analysis."""
    await websocket.accept()
    await websocket.send_json({"type": "status", "session_id": session_id, "status": "live"})
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

            text = message.get("text")
            if text and orchestrator:
                await websocket.send_json({"type": "trace", "step": "received"})
                result = await orchestrator.process(text=text)
                await websocket.send_json({"type": "verdict", "result": result})
    except WebSocketDisconnect:
        logger.info(f"WebSocket session disconnected: {session_id}")


# ─── Demo Data Endpoints ────────────────────────────────────────────────


@app.get("/api/demo/scam-transcript")
async def get_demo_transcript():
    """Return a sample scam transcript for demo purposes."""
    return {
        "title": "Digital Arrest Scam — Sample Transcript",
        "turns": [
            "Hello, this is Inspector Sharma from CBI Cyber Cell. Am I speaking to the account holder of State Bank account ending 4521?",
            "Sir, we have detected suspicious transactions from your account linked to a money laundering case. This is a very serious matter under PMLA Act.",
            "I need you to stay on this video call for the verification process. Do NOT disconnect or contact anyone else — this is a confidential investigation under Section 45 of PMLA.",
            "Your Aadhaar number has been misused to open 17 fraudulent bank accounts. An arrest warrant has already been issued in your name by the Delhi High Court.",
            "To avoid immediate arrest, you need to transfer your funds to an RBI safe custody account for asset verification. This is standard procedure. I will share the account details now.",
            "The amount needs to be transferred within the next 30 minutes or I will have no choice but to dispatch officers to your location for physical arrest.",
        ],
        "expected_result": "High confidence scam detection — escalating urgency, authority impersonation, secrecy demands, and financial pressure all match known digital arrest patterns.",
    }


@app.get("/api/demo/benign-transcript")
async def get_demo_benign():
    """Return a benign conversation transcript for comparison."""
    return {
        "title": "Legitimate Customer Service Call — Sample",
        "turns": [
            "Good afternoon, thank you for calling State Bank customer service. My name is Priya, employee ID 45221. How can I help you today?",
            "I can see your account details. You mentioned a pending transaction — let me look that up for you.",
            "I can see the transaction of Rs 5,000 from yesterday. It was a regular UPI transfer. Would you like me to send you a detailed statement?",
            "If you have any concerns about unauthorized transactions, I'd recommend visiting your nearest branch with your ID for a detailed review. Our branch at MG Road is open until 4 PM.",
            "Is there anything else I can help you with today? Thank you for banking with us.",
        ],
        "expected_result": "Low confidence — legitimate customer service interaction with no scam indicators.",
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
