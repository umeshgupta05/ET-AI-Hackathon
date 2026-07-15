"""
Whisper Transcriber — Speech-to-text using OpenAI Whisper.

Uses either:
1. Groq's free Whisper endpoint (fastest, for live demo)
2. Local HuggingFace Whisper (offline fallback)

Supports streaming transcription via chunked audio processing.
"""

import logging
import base64
from typing import Optional
from pathlib import Path

import numpy as np

from config import config

logger = logging.getLogger(__name__)


class Transcriber:
    """
    Whisper-based speech-to-text transcription.

    Modes:
    1. Groq API: Ultra-fast inference via Groq's free Whisper endpoint
    2. Local: HuggingFace transformers pipeline (GPU/CPU)
    """

    def __init__(self):
        self._pipeline = None
        self._groq_available = False
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Whisper transcription."""
        if self._initialized:
            return

        logger.info(" Initializing Whisper transcriber...")

        # Check if Groq API is available for fast Whisper
        if config.groq.api_key:
            self._groq_available = True
            logger.info(" Groq Whisper endpoint available (fast mode)")
            self._initialized = True
            return

        self._initialize_local_pipeline()
        self._initialized = True

    def _initialize_local_pipeline(self) -> None:
        """Load local Whisper only when the hosted endpoint is unavailable."""
        try:
            from transformers import pipeline as hf_pipeline

            self._pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",  # Use base for hackathon (faster loading)
                device=-1,  # CPU
                chunk_length_s=30,
                return_timestamps=True,
            )
            logger.info(" Local Whisper pipeline loaded (whisper-base)")
        except Exception as e:
            logger.warning(f"Local Whisper not loaded: {e}")
            if not self._groq_available:
                raise RuntimeError("No Whisper model available")

    async def transcribe(
        self,
        audio_data: bytes,
        language: str = "en",
        use_groq: bool = True,
    ) -> dict:
        """
        Transcribe audio to text.

        Args:
        audio_data: Raw audio bytes (WAV or MP3 format)
        language: Language code
        use_groq: Whether to try Groq API first

        Returns:
        {
        "text": str,
        "segments": [{
        "start": float,
        "end": float,
        "text": str,
        }],
        "language": str,
        "duration": float,
        "provider": "groq" | "local",
        }
        """
        if not self._initialized:
            raise RuntimeError("Transcriber not initialized")

        # Try Groq first (faster)
        if use_groq and self._groq_available:
            try:
                return await self._transcribe_groq(audio_data, language)
            except Exception as e:
                logger.warning(f"Groq Whisper failed: {e}, falling back to local")

            # Fall back to local
            if self._pipeline is None:
                self._initialize_local_pipeline()
            if self._pipeline:
                return await self._transcribe_local(audio_data, language)

        raise RuntimeError("No Whisper transcription available")

    async def _transcribe_groq(self, audio_data: bytes, language: str) -> dict:
        """Transcribe via Groq's free Whisper API endpoint."""
        import httpx

        url = f"{config.groq.base_url}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {config.groq.api_key}"}

        filename, content_type = self._audio_file_info(audio_data)
        files = {"file": (filename, audio_data, content_type)}
        data = {
            "model": config.groq.whisper_model,
            "language": language,
            "response_format": "verbose_json",
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            result = response.json()

            segments = []
            if "segments" in result:
                segments = [
                    {
                        "start": seg.get("start", 0.0),
                        "end": seg.get("end", 0.0),
                        "text": seg.get("text", "").strip(),
                    }
                    for seg in result["segments"]
                ]

            return {
                "text": result.get("text", "").strip(),
                "segments": segments,
                "language": result.get("language", language),
                "duration": result.get("duration", 0.0),
                "provider": "groq",
            }

    @staticmethod
    def _audio_file_info(audio_data: bytes) -> tuple[str, str]:
        """Infer the container so the hosted Whisper API receives accurate metadata."""
        if audio_data.startswith(b"fLaC"):
            return "audio.flac", "audio/flac"
        if audio_data.startswith(b"OggS"):
            return "audio.ogg", "audio/ogg"
        if audio_data.startswith(b"\x1a\x45\xdf\xa3"):
            return "audio.webm", "audio/webm"
        if audio_data.startswith(b"ID3") or audio_data[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}:
            return "audio.mp3", "audio/mpeg"
        return "audio.wav", "audio/wav"

    async def _transcribe_local(self, audio_data: bytes, language: str) -> dict:
        """Transcribe using local HuggingFace Whisper pipeline."""
        import soundfile as sf
        from io import BytesIO

        # Load audio
        buffer = BytesIO(audio_data)
        try:
            audio_array, sample_rate = sf.read(buffer)
        except Exception:
            # Try with librosa if soundfile fails
            import librosa

            buffer.seek(0)
            audio_array, sample_rate = librosa.load(buffer, sr=16000, mono=True)

        # Ensure mono
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        # Run pipeline
        result = self._pipeline(
            {"raw": audio_array, "sampling_rate": sample_rate},
            generate_kwargs={"language": language},
        )

        # Extract segments
        segments = []
        if "chunks" in result:
            for chunk in result["chunks"]:
                timestamp = chunk.get("timestamp", (0.0, 0.0))
                segments.append(
                    {
                        "start": timestamp[0] if timestamp[0] is not None else 0.0,
                        "end": timestamp[1] if timestamp[1] is not None else 0.0,
                        "text": chunk.get("text", "").strip(),
                    }
                )

            duration = len(audio_array) / sample_rate

        return {
            "text": result.get("text", "").strip(),
            "segments": segments,
            "language": language,
            "duration": round(duration, 2),
            "provider": "local",
        }

    async def transcribe_chunks(
        self,
        audio_data: bytes,
        chunk_duration: float = 5.0,
    ) -> list[dict]:
        """
        Transcribe audio in chunks for streaming/real-time analysis.
        Returns list of per-chunk transcriptions.
        """
        import soundfile as sf
        from io import BytesIO

        buffer = BytesIO(audio_data)
        try:
            audio_array, sample_rate = sf.read(buffer)
        except Exception:
            import librosa

            buffer.seek(0)
            audio_array, sample_rate = librosa.load(buffer, sr=16000, mono=True)

        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        chunk_samples = int(chunk_duration * sample_rate)
        total_samples = len(audio_array)
        chunks = []

        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            chunk = audio_array[start:end]

            if len(chunk) < sample_rate * 0.5:  # Skip chunks < 0.5s
                continue

            # Convert chunk to wav bytes
            chunk_buffer = BytesIO()
            sf.write(chunk_buffer, chunk, sample_rate, format="WAV")
            chunk_bytes = chunk_buffer.getvalue()

        result = await self.transcribe(chunk_bytes, use_groq=self._groq_available)
        result["chunk_start"] = round(start / sample_rate, 2)
        result["chunk_end"] = round(end / sample_rate, 2)
        chunks.append(result)

        return chunks

    def get_stats(self) -> dict:
        return {
            "status": "ready" if self._initialized else "not_initialized",
            "groq_available": self._groq_available,
            "local_model": "openai/whisper-base",
            "groq_model": config.groq.whisper_model,
        }


# Module singleton
_transcriber: Optional[Transcriber] = None


def get_transcriber() -> Transcriber:
    global _transcriber
    if _transcriber is None:
        _transcriber = Transcriber()
    return _transcriber
