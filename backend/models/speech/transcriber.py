"""
Whisper Transcriber — speech-to-text using OpenAI Whisper.

Uses either:
1. Groq's hosted Whisper endpoint for fast inference.
2. A local Hugging Face Whisper pipeline as an offline fallback.

Supports chunk-based transcription for near-real-time analysis.
"""

import asyncio
import logging
from io import BytesIO
from typing import Optional

from config import config
from localization import normalize_language

logger = logging.getLogger(__name__)


class Transcriber:
    """
    Whisper-based speech-to-text transcription.

    Modes:
    1. Groq API: Fast hosted inference.
    2. Local: Hugging Face Transformers pipeline on GPU or CPU.
    """

    LOCAL_MODEL = "openai/whisper-base"
    MIN_CHUNK_DURATION_SECONDS = 0.5

    def __init__(self) -> None:
        self._pipeline = None
        self._groq_available = False
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the hosted or local Whisper transcription provider."""
        if self._initialized:
            return

        logger.info("Initializing Whisper transcriber...")

        if config.groq.api_key:
            self._groq_available = True
            logger.info("Groq Whisper endpoint available (fast mode)")
        else:
            # No hosted provider is configured, so the local model must be
            # available before initialization can be considered successful.
            await asyncio.to_thread(self._initialize_local_pipeline)

        self._initialized = True

    def _initialize_local_pipeline(self) -> None:
        """Load the local Whisper pipeline when it has not already been loaded."""
        if self._pipeline is not None:
            return

        try:
            import torch
            from transformers import pipeline as hf_pipeline

            device = 0 if torch.cuda.is_available() else -1
            self._pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=self.LOCAL_MODEL,
                device=device,
                chunk_length_s=30,
                return_timestamps=True,
            )
            logger.info(
                "Local Whisper pipeline loaded (%s, device=%s)",
                self.LOCAL_MODEL,
                "cuda" if device == 0 else "cpu",
            )
        except Exception as exc:
            logger.warning("Local Whisper not loaded: %s", exc)
            if not self._groq_available:
                raise RuntimeError("No Whisper model available") from exc

    @staticmethod
    def _validate_audio(audio_data: bytes) -> None:
        """Validate that non-empty binary audio data was supplied."""
        if not isinstance(audio_data, (bytes, bytearray)):
            raise TypeError("audio_data must be bytes or bytearray")
        if not audio_data:
            raise ValueError("audio_data cannot be empty")

    async def transcribe(
        self,
        audio_data: bytes,
        language: str = "en",
        use_groq: bool = True,
    ) -> dict:
        """
        Transcribe audio to text.

        Args:
            audio_data: Raw audio bytes in a supported audio container.
            language: Source-language code.
            use_groq: Try the Groq endpoint before the local model.

        Returns:
            A dictionary containing text, timestamped segments, language,
            duration, and the provider used.
        """
        if not self._initialized:
            raise RuntimeError("Transcriber not initialized")

        self._validate_audio(audio_data)
        normalized_language = normalize_language(language)
        groq_error: Optional[Exception] = None

        if use_groq and self._groq_available:
            try:
                return await self._transcribe_groq(audio_data, normalized_language)
            except Exception as exc:
                groq_error = exc
                logger.warning(
                    "Groq Whisper failed: %s; falling back to local Whisper",
                    exc,
                )

        # Local transcription must also work when Groq is disabled, not
        # configured, or temporarily unavailable.
        if self._pipeline is None:
            try:
                await asyncio.to_thread(self._initialize_local_pipeline)
            except Exception as exc:
                if groq_error is not None:
                    raise RuntimeError(
                        "Groq transcription failed and local Whisper could not be loaded"
                    ) from exc
                raise

        if self._pipeline is not None:
            return await asyncio.to_thread(
                self._transcribe_local_sync,
                bytes(audio_data),
                normalized_language,
            )

        raise RuntimeError("No Whisper transcription provider is available")

    async def transcribe_and_translate(
        self,
        audio_data: bytes,
        language: str = "en",
        use_groq: bool = True,
    ) -> dict:
        """
        Return the source-language transcript and English-normalized text.

        The fraud-classification and retrieval stack can analyze the English
        text while the citizen-facing interface preserves the original speech.
        """
        source_language = normalize_language(language)
        transcript = await self.transcribe(
            audio_data,
            language=source_language,
            use_groq=use_groq,
        )
        original_text = transcript.get("text", "").strip()

        if source_language == "en":
            transcript.update(
                {
                    "original_text": original_text,
                    "english_text": original_text,
                    "translation_provider": transcript.get("provider"),
                    "translated_to_english": False,
                    "analysis_language": "en",
                    "translation_error": None,
                }
            )
            return transcript

        translated_text = ""
        translation_provider = None
        translation_error = None

        if use_groq and self._groq_available:
            try:
                translation = await self._translate_groq(audio_data)
                translated_text = translation.get("text", "").strip()
                translation_provider = translation.get("provider")
            except Exception as exc:
                translation_error = str(exc)
                logger.warning("Groq Whisper translation failed: %s", exc)

        if not translated_text and self._pipeline is None:
            try:
                await asyncio.to_thread(self._initialize_local_pipeline)
            except Exception as exc:
                translation_error = translation_error or str(exc)

        if not translated_text and self._pipeline is not None:
            try:
                translation = await asyncio.to_thread(
                    self._translate_local_sync,
                    bytes(audio_data),
                    source_language,
                )
                translated_text = translation.get("text", "").strip()
                translation_provider = translation.get("provider")
            except Exception as exc:
                translation_error = translation_error or str(exc)
                logger.warning("Local Whisper translation failed: %s", exc)

        transcript.update(
            {
                "original_text": original_text,
                # Preserve the original transcript if translation is unavailable
                # so downstream analysis still receives meaningful text.
                "english_text": translated_text or original_text,
                "translation_provider": translation_provider,
                "translated_to_english": bool(
                    translated_text and translated_text != original_text
                ),
                "analysis_language": "en",
                "translation_error": translation_error,
            }
        )
        return transcript

    async def _transcribe_groq(self, audio_data: bytes, language: str) -> dict:
        """Transcribe audio through Groq's Whisper-compatible endpoint."""
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
            response = await client.post(
                url,
                headers=headers,
                files=files,
                data=data,
            )
            response.raise_for_status()
            result = response.json()

        segments = [
            {
                "start": segment.get("start", 0.0),
                "end": segment.get("end", 0.0),
                "text": segment.get("text", "").strip(),
            }
            for segment in result.get("segments", [])
        ]

        return {
            "text": result.get("text", "").strip(),
            "segments": segments,
            "language": result.get("language", language),
            "duration": result.get("duration", 0.0),
            "provider": "groq",
        }

    async def _translate_groq(self, audio_data: bytes) -> dict:
        """Translate spoken audio to English through Groq's Whisper endpoint."""
        import httpx

        url = f"{config.groq.base_url}/audio/translations"
        headers = {"Authorization": f"Bearer {config.groq.api_key}"}
        filename, content_type = self._audio_file_info(audio_data)
        files = {"file": (filename, audio_data, content_type)}
        data = {
            "model": config.groq.whisper_model,
            "response_format": "verbose_json",
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                url,
                headers=headers,
                files=files,
                data=data,
            )
            response.raise_for_status()
            result = response.json()

        return {
            "text": result.get("text", "").strip(),
            "language": "en",
            "duration": result.get("duration", 0.0),
            "provider": "groq_whisper_translation",
        }

    @staticmethod
    def _audio_file_info(audio_data: bytes) -> tuple[str, str]:
        """Infer a likely filename and MIME type from common container headers."""
        if audio_data.startswith(b"fLaC"):
            return "audio.flac", "audio/flac"
        if audio_data.startswith(b"OggS"):
            return "audio.ogg", "audio/ogg"
        if audio_data.startswith(b"\x1a\x45\xdf\xa3"):
            return "audio.webm", "audio/webm"
        if audio_data.startswith(b"RIFF") and audio_data[8:12] == b"WAVE":
            return "audio.wav", "audio/wav"
        if audio_data.startswith(b"ID3") or audio_data[:2] in {
            b"\xff\xfb",
            b"\xff\xf3",
            b"\xff\xf2",
        }:
            return "audio.mp3", "audio/mpeg"
        if len(audio_data) >= 12 and audio_data[4:8] == b"ftyp":
            return "audio.m4a", "audio/mp4"
        if audio_data[:2] in {
            b"\xff\xf1",
            b"\xff\xf9",
        }:
            return "audio.aac", "audio/aac"

        # WAV is retained as the default for backwards compatibility. The
        # receiving API may still inspect the actual binary content.
        return "audio.wav", "audio/wav"

    @staticmethod
    def _load_audio(audio_data: bytes):
        """Decode audio bytes and return a mono float array and sample rate."""
        import soundfile as sf

        buffer = BytesIO(audio_data)
        try:
            audio_array, sample_rate = sf.read(buffer)
        except Exception:
            import librosa

            buffer.seek(0)
            audio_array, sample_rate = librosa.load(
                buffer,
                sr=16000,
                mono=True,
            )

        if sample_rate <= 0:
            raise ValueError("Invalid audio sample rate")
        if getattr(audio_array, "size", 0) == 0:
            raise ValueError("Decoded audio is empty")
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        return audio_array, sample_rate

    def _transcribe_local_sync(self, audio_data: bytes, language: str) -> dict:
        """Run local Whisper transcription synchronously in a worker thread."""
        if self._pipeline is None:
            raise RuntimeError("Local Whisper pipeline is not initialized")

        audio_array, sample_rate = self._load_audio(audio_data)
        duration = len(audio_array) / sample_rate

        result = self._pipeline(
            {"raw": audio_array, "sampling_rate": sample_rate},
            generate_kwargs={"language": language, "task": "transcribe"},
        )

        segments = []
        for chunk in result.get("chunks", []):
            timestamp = chunk.get("timestamp", (0.0, 0.0))
            if not isinstance(timestamp, (tuple, list)) or len(timestamp) != 2:
                timestamp = (0.0, 0.0)

            segments.append(
                {
                    "start": timestamp[0] if timestamp[0] is not None else 0.0,
                    "end": timestamp[1] if timestamp[1] is not None else 0.0,
                    "text": chunk.get("text", "").strip(),
                }
            )

        return {
            "text": result.get("text", "").strip(),
            "segments": segments,
            "language": language,
            "duration": round(duration, 2),
            "provider": "local",
        }

    def _translate_local_sync(self, audio_data: bytes, language: str) -> dict:
        """Run local Whisper speech translation synchronously in a worker thread."""
        if self._pipeline is None:
            raise RuntimeError("Local Whisper pipeline is not initialized")

        audio_array, sample_rate = self._load_audio(audio_data)
        duration = len(audio_array) / sample_rate

        result = self._pipeline(
            {"raw": audio_array, "sampling_rate": sample_rate},
            generate_kwargs={"language": language, "task": "translate"},
        )

        return {
            "text": result.get("text", "").strip(),
            "language": "en",
            "duration": round(duration, 2),
            "provider": "local_whisper_translation",
        }

    async def transcribe_chunks(
        self,
        audio_data: bytes,
        chunk_duration: float = 5.0,
        language: str = "en",
        use_groq: bool = True,
    ) -> list[dict]:
        """
        Split an audio file into chunks and transcribe each eligible chunk.

        This method processes a complete audio payload chunk by chunk. For true
        live streaming, call it repeatedly as new audio frames arrive or expose
        a WebSocket endpoint that buffers incoming frames.
        """
        if not self._initialized:
            raise RuntimeError("Transcriber not initialized")

        self._validate_audio(audio_data)
        if chunk_duration <= 0:
            raise ValueError("chunk_duration must be greater than zero")

        audio_array, sample_rate = await asyncio.to_thread(
            self._load_audio,
            bytes(audio_data),
        )

        chunk_samples = int(chunk_duration * sample_rate)
        if chunk_samples <= 0:
            raise ValueError("chunk_duration is too small for the audio sample rate")

        minimum_samples = int(self.MIN_CHUNK_DURATION_SECONDS * sample_rate)
        total_samples = len(audio_array)
        chunks: list[dict] = []

        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            chunk = audio_array[start:end]

            if len(chunk) < minimum_samples:
                logger.debug(
                    "Skipping short audio chunk %.2f-%.2f seconds",
                    start / sample_rate,
                    end / sample_rate,
                )
                continue

            chunk_bytes = await asyncio.to_thread(
                self._encode_wav,
                chunk,
                sample_rate,
            )
            result = await self.transcribe(
                chunk_bytes,
                language=language,
                use_groq=use_groq,
            )
            result["chunk_start"] = round(start / sample_rate, 2)
            result["chunk_end"] = round(end / sample_rate, 2)
            chunks.append(result)

        return chunks

    @staticmethod
    def _encode_wav(audio_array, sample_rate: int) -> bytes:
        """Encode an audio array as in-memory WAV bytes."""
        import soundfile as sf

        chunk_buffer = BytesIO()
        sf.write(chunk_buffer, audio_array, sample_rate, format="WAV")
        return chunk_buffer.getvalue()

    def get_stats(self) -> dict:
        """Return the current transcriber configuration and readiness state."""
        return {
            "status": "ready" if self._initialized else "not_initialized",
            "groq_available": self._groq_available,
            "local_model_loaded": self._pipeline is not None,
            "local_model": self.LOCAL_MODEL,
            "groq_model": config.groq.whisper_model,
            "multilingual_input": True,
            "english_normalization": (
                "Groq Whisper translation with local Whisper fallback"
            ),
            "chunk_processing": True,
            "minimum_chunk_duration_seconds": self.MIN_CHUNK_DURATION_SECONDS,
        }


# Module singleton
_transcriber: Optional[Transcriber] = None


def get_transcriber() -> Transcriber:
    """Return the process-wide Transcriber singleton."""
    global _transcriber
    if _transcriber is None:
        _transcriber = Transcriber()
    return _transcriber