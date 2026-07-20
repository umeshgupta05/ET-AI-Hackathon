"""
Speech Agent — Voice Spoofing Detection + Transcription.

Orchestrates:
Audio → Whisper transcription → WavLM spoof detection → Structured output

Two AI models:
1. Whisper (speech-to-text transcription)
2. WavLM/wav2vec2 (voice spoofing / deepfake detection)
"""

import logging
from typing import Optional

import numpy as np

from models.speech.transcriber import get_transcriber
from models.speech.spoof_detector import get_spoof_detector

logger = logging.getLogger(__name__)


class SpeechAgent:
    """
    Multi-model speech analysis agent.

    Handles:
    1. Speech-to-text transcription (Whisper)
    2. Voice spoofing / AI-voice detection (WavLM/AASIST)
    3. Returns both transcript and spoof analysis
    """

    def __init__(self):
        self._transcriber = get_transcriber()
        self._spoof_detector = get_spoof_detector()
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        logger.info(" Initializing Speech Agent...")
        await self._transcriber.initialize()
        await self._spoof_detector.initialize()
        self._initialized = True
        logger.info(" Speech Agent ready (Whisper + WavLM)")

    async def analyze(self, audio_bytes: bytes, language: str = "en") -> dict:
        """
        Full speech analysis: transcribe + detect spoofing.

        Args:
        audio_bytes: Raw audio bytes (WAV, MP3, etc.)

        Returns:
        {
        "agent": "speech",
        "transcript": {...},
        "spoof_detection": {...},
        "techniques_used": [...]
        }
        """
        if not self._initialized:
            await self.initialize()

        logger.info(" Speech Agent analyzing audio...")

        # Step 1: Transcribe + translate to English (NLP classifier is English-trained)
        transcript = await self._transcriber.transcribe_and_translate(
            audio_bytes,
            language=language,
        )

        # Step 2: Spoof detection
        # Load audio array for spoof detector
        spoof_result = {
            "spoof_score": 0.5,
            "verdict": "uncertain",
            "confidence": 0.0,
            "analysis": {},
        }
        try:
            import soundfile as sf
            from io import BytesIO

            buffer = BytesIO(audio_bytes)
            try:
                audio_array, sample_rate = sf.read(buffer)
            except Exception:
                import librosa

                buffer.seek(0)
                audio_array, sample_rate = librosa.load(buffer, sr=16000, mono=True)

            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)

            spoof_result = self._spoof_detector.detect_spoof(audio_array, sample_rate)

        except Exception as e:
            logger.warning(f"Spoof detection failed: {e}")

        return {
            "agent": "speech",
            "transcript": transcript,
            "spoof_detection": spoof_result,
            "techniques_used": [
                "Whisper (speech-to-text)",
                "Whisper speech translation to English",
                "WavLM/wav2vec2 (voice spoof detection)",
            ],
        }

    async def analyze_streaming(
        self, audio_bytes: bytes, chunk_duration: float = 5.0, language: str = "en"
    ) -> list[dict]:
        """
        Streaming analysis: transcribe in chunks with per-chunk spoof detection.
        For live-demo confidence trajectory feature.
        """
        if not self._initialized:
            await self.initialize()

        chunks = await self._transcriber.transcribe_chunks(
            audio_bytes,
            chunk_duration,
            language=language,
        )

        # Add spoof detection to each chunk's time range
        # (simplified: run spoof on full audio, report per chunk)
        try:
            import soundfile as sf
            from io import BytesIO

            buffer = BytesIO(audio_bytes)
            audio_array, sample_rate = sf.read(buffer)
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)

            full_spoof = self._spoof_detector.detect_spoof(audio_array, sample_rate)

            for chunk in chunks:
                chunk["spoof_score"] = full_spoof["spoof_score"]
                chunk["spoof_verdict"] = full_spoof["verdict"]

        except Exception as e:
            for chunk in chunks:
                chunk["spoof_score"] = 0.5
                chunk["spoof_verdict"] = "uncertain"

            return chunks

    def get_stats(self) -> dict:
        return {
            "agent": "speech",
            "status": "ready" if self._initialized else "not_initialized",
            "sub_models": {
                "transcriber": self._transcriber.get_stats(),
                "spoof_detector": self._spoof_detector.get_stats(),
            },
        }
