"""
Production-Ready Transcription Provider
"""

import base64
import os
from typing import Optional
from abc import ABC, abstractmethod
from helpers.utils import get_logger
from helpers.langfuse_client import observe, update_current_observation
logger = get_logger(__name__)


# Custom Exceptions
class TranscriptionException(Exception):
    """Base exception for transcription errors"""
    pass


class ModelLoadException(TranscriptionException):
    """Exception raised when model fails to load"""
    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Failed to load model: {model_name}")


class InvalidAudioException(TranscriptionException):
    """Exception raised for invalid audio input"""
    pass


class TranscriptionProvider(ABC):
    """Abstract base class for transcription providers"""

    @abstractmethod
    async def transcribe(self, audio_content: str, lang: str = "en") -> str:
        """
        Transcribe audio content

        Args:
            audio_content: Base64 encoded audio string
            lang: Language code

        Returns:
            str: Transcribed text
        """
        pass

    @abstractmethod
    def validate_audio(self, audio_content: str) -> bool:
        """Validate audio input"""
        pass


class FasterWhisperTranscriptionProvider(TranscriptionProvider):
    """Transcription using faster-whisper-server (OpenAI-compatible /v1/audio/transcriptions)."""

    def __init__(self, base_url: str = None):
        import httpx  # noqa: F401 — ensure httpx is available
        self.base_url = (base_url or os.getenv("FASTER_WHISPER_URL", "http://localhost:8000")).rstrip('/')
        logger.info(f"✅ FasterWhisper Transcription Provider initialized: {self.base_url}")

    def validate_audio(self, audio_content: str) -> bytes:
        if not audio_content:
            raise InvalidAudioException("Audio content is empty")
        try:
            audio_bytes = base64.b64decode(audio_content)
            if len(audio_bytes) > 50 * 1024 * 1024:
                raise InvalidAudioException("Audio too large (max 50MB)")
            if len(audio_bytes) < 100:
                raise InvalidAudioException("Audio data too short")
            return audio_bytes
        except base64.binascii.Error as e:
            raise InvalidAudioException(f"Invalid base64 audio data: {e}")

    @observe(name="stt.faster_whisper", as_type="generation")
    async def transcribe(self, audio_content: str, lang: str = "en") -> str:
        import httpx
        audio_bytes = self.validate_audio(audio_content)
        whisper_lang = lang.split("-")[0]
        update_current_observation(
            metadata={"provider": "faster_whisper", "lang": whisper_lang, "audio_bytes": len(audio_bytes)},
        )
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.base_url}/v1/audio/transcriptions",
                    files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                    data={"model": os.getenv("FASTER_WHISPER_MODEL", "Systran/faster-whisper-medium"),
                          "language": whisper_lang}
                )
                resp.raise_for_status()
                text = resp.json().get("text", "").strip()
                logger.info(f"Transcription (faster-whisper): '{text[:50]}'")
                update_current_observation(output=text)
                return text
        except Exception as e:
            raise TranscriptionException(str(e))


# Singleton instance - initialized once at startup
_transcription_provider: Optional[TranscriptionProvider] = None


def get_transcription_provider() -> TranscriptionProvider:
    """
    Get or create transcription provider singleton

    Returns:
        TranscriptionProvider: Transcription provider instance
    """
    global _transcription_provider
    if _transcription_provider is None:
        _transcription_provider = FasterWhisperTranscriptionProvider()
    return _transcription_provider
