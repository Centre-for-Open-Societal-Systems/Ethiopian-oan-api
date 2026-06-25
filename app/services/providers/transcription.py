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


class AzureSTTProvider(TranscriptionProvider):
    """Transcription using Azure Speech-to-Text REST API (short audio, <= 60s)."""

    _LANG_MAP = {"en": "en-US", "am": "am-ET", "hi": "hi-IN", "mr": "mr-IN"}

    def __init__(self, api_key: str = None, region: str = None):
        self.api_key = api_key or os.getenv("azure_foundary_api_key")
        self.region = region or os.getenv("azure_foundary_region")
        if not self.api_key or not self.region:
            raise ValueError("Azure STT requires azure_foundary_api_key and azure_foundary_region in env")
        self.endpoint = (
            f"https://{self.region}.stt.speech.microsoft.com"
            f"/speech/recognition/conversation/cognitiveservices/v1"
        )
        logger.info(f"✅ Azure STT Provider initialized: region={self.region}")

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

    def _bcp47(self, lang: str) -> str:
        if "-" in lang:
            return lang
        return self._LANG_MAP.get(lang.lower(), lang)

    @staticmethod
    def _detect_content_type(audio_bytes: bytes) -> str:
        """Pick the Azure-compatible Content-Type from the audio's magic bytes.

        Azure REST short-audio supports audio/wav (PCM) and audio/ogg;codecs=opus.
        WebM/Opus is NOT decoded by the REST endpoint and will return empty text.
        """
        head = audio_bytes[:16]
        if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
            return "audio/wav; codecs=audio/pcm; samplerate=16000"
        if head[:4] == b"OggS":
            return "audio/ogg; codecs=opus"
        return "audio/wav; codecs=audio/pcm; samplerate=16000"

    @staticmethod
    def _inspect_wav(audio_bytes: bytes) -> dict:
        """Parse the WAV fmt chunk + compute peak/RMS of the int16 PCM data.

        Returns {} for non-WAV input. Used purely for diagnostics so we can
        tell silent-mic from sample-rate-mismatch from format errors.
        """
        import struct
        if audio_bytes[:4] != b"RIFF" or audio_bytes[8:12] != b"WAVE":
            return {}
        # Walk RIFF chunks to find 'fmt ' and 'data'.
        i = 12
        fmt = {}
        data_off = data_size = None
        while i + 8 <= len(audio_bytes):
            cid = audio_bytes[i:i+4]
            csize = struct.unpack("<I", audio_bytes[i+4:i+8])[0]
            body = audio_bytes[i+8:i+8+csize]
            if cid == b"fmt " and len(body) >= 16:
                audio_fmt, channels, rate, byte_rate, block_align, bps = struct.unpack("<HHIIHH", body[:16])
                fmt = {"audio_fmt": audio_fmt, "channels": channels, "sample_rate": rate,
                       "bits_per_sample": bps, "byte_rate": byte_rate, "block_align": block_align}
            elif cid == b"data":
                data_off, data_size = i + 8, csize
                break
            # chunks are word-aligned
            i += 8 + csize + (csize & 1)
        if data_off is None or not fmt or fmt.get("bits_per_sample") != 16:
            return fmt
        end = min(data_off + data_size, len(audio_bytes))
        n = (end - data_off) // 2
        if n <= 0:
            return {**fmt, "samples": 0}
        # peak/RMS over a sampled subset to keep it cheap on big files
        step = max(1, n // 4096)
        peak = 0
        sq_sum = 0
        count = 0
        for k in range(0, n, step):
            s = struct.unpack_from("<h", audio_bytes, data_off + 2 * k)[0]
            a = -s if s < 0 else s
            if a > peak:
                peak = a
            sq_sum += s * s
            count += 1
        rms = (sq_sum / count) ** 0.5 if count else 0.0
        duration_s = n / fmt["sample_rate"] / fmt["channels"]
        return {**fmt, "samples": n, "duration_s": round(duration_s, 2),
                "peak_int16": peak, "rms_int16": round(rms, 1)}

    @observe(name="stt.azure", as_type="generation")
    async def transcribe(self, audio_content: str, lang: str = "en") -> str:
        import httpx
        audio_bytes = self.validate_audio(audio_content)
        recognition_lang = self._bcp47(lang)
        content_type = self._detect_content_type(audio_bytes)
        magic = audio_bytes[:8].hex()
        wav_info = self._inspect_wav(audio_bytes)
        logger.info(
            f"Azure STT request: lang={recognition_lang}, bytes={len(audio_bytes)}, "
            f"content_type={content_type}, magic={magic}, wav={wav_info}"
        )
        update_current_observation(
            metadata={
                "provider": "azure",
                "lang": recognition_lang,
                "audio_bytes": len(audio_bytes),
                "content_type": content_type,
            },
        )
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": content_type,
            "Accept": "application/json",
        }
        params = {"language": recognition_lang, "format": "detailed"}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(self.endpoint, params=params, headers=headers, content=audio_bytes)
                resp.raise_for_status()
                data = resp.json()
                status = data.get("RecognitionStatus")
                text = (data.get("DisplayText") or "").strip()
                if not text:
                    logger.warning(f"Azure STT empty result: status={status}, body={data}")
                if status and status != "Success":
                    raise TranscriptionException(f"Azure STT status={status} body={data}")
                logger.info(f"Transcription (azure): '{text[:50]}'")
                update_current_observation(output=text)
                return text
        except httpx.HTTPStatusError as e:
            raise TranscriptionException(f"Azure STT HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise TranscriptionException(str(e))


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


class OmniASRTranscriptionProvider(TranscriptionProvider):
    """Transcription using OmniASR via Triton-compatible HTTP inference API.

    Endpoint: POST {url}/v2/models/{model}/infer
    Auth:     Bearer token via OMNIASR_API_KEY
    Lang map: "am" -> "amh_Ethi", "en" -> "eng_Latn"
    """

    _LANG_MAP = {
        "am": "amh_Ethi",
        "en": "eng_Latn",
        "amh_ethi": "amh_Ethi",
        "eng_latn": "eng_Latn",
    }

    def __init__(self, base_url: str = None, model_name: str = None, api_key: str = None):
        self.base_url = (base_url or os.getenv("OMNIASR_URL", "http://52.66.116.220:8080")).rstrip("/")
        self.model_name = model_name or os.getenv("OMNIASR_MODEL_NAME", "omniasr-amh")
        self.api_key = api_key or os.getenv("OMNIASR_API_KEY")
        self.infer_url = f"{self.base_url}/v2/models/{self.model_name}/infer"
        logger.info(f"✅ OmniASR Transcription Provider initialized: {self.infer_url}")

    def _triton_lang(self, lang: str) -> str:
        return self._LANG_MAP.get(lang.lower(), "amh_Ethi")

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

    @observe(name="stt.omniasr", as_type="generation")
    async def transcribe(self, audio_content: str, lang: str = "am") -> str:
        import httpx
        audio_bytes = self.validate_audio(audio_content)
        triton_lang = self._triton_lang(lang)

        update_current_observation(
            metadata={"provider": "omniasr", "lang": triton_lang, "audio_bytes": len(audio_bytes)},
        )

        payload = {
            "inputs": [
                {
                    "name": "audio_bytes",
                    "datatype": "BYTES",
                    "shape": [1],
                    "data": [base64.b64encode(audio_bytes).decode()],
                },
                {
                    "name": "language",
                    "datatype": "STRING",
                    "shape": [1],
                    "data": [triton_lang],
                },
            ]
        }

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(self.infer_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                outputs = data.get("outputs", [])
                transcript_out = next((o for o in outputs if o.get("name") == "transcript"), None)
                text = (transcript_out["data"][0] if transcript_out and transcript_out.get("data") else "").strip()
                logger.info(f"Transcription (omniasr): '{text[:50]}'")
                update_current_observation(output=text)
                return text
        except httpx.HTTPStatusError as e:
            raise TranscriptionException(f"OmniASR HTTP {e.response.status_code}: {e.response.text}")
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
        provider_name = os.getenv("STT_PROVIDER", "faster_whisper").strip().lower()
        if provider_name == "azure":
            _transcription_provider = AzureSTTProvider()
        elif provider_name == "omniasr":
            _transcription_provider = OmniASRTranscriptionProvider()
        else:
            _transcription_provider = FasterWhisperTranscriptionProvider()
    return _transcription_provider
