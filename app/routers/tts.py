from app.services.providers.tts import get_tts_provider
from helpers.utils import get_logger
import uuid
import base64
import struct
from fastapi import APIRouter, HTTPException
from app.models.requests import TTSRequest
from app.models.responses import TTSResponse
from helpers.langfuse_client import observe, update_current_trace, update_current_observation

logger = get_logger(__name__)

router = APIRouter(prefix="/tts", tags=["tts"])

PCM_SAMPLE_RATE = 24000
PCM_CHANNELS = 1
PCM_BITS_PER_SAMPLE = 16


def _pcm_to_wav(pcm: bytes,
                sample_rate: int = PCM_SAMPLE_RATE,
                channels: int = PCM_CHANNELS,
                bits_per_sample: int = PCM_BITS_PER_SAMPLE) -> bytes:
    """Wrap int16 PCM bytes in a RIFF/WAVE header so browsers can play it."""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm)
    return (
        b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                                byte_rate, block_align, bits_per_sample)
        + b"data" + struct.pack("<I", data_size) + pcm
    )

@router.post("/", response_model=TTSResponse)
@observe(name="tts.synthesize", as_type="generation")
async def tts(request: TTSRequest):
    """Handles text to speech conversion using Bhashini service."""

    if not request.text:
        raise HTTPException(status_code=400, detail="text is required")

    update_current_trace(
        name="tts",
        session_id=request.session_id,
        tags=["tts", f"lang:{request.lang_code}"],
    )

    requested_model = (request.model or "").strip() or None

    raw_lang = (request.lang_code or "").lower()
    if requested_model == "mms-tts-orm" or raw_lang.startswith("om") or raw_lang.startswith("orm"):
        raise HTTPException(
            status_code=501,
            detail="Oromo TTS is temporarily disabled. Use an English or Amharic voice for now.",
        )

    try:
        provider = get_tts_provider()
        lang_code = "am" if raw_lang.startswith("am") else "en"
        update_current_observation(
            input=request.text,
            metadata={
                "provider": type(provider).__name__,
                "lang_code": lang_code,
                "voice_id": request.voice_id,
                "model": requested_model,
                "speaker_id": request.speaker_id,
            },
        )
        audio_bytes = await provider._synthesize(
            request.text, lang_code, requested_model, request.speaker_id
        )
        wav_bytes = _pcm_to_wav(audio_bytes)
        logger.info(
            f"TTS synth: voice_id={request.voice_id}, lang={lang_code}, "
            f"model={requested_model or 'default'}, speaker_id={request.speaker_id}, "
            f"pcm_bytes={len(audio_bytes)}, wav_bytes={len(wav_bytes)}"
        )

        # Base64 encode the binary audio data for JSON serialization
        audio_data = base64.b64encode(wav_bytes).decode('utf-8')

        return TTSResponse(
            status='success',
            audio_content=audio_data,
            session_id=request.session_id or str(uuid.uuid4())
        )
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"TTS bad request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")
