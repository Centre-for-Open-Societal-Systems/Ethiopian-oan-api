import uuid
from app.services.providers.transcription import get_transcription_provider
from helpers.utils import get_logger
from fastapi import APIRouter, HTTPException
from app.models.requests import TranscribeRequest
from app.models.responses import TranscribeResponse
from app.services.pii_masker import pii_masker
from helpers.langfuse_client import observe, update_current_trace, update_current_observation

logger = get_logger(__name__)

router = APIRouter(prefix="/transcribe", tags=["transcribe"])

@router.post("/", response_model=TranscribeResponse)
@observe(name="stt.transcribe", as_type="generation")
async def transcribe(request: TranscribeRequest):
    """Handles transcription of audio using the configured transcription provider."""

    if not request.audio_content:
        raise HTTPException(status_code=400, detail="audio_content is required")

    update_current_trace(
        name="stt",
        session_id=request.session_id,
        tags=["stt"],
    )

    try:
        provider = get_transcription_provider()
        transcription = await provider.transcribe(request.audio_content)
        logger.info(f"Transcription: {pii_masker.mask(transcription)}")
        update_current_observation(
            output=transcription,
            metadata={"provider": type(provider).__name__},
        )

        return TranscribeResponse(
            status='success',
            text=transcription,
            lang_code='en',
            session_id=request.session_id or str(uuid.uuid4())
        )
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
