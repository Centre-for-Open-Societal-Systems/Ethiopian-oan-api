from app.services.providers.tts import get_tts_provider
from helpers.utils import get_logger
import uuid
import base64
from fastapi import APIRouter, HTTPException
from app.models.requests import TTSRequest
from app.models.responses import TTSResponse
from helpers.langfuse_client import observe, update_current_trace, update_current_observation

logger = get_logger(__name__)

router = APIRouter(prefix="/tts", tags=["tts"])

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

    try:
        provider = get_tts_provider()
        lang_code = "en" if request.lang_code.startswith("en") else "am"
        update_current_observation(
            input=request.text,
            metadata={"provider": type(provider).__name__, "lang_code": lang_code},
        )
        audio_bytes = await provider._synthesize(request.text, lang_code)

        # Base64 encode the binary audio data for JSON serialization
        audio_data = base64.b64encode(audio_bytes).decode('utf-8')

        return TTSResponse(
            status='success',
            audio_content=audio_data,
            session_id=request.session_id or str(uuid.uuid4())
        )
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")
