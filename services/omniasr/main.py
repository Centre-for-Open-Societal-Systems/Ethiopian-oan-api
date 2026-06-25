"""
omniASR wrapper service — exposes POST /v1/audio/transcriptions
(same interface as faster-whisper-server) so the main app needs zero changes.

Start: uvicorn main:app --host 0.0.0.0 --port 8001
Env:
  OMNIASR_MODEL  — model card name (default: omniASR_LLM_7B_v2)
                   CPU fallback: omniASR_CTC_300M_v2
"""

import os
import tempfile

from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="omniASR wrapper", version="1.0.0")

MODEL_CARD = os.getenv("OMNIASR_MODEL", "omniASR_LLM_7B_v2")

# Populated at startup — avoids loading the model on every import.
pipeline = None

# BCP-47 short code → fairseq2 FLORES-200 language tag
LANG_MAP: dict[str, str] = {
    "am": "amh_Ethi",   # Amharic
    "en": "eng_Latn",   # English
    "mr": "mar_Deva",   # Marathi
    "hi": "hin_Deva",   # Hindi
    "sw": "swh_Latn",   # Swahili
    "ti": "tir_Ethi",   # Tigrinya
    "or": "ory_Orya",   # Odia
    "kn": "kan_Knda",   # Kannada
    "ta": "tam_Taml",   # Tamil
    "te": "tel_Telu",   # Telugu
    "gu": "guj_Gujr",   # Gujarati
    "pa": "pan_Guru",   # Punjabi
    "bn": "ben_Beng",   # Bengali
    "ml": "mal_Mlym",   # Malayalam
    "ur": "urd_Arab",   # Urdu
    "ar": "arb_Arab",   # Arabic
    "fr": "fra_Latn",   # French
    "de": "deu_Latn",   # German
    "es": "spa_Latn",   # Spanish
    "pt": "por_Latn",   # Portuguese
    "zh": "zho_Hans",   # Chinese (Simplified)
    "ja": "jpn_Jpan",   # Japanese
    "ko": "kor_Hang",   # Korean
    "ru": "rus_Cyrl",   # Russian
}


@app.on_event("startup")
async def load_model() -> None:
    global pipeline
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline  # noqa: PLC0415

    print(f"[omniASR] Loading model: {MODEL_CARD}")
    pipeline = ASRInferencePipeline(model_card=MODEL_CARD)
    print(f"[omniASR] Model ready: {MODEL_CARD}")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": MODEL_CARD, "ready": pipeline is not None}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile,
    language: str = Form("en"),
    model: str = Form(""),          # ignored — model fixed at startup
    response_format: str = Form("json"),
) -> JSONResponse:
    if pipeline is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded yet"})

    # Map BCP-47 short code (e.g. "am-ET" → "am") to FLORES-200 tag
    lang_short = language.split("-")[0].lower()
    lang_code = LANG_MAP.get(lang_short, "eng_Latn")

    audio_bytes = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        results = pipeline.transcribe([tmp_path], lang=[lang_code], batch_size=1)
        text: str = results[0] if results else ""
    finally:
        os.unlink(tmp_path)

    return JSONResponse({"text": text.strip()})
