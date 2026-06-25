# OpenAgriNet AI API

OpenAgriNet is part of Maharashtra's smart farming Digital Public Infrastructure (DPI) initiative. OAN powers oan-api, an AI-driven agricultural assistant that brings expert farming knowledge to every farmer in simple language.

For more information about OpenAgriNet, visit: https://openagrinet.global/

## Features

- Location-based market prices for crops
- Current and upcoming weather information
- Nearest storage facility lookup
- Crop selection guidance by region
- Pest and disease management advice
- Best practices for specific crops

## Benefits

- Multi-language support (Marathi and English)
- 24/7 accessibility via mobile or computer
- Integration with trusted agricultural sources
- Location-specific personalized advice
- Continuous improvement based on farmer feedback

## Data Sources

- Agricultural universities' Package of Practices (PoP)
- IMD (India Meteorological Department) weather data
- APMC (Agricultural Produce Market Committee) market prices
- Registered warehouse database

---

## Getting Started

### Prerequisites

- [Ollama](https://ollama.ai) installed locally (for the LLM)
- [Docker](https://docker.com) running (for all supporting services)
- `conda activate protean` (Python environment with all dependencies)

---

### Step 1 — Start Required Services

**Create a shared Docker network (first time only):**
```bash
docker network create oannetwork
```

**Redis (session cache):**
```bash
docker run -d --name redis-stack --network oannetwork \
    -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

**PostgreSQL (market data):**
```bash
docker compose up postgres -d
```

**Ollama (LLM):**
```bash
ollama serve
ollama pull qwen2.5:7b   # first time only
```

**faster-whisper (voice STT):**
```bash
docker run -d -p 8000:8000 fedirz/faster-whisper-server:latest-cpu
```

**Coqui XTTS (voice TTS):**
```bash
docker run -d -p 8020:8020 daswer123/xtts-api-server
```

**Nominatim (geocoding, optional):**

For detailed setup instructions, system requirements, and troubleshooting, see [docs/nominatim.md](docs/nominatim.md).
```bash
docker-compose up nominatim -d
docker-compose logs -f nominatim   # monitor import (30-60+ min first time)
```

**Marqo (vector search, optional):**
```bash
docker run --name marqo -p 8882:8882 \
    -e MARQO_MAX_CONCURRENT_SEARCH=50 \
    -e VESPA_POOL_SIZE=50 \
    marqoai/marqo:latest
```

---

### Step 2 — Configure `.env`

Create or edit `.env` in the project root:

```ini
# LLM
LLM_PROVIDER=ollama
LLM_MODEL_NAME=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_BASE_URL=http://localhost:11434/v1

# Voice STT
STT_PROVIDER=faster_whisper
FASTER_WHISPER_URL=http://localhost:8000

# Voice TTS
TTS_PROVIDER=coqui_xtts
XTTS_URL=http://localhost:8020
XTTS_SPEAKER_WAV=/path/to/reference_voice.wav   # 3-10s WAV for voice cloning

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/load_agri

# Moderation (optional)
ENABLE_MODERATION=false
```

See `.env.example` for all available options including vLLM and OpenRouter configuration.

---

### Step 3 — Database Migrations

Run migrations to set up the PostgreSQL schema:

```bash
alembic upgrade head
```

Other migration commands:
```bash
alembic current    # check current version
alembic history    # view migration history

# Auto-generate migration from model changes
alembic revision --autogenerate -m "description of changes"
```

---

### Step 4 — Start the API

```bash
cd D:/oan-ai-api-feature-ATI
conda activate protean
python main.py
```

The server starts on `http://localhost:8000`.

---

## Testing

### Health check
Confirms the API is up (no Ollama dependency):
```bash
curl http://localhost:8000/api/health/live
# Expected: {"status": "alive"}
```

### Text chat — LLM + tool calling
Tests the full Ollama → tool dispatch → response path:
```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the price of Teff in Adama?",
    "session_id": "test-1",
    "source_lang": "en",
    "target_lang": "en",
    "user_id": "test"
  }'
```

Expected server logs:
```
LLM call round 1...
Tool call #1: get_crop_price_quick({'crop_name': 'Teff', 'marketplace_name': 'Adama'})
Tool get_crop_price_quick completed in ...ms
Response complete after 2 round(s)
```

### Quickest sanity check
```bash
curl -s -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What crops are available in Adama?",
    "session_id": "test-1",
    "source_lang": "en",
    "target_lang": "en",
    "user_id": "test"
  }' | head -c 500
```
If this returns streamed text with crop/market data, the LLM + tool pipeline is fully working.

### STT smoke test — faster-whisper directly
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@some_audio.wav" \
  -F "model=Systran/faster-whisper-medium" \
  -F "language=en"
# Expected: {"text": "your transcribed speech"}
```

### TTS smoke test — Coqui XTTS directly
```bash
curl -X POST http://localhost:8020/tts_stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello farmer", "language": "en", "speaker_wav": ""}' \
  --output test_out.wav
# Expected: test_out.wav contains audible speech
```

### Voice pipeline — full end-to-end
Connect a WebSocket to `ws://localhost:8000/api/conv/ws?lang=en` and stream PCM audio.

Expected server logs for a complete turn:
```
FasterWhisperSTTService: url=http://localhost:8000
STT: 'price of teff in adama'
Tool call #1: get_crop_price_quick(...)
TTS START: 'The current price...'
TTS: Complete (N frames)
```

---

## Switching Providers

All providers are swappable via `.env` without code changes:

| Service | Env var | Options |
|---|---|---|
| LLM | `LLM_PROVIDER` | `ollama` · `vllm` · `openai` |
| STT | `STT_PROVIDER` | `faster_whisper` |
| TTS | `TTS_PROVIDER` | `coqui_xtts` |

---

## Market Data Scraper

Syncs crop and livestock prices from NMIS API (nmis.et) with bilingual support (English/Amharic). Clears prices and logs each run to database.

```bash
python scripts/run_all_scrapers.py
```

Cron (daily 6 AM):
```
0 6 * * * cd /path/to/load_agri && python scripts/run_all_scrapers.py >> /var/log/scraper.log 2>&1
```

---

## Deploying with Docker Compose

Start the full application stack:
```bash
docker compose up --build --force-recreate --detach
```

Stop the application:
```bash
docker compose down --remove-orphans
```

View application logs:
```bash
docker logs -f oan_app
```

---

## API Reference

### Base URL
```
http://localhost:8000
```

### Authentication
All endpoints require JWT authentication. Include the JWT token in the Authorization header:
```
Authorization: Bearer your_jwt_token
```

### Available Endpoints

#### 1. Transcribe Audio
```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_jwt_token" \
  -d '{
    "audio_content": "your_base64_encoded_audio",
    "session_id": "optional_session_id"
  }'
```

#### 2. Text to Speech
```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_jwt_token" \
  -d '{
    "text": "your text here",
    "target_lang": "mr",
    "session_id": "optional_session_id"
  }'
```

#### 3. Chat
```bash
curl -X GET "http://localhost:8000/chat?query=your_question&session_id=optional_session_id&source_lang=mr&target_lang=mr&user_id=user123" \
  -H "Authorization: Bearer your_jwt_token"
```

#### 4. Suggestions
```bash
curl -X GET "http://localhost:8000/suggestions?session_id=your_session_id&target_lang=mr" \
  -H "Authorization: Bearer your_jwt_token"
```

### Supported Languages
- Marathi (mr)
- English (en)

For detailed API documentation including request/response schemas, please see [API_REFERENCE.md](docs/api.md)

## Contributing

Contribution guidelines will be added soon.
