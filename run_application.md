# OAN AI API — Run Guide

Complete step-by-step guide to get the OAN AI API running from scratch.

For more information about OpenAgriNet, visit: https://openagrinet.global/

---

## Quick Start (Single Command)

After cloning the repo and filling in `.env`, run everything with one command:

```bash
bash start.sh
```

This handles all steps in sequence: starts Docker services → runs migrations → scrapes NMIS data → indexes Cosdata → starts the app.

**Flags (for subsequent runs when data already exists):**

```bash
bash start.sh --skip-scrape             # Skip NMIS scraping
bash start.sh --skip-index              # Skip Cosdata indexing
bash start.sh --skip-scrape --skip-index  # Just start the app (fastest)
```

> **Prerequisites before running:** Docker must be running, Ollama must be running (`ollama serve`), and `.env` must be configured. See the steps below for details.

---

## Prerequisites

- [Docker](https://docker.com) installed and running
- [Ollama](https://ollama.ai) installed locally
- 
Python 3.10+ environment (`conda activate protean`)
- Git

---

## Overview of Services

| Service | Port | Purpose |
|---|---|---|
| PostgreSQL | 5432 | Primary database (market prices, scraped data) |
| Redis | 6379 | Session cache & conversation history |
| Cosdata | 8443 | Vector database for RAG (semantic search) |
| faster-whisper-server | 8010 | Open-source STT (Speech-to-Text) |
| Ollama | 11434 | Local LLM inference |
| OAN AI API (FastAPI) | 8000 | Main application |

---

## Step 1 — Clone & Configure

```bash
git clone <repo-url>
cd oan-ai-api-feature-ATI
cp .env.example .env
```

Edit `.env` and fill in the required values. The minimum configuration:

```env
# LLM
LLM_PROVIDER=ollama
LLM_MODEL_NAME=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
OPENAI_BASE_URL=http://localhost:11434/v1

# Voice STT
STT_PROVIDER=faster_whisper
FASTER_WHISPER_URL=http://localhost:8010
FASTER_WHISPER_MODEL=Systran/faster-whisper-medium

# Voice TTS (local VITS models via HuggingFace — no external server needed)
TTS_MODEL_AM=facebook/mms-tts-amh    # Any HF VITS model ID for Amharic
TTS_MODEL_EN=facebook/mms-tts-eng    # Any HF VITS model ID for English
TTS_SAMPLE_RATE=16000                 # Native output rate of models
TTS_DEVICE=cpu                        # "cpu" or "cuda"

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/oan

# RAG (Vector Search)
RAG_PROVIDER=cosdata
COSDATA_ENDPOINT_URL=http://127.0.0.1:8443
COSDATA_USERNAME=admin
COSDATA_PASSWORD=admin
COSDATA_COLLECTION_NAME=oan-collection
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-large

# Scraper
SCRAPER_ENABLED=true

# Moderation
ENABLE_MODERATION=false
```

---

## Step 2 — Start Infrastructure Services (Docker)

### Create shared Docker network (first time only)

```bash
docker network create oannetwork
```

### Redis (session cache)

```bash
docker run -d --name redis-stack --network oannetwork \
    -p 6379:6379 -p 8001:8001 \
    redis/redis-stack:latest
```

### PostgreSQL (market data)

```bash
docker compose up postgres -d
```

### Cosdata (vector database)

```bash
docker compose up cosdata -d
```

Or run Cosdata standalone:

```bash
docker run -d --name oan_cosdata --network oannetwork \
    -p 8443:8443 -p 50051:50051 \
    -v cosdata_data:/opt/cosdata/data \
    cosdatateam/cosdata:latest \
    /opt/cosdata/bin/cosdata --admin-key admin --skip-confirmation
```

### Verify infrastructure is healthy

```bash
docker compose ps
```

**PostgreSQL:**
```bash
docker exec oan_postgres pg_isready -U postgres
# Expected: /var/run/postgresql:5432 - accepting connections
```

**Redis:**
```bash
docker exec redis-stack redis-cli ping
# Expected: PONG
```

**Cosdata:**
```bash
curl http://localhost:8443/vectordb/collections
# Expected: "Invalid auth token!" (means it's running)
```

---

## Step 3 — Start AI Inference Services

Open a separate terminal for each service.

### Ollama (LLM)

Ollama starts automatically as a system service after install. If it's not running:

```bash
ollama serve
```

Pull the model (first time only — ~4GB download):

```bash
ollama pull qwen2.5:7b
```

Verify:
```bash
curl http://localhost:11434/api/tags
```

### faster-whisper-server (STT)

```bash
docker run -d --name faster-whisper --network oannetwork \
    -p 8010:8000 \
    fedirz/faster-whisper-server:latest-cpu
```

> On GPU (if you have CUDA):
> ```bash
> docker run -d --name faster-whisper --network oannetwork \
>     -p 8010:8000 --gpus all \
>     fedirz/faster-whisper-server:latest-cuda
> ```

Verify:
```bash
curl http://localhost:8010/health
# Expected: OK
```

### TTS (MMS-TTS — runs locally, no Docker needed)

TTS now runs inside the main application using HuggingFace VITS models (`facebook/mms-tts-amh`, `facebook/mms-tts-eng`). No separate server or Docker container is needed.

Models are downloaded automatically on first use (~300-500MB per language) and cached at `~/.cache/huggingface/hub/`.

To use a different model, change the env var:
```env
TTS_MODEL_AM=facebook/mms-tts-orm   # e.g. switch Amharic to Oromo
```

---

## Step 4 — Install Python Dependencies & Run Migrations

```bash
conda activate protean
pip install -r requirements.txt
```

Run database migrations to create all tables:

```bash
alembic upgrade head
```

Expected output:
```
INFO  [alembic.runtime.migration] Running upgrade  -> bf13d07dbb2d, initial schema
```

Other useful migration commands:
```bash
alembic current     # check current migration version
alembic history     # view all migration history

# Auto-generate a new migration after model changes
alembic revision --autogenerate -m "description"
```

---

## Step 5 — Pull Market Data from NMIS

Scrape all agricultural market data (crops, livestock, prices) from the Ethiopian NMIS API and sync to PostgreSQL. **This takes 10–20 minutes on first run.**

```bash
python scripts/run_all_scrapers.py
```

This runs 9 scrapers in sequence:

| Scraper | What it syncs |
|---|---|
| `marketplaces` | All crop & livestock market locations (309 crop, 22 livestock) |
| `crops` | Crop types (teff, wheat, maize, etc.) |
| `livestock` | Livestock types (ox, cattle, sheep, goat, etc.) |
| `crop_varieties` | Crop variety data per marketplace |
| `livestock_varieties` | Livestock breed data per marketplace |
| `crop_prices` | Latest crop prices per marketplace |
| `livestock_prices` | Latest livestock prices per marketplace |
| `crop_prices_collected_at` | Historical crop price dates |
| `livestock_prices_collected_at` | Historical livestock price dates |

Expected final output:
```
Completed in Xs
  ✓ marketplaces
  ✓ crops
  ✓ livestock
  ✓ crop_varieties
  ✓ livestock_varieties
  ✓ crop_prices
  ✓ livestock_prices
  ✓ crop_prices_collected_at
  ✓ livestock_prices_collected_at
```

> **Note:** This script is safe to re-run at any time — all scrapers use upserts, no data is deleted. Run it on a daily schedule to keep prices current:
> ```bash
> # Cron — every day at 6 AM
> 0 6 * * * cd /path/to/oan-ai-api && python scripts/run_all_scrapers.py >> /var/log/scraper.log 2>&1
> ```

---

## Step 6 — Index Documents into Cosdata (RAG)

Index the agricultural knowledge base into Cosdata for semantic search (RAG).

```bash
python scripts/index_cosdata.py --file assets/all_agricultural_docs.json
```

**First run** downloads the embedding model (~1.1GB):
```
INFO - Loading embedding model: intfloat/multilingual-e5-large
INFO - Model loaded. Embedding dimension: 1024
```

Then creates the collection and indexes 47 documents:
```
INFO - Creating new collection: oan-collection with dimension 1024
INFO - Index created successfully
INFO - Indexing 47 documents in batches of 50
INFO - Batch 1 indexed successfully
INFO - Indexing complete. Total documents indexed: 47
```

To recreate the collection from scratch (e.g. after updating documents):
```bash
python scripts/index_cosdata.py --file assets/all_agricultural_docs.json --recreate
```

To verify the index works after indexing:
```bash
python scripts/index_cosdata.py --file assets/all_agricultural_docs.json --verify
```

> **Note:** The embedding model (`intfloat/multilingual-e5-large`, ~1.1GB) is downloaded from HuggingFace on first run and cached at `~/.cache/huggingface/hub/`. Subsequent runs (and the running app) use the local cache — no re-download needed.

---

## Step 7 — Start the Application

### Local (development)

```bash
conda activate protean
cd D:/oan-ai-api-feature-ATI
python main.py
```

The server starts on `http://localhost:8000`.

### Docker (production)

Build and start the app container (after completing Steps 2–6):

```bash
docker compose up --build -d app
```

To also run scrapers automatically on every startup:
```bash
RUN_SCRAPERS_ON_STARTUP=true docker compose up --build -d app
```

Full stack (build everything):
```bash
docker compose up --build --force-recreate --detach
```

View logs:
```bash
docker logs -f oan_app
```

Stop everything:
```bash
docker compose down --remove-orphans
```

---

## Step 8 — Verify Everything Works

### Health check
```bash
curl http://localhost:8000/api/health/live
# Expected: {"status":"alive"}
```

### Text chat (English)
```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the price of oxen in Nagelle Borana?",
    "source_lang": "en",
    "target_lang": "en",
    "user_id": "test"
  }'
```

Expected response:
```json
{
  "response": "Oxen in Nagelle Borana are trading between 70,000 and 88,000 Birr...",
  "status": "success",
  "metrics": { "tool_calls": 1, "total_e2e_latency": 5000 }
}
```

### STT smoke test
```bash
curl -X POST http://localhost:8010/v1/audio/transcriptions \
  -F "file=@some_audio.wav" \
  -F "model=Systran/faster-whisper-medium" \
  -F "language=en"
# Expected: {"text": "your transcribed speech"}
```

### Voice pipeline (WebSocket)

The voice pipeline is available at:
```
ws://localhost:8000/api/conv/ws?lang=en   # English
ws://localhost:8000/api/conv/ws?lang=am   # Amharic
```

Stream raw **float32 PCM audio at 16kHz** in 512-sample chunks. The server returns JSON events:

| Event | Meaning |
|---|---|
| `{"type": "speech_start"}` | VAD detected voice start |
| `{"type": "speech_end"}` | VAD detected voice stop |
| `{"type": "transcription", "text": "...", "is_final": true}` | STT transcript |
| `{"type": "llm_chunk", "text": "..."}` | AI response text |
| `{"type": "metrics", "data": {...}}` | Latency breakdown |
| binary bytes | TTS audio (raw PCM at 24kHz) |

Expected server logs for a complete voice turn:
```
🟢 Speech STARTED
FasterWhisperSTTService: transcribed 'price of teff in Adama'
🚀 AgriNet: Proceeding with query: 'price of teff in Adama'
Tool call: get_livestock_price_quick(...)
📤 Sending response to frontend
```

---

## Switching Providers

All providers are swappable via `.env` without code changes:

| Service | Env var | Options |
|---|---|---|
| LLM | `LLM_PROVIDER` | `ollama` · `openai` · `openrouter` |
| STT | `STT_PROVIDER` | `faster_whisper` |
| TTS | `TTS_MODEL_AM` / `TTS_MODEL_EN` | Any HuggingFace VITS model ID |
| RAG | `RAG_PROVIDER` | `cosdata` · `marqo` |

---

## Troubleshooting

### "Collection Not Found" on RAG queries
The Cosdata collection hasn't been indexed yet. Run Step 6.

### Chat responses are slow (>10s)
- Ollama loads the model on first request (~5s cold start) — subsequent requests are fast
- Check GPU availability: `nvidia-smi`
- If running CPU-only, expect 10–30s per response

### STT returns garbled text
- For English: ensure `FASTER_WHISPER_MODEL=Systran/faster-whisper-medium`
- For Amharic: medium model has poor accuracy; a fine-tuned Amharic Whisper model is recommended

### Scrapers fail mid-run
Each scraper is independent — a failure is logged and the run continues. Re-running the script is safe due to upserts.

### UnicodeEncodeError on Windows
```bash
set PYTHONIOENCODING=utf-8
python scripts/run_all_scrapers.py
```

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/load_agri` | PostgreSQL connection string |
| `LLM_PROVIDER` | `ollama` | LLM backend: `ollama`, `openai`, `openrouter` |
| `LLM_MODEL_NAME` | `qwen2.5:7b` | Model name for the chosen LLM provider |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OPENAI_BASE_URL` | — | OpenAI-compatible API base URL |
| `OPENAI_API_KEY` | — | API key for OpenAI-compatible provider |
| `STT_PROVIDER` | `faster_whisper` | STT backend: `faster_whisper` |
| `FASTER_WHISPER_URL` | `http://localhost:8010` | faster-whisper-server URL |
| `FASTER_WHISPER_MODEL` | `Systran/faster-whisper-medium` | Whisper model to use for transcription |
| `TTS_MODEL_AM` | `facebook/mms-tts-amh` | HuggingFace VITS model ID for Amharic TTS |
| `TTS_MODEL_EN` | `facebook/mms-tts-eng` | HuggingFace VITS model ID for English TTS |
| `TTS_SAMPLE_RATE` | `16000` | Native output sample rate of TTS models |
| `TTS_DEVICE` | `cpu` | Device for TTS inference: `cpu` or `cuda` |
| `RAG_PROVIDER` | `cosdata` | RAG backend: `cosdata` or `marqo` |
| `COSDATA_ENDPOINT_URL` | `http://127.0.0.1:8443` | Cosdata server URL |
| `COSDATA_USERNAME` | `admin` | Cosdata username |
| `COSDATA_PASSWORD` | `admin` | Cosdata password |
| `COSDATA_COLLECTION_NAME` | `oan-collection` | Cosdata collection name |
| `EMBEDDING_MODEL_NAME` | `intfloat/multilingual-e5-large` | Sentence transformer model for embeddings |
| `ENABLE_MODERATION` | `false` | Enable content moderation classifier |
| `SCRAPER_ENABLED` | `true` | Enable NMIS data scraper |
| `RUN_SCRAPERS_ON_STARTUP` | `false` | Auto-run scrapers when Docker container starts |
| `NOMINATIM_DOMAIN` | — | Self-hosted Nominatim IP for geocoding |
| `OPENWEATHERMAP_API_KEY` | — | OpenWeatherMap API key for weather queries |
