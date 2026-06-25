#!/bin/bash
# OAN AI API — Full Setup & Start Script
# Runs all steps in sequence: infrastructure → migrations → scrape → index → start app
#
# Usage:
#   bash start.sh            # Full setup + start app
#   bash start.sh --skip-scrape   # Skip NMIS scraping (use existing DB data)
#   bash start.sh --skip-index   # Skip Cosdata indexing (collection already exists)
#   bash start.sh --skip-scrape --skip-index   # Just start the app

set -e

SKIP_SCRAPE=false
SKIP_INDEX=false

for arg in "$@"; do
  case $arg in
    --skip-scrape) SKIP_SCRAPE=true ;;
    --skip-index)  SKIP_INDEX=true ;;
  esac
done

echo "========================================"
echo "  OAN AI API — Setup & Start"
echo "========================================"

# ── Step 1: Start Docker infrastructure ──────────────────────────
echo ""
echo "[1/5] Starting Docker services (postgres, redis, cosdata)..."
docker compose up -d postgres redis cosdata

echo "      Waiting for services to be healthy..."
until docker compose ps postgres | grep -q "healthy"; do sleep 2; done
until docker compose ps redis    | grep -q "healthy"; do sleep 2; done
until docker compose ps cosdata  | grep -q "healthy"; do sleep 2; done
echo "      ✓ All Docker services healthy"

# ── Step 2: Check Ollama ──────────────────────────────────────────
echo ""
echo "[2/5] Checking Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "      ✓ Ollama is running"
else
  echo "      ⚠ Ollama not detected at localhost:11434"
  echo "        Start it with: ollama serve"
  echo "        Then re-run this script."
  exit 1
fi

# ── Step 3: DB migrations ─────────────────────────────────────────
echo ""
echo "[3/5] Running database migrations..."
alembic upgrade head
echo "      ✓ Migrations complete"

# ── Step 4: NMIS data scraping ────────────────────────────────────
echo ""
if [ "$SKIP_SCRAPE" = true ]; then
  echo "[4/5] Skipping NMIS scraping (--skip-scrape)"
else
  echo "[4/5] Scraping market data from NMIS (this takes 10-20 min)..."
  python scripts/run_all_scrapers.py
  echo "      ✓ NMIS data synced"
fi

# ── Step 5: Cosdata indexing ──────────────────────────────────────
echo ""
if [ "$SKIP_INDEX" = true ]; then
  echo "[5/5] Skipping Cosdata indexing (--skip-index)"
else
  echo "[5/5] Indexing agricultural docs into Cosdata..."
  python scripts/index_cosdata.py --file assets/all_agricultural_docs.json
  echo "      ✓ Cosdata indexed"
fi

# ── Start app ─────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Starting OAN AI API on 0.0.0.0:8000"
echo "========================================"
python main.py
