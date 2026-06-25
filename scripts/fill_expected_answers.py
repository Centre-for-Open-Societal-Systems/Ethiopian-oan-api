"""
Fill the expected_answer column in benchmark_samples.csv by querying
the live DB via the app's intent cache + SQLAlchemy.
Amharic rows are translated to Amharic using the Gemma vLLM endpoint.

Run inside the app container:
    docker exec oan_app python3 /app/scripts/fill_expected_answers.py

Writes back to both:
    /app/downloads/benchmark_samples.csv
    /app/results/benchmark_samples.csv

Rows that already have a correct-language answer are left unchanged.
English rows with a missing answer → fill from DB.
Amharic rows with an English answer (or missing) → fill from DB then translate.
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, "/app")
from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import select, func
import re
from app.database import async_session_maker
from app.models.market import (
    Crop, Livestock, Marketplace, MarketPrice,
    CropVariety, LivestockBreed,
)
from app.services.intent_router.cache import intent_cache
from app.services.intent_router.entities import extract

DOWNLOADS_CSV = Path("/app/downloads/benchmark_samples.csv")
RESULTS_CSV   = Path("/app/results/benchmark_samples.csv")

TRITON_URL = os.getenv("OPENAI_BASE_URL", "http://52.66.116.220:8080").rstrip("/") + "/v1/chat/completions"
TRITON_KEY = os.getenv("TRITON_LLM_API_KEY", "")
LLM_MODEL  = os.getenv("LLM_MODEL_NAME", "gemma-4-26b-a4b")


# ── Language helpers ──────────────────────────────────────────────────────────

def _is_amharic(text: str) -> bool:
    return any(0x1200 <= ord(c) <= 0x137F for c in text)


# ── Translation ───────────────────────────────────────────────────────────────

def _translate_sync(text: str) -> str:
    """Call Gemma to translate English agricultural text to Amharic."""
    headers = {"Content-Type": "application/json"}
    if TRITON_KEY:
        headers["Authorization"] = f"Bearer {TRITON_KEY}"
    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise translator for Ethiopian agricultural market data. "
                    "Translate the given English sentence to Amharic. "
                    "Keep all numbers, ETB currency, Quintal, Head units, dates, "
                    "marketplace names, and crop/livestock names exactly as they appear. "
                    "Return only the Amharic translation — no explanations, no extra text."
                ),
            },
            {"role": "user", "content": text},
        ],
        "max_tokens": 512,
        "temperature": 0.1,
    }).encode()
    req = urllib.request.Request(TRITON_URL, data=payload, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        r = json.loads(resp.read().decode())
        return r["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"    [WARN] Translation failed: {e} — keeping English")
        return text


async def translate_to_amharic(text: str) -> str:
    return await asyncio.to_thread(_translate_sync, text)


# ── DB helpers ────────────────────────────────────────────────────────────────

async def get_crop_price(crop_id: int, marketplace_id: int) -> dict | None:
    async with async_session_maker() as db:
        result = await db.execute(
            select(MarketPrice)
            .where(MarketPrice.crop_id == crop_id,
                   MarketPrice.marketplace_id == marketplace_id)
            .order_by(MarketPrice.price_date.desc())
            .limit(1)
        )
        row = result.scalar_one_or_none()
        if row:
            return {
                "avg": float(row.avg_price or 0),
                "min": float(row.min_price or 0),
                "max": float(row.max_price or 0),
                "unit": row.unit or "Quintal",
                "date": str(row.price_date),
                "currency": row.currency or "ETB",
            }
    return None


async def get_livestock_price(livestock_id: int, marketplace_id: int) -> dict | None:
    async with async_session_maker() as db:
        result = await db.execute(
            select(MarketPrice)
            .where(MarketPrice.livestock_id == livestock_id,
                   MarketPrice.marketplace_id == marketplace_id)
            .order_by(MarketPrice.price_date.desc())
            .limit(1)
        )
        row = result.scalar_one_or_none()
        if row:
            return {
                "avg": float(row.avg_price or 0),
                "min": float(row.min_price or 0),
                "max": float(row.max_price or 0),
                "unit": row.unit or "Head",
                "date": str(row.price_date),
                "currency": row.currency or "ETB",
            }
    return None


async def get_crops_at_market(marketplace_id: int) -> list[str]:
    async with async_session_maker() as db:
        result = await db.execute(
            select(Crop.name)
            .join(MarketPrice, Crop.crop_id == MarketPrice.crop_id)
            .where(MarketPrice.marketplace_id == marketplace_id)
            .distinct()
            .limit(15)
        )
        return [r[0] for r in result.all()]


async def get_livestock_at_market(marketplace_id: int) -> list[str]:
    async with async_session_maker() as db:
        result = await db.execute(
            select(Livestock.name)
            .join(MarketPrice, Livestock.livestock_id == MarketPrice.livestock_id)
            .where(MarketPrice.marketplace_id == marketplace_id)
            .distinct()
            .limit(15)
        )
        return [r[0] for r in result.all()]


async def fuzzy_marketplace(token: str) -> "MarketplaceEntity | None":
    """Fallback: find marketplace whose name contains the token (case-insensitive)."""
    from app.services.intent_router.cache import MarketplaceEntity as ME
    token_lower = token.strip().lower()
    for key, ent in intent_cache.marketplaces.items():
        if token_lower in key.lower():
            return ent
    async with async_session_maker() as db:
        result = await db.execute(
            select(Marketplace)
            .where(func.lower(Marketplace.name).contains(token_lower))
            .where(Marketplace.is_active == True)
            .limit(1)
        )
        row = result.scalar_one_or_none()
        if row:
            return ME(id=row.marketplace_id, en=row.name, am=row.name_amharic,
                      region=row.region, marketplace_type=row.marketplace_type)
    return None


async def get_marketplaces_by_type(mtype: str | None = None) -> list[str]:
    async with async_session_maker() as db:
        q = select(Marketplace.name).where(Marketplace.is_active == True)
        if mtype:
            q = q.where(Marketplace.marketplace_type == mtype)
        result = await db.execute(q.order_by(Marketplace.name).limit(20))
        return [r[0] for r in result.all()]


# ── Answer builder ────────────────────────────────────────────────────────────

async def resolve_marketplace(ents, query: str, lang: str):
    if ents.marketplace:
        return ents.marketplace
    tokens = re.findall(r"[ሀ-፿]+|[A-Za-z][A-Za-z']*", query)
    for tok in tokens:
        if len(tok) >= 3:
            mp = await fuzzy_marketplace(tok)
            if mp:
                return mp
    return None


async def build_answer(lang: str, query: str, intent: str) -> str:
    ents = extract(query, lang, intent_cache)
    if not ents.marketplace:
        ents.marketplace = await resolve_marketplace(ents, query, lang)

    if intent == "crop_price":
        if ents.crop and ents.marketplace:
            p = await get_crop_price(ents.crop.id, ents.marketplace.id)
            if p:
                return (
                    f"The price of {ents.crop.en} at {ents.marketplace.en} is "
                    f"{p['avg']:,.0f} {p['currency']} per {p['unit']} "
                    f"(min: {p['min']:,.0f}, max: {p['max']:,.0f}) as of {p['date']}."
                )
            return f"No price data found for {ents.crop.en} at {ents.marketplace.en}."
        if ents.crop:
            return (
                f"Please specify a marketplace to get the price of {ents.crop.en}. "
                f"For example: What is the price of {ents.crop.en} in Adama?"
            )
        return "Please specify both a crop and a marketplace."

    if intent == "livestock_price":
        if ents.livestock and ents.marketplace:
            p = await get_livestock_price(ents.livestock.id, ents.marketplace.id)
            if p:
                return (
                    f"The price of {ents.livestock.en} at {ents.marketplace.en} is "
                    f"{p['avg']:,.0f} {p['currency']} per {p['unit']} "
                    f"(min: {p['min']:,.0f}, max: {p['max']:,.0f}) as of {p['date']}."
                )
            return f"No price data found for {ents.livestock.en} at {ents.marketplace.en}."
        if ents.livestock:
            return (
                f"Please specify a marketplace to get the price of {ents.livestock.en}. "
                f"For example: What is the price of {ents.livestock.en} in Miyo?"
            )
        return "Please specify both a livestock type and a marketplace."

    if intent == "crop_listing":
        if ents.marketplace:
            crops = await get_crops_at_market(ents.marketplace.id)
            if crops:
                return f"Crops available at {ents.marketplace.en}: {', '.join(crops)}."
            return f"No crop data found for {ents.marketplace.en}."
        crops_sample = await get_marketplaces_by_type("crop")
        return (
            f"Please specify a marketplace. Crop markets include: "
            f"{', '.join(crops_sample[:8])}."
        )

    if intent == "livestock_listing":
        if ents.marketplace:
            animals = await get_livestock_at_market(ents.marketplace.id)
            if animals:
                return f"Livestock available at {ents.marketplace.en}: {', '.join(animals)}."
            return f"No livestock data found for {ents.marketplace.en}."
        ls_markets = await get_marketplaces_by_type("livestock")
        return (
            f"Please specify a livestock marketplace. Options include: "
            f"{', '.join(ls_markets[:8])}."
        )

    if intent == "marketplace_listing":
        q_lower = query.lower()
        if "crop" in q_lower or "ሰብል" in query:
            markets = await get_marketplaces_by_type("crop")
            return f"Active crop marketplaces: {', '.join(markets[:15])}."
        if any(w in q_lower for w in ("livestock", "cattle", "animal", "ከብት")):
            markets = await get_marketplaces_by_type("livestock")
            return f"Active livestock marketplaces: {', '.join(markets[:15])}."
        crop_m = await get_marketplaces_by_type("crop")
        ls_m   = await get_marketplaces_by_type("livestock")
        return (
            f"Active crop marketplaces: {', '.join(crop_m[:10])}. "
            f"Active livestock marketplaces: {', '.join(ls_m[:10])}."
        )

    return ""


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("Warming intent cache from DB ...")
    await intent_cache.warmup()
    print(
        f"Cache ready: {len(intent_cache.crops)} crop keys, "
        f"{len(intent_cache.livestock)} livestock keys, "
        f"{len(intent_cache.marketplaces)} marketplace keys"
    )

    # Use downloads CSV as source of truth; fall back to results if not found
    src = DOWNLOADS_CSV if DOWNLOADS_CSV.exists() else RESULTS_CSV
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    filled = translated = skipped = 0

    for i, row in enumerate(rows, 1):
        lang     = row["lang"]
        existing = row.get("expected_answer", "").strip()

        needs_fill      = not existing
        needs_translate = lang == "am" and existing and not _is_amharic(existing)

        if not needs_fill and not needs_translate:
            skipped += 1
            continue

        if needs_fill:
            answer = await build_answer(lang, row["query"], row["expected_intent"])
            print(f"  [{i:3d}] {row['query'][:60]}")
            print(f"        → {answer[:100]}")
            filled += 1
        else:
            answer = existing  # already English, just needs translation

        if lang == "am" and answer and not _is_amharic(answer):
            print(f"  [{i:3d}] Translating to Amharic: {answer[:70]}")
            answer = await translate_to_amharic(answer)
            print(f"        → {answer[:100]}")
            translated += 1

        row["expected_answer"] = answer

    # Write back to both paths
    fieldnames = ["label", "lang", "query", "expected_intent", "expected_answer"]
    for out_path in [DOWNLOADS_CSV, RESULTS_CSV]:
        if not out_path.parent.exists():
            continue
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV updated: {out_path}")

    print(
        f"\nDone — filled {filled} answers, "
        f"translated {translated} to Amharic, "
        f"skipped {skipped} (already correct)."
    )


if __name__ == "__main__":
    asyncio.run(main())
