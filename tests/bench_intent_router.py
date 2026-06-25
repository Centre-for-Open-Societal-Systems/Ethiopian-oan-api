"""Benchmark for the regex Intent Router fast path.

Measures:
  1. Pure pattern detection latency (regex only)
  2. Entity extraction latency (with seeded in-memory cache)
  3. End-to-end IntentRouter.route() latency (no DB - handlers stubbed)
  4. LLM baseline: a single Gemini/OpenRouter chat completion to mirror
     what the LLM path would cost just for routing intent.

Cache and session layers are stubbed (in-memory) so we measure only the
algorithmic overhead of the router, not network/DB. For LLM, we issue a
real call so the comparison is fair.

Run inside the container:
    docker exec oan_app python tests/bench_intent_router.py
"""
from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.intent_router import router as router_mod
from app.services.intent_router import session as session_mod
from app.services.intent_router.cache import (
    CropEntity,
    LivestockEntity,
    MarketplaceEntity,
    intent_cache,
)
from app.services.intent_router.entities import extract
from app.services.intent_router.patterns import Intent, detect_intent
from app.services.intent_router.router import IntentRouter
from app.services.intent_router.session import SessionState

# ---------------------------------------------------------------------------
# Test corpus — representative agricultural queries
# ---------------------------------------------------------------------------
EN_FAST_PATH_QUERIES = [
    "What is the price of teff in Adama?",
    "How much does wheat cost in Merkato?",
    "price of maize in Adama",
    "cattle rate in Adama",
    "What is the weather in Adama?",
    "weather forecast for tomorrow in Adama",
    "list crops available in Adama",
    "show me livestock in Adama",
    "list marketplaces",
    "How much is goat in Merkato?",
]

EN_LOW_CONFIDENCE_QUERIES = [
    "How do I prevent mastitis in cattle?",
    "what fertilizer should I use for teff",
    "tell me about crop rotation",
    "best time to plant maize?",
    "how to deal with locust attacks",
]

AM_QUERIES = [
    "የጤፍ ዋጋ በአዳማ ስንት ነው?",
    "ስንዴ ዋጋ",
    "የከብት ዋጋ ስንት ነው?",
    "የነገ የአየር ሁኔታ ትንበያ",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def seed_cache():
    intent_cache.crops.clear()
    intent_cache.livestock.clear()
    intent_cache.marketplaces.clear()
    intent_cache.regions.clear()

    crops = [
        CropEntity(id=1, en="Teff", am="ጤፍ"),
        CropEntity(id=2, en="Wheat", am="ስንዴ"),
        CropEntity(id=3, en="Maize", am="በቆሎ"),
        CropEntity(id=4, en="Barley", am="ገብስ"),
        CropEntity(id=5, en="Coffee", am="ቡና"),
    ]
    for c in crops:
        intent_cache.crops[c.en.lower()] = c
        if c.am:
            intent_cache.crops[c.am] = c

    ls = [
        LivestockEntity(id=10, en="Cattle", am="ከብት"),
        LivestockEntity(id=11, en="Goat", am="ፍየል"),
        LivestockEntity(id=12, en="Sheep", am="በግ"),
    ]
    for l in ls:
        intent_cache.livestock[l.en.lower()] = l
        if l.am:
            intent_cache.livestock[l.am] = l

    mps = [
        MarketplaceEntity(id=100, en="Adama", am="አዳማ", region="Oromia"),
        MarketplaceEntity(id=101, en="Merkato", am="መርካቶ", region="Addis Ababa"),
    ]
    for mp in mps:
        intent_cache.marketplaces[mp.en.lower()] = mp
        if mp.am:
            intent_cache.marketplaces[mp.am] = mp
        if mp.region:
            intent_cache.regions.add(mp.region.lower())

    intent_cache._warmed = True


class FakeSession:
    def __init__(self):
        self.store = {}

    async def load(self, session_id):
        raw = self.store.get(session_id)
        return SessionState.from_json(raw) if raw else None

    async def save(self, session_id, state):
        self.store[session_id] = state.to_json()

    async def clear(self, session_id):
        self.store.pop(session_id, None)


def install_fake_session():
    fake = FakeSession()
    router_mod.session.load = fake.load
    router_mod.session.save = fake.save
    router_mod.session.clear = fake.clear
    session_mod.load = fake.load
    session_mod.save = fake.save
    session_mod.clear = fake.clear
    return fake


def stub_handlers():
    """Replace handlers with fast no-DB stubs to isolate router overhead."""
    async def ok(entities, lang):
        return "OK"
    router_mod.HANDLERS = {intent: ok for intent in router_mod.HANDLERS}


# ---------------------------------------------------------------------------
# Bench helpers
# ---------------------------------------------------------------------------
def _summarize(label: str, samples_ms: list[float]) -> dict:
    n = len(samples_ms)
    avg = statistics.mean(samples_ms)
    p50 = statistics.median(samples_ms)
    p95 = statistics.quantiles(samples_ms, n=20)[-1] if n >= 20 else max(samples_ms)
    p99 = statistics.quantiles(samples_ms, n=100)[-1] if n >= 100 else max(samples_ms)
    mn = min(samples_ms)
    mx = max(samples_ms)
    print(
        f"  {label:35s}  n={n:4d}  avg={avg:7.3f}ms  p50={p50:7.3f}  "
        f"p95={p95:7.3f}  p99={p99:7.3f}  min={mn:6.3f}  max={mx:7.3f}"
    )
    return {"label": label, "n": n, "avg_ms": avg, "p50_ms": p50,
            "p95_ms": p95, "p99_ms": p99, "min_ms": mn, "max_ms": mx}


def bench_sync(label: str, fn: Callable, queries: list[str], iterations: int) -> dict:
    samples = []
    for _ in range(iterations):
        for q in queries:
            t0 = time.perf_counter()
            fn(q)
            samples.append((time.perf_counter() - t0) * 1000)
    return _summarize(label, samples)


async def bench_async(label: str, fn, queries: list[str], iterations: int) -> dict:
    samples = []
    for i in range(iterations):
        for j, q in enumerate(queries):
            t0 = time.perf_counter()
            await fn(q, i, j)
            samples.append((time.perf_counter() - t0) * 1000)
    return _summarize(label, samples)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
async def main():
    seed_cache()
    install_fake_session()
    stub_handlers()

    print("=" * 88)
    print("Intent Router micro-benchmark (in-memory cache + stubbed handlers)")
    print("=" * 88)

    results = []

    # 1) Pattern detection only (regex)
    results.append(bench_sync(
        "regex.detect_intent (en, fast-path queries)",
        lambda q: detect_intent(q, "en"),
        EN_FAST_PATH_QUERIES,
        iterations=200,
    ))
    results.append(bench_sync(
        "regex.detect_intent (en, low-confidence)",
        lambda q: detect_intent(q, "en"),
        EN_LOW_CONFIDENCE_QUERIES,
        iterations=200,
    ))
    results.append(bench_sync(
        "regex.detect_intent (am)",
        lambda q: detect_intent(q, "am"),
        AM_QUERIES,
        iterations=200,
    ))

    # 2) Entity extraction
    results.append(bench_sync(
        "extract (en)",
        lambda q: extract(q, "en", intent_cache),
        EN_FAST_PATH_QUERIES,
        iterations=200,
    ))
    results.append(bench_sync(
        "extract (am)",
        lambda q: extract(q, "am", intent_cache),
        AM_QUERIES,
        iterations=200,
    ))

    # 3) Full router
    router = IntentRouter()
    results.append(await bench_async(
        "IntentRouter.route (en, fast-path)",
        lambda q, i, j: router.route(query=q, lang="en", session_id=f"bench-{i}-{j}"),
        EN_FAST_PATH_QUERIES,
        iterations=100,
    ))
    results.append(await bench_async(
        "IntentRouter.route (en, low-conf)",
        lambda q, i, j: router.route(query=q, lang="en", session_id=f"bench-low-{i}-{j}"),
        EN_LOW_CONFIDENCE_QUERIES,
        iterations=100,
    ))
    results.append(await bench_async(
        "IntentRouter.route (am)",
        lambda q, i, j: router.route(query=q, lang="am", session_id=f"bench-am-{i}-{j}"),
        AM_QUERIES,
        iterations=100,
    ))

    # 4) LLM baseline — issue real chat completions on the configured backend
    do_llm = os.getenv("BENCH_LLM", "1") != "0"
    if do_llm:
        print()
        print("─" * 88)
        print("LLM baseline (single chat completion per query)")
        print("─" * 88)
        try:
            import httpx
            base_url = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
            api_key = os.getenv("OPENAI_API_KEY", "")
            model = os.getenv("LLM_MODEL_NAME", "qwen/qwen3-14b")
            url = f"{base_url}/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

            async def call_llm(q, i, j):
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a router. Reply with JSON {\"intent\": \"crop_price\"|\"livestock_price\"|\"weather_current\"|\"weather_forecast\"|\"crop_listing\"|\"livestock_listing\"|\"marketplace_listing\"|\"unknown\"}."},
                        {"role": "user", "content": q},
                    ],
                    "max_tokens": 32,
                    "temperature": 0,
                }
                async with httpx.AsyncClient(timeout=30.0) as client:
                    r = await client.post(url, json=payload, headers=headers)
                    r.raise_for_status()

            n_llm = int(os.getenv("BENCH_LLM_N", "3"))
            results.append(await bench_async(
                f"LLM.chat ({model}, route-classify only)",
                call_llm,
                EN_FAST_PATH_QUERIES,
                iterations=n_llm,
            ))
        except Exception as e:
            print(f"  [LLM bench skipped: {e}]")

    print()
    print("=" * 88)


if __name__ == "__main__":
    asyncio.run(main())
