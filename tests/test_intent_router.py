"""Unit tests for the Intent Router fast path.

Runs as a standalone script (`python tests/test_intent_router.py`) to match the
conventions of the other files in `tests/`. No pytest / pytest-asyncio
dependency required. The tests seed the IntentCache in-memory (skipping DB
warmup), stub the Redis-backed session layer with an in-process dict, and
replace handlers with deterministic stubs so routing logic can be verified in
isolation.
"""
from __future__ import annotations

import asyncio
import os
import sys
import traceback
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.services.intent_router import confidence as confidence_mod
from app.services.intent_router import session as session_mod
from app.services.intent_router import templates as templates_mod
from app.services.intent_router import router as router_mod
from app.services.intent_router.cache import (
    CropEntity,
    IntentCache,
    LivestockEntity,
    MarketplaceEntity,
    intent_cache,
)
from app.services.intent_router.entities import ExtractedEntities, extract
from app.services.intent_router.patterns import Intent, detect_intent
from app.services.intent_router.router import IntentRouter
from app.services.intent_router.session import SessionState


# ---------------------------------------------------------------------------
# Fixture setup
# ---------------------------------------------------------------------------

def seed_cache(cache: IntentCache) -> None:
    cache.crops.clear()
    cache.livestock.clear()
    cache.marketplaces.clear()
    cache.regions.clear()

    teff = CropEntity(id=1, en="Teff", am="ጤፍ")
    wheat = CropEntity(id=2, en="Wheat", am="ስንዴ")
    maize = CropEntity(id=3, en="Maize", am="በቆሎ")
    for c in (teff, wheat, maize):
        cache.crops[c.en.lower()] = c
        if c.am:
            cache.crops[c.am] = c

    cattle = LivestockEntity(id=10, en="Cattle", am="ከብት")
    goat = LivestockEntity(id=11, en="Goat", am="ፍየል")
    for ls in (cattle, goat):
        cache.livestock[ls.en.lower()] = ls
        if ls.am:
            cache.livestock[ls.am] = ls

    adama = MarketplaceEntity(id=100, en="Adama", am="አዳማ", region="Oromia", marketplace_type="crop")
    merkato = MarketplaceEntity(id=101, en="Merkato", am="መርካቶ", region="Addis Ababa", marketplace_type="crop")
    for mp in (adama, merkato):
        cache.marketplaces[mp.en.lower()] = mp
        if mp.am:
            cache.marketplaces[mp.am] = mp
        if mp.region:
            cache.regions.add(mp.region.lower())

    cache._warmed = True


class FakeSession:
    """In-process replacement for the Redis-backed session module."""

    NAMESPACE = "intent_session"

    def __init__(self):
        self.store: dict[str, str] = {}

    def _key(self, session_id: str) -> str:
        return f"{self.NAMESPACE}:{session_id}"

    async def load(self, session_id: str) -> Optional[SessionState]:
        raw = self.store.get(self._key(session_id))
        if not raw:
            return None
        return SessionState.from_json(raw)

    async def save(self, session_id: str, state: SessionState) -> None:
        self.store[self._key(session_id)] = state.to_json()

    async def clear(self, session_id: str) -> None:
        self.store.pop(self._key(session_id), None)


def install_fake_session() -> FakeSession:
    fake = FakeSession()
    # The router module imports session as a module reference, so patching the
    # functions on that module object is sufficient.
    router_mod.session.load = fake.load
    router_mod.session.save = fake.save
    router_mod.session.clear = fake.clear
    session_mod.load = fake.load
    session_mod.save = fake.save
    session_mod.clear = fake.clear
    return fake


def install_stub_handler(intent: Intent, handler: Callable) -> None:
    router_mod.HANDLERS = {**router_mod.HANDLERS, intent: handler}


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

RESULTS: list[tuple[str, bool, str]] = []


def run_case(name: str, fn: Callable):
    try:
        if asyncio.iscoroutinefunction(fn):
            asyncio.run(fn())
        else:
            fn()
        RESULTS.append((name, True, ""))
        print(f"  PASS  {name}")
    except AssertionError as e:
        RESULTS.append((name, False, f"assertion: {e}"))
        print(f"  FAIL  {name} :: assertion: {e}")
    except Exception:
        RESULTS.append((name, False, traceback.format_exc()))
        print(f"  FAIL  {name}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------

def test_detect_crop_price_en():
    for q in [
        "What is the price of teff in Adama?",
        "How much does wheat cost?",
        "teff rate today",
        "maize going for how much",
    ]:
        assert detect_intent(q, "en") == Intent.CROP_PRICE, q


def test_detect_crop_price_am():
    assert detect_intent("የጤፍ ዋጋ በአዳማ ስንት ነው?", "am") == Intent.CROP_PRICE
    assert detect_intent("ስንዴ ዋጋ", "am") == Intent.CROP_PRICE


def test_detect_livestock_price_beats_crop_price():
    assert detect_intent("What is the price of cattle in Adama?", "en") == Intent.LIVESTOCK_PRICE


def test_detect_livestock_price_am():
    assert detect_intent("የከብት ዋጋ ስንት ነው?", "am") == Intent.LIVESTOCK_PRICE


def test_detect_weather_current():
    assert detect_intent("What is the weather in Adama?", "en") == Intent.WEATHER_CURRENT


def test_detect_weather_forecast_en():
    assert detect_intent("What is the weather forecast for tomorrow?", "en") == Intent.WEATHER_FORECAST


def test_detect_weather_forecast_am():
    assert detect_intent("የነገ የአየር ሁኔታ ትንበያ", "am") == Intent.WEATHER_FORECAST


def test_detect_marketplace_listing():
    assert detect_intent("List marketplaces", "en") == Intent.MARKETPLACE_LISTING


def test_detect_crop_listing():
    assert detect_intent("List crops available in Adama", "en") == Intent.CROP_LISTING


def test_detect_livestock_listing():
    assert detect_intent("Show me livestock in Adama", "en") == Intent.LIVESTOCK_LISTING


def test_detect_unknown_advisory_question():
    assert detect_intent("How do I prevent mastitis in cattle?", "en") == Intent.UNKNOWN


def test_detect_empty_query():
    assert detect_intent("", "en") == Intent.UNKNOWN
    assert detect_intent("   ", "en") == Intent.UNKNOWN


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def test_extract_crop_and_marketplace_en():
    seed_cache(intent_cache)
    e = extract("price of teff in Adama", "en", intent_cache)
    assert e.crop is not None and e.crop.id == 1
    assert e.marketplace is not None and e.marketplace.en == "Adama"
    assert e.location == "Adama"
    assert e.region == "Oromia"


def test_extract_crop_and_marketplace_am():
    seed_cache(intent_cache)
    e = extract("የጤፍ ዋጋ በአዳማ", "am", intent_cache)
    assert e.crop is not None and e.crop.id == 1
    assert e.marketplace is not None and e.marketplace.id == 100


def test_extract_livestock():
    seed_cache(intent_cache)
    e = extract("price of cattle in Merkato", "en", intent_cache)
    assert e.livestock is not None and e.livestock.id == 10
    assert e.marketplace is not None and e.marketplace.en == "Merkato"


def test_extract_timeframe_en():
    seed_cache(intent_cache)
    e = extract("weather in Adama tomorrow", "en", intent_cache)
    assert e.timeframe == "tomorrow"


def test_extract_timeframe_am():
    seed_cache(intent_cache)
    e = extract("የአየር ሁኔታ ነገ", "am", intent_cache)
    assert e.timeframe == "tomorrow"


def test_extract_no_match():
    seed_cache(intent_cache)
    e = extract("something unrelated", "en", intent_cache)
    assert e.crop is None
    assert e.livestock is None
    assert e.marketplace is None


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def test_confidence_crop_price_high():
    seed_cache(intent_cache)
    e = extract("price of teff in Adama", "en", intent_cache)
    s = confidence_mod.score(Intent.CROP_PRICE, e)
    assert s >= confidence_mod.HIGH_THRESHOLD
    assert confidence_mod.decision(s) == "high"


def test_confidence_crop_price_medium():
    seed_cache(intent_cache)
    e = extract("price of teff", "en", intent_cache)
    s = confidence_mod.score(Intent.CROP_PRICE, e)
    assert confidence_mod.MEDIUM_THRESHOLD <= s < confidence_mod.HIGH_THRESHOLD
    assert confidence_mod.decision(s) == "medium"


def test_confidence_crop_price_low():
    s = confidence_mod.score(Intent.CROP_PRICE, ExtractedEntities())
    assert s < confidence_mod.MEDIUM_THRESHOLD
    assert confidence_mod.decision(s) == "low"


def test_confidence_weather_high_with_location():
    seed_cache(intent_cache)
    e = extract("weather in Adama", "en", intent_cache)
    s = confidence_mod.score(Intent.WEATHER_CURRENT, e)
    assert confidence_mod.decision(s) == "high"


def test_confidence_weather_low_without_location():
    s = confidence_mod.score(Intent.WEATHER_CURRENT, ExtractedEntities())
    assert confidence_mod.decision(s) == "low"


def test_confidence_threshold_boundary():
    assert confidence_mod.decision(confidence_mod.HIGH_THRESHOLD - 0.01) != "high"
    assert confidence_mod.decision(confidence_mod.HIGH_THRESHOLD) == "high"


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def test_template_price_followup_crop_en():
    out = templates_mod.render("price_followup", lang="en", missing="crop")
    assert "crop" in out.lower()


def test_template_price_followup_marketplace_en():
    out = templates_mod.render("price_followup", lang="en", missing="marketplace")
    assert "market" in out.lower()


def test_template_price_followup_am_nonempty():
    out = templates_mod.render("price_followup", lang="am", missing="crop")
    assert out  # non-empty Amharic string


def test_template_listing_followup_marketplace():
    out = templates_mod.render("listing_followup", lang="en", missing="marketplace")
    assert "market" in out.lower()


def test_template_weather_forecast_fills_placeholders():
    out = templates_mod.render(
        "weather_forecast",
        lang="en",
        location="Adama",
        timeframe="tomorrow",
        forecast_text="Sunny, 25C",
    )
    assert "Adama" in out
    assert "Sunny" in out


# ---------------------------------------------------------------------------
# IntentRouter.route — end-to-end routing
# ---------------------------------------------------------------------------

async def test_route_high_confidence_crop_price():
    seed_cache(intent_cache)
    install_fake_session()

    async def _ok(entities, lang):
        assert entities.crop is not None
        assert entities.marketplace is not None
        return "PRICE-OK"

    install_stub_handler(Intent.CROP_PRICE, _ok)
    router = IntentRouter()
    result = await router.route(query="price of teff in Adama", lang="en", session_id="s1")
    assert result.matched is True
    assert result.response == "PRICE-OK"
    assert result.decision == "high"
    assert result.intent == Intent.CROP_PRICE.value
    assert result.latency_ms >= 0
    assert "https://nmis.et/" in result.sources


async def test_route_medium_confidence_followup():
    seed_cache(intent_cache)
    fake = install_fake_session()
    router = IntentRouter()

    result = await router.route(query="price of teff", lang="en", session_id="s2")
    assert result.matched is True
    assert result.followup is True
    assert result.decision == "medium"
    assert "market" in (result.response or "").lower()
    saved = fake.store.get("intent_session:s2")
    assert saved is not None and "marketplace" in saved


async def test_route_low_confidence_falls_through():
    seed_cache(intent_cache)
    install_fake_session()
    router = IntentRouter()
    result = await router.route(
        query="How do I prevent mastitis in cattle?", lang="en", session_id="s3"
    )
    assert result.matched is False


async def test_route_cache_not_warm_skips():
    seed_cache(intent_cache)
    install_fake_session()
    intent_cache._warmed = False
    try:
        router = IntentRouter()
        result = await router.route(query="price of teff in Adama", lang="en", session_id="s4")
        assert result.matched is False
    finally:
        intent_cache._warmed = True


async def test_route_multi_turn_merge():
    seed_cache(intent_cache)
    fake = install_fake_session()
    captured: dict = {}

    async def _ok(entities, lang):
        captured["crop"] = entities.crop
        captured["marketplace"] = entities.marketplace
        return "PRICE-OK"

    install_stub_handler(Intent.CROP_PRICE, _ok)
    router = IntentRouter()

    r1 = await router.route(query="price of teff", lang="en", session_id="s-multi")
    assert r1.followup is True

    r2 = await router.route(query="Adama", lang="en", session_id="s-multi")
    assert r2.matched is True
    assert r2.decision == "high"
    assert captured["crop"] is not None and captured["crop"].en == "Teff"
    assert captured["marketplace"] is not None and captured["marketplace"].en == "Adama"
    assert "intent_session:s-multi" not in fake.store


async def test_route_handler_exception_returns_unmatched():
    seed_cache(intent_cache)
    install_fake_session()

    async def _boom(entities, lang):
        raise RuntimeError("upstream API failed")

    install_stub_handler(Intent.CROP_PRICE, _boom)
    router = IntentRouter()
    result = await router.route(query="price of teff in Adama", lang="en", session_id="s-boom")
    assert result.matched is False


async def test_route_weather_high_confidence():
    seed_cache(intent_cache)
    install_fake_session()

    async def _ok(entities, lang):
        assert entities.location == "Adama"
        return "Sunny 25C"

    install_stub_handler(Intent.WEATHER_CURRENT, _ok)
    router = IntentRouter()
    result = await router.route(query="what is the weather in Adama", lang="en", session_id="s-weather")
    assert result.matched is True
    assert result.response == "Sunny 25C"
    assert "OpenWeatherMap" in result.sources


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

ALL_TESTS = [
    # Pattern detection
    ("detect.crop_price_en",              test_detect_crop_price_en),
    ("detect.crop_price_am",              test_detect_crop_price_am),
    ("detect.livestock_beats_crop",       test_detect_livestock_price_beats_crop_price),
    ("detect.livestock_price_am",         test_detect_livestock_price_am),
    ("detect.weather_current",            test_detect_weather_current),
    ("detect.weather_forecast_en",        test_detect_weather_forecast_en),
    ("detect.weather_forecast_am",        test_detect_weather_forecast_am),
    ("detect.marketplace_listing",        test_detect_marketplace_listing),
    ("detect.crop_listing",               test_detect_crop_listing),
    ("detect.livestock_listing",          test_detect_livestock_listing),
    ("detect.unknown_advisory",           test_detect_unknown_advisory_question),
    ("detect.empty_query",                test_detect_empty_query),
    # Entity extraction
    ("extract.crop_marketplace_en",       test_extract_crop_and_marketplace_en),
    ("extract.crop_marketplace_am",       test_extract_crop_and_marketplace_am),
    ("extract.livestock",                 test_extract_livestock),
    ("extract.timeframe_en",              test_extract_timeframe_en),
    ("extract.timeframe_am",              test_extract_timeframe_am),
    ("extract.no_match",                  test_extract_no_match),
    # Confidence
    ("confidence.crop_price_high",        test_confidence_crop_price_high),
    ("confidence.crop_price_medium",      test_confidence_crop_price_medium),
    ("confidence.crop_price_low",         test_confidence_crop_price_low),
    ("confidence.weather_high",           test_confidence_weather_high_with_location),
    ("confidence.weather_low",            test_confidence_weather_low_without_location),
    ("confidence.threshold_boundary",     test_confidence_threshold_boundary),
    # Templates
    ("template.price_followup_crop_en",   test_template_price_followup_crop_en),
    ("template.price_followup_mp_en",     test_template_price_followup_marketplace_en),
    ("template.price_followup_am",        test_template_price_followup_am_nonempty),
    ("template.listing_followup_mp",      test_template_listing_followup_marketplace),
    ("template.weather_forecast",         test_template_weather_forecast_fills_placeholders),
    # End-to-end routing
    ("route.crop_price_high",             test_route_high_confidence_crop_price),
    ("route.medium_followup",             test_route_medium_confidence_followup),
    ("route.low_falls_through",           test_route_low_confidence_falls_through),
    ("route.cache_not_warm",              test_route_cache_not_warm_skips),
    ("route.multi_turn_merge",            test_route_multi_turn_merge),
    ("route.handler_exception",           test_route_handler_exception_returns_unmatched),
    ("route.weather_high",                test_route_weather_high_confidence),
]


def main() -> int:
    print(f"Running {len(ALL_TESTS)} intent-router test cases\n")
    for name, fn in ALL_TESTS:
        run_case(name, fn)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = sum(1 for _, ok, _ in RESULTS if not ok)
    total = len(RESULTS)
    print(f"\n{'=' * 60}")
    print(f"Summary: {passed}/{total} passed, {failed} failed")
    if failed:
        print("Failed cases:")
        for name, ok, msg in RESULTS:
            if not ok:
                first = msg.splitlines()[0] if msg else ""
                print(f"  - {name}: {first}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
