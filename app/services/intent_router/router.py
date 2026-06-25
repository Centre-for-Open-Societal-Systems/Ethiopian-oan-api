"""Main IntentRouter orchestrator.

Ties together: pattern detection -> entity extraction -> confidence scoring ->
handler dispatch (high) OR follow-up template (medium) OR fall-through (low).
Maintains 5-minute Redis session context for multi-turn merges.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional
import time

from app.services.intent_router import confidence, handlers, session, templates
from app.services.intent_router.cache import intent_cache
from app.services.intent_router.entities import ExtractedEntities, extract
from app.services.intent_router.patterns import Intent, detect_intent
from app.services.intent_router.session import SessionEntities, SessionState
from helpers.utils import get_logger

logger = get_logger(__name__)


@dataclass
class FastPathResult:
    matched: bool
    response: Optional[str] = None
    intent: Optional[str] = None
    confidence: float = 0.0
    decision: str = "low"
    followup: bool = False
    latency_ms: float = 0.0
    sources: list[str] = field(default_factory=list)


HANDLERS = {
    Intent.CROP_PRICE: handlers.handle_crop_price,
    Intent.LIVESTOCK_PRICE: handlers.handle_livestock_price,
    Intent.CROP_LISTING: handlers.handle_crop_listing,
    Intent.LIVESTOCK_LISTING: handlers.handle_livestock_listing,
    Intent.MARKETPLACE_LISTING: handlers.handle_marketplace_listing,
    Intent.WEATHER_CURRENT: handlers.handle_weather_current,
    Intent.WEATHER_FORECAST: handlers.handle_weather_forecast,
}


FOLLOWUP_TEMPLATES = {
    Intent.CROP_PRICE: "price_followup",
    Intent.LIVESTOCK_PRICE: "price_followup",
    Intent.CROP_LISTING: "listing_followup",
    Intent.LIVESTOCK_LISTING: "listing_followup",
}


def _missing_entity(intent: Intent, entities: ExtractedEntities) -> Optional[str]:
    if intent == Intent.CROP_PRICE:
        if not entities.crop:
            return "crop"
        if not entities.marketplace:
            return "marketplace"
    if intent == Intent.LIVESTOCK_PRICE:
        if not entities.livestock:
            return "livestock"
        if not entities.marketplace:
            return "marketplace"
    if intent in (Intent.CROP_LISTING, Intent.LIVESTOCK_LISTING):
        if not entities.marketplace:
            return "marketplace"
    return None


def _merge_session(entities: ExtractedEntities, state: SessionState, cache) -> ExtractedEntities:
    """Hydrate entity dataclasses from saved dicts and fill gaps."""
    saved = state.entities
    if not entities.crop and saved.crop:
        entities.crop = cache.lookup_crop(saved.crop.get("en", ""))
    if not entities.livestock and saved.livestock:
        entities.livestock = cache.lookup_livestock(saved.livestock.get("en", ""))
    if not entities.marketplace and saved.marketplace:
        entities.marketplace = cache.lookup_marketplace(saved.marketplace.get("en", ""))
    if not entities.region and saved.region:
        entities.region = saved.region
    if not entities.location and saved.location:
        entities.location = saved.location
    if not entities.timeframe and saved.timeframe:
        entities.timeframe = saved.timeframe
    return entities


def _entities_to_session(entities: ExtractedEntities) -> SessionEntities:
    return SessionEntities(
        crop={"id": entities.crop.id, "en": entities.crop.en, "am": entities.crop.am}
        if entities.crop else None,
        livestock={"id": entities.livestock.id, "en": entities.livestock.en, "am": entities.livestock.am}
        if entities.livestock else None,
        marketplace={
            "id": entities.marketplace.id,
            "en": entities.marketplace.en,
            "am": entities.marketplace.am,
        } if entities.marketplace else None,
        region=entities.region,
        location=entities.location,
        timeframe=entities.timeframe,
    )


class IntentRouter:
    async def route(self, query: str, lang: str, session_id: str) -> FastPathResult:
        start = time.perf_counter()
        lang = lang if lang in ("en", "am") else "en"

        if not intent_cache.warmed:
            logger.debug("IntentCache not warm, skipping fast path")
            return FastPathResult(matched=False, latency_ms=(time.perf_counter() - start) * 1000)

        intent = detect_intent(query, lang)
        saved_state = await session.load(session_id)

        if intent == Intent.UNKNOWN and saved_state:
            try:
                intent = Intent(saved_state.intent)
            except ValueError:
                intent = Intent.UNKNOWN

        if intent == Intent.UNKNOWN:
            return FastPathResult(matched=False, latency_ms=(time.perf_counter() - start) * 1000)

        entities = extract(query, lang, intent_cache)

        if saved_state:
            entities = _merge_session(entities, saved_state, intent_cache)

        score = confidence.score(intent, entities)
        dec = confidence.decision(score)

        logger.info(
            f"IntentRouter: intent={intent.value}, score={score:.2f}, decision={dec}, "
            f"entities=crop={getattr(entities.crop, 'en', None)}, "
            f"livestock={getattr(entities.livestock, 'en', None)}, "
            f"marketplace={getattr(entities.marketplace, 'en', None)}"
        )

        if dec == "high":
            handler = HANDLERS.get(intent)
            if not handler:
                return FastPathResult(matched=False, intent=intent.value, confidence=score, decision=dec,
                                      latency_ms=(time.perf_counter() - start) * 1000)
            try:
                response = await handler(entities, lang)
            except Exception as e:
                logger.error(f"Fast-path handler failed for {intent.value}: {e}")
                return FastPathResult(matched=False, intent=intent.value, confidence=score, decision=dec,
                                      latency_ms=(time.perf_counter() - start) * 1000)

            if not response:
                return FastPathResult(matched=False, intent=intent.value, confidence=score, decision=dec,
                                      latency_ms=(time.perf_counter() - start) * 1000)

            await session.clear(session_id)
            return FastPathResult(
                matched=True,
                response=response,
                intent=intent.value,
                confidence=score,
                decision=dec,
                latency_ms=(time.perf_counter() - start) * 1000,
                sources=["https://nmis.et/"] if intent in (
                    Intent.CROP_PRICE, Intent.LIVESTOCK_PRICE,
                    Intent.CROP_LISTING, Intent.LIVESTOCK_LISTING,
                ) else ["OpenWeatherMap"] if intent in (
                    Intent.WEATHER_CURRENT, Intent.WEATHER_FORECAST,
                ) else [],
            )

        if dec == "medium":
            missing = _missing_entity(intent, entities)
            if not missing:
                return FastPathResult(matched=False, intent=intent.value, confidence=score, decision=dec,
                                      latency_ms=(time.perf_counter() - start) * 1000)

            template_name = FOLLOWUP_TEMPLATES.get(intent)
            if not template_name:
                return FastPathResult(matched=False, intent=intent.value, confidence=score, decision=dec,
                                      latency_ms=(time.perf_counter() - start) * 1000)

            try:
                response = templates.render(template_name, lang=lang, missing=missing)
            except Exception as e:
                logger.error(f"Follow-up template render failed: {e}")
                return FastPathResult(matched=False, intent=intent.value, confidence=score, decision=dec,
                                      latency_ms=(time.perf_counter() - start) * 1000)

            await session.save(
                session_id,
                SessionState(
                    intent=intent.value,
                    entities=_entities_to_session(entities),
                    awaiting=missing,
                ),
            )

            return FastPathResult(
                matched=True,
                response=response,
                intent=intent.value,
                confidence=score,
                decision=dec,
                followup=True,
                latency_ms=(time.perf_counter() - start) * 1000,
            )

        return FastPathResult(matched=False, intent=intent.value, confidence=score, decision=dec,
                              latency_ms=(time.perf_counter() - start) * 1000)


intent_router = IntentRouter()
