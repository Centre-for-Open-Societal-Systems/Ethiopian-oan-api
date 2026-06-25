"""Confidence scoring for routed intents.

HIGH (>=HIGH_THRESHOLD) → direct lookup + formatted response
MEDIUM (>=MEDIUM_THRESHOLD) → ask a follow-up question using a template (no LLM)
LOW → fall through to the main LLM pipeline
"""
from app.config import settings
from app.services.intent_router.entities import ExtractedEntities
from app.services.intent_router.patterns import Intent

HIGH_THRESHOLD = settings.intent_router_high_threshold
MEDIUM_THRESHOLD = settings.intent_router_medium_threshold


def score(intent: Intent, entities: ExtractedEntities) -> float:
    if intent == Intent.CROP_PRICE:
        if entities.crop and entities.marketplace:
            return 0.95
        if entities.crop:
            return 0.65
        if entities.marketplace:
            return 0.55
        return 0.3

    if intent == Intent.LIVESTOCK_PRICE:
        if entities.livestock and entities.marketplace:
            return 0.95
        if entities.livestock:
            return 0.65
        if entities.marketplace:
            return 0.55
        return 0.3

    if intent == Intent.CROP_LISTING:
        if entities.marketplace:
            return 0.9
        if entities.region:
            return 0.7
        return 0.4

    if intent == Intent.LIVESTOCK_LISTING:
        if entities.marketplace:
            return 0.9
        if entities.region:
            return 0.7
        return 0.4

    if intent == Intent.MARKETPLACE_LISTING:
        return 0.85

    if intent == Intent.WEATHER_CURRENT:
        if entities.location:
            return 0.9
        return 0.4

    if intent == Intent.WEATHER_FORECAST:
        if entities.location:
            return 0.9
        return 0.4

    return 0.0


def decision(score_value: float) -> str:
    if score_value >= HIGH_THRESHOLD:
        return "high"
    if score_value >= MEDIUM_THRESHOLD:
        return "medium"
    return "low"
