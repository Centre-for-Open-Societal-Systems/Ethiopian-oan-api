"""Entity extraction from free-text queries using the warmed IntentCache.

Walks unigrams and bigrams looking for exact matches in the cache. No fuzzy
match here (kept deterministic); fuzzy tolerance is intentionally out of scope
for v1 to avoid false positives on Amharic.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import re

from app.services.intent_router.cache import (
    IntentCache,
    CropEntity,
    LivestockEntity,
    MarketplaceEntity,
)


@dataclass
class ExtractedEntities:
    crop: Optional[CropEntity] = None
    livestock: Optional[LivestockEntity] = None
    marketplace: Optional[MarketplaceEntity] = None
    region: Optional[str] = None
    location: Optional[str] = None
    timeframe: Optional[str] = None


AM_PREFIXES = ("የ", "በ", "ለ", "ከ", "ወደ")

# Query terms that don't appear verbatim in the cache → alternate keys to try
EN_LIVESTOCK_SYNONYMS: dict[str, list[str]] = {
    "oxen":   ["ox", "bull", "cow"],
    "cattle": ["ox", "bull", "cow", "heifer", "calf"],
    "bovine": ["ox", "bull", "cow"],
    "herd":   ["ox", "bull", "cow"],
    "lamb":   ["male young sheep", "female young sheep"],
    "kid":    ["male young goat", "female young goat"],
    "poultry": ["chicken", "hen", "rooster"],
}
# Amharic single-word animal terms → alternate cache keys (multi-word breed names)
AM_LIVESTOCK_SYNONYMS: dict[str, list[str]] = {
    "ከብቶች": ["ox", "bull", "cow", "heifer"],  # cattle (plural)
}


def _strip_am_prefix(token: str) -> str:
    for p in AM_PREFIXES:
        if token.startswith(p) and len(token) > len(p):
            return token[len(p):]
    return token


TIMEFRAME_EN = {
    "today": "today",
    "now": "today",
    "tomorrow": "tomorrow",
    "this week": "week",
    "next week": "week",
    "weekly": "week",
    "7 days": "week",
}
TIMEFRAME_AM = {
    "ዛሬ": "today",
    "ነገ": "tomorrow",
    "በዚህ ሳምንት": "week",
    "በሚቀጥለው ሳምንት": "week",
}


def _tokenize(query: str) -> list[str]:
    query = query.strip()
    tokens = re.findall(r"[\u1200-\u137F]+|[A-Za-z][A-Za-z'-]*", query)
    return tokens


def _bigrams(tokens: list[str]) -> list[str]:
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]


def extract(query: str, lang: str, cache: IntentCache) -> ExtractedEntities:
    entities = ExtractedEntities()
    tokens = _tokenize(query)
    candidates = _bigrams(tokens) + tokens

    for cand in candidates:
        stripped = _strip_am_prefix(cand) if lang == "am" else cand
        if entities.crop is None:
            crop = cache.lookup_crop(cand) or (cache.lookup_crop(stripped) if stripped != cand else None)
            if crop:
                entities.crop = crop
        if entities.livestock is None:
            ls = cache.lookup_livestock(cand) or (cache.lookup_livestock(stripped) if stripped != cand else None)
            if ls is None:
                # Try synonym aliases for terms not stored verbatim in the cache
                syn_map = AM_LIVESTOCK_SYNONYMS if lang == "am" else EN_LIVESTOCK_SYNONYMS
                lower_cand = (stripped if lang == "am" else cand).lower()
                for alt in syn_map.get(lower_cand, []):
                    ls = cache.lookup_livestock(alt)
                    if ls:
                        break
            if ls:
                entities.livestock = ls
        if entities.marketplace is None:
            mp = cache.lookup_marketplace(cand) or (cache.lookup_marketplace(stripped) if stripped != cand else None)
            if mp:
                entities.marketplace = mp

    lower_q = query.lower()
    tf_table = TIMEFRAME_AM if lang == "am" else TIMEFRAME_EN
    for phrase, tf in tf_table.items():
        if phrase in lower_q if lang != "am" else phrase in query:
            entities.timeframe = tf
            break

    if entities.marketplace is not None:
        entities.location = entities.marketplace.en
        entities.region = entities.marketplace.region
    else:
        for cand in candidates:
            stripped = _strip_am_prefix(cand) if lang == "am" else cand
            normalized = cand.strip().lower()
            stripped_norm = stripped.strip().lower()
            if (
                normalized in cache.regions
                or cand.strip() in cache.regions
                or stripped_norm in cache.regions
                or stripped.strip() in cache.regions
            ):
                entities.region = stripped if stripped != cand else cand
                entities.location = entities.region
                break

    return entities
