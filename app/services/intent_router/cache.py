"""Entity registry warmed from the DB on startup.

Holds crops, livestock, marketplaces, and regions indexed by both English and
Amharic names for O(1) token lookup during entity extraction.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import asyncio

from sqlalchemy import select

from app.database import async_session_maker
from app.models.market import Crop, Livestock, Marketplace, CropVariety, LivestockBreed
from helpers.utils import get_logger

logger = get_logger(__name__)


@dataclass
class CropEntity:
    id: int
    en: str
    am: Optional[str] = None


@dataclass
class LivestockEntity:
    id: int
    en: str
    am: Optional[str] = None


@dataclass
class MarketplaceEntity:
    id: int
    en: str
    am: Optional[str] = None
    region: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    marketplace_type: str = "crop"


def _normalize_en(s: str) -> str:
    return s.strip().lower()


def _normalize_am(s: str) -> str:
    return s.strip()


class IntentCache:
    def __init__(self):
        self.crops: dict[str, CropEntity] = {}
        self.livestock: dict[str, LivestockEntity] = {}
        self.marketplaces: dict[str, MarketplaceEntity] = {}
        self.regions: set[str] = set()
        self._lock = asyncio.Lock()
        self._warmed = False

    @property
    def warmed(self) -> bool:
        return self._warmed

    async def warmup(self) -> None:
        async with self._lock:
            try:
                async with async_session_maker() as db:
                    await self._load_crops(db)
                    await self._load_livestock(db)
                    await self._load_marketplaces(db)
                self._warmed = True
                logger.info(
                    f"IntentCache warmed: {len(self.crops)} crop keys, "
                    f"{len(self.livestock)} livestock keys, "
                    f"{len(self.marketplaces)} marketplace keys, "
                    f"{len(self.regions)} regions"
                )
            except Exception as e:
                logger.error(f"IntentCache warmup failed: {e}")
                self._warmed = False

    async def refresh(self) -> None:
        self._warmed = False
        self.crops.clear()
        self.livestock.clear()
        self.marketplaces.clear()
        self.regions.clear()
        await self.warmup()

    async def _load_crops(self, db) -> None:
        result = await db.execute(select(Crop).where(Crop.is_active == True))
        for crop in result.scalars().all():
            entity = CropEntity(id=crop.crop_id, en=crop.name, am=crop.name_amharic)
            self.crops[_normalize_en(crop.name)] = entity
            if crop.name_amharic:
                self.crops[_normalize_am(crop.name_amharic)] = entity

        variety_result = await db.execute(select(CropVariety).where(CropVariety.is_active == True))
        for v in variety_result.scalars().all():
            parent = next((c for c in self.crops.values() if c.id == v.crop_id), None)
            if not parent:
                continue
            self.crops[_normalize_en(v.name)] = parent
            if v.name_amharic:
                self.crops[_normalize_am(v.name_amharic)] = parent

    async def _load_livestock(self, db) -> None:
        result = await db.execute(select(Livestock).where(Livestock.is_active == True))
        for ls in result.scalars().all():
            entity = LivestockEntity(id=ls.livestock_id, en=ls.name, am=ls.name_amharic)
            en_key = _normalize_en(ls.name)
            self.livestock[en_key] = entity
            # Last-word alias for gendered multi-word names (e.g. "Male Adult Goat" → "goat")
            words = en_key.split()
            if len(words) > 1 and words[-1] not in self.livestock:
                self.livestock[words[-1]] = entity
            if ls.name_amharic:
                am_key = _normalize_am(ls.name_amharic)
                self.livestock[am_key] = entity
                # Amharic last-word alias (e.g. "ወንድ ጠቦት ፍየል" → "ፍየል")
                am_words = am_key.split()
                if len(am_words) > 1 and am_words[-1] not in self.livestock:
                    self.livestock[am_words[-1]] = entity

        breed_result = await db.execute(select(LivestockBreed).where(LivestockBreed.is_active == True))
        for b in breed_result.scalars().all():
            parent = next((l for l in self.livestock.values() if l.id == b.livestock_id), None)
            if not parent:
                continue
            en_key = _normalize_en(b.name)
            self.livestock[en_key] = parent
            # Also index by last word of multi-word breed names (e.g. "male adult sheep" → "sheep")
            words = en_key.split()
            if len(words) > 1 and words[-1] not in self.livestock:
                self.livestock[words[-1]] = parent
            if b.name_amharic:
                am_key = _normalize_am(b.name_amharic)
                self.livestock[am_key] = parent
                # Also index by last word of multi-word Amharic breed names
                am_words = am_key.split()
                if len(am_words) > 1 and am_words[-1] not in self.livestock:
                    self.livestock[am_words[-1]] = parent

    async def _load_marketplaces(self, db) -> None:
        result = await db.execute(select(Marketplace).where(Marketplace.is_active == True))
        for mp in result.scalars().all():
            entity = MarketplaceEntity(
                id=mp.marketplace_id,
                en=mp.name,
                am=mp.name_amharic,
                region=mp.region,
                latitude=float(mp.latitude) if mp.latitude is not None else None,
                longitude=float(mp.longitude) if mp.longitude is not None else None,
                marketplace_type=mp.marketplace_type,
            )
            self.marketplaces[_normalize_en(mp.name)] = entity
            # Also index by short name after " - " (e.g. "Addis Ababa - Merkato" → "merkato")
            if " - " in mp.name:
                short = mp.name.split(" - ", 1)[1].strip()
                short_key = _normalize_en(short)
                if short_key not in self.marketplaces:
                    self.marketplaces[short_key] = entity
            if mp.name_amharic:
                self.marketplaces[_normalize_am(mp.name_amharic)] = entity
                if " - " in mp.name_amharic:
                    short_am = mp.name_amharic.split(" - ", 1)[1].strip()
                    if short_am not in self.marketplaces:
                        self.marketplaces[_normalize_am(short_am)] = entity
            if mp.region:
                self.regions.add(_normalize_en(mp.region))
            if mp.region_amharic:
                self.regions.add(_normalize_am(mp.region_amharic))

    def lookup_crop(self, token: str) -> Optional[CropEntity]:
        return self.crops.get(_normalize_en(token)) or self.crops.get(_normalize_am(token))

    def lookup_livestock(self, token: str) -> Optional[LivestockEntity]:
        return self.livestock.get(_normalize_en(token)) or self.livestock.get(_normalize_am(token))

    def lookup_marketplace(self, token: str) -> Optional[MarketplaceEntity]:
        return self.marketplaces.get(_normalize_en(token)) or self.marketplaces.get(_normalize_am(token))


intent_cache = IntentCache()
