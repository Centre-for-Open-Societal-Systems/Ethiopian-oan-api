"""Fast-path handlers: one per intent.

Each handler runs a direct DB query or API call and returns a rendered
response via Jinja templates. Returns None when data is not available so the
caller can fall through to the LLM.
"""
from __future__ import annotations
from typing import Optional
import os

from sqlalchemy import select, desc, distinct

from app.core.cache import cache as redis_cache
from app.database import async_session_maker
from app.models.market import Crop, Livestock, MarketPrice, Marketplace
from app.services.intent_router import templates
from app.services.intent_router.entities import ExtractedEntities
from helpers.utils import get_logger

logger = get_logger(__name__)

CROP_UNIT_EN = "per quintal"
CROP_UNIT_AM = "በኩንታል"
LIVESTOCK_UNIT_EN = "per head"
LIVESTOCK_UNIT_AM = "በራስ"
CURRENCY = "ETB"

PRICE_CACHE_TTL = 900
LIST_CACHE_TTL = 3600


def _fmt_price(v) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


async def handle_crop_price(entities: ExtractedEntities, lang: str) -> Optional[str]:
    if not (entities.crop and entities.marketplace):
        return None

    cache_key = f"fastpath:crop_price:{entities.crop.id}:{entities.marketplace.id}:{lang}"
    cached = await redis_cache.get(cache_key)
    if cached:
        return cached

    async with async_session_maker() as db:
        stmt = (
            select(MarketPrice)
            .where(
                MarketPrice.crop_id == entities.crop.id,
                MarketPrice.marketplace_id == entities.marketplace.id,
            )
            .order_by(desc(MarketPrice.price_date))
            .limit(1)
        )
        result = await db.execute(stmt)
        row = result.scalar_one_or_none()

    if not row:
        return None

    crop_name = entities.crop.am if lang == "am" and entities.crop.am else entities.crop.en
    marketplace_name = (
        entities.marketplace.am if lang == "am" and entities.marketplace.am else entities.marketplace.en
    )

    rendered = templates.render(
        "price_found",
        lang=lang,
        item_type="crop",
        item=crop_name,
        marketplace=marketplace_name,
        min_price=_fmt_price(row.min_price),
        max_price=_fmt_price(row.max_price),
        avg_price=_fmt_price(row.avg_price),
        currency=CURRENCY,
        unit=CROP_UNIT_AM if lang == "am" else CROP_UNIT_EN,
        updated_at=row.price_date.strftime("%Y-%m-%d"),
    )
    return await _cache_and_return(cache_key, rendered, PRICE_CACHE_TTL)


async def _cache_and_return(key: str, value: str, ttl: int) -> str:
    try:
        await redis_cache.set(key, value, ttl=ttl)
    except Exception as e:
        logger.warning(f"fastpath cache set failed: {e}")
    return value


async def handle_livestock_price(entities: ExtractedEntities, lang: str) -> Optional[str]:
    if not (entities.livestock and entities.marketplace):
        return None

    cache_key = f"fastpath:livestock_price:{entities.livestock.id}:{entities.marketplace.id}:{lang}"
    cached = await redis_cache.get(cache_key)
    if cached:
        return cached

    async with async_session_maker() as db:
        stmt = (
            select(MarketPrice)
            .where(
                MarketPrice.livestock_id == entities.livestock.id,
                MarketPrice.marketplace_id == entities.marketplace.id,
            )
            .order_by(desc(MarketPrice.price_date))
            .limit(1)
        )
        result = await db.execute(stmt)
        row = result.scalar_one_or_none()

    if not row:
        return None

    livestock_name = (
        entities.livestock.am if lang == "am" and entities.livestock.am else entities.livestock.en
    )
    marketplace_name = (
        entities.marketplace.am if lang == "am" and entities.marketplace.am else entities.marketplace.en
    )

    rendered = templates.render(
        "price_found",
        lang=lang,
        item_type="livestock",
        item=livestock_name,
        marketplace=marketplace_name,
        min_price=_fmt_price(row.min_price),
        max_price=_fmt_price(row.max_price),
        avg_price=_fmt_price(row.avg_price),
        currency=CURRENCY,
        unit=LIVESTOCK_UNIT_AM if lang == "am" else LIVESTOCK_UNIT_EN,
        updated_at=row.price_date.strftime("%Y-%m-%d"),
    )
    return await _cache_and_return(cache_key, rendered, PRICE_CACHE_TTL)


async def handle_crop_listing(entities: ExtractedEntities, lang: str) -> Optional[str]:
    if not entities.marketplace:
        return None

    cache_key = f"fastpath:crop_listing:{entities.marketplace.id}:{lang}"
    cached = await redis_cache.get(cache_key)
    if cached:
        return cached

    async with async_session_maker() as db:
        stmt = (
            select(Crop.name, Crop.name_amharic)
            .join(MarketPrice, MarketPrice.crop_id == Crop.crop_id)
            .where(MarketPrice.marketplace_id == entities.marketplace.id)
            .where(Crop.is_active == True)
            .distinct()
            .order_by(Crop.name)
        )
        result = await db.execute(stmt)
        rows = result.all()

    if not rows:
        return None

    items = [
        {"en": r.name, "am": r.name_amharic or r.name}
        for r in rows
    ]
    marketplace_name = (
        entities.marketplace.am if lang == "am" and entities.marketplace.am else entities.marketplace.en
    )

    rendered = templates.render(
        "listing_crops",
        lang=lang,
        marketplace=marketplace_name,
        items=items,
    )
    return await _cache_and_return(cache_key, rendered, LIST_CACHE_TTL)


async def handle_livestock_listing(entities: ExtractedEntities, lang: str) -> Optional[str]:
    if not entities.marketplace:
        return None

    cache_key = f"fastpath:livestock_listing:{entities.marketplace.id}:{lang}"
    cached = await redis_cache.get(cache_key)
    if cached:
        return cached

    async with async_session_maker() as db:
        stmt = (
            select(Livestock.name, Livestock.name_amharic)
            .join(MarketPrice, MarketPrice.livestock_id == Livestock.livestock_id)
            .where(MarketPrice.marketplace_id == entities.marketplace.id)
            .where(Livestock.is_active == True)
            .distinct()
            .order_by(Livestock.name)
        )
        result = await db.execute(stmt)
        rows = result.all()

    if not rows:
        return None

    items = [
        {"en": r.name, "am": r.name_amharic or r.name}
        for r in rows
    ]
    marketplace_name = (
        entities.marketplace.am if lang == "am" and entities.marketplace.am else entities.marketplace.en
    )

    rendered = templates.render(
        "listing_livestock",
        lang=lang,
        marketplace=marketplace_name,
        items=items,
    )
    return await _cache_and_return(cache_key, rendered, LIST_CACHE_TTL)


async def handle_marketplace_listing(entities: ExtractedEntities, lang: str) -> Optional[str]:
    cache_key = f"fastpath:marketplace_listing:{entities.region or 'all'}:{lang}"
    cached = await redis_cache.get(cache_key)
    if cached:
        return cached

    async with async_session_maker() as db:
        stmt = select(Marketplace).where(Marketplace.is_active == True)
        if entities.region:
            from sqlalchemy import or_, func
            region = entities.region.strip().lower()
            stmt = stmt.where(
                or_(
                    func.lower(Marketplace.region) == region,
                    func.lower(Marketplace.region).contains(region),
                    Marketplace.region_amharic == entities.region,
                )
            )
        stmt = stmt.order_by(Marketplace.name)
        result = await db.execute(stmt)
        marketplaces = result.scalars().all()

    if not marketplaces:
        return None

    items = [
        {
            "en": mp.name,
            "am": mp.name_amharic or mp.name,
            "region": mp.region,
            "type": mp.marketplace_type,
        }
        for mp in marketplaces
    ]

    rendered = templates.render(
        "listing_marketplaces",
        lang=lang,
        region=entities.region,
        items=items,
    )
    return await _cache_and_return(cache_key, rendered, LIST_CACHE_TTL)


async def handle_weather_current(entities: ExtractedEntities, lang: str) -> Optional[str]:
    if not entities.location:
        return None

    lat, lon = None, None
    if entities.marketplace and entities.marketplace.latitude and entities.marketplace.longitude:
        lat, lon = entities.marketplace.latitude, entities.marketplace.longitude

    from agents.tools.weather_tool import get_current_weather, CurrentWeatherInput

    try:
        weather = await get_current_weather(
            CurrentWeatherInput(
                latitude=lat,
                longitude=lon,
                location=None if lat and lon else entities.location,
                units="metric",
                language=lang,
            )
        )
    except Exception as e:
        logger.warning(f"weather fast-path failed: {e}")
        return None

    return templates.render(
        "weather_current",
        lang=lang,
        location=entities.location,
        temperature=f"{weather.temperature:.1f}",
        feels_like=f"{weather.feels_like:.1f}",
        humidity=weather.humidity,
        wind_speed=f"{weather.wind_speed:.1f}",
        description=weather.description,
    )


async def handle_weather_forecast(entities: ExtractedEntities, lang: str) -> Optional[str]:
    if not entities.location:
        return None

    lat, lon = None, None
    if entities.marketplace and entities.marketplace.latitude and entities.marketplace.longitude:
        lat, lon = entities.marketplace.latitude, entities.marketplace.longitude

    from agents.tools.weather_tool import get_weather_forecast, ForecastInput

    try:
        forecast_text = await get_weather_forecast(
            ForecastInput(
                latitude=lat,
                longitude=lon,
                location=None if lat and lon else entities.location,
                units="metric",
                language=lang,
            )
        )
    except Exception as e:
        logger.warning(f"weather forecast fast-path failed: {e}")
        return None

    return templates.render(
        "weather_forecast",
        lang=lang,
        location=entities.location,
        timeframe=entities.timeframe or "upcoming",
        forecast_text=forecast_text,
    )
