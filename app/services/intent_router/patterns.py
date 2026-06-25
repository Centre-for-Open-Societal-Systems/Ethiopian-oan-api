import re
from enum import Enum


class Intent(str, Enum):
    CROP_PRICE = "crop_price"
    LIVESTOCK_PRICE = "livestock_price"
    CROP_LISTING = "crop_listing"
    LIVESTOCK_LISTING = "livestock_listing"
    MARKETPLACE_LISTING = "marketplace_listing"
    WEATHER_CURRENT = "weather_current"
    WEATHER_FORECAST = "weather_forecast"
    UNKNOWN = "unknown"


PRICE_EN = re.compile(r"\b(price|cost|rate|how much|selling for|worth|going for)\b", re.I)
PRICE_AM = re.compile(r"(ዋጋ|ስንት|ምን ያህል)")

WEATHER_EN = re.compile(r"\b(weather|forecast|rain|temperature|climate|humidity|wind)\b", re.I)
WEATHER_AM = re.compile(r"(የአየር ሁኔታ|ዝናብ|ሙቀት|ነፋስ)")

FORECAST_EN = re.compile(r"\b(forecast|tomorrow|next (week|day)|will it|upcoming)\b", re.I)
FORECAST_AM = re.compile(r"(ነገ|በሚቀጥለው|ትንበያ)")

LISTING_EN = re.compile(r"\b(list|show me|what.*(selling|available|crops|livestock|markets|marketplaces)|which.*(market|marketplace)|selling at)\b", re.I)
LISTING_AM = re.compile(r"(ዝርዝር|አሳየኝ|የትኞቹ|የሚገኙ)")

LIVESTOCK_KEYWORDS_EN = re.compile(r"\b(livestock|cattle|cow|ox|oxen|bull|goat|sheep|camel|chicken|hen|rooster|donkey|horse|mule)\b", re.I)
LIVESTOCK_KEYWORDS_AM = re.compile(r"(ከብት|በሬ|ላም|ፍየል|በግ|ግመል|ዶሮ|አህያ|ፈረስ|በቅሎ)")

CROP_KEYWORDS_HINT_EN = re.compile(r"\b(crop|teff|wheat|barley|maize|corn|sorghum|coffee|onion|potato|tomato|bean|pea|lentil)\b", re.I)
CROP_KEYWORDS_HINT_AM = re.compile(r"(ጤፍ|ስንዴ|ገብስ|በቆሎ|ማሽላ|ቡና|ሽንኩርት|ድንች|ቲማቲም|ባቄላ|ምስር|ሰብል)")

MARKETPLACE_LISTING_EN = re.compile(r"\b(markets|marketplaces)\b", re.I)
MARKETPLACE_LISTING_AM = re.compile(r"(ገበያዎች|የገበያ ቦታዎች)")


def detect_intent(query: str, lang: str) -> Intent:
    q = query.strip()
    if not q:
        return Intent.UNKNOWN

    am = lang == "am"
    has_price = bool((PRICE_AM if am else PRICE_EN).search(q)) or (not am and bool(PRICE_EN.search(q)))
    has_weather = bool((WEATHER_AM if am else WEATHER_EN).search(q))
    has_forecast = bool((FORECAST_AM if am else FORECAST_EN).search(q))
    has_listing = bool((LISTING_AM if am else LISTING_EN).search(q))
    has_livestock = bool((LIVESTOCK_KEYWORDS_AM if am else LIVESTOCK_KEYWORDS_EN).search(q))
    has_crop_hint = bool((CROP_KEYWORDS_HINT_AM if am else CROP_KEYWORDS_HINT_EN).search(q))
    has_marketplace_listing = bool((MARKETPLACE_LISTING_AM if am else MARKETPLACE_LISTING_EN).search(q))

    if has_weather:
        return Intent.WEATHER_FORECAST if has_forecast else Intent.WEATHER_CURRENT

    if has_price:
        if has_livestock:
            return Intent.LIVESTOCK_PRICE
        return Intent.CROP_PRICE

    if has_listing:
        if has_marketplace_listing:
            return Intent.MARKETPLACE_LISTING
        if has_livestock:
            return Intent.LIVESTOCK_LISTING
        if has_crop_hint or "crop" in q.lower() or "ሰብል" in q:
            return Intent.CROP_LISTING
        return Intent.MARKETPLACE_LISTING

    return Intent.UNKNOWN
