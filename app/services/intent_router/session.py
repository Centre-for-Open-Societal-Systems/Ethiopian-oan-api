"""Short-lived Redis session for multi-turn follow-ups in the fast path.

Uses the existing aiocache Redis client with a dedicated namespace.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Optional
import json

from app.config import settings
from app.core.cache import cache
from helpers.utils import get_logger

logger = get_logger(__name__)

NAMESPACE = "intent_session"


@dataclass
class SessionEntities:
    crop: Optional[dict] = None
    livestock: Optional[dict] = None
    marketplace: Optional[dict] = None
    region: Optional[str] = None
    location: Optional[str] = None
    timeframe: Optional[str] = None


@dataclass
class SessionState:
    intent: str
    entities: SessionEntities = field(default_factory=SessionEntities)
    awaiting: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps({
            "intent": self.intent,
            "entities": asdict(self.entities),
            "awaiting": self.awaiting,
        })

    @classmethod
    def from_json(cls, raw: str) -> "SessionState":
        data = json.loads(raw)
        return cls(
            intent=data["intent"],
            entities=SessionEntities(**(data.get("entities") or {})),
            awaiting=data.get("awaiting"),
        )


def _key(session_id: str) -> str:
    return f"{NAMESPACE}:{session_id}"


async def load(session_id: str) -> Optional[SessionState]:
    try:
        raw = await cache.get(_key(session_id))
    except Exception as e:
        logger.warning(f"session.load failed: {e}")
        return None
    if not raw:
        return None
    try:
        return SessionState.from_json(raw)
    except Exception as e:
        logger.warning(f"session.load parse failed: {e}")
        return None


async def save(session_id: str, state: SessionState) -> None:
    try:
        await cache.set(_key(session_id), state.to_json(), ttl=settings.intent_router_session_ttl)
    except Exception as e:
        logger.warning(f"session.save failed: {e}")


async def clear(session_id: str) -> None:
    try:
        await cache.delete(_key(session_id))
    except Exception as e:
        logger.warning(f"session.clear failed: {e}")
