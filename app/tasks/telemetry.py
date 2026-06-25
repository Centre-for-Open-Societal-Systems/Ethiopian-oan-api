import json
import asyncpg
from typing import Dict, Optional
from helpers.utils import get_logger

logger = get_logger(__name__)

_pool: Optional[asyncpg.Pool] = None


async def init_telemetry_pool(dsn: str):
    global _pool
    _pool = await asyncpg.create_pool(dsn, min_size=2, max_size=5)
    logger.info("Telemetry DB pool initialized")


async def close_telemetry_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def send_telemetry(telemetry_data: Dict) -> Dict:
    if not _pool:
        logger.warning("Telemetry pool not initialized, skipping")
        return {"status": "skipped"}

    try:
        events = telemetry_data.get("events", [])
        message_json = json.dumps({"events": events})

        await _pool.execute(
            "INSERT INTO winston_logs (level, message, timestamp, meta) "
            "VALUES ($1, $2, NOW(), $3::jsonb)",
            "info", message_json, "{}"
        )
        
        return {"status": "inserted", "event_count": len(events)}
    except Exception as e:
        logger.error(f"Telemetry insert failed: {e}")
        return {"status": "error", "error": str(e)}
