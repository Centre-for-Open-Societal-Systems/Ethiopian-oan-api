"""
FastAPI Main Application

This is the entry point for the MahaVistaar AI API FastAPI application.
"""
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routers import chat_router, suggestions_router, transcribe_router, tts_router, conversation_router
from app.routers.health import router as health_router
from app.core.cache import cache
from app.database import close_db
from helpers.utils import get_logger

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting up MahaVistaar AI API...")

    # Initialize Langfuse client (no-op if disabled/misconfigured)
    try:
        from helpers.langfuse_client import get_client, is_enabled
        if is_enabled():
            get_client()
            logger.info("✅ Langfuse tracing enabled")
        else:
            logger.info("ℹ️  Langfuse tracing disabled")
    except Exception as e:
        logger.warning(f"⚠️ Langfuse init skipped: {e}")

    # Test cache connection
    try:
        await cache.set("health_check", "ok", ttl=60)
        test_value = await cache.get("health_check")
        if test_value == "ok":
            logger.info("✅ Cache connection successful")
        else:
            logger.warning("⚠️ Cache connection issue - values not persisting correctly")
    except Exception as e:
        logger.error(f"❌ Cache connection failed: {str(e)}")

    # Initialize database connection pool
    logger.info("✅ Database engine initialized")

    # Initialize telemetry DB pool
    if settings.telemetry_db_url:
        try:
            from app.tasks.telemetry import init_telemetry_pool
            await init_telemetry_pool(settings.telemetry_db_url)
            logger.info("✅ Telemetry DB pool initialized")
        except Exception as e:
            logger.warning(f"⚠️ Telemetry DB pool init failed: {e}")

    # Pre-load TTS models to avoid first-request memory/latency spike
    try:
        from app.services.providers.tts import get_tts_provider
        tts = get_tts_provider()
        await tts._load_model("en")
        await tts._load_model("am")
        logger.info("✅ TTS models pre-loaded (en, am)")
    except Exception as e:
        logger.warning(f"⚠️ TTS pre-load failed (will load on first request): {e}")

    logger.info("✅ Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down MahaVistaar AI API...")

    # Close telemetry pool
    try:
        from app.tasks.telemetry import close_telemetry_pool
        await close_telemetry_pool()
        logger.info("✅ Telemetry pool closed")
    except Exception as e:
        logger.warning(f"⚠️ Telemetry pool close failed: {e}")

    # Cleanup TTS provider
    try:
        from app.services.providers.tts import cleanup_tts_provider
        cleanup_tts_provider()
        logger.info("✅ TTS provider cleaned up")
    except Exception as e:
        logger.warning(f"⚠️ TTS cleanup failed: {e}")

    # Flush Langfuse events
    try:
        from helpers.langfuse_client import shutdown as langfuse_shutdown
        langfuse_shutdown()
        logger.info("✅ Langfuse flushed")
    except Exception as e:
        logger.warning(f"⚠️ Langfuse flush failed: {e}")

    # Close database connections
    await close_db()
    logger.info("✅ Database connections closed")

    logger.info("✅ Application shutdown complete")

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    app = FastAPI(
        title=settings.app_name,
        description="AI-powered agricultural assistant API for Maharashtra farmers",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=settings.allowed_credentials,
        allow_methods=settings.allowed_methods,
        allow_headers=settings.allowed_headers,
    )
    
    # Include routers
    app.include_router(chat_router, prefix=settings.api_prefix)
    app.include_router(suggestions_router, prefix=settings.api_prefix)
    app.include_router(transcribe_router, prefix=settings.api_prefix)
    app.include_router(tts_router, prefix=settings.api_prefix)
    app.include_router(health_router, prefix=settings.api_prefix)
    app.include_router(conversation_router, prefix=settings.api_prefix)
    return app

# Create the app instance
app = create_app()

if __name__ == "__main__":
    logger.info(f"Starting {settings.app_name} server...")
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.uvicorn_workers,
        log_level=settings.log_level.lower()
    )
