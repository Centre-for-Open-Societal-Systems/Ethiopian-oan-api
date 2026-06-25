"""
Langfuse client bootstrap (Langfuse v3/v4 OTel-based API).

Provides:
- A singleton `Langfuse` client
- A `@observe` decorator that degrades to a no-op when Langfuse is disabled
- Safe helpers to attach trace/observation metadata, input, output, user_id, session_id, tags

All helpers silently no-op if Langfuse is not configured, so call sites can use them
unconditionally.
"""

import os
from typing import Any, Callable, List, Optional

from helpers.utils import get_logger

logger = get_logger(__name__)

_client = None
_enabled: Optional[bool] = None


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def is_enabled() -> bool:
    """Return True if Langfuse is configured and enabled via env."""
    global _enabled
    if _enabled is not None:
        return _enabled

    if not _truthy(os.getenv("LANGFUSE_ENABLED", "true")):
        _enabled = False
        return False

    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
    _enabled = bool(pk and sk and host)
    if not _enabled:
        logger.info("Langfuse disabled: missing LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL")
    return _enabled


def get_client():
    """Return the shared Langfuse client, or None if disabled/unavailable."""
    global _client
    if _client is not None:
        return _client
    if not is_enabled():
        return None
    try:
        from langfuse import Langfuse

        host = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
        _client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=host,
        )
        logger.info(f"Langfuse client initialized (host={host})")
    except Exception as e:
        logger.warning(f"Langfuse init failed, tracing disabled: {e}")
        _client = None
    return _client


def observe(*dec_args, **dec_kwargs) -> Callable:
    """
    `@observe` wrapper that degrades to a no-op when Langfuse is disabled.
    Supports `@observe` and `@observe(name=..., as_type=...)`.
    """
    # Bare `@observe` usage
    if len(dec_args) == 1 and callable(dec_args[0]) and not dec_kwargs:
        return _wrap(dec_args[0])

    def decorator(func: Callable) -> Callable:
        return _wrap(func, *dec_args, **dec_kwargs)

    return decorator


def _wrap(func: Callable, *dec_args, **dec_kwargs) -> Callable:
    if not is_enabled():
        return func
    try:
        from langfuse import observe as _real_observe
    except Exception:
        return func

    try:
        if dec_args or dec_kwargs:
            return _real_observe(*dec_args, **dec_kwargs)(func)
        return _real_observe(func)
    except Exception as e:
        logger.debug(f"Langfuse observe decoration failed on {func.__name__}: {e}")
        return func


from contextlib import contextmanager


@contextmanager
def span_context(
    name: str,
    *,
    as_type: Optional[str] = None,
    input: Any = None,
    metadata: Any = None,
):
    """
    Context manager that opens a Langfuse span as the current observation and
    closes it on exit. Works inside async generators where `@observe` loses the
    span across `yield` points. No-op when Langfuse is disabled.

    Usage:
        with span_context("chat.stream", input={...}) as span:
            ...
            yield chunk
    """
    if not is_enabled():
        yield None
        return
    client = get_client()
    if client is None:
        yield None
        return
    try:
        kwargs: dict = {"name": name}
        if as_type is not None:
            kwargs["as_type"] = as_type
        if input is not None:
            kwargs["input"] = input
        if metadata is not None:
            kwargs["metadata"] = metadata
        with client.start_as_current_observation(**kwargs) as span:
            yield span
    except Exception as e:
        logger.debug(f"span_context({name}) failed, continuing without trace: {e}")
        yield None


def get_openai_module():
    """
    Return the Langfuse-wrapped `openai` module when tracing is enabled,
    otherwise fall back to the real `openai`. Usage:
        openai = get_openai_module()
        client = openai.AsyncOpenAI(...)
    """
    if is_enabled():
        try:
            from langfuse import openai as lf_openai  # type: ignore

            return lf_openai
        except Exception as e:
            logger.debug(f"Langfuse OpenAI wrapper unavailable: {e}")
    import openai

    return openai


def update_current_trace(
    *,
    name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    input: Any = None,
    output: Any = None,
    metadata: Any = None,
) -> None:
    """
    Attach trace-level attributes (user_id, session_id, tags, name, io) to the
    currently active span. In Langfuse v4, these are OTel span attributes.
    """
    if not is_enabled():
        return
    try:
        from opentelemetry import trace
        from langfuse import LangfuseOtelSpanAttributes as A

        span = trace.get_current_span()
        if span is None or not getattr(span, "is_recording", lambda: False)():
            return
        if name is not None:
            span.set_attribute(A.TRACE_NAME, str(name))
        if user_id is not None:
            span.set_attribute(A.TRACE_USER_ID, str(user_id))
        if session_id is not None:
            span.set_attribute(A.TRACE_SESSION_ID, str(session_id))
        if tags:
            span.set_attribute(A.TRACE_TAGS, list(tags))
        client = get_client()
        if client is not None and (input is not None or output is not None):
            client.set_current_trace_io(input=input, output=output)
        if metadata is not None and client is not None:
            # Metadata lives on the observation/span; mirror onto root span
            client.update_current_span(metadata=metadata)
    except Exception as e:
        logger.debug(f"update_current_trace failed: {e}")


def update_current_observation(
    *,
    input: Any = None,
    output: Any = None,
    metadata: Any = None,
    model: Optional[str] = None,
    usage: Any = None,
) -> None:
    """
    Update the currently active observation. Prefers `update_current_generation`
    when generation-specific fields (model, usage) are provided; otherwise
    falls back to `update_current_span`.
    """
    if not is_enabled():
        return
    client = get_client()
    if client is None:
        return
    try:
        if model is not None or usage is not None:
            kwargs: dict = {}
            if input is not None:
                kwargs["input"] = input
            if output is not None:
                kwargs["output"] = output
            if metadata is not None:
                kwargs["metadata"] = metadata
            if model is not None:
                kwargs["model"] = model
            if usage is not None:
                kwargs["usage_details"] = usage
            client.update_current_generation(**kwargs)
        else:
            kwargs = {}
            if input is not None:
                kwargs["input"] = input
            if output is not None:
                kwargs["output"] = output
            if metadata is not None:
                kwargs["metadata"] = metadata
            if kwargs:
                client.update_current_span(**kwargs)
    except Exception as e:
        logger.debug(f"update_current_observation failed: {e}")


def flush() -> None:
    """Flush pending events. Call on graceful shutdown."""
    client = get_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as e:
        logger.warning(f"Langfuse flush failed: {e}")


def shutdown() -> None:
    """Flush and shutdown the Langfuse client."""
    client = get_client()
    if client is None:
        return
    try:
        client.flush()
        if hasattr(client, "shutdown"):
            client.shutdown()
    except Exception as e:
        logger.warning(f"Langfuse shutdown failed: {e}")
