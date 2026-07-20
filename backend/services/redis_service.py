"""Optional Redis coordination for distributed rate limiting.

Redis deliberately stores only hashed limiter identifiers and short-lived counters.
Analysis payloads and model results remain outside Redis.
"""

from __future__ import annotations

import hashlib
import logging
import os
import asyncio
from typing import Any


logger = logging.getLogger(__name__)
_client = None
_last_error: str | None = None
_initialization_lock = asyncio.Lock()

RATE_LIMIT_SCRIPT = """
local current = redis.call('INCR', KEYS[1])
if current == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
local ttl = redis.call('TTL', KEYS[1])
return {current, ttl}
"""


def redis_enabled() -> bool:
    return os.getenv("REDIS_ENABLED", "false").lower() == "true"


def _redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://:change-me@localhost:6379/0")


def _key(namespace: str, identifier: str) -> str:
    digest = hashlib.sha256(identifier.strip().lower().encode("utf-8")).hexdigest()
    return f"shield:limit:{namespace}:{digest}"


async def initialize_redis() -> bool:
    """Create and validate the process-wide async connection pool."""
    global _client, _last_error
    if not redis_enabled():
        return False
    async with _initialization_lock:
        if _client is not None:
            try:
                await _client.ping()
                _last_error = None
                return True
            except Exception:
                stale_client = _client
                _client = None
                await stale_client.aclose()
        try:
            import redis.asyncio as redis

            _client = redis.from_url(
                _redis_url(),
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
                health_check_interval=30,
                max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "20")),
            )
            await _client.ping()
            _last_error = None
            logger.info("Redis coordination ready")
            return True
        except Exception as exc:
            _last_error = str(exc)
            failed_client = _client
            _client = None
            if failed_client is not None:
                await failed_client.aclose()
            logger.warning("Redis unavailable; local fallbacks remain active: %s", exc)
            return False


async def close_redis() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def consume_rate_limit(
    namespace: str,
    identifier: str,
    *,
    limit: int,
    window_seconds: int,
) -> dict[str, int | bool | str] | None:
    """Atomically consume one allowance, or return None when Redis is optional/unavailable."""
    if not redis_enabled() or not await initialize_redis():
        return None
    try:
        current, ttl = await _client.eval(
            RATE_LIMIT_SCRIPT,
            1,
            _key(namespace, identifier),
            max(1, window_seconds),
        )
        remaining = max(0, limit - int(current))
        return {
            "allowed": int(current) <= limit,
            "limit": limit,
            "remaining": remaining,
            "retry_after": max(1, int(ttl)),
            "backend": "redis",
        }
    except Exception as exc:
        global _last_error
        _last_error = str(exc)
        logger.warning("Redis rate limiter failed open: %s", exc)
        return None


async def reset_rate_limit(namespace: str, identifier: str) -> None:
    if not redis_enabled() or not await initialize_redis():
        return
    try:
        await _client.delete(_key(namespace, identifier))
    except Exception as exc:
        logger.warning("Could not reset Redis limiter: %s", exc)


async def redis_status() -> dict[str, Any]:
    if not redis_enabled():
        return {"enabled": False, "status": "disabled", "purpose": "distributed_rate_limiting"}
    ready = await initialize_redis()
    return {
        "enabled": True,
        "status": "ready" if ready else "unavailable",
        "purpose": "distributed_rate_limiting",
        "stores_analysis_payloads": False,
        **({"detail": _last_error} if not ready and _last_error else {}),
    }
