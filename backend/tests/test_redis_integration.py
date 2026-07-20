"""Redis integration test for pooled readiness and atomic distributed limits."""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
os.environ["REDIS_ENABLED"] = "true"
os.environ.setdefault("REDIS_URL", "redis://:change-me@localhost:16379/0")

from services.redis_service import (
    close_redis,
    consume_rate_limit,
    initialize_redis,
    redis_status,
    reset_rate_limit,
)


async def smoke() -> None:
    assert await initialize_redis()
    identifier = f"integration-{uuid.uuid4()}"
    first = await consume_rate_limit("test", identifier, limit=2, window_seconds=30)
    second = await consume_rate_limit("test", identifier, limit=2, window_seconds=30)
    third = await consume_rate_limit("test", identifier, limit=2, window_seconds=30)
    assert first and first["allowed"] and first["remaining"] == 1
    assert second and second["allowed"] and second["remaining"] == 0
    assert third and not third["allowed"] and third["retry_after"] > 0
    await reset_rate_limit("test", identifier)
    reset = await consume_rate_limit("test", identifier, limit=2, window_seconds=30)
    assert reset and reset["allowed"] and reset["remaining"] == 1
    status = await redis_status()
    assert status["status"] == "ready"
    assert status["stores_analysis_payloads"] is False
    await reset_rate_limit("test", identifier)
    await close_redis()
    print("Redis pooled readiness and distributed rate limit: PASS")


if __name__ == "__main__":
    asyncio.run(smoke())
