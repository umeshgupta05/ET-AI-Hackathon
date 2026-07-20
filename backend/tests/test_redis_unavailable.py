"""Regression check for optional Redis when the service is unreachable."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

os.environ["REDIS_ENABLED"] = "true"
os.environ["REDIS_URL"] = "redis://:change-me@127.0.0.1:6399/0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.redis_service import close_redis, initialize_redis


async def main() -> None:
    results = await asyncio.gather(initialize_redis(), initialize_redis())
    assert results == [False, False], results
    await close_redis()


if __name__ == "__main__":
    asyncio.run(main())
    print("Unavailable Redis concurrency fallback: PASS")
