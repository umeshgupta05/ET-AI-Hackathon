"""Opt-in RabbitMQ integration smoke test.

Run with ASYNC_JOBS_ENABLED=true and a reachable RABBITMQ_URL.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.broker import QUEUE_NAME, close_broker, connect, declare_topology, publish_job


async def smoke() -> None:
    if os.getenv("ASYNC_JOBS_ENABLED", "false").lower() != "true":
        raise RuntimeError("Set ASYNC_JOBS_ENABLED=true before running this test")
    job_id = str(uuid.uuid4())
    await publish_job(job_id, "broker-integration-test")

    connection = await connect()
    try:
        channel = await connection.channel()
        await declare_topology(channel)
        queue = await channel.get_queue(QUEUE_NAME)
        message = await queue.get(timeout=5, fail=False)
        assert message is not None, "Published message was not delivered"
        payload = json.loads(message.body.decode("utf-8"))
        assert payload == {"job_id": job_id}, "Broker payload leaked analysis data"
        assert message.message_id == job_id
        await message.ack()
    finally:
        await connection.close()
        await close_broker()

    print("RabbitMQ publisher/consumer privacy contract: PASS")


if __name__ == "__main__":
    asyncio.run(smoke())
