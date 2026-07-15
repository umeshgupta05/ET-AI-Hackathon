"""Optional RabbitMQ transport for durable asynchronous analysis."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any


EXCHANGE_NAME = "shield.analysis"
QUEUE_NAME = "shield.analysis.jobs"
ROUTING_KEY = "analysis.requested"
DLX_NAME = "shield.analysis.dlx"
DLQ_NAME = "shield.analysis.dead"
DLQ_ROUTING_KEY = "analysis.dead"
_publisher_connection = None
_publisher_lock = asyncio.Lock()


def jobs_enabled() -> bool:
    return os.getenv("ASYNC_JOBS_ENABLED", "false").lower() == "true"


def rabbitmq_url() -> str:
    return os.getenv("RABBITMQ_URL", "amqp://shield:change-me@localhost:5672/")


async def connect():
    """Open a recovering connection. Import is lazy so RabbitMQ stays optional."""
    if not jobs_enabled():
        raise RuntimeError("Asynchronous jobs are disabled")
    try:
        import aio_pika
    except ImportError as exc:
        raise RuntimeError("aio-pika is not installed") from exc
    return await aio_pika.connect_robust(rabbitmq_url(), timeout=5)


async def _get_publisher_connection():
    global _publisher_connection
    if _publisher_connection is not None and not _publisher_connection.is_closed:
        return _publisher_connection
    async with _publisher_lock:
        if _publisher_connection is None or _publisher_connection.is_closed:
            _publisher_connection = await connect()
    return _publisher_connection


async def close_broker() -> None:
    global _publisher_connection
    if _publisher_connection is not None and not _publisher_connection.is_closed:
        await _publisher_connection.close()
    _publisher_connection = None


async def declare_topology(channel) -> None:
    import aio_pika

    exchange = await channel.declare_exchange(
        EXCHANGE_NAME, aio_pika.ExchangeType.DIRECT, durable=True
    )
    dead_exchange = await channel.declare_exchange(
        DLX_NAME, aio_pika.ExchangeType.DIRECT, durable=True
    )
    dead_queue = await channel.declare_queue(
        DLQ_NAME,
        durable=True,
        arguments={"x-queue-type": "quorum"},
    )
    await dead_queue.bind(dead_exchange, DLQ_ROUTING_KEY)
    queue = await channel.declare_queue(
        QUEUE_NAME,
        durable=True,
        arguments={
            "x-queue-type": "quorum",
            "x-dead-letter-exchange": DLX_NAME,
            "x-dead-letter-routing-key": DLQ_ROUTING_KEY,
            "x-delivery-limit": int(os.getenv("RABBITMQ_DELIVERY_LIMIT", "3")),
            "x-overflow": "reject-publish",
            "x-max-length": int(os.getenv("RABBITMQ_MAX_QUEUE_LENGTH", "10000")),
        },
    )
    await queue.bind(exchange, ROUTING_KEY)


async def publish_job(job_id: str, correlation_id: str | None = None) -> None:
    import aio_pika

    connection = await _get_publisher_connection()
    channel = None
    try:
        channel = await connection.channel(
            publisher_confirms=True,
            on_return_raises=True,
        )
        await declare_topology(channel)
        exchange = await channel.get_exchange(EXCHANGE_NAME)
        message = aio_pika.Message(
            body=json.dumps({"job_id": job_id}).encode("utf-8"),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            message_id=job_id,
            correlation_id=correlation_id or job_id,
        )
        await exchange.publish(message, routing_key=ROUTING_KEY, mandatory=True)
    finally:
        if channel is not None and not channel.is_closed:
            await channel.close()


async def broker_status() -> dict[str, Any]:
    if not jobs_enabled():
        return {"enabled": False, "status": "disabled"}
    try:
        connection = await connect()
        try:
            channel = await connection.channel(publisher_confirms=True)
            await declare_topology(channel)
        finally:
            await connection.close()
        return {"enabled": True, "status": "ready", "queue": QUEUE_NAME}
    except Exception as exc:
        return {"enabled": True, "status": "unavailable", "detail": str(exc)}
