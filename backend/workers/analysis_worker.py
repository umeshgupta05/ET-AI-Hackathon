"""RabbitMQ worker for authenticated text-analysis jobs."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
load_dotenv(BACKEND_DIR / ".env")

from agents.orchestrator import FusionOrchestrator
from services.broker import QUEUE_NAME, close_broker, connect, declare_topology, publish_job
from stores.job_store import (
    claim_job,
    complete_job,
    fail_job,
    init_job_db,
    mark_retrying,
    recover_stale_jobs,
    renew_lease,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("analysis-worker")
DELIVERY_LIMIT = int(os.getenv("RABBITMQ_DELIVERY_LIMIT", "3"))


async def run_worker() -> None:
    init_job_db()
    orchestrator = FusionOrchestrator()
    await orchestrator.initialize()
    connection = await connect()
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=int(os.getenv("RABBITMQ_PREFETCH", "1")))
    await declare_topology(channel)
    queue = await channel.get_queue(QUEUE_NAME)

    async def recover_expired_leases() -> None:
        while True:
            await asyncio.sleep(30)
            for stale_job_id in recover_stale_jobs():
                try:
                    await publish_job(stale_job_id, "worker-lease-recovery")
                    logger.warning("Requeued stale analysis job %s", stale_job_id)
                except Exception:
                    logger.exception("Could not requeue stale job %s", stale_job_id)

    async def process_message(message) -> None:
        try:
            envelope = json.loads(message.body.decode("utf-8"))
            job_id = str(envelope["job_id"])
        except (ValueError, KeyError, UnicodeDecodeError):
            logger.error("Rejecting malformed message %s", message.message_id)
            await message.reject(requeue=False)
            return

        job = claim_job(job_id)
        if not job:
            logger.info("Ignoring already-claimed or completed job %s", job_id)
            await message.ack()
            return

        async def keep_lease_alive() -> None:
            while True:
                await asyncio.sleep(30)
                if not renew_lease(job_id):
                    logger.warning("Stopped lease heartbeat for inactive job %s", job_id)
                    return

        lease_task = asyncio.create_task(keep_lease_alive())
        try:
            payload = job["input"] or {}
            result = await orchestrator.process(
                text=payload["text"],
                context=payload.get("context"),
                language=payload.get("language", "en"),
            )
            complete_job(job_id, result)
            await message.ack()
            logger.info("Completed analysis job %s", job_id)
        except Exception as exc:
            delivery_count = int(message.headers.get("x-delivery-count", 0))
            if delivery_count + 1 >= DELIVERY_LIMIT:
                fail_job(job_id, str(exc))
                await message.reject(requeue=False)
                logger.exception("Job %s exhausted retries", job_id)
            else:
                mark_retrying(job_id, str(exc))
                await message.reject(requeue=True)
                logger.warning("Retrying job %s after attempt %s", job_id, delivery_count + 1)
        finally:
            lease_task.cancel()
            with suppress(asyncio.CancelledError):
                await lease_task

    await queue.consume(process_message, no_ack=False)
    recovery_task = asyncio.create_task(recover_expired_leases())
    logger.info("Worker consuming %s", QUEUE_NAME)
    try:
        await asyncio.Future()
    finally:
        recovery_task.cancel()
        with suppress(asyncio.CancelledError):
            await recovery_task
        await close_broker()
        await connection.close()


if __name__ == "__main__":
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.info("Worker stopped")
