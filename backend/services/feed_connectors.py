"""Polling connectors for authorized geospatial, bank, telecom, and state feeds."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import httpx

from config import config
from stores.operational_store import (
    upsert_geospatial_incident,
    upsert_graph_edge,
    upsert_graph_entity,
)


logger = logging.getLogger(__name__)
_task: asyncio.Task | None = None
_stop = asyncio.Event()
_status: dict[str, Any] = {
    "running": False,
    "last_poll_at": None,
    "connectors": {},
}


def _definitions() -> list[dict[str, str]]:
    deployment = config.deployment
    return [
        {"name": "ncrb", "url": deployment.ncrb_feed_url, "token": os.getenv("NCRB_FEED_TOKEN", "")},
        {"name": "bank", "url": deployment.bank_feed_url, "token": os.getenv("BANK_FEED_TOKEN", "")},
        {"name": "telecom", "url": deployment.telecom_feed_url, "token": os.getenv("TELECOM_FEED_TOKEN", "")},
        {"name": "state", "url": deployment.state_feed_url, "token": os.getenv("STATE_FEED_TOKEN", "")},
    ]


def _validate_endpoint(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"}:
        raise ValueError("Feed URL must use HTTP(S)")
    if config.deployment.is_production and parsed.scheme != "https":
        raise ValueError("Production feed URL must use HTTPS")


def _records(payload: Any, *keys: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _ingest(payload: Any, connector: str) -> dict[str, int]:
    counts = {"geospatial_incidents": 0, "graph_entities": 0, "graph_edges": 0}
    incidents = _records(payload, "geospatial_incidents", "incidents", "hotspots")
    entities = _records(payload, "graph_entities", "entities", "nodes")
    edges = _records(payload, "graph_edges", "edges", "relationships")
    source = f"authorized_{connector}_feed"
    for item in incidents:
        record = {**item, "source": item.get("source") or source, "provenance_tier": "authorized"}
        upsert_geospatial_incident(record)
        counts["geospatial_incidents"] += 1
    for item in entities:
        record = {**item, "source": item.get("source") or source, "provenance_tier": "authorized"}
        upsert_graph_entity(record)
        counts["graph_entities"] += 1
    for item in edges:
        record = {**item, "source": item.get("source") or source, "provenance_tier": "authorized"}
        upsert_graph_edge(record)
        counts["graph_edges"] += 1
    return counts


async def poll_once() -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    connector_status: dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=False) as client:
        for definition in _definitions():
            name, url, token = definition["name"], definition["url"].strip(), definition["token"].strip()
            if not url:
                connector_status[name] = {"status": "not_configured"}
                continue
            try:
                _validate_endpoint(url)
                headers = {"Accept": "application/json"}
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "json" not in content_type:
                    raise ValueError("Feed response must be JSON")
                counts = _ingest(response.json(), name)
                connector_status[name] = {
                    "status": "ready",
                    "last_success_at": now,
                    "ingested": counts,
                }
            except Exception as exc:
                connector_status[name] = {"status": "error", "error": str(exc)[:500]}
                logger.warning("%s feed poll failed: %s", name, exc)
    _status.update({"last_poll_at": now, "connectors": connector_status})
    return feed_status()


async def _poll_loop() -> None:
    interval = max(15, int(os.getenv("FEED_POLL_INTERVAL_SECONDS", "300")))
    _status["running"] = True
    try:
        while not _stop.is_set():
            await poll_once()
            try:
                await asyncio.wait_for(_stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
    finally:
        _status["running"] = False


def start_feed_pollers() -> None:
    global _task
    enabled = os.getenv("FEED_POLLING_ENABLED", "false").lower() == "true"
    if not enabled or _task:
        return
    _stop.clear()
    _task = asyncio.create_task(_poll_loop(), name="authorized-feed-pollers")


async def stop_feed_pollers() -> None:
    global _task
    if not _task:
        return
    _stop.set()
    await _task
    _task = None


def feed_status() -> dict[str, Any]:
    configured = [item["name"] for item in _definitions() if item["url"].strip()]
    return {
        **_status,
        "enabled": os.getenv("FEED_POLLING_ENABLED", "false").lower() == "true",
        "configured_connectors": configured,
        "poll_interval_seconds": max(15, int(os.getenv("FEED_POLL_INTERVAL_SECONDS", "300"))),
    }
