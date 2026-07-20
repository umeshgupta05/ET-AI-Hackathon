"""Optional MCP adapter for authorized analyst clients.

Run separately from the public API with ``python mcp_server.py``. The server uses
stdio and calls the existing HTTP API, keeping authentication and audit controls
at the application boundary.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations


load_dotenv(Path(__file__).with_name(".env"))
API_BASE = os.getenv("SHIELD_API_BASE_URL", "http://localhost:8000")
API_TOKEN = os.getenv("SHIELD_API_TOKEN", "")
mcp = FastMCP(
    "Digital Public Safety Shield",
    instructions=(
        "Use Shield tools for fraud triage and evidence review. Treat every verdict as "
        "decision support, preserve human review, and never claim that a report was filed "
        "or a banknote was certified. Authenticated tools require SHIELD_API_TOKEN."
    ),
)

READ_ONLY_CLOSED = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
READ_ONLY_EXTERNAL = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
)
ADDITIVE_EXTERNAL = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=True,
)
ADDITIVE_CLOSED = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)


async def _request(method: str, path: str, **kwargs) -> dict[str, Any]:
    headers = kwargs.pop("headers", {})
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    async with httpx.AsyncClient(base_url=API_BASE, timeout=120) as client:
        response = await client.request(method, path, headers=headers, **kwargs)
        response.raise_for_status()
        return response.json()


def _require_token() -> None:
    if not API_TOKEN:
        raise RuntimeError("SHIELD_API_TOKEN is required for owner-scoped analyst tools")


@mcp.tool(
    title="Shield Service Readiness",
    annotations=READ_ONLY_CLOSED,
    structured_output=True,
)
async def shield_health() -> dict[str, Any]:
    """Return public model, broker, and service readiness information."""
    return await _request("GET", "/api/health")


@mcp.tool(
    title="Analyze Suspicious Text",
    annotations=ADDITIVE_EXTERNAL,
    structured_output=True,
)
async def analyze_fraud_text(text: str, language: str = "en") -> dict[str, Any]:
    """Analyze suspicious text and save the resulting case to the token owner's audit history."""
    _require_token()
    return await _request(
        "POST",
        "/api/analyze/text",
        json={"text": text, "language": language},
    )


@mcp.tool(
    title="Queue Fraud Text Analysis",
    annotations=ADDITIVE_CLOSED,
    structured_output=True,
)
async def queue_fraud_text(text: str, language: str = "en") -> dict[str, Any]:
    """Submit durable background text analysis through RabbitMQ and return its job ID."""
    _require_token()
    return await _request(
        "POST",
        "/api/jobs/analyze/text",
        json={"text": text, "language": language},
    )


@mcp.tool(
    title="Get Analysis Job",
    annotations=READ_ONLY_CLOSED,
    structured_output=True,
)
async def get_analysis_job(job_id: str) -> dict[str, Any]:
    """Get an owner-scoped queued analysis status and result."""
    _require_token()
    return await _request("GET", f"/api/jobs/{quote(job_id, safe='')}")


@mcp.tool(
    title="List Recent Shield Cases",
    annotations=READ_ONLY_CLOSED,
    structured_output=True,
)
async def list_recent_cases(limit: int = 20) -> dict[str, Any]:
    """List recent cases owned by the authenticated analyst, capped at 100."""
    _require_token()
    payload = await _request("GET", "/api/history")
    bounded_limit = max(1, min(100, limit))
    return {"items": payload.get("items", [])[:bounded_limit], "limit": bounded_limit}


@mcp.tool(
    title="Get Case Evidence Package",
    annotations=READ_ONLY_CLOSED,
    structured_output=True,
)
async def get_case_evidence(case_id: str) -> dict[str, Any]:
    """Retrieve the owner-scoped integrity-hashed evidence package for a saved case."""
    _require_token()
    return await _request("GET", f"/api/cases/{quote(case_id, safe='')}/evidence")


@mcp.tool(
    title="Fraud Hotspot Overview",
    annotations=READ_ONLY_CLOSED,
    structured_output=True,
)
async def hotspot_overview(
    latitude: float | None = None,
    longitude: float | None = None,
) -> dict[str, Any]:
    """Return demo hotspot intelligence, optionally ranked around a supplied location."""
    if (latitude is None) != (longitude is None):
        raise ValueError("latitude and longitude must be supplied together")
    params = None if latitude is None else {"latitude": latitude, "longitude": longitude}
    return await _request("GET", "/api/intelligence/hotspots", params=params)


@mcp.tool(
    title="Fraud Network Summary",
    annotations=READ_ONLY_CLOSED,
    structured_output=True,
)
async def fraud_network_summary() -> dict[str, Any]:
    """Return risk and community analysis for the demonstration fraud graph."""
    return await _request("GET", "/api/graph/analyze")


@mcp.tool(
    title="Official Reporting Guidance",
    annotations=READ_ONLY_EXTERNAL,
    structured_output=True,
)
async def reporting_guidance(risk_level: str = "medium") -> dict[str, Any]:
    """Return official reporting channels and evidence-preservation guidance."""
    return await _request(
        "GET", "/api/reporting/guidance", params={"risk_level": risk_level}
    )


@mcp.resource(
    "shield://capabilities",
    title="Shield Capabilities and Readiness",
    description="Current public model, integration, and safety readiness as JSON.",
    mime_type="application/json",
)
async def capabilities_resource() -> str:
    return json.dumps(await shield_health(), ensure_ascii=False)


@mcp.resource(
    "shield://reporting-guidance",
    title="Fraud Reporting Guidance",
    description="Official reporting and evidence-preservation guidance for medium risk.",
    mime_type="application/json",
)
async def reporting_resource() -> str:
    return json.dumps(await reporting_guidance("medium"), ensure_ascii=False)


@mcp.prompt(
    title="Human-Reviewed Fraud Triage",
    description="Guide an analyst through evidence-preserving, human-reviewed fraud triage.",
)
def fraud_triage(language: str = "en") -> str:
    return (
        f"Triage the citizen report in language '{language}'. First analyze the supplied text or "
        "queue it when latency matters. Explain the strongest evidence and uncertainty, review "
        "related case evidence only when authorized, then provide official reporting guidance. "
        "Keep a human decision-maker in control. Never claim autonomous enforcement, complaint "
        "filing, identity verification, or currency certification."
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
