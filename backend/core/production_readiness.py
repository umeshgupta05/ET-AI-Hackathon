"""Production readiness guardrails.

The app can run as a hackathon demo or as a production candidate. In production
mode, endpoints that would otherwise use static/demo intelligence are blocked
until real integrations are configured.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from config import config


def _present(value: str | None) -> bool:
    return bool((value or "").strip())


def _database_ready() -> bool:
    # The current operational stores use SQLite directly. A configured URL alone
    # must not claim PostgreSQL readiness until those stores use that adapter.
    return False


def _certified_currency_ready() -> bool:
    manifest = config.deployment.currency_certified_manifest.strip()
    return bool(manifest) and Path(manifest).exists()


def demo_intelligence_allowed() -> bool:
    return not config.deployment.is_production or config.deployment.allow_demo_intelligence


def require_operational_integration(feature: str) -> None:
    """Block demo-only features in production mode."""
    if demo_intelligence_allowed():
        return
    try:
        from stores.operational_store import trusted_operational_counts

        counts = trusted_operational_counts()
    except Exception:
        counts = {}
    if feature == "geospatial" and counts.get("geospatial_incidents", 0) > 0:
        return
    if feature == "graph" and counts.get("graph_entities", 0) > 0 and counts.get("graph_edges", 0) > 0:
        return
    if feature == "currency" and counts.get("currency_certified_specimens", 0) > 0:
        return

    details = {
        "geospatial": "Ingest records marked authorized from an approved NCRB/state/bank/telecom source.",
        "graph": "Ingest authorized transaction, call-record, device-fingerprint, and case records.",
        "demo": "Demo fixtures are disabled when DEPLOYMENT_MODE=production.",
        "currency": "Set CURRENCY_CERTIFIED_MANIFEST to an independently governed bank/RBI/lab manifest.",
    }
    raise HTTPException(
        status_code=503,
        detail={
            "error": "production_integration_required",
            "feature": feature,
            "message": details.get(feature, "A real operational integration is required."),
            "deployment_mode": config.deployment.mode,
        },
    )


def readiness_report(
    *,
    queue_status: dict[str, Any] | None = None,
    redis_status: dict[str, Any] | None = None,
    currency_manifest_installed: bool = False,
    certified_currency_specimens: int = 0,
) -> dict[str, Any]:
    """Return a transparent production-readiness report."""
    queue_status = queue_status or {}
    redis_status = redis_status or {}
    webhook_token = os.getenv("MULTICHANNEL_WEBHOOK_TOKEN", "").strip()
    channel_provider = os.getenv("CHANNEL_WEBHOOK_PROVIDER", "shared").strip().lower()
    twilio_signed = (
        channel_provider == "twilio"
        and _present(os.getenv("TWILIO_AUTH_TOKEN"))
        and _present(os.getenv("TWILIO_WEBHOOK_BASE_URL"))
    )
    from stores.evidence_store import evidence_store_status

    evidence_status = evidence_store_status()

    checks = [
        {
            "id": "jwt_secret",
            "label": "JWT signing secret",
            "required_for_production": True,
            "status": "ready" if len(os.getenv("JWT_SECRET", "")) >= 32 else "missing",
            "detail": "Set JWT_SECRET to a long random value.",
        },
        {
            "id": "debug_disabled",
            "label": "Debug mode disabled",
            "required_for_production": True,
            "status": "ready" if not config.debug else "unsafe",
            "detail": "DEBUG must be false in production.",
        },
        {
            "id": "database",
            "label": "Durable relational database",
            "required_for_production": True,
            "status": "ready" if _database_ready() else "missing",
            "detail": (
                "Current auth, case, job, and real-time stores are SQLite-backed. "
                "A managed PostgreSQL adapter and migration are required for multi-host production."
            ),
        },
        {
            "id": "evidence_store",
            "label": "Evidence object storage",
            "required_for_production": True,
            "status": "ready" if evidence_status.get("status") == "ready" else evidence_status.get("status", "missing"),
            "detail": "Configure a reachable file:// mount or encrypted s3:// evidence store with retention policy.",
        },
        {
            "id": "redis",
            "label": "Distributed rate limiting",
            "required_for_production": True,
            "status": "ready" if redis_status.get("status") == "ready" else redis_status.get("status", "missing"),
            "detail": "Set REDIS_ENABLED=true and provide a managed Redis URL.",
        },
        {
            "id": "rabbitmq",
            "label": "Durable async analysis queue",
            "required_for_production": True,
            "status": "ready" if queue_status.get("status") == "ready" else queue_status.get("status", "missing"),
            "detail": "Set ASYNC_JOBS_ENABLED=true and provide a managed RabbitMQ URL.",
        },
        {
            "id": "geospatial_feeds",
            "label": "Authorized geospatial and complaint feeds",
            "required_for_production": True,
            "status": "ready"
            if all(
                _present(value)
                for value in (
                    config.deployment.ncrb_feed_url,
                    config.deployment.state_feed_url,
                    config.deployment.bank_feed_url,
                    config.deployment.telecom_feed_url,
                )
            )
            else "missing",
            "detail": "Set NCRB_FEED_URL, STATE_FEED_URL, BANK_FEED_URL, and TELECOM_FEED_URL.",
        },
        {
            "id": "graph_feeds",
            "label": "Fraud graph live data feeds",
            "required_for_production": True,
            "status": "ready"
            if _present(config.deployment.bank_feed_url) and _present(config.deployment.telecom_feed_url)
            else "missing",
            "detail": "Graph intelligence requires bank and telecom/entity feeds.",
        },
        {
            "id": "currency_certification",
            "label": "Certified currency validation data",
            "required_for_production": True,
            "status": "ready" if _certified_currency_ready() or certified_currency_specimens > 0 else "research_only",
            "detail": "Current local dataset can be research-ready but is not RBI/lab certified.",
            "research_manifest_installed": currency_manifest_installed,
            "certified_specimens": certified_currency_specimens,
        },
        {
            "id": "channel_security",
            "label": "Signed public channel webhooks",
            "required_for_production": True,
            "status": "ready" if twilio_signed or _present(webhook_token) else "missing",
            "detail": "Use Twilio signature validation in production or configure the shared token for a controlled sandbox.",
        },
        {
            "id": "realtime_alert_delivery",
            "label": "Citizen, telecom, and MHA alert delivery",
            "required_for_production": True,
            "status": "ready"
            if all(_present(os.getenv(name)) for name in (
                "CITIZEN_ALERT_WEBHOOK_URL", "TELECOM_ALERT_WEBHOOK_URL",
                "MHA_ALERT_WEBHOOK_URL", "ALERT_WEBHOOK_SECRET",
            ))
            else "missing",
            "detail": "Configure all signed alert destination webhooks before claiming automated external alert delivery.",
        },
        {
            "id": "pii_hashing",
            "label": "PII correlation hashing",
            "required_for_production": True,
            "status": "ready" if len(os.getenv("PII_HASH_SECRET", "")) >= 32 else "missing",
            "detail": "Set a dedicated PII_HASH_SECRET of at least 32 characters.",
        },
        {
            "id": "official_reporting_bridge",
            "label": "Official reporting API bridge",
            "required_for_production": True,
            "status": "ready"
            if _present(config.deployment.official_reporting_api_url)
            and _present(config.deployment.official_reporting_api_token)
            else "missing",
            "detail": "Set OFFICIAL_REPORTING_API_URL and OFFICIAL_REPORTING_API_TOKEN for authorized complaint submission.",
        },
        {
            "id": "observability",
            "label": "OpenTelemetry exporter",
            "required_for_production": True,
            "status": "ready" if _present(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")) else "missing",
            "detail": "Set OTEL_EXPORTER_OTLP_ENDPOINT and route traces through a redacted collector.",
        },
        {
            "id": "whatsapp_media",
            "label": "WhatsApp media ingestion",
            "required_for_production": True,
            "status": "ready" if config.deployment.whatsapp_media_integration else "missing",
            "detail": "Enable verified provider media download before claiming WhatsApp image/audio analysis.",
        },
        {
            "id": "ivr_provider",
            "label": "IVR provider configuration",
            "required_for_production": True,
            "status": "ready" if config.deployment.ivr_provider_configured else "missing",
            "detail": "Set IVR_PROVIDER_CONFIGURED=true only after provider numbers/webhooks are live.",
        },
    ]

    blockers = [
        check
        for check in checks
        if check["required_for_production"] and check["status"] != "ready"
    ]
    return {
        "deployment_mode": config.deployment.mode,
        "is_production": config.deployment.is_production,
        "production_ready": not blockers,
        "demo_intelligence_allowed": demo_intelligence_allowed(),
        "checks": checks,
        "blockers": blockers,
        "claim_guardrail": (
            "Production claims are blocked until every required check is ready."
            if blockers
            else "All configured production readiness checks are ready."
        ),
    }
