"""Real-time call risk sessions and signed alert delivery.

The service stores a minimal, privacy-preserving event trail. Raw caller and
account identifiers are never persisted; callers supply values and this module
stores keyed hashes suitable for correlation across authorized feeds.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from stores.database import SessionLocal
from stores.models import RealtimeSession, RealtimeEvent, AlertOutbox

ALERT_DESTINATIONS = ("citizen", "telecom", "mha")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _keyed_hash(value: str | None) -> str | None:
    if not value:
        return None
    secret = os.getenv("PII_HASH_SECRET") or os.getenv("JWT_SECRET") or "demo-only-change-me"
    return hmac.new(secret.encode(), value.strip().encode(), hashlib.sha256).hexdigest()


def create_session(
    *,
    channel: str,
    language: str,
    caller_id: str | None = None,
    participant_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    now = _now()
    clean_metadata = dict(metadata or {})
    clean_metadata.pop("caller_id", None)
    clean_metadata.pop("participant_id", None)
    
    with SessionLocal() as db:
        new_session = RealtimeSession(
            id=session_id,
            channel=channel,
            language=language,
            status="active",
            caller_hash=_keyed_hash(caller_id),
            participant_hash=_keyed_hash(participant_id),
            risk_score=0.0,
            risk_level="safe",
            event_count=0,
            started_at=now,
            updated_at=now,
            metadata_json=clean_metadata
        )
        db.add(new_session)
        db.commit()
        db.refresh(new_session)
        
    return get_session(session_id)


def _risk_level(score: float) -> str:
    if score >= 0.80:
        return "critical"
    if score >= 0.65:
        return "high"
    if score >= 0.45:
        return "review"
    if score >= 0.25:
        return "low"
    return "safe"


def score_call_signals(metadata: dict[str, Any]) -> tuple[float, list[str]]:
    """Score provider and behavioral signals independent of transcript text."""
    weighted: list[tuple[float, float, str]] = []
    verification = str(metadata.get("caller_verification", "unknown")).lower()
    attestation = str(metadata.get("stir_shaken_attestation", "unavailable")).upper()
    if verification == "failed":
        weighted.append((1.0, 0.18, "caller identity verification failed"))
    elif verification == "verified":
        weighted.append((0.0, 0.12, "caller identity independently verified"))
    if attestation in {"C", "FAILED"}:
        weighted.append((0.9, 0.16, "weak or failed calling-number attestation"))
    elif attestation == "A":
        weighted.append((0.0, 0.10, "full calling-number attestation"))

    direct_signals = (
        ("spoof_risk", 0.14, "telecom spoofing signal"),
        ("face_swap_score", 0.12, "video face-swap signal"),
        ("video_identity_mismatch", 0.12, "video identity mismatch"),
        ("payment_requested", 0.14, "payment requested during call"),
        ("secrecy_requested", 0.10, "caller demanded secrecy"),
        ("screen_share_requested", 0.08, "screen sharing requested"),
        ("remote_app_requested", 0.12, "remote-access app requested"),
    )
    for key, weight, reason in direct_signals:
        value = metadata.get(key, False)
        numeric = float(value) if isinstance(value, (int, float)) else (1.0 if value else 0.0)
        numeric = max(0.0, min(1.0, numeric))
        if numeric:
            weighted.append((numeric, weight, reason))

    urgency_seconds = metadata.get("urgency_seconds")
    if urgency_seconds is not None and float(urgency_seconds) <= 1800:
        weighted.append((1.0, 0.10, "short payment or compliance deadline"))
    authority = str(metadata.get("claimed_authority", "")).strip()
    if authority and verification != "verified":
        weighted.append((0.75, 0.10, f"unverified authority claim: {authority[:48]}"))

    if not weighted:
        return 0.0, []
    total_weight = sum(weight for _, weight, _ in weighted)
    score = sum(value * weight for value, weight, _ in weighted) / max(total_weight, 0.01)
    reasons = [reason for value, _, reason in weighted if value >= 0.5]
    return round(max(0.0, min(1.0, score)), 4), reasons


def append_event(
    session_id: str,
    *,
    transcript: str,
    metadata: dict[str, Any],
    model_score: float,
    model_verdict: str,
) -> dict[str, Any]:
    with SessionLocal() as db:
        session = db.query(RealtimeSession).filter(RealtimeSession.id == session_id).first()
        if not session:
            raise KeyError("Realtime session not found")
        if session.status != "active":
            raise ValueError("Realtime session is closed")
        sequence = session.event_count + 1

        clean_metadata = dict(metadata or {})
        for key in ("caller_id", "participant_id", "destination_account", "phone_number", "account_id"):
            if clean_metadata.get(key):
                clean_metadata[f"{key}_hash"] = _keyed_hash(str(clean_metadata.pop(key)))

        signal_score, reasons = score_call_signals(clean_metadata)
        model_score = max(0.0, min(1.0, float(model_score)))
        # Content remains primary, but verified provider signals can move a live decision quickly.
        combined = model_score * 0.68 + signal_score * 0.32
        previous = session.risk_score
        combined = max(combined, previous * 0.92)
        level = _risk_level(combined)
        occurred_at = str(clean_metadata.get("occurred_at") or _now())
        event_id = str(uuid.uuid4())
        evidence = {
            "session_id": session_id,
            "sequence": sequence,
            "transcript": transcript,
            "metadata": clean_metadata,
            "signal_score": round(signal_score, 4),
            "model_score": round(model_score, 4),
            "model_verdict": model_verdict,
            "combined_score": round(combined, 4),
            "occurred_at": occurred_at,
        }
        evidence_hash = hashlib.sha256(
            json.dumps(evidence, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

        event = RealtimeEvent(
            id=event_id,
            session_id=session_id,
            sequence=sequence,
            transcript=transcript,
            metadata_json=clean_metadata,
            signal_score=signal_score,
            model_score=model_score,
            combined_score=combined,
            reasons_json=reasons,
            occurred_at=occurred_at,
            evidence_hash=evidence_hash
        )
        db.add(event)
        
        session.risk_score = combined
        session.risk_level = level
        session.event_count = sequence
        session.updated_at = _now()
        
        db.commit()

    return {
        **evidence,
        "event_id": event_id,
        "risk_level": level,
        "signal_reasons": reasons,
        "evidence_hash": evidence_hash,
    }


def get_transcript(session_id: str) -> str:
    with SessionLocal() as db:
        events = db.query(RealtimeEvent).filter(RealtimeEvent.session_id == session_id).order_by(RealtimeEvent.sequence).all()
        return "\n".join(e.transcript for e in events if e.transcript.strip())


def get_session(session_id: str) -> dict[str, Any]:
    with SessionLocal() as db:
        row = db.query(RealtimeSession).filter(RealtimeSession.id == session_id).first()
        if not row:
            raise KeyError("Realtime session not found")
        events = db.query(RealtimeEvent).filter(RealtimeEvent.session_id == session_id).order_by(RealtimeEvent.sequence).all()
        
        return {
            "session_id": row.id,
            "channel": row.channel,
            "language": row.language,
            "status": row.status,
            "risk_score": round(row.risk_score, 4),
            "risk_level": row.risk_level,
            "event_count": row.event_count,
            "started_at": row.started_at,
            "updated_at": row.updated_at,
            "closed_at": row.closed_at,
            "metadata": row.metadata_json,
            "events": [
                {
                    "event_id": event.id,
                    "sequence": event.sequence,
                    "transcript": event.transcript,
                    "metadata": event.metadata_json,
                    "signal_score": event.signal_score,
                    "model_score": event.model_score,
                    "combined_score": event.combined_score,
                    "signal_reasons": event.reasons_json,
                    "occurred_at": event.occurred_at,
                    "evidence_hash": event.evidence_hash,
                }
                for event in events
            ],
        }


def close_session(session_id: str) -> dict[str, Any]:
    now = _now()
    with SessionLocal() as db:
        session = db.query(RealtimeSession).filter(RealtimeSession.id == session_id).first()
        if not session:
            raise KeyError("Realtime session not found")
        session.status = 'closed'
        session.closed_at = now
        session.updated_at = now
        db.commit()
        
    return get_session(session_id)


def ensure_alerts(session_id: str, event: dict[str, Any]) -> list[dict[str, Any]]:
    if float(event["combined_score"]) < float(os.getenv("REALTIME_ALERT_THRESHOLD", "0.65")):
        return []
    payload = {
        "schema": "shield.alert.v1",
        "session_id": session_id,
        "risk_score": event["combined_score"],
        "risk_level": event["risk_level"],
        "signal_reasons": event["signal_reasons"],
        "evidence_hash": event["evidence_hash"],
        "occurred_at": event["occurred_at"],
        "recommended_action": "Interrupt payment, preserve evidence, and contact 1930/NCRP.",
    }
    created: list[dict[str, Any]] = []
    
    with SessionLocal() as db:
        for destination in ALERT_DESTINATIONS:
            idempotency_key = hashlib.sha256(
                f"{session_id}:{destination}:{event['event_id']}".encode()
            ).hexdigest()
            alert_id = str(uuid.uuid4())
            now = _now()
            
            # Check if exists
            existing = db.query(AlertOutbox).filter(AlertOutbox.idempotency_key == idempotency_key).first()
            if existing:
                created.append(_alert_dict(existing))
                continue
                
            outbox = AlertOutbox(
                id=alert_id,
                session_id=session_id,
                destination=destination,
                idempotency_key=idempotency_key,
                status="pending",
                attempt_count=0,
                payload_json={**payload, "destination": destination},
                payload_hash=hashlib.sha256(
                    json.dumps({**payload, "destination": destination}, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest(),
                created_at=now,
                updated_at=now
            )
            db.add(outbox)
            try:
                db.commit()
                db.refresh(outbox)
                created.append(_alert_dict(outbox))
            except IntegrityError:
                db.rollback()
                existing = db.query(AlertOutbox).filter(AlertOutbox.idempotency_key == idempotency_key).first()
                if existing:
                    created.append(_alert_dict(existing))
                
    return created


def _alert_dict(row: AlertOutbox) -> dict[str, Any]:
    return {
        "alert_id": row.id,
        "session_id": row.session_id,
        "destination": row.destination,
        "idempotency_key": row.idempotency_key,
        "status": row.status,
        "attempt_count": row.attempt_count,
        "payload": row.payload_json,
        "payload_hash": row.payload_hash,
        "last_error": row.last_error,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def list_alerts(session_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    with SessionLocal() as db:
        query = db.query(AlertOutbox).order_by(AlertOutbox.created_at.desc())
        if session_id:
            query = query.filter(AlertOutbox.session_id == session_id)
        rows = query.limit(limit).all()
        return [_alert_dict(row) for row in rows]


async def dispatch_alerts(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    endpoints = {
        "citizen": os.getenv("CITIZEN_ALERT_WEBHOOK_URL", "").strip(),
        "telecom": os.getenv("TELECOM_ALERT_WEBHOOK_URL", "").strip(),
        "mha": os.getenv("MHA_ALERT_WEBHOOK_URL", "").strip(),
    }
    secret = os.getenv("ALERT_WEBHOOK_SECRET", "").strip()
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        for alert in alerts:
            endpoint = endpoints.get(alert["destination"], "")
            if not endpoint:
                status, error = "pending_integration", "Destination webhook is not configured"
            else:
                body = json.dumps(alert["payload"], sort_keys=True, ensure_ascii=False).encode()
                signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest() if secret else ""
                try:
                    response = await client.post(
                        endpoint,
                        content=body,
                        headers={
                            "Content-Type": "application/json",
                            "Idempotency-Key": alert["idempotency_key"],
                            "X-Shield-Signature": f"sha256={signature}",
                        },
                    )
                    response.raise_for_status()
                    status, error = "delivered", None
                except Exception as exc:
                    status, error = "retry_required", str(exc)[:500]
                    
            with SessionLocal() as db:
                outbox = db.query(AlertOutbox).filter(AlertOutbox.id == alert["alert_id"]).first()
                if outbox:
                    outbox.status = status
                    outbox.attempt_count += 1
                    outbox.last_error = error
                    outbox.updated_at = _now()
                    db.commit()
                    db.refresh(outbox)
                    results.append(_alert_dict(outbox))
    return results
