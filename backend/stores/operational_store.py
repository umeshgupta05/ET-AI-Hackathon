"""Persistent operational intelligence store."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.dialects.postgresql import insert
from .database import engine, SessionLocal
from .models import (
    Base, GeospatialIncident, GraphEntity, GraphEdge,
    ReportingDraft, CurrencyCertifiedSpecimen, ModelFeedback
)

PROVENANCE_TIERS = {"authorized", "public_research", "synthetic_sandbox"}

def _provenance_tier(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or record.get("attrs") or {}
    tier = str(record.get("provenance_tier") or metadata.get("provenance_tier") or "").strip().lower()
    if tier in PROVENANCE_TIERS:
        return tier
    source = str(record.get("source") or "").strip().lower()
    if source.startswith("authorized"):
        return "authorized"
    if source.startswith("public"):
        return "public_research"
    return "synthetic_sandbox"

def init_operational_db() -> None:
    Base.metadata.create_all(bind=engine)

def upsert_geospatial_incident(record: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    metadata = dict(record.get("metadata") or {})
    provenance_tier = _provenance_tier(record)
    metadata["provenance_tier"] = provenance_tier
    clean = {
        "id": str(record["id"]),
        "district": str(record["district"]),
        "lat": float(record["lat"]),
        "lon": float(record["lon"]),
        "type": str(record["type"]),
        "reports": int(record.get("reports", 1)),
        "severity": max(0.0, min(1.0, float(record.get("severity", 0.5)))),
        "source": str(record.get("source") or "authorized_feed"),
        "source_reference": record.get("source_reference"),
        "occurred_at": record.get("occurred_at"),
        "received_at": now,
        "metadata": metadata,
        "provenance_tier": provenance_tier,
    }
    with SessionLocal() as db:
        stmt = insert(GeospatialIncident).values(
            id=clean["id"],
            district=clean["district"],
            lat=clean["lat"],
            lon=clean["lon"],
            type=clean["type"],
            reports=clean["reports"],
            severity=clean["severity"],
            source=clean["source"],
            source_reference=clean["source_reference"],
            occurred_at=clean["occurred_at"],
            received_at=clean["received_at"],
            metadata_json=metadata
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=['id'],
            set_={
                'district': stmt.excluded.district,
                'lat': stmt.excluded.lat,
                'lon': stmt.excluded.lon,
                'type': stmt.excluded.type,
                'reports': stmt.excluded.reports,
                'severity': stmt.excluded.severity,
                'source': stmt.excluded.source,
                'source_reference': stmt.excluded.source_reference,
                'occurred_at': stmt.excluded.occurred_at,
                'received_at': stmt.excluded.received_at,
                'metadata_json': stmt.excluded.metadata_json,
            }
        )
        db.execute(stmt)
        db.commit()
    return clean

def list_geospatial_incidents(limit: int = 1000) -> list[dict[str, Any]]:
    with SessionLocal() as db:
        incidents = db.query(GeospatialIncident).order_by(GeospatialIncident.received_at.desc()).limit(limit).all()
    return [
        {
            "id": row.id,
            "district": row.district,
            "lat": row.lat,
            "lon": row.lon,
            "type": row.type,
            "reports": row.reports,
            "severity": row.severity,
            "source": row.source,
            "source_reference": row.source_reference,
            "occurred_at": row.occurred_at,
            "received_at": row.received_at,
            "metadata": row.metadata_json or {},
            "provenance_tier": (row.metadata_json or {}).get(
                "provenance_tier", "synthetic_sandbox"
            ),
        }
        for row in incidents
    ]

def upsert_graph_entity(record: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    attrs = dict(record.get("attrs") or {})
    provenance_tier = _provenance_tier(record)
    attrs["provenance_tier"] = provenance_tier
    entity_id = str(record["id"])
    entity_type = str(record.get("type") or record.get("entity_type") or "unknown")
    label = str(record.get("label") or "unknown")
    attrs.update({"type": entity_type, "label": label})
    clean = {
        "id": entity_id,
        "type": entity_type,
        "label": label,
        "source": str(record.get("source") or "authorized_feed"),
        "source_reference": record.get("source_reference"),
        "updated_at": now,
        "attrs": attrs,
        "provenance_tier": provenance_tier,
    }
    with SessionLocal() as db:
        stmt = insert(GraphEntity).values(
            id=clean["id"],
            entity_type=clean["type"],
            label=clean["label"],
            source=clean["source"],
            source_reference=clean["source_reference"],
            updated_at=clean["updated_at"],
            attrs_json=attrs
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=['id'],
            set_={
                'entity_type': stmt.excluded.entity_type,
                'label': stmt.excluded.label,
                'source': stmt.excluded.source,
                'source_reference': stmt.excluded.source_reference,
                'updated_at': stmt.excluded.updated_at,
                'attrs_json': stmt.excluded.attrs_json,
            }
        )
        db.execute(stmt)
        db.commit()
    return clean

def upsert_graph_edge(record: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    source_id = str(record["source_id"])
    target_id = str(record["target_id"])
    edge_type = str(record.get("type") or record.get("edge_type") or "connected")
    edge_id = str(record.get("id") or f"{source_id}:{edge_type}:{target_id}")
    attrs = dict(record.get("attrs") or {})
    provenance_tier = _provenance_tier(record)
    attrs["provenance_tier"] = provenance_tier
    clean = {
        "id": edge_id,
        "source_id": source_id,
        "target_id": target_id,
        "type": edge_type,
        "weight": float(record.get("weight", 1.0)),
        "source": str(record.get("source") or "authorized_feed"),
        "source_reference": record.get("source_reference"),
        "observed_at": record.get("observed_at"),
        "updated_at": now,
        "attrs": attrs,
        "provenance_tier": provenance_tier,
    }
    with SessionLocal() as db:
        stmt = insert(GraphEdge).values(
            id=clean["id"],
            source_id=clean["source_id"],
            target_id=clean["target_id"],
            edge_type=clean["type"],
            weight=clean["weight"],
            source=clean["source"],
            source_reference=clean["source_reference"],
            observed_at=clean["observed_at"],
            updated_at=clean["updated_at"],
            attrs_json=attrs
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=['id'],
            set_={
                'source_id': stmt.excluded.source_id,
                'target_id': stmt.excluded.target_id,
                'edge_type': stmt.excluded.edge_type,
                'weight': stmt.excluded.weight,
                'source': stmt.excluded.source,
                'source_reference': stmt.excluded.source_reference,
                'observed_at': stmt.excluded.observed_at,
                'updated_at': stmt.excluded.updated_at,
                'attrs_json': stmt.excluded.attrs_json,
            }
        )
        db.execute(stmt)
        db.commit()
    return clean

def list_graph_entities(limit: int = 10000) -> list[dict[str, Any]]:
    with SessionLocal() as db:
        entities = db.query(GraphEntity).limit(limit).all()
    return [
        {
            "id": row.id,
            "type": row.entity_type,
            "label": row.label,
            "source": row.source,
            "source_reference": row.source_reference,
            "updated_at": row.updated_at,
            "attrs": row.attrs_json or {},
            "provenance_tier": (row.attrs_json or {}).get(
                "provenance_tier", "synthetic_sandbox"
            ),
        }
        for row in entities
    ]

def list_graph_edges(limit: int = 20000) -> list[dict[str, Any]]:
    with SessionLocal() as db:
        edges = db.query(GraphEdge).limit(limit).all()
    return [
        {
            "id": row.id,
            "source_id": row.source_id,
            "target_id": row.target_id,
            "type": row.edge_type,
            "weight": row.weight,
            "source": row.source,
            "source_reference": row.source_reference,
            "observed_at": row.observed_at,
            "updated_at": row.updated_at,
            "attrs": row.attrs_json or {},
            "provenance_tier": (row.attrs_json or {}).get(
                "provenance_tier", "synthetic_sandbox"
            ),
        }
        for row in edges
    ]

def operational_counts() -> dict[str, int]:
    with SessionLocal() as db:
        return {
            "geospatial_incidents": db.query(GeospatialIncident).count(),
            "graph_entities": db.query(GraphEntity).count(),
            "graph_edges": db.query(GraphEdge).count(),
            "reporting_drafts": db.query(ReportingDraft).count(),
            "currency_certified_specimens": db.query(CurrencyCertifiedSpecimen).count(),
        }

def operational_provenance_counts() -> dict[str, dict[str, int]]:
    result = {
        "geospatial_incidents": {tier: 0 for tier in PROVENANCE_TIERS},
        "graph_entities": {tier: 0 for tier in PROVENANCE_TIERS},
        "graph_edges": {tier: 0 for tier in PROVENANCE_TIERS},
        "currency_certified_specimens": {tier: 0 for tier in PROVENANCE_TIERS},
    }
    with SessionLocal() as db:
        table_specs = (
            ("geospatial_incidents", db.query(GeospatialIncident).all(), "metadata_json"),
            ("graph_entities", db.query(GraphEntity).all(), "attrs_json"),
            ("graph_edges", db.query(GraphEdge).all(), "attrs_json"),
            ("currency_certified_specimens", db.query(CurrencyCertifiedSpecimen).all(), "metadata_json"),
        )
        for table, rows, column in table_specs:
            for row in rows:
                metadata = getattr(row, column) or {}
                tier = _provenance_tier({
                    "source": getattr(row, "source", ""),
                    "metadata": metadata,
                })
                result[table][tier] += 1
    return result

def trusted_operational_counts() -> dict[str, int]:
    grouped = operational_provenance_counts()
    return {name: tiers["authorized"] for name, tiers in grouped.items()}

def save_reporting_draft(record: dict[str, Any]) -> dict[str, Any]:
    with SessionLocal() as db:
        stmt = insert(ReportingDraft).values(
            id=record["id"],
            case_id=record.get("case_id"),
            user_id=record.get("user_id"),
            destination=record["destination"],
            status=record["status"],
            payload_json=record["payload"],
            integrity_hash=record["integrity_hash"],
            created_at=record["created_at"]
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=['id'],
            set_={
                'case_id': stmt.excluded.case_id,
                'user_id': stmt.excluded.user_id,
                'destination': stmt.excluded.destination,
                'status': stmt.excluded.status,
                'payload_json': stmt.excluded.payload_json,
                'integrity_hash': stmt.excluded.integrity_hash,
                'created_at': stmt.excluded.created_at,
            }
        )
        db.execute(stmt)
        db.commit()
    return record

def upsert_currency_specimen(record: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    metadata = dict(record.get("metadata") or {})
    provenance_tier = _provenance_tier(record)
    metadata["provenance_tier"] = provenance_tier
    clean = {
        "id": str(record["id"]),
        "denomination": str(record["denomination"]),
        "label": str(record["label"]),
        "issuer": str(record["issuer"]),
        "image_sha256": str(record["image_sha256"]).lower(),
        "certification_reference": str(record["certification_reference"]),
        "captured_device": record.get("captured_device"),
        "captured_at": record.get("captured_at"),
        "received_at": now,
        "metadata": metadata,
        "provenance_tier": provenance_tier,
    }
    with SessionLocal() as db:
        stmt = insert(CurrencyCertifiedSpecimen).values(
            id=clean["id"],
            denomination=clean["denomination"],
            label=clean["label"],
            issuer=clean["issuer"],
            image_sha256=clean["image_sha256"],
            certification_reference=clean["certification_reference"],
            captured_device=clean["captured_device"],
            captured_at=clean["captured_at"],
            received_at=clean["received_at"],
            metadata_json=metadata
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=['id'],
            set_={
                'denomination': stmt.excluded.denomination,
                'label': stmt.excluded.label,
                'issuer': stmt.excluded.issuer,
                'image_sha256': stmt.excluded.image_sha256,
                'certification_reference': stmt.excluded.certification_reference,
                'captured_device': stmt.excluded.captured_device,
                'captured_at': stmt.excluded.captured_at,
                'received_at': stmt.excluded.received_at,
                'metadata_json': stmt.excluded.metadata_json,
            }
        )
        db.execute(stmt)
        db.commit()
    return clean

def certified_currency_count() -> int:
    return trusted_operational_counts()["currency_certified_specimens"]

def save_model_feedback(record: dict[str, Any]) -> dict[str, Any]:
    with SessionLocal() as db:
        stmt = insert(ModelFeedback).values(
            id=record["id"],
            case_id=record["case_id"],
            user_id=record["user_id"],
            predicted_verdict=record["predicted_verdict"],
            predicted_confidence=record["predicted_confidence"],
            observed_outcome=record["observed_outcome"],
            reporting_reference=record.get("reporting_reference"),
            notes=record.get("notes"),
            review_status=record["review_status"],
            created_at=record["created_at"]
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=['id'],
            set_={
                'case_id': stmt.excluded.case_id,
                'user_id': stmt.excluded.user_id,
                'predicted_verdict': stmt.excluded.predicted_verdict,
                'predicted_confidence': stmt.excluded.predicted_confidence,
                'observed_outcome': stmt.excluded.observed_outcome,
                'reporting_reference': stmt.excluded.reporting_reference,
                'notes': stmt.excluded.notes,
                'review_status': stmt.excluded.review_status,
                'created_at': stmt.excluded.created_at,
            }
        )
        db.execute(stmt)
        db.commit()
    return record

def list_model_feedback(user_id: str, limit: int = 100) -> list[dict[str, Any]]:
    with SessionLocal() as db:
        feedback = db.query(ModelFeedback).filter(ModelFeedback.user_id == user_id).order_by(ModelFeedback.created_at.desc()).limit(limit).all()
    return [
        {
            "id": row.id,
            "case_id": row.case_id,
            "user_id": row.user_id,
            "predicted_verdict": row.predicted_verdict,
            "predicted_confidence": row.predicted_confidence,
            "observed_outcome": row.observed_outcome,
            "reporting_reference": row.reporting_reference,
            "notes": row.notes,
            "review_status": row.review_status,
            "created_at": row.created_at,
        }
        for row in feedback
    ]
