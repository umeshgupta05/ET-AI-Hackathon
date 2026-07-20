"""Persistent operational intelligence store.

This module is the bridge between external authorized feeds and the AI agents.
It stores only normalized intelligence records and provenance metadata; raw
citizen payloads remain in the existing case/evidence flow.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DB_PATH = Path(__file__).resolve().parent / "data" / "operational_intelligence.db"
PROVENANCE_TIERS = {"authorized", "public_research", "synthetic_sandbox"}


def _provenance_tier(record: dict[str, Any]) -> str:
    """Normalize provenance so sandbox rows can never satisfy production gates."""
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


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    return connection


def init_operational_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS geospatial_incidents (
                id TEXT PRIMARY KEY,
                district TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                type TEXT NOT NULL,
                reports INTEGER NOT NULL,
                severity REAL NOT NULL,
                source TEXT NOT NULL,
                source_reference TEXT,
                occurred_at TEXT,
                received_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_entities (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                label TEXT NOT NULL,
                source TEXT NOT NULL,
                source_reference TEXT,
                updated_at TEXT NOT NULL,
                attrs_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS graph_edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL NOT NULL,
                source TEXT NOT NULL,
                source_reference TEXT,
                observed_at TEXT,
                updated_at TEXT NOT NULL,
                attrs_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reporting_drafts (
                id TEXT PRIMARY KEY,
                case_id TEXT,
                user_id TEXT,
                destination TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                integrity_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS currency_certified_specimens (
                id TEXT PRIMARY KEY,
                denomination TEXT NOT NULL,
                label TEXT NOT NULL,
                issuer TEXT NOT NULL,
                image_sha256 TEXT NOT NULL,
                certification_reference TEXT NOT NULL,
                captured_device TEXT,
                captured_at TEXT,
                received_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_feedback (
                id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                predicted_verdict TEXT NOT NULL,
                predicted_confidence REAL NOT NULL,
                observed_outcome TEXT NOT NULL,
                reporting_reference TEXT,
                notes TEXT,
                review_status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


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
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO geospatial_incidents
            (id, district, lat, lon, type, reports, severity, source, source_reference,
             occurred_at, received_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clean["id"],
                clean["district"],
                clean["lat"],
                clean["lon"],
                clean["type"],
                clean["reports"],
                clean["severity"],
                clean["source"],
                clean["source_reference"],
                clean["occurred_at"],
                clean["received_at"],
                json.dumps(metadata, ensure_ascii=False),
            ),
        )
    return clean


def list_geospatial_incidents(limit: int = 1000) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM geospatial_incidents
            ORDER BY COALESCE(occurred_at, received_at) DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "id": row["id"],
            "district": row["district"],
            "lat": row["lat"],
            "lon": row["lon"],
            "type": row["type"],
            "reports": row["reports"],
            "severity": row["severity"],
            "source": row["source"],
            "source_reference": row["source_reference"],
            "occurred_at": row["occurred_at"],
            "received_at": row["received_at"],
            "metadata": json.loads(row["metadata_json"] or "{}"),
            "provenance_tier": json.loads(row["metadata_json"] or "{}").get(
                "provenance_tier", "synthetic_sandbox"
            ),
        }
        for row in rows
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
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO graph_entities
            (id, entity_type, label, source, source_reference, updated_at, attrs_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clean["id"],
                clean["type"],
                clean["label"],
                clean["source"],
                clean["source_reference"],
                clean["updated_at"],
                json.dumps(attrs, ensure_ascii=False),
            ),
        )
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
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO graph_edges
            (id, source_id, target_id, edge_type, weight, source, source_reference,
             observed_at, updated_at, attrs_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clean["id"],
                clean["source_id"],
                clean["target_id"],
                clean["type"],
                clean["weight"],
                clean["source"],
                clean["source_reference"],
                clean["observed_at"],
                clean["updated_at"],
                json.dumps(attrs, ensure_ascii=False),
            ),
        )
    return clean


def list_graph_entities(limit: int = 10000) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM graph_entities LIMIT ?", (limit,)).fetchall()
    return [
        {
            "id": row["id"],
            "type": row["entity_type"],
            "label": row["label"],
            "source": row["source"],
            "source_reference": row["source_reference"],
            "updated_at": row["updated_at"],
            "attrs": json.loads(row["attrs_json"] or "{}"),
            "provenance_tier": json.loads(row["attrs_json"] or "{}").get(
                "provenance_tier", "synthetic_sandbox"
            ),
        }
        for row in rows
    ]


def list_graph_edges(limit: int = 20000) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM graph_edges LIMIT ?", (limit,)).fetchall()
    return [
        {
            "id": row["id"],
            "source_id": row["source_id"],
            "target_id": row["target_id"],
            "type": row["edge_type"],
            "weight": row["weight"],
            "source": row["source"],
            "source_reference": row["source_reference"],
            "observed_at": row["observed_at"],
            "updated_at": row["updated_at"],
            "attrs": json.loads(row["attrs_json"] or "{}"),
            "provenance_tier": json.loads(row["attrs_json"] or "{}").get(
                "provenance_tier", "synthetic_sandbox"
            ),
        }
        for row in rows
    ]


def operational_counts() -> dict[str, int]:
    with _connect() as conn:
        return {
            "geospatial_incidents": conn.execute("SELECT COUNT(*) FROM geospatial_incidents").fetchone()[0],
            "graph_entities": conn.execute("SELECT COUNT(*) FROM graph_entities").fetchone()[0],
            "graph_edges": conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0],
            "reporting_drafts": conn.execute("SELECT COUNT(*) FROM reporting_drafts").fetchone()[0],
            "currency_certified_specimens": conn.execute("SELECT COUNT(*) FROM currency_certified_specimens").fetchone()[0],
        }


def operational_provenance_counts() -> dict[str, dict[str, int]]:
    """Return totals grouped by trust tier without relying on SQLite JSON extensions."""
    result = {
        "geospatial_incidents": {tier: 0 for tier in PROVENANCE_TIERS},
        "graph_entities": {tier: 0 for tier in PROVENANCE_TIERS},
        "graph_edges": {tier: 0 for tier in PROVENANCE_TIERS},
        "currency_certified_specimens": {tier: 0 for tier in PROVENANCE_TIERS},
    }
    with _connect() as conn:
        table_specs = (
            ("geospatial_incidents", "metadata_json", True),
            ("graph_entities", "attrs_json", True),
            ("graph_edges", "attrs_json", True),
            ("currency_certified_specimens", "metadata_json", False),
        )
        for table, column, has_source in table_specs:
            source_column = "source, " if has_source else ""
            for row in conn.execute(f"SELECT {source_column}{column} AS metadata_json FROM {table}"):
                try:
                    metadata = json.loads(row["metadata_json"] or "{}")
                except json.JSONDecodeError:
                    metadata = {}
                tier = _provenance_tier({
                    "source": row["source"] if "source" in row.keys() else "",
                    "metadata": metadata,
                })
                result[table][tier] += 1
    return result


def trusted_operational_counts() -> dict[str, int]:
    """Counts that are permitted to unlock strict production intelligence."""
    grouped = operational_provenance_counts()
    return {name: tiers["authorized"] for name, tiers in grouped.items()}


def save_reporting_draft(record: dict[str, Any]) -> dict[str, Any]:
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO reporting_drafts
            (id, case_id, user_id, destination, status, payload_json, integrity_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record.get("case_id"),
                record.get("user_id"),
                record["destination"],
                record["status"],
                json.dumps(record["payload"], ensure_ascii=False),
                record["integrity_hash"],
                record["created_at"],
            ),
        )
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
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO currency_certified_specimens
            (id, denomination, label, issuer, image_sha256, certification_reference,
             captured_device, captured_at, received_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clean["id"],
                clean["denomination"],
                clean["label"],
                clean["issuer"],
                clean["image_sha256"],
                clean["certification_reference"],
                clean["captured_device"],
                clean["captured_at"],
                clean["received_at"],
                json.dumps(metadata, ensure_ascii=False),
            ),
        )
    return clean


def certified_currency_count() -> int:
    return trusted_operational_counts()["currency_certified_specimens"]


def save_model_feedback(record: dict[str, Any]) -> dict[str, Any]:
    with _connect() as conn:
        conn.execute(
            """INSERT INTO model_feedback
               (id, case_id, user_id, predicted_verdict, predicted_confidence,
                observed_outcome, reporting_reference, notes, review_status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record["id"], record["case_id"], record["user_id"],
                record["predicted_verdict"], record["predicted_confidence"],
                record["observed_outcome"], record.get("reporting_reference"),
                record.get("notes"), record["review_status"], record["created_at"],
            ),
        )
    return record


def list_model_feedback(user_id: str, limit: int = 100) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM model_feedback WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]
