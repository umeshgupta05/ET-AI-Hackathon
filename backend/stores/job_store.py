"""Durable state for asynchronous fraud-analysis jobs."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DB_PATH = Path(__file__).resolve().parent / "runtime" / "jobs.db"
TERMINAL_STATUSES = {"completed", "failed"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def init_job_db() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_jobs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                status TEXT NOT NULL,
                input_json TEXT,
                result_json TEXT,
                error TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                lease_expires_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_analysis_jobs_user_created "
            "ON analysis_jobs(user_id, created_at DESC)"
        )
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(analysis_jobs)").fetchall()
        }
        if "lease_expires_at" not in columns:
            connection.execute(
                "ALTER TABLE analysis_jobs ADD COLUMN lease_expires_at TEXT"
            )


def create_job(user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    now = _now()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO analysis_jobs
                (id, user_id, status, input_json, created_at, updated_at)
            VALUES (?, ?, 'queued', ?, ?, ?)
            """,
            (job_id, user_id, json.dumps(payload, ensure_ascii=False), now, now),
        )
    return get_job(user_id, job_id) or {}


def _public_job(row: sqlite3.Row, include_input: bool = False) -> dict[str, Any]:
    job = {
        "job_id": row["id"],
        "status": row["status"],
        "attempts": row["attempts"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "error": row["error"],
        "result": json.loads(row["result_json"]) if row["result_json"] else None,
    }
    if include_input:
        job["input"] = json.loads(row["input_json"]) if row["input_json"] else None
    return job


def get_job(user_id: str, job_id: str) -> dict[str, Any] | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM analysis_jobs WHERE id = ? AND user_id = ?",
            (job_id, user_id),
        ).fetchone()
    return _public_job(row) if row else None


def claim_job(job_id: str) -> dict[str, Any] | None:
    """Atomically claim a queued/retrying job and return its private input."""
    now = _now()
    lease_expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=120)
    ).isoformat()
    with _connect() as connection:
        cursor = connection.execute(
            """
            UPDATE analysis_jobs
            SET status = 'processing', attempts = attempts + 1, updated_at = ?,
                lease_expires_at = ?, error = NULL
            WHERE id = ? AND (
                status IN ('queued', 'retrying')
                OR (status = 'processing' AND lease_expires_at <= ?)
            )
            """,
            (now, lease_expires_at, job_id, now),
        )
        if cursor.rowcount != 1:
            return None
        row = connection.execute(
            "SELECT * FROM analysis_jobs WHERE id = ?", (job_id,)
        ).fetchone()
    return _public_job(row, include_input=True) if row else None


def mark_retrying(job_id: str, error: str) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE analysis_jobs SET status = 'retrying', lease_expires_at = NULL, error = ?, updated_at = ? WHERE id = ?",
            (error[:1000], _now(), job_id),
        )


def renew_lease(job_id: str, lease_seconds: int = 120) -> bool:
    """Extend an active processing lease while a long model call is still running."""
    lease_expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
    ).isoformat()
    with _connect() as connection:
        cursor = connection.execute(
            """
            UPDATE analysis_jobs
            SET lease_expires_at = ?, updated_at = ?
            WHERE id = ? AND status = 'processing'
            """,
            (lease_expires_at, _now(), job_id),
        )
    return cursor.rowcount == 1


def complete_job(job_id: str, result: dict[str, Any]) -> None:
    """Persist the result and erase the sensitive source payload."""
    with _connect() as connection:
        connection.execute(
            """
            UPDATE analysis_jobs
            SET status = 'completed', result_json = ?, input_json = NULL,
                lease_expires_at = NULL, error = NULL, updated_at = ?
            WHERE id = ?
            """,
            (json.dumps(result, ensure_ascii=False), _now(), job_id),
        )


def fail_job(job_id: str, error: str) -> None:
    with _connect() as connection:
        connection.execute(
            """
            UPDATE analysis_jobs
            SET status = 'failed', input_json = NULL, lease_expires_at = NULL,
                error = ?, updated_at = ?
            WHERE id = ?
            """,
            (error[:1000], _now(), job_id),
        )


def recover_stale_jobs(limit: int = 100) -> list[str]:
    """Return expired processing leases to the queueable state."""
    now = _now()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT id FROM analysis_jobs
            WHERE status = 'processing' AND lease_expires_at <= ?
            ORDER BY updated_at ASC LIMIT ?
            """,
            (now, limit),
        ).fetchall()
        job_ids = [row["id"] for row in rows]
        if job_ids:
            placeholders = ",".join("?" for _ in job_ids)
            connection.execute(
                f"""
                UPDATE analysis_jobs
                SET status = 'retrying', lease_expires_at = NULL,
                    error = 'Recovered after worker lease expired', updated_at = ?
                WHERE id IN ({placeholders})
                """,
                [now, *job_ids],
            )
    return job_ids
