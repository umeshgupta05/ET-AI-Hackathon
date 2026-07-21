"""Durable state for asynchronous fraud-analysis jobs."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from sqlalchemy import or_, and_

from .database import engine, SessionLocal
from .models import Base, AnalysisJob

TERMINAL_STATUSES = {"completed", "failed"}

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def init_job_db() -> None:
    Base.metadata.create_all(bind=engine)

def create_job(user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    now = _now()
    with SessionLocal() as db:
        job = AnalysisJob(
            id=job_id,
            user_id=user_id,
            status="queued",
            input_json=payload,
            created_at=now,
            updated_at=now
        )
        db.add(job)
        db.commit()
    return get_job(user_id, job_id) or {}

def _public_job(job: AnalysisJob, include_input: bool = False) -> dict[str, Any]:
    j = {
        "job_id": job.id,
        "status": job.status,
        "attempts": job.attempts,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "error": job.error,
        "result": job.result_json,
    }
    if include_input:
        j["input"] = job.input_json
    return j

def get_job(user_id: str, job_id: str) -> dict[str, Any] | None:
    with SessionLocal() as db:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id, AnalysisJob.user_id == user_id).first()
    return _public_job(job) if job else None

def claim_job(job_id: str) -> dict[str, Any] | None:
    now = _now()
    lease_expires_at = (datetime.now(timezone.utc) + timedelta(seconds=120)).isoformat()
    with SessionLocal() as db:
        job = db.query(AnalysisJob).filter(
            AnalysisJob.id == job_id,
            or_(
                AnalysisJob.status.in_(["queued", "retrying"]),
                and_(AnalysisJob.status == "processing", AnalysisJob.lease_expires_at <= now)
            )
        ).with_for_update(skip_locked=True).first()
        
        if not job:
            return None
            
        job.status = "processing"
        job.attempts += 1
        job.updated_at = now
        job.lease_expires_at = lease_expires_at
        job.error = None
        db.commit()
        db.refresh(job)
        return _public_job(job, include_input=True)

def mark_retrying(job_id: str, error: str) -> None:
    with SessionLocal() as db:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if job:
            job.status = "retrying"
            job.lease_expires_at = None
            job.error = error[:1000]
            job.updated_at = _now()
            db.commit()

def renew_lease(job_id: str, lease_seconds: int = 120) -> bool:
    lease_expires_at = (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
    with SessionLocal() as db:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id, AnalysisJob.status == "processing").first()
        if job:
            job.lease_expires_at = lease_expires_at
            job.updated_at = _now()
            db.commit()
            return True
    return False

def complete_job(job_id: str, result: dict[str, Any]) -> None:
    with SessionLocal() as db:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if job:
            job.status = "completed"
            job.result_json = result
            job.input_json = None
            job.lease_expires_at = None
            job.error = None
            job.updated_at = _now()
            db.commit()

def fail_job(job_id: str, error: str) -> None:
    with SessionLocal() as db:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.input_json = None
            job.lease_expires_at = None
            job.error = error[:1000]
            job.updated_at = _now()
            db.commit()

def recover_stale_jobs(limit: int = 100) -> list[str]:
    now = _now()
    with SessionLocal() as db:
        jobs = db.query(AnalysisJob).filter(
            AnalysisJob.status == "processing",
            AnalysisJob.lease_expires_at <= now
        ).order_by(AnalysisJob.updated_at.asc()).limit(limit).all()
        
        job_ids = []
        for job in jobs:
            job.status = "retrying"
            job.lease_expires_at = None
            job.error = "Recovered after worker lease expired"
            job.updated_at = now
            job_ids.append(job.id)
        db.commit()
        return job_ids
