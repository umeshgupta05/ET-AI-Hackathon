"""Optional external evidence object storage.

For production deployments this can mirror integrity-hashed case evidence to a
retention-controlled location. The first supported backend is file:// for
on-prem or mounted object-store gateways; cloud SDK backends can be added behind
the same interface without changing API routes.
"""

from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _s3_parts(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    return parsed.netloc, parsed.path.strip("/")


def _s3_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL") or None,
        region_name=os.getenv("AWS_REGION") or None,
    )


def evidence_store_status() -> dict[str, Any]:
    url = os.getenv("EVIDENCE_STORE_URL", "").strip()
    if not url:
        return {"enabled": False, "status": "disabled"}
    if url.startswith("file://"):
        path = Path(url.removeprefix("file://"))
        try:
            path.mkdir(parents=True, exist_ok=True)
            return {"enabled": True, "status": "ready", "backend": "file", "path": str(path)}
        except OSError as exc:
            return {"enabled": True, "status": "unavailable", "backend": "file", "detail": str(exc)}
    if url.startswith("s3://"):
        bucket, prefix = _s3_parts(url)
        try:
            _s3_client().head_bucket(Bucket=bucket)
            return {
                "enabled": True,
                "status": "ready",
                "backend": "s3",
                "bucket": bucket,
                "prefix": prefix,
                "server_side_encryption": os.getenv("EVIDENCE_S3_SSE", "AES256"),
                "retention_days": int(os.getenv("EVIDENCE_RETENTION_DAYS", "365")),
            }
        except Exception as exc:
            return {"enabled": True, "status": "unavailable", "backend": "s3", "detail": str(exc)[:500]}
    return {
        "enabled": True,
        "status": "unsupported",
        "detail": "Supported evidence store schemes are file:// and s3://.",
    }


def mirror_case_evidence(user_id: str, case_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = os.getenv("EVIDENCE_STORE_URL", "").strip()
    if not url:
        return {"enabled": False, "status": "disabled"}

    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if url.startswith("s3://"):
        bucket, prefix = _s3_parts(url)
        key = "/".join(part for part in (prefix, user_id, f"{case_id}.json") if part)
        args: dict[str, Any] = {
            "Bucket": bucket,
            "Key": key,
            "Body": encoded,
            "ContentType": "application/json",
            "ServerSideEncryption": os.getenv("EVIDENCE_S3_SSE", "AES256"),
            "Metadata": {
                "sha256": digest,
                "case-id": case_id,
                "stored-at": datetime.now(timezone.utc).isoformat(),
            },
        }
        kms_key = os.getenv("EVIDENCE_S3_KMS_KEY_ID", "").strip()
        if args["ServerSideEncryption"] == "aws:kms" and kms_key:
            args["SSEKMSKeyId"] = kms_key
        try:
            _s3_client().put_object(**args)
            return {
                "enabled": True,
                "status": "stored",
                "backend": "s3",
                "bucket": bucket,
                "key": key,
                "sha256": digest,
            }
        except Exception as exc:
            return {"enabled": True, "status": "failed", "backend": "s3", "detail": str(exc)[:500]}
    if not url.startswith("file://"):
        return {"enabled": True, "status": "unsupported"}

    root = Path(url.removeprefix("file://"))
    case_dir = root / user_id
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / f"{case_id}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return {
        "enabled": True,
        "status": "stored",
        "backend": "file",
        "path": str(path),
        "sha256": digest,
    }
