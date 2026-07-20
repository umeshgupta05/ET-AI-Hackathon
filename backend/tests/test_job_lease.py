"""Durable job claim and processing-lease heartbeat contract."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stores.job_store import claim_job, create_job, fail_job, init_job_db, renew_lease


def main() -> None:
    init_job_db()
    created = create_job(
        f"lease-test-{uuid.uuid4()}",
        {"text": "lease contract", "language": "en"},
    )
    job_id = created["job_id"]
    claimed = claim_job(job_id)
    assert claimed and claimed["status"] == "processing"
    assert renew_lease(job_id)
    fail_job(job_id, "integration cleanup")
    assert not renew_lease(job_id)
    print("Durable job claim and lease heartbeat: PASS")


if __name__ == "__main__":
    main()
