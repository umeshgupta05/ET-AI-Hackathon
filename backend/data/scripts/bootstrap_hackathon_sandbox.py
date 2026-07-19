"""Prepare a no-login, provenance-safe hackathon environment.

This command downloads a public real-SMS corpus, creates schema-equivalent
privacy-safe fraud events, and seeds the same operational store consumed by the
API and graph/geospatial agents. Sandbox records are explicitly marked and can
never satisfy strict production-readiness gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import secrets
import sys
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_DIR / "data"
PUBLIC_DIR = DATA_DIR / "public_sources" / "uci_sms_spam_collection"
UCI_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
UCI_DOI = "10.24432/C5CC84"
GENERATOR_VERSION = "1.0.0"
SEED = 2026

sys.path.insert(0, str(BACKEND_DIR))

from operational_store import (  # noqa: E402
    init_operational_db,
    operational_counts,
    operational_provenance_counts,
    upsert_geospatial_incident,
    upsert_graph_edge,
    upsert_graph_entity,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_public_sms(force: bool = False) -> dict:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    archive = PUBLIC_DIR / "sms_spam_collection.zip"
    raw_path = PUBLIC_DIR / "SMSSpamCollection"
    normalized_path = PUBLIC_DIR / "sms_spam_collection.jsonl"

    if force or not archive.exists():
        request = urllib.request.Request(UCI_URL, headers={"User-Agent": "DigitalSafetyShield/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)

    with zipfile.ZipFile(archive) as bundle:
        member = next((name for name in bundle.namelist() if name.endswith("SMSSpamCollection")), None)
        if not member:
            raise RuntimeError("UCI archive does not contain SMSSpamCollection")
        raw_path.write_bytes(bundle.read(member))

    rows = []
    for index, line in enumerate(raw_path.read_text(encoding="utf-8").splitlines()):
        label, text = line.split("\t", 1)
        rows.append({
            "id": f"uci-sms-{index:05d}",
            "text": text,
            "source_label": label,
            "task_label": "spam" if label == "spam" else "legitimate",
            "provenance_tier": "public_research",
            "source_dataset": "UCI SMS Spam Collection",
        })
    normalized_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest = {
        "name": "UCI SMS Spam Collection",
        "download_url": UCI_URL,
        "dataset_page": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        "doi": UCI_DOI,
        "license": "CC BY 4.0",
        "records": len(rows),
        "real_or_synthetic": "real donated SMS corpus",
        "intended_use": "auxiliary spam/legitimate research; not relabelled as digital-arrest ground truth",
        "archive_sha256": _sha256(archive),
        "normalized_sha256": _sha256(normalized_path),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    (PUBLIC_DIR / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _metadata(record_kind: str) -> dict:
    return {
        "provenance_tier": "synthetic_sandbox",
        "sandbox": True,
        "record_kind": record_kind,
        "generator": "bootstrap_hackathon_sandbox.py",
        "generator_version": GENERATOR_VERSION,
        "contains_real_pii": False,
        "allowed_use": "hackathon demonstration and integration testing",
    }


def _seed_geospatial() -> int:
    locations = [
        ("Delhi", 28.6139, 77.2090, "digital_arrest", 38, 0.91),
        ("Mumbai", 19.0760, 72.8777, "investment_fraud", 34, 0.84),
        ("Bengaluru", 12.9716, 77.5946, "kyc_otp", 29, 0.78),
        ("Hyderabad", 17.3850, 78.4867, "parcel_customs", 27, 0.76),
        ("Kolkata", 22.5726, 88.3639, "impersonation", 22, 0.69),
        ("Chennai", 13.0827, 80.2707, "job_fraud", 20, 0.64),
        ("Pune", 18.5204, 73.8567, "investment_fraud", 18, 0.61),
        ("Ahmedabad", 23.0225, 72.5714, "counterfeit", 16, 0.58),
        ("Lucknow", 26.8467, 80.9462, "digital_arrest", 15, 0.57),
        ("Jaipur", 26.9124, 75.7873, "kyc_otp", 13, 0.52),
        ("Bhopal", 23.2599, 77.4126, "counterfeit", 11, 0.48),
        ("Guwahati", 26.1445, 91.7362, "parcel_customs", 10, 0.44),
    ]
    occurred_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    for index, (district, lat, lon, incident_type, reports, severity) in enumerate(locations):
        upsert_geospatial_incident({
            "id": f"sandbox-geo-{index:03d}",
            "district": district,
            "lat": lat,
            "lon": lon,
            "type": incident_type,
            "reports": reports,
            "severity": severity,
            "source": "synthetic_hackathon",
            "source_reference": f"sandbox://geospatial/{index}",
            "occurred_at": occurred_at,
            "provenance_tier": "synthetic_sandbox",
            "metadata": _metadata("aggregate_geospatial_scenario"),
        })
    return len(locations)


def _entity(entity_id: str, entity_type: str, label: str, **attrs) -> None:
    upsert_graph_entity({
        "id": entity_id,
        "type": entity_type,
        "label": label,
        "source": "synthetic_hackathon",
        "source_reference": f"sandbox://entity/{entity_id}",
        "provenance_tier": "synthetic_sandbox",
        "attrs": {**_metadata("fraud_graph_entity"), **attrs},
    })


def _edge(source_id: str, target_id: str, relationship: str, index: int, weight: float = 1.0, **attrs) -> None:
    upsert_graph_edge({
        "id": f"sandbox-edge-{index:04d}",
        "source_id": source_id,
        "target_id": target_id,
        "type": relationship,
        "weight": weight,
        "source": "synthetic_hackathon",
        "source_reference": f"sandbox://event/{index}",
        "observed_at": (datetime.now(timezone.utc) - timedelta(minutes=index)).isoformat(),
        "provenance_tier": "synthetic_sandbox",
        "attrs": {**_metadata("fraud_graph_relationship"), **attrs},
    })


def _seed_graph() -> tuple[int, int]:
    rng = random.Random(SEED)
    scammers = [f"sandbox-phone-s{i:02d}" for i in range(10)]
    victims = [f"sandbox-phone-v{i:02d}" for i in range(20)]
    accounts = [f"sandbox-account-m{i:02d}" for i in range(10)]
    devices = [f"sandbox-device-{i:02d}" for i in range(10)]
    reports = [f"sandbox-report-{i:02d}" for i in range(10)]

    for index, item in enumerate(scammers):
        _entity(item, "phone", "scammer", call_count=40 + index * 5, reported_count=3 + index, voip=1)
    for index, item in enumerate(victims):
        _entity(item, "phone", "victim", call_count=1 + index % 3, reported_count=index % 2, voip=0)
    for index, item in enumerate(accounts):
        _entity(item, "account", "mule", inflow=250000 + index * 45000, outflow=230000 + index * 43000, tx_count=12 + index)
    for index, item in enumerate(devices):
        _entity(item, "device", "suspicious", linked_accounts=2 + index % 4, risk_score=0.65 + index * 0.02)
    for index, item in enumerate(reports):
        _entity(item, "report", "confirmed_scam", report_count=1, severity=0.7 + index * 0.02)

    edge_index = 0
    for index, scammer in enumerate(scammers):
        account = accounts[index]
        device = devices[index]
        _edge(scammer, account, "controls", edge_index, 0.95)
        edge_index += 1
        _edge(scammer, device, "uses", edge_index, 0.9)
        edge_index += 1
        _edge(device, account, "accessed", edge_index, 0.88)
        edge_index += 1
        _edge(scammer, reports[index], "reported_in", edge_index, 1.0)
        edge_index += 1
        for victim in rng.sample(victims, 4):
            _edge(scammer, victim, "called", edge_index, rng.uniform(0.55, 0.95), duration_seconds=rng.randint(40, 420))
            edge_index += 1
    for index, account in enumerate(accounts):
        _edge(account, accounts[(index + 1) % len(accounts)], "transferred", edge_index, 0.8, amount=50000 + index * 7500)
        edge_index += 1

    return len(scammers + victims + accounts + devices + reports), edge_index


def _ensure_local_env() -> list[str]:
    env_path = BACKEND_DIR / ".env"
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    keys = {
        line.split("=", 1)[0].strip()
        for line in existing.splitlines()
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    }
    additions = []
    values = {
        "JWT_SECRET": secrets.token_urlsafe(48),
        "SHIELD_INGEST_TOKEN": secrets.token_urlsafe(36),
        "MULTICHANNEL_WEBHOOK_TOKEN": secrets.token_urlsafe(36),
        "DEPLOYMENT_MODE": "demo",
        "ALLOW_DEMO_INTELLIGENCE": "true",
        "REDIS_ENABLED": "true",
        "REDIS_URL": "redis://:change-me@localhost:16379/0",
        "ASYNC_JOBS_ENABLED": "true",
        "RABBITMQ_URL": "amqp://shield:change-me@localhost:5672/",
    }
    for key, value in values.items():
        if key not in keys:
            additions.append(f"{key}={value}")
    if additions:
        separator = "" if not existing or existing.endswith("\n") else "\n"
        env_path.write_text(existing + separator + "\n# Generated local hackathon sandbox settings\n" + "\n".join(additions) + "\n", encoding="utf-8")
    return [item.split("=", 1)[0] for item in additions]


def bootstrap(*, download_public: bool, configure_env: bool, force_download: bool) -> dict:
    init_operational_db()
    existing_sms_manifest = PUBLIC_DIR / "source_manifest.json"
    sms_manifest = (
        _download_public_sms(force=force_download)
        if download_public
        else json.loads(existing_sms_manifest.read_text(encoding="utf-8"))
        if existing_sms_manifest.exists()
        else None
    )
    geospatial_count = _seed_geospatial()
    graph_nodes, graph_edges = _seed_graph()
    env_keys_added = _ensure_local_env() if configure_env else []
    result = {
        "status": "ready",
        "sandbox_disclosure": "All seeded event-level graph and geospatial records are synthetic_hackathon data with no real PII.",
        "public_sms": sms_manifest,
        "seeded": {
            "geospatial_incidents": geospatial_count,
            "graph_entities": graph_nodes,
            "graph_edges": graph_edges,
        },
        "store_counts": operational_counts(),
        "provenance_counts": operational_provenance_counts(),
        "env_keys_added": env_keys_added,
        "production_claim": False,
        "remaining_external_steps": [
            "Meta/Twilio/Exotel channel credentials require an account owner and provider verification.",
            "NCRB, bank, telecom, and certified-currency feeds require institutional authorization.",
        ],
    }
    manifest_path = DATA_DIR / "sandbox_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-download", action="store_true", help="Do not download the public UCI SMS corpus")
    parser.add_argument("--configure-env", action="store_true", help="Add missing local-only secrets and sandbox settings to backend/.env")
    parser.add_argument("--force-download", action="store_true", help="Refresh public downloads")
    args = parser.parse_args()
    result = bootstrap(
        download_public=not args.skip_download,
        configure_env=args.configure_env,
        force_download=args.force_download,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
