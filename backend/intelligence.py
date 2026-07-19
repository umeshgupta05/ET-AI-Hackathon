"""Operational geospatial intelligence and evidence-package helpers."""

import hashlib
import json
import math
import secrets
from datetime import datetime, timezone
from typing import Any, Optional

from operational_store import list_geospatial_incidents, save_reporting_draft


INCIDENTS = [
    {"district": "Delhi", "lat": 28.6139, "lon": 77.2090, "type": "digital_arrest", "reports": 38, "severity": 0.91},
    {"district": "Mumbai", "lat": 19.0760, "lon": 72.8777, "type": "investment_fraud", "reports": 34, "severity": 0.84},
    {"district": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "type": "kyc_otp", "reports": 29, "severity": 0.78},
    {"district": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "type": "parcel_customs", "reports": 27, "severity": 0.76},
    {"district": "Kolkata", "lat": 22.5726, "lon": 88.3639, "type": "impersonation", "reports": 22, "severity": 0.69},
    {"district": "Chennai", "lat": 13.0827, "lon": 80.2707, "type": "job_fraud", "reports": 20, "severity": 0.64},
    {"district": "Pune", "lat": 18.5204, "lon": 73.8567, "type": "investment_fraud", "reports": 18, "severity": 0.61},
    {"district": "Ahmedabad", "lat": 23.0225, "lon": 72.5714, "type": "counterfeit", "reports": 16, "severity": 0.58},
    {"district": "Lucknow", "lat": 26.8467, "lon": 80.9462, "type": "digital_arrest", "reports": 15, "severity": 0.57},
    {"district": "Jaipur", "lat": 26.9124, "lon": 75.7873, "type": "kyc_otp", "reports": 13, "severity": 0.52},
    {"district": "Bhopal", "lat": 23.2599, "lon": 77.4126, "type": "counterfeit", "reports": 11, "severity": 0.48},
    {"district": "Guwahati", "lat": 26.1445, "lon": 91.7362, "type": "parcel_customs", "reports": 10, "severity": 0.44},
]


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _active_incidents() -> tuple[list[dict[str, Any]], str, str]:
    operational = list_geospatial_incidents()
    if operational:
        tiers = {item.get("provenance_tier", "synthetic_sandbox") for item in operational}
        if tiers == {"authorized"}:
            return (
                operational,
                "authorized operational intelligence feeds",
                "Live feed records normalized from authorized ingest APIs.",
            )
        source = "hackathon sandbox feed" if "synthetic_sandbox" in tiers else "public research feed"
        return (
            operational,
            source,
            "Non-authorized data for demonstration and research only; not operational law-enforcement intelligence.",
        )
    return (
        INCIDENTS,
        "anonymized demonstration intelligence feed",
        "Demo data only; connect authorized NCRB, bank, telecom, and state feeds for operational deployment.",
    )


def geospatial_overview(latitude: Optional[float] = None, longitude: Optional[float] = None) -> dict[str, Any]:
    incidents, source, limitations = _active_incidents()
    hotspots = []
    for incident in incidents:
        item = dict(incident)
        item["risk_score"] = round(incident["severity"] * 0.65 + min(incident["reports"] / 40, 1) * 0.35, 3)
        if latitude is not None and longitude is not None:
            item["distance_km"] = round(_distance_km(latitude, longitude, item["lat"], item["lon"]), 1)
        hotspots.append(item)
    hotspots.sort(key=lambda item: item.get("distance_km", -item["risk_score"]))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "hotspots": hotspots,
        "summary": {
            "reports": sum(item["reports"] for item in incidents),
            "districts": len({item["district"] for item in incidents}),
            "highest_risk_district": max(incidents, key=lambda item: item["severity"])["district"] if incidents else None,
        },
        "limitations": limitations,
    }


def command_center_plan(available_units: int = 10) -> dict[str, Any]:
    """Allocate patrol/analyst capacity from current hotspot risk and report volume."""
    overview = geospatial_overview()
    hotspots = overview["hotspots"]
    total_weight = sum(max(0.01, item["risk_score"] * item["reports"]) for item in hotspots)
    remaining = max(1, available_units)
    allocations = []
    for index, hotspot in enumerate(hotspots):
        weight = max(0.01, hotspot["risk_score"] * hotspot["reports"])
        units = max(1, round(available_units * weight / total_weight)) if available_units >= len(hotspots) else 0
        if index < available_units and units == 0:
            units = 1
        units = min(units, remaining)
        remaining -= units
        allocations.append({
            "district": hotspot["district"],
            "incident_type": hotspot["type"],
            "priority": "critical" if hotspot["risk_score"] >= 0.8 else "high" if hotspot["risk_score"] >= 0.6 else "monitor",
            "recommended_units": units,
            "risk_score": hotspot["risk_score"],
            "reports": hotspot["reports"],
            "recommended_action": (
                "Coordinate cyber cell, bank liaison, and telecom nodal officer"
                if hotspot["type"] != "counterfeit"
                else "Coordinate field seizure team, bank branch alerts, and forensic screening"
            ),
        })
    return {
        "generated_at": overview["generated_at"],
        "source": overview["source"],
        "available_units": available_units,
        "allocations": allocations,
        "inter_district_sharing": [
            {
                "lead_district": item["district"],
                "share_with": [other["district"] for other in hotspots if other["type"] == item["type"] and other["district"] != item["district"]][:3],
                "threat_type": item["type"],
            }
            for item in hotspots[:5]
        ],
        "human_approval_required": True,
        "limitations": overview["limitations"],
    }


def build_evidence_package(case: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    result = case["result"]
    package = {
        "package_version": "1.0",
        "case_id": case["id"],
        "case_type": case["case_type"],
        "captured_at": case["created_at"],
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "custodian": {"user_id": user["id"], "name": user["name"]},
        "decision": {"verdict": result.get("verdict"), "risk_level": result.get("risk_level"), "confidence": result.get("confidence")},
        "agents_invoked": result.get("agents_invoked", []),
        "fusion_details": result.get("fusion_details", {}),
        "trace": result.get("trace", []),
        "source_integrity": result.get("evidence_integrity", {}),
        "disclosure": "AI-assisted lead intelligence. Human verification is required before enforcement or legal submission.",
    }
    canonical = json.dumps(package, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    package["package_integrity"] = {"algorithm": "SHA-256", "hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}
    return package


def reporting_guidance(risk_level: str) -> dict[str, Any]:
    urgent = risk_level.lower() in {"critical", "high"}
    return {
        "immediate_actions": [
            "Stop communication and do not transfer funds or disclose OTP, PIN, or credentials.",
            "Call 1930 immediately if money was transferred.",
            "Preserve screenshots, phone numbers, transaction IDs, audio, and chat exports.",
        ],
        "official_channels": [
            {"name": "National Cyber Crime Reporting Portal", "url": "https://cybercrime.gov.in"},
            {"name": "Cyber fraud helpline", "phone": "1930"},
            {"name": "Emergency response", "phone": "112", "when": "Immediate physical danger"},
        ],
        "priority": "immediate" if urgent else "standard",
        "note": "This application does not submit a police complaint automatically.",
    }


def build_reporting_draft(case: dict[str, Any], user: dict[str, Any], destination: str = "NCRP") -> dict[str, Any]:
    """Create an evidence-preserving reporting draft for human submission."""
    evidence = build_evidence_package(case, user)
    payload = {
        "destination": destination,
        "case_id": case["id"],
        "created_for": {"user_id": user["id"], "name": user["name"]},
        "official_channels": reporting_guidance(case["result"].get("risk_level", "medium"))["official_channels"],
        "summary": {
            "case_type": case["case_type"],
            "verdict": case["result"].get("verdict") or case["result"].get("final_verdict"),
            "risk_level": case["result"].get("risk_level"),
            "confidence": case["result"].get("confidence"),
            "agents_invoked": case["result"].get("agents_invoked", []),
        },
        "evidence_package": evidence,
        "submission_status": "draft_requires_human_review",
        "disclosure": "Prepared for manual official submission; no automatic complaint filing occurred.",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    created_at = datetime.now(timezone.utc).isoformat()
    draft = {
        "id": secrets.token_urlsafe(12),
        "case_id": case["id"],
        "user_id": user["id"],
        "destination": destination,
        "status": "draft_requires_human_review",
        "payload": payload,
        "integrity_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "created_at": created_at,
    }
    save_reporting_draft(draft)
    return draft
