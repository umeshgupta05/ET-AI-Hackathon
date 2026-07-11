"""Operational geospatial intelligence and evidence-package helpers."""

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Optional


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


def geospatial_overview(latitude: Optional[float] = None, longitude: Optional[float] = None) -> dict[str, Any]:
    hotspots = []
    for incident in INCIDENTS:
        item = dict(incident)
        item["risk_score"] = round(incident["severity"] * 0.65 + min(incident["reports"] / 40, 1) * 0.35, 3)
        if latitude is not None and longitude is not None:
            item["distance_km"] = round(_distance_km(latitude, longitude, item["lat"], item["lon"]), 1)
        hotspots.append(item)
    hotspots.sort(key=lambda item: item.get("distance_km", -item["risk_score"]))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "anonymized demonstration intelligence feed",
        "hotspots": hotspots,
        "summary": {
            "reports": sum(item["reports"] for item in INCIDENTS),
            "districts": len(INCIDENTS),
            "highest_risk_district": max(INCIDENTS, key=lambda item: item["severity"])["district"],
        },
        "limitations": "Demo data only; connect authorized NCRB, bank, telecom, and state feeds for operational deployment.",
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
