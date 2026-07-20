"""
Real-time Analytics Tracker.

Logs every analysis performed by the system and provides
live aggregated intelligence for the Command Centre.
No hardcoded numbers — everything derives from actual system usage.
"""

import threading
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Optional


class AnalyticsTracker:
    """Thread-safe in-memory tracker for real-time threat intelligence."""

    def __init__(self):
        self._lock = threading.Lock()
        self._analyses: list[dict] = []
        self._pattern_counts: dict[str, int] = defaultdict(int)
        self._city_detections: dict[str, int] = defaultdict(int)
        self._scam_type_timeline: list[dict] = []
        self._start_time = datetime.now(timezone.utc)

    def log_analysis(
        self,
        verdict: str,
        confidence: float,
        risk_level: str,
        scam_types: list[str] | None = None,
        modality: str = "text",
        agents_invoked: list[str] | None = None,
        processing_time: float = 0.0,
        source_city: str | None = None,
    ) -> None:
        """Log an analysis result for real-time tracking."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "epoch": time.time(),
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "risk_level": risk_level,
            "scam_types": scam_types or [],
            "modality": modality,
            "agents_invoked": agents_invoked or [],
            "processing_time": round(processing_time, 3),
        }

        with self._lock:
            self._analyses.append(entry)

            # Track pattern counts
            for stype in entry["scam_types"]:
                self._pattern_counts[stype] += 1

            # Track city if provided
            if source_city:
                self._city_detections[source_city] += 1

            # Track scam detections on timeline
            if verdict in ("high_risk", "medium_risk"):
                self._scam_type_timeline.append({
                    "timestamp": entry["timestamp"],
                    "types": entry["scam_types"],
                    "confidence": confidence,
                    "risk_level": risk_level,
                })

    def get_live_stats(self) -> dict[str, Any]:
        """Get real-time aggregated stats for the Command Centre."""
        now = datetime.now(timezone.utc)
        cutoff_24h = (now - timedelta(hours=24)).isoformat()
        cutoff_1h = (now - timedelta(hours=1)).isoformat()

        with self._lock:
            total = len(self._analyses)
            recent_24h = [a for a in self._analyses if a["timestamp"] >= cutoff_24h]
            recent_1h = [a for a in self._analyses if a["timestamp"] >= cutoff_1h]

            # A review verdict is deliberately distinct from an actionable threat.
            actionable_verdicts = {"high_risk", "medium_risk"}
            threats_24h = [a for a in recent_24h if a["verdict"] in actionable_verdicts]
            threats_1h = [a for a in recent_1h if a["verdict"] in actionable_verdicts]
            safe_24h = [a for a in recent_24h if a["verdict"] in ("safe", "low_risk")]
            review_24h = [a for a in recent_24h if a["verdict"] == "needs_review"]

            # Active scam patterns detected
            active_patterns: dict[str, dict] = {}
            for a in recent_24h:
                for stype in a.get("scam_types", []):
                    if stype not in active_patterns:
                        active_patterns[stype] = {
                            "pattern": stype.replace("_", " ").title(),
                            "count_24h": 0,
                            "count_1h": 0,
                            "max_confidence": 0.0,
                            "avg_confidence": 0.0,
                            "confidences": [],
                            "verdicts": [],
                        }
                    active_patterns[stype]["count_24h"] += 1
                    active_patterns[stype]["confidences"].append(a["confidence"])
                    active_patterns[stype]["verdicts"].append(a["verdict"])
                    active_patterns[stype]["max_confidence"] = max(
                        active_patterns[stype]["max_confidence"], a["confidence"]
                    )

            # Calculate averages and trends
            for key, pat in active_patterns.items():
                confs = pat.pop("confidences")
                pat["avg_confidence"] = round(sum(confs) / len(confs), 4) if confs else 0.0
                # Count in last hour for trend
                for a in recent_1h:
                    if key in a.get("scam_types", []):
                        pat["count_1h"] += 1
                # Determine trend
                if pat["count_1h"] > 0 and pat["count_24h"] > 2:
                    hourly_rate = pat["count_1h"]
                    daily_avg_hourly = pat["count_24h"] / 24
                    if daily_avg_hourly > 0 and hourly_rate > daily_avg_hourly * 1.5:
                        pat["trend"] = "surging"
                    elif hourly_rate > daily_avg_hourly:
                        pat["trend"] = "rising"
                    else:
                        pat["trend"] = "steady"
                else:
                    pat["trend"] = "new" if pat["count_24h"] <= 2 else "steady"

            # Modality breakdown
            modality_counts = defaultdict(int)
            for a in recent_24h:
                modality_counts[a["modality"]] += 1

            # Average processing time
            proc_times = [a["processing_time"] for a in recent_24h if a["processing_time"] > 0]
            avg_proc_time = round(sum(proc_times) / len(proc_times), 2) if proc_times else 0.0

            # Uptime
            uptime_seconds = (now - self._start_time).total_seconds()

            return {
                "generated_at": now.isoformat(),
                "is_live": True,
                "data_source": "real-time system analytics",
                "uptime_seconds": round(uptime_seconds),
                "summary": {
                    "total_analyses": total,
                    "analyses_24h": len(recent_24h),
                    "analyses_1h": len(recent_1h),
                    "threats_detected_24h": len(threats_24h),
                    "safe_cleared_24h": len(safe_24h),
                    "needs_review_24h": len(review_24h),
                    "active_patterns": len(active_patterns),
                    "avg_processing_time": avg_proc_time,
                },
                "active_campaigns": sorted(
                    active_patterns.values(),
                    key=lambda p: p["count_24h"],
                    reverse=True,
                ),
                "modality_breakdown": dict(modality_counts),
                "detection_rate": {
                    "threat_rate_24h": round(
                        len(threats_24h) / max(len(recent_24h), 1), 4
                    ),
                    "safe_clear_rate_24h": round(
                        len(safe_24h) / max(len(recent_24h), 1), 4
                    ),
                },
                "all_time": {
                    "total_analyses": total,
                    "pattern_counts": dict(self._pattern_counts),
                    "city_detections": dict(self._city_detections),
                },
            }


# Module singleton
_tracker: AnalyticsTracker | None = None


def get_analytics_tracker() -> AnalyticsTracker:
    global _tracker
    if _tracker is None:
        _tracker = AnalyticsTracker()
    return _tracker
