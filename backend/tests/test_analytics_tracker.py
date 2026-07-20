"""Unit checks for live Command Centre analytics semantics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.analytics import AnalyticsTracker


def main() -> None:
    tracker = AnalyticsTracker()
    tracker.log_analysis("needs_review", 0.48, "review", ["digital_arrest"])
    tracker.log_analysis("high_risk", 0.91, "critical", ["digital_arrest"])
    tracker.log_analysis("safe", 0.08, "safe", [])
    stats = tracker.get_live_stats()

    assert stats["summary"]["needs_review_24h"] == 1
    assert stats["summary"]["threats_detected_24h"] == 1
    assert stats["summary"]["safe_cleared_24h"] == 1
    assert "false_positive_estimate" not in stats["detection_rate"]
    assert stats["active_campaigns"][0]["pattern"] == "Digital Arrest"
    print("Command Centre analytics semantics: PASS")


if __name__ == "__main__":
    main()
