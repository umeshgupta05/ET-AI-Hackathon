import asyncio

import realtime_safety


def test_realtime_risk_creates_privacy_preserving_alerts(tmp_path, monkeypatch):
    monkeypatch.setattr(realtime_safety, "DB_PATH", tmp_path / "realtime.db")
    monkeypatch.setenv("PII_HASH_SECRET", "x" * 48)
    realtime_safety.init_realtime_db()
    session = realtime_safety.create_session(
        channel="web",
        language="en",
        caller_id="+919999999999",
    )
    event = realtime_safety.append_event(
        session["session_id"],
        transcript="Stay on the call and transfer money now or you will be arrested.",
        metadata={
            "caller_verification": "failed",
            "stir_shaken_attestation": "failed",
            "payment_requested": True,
            "secrecy_requested": True,
            "urgency_seconds": 300,
            "destination_account": "raw-account-must-not-persist",
        },
        model_score=0.91,
        model_verdict="high_risk",
    )
    assert event["risk_level"] == "critical"
    assert "raw-account-must-not-persist" not in str(event)
    alerts = realtime_safety.ensure_alerts(session["session_id"], event)
    assert {item["destination"] for item in alerts} == {"citizen", "telecom", "mha"}
    dispatched = asyncio.run(realtime_safety.dispatch_alerts(alerts))
    assert all(item["status"] == "pending_integration" for item in dispatched)

