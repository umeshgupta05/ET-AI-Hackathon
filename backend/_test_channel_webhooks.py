"""Smoke tests for app/WhatsApp/IVR channel webhooks."""

from __future__ import annotations

from fastapi.testclient import TestClient

import main


class DummyOrchestrator:
    def get_stats(self) -> dict:
        return {"status": "dummy"}

    async def process(self, **kwargs) -> dict:
        return {
            "verdict": "high_risk",
            "risk_level": "high",
            "confidence": 0.82,
            "guided_reporting": main.reporting_guidance("high"),
            "agents_invoked": ["nlp"],
            "processing_time_seconds": 0.01,
            "agent_results": {
                "nlp": {
                    "recommended_action": "Do not transfer funds. Preserve evidence and call 1930 if money was sent.",
                    "retrieved_pattern_matches": [],
                }
            },
            "fusion_details": {},
        }


def main_test() -> None:
    main.orchestrator = DummyOrchestrator()
    client = TestClient(main.app)

    capabilities = client.get("/api/channels")
    assert capabilities.status_code == 200
    assert capabilities.json()["channels"]["whatsapp"]["status"] == "ready"
    assert capabilities.json()["channels"]["ivr"]["status"] == "ready"

    whatsapp = client.post(
        "/api/channels/whatsapp",
        data={
            "Body": "CBI officer says I must transfer money to avoid digital arrest.",
            "From": "whatsapp:+919999999999",
            "language": "en",
        },
    )
    assert whatsapp.status_code == 200
    assert whatsapp.headers["content-type"].startswith("application/xml")
    assert "HIGH risk" in whatsapp.text or "HIGH RISK" in whatsapp.text
    assert "1930" in whatsapp.text

    ivr_start = client.get("/api/channels/ivr/start")
    assert ivr_start.status_code == 200
    assert "<Gather" in ivr_start.text

    ivr_guidance = client.post("/api/channels/ivr/analyze", data={"Digits": "2"})
    assert ivr_guidance.status_code == 200
    assert "1930" in ivr_guidance.text

    ivr_analysis = client.post(
        "/api/channels/ivr/analyze",
        data={
            "SpeechResult": "A caller says I am under digital arrest and must transfer money.",
            "From": "+919999999999",
        },
    )
    assert ivr_analysis.status_code == 200
    assert "Shield verdict" in ivr_analysis.text

    print("Channel webhook contract checks passed")


if __name__ == "__main__":
    main_test()
