"""Fast operational API test that does not load heavyweight model weights."""

import uuid

from fastapi.testclient import TestClient

import main


class StubOrchestrator:
    async def process(self, **kwargs):
        return {
            "verdict": "high_risk",
            "confidence": 0.91,
            "risk_level": "critical",
            "agents_invoked": ["nlp", "graph"],
            "fusion_details": {"fusion_method": "test_stub"},
            "trace": [{"step": "test"}],
            "processing_time_seconds": 0.01,
        }

    def get_stats(self):
        return {"status": "test_ready"}


def main_test() -> None:
    main.init_db()
    main.orchestrator = StubOrchestrator()
    client = TestClient(main.app)

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["capabilities"]["languages"] == 12
    assert health.headers["x-request-id"]

    hotspots = client.get("/api/intelligence/hotspots", params={"latitude": 28.61, "longitude": 77.20})
    assert hotspots.status_code == 200
    assert len(hotspots.json()["hotspots"]) >= 10
    assert hotspots.json()["hotspots"][0]["distance_km"] >= 0

    email = f"audit-{uuid.uuid4().hex[:10]}@example.test"
    auth = client.post(
        "/api/auth/register",
        json={"name": "Audit User", "email": email, "password": "StrongPass123!", "preferred_language": "hi"},
    )
    assert auth.status_code == 200, auth.text
    headers = {"Authorization": f"Bearer {auth.json()['access_token']}"}

    analysis = client.post("/api/analyze/text", json={"text": "Digital arrest. Transfer money now."}, headers=headers)
    assert analysis.status_code == 200, analysis.text
    case_id = analysis.json()["case_id"]
    assert len(analysis.json()["evidence_integrity"]["hash"]) == 64

    evidence = client.get(f"/api/cases/{case_id}/evidence", headers=headers)
    assert evidence.status_code == 200, evidence.text
    assert len(evidence.json()["package_integrity"]["hash"]) == 64

    guidance = client.get("/api/reporting/guidance", params={"risk_level": "critical"})
    assert guidance.status_code == 200
    assert guidance.json()["priority"] == "immediate"

    logout = client.post("/api/auth/logout", headers=headers)
    assert logout.status_code == 200
    assert client.get("/api/auth/me", headers=headers).status_code == 401
    print("Operational API tests passed")


if __name__ == "__main__":
    main_test()
