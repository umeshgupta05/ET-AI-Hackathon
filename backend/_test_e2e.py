"""Full E2E audit of all API endpoints."""
import requests
import json
import time

BASE = "http://localhost:8000"

def test(name, method, url, **kwargs):
    try:
        r = getattr(requests, method)(url, timeout=90, **kwargs)
        status = r.status_code
        try:
            data = r.json()
        except:
            data = r.text[:200]
        if status < 400:
            print(f"  PASS [{status}] {name}")
            return data
        else:
            print(f"  FAIL [{status}] {name}: {str(data)[:100]}")
            return None
    except Exception as e:
        print(f"  FAIL [ERR] {name}: {e}")
        return None

print("=" * 60)
print("FULL E2E AUDIT")
print("=" * 60)

# 1. Health
print("\n--- Core Endpoints ---")
data = test("Health Check", "get", f"{BASE}/api/health")
if data:
    print(f"    Orchestrator: {data.get('agents',{}).get('orchestrator','?')}")
    topo = data.get('agents',{}).get('graph_topology',{})
    if topo:
        print(f"    Graph: {topo.get('type','?')}, nodes={topo.get('nodes','?')}")

# 2. Demo endpoints
print("\n--- Demo Endpoints ---")
test("Scam Transcript", "get", f"{BASE}/api/demo/scam-transcript")
test("Benign Transcript", "get", f"{BASE}/api/demo/benign-transcript")

# 3. Text analysis (scam)
print("\n--- Scam Text Analysis ---")
data = test("Scam Text", "post", f"{BASE}/api/analyze/text", json={
    "text": "This is Inspector Sharma from CBI. Your Aadhaar has been linked to money laundering. Transfer Rs 50,000 immediately or face digital arrest. Do not tell anyone."
})
if data:
    print(f"    Verdict: {data.get('verdict')}")
    print(f"    Confidence: {data.get('confidence')}")
    print(f"    Risk: {data.get('risk_level')}")
    print(f"    Agents: {data.get('agents_invoked')}")
    lg = data.get('langgraph', {})
    if lg:
        print(f"    LangGraph: type={lg.get('graph_type')}, iter={lg.get('iterations')}")
    print(f"    Time: {data.get('processing_time_seconds')}s")

# 4. Legitimate text
print("\n--- Legitimate Text Analysis ---")
data = test("Safe Text", "post", f"{BASE}/api/analyze/text", json={
    "text": "Thank you for calling SBI. Your account balance is Rs 1,50,000. Next EMI due July 15th. Visit your branch for queries."
})
if data:
    print(f"    Verdict: {data.get('verdict')}")
    print(f"    Confidence: {data.get('confidence')}")
    print(f"    Risk: {data.get('risk_level')}")

# 5. Turn-by-turn
print("\n--- Turn-by-Turn Analysis ---")
data = test("Turn Analysis", "post", f"{BASE}/api/analyze/turns", json={
    "turns": [
        "Hello, this is Inspector Sharma from CBI Cyber Cell.",
        "Your Aadhaar has been used for money laundering. An arrest warrant has been issued.",
        "Transfer Rs 2 lakhs to the RBI safe custody account immediately."
    ]
})
if data:
    traj = data.get('trajectory', [])
    print(f"    Turns analyzed: {len(traj)}")
    for t in traj:
        print(f"    Turn {t.get('turn',0)}: confidence={t.get('fused_confidence','?')}")

# 6. Graph
print("\n--- Graph Endpoints ---")
data = test("Graph Analysis", "get", f"{BASE}/api/graph/analyze")
if data:
    print(f"    Risk Score: {data.get('network_risk_score')}")
    print(f"    High-risk nodes: {len(data.get('high_risk_nodes',[]))}")

test("Graph Visualization", "get", f"{BASE}/api/graph/visualization")

# 7. Realtime call flow and pre-transfer alerting
print("\n--- Realtime Intervention ---")
session = test("Start Realtime Session", "post", f"{BASE}/api/realtime/sessions", json={
    "channel": "web", "language": "en", "metadata": {"test": "e2e"}
})
if session:
    event = test("High-risk Call Event", "post", f"{BASE}/api/realtime/sessions/{session['session_id']}/events", json={
        "transcript": "This is CBI. Do not tell anyone. Transfer money within five minutes or you will be arrested.",
        "caller_verification": "failed",
        "stir_shaken_attestation": "failed",
        "spoof_risk": 0.9,
        "claimed_authority": "CBI",
        "payment_requested": True,
        "secrecy_requested": True,
        "urgency_seconds": 300,
    })
    if event:
        assert event["pre_transfer_intervention"] is True
        assert {alert["destination"] for alert in event["alerts"]} == {"citizen", "telecom", "mha"}
        print(f"    Combined risk: {event['event']['combined_score']}")
        print(f"    Alert states: {[alert['status'] for alert in event['alerts']]}")
    test("Close Realtime Session", "post", f"{BASE}/api/realtime/sessions/{session['session_id']}/close")

# 8. Operational interfaces
print("\n--- Operational Interfaces ---")
test("Channel Capabilities", "get", f"{BASE}/api/channels")
test("Hotspot Intelligence", "get", f"{BASE}/api/intelligence/hotspots")
test("Command Centre", "get", f"{BASE}/api/intelligence/command-centre")
test("Feed Connector Status", "get", f"{BASE}/api/feeds/status")
test("Production Readiness", "get", f"{BASE}/api/readiness")

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
