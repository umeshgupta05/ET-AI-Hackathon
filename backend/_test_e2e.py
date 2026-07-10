"""End-to-end test of LangGraph orchestrator via API."""

import requests

BASE = "http://localhost:8000"


def main() -> None:
    print("=" * 60)
    print("TEST 1: Health Check")
    response = requests.get(f"{BASE}/api/health", timeout=30)
    response.raise_for_status()
    data = response.json()
    print(f"  Status: {data['status']}")
    print(f"  Orchestrator: {data['agents']['orchestrator']}")
    if "graph_topology" in data["agents"]:
        topology = data["agents"]["graph_topology"]
        print(
            "  Graph: "
            f"{topology['type']} ({topology['nodes']} nodes, "
            f"{topology['edges']} edges, cyclic={topology['cyclic']})"
        )
    print()

    print("=" * 60)
    print("TEST 2: Scam Text Analysis (LangGraph)")
    response = requests.post(
        f"{BASE}/api/analyze/text",
        json={
            "text": (
                "This is Inspector Sharma from CBI. Your Aadhaar has been linked "
                "to money laundering case 4527. Transfer Rs 50,000 to this account "
                "immediately or face digital arrest. Do not tell anyone."
            )
        },
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    print(f"  Verdict: {data['verdict']}")
    print(f"  Confidence: {data['confidence']}")
    print(f"  Risk Level: {data['risk_level']}")
    print(f"  Agents: {data['agents_invoked']}")
    if "langgraph" in data:
        langgraph = data["langgraph"]
        print(
            "  LangGraph: "
            f"{langgraph['graph_type']}, iterations={langgraph['iterations']}, "
            f"self_correction={langgraph['self_correction']}"
        )
    print(f"  Trace steps: {len(data['trace'])}")
    for step in data["trace"]:
        name = step.get("step", "?")
        if "techniques" in step:
            print(f"    -> {name}: {step.get('techniques', [])}")
        elif "confidence" in step:
            print(f"    -> {name}: confidence={step['confidence']}")
        elif "fused_score" in step:
            print(f"    -> {name}: score={step['fused_score']}")
        else:
            print(f"    -> {name}")
    print(f"  Time: {data['processing_time_seconds']}s")
    print()

    print("=" * 60)
    print("TEST 3: Legitimate Text Analysis")
    response = requests.post(
        f"{BASE}/api/analyze/text",
        json={
            "text": (
                "Thank you for calling State Bank customer service. Your account "
                "balance is Rs 1,50,000. Your next EMI is due on July 15th. Visit "
                "your branch for any queries."
            )
        },
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    print(f"  Verdict: {data['verdict']}")
    print(f"  Confidence: {data['confidence']}")
    print(f"  Risk Level: {data['risk_level']}")
    print()

    print("=" * 60)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
