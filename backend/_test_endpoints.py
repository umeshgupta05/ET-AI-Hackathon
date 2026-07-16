"""Smoke test the public Command Centre contracts against a running backend."""

import sys

import requests

BASE = "http://localhost:8000"

tests = [
    ("Threat Feed", "/api/intelligence/threat-feed"),
    ("Command Centre", "/api/intelligence/command-centre"),
    ("Benchmarks", "/api/benchmarks"),
    ("Health", "/api/health"),
    ("Demo Scam", "/api/demo/scam-transcript"),
    ("Demo Benign", "/api/demo/benign-transcript"),
]

failures = []
for name, path in tests:
    try:
        r = requests.get(BASE + path, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if name == "Threat Feed":
                s = data.get("summary", {})
                required = {"total_analyses", "threats_detected_24h", "needs_review_24h", "active_patterns"}
                assert required <= set(s), f"missing summary fields: {required - set(s)}"
                assert "false_positive_estimate" not in data.get("detection_rate", {})
                print(f"  PASS {name}: {s['active_patterns']} active patterns, {s['total_analyses']} analyses")
            elif name == "Command Centre":
                net = data.get("network", {})
                assert {"total_nodes", "total_edges", "high_risk_entities"} <= set(net)
                assert data.get("geospatial", {}).get("source")
                print(f"  PASS {name}: {net['total_nodes']} nodes, {net['total_edges']} edges")
            elif name == "Benchmarks":
                models = data.get("models", [])
                assert models, "no metadata-backed models returned"
                assert all(model.get("evaluation_set") for model in models)
                print(f"  PASS {name}: {len(models)} locally benchmarked models")
            elif "Demo" in name:
                turns = data.get("turns", [])
                print(f"  PASS {name}: {len(turns)} turns")
            else:
                print(f"  PASS {name}")
        else:
            print(f"  FAIL {name}: [{r.status_code}]")
            failures.append(name)
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        failures.append(name)

if failures:
    print(f"Endpoint smoke test failed: {', '.join(failures)}")
    sys.exit(1)
print("All endpoint tests passed")
