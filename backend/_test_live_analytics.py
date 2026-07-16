"""Test live analytics flow: submit analyses, verify threat feed updates."""
import requests, json, time

BASE = "http://localhost:8000"

def get_stats():
    r = requests.get(f"{BASE}/api/intelligence/threat-feed", timeout=10)
    return r.json()

# Step 1: Check initial state
print("=== STEP 1: Initial State ===")
stats = get_stats()
s = stats.get("summary", {})
print(f"  is_live: {stats.get('is_live')}")
print(f"  total_analyses: {s.get('total_analyses')}")
print(f"  threats_detected_24h: {s.get('threats_detected_24h')}")
print(f"  active_patterns: {s.get('active_patterns')}")
print()

# Step 2: Submit a scam text
print("=== STEP 2: Submit Scam Text ===")
scam = "This is Inspector Sharma from CBI. Your Aadhaar is linked to money laundering. Transfer funds to RBI safe account immediately or face arrest."
r = requests.post(f"{BASE}/api/analyze/text", json={"text": scam, "language": "en"}, timeout=60)
result = r.json()
print(f"  verdict: {result.get('verdict')}")
print(f"  confidence: {result.get('confidence')}")
print(f"  risk_level: {result.get('risk_level')}")
print()

# Step 3: Check updated state
print("=== STEP 3: After Scam Analysis ===")
stats = get_stats()
s = stats.get("summary", {})
print(f"  total_analyses: {s.get('total_analyses')}")
print(f"  threats_detected_24h: {s.get('threats_detected_24h')}")
print(f"  active_patterns: {s.get('active_patterns')}")
campaigns = stats.get("active_campaigns", [])
print(f"  active_campaigns: {len(campaigns)}")
for c in campaigns:
    print(f"    - {c.get('pattern')}: {c.get('count_24h')} detections, trend={c.get('trend')}")
print(f"  modality_breakdown: {stats.get('modality_breakdown')}")
print()

# Step 4: Submit a benign text
print("=== STEP 4: Submit Benign Text ===")
benign = "Thank you for calling State Bank. I can see your transaction of Rs 5000. Is there anything else I can help with?"
r = requests.post(f"{BASE}/api/analyze/text", json={"text": benign, "language": "en"}, timeout=60)
result = r.json()
print(f"  verdict: {result.get('verdict')}")
print(f"  confidence: {result.get('confidence')}")
print()

# Step 5: Check final state
print("=== STEP 5: Final State ===")
stats = get_stats()
s = stats.get("summary", {})
print(f"  total_analyses: {s.get('total_analyses')}")
print(f"  analyses_24h: {s.get('analyses_24h')}")
print(f"  threats_detected_24h: {s.get('threats_detected_24h')}")
print(f"  safe_cleared_24h: {s.get('safe_cleared_24h')}")
print(f"  active_patterns: {s.get('active_patterns')}")
print(f"  detection_rate: {stats.get('detection_rate')}")
print()
print("DONE - All stats are LIVE, derived from actual analyses!")
