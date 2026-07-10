import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

from agents.orchestrator import FusionOrchestrator, FraudAnalysisState
print("LangGraph orchestrator imported OK")

o = FusionOrchestrator()
s = o.get_stats()
print(f"Orchestrator type: {s['orchestrator']}")
print(f"Topology: {s['graph_topology']}")
print(f"Total AI techniques: {s['total_ai_techniques']}")
print("SUCCESS")
