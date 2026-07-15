# Digital Public Safety Shield Architecture

```mermaid
flowchart LR
    Citizen[Citizen / Bank / Officer] --> API[FastAPI Gateway]
    API --> Auth[JWT + SQLite Case Store]
    API --> Jobs[Authenticated Async Job API]
    Jobs --> RMQ[RabbitMQ Quorum Queue + DLQ]
    RMQ --> Worker[Analysis Worker]
    Worker --> LG
    API --> Redis[Redis Atomic Rate Limits]
    MCP[Trusted MCP stdio Adapter: Tools + Resources + Prompt] --> API
    API --> LG[LangGraph Orchestrator]
    LG --> V[Vision Agent\nYOLO + EfficientNet + CLIP + Forensics]
    LG --> S[Speech Agent\nWhisper + Spoof Detection]
    LG --> N[NLP Agent\nDistilBERT + Hybrid RAG + Multi-role LLM]
    LG --> G[Graph Agent\nGAT + Community Detection]
    V --> F[XGBoost / Weighted Fusion]
    S --> F
    N --> F
    G --> F
    F --> C[Probability Calibration]
    C --> D[Explainable Verdict + Trace]
    D --> E[SHA-256 Evidence Package]
    D --> R[1930 / NCRP Reporting Guidance]
    API --> Geo[Geospatial Hotspot Intelligence]
    API <--> WS[Realtime WebSocket Session]
```

## Deployment Boundaries

- The API is stateless for inference; authenticated profile and case history currently use SQLite for the prototype.
- Model adapters lazy-load heavyweight vision and speech models and degrade to documented fallback paths.
- RabbitMQ is optional. Queue messages contain only opaque job IDs; sensitive payloads remain in the owner-scoped job store and are erased after completion or terminal failure. Manual acknowledgement, confirms, lease heartbeats, retries, recovery, and a DLQ provide at-least-once processing with idempotent job claiming.
- Redis is optional and does not duplicate RabbitMQ. It holds only SHA-256-keyed, expiring counters for shared login and inference rate limits; no analysis payloads or results are cached.
- MCP runs as a separate stdio adapter and calls authenticated HTTP routes, preserving the API's ownership and audit boundary. It exposes annotated tools, read-only resources, and a human-reviewed triage prompt; queued tools use the same RabbitMQ job API.
- Every saved verdict receives a SHA-256 integrity record. Evidence exports add custody, model trace, fusion details, and an explicit human-review disclosure.
- Production deployment should replace SQLite with managed PostgreSQL, use managed Redis/RabbitMQ with TLS and credential rotation, terminate TLS at an API gateway, use a secrets manager, add redacted OpenTelemetry, and ingest only authorized government, bank, and telecom feeds.

## Privacy and Safety

- Anonymous analysis is supported and is not persisted.
- Authenticated analysis is persisted only for the account that submitted it.
- Evidence packages are ownership checked and are decision-support artifacts, not autonomous enforcement decisions.
- Uploaded media is processed in memory by the API and file size/content type are restricted.
