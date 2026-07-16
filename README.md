# Digital Public Safety Shield

ET AI Hackathon 2026, Problem Statement 6. This repository contains a working multimodal fraud-analysis prototype with a FastAPI backend and React/Vite frontend.

## Capabilities

- Digital-arrest, impersonation, OTP/KYC, parcel, investment, and financial scam analysis.
- Currency-image screening with YOLO localization, EfficientNet/Transformer classification, CLIP, ELA, FFT, NPR, and Grad-CAM.
- Audio transcription and spoof-risk signals with Whisper and WavLM/AASIST-style analysis.
- Hybrid BM25 + dense RAG with cross-encoder reranking.
- LangGraph agent routing, traceable fusion, calibration, and modality-gated XGBoost fallback.
- Fraud-network analysis with a Graph Attention Network and community/centrality signals.
- **Threat Intelligence Command Centre** — interactive geospatial heatmap (Leaflet), D3 force-directed fraud network graph, predictive threat feed, and AI model benchmark dashboard.
- **Real-time analytics** — every analysis is tracked; Command Centre stats update live (auto-refresh every 10 seconds). The map remains explicitly labelled as demonstration intelligence until authorized feeds are connected.
- **False-positive reduction** — `needs_review` intermediate verdict tier prevents borderline cases from triggering false alarms.
- Authentication, case history, evidence hashing, reporting guidance, WebSockets, hotspots, and multilingual UI support.
- Optional RabbitMQ quorum-queue processing for authenticated, durable text-analysis jobs.
- Optional Redis coordination for shared login/analysis rate limits without storing citizen payloads.
- Optional MCP stdio analyst adapter with tools, resources, prompts, case evidence, and queued analysis.

This is a research prototype. It does not automatically file police complaints, certify banknotes, replace a forensic examiner, or claim production accuracy.

## Repository Layout

```text
backend/                 FastAPI API, agents, models, data and training scripts
backend/analytics.py     Real-time analytics tracker (logs every analysis for live Command Centre)
frontend/                React 19 + Vite application
frontend/src/CommandCentre.jsx   Threat Intelligence Command Centre component
frontend/src/CommandCentre.css   Command Centre styling
docs/                    Architecture and problem-statement checklist
backend/data/training/   Small tracked datasets and reproducible manifests
```

## Prerequisites

- Python 3.11 or newer
- Node.js 20.19+ or 22.12+
- Git
- About 5 GB free disk space for Python packages and downloaded model caches
- Kaggle credentials only if preparing the currency dataset
- Docker Desktop only if using the optional RabbitMQ/Redis services

Windows PowerShell commands are shown below. Run them from the repository root unless a step says otherwise.

## Backend Setup

```powershell
cd "D:\ET AI Hackathon"
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
Copy-Item backend\.env.example backend\.env
```

Edit `backend/.env` and set:

```dotenv
OPENROUTER_API_KEY=your_openrouter_key
GROQ_API_KEY=your_groq_key
JWT_SECRET=your_long_random_secret
DEBUG=false
```

- OpenRouter uses `moonshotai/kimi-k2.5` first, then `moonshotai/kimi-k2.6:free` before the existing Groq fallbacks for reasoning, routing, and vision.
- Groq provides hosted Whisper transcription and the final configured fallback services.
- Generate a JWT secret with `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
- Never commit `backend/.env`.
- Set `ASYNC_JOBS_ENABLED=true` and `RABBITMQ_URL` only when running the optional broker and worker.
- Set `REDIS_ENABLED=true` and `REDIS_URL` only when running shared distributed limits.

Start the backend in terminal 1:

```powershell
cd "D:\ET AI Hackathon\backend"
..\.venv\Scripts\Activate.ps1
python main.py
```

Backend URLs:

- API: http://localhost:8000
- Health: http://localhost:8000/api/health
- OpenAPI: http://localhost:8000/docs

The first startup can take several minutes while local Hugging Face, CLIP, and vision assets are downloaded or loaded. Keep `DEBUG=false` while testing models so file changes do not repeatedly reload them.

## Optional Infrastructure

RabbitMQ and Redis have distinct jobs and are not interchangeable:

| Service | Purpose | Data boundary | Required for normal API? |
|---|---|---|---|
| RabbitMQ | Durable background analysis delivery, retries, dead-lettering | Queue messages contain only an opaque job ID | No |
| Redis | Atomic shared rate limits across API processes | Hashed identifiers and expiring counters only | No |
| SQLite | Prototype users, cases, and durable job/result records | Owner-scoped application data | Yes for current prototype auth/history |

Start both optional services with `docker compose up -d rabbitmq redis`. The compose defaults are development credentials; set `RABBITMQ_USER`, `RABBITMQ_PASSWORD`, and `REDIS_PASSWORD` in a root-level untracked `.env` before shared deployment.

### Durable RabbitMQ Jobs

The normal synchronous API does not require RabbitMQ. For long-running authenticated text jobs, start the broker and a separate worker:

```powershell
cd "D:\ET AI Hackathon"
docker compose up -d rabbitmq redis

# In backend/.env:
# ASYNC_JOBS_ENABLED=true
# RABBITMQ_URL=amqp://shield:change-me@localhost:5672/

cd backend
python workers\analysis_worker.py
```

The API stores sensitive input server-side and publishes only an opaque job ID. The worker uses a long-lived consumer, manual acknowledgements, publisher-confirmed durable messages, pooled publishing, processing-lease heartbeats, retry limits, stale-lease recovery, and a dead-letter quorum queue. RabbitMQ management is available at http://localhost:15672.

### Redis Coordination

Enable Redis in `backend/.env`:

```dotenv
REDIS_ENABLED=true
REDIS_URL=redis://:change-me@localhost:6379/0
REDIS_MAX_CONNECTIONS=20
LOGIN_RATE_LIMIT=8
LOGIN_RATE_WINDOW_SECONDS=300
ANALYSIS_RATE_LIMIT=30
ANALYSIS_RATE_WINDOW_SECONDS=60
```

FastAPI creates one shared asynchronous Redis pool and closes it at shutdown. Rate-limit increments and expiry are atomic. If Redis is disabled or unavailable, login protection falls back to the process-local limiter and analysis remains available; health reports the degraded state. Redis does not cache analysis text, media, model output, JWTs, or case evidence.

## Optional MCP Adapter

MCP is a separate local stdio process for trusted analyst clients. It reuses the HTTP API's authentication, ownership, persistence, and audit boundary instead of accessing databases or models directly:

```powershell
$env:SHIELD_API_BASE_URL="http://localhost:8000"
$env:SHIELD_API_TOKEN="your_login_access_token"
python backend\mcp_server.py
```

Obtain the access token by signing in through the frontend or `POST /api/auth/login`. The token determines case ownership and expires according to `ACCESS_TOKEN_MINUTES`. Do not expose the stdio adapter directly to untrusted users or place access tokens in source control.

| MCP primitive | Implemented capability |
|---|---|
| Tools | Health, synchronous analysis, RabbitMQ job submission/status, recent cases, evidence package, hotspots, graph summary, reporting guidance |
| Resources | `shield://capabilities`, `shield://reporting-guidance` |
| Prompt | Human-reviewed fraud triage with explicit non-enforcement limitations |

Every tool has MCP risk annotations. Owner-scoped tools require `SHIELD_API_TOKEN`; analysis is marked additive and external because it saves a case and may call hosted models. MCP is an integration surface for analyst assistants, not another model inside the classifier.

## Frontend Setup

Start the frontend in terminal 2:

```powershell
cd "D:\ET AI Hackathon\frontend"
npm install
npm run dev
```

Open http://localhost:5173. The frontend uses `http://localhost:8000` by default. For another backend URL, create `frontend/.env.local`:

```dotenv
VITE_API_BASE_URL=http://localhost:8000
```

## Dataset Preparation

### Text

The tracked generator creates 240 balanced records across six scam and six legitimate categories. An optional hard-negative mode adds legitimate texts containing scam-adjacent keywords (arrest, CBI, customs, OTP) for false-positive reduction research. Stable template groups prevent variants of one template from crossing train and validation folds.

```powershell
python backend\data\scripts\generate_text_dataset.py
```

### Currency

Synthetic pattern images were removed. The preparation script selectively downloads a balanced subset of the publisher-labelled Indian Currency Real vs Fake Notes Dataset, validates dimensions/readability, rejects exact duplicates, and records SHA-256/provenance in `source_manifest.json`.

1. Put Kaggle credentials at `%USERPROFILE%\.kaggle\kaggle.json`.
2. Run:

```powershell
python backend\data\scripts\prepare_currency_dataset.py
```

The current target is at least 500 images across INR 10, 20, 50, 100, 500, and 2000. Images and downloaded archives are Git-ignored; the source manifest and preparation code are tracked. Labels are publisher-provided research labels, not RBI or laboratory certifications. The source license is CC BY-NC-SA 4.0, so review its noncommercial/share-alike requirements before use.

## Model Training

Run quick smoke training first:

```powershell
python backend\data\scripts\train_text_classifier.py --smoke
python backend\data\scripts\train_vision_classifier.py --smoke
python backend\data\scripts\train_graph_model.py --smoke
python backend\data\scripts\prepare_fusion_validation.py
python backend\data\scripts\train_xgboost_fusion.py --smoke
```

Run the complete command-ready pipeline after the currency dataset is prepared:

```powershell
python backend\data\scripts\train_all.py
```

Or run full models individually:

```powershell
python backend\data\scripts\train_text_classifier.py
python backend\data\scripts\train_vision_classifier.py
python backend\data\scripts\train_graph_model.py
python backend\data\scripts\prepare_fusion_validation.py
python backend\data\scripts\train_xgboost_fusion.py
python backend\data\scripts\evaluate_text_benchmark.py
```

Training metadata is saved beside each model. Large weights are intentionally ignored by Git and must be generated locally. Full CPU-only training can take a long time.

## Verification

With the backend running:

```powershell
python backend\_test_ml_contracts.py
python backend\_test_operational.py
python backend\_test_langgraph.py
python backend\_test_e2e.py
```

Frontend checks:

```powershell
cd frontend
npm run lint
npm run i18n:check
npm run build
```

Live analytics end-to-end test (requires running backend):

```powershell
python backend\_test_live_analytics.py
```

RabbitMQ, Redis, and MCP integration checks:

```powershell
$env:ASYNC_JOBS_ENABLED="true"
$env:RABBITMQ_URL="amqp://shield:change-me@localhost:5672/"
python backend\tests\test_broker_integration.py
python backend\tests\test_job_lease.py
python backend\tests\test_redis_integration.py
python backend\tests\test_mcp_integration.py
```

Set `SHIELD_API_TOKEN` before the MCP test to additionally verify MCP -> API -> RabbitMQ -> worker -> completed result. Without a token, it verifies public discovery, annotations, resources, prompts, health, and hotspots. The E2E suite checks health, scam/benign text, turn-by-turn analysis, demo contracts, and real genuine/counterfeit dataset images. Hosted-model tests require valid API keys and internet access.

## Main API Routes

| Route | Purpose |
|---|---|
| `GET /api/health` | Agent, dataset, model, and security readiness |
| `POST /api/analyze` | Combined text/image/audio analysis |
| `POST /api/analyze/text` | Text scam analysis |
| `POST /api/analyze/image` | Currency/document image screening |
| `POST /api/analyze/audio` | Audio transcription and spoof/scam analysis |
| `POST /api/analyze/turns` | Turn-by-turn risk trajectory |
| `POST /api/jobs/analyze/text` | Authenticated durable text-job submission (optional RabbitMQ) |
| `GET /api/jobs/{job_id}` | Owner-scoped asynchronous job status/result |
| `GET /api/graph/analyze` | Fraud-network analysis |
| `GET /api/graph/visualization` | Graph visualization payload |
| `GET /api/intelligence/threat-feed` | **Real-time** threat intelligence (live analytics from actual system analyses) |
| `GET /api/intelligence/command-centre` | Unified command centre data (geospatial + network + system) |
| `GET /api/benchmarks` | AI model benchmark metrics (precision, recall, F1, FP rate per model) |
| `GET /api/demo/scam-transcript` | Scam demo fixture |
| `GET /api/demo/benign-transcript` | Benign demo fixture |
| `WS /ws/trace` | Live analysis trace |

## Command Centre

The Command Centre provides a unified threat intelligence dashboard accessible via the "Command Centre" tab in the frontend. It consists of four panels:

| Panel | Technology | Data Source |
|---|---|---|
| **Geospatial Intelligence** | Leaflet + OpenStreetMap | Demonstration intelligence feed with 12 reference hotspots; authorized feeds are required for operations |
| **Fraud Network Graph** | D3 force-directed simulation | `/api/graph/visualization` — current local GAT demonstration graph |
| **Live Threat Feed** | Auto-refresh every 10s | `/api/intelligence/threat-feed` — real-time from `analytics.py` tracker |
| **AI Benchmarks** | Visual progress bars | `/api/benchmarks` — precision, recall, F1, FP rate per model |

### Real-Time Analytics

The threat feed is **not hardcoded**. Every analysis submitted through any endpoint (`/api/analyze/text`, `/api/analyze/image`, `/api/analyze/audio`, `/api/analyze`) is logged to an in-memory analytics tracker (`backend/analytics.py`). The Command Centre reads from this tracker and auto-refreshes every 10 seconds.

- **Total Analyses**: Count of all analyses performed since server start
- **Threats Detected**: Analyses returning `high_risk` or `medium_risk` verdicts
- **Cleared Safe**: Analyses returning `safe` or `low_risk` verdicts
- **Scam Patterns**: Distinct scam types extracted from NLP RAG matches
- **Modality Breakdown**: Count of text/image/audio analyses
- **Active Campaigns**: Scam patterns grouped with detection count, trend (surging/rising/steady/new), and confidence stats

### Verdict Tiers

The fusion orchestrator uses six verdict tiers to minimize false positives:

| Calibrated Score | Verdict | Risk Level |
|---|---|---|
| > 0.80 | `high_risk` | critical |
| > 0.60 | `high_risk` | high |
| > 0.45 | `medium_risk` | medium |
| > 0.30 | `needs_review` | review |
| > 0.20 | `low_risk` | low |
| ≤ 0.20 | `safe` | safe |

The `needs_review` tier prevents borderline cases (0.30–0.45) from triggering false alarms, routing them for human review instead.

## Current Data and Model Boundaries

- Text training data is curated/template-generated; the separate Chakravyuh benchmark is test-only and never used for training.
- Currency training uses 510 validated, deduplicated publisher-labelled research images in the current local preparation.
- Fixed UI text has complete checked-in coverage for 12 languages; Kimi generates explanations, indicators, and recommendations in the selected language at analysis time, with localized deterministic fallbacks.
- The graph is a 69-node demonstration network, not live law-enforcement intelligence.
- CLIP and XGBoost load lazily or fall back safely when artifacts are unavailable.
- Current grouped currency validation: 93.9% accuracy, 93.5% F1, and 0.984 ROC-AUC. These are research-split metrics, not currency-certification accuracy.
- Current Chakravyuh test-only text result: 0.853 ROC-AUC, 0.746 F1, 95.7% precision, 61.1% recall, and 12.9% false-positive rate (4/31 benign) across 175 scenarios at the action threshold. Top category scores include OTP theft (F1: 0.957) and KYC fraud (F1: 0.902). The `needs_review` tier captures 8 borderline cases (7 scam, 1 benign). Regional-language subsets are too small for language-level claims.
- Current fusion meta-learner: 1,382 labelled rows and 0.723 overall validation ROC-AUC. The deployment quality gate enables XGBoost only for image signatures (0.951 validation ROC-AUC); text, audio, and unseen combinations use weighted fallback.
- Court admissibility and government deployment require authorized acquisition, retention, audit, privacy, accessibility, security review, human oversight, and independent validation.

## Evaluated Production Additions

- **OpenTelemetry:** relevant next step for API, worker, hosted-model, RabbitMQ, and Redis traces/metrics. It is not enabled because exporter, collector, sampling, and PII-redaction policies must be chosen together.
- **PostgreSQL:** recommended replacement for SQLite when API and workers run on multiple hosts. Redis is intentionally not used as the durable case database.
- **Object storage:** appropriate for encrypted media/evidence retention only after retention, consent, access-control, and deletion policies are defined. Uploads currently stay in memory.
- **Kafka, Celery, and another vector database:** not added. RabbitMQ already handles work delivery, the worker has explicit ownership/lease semantics, and ChromaDB plus BM25 already serves prototype RAG.
- **Remote MCP over HTTP:** not added. Local stdio minimizes exposure; remote use would require OAuth 2.1 scopes, TLS, consent UI, and per-tool authorization.

See [docs/PROBLEM_STATEMENT_CHECKLIST.md](docs/PROBLEM_STATEMENT_CHECKLIST.md) for requirement coverage and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system boundaries.

## Common Problems

- `401` or hosted-model failures: verify API keys in `backend/.env` and restart the backend.
- Port already in use: stop the existing process or launch Uvicorn on another port and set `VITE_API_BASE_URL`.
- Slow first request: local model initialization and downloads are intentionally lazy.
- Currency classifier unavailable: prepare the dataset and train `train_vision_classifier.py`.
- XGBoost shows `weighted_fallback`: generate held-out fusion predictions and train `train_xgboost_fusion.py`.
- Redis shows `unavailable`: verify `docker compose ps redis`, the password in `REDIS_URL`, and `REDIS_ENABLED=true`.
- RabbitMQ jobs return `503`: enable `ASYNC_JOBS_ENABLED`, start RabbitMQ and the worker, and verify the broker URL.
- Kaggle authentication fails: verify `%USERPROFILE%\.kaggle\kaggle.json` and its key permissions.
