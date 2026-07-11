# 🛡️ Digital Public Safety Shield
### ET AI Hackathon 2026 — Problem Statement 6
### Multi-Agent AI System for Citizen Fraud Detection

---

## 🧠 Architecture

A **multi-agent AI system** — not a rules engine. Every verdict is produced by real models:

| Agent | Models | Purpose |
|-------|--------|---------|
| **Vision Agent** | YOLOv8 + EfficientNet + Transformer Attention + Contrastive Learning + ELA + FFT + NPR + CLIP + Grad-CAM | Counterfeit currency & deepfake document detection |
| **Speech Agent** | Whisper + WavLM/AASIST | Voice transcription & spoofing detection |
| **NLP Agent** | Kimi K2 (Multi-Role CoT) + DistilBERT + RAG + ChromaDB | Scam pattern analysis with agentic reasoning |
| **Fusion Orchestrator** | LangGraph routing + XGBoost stacking + weighted fallback + Isotonic calibration | Multi-agent verdict fusion |

**21 distinct AI techniques/signals** across 5 AI domains. All at **$0.00 cost** (free-tier APIs + open-weight models).

---

## 🚀 Quick Start

### 1. Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure API keys (free!)
copy .env.example .env
# Edit .env with your Groq key from https://console.groq.com

# Run server
python main.py
```

---

## ML Upgrade Status

- Text dataset generator now writes the full balanced corpus: 240 records across digital arrest, financial fraud, impersonation, KYC/OTP, parcel/customs, job/investment scams, and legitimate-service calls.
- Vision synthetic currency generation now produces 500 images: 250 genuine and 250 counterfeit with richer blur, noise, color-shift, security-thread, watermark, and print-defect variations.
- Graph training now uses an expanded fraud graph with 69 nodes and 212 adjacency edges, with metadata saved beside the trained GAT artifacts.
- Vision analysis includes a lazy CLIP/open-clip zero-shot signal when `open-clip-torch` and model weights are available. If CLIP cannot load, the app continues with the trained classifier and forensic stack.
- Fusion includes an optional XGBoost meta-learner. If `backend/data/trained_models/xgboost_fusion/model.json` is missing or cannot load, the orchestrator falls back to the existing weighted fusion path.

Large generated model weights, local databases, `.env`, `node_modules`, and frontend build output are intentionally ignored by Git. Training scripts regenerate those artifacts locally. CPU-only full retraining may take longer, so smoke training remains available for quick validation.

Problem-statement coverage and honest deployment limitations are tracked in [`docs/PROBLEM_STATEMENT_CHECKLIST.md`](docs/PROBLEM_STATEMENT_CHECKLIST.md). The system diagram and production boundaries are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

### 3. Open

Navigate to http://localhost:5173

---

## 🔑 API Keys (All Free!)

| Provider | Get Key At | Cost |
|----------|-----------|------|
| **Groq** | [console.groq.com](https://console.groq.com) | Free (no credit card) |
| **OpenRouter** | [openrouter.ai](https://openrouter.ai) | Free (no credit card) |

---

## 🎯 Demo Scenarios

1. **Scam Call Transcript**: Click "Demo: Scam Call" to analyze a real digital arrest scam script turn-by-turn. Watch confidence climb as scam indicators accumulate.

2. **Legitimate Call**: Click "Demo: Legitimate Call" to analyze a normal customer service conversation. Confidence stays low.

3. **Currency Image**: Upload a currency note image for multi-model forgery analysis with Grad-CAM attention heatmaps.

4. **Voice Recording**: Upload a call recording for simultaneous transcription, scam detection, and voice spoofing analysis.

---

## 📊 AI Techniques (17 Total)

### Computer Vision (8)
1. **YOLOv8** — Object detection for currency note localization
2. **EfficientNet-B0** — Transfer learning for forgery classification
3. **Contrastive Learning** (SimCLR) — Similarity space for counterfeit detection
4. **ELA** — Error Level Analysis for digital manipulation detection
5. **FFT** — Frequency domain analysis for print artifact detection
6. **NPR** — Neighboring Pixel Relationship for AI-generated detection
7. **Grad-CAM** — Attention visualization showing model focus areas
8. **CLIP** — Vision-language deepfake document detection

### Speech AI (2)
9. **Whisper** — Speech-to-text transcription (Groq + local)
10. **WavLM/AASIST** — Voice spoofing / deepfake audio detection

### NLP / LLMs (5)
11. **Kimi K2** — Agentic reasoning via Groq (primary model)
12. **Llama 4 Maverick** — Multimodal vision-language reasoning
13. **DistilBERT** — Zero-shot NLI scam classification
14. **RAG** — ChromaDB + sentence-transformers retrieval
15. **Multi-Role CoT** — Investigator → Policy Checker → Risk Assessor

### Fusion / Calibration (2)
16. **XGBoost Stacking** — Ensemble meta-learner
17. **Isotonic Regression** — Probability calibration

---

## 🏗️ Project Structure

```
digital-public-safety-shield/
├── backend/
│   ├── main.py                    # FastAPI server
│   ├── config.py                  # Configuration
│   ├── requirements.txt           # Python dependencies
│   ├── agents/
│   │   ├── orchestrator.py        # Agentic Fusion Orchestrator
│   │   ├── vision_agent.py        # 8-technique vision pipeline
│   │   ├── speech_agent.py        # Whisper + WavLM
│   │   ├── nlp_agent.py           # Kimi K2 multi-role CoT
│   │   └── calibration.py         # Score calibration
│   └── models/
│       ├── vision/
│       │   ├── detector.py        # YOLOv8 currency detector
│       │   ├── classifier.py      # EfficientNet forgery classifier
│       │   ├── forensics.py       # ELA + FFT + NPR
│       │   └── explainability.py  # Grad-CAM
│       ├── speech/
│       │   ├── transcriber.py     # Whisper transcription
│       │   └── spoof_detector.py  # WavLM spoof detection
│       └── nlp/
│           ├── llm_client.py      # Groq/OpenRouter multi-provider
│           ├── rag_engine.py      # ChromaDB RAG
│           └── text_classifier.py # DistilBERT NLI
├── frontend/
│   ├── src/
│   │   ├── App.jsx                # Main UI (chat + verdict cards)
│   │   ├── index.css              # Design system
│   │   └── utils/api.js           # Backend API client
│   └── index.html
└── README.md
```
