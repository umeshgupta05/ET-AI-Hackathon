"""
RAG Engine — ChromaDB + sentence-transformers for scam pattern retrieval.

Retrieval-Augmented Generation grounded on:
- MHA advisories on digital arrest scams
- Known scam script templates
- Publicly reported fraud patterns

This ensures LLM verdicts are retrieval-grounded, not hallucinated.
Includes semantic deviation scoring for adaptive thresholds.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

try:
    import chromadb

    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer

    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

try:
    from rank_bm25 import BM25Okapi

    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

from config import config, SCAM_PATTERNS_DIR, CHROMA_DB_DIR

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Retrieval engine for scam pattern matching.

    Uses:
    - sentence-transformers (all-MiniLM-L6-v2) for embeddings — 22M params, runs on CPU
    - ChromaDB for persistent vector storage
    - Semantic deviation scoring for adaptive anomaly detection
    """

    def __init__(self):
        self._embedder = None
        self._reranker = None
        self._chroma_client = None
        self._collection = None
        self._bm25 = None
        self._bm25_records: list[dict] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Load embedding model and initialize ChromaDB collection."""
        if self._initialized:
            return

        if not HAS_CHROMADB or not HAS_SBERT:
            logger.warning(
                "ChromaDB or sentence-transformers not installed — RAG engine disabled"
            )
            self._initialized = True
            return

        logger.info("📚 Initializing RAG engine...")

        # Load sentence-transformer embedding model (runs on CPU, ~90MB)
        self._embedder = SentenceTransformer(
            config.local_models.embedding_model,
            cache_folder=str(config.MODELS_CACHE_DIR)
            if hasattr(config, "MODELS_CACHE_DIR")
            else None,
        )
        logger.info(f" Embedding model loaded: {config.local_models.embedding_model}")

        # Initialize ChromaDB persistent client
        self._chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        self._collection = self._chroma_client.get_or_create_collection(
            name="scam_patterns",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f" ChromaDB collection ready ({self._collection.count()} documents)"
        )

        # Seed corpus if empty
        if self._collection.count() == 0:
            await self._seed_corpus()

        self._bm25_records = self._load_pattern_records()
        self._build_bm25_index()
        self._load_reranker()

        self._initialized = True

    async def _seed_corpus(self) -> None:
        """Load scam pattern corpus into ChromaDB on first run."""
        logger.info(" Seeding scam pattern corpus...")

        patterns = self._get_builtin_patterns()

        # Also load any JSON files from the data directory
        for json_file in SCAM_PATTERNS_DIR.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    file_patterns = json.load(f)
                    if isinstance(file_patterns, list):
                        patterns.extend(file_patterns)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

        if not patterns:
            logger.warning("No scam patterns found to seed")
            return

        # Embed and store
        documents = [p["text"] for p in patterns]
        ids = [p["id"] for p in patterns]
        metadatas = [
            {
                "category": p.get("category", "unknown"),
                "severity": p.get("severity", "high"),
                "source": p.get("source", "builtin"),
            }
            for p in patterns
        ]

        embeddings = self._embedder.encode(documents, show_progress_bar=False).tolist()

        self._collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )
        logger.info(f" Seeded {len(patterns)} scam patterns into ChromaDB")

    def _load_pattern_records(self) -> list[dict]:
        """Load local pattern records for lexical retrieval and reranking."""
        records = self._get_builtin_patterns()

        for json_file in SCAM_PATTERNS_DIR.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    file_patterns = json.load(f)
                if isinstance(file_patterns, list):
                    records.extend(file_patterns)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

        deduped = {}
        for i, record in enumerate(records):
            text = str(record.get("text", "")).strip()
            if not text:
                continue
            record_id = str(record.get("id") or f"pattern_{i}")
            deduped[record_id] = {
                "id": record_id,
                "document": text,
                "category": record.get("category", "unknown"),
                "severity": record.get("severity", "unknown"),
                "source": record.get("source", "unknown"),
            }

        return list(deduped.values())

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25 with light normalization."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build_bm25_index(self) -> None:
        """Build a lexical BM25 index over the scam pattern corpus."""
        if not HAS_BM25:
            logger.warning("rank_bm25 not installed; lexical RAG disabled")
            return
        if not self._bm25_records:
            logger.warning("No scam patterns available for BM25 indexing")
            return

        tokenized_docs = [self._tokenize(record["document"]) for record in self._bm25_records]
        self._bm25 = BM25Okapi(tokenized_docs)
        logger.info(f" BM25 lexical index ready ({len(self._bm25_records)} documents)")

    def _load_reranker(self) -> None:
        """Load compact cross-encoder reranker when available."""
        if not HAS_SBERT:
            return

        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        try:
            self._reranker = CrossEncoder(
                model_name,
                max_length=256,
                automodel_args={"cache_dir": str(config.MODELS_CACHE_DIR)}
                if hasattr(config, "MODELS_CACHE_DIR")
                else None,
            )
            logger.info(f" Cross-encoder reranker loaded: {model_name}")
        except Exception as e:
            self._reranker = None
            logger.warning(f"Cross-encoder reranker unavailable; using hybrid score only: {e}")

    def _get_builtin_patterns(self) -> list[dict]:
        """Built-in scam pattern corpus based on MHA advisories and public reports."""
        return [
            # ─── Digital Arrest Scam Patterns ─────────────────────────
            {
                "id": "da_001",
                "category": "digital_arrest",
                "severity": "critical",
                "source": "MHA Advisory 2024",
                "text": (
                    "Digital arrest scam pattern: Caller impersonates CBI officer, claims victim's "
                    "Aadhaar is linked to money laundering. Demands victim stay on video call for "
                    "'verification process'. Threatens immediate arrest if call is disconnected. "
                    "Escalates to demanding fund transfer to 'safe RBI custody account' for "
                    "'asset verification'. Uses official-sounding terminology and fake badge numbers."
                ),
            },
            {
                "id": "da_002",
                "category": "digital_arrest",
                "severity": "critical",
                "source": "MHA Advisory 2024",
                "text": (
                    "Digital arrest scam pattern: Caller claims to be from Customs department. "
                    "States a parcel in victim's name containing illegal items (drugs, fake "
                    "passports) has been intercepted. Transfers call to 'senior officer' who "
                    "demands secrecy under Section 'XYZ' of IT Act. Requests victim to transfer "
                    "funds for 'investigation clearance' and threatens FIR filing."
                ),
            },
            {
                "id": "da_003",
                "category": "digital_arrest",
                "severity": "critical",
                "source": "I4C Report 2024",
                "text": (
                    "Digital arrest scam pattern: Caller impersonates ED (Enforcement Directorate) "
                    "officer. Claims victim's bank account is under investigation for PMLA violation. "
                    "Victim is instructed to install remote-access app for 'account verification'. "
                    "Demands continuous video call monitoring. Multiple 'officers' take turns "
                    "threatening the victim. Instructs fund transfer to 'government escrow account'."
                ),
            },
            {
                "id": "da_004",
                "category": "digital_arrest",
                "severity": "critical",
                "source": "News Reports 2024",
                "text": (
                    "Digital arrest scam pattern: Scammers set up fake police station background "
                    "on video call. Caller wears police uniform. Shows fake arrest warrant document "
                    "with victim's name. Threatens victim's family members will also be arrested. "
                    "Instructs victim not to contact anyone else claiming matter is 'sub-judice'. "
                    "Multi-day psychological hostage situation over continuous video call."
                ),
            },
            {
                "id": "da_005",
                "category": "digital_arrest",
                "severity": "high",
                "source": "MHA Advisory 2024",
                "text": (
                    "Digital arrest scam pattern: Caller claims victim's mobile number is about to "
                    "be disconnected due to 'illegal activities'. Call is transferred to fake 'TRAI "
                    "officer' then to fake 'police officer'. Script follows: telecom threat → law "
                    "enforcement impersonation → financial data collection → fund transfer demand. "
                    "Often uses AI-generated voice to sound more authoritative."
                ),
            },
            # ─── Financial Fraud Patterns ─────────────────────────────
            {
                "id": "ff_001",
                "category": "financial_fraud",
                "severity": "high",
                "source": "RBI Alert 2024",
                "text": (
                    "Banking fraud pattern: Caller claims to be from victim's bank. States account "
                    "is compromised and needs immediate 'security update'. Requests OTP, CVV, or "
                    "asks victim to click verification link. Uses spoofed caller ID matching bank's "
                    "official number. May reference actual recent transactions to build credibility."
                ),
            },
            {
                "id": "ff_002",
                "category": "financial_fraud",
                "severity": "high",
                "source": "Cyber Police Reports",
                "text": (
                    "UPI fraud pattern: Victim receives 'collect request' disguised as payment. "
                    "Scammer claims to be buyer on OLX/marketplace and says they are 'sending money' "
                    "but actually sends collect request. Victim approves thinking they are receiving "
                    "payment. Alternatively, scammer sends QR code claiming 'scan to receive money' "
                    "which actually debits victim's account."
                ),
            },
            {
                "id": "ff_003",
                "category": "financial_fraud",
                "severity": "critical",
                "source": "I4C Report 2025",
                "text": (
                    "Investment fraud pattern: Victim is added to WhatsApp/Telegram group showing "
                    "fake trading profits. Initial small investments yield returns to build trust. "
                    "Victim is encouraged to invest larger amounts. When trying to withdraw, "
                    "told to pay 'taxes' or 'processing fees'. Platform is entirely fake. "
                    "Uses AI-generated testimonials and deepfake video endorsements."
                ),
            },
            # ─── Impersonation Patterns ───────────────────────────────
            {
                "id": "imp_001",
                "category": "impersonation",
                "severity": "high",
                "source": "CBI Advisory 2024",
                "text": (
                    "Government impersonation indicators: Use of terms like 'digital arrest', "
                    "'cyber cell custody', 'PMLA investigation', 'Aadhaar verification warrant'. "
                    "Legitimate law enforcement NEVER conducts arrests via video call, never demands "
                    "money for 'case clearance', never asks victims to stay on call continuously, "
                    "and never threatens immediate arrest without proper legal process."
                ),
            },
            {
                "id": "imp_002",
                "category": "impersonation",
                "severity": "high",
                "source": "DoT Advisory 2024",
                "text": (
                    "Telecom impersonation indicators: Claims of SIM deactivation, mobile number "
                    "linked to crime, TRAI disconnection notice. Legitimate telecom operators and "
                    "TRAI never make threatening calls, never demand immediate payment to prevent "
                    "disconnection, and never transfer calls to 'police' for resolution."
                ),
            },
            # ─── Scam Script Linguistic Markers ───────────────────────
            {
                "id": "ling_001",
                "category": "linguistic_markers",
                "severity": "medium",
                "source": "Research Analysis",
                "text": (
                    "Scam linguistic markers: Urgency creation ('immediately', 'right now', "
                    "'within the hour'), authority claims ('I am calling from CBI/ED/Police'), "
                    "secrecy demands ('do not tell anyone', 'this is confidential', 'sub-judice "
                    "matter'), threat escalation ('you will be arrested', 'your family will be "
                    "affected'), financial pressure ('transfer to safe account', 'verification "
                    "deposit required')."
                ),
            },
            {
                "id": "ling_002",
                "category": "linguistic_markers",
                "severity": "medium",
                "source": "Research Analysis",
                "text": (
                    "Legitimate vs. scam call differentiators: Legitimate officers provide verifiable "
                    "badge numbers, station details, and invite victims to visit station in person. "
                    "Scammers resist verification, demand the victim stay on call, create artificial "
                    "urgency, refuse to provide verifiable contact details, and escalate emotional "
                    "pressure progressively across the call."
                ),
            },
            # ─── Counterfeit Currency Patterns ────────────────────────
            {
                "id": "cf_001",
                "category": "counterfeit_currency",
                "severity": "high",
                "source": "RBI Annual Report 2025",
                "text": (
                    "Counterfeit Rs 500 note indicators: High-quality FICN (Fake Indian Currency "
                    "Notes) show discontinuous security thread, blurred or absent micro-lettering "
                    "'RBI' and 'भारतीय', inconsistent color-shifting ink on denomination numeral, "
                    "missing or poorly reproduced latent image of denomination, low-quality "
                    "intaglio printing texture on Mahatma Gandhi portrait."
                ),
            },
            {
                "id": "cf_002",
                "category": "counterfeit_currency",
                "severity": "high",
                "source": "RBI Annual Report 2025",
                "text": (
                    "Counterfeit Rs 2000 note indicators: Missing or non-functional see-through "
                    "register, absent or non-fluorescent security fiber, bleed-through of printing "
                    "on reverse side, inconsistent serial number font, missing windowed security "
                    "thread with inscription 'RBI 2000', absence of angular lines for visually "
                    "impaired identification on both sides."
                ),
            },
            # ─── Benign Conversation Templates ────────────────────────
            {
                "id": "benign_001",
                "category": "benign",
                "severity": "none",
                "source": "Training Data",
                "text": (
                    "Normal customer service call: Agent greets customer, asks how they can help, "
                    "discusses product inquiry, provides information about services, offers to "
                    "transfer to specific department, provides reference number, thanks customer "
                    "and ends call politely. No urgency, threats, or financial demands."
                ),
            },
            {
                "id": "benign_002",
                "category": "benign",
                "severity": "none",
                "source": "Training Data",
                "text": (
                    "Legitimate bank call: Bank representative identifies themselves with employee "
                    "ID, confirms last few digits of account for verification, discusses pending "
                    "transaction or card renewal, never asks for full card number, CVV, OTP or "
                    "password, provides branch contact for in-person verification."
                ),
            },
        ]

    def _query_dense(
        self,
        text: str,
        top_k: int = 3,
        category_filter: Optional[str] = None,
        min_similarity: float = 0.3,
    ) -> list[dict]:
        """
        Query the RAG corpus for matching scam patterns.

        Returns list of matches with:
        - document: the matching pattern text
        - similarity: cosine similarity score
        - category: pattern category
        - severity: pattern severity level
        """
        if not self._initialized or not self._embedder or not self._collection:
            raise RuntimeError("RAG engine not initialized. Call initialize() first.")

        # Embed query
        query_embedding = self._embedder.encode([text]).tolist()

        # Build ChromaDB query
        where_filter = {"category": category_filter} if category_filter else None

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to structured output
        matches = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                # ChromaDB returns distances; convert to similarity
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = 1 - distance  # cosine distance → cosine similarity

                if similarity >= min_similarity:
                    metadata = (
                        results["metadatas"][0][i] if results["metadatas"] else {}
                    )
                    matches.append(
                        {
                            "document": doc,
                            "similarity": round(similarity, 4),
                            "category": metadata.get("category", "unknown"),
                            "severity": metadata.get("severity", "unknown"),
                            "source": metadata.get("source", "unknown"),
                            "id": results["ids"][0][i]
                            if results["ids"]
                            else f"match_{i}",
                        }
                    )

        # Sort by similarity descending
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches

    def _query_bm25(
        self,
        text: str,
        top_k: int,
        category_filter: Optional[str] = None,
    ) -> list[dict]:
        """Query local BM25 lexical index."""
        if not self._bm25:
            return []

        tokens = self._tokenize(text)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        max_score = float(max(scores)) if len(scores) else 0.0
        if max_score <= 0:
            return []

        matches = []
        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)
        for idx, score in ranked:
            record = self._bm25_records[idx]
            if category_filter and record["category"] != category_filter:
                continue

            bm25_score = float(score) / max_score
            matches.append(
                {
                    **record,
                    "similarity": round(bm25_score, 4),
                    "dense_score": 0.0,
                    "bm25_score": round(bm25_score, 4),
                    "hybrid_score": round(bm25_score, 4),
                    "rerank_score": None,
                    "retrieval_method": "bm25",
                }
            )
            if len(matches) >= top_k:
                break
        return matches

    def _merge_hybrid_results(
        self,
        dense_matches: list[dict],
        bm25_matches: list[dict],
    ) -> list[dict]:
        """Merge dense and lexical candidates into one ranked candidate set."""
        merged = {}
        for match in dense_matches:
            match = match.copy()
            match["dense_score"] = match.get("similarity", 0.0)
            match["bm25_score"] = 0.0
            match["hybrid_score"] = match.get("similarity", 0.0)
            match["rerank_score"] = None
            match["retrieval_method"] = "dense"
            merged[match["id"]] = match

        for match in bm25_matches:
            existing = merged.get(match["id"])
            if not existing:
                merged[match["id"]] = match.copy()
                continue

            existing["bm25_score"] = max(existing.get("bm25_score", 0.0), match["bm25_score"])
            existing["similarity"] = max(existing.get("similarity", 0.0), match["similarity"])
            existing["retrieval_method"] = "hybrid"

        for match in merged.values():
            dense_score = float(match.get("dense_score") or 0.0)
            bm25_score = float(match.get("bm25_score") or 0.0)
            match["hybrid_score"] = round((0.65 * dense_score) + (0.35 * bm25_score), 4)

        return list(merged.values())

    def _rerank(self, text: str, candidates: list[dict]) -> list[dict]:
        """Rerank candidates with cross-encoder when available."""
        if not self._reranker or not candidates:
            return sorted(candidates, key=lambda x: x.get("hybrid_score", 0.0), reverse=True)

        try:
            pairs = [(text, candidate["document"]) for candidate in candidates]
            scores = self._reranker.predict(pairs)
            min_score = float(min(scores))
            max_score = float(max(scores))
            span = max(max_score - min_score, 1e-6)

            for candidate, score in zip(candidates, scores):
                normalized = (float(score) - min_score) / span
                candidate["rerank_score"] = round(normalized, 4)
                candidate["hybrid_score"] = round(
                    (0.70 * normalized)
                    + (0.30 * float(candidate.get("hybrid_score", 0.0))),
                    4,
                )
                candidate["retrieval_method"] = f"{candidate['retrieval_method']}+rerank"
        except Exception as e:
            logger.warning(f"RAG reranking failed; falling back to hybrid score: {e}")

        return sorted(candidates, key=lambda x: x.get("hybrid_score", 0.0), reverse=True)

    def query(
        self,
        text: str,
        top_k: int = 3,
        category_filter: Optional[str] = None,
        min_similarity: float = 0.3,
    ) -> list[dict]:
        """
        Query the RAG corpus using dense retrieval + BM25 + optional reranking.
        """
        if not self._initialized or not self._embedder or not self._collection:
            raise RuntimeError("RAG engine not initialized. Call initialize() first.")

        candidate_k = max(top_k * 4, 8)
        dense_matches = self._query_dense(
            text,
            top_k=candidate_k,
            category_filter=category_filter,
            min_similarity=0.0,
        )
        bm25_matches = self._query_bm25(
            text,
            top_k=candidate_k,
            category_filter=category_filter,
        )
        candidates = self._merge_hybrid_results(dense_matches, bm25_matches)
        candidates = self._rerank(text, candidates)

        matches = []
        for candidate in candidates:
            score = float(candidate.get("hybrid_score", candidate.get("similarity", 0.0)))
            if score < min_similarity:
                continue
            candidate["similarity"] = round(score, 4)
            matches.append(candidate)
            if len(matches) >= top_k:
                break

        return matches

    def compute_semantic_deviation(
        self,
        text: str,
        baseline_category: str = "benign",
    ) -> float:
        """
        Compute semantic deviation score — how far the text deviates from
        'normal' conversation patterns toward known scam patterns.

        Returns: 0.0 (benign-like) to 1.0 (scam-like)

        This is an advanced RAG technique: instead of just retrieving similar
        documents, we compute a directional similarity score that measures
        how much closer the text is to scam patterns vs. benign patterns.
        """
        if not self._initialized or not self._embedder or not self._collection:
            raise RuntimeError("RAG engine not initialized. Call initialize() first.")

        # Get similarity to benign patterns
        benign_matches = self.query(
            text, top_k=3, category_filter="benign", min_similarity=0.0
        )
        benign_sim = max((m["similarity"] for m in benign_matches), default=0.0)

        # Get similarity to scam patterns (all non-benign)
        scam_matches = self.query(text, top_k=3, min_similarity=0.0)
        scam_matches = [m for m in scam_matches if m["category"] != "benign"]
        scam_sim = max((m["similarity"] for m in scam_matches), default=0.0)

        # Compute deviation: normalized distance from benign toward scam
        if scam_sim + benign_sim == 0:
            return 0.5  # no signal
        deviation = scam_sim / (scam_sim + benign_sim)

        return round(deviation, 4)

    def get_stats(self) -> dict:
        """Return corpus statistics."""
        if not self._collection:
            return {"status": "not_initialized"}
        return {
            "status": "ready",
            "total_documents": self._collection.count(),
            "embedding_model": config.local_models.embedding_model,
            "vector_store": "ChromaDB",
            "lexical_index": "BM25" if self._bm25 else "disabled",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"
            if self._reranker
            else "disabled",
            "retrieval": "hybrid_dense_bm25_cross_encoder",
        }


# Module-level singleton
_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """Get or create the singleton RAG engine."""
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine
