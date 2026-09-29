"""
app/main.py
────────────
FastAPI application entry point for Pramana.

Endpoints:
  GET  /health  → { status, provider, dev_mode, use_live_search }
  POST /verify  → { input_text, claims_found, results: [VerificationResult] }

Pipeline stages wired here (Weeks 1–5):
  Week 1: /health ✓  /verify (mocked) ✓
  Week 2: check_worthiness + claim_matching + queries wired in ✓
  Week 3: retrieval + source_trust wired in ✓
  Week 4: evidence + reranking ✓
  Week 5: verification + store → full pipeline end-to-end ✓

USE_LIVE_SEARCH is forced to True for the FastAPI server.
Eval scripts override it to False before calling the pipeline directly.
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Force live search for the server ──────────────────────────────────────────
# Eval scripts will set USE_LIVE_SEARCH=false before importing pipeline modules.
# The FastAPI server always uses live Tavily search for real user submissions.
os.environ.setdefault("USE_LIVE_SEARCH", "true")

# Protects your Groq/Tavily quota: one request can verify at most this many claims.
MAX_CLAIMS_PER_REQUEST: int = int(os.getenv("MAX_CLAIMS_PER_REQUEST", "8"))


# ── Deferred imports (after env setup) ────────────────────────────────────────
from app.models.database import init_db
from app.models.schemas import PipelineResult, VerificationResult, VerdictLabel
from app.pipeline.check_worthiness import filter_check_worthy
from app.pipeline.claim_matching import match_claim, store_claim
from app.pipeline.claims.extractor import extract_claims
from app.pipeline.evidence import extract_evidence
from app.pipeline.queries import generate_queries
from app.pipeline.retrieval import retrieve_evidence
from app.pipeline.verification import verify_claim
from app.services.llm_client import DEV_MODE, LLM_PROVIDER
from app.services.source_trust import score_and_filter


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hook. Init DB tables on startup."""
    logger.info("Pramana starting up — provider=%s dev_mode=%s", LLM_PROVIDER, DEV_MODE)
    await init_db()
    yield
    logger.info("Pramana shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Pramana",
    description="Research-grade AI fact-checking pipeline. $0. Open source.",
    version="3.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────
class VerifyRequest(BaseModel):
    text: str
    """Input text to fact-check. Can be an article, speech excerpt, or social post."""


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Health check. Confirms the server is up and shows active configuration."""
    return {
        "status": "ok",
        "provider": LLM_PROVIDER,
        "dev_mode": DEV_MODE,
        "use_live_search": os.getenv("USE_LIVE_SEARCH", "true"),
        "version": "3.1.0",
    }


@app.post("/verify", response_model=PipelineResult)
async def verify(request: VerifyRequest):
    """
    Full fact-checking pipeline.

    Stage 1 extract → 2 check-worthy → 3 cache match → 4 queries → 5 retrieval
    → 6 source scoring → 7 evidence (scrape + rerank + refine) → 8 verdict → 9 store.
    """
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Input text cannot be empty.")

    # ── Stage 1: Extract claims ────────────────────────────────────────────────
    extraction = await extract_claims(request.text)
    if not extraction.claims:
        return PipelineResult(
            input_text=request.text,
            claims_found=0,
            results=[],
        )

    # De-duplicate while keeping order, and cap the work per request.
    claim_texts = list(dict.fromkeys(c.text for c in extraction.claims))

    # ── Stage 2: Check-worthiness filter ──────────────────────────────────────
    worthy_claims = (await filter_check_worthy(claim_texts))[:MAX_CLAIMS_PER_REQUEST]
    if not worthy_claims:
        return PipelineResult(
            input_text=request.text,
            claims_found=0,
            results=[],
        )

    results: list[VerificationResult] = []
    cached_count = 0

    for claim in worthy_claims:
        try:
            # ── Stage 3: Claim matching ────────────────────────────────────────
            match = await match_claim(claim)
            if match.matched and match.cached_verdict is not None:
                results.append(match.cached_verdict)
                cached_count += 1
                continue

            # ── Stage 4: Query generation ──────────────────────────────────────
            query_result = await generate_queries(claim)

            # ── Stage 5: Retrieval ─────────────────────────────────────────────
            retrieval_result = await retrieve_evidence(
                claim=claim,
                queries=query_result.queries,
            )

            # ── Stage 6: Source quality scoring ───────────────────────────────
            scored_sources = score_and_filter(retrieval_result.sources)

            # ── Stage 7: Evidence extraction (scrape → rerank → refine) ───────
            passages = await extract_evidence(claim, scored_sources)

            # ── Stage 8: Verification + chain-of-thought ──────────────────────
            result = await verify_claim(claim, passages)

            # ── Stage 9: Store (only real verdicts — never cache "no evidence") ─
            if result.verdict != VerdictLabel.INSUFFICIENT_EVIDENCE:
                await store_claim(str(uuid.uuid4()), claim, result)

            results.append(result)

        except Exception as exc:  # One bad claim must not kill the whole request.
            logger.exception("Pipeline failed for claim=%r", claim[:80])
            results.append(
                VerificationResult(
                    claim=claim,
                    verdict=VerdictLabel.INSUFFICIENT_EVIDENCE,
                    confidence=0.0,
                    explanation=f"Verification failed: {type(exc).__name__}: {exc}",
                )
            )

    return PipelineResult(
        input_text=request.text,
        claims_found=len(results),
        results=results,
        cached_count=cached_count,
    )
