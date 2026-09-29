"""
app/pipeline/claim_matching.py
───────────────────────────────
Stage 3 of the Pramana pipeline.

Semantic deduplication using MiniLM embeddings + Supabase pgvector.

  NEW claim → embed → cosine similarity search → NO match (> 0.92) → proceed to Step 4
  DUPLICATE  → embed → cosine similarity search → match found     → return cached verdict

Model: sentence-transformers/all-MiniLM-L6-v2
  - 384-dimensional embeddings
  - Runs locally (no API calls, no cost, no rate limits)
  - ~80 MB download on first use; cached by HuggingFace thereafter

pgvector (Supabase):
  - Index: HNSW with cosine distance (created by database.init_db())
  - Threshold: 0.92 cosine similarity (tunable via CLAIM_MATCH_THRESHOLD env var)

In-memory fallback:
  - Active when DATABASE_URL is not set (offline dev, CI)
  - Uses a simple dict: { claim_text → VerificationResult }
  - Cleared on process restart (not persistent)
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

from app.models.database import find_similar_claim, save_verified_claim
from app.models.schemas import ClaimMatchResult, VerificationResult

load_dotenv()

logger = logging.getLogger(__name__)

CLAIM_MATCH_THRESHOLD: float = float(os.getenv("CLAIM_MATCH_THRESHOLD", "0.92"))

# ── In-memory fallback store ───────────────────────────────────────────────────
# List of (embedding, VerificationResult) tuples.
# Used when DATABASE_URL is not configured.
_memory_store: list[tuple[list[float], VerificationResult]] = []


# ── Embedder (lazy-loaded) ─────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_embedder():
    """
    Lazy-load the MiniLM sentence transformer.
    Cached after first call — model stays in memory for the process lifetime.

    Returns:
        SentenceTransformer instance, or None if the library is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("MiniLM embedder loaded: all-MiniLM-L6-v2 (384-dim)")
        return model
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. "
            "pip install sentence-transformers — claim matching disabled."
        )
        return None


# ── Public API ─────────────────────────────────────────────────────────────────
async def match_claim(claim: str) -> ClaimMatchResult:
    """
    Look up a claim against the verified claims cache.

    Returns a ClaimMatchResult with:
      - matched=False  → claim is new, proceed to Steps 4–8
      - matched=True   → cached_verdict contains the full VerificationResult

    Args:
        claim: Single atomic claim string (from check_worthiness filter).
    """
    embedding = _embed(claim)
    if embedding is None:
        # Embedder unavailable — pass through to avoid blocking the pipeline.
        logger.warning("match_claim: embedder unavailable, skipping match.")
        return ClaimMatchResult(claim=claim, matched=False)

    # 1. Try pgvector (Supabase)
    db_match = await find_similar_claim(
        embedding=embedding,
        similarity_threshold=CLAIM_MATCH_THRESHOLD,
    )
    if db_match is not None:
        try:
            cached = VerificationResult(**db_match["result_json"])
            cached.matched_claim_id = db_match["matched_claim_id"]
        except Exception as exc:
            logger.warning("match_claim: could not deserialise cached result: %s", exc)
            return ClaimMatchResult(claim=claim, matched=False)

        logger.info(
            "match_claim: HIT (pgvector) — claim=%r similarity=%.4f cached_id=%s",
            claim[:60],
            db_match["similarity_score"],
            db_match["matched_claim_id"],
        )
        return ClaimMatchResult(
            claim=claim,
            matched=True,
            similarity_score=db_match["similarity_score"],
            matched_claim_id=db_match["matched_claim_id"],
            cached_verdict=cached,
        )

    # 2. Try in-memory fallback (offline dev / no DATABASE_URL)
    memory_match = _search_memory(embedding)
    if memory_match is not None:
        similarity, cached = memory_match
        logger.info(
            "match_claim: HIT (in-memory) — claim=%r similarity=%.4f",
            claim[:60],
            similarity,
        )
        return ClaimMatchResult(
            claim=claim,
            matched=True,
            similarity_score=similarity,
            matched_claim_id=cached.matched_claim_id,
            cached_verdict=cached,
        )

    logger.debug("match_claim: MISS — claim=%r (new claim)", claim[:60])
    return ClaimMatchResult(claim=claim, matched=False)


async def store_claim(
    claim_id: str,
    claim: str,
    result: VerificationResult,
) -> None:
    """
    Persist a new VerificationResult after successful verification (Stage 9).

    Stores the embedding in both:
      - Supabase pgvector (if DATABASE_URL is set)
      - In-memory fallback store (always, for the current process lifetime)

    Args:
        claim_id: UUID string for this claim.
        claim:    The atomic claim text.
        result:   Full VerificationResult from Stage 8.
    """
    embedding = _embed(claim)
    if embedding is None:
        logger.warning("store_claim: embedder unavailable, skipping store.")
        return

    # Persist to Supabase.
    await save_verified_claim(
        claim_id=claim_id,
        claim_text=claim,
        result_dict=result.model_dump(),
        embedding=embedding,
    )

    # Always update in-memory store (fast path for same-session duplicates).
    _memory_store.append((embedding, result))
    logger.debug("store_claim: stored claim_id=%s in-memory and DB.", claim_id)


# ── Embedding helpers ──────────────────────────────────────────────────────────
def _embed(text: str) -> Optional[list[float]]:
    """
    Encode text → 384-dim float list using MiniLM.
    Returns None if the model is unavailable.
    """
    model = _get_embedder()
    if model is None:
        return None

    # encode() returns a numpy array — convert to plain Python list for JSON/pgvector compat.
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Dot product of two L2-normalised vectors = cosine similarity."""
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _search_memory(
    query_embedding: list[float],
) -> Optional[tuple[float, VerificationResult]]:
    """
    Linear scan of the in-memory store.
    Returns (similarity, VerificationResult) if the best match exceeds threshold.
    O(n) — only used for small offline dev sessions; pgvector handles production.
    """
    if not _memory_store:
        return None

    best_sim = -1.0
    best_result: Optional[VerificationResult] = None

    for stored_embedding, stored_result in _memory_store:
        sim = _cosine_similarity(query_embedding, stored_embedding)
        if sim > best_sim:
            best_sim = sim
            best_result = stored_result

    if best_sim >= CLAIM_MATCH_THRESHOLD and best_result is not None:
        return best_sim, best_result

    return None


def get_memory_store_size() -> int:
    """Return the number of claims currently in the in-memory fallback store."""
    return len(_memory_store)


def clear_memory_store() -> None:
    """Clear the in-memory store. Primarily for use in tests."""
    _memory_store.clear()
    logger.debug("In-memory claim store cleared.")
