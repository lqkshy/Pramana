"""
app/pipeline/reranking.py
──────────────────────────
Stage 7 (part 1) of the Pramana pipeline.

Chunk source text into ~100-200 word passages, score every (claim, passage)
pair with a LOCAL cross-encoder, and return the top-k passages.

No LLM calls. No API calls. Model: cross-encoder/ms-marco-MiniLM-L-6-v2
(~90 MB, downloaded once by HuggingFace, then cached).

If sentence-transformers is not installed, falls back to a simple word-overlap
score so the pipeline (and unit tests) still run.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from functools import lru_cache
from typing import Optional

from app.models.schemas import EvidencePassage, ScoredSource

logger = logging.getLogger(__name__)

RERANK_MODEL: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHUNK_MAX_WORDS: int = 150
CHUNK_MIN_WORDS: int = 8
MAX_CHUNKS_PER_SOURCE: int = 30   # bounds CPU cost on very long pages

_STOPWORDS = frozenset(
    "a an the of in on at to for from by with and or but is are was were be been "
    "it its this that these those as than then which who whom".split()
)


# ── Model loading ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Lazy-load the cross-encoder. Returns None if unavailable."""
    try:
        from sentence_transformers import CrossEncoder  # type: ignore[import]

        model = CrossEncoder(RERANK_MODEL)
        logger.info("Cross-encoder loaded: %s", RERANK_MODEL)
        return model
    except Exception as exc:  # ImportError, download failure, etc.
        logger.warning("Cross-encoder unavailable (%s). Using word-overlap fallback.", exc)
        return None


# ── Chunking ───────────────────────────────────────────────────────────────────
def chunk_text(text: str, max_words: int = CHUNK_MAX_WORDS) -> list[str]:
    """Split text into sentence-aligned chunks of at most `max_words` words."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    count = 0

    def flush() -> None:
        nonlocal current, count
        if current:
            chunks.append(" ".join(current))
        current, count = [], 0

    for sentence in sentences:
        words = sentence.split()
        if len(words) > max_words:  # pathological sentence → hard split
            flush()
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i : i + max_words]))
            continue
        if count + len(words) > max_words:
            flush()
        current.append(sentence)
        count += len(words)
    flush()

    return [c for c in chunks if len(c.split()) >= CHUNK_MIN_WORDS]


# ── Scoring ────────────────────────────────────────────────────────────────────
def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS}


def _overlap_score(claim: str, passage: str) -> float:
    """Fallback relevance: fraction of claim tokens found in the passage."""
    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return 0.0
    return len(claim_tokens & _tokens(passage)) / len(claim_tokens)


def _score_pairs(claim: str, passages: list[str]) -> list[float]:
    model = _get_cross_encoder()
    if model is None:
        return [_overlap_score(claim, p) for p in passages]
    scores = model.predict([(claim, p) for p in passages])
    return [float(s) for s in scores]


# ── Public API ─────────────────────────────────────────────────────────────────
async def rerank_passages(
    claim: str,
    sources: list[ScoredSource],
    top_k: int = 3,
    max_per_source: int = 2,
) -> list[EvidencePassage]:
    """
    Return the top-k most claim-relevant passages across all sources.

    max_per_source keeps one long page from filling all k slots. That matters
    because evidence_strength rewards citing MULTIPLE distinct sources.
    """
    candidates: list[tuple[str, ScoredSource]] = []
    for source in sources:
        body = source.raw_text or source.snippet
        for chunk in chunk_text(body)[:MAX_CHUNKS_PER_SOURCE]:
            candidates.append((chunk, source))

    if not candidates:
        logger.warning("rerank_passages: no candidate passages for claim=%r", claim[:60])
        return []

    # Cross-encoder inference is blocking CPU work → keep it off the event loop.
    scores = await asyncio.to_thread(_score_pairs, claim, [c for c, _ in candidates])

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    selected: list[EvidencePassage] = []
    per_source: dict[str, int] = {}
    for score, (chunk, source) in ranked:
        if per_source.get(source.url, 0) >= max_per_source:
            continue
        per_source[source.url] = per_source.get(source.url, 0) + 1
        selected.append(
            EvidencePassage(
                text=chunk,
                source_url=source.url,
                source_quality=source.quality_score,
                rerank_score=score,
            )
        )
        if len(selected) >= top_k:
            break

    logger.info(
        "rerank_passages: %d candidates → top %d (best score %.3f)",
        len(candidates),
        len(selected),
        selected[0].rerank_score if selected else 0.0,
    )
    return selected
