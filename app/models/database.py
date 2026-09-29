"""
app/models/database.py
──────────────────────
SQLAlchemy async ORM layer for Pramana.

Tables:
  verified_claims   — stores full VerificationResult JSON + claim embedding (pgvector)
  source_cache      — optional: cache scraped page text by URL to avoid re-scraping

pgvector is used for Step 3 (claim_matching.py):
  Each verified claim stores a 384-dim MiniLM embedding.
  New claims run cosine similarity search — match > 0.92 returns the cached verdict.

Connection: Supabase free tier (postgresql://...).
  pgvector extension must be enabled in Supabase dashboard → Extensions → pgvector.

Week 1: in-memory dict fallback (SUPABASE_URL not set → uses SQLite-backed in-memory mode).
Week 2: pgvector fully enabled.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL: str = os.getenv("DATABASE_URL", "")

# ── SQLAlchemy async setup ─────────────────────────────────────────────────────
try:
    from sqlalchemy import (
        Column,
        DateTime,
        Float,
        String,
        Text,
        text,
    )
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )
    from sqlalchemy.orm import DeclarativeBase

    _SQLALCHEMY_AVAILABLE = True
except ImportError:
    _SQLALCHEMY_AVAILABLE = False
    logger.warning("sqlalchemy / asyncpg not installed. DB layer disabled.")

# pgvector type — installed separately
try:
    from pgvector.sqlalchemy import Vector  # type: ignore[import]
    _PGVECTOR_AVAILABLE = True
except ImportError:
    _PGVECTOR_AVAILABLE = False
    logger.warning(
        "pgvector not installed (pip install pgvector). "
        "Claim matching will use in-memory fallback."
    )

EMBEDDING_DIM = 384  # sentence-transformers/all-MiniLM-L6-v2 output dimension


# ── ORM models ─────────────────────────────────────────────────────────────────
if _SQLALCHEMY_AVAILABLE:

    class Base(DeclarativeBase):
        pass

    class VerifiedClaim(Base):
        """
        Stores one verified claim per row.
        embedding: 384-dim float32 vector for pgvector cosine similarity search.
        result_json: full VerificationResult serialised as JSONB.
        """

        __tablename__ = "verified_claims"

        id: str = Column(String(64), primary_key=True)
        claim_text: str = Column(Text, nullable=False)
        verdict: str = Column(String(32), nullable=False)
        confidence: float = Column(Float, nullable=False)
        evidence_strength: float = Column(Float, default=0.0)
        result_json: dict = Column(JSONB, nullable=False)
        created_at: datetime = Column(
            DateTime(timezone=True),
            default=lambda: datetime.now(timezone.utc),
            nullable=False,
        )

        if _PGVECTOR_AVAILABLE:
            embedding = Column(Vector(EMBEDDING_DIM), nullable=True)
        else:
            embedding = Column(Text, nullable=True)  # stub — stores nothing useful

    class SourceCache(Base):
        """Caches scraped page text by URL to avoid redundant HTTP calls."""

        __tablename__ = "source_cache"

        url: str = Column(String(2048), primary_key=True)
        title: str = Column(Text, default="")
        raw_text: str = Column(Text, default="")
        quality_score: float = Column(Float, default=0.0)
        fetched_at: datetime = Column(
            DateTime(timezone=True),
            default=lambda: datetime.now(timezone.utc),
            nullable=False,
        )


# ── Engine / session factory ───────────────────────────────────────────────────
_engine = None
_AsyncSessionLocal = None


def _get_engine():
    global _engine, _AsyncSessionLocal
    if _engine is not None:
        return _engine

    if not DATABASE_URL:
        logger.warning(
            "DATABASE_URL not set — database persistence disabled. "
            "Claim matching will use in-memory dict."
        )
        return None

    if not _SQLALCHEMY_AVAILABLE:
        return None

    # Supabase uses postgresql:// — convert to asyncpg dialect.
    url = DATABASE_URL
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)

    _engine = create_async_engine(
        url,
        pool_size=5,
        max_overflow=2,
        pool_timeout=30,
        pool_recycle=1800,
        echo=False,
    )
    _AsyncSessionLocal = async_sessionmaker(
        _engine, class_=AsyncSession, expire_on_commit=False
    )
    logger.info("Database engine created. pgvector=%s", _PGVECTOR_AVAILABLE)
    return _engine


@asynccontextmanager
async def get_session() -> AsyncGenerator[Optional["AsyncSession"], None]:
    """
    Async context manager yielding a database session.
    Yields None if DATABASE_URL is not configured (safe for offline dev).

    Usage:
        async with get_session() as session:
            if session is None:
                # fallback to in-memory
                ...
    """
    engine = _get_engine()
    if engine is None or _AsyncSessionLocal is None:
        yield None
        return

    async with _AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── Schema initialisation ──────────────────────────────────────────────────────
async def init_db() -> None:
    """
    Create all tables and enable pgvector extension.
    Call once at application startup (lifespan hook in main.py).
    """
    engine = _get_engine()
    if engine is None:
        logger.info("init_db: no DATABASE_URL — skipping schema creation.")
        return

    if not _SQLALCHEMY_AVAILABLE:
        return

    async with engine.begin() as conn:
        # Enable pgvector extension (requires Supabase Pro or self-hosted with vector).
        # On Supabase free tier: enable manually in Dashboard → Extensions → pgvector.
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            logger.info("pgvector extension ensured.")
        except Exception as exc:
            logger.warning(
                "Could not auto-enable pgvector extension: %s. "
                "Enable it manually in the Supabase dashboard.",
                exc,
            )

        # Create tables.
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created / verified.")

    # Create pgvector HNSW index for fast ANN search on claim embeddings.
    # HNSW is faster than IVFFlat for small-to-medium datasets (< 1M rows).
    await _ensure_vector_index()


async def _ensure_vector_index() -> None:
    """Create cosine-similarity HNSW index on verified_claims.embedding if it does not exist."""
    if not _PGVECTOR_AVAILABLE:
        return

    engine = _get_engine()
    if engine is None:
        return

    index_sql = """
    CREATE INDEX IF NOT EXISTS idx_verified_claims_embedding
    ON verified_claims
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
    """
    async with engine.begin() as conn:
        try:
            await conn.execute(text(index_sql))
            logger.info("HNSW index on verified_claims.embedding ensured.")
        except Exception as exc:
            logger.warning("Could not create HNSW index: %s", exc)


# ── CRUD helpers ───────────────────────────────────────────────────────────────
async def save_verified_claim(
    claim_id: str,
    claim_text: str,
    result_dict: dict,
    embedding: Optional[list[float]] = None,
) -> None:
    """
    Persist a VerificationResult to the verified_claims table.

    Args:
        claim_id:    UUID string for this claim.
        claim_text:  The original atomic claim text.
        result_dict: VerificationResult.model_dump() output.
        embedding:   384-dim MiniLM embedding as a list[float]. None = not stored.
    """
    async with get_session() as session:
        if session is None:
            logger.debug("save_verified_claim: no DB session, skipping persist.")
            return

        if not _SQLALCHEMY_AVAILABLE:
            return

        row = VerifiedClaim(
            id=claim_id,
            claim_text=claim_text,
            verdict=result_dict.get("verdict", ""),
            confidence=result_dict.get("confidence", 0.0),
            evidence_strength=result_dict.get("evidence_strength", 0.0),
            result_json=result_dict,
            embedding=embedding,
        )
        session.add(row)
    logger.debug("Saved claim %s to database.", claim_id)


async def find_similar_claim(
    embedding: list[float],
    similarity_threshold: float = 0.92,
    limit: int = 1,
) -> Optional[dict]:
    """
    Query verified_claims for the most similar claim using pgvector cosine similarity.

    Returns the result_json dict if similarity ≥ threshold, else None.

    Args:
        embedding:            384-dim MiniLM embedding for the new claim.
        similarity_threshold: Minimum cosine similarity to be considered a match (default 0.92).
        limit:                Max rows to inspect (default 1 — we only need the top hit).
    """
    if not _PGVECTOR_AVAILABLE:
        logger.debug("find_similar_claim: pgvector unavailable, returning None.")
        return None

    async with get_session() as session:
        if session is None:
            return None

        # pgvector cosine similarity: 1 - (embedding <=> query_vector)
        # <=> operator returns cosine distance, so similarity = 1 - distance.
        sql = text(
            """
            SELECT id, claim_text, result_json,
                   1 - (embedding <=> CAST(:vec AS vector)) AS similarity
            FROM verified_claims
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :limit
            """
        )

        vec_str = f"[{','.join(str(v) for v in embedding)}]"
        result = await session.execute(sql, {"vec": vec_str, "limit": limit})
        row = result.fetchone()

        if row is None:
            return None

        similarity: float = float(row.similarity)
        logger.debug(
            "Closest claim match: id=%s similarity=%.4f threshold=%.2f",
            row.id,
            similarity,
            similarity_threshold,
        )

        if similarity >= similarity_threshold:
            return {
                "matched_claim_id": row.id,
                "similarity_score": similarity,
                "result_json": row.result_json,
            }

        return None
