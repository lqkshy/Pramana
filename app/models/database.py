import os
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Text,
    Float,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

DATABASE_URL = "sqlite:///claim_verifier.db"
engine = create_engine(DATABASE_URL, echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Runs(Base):
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True, index=True)
    input_text = Column(String, nullable=False)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class Claims(Base):
    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=False)
    claim_text = Column(String, nullable=False)
    verdict = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    reason = Column(Text, nullable=False)

    run = relationship("Runs", back_populates="claims")


Runs.claims = relationship("Claims", order_by=Claims.id, back_populates="run")


class Sources(Base):
    __tablename__ = "sources"

    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(Integer, ForeignKey("claims.id"), nullable=False)
    url = Column(String, nullable=False)
    snippet = Column(Text, nullable=False)


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency to get a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_run(input_text: str, results: list[dict]) -> int:
    """Save a full pipeline run and return the run_id.

    Args:
        input_text: The original text that was processed.
        results: The list of claim result dicts from run_pipeline.

    Returns:
        The integer run_id of the newly created run record.
    """
    db = SessionLocal()
    try:
        run = Runs(input_text=input_text)
        db.add(run)
        db.commit()
        db.refresh(run)

        for result in results:
            claim = Claims(
                run_id=run.id,
                claim_text=result.get("claim", ""),
                verdict=result.get("verdict", ""),
                confidence=result.get("confidence", 0.0),
                reason=result.get("reason", ""),
            )
            db.add(claim)
            db.commit()
            db.refresh(claim)

            sources = result.get("sources", [])
            for url in sources:
                source = Sources(claim_id=claim.id, url=url, snippet="")
                db.add(source)
            db.commit()

        return run.id
    except Exception as e:
        db.rollback()
        raise RuntimeError(f"Failed to save pipeline run: {e}") from e
    finally:
        db.close()