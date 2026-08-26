"""
Custom exception hierarchy for Pramana.

All application-level errors should subclass PramanaError so that
FastAPI exception handlers can catch and format them consistently.
"""
__all__ = [
    "PramanaError",
    "ClaimExtractionError",
    "RetrievalError",
    "VerificationError",
    "PipelineError",
]

class PramanaError(Exception):
    """Base exception for all Pramana application errors."""

class ClaimExtractionError(PramanaError):
    """Raised when claim extraction fails."""

class RetrievalError(PramanaError):
    """Raised when evidence retrieval fails."""

class VerificationError(PramanaError):
    """Raised when claim verification fails."""

class PipelineError(PramanaError):
    """Raised for unrecoverable pipeline-level failures."""
