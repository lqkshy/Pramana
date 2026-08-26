from sentence_transformers import CrossEncoder

import os
import logging

logger = logging.getLogger(__name__)

_model = None
_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_model():
    global _model
    if _model is None:
        try:
            _model = CrossEncoder(_model_name)
        except Exception as e:
            logger.warning(f"Failed to load reranking model {_model_name}: {e}")
            _model = None
    return _model


def rerank_evidence(claim: str, passages: list[str], top_k: int = 3) -> list[str]:
    """Rank (claim, passage) pairs for relevance and return top_k passages.

    Args:
        claim: A single atomic verifiable claim.
        passages: List of text passages to rank against the claim.
        top_k: Number of top passages to return (default 3).

    Returns:
        List of the top_k most relevant passage strings.
        If the model fails to load, returns the original passages as-is.
    """
    if not passages or not any(p.strip() for p in passages):
        return []

    model = _get_model()

    if model is None:
        # Model failed to load; return passages as-is (filtered to top_k)
        return passages[:top_k]

    # Build feature pairs of (claim, passage)
    features = [(claim.strip(), passage.strip()) for passage in passages if passage.strip()]

    if not features:
        return []

    try:
        # Score each pair; higher score = more relevant
        scores = model.predict(features)

        # Pair scores with passages
        scored = list(zip(scores, features))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Extract top_k passages
        top_passages = [passage for _, passage in scored[:top_k]]

        return top_passages

    except Exception as e:
        logger.warning(f"Reranking prediction failed: {e}")
        # Fall back to returning first top_k passages
        return passages[:top_k]