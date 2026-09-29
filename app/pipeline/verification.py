"""
app/pipeline/verification.py
─────────────────────────────
Stage 8 of the Pramana pipeline: verdict + chain-of-thought explanation.

ONE LLM call. task_type="reasoning" → llm_client.py routes it by DEV_MODE:
    DEV_MODE=true  → Llama 3.1 8B  (14,400 req/day)
    DEV_MODE=false → Llama 3.3 70B (1,000 req/day)

Design choices:
  * The model cites sources by NUMBER ([1], [2]), never by URL, so it cannot
    invent URLs. We map numbers back to real URLs ourselves.
  * evidence_strength is computed HERE with the locked v3.1 formula:
        avg(quality of cited sources) × min(1.0, cited_count / 3)
    Two passages from the same URL count as ONE source.
  * A SUPPORTED / CONTRADICTED verdict with no valid citation is downgraded to
    INSUFFICIENT_EVIDENCE — a verdict must be backed by something.
"""

from __future__ import annotations

import json
import logging

from app.models.schemas import (
    EvidencePassage,
    VerdictLabel,
    VerificationResult,
    compute_evidence_strength,
)
from app.services.llm_client import call_llm

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a rigorous fact-checking analyst. Judge the claim using ONLY the "
    "numbered evidence provided. Never use outside knowledge. "
    "Return ONLY valid JSON. No preamble. No markdown fences."
)

_PROMPT = """Claim: {claim}

Evidence (each has a source trust score from 0 to 1; higher = more reliable):
{evidence}

Choose exactly ONE verdict:
- SUPPORTED: the evidence clearly confirms the claim.
- CONTRADICTED: the evidence clearly shows the claim is false.
- CONFLICTING_EVIDENCE: credible evidence points in BOTH directions.
- INSUFFICIENT_EVIDENCE: the evidence is unrelated, vague, or does not settle the claim.

Think step by step: what does each relevant source say, do they agree, and does
it match every part of the claim (numbers, dates, names)? Prefer higher-trust sources.

Write the explanation for a non-technical reader and refer to sources as [1], [2].
Return ONLY this JSON:
{{
  "verdict": "SUPPORTED | CONTRADICTED | CONFLICTING_EVIDENCE | INSUFFICIENT_EVIDENCE",
  "confidence": 0.0,
  "explanation": "step-by-step reasoning citing [1], [2]...",
  "cited_sources": [1, 2]
}}"""

# Tolerate the model (or AVeriTeC-style labels) using other wording.
_LABEL_MAP = {
    "SUPPORTED": VerdictLabel.SUPPORTED,
    "SUPPORTS": VerdictLabel.SUPPORTED,
    "TRUE": VerdictLabel.SUPPORTED,
    "CONTRADICTED": VerdictLabel.CONTRADICTED,
    "REFUTED": VerdictLabel.CONTRADICTED,
    "REFUTES": VerdictLabel.CONTRADICTED,
    "FALSE": VerdictLabel.CONTRADICTED,
    "CONFLICTING_EVIDENCE": VerdictLabel.CONFLICTING_EVIDENCE,
    "CONFLICTING": VerdictLabel.CONFLICTING_EVIDENCE,
    "INSUFFICIENT_EVIDENCE": VerdictLabel.INSUFFICIENT_EVIDENCE,
    "NOT_ENOUGH_EVIDENCE": VerdictLabel.INSUFFICIENT_EVIDENCE,
    "NEE": VerdictLabel.INSUFFICIENT_EVIDENCE,
}


async def verify_claim(claim: str, passages: list[EvidencePassage]) -> VerificationResult:
    """Produce the final VerificationResult for one atomic claim."""
    if not passages:
        return _insufficient(claim, "No usable evidence passages were found for this claim.")

    evidence_block = "\n\n".join(
        f"[{i}] (trust {p.source_quality:.2f}) {p.text}" for i, p in enumerate(passages, 1)
    )
    raw = await call_llm(
        prompt=_PROMPT.format(claim=claim, evidence=evidence_block),
        task_type="reasoning",
        system_prompt=_SYSTEM,
        max_tokens=900,
        temperature=0.0,
        json_mode=True,
    )
    return build_result(claim, passages, raw)


def build_result(claim: str, passages: list[EvidencePassage], raw: str) -> VerificationResult:
    """Parse the LLM JSON, validate citations, compute evidence_strength."""
    try:
        data = json.loads(_strip_fences(raw))
        if not isinstance(data, dict):
            raise ValueError("top-level JSON is not an object")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("verify_claim: unparseable LLM output (%s): %r", exc, raw[:200])
        return _insufficient(claim, "The verifier returned an unreadable response.", passages)

    verdict = _LABEL_MAP.get(
        str(data.get("verdict", "")).strip().upper().replace(" ", "_"),
        VerdictLabel.INSUFFICIENT_EVIDENCE,
    )
    explanation = str(data.get("explanation", "")).strip() or "No explanation was provided."
    try:
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0

    # Map cited numbers → real URLs (unique, valid, in order).
    cited_urls: list[str] = []
    cited_qualities: list[float] = []
    quality_by_url: dict[str, float] = {}
    for p in passages:
        quality_by_url[p.source_url] = max(quality_by_url.get(p.source_url, 0.0), p.source_quality)

    cited = data.get("cited_sources", [])
    for n in cited if isinstance(cited, list) else []:
        try:
            idx = int(n) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(passages):
            url = passages[idx].source_url
            if url not in cited_urls:
                cited_urls.append(url)
                cited_qualities.append(quality_by_url[url])

    # A verdict needs backing.
    if verdict in (VerdictLabel.SUPPORTED, VerdictLabel.CONTRADICTED) and not cited_urls:
        explanation += " (Downgraded: the verdict cited no valid source.)"
        verdict = VerdictLabel.INSUFFICIENT_EVIDENCE
        confidence = min(confidence, 0.3)

    return VerificationResult(
        claim=claim,
        verdict=verdict,
        confidence=confidence,
        explanation=explanation,
        evidence_passages=passages,
        cited_source_urls=cited_urls,
        cited_source_qualities=cited_qualities,
        evidence_strength=compute_evidence_strength(cited_qualities),
    )


def _insufficient(
    claim: str, why: str, passages: list[EvidencePassage] | None = None
) -> VerificationResult:
    return VerificationResult(
        claim=claim,
        verdict=VerdictLabel.INSUFFICIENT_EVIDENCE,
        confidence=0.0,
        explanation=why,
        evidence_passages=passages or [],
        cited_source_urls=[],
        cited_source_qualities=[],
        evidence_strength=0.0,
    )


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()
    return text
