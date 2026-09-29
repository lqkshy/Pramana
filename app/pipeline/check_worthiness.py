"""
app/pipeline/check_worthiness.py
─────────────────────────────────
Stage 2 of the Pramana pipeline.

Binary filter: is this claim a specific, falsifiable, real-world assertion?

  KEEP  → specific, falsifiable assertion about the real world
  DROP  → opinions, predictions, rhetorical questions, vague generalities,
           normative statements, tautologies

One Groq call (task_type="fast" → Llama 3.1 8B).

The prompt is deliberately terse. The model returns a single JSON object:
  { "is_check_worthy": true/false, "reason": "one-sentence explanation" }
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from app.models.schemas import CheckWorthinessResult
from app.services.llm_client import call_llm

logger = logging.getLogger(__name__)

# ── Prompt ─────────────────────────────────────────────────────────────────────
_SYSTEM = (
    "You are a check-worthiness classifier for a fact-checking pipeline. "
    "You decide whether a claim is worth fact-checking: "
    "it must be a specific, falsifiable, real-world assertion that can be "
    "verified by looking at evidence. "
    "Return ONLY valid JSON with no extra text."
)

_PROMPT_TEMPLATE = """Classify the following claim as check-worthy or not.

Claim: {claim}

Check-worthy (return true) if the claim is:
- A specific, verifiable assertion about the real world
- Something that is either true or false (falsifiable)
- Based on past or present events, statistics, actions, or properties

Not check-worthy (return false) if the claim is:
- A personal opinion or subjective judgment ("X is the best…")
- A future prediction ("X will happen…")
- A rhetorical question
- A vague generality with no specific, checkable content
- A normative/ethical statement ("X should…" / "X ought to…")
- A tautology or definitional truth

Return ONLY this JSON object and nothing else:
{{
  "is_check_worthy": true or false,
  "reason": "one concise sentence explaining your decision"
}}"""


# ── Public API ─────────────────────────────────────────────────────────────────
async def check_worthiness(claim: str) -> CheckWorthinessResult:
    """
    Determine whether a claim is worth fact-checking.

    Args:
        claim: A single atomic claim string from Stage 1 (extractor.py).

    Returns:
        CheckWorthinessResult with is_check_worthy bool and reason string.
    """
    prompt = _PROMPT_TEMPLATE.format(claim=claim.strip())

    raw = await call_llm(
        prompt=prompt,
        task_type="fast",
        system_prompt=_SYSTEM,
        max_tokens=256,
        temperature=0.0,   # Fully deterministic — classification should not vary.
        json_mode=True,
    )

    is_worthy, reason = _parse_response(raw, claim)

    result = CheckWorthinessResult(
        claim=claim,
        is_check_worthy=is_worthy,
        reason=reason,
    )

    logger.info(
        "check_worthiness: claim=%r → %s | reason=%s",
        claim[:60],
        "KEEP" if is_worthy else "DROP",
        reason,
    )
    return result


async def filter_check_worthy(claims: list[str]) -> list[str]:
    """
    Filter a list of atomic claims, returning only those that are check-worthy.

    Processes claims sequentially to stay within Groq RPM limits.
    For parallelism on large lists, add a semaphore — but sequential is safe here.

    Args:
        claims: List of atomic claim strings from Stage 1.

    Returns:
        Filtered list containing only check-worthy claims.
    """
    worthy: list[str] = []
    for claim in claims:
        result = await check_worthiness(claim)
        if result.is_check_worthy:
            worthy.append(claim)

    logger.info(
        "filter_check_worthy: %d / %d claims passed", len(worthy), len(claims)
    )
    return worthy


# ── Response parsing ───────────────────────────────────────────────────────────
def _parse_response(raw: str, claim: str) -> tuple[bool, str]:
    """
    Parse the LLM JSON response into (is_check_worthy, reason).

    Gracefully degrades:
      - If JSON is malformed → log warning, default to True (pass-through)
        so downstream steps can still try to verify the claim.
      - If 'is_check_worthy' key is missing → default True.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from inside markdown fences or extra text.
        cleaned = _strip_markdown_fences(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(
                "check_worthiness: could not parse JSON for claim=%r. "
                "Raw response: %r. Defaulting to check_worthy=True.",
                claim[:60],
                raw[:200],
            )
            return True, "parse_error — defaulting to check_worthy"

    is_worthy: bool = bool(data.get("is_check_worthy", True))
    reason: str = str(data.get("reason", ""))
    return is_worthy, reason


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences from model output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (``` or ```json) and last line (```)
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()
    return text
