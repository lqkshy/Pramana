"""
app/pipeline/claims/extractor.py
──────────────────────────────────
Stage 1 of the Pramana pipeline.

ONE structured Groq call that returns all three extraction outputs as a single JSON object:
  {
    "selected_sentences": [...],   // verifiable sentences from the input
    "disambiguated":      [...],   // pronouns/context resolved
    "decomposed":         [...]    // atomic claims (one falsifiable fact each)
  }

This is Fix 1 from the build plan: previously 4 separate Ollama calls (20–40s latency).
Now 1 Groq call. All three outputs are produced in a single request.

Model: Groq Llama 3.1 8B (task_type="fast") — 14,400 req/day free tier.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from app.models.schemas import AtomicClaim, ExtractionResult
from app.services.llm_client import call_llm

logger = logging.getLogger(__name__)

# ── Prompt ─────────────────────────────────────────────────────────────────────
_SYSTEM = (
    "You are a claim extraction engine for a fact-checking pipeline. "
    "You identify verifiable factual claims in text, resolve ambiguities, "
    "and decompose compound claims into atomic sub-claims. "
    "Return ONLY valid JSON. No preamble. No markdown fences."
)

_PROMPT_TEMPLATE = """Extract verifiable claims from the following text.

Text:
\"\"\"{text}\"\"\"

Perform three tasks and return them in a single JSON object:

1. selected_sentences: Select sentences that contain specific, verifiable factual claims.
   Skip opinions, predictions, rhetorical questions, and vague generalities.

2. disambiguated: For each selected sentence, rewrite it so it is self-contained.
   Replace all pronouns (he, she, they, it, this, that) with the specific entity they refer to.
   Replace "the company", "the country", "the study" etc. with the actual named entity.
   If a sentence is already clear with no pronouns or ambiguous references, copy it unchanged.

3. decomposed: Break each disambiguated sentence into atomic claims.
   Each atomic claim = exactly ONE falsifiable fact.
   Split compound sentences with "and", "but", "while", "also" into separate claims.
   Each claim must be self-contained and make sense on its own.

Return ONLY this JSON structure:
{{
  "selected_sentences": [
    "sentence 1 exactly as it appears in the text",
    "sentence 2 exactly as it appears in the text"
  ],
  "disambiguated": [
    "sentence 1 with all pronouns and references resolved",
    "sentence 2 with all pronouns and references resolved"
  ],
  "decomposed": [
    "atomic claim 1 (one falsifiable fact)",
    "atomic claim 2 (one falsifiable fact)",
    "atomic claim 3 (one falsifiable fact)"
  ]
}}

If the text contains NO verifiable factual claims, return:
{{"selected_sentences": [], "disambiguated": [], "decomposed": []}}"""


# ── Public API ─────────────────────────────────────────────────────────────────
async def extract_claims(text: str) -> ExtractionResult:
    """
    Stage 1: Extract, disambiguate, and decompose claims from input text.

    Args:
        text: Raw input text from the user (article, speech, social media post, etc.)

    Returns:
        ExtractionResult with selected_sentences, disambiguated text, and atomic claims.
    """
    if not text.strip():
        logger.warning("extract_claims: empty input text.")
        return ExtractionResult(raw_input=text)

    prompt = _PROMPT_TEMPLATE.format(text=text.strip())

    raw = await call_llm(
        prompt=prompt,
        task_type="fast",
        system_prompt=_SYSTEM,
        max_tokens=2048,
        temperature=0.0,
        json_mode=True,
    )

    result = _parse_response(raw, text)
    logger.info(
        "extract_claims: found %d atomic claims from %d chars of text",
        len(result.claims),
        len(text),
    )
    return result


# ── Response parsing ───────────────────────────────────────────────────────────
def _parse_response(raw: str, original_text: str) -> ExtractionResult:
    cleaned = _strip_fences(raw)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(
            "extract_claims: JSON parse failed. Raw: %r", raw[:300]
        )
        return ExtractionResult(raw_input=original_text)

    selected: list[str] = data.get("selected_sentences", [])
    disambiguated: list[str] = data.get("disambiguated", [])
    decomposed: list[str] = data.get("decomposed", [])

    # Build AtomicClaim objects, mapping each decomposed claim back to its
    # most likely original sentence (by index alignment when counts match).
    claims: list[AtomicClaim] = []
    for i, claim_text in enumerate(decomposed):
        if not claim_text.strip():
            continue
        original = selected[i] if i < len(selected) else (selected[-1] if selected else "")
        disambig = disambiguated[i] if i < len(disambiguated) else claim_text
        claims.append(
            AtomicClaim(
                text=claim_text.strip(),
                original_sentence=original,
                disambiguated=disambig.strip(),
            )
        )

    return ExtractionResult(
        selected_sentences=selected,
        claims=claims,
        raw_input=original_text,
    )


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()
    return text
