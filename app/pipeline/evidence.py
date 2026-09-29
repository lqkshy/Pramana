"""
app/pipeline/evidence.py
─────────────────────────
Stage 7 of the Pramana pipeline: evidence extraction.

  7a  httpx fetch pages          (live mode only; skipped if raw_text/snippet exists offline)
  7b  BeautifulSoup → clean text
  7c  chunk + cross-encoder rerank → top-3 passages   (reranking.py, local)
  7e  ONE Groq 8B call → pull the precise sentence(s) out of each passage

Exactly 1 LLM call per claim (plan Fix 6). If that call fails or hallucinates,
we keep the original passages — the pipeline never breaks because of Stage 7e.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from app.models.schemas import EvidencePassage, ScoredSource
from app.pipeline.reranking import rerank_passages
from app.services.llm_client import call_llm

logger = logging.getLogger(__name__)

MAX_SOURCES_TO_SCRAPE = 6
FETCH_TIMEOUT = 10.0
MAX_PAGE_BYTES = 1_500_000
MAX_PAGE_CHARS = 20_000
_FETCH_SEMAPHORE = asyncio.Semaphore(4)
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PramanaBot/1.0; +research fact-checker)"}


# ── 7a/7b: fetch + extract ─────────────────────────────────────────────────────
async def enrich_with_page_text(sources: list[ScoredSource]) -> list[ScoredSource]:
    """Fetch full page text for live sources that don't have raw_text yet."""
    return list(await asyncio.gather(*[_enrich_one(s) for s in sources]))


async def _enrich_one(source: ScoredSource) -> ScoredSource:
    if source.raw_text or not source.url.startswith("http"):
        return source  # benchmark item, or already scraped
    async with _FETCH_SEMAPHORE:
        text = await _fetch_text(source.url)
    if not text:
        return source  # fall back to Tavily snippet
    return source.model_copy(update={"raw_text": text})


async def _fetch_text(url: str) -> str:
    try:
        import httpx  # type: ignore[import]

        async with httpx.AsyncClient(
            timeout=FETCH_TIMEOUT, follow_redirects=True, headers=_HEADERS
        ) as client:
            resp = await client.get(url)
        ctype = resp.headers.get("content-type", "")
        if resp.status_code != 200 or "html" not in ctype.lower():
            return ""  # skip PDFs, images, errors
        return html_to_text(resp.content[:MAX_PAGE_BYTES].decode(resp.encoding or "utf-8", "ignore"))
    except Exception as exc:
        logger.debug("fetch failed for %s: %s", url, exc)
        return ""


def html_to_text(html: str) -> str:
    """Extract readable article text from raw HTML (paragraph-first)."""
    try:
        from bs4 import BeautifulSoup  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("pip install beautifulsoup4") from exc

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    paragraphs = [p for p in paragraphs if len(p.split()) >= 8]
    text = " ".join(paragraphs) if paragraphs else soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()[:MAX_PAGE_CHARS]


# ── 7e: refinement (1 Groq call) ───────────────────────────────────────────────
_SYSTEM = (
    "You extract evidence for a fact-checking pipeline. "
    "Return ONLY valid JSON. No preamble. No markdown fences."
)

_PROMPT = """Claim: {claim}

Below are numbered passages. For EACH passage, copy the 1-2 sentences that are most
relevant to verifying the claim (evidence for OR against). Rules:
- Copy the sentences VERBATIM from the passage. Do not paraphrase.
- Keep numbers, dates, names and negations exactly as written.
- If nothing in the passage is relevant, return an empty string for it.

{passages}

Return ONLY this JSON:
{{"evidence": [{{"id": 1, "sentence": "..."}}, {{"id": 2, "sentence": "..."}}]}}"""


async def refine_passages(claim: str, passages: list[EvidencePassage]) -> list[EvidencePassage]:
    if not passages:
        return passages

    block = "\n\n".join(f"[{i}] {p.text}" for i, p in enumerate(passages, 1))
    try:
        raw = await call_llm(
            prompt=_PROMPT.format(claim=claim, passages=block),
            task_type="fast",
            system_prompt=_SYSTEM,
            max_tokens=700,
            temperature=0.0,
            json_mode=True,
        )
        data = json.loads(_strip_fences(raw))
        items = data.get("evidence", [])
    except Exception as exc:
        logger.warning("refine_passages failed (%s) — keeping original passages.", exc)
        return passages

    refined = list(passages)
    for item in items if isinstance(items, list) else []:
        try:
            idx = int(item.get("id")) - 1
            sentence = str(item.get("sentence", "")).strip()
        except (TypeError, ValueError, AttributeError):
            continue
        if not (0 <= idx < len(passages)) or not sentence:
            continue
        # Anti-hallucination: accept only if it really appears in the passage.
        if _norm(sentence) in _norm(passages[idx].text):
            refined[idx] = passages[idx].model_copy(update={"text": sentence})
    return refined


# ── Public API ─────────────────────────────────────────────────────────────────
async def extract_evidence(
    claim: str,
    sources: list[ScoredSource],
    top_k: int = 3,
    refine: bool = True,
) -> list[EvidencePassage]:
    """Stage 7 entry point: sources (already scored) → top-k evidence passages."""
    sources = sources[:MAX_SOURCES_TO_SCRAPE]  # already sorted best-first by Stage 6
    sources = await enrich_with_page_text(sources)
    passages = await rerank_passages(claim, sources, top_k=top_k)
    if refine:
        passages = await refine_passages(claim, passages)
    return passages


# ── Helpers ────────────────────────────────────────────────────────────────────
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')).strip()


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()
    return text
