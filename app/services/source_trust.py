"""
app/services/source_trust.py
─────────────────────────────
Stage 6 of the Pramana pipeline.

Deterministic domain trust scoring. No LLM. No API calls. Pure Python.

Trust tiers (in evaluation order — first matching rule wins):
  .gov / intergovernmental → 0.90  (government, highest institutional authority)
  .edu / academic journals  → 0.85  (university, peer-reviewed)
  Major news outlets (~30)  → 0.80  (established journalism with editorial standards)
  Wikipedia                 → 0.75  (crowd-sourced, well-cited)
  .org non-profit           → 0.60  (generally credible, varies)
  Unknown domain            → 0.40  (not zero — might be a legitimate specialist source)
  Known disinformation      → 0.10  (active misinformation — filter practically out)

Threshold: sources scoring < 0.30 are dropped BEFORE being passed to the LLM verifier.
  This saves tokens and prevents low-quality sources from influencing the verdict.

Unknown = 0.40, not 0.00:
  An unknown domain might be a legitimate specialised source (a university department blog,
  a government sub-agency, a respected industry publication). Filtering it out entirely
  would harm recall on niche scientific or policy topics.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from app.models.schemas import RetrievedSource, ScoredSource

logger = logging.getLogger(__name__)

# ── Score constants ────────────────────────────────────────────────────────────
SCORE_GOVERNMENT = 0.90
SCORE_ACADEMIC = 0.85
SCORE_MAJOR_NEWS = 0.80
SCORE_WIKIPEDIA = 0.75
SCORE_NONPROFIT = 0.60
SCORE_UNKNOWN = 0.40
SCORE_DISINFORMATION = 0.10

# Sources below this threshold are dropped before being sent to the LLM.
DROP_THRESHOLD = 0.30


# ── Major news outlets (hardcoded — ~30 outlets) ───────────────────────────────
# Criterion: global editorial standards, significant audience, fact-checking track record.
# Add more via TRUSTED_NEWS_DOMAINS env var if needed (comma-separated).
_MAJOR_NEWS_DOMAINS: frozenset[str] = frozenset(
    {
        # International wires
        "reuters.com",
        "apnews.com",
        "afp.com",
        # US national
        "nytimes.com",
        "washingtonpost.com",
        "wsj.com",
        "usatoday.com",
        "npr.org",
        "pbs.org",
        "theatlantic.com",
        "politico.com",
        # UK national
        "bbc.com",
        "bbc.co.uk",
        "theguardian.com",
        "economist.com",
        "ft.com",
        "thetimes.co.uk",
        "telegraph.co.uk",
        "independent.co.uk",
        # Broadcast
        "cnn.com",
        "nbcnews.com",
        "abcnews.go.com",
        "cbsnews.com",
        # Science / health
        "sciencedaily.com",
        "newscientist.com",
        "statnews.com",
        "healthline.com",
        # Fact-checkers (full trust — this is literally what they do)
        "snopes.com",
        "factcheck.org",
        "politifact.com",
        "fullfact.org",
        "africacheck.org",
    }
)

# ── Academic / journal domains ─────────────────────────────────────────────────
# These supplement the .edu TLD rule.
_ACADEMIC_DOMAINS: frozenset[str] = frozenset(
    {
        "nature.com",
        "science.org",
        "thelancet.com",
        "nejm.org",
        "bmj.com",
        "cell.com",
        "pnas.org",
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "scholar.google.com",
        "semanticscholar.org",
        "arxiv.org",
        "biorxiv.org",
        "medrxiv.org",
        "jstor.org",
        "springer.com",
        "wiley.com",
        "tandfonline.com",
        "oxfordjournals.org",
        "cambridge.org",
        "ssrn.com",
    }
)

# ── Intergovernmental / quasi-governmental domains ─────────────────────────────
_INTERGOVERNMENTAL_DOMAINS: frozenset[str] = frozenset(
    {
        "who.int",
        "un.org",
        "worldbank.org",
        "imf.org",
        "oecd.org",
        "europa.eu",
        "ec.europa.eu",
        "unicef.org",
        "unhcr.org",
        "wto.org",
        "nato.int",
    }
)

# ── Known disinformation domains ───────────────────────────────────────────────
# Keep this list small and conservative — only well-documented cases.
# Sources scoring 0.10 are not dropped (threshold is 0.30) but are heavily
# discounted when evidence_strength is computed.
_DISINFO_DOMAINS: frozenset[str] = frozenset(
    {
        "infowars.com",
        "naturalnews.com",
        "beforeitsnews.com",
        "worldnewsdailyreport.com",
        "empirenews.net",
        "theonion.com",          # Satire — not malicious, but not factual
        "babylonbee.com",        # Satire
        "clickhole.com",         # Satire
        "nationalreport.net",
        "abcnews.com.co",        # Impersonation of ABC News
        "ksat12.com.co",
    }
)


# ── Public API ─────────────────────────────────────────────────────────────────
def score_source(source: RetrievedSource) -> ScoredSource:
    """
    Assign a trust score to a single source based on its domain.

    Args:
        source: A RetrievedSource from Stage 5 (retrieval.py).

    Returns:
        ScoredSource with quality_score and trust_tier populated.
        Returns score=0.0 if the URL cannot be parsed.
    """
    url = source.url.strip()
    domain, score, tier = _score_url(url)

    return ScoredSource(
        url=url,
        title=source.title,
        snippet=source.snippet,
        raw_text=source.raw_text,
        quality_score=score,
        trust_tier=tier,
    )


def score_and_filter(
    sources: list[RetrievedSource],
    threshold: float = DROP_THRESHOLD,
) -> list[ScoredSource]:
    """
    Score all sources and drop those below the threshold.

    Args:
        sources:   List of RetrievedSource from Stage 5.
        threshold: Minimum quality_score to keep a source (default 0.30).

    Returns:
        Filtered list of ScoredSource, sorted by quality_score descending.
    """
    scored: list[ScoredSource] = [score_source(s) for s in sources]
    filtered: list[ScoredSource] = [s for s in scored if s.quality_score >= threshold]
    filtered.sort(key=lambda s: s.quality_score, reverse=True)

    dropped = len(scored) - len(filtered)
    if dropped > 0:
        logger.debug(
            "score_and_filter: dropped %d / %d sources below threshold %.2f",
            dropped,
            len(scored),
            threshold,
        )

    return filtered


def score_url(url: str) -> tuple[float, str]:
    """
    Public helper: score a URL string directly.

    Returns:
        (quality_score, trust_tier) tuple.
    """
    _, score, tier = _score_url(url)
    return score, tier


# ── Scoring logic ──────────────────────────────────────────────────────────────
def _score_url(url: str) -> tuple[str, float, str]:
    """
    Core scoring function. Returns (domain, quality_score, trust_tier).

    Rules are evaluated in priority order — first match wins.
    This means .gov domains inside .org paths still score as government.
    """
    # Synthetic URLs created by retrieval.py in benchmark (pre_retrieved) mode.
    # They have no domain, so score them as "unknown" instead of 0.0 -- otherwise
    # every benchmark passage is dropped by the 0.30 threshold and the eval
    # runs with zero evidence.
    if url.startswith("pre_retrieved://"):
        return "pre_retrieved", SCORE_UNKNOWN, "pre_retrieved"

    domain = _extract_domain(url)
    if not domain:
        return "", 0.0, "invalid"

    # Rule 1: Known disinformation sites.
    if _matches_domain(domain, _DISINFO_DOMAINS):
        return domain, SCORE_DISINFORMATION, "disinformation"

    # Rule 2: Intergovernmental organisations (.int, .un.org, worldbank.org, etc.)
    if _matches_domain(domain, _INTERGOVERNMENTAL_DOMAINS):
        return domain, SCORE_GOVERNMENT, "intergovernmental"

    # Rule 3: Government TLDs — .gov fires before .org check.
    if _is_government_domain(domain):
        return domain, SCORE_GOVERNMENT, "government"

    # Rule 4: Academic — known journal domains or .edu TLD.
    if _matches_domain(domain, _ACADEMIC_DOMAINS) or _is_academic_domain(domain):
        return domain, SCORE_ACADEMIC, "academic"

    # Rule 5: Major news outlets (hardcoded list).
    if _matches_domain(domain, _MAJOR_NEWS_DOMAINS):
        return domain, SCORE_MAJOR_NEWS, "major_news"

    # Rule 6: Wikipedia.
    if "wikipedia.org" in domain:
        return domain, SCORE_WIKIPEDIA, "wikipedia"

    # Rule 7: .org non-profit (only after all specific .org domains above).
    if domain.endswith(".org"):
        return domain, SCORE_NONPROFIT, "nonprofit_org"

    # Rule 8: Everything else — unknown.
    return domain, SCORE_UNKNOWN, "unknown"


def _extract_domain(url: str) -> str:
    """
    Extract the netloc from a URL and return the bare domain (no www. prefix).

    Returns empty string if URL is malformed or has no netloc
    (e.g. synthetic pre_retrieved:// URLs from eval mode).
    """
    if url.startswith("pre_retrieved://"):
        return ""  # Synthetic eval URL — no domain to score. Will score as 0.0/invalid.

    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        netloc = parsed.netloc or parsed.path
        # Strip port numbers (e.g. example.com:8080 → example.com).
        netloc = netloc.split(":")[0]
        # Normalise: lowercase, remove www. prefix.
        netloc = netloc.lower().strip()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def _matches_domain(domain: str, domain_set: frozenset[str]) -> bool:
    """
    Check if domain matches any entry in domain_set.
    Handles subdomains: pubmed.ncbi.nlm.nih.gov matches ncbi.nlm.nih.gov.
    """
    if domain in domain_set:
        return True
    # Check all suffixes (e.g. sub.domain.tld → domain.tld, tld)
    parts = domain.split(".")
    for i in range(1, len(parts)):
        suffix = ".".join(parts[i:])
        if suffix in domain_set:
            return True
    return False


def _is_government_domain(domain: str) -> bool:
    """
    True for .gov TLD and common government domain patterns.

    Examples:
      cdc.gov → True
      data.gov.uk → True (UK government)
      health.govt.nz → True (NZ government)
      census.gov → True
    """
    if domain.endswith(".gov"):
        return True
    # Country-specific government patterns: .gov.uk, .gov.au, .gov.in, .govt.nz, etc.
    if re.search(r"\.gov\.[a-z]{2}$", domain):
        return True
    if re.search(r"\.govt\.[a-z]{2}$", domain):
        return True
    return False


def _is_academic_domain(domain: str) -> bool:
    """
    True for .edu TLD and academic sub-TLDs in other countries (.ac.uk, .edu.au, etc.)
    """
    if domain.endswith(".edu"):
        return True
    if re.search(r"\.ac\.[a-z]{2}$", domain):
        return True  # UK: .ac.uk, India: .ac.in, etc.
    if re.search(r"\.edu\.[a-z]{2}$", domain):
        return True  # Australia: .edu.au, etc.
    return False


# ── Convenience: score a list of URLs directly ────────────────────────────────
def score_urls(urls: list[str]) -> dict[str, float]:
    """
    Score a list of URL strings.

    Returns:
        Dict mapping URL → quality_score.
        Useful for looking up quality scores after the verification LLM call
        returns its list of cited_source_urls.
    """
    return {url: _score_url(url)[1] for url in urls}
