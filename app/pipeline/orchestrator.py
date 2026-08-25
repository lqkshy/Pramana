from app.pipeline.claims import extract_claims
from app.pipeline.queries import generate_queries
from app.pipeline.retrieval import retrieve_sources
from app.pipeline.evidence import extract_evidence
from app.pipeline.verification import verify_claim


def run_pipeline(text: str) -> list[dict]:
    """Run the full verification pipeline on a paragraph of text.

    Steps:
    1. Extract atomic claims from the text
    2. For each claim, generate web search queries
    3. Retrieve search results for each query
    4. Extract evidence from each source URL
    5. Verify the claim against the evidence

    Returns a list of dicts, one per claim. If a step fails for an individual
    claim, that claim is skipped (not included in the return list) so one failed
    claim does not crash the whole pipeline.

    Each returned dict has keys: claim, verdict, confidence, reason, sources.
    """
    results = []

    # Step 1: Extract claims
    try:
        claims = extract_claims(text)
    except Exception:
        print("Warning: Failed to extract claims from text.")
        return results

    # Process each claim individually
    for claim in claims:
        claim_result = None

        # Step 2: Generate queries
        try:
            queries = generate_queries(claim)
        except Exception as e:
            print(f"Warning: Failed to generate queries for claim: '{claim[:60]}...': {e}")
            continue

        # Step 3: Retrieve sources
        try:
            sources_list = retrieve_sources(queries)
        except Exception as e:
            print(f"Warning: Failed to retrieve sources for claim: '{claim[:60]}...': {e}")
            continue

        # Collect evidence from all source URLs
        evidence_passages = []
        used_urls = []
        for src in sources_list:
            url = src.get("url", "")
            if not url:
                continue
            try:
                evidence = extract_evidence(url, claim)
                if evidence and evidence.strip():
                    evidence_passages.append(evidence.strip())
                    used_urls.append(url)
            except Exception as e:
                print(f"Warning: Failed to extract evidence from {url}: {e}")
                continue

        # Need at least some evidence to verify
        if not evidence_passages:
            print(f"Warning: No evidence extracted for claim: '{claim[:60]}...'")
            continue

        # Step 5: Verify claim against evidence
        try:
            claim_verdict = verify_claim(claim, evidence_passages)
            claim_result = {
                "claim": claim,
                "verdict": claim_verdict["verdict"],
                "confidence": claim_verdict["confidence"],
                "reason": claim_verdict["reason"],
                "sources": used_urls,
            }
        except Exception as e:
            print(f"Warning: Failed to verify claim: '{claim[:60]}...': {e}")
            continue

        if claim_result:
            results.append(claim_result)

    return results