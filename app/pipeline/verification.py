import anthropic
import os

from dotenv import load_dotenv

load_dotenv()

_ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def verify_claim(claim: str, evidence_passages: list[str]) -> dict:
    """Verify a claim against evidence passages using Claude.

    Args:
        claim: A single atomic verifiable claim.
        evidence_passages: List of text passages extracted from web pages.

    Returns:
        A dict with keys: verdict, confidence, reason.
        verdict is "SUPPORTED", "CONTRADICTED", or "UNVERIFIED".
        confidence is a float between 0.0 and 1.0.
        reason is a one-sentence explanation.

    Raises:
        ValueError: If claim is empty or evidence_passages is empty/None.
        RuntimeError: If the API call fails or returns an unexpected response.
    """
    if not claim or not claim.strip():
        raise ValueError("Claim cannot be empty or whitespace-only.")

    if not evidence_passages or not any(p.strip() for p in evidence_passages):
        raise ValueError("Evidence passages cannot be empty or whitespace-only.")

    if not _ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not configured.")

    client = anthropic.Anthropic(api_key=_ANTHROPIC_API_KEY)

    # Build the evidence context from passages
    evidence_context = "\n\n".join(
        f"Evidence {i+1}: {passage.strip()}"
        for i, passage in enumerate(evidence_passages)
        if passage.strip()
    )

    prompt = (
    f"Claim: \"{claim.strip()}\"\n\n"
    f"Evidence passages:\n{evidence_context}\n\n"
    "Based on the evidence passages above, determine if the claim is "
    "either 'SUPPORTED', 'CONTRADICTED', or 'UNVERIFIED'.\n\n"
    "Respond with a JSON object with exactly these keys:\n"
    "  - verdict: one of 'SUPPORTED', 'CONTRADICTED', or 'UNVERIFIED'\n"
    "  - confidence: a float between 0.0 and 1.0\n"
    "  - reason: one concise sentence explaining the verdict\n\n"
    "Do not include any other text, analysis, or reasoning. Only return the JSON."
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text.strip()

        # Parse the JSON response
        result = json.loads(content)

        if "verdict" not in result or "confidence" not in result or "reason" not in result:
            raise RuntimeError("API response missing required keys (verdict, confidence, reason).")

        valid_verdicts = {"SUPPORTED", "CONTRADICTED", "UNVERIFIED"}
        if result["verdict"] not in valid_verdicts:
            raise RuntimeError(f"Invalid verdict: {result['verdict']}. Must be one of {valid_verdicts}.")

        if not isinstance(result["confidence"], (int, float)) or not (0.0 <= result["confidence"] <= 1.0):
            raise RuntimeError(f"Invalid confidence: {result['confidence']}. Must be a float between 0.0 and 1.0.")

        if not isinstance(result["reason"], str) or not result["reason"].strip():
            raise RuntimeError("Invalid reason: must be a non-empty string.")

        # Ensure reason is one sentence (trim to first period)
        sentence = result["reason"].split(".")[0] + "."
        if len(sentence.strip()) < 5:
            raise RuntimeError("Reason sentence too short.")

        return {
            "verdict": result["verdict"],
            "confidence": float(result["confidence"]),
            "reason": sentence,
        }

    except anthropic.APIError as e:
        raise RuntimeError(f"Anthropic API error: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse API response as JSON: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during claim verification: {e}")