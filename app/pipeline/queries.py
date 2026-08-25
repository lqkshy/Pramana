import anthropic
import json
import os
from dotenv import load_dotenv

load_dotenv()

_ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def generate_queries(claim: str) -> list[str]:
    """Generate 2-3 specific web search queries for a given claim.

    Args:
        claim: A single atomic verifiable claim.

    Returns:
        A list of 2-3 search query strings.

    Raises:
        ValueError: If the claim is empty or None.
        RuntimeError: If the API call fails or returns an unexpected response.
    """
    if not claim or not claim.strip():
        raise ValueError("Claim cannot be empty or whitespace-only.")

    if not _ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not configured.")

    client = anthropic.Anthropic(api_key=_ANTHROPIC_API_KEY)

    prompt = (
        f"Generate exactly 2-3 specific web search queries (as concise strings, no numbers or bullet points) "
        f"that would help verify the following claim:\n\n"
        f"Claim: \"{claim}\"\n\n"
        "Return ONLY a JSON array of strings, e.g. [\"GPT-4 pricing page\", \"GPT-4 vs Claude price comparison\"]. "
        "No surrounding text, no headings."
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text.strip()
        queries = json.loads(content)

        if not isinstance(queries, list):
            raise RuntimeError("API did not return a JSON array.")

        if len(queries) < 2 or len(queries) > 3:
            raise RuntimeError(f"Expected 2-3 queries, got {len(queries)}.")

        for q in queries:
            if not isinstance(q, str) or not q.strip():
                raise RuntimeError("API returned invalid query items.")

        return queries

    except anthropic.APIError as e:
        raise RuntimeError(f"Anthropic API error: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse API response as JSON: {e}")