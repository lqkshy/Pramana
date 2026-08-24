from dotenv import load_dotenv
import os
import anthropic
import json

load_dotenv()

_key = os.getenv("ANTHROPIC_API_KEY")


def extract_claims(paragraph):
    """Extract atomic, independently verifiable factual claims from text."""
    if not paragraph or not paragraph.strip():
        raise ValueError("Input cannot be empty.")

    if not _key:
        raise RuntimeError("ANTHROPIC_API_KEY not set.")

    client = anthropic.Anthropic(api_key=_key)

    prompt = (
        "Extract atomic, independently verifiable factual claims from the following text. "
        "Each claim should be a single checkable statement. Return ONLY a JSON array of strings, "
        "with no surrounding text, no numbers, no bullet points.\n\n"
        f"Text: \"{paragraph}\""
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text.strip()
        claims = json.loads(content)

        if not isinstance(claims, list):
            raise RuntimeError("API did not return a JSON array.")

        for claim in claims:
            if not isinstance(claim, str) or not claim.strip():
                raise RuntimeError("API returned invalid claim items.")

        return claims

    except anthropic.APIError as e:
        raise RuntimeError(f"Anthropic API error: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse API response: {e}")