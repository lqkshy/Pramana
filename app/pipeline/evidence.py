import json
import os
from bs4 import BeautifulSoup

import httpx
import anthropic
from dotenv import load_dotenv

load_dotenv()

_ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def extract_evidence(url: str, claim: str) -> str:
    """Fetch a webpage and use Claude to extract the most relevant passage supporting or refuting a claim.

    Args:
        url: The page URL to fetch.
        claim: The claim to find evidence for or against.

    Returns:
        A single string representing the most relevant evidence passage from the page.

    Raises:
        ValueError: If URL or claim is empty.
        RuntimeError: If the page fetch or API call fails.
    """
    if not url or not url.strip():
        raise ValueError("URL cannot be empty or whitespace-only.")

    if not claim or not claim.strip():
        raise ValueError("Claim cannot be empty or whitespace-only.")

    if not _ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not configured.")

    try:
        # Fetch the page
        response = httpx.get(url, timeout=15)
        response.raise_for_status()

        # Extract main text using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text content
        text = soup.get_text(separator=" ", strip=True)

        if not text:
            raise RuntimeError("No readable text found on the page.")

        # Call Claude to find the most relevant passage
        client = anthropic.Anthropic(api_key=_ANTHROPIC_API_KEY)

        prompt = (
            f"Here is a webpage text extracted from: {url}\n\n"
            f"{text[:3000]}\n\n"
            f"Claim: \"{claim}\"\n\n"
            "Find the single most relevant passage from the page that supports or refutes the claim. "
            "Return ONLY the passage text, no analysis, no introduction, no conclusion. "
            "If no relevant passage exists, return an empty string."
        )

        api_response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )

        passage = api_response.content[0].text.strip()

        return passage

    except httpx.HTTPError as e:
        raise RuntimeError(f"HTTP error fetching URL '{url}': {e}")
    except anthropic.APIError as e:
        raise RuntimeError(f"Anthropic API error: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error extracting evidence: {e}")