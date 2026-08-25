import json
import os
from collections import OrderedDict

import httpx
from dotenv import load_dotenv

load_dotenv()

_SERPER_API_KEY = os.getenv("SERPER_API_KEY")


def retrieve_sources(queries: list[str]) -> list[dict]:
    """Call Serper API for each query and return deduplicated search results.

    Args:
        queries: A list of search query strings.

    Returns:
        A list of dicts with keys: url, title, snippet.
        Duplicates are removed by URL.

    Raises:
        ValueError: If queries is empty.
        RuntimeError: If the Serper API call fails.
    """
    if not queries or not any(q.strip() for q in queries):
        raise ValueError("Queries list cannot be empty or whitespace-only.")

    if not _SERPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY not configured.")

    results = []

    for query in queries:
        try:
            url = "https://google.serper.com/search"
            payload = json.dumps({"q": query})
            headers = {
                "X-API-KEY": _SERPER_API_KEY,
                "Content-Type": "application/json",
            }

            response = httpx.post(url, headers=headers, data=payload, timeout=15)
            response.raise_for_status()

            data = response.json()
            organic = data.get("organic", [])

            for result in organic:
                link = result.get("link", "")
                title = result.get("title", "")
                snippet = result.get("snippet", "")

                if link:
                    results.append({
                        "url": link,
                        "title": title,
                        "snippet": snippet,
                    })

        except httpx.HTTPError as e:
            raise RuntimeError(f"Serper API HTTP error for query '{query}': {e}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse Serper API response for query '{query}': {e}")

    # Deduplicate by URL while preserving order
    seen = set()
    deduped = []
    for r in results:
        url = r["url"]
        if url not in seen:
            seen.add(url)
            deduped.append(r)

    return deduped