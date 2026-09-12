import re

import requests

DDG_ENDPOINT = "https://html.duckduckgo.com/html/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "do", "does", "did", "can", "could", "should", "would", "will", "shall",
    "of", "in", "on", "for", "to", "with", "about", "as", "by", "at", "from",
    "and", "or", "but", "not", "it", "its", "this", "that", "these", "those",
    "my", "your", "our", "their",
}


def search_keywords(question: str, max_terms: int = 12) -> str:
    """Convert a natural-language question into keyword search form
    (the CRAG paper rewrites queries into keywords to mimic real
    search-engine usage — no extra LLM call needed for this cheap part)."""

    tokens = re.findall(r"[A-Za-z0-9._-]+", question)
    keep = [
        t for t in tokens
        if len(t) > 1 and t.lower() not in _STOPWORDS
    ]
    return " ".join(keep[:max_terms])

def web_search(query: str, max_results: int = 3) -> str:
    """Search DuckDuckGo and return concatenated snippets, or '' on failure.

    Uses the lightweight HTML endpoint — no API key required.
    """

    try:
        response = requests.post(
            DDG_ENDPOINT,
            data={"q": query},
            headers=HEADERS,
            timeout=10,
        )
        response.raise_for_status()
    except Exception as e:
        print(f"[WEB] search failed: {e}")
        return ""

    results = _parse_results(response.text, max_results)

    if not results:
        print("[WEB] no results parsed")
        return ""

    return "\n\n".join(
        f"Source: {title}\n{snippet}"
        for title, snippet in results
    )


def _parse_results(html: str, limit: int) -> list[tuple[str, str]]:
    """Extract (title, snippet) pairs from the DDG HTML results page."""

    results = []

    # result blocks: <a rel="nofollow" class="result__a" ...>Title</a>
    # snippets:      <a class="result__snippet" ...>snippet</a>
    blocks = re.findall(
        r'class="result__a"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</a>',
        html,
        re.DOTALL,
    )

    for title, snippet in blocks[:limit]:
        results.append((
            _strip_tags(title),
            _strip_tags(snippet),
        ))

    return results


def _strip_tags(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text)).strip()
