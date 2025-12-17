import requests
from urllib.parse import urlencode
from langchain_core.tools import tool


@tool
def internet_search(query: str, max_related: int = 6) -> str:
    """
    Lightweight web lookup via DuckDuckGo Instant Answer API.
    Returns a compact summary + a few source links when available.
    """
    if not query or not query.strip():
        return "Error: empty query."

    base_url = "https://api.duckduckgo.com/"
    params = {
        "q": query.strip(),
        "format": "json",
        "no_html": 1,
        "no_redirect": 1,
        "skip_disambig": 1,
    }

    try:
        url = f"{base_url}?{urlencode(params)}"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        return f"Internet lookup failed (network/http): {e}"
    except ValueError:
        return "Internet lookup failed: response was not valid JSON."

    heading = (data.get("Heading") or "").strip()
    abstract = (data.get("AbstractText") or data.get("Abstract") or "").strip()
    answer = (data.get("Answer") or "").strip()
    definition = (data.get("Definition") or "").strip()
    abstract_url = (data.get("AbstractURL") or "").strip()

    related_texts = []
    related = data.get("RelatedTopics") or []
    for item in related:
        if isinstance(item, dict) and "Topics" in item and isinstance(item["Topics"], list):
            for t in item["Topics"]:
                txt = (t.get("Text") or "").strip()
                u = (t.get("FirstURL") or "").strip()
                if txt:
                    related_texts.append((txt, u))
        elif isinstance(item, dict):
            txt = (item.get("Text") or "").strip()
            u = (item.get("FirstURL") or "").strip()
            if txt:
                related_texts.append((txt, u))

    lines = []
    title = heading if heading else query.strip()
    lines.append(f"Title: {title}")

    if answer:
        lines.append(f"Answer: {answer}")
    if definition:
        lines.append(f"Definition: {definition}")
    if abstract:
        lines.append(f"Abstract: {abstract}")
    if abstract_url:
        lines.append(f"Source: {abstract_url}")

    if related_texts:
        lines.append("Related:")
        for txt, u in related_texts[:max_related]:
            lines.append(f"- {txt}" + (f" ({u})" if u else ""))

    if len(lines) <= 1:
        return "No instant-answer content found for this query. Try a more specific query."

    return "\n".join(lines)
