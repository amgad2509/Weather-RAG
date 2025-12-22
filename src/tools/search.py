# src/tools/internet_search.py
import os
import requests
import time
import threading
from urllib.parse import urlencode
from langchain_core.tools import tool
from src.schema import InternetSearchInput

# simple semaphore to cap concurrent network calls
_SEARCH_CONCURRENCY = int((os.getenv("SEARCH_MAX_CONCURRENCY") or 3))
_SEARCH_SEM = threading.Semaphore(_SEARCH_CONCURRENCY if _SEARCH_CONCURRENCY > 0 else 3)

@tool(args_schema=InternetSearchInput)
def internet_search(query: str, max_related: int = 6) -> str:
    """
    Lightweight web lookup via DuckDuckGo Instant Answer API.
    Best-effort for quick facts, definitions, entities.
    Includes simple retries, timeouts, and a concurrency cap.
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
    url = f"{base_url}?{urlencode(params)}"

    attempts = 3
    backoff = 1.0
    resp_data = None
    err_msg = ""

    with _SEARCH_SEM:
        for i in range(attempts):
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                resp_data = r.json()
                break
            except requests.RequestException as e:
                err_msg = f"Internet lookup failed (network/http): {e}"
            except ValueError:
                err_msg = "Internet lookup failed: response was not valid JSON."

            if i < attempts - 1:
                time.sleep(backoff)
                backoff *= 2

    if resp_data is None:
        return err_msg or "Internet lookup failed after retries."

    heading = (resp_data.get("Heading") or "").strip()
    abstract = (resp_data.get("AbstractText") or resp_data.get("Abstract") or "").strip()
    answer = (resp_data.get("Answer") or "").strip()
    definition = (resp_data.get("Definition") or "").strip()
    abstract_url = (resp_data.get("AbstractURL") or "").strip()

    related_texts = []
    related = resp_data.get("RelatedTopics") or []
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
