from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Set


def parse_sources_from_internet_output(text: str) -> List[Dict[str, str]]:
    """
    Extract URLs from internet_search output where lines look like:
      - 'Source: https://...'
      - '- title (https://...)'
    """
    sources: List[Dict[str, str]] = []
    seen: Set[str] = set()

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lower = line.lower()
        if lower.startswith("source:"):
            url = line.split(":", 1)[1].strip()
            if url and url not in seen:
                sources.append({"name": url, "url": url})
                seen.add(url)
            continue

        if line.startswith("- ") and "(" in line and line.endswith(")"):
            try:
                title_part, url_part = line[2:].rsplit("(", 1)
                url = url_part[:-1].strip()
                title = title_part.strip() or url
                if url and url not in seen:
                    sources.append({"name": title, "url": url})
                    seen.add(url)
            except ValueError:
                continue

    return sources


def parse_sources_from_retriever_output(obj: Any, limit: int = 5) -> List[Dict[str, str]]:
    """
    Best-effort extraction of URLs from retriever tool output (Documents/dicts/strings).
    Limit to first `limit` unique URLs to avoid flooding.
    """
    sources: List[Dict[str, str]] = []
    seen: Set[str] = set()

    def add(url: str, name: str = None):
        if not url or url in seen:
            return
        if len(sources) >= limit:
            return
        sources.append({"name": name or url, "url": url})
        seen.add(url)

    if obj is None or limit <= 0:
        return sources

    if isinstance(obj, str):
        for m in re.findall(r"https?://[^\s)]+", obj):
            add(m)
        try:
            parsed = json.loads(obj)
            sources.extend(parse_sources_from_retriever_output(parsed, limit=limit - len(sources)))
        except Exception:
            pass
        return sources

    if isinstance(obj, (list, tuple)):
        for item in obj:
            if len(sources) >= limit:
                break
            sources.extend(parse_sources_from_retriever_output(item, limit=limit - len(sources)))
        return sources

    if isinstance(obj, dict):
        url = obj.get("url") or obj.get("source") or obj.get("link") or obj.get("path")
        name = obj.get("title") or obj.get("name") or obj.get("file_name") or obj.get("filename") or url
        if url:
            add(str(url), str(name) if name else None)
        return sources

    meta = getattr(obj, "metadata", None)
    if isinstance(meta, dict):
        url = meta.get("source") or meta.get("url") or meta.get("link") or meta.get("path")
        name = meta.get("title") or meta.get("file_name") or meta.get("filename") or url
        if url:
            add(str(url), str(name) if name else None)
        if len(sources) >= limit:
            return sources

    try:
        rep = str(obj)
        for m in re.findall(r"https?://[^\s)]+", rep):
            add(m)
    except Exception:
        pass

    return sources


__all__ = ["parse_sources_from_internet_output", "parse_sources_from_retriever_output"]
