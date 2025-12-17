from typing import List, Dict

def extract_sources_from_ddg_text(text: str) -> List[Dict[str, str]]:
    """
    Parses lines like:
    Source: https://...
    - something (https://...)
    """
    sources: List[Dict[str, str]] = []
    if not text:
        return sources

    for line in text.splitlines():
        line = line.strip()

        if line.lower().startswith("source:"):
            url = line.split(":", 1)[1].strip()
            if url:
                sources.append({"name": "DuckDuckGo Source", "url": url})
            continue

        if line.startswith("- ") and "(" in line and line.endswith(")"):
            try:
                name_part, url_part = line.rsplit("(", 1)
                url = url_part[:-1].strip()
                name = name_part.replace("- ", "").strip() or "Related"
                if url:
                    sources.append({"name": name, "url": url})
            except Exception:
                pass

    # Dedup by url
    seen = set()
    deduped = []
    for s in sources:
        u = s.get("url", "")
        if u and u not in seen:
            seen.add(u)
            deduped.append(s)

    return deduped
