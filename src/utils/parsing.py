import re
from typing import Optional, Tuple

_REASONING_RE = re.compile(r"(?is)<reasoning>\s*(.*?)\s*</reasoning>")

def split_reasoning(content: str) -> Tuple[Optional[str], str]:
    if not content:
        return None, ""
    m = _REASONING_RE.search(content)
    if not m:
        return None, content.strip()
    reasoning = (m.group(1) or "").strip()
    answer = _REASONING_RE.sub("", content, count=1).strip()
    return reasoning, answer

def strip_reasoning_during_stream(raw: str) -> str:
    if not raw:
        return ""
    lower = raw.lower()
    if "<reasoning>" in lower and "</reasoning>" not in lower:
        return raw.split("<reasoning>", 1)[0].strip()
    _, answer = split_reasoning(raw)
    return answer
