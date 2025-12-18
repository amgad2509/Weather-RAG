# src/utils/telemetry.py
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_ms() -> int:
    return int(time.time() * 1000)


def _truncate(value: Any, max_len: int = 320) -> Any:
    if value is None:
        return None
    try:
        s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        s = str(value)
    s = s.replace("\n", " ").strip()
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def emit(event: str, *, trace_id: str, **fields: Any) -> None:
    """
    Print one JSON line to stdout (structured logs).
    """
    payload: Dict[str, Any] = {
        "ts": _utc_ts(),
        "event": event,
        "trace_id": trace_id,
        **fields,
    }
    # Ensure one-line JSON
    line = json.dumps(payload, ensure_ascii=False, default=str)
    print(line, file=sys.stdout, flush=True)


class Stopwatch:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def ms(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000)
