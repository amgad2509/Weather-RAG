# src/api/tracing_logger.py
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.utils.source_parsers import (
    parse_sources_from_internet_output,
    parse_sources_from_retriever_output,
)
from src.utils.telemetry import emit

class JsonLineFormatter(logging.Formatter):
    """Formats log records as single-line JSON (JSONL)."""

    def format(self, record: logging.LogRecord) -> str:
        now = datetime.now(timezone.utc).isoformat()

        # لو انت بتعمل logger.info({...}) بدكت، نخليه هو الـ payload
        if isinstance(record.msg, dict):
            payload: Dict[str, Any] = dict(record.msg)
            payload.setdefault("ts", now)
            payload.setdefault("level", record.levelname)
            payload.setdefault("logger", record.name)
        else:
            payload = {
                "ts": now,
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }

        # لو في Exception
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def setup_tracing_logger(
    log_path: str = "tracing.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Creates a dedicated tracing logger that writes JSONL to:
      - stdout (normal logs)
      - tracing.log (file)
    """
    logger = logging.getLogger("tracing")
    logger.setLevel(level)
    logger.propagate = False  # مهم عشان مايتكررش مع root logger

    # منع تكرار الهاندلرز في حالة uvicorn --reload
    logger.handlers.clear()

    formatter = JsonLineFormatter()

    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(formatter)

    # File handler
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)

    return logger


# -------------------------
# Helpers (shared across routes)
# -------------------------
def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


# -------------------------
# Tracing -> file (JSONL)
# -------------------------
TRACING_LOG_PATH = os.getenv("TRACING_LOG_PATH", "tracing.log")

_tracing_file_logger = logging.getLogger("tracing.file")
_tracing_file_logger.setLevel(logging.INFO)
_tracing_file_logger.propagate = False

if not _tracing_file_logger.handlers:
    log_file = Path(TRACING_LOG_PATH)
    if str(log_file.parent) not in ("", "."):
        log_file.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    _tracing_file_logger.addHandler(fh)


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit_trace(event: str, **fields: Any) -> None:
    """
    Emits tracing twice:
      1) stdout/normal logs via existing `emit(...)`
      2) writes JSONL line to tracing.log (or TRACING_LOG_PATH)
    """
    payload = {"ts": _now_iso_utc(), "event": event, **fields}

    try:
        emit(event, **fields)
    except Exception:
        try:
            logging.getLogger(__name__).info(_json_dumps(payload))
        except Exception:
            pass

    try:
        _tracing_file_logger.info(_json_dumps(payload))
    except Exception:
        pass


def sse(payload: Dict[str, Any]) -> str:
    return f"data: {_json_dumps(payload)}\n\n"


__all__ = [
    "JsonLineFormatter",
    "setup_tracing_logger",
    "emit_trace",
    "parse_sources_from_internet_output",
    "parse_sources_from_retriever_output",
    "sse",
]
