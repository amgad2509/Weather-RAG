# src/api/tracing_logger.py
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


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
