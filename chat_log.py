from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_LOG_DIR = Path(__file__).parent / "workspace" / "chat_history"


def log_message(
    *,
    direction: str,
    chat_id: int,
    text: str,
    media: dict | None = None,
) -> None:
    """Append a message record to today's JSONL chat log."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    record = {
        "ts": now.isoformat(),
        "dir": direction,
        "chat_id": chat_id,
        "text": text,
        "media": media,
    }
    filepath = _LOG_DIR / f"{now.strftime('%Y-%m-%d')}.jsonl"
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        log.exception("Failed to write chat log")
