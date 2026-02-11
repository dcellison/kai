import json
import logging
from datetime import UTC, datetime, timedelta

from kai.config import PROJECT_ROOT

log = logging.getLogger(__name__)

_LOG_DIR = PROJECT_ROOT / "workspace" / ".claude" / "history"

_MAX_RECENT_MESSAGES = 20
_MAX_CHARS_PER_MESSAGE = 500


def log_message(
    *,
    direction: str,
    chat_id: int,
    text: str,
    media: dict | None = None,
) -> None:
    """Append a message record to today's JSONL chat log."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
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


def get_recent_history() -> str:
    """Return a formatted summary of recent messages from today and yesterday."""
    now = datetime.now(UTC)
    dates = [now.strftime("%Y-%m-%d"), (now - timedelta(days=1)).strftime("%Y-%m-%d")]

    messages: list[dict] = []
    for date_str in reversed(dates):  # yesterday first, then today
        path = _LOG_DIR / f"{date_str}.jsonl"
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    messages.append(json.loads(line))
        except (OSError, json.JSONDecodeError):
            log.exception("Failed to read history file %s", path)

    if not messages:
        return ""

    # Take the most recent N messages
    messages = messages[-_MAX_RECENT_MESSAGES:]

    lines = []
    for msg in messages:
        ts = msg.get("ts", "")[:16].replace("T", " ")  # "2026-02-11 07:00"
        speaker = "You" if msg.get("dir") == "user" else "Kai"
        text = msg.get("text", "")
        if len(text) > _MAX_CHARS_PER_MESSAGE:
            text = text[:_MAX_CHARS_PER_MESSAGE] + "..."
        lines.append(f"[{ts}] {speaker}: {text}")

    return "\n".join(lines)
