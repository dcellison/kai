"""
Conversation history logging and retrieval.

Provides functionality to:
1. Log every user and assistant message as JSONL (one file per day)
2. Retrieve recent messages for injection into new Claude sessions
3. Serve as the "episodic memory" layer of Kai's three-layer memory system

Log files are stored in workspace/.claude/history/ as date-stamped JSONL files
(e.g., 2026-02-11.jsonl). Each line is a JSON object with fields:
    ts       — ISO 8601 timestamp
    dir      — "user" or "assistant"
    chat_id  — Telegram chat ID
    text     — message text
    media    — optional dict with media metadata (type, filename, duration)

The inner Claude Code instance can search these files directly with grep or jq
when asked about past conversations. get_recent_history() provides a formatted
summary of the last few messages for ambient recall at session start.
"""

import json
import logging
from datetime import UTC, datetime, timedelta

from kai.config import PROJECT_ROOT

log = logging.getLogger(__name__)

# History files live inside the workspace so the inner Claude can access them
_LOG_DIR = PROJECT_ROOT / "workspace" / ".claude" / "history"

# Limits for the recent-history summary injected at session start
_MAX_RECENT_MESSAGES = 20
_MAX_CHARS_PER_MESSAGE = 500


def log_message(
    *,
    direction: str,
    chat_id: int,
    text: str,
    media: dict | None = None,
) -> None:
    """
    Append a single message record to today's JSONL chat log.

    Called from bot.py for every inbound user message and outbound assistant
    response. Each message is written immediately (not batched) so the log
    stays current even if the process crashes mid-conversation.

    Args:
        direction: "user" for inbound messages, "assistant" for Kai's responses.
        chat_id: Telegram chat ID the message belongs to.
        text: The message text content.
        media: Optional metadata dict for non-text messages (photos, voice, documents).
    """
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
    """
    Return a formatted summary of recent messages from today and yesterday.

    Injected into the first prompt of each new Claude session (in claude.py)
    to give Kai ambient awareness of recent conversations without loading the
    full history. Long messages are truncated and the total count is capped.

    Returns:
        A newline-separated string of formatted messages like
        "[2026-02-11 07:00] You: hello", or an empty string if no history exists.
    """
    now = datetime.now(UTC)
    dates = [now.strftime("%Y-%m-%d"), (now - timedelta(days=1)).strftime("%Y-%m-%d")]

    # Read yesterday first, then today, so messages are in chronological order
    messages: list[dict] = []
    for date_str in reversed(dates):
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

    # Take only the most recent N messages
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
