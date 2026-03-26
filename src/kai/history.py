"""
Conversation history logging and retrieval.

Provides functionality to:
1. Log every user and assistant message as JSONL (one file per day)
2. Retrieve recent messages for injection into new Claude sessions
3. Serve as the "episodic memory" layer of Kai's three-layer memory system

Log files are stored in per-user subdirectories under DATA_DIR/history/
(e.g., DATA_DIR/history/<chat_id>/2026-02-11.jsonl). Each line is a JSON
object with fields:
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
from datetime import UTC, datetime

from kai.config import DATA_DIR

log = logging.getLogger(__name__)

# History files live under DATA_DIR (alongside the database and logs) so they
# survive installs. In production this is /var/lib/kai/history/, in dev it's
# PROJECT_ROOT/history/. Intentionally NOT updated when workspace switches -
# all conversation history stays in one canonical location.
_LOG_DIR = DATA_DIR / "history"

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
    # Per-user subdirectory: DATA_DIR/history/<chat_id>/YYYY-MM-DD.jsonl
    # Separates users on disk so grep/jq searches are naturally scoped
    # and one user's history can be managed independently.
    user_dir = _LOG_DIR / str(chat_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    record = {
        "ts": now.isoformat(),
        "dir": direction,
        "chat_id": chat_id,
        "text": text,
        "media": media,
    }
    filepath = user_dir / f"{now.strftime('%Y-%m-%d')}.jsonl"
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        log.exception("Failed to write chat log")


def get_recent_history(chat_id: int | None = None) -> str:
    """
    Return a formatted summary of recent messages, scanning back as needed.

    Scans date-stamped JSONL files from newest to oldest, collecting up to
    _MAX_RECENT_MESSAGES messages. This ensures Kai has ambient recall even
    after gaps of several days without conversation.

    When chat_id is provided, scans only that user's subdirectory
    (DATA_DIR/history/<chat_id>/), plus any legacy flat files from before
    per-user isolation was added. When None, scans all subdirectories.

    Injected into the first prompt of each new Claude session (in claude.py)
    to give Kai ambient awareness of recent conversations without loading the
    full history. Long messages are truncated and the total count is capped.

    Args:
        chat_id: When provided, only include messages from this chat.
            When None, include all messages (backward-compatible for
            single-user deployments).

    Returns:
        A newline-separated string of formatted messages like
        "[2026-02-11 07:00] You: hello", or an empty string if no history exists.
    """
    if not _LOG_DIR.exists():
        return ""

    if chat_id is not None:
        # Scan only this user's subdirectory for per-user isolation.
        user_dir = _LOG_DIR / str(chat_id)
        files = sorted(user_dir.glob("*.jsonl"), reverse=True) if user_dir.exists() else []
        # Also include legacy flat files (pre-per-user migration) - the
        # chat_id filter in the parsing loop handles mixed records correctly.
        # These age out naturally as new writes go to per-user directories.
        legacy = [f for f in _LOG_DIR.glob("*.jsonl") if f.is_file()]
        files = sorted(set(files) | set(legacy), key=lambda p: p.name, reverse=True)
    else:
        # Scan all user directories (backward compat / admin view).
        # Sort by filename (date) not full path, so files from different
        # user directories on the same date interleave chronologically.
        files = sorted(_LOG_DIR.rglob("*.jsonl"), key=lambda p: p.name, reverse=True)

    if not files:
        return ""

    # Read files newest-first, collecting messages until we have enough.
    # We read entire files since individual files are small (one day of chat),
    # then take the last N from the combined pool.
    messages: list[dict] = []
    for path in files:
        file_messages: list[dict] = []
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            log.exception("Failed to read history file %s", path)
            continue
        for line in raw.splitlines():
            if line.strip():
                try:
                    record = json.loads(line)
                    # Skip messages from other users when filtering.
                    # Records without a chat_id field predate Phase 2
                    # and are included for all users (see docstring note).
                    record_chat_id = record.get("chat_id")
                    if chat_id is not None and record_chat_id is not None and record_chat_id != chat_id:
                        continue
                    file_messages.append(record)
                except json.JSONDecodeError:
                    # Skip individual bad lines rather than discarding the whole file
                    log.debug("Skipping malformed JSON line in %s: %s", path.name, line[:100])

        # Prepend this file's messages (older days go before newer days)
        messages = file_messages + messages

        # Stop scanning once we have more than enough
        if len(messages) >= _MAX_RECENT_MESSAGES:
            break

    if not messages:
        return ""

    # Take only the most recent N messages (chronological order preserved)
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
