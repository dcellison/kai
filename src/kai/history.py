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
import re
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

# Synthetic placeholder markers written by the failure-path log_message
# calls in bot.py - `/stop` aborts ("[stopped by user]"), empty results
# ("[no response]"), and error paths ("[error: <type/message>]"). These
# are formatted-text-only entries with no real assistant content;
# `get_recent_pairs` skips them so a windowed extraction payload does
# not feed a "botched exchange" prior turn into the episode classifier
# (issue #392). Matched via `re.fullmatch` (which implicitly anchors
# at both ends, so the regex body has no leading `^`/trailing `$`)
# against the FULL line so a legitimate message that happens to
# contain "[error: ...]" as quoted prose - e.g. a user asking "what
# does [error: foo] mean?" - is NOT mis-skipped. The DOTALL flag
# lets `.` match newlines so a multi-line error string (a Python
# traceback rendered into the placeholder) still matches; without
# DOTALL, `.+` would stop at the first `\n` and the closing `]`
# would fail to match. `error: .+` rather than `error: [^\]]+`
# because the inner content can itself contain `]` characters
# (tracebacks frequently do); the trailing `]` requirement still
# forces the final `]` to be the last character of the entire line.
_SYNTHETIC_ASSISTANT_MARKERS: re.Pattern[str] = re.compile(
    r"\[(stopped by user|no response|error: .+)\]",
    re.DOTALL,
)


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


def get_recent_pairs(chat_id: int, n: int) -> list[tuple[str, str]]:
    """
    Return the most recent N (user_text, assistant_text) pairs from the
    user's JSONL chat history, in CHRONOLOGICAL order (oldest first).

    Used by the stage-1 memory extractor (issue #392) to feed prior
    turns to the episode classifier as PRIOR CONTEXT background. The
    classifier needs structured pairs (not the formatted text that
    `get_recent_history` produces) and stricter filtering (synthetic
    failure-path markers and empty-text records would distort the
    closure heuristic).

    The implementation walks JSONL files newest-first internally to
    bound the work when history is long, then reverses the collected
    records to deliver oldest-first so callers can render PRIOR USER 1,
    2, 3 ... naturally in the prompt.

    Pairing semantics:
    - A user record followed (eventually) by an assistant record forms
      one pair. Records between the two are skipped per the rules below.
    - Multiple user records before a single assistant record collapse:
      only the LAST user record (the one immediately preceding the
      assistant) is paired. Earlier orphan user messages are dropped.
      This matches the "user can send multiple messages before getting
      a reply" pattern in Telegram.
    - An assistant record with no prior pending user record is dropped
      (orphan; legacy or restored-from-backup data).

    Records skipped before pairing:
    - Records with empty `text` (after strip). Image-only and voice-only
      messages with no transcribed text would otherwise produce
      meaningless prior turns.
    - Assistant records whose text matches `_SYNTHETIC_ASSISTANT_MARKERS`
      (the failure-path placeholders written by bot.py's `/stop`,
      empty-result, and error paths). Pairing one of these into a
      windowed payload would feed the classifier a "botched exchange"
      prior context that distorts the closure heuristic.
    - Records with no `chat_id` field (pre-Phase-2 legacy records).
      Greenfield helper sets the deliberate convention: skip rather
      than risk mixing chats. `get_recent_history` includes such
      records for backward compat; this helper diverges because the
      classifier path is sensitive to mis-attributed prior turns in
      a way the display path is not.

    Both halves of the current in-flight exchange are already written
    to JSONL by the time extraction runs (the user message is logged
    when it arrives in `bot.py`; the assistant message is logged
    immediately before `_ingest_memory` is dispatched as a background
    task). So the most recent pair returned by this helper IS the
    current exchange. Callers MUST drop the most recent pair before
    passing prior turns to the classifier - see `_ingest_memory` in
    `bot.py` for the +1/drop pattern.

    Args:
        chat_id: Telegram chat ID. Records without this exact id are
            filtered out (alongside no-chat_id legacy records).
        n: Maximum number of pairs to return. Non-positive values
            return an empty list (used by callers that want to disable
            windowing without a special-case branch).

    Returns:
        Up to N pairs, oldest-first. Returns fewer than N pairs when
        history is short, the user is new, or filters drop rows.
        Never raises; OS errors are logged and the affected file is
        skipped.
    """
    if n <= 0:
        return []
    if not _LOG_DIR.exists():
        return []

    # Per-user subdirectory; ignore legacy flat files entirely. The
    # classifier path's strictness on chat_id provenance means
    # admin-restored backups in user_dir are also filtered out below
    # (records without `chat_id` are dropped) so even a misplaced
    # legacy file in the wrong subdirectory cannot leak across users.
    user_dir = _LOG_DIR / str(chat_id)
    if not user_dir.exists():
        return []

    files = sorted(user_dir.glob("*.jsonl"), reverse=True)  # newest first
    if not files:
        return []

    # Collect filtered records, building oldest-first by prepending
    # each newer file's records to the running list. After each file
    # we re-pair the running list and check the PAIR count (not the
    # raw record count) against the requested N - the early-break
    # condition has to be expressed in pairs because the structural
    # cost is per-pair: a heavy stop-and-retry run can produce many
    # filtered records that all collapse to the same multi-user
    # pending slot or get dropped as orphan assistants. Files are
    # small (one day apiece), and re-pairing the running list is
    # bounded by total filtered records, so the per-file overhead
    # stays cheap even on heavy users.
    records: list[dict] = []
    pairs: list[tuple[str, str]] = []
    for path in files:
        file_records: list[dict] = []
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            log.exception("Failed to read history file %s", path)
            continue
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Skip individual bad lines rather than discarding the
                # whole file - matches the existing get_recent_history
                # tolerance for partial corruption.
                log.debug("Skipping malformed JSON line in %s", path.name)
                continue
            # Drop records without a `chat_id` field (legacy pre-Phase-2).
            # Greenfield divergence from get_recent_history: the
            # classifier path needs strict provenance; legacy records
            # have no way to confirm they belong to this chat.
            if "chat_id" not in rec:
                continue
            # User partition: drop records belonging to other chats.
            # Possible if a backup/restore mis-placed a file, or if a
            # future operator action stages JSONL across chats.
            if rec.get("chat_id") != chat_id:
                continue
            text = (rec.get("text") or "").strip()
            if not text:
                continue
            direction = rec.get("dir")
            # Synthetic-marker filter applies only to assistant records.
            # A user message that quotes one of the marker shapes is
            # legitimate prior context (the anchored regex shape catches
            # only exact full-line matches; quoted prose is safe).
            if direction == "assistant" and _SYNTHETIC_ASSISTANT_MARKERS.fullmatch(text):
                continue
            file_records.append({"dir": direction, "text": text})
        # Prepend so the running list stays oldest-first across files,
        # then re-pair. Pair count grows monotonically as we add
        # older records (newer pairs already exist; older records can
        # only add older pairs at the head), so checking after each
        # file is sound.
        records = file_records + records
        pairs = _pair_records_chronologically(records)
        if len(pairs) >= n:
            break

    # Cap at the N most recent. Slicing from the tail preserves
    # chronological order among the kept pairs, which is what the
    # PRIOR USER 1, 2, 3 ... rendering relies on.
    if len(pairs) > n:
        pairs = pairs[-n:]
    return pairs


def _pair_records_chronologically(records: list[dict]) -> list[tuple[str, str]]:
    """
    Walk chronologically and emit (user, assistant) pairs.

    Multi-user-before-assistant collapse: the `pending_user` slot
    keeps overwriting on consecutive user records, so the LAST user
    before an assistant is the one that pairs. Earlier orphan user
    messages are dropped. This matches the Telegram pattern where a
    user can send several messages before getting a reply.

    Assistant records pair only when a pending user exists. An
    assistant record with no pending user is an orphan (legacy data,
    restored-from-backup mishap, or partial JSONL truncation) and
    is silently dropped - the falling-out branch of the elif is the
    no-op orphan handler.

    Used both by `get_recent_pairs` for the early-break pair-count
    check and for the final returned pairs. Single source of truth
    for the pairing semantics.
    """
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    for rec in records:
        direction = rec["dir"]
        text = rec["text"]
        if direction == "user":
            pending_user = text
        elif direction == "assistant" and pending_user is not None:
            pairs.append((pending_user, text))
            pending_user = None
    return pairs
