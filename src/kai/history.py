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

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

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


@dataclass(frozen=True)
class LogEntry:
    """
    Receipt for a successfully appended JSONL line.

    Returned by `log_message` so the transcript provenance writer
    (`memory_extraction.extract_and_store`) can stamp the exact
    timestamp, date filename, and content hash that landed on disk,
    without re-deriving any of them and risking near-midnight skew
    or post-strip variants of the persisted text.

    Attributes:
        ts: The exact `ts` value persisted to the JSONL record.
        date: The UTC date used as the `YYYY-MM-DD.jsonl` filename.
            Carried separately from `ts` so callers do not slice the
            ISO string; both are derived from the same
            `datetime.now(UTC)` call so a wraparound between them is
            structurally impossible.
        chat_id: Telegram chat id, matching the JSONL `chat_id` field.
        direction: "user" or "assistant", matching the JSONL `dir`.
        text: The exact `text` field persisted to the JSONL line.
        sha256: SHA-256 hex digest of `text.encode("utf-8")`. Cached
            here so the provenance writer does not have to re-hash;
            the transcript helper compares against this fingerprint
            at lookup time.
    """

    ts: str
    date: str
    chat_id: int
    direction: str
    text: str
    sha256: str


def log_message(
    *,
    direction: str,
    chat_id: int,
    text: str,
    media: dict | None = None,
) -> LogEntry | None:
    """
    Append a single message record to today's JSONL chat log.

    Called from bot.py for every inbound user message and outbound assistant
    response. Each message is written immediately (not batched) so the log
    stays current even if the process crashes mid-conversation.

    Returns:
        A `LogEntry` describing the persisted line on success, or `None`
        when the JSONL append failed (an `OSError` from the filesystem
        layer is logged via the existing warning path; the return value
        is the only signal a caller has that the line did NOT make it
        to disk). Callers that stamp transcript provenance must skip
        the stamp when the return is `None`, so a write failure never
        produces a row pointing at a JSONL line that does not exist.

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
    # Single `now` call so the ts written into the record and the date
    # baked into the filename can never drift across a midnight boundary
    # within one log_message invocation. The returned LogEntry carries
    # both, derived from the same instant, so the provenance writer
    # stamps the file path the line actually landed in.
    now = datetime.now(UTC)
    ts = now.isoformat()
    date = now.strftime("%Y-%m-%d")
    record = {
        "ts": ts,
        "dir": direction,
        "chat_id": chat_id,
        "text": text,
        "media": media,
    }
    filepath = user_dir / f"{date}.jsonl"
    # mkdir lives inside the try so a filesystem permission or
    # availability failure BEFORE the append still produces the same
    # None signal as an append failure. Without this, a directory
    # creation error would escape `log_message` and crash the
    # caller, defeating the safety property the LogEntry | None
    # contract is meant to provide.
    try:
        user_dir.mkdir(parents=True, exist_ok=True)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        log.exception("Failed to write chat log")
        # Returning None (not a populated LogEntry) is the contract
        # that lets the transcript-provenance writer skip stamping for
        # this exchange. A populated entry here would produce a row
        # pointing at a JSONL line that does not exist, which would
        # later surface as a `memory.provenance.drift` file-missing or
        # ts-not-found event.
        return None
    return LogEntry(
        ts=ts,
        date=date,
        chat_id=chat_id,
        direction=direction,
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


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


# ── Transcript provenance reader ────────────────────────────────────
#
# Memory rows written from real bot paths carry source_* metadata
# pointing back at the JSONL line(s) that produced them. The reader
# below resolves a TranscriptProvenance value (built from a row's
# metadata by `kai.memory.read_transcript_provenance`) into the
# originating turns plus a small surrounding window, with fail-closed
# semantics: a missing file, missing timestamp, or content-hash drift
# returns the failure reason rather than guessing at a similar turn.


LookupReason = Literal[
    "ok",
    "legacy",
    "file_missing",
    "unreadable",
    "ts_not_found",
    "hash_mismatch",
    "chat_mismatch",
]

_TRANSCRIPT_WINDOW_TURN_CAP = 50

_PROVENANCE_DRIFT_EVENT = "memory.provenance.drift"


@dataclass(frozen=True)
class TranscriptTurn:
    """One JSONL record reduced to the fields the source-view consumers need."""

    ts: str
    direction: str
    text: str


@dataclass(frozen=True)
class TranscriptContext:
    """
    The originating turns plus a chronological surrounding window.

    `target_assistant` is None when the row's stored provenance has
    no `source_assistant_ts` (a corner case the spec admits but does
    not produce from real bot paths today); an assistant ts that IS
    set but no longer resolves in the JSONL is reported as a drift
    via TranscriptLookup.reason, not by setting this to None.

    `truncated` flags the episode-window cap and is always False on
    fact lookups (whose context is fixed by the `before` / `after`
    parameters).
    """

    chat_id: int
    target_user: TranscriptTurn
    target_assistant: TranscriptTurn | None
    before: list[TranscriptTurn]
    after: list[TranscriptTurn]
    truncated: bool


@dataclass(frozen=True)
class TranscriptLookup:
    """
    Typed result for `fetch_transcript_context`.

    `reason` always carries one of the LookupReason values; `context`
    is populated iff `reason == "ok"`. The two are bundled so callers
    (the /memory source view, the reclassification dry-run report)
    can branch on the reason without scraping logs.
    """

    reason: LookupReason
    context: TranscriptContext | None


def _read_jsonl_file(path) -> list[dict] | None:
    """
    Best-effort JSONL reader.

    Returns the parsed records on success, None on any I/O failure.
    Malformed lines are skipped individually (mirroring
    `get_recent_history`'s own posture); only a file-level error
    collapses the return to None so the caller can distinguish
    "file present but garbled rows" from "file unreadable."
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    out: list[dict] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            log.debug("Skipping malformed JSON line in %s: %s", path.name, line[:100])
    return out


def _shift_date(date: str, days: int) -> str:
    """
    Add `days` to a `YYYY-MM-DD` string, returning the same shape.

    Used to walk one file forward or back when the target window
    crosses midnight. Goes through `datetime` rather than string
    math so leap years and month rollovers behave correctly without
    a calendar table.
    """
    parsed = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC)
    return (parsed + timedelta(days=days)).strftime("%Y-%m-%d")


def _emit_drift_log(
    *,
    memory_id: str | None,
    chat_id: int | None,
    reason: LookupReason,
    side: str | None = None,
) -> None:
    """
    Structured log line for a non-ok, non-legacy lookup.

    Every drift event names the row (when known), the chat, the
    stable reason key, and the `side` discriminator that distinguishes
    user-line and assistant-line misses for `ts_not_found`. Keeping
    one event name across every failure mode means downstream log
    queries do not have to enumerate sub-events as the helper grows.
    """
    payload: dict = {
        "memory_id": memory_id,
        "chat_id": chat_id,
        "reason": reason,
        "side": side,
    }
    log.info("%s %s", _PROVENANCE_DRIFT_EVENT, json.dumps(payload, separators=(",", ":")))


def fetch_transcript_context(
    provenance,
    *,
    before: int = 3,
    after: int = 1,
    memory_id: str | None = None,
    expected_chat_id: int | None = None,
) -> TranscriptLookup:
    """
    Resolve a row's transcript provenance into surrounding turns.

    Args:
        provenance: A `kai.memory.TranscriptProvenance` value. Typed
            as Any here to avoid an import cycle (memory.py imports
            history.py for log writes, and the resolver lives in
            memory.py); duck-typed access to `present`, `chat_id`,
            `date`, `user_ts`, `user_text_sha256`, `assistant_ts`,
            and `date_end` is the contract.
        before: Maximum non-synthetic turns to include in the
            preceding context window. Walks at most one file backward
            when today's file does not contain `before` predecessors.
        after: Maximum non-synthetic turns to include after the
            assistant turn. Walks at most one file forward.
        memory_id: Optional row id, carried into the drift log so log
            queries can correlate failures back to the affected row.
        expected_chat_id: Ownership gate for the consumer. When
            provided AND it differs from `provenance.chat_id`, the
            helper returns `chat_mismatch` BEFORE touching disk: row
            ownership (verified by Mem0's user_id partition at
            `get_by_id` time) does not extend to the `source_chat_id`
            field, which is just metadata the row carries. A
            malformed, restored, or forged row whose `source_chat_id`
            points at another chat would otherwise let the consumer
            dereference an unrelated chat's JSONL. Callers that
            already trust the provenance (admin tooling that scans
            cross-chat by design) can omit the kwarg.

    Returns:
        TranscriptLookup with `reason="ok"` and a populated
        `TranscriptContext` on success. Every other path returns a
        non-ok reason with `context=None`; non-ok, non-legacy reasons
        emit a single structured log line.

    Failure mode contract:
        - legacy:        `provenance.present is False`. No log emitted.
        - file_missing:  the per-day JSONL file is absent from disk.
        - unreadable:    the file exists but cannot be read.
        - ts_not_found:  the named timestamp is absent from the file
                         (the `side` field of the log distinguishes
                         user-side miss from assistant-side miss).
        - hash_mismatch: the user line was found but its text no
                         longer matches the stored fingerprint.
        - chat_mismatch: `expected_chat_id` was supplied and disagrees
                         with `provenance.chat_id`. No disk access
                         occurs; the drift log fires so a forged or
                         corrupted pointer surfaces in observability.

    The helper never returns a "maybe matched" turn. A wrong-turn
    render would be worse than no render at all.
    """
    if not provenance.present:
        return TranscriptLookup(reason="legacy", context=None)

    chat_id: int = provenance.chat_id
    if expected_chat_id is not None and chat_id != expected_chat_id:
        # Fail closed BEFORE the filesystem read so a cross-chat
        # pointer cannot leak even one byte of another chat's
        # transcript. The drift log surfaces the attempt so an
        # operator scanning observability for provenance issues can
        # find a forged or restored-from-bad-backup row.
        _emit_drift_log(memory_id=memory_id, chat_id=chat_id, reason="chat_mismatch")
        return TranscriptLookup(reason="chat_mismatch", context=None)

    date: str = provenance.date
    user_ts: str = provenance.user_ts
    user_text_sha256: str = provenance.user_text_sha256
    assistant_ts: str | None = provenance.assistant_ts

    primary_path = _LOG_DIR / str(chat_id) / f"{date}.jsonl"
    if not primary_path.exists():
        _emit_drift_log(memory_id=memory_id, chat_id=chat_id, reason="file_missing")
        return TranscriptLookup(reason="file_missing", context=None)
    primary_records = _read_jsonl_file(primary_path)
    if primary_records is None:
        _emit_drift_log(memory_id=memory_id, chat_id=chat_id, reason="unreadable")
        return TranscriptLookup(reason="unreadable", context=None)

    # Locate the user line by exact ts + direction. The lookup is by
    # ts (not by index) so any future history-management tool that
    # reorders or deduplicates lines without changing them does not
    # break provenance.
    user_index = _find_record_index(primary_records, ts=user_ts, direction="user")
    if user_index is None:
        _emit_drift_log(memory_id=memory_id, chat_id=chat_id, reason="ts_not_found", side="user")
        return TranscriptLookup(reason="ts_not_found", context=None)
    user_record = primary_records[user_index]
    user_text = user_record.get("text", "")
    if hashlib.sha256(user_text.encode("utf-8")).hexdigest() != user_text_sha256:
        _emit_drift_log(memory_id=memory_id, chat_id=chat_id, reason="hash_mismatch")
        return TranscriptLookup(reason="hash_mismatch", context=None)
    target_user = TranscriptTurn(ts=user_record["ts"], direction="user", text=user_text)

    target_assistant: TranscriptTurn | None = None
    if assistant_ts is not None:
        # Same-day search first; the assistant turn usually lives in
        # the same file. When it does not, walk one file forward
        # (episode midnight cross) before giving up.
        assistant_index = _find_record_index(primary_records, ts=assistant_ts, direction="assistant")
        if assistant_index is None:
            forward_path = _LOG_DIR / str(chat_id) / f"{_shift_date(date, 1)}.jsonl"
            forward_records = _read_jsonl_file(forward_path) if forward_path.exists() else None
            if forward_records is not None:
                forward_assistant_index = _find_record_index(forward_records, ts=assistant_ts, direction="assistant")
                if forward_assistant_index is not None:
                    rec = forward_records[forward_assistant_index]
                    target_assistant = TranscriptTurn(ts=rec["ts"], direction="assistant", text=rec.get("text", ""))
            if target_assistant is None:
                _emit_drift_log(memory_id=memory_id, chat_id=chat_id, reason="ts_not_found", side="assistant")
                return TranscriptLookup(reason="ts_not_found", context=None)
        else:
            rec = primary_records[assistant_index]
            target_assistant = TranscriptTurn(ts=rec["ts"], direction="assistant", text=rec.get("text", ""))

    before_turns = _collect_before(
        chat_id=chat_id, date=date, primary_records=primary_records, user_index=user_index, before=before
    )
    after_turns = _collect_after(
        chat_id=chat_id,
        date=date,
        primary_records=primary_records,
        anchor_index=user_index
        if target_assistant is None
        else _find_record_index(primary_records, ts=target_assistant.ts, direction="assistant"),
        target_assistant=target_assistant,
        after=after,
    )

    return TranscriptLookup(
        reason="ok",
        context=TranscriptContext(
            chat_id=chat_id,
            target_user=target_user,
            target_assistant=target_assistant,
            before=before_turns,
            after=after_turns,
            truncated=False,
        ),
    )


def _find_record_index(records: list[dict], *, ts: str, direction: str) -> int | None:
    """
    Return the index of the first record matching exact ts + direction.

    Linear scan; per-day JSONL files are small enough that an index
    would add complexity without measurable benefit. Returns None
    when no record matches.
    """
    for i, rec in enumerate(records):
        if rec.get("ts") == ts and rec.get("dir") == direction:
            return i
    return None


def _turn_is_synthetic(text: str) -> bool:
    """Skip synthetic assistant placeholders in context windows."""
    return bool(_SYNTHETIC_ASSISTANT_MARKERS.fullmatch(text or ""))


def _to_turn(rec: dict) -> TranscriptTurn:
    return TranscriptTurn(ts=rec.get("ts", ""), direction=rec.get("dir", ""), text=rec.get("text", ""))


def _collect_before(
    *, chat_id: int, date: str, primary_records: list[dict], user_index: int, before: int
) -> list[TranscriptTurn]:
    """
    Walk backwards from the target user line, including up to `before`
    non-synthetic turns. When today's file does not provide enough,
    walk one file back. Returns chronological order (oldest first).
    """
    if before <= 0:
        return []
    collected: list[TranscriptTurn] = []
    for rec in reversed(primary_records[:user_index]):
        if _turn_is_synthetic(rec.get("text", "")):
            continue
        collected.append(_to_turn(rec))
        if len(collected) >= before:
            break
    if len(collected) < before:
        prev_path = _LOG_DIR / str(chat_id) / f"{_shift_date(date, -1)}.jsonl"
        if prev_path.exists():
            prev_records = _read_jsonl_file(prev_path)
            if prev_records is not None:
                for rec in reversed(prev_records):
                    if _turn_is_synthetic(rec.get("text", "")):
                        continue
                    collected.append(_to_turn(rec))
                    if len(collected) >= before:
                        break
    collected.reverse()
    return collected


def _collect_after(
    *,
    chat_id: int,
    date: str,
    primary_records: list[dict],
    anchor_index: int | None,
    target_assistant: TranscriptTurn | None,
    after: int,
) -> list[TranscriptTurn]:
    """
    Walk forward from the anchor index, including up to `after`
    non-synthetic turns. The anchor is the assistant turn when
    present, otherwise the user turn (so the "after" window starts
    right after whichever target is the rightmost one in the primary
    file). When today's file does not provide enough, walk one file
    forward.

    Midnight-cross edge case: an episode whose user lands on day D
    and whose assistant lands on day D+1 reaches here with
    `anchor_index=None` (the assistant is not in `primary_records`)
    but `target_assistant` populated. The early return for
    `anchor_index is None` is therefore gated on having no
    target_assistant at all; when the target lives in the next file,
    we skip the primary-file scan and walk straight into the
    next-file path below.
    """
    if after <= 0:
        return []
    if anchor_index is None and target_assistant is None:
        return []
    collected: list[TranscriptTurn] = []
    if anchor_index is not None:
        for rec in primary_records[anchor_index + 1 :]:
            if _turn_is_synthetic(rec.get("text", "")):
                continue
            collected.append(_to_turn(rec))
            if len(collected) >= after:
                break
    if len(collected) < after:
        # The episode midnight-cross path may have already populated
        # the assistant from the next file. We still want to include
        # any "after" turns following that assistant in the next file
        # itself, which is the same path the same-day branch would have
        # taken if the assistant lived locally.
        next_path = _LOG_DIR / str(chat_id) / f"{_shift_date(date, 1)}.jsonl"
        if next_path.exists():
            next_records = _read_jsonl_file(next_path)
            if next_records is not None:
                # When the target_assistant lives in next_records, start
                # after it; otherwise start at the beginning of the file.
                start = 0
                if target_assistant is not None:
                    forward_assistant_index = _find_record_index(
                        next_records, ts=target_assistant.ts, direction="assistant"
                    )
                    if forward_assistant_index is not None:
                        start = forward_assistant_index + 1
                for rec in next_records[start:]:
                    if _turn_is_synthetic(rec.get("text", "")):
                        continue
                    collected.append(_to_turn(rec))
                    if len(collected) >= after:
                        break
    return collected
