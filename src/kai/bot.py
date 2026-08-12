"""
Telegram bot interface — command handlers, message routing, and streaming responses.

Provides functionality to:
1. Handle all Telegram slash commands (/new, /model, /workspace, /voice, etc.)
2. Process text, photo, document, and voice messages from the user
3. Stream Claude's responses in real-time with progressive message edits
4. Manage model switching, voice TTS output, and workspace navigation
5. Enforce authorization (only allowed user IDs can interact)

This module is the "presentation layer" of Kai — it receives Telegram updates,
translates them into prompts for the Claude process (claude.py), streams the
response back to the user, and handles all Telegram-specific concerns like
message length limits, Markdown fallback, inline keyboards, and typing indicators.

The response flow for a text message:
    1. User message arrives → handle_message()
    2. Message logged to JSONL history
    3. Per-chat lock acquired (prevents concurrent Claude interactions)
    4. Flag file written (for crash recovery)
    5. Prompt sent to ClaudeCodeBackend.send() → streaming begins
    6. Live message created and progressively edited (2-second intervals)
    7. Final response delivered (text, voice, or both depending on voice mode)
    8. Session saved to database
    9. Flag file cleared

Handler registration order in create_bot() matters: python-telegram-bot matches
the first handler whose filters pass, so specific commands are registered before
the catch-all text message handler.
"""

import asyncio
import base64
import functools
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from kai import github_api, memory_command, review, services, sessions, webhook
from kai.agent_failure import render_agent_failure
from kai.backend import require_backend_name, resolve_home_workspace
from kai.config import (
    DATA_DIR,
    ONESHOT_REASONER_BACKENDS,
    OPEN_ENDED_PROVIDERS,
    PROVIDER_DEFAULTS,
    Config,
    ModelRole,
    WorkspaceConfig,
    get_effective_provider,
    get_user_backend_and_provider,
    models_for_backend,
    resolve_user_model,
    validate_model_for_backend,
)
from kai.history import LogEntry, log_message
from kai.locks import get_lock, get_stop_event
from kai.pool import SubprocessPool
from kai.sessions import WorkshopFinalizationCommitUncertainError
from kai.telegram_utils import chunk_text
from kai.transcribe import TranscriptionError, transcribe_voice
from kai.tts import DEFAULT_VOICE, VOICES, TTSError, synthesize_speech
from kai.workshop.artifacts import InboundArtifact
from kai.workshop.conversation_runs import PreparedConversationRun, WorkshopConversationRunService
from kai.workshop.domain import MessageId
from kai.workshop.inbound import InboundMessage
from kai.workshop.outbound import DeliveryObservation, OutboundMessage
from kai.workshop.streaming_preview import ConfirmedTelegramStreamingPreview
from kai.workspace_utils import is_workspace_allowed

_UPLOAD_ROOT_MODE = 0o711
_UPLOAD_FILE_MODE = 0o600

# TOTP is optional for single-user development (requires pip install
# -e '.[totp]'). A missing extra may disable the gate only when neither
# protected TOTP file exists. If protected state exists, fail closed.
try:
    from kai.totp import TotpStateError, get_failure_count, get_lockout_remaining, is_totp_configured, verify_code
except ImportError as exc:
    _TOTP_IMPORT_FAILURE = exc

    class TotpStateError(RuntimeError):  # type: ignore[no-redef]
        """TOTP protected state exists but the optional implementation is unavailable."""

    def is_totp_configured() -> bool:  # type: ignore[misc]
        try:
            state_exists = any(os.path.exists(path) for path in ("/etc/kai/totp.secret", "/etc/kai/totp.attempts"))
        except OSError as exc:
            raise TotpStateError("Could not determine whether protected TOTP state exists") from exc
        if state_exists:
            raise TotpStateError("Protected TOTP state exists but TOTP support could not be imported") from (
                _TOTP_IMPORT_FAILURE
            )
        return False

    def get_lockout_remaining(principal_id: int) -> int:  # type: ignore[misc]
        raise TotpStateError("TOTP support is unavailable")

    def verify_code(  # type: ignore[misc]
        code: str,
        principal_id: int,
        lockout_attempts: int = 3,
        lockout_minutes: int = 15,
    ) -> bool:
        raise TotpStateError("TOTP support is unavailable")

    def get_failure_count(principal_id: int) -> int:  # type: ignore[misc]
        raise TotpStateError("TOTP support is unavailable")


log = logging.getLogger(__name__)

# Minimum interval between Telegram message edits (seconds).
# Telegram rate-limits message edits; 2 seconds keeps us safely below the limit
# while still giving the user a sense of streaming output.
EDIT_INTERVAL = 2.0


class ResponseDeliveryRoute(StrEnum):
    """Explicit final-delivery authority selected by an ingress handler."""

    LEGACY = "legacy"
    WORKSHOP_PRIVATE_TEXT = "workshop_private_text"


# Flag file written while processing a message. If the process crashes mid-response,
# main.py detects this file at startup and notifies the user to resend. Lives under
# DATA_DIR so it's writable even when source is in read-only /opt/kai/.
# Directory for per-user crash recovery flags. Each file is named by
# chat_id and exists only while that user's response is in-flight.
# Using a directory of files (not a single JSON file) avoids locking
# and allows atomic per-user create/delete.
_RESPONDING_DIR = DATA_DIR / ".responding"


# ── Crash recovery flag ──────────────────────────────────────────────


def _set_responding(chat_id: int) -> None:
    """Mark a response as in-flight for crash recovery."""
    _RESPONDING_DIR.mkdir(exist_ok=True)
    (_RESPONDING_DIR / str(chat_id)).touch()


def _clear_responding(chat_id: int) -> None:
    """Mark a response as complete for a specific user."""
    (_RESPONDING_DIR / str(chat_id)).unlink(missing_ok=True)


async def _notify_if_queued(update: Update, chat_id: int) -> bool:
    """Send a notification if the user's message will queue behind the lock.

    Called immediately before acquiring the per-chat lock. If the lock is
    already held (Kai is mid-response), sends a one-line Telegram message
    so the user knows their message was received. The notification goes
    directly to Telegram via _reply_safe - Claude never sees it. Do NOT
    add a log_message call here; the notification is purely for the user.

    Returns True if the message is queuing (lock was held), False otherwise.
    The caller uses this to decide whether to prepend a context-switch
    marker to the prompt via _prepend_queue_marker().

    There is a harmless TOCTOU gap: if the lock holder releases between
    the locked() check and the subsequent acquire, the user sees "finishing
    something up" followed by an instant response, and Claude gets a
    context-switch marker for a task that already finished. Both are
    harmless and not worth fixing.
    """
    if get_lock(chat_id).locked():
        assert update.message is not None
        await _reply_safe(
            update.message,
            "Got your message - finishing something up. /stop to interrupt.",
        )
        return True
    return False


# Prepended to prompts that waited behind the lock, so Claude focuses on the
# new message instead of continuing from the previous task's tool output.
_QUEUED_MESSAGE_MARKER = (
    "[The user sent this while you were working on something else. "
    "Their previous task is done. Focus on this new message.]\n\n"
)

# Safety-net timeout for acquiring the per-chat lock (seconds). If the
# idle timeout in claude.py doesn't fire for some reason, this prevents
# a stuck interaction from blocking all future messages indefinitely.
# Set generously: the idle timer in claude.py is the real safety net
# (fires after timeout_seconds * 5 of silence); this is a last-resort
# backstop for interactions that run legitimately long with active output.
_LOCK_ACQUIRE_TIMEOUT = 3600  # 1 hour


# Strong references to in-flight memory-ingestion tasks.
#
# `asyncio.create_task` returns a Task that the runtime holds only by
# WEAK reference. Without a strong reference somewhere, a heap-pressure
# GC cycle can reap a still-running task; the failure mode is silent
# because the task simply disappears and the exchange never makes it
# into semantic memory. The set holds a strong reference until the
# task completes; the done-callback `_pending_memory_tasks.discard`
# registered at spawn keeps the set self-pruning so a long-running
# deployment does not accumulate references without bound.
#
# Same pattern as `webhook.py`'s `_background_tasks` and
# `memory_extraction.py`'s `_pending_episode_tasks`. Each module owns
# its own set so the ownership boundary stays explicit; consolidating
# into a single global set would obscure which subsystem is responsible
# for a given task's lifetime.
_pending_memory_tasks: set[asyncio.Task[None]] = set()


async def _acquire_lock_or_kill(
    chat_id: int,
    pool: "SubprocessPool",
    update: Update,
) -> asyncio.Lock | None:
    """Acquire the per-chat lock with a timeout, force-killing if stuck.

    Returns the acquired lock on success (caller must call lock.release()
    in a finally block). Returns None if the lock timed out, in which case
    the stuck Claude process was killed and the user was notified - caller
    should return without further action.

    Returns the lock object directly rather than a bool so the caller
    releases the same object that was acquired (avoids issues if get_lock
    is called again and returns a different instance).
    """
    lock = get_lock(chat_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_ACQUIRE_TIMEOUT)
        return lock
    except TimeoutError:
        log.error(
            "Lock acquisition timed out for chat %d after %ds; force-killing agent subprocess",
            chat_id,
            _LOCK_ACQUIRE_TIMEOUT,
        )
        await pool.force_kill(chat_id)
        # update.message can be None for edited messages or callback
        # queries, so guard rather than assert.
        if update.message is not None:
            await _reply_safe(
                update.message,
                "Previous task timed out and was stopped. Please send your message again.",
            )
        return None


def _prepend_queue_marker(prompt: str | list[dict[str, str]]) -> str | list[dict[str, str]]:
    """Prepend context-switch marker to a prompt that waited behind the lock.

    Handles both plain string prompts (text, document, voice) and multimodal
    content lists (photo). For lists, prepends to the first text block's text
    field and passes subsequent blocks (e.g., base64 image) through unchanged.
    """
    if isinstance(prompt, list):
        # Multimodal content (photo): prepend to the first text block
        first = prompt[0]
        return [{"type": "text", "text": _QUEUED_MESSAGE_MARKER + first["text"]}] + prompt[1:]
    return _QUEUED_MESSAGE_MARKER + prompt


# ── Update property helpers (Pyright can't narrow @property returns) ─


def _chat_id(update: Update) -> int:
    """Extract the chat ID from an update, with type narrowing for static analysis."""
    chat = update.effective_chat
    assert chat is not None
    return chat.id


def _user_id(update: Update) -> int:
    """Extract the user ID from an update, with type narrowing for static analysis."""
    user = update.effective_user
    assert user is not None
    return user.id


# ── Authorization ────────────────────────────────────────────────────


def _is_authorized(config: Config, user_id: int) -> bool:
    """Check if a Telegram user ID is in the allowed list."""
    return user_id in config.allowed_user_ids


def _require_auth(func):
    """
    Decorator that silently drops updates from unauthorized users.

    Wraps a Telegram handler function to check the sender's user ID against
    the allowed list before executing. Unauthorized messages are ignored
    without any response (to avoid revealing the bot's existence).
    """

    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        config: Config = context.bot_data["config"]
        if not _is_authorized(config, _user_id(update)):
            return
        return await func(update, context)

    return wrapper


def _require_sensitive_authentication(func):
    """Wrap a registered handler with authorization and TOTP middleware.

    The wrapper lives at the Telegram registration boundary so it can protect
    handlers defined in other modules (notably memory_command) without a
    circular import.  Authorization runs first to avoid disclosing the bot or
    issuing TOTP challenges to unknown Telegram users.
    """

    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        config: Config = context.bot_data["config"]
        if not _is_authorized(config, _user_id(update)):
            if update.callback_query is not None:
                await update.callback_query.answer("Not authorized.")
            return
        if not await _check_totp(update, context):
            return
        return await func(update, context)

    # Tests and registration audits use this explicit marker instead of
    # inferring security from wrapper names or decorator order.
    wrapper._kai_totp_sensitive = True  # type: ignore[attr-defined]
    return wrapper


# ── Telegram message utilities ───────────────────────────────────────


def _truncate_for_telegram(text: str, max_len: int = 4096) -> str:
    """Truncate text to Telegram's message length limit, appending '...' if cut."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 4] + "\n..."


# ── Streaming publishable-prefix filter ──────────────────────────────
#
# Telegram's live message edits operate on raw accumulated text from each
# StreamEvent. Backend chunks arrive at protocol boundaries (tokens,
# model-defined messages) which rarely line up with human boundaries. The
# helper below answers "is there a stable prefix of this accumulated text
# I can publish right now?" and returns the longest such prefix or None.
#
# The transport layer (_handle_response) consults this helper for every
# streamed update; a None answer means "withhold this update; the final
# response path will still deliver everything at completion". A non-None
# answer is the candidate text to create or edit the live message with.
# Backends are untouched - this is purely a Telegram-side policy.

# Triple-backtick fence delimiter, allowing up to three leading spaces
# (CommonMark allows 0-3) before the opening sequence.
_FENCE_LINE_RE = re.compile(r"^[ ]{0,3}```")
# Sentence terminators.
_SENTENCE_END_CHARS = frozenset(".?!")
# Closing punctuation that may follow a sentence terminator and stay part
# of the same sentence (quote-then-period style).
_CLOSE_PUNCT_CHARS = frozenset("\"')]`")
# Markdown list-item line shapes.
_LIST_LINE_WITH_CONTENT_RE = re.compile(r"^[ ]*([-*+]|\d+\.)\s+\S")
_LIST_LINE_RE = re.compile(r"^[ ]*([-*+]|\d+\.)(\s|$)")
# Ordered-list marker at the start of a line. Used in _sentence_cuts to
# advance past the marker period; without this, `1. Item one` would
# treat the `.` in `1.` as a sentence boundary and publish a bare
# numbered marker as if it were a completed sentence.
_ORDERED_LIST_MARKER_RE = re.compile(r"^[ ]*\d+\.(\s|$)")
# Looser pattern that also matches a forming next-item marker. The
# stream may have emitted just the digits of the next ordered-list
# marker (`3`) before the period and following text arrive; treating
# the partial marker as a list-item signal lets list_item cuts fire on
# the previous complete items even while the next marker is still
# mid-emission.
_LIST_LINE_FORMING_RE = re.compile(r"^[ ]*(\d+\.?|[-*+])(\s|$)")
# Minimum length for the long-span fallback to fire. Picked to be long
# enough that a coherent paragraph is likely visible, but short enough
# that streamed paragraphs reach it before the user gives up watching.
_LONG_SPAN_MIN_CHARS = 240
# Candidate kinds ordered by preference when two kinds resolve to the
# same cut position. Lower number = stronger boundary, evaluated first.
_KIND_PRIORITY = {
    "sentence": 1,
    "paragraph": 2,
    "list_item": 3,
    "closed_fence": 4,
    "long_span": 5,
    "full": 6,
}


def _is_fence_line(line: str) -> bool:
    return bool(_FENCE_LINE_RE.match(line))


def _has_open_fenced_code(text: str) -> bool:
    """True iff `text` has an odd number of triple-backtick fence lines.

    An odd count means the last fence opened is still unclosed; the helper
    must not publish through an open fenced block because the next chunk
    is going to land inside it.
    """
    fences = sum(1 for line in text.splitlines() if _is_fence_line(line))
    return (fences % 2) == 1


def _segments_outside_fences(text: str) -> str:
    """Return the lines of `text` that sit outside fenced code blocks.

    Used by the inline-Markdown checks: backticks and brackets inside a
    fenced code block are content, not Markdown delimiters, and must not
    affect the balance counts.
    """
    parts = []
    in_fence = False
    for line in text.splitlines(keepends=True):
        if _is_fence_line(line.rstrip("\n")):
            in_fence = not in_fence
            continue
        if not in_fence:
            parts.append(line)
    return "".join(parts)


def _has_unbalanced_inline_markdown(text: str) -> bool:
    """True iff `text` has an unbalanced inline-code span or link.

    Checks (all candidate-wide per spec §6 D6 N-4 fix):
      - Odd count of unescaped single backticks outside fenced regions.
      - Unmatched `[` (no later `]`) anywhere outside fenced regions.
      - Unmatched `](` (no later closing `)`) anywhere outside fenced
        regions, which catches links whose target was cut mid-stream.
    """
    outside = _segments_outside_fences(text)
    n = len(outside)

    # Backtick parity, skipping backslash-escaped chars.
    backticks = 0
    i = 0
    while i < n:
        if outside[i] == "\\" and i + 1 < n:
            i += 2
            continue
        if outside[i] == "`":
            backticks += 1
        i += 1
    if backticks % 2 == 1:
        return True

    # Bracket + link-target balance. A `[label]` reference link closes
    # cleanly; `[label](target)` opens a target on the matching `]` and
    # tracks paren depth until it closes. An unbalanced state at end of
    # text means the stream cut inside an open link.
    open_brackets = 0
    in_link_target = False
    paren_depth = 0
    i = 0
    while i < n:
        if outside[i] == "\\" and i + 1 < n:
            i += 2
            continue
        c = outside[i]
        if not in_link_target:
            if c == "[":
                open_brackets += 1
            elif c == "]" and open_brackets > 0:
                open_brackets -= 1
                if i + 1 < n and outside[i + 1] == "(":
                    in_link_target = True
                    paren_depth = 1
                    i += 2
                    continue
        else:
            if c == "(":
                paren_depth += 1
            elif c == ")":
                paren_depth -= 1
                if paren_depth == 0:
                    in_link_target = False
        i += 1
    return open_brackets > 0 or in_link_target


def _line_ends_inside_unmatched_inline(line: str) -> bool:
    """True iff `line` has an odd backtick count or open `[`."""
    bt = 0
    open_brackets = 0
    i = 0
    n = len(line)
    while i < n:
        if line[i] == "\\" and i + 1 < n:
            i += 2
            continue
        c = line[i]
        if c == "`":
            bt += 1
        elif c == "[":
            open_brackets += 1
        elif c == "]" and open_brackets > 0:
            open_brackets -= 1
        i += 1
    return (bt % 2 == 1) or open_brackets > 0


def _has_dangling_final_line(candidate: str, kind: str) -> bool:
    """True iff the final visible line of `candidate` is a dangling fragment.

    For non-`full` candidates the boundary collection in Phase 2 already
    validated the final line as a stable unit (a complete sentence,
    paragraph end, list item, etc.), so only the unmatched-inline-Markdown
    rule on the final line runs. For `full` candidates the cut sits at
    the stream's tail and the full §D5 guard applies: bare list/heading
    markers, short or single-word lines without sentence punctuation, and
    mid-line broken inline Markdown all reject.
    """
    stripped = candidate.rstrip()
    if not stripped:
        return False
    lines = stripped.splitlines()
    final_line = lines[-1] if lines else ""
    final_stripped = final_line.strip()
    if not final_stripped:
        return False

    # Look past trailing closing punctuation runs to find the true
    # terminator. `He said "Yes."` ends with `"` but the real terminator
    # is the `.`; that distinction lets compound endings like `?!` and
    # quote-wrapped sentences satisfy the "ends in sentence punctuation"
    # check below.
    last_real = final_stripped
    while last_real and last_real[-1] in _CLOSE_PUNCT_CHARS:
        last_real = last_real[:-1]
    ends_in_sentence = bool(last_real) and last_real[-1] in _SENTENCE_END_CHARS

    # The `full` candidate sits at the very tail of the accumulated
    # buffer; it has no preceding boundary marker to vouch for it. It
    # is safe to publish only when the final visible line ends in
    # sentence punctuation (with optional closing quotes/parens). Any
    # other tail shape (bare list/heading markers, mid-sentence prose
    # of any length, a dangling fragment after an earlier sentence on
    # the same line) risks shipping unstable text that the next stream
    # chunk will overwrite. Other stable boundary kinds (paragraph,
    # sentence, closed_fence, list_item) produce their own candidates
    # at lower or equal cut positions and win via the priority sort
    # when they coincide.
    if kind == "full" and not ends_in_sentence:
        return True

    # Rule 5 applies to every candidate kind: an open inline span on the
    # final line is broken Markdown regardless of cut origin.
    if _is_fence_line(final_line):
        # Triple-backtick fence delimiter; the trailing backticks are the
        # closing fence marker, not an unclosed inline code span. Without
        # this carve-out, closed-fence candidates get rejected for ending
        # with three "unmatched" backticks.
        return False
    return _line_ends_inside_unmatched_inline(final_line)


def _paragraph_cuts(working: str) -> list[int]:
    """Positions of `\\n\\n` separators that are followed by more content."""
    cuts: list[int] = []
    i = 0
    n = len(working)
    while True:
        j = working.find("\n\n", i)
        if j < 0:
            break
        if working[j + 2 : n].strip():
            cuts.append(j)
        i = j + 1
    return cuts


def _sentence_cuts(working: str) -> list[int]:
    """End-of-sentence positions outside any open fenced code block.

    A `.?!` run (optionally followed by closing punctuation like quotes
    or parens) qualifies as a sentence boundary only when the run ends
    at end-of-line or is followed by whitespace. Mid-token periods
    (decimals like `3.13`, version strings like `v1.2.3`, file paths
    like `src/bot.py`, domain names) are NOT sentence ends; cutting at
    them would publish a misleading prefix that splits the token in
    half (e.g. `Use Python 3.` while the stream still has `13` to come).
    The next stream chunk would then overwrite the visible message with
    the correctly-joined text, but the user has already seen the wrong
    prefix flash by.
    """
    cuts: list[int] = []
    in_fence = False
    pos = 0
    n = len(working)
    while pos < n:
        nl = working.find("\n", pos)
        line_end = nl if nl >= 0 else n
        line = working[pos:line_end]
        if _is_fence_line(line):
            in_fence = not in_fence
        elif not in_fence:
            line_len = len(line)
            # Advance past an ordered-list marker so its period isn't
            # treated as a sentence terminator. The marker `.` is
            # followed by whitespace and would otherwise satisfy the
            # sentence-end predicate, letting `1.` publish as if it
            # were a complete sentence while the list item is still
            # being typed.
            list_match = _ORDERED_LIST_MARKER_RE.match(line)
            j = list_match.end() if list_match else 0
            while j < line_len:
                if line[j] in _SENTENCE_END_CHARS:
                    # Extend the cut through any closing-punctuation or
                    # compound sentence-end run so `Yes."` and `?!` keep
                    # the closer as part of the published prefix.
                    k = j + 1
                    while k < line_len and (line[k] in _CLOSE_PUNCT_CHARS or line[k] in _SENTENCE_END_CHARS):
                        k += 1
                    # Sentence-boundary predicate: the run must land at
                    # end-of-line or be followed by a whitespace
                    # separator. Anything else means the punctuation is
                    # internal to a token and should not become a cut.
                    if k == line_len or line[k].isspace():
                        cuts.append(pos + k)
                    j = k
                else:
                    j += 1
        pos = line_end + 1 if nl >= 0 else line_end
    return cuts


def _closed_fence_cuts(working: str) -> list[int]:
    """Positions immediately after each closing fence line."""
    cuts: list[int] = []
    in_fence = False
    pos = 0
    n = len(working)
    while pos < n:
        nl = working.find("\n", pos)
        line_end = nl if nl >= 0 else n
        line = working[pos:line_end]
        if _is_fence_line(line):
            in_fence = not in_fence
            if not in_fence:
                cuts.append(line_end)
        pos = line_end + 1 if nl >= 0 else line_end
    return cuts


def _list_item_cuts(working: str) -> list[int]:
    """End-of-line positions where a complete list item closes."""
    cuts: list[int] = []
    lines = working.splitlines()
    if not lines:
        return cuts
    starts = [0]
    for line in lines[:-1]:
        starts.append(starts[-1] + len(line) + 1)
    for i, line in enumerate(lines):
        if not _LIST_LINE_WITH_CONTENT_RE.match(line):
            continue
        if i + 1 >= len(lines):
            continue
        next_line = lines[i + 1]
        # A list-item boundary fires when the next line is either
        # blank, a fully-formed list-item line, or a still-forming
        # marker (bare digits like `3` before the period arrives, a
        # lone bullet character). That keeps prose paragraphs after a
        # list intact (the paragraph boundary handles them), avoids
        # cutting a list short when the next line is still building,
        # and accepts the partial next-item marker as evidence that
        # the previous item is complete.
        if not next_line.strip() or _LIST_LINE_FORMING_RE.match(next_line):
            cuts.append(starts[i] + len(line))
    return cuts


def _long_span_cut(working: str) -> int | None:
    """Cut at a whitespace boundary in long unpunctuated prose.

    Long-form responses without paragraph or sentence breaks still need
    a streaming surface so the user sees progress. Two shapes are
    handled:

      1. Multi-line: when a newline exists and the prefix before the
         final newline already holds at least ``_LONG_SPAN_MIN_CHARS``
         of content, cut before the final newline. This preserves the
         dangling-line guard's protection on the final (possibly
         in-progress) line.
      2. Single-line: when no newline exists, cut at the rightmost
         whitespace whose position is at or beyond the threshold.
         Without this fallback, an inner-Claude monologue streamed as
         one long unpunctuated paragraph would have no stable prefix
         until a sentence terminator finally appears; the user sees a
         stalled message for the entire run.
    """
    last_nl = working.rfind("\n")
    if last_nl >= 0:
        prefix = working[:last_nl].rstrip()
        if len(prefix) >= _LONG_SPAN_MIN_CHARS:
            return last_nl
        return None
    # Single-line fallback. Scan right-to-left for a whitespace at
    # position ≥ threshold; the word immediately before such a
    # whitespace is guaranteed complete (it has a separator after it),
    # whereas the final word at the buffer tail may still be growing.
    n = len(working)
    if n <= _LONG_SPAN_MIN_CHARS:
        return None
    for i in range(n - 1, _LONG_SPAN_MIN_CHARS - 1, -1):
        if working[i].isspace():
            prefix = working[:i].rstrip()
            if len(prefix) >= _LONG_SPAN_MIN_CHARS:
                return i
    return None


def _stream_publishable_prefix(text: str) -> str | None:
    """
    Return the longest stable prefix of `text` safe for a live Telegram
    update, or None when no stable prefix exists yet.

    Stable means the prefix ends at a coherent boundary (paragraph,
    sentence, closed fenced code block, list-item boundary, or a long
    whitespace-aligned span) and its final visible line is not a
    dangling fragment. The helper is pure and deterministic; transport
    callers may invoke it on every streamed update and either publish
    the returned prefix or wait for the next event.
    """
    if not text or not text.strip():
        return None
    working = text.rstrip()
    if not working:
        return None

    candidates: list[tuple[int, str]] = []
    candidates.extend((p, "paragraph") for p in _paragraph_cuts(working))
    candidates.extend((p, "sentence") for p in _sentence_cuts(working))
    candidates.extend((p, "closed_fence") for p in _closed_fence_cuts(working))
    candidates.extend((p, "list_item") for p in _list_item_cuts(working))
    ls = _long_span_cut(working)
    if ls is not None:
        candidates.append((ls, "long_span"))
    candidates.append((len(working), "full"))

    # Evaluate longest-first; when two kinds resolve to the same cut
    # position, prefer the stronger boundary so the dangling-line guard
    # runs in its lighter mode for the chosen candidate.
    candidates.sort(key=lambda pk: (-pk[0], _KIND_PRIORITY[pk[1]]))

    seen_positions: set[int] = set()
    for cut, kind in candidates:
        if cut in seen_positions:
            continue
        seen_positions.add(cut)
        candidate = working[:cut].rstrip()
        if not candidate:
            continue
        if _has_open_fenced_code(candidate):
            continue
        if _has_unbalanced_inline_markdown(candidate):
            continue
        if _has_dangling_final_line(candidate, kind):
            continue
        return candidate
    return None


async def _reply_safe(msg: Message, text: str) -> Message:
    """
    Reply with Markdown formatting, falling back to plain text on parse failure.

    Telegram's Markdown parser is strict about balanced formatting characters.
    Rather than trying to escape everything, we just retry without parse_mode
    if the first attempt fails. Only catches BadRequest (Telegram rejecting the
    markup) - network errors and timeouts propagate normally to avoid sending
    duplicate messages.
    """
    try:
        return await msg.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    except BadRequest:
        return await msg.reply_text(text)


async def _edit_message_safe(msg: Message, text: str) -> bool:
    """
    Edit an existing message with Markdown, falling back to plain text.

    Used during streaming to update the live response message. On BadRequest
    (Telegram rejecting the markup), retries without parse_mode. All other
    errors are silently ignored since edits are best-effort during streaming
    (e.g., message not modified, message deleted by user, network blip). Returns
    whether either edit attempt succeeded so shadow delivery observations can
    describe the outcome without changing this best-effort behavior.
    """
    truncated = _truncate_for_telegram(text)
    try:
        await msg.edit_text(truncated, parse_mode=ParseMode.MARKDOWN)
        return True
    except BadRequest:
        try:
            await msg.edit_text(truncated)
            return True
        except Exception:
            # Editing is best-effort during streaming; log at debug so persistent
            # issues (e.g., revoked bot token) leave a diagnostic trail
            log.debug("Failed to edit message (plain-text fallback)", exc_info=True)
            return False
    except Exception:
        log.debug("Failed to edit message", exc_info=True)
        return False


async def _send_response(update: Update, text: str) -> None:
    """Send a potentially long response as multiple chunked messages."""
    assert update.message is not None
    for chunk in chunk_text(text):
        await _reply_safe(update.message, chunk)


def _get_pool(context: ContextTypes.DEFAULT_TYPE) -> "SubprocessPool":
    """Retrieve the SubprocessPool from bot_data."""
    return context.bot_data["pool"]


# ── Basic command handlers ───────────────────────────────────────────


@_require_auth
async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — the initial greeting when a user first messages the bot."""
    assert update.message is not None
    await update.message.reply_text("Kai is ready. Send me a message.")


async def _end_session(chat_id: int) -> None:
    """Session-end hook: clear the session record.

    The previous Track 1 verification-window flush that used to run
    here is gone. Track 2 extraction (`memory_extraction.extract_and_store`)
    is fire-and-forget per turn and has no end-of-session drain step,
    so the only thing left to do at session end is the session-state
    cleanup itself. Kept as a thin wrapper rather than inlining the
    `sessions.clear_session` call because it has multiple call sites
    (grep `_end_session(`); inlining would touch every one.
    """
    await sessions.clear_session(chat_id)


@_require_auth
async def handle_new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /new — kill the Claude process and start a fresh session.

    Clears the session from the database and kills the subprocess so
    the next message launches a new one.
    """
    assert update.message is not None
    chat_id = _chat_id(update)
    pool = _get_pool(context)
    await pool.restart(chat_id)
    await _end_session(chat_id)
    await update.message.reply_text("Session cleared. Starting fresh.")


# ── Model selection ──────────────────────────────────────────────────


def _backend_name_for_instance(instance: object) -> str:
    """Return a live instance's validated config-key backend name."""
    return require_backend_name(instance)


def _get_user_backend_provider(pool: SubprocessPool, chat_id: int, config: Config) -> tuple[str, str]:
    """Derive the effective (backend, provider) for a user without creating an instance.

    Checks the running instance first; falls back to the canonical
    get_user_backend_and_provider helper so read-only commands avoid
    spawning a subprocess. The pair drives both the model-keyboard
    listing and model validation, so they cannot drift.
    """
    instance = pool.get_if_exists(chat_id)
    if instance:
        return _backend_name_for_instance(instance), instance.provider
    user_config = config.get_user_config(chat_id)
    return get_user_backend_and_provider(user_config, config)


def _get_user_provider(pool: SubprocessPool, chat_id: int, config: Config) -> str:
    """Derive the effective provider for a user without creating an instance.

    Thin wrapper around _get_user_backend_provider, kept for the
    handful of callers that only need the provider value (display in
    /stats, log messages). Backend-aware callers should use
    _get_user_backend_provider directly.
    """
    _, provider = _get_user_backend_provider(pool, chat_id, config)
    return provider


def _get_user_models(pool: SubprocessPool, chat_id: int, config: Config) -> dict[str, str] | None:
    """Get the curated model dict for a user's backend/provider, or None if open-ended.

    Codex installs see CODEX_MODELS; other backends see PROVIDER_MODELS
    for their effective provider. None means open-ended (no keyboard).
    The codex-side surface is fully separate from PROVIDER_MODELS["openai"]
    so a codex install never offers a goose-only model.
    """
    backend, provider = _get_user_backend_provider(pool, chat_id, config)
    models = models_for_backend(backend, provider)
    if models is None and backend != "codex" and backend != "opencode" and provider not in OPEN_ENDED_PROVIDERS:
        # Provider is not open-ended but has no curated list. This means
        # PROVIDER_MODELS is missing an entry for a valid provider -
        # programming oversight, not user error. OpenCode is excluded
        # alongside codex because its None return is intentional (full
        # provider/model IDs are open-ended, not a missing registry
        # entry), and the global provider for opencode is usually
        # empty so the OPEN_ENDED_PROVIDERS check above would not catch
        # it.
        log.warning(
            "Provider '%s' has no curated model list; falling back to text input",
            provider,
        )
    return models


def _models_keyboard(current: str, models: dict[str, str]) -> InlineKeyboardMarkup:
    """Build an inline keyboard with model choices, highlighting the current model."""
    buttons = []
    for key, name in models.items():
        label = f"{name} \U0001f7e2" if key == current else name
        buttons.append([InlineKeyboardButton(label, callback_data=f"model:{key}")])
    return InlineKeyboardMarkup(buttons)


@_require_auth
async def handle_models(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /models - show model selection UI appropriate for the user's provider."""
    assert update.message is not None
    pool = _get_pool(context)
    chat_id = _chat_id(update)
    config: Config = context.bot_data["config"]
    models = _get_user_models(pool, chat_id, config)

    if models is None:
        # Open-ended provider - no keyboard, show current model.
        # Use get_effective_model() so the displayed model reflects
        # persisted settings even before the first message (when no
        # subprocess instance exists yet after a service restart).
        current = await pool.get_effective_model(chat_id)
        await update.message.reply_text(
            f"Current model: {current}\nUse /model <id> to switch to any model your provider supports."
        )
        return

    await update.message.reply_text(
        "Choose a model:",
        reply_markup=_models_keyboard(await pool.get_effective_model(chat_id), models),
    )


async def _switch_model(context: ContextTypes.DEFAULT_TYPE, chat_id: int, model: str) -> None:
    """
    Switch the Claude model, persist the choice, restart the process,
    and clear the session.

    Called by the inline keyboard callback, /model text command, and
    /settings model handler. The model choice is written to the DB so
    it survives restarts (behavior change from session-only).
    """
    pool = _get_pool(context)
    # Resolve the effective workspace BEFORE setting the model. The
    # resolver may switch the live instance into the saved workspace
    # which resets `instance.model` to the default and reapplies any
    # workspace-level override; doing that after set_model would undo
    # the just-set model. With this ordering, set_model operates on
    # the already-restored instance and the model assignment sticks.
    workspace = str(await pool.get_effective_workspace(chat_id))
    pool.set_model(chat_id, model)
    # Persist to settings table so the choice survives restarts
    await sessions.set_user_setting(chat_id, "model", model)
    # Clear any workspace-level model override so the switch actually
    # takes effect. Without this, a prior /workspace config model entry
    # shadows the user setting (workspace config has higher precedence)
    # and the model would revert on the next service restart.
    await sessions.delete_workspace_config_setting(chat_id, workspace, "model")
    await pool.restart(chat_id)
    await _end_session(chat_id)


async def handle_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle inline keyboard model selection.

    Validates authorization and the selected model against the user's
    provider, switches if different from current, and updates the
    keyboard message with confirmation text.
    """
    assert update.callback_query is not None
    query = update.callback_query
    config: Config = context.bot_data["config"]
    if not _is_authorized(config, _user_id(update)):
        await query.answer("Not authorized.")
        return

    assert query.data is not None
    model = query.data.removeprefix("model:")
    pool = _get_pool(context)
    chat_id = _chat_id(update)

    # Validate against the user's effective backend - codex installs
    # check CODEX_MODELS only, no fallback to PROVIDER_MODELS["openai"].
    backend, provider = _get_user_backend_provider(pool, chat_id, config)
    if not validate_model_for_backend(model, backend, provider):
        await query.answer("Invalid model.")
        return

    # Use get_effective_model so the comparison matches the keyboard highlight,
    # which also uses get_effective_model to mark the active model.
    if model == await pool.get_effective_model(chat_id):
        await query.answer()
        await query.edit_message_text("No change.", reply_markup=InlineKeyboardMarkup([]))
        return

    await query.answer()
    await _switch_model(context, chat_id, model)
    models = _get_user_models(pool, chat_id, config)
    display = models.get(model, model) if models else model
    await query.edit_message_text(
        f"Switched to {display}. Session restarted.",
        reply_markup=InlineKeyboardMarkup([]),
    )


@_require_auth
async def handle_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /model <name> - switch model directly via text command."""
    assert update.message is not None
    pool = _get_pool(context)
    chat_id = _chat_id(update)
    config: Config = context.bot_data["config"]

    if not context.args:
        models = _get_user_models(pool, chat_id, config)
        if models:
            opts = " | ".join(sorted(models.keys()))
            await update.message.reply_text(f"Usage: /model <{opts}>")
        else:
            await update.message.reply_text("Usage: /model <model_id>")
        return

    model = context.args[0].lower()

    # Backend-aware validation: codex installs use CODEX_MODELS only;
    # goose / claude delegate to the provider surface.
    backend, provider = _get_user_backend_provider(pool, chat_id, config)
    if not validate_model_for_backend(model, backend, provider):
        valid_models = models_for_backend(backend, provider) or {}
        valid = sorted(valid_models.keys())
        await update.message.reply_text(f"Choose: {', '.join(valid)}")
        return

    await _switch_model(context, chat_id, model)
    models = _get_user_models(pool, chat_id, config)
    display = models.get(model, model) if models else model
    await update.message.reply_text(f"Model set to {display}. Session restarted.")


# ── Per-user settings ──────────────────────────────────────────────


@_require_auth
async def handle_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /settings - view or modify per-user default settings.

    Dispatches to show, set, or reset based on arguments. Settings
    persist in the database and survive restarts.
    """
    assert update.message is not None
    chat_id = _chat_id(update)
    config: Config = context.bot_data["config"]

    # Parse: "/settings [field] [value...]"
    args = context.args or []
    field = args[0].lower() if args else None
    value = args[1] if len(args) > 1 else None

    # /settings - show current
    if field is None:
        await _show_settings(update, context, chat_id, config)
        return

    # /settings reset [field]
    if field == "reset":
        await _handle_settings_reset(update, context, chat_id, config, value)
        return

    # /settings model <name>
    if field == "model":
        pool = _get_pool(context)
        if not value:
            user_models = _get_user_models(pool, chat_id, config)
            if user_models:
                opts = " | ".join(sorted(user_models.keys()))
                await update.message.reply_text(f"Usage: /settings model <{opts}>")
            else:
                await update.message.reply_text("Usage: /settings model <model_id>")
            return

        model_key = value.lower()
        # Backend-aware validation
        backend, provider = _get_user_backend_provider(pool, chat_id, config)
        if not validate_model_for_backend(model_key, backend, provider):
            valid_models = models_for_backend(backend, provider) or {}
            valid = sorted(valid_models.keys())
            await update.message.reply_text(f"Unknown model. Choose from: {', '.join(valid)}")
            return

        # Funnel through _switch_model() - same path as /model and /models
        # keyboard. _switch_model() handles DB write, instance update,
        # process restart, and session clear.
        await _switch_model(context, chat_id, model_key)
        user_models = _get_user_models(pool, chat_id, config)
        display = user_models.get(model_key, model_key) if user_models else model_key
        await update.message.reply_text(f"Default model set to {display}. Session restarted.")
        return

    # /settings timeout <n>
    if field == "timeout":
        if not value:
            await update.message.reply_text("Usage: /settings timeout <seconds>")
            return
        try:
            timeout = int(value)
            if timeout <= 0:
                raise ValueError
        except ValueError:
            await update.message.reply_text("Timeout must be a positive integer (seconds).")
            return
        # Cap at 600s (10 minutes). The timeout is the runaway guard
        # for a stuck subprocess; this cap prevents a single stuck
        # request from holding the per-chat lock indefinitely.
        if timeout > 600:
            await update.message.reply_text("Timeout cannot exceed 600 seconds.")
            return
        await sessions.set_user_setting(chat_id, "timeout", str(timeout))
        # Apply to running instance if one exists. Don't use pool.get()
        # here - it would create a new instance just to set an attribute.
        pool = _get_pool(context)
        instance = pool.get_if_exists(chat_id)
        if instance:
            instance.timeout_seconds = timeout
        await update.message.reply_text(f"Default timeout set to {timeout}s.")
        return

    await update.message.reply_text(f"Unknown setting: {field}\nSettings: model, timeout, reset")


async def _show_settings(update: Update, context: ContextTypes.DEFAULT_TYPE, chat_id: int, config: Config) -> None:
    """Display the user's effective settings with source attribution."""
    assert update.message is not None
    db_settings = await sessions.get_user_settings(chat_id)
    user_config = config.get_user_config(chat_id)

    def _resolve(db_key: str, yaml_val: object, global_val: object, fmt: object) -> tuple[str, str]:
        """Resolve effective value and source for a setting."""
        # Wrap fmt in try/except so a corrupt DB row doesn't crash the
        # display command. Matches the defensive parsing in _restore_workspace.
        if db_key in db_settings:
            try:
                return fmt(db_settings[db_key]), "user override"  # type: ignore[operator]
            except (ValueError, TypeError):
                pass  # fall through to yaml/global
        if yaml_val is not None:
            return fmt(yaml_val), "users.yaml"  # type: ignore[operator]
        return fmt(global_val), "global default"  # type: ignore[operator]

    # Model
    yaml_model = user_config.model if user_config else None
    model, model_src = _resolve("model", yaml_model, config.default_model, str)

    # Timeout
    yaml_timeout = user_config.timeout if user_config else None
    timeout, timeout_src = _resolve("timeout", yaml_timeout, config.default_timeout, lambda v: f"{int(v)}s")

    # Provider info - always show so users know their configuration.
    # Uses the shared helper that checks the running instance first,
    # falling back to config cascade so new users see their provider
    # before any session starts.
    pool = _get_pool(context)
    provider = _get_user_provider(pool, chat_id, config)
    provider_line = f"\n  Provider: {provider}"

    await update.message.reply_text(
        f"Your settings:\n  Model: {model} ({model_src}){provider_line}\n  Timeout: {timeout} ({timeout_src})"
    )


def _revert_instance_field(pool: SubprocessPool, chat_id: int, field: str, config: Config) -> None:
    """
    Write the resolved default value for a single field back onto the
    live ClaudeCodeBackend instance.

    Called before restart so that stale in-memory overrides don't
    persist after a DB entry is deleted. Resolution order mirrors
    _create_instance(): users.yaml > global config.
    """
    instance = pool.get_if_exists(chat_id)
    if not instance:
        return
    user = config.get_user_config(chat_id)
    if field == "model":
        if user and user.model:
            instance.model = user.model
        else:
            # Fall back to provider default, not necessarily config.default_model.
            # If the user's provider differs from the global provider, the
            # global default_model may be invalid for their provider.
            provider = instance.provider
            effective_global = get_effective_provider(config.default_backend, config.default_provider)
            if provider == effective_global:
                instance.model = config.default_model
            else:
                fallback = PROVIDER_DEFAULTS.get(provider, "")
                if not fallback:
                    # Open-ended provider with no default - same warning
                    # as pool.py _create_instance for consistency.
                    log.warning(
                        "No default model for provider '%s'; using global default '%s' which may not be valid",
                        provider,
                        config.default_model,
                    )
                    fallback = config.default_model
                instance.model = fallback
    elif field == "timeout":
        instance.timeout_seconds = user.timeout if user and user.timeout is not None else config.default_timeout


async def _handle_settings_reset(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    config: Config,
    field: str | None,
) -> None:
    """
    Handle /settings reset [field].

    Always restarts the process even for non-flag settings (timeout)
    where a restart isn't strictly necessary. The simplicity of
    "reset always restarts" outweighs the minor overhead of one
    extra process restart during an infrequent operation.
    """
    assert update.message is not None
    valid_fields = {"model", "timeout"}

    if field:
        field = field.lower()
        if field not in valid_fields:
            await update.message.reply_text(f"Unknown field: {field}\nFields: {', '.join(sorted(valid_fields))}")
            return
        await sessions.delete_user_setting(chat_id, field)
        pool = _get_pool(context)
        # Write the resolved default back onto the live instance before
        # restarting. restart() preserves the Python object, so stale
        # in-memory attributes would persist without this step.
        _revert_instance_field(pool, chat_id, field, config)
        await pool.restart(chat_id)
        await _end_session(chat_id)
        await update.message.reply_text(f"Cleared {field} override. Using default. Session restarted.")
    else:
        await sessions.delete_all_user_settings(chat_id)
        pool = _get_pool(context)
        # Revert all fields to their resolved defaults before
        # restarting (same rationale as single-field reset above).
        for f in ("model", "timeout"):
            _revert_instance_field(pool, chat_id, f, config)
        await pool.restart(chat_id)
        await _end_session(chat_id)
        await update.message.reply_text("All settings cleared. Using defaults. Session restarted.")


# ── Voice TTS ────────────────────────────────────────────────────────


def _voices_keyboard(current: str) -> InlineKeyboardMarkup:
    """Build an inline keyboard with voice choices, highlighting the current voice."""
    buttons = []
    for key, name in VOICES.items():
        label = f"{name} \U0001f7e2" if key == current else name
        buttons.append([InlineKeyboardButton(label, callback_data=f"voice:{key}")])
    return InlineKeyboardMarkup(buttons)


# Voice mode options: "off" (text only), "on" (text + voice), "only" (voice only)
_VOICE_MODES = {"off", "on", "only"}
_VOICE_MODE_LABELS = {"off": "OFF", "on": "ON (text + voice)", "only": "ONLY (voice only)"}


@_require_auth
async def handle_voice_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /voice — toggle voice mode or set a specific voice.

    Supports multiple subcommands:
        /voice          — toggle off ↔ only
        /voice on       — enable text + voice mode
        /voice only     — enable voice-only mode (no text)
        /voice off      — disable voice
        /voice <name>   — set a specific voice (enables voice if off)
    """
    assert update.message is not None
    config: Config = context.bot_data["config"]
    if not config.tts_enabled:
        await update.message.reply_text("TTS is not enabled. Set TTS_ENABLED=true in .env")
        return

    chat_id = _chat_id(update)
    current_mode = await sessions.get_setting(f"voice_mode:{chat_id}") or "off"
    current_voice = await sessions.get_setting(f"voice_name:{chat_id}") or DEFAULT_VOICE

    if context.args:
        arg = context.args[0].lower()
        if arg in _VOICE_MODES:
            # /voice on|only|off — set mode directly
            await sessions.set_setting(f"voice_mode:{chat_id}", arg)
            await update.message.reply_text(f"Voice mode: {_VOICE_MODE_LABELS[arg]} (voice: {VOICES[current_voice]})")
        elif arg in VOICES:
            # /voice <name> — set voice (enable in current mode, or default to "only")
            await sessions.set_setting(f"voice_name:{chat_id}", arg)
            if current_mode == "off":
                await sessions.set_setting(f"voice_mode:{chat_id}", "only")
                current_mode = "only"
            await update.message.reply_text(
                f"Voice set to {VOICES[arg]}. Voice mode: {_VOICE_MODE_LABELS[current_mode]}"
            )
        else:
            names = ", ".join(VOICES.keys())
            await update.message.reply_text(
                f"Unknown voice or mode. Usage:\n"
                f"/voice on — text + voice\n"
                f"/voice only — voice only\n"
                f"/voice off — text only\n"
                f"/voice <name> — set voice\n\n"
                f"Voices: {names}"
            )
    else:
        # /voice — toggle: off → only → off
        new_mode = "off" if current_mode != "off" else "only"
        await sessions.set_setting(f"voice_mode:{chat_id}", new_mode)
        await update.message.reply_text(f"Voice mode: {_VOICE_MODE_LABELS[new_mode]} (voice: {VOICES[current_voice]})")


@_require_auth
async def handle_voices(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /voices — show an inline keyboard of available TTS voices."""
    assert update.message is not None
    config: Config = context.bot_data["config"]
    if not config.tts_enabled:
        await update.message.reply_text("TTS is not enabled. Set TTS_ENABLED=true in .env")
        return

    chat_id = _chat_id(update)
    current_voice = await sessions.get_setting(f"voice_name:{chat_id}") or DEFAULT_VOICE
    await update.message.reply_text(
        "Choose a voice:",
        reply_markup=_voices_keyboard(current_voice),
    )


async def handle_voice_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle inline keyboard voice selection.

    Sets the chosen voice in settings and auto-enables voice mode if it
    was off (defaults to "only" mode).
    """
    assert update.callback_query is not None
    query = update.callback_query
    config: Config = context.bot_data["config"]
    if not _is_authorized(config, _user_id(update)):
        await query.answer("Not authorized.")
        return

    assert query.data is not None
    voice = query.data.removeprefix("voice:")
    if voice not in VOICES:
        await query.answer("Invalid voice.")
        return

    chat_id = _chat_id(update)
    current_voice = await sessions.get_setting(f"voice_name:{chat_id}") or DEFAULT_VOICE

    if voice == current_voice:
        await query.answer()
        await query.edit_message_text("No change.", reply_markup=InlineKeyboardMarkup([]))
        return

    current_mode = await sessions.get_setting(f"voice_mode:{chat_id}") or "off"
    await sessions.set_setting(f"voice_name:{chat_id}", voice)
    # Auto-enable voice if it was off
    if current_mode == "off":
        await sessions.set_setting(f"voice_mode:{chat_id}", "only")
        current_mode = "only"
    await query.answer()
    await query.edit_message_text(
        f"Voice set to {VOICES[voice]}. Voice mode: {_VOICE_MODE_LABELS[current_mode]}",
        reply_markup=InlineKeyboardMarkup([]),
    )


# ── Info and management commands ─────────────────────────────────────


@_require_auth
async def handle_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stats: show session info, model, and process status."""
    assert update.message is not None
    chat_id = _chat_id(update)
    pool = _get_pool(context)
    stats = await sessions.get_stats(chat_id)
    alive = pool.is_alive(chat_id)
    if not stats:
        await update.message.reply_text(f"No active session.\nProcess alive: {alive}")
        return
    await update.message.reply_text(
        f"Session: {stats['session_id'][:8]}...\n"
        f"Model: {stats['model']}\n"
        f"Started: {stats['created_at']}\n"
        f"Last used: {stats['last_used_at']}\n"
        f"Process alive: {alive}"
    )


async def _list_jobs(update: Update, chat_id: int) -> None:
    """
    List all active scheduled jobs for a chat.

    Formats each job with an emoji tag (bell for reminders, robot for Claude
    jobs), the job ID, name, and a human-readable schedule description.
    Shared by /job (list branch) and /jobs (alias).
    """
    assert update.message is not None
    jobs = await sessions.get_jobs(chat_id)
    if not jobs:
        await update.message.reply_text("No active scheduled jobs.")
        return
    lines = []
    for j in jobs:
        type_tag = "\U0001f514" if j["job_type"] == "reminder" else "\U0001f916"
        lines.append(f"{type_tag} #{j['id']} {j['name']} ({_format_schedule(j)})")
    await update.message.reply_text("Active jobs:\n" + "\n".join(lines))


def _format_schedule(job: dict) -> str:
    """Format a job's schedule as a human-readable string."""
    sched = job["schedule_type"]
    try:
        data = json.loads(job["schedule_data"])
    except (json.JSONDecodeError, TypeError):
        # Malformed schedule_data - fall back to raw schedule type name
        return sched
    if sched == "once":
        return f"once at {data.get('run_at', '?')}"
    if sched == "interval":
        secs = data.get("seconds", 0)
        if secs >= 3600:
            return f"every {secs // 3600}h"
        if secs >= 60:
            return f"every {secs // 60}m"
        return f"every {secs}s"
    if sched == "daily":
        times = data.get("times", [])
        return f"daily at {', '.join(times)} UTC" if times else "daily"
    return sched


@_require_auth
async def handle_job(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /job - manage scheduled jobs.

    Subcommands:
        /job             - list all active jobs (same as /job list)
        /job list        - list all active jobs
        /job info <id>   - show details for a specific job
        /job cancel <id> - cancel (delete) a job
    """
    assert update.message is not None
    chat_id = _chat_id(update)
    args = context.args or []
    subcommand = args[0].lower() if args else None

    # No subcommand or "list": show all jobs
    if subcommand is None or subcommand == "list":
        await _list_jobs(update, chat_id)
        return

    if subcommand == "info":
        if len(args) < 2:
            await update.message.reply_text("Usage: /job info <id>")
            return
        try:
            job_id = int(args[1])
        except ValueError:
            await update.message.reply_text("Job ID must be a number.")
            return
        job = await sessions.get_job_by_id(job_id)
        # Ownership check: only show jobs belonging to this chat
        if not job or job["chat_id"] != chat_id:
            await update.message.reply_text(f"Job #{job_id} not found.")
            return
        auto_remove = "yes" if job["auto_remove"] else "no"
        await update.message.reply_text(
            f"Job #{job['id']} - {job['name']}\n"
            f"Type: {job['job_type']}\n"
            f"Schedule: {_format_schedule(job)}\n"
            f"Auto-remove: {auto_remove}\n"
            f"\n"
            f"Prompt:\n"
            f"{job['prompt']}"
        )
        return

    if subcommand == "cancel":
        if len(args) < 2:
            await update.message.reply_text("Usage: /job cancel <id>")
            return
        try:
            job_id = int(args[1])
        except ValueError:
            await update.message.reply_text("Job ID must be a number.")
            return
        # Pass user's chat_id for ownership check - users can only cancel
        # their own jobs (prevents cross-user job manipulation).
        deleted = await sessions.delete_job(job_id, chat_id=chat_id)
        if not deleted:
            await update.message.reply_text(f"Job #{job_id} not found.")
            return
        # Remove from APScheduler's in-memory queue. Daily jobs with multiple
        # times get suffixed names (cron_19_0, cron_19_1), so match both the
        # exact name and any suffixed variants - same pattern as cron.py.
        jq = context.application.job_queue
        assert jq is not None
        prefix = f"cron_{job_id}"
        current = [j for j in jq.jobs() if j.name == prefix or (j.name and j.name.startswith(f"{prefix}_"))]
        for j in current:
            j.schedule_removal()
        await update.message.reply_text(f"Job #{job_id} cancelled.")
        return

    # Unknown subcommand
    await update.message.reply_text(
        "Usage:\n/job - List all jobs\n/job info <id> - Show job details\n/job cancel <id> - Cancel a job"
    )


@_require_auth
async def handle_jobs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Alias for /job list. Kept for backward compatibility."""
    assert update.message is not None
    await _list_jobs(update, _chat_id(update))


@_require_auth
async def handle_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /stop — abort the current Claude response.

    Sets the per-chat stop event (checked by the streaming loop) and kills
    the Claude process immediately. The streaming loop in _handle_response()
    sees the stop event and appends "(stopped)" to the live message.
    """
    assert update.message is not None
    chat_id = _chat_id(update)
    pool = _get_pool(context)
    stop_event = get_stop_event(chat_id)
    stop_event.set()
    await pool.force_kill(chat_id)
    await update.message.reply_text("Stopping...")


# ── Workspace management ─────────────────────────────────────────────


def _resolve_workspace_path(target: str, base: Path | None) -> Path | None:
    """
    Resolve a workspace name to an absolute path under the base directory.

    Only relative names are allowed (e.g., "my-project", not "/tmp/evil").
    Returns None if no base is set or if the resolved path would escape
    the base directory (path traversal prevention).

    Args:
        target: The workspace name or relative path.
        base: The WORKSPACE_BASE directory, or None if unset.

    Returns:
        The resolved absolute path, or None if invalid.
    """
    if not base:
        return None
    # expanduser() handles ~ in the target path (e.g., "~/Projects/foo")
    resolved = (base / target).expanduser().resolve()
    # Resolve base too so symlinks in the base path don't bypass the check
    resolved_base = base.resolve()
    if not str(resolved).startswith(str(resolved_base) + "/") and resolved != resolved_base:
        return None
    return resolved


def _short_workspace_name(path: str, base: Path | None) -> str:
    """
    Shorten a workspace path for display in Telegram messages and keyboards.

    If the path is under WORKSPACE_BASE, strips the base prefix to show just
    the relative name. Otherwise falls back to showing just the directory name.
    """
    base_str = str(base) if base else None
    if base_str and path.startswith(base_str.rstrip("/") + "/"):
        return path[len(base_str.rstrip("/")) + 1 :]
    return Path(path).name


def _workspace_config_suffix(ws_config: WorkspaceConfig | None) -> str:
    """Build a parenthesized suffix showing workspace config details.

    Returns e.g. " (model: opus)" or "" if no config.
    """
    extras = []
    if ws_config and ws_config.model:
        extras.append(f"model: {ws_config.model}")
    return f" ({', '.join(extras)})" if extras else ""


async def _do_switch_workspace(context: ContextTypes.DEFAULT_TYPE, chat_id: int, path: Path) -> WorkspaceConfig | None:
    """
    Core workspace switch logic shared by command and callback handlers.

    Kills the Claude process (it will restart in the new directory on next
    message), clears the session, and persists the new workspace to settings.
    Switching to home deletes the setting (home is the default). Looks up
    per-workspace config from workspaces.yaml and passes it to Claude.

    Returns the WorkspaceConfig for the target workspace (or None) so
    callers can display config details without a redundant lookup.
    """
    pool = _get_pool(context)
    config: Config = context.bot_data["config"]
    # "home" for this user is per-user post-#353. Same resolver pool.py
    # uses, so the equality check below matches the directory the user
    # would land in on `/workspace home` or session restart.
    home = resolve_home_workspace(chat_id, config)

    # Layer DB overrides (from /workspace config) on top of YAML baseline.
    yaml_config = config.get_workspace_config(path)
    ws_config = await sessions.build_workspace_config(yaml_config, path, chat_id)
    await pool.change_workspace(chat_id, path, workspace_config=ws_config)
    # Per-user file confinement is handled at request time in webhook.py
    # via pool.get_effective_workspace(chat_id), so no global update needed
    # here.
    await _end_session(chat_id)

    if path == home:
        await sessions.delete_setting(f"workspace:{chat_id}")
    else:
        await sessions.set_setting(f"workspace:{chat_id}", str(path))
        await sessions.upsert_workspace_history(str(path), chat_id)

    return ws_config


async def _switch_workspace(update: Update, context: ContextTypes.DEFAULT_TYPE, path: Path) -> None:
    """
    Switch to a workspace path and send a confirmation reply.

    Wraps _do_switch_workspace with user-facing feedback including workspace
    metadata (git repo detection, CLAUDE.md presence).
    """
    assert update.message is not None
    pool = _get_pool(context)
    config: Config = context.bot_data["config"]
    # Per-user resolution: this is what `/workspace home` would route to
    # for THIS user, not a shared default.
    home = resolve_home_workspace(_chat_id(update), config)

    if path == await pool.get_effective_workspace(_chat_id(update)):
        await update.message.reply_text("Already in that workspace.")
        return

    # Guard against directories deleted after startup (matches keyboard path behavior)
    if not path.is_dir():
        await update.message.reply_text("That workspace no longer exists.")
        return

    ws_config = await _do_switch_workspace(context, _chat_id(update), path)

    config_suffix = _workspace_config_suffix(ws_config)

    if path == home:
        await update.message.reply_text(f"Switched to home workspace{config_suffix}. Session cleared.")
    else:
        # Show filesystem metadata alongside config details
        notes = []
        if (path / ".git").is_dir():
            notes.append("Git repo")
        if (path / ".claude" / "CLAUDE.md").exists():
            notes.append("Has CLAUDE.md")
        note_suffix = f" ({', '.join(notes)})" if notes else ""
        await update.message.reply_text(f"Workspace: {path}{note_suffix}{config_suffix}\nSession cleared.")


def _workspaces_keyboard(
    history: list[dict],
    current_path: str,
    home_path: str,
    base: Path | None,
    allowed_workspaces: list[Path],
) -> InlineKeyboardMarkup:
    """
    Build an inline keyboard for workspace switching.

    Layout (top to bottom):
    1. Home button (always first)
    2. Allowed (pinned) workspaces from ALLOWED_WORKSPACES config, in order
    3. Recent workspace history, deduplicated against allowed workspaces and home

    The current workspace is marked with a green dot. Callback data:
    - "ws:home" for the home button
    - "ws:allowed:<index>" for pinned workspaces (index into allowed_workspaces)
    - "ws:<index>" for history entries (index into the history list)
    """
    buttons = []

    # Collect allowed paths as strings for deduplication checks below
    allowed_path_strs = {str(p) for p in allowed_workspaces}

    # Home button (always first)
    home_label = "\U0001f3e0 Home"
    if current_path == home_path:
        home_label += " \U0001f7e2"
    buttons.append([InlineKeyboardButton(home_label, callback_data="ws:home")])

    # Detect name collisions within the allowed list so labels can be disambiguated.
    # If two entries share the same directory name, show "parent/name" instead of "name".
    name_counts: dict[str, int] = {}
    for p in allowed_workspaces:
        name_counts[p.name] = name_counts.get(p.name, 0) + 1
    duplicate_names = {name for name, count in name_counts.items() if count > 1}

    # Pinned workspaces from ALLOWED_WORKSPACES (shown above history)
    for i, p in enumerate(allowed_workspaces):
        if p.name in duplicate_names:
            # Include parent directory name to make the button unambiguous
            short = f"{p.parent.name}/{p.name}"
        else:
            short = _short_workspace_name(str(p), base)
        label = short
        if str(p) == current_path:
            label += " \U0001f7e2"
        buttons.append([InlineKeyboardButton(label, callback_data=f"ws:allowed:{i}")])

    # History entries — skip home and any path already shown in the allowed section
    for i, entry in enumerate(history):
        p = entry["path"]
        if p == home_path or p in allowed_path_strs:
            continue
        short = _short_workspace_name(p, base)
        label = short
        if p == current_path:
            label += " \U0001f7e2"
        buttons.append([InlineKeyboardButton(label, callback_data=f"ws:{i}")])

    return InlineKeyboardMarkup(buttons)


# ── Workspace config (/workspace config) ───────────────────────────


async def _handle_workspace_config(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    target: str,
) -> None:
    """
    Handle /workspace config subcommands.

    Dispatches to show, set, or reset workspace configuration fields.
    All changes apply to the current workspace.
    """
    assert update.message is not None
    chat_id = _chat_id(update)
    pool = _get_pool(context)
    config: Config = context.bot_data["config"]
    workspace = await pool.get_effective_workspace(chat_id)
    workspace_str = str(workspace)

    # Parse: "config [field] [value...]"
    parts = target.split(None, 2)  # ["config"], ["config", field], or ["config", field, value]
    field = parts[1].lower() if len(parts) > 1 else None
    value = parts[2] if len(parts) > 2 else None

    # /workspace config - show current settings
    if field is None:
        await _show_workspace_config(update, workspace, config)
        return

    # /workspace config reset [field]
    if field == "reset":
        if value:
            field_to_reset = value.lower()
            await sessions.delete_workspace_config_setting(chat_id, workspace_str, field_to_reset)
            await _apply_config_change(context, chat_id, workspace, config)
            await update.message.reply_text(f"{field_to_reset} reset to default.")
        else:
            await sessions.delete_all_workspace_config(chat_id, workspace_str)
            await _apply_config_change(context, chat_id, workspace, config)
            await update.message.reply_text("All workspace config cleared. Using global defaults.")
        return

    # /workspace config model <name>
    if field == "model":
        pool = _get_pool(context)
        if not value:
            user_models = _get_user_models(pool, chat_id, config)
            if user_models:
                opts = " | ".join(sorted(user_models.keys()))
                await update.message.reply_text(f"Usage: /workspace config model <{opts}>")
            else:
                await update.message.reply_text("Usage: /workspace config model <model_id>")
            return
        # Backend-aware validation
        backend, provider = _get_user_backend_provider(pool, chat_id, config)
        if not validate_model_for_backend(value.lower(), backend, provider):
            valid_models = models_for_backend(backend, provider) or {}
            valid = sorted(valid_models.keys())
            await update.message.reply_text(f"Unknown model. Choose from: {', '.join(valid)}")
            return
        await sessions.set_workspace_config_setting(chat_id, workspace_str, "model", value.lower())
        await _apply_config_change(context, chat_id, workspace, config)
        await update.message.reply_text(f"Model set to {value.lower()}.")
        return

    # /workspace config timeout <n>
    if field == "timeout":
        if not value:
            await update.message.reply_text("Usage: /workspace config timeout <seconds>")
            return
        try:
            timeout = int(value)
            if timeout <= 0:
                raise ValueError
        except ValueError:
            await update.message.reply_text("Timeout must be a positive integer (seconds).")
            return
        await sessions.set_workspace_config_setting(chat_id, workspace_str, "timeout", str(timeout))
        await _apply_config_change(context, chat_id, workspace, config)
        await update.message.reply_text(f"Timeout set to {timeout}s.")
        return

    # /workspace config env [KEY=VALUE | -KEY]
    if field == "env":
        changed = await _handle_workspace_env(update, chat_id, workspace_str, value)
        if changed:
            await _apply_config_change(context, chat_id, workspace, config)
        return

    # /workspace config prompt [text | clear]
    if field == "prompt":
        changed = await _handle_workspace_prompt(update, chat_id, workspace_str, value)
        if changed:
            await _apply_config_change(context, chat_id, workspace, config)
        return

    await update.message.reply_text(f"Unknown config field: {field}\nFields: model, timeout, env, prompt, reset")


async def _show_workspace_config(
    update: Update,
    workspace: Path,
    config: Config,
) -> None:
    """Display the effective config for the current workspace with source."""
    assert update.message is not None
    chat_id = _chat_id(update)
    yaml_config = config.get_workspace_config(workspace)
    db_settings = await sessions.get_workspace_config_settings(chat_id, str(workspace))

    # Fetch user-level settings so the display reflects the full
    # precedence chain: workspace DB > workspaces.yaml > user DB >
    # users.yaml > global default. Without these, fields that the user
    # set via /settings or /model would show "global default" instead
    # of the actual effective value.
    user_settings = await sessions.get_user_settings(chat_id)
    user_config = config.get_user_config(chat_id)

    lines = [f"Config for {workspace.name}:"]

    # Model: workspace DB > workspaces.yaml > user DB > users.yaml > global
    if "model" in db_settings:
        model, model_src = db_settings["model"], "workspace override"
    elif yaml_config and yaml_config.model:
        model, model_src = yaml_config.model, "workspaces.yaml"
    elif "model" in user_settings:
        model, model_src = user_settings["model"], "user setting"
    elif user_config and user_config.model:
        model, model_src = user_config.model, "users.yaml"
    else:
        model, model_src = config.default_model, "global default"
    lines.append(f"  Model: {model} ({model_src})")

    # Timeout: workspace DB > workspaces.yaml > user DB > users.yaml > global
    try:
        if "timeout" in db_settings:
            timeout, timeout_src = int(db_settings["timeout"]), "workspace override"
        elif yaml_config and yaml_config.timeout is not None:
            timeout, timeout_src = yaml_config.timeout, "workspaces.yaml"
        elif "timeout" in user_settings:
            timeout, timeout_src = int(user_settings["timeout"]), "user setting"
        elif user_config and user_config.timeout is not None:
            timeout, timeout_src = user_config.timeout, "users.yaml"
        else:
            timeout, timeout_src = config.default_timeout, "global default"
        lines.append(f"  Timeout: {timeout}s ({timeout_src})")
    except (ValueError, TypeError):
        lines.append("  Timeout: (corrupted - reset with /workspace config reset timeout)")

    # Env vars (show keys only, not values - may contain secrets)
    env_keys: list[str] = []
    if yaml_config and yaml_config.env:
        env_keys.extend(yaml_config.env.keys())
    env_corrupted = False
    if "env" in db_settings:
        try:
            db_env = json.loads(db_settings["env"])
            env_keys.extend(k for k in db_env if k not in env_keys)
        except json.JSONDecodeError:
            env_corrupted = True
    if env_corrupted:
        lines.append("  Env vars: (DB override corrupted - reset to clear)")
    elif env_keys:
        lines.append(f"  Env vars: {', '.join(sorted(env_keys))}")

    # System prompt
    prompt = db_settings.get("prompt")
    if prompt:
        preview = prompt[:100] + ("..." if len(prompt) > 100 else "")
        lines.append(f"  Prompt: {preview} (workspace override)")
    elif yaml_config and yaml_config.system_prompt:
        preview = yaml_config.system_prompt[:100]
        if len(yaml_config.system_prompt) > 100:
            preview += "..."
        lines.append(f"  Prompt: {preview} (workspaces.yaml)")

    await update.message.reply_text("\n".join(lines))


async def _handle_workspace_env(
    update: Update,
    chat_id: int,
    workspace_str: str,
    value: str | None,
) -> bool:
    """Handle /workspace config env subcommands. Returns True if config changed."""
    assert update.message is not None

    # Load existing env vars from the database
    settings = await sessions.get_workspace_config_settings(chat_id, workspace_str)
    env: dict[str, str] = {}
    if "env" in settings:
        try:
            env = json.loads(settings["env"])
        except json.JSONDecodeError:
            # Corrupted entry; start fresh
            env = {}

    # /workspace config env - list current vars
    if not value:
        if not env:
            await update.message.reply_text("No workspace env vars set.")
        else:
            # Show keys only for security
            key_lines = [f"  {k}" for k in sorted(env.keys())]
            await update.message.reply_text("Workspace env vars:\n" + "\n".join(key_lines))
        return False

    # /workspace config env -KEY - remove a var
    if value.startswith("-"):
        key = value[1:]
        if key in env:
            del env[key]
            if env:
                await sessions.set_workspace_config_setting(chat_id, workspace_str, "env", json.dumps(env))
            else:
                await sessions.delete_workspace_config_setting(chat_id, workspace_str, "env")
            await update.message.reply_text(f"Removed {key}.")
            return True
        await update.message.reply_text(f"{key} is not set.")
        return False

    # /workspace config env KEY=VALUE - set a var
    if "=" not in value:
        await update.message.reply_text("Usage: /workspace config env KEY=VALUE")
        return False

    key, val = value.split("=", 1)
    key = key.strip()
    if not key:
        await update.message.reply_text("Key cannot be empty.")
        return False

    env[key] = val
    await sessions.set_workspace_config_setting(chat_id, workspace_str, "env", json.dumps(env))
    await update.message.reply_text(f"Set {key}.")
    return True


async def _handle_workspace_prompt(
    update: Update,
    chat_id: int,
    workspace_str: str,
    value: str | None,
) -> bool:
    """Handle /workspace config prompt subcommands. Returns True if config changed."""
    assert update.message is not None

    # Check for file attachment (caption-based prompt).
    # When a document is attached, Telegram puts the command text in
    # update.message.caption, not update.message.text. CommandHandler
    # handles caption-based dispatch, so the handler fires, but
    # context.args is populated from the caption.
    if update.message.document:
        # Reject oversized files before downloading into memory.
        # Telegram allows up to 20 MB; 100 KB is generous for a
        # system prompt that will be stored in SQLite.
        max_prompt_bytes = 100 * 1024
        file_size = update.message.document.file_size
        if file_size and file_size > max_prompt_bytes:
            await update.message.reply_text(f"File too large ({file_size // 1024}KB). Max prompt file size is 100KB.")
            return False
        file = await update.message.document.get_file()
        raw = await file.download_as_bytearray()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            await update.message.reply_text("File must be UTF-8 text.")
            return False
        await sessions.set_workspace_config_setting(chat_id, workspace_str, "prompt", content.strip())
        await update.message.reply_text(f"Prompt set from file ({len(content)} chars).")
        return True

    # /workspace config prompt (no value) - show current
    if not value:
        settings = await sessions.get_workspace_config_settings(chat_id, workspace_str)
        prompt = settings.get("prompt")
        if prompt:
            await update.message.reply_text(f"Current prompt:\n{prompt}")
        else:
            await update.message.reply_text("No workspace prompt set.")
        return False

    # /workspace config prompt clear
    if value.strip().lower() == "clear":
        await sessions.delete_workspace_config_setting(chat_id, workspace_str, "prompt")
        await update.message.reply_text("Prompt cleared.")
        return True

    # /workspace config prompt <text>
    await sessions.set_workspace_config_setting(chat_id, workspace_str, "prompt", value.strip())
    await update.message.reply_text("Prompt set.")
    return True


async def _apply_config_change(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    workspace: Path,
    config: Config,
) -> None:
    """
    Rebuild and apply workspace config after a setting change.

    Kills the current Claude process so the next message starts fresh
    with the new config. Reuses change_workspace() since it handles
    the full reset-then-override cycle.
    """
    pool = _get_pool(context)
    yaml_config = config.get_workspace_config(workspace)
    ws_config = await sessions.build_workspace_config(yaml_config, workspace, chat_id)
    await pool.change_workspace(chat_id, workspace, workspace_config=ws_config)


@_require_auth
async def handle_workspaces(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /workspaces - show an inline keyboard of recent workspaces."""
    assert update.message is not None
    chat_id = _chat_id(update)
    config: Config = context.bot_data["config"]
    # Resolve per-user workspace access (workspace_base + allowed list)
    base, allowed = await sessions.resolve_workspace_access(chat_id, config)
    history = await sessions.get_workspace_history(chat_id)
    pool = _get_pool(context)
    current = str(await pool.get_effective_workspace(chat_id))
    # Per-user home for the listing's "Home" pin and the empty-state
    # short-circuit below.
    home = str(resolve_home_workspace(chat_id, config))

    if not history and not allowed and current == home:
        await update.message.reply_text("No workspace history yet.\nUse /workspace new <name> to create one.")
        return

    keyboard = _workspaces_keyboard(history, current, home, base, allowed)
    await update.message.reply_text("Workspaces:", reply_markup=keyboard)


async def handle_workspace_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle inline keyboard workspace selection.

    Resolves the selected workspace from the callback data, validates it
    still exists, switches to it, and updates the keyboard message.
    Removes stale entries from history if the directory no longer exists.
    """
    assert update.callback_query is not None
    query = update.callback_query
    chat_id = _chat_id(update)
    config: Config = context.bot_data["config"]
    if not _is_authorized(config, _user_id(update)):
        await query.answer("Not authorized.")
        return

    assert query.data is not None
    data = query.data.removeprefix("ws:")
    pool = _get_pool(context)
    # Per-user resolution: the "Home" keyboard button maps to THIS user's
    # home directory, not a shared one.
    home = resolve_home_workspace(chat_id, config)

    # Resolve per-user workspace access for this user
    base, allowed = await sessions.resolve_workspace_access(chat_id, config)

    # Resolve target path from callback data
    if data == "home":
        path = home
        label = "Home"
    elif data.startswith("allowed:"):
        # Pinned workspace from the user's effective allowed list
        try:
            idx = int(data.removeprefix("allowed:"))
        except ValueError:
            await query.answer("Invalid selection.")
            await query.edit_message_text("No change.", reply_markup=InlineKeyboardMarkup([]))
            return
        if idx < 0 or idx >= len(allowed):
            await query.answer("Workspace no longer available.")
            await query.edit_message_text("No change.", reply_markup=InlineKeyboardMarkup([]))
            return
        path = allowed[idx]
        if not path.is_dir():
            await query.answer("That workspace no longer exists.")
            await query.edit_message_text("No change.", reply_markup=InlineKeyboardMarkup([]))
            return
        label = _short_workspace_name(str(path), base)
    else:
        try:
            idx = int(data)
        except ValueError:
            await query.answer("Invalid selection.")
            await query.edit_message_text("No change.", reply_markup=InlineKeyboardMarkup([]))
            return
        history = await sessions.get_workspace_history(chat_id)
        if idx < 0 or idx >= len(history):
            await query.answer("Workspace no longer in history.")
            await query.edit_message_text("No change.", reply_markup=InlineKeyboardMarkup([]))
            return
        path = Path(history[idx]["path"])
        # Reject history entries that are no longer in an allowed workspace
        # source. This handles the case where a path was removed from the
        # user's allowed list after they visited it - the history entry
        # persists but access is revoked.
        if not is_workspace_allowed(path, base, allowed):
            await sessions.delete_workspace_history(str(path), chat_id)
            await query.answer("That workspace is no longer allowed.")
            history = await sessions.get_workspace_history(chat_id)
            keyboard = _workspaces_keyboard(
                history, str(await pool.get_effective_workspace(chat_id)), str(home), base, allowed
            )
            await query.edit_message_reply_markup(reply_markup=keyboard)
            return
        # Remove stale entries where the directory no longer exists
        if not path.is_dir():
            await sessions.delete_workspace_history(str(path), chat_id)
            await query.answer("That workspace no longer exists.")
            history = await sessions.get_workspace_history(chat_id)
            keyboard = _workspaces_keyboard(
                history, str(await pool.get_effective_workspace(chat_id)), str(home), base, allowed
            )
            await query.edit_message_reply_markup(reply_markup=keyboard)
            return
        label = _short_workspace_name(str(path), base)

    # Already there — dismiss the keyboard
    if path == await pool.get_effective_workspace(chat_id):
        await query.answer()
        await query.edit_message_text("No change.", reply_markup=InlineKeyboardMarkup([]))
        return

    # Switch and confirm, showing any per-workspace config details
    await query.answer()
    ws_config = await _do_switch_workspace(context, _chat_id(update), path)
    suffix = _workspace_config_suffix(ws_config)
    await query.edit_message_text(
        f"Switched to {label}{suffix}. Session cleared.",
        reply_markup=InlineKeyboardMarkup([]),
    )


# ── Workspace allow/deny/allowed ────────────────────────────────────


async def _handle_workspace_allow(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    target: str,
) -> None:
    """Handle /workspace allow <path> - add an allowed workspace."""
    assert update.message is not None
    chat_id = _chat_id(update)
    config: Config = context.bot_data["config"]

    # Parse path from target string ("allow /path/to/dir")
    parts = target.split(None, 1)
    if len(parts) < 2:
        await update.message.reply_text("Usage: /workspace allow <path>")
        return

    raw_path = parts[1].strip()

    # Require fully absolute path. Reject ~ because expanduser() resolves
    # to the bot process's $HOME, not the requesting user's home directory.
    # In multi-user with separate os_user values, ~/projects would point to
    # the wrong location. Requiring / avoids the ambiguity entirely.
    if not raw_path.startswith("/"):
        await update.message.reply_text("Path must be absolute (start with /).")
        return

    # Resolve to canonical form
    resolved = Path(raw_path).resolve()

    # Must exist and be a directory
    if not resolved.is_dir():
        await update.message.reply_text(f"Not a directory: {resolved}")
        return

    # Check for redundancy: already under workspace_base?
    base, allowed = await sessions.resolve_workspace_access(chat_id, config)
    if base:
        resolved_base = base.resolve()
        if str(resolved).startswith(str(resolved_base) + "/") or resolved == resolved_base:
            await update.message.reply_text(
                f"Already covered by your workspace base:\n{base}\n\n"
                "Use /workspace <name> to access directories under it."
            )
            return

    # Check for duplicates (already in the effective list)
    # allowed list is pre-resolved by resolve_workspace_access()
    if resolved in allowed:
        await update.message.reply_text("Already in your allowed list.")
        return

    await sessions.add_allowed_workspace(chat_id, str(resolved))
    await update.message.reply_text(f"Added: {resolved}")


async def _handle_workspace_deny(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    target: str,
) -> None:
    """Handle /workspace deny <path> - remove an allowed workspace."""
    assert update.message is not None
    chat_id = _chat_id(update)

    parts = target.split(None, 1)
    if len(parts) < 2:
        await update.message.reply_text("Usage: /workspace deny <path>")
        return

    raw_path = parts[1].strip()

    # Same absolute-path requirement as _handle_workspace_allow:
    # relative paths resolve against cwd and will never match a stored
    # entry, producing a confusing "not in your list" response.
    if not raw_path.startswith("/"):
        await update.message.reply_text("Path must be absolute (start with /).")
        return

    resolved = Path(raw_path).resolve()

    # Check if this is a user-added path (in the database)
    removed = await sessions.remove_allowed_workspace(chat_id, str(resolved))
    if removed:
        await update.message.reply_text(f"Removed: {resolved}")
    else:
        # Check if it's a global entry (can't be removed via Telegram)
        config: Config = context.bot_data["config"]
        if resolved in [p.resolve() for p in config.allowed_workspaces]:
            await update.message.reply_text("That workspace is configured globally and cannot be removed via Telegram.")
        else:
            await update.message.reply_text("Not in your allowed workspace list.")


async def _handle_workspace_allowed(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle /workspace allowed - list all allowed workspaces."""
    assert update.message is not None
    chat_id = _chat_id(update)
    config: Config = context.bot_data["config"]
    user_config = config.get_user_config(chat_id)

    # Resolve workspace_base: users.yaml > env
    base = user_config.workspace_base if user_config and user_config.workspace_base else config.workspace_base

    # Build allowed list with source attribution. Avoids calling
    # resolve_workspace_access() + get_allowed_workspaces() which
    # would query the DB twice. Only this handler needs attribution.
    db_paths = await sessions.get_allowed_workspaces(chat_id)
    db_path_set = {p.resolve() for p in db_paths}
    # Per-user yaml allowed_workspaces tier (#460). Tracked
    # separately from db_path_set so the source attribution
    # below can distinguish "your yaml" from "your DB entries".
    yaml_path_set: set[Path] = set()
    if user_config and user_config.allowed_workspaces:
        yaml_path_set = {p.resolve() for p in user_config.allowed_workspaces}

    # Combined list: DB > yaml-per-user > global. Same order as
    # resolve_workspace_access so this view matches what name
    # resolution actually traverses.
    seen: set[Path] = set()
    allowed: list[Path] = []
    for p in db_paths:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            allowed.append(resolved)
    if user_config and user_config.allowed_workspaces:
        for p in user_config.allowed_workspaces:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                allowed.append(resolved)
    for p in config.allowed_workspaces:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            allowed.append(resolved)

    lines = []
    if base:
        lines.append(f"Workspace base: {base}")
    else:
        lines.append("Workspace base: not set")

    if allowed:
        lines.append("")
        lines.append("Allowed workspaces:")
        for p in allowed:
            # Source attribution priority matches the union order:
            # DB beats yaml beats global. A path could in principle
            # appear in multiple tiers (operator pinned via
            # /workspace allow AND listed in users.yaml); the most
            # specific tier wins the label.
            if p in db_path_set:
                source = "you"
            elif p in yaml_path_set:
                source = "yaml"
            else:
                source = "global"
            lines.append(f"  {p} ({source})")
    elif base:
        lines.append("\nNo additional allowed paths beyond workspace base.")
    else:
        lines.append("\nNo allowed workspaces configured.")

    if not base and not allowed:
        lines.append("\nAll directories are accessible (permissive mode).")

    await update.message.reply_text("\n".join(lines))


_NO_BASE_MSG = "No workspace base configured. Set workspace_base in users.yaml or WORKSPACE_BASE in .env."


# Project ids are retrieval-time labels stored in memory rows, so
# they get the same conservative shape discipline as tag slugs:
# lowercase, digits, hyphen/underscore, no leading separator, 64-char
# ceiling. Display names keep the user's original casing.
_PROJECT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


async def _register_memory_project_for(
    config: Config,
    chat_id: int,
    root: Path,
    raw_name: str,
) -> tuple[bool, str]:
    """
    Shared registration core for /project register and the
    /workspace new auto-hook. Returns (ok, user_message); never
    raises, because the auto-hook must not fail workspace creation
    over a registration problem.

    Guards, in order:
    - project_id shape (lowercased raw_name must match _PROJECT_ID_RE).
    - nested-root guard: if ANY registered project (merged view)
      already owns this root, including via containment, the
      registration is rejected naming the owner. A nested project
      would steal scope from its parent through longest-prefix
      detection.
    - id collision against the merged view.

    On success the row is persisted AND pushed into the detection
    cache, so the user's next message in the workspace routes to the
    new project with no restart.
    """
    from kai.memory_projects import (
        db_registry_upsert,
        detect_active_memory_project,
        merged_registry,
        registry_mutation_lock,
    )

    project_id = raw_name.strip().lower()
    if not _PROJECT_ID_RE.match(project_id):
        return False, f"Invalid project name {raw_name!r}: use letters, digits, - or _ (max 64 chars)."

    # Guard + persist + cache update under the registry mutation
    # lock: the guards read the merged view, and without the lock a
    # second registration can pass its own guards against the same
    # stale view while this one is awaiting the DB insert, committing
    # a parent/child pair the nested-root guard exists to prevent.
    async with registry_mutation_lock():
        merged = merged_registry(config.memory_projects)
        owner = detect_active_memory_project(root, merged)
        if owner is not None:
            return False, f"This workspace is already inside project '{owner.project_id}'."
        if project_id in merged:
            return False, f"Project id '{project_id}' is already registered; pick another name."

        resolved_root = root.expanduser().resolve()
        row = {
            "project_id": project_id,
            "display_name": raw_name.strip(),
            "workspace_root": str(resolved_root),
            "memory_enabled": True,
            "default_scope_for_new_facts": "project",
            "created_by": chat_id,
        }
        try:
            await sessions.register_memory_project(
                project_id=project_id,
                display_name=raw_name.strip(),
                workspace_root=str(resolved_root),
                created_by=chat_id,
            )
        except Exception as e:
            # IntegrityError covers id/root collisions that raced a
            # restart-era row the cache never saw; anything else is a
            # DB-layer failure. Both collapse to a user-facing message
            # because the caller may be the auto-hook, which must not
            # raise.
            log.warning("memory project registration failed for %r: %s", project_id, e)
            return False, f"Could not register project '{project_id}': {e}"
        if not db_registry_upsert(row):
            # The handler validated every field above, so a cache
            # rejection here means validation drift between this guard
            # and _row_to_config; the row is persisted and will load on
            # next restart regardless.
            log.error("memory project cache rejected validated row %r", project_id)
    log.info(
        "memory.project.registry %s",
        json.dumps(
            {"action": "register", "project_id": project_id, "root": str(resolved_root), "by": chat_id},
            separators=(",", ":"),
        ),
    )
    return True, f"Registered memory project '{project_id}' for this workspace."


@_require_auth
async def handle_project(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /project - register, unregister, or list memory projects.

    Subcommands:
        /project                    - list registered projects
        /project list               - same
        /project register [name]    - register the CURRENT workspace
                                      (name defaults to its directory name)
        /project unregister <name>  - remove a chat-registered project

    Operator-pinned projects (from the host-side registry file) are
    visible in the listing but cannot be unregistered from chat.
    """
    from kai.memory_projects import (
        db_registry_creator,
        db_registry_remove,
        detect_active_memory_project,
        merged_registry,
    )

    assert update.message is not None
    chat_id = _chat_id(update)
    pool = _get_pool(context)
    config: Config = context.bot_data["config"]

    args = context.args or []
    sub = args[0].lower() if args else "list"

    if sub == "register":
        current = Path(await pool.get_effective_workspace(chat_id))
        raw_name = args[1] if len(args) > 1 else current.name
        # The message text alone distinguishes success from rejection
        # for the user; the boolean exists for the /workspace new
        # auto-hook, which prefixes failures with a "Note:" wrapper.
        _ok, message = await _register_memory_project_for(config, chat_id, current, raw_name)
        await update.message.reply_text(message)
        return

    if sub == "unregister":
        if len(args) < 2:
            await update.message.reply_text("Usage: /project unregister <name>")
            return
        project_id = args[1].strip().lower()
        from kai.memory_projects import registry_mutation_lock

        # The role comes from static config and cannot go stale;
        # everything keyed on the registry row must be read under the
        # mutation lock below.
        user = config.get_user_config(chat_id)
        is_admin = user is not None and user.role == "admin"

        # Pinned check, creator lookup, authorization, AND the delete
        # all under the registry mutation lock: a creator read before
        # the lock can authorize against a row that an earlier queued
        # mutation deletes and a different user re-registers under
        # the same id, letting stale authorization delete the new
        # owner's project. The denial is computed inside the lock and
        # replied outside it, so Telegram I/O never holds the lock.
        denial: str | None = None
        removed = False
        async with registry_mutation_lock():
            if project_id in config.memory_projects:
                denial = f"Project '{project_id}' is operator-pinned; it cannot be unregistered from chat."
            else:
                creator = db_registry_creator(project_id)
                if creator is None:
                    denial = f"No chat-registered project named '{project_id}'."
                elif chat_id != creator and not is_admin:
                    denial = (
                        f"Project '{project_id}' was registered by another user; "
                        "only they or an admin can unregister it."
                    )
                else:
                    removed = await sessions.unregister_memory_project(project_id)
                    db_registry_remove(project_id)
        if denial is not None:
            await update.message.reply_text(denial)
            return
        log.info(
            "memory.project.registry %s",
            json.dumps(
                {"action": "unregister", "project_id": project_id, "by": chat_id},
                separators=(",", ":"),
            ),
        )
        if removed:
            await update.message.reply_text(f"Unregistered memory project '{project_id}'.")
        else:
            await update.message.reply_text(f"Project '{project_id}' was already gone.")
        return

    if sub != "list":
        await update.message.reply_text("Usage: /project [list | register [name] | unregister <name>]")
        return

    merged = merged_registry(config.memory_projects)
    if not merged:
        await update.message.reply_text(
            "No memory projects registered.\nUse /project register [name] in a workspace to create one."
        )
        return
    current = Path(await pool.get_effective_workspace(chat_id))
    active = detect_active_memory_project(current, merged)
    lines = ["Memory projects:"]
    for project_id in sorted(merged):
        cfg = merged[project_id]
        provenance = "pinned" if project_id in config.memory_projects else "user"
        marker = " (active)" if active is not None and active.project_id == project_id else ""
        lines.append(f"- {project_id} [{provenance}]{marker}")
        for project_root in cfg.workspace_roots:
            lines.append(f"    {project_root}")
    await update.message.reply_text("\n".join(lines))


@_require_auth
async def handle_workspace(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /workspace - show, switch, or create workspaces.

    Subcommands:
        /workspace                - show current workspace
        /workspace home           - switch to home workspace
        /workspace <name>         - switch by name (workspace_base, then allowed list)
        /workspace new <name>     - create a new workspace with git init
        /workspace allow <path>   - add an allowed workspace path
        /workspace deny <path>    - remove an allowed workspace path
        /workspace allowed        - list all allowed workspaces with sources

    Absolute paths and ~ expansion are rejected for security. Name resolution
    checks workspace_base first, then the allowed list (by directory name).
    """
    assert update.message is not None
    chat_id = _chat_id(update)
    pool = _get_pool(context)
    config: Config = context.bot_data["config"]
    # Per-user `/workspace home` destination. The user-visible fix for
    # #353: prior to this, "home" was a shared directory.
    home = resolve_home_workspace(chat_id, config)

    # Resolve per-user workspace access (workspace_base + allowed list)
    base, allowed = await sessions.resolve_workspace_access(chat_id, config)

    # No args: show current workspace
    if not context.args:
        current = await pool.get_effective_workspace(chat_id)
        short = _short_workspace_name(str(current), base)
        if current == home:
            short = "Home"
        await update.message.reply_text(f"Workspace: {short}\n{current}")
        return

    target = " ".join(context.args)

    # "home" keyword: always allowed
    if target.lower() == "home":
        await _switch_workspace(update, context, home)
        return

    # Route workspace subcommands. Exact word boundary check
    # (same pattern as "config") to avoid collisions with workspace
    # names like "allowlist" or "denied-access".
    target_lower = target.lower()
    if target_lower == "allow" or target_lower.startswith("allow "):
        await _handle_workspace_allow(update, context, target)
        return
    if target_lower == "deny" or target_lower.startswith("deny "):
        await _handle_workspace_deny(update, context, target)
        return
    if target_lower == "allowed":
        await _handle_workspace_allowed(update, context)
        return

    # Reject absolute paths and ~ expansion for security
    if target.startswith("/") or target.startswith("~"):
        await update.message.reply_text("Absolute paths are not allowed. Use a workspace name.")
        return

    # "new" keyword: create a new workspace directory with git init.
    # Exact word boundary so names like "newsletter" aren't caught.
    if target_lower == "new" or target_lower.startswith("new "):
        parts = target.split(None, 1)
        if len(parts) < 2:
            await update.message.reply_text("Usage: /workspace new <name>")
            return
        if not base:
            await update.message.reply_text(_NO_BASE_MSG)
            return
        name = parts[1]
        resolved = _resolve_workspace_path(name, base)
        if resolved is None:
            await update.message.reply_text("Invalid workspace name.")
            return
        if resolved.exists():
            await update.message.reply_text(f"Already exists:\n{resolved}")
            return
        resolved.mkdir(parents=True)
        proc = await asyncio.create_subprocess_exec(
            "git",
            "init",
            cwd=str(resolved),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        rc = await proc.wait()
        if rc != 0:
            # Directory was created but git init failed (git missing,
            # permissions, etc.). Warn the user but still switch - the
            # workspace is usable without version control.
            await update.message.reply_text(
                f"Warning: git init failed (exit code {rc}). The workspace was created but has no git repo."
            )
        # Auto-register the new workspace as a memory project:
        # /workspace new is an explicit "starting a project here"
        # signal, and registering at creation time means the very
        # first exchange routes its memories to the project instead
        # of leaking them global. Failure warns but never blocks the
        # creation; the workspace is fully usable unregistered, and
        # /project register remains the manual path. The BASENAME,
        # not the raw argument: /workspace new accepts relative
        # subpaths ("sub/project"), whose separator would fail the
        # project-id slug validation; the final path component
        # matches /project register's default for the same directory.
        registered, register_note = await _register_memory_project_for(config, chat_id, resolved, resolved.name)
        await _switch_workspace(update, context, resolved)
        if registered:
            await update.message.reply_text(register_note)
        else:
            await update.message.reply_text(f"Note: memory project not registered: {register_note}")
        return

    # "config" keyword: view or modify workspace settings.
    # Exact word boundary check to avoid collisions with workspace
    # names starting with "config" (e.g., "configs", "config-backup").
    if target_lower == "config" or target_lower.startswith("config "):
        await _handle_workspace_config(update, context, target)
        return

    # Try workspace_base first (base wins on name collision per spec)
    resolved: Path | None = None
    base_candidate = _resolve_workspace_path(target, base)
    if base_candidate is not None and base_candidate.is_dir():
        resolved = base_candidate

    # Fall back to allowed workspaces - match by directory name.
    # Multiple matches means the user needs to pick via /workspaces.
    if resolved is None:
        matches = [p for p in allowed if p.name == target]
        if len(matches) > 1:
            paths = "\n".join(f"  {p}" for p in matches)
            await update.message.reply_text(
                f"Multiple workspaces named '{target}':\n{paths}\nUse /workspaces to pick one."
            )
            return
        resolved = matches[0] if matches else None

    if resolved is None:
        # Give a helpful message if neither source is configured
        if not base and not allowed:
            await update.message.reply_text(_NO_BASE_MSG)
        else:
            await update.message.reply_text(f"Workspace '{target}' not found.")
        return

    await _switch_workspace(update, context, resolved)


# ── GitHub notification settings ─────────────────────────────────────

# Permissive sanity check for owner/repo format. Accepts alphanumeric
# characters, hyphens, underscores, and periods in each component.
# More permissive than GitHub's actual rules (e.g., allows leading
# dots), but catches obvious typos. The GitHub API returns 404 for
# names that pass this regex but aren't valid repos.
_REPO_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")


def _derive_webhook_url(telegram_webhook_url: str) -> str:
    """
    Derive the GitHub webhook URL from the Telegram webhook URL.

    Replaces the path component with /webhook/github, preserving the
    scheme and authority. For example:
        https://api.syrinx.net/webhook/telegram -> https://api.syrinx.net/webhook/github

    Callers must guard against telegram_webhook_url being None (polling
    mode) before calling this function.
    """
    parsed = urlparse(telegram_webhook_url)
    return urlunparse(parsed._replace(path="/webhook/github"))


async def _github_api_ensure_webhook(
    repo: str,
    token: str,
    config: Config,
) -> None:
    """
    Register a webhook on repo if one doesn't already exist.

    Checks for an existing hook first (idempotent). On success, stores
    the hook ID in the settings table so deregistration doesn't need to
    re-query GitHub.

    Raises:
        github_api.GitHubAPIError: On failure (401, 403, 404, etc.), or
            status 0 if telegram_webhook_url is None (polling mode).
    """
    if config.telegram_webhook_url is None:
        raise github_api.GitHubAPIError(0, "No webhook URL configured (polling mode)")
    if not config.github_webhook_secret:
        raise github_api.GitHubAPIError(0, "GITHUB_WEBHOOK_SECRET is not configured")
    owner, name = repo.split("/", 1)
    webhook_url = _derive_webhook_url(config.telegram_webhook_url)

    # Check if already registered (idempotent)
    exists, hook_id = await github_api.check_webhook_exists(owner, name, token, webhook_url)
    if exists:
        # Store the hook ID in case it wasn't stored before (e.g.,
        # manually created webhook)
        if hook_id is not None:
            await sessions.set_setting(f"github_hook_id:{repo}", str(hook_id))
        return

    # Register the webhook and store the returned ID
    hook_id = await github_api.register_webhook(
        owner,
        name,
        token,
        webhook_url,
        config.github_webhook_secret,
    )
    await sessions.set_setting(f"github_hook_id:{repo}", str(hook_id))


async def _github_api_remove_webhook(
    repo: str,
    token: str,
    config: Config,
) -> None:
    """
    Remove the Kai webhook from repo.

    Looks up the stored hook ID first. If not stored (webhook was
    manually created), falls back to querying GitHub to find it.
    No-op if telegram_webhook_url is None (polling mode) or the hook
    is already gone.

    Raises:
        github_api.GitHubAPIError: On unexpected errors.
    """
    if config.telegram_webhook_url is None:
        return
    owner, name = repo.split("/", 1)

    # Try stored hook ID first
    stored_id = await sessions.get_setting(f"github_hook_id:{repo}")
    if stored_id:
        await github_api.deregister_webhook(owner, name, int(stored_id), token)
        await sessions.delete_setting(f"github_hook_id:{repo}")
        return

    # Fall back: find the hook ID by querying GitHub
    webhook_url = _derive_webhook_url(config.telegram_webhook_url)
    exists, hook_id = await github_api.check_webhook_exists(owner, name, token, webhook_url)
    if not exists or hook_id is None:
        return  # Already gone
    await github_api.deregister_webhook(owner, name, hook_id, token)


async def _handle_github_token(
    update: Update,
    chat_id: int,
    args: list[str],
) -> None:
    """
    Handle /github token <ghp_...> and /github token clear.

    Stores or clears the user's GitHub PAT. The token is stored in
    owner-only SQLite state, never logged, never echoed, and the
    source Telegram command is deleted after a successful store when
    Telegram permits deletion.
    """
    assert update.message is not None

    if not args:
        await update.message.reply_text("Usage: /github token <token> or /github token clear")
        return

    if args[0].lower() == "clear":
        await sessions.delete_setting(f"github_token:{chat_id}")
        await update.message.reply_text("GitHub token removed.")
        return

    # Store the token in owner-only SQLite state, then delete the
    # token-bearing Telegram command where Telegram permits it.
    await sessions.set_setting(f"github_token:{chat_id}", args[0])
    try:
        await update.message.delete()
    except BadRequest as exc:
        log.warning("Could not delete GitHub token command message: %s", exc)
    except Exception:
        log.exception("Could not delete GitHub token command message")
    await update.message.reply_text("GitHub token stored.")


def _manual_webhook_text(repo: str, webhook_url: str | None) -> str:
    """
    Build the manual fallback text shown when automatic webhook
    registration is not possible (no token, 403, or polling mode).
    """
    lines = [
        "Webhook registration requires a GitHub token with admin:repo_hook scope. To set one: /github token <your_pat>",
        "",
        "To register the webhook manually, go to:",
        f"  {repo} Settings > Webhooks > Add webhook",
    ]
    if webhook_url:
        lines.append(f"  URL: {webhook_url}")
    lines.extend(
        [
            "  Content type: application/json",
            "  Secret: (ask your Kai admin)",
            "  Events: Pushes, Pull requests, Issues, Issue comments, PR reviews",
        ]
    )
    return "\n".join(lines)


async def _handle_github_add(
    update: Update,
    chat_id: int,
    args: list[str],
    config: Config,
) -> None:
    """
    Handle /github add <owner/repo> - subscribe to a repo's notifications.

    If the user has a stored GitHub PAT, attempts to auto-register the
    webhook. Falls back to manual instructions on 403 (no admin access)
    or when no token is stored.
    """
    assert update.message is not None

    if not args:
        await update.message.reply_text("Usage: /github add <owner/repo>")
        return

    # Validate owner/repo format
    raw_repo = args[0]
    if not _REPO_PATTERN.match(raw_repo):
        await update.message.reply_text("Invalid repo format. Expected: owner/repo (e.g., dcellison/kai)")
        return

    repo = raw_repo.lower()

    # Get the user's yaml baseline for effective computation
    user_config = config.get_user_config(chat_id)
    yaml_repos = user_config.github_repos if user_config else []
    effective = await sessions.get_effective_repos(chat_id, yaml_repos)
    added = await sessions.get_github_added_repos(chat_id)
    removed = await sessions.get_github_removed_repos(chat_id)

    # Check if already subscribed
    if repo in effective:
        await update.message.reply_text(f"Already subscribed to `{repo}`.")
        return

    # Determine if this is a re-add (cancels a previous remove)
    is_readd = repo in removed
    if is_readd:
        # Remove from the removed list (the add cancels the remove).
        # Don't skip webhook registration - it may have been deregistered.
        removed = [r for r in removed if r != repo]
        await sessions.set_github_removed_repos(chat_id, removed)
    else:
        # Add to the added list
        added.append(repo)
        await sessions.set_github_added_repos(chat_id, added)

    # Derive the webhook URL for manual fallback text
    webhook_url = _derive_webhook_url(config.telegram_webhook_url) if config.telegram_webhook_url else None

    # Attempt automatic webhook registration if the user has a token
    token = await sessions.get_setting(f"github_token:{chat_id}")
    verb = "Re-subscribed" if is_readd else "Subscribed"

    if token:
        try:
            await _github_api_ensure_webhook(repo, token, config)
            await update.message.reply_text(f"{verb} to `{repo}` notifications. Webhook registered.")
            return
        except github_api.GitHubAPIError as e:
            if e.status == 401:
                await update.message.reply_text(
                    f"{verb} to `{repo}` notifications.\n\n"
                    "GitHub token is invalid or expired. "
                    "Update it with /github token <new_pat>"
                )
                return
            if e.status == 403:
                # No admin access - subscription succeeds, show manual fallback
                await update.message.reply_text(
                    f"{verb} to `{repo}` notifications.\n\n" + _manual_webhook_text(repo, webhook_url)
                )
                return
            if e.status == 404:
                # Repo not found - roll back the subscription
                if is_readd:
                    removed.append(repo)
                    await sessions.set_github_removed_repos(chat_id, removed)
                else:
                    added = [r for r in added if r != repo]
                    await sessions.set_github_added_repos(chat_id, added)
                await update.message.reply_text(f"Repository `{repo}` not found. Check the name and try again.")
                return
            # Network error or other - subscription succeeds, warn
            log.warning("GitHub API error registering webhook for %s: %s", repo, e)
            await update.message.reply_text(
                f"{verb} to `{repo}` notifications.\n\n"
                "Could not register webhook automatically (network error). "
                f"You can retry with /github add {repo}."
            )
            return

    # No token - subscription succeeds, show manual fallback
    await update.message.reply_text(f"{verb} to `{repo}` notifications.\n\n" + _manual_webhook_text(repo, webhook_url))


async def _handle_github_remove(
    update: Update,
    chat_id: int,
    args: list[str],
    config: Config,
) -> None:
    """
    Handle /github remove <owner/repo> - unsubscribe from repo notifications.

    If this user was the last subscriber and has a stored GitHub PAT,
    attempts to deregister the webhook. Otherwise notifies the user
    about manual cleanup.
    """
    assert update.message is not None

    if not args:
        await update.message.reply_text("Usage: /github remove <owner/repo>")
        return

    # Validate owner/repo format
    raw_repo = args[0]
    if not _REPO_PATTERN.match(raw_repo):
        await update.message.reply_text("Invalid repo format. Expected: owner/repo (e.g., dcellison/kai)")
        return

    repo = raw_repo.lower()

    # Get the user's current effective repos
    user_config = config.get_user_config(chat_id)
    yaml_repos = user_config.github_repos if user_config else []
    effective = await sessions.get_effective_repos(chat_id, yaml_repos)

    if repo not in effective:
        await update.message.reply_text(f"Not subscribed to `{repo}`.")
        return

    # Remove the subscription. If repo is in the DB-added list, remove
    # it from there. Otherwise, add it to the removed list (to override
    # the yaml baseline).
    added = await sessions.get_github_added_repos(chat_id)
    if repo in added:
        added = [r for r in added if r != repo]
        await sessions.set_github_added_repos(chat_id, added)
    else:
        removed = await sessions.get_github_removed_repos(chat_id)
        removed.append(repo)
        await sessions.set_github_removed_repos(chat_id, removed)

    # Check if any other user is still subscribed to this repo.
    # A linear scan of all users is fine for small deployments.
    other_subscribers = False
    for uid, uc in config.user_configs.items():
        if uid == chat_id:
            continue
        uc_effective = await sessions.get_effective_repos(uc.telegram_id, uc.github_repos)
        if repo in uc_effective:
            other_subscribers = True
            break

    token = await sessions.get_setting(f"github_token:{chat_id}")

    if other_subscribers:
        await update.message.reply_text(f"Unsubscribed from `{repo}`. Webhook kept (other users are still subscribed).")
        return

    # Last subscriber - try to remove the webhook
    if token:
        try:
            await _github_api_remove_webhook(repo, token, config)
            await update.message.reply_text(f"Unsubscribed from `{repo}`. Webhook removed (no other subscribers).")
        except github_api.GitHubAPIError as e:
            log.warning("GitHub API error removing webhook for %s: %s", repo, e)
            await update.message.reply_text(
                f"Unsubscribed from `{repo}`. "
                "Could not remove webhook automatically - you may want to remove it manually."
            )
    else:
        await update.message.reply_text(
            f"Unsubscribed from `{repo}`. No other subscribers. "
            "To remove the webhook, go to the repo's Settings > Webhooks."
        )


async def _show_github(update: Update, chat_id: int, config: Config) -> None:
    """Display the user's effective GitHub notification settings with source attribution."""
    assert update.message is not None
    user_config = config.get_user_config(chat_id)

    # GitHub identity (from users.yaml only, not user-settable)
    github_user = user_config.github if user_config else None

    # Resolve effective settings using the same precedence as webhook routing
    effective = await sessions.resolve_github_settings(chat_id, config)

    lines = []
    if github_user:
        lines.append(f"GitHub: {github_user}")
    else:
        lines.append("GitHub: not configured")

    # Notification destination
    notify = effective["notify_chat_id"]
    if notify and notify != chat_id:
        lines.append(f"Notifications: {notify}")
    else:
        lines.append("Notifications: this chat")

    # Feature toggles with source attribution. Read DB settings directly
    # so we can tell the user where each value comes from.
    db_settings = await sessions.get_github_db_settings(chat_id)

    def _toggle_line(
        label: str,
        db_key: str,
        yaml_val: bool | None,
        effective_val: bool,
    ) -> str:
        """Format a toggle line with its source (DB override, yaml, or global default)."""
        state = "on" if effective_val else "off"
        if db_key in db_settings:
            source = "user override"
        elif yaml_val is not None:
            source = "users.yaml"
        else:
            source = "global default"
        return f"{label}: {state} ({source})"

    yaml_pr = user_config.pr_review if user_config else None
    yaml_triage = user_config.issue_triage if user_config else None

    lines.append(
        _toggle_line(
            "PR reviews",
            "pr_review",
            yaml_pr,
            effective["pr_review"],
        )
    )
    lines.append(
        _toggle_line(
            "Issue triage",
            "issue_triage",
            yaml_triage,
            effective["issue_triage"],
        )
    )

    # Subscribed repos with source attribution. Build sets from each
    # source so we can label each repo's origin in the display.
    yaml_repos_set = set(r.lower() for r in (user_config.github_repos if user_config else []))
    db_added = await sessions.get_github_added_repos(chat_id)
    db_added_set = set(db_added)

    repos = effective["repos"]
    if repos:
        lines.append("")
        lines.append("Subscribed repos:")
        for repo in repos:
            # A repo in the DB-added set was added via /github add.
            # Everything else comes from users.yaml (DB-removed repos
            # are already excluded from the effective list).
            if repo in db_added_set:
                lines.append(f"  {repo}  (added via /github add)")
            elif repo.lower() in yaml_repos_set:
                lines.append(f"  {repo}  (users.yaml)")
            else:
                lines.append(f"  {repo}")
    else:
        lines.append("\nNo repo subscriptions configured.")

    # Token status (never show the actual token value)
    token = await sessions.get_setting(f"github_token:{chat_id}")
    lines.append(f"\nGitHub token: {'stored' if token else 'not set'}")

    await update.message.reply_text("\n".join(lines))


async def _is_notify_chat_used(
    notify_chat_id: int,
    exclude_user: int,
    config: Config,
) -> bool:
    """
    Check if any user other than exclude_user still uses this chat_id
    as their GitHub notification destination.

    Checks users.yaml (via config) and the database. Returns True if
    at least one other source references this chat_id.

    Note: this does a linear scan of all users with one DB query per
    user. Fine for a personal assistant with a handful of users.
    """
    for uid, uc in config.user_configs.items():
        if uid == exclude_user:
            continue
        # Check the users.yaml entry for this user
        if uc.github_notify_chat_id == notify_chat_id:
            return True
        # Check DB override for this user
        val = await sessions.get_setting(f"github_notify_chat:{uid}")
        if val:
            try:
                if int(val) == notify_chat_id:
                    return True
            except ValueError:
                continue
    return False


async def _handle_github_notify(
    update: Update,
    chat_id: int,
    args: list[str],
    config: Config,
) -> None:
    """Handle /github notify <chat_id|reset> - set or clear notification destination."""
    assert update.message is not None

    if not args:
        await update.message.reply_text("Usage: /github notify <chat_id> or /github notify reset")
        return

    value = args[0].lower()

    if value == "reset":
        # Read the current notify chat_id before deleting it, so we can
        # remove it from the live notification-destination registry.
        old_val = await sessions.get_setting(f"github_notify_chat:{chat_id}")
        await sessions.delete_setting(f"github_notify_chat:{chat_id}")
        if old_val:
            try:
                old_chat_id = int(old_val)
            except ValueError:
                old_chat_id = None
            # Never remove the user's own chat_id from the legacy registry.
            # The registry is no longer an authorization source, but keeping
            # this guard preserves its outbound-destination semantics.
            if old_chat_id is not None and old_chat_id != chat_id:
                still_used = await _is_notify_chat_used(
                    old_chat_id,
                    exclude_user=chat_id,
                    config=config,
                )
                if not still_used:
                    webhook.remove_notification_chat_id(old_chat_id)
        await update.message.reply_text("Notification destination reset to this chat.")
        return

    # Validate chat_id is a valid integer (can be negative for groups)
    try:
        notify_id = int(value)
    except ValueError:
        await update.message.reply_text("Chat ID must be an integer.")
        return

    # If there is an existing notify destination that differs from the
    # new one, clean it up from the live registry before overwriting.
    # Without this, the old chat_id would linger until restart.
    old_val = await sessions.get_setting(f"github_notify_chat:{chat_id}")
    if old_val:
        try:
            old_notify = int(old_val)
        except ValueError:
            old_notify = None
        if old_notify is not None and old_notify != notify_id and old_notify != chat_id:
            still_used = await _is_notify_chat_used(
                old_notify,
                exclude_user=chat_id,
                config=config,
            )
            if not still_used:
                webhook.remove_notification_chat_id(old_notify)

    await sessions.set_setting(f"github_notify_chat:{chat_id}", str(notify_id))

    # Keep the live outbound-destination registry synchronized. This does not
    # expand Telegram inbound authorization.
    webhook.add_notification_chat_id(notify_id)

    await update.message.reply_text(f"GitHub notifications will go to chat {notify_id}.")


async def _handle_github_toggle(
    update: Update,
    chat_id: int,
    field: str,
    args: list[str],
) -> None:
    """Handle /github reviews on|off and /github triage on|off."""
    assert update.message is not None
    label = "PR reviews" if field == "pr_review" else "Issue triage"

    if not args or args[0].lower() not in ("on", "off"):
        # Usage hint uses the subcommand name, not the internal field name
        subcmd = "reviews" if field == "pr_review" else "triage"
        await update.message.reply_text(f"Usage: /github {subcmd} on|off")
        return

    value = args[0].lower() == "on"
    await sessions.set_setting(f"{field}:{chat_id}", "true" if value else "false")
    state = "enabled" if value else "disabled"
    await update.message.reply_text(f"{label} {state}.")


@_require_auth
async def handle_github(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /github - view and manage GitHub notification settings."""
    assert update.message is not None
    chat_id = _chat_id(update)
    config: Config = context.bot_data["config"]

    args = context.args or []
    subcommand = args[0].lower() if args else None

    # No subcommand: display current settings
    if subcommand is None:
        await _show_github(update, chat_id, config)
        return

    if subcommand == "notify":
        await _handle_github_notify(update, chat_id, args[1:], config)
        return

    if subcommand == "reviews":
        await _handle_github_toggle(update, chat_id, "pr_review", args[1:])
        return

    if subcommand == "triage":
        await _handle_github_toggle(update, chat_id, "issue_triage", args[1:])
        return

    if subcommand == "token":
        await _handle_github_token(update, chat_id, args[1:])
        return

    if subcommand == "add":
        await _handle_github_add(update, chat_id, args[1:], config)
        return

    if subcommand == "remove":
        await _handle_github_remove(update, chat_id, args[1:], config)
        return

    await update.message.reply_text(
        "Unknown subcommand. Valid: notify, reviews, triage, add, remove, token\nRun /github for current settings."
    )


# ── Server info and help ─────────────────────────────────────────────


@_require_auth
async def handle_webhooks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /webhooks — show webhook server status and endpoint info."""
    assert update.message is not None
    config: Config = context.bot_data["config"]
    running = webhook.is_running()
    status = "running" if running else "not running"
    has_github_secret = bool(config.github_webhook_secret)
    has_generic_secret = bool(config.generic_webhook_secret)
    lines = [
        f"Webhook server: {status}",
        f"Port: {config.webhook_port}",
        "",
        "Endpoints:",
        "  GET  /health          (health check)",
        "  POST /api/schedule    (principal-bound scheduling API)",
        "  POST /api/services/*  (principal-bound service proxy)",
    ]
    if has_github_secret:
        lines.append("  POST /webhook/github  (GitHub events)")
    if has_generic_secret:
        lines.append("  POST /webhook         (generic)")
    if not has_github_secret and not has_generic_secret:
        lines += [
            "",
            "No external webhook secrets are configured.",
            "Internal APIs remain active with per-user process credentials.",
        ]
    if running and has_github_secret:
        lines += [
            "",
            "GitHub setup:",
            "1. Set Payload URL to https://your-host/webhook/github",
            "2. Content type: application/json",
            "3. Set the secret to match GITHUB_WEBHOOK_SECRET",
            "4. Choose events: Pushes, Pull requests, Issues, Comments",
        ]
    await update.message.reply_text("\n".join(lines))


# Canonical local directory for the /review command's review artifact.
# The path `/tmp/pr-<N>-review.md` is a deliberate echo of the
# operator's chat-review-loop convention so manual and product-generated
# review artifacts land where the operator already expects them. Kept
# as a module-level constant so tests can redirect to a tmp_path
# without colliding with locally-owned `/tmp/pr-<N>-review.md` files.
_REVIEW_TMP_DIR = Path("/tmp")


async def _resolve_review_repo(
    actor_id: int,
    workspace: str,
    config: Config,
) -> tuple[str, str]:
    """
    Resolve the short-form `/review <pr-number>` to ``(repo, workspace_remote)``.

    Walks the conservative ladder per the spec:

    1. If the active workspace is a git checkout whose `origin` remote
       normalizes to exactly one repo in the user's admin-configured
       GitHub repo list, return that repo.
    2. Otherwise, if the user's configured GitHub repo list contains
       exactly one repo, return that repo.
    3. Otherwise, return the empty string so the caller can prompt
       for the explicit `owner/repo` form.

    No fallback to ``Config.github_repo`` (deprecated), no
    first-of-list, and no new workspace-to-repo mapping.

    The second tuple element is the normalized workspace `origin`
    remote (or "" when the workspace is not a GitHub checkout). The
    caller compares it to the chosen repo to decide whether the
    workspace is safe to pass as ``local_repo_path``; passing a
    workspace that does not match the target repo would surface the
    wrong spec / conventions / surrounding-code excerpts into the
    review, regardless of which ladder rung produced the repo.

    Args:
        actor_id: The authorized Telegram user id; drives user-scoped
            config and effective-repo lookups so a group chat does
            not read the wrong user's settings.
        workspace: The active workspace path (already resolved via
            the pool); kept as a parameter so the caller can also
            check workspace-matches in the explicit-repo branch
            without re-fetching.
    """
    user_config = config.get_user_config(actor_id)
    authorized_repos = sorted({repo.strip().lower() for repo in (user_config.github_repos if user_config else [])})

    workspace_remote_raw = await review._resolve_workspace_remote_repo(workspace)
    workspace_remote = workspace_remote_raw.lower() if workspace_remote_raw else ""

    if workspace_remote and workspace_remote in authorized_repos:
        return workspace_remote, workspace_remote

    if len(authorized_repos) == 1:
        return authorized_repos[0].lower(), workspace_remote

    return "", workspace_remote


@_require_auth
async def handle_review_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle the /review command; run a deep PR review on demand.

    Shapes:

        /review <pr-number>              repository is inferred from
                                         the active workspace's git
                                         remote (when it matches one
                                         of the user's configured
                                         GitHub repos) or from the
                                         sole configured repo.
        /review <owner/repo> <pr-number> explicit repository.

    Runtime registration places this handler behind the common authorization
    and TOTP middleware. The review runs inside the update handler (not a detached
    background task). On success the review text lands at
    ``/tmp/pr-<N>-review.md`` as the canonical artifact, a
    timestamped copy is staged under the chat's file area via
    ``_save_upload()``, and the staged copy is uploaded to Telegram
    as a document attachment so phone-only use can read the full
    review. No GitHub comment is posted; the webhook cooldown map is
    untouched.
    """
    assert update.message is not None

    # `actor_id` drives every user-scoped lookup (config, effective
    # repos, backend, provider, model override); `chat_id` drives
    # chat-scoped state (workspace path, per-chat file staging, the
    # reply target, document upload). In a notification group the two
    # differ: chat_id is the group's id and would silently miss the
    # authorized operator's user config if it were used for the
    # user-scoped reads, dropping the review onto global defaults.
    actor_id = _user_id(update)
    chat_id = _chat_id(update)
    config: Config = context.bot_data["config"]
    pool = _get_pool(context)
    workspace = str(await pool.get_effective_workspace(chat_id))

    args = list(context.args or [])
    usage = "Usage: /review [owner/repo] <pr-number>"

    # Two shapes: one arg (PR number, inferred repo) or two args
    # (explicit repo + PR number). Anything else is a usage error.
    # `workspace_remote` captures the normalized active-workspace
    # `origin` so the post-parse step can decide whether the
    # workspace is safe to pass as local_repo_path.
    repo: str = ""
    workspace_remote: str = ""
    pr_number: int = 0
    if len(args) == 1:
        try:
            pr_number = int(args[0])
        except ValueError:
            await update.message.reply_text(usage)
            return
        repo, workspace_remote = await _resolve_review_repo(actor_id, workspace, config)
        if not repo:
            await update.message.reply_text(
                f"{usage}\n"
                "Could not infer the repository from your active workspace or your "
                "admin-configured GitHub repo list."
            )
            return
    elif len(args) == 2:
        candidate = args[0].lower()
        if not _REPO_PATTERN.match(candidate):
            await update.message.reply_text("Invalid repo format. Expected: owner/repo (e.g., dcellison/kai)")
            return
        repo = candidate
        try:
            pr_number = int(args[1])
        except ValueError:
            await update.message.reply_text(usage)
            return
        workspace_remote_raw = await review._resolve_workspace_remote_repo(workspace)
        workspace_remote = workspace_remote_raw.lower() if workspace_remote_raw else ""
    else:
        await update.message.reply_text(usage)
        return

    # Review collection shells out to `gh` in the outer Kai process. The
    # target must therefore come from the user's admin-controlled users.yaml
    # grant, never merely from a self-service notification subscription. This
    # check covers both the inferred and explicit command forms and defaults
    # to no authority for missing users and admins with an empty list.
    user_config = config.get_user_config(actor_id)
    if user_config is None or not user_config.authorizes_github_repo(repo):
        log.warning(
            "Denied manual GitHub review for user %d: repository %s is not admin-authorized",
            actor_id,
            repo,
        )
        await update.message.reply_text(
            f"Repository `{repo}` is not authorized for GitHub review. "
            "Ask the Kai administrator to add its exact name to your github_repos entry in users.yaml."
        )
        return

    # Per-user backend / provider / claude-user / model-override
    # resolution. Use the shared `get_user_backend_and_provider`
    # helper so a manual /review picks up exactly the same effective
    # backend the rest of the bot uses for this user; hand-rolling
    # the fallback would drift if config.py's resolution rules ever
    # change.
    claude_user = user_config.os_user if user_config and user_config.os_user else None
    agent_backend, provider = get_user_backend_and_provider(user_config, config)
    model_override = resolve_user_model(ModelRole.PR_REVIEW, user_config, config)
    github_token = await sessions.get_setting(f"github_token:{actor_id}")
    if getattr(config, "protected_install", False) is True and not github_token:
        await update.message.reply_text(
            "PR review requires a stored per-user GitHub token in protected installs. "
            "Send `/github token <token>` first."
        )
        return

    # Only pass the workspace as `local_repo_path` when its `origin`
    # remote actually matches the target repo. Otherwise the bundle
    # would load spec/conventions from an unrelated checkout and
    # the surrounding-code search would either misdirect (rare path
    # collisions) or emit a noisy unavailable warning even though
    # the workspace and the PR repo are intentionally unrelated.
    # Passing None makes the bundle skip spec, conventions, and
    # related-context cleanly.
    local_repo_path = workspace if workspace_remote == repo else None

    # Start ack: always sent before the backend invocation; a single
    # review can run up to PR_REVIEW_TIMEOUT_S seconds and silent
    # Telegram during that wait would be confusing.
    await update.message.reply_text(f"Reviewing {repo}#{pr_number}…")

    try:
        result = await review.generate_pr_review(
            repo,
            pr_number,
            local_repo_path=local_repo_path,
            spec_dir=config.spec_dir,
            include_prior_comments=True,
            claude_user=claude_user,
            agent_backend=agent_backend,
            provider=provider,
            timeout_s=config.pr_review_timeout_s,
            model_override=model_override,
            github_token=github_token,
        )
    except Exception as exc:
        log.exception("Manual review failed for %s#%d", repo, pr_number)
        await update.message.reply_text(f"Review failed for {repo}#{pr_number}: {exc}")
        return

    if not result.review_text.strip():
        await update.message.reply_text(f"Review returned no output for {repo}#{pr_number}.")
        return

    # File body: short metadata header in front of the raw review
    # text so the standalone artifact identifies the PR it covers.
    body = f"# PR #{pr_number} review\n\nRepository: {repo}\nURL: {result.pr_url}\n\n{result.review_text}\n"

    canonical = _REVIEW_TMP_DIR / f"pr-{pr_number}-review.md"
    try:
        canonical.write_text(body)
    except OSError as exc:
        # The canonical artifact is the contract: if we cannot write
        # it the review effectively does not exist for the operator
        # (no /tmp file to read, nothing to stage or upload). Surface
        # the failure as a clear chat error so the operator does not
        # see only the "Reviewing…" ack and silence.
        log.exception("Failed to write canonical review artifact for %s#%d", repo, pr_number)
        await update.message.reply_text(f"Review backend succeeded but writing {canonical} failed: {exc}")
        return

    # Stage a timestamped copy under DATA_DIR/files/<chat_id>/ using
    # the existing upload-file naming convention so the staged
    # artifact composes with the per-chat file area and never
    # overwrites a previous review's staged copy. Staging failure
    # is non-fatal: the canonical /tmp artifact still exists and
    # the reply will surface the staging gap explicitly.
    staged: Path | None = None
    staging_failed = False
    try:
        staged = _save_upload(
            body.encode(),
            f"pr-{pr_number}-review.md",
            user_id=chat_id,
            reader_user=claude_user,
        )
    except OSError:
        staging_failed = True
        log.exception("Failed to stage review copy for %s#%d", repo, pr_number)

    # Upload the staged Markdown file to Telegram so phone-only use
    # can read the full review. The staged file is the upload source
    # rather than /tmp because the send-file allowlist may reject
    # /tmp and the staged path lives under the configured per-chat
    # data area. Failure is non-fatal (the canonical /tmp artifact
    # is already written) but it is NOT silent: a phone-only
    # operator cannot read /tmp, so the final reply must say the
    # attachment failed if it did.
    upload_failed = False
    if staged is not None:
        try:
            with open(staged, "rb") as f:
                await context.bot.send_document(
                    chat_id,
                    document=f,
                    caption=f"PR #{pr_number} review\n{canonical}",
                    filename=f"pr-{pr_number}-review.md",
                )
        except Exception:
            upload_failed = True
            log.exception("Failed to upload staged review document for %s#%d", repo, pr_number)

    # Reply 1: short status line. Always sent first so the operator
    # sees the canonical path immediately, before any potentially
    # long warning block. Per the spec the status comes before
    # warnings, and per Telegram's 4096-char message limit it has to
    # stand alone so a flood of warnings does not crowd it out.
    status_lines = [f"Review written to {canonical}"]
    if staging_failed:
        status_lines.append("Staging copy failed; document attachment skipped.")
    elif upload_failed:
        status_lines.append("Attachment failed; open the file at the path above.")
    await update.message.reply_text("\n".join(status_lines))

    # Reply 2+: collection warnings, chunked so the Telegram message
    # limit (4096 chars) cannot truncate the final reply when a
    # large bundle emits many file/fetch failures. chunk_text breaks
    # at paragraph or line boundaries when possible.
    if result.collection_warnings:
        warnings_block = "Warnings:\n" + "\n".join(f"  [{w.source}] {w.message}" for w in result.collection_warnings)
        for chunk in chunk_text(warnings_block):
            await update.message.reply_text(chunk)


@_require_auth
async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help — show all available commands."""
    assert update.message is not None
    await update.message.reply_text(
        "/stop - Interrupt current response\n"
        "/new - Start a fresh session\n"
        "\n"
        "/models - Choose a model\n"
        "/model <name> - Switch model directly\n"
        "\n"
        "/settings - Show your settings\n"
        "/settings model <name> - Default model\n"
        "/settings timeout <n> - Response timeout (seconds)\n"
        "/settings reset [field] - Clear overrides\n"
        "\n"
        "/workspace (or /ws) - Show current workspace\n"
        "/workspace <name> - Switch by name\n"
        "/workspace home - Return to default\n"
        "/workspace new <name> - Create + git init + switch\n"
        "/project - List memory projects\n"
        "/project register [name] - Register current workspace as a memory project\n"
        "/project unregister <name> - Remove a chat-registered project\n"
        "/workspace allow <path> - Add an allowed workspace\n"
        "/workspace deny <path> - Remove an allowed workspace\n"
        "/workspace allowed - List your workspaces\n"
        "/workspace config - Show workspace settings\n"
        "/workspace config <field> <value> - Override a setting\n"
        "/workspace config env KEY=VALUE - Set an env var\n"
        "/workspace config prompt <text> - Set system prompt\n"
        "/workspace config reset [field] - Clear overrides\n"
        "/workspaces - Switch workspace (inline buttons)\n"
        "\n"
        "/github - Show GitHub settings\n"
        "/github notify <chat_id|reset> - Route or reset notifications\n"
        "/github reviews [on|off] - Toggle PR reviews\n"
        "/github triage [on|off] - Toggle issue triage\n"
        "/github token [<token>] - Manage access token\n"
        "/github add <repo> - Watch a repo\n"
        "/github remove <repo> - Unwatch a repo\n"
        "/review <pr-number> - Review a PR on the inferred repo\n"
        "/review <owner/repo> <pr-number> - Review an explicit PR\n"
        "\n"
        "/memory - Browse remembered facts and episodes\n"
        "/memory search <q> - Semantic search over memories\n"
        "/memory stats - Counts and confidence distribution\n"
        "/memory help - /memory subcommand reference\n"
        "\n"
        "/voice - Toggle voice off / voice-only\n"
        "/voice only - Voice only (no text)\n"
        "/voice on - Text + voice\n"
        "/voice off - Text only\n"
        "/voice <name> - Set voice\n"
        "/voices - Choose a voice (inline buttons)\n"
        "\n"
        "/stats - Show session info\n"
        "/job - List scheduled jobs\n"
        "/job info <id> - Show job details\n"
        "/job cancel <id> - Cancel a job\n"
        "/webhooks - Show webhook server status\n"
        "/help - This message"
    )


@_require_auth
async def handle_unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle unrecognized slash commands with a helpful redirect to /help."""
    assert update.message is not None
    await update.message.reply_text(
        f"Unknown command: {(update.message.text or '').split()[0]}\nTry /help for available commands."
    )


# ── Media message handlers ──────────────────────────────────────────


def _grant_upload_read_access(path: Path, reader_user: str) -> None:
    """Grant one target OS user read access to a private upload file."""
    if sys.platform == "darwin":
        command = [
            "/bin/chmod",
            "+a",
            f"user:{reader_user} allow read,readattr,readextattr,readsecurity",
            str(path),
        ]
    elif sys.platform.startswith("linux"):
        setfacl = shutil.which("setfacl")
        if setfacl is None:
            raise OSError("setfacl is required for isolated uploaded-file handoff on Linux")
        command = [setfacl, "-m", f"u:{reader_user}:r--", str(path)]
    else:
        raise OSError(f"isolated uploaded-file handoff is unsupported on {sys.platform}")

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        raise OSError(f"could not grant upload access to {reader_user}: {detail}")


def _upload_reader_user(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> str | None:
    """Return the configured target OS user that must read uploaded files."""
    config: Config = context.bot_data["config"]
    user_config = config.get_user_config(chat_id)
    return user_config.os_user if user_config and user_config.os_user else None


def _save_upload(
    data: bytes,
    filename: str,
    user_id: int | None = None,
    *,
    reader_user: str | None = None,
) -> Path:
    """
    Save file bytes to DATA_DIR/files/ with a timestamped name.

    Creates the files/ directory if it doesn't exist. Filenames are prefixed
    with a timestamp to avoid collisions and sanitized to remove slashes and
    spaces. Returns the absolute path to the saved file so Claude can
    reference it in subsequent commands.

    When user_id is provided, files are saved to a per-user subdirectory
    (DATA_DIR/files/{user_id}/) to prevent cross-user file access.
    When None, uses the shared DATA_DIR/files/ directory (backward-
    compatible for single-user deployments).

    Upload directories are traversal-only and files are private. In protected
    installs where the agent runs as a different os_user, reader_user gets a
    named read ACL so the exact file path can be consumed without making the
    upload listable or world-readable.

    Args:
        data: Raw file bytes to write.
        filename: Original filename from Telegram (sanitized before use).
        user_id: Optional Telegram user ID for per-user file isolation.
        reader_user: Optional OS user that should receive read access.

    Returns:
        Absolute path to the saved file.
    """
    if user_id is not None:
        files_dir = DATA_DIR / "files" / str(user_id)
    else:
        files_dir = DATA_DIR / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "files").chmod(_UPLOAD_ROOT_MODE)
    if user_id is not None:
        files_dir.chmod(_UPLOAD_ROOT_MODE)

    # Timestamp prefix ensures unique names even if the same file is sent twice
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # Strip directory components entirely rather than replacing slashes.
    # Path.name returns only the final component, handling "/" and "..".
    safe_name = Path(filename).name.replace(" ", "_")
    if not safe_name:
        safe_name = "unnamed_file"
    dest = files_dir / f"{ts}_{safe_name}"
    try:
        dest.write_bytes(data)
        dest.chmod(_UPLOAD_FILE_MODE)
        if reader_user is not None:
            _grant_upload_read_access(dest, reader_user)
    except OSError:
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return dest


@_require_auth
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle photo messages — download, base64-encode, and send to Claude.

    Downloads the highest-resolution version of the photo, encodes it as
    base64, and sends it to Claude as a multi-modal content block alongside
    the caption (or "What's in this image?" if no caption).
    """
    if not update.message or not update.message.photo:
        return

    # TOTP gate: require valid session for content that invokes Claude
    if not await _check_totp(update, context):
        return

    chat_id = _chat_id(update)
    pool = _get_pool(context)
    model = pool.get_model(chat_id)
    reader_user = _upload_reader_user(context, chat_id)

    # Download the largest available resolution (last in the list)
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    data = await file.download_as_bytearray()
    raw = bytes(data)
    b64 = base64.b64encode(raw).decode()

    # Save to DATA_DIR/files/ so Claude can access the file via shell tools
    try:
        saved = _save_upload(raw, f"photo_{photo.file_unique_id}.jpg", user_id=chat_id, reader_user=reader_user)
    except OSError as exc:
        log.exception("Failed to save uploaded photo for chat %d", chat_id)
        await update.message.reply_text(f"Couldn't save the upload for agent access: {exc}")
        return

    caption = update.message.caption or "What's in this image?"
    caption += f"\n[File saved to: {saved}]"
    workshop_inbound_message_id: MessageId | None = None
    inbound_recorder = context.bot_data.get("workshop_inbound_recorder")
    if inbound_recorder is not None:
        try:
            result = await inbound_recorder(
                InboundMessage(
                    transport="telegram",
                    update_id=str(update.update_id),
                    message_id=str(update.message.message_id),
                    sender_subject=str(_user_id(update)),
                    channel_subject=str(chat_id),
                    body=caption,
                    occurred_at=update.message.date,
                )
            )
            aggregate_id = result.event.envelope.aggregate_id
            if not isinstance(aggregate_id, MessageId):
                raise RuntimeError("Workshop inbound recorder returned a non-message aggregate")
            workshop_inbound_message_id = aggregate_id
        except Exception:
            log.exception(
                "Workshop photo message shadow write failed (update_id=%s, message_id=%s)",
                update.update_id,
                update.message.message_id,
            )
    artifact_recorder = context.bot_data.get("workshop_artifact_recorder")
    if workshop_inbound_message_id is not None and artifact_recorder is not None:
        try:
            await artifact_recorder(
                InboundArtifact(
                    message_id=workshop_inbound_message_id,
                    kind="photo",
                    media_type="image/jpeg",
                    storage_path=saved,
                    source_transport="telegram",
                    source_unique_id=photo.file_unique_id,
                    occurred_at=update.message.date,
                    original_filename=None,
                ),
                storage_root=DATA_DIR / "files",
            )
        except Exception:
            log.exception(
                "Workshop photo artifact shadow write failed (update_id=%s, message_id=%s)",
                update.update_id,
                update.message.message_id,
            )
    # Capture the user LogEntry for transcript provenance threading.
    user_log = log_message(
        direction="user",
        chat_id=chat_id,
        text=caption,
        reader_user=reader_user,
        media={
            "type": "photo",
            "workshop_message_shadowed": workshop_inbound_message_id is not None,
        },
    )
    content = [
        {"type": "text", "text": caption},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
    ]

    was_queued = await _notify_if_queued(update, chat_id)
    lock = await _acquire_lock_or_kill(chat_id, pool, update)
    if lock is None:
        return
    try:
        _set_responding(chat_id)
        try:
            await _handle_response(
                update,
                context,
                chat_id,
                _prepend_queue_marker(content) if was_queued else content,
                pool,
                model,
                user_log=user_log,
                workshop_inbound_message_id=workshop_inbound_message_id,
            )
        finally:
            _clear_responding(chat_id)
    finally:
        lock.release()


# File extensions treated as readable text (sent to Claude as code blocks)
_TEXT_EXTENSIONS = {
    ".txt",
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".json",
    ".csv",
    ".tsv",
    ".md",
    ".rst",
    ".xml",
    ".html",
    ".htm",
    ".css",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".sql",
    ".log",
    ".env",
    ".gitignore",
    ".dockerfile",
    ".makefile",
    ".rb",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".swift",
    ".r",
    ".lua",
    ".pl",
    ".php",
    ".ex",
    ".exs",
    ".erl",
}

# Map image file extensions to MIME types for Claude's image content blocks
_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
_DOCUMENT_MEDIA_TYPE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")


def _canonical_document_media_type(file_name: str, claimed_media_type: str | None) -> str:
    """Return stable artifact MIME metadata without trusting malformed input."""
    suffix = Path(file_name).suffix.lower()
    if suffix in _IMAGE_MEDIA_TYPES:
        return _IMAGE_MEDIA_TYPES[suffix]
    if isinstance(claimed_media_type, str):
        normalized = claimed_media_type.split(";", 1)[0].strip().lower()
        if _DOCUMENT_MEDIA_TYPE_PATTERN.fullmatch(normalized):
            return normalized
    if suffix in _TEXT_EXTENSIONS:
        return "text/plain"
    return "application/octet-stream"


def _canonical_document_filename(file_name: str) -> str | None:
    """Return a bounded cross-platform basename for artifact provenance."""
    filename = Path(file_name.replace("\\", "/")).name.strip()
    if not filename or filename in {".", ".."} or len(filename) > 255 or "\0" in filename:
        return None
    return filename


@_require_auth
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle document (file) uploads -- images, text files, and everything else.

    All files are saved to workspace/files/ so Claude can access them via
    shell tools. Routes based on file extension for content presentation:
    - Image files -- base64-encoded and sent as multi-modal content
    - Text/code files -- decoded as UTF-8 and sent as a code block
    - Other files -- saved to disk, Claude gets the path to work with
    """
    if not update.message or not update.message.document:
        return

    # TOTP gate: require valid session for content that invokes Claude
    if not await _check_totp(update, context):
        return

    doc = update.message.document
    file_name = doc.file_name or "unknown"
    suffix = Path(file_name).suffix.lower()
    caption = update.message.caption or ""

    chat_id = _chat_id(update)
    pool = _get_pool(context)
    model = pool.get_model(chat_id)
    reader_user = _upload_reader_user(context, chat_id)
    artifact_media_type = _canonical_document_media_type(file_name, doc.mime_type)

    if suffix in _IMAGE_MEDIA_TYPES:
        # Handle images sent as documents (uncompressed upload)
        file = await context.bot.get_file(doc.file_id)
        data = await file.download_as_bytearray()
        raw = bytes(data)
        b64 = base64.b64encode(raw).decode()
        media_type = _IMAGE_MEDIA_TYPES[suffix]

        # Save to DATA_DIR/files/ so Claude can access the file via shell tools
        try:
            saved = _save_upload(raw, file_name, user_id=chat_id, reader_user=reader_user)
        except OSError as exc:
            log.exception("Failed to save uploaded image document for chat %d", chat_id)
            await update.message.reply_text(f"Couldn't save the upload for agent access: {exc}")
            return
        img_caption = caption or f"What's in this image ({file_name})?"
        img_caption += f"\n[File saved to: {saved}]"

        history_text = caption or file_name
        content = [
            {"type": "text", "text": img_caption},
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
        ]
    elif suffix in _TEXT_EXTENSIONS or (doc.mime_type and doc.mime_type.startswith("text/")):
        # Handle text/code files -- decode and wrap in a code block
        file = await context.bot.get_file(doc.file_id)
        data = await file.download_as_bytearray()
        raw = bytes(data)
        try:
            text_content = raw.decode("utf-8")
        except UnicodeDecodeError:
            await update.message.reply_text(f"Couldn't decode {file_name} as text.")
            return

        # Save to DATA_DIR/files/ so Claude can access the file via shell tools
        try:
            saved = _save_upload(raw, file_name, user_id=chat_id, reader_user=reader_user)
        except OSError as exc:
            log.exception("Failed to save uploaded text document for chat %d", chat_id)
            await update.message.reply_text(f"Couldn't save the upload for agent access: {exc}")
            return
        header = f"File: {file_name}\n```\n{text_content}\n```\n[File saved to: {saved}]"

        history_text = caption or f"[file: {file_name}]"
        if caption:
            content = f"{caption}\n\n{header}"
        else:
            content = header
    else:
        # Any other file type -- save to disk and tell Claude the path so it
        # can work with the file via shell tools (e.g., unzip, pdftotext, etc.)
        file = await context.bot.get_file(doc.file_id)
        data = await file.download_as_bytearray()
        try:
            saved = _save_upload(bytes(data), file_name, user_id=chat_id, reader_user=reader_user)
        except OSError as exc:
            log.exception("Failed to save uploaded document for chat %d", chat_id)
            await update.message.reply_text(f"Couldn't save the upload for agent access: {exc}")
            return

        history_text = caption or f"[file: {file_name}]"
        content = (caption or f"File received: {file_name}") + f"\n[File saved to: {saved}]"

    workshop_inbound_message_id: MessageId | None = None
    inbound_recorder = context.bot_data.get("workshop_inbound_recorder")
    if inbound_recorder is not None:
        try:
            result = await inbound_recorder(
                InboundMessage(
                    transport="telegram",
                    update_id=str(update.update_id),
                    message_id=str(update.message.message_id),
                    sender_subject=str(_user_id(update)),
                    channel_subject=str(chat_id),
                    body=history_text,
                    occurred_at=update.message.date,
                )
            )
            aggregate_id = result.event.envelope.aggregate_id
            if not isinstance(aggregate_id, MessageId):
                raise RuntimeError("Workshop inbound recorder returned a non-message aggregate")
            workshop_inbound_message_id = aggregate_id
        except Exception:
            log.exception(
                "Workshop document message shadow write failed (update_id=%s, message_id=%s)",
                update.update_id,
                update.message.message_id,
            )
    artifact_recorder = context.bot_data.get("workshop_artifact_recorder")
    if workshop_inbound_message_id is not None and artifact_recorder is not None:
        try:
            await artifact_recorder(
                InboundArtifact(
                    message_id=workshop_inbound_message_id,
                    kind="document",
                    media_type=artifact_media_type,
                    storage_path=saved,
                    source_transport="telegram",
                    source_unique_id=doc.file_unique_id,
                    occurred_at=update.message.date,
                    original_filename=_canonical_document_filename(file_name),
                ),
                storage_root=DATA_DIR / "files",
            )
        except Exception:
            log.exception(
                "Workshop document artifact shadow write failed (update_id=%s, message_id=%s)",
                update.update_id,
                update.message.message_id,
            )
    user_log = log_message(
        direction="user",
        chat_id=chat_id,
        text=history_text,
        reader_user=reader_user,
        media={
            "type": "document",
            "filename": file_name,
            "workshop_message_shadowed": workshop_inbound_message_id is not None,
        },
    )

    was_queued = await _notify_if_queued(update, chat_id)
    lock = await _acquire_lock_or_kill(chat_id, pool, update)
    if lock is None:
        return
    try:
        _set_responding(chat_id)
        try:
            await _handle_response(
                update,
                context,
                chat_id,
                _prepend_queue_marker(content) if was_queued else content,
                pool,
                model,
                user_log=user_log,
                workshop_inbound_message_id=workshop_inbound_message_id,
            )
        finally:
            _clear_responding(chat_id)
    finally:
        lock.release()


@_require_auth
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle voice messages — transcribe via whisper-cpp and send to the agent.

    Pipeline: download audio → check dependencies → transcribe → echo
    transcription to user → send to the agent as "[Voice message transcription]: ..."

    The echo step shows the user what was heard before the agent processes it,
    providing transparency and a chance to correct misheard speech.
    """
    if not update.message or not update.message.voice:
        return

    # TOTP gate: require valid session for content that invokes Claude
    if not await _check_totp(update, context):
        return

    chat_id = _chat_id(update)
    reader_user = _upload_reader_user(context, chat_id)
    pool = _get_pool(context)
    config: Config = context.bot_data["config"]

    if not config.voice_enabled:
        await update.message.reply_text("Voice messages are not enabled.")
        return

    # Check that all required external tools are available
    missing = []
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg")
    if not shutil.which("whisper-cli"):
        missing.append("whisper-cpp")
    if not config.whisper_model_path.exists():
        missing.append("whisper model")
    if missing:
        await update.message.reply_text(
            f"Voice is enabled but dependencies are missing: {', '.join(missing)}. "
            "See the wiki for setup instructions: Voice-Message-Setup"
        )
        return

    voice = update.message.voice
    file = await context.bot.get_file(voice.file_id)
    audio_data = bytes(await file.download_as_bytearray())

    voice_media: dict[str, object] = {"type": "voice", "duration": voice.duration}
    voice_placeholder = f"[voice message, {voice.duration}s]"

    # Transcription failure paths preserve the historical placeholder
    # entry so an operator grepping history sees that a voice message
    # came in even when whisper failed. Extraction never runs on those
    # paths, so the placeholder's lack of recoverable content is not a
    # provenance gap; the only paths that ever stamp provenance are
    # below, after a real transcript exists.
    try:
        transcript = await transcribe_voice(audio_data, config.whisper_model_path)
    except TranscriptionError as e:
        log_message(
            direction="user",
            chat_id=chat_id,
            text=voice_placeholder,
            media=voice_media,
            reader_user=reader_user,
        )
        await update.message.reply_text(f"Transcription failed: {e}")
        return

    if not transcript:
        log_message(
            direction="user",
            chat_id=chat_id,
            text=voice_placeholder,
            media=voice_media,
            reader_user=reader_user,
        )
        await update.message.reply_text("Couldn't make out any speech in that voice message.")
        return

    # Echo the transcription so the user sees what Kai heard
    await _reply_safe(update.message, f"_Heard:_ {transcript}")

    workshop_inbound_message_id: MessageId | None = None
    inbound_recorder = context.bot_data.get("workshop_inbound_recorder")
    if inbound_recorder is not None:
        try:
            result = await inbound_recorder(
                InboundMessage(
                    transport="telegram",
                    update_id=str(update.update_id),
                    message_id=str(update.message.message_id),
                    sender_subject=str(_user_id(update)),
                    channel_subject=str(chat_id),
                    body=transcript,
                    occurred_at=update.message.date,
                )
            )
            aggregate_id = result.event.envelope.aggregate_id
            if not isinstance(aggregate_id, MessageId):
                raise RuntimeError("Workshop inbound recorder returned a non-message aggregate")
            workshop_inbound_message_id = aggregate_id
        except Exception:
            log.exception(
                "Workshop voice message shadow write failed (update_id=%s, message_id=%s)",
                update.update_id,
                update.message.message_id,
            )

    artifact_recorder = context.bot_data.get("workshop_artifact_recorder")
    if workshop_inbound_message_id is not None and artifact_recorder is not None:
        try:
            # whisper-cpp uses a temporary copy that disappears after
            # transcription. Preserve the original Telegram Ogg/Opus bytes
            # inside the existing per-user upload boundary so canonical
            # artifact provenance never points at an ephemeral path. This
            # path is not added to the backend prompt.
            saved_voice = _save_upload(audio_data, "voice.oga", user_id=chat_id)
            await artifact_recorder(
                InboundArtifact(
                    message_id=workshop_inbound_message_id,
                    kind="voice",
                    media_type="audio/ogg",
                    storage_path=saved_voice,
                    source_transport="telegram",
                    source_unique_id=voice.file_unique_id,
                    occurred_at=update.message.date,
                    original_filename=None,
                ),
                storage_root=DATA_DIR / "files",
            )
        except Exception:
            log.exception(
                "Workshop voice artifact shadow write failed (update_id=%s, message_id=%s)",
                update.update_id,
                update.message.message_id,
            )

    # Log the transcript itself as the user's message so the JSONL line
    # carries what the extractor actually saw. This is the only history-
    # output behaviour change in the provenance work: previous behaviour
    # wrote only the duration placeholder, which silently lost the user's
    # actual words and made source view useless for voice-derived rows.
    user_log = log_message(
        direction="user",
        chat_id=chat_id,
        text=transcript,
        media={
            **voice_media,
            "workshop_message_shadowed": workshop_inbound_message_id is not None,
        },
        reader_user=reader_user,
    )

    prompt = f"[Voice message transcription]: {transcript}"
    model = pool.get_model(chat_id)

    was_queued = await _notify_if_queued(update, chat_id)
    lock = await _acquire_lock_or_kill(chat_id, pool, update)
    if lock is None:
        return
    try:
        _set_responding(chat_id)
        try:
            await _handle_response(
                update,
                context,
                chat_id,
                _prepend_queue_marker(prompt) if was_queued else prompt,
                pool,
                model,
                user_log=user_log,
                workshop_inbound_message_id=workshop_inbound_message_id,
            )
        finally:
            _clear_responding(chat_id)
    finally:
        lock.release()


# ── Main message handler ─────────────────────────────────────────────


async def _deny_totp_state_error(update: Update, exc: TotpStateError) -> None:
    """Deny an update when protected TOTP state cannot be trusted."""
    log.error("TOTP state unavailable; denying Telegram update: %s", exc)
    if update.callback_query is not None:
        await update.callback_query.answer("Authentication unavailable.", show_alert=True)
    message = update.message or update.effective_message
    if message is not None:
        await message.reply_text("Authentication service unavailable. Access denied; contact the Kai administrator.")


async def _check_totp(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """
    Check TOTP authentication if configured. Returns True if the request
    should proceed, False if a challenge was sent or access denied.

    Used directly by content handlers and by the common sensitive-handler
    middleware. For non-text updates, this sends the challenge prompt and
    returns False; the user then types their code as a text message, which
    handle_message processes.
    """
    try:
        if not await asyncio.to_thread(is_totp_configured):
            return True
    except TotpStateError as exc:
        await _deny_totp_state_error(update, exc)
        return False

    assert context.user_data is not None
    assert update.effective_chat is not None
    message = update.message or update.effective_message
    if message is None:
        log.error("TOTP challenge denied: Telegram update has no effective message")
        return False

    totp_cfg: Config = context.bot_data["config"]
    session_min = totp_cfg.totp_session_minutes
    auth_time = context.user_data.get("totp_authenticated_at", 0)
    totp_expired = time.time() - auth_time > session_min * 60

    if not totp_expired:
        # Auth is still valid - refresh the timestamp so the session
        # timeout measures inactivity, not time since login.
        context.user_data["totp_authenticated_at"] = time.time()
        return True

    # Session expired. For non-text messages (photos, documents, voice),
    # just send the challenge prompt. The user must type their code as
    # text, which handle_message will process via the full TOTP gate.
    if not context.user_data.get("totp_pending"):
        challenge_sec = totp_cfg.totp_challenge_seconds
        context.user_data["totp_pending"] = {
            "expires_at": time.time() + challenge_sec,
        }
        await message.reply_text("Session expired. Enter code from authenticator.")
    if update.callback_query is not None:
        await update.callback_query.answer("Authentication required.")
    return False


async def _check_totp_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Run the complete text-message TOTP challenge and verification flow."""
    try:
        if not await asyncio.to_thread(is_totp_configured):
            return True

        assert context.user_data is not None
        assert update.effective_chat is not None
        assert update.message is not None

        principal_id = _user_id(update)
        totp_cfg: Config = context.bot_data["config"]
        auth_time = context.user_data.get("totp_authenticated_at", 0)
        totp_expired = time.time() - auth_time > totp_cfg.totp_session_minutes * 60

        if not totp_expired:
            # Auth is still valid - refresh the timestamp so the session
            # timeout measures inactivity, not time since login.
            context.user_data["totp_authenticated_at"] = time.time()
            return True

        pending = context.user_data.get("totp_pending")
        if not pending:
            # First message after expiry - send challenge via the shared helper.
            await _check_totp(update, context)
            return False

        if time.time() > pending["expires_at"]:
            del context.user_data["totp_pending"]
            await update.message.reply_text("TOTP challenge expired. Send another message to try again.")
            return False

        code = update.message.text.strip() if update.message.text else ""

        # Only treat 6-digit ASCII strings as code attempts. Other messages
        # are kept out of the verifier and are not deleted from Telegram.
        if not (code.isascii() and code.isdigit() and len(code) == 6):
            await update.effective_chat.send_message("Authentication required. Enter your 6-digit code.")
            return False

        try:
            await update.message.delete()
        except Exception:
            pass

        lockout_remaining = await asyncio.to_thread(get_lockout_remaining, principal_id)
        if lockout_remaining > 0:
            minutes = math.ceil(lockout_remaining / 60)
            await update.effective_chat.send_message(
                f"Too many failed attempts. Locked out for {minutes} more minute{'s' if minutes != 1 else ''}."
            )
            return False

        lockout_attempts = totp_cfg.totp_lockout_attempts
        lockout_minutes = totp_cfg.totp_lockout_minutes
        if await asyncio.to_thread(verify_code, code, principal_id, lockout_attempts, lockout_minutes):
            del context.user_data["totp_pending"]
            context.user_data["totp_authenticated_at"] = time.time()
            await update.effective_chat.send_message("Authenticated.")
            return False

        lockout_remaining = await asyncio.to_thread(get_lockout_remaining, principal_id)
        if lockout_remaining > 0:
            del context.user_data["totp_pending"]
            await update.effective_chat.send_message(
                f"Too many failed attempts. Locked out for {lockout_minutes} minutes."
            )
        else:
            failures = await asyncio.to_thread(get_failure_count, principal_id)
            remaining = max(0, lockout_attempts - failures)
            await update.effective_chat.send_message(f"Invalid code. {remaining} attempt(s) remaining.")
        return False
    except TotpStateError as exc:
        await _deny_totp_state_error(update, exc)
        return False


@_require_auth
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle plain text messages — the primary interaction path.

    Logs the message, acquires the per-chat lock, and delegates accepted
    private text to the canonical Workshop run service. Telegram remains the
    renderer; the run service owns canonical execution-target resolution.
    """
    if not update.message or not update.message.text:
        return

    if not await _check_totp_text(update, context):
        return

    chat_id = _chat_id(update)
    prompt = update.message.text
    reader_user = _upload_reader_user(context, chat_id)
    # Capture the user LogEntry so _handle_response can thread it to
    # the provenance writer. None on JSONL write failure; the extraction
    # path then skips provenance stamping for this exchange.
    user_log = log_message(direction="user", chat_id=chat_id, text=prompt, reader_user=reader_user)
    workshop_inbound_message_id: MessageId | None = None
    recorder = context.bot_data.get("workshop_inbound_recorder")
    if recorder is not None:
        try:
            result = await recorder(
                InboundMessage(
                    transport="telegram",
                    update_id=str(update.update_id),
                    message_id=str(update.message.message_id),
                    sender_subject=str(_user_id(update)),
                    channel_subject=str(chat_id),
                    body=prompt,
                    occurred_at=update.message.date,
                )
            )
            aggregate_id = result.event.envelope.aggregate_id
            if not isinstance(aggregate_id, MessageId):
                raise RuntimeError("Workshop inbound recorder returned a non-message aggregate")
            workshop_inbound_message_id = aggregate_id
        except Exception:
            log.exception(
                "Workshop inbound shadow write failed (update_id=%s, message_id=%s)",
                update.update_id,
                update.message.message_id,
            )
    pool = _get_pool(context)
    delivery_route = (
        ResponseDeliveryRoute.WORKSHOP_PRIVATE_TEXT if chat_id == _user_id(update) else ResponseDeliveryRoute.LEGACY
    )
    workshop_run: PreparedConversationRun | None = None
    if delivery_route == ResponseDeliveryRoute.WORKSHOP_PRIVATE_TEXT and workshop_inbound_message_id is not None:
        run_service: WorkshopConversationRunService | None = context.bot_data.get("workshop_conversation_run_service")
        if run_service is not None:
            try:
                workshop_run = await run_service.prepare(workshop_inbound_message_id)
            except Exception:
                log.exception(
                    "Workshop conversation run preparation failed (inbound_message_id=%s)",
                    workshop_inbound_message_id,
                )
    model = workshop_run.model if workshop_run is not None else pool.get_model(chat_id)

    was_queued = await _notify_if_queued(update, chat_id)
    lock = await _acquire_lock_or_kill(chat_id, pool, update)
    if lock is None:
        return
    try:
        _set_responding(chat_id)
        try:
            await _handle_response(
                update,
                context,
                chat_id,
                _prepend_queue_marker(prompt) if was_queued else prompt,
                pool,
                model,
                user_log=user_log,
                workshop_inbound_message_id=workshop_inbound_message_id,
                delivery_route=delivery_route,
                workshop_run=workshop_run,
            )
        finally:
            _clear_responding(chat_id)
    finally:
        lock.release()


# ── Streaming response handler ───────────────────────────────────────


async def _handle_response(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    prompt: str | list,
    pool: SubprocessPool,
    model: str,
    user_log: LogEntry | None = None,
    workshop_inbound_message_id: MessageId | None = None,
    delivery_route: ResponseDeliveryRoute = ResponseDeliveryRoute.LEGACY,
    workshop_run: PreparedConversationRun | None = None,
) -> None:
    """
    Stream Claude's response and deliver it to the user.

    This is the central response handler used by all message types (text,
    photo, document, voice). It manages the full response lifecycle:

    1. Check voice mode to determine output format
    2. Start a background typing indicator task
    3. Stream events from Claude, creating/editing a live Telegram message
    4. Handle /stop interruptions via the per-chat stop event
    5. On completion: save session, log response, deliver final text/voice
    6. Handle errors gracefully with user-visible error messages

    In voice-only mode, streaming text edits are skipped (no live message)
    and the final response is synthesized to speech via Piper TTS.

    In text+voice mode, the text response is delivered normally, then a
    voice note is sent as a follow-up.

    Args:
        update: The Telegram Update that triggered this response.
        context: Telegram callback context.
        chat_id: The Telegram chat ID.
        prompt: Text string or list of content blocks to send to Claude.
        claude: The ClaudeCodeBackend instance.
        model: Current model name (for session tracking).
    """
    assert update.message is not None
    if not isinstance(delivery_route, ResponseDeliveryRoute):
        raise ValueError("delivery_route must be a ResponseDeliveryRoute")
    # Check voice mode before starting
    config: Config = context.bot_data["config"]
    reader_user = _upload_reader_user(context, chat_id)
    voice_mode = "off"
    if config.tts_enabled:
        voice_mode = await sessions.get_setting(f"voice_mode:{chat_id}") or "off"
    voice_only = voice_mode == "only"
    workshop_delivery_requested = delivery_route == ResponseDeliveryRoute.WORKSHOP_PRIVATE_TEXT and voice_mode == "off"
    workshop_delivery_candidate = (
        workshop_delivery_requested
        and workshop_inbound_message_id is not None
        and workshop_run is not None
        and context.bot_data.get("workshop_streaming_preview_recorder") is not None
        and context.bot_data.get("workshop_streaming_finalizer") is not None
    )
    if workshop_delivery_requested and not workshop_delivery_candidate:
        log.error(
            "Workshop authoritative delivery prerequisites are unavailable; refusing direct fallback "
            "(inbound_message_id=%s)",
            workshop_inbound_message_id,
        )
        log_message(
            direction="assistant",
            chat_id=chat_id,
            text="[error: durable delivery preparation failed]",
            reader_user=reader_user,
        )
        await _reply_safe(
            update.message,
            "Kai could not safely prepare durable delivery for this reply. Please try again.",
        )
        return

    # Keep activity indicator visible until the response completes.
    # Telegram hides the typing indicator after ~5 seconds, so we
    # re-send it every 4 seconds in a background task.
    chat_action = ChatAction.RECORD_VOICE if voice_only else ChatAction.TYPING

    async def _keep_typing():
        # Loop runs until the task is cancelled via typing_task.cancel().
        # No shared mutable flag needed - task cancellation is the proper
        # async mechanism and avoids fragile closure-captured booleans.
        while True:
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action=chat_action)
            except Exception:
                pass
            await asyncio.sleep(4)

    typing_task = asyncio.create_task(_keep_typing())

    live_msg = None
    last_edit_time = 0.0
    last_edit_text = ""
    final_response = None
    stopped_by_user = False

    try:
        # Reset the stop event (in case /stop was sent between messages)
        stop_event = get_stop_event(chat_id)
        stop_event.clear()

        # Stream events from Claude. Pass chat_id so the inner Claude
        # can include it in API calls for correct multi-user routing.
        event_stream = (
            workshop_run.stream(prompt)
            if workshop_delivery_candidate and workshop_run is not None
            else pool.send(prompt, chat_id=chat_id)
        )
        async for event in event_stream:
            # Check for /stop between stream chunks
            if stop_event.is_set():
                stop_event.clear()
                stopped_by_user = True
                if live_msg:
                    await _edit_message_safe(live_msg, last_edit_text + "\n\n_(stopped)_")
                final_response = None
                break

            if event.done:
                final_response = event.response
                break

            # In voice-only mode, skip live text updates
            if voice_only:
                continue

            now = time.monotonic()
            if not event.text_so_far:
                continue

            # Stable-prefix gate: only create or edit the live message
            # when the accumulated text has a coherent prefix to show.
            # Withholding unstable chunks here means /stop, final
            # delivery, and edit suppression always operate against text
            # Telegram actually saw, which is the invariant the rest of
            # this function relies on. `last_edit_text` is the last
            # PUBLISHED stable prefix, never raw accumulated text.
            publishable = _stream_publishable_prefix(event.text_so_far)
            if publishable is None or publishable == last_edit_text:
                continue

            if live_msg is None:
                live_msg = await _reply_safe(update.message, _truncate_for_telegram(publishable))
                last_edit_time = now
                last_edit_text = publishable
                if workshop_delivery_candidate:
                    preview_recorder = context.bot_data["workshop_streaming_preview_recorder"]
                    try:
                        await preview_recorder(
                            ConfirmedTelegramStreamingPreview(
                                inbound_message_id=workshop_inbound_message_id,
                                external_message_id=live_msg.message_id,
                                confirmed_at=datetime.now(UTC),
                            )
                        )
                    except Exception:
                        log.exception(
                            "Workshop streaming-preview binding failed; refusing direct fallback "
                            "(inbound_message_id=%s)",
                            workshop_inbound_message_id,
                        )
                        log_message(
                            direction="assistant",
                            chat_id=chat_id,
                            text="[error: durable delivery preparation failed]",
                            reader_user=reader_user,
                        )
                        notice = "Kai could not safely prepare durable delivery for this reply. Please try again."
                        if not await _edit_message_safe(live_msg, notice):
                            await _reply_safe(update.message, notice)
                        return
            elif now - last_edit_time >= EDIT_INTERVAL:
                await _edit_message_safe(live_msg, publishable)
                last_edit_time = now
                last_edit_text = publishable
    finally:
        # Always cancel the typing indicator, even if the streaming loop
        # exits with an exception. Without this, a leaked _keep_typing task
        # sends typing indicators to the chat indefinitely.
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

    # Handle error cases. Skip the error message if /stop was used -
    # the user already saw the "(stopped)" edit and doesn't need a false alarm.
    # Failed responses are logged to history so that after a session restart,
    # the injected history shows the message was attempted (not unanswered).
    # Without this, the agent sees an unanswered user message in history and
    # may try to address it instead of the current message.
    if final_response is None:
        if stopped_by_user:
            log_message(
                direction="assistant",
                chat_id=chat_id,
                text="[stopped by user]",
                reader_user=reader_user,
            )
        else:
            log_message(
                direction="assistant",
                chat_id=chat_id,
                text="[no response]",
                reader_user=reader_user,
            )
            await update.message.reply_text("Error: No response from agent")
        return

    if not final_response.success:
        # Defensive fallback: claude.py now always populates `error`
        # with a non-None string for is_error events (see the
        # response_error resolution there). The `or` here is belt-and-
        # suspenders against a future change that re-introduces None,
        # so the literal "Error: None" string can't reappear via this
        # surface even on a regression.
        error_text = render_agent_failure(final_response.failure_kind, final_response.error, config, chat_id)
        visible_error = error_text.removeprefix("Error: ")
        log_message(
            direction="assistant",
            chat_id=chat_id,
            text=f"[error: {visible_error}]",
            reader_user=reader_user,
        )
        # Send the error notice as a NEW message (not an edit of the
        # live streamed message), so any tool-use, partial reasoning,
        # and intermediate output the user was watching stays visible.
        # The previous in-place edit erased that context entirely,
        # which on long sessions could mean minutes of visible work
        # disappearing into a single error line. _reply_safe is the
        # right wrapper here: error strings can carry markdown-like
        # characters (parens, dollar signs, brackets) that Telegram's
        # Markdown parser sometimes rejects, and the wrapper falls
        # back to plain text on BadRequest while letting network
        # errors propagate naturally.
        await _reply_safe(update.message, error_text)
        return

    # Persist session info for /stats
    if final_response.session_id:
        await sessions.save_session(chat_id, final_response.session_id, model)

    final_text = final_response.text
    # Authoritative Workshop replies enter JSONL only after their durable
    # finalization commit is confirmed. Legacy routes retain their existing
    # write timing. None means the append failed (logged by log_message itself);
    # the extraction path then skips provenance stamping for this exchange.
    if workshop_delivery_candidate:
        assistant_log = None
    else:
        assistant_log = log_message(
            direction="assistant",
            chat_id=chat_id,
            text=final_text,
            reader_user=reader_user,
        )

    workshop_outbound_message_id: MessageId | None = None
    workshop_delivery_committed = False
    if workshop_delivery_candidate:
        finalizer = context.bot_data["workshop_streaming_finalizer"]
        try:
            finalization_result = await finalizer(
                OutboundMessage(
                    in_reply_to_message_id=workshop_inbound_message_id,
                    body=final_text,
                    occurred_at=datetime.now(UTC),
                )
            )
            aggregate_id = finalization_result.message.event.envelope.aggregate_id
            if not isinstance(aggregate_id, MessageId):
                raise RuntimeError("Workshop finalizer returned a non-message aggregate")
            workshop_outbound_message_id = aggregate_id
            workshop_delivery_committed = True
        except WorkshopFinalizationCommitUncertainError:
            log.critical(
                "Workshop finalization commit outcome is uncertain; refusing direct fallback (inbound_message_id=%s)",
                workshop_inbound_message_id,
                exc_info=True,
            )
            # Preserve the answer for reconciliation: the durable transaction
            # may already contain the identical canonical assistant message.
            log_message(
                direction="assistant",
                chat_id=chat_id,
                text=final_text,
                reader_user=reader_user,
            )
            await _reply_safe(
                update.message,
                "Kai could not safely confirm final delivery. The reply was not sent again to avoid a duplicate.",
            )
            return
        except Exception:
            log.exception(
                "Workshop authoritative finalization failed; refusing direct fallback (inbound_message_id=%s)",
                workshop_inbound_message_id,
            )
            log_message(
                direction="assistant",
                chat_id=chat_id,
                text="[error: durable delivery finalization failed]",
                reader_user=reader_user,
            )
            notice = "Kai could not safely finalize durable delivery for this reply. Please try again."
            if live_msg is None or not await _edit_message_safe(live_msg, notice):
                await _reply_safe(update.message, notice)
            return

    if workshop_delivery_candidate:
        assistant_log = log_message(
            direction="assistant",
            chat_id=chat_id,
            text=final_text,
            reader_user=reader_user,
        )

    outbound_recorder = context.bot_data.get("workshop_outbound_recorder")
    if not workshop_delivery_committed and workshop_inbound_message_id is not None and outbound_recorder is not None:
        try:
            outbound_result = await outbound_recorder(
                OutboundMessage(
                    in_reply_to_message_id=workshop_inbound_message_id,
                    body=final_text,
                    occurred_at=datetime.now(UTC),
                )
            )
            aggregate_id = outbound_result.event.envelope.aggregate_id
            if not isinstance(aggregate_id, MessageId):
                raise RuntimeError("Workshop outbound recorder returned a non-message aggregate")
            workshop_outbound_message_id = aggregate_id
        except Exception:
            log.exception(
                "Workshop outbound shadow write failed (inbound_message_id=%s)",
                workshop_inbound_message_id,
            )

    async def _observe_delivery(mode: str, succeeded: bool) -> None:
        if workshop_outbound_message_id is None:
            return
        delivery_recorder = context.bot_data.get("workshop_delivery_recorder")
        if delivery_recorder is None:
            return
        try:
            await delivery_recorder(
                DeliveryObservation(
                    message_id=workshop_outbound_message_id,
                    transport="telegram",
                    mode=mode,
                    succeeded=succeeded,
                    occurred_at=datetime.now(UTC),
                )
            )
        except Exception:
            log.exception(
                "Workshop delivery shadow write failed (message_id=%s, mode=%s, succeeded=%s)",
                workshop_outbound_message_id,
                mode,
                succeeded,
            )

    # Fire-and-forget: embed this exchange in semantic memory.
    # Runs in a background task so it does not delay response delivery.
    # Failures are logged but never propagate to the user.
    from kai.memory import is_enabled as memory_is_enabled

    if memory_is_enabled() and chat_id is not None:

        async def _ingest_memory() -> None:
            try:
                from kai import memory_extraction

                # Extract user text from prompt. For multimodal prompts
                # (image + text), pull the first text block rather than
                # storing the Python repr of the list.
                if isinstance(prompt, str):
                    user_text = prompt
                else:
                    user_text = next(
                        (block["text"] for block in prompt if block.get("type") == "text"),
                        "",
                    )
                # Skip image-only exchanges - no meaningful text to embed.
                if not user_text:
                    return

                # Track 2: Haiku extraction. Runs only under the Claude
                # backend and only when explicitly enabled. Fire-and-
                # forget INSIDE the existing fire-and-forget task - the
                # subprocess latency never blocks reply delivery.
                #
                # Spec 360 removed the previous Track 1 raw-user write
                # that used to run alongside this call. The verbatim
                # user storage was producing retrieval blocks dense
                # with `User said:` lines that mimicked real user input
                # and confused the inner agent on memory-adjacent
                # topics. Extracted facts (Track 2) remain the only
                # write path.
                #
                # Backend check uses the same per-user fall-through
                # pattern as `_get_user_provider` (user override wins,
                # else global). A global-only check would miss users
                # with a per-user override, so the explicit fall-through
                # is mandatory here.
                user_config = config.get_user_config(chat_id)
                effective_backend = (
                    user_config.backend if user_config and user_config.backend else config.default_backend
                )
                if config.memory_extraction_enabled and effective_backend in ONESHOT_REASONER_BACKENDS:
                    # Windowed PRIOR CONTEXT for the episode classifier
                    # (issue #392). Fetch one extra pair beyond the
                    # configured window and drop the most recent: the
                    # current exchange has already been written to JSONL
                    # by the log_message(direction="assistant", ...) call
                    # above, so the newest pair returned by
                    # `get_recent_pairs` IS the current exchange. The
                    # `[:-1]` slice handles short-history cases
                    # gracefully without a guard - on an empty `fetched`
                    # list the slice is `[][:-1] == []`, on a single
                    # element list it is `[]` (the only pair IS the
                    # current exchange, nothing prior to drop into
                    # `prior_pairs`), and on a multi-pair list it
                    # drops only the most-recent entry. N=0 disables
                    # windowing entirely; skip the disk read in that
                    # case.
                    prior_pairs: list[tuple[str, str]] = []
                    if config.episode_classifier_context_turns > 0:
                        from kai.history import get_recent_pairs

                        fetched = get_recent_pairs(chat_id, config.episode_classifier_context_turns + 1)
                        prior_pairs = fetched[:-1]
                    await memory_extraction.extract_and_store(
                        user_text=user_text,
                        assistant_text=final_text,
                        user_id=str(chat_id),
                        session_id=final_response.session_id,
                        config=config,
                        prior_pairs=prior_pairs,
                        workspace=ingest_workspace,
                        user_log=user_log,
                        assistant_log=assistant_log,
                    )
            except Exception:
                log.warning("Memory ingestion failed", exc_info=True)

        # Workspace for write-scope routing, captured BEFORE the
        # fire-and-forget task is scheduled. The exchange being
        # extracted happened in THIS workspace; reading
        # pool.get_effective_workspace inside the task instead would
        # let a /workspace switch that lands during the ingestion
        # delay re-route the exchange's facts to a project the
        # conversation never touched.
        if workshop_delivery_candidate and workshop_run is not None:
            ingest_workspace = str(await workshop_run.effective_workspace())
        else:
            ingest_workspace = str(await pool.get_effective_workspace(chat_id))
        task = asyncio.create_task(_ingest_memory())
        _pending_memory_tasks.add(task)
        task.add_done_callback(_pending_memory_tasks.discard)

    if workshop_delivery_committed:
        return

    # Voice-only mode: synthesize and send voice, fall back to text on failure
    if voice_only and final_text:
        voice_name = await sessions.get_setting(f"voice_name:{chat_id}") or DEFAULT_VOICE
        try:
            audio = await synthesize_speech(final_text, config.piper_model_dir, voice_name)
        except TTSError as e:
            log.warning("TTS failed, falling back to text: %s", e)
        else:
            try:
                await context.bot.send_voice(chat_id=chat_id, voice=audio)
            except Exception:
                await _observe_delivery("voice", False)
                raise
            await _observe_delivery("voice", True)
            return

    # Send text response (normal mode, or voice-only fallback)
    text_delivery_succeeded = True
    try:
        if live_msg:
            # Update the live message with the final text
            if len(final_text) <= 4096:
                if final_text != last_edit_text:
                    text_delivery_succeeded = await _edit_message_safe(live_msg, final_text)
            else:
                # Response exceeds Telegram's limit — edit first chunk, send the rest
                chunks = chunk_text(final_text)
                text_delivery_succeeded = await _edit_message_safe(live_msg, chunks[0])
                for chunk in chunks[1:]:
                    await _reply_safe(update.message, chunk)
        else:
            await _send_response(update, final_text)
    except Exception:
        await _observe_delivery("text", False)
        raise
    await _observe_delivery("text", text_delivery_succeeded)

    # Text+voice mode: send voice note after text
    if voice_mode == "on" and final_text:
        voice_name = await sessions.get_setting(f"voice_name:{chat_id}") or DEFAULT_VOICE
        try:
            audio = await synthesize_speech(final_text, config.piper_model_dir, voice_name)
        except TTSError as e:
            log.warning("TTS failed: %s", e)
        else:
            try:
                await context.bot.send_voice(chat_id=chat_id, voice=audio)
            except Exception:
                await _observe_delivery("voice", False)
                raise
            await _observe_delivery("voice", True)


# ── Application factory ─────────────────────────────────────────────


def create_bot(config: Config, *, use_webhook: bool = True) -> Application:
    """
    Build and configure the Telegram Application with all handlers.

    Creates the python-telegram-bot Application, initializes the ClaudeCodeBackend
    subprocess manager, stores both in bot_data, and registers all command,
    callback, and message handlers.

    concurrent_updates=True is required so /stop can be processed while a
    message handler is blocked waiting on Claude's response.

    Handler registration order matters: specific handlers (commands, photos,
    documents, voice) are registered before the catch-all text handler.

    Args:
        config: The application Config instance.
        use_webhook: If True, suppress the default Updater (updates arrive via
            webhook POST). If False, keep the Updater for long-polling mode.

    Returns:
        A fully configured Telegram Application ready to be started.
    """
    builder = Application.builder().token(config.telegram_bot_token).concurrent_updates(True)

    # PTB's ApplicationBuilder creates an Updater by default. In webhook mode,
    # updates arrive via HTTP POST so the Updater is dead weight - suppress it.
    # In polling mode, the Updater drives the update loop and must be kept.
    if use_webhook:
        builder = builder.updater(None)

    app = builder.build()
    app.bot_data["config"] = config
    app.bot_data["workshop_inbound_recorder"] = sessions.record_workshop_inbound_message
    app.bot_data["workshop_artifact_recorder"] = sessions.record_workshop_inbound_artifact
    app.bot_data["workshop_outbound_recorder"] = sessions.record_workshop_outbound_message
    app.bot_data["workshop_delivery_recorder"] = sessions.record_workshop_delivery_observation
    app.bot_data["workshop_streaming_preview_recorder"] = sessions.record_workshop_streaming_preview
    app.bot_data["workshop_streaming_finalizer"] = sessions.record_workshop_streaming_finalization
    pool = SubprocessPool(
        config=config,
        services_info=services.get_available_services(),
    )
    app.bot_data["pool"] = pool
    app.bot_data["workshop_conversation_run_service"] = WorkshopConversationRunService(
        pool,
        sessions.resolve_workshop_conversation_run,
    )

    # Default every recognized command to sensitive. `/start` and `/help`
    # disclose no user state and remain available so an authorized operator can
    # get recovery guidance before authenticating. Adding a future command to
    # this table automatically puts it behind TOTP unless it is deliberately
    # added to the narrow exemption set.
    command_handlers = [
        ("start", handle_start),
        ("new", handle_new),
        ("models", handle_models),
        ("model", handle_model),
        ("stats", handle_stats),
        ("help", handle_help),
        ("job", handle_job),
        ("jobs", handle_jobs),
        ("settings", handle_settings),
        ("project", handle_project),
        ("workspace", handle_workspace),
        ("ws", handle_workspace),
        ("workspaces", handle_workspaces),
        ("voice", handle_voice_command),
        ("voices", handle_voices),
        ("webhooks", handle_webhooks),
        ("github", handle_github),
        ("review", handle_review_command),
        ("memory", memory_command.handle_memory_command),
        ("stop", handle_stop),
    ]
    totp_exempt_commands = {"start", "help"}
    for command, callback in command_handlers:
        registered_callback = (
            callback if command in totp_exempt_commands else _require_sensitive_authentication(callback)
        )
        app.add_handler(CommandHandler(command, registered_callback))

    # Every callback either discloses user state or mutates it, so all callback
    # families share the same authorization/TOTP middleware.
    app.add_handler(CallbackQueryHandler(_require_sensitive_authentication(handle_model_callback), pattern=r"^model:"))
    app.add_handler(CallbackQueryHandler(_require_sensitive_authentication(handle_voice_callback), pattern=r"^voice:"))
    app.add_handler(CallbackQueryHandler(_require_sensitive_authentication(handle_workspace_callback), pattern=r"^ws:"))
    app.add_handler(
        CallbackQueryHandler(
            _require_sensitive_authentication(memory_command.handle_memory_callback),
            pattern=r"^mem:",
        )
    )

    # Media handlers (must be before the catch-all text handler)
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    # Unknown command handler (catches unrecognized /commands)
    app.add_handler(MessageHandler(filters.COMMAND, handle_unknown_command))

    # Catch-all text message handler (must be last)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return app
