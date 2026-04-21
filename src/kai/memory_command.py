"""
Telegram-facing surface for the `/memory` command (spec 310).

Wraps the Mem0-backed memory store in a per-user browse, search, and
forget UI. Every read and delete is scoped to the calling user's
`chat_id` - there is no cross-user inspection. Every write path is
guarded by an explicit confirmation step rendered as inline buttons,
because deletions are irreversible.

Architectural shape:
- Pure builders (`_build_*`) take fixture data and return
  `(text, keyboard)` tuples. They do no I/O and have no state. This
  lets the unit tests assert on rendering without standing up an
  Update/Bot fixture or touching Mem0.
- Top-level handlers (`handle_memory_command`,
  `handle_memory_callback`) own the I/O: they fetch from `memory.py`,
  call the builders, and dispatch send/edit calls to Telegram.
- A module-level dict `_screen_cache` holds the per-chat navigation
  state. The cache is lazy-expired on every access - no background
  reaper task. Losing the cache on process restart is acceptable
  (spec 310 §7.4); the user just retypes `/memory`.

The `source == "extracted"` filter is applied by the underlying
`memory.get_by_tag` / `memory.delete_by_id` helpers (memory.py
§7.2). This module never references `metadata.source` directly so
that a future change to the source-filter rule lives in exactly one
place.

See `home/specs/310-memory-command.md` for the canonical UX flows
and design rationale.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from kai import memory
from kai.config import Config
from kai.memory import MemoryResult, MemoryStats

if TYPE_CHECKING:
    # Only used for type hints; importing at runtime would create a
    # cycle since bot.py imports this module.
    pass

log = logging.getLogger(__name__)


# ── Constants ───────────────────────────────────────────────────────

# The closed enum of tags accepted by the Haiku extractor, in the same
# order as `_FACT_SCHEMA["properties"]["facts"]["items"]["properties"]
# ["tags"]["items"]["enum"]` at memory_extraction.py:153-163. Mirrored
# here (rather than imported) because importing memory_extraction would
# pull in the Anthropic SDK, an unnecessary dependency for the UI path.
# A drift between the two lists shows up immediately in `/memory stats`
# (a tag with rows but no enum entry would be silently absent from the
# stats view), and is also covered by a unit test in test_memory_command.
_TAG_ENUM: tuple[str, ...] = (
    "preference",
    "decision",
    "fact",
    "constraint",
    "confirmed_action",
    "project",
    "location",
    "schedule",
    "relationship",
)

# Page size for the tag drill-down view (spec §6.2). Five fits on a
# single phone screen with the surrounding buttons; raising it forces
# scrolling.
_TAG_PAGE_SIZE = 5

# Default number of search hits returned by `/memory search`, before
# the relevance floor is applied (spec §5: "default N = 8"). Floor
# filtering happens after the fetch so a small N does not silently
# eliminate borderline-relevant rows that the floor would have kept.
_SEARCH_LIMIT = 8

# Maximum line length for fact text in list views (tag drill-down,
# search results). Spec §6.2 mandates 80 chars + ellipsis. Fact detail
# view shows the full text (no truncation).
_LIST_TRUNCATE_LEN = 80

# Per-chat cache TTL in seconds (spec §7.4: "30 minutes"). Checked
# lazily on every callback access. Note this is an *idle* timeout,
# not an absolute session timeout: `_set_cache` resets `created_at`
# on every write, so a user who keeps tapping buttons inside the
# 30-minute window keeps the cache alive indefinitely. The semantics
# match the user-facing "your screen state is forgotten after a
# pause" intent of the spec; the implementation does not enforce a
# hard ceiling on session duration.
_CACHE_TTL_S = 30 * 60

# ── Per-chat navigation cache ───────────────────────────────────────


@dataclass
class _ScreenCache:
    """Per-chat ephemeral state backing /memory navigation.

    Holds the screen the user is currently looking at plus the
    arguments needed to re-render it (tag name + page, search query,
    fact id list). The numbered buttons in tag and search views are
    stored as a list of memory ids indexed 0..n; the callback data
    only carries the integer index, keeping every callback under the
    Telegram 64-byte limit even when memory ids are 36-char UUIDs.

    `return_to` carries the encoding of the screen the back button on
    a fact view should land on. It is a tuple of (verb, args) where
    args is a list of strings that get re-encoded into a callback
    on dispatch. Storing the structured form (rather than a callback
    string) means the back-target representation can change without
    needing to reparse stale entries.
    """

    screen: str
    memory_ids: list[str] = field(default_factory=list)
    tag: str | None = None
    page: int = 0
    query: str | None = None
    return_to: tuple[str, list[str]] | None = None
    created_at: float = field(default_factory=time.monotonic)


# Module-level cache: chat_id -> _ScreenCache. Single entry per chat,
# overwritten on every fresh `/memory` invocation (spec §7.4). Keeps
# memory bounded by chat count, not by historical screen count.
_screen_cache: dict[int, _ScreenCache] = {}


def _get_cache(chat_id: int) -> _ScreenCache | None:
    """Return the cache entry for `chat_id`, expiring it if stale.

    Lazy expiry: every callback access checks the entry's age and
    drops it if older than `_CACHE_TTL_S`. There is no background
    reaper because cache entries are tiny (a few hundred bytes each)
    and bounded by the number of distinct active chats.
    """
    entry = _screen_cache.get(chat_id)
    if entry is None:
        return None
    if time.monotonic() - entry.created_at > _CACHE_TTL_S:
        # Expired - drop it. The caller will treat this as "no cache"
        # and route the user back to the dashboard with an explanatory
        # toast (see `handle_memory_callback`).
        _screen_cache.pop(chat_id, None)
        return None
    return entry


def _set_cache(chat_id: int, entry: _ScreenCache) -> None:
    """Install a fresh cache entry, overwriting any prior one.

    The created_at timestamp is reset on every assignment so that
    each successful navigation extends the TTL window. Without this,
    a long browse session that crosses 30 minutes would suddenly
    expire mid-flow.
    """
    entry.created_at = time.monotonic()
    _screen_cache[chat_id] = entry


def _clear_cache(chat_id: int) -> None:
    """Remove the cache entry for `chat_id` (used by tests and reset)."""
    _screen_cache.pop(chat_id, None)


# ── Callback data encoding ──────────────────────────────────────────
#
# Telegram limits `callback_data` to 64 bytes. The grammar is:
#   mem:<verb>[:<arg>[:<arg>...]]
# Verbs are short tokens; arguments are tag names (max 16 chars,
# "confirmed_action"), short integers (page numbers, list indices),
# and the literal string "back" for navigation actions. Memory ids
# never appear in callback data - they live in the per-chat cache and
# are referenced by integer index.


_CALLBACK_PREFIX = "mem:"


@dataclass(frozen=True)
class _CallbackAction:
    """Decoded callback action from a Telegram inline-button tap."""

    verb: str
    args: list[str]


def _encode_callback(verb: str, *args: str) -> str:
    """Build a `mem:<verb>:<arg>:...` callback string.

    Centralized so that the 64-byte ceiling can be asserted in one
    place. Telegram silently truncates over-long `callback_data`,
    which would manifest as confusing "invalid action" errors at
    runtime; assert at construction time instead so tests catch it.
    """
    parts = [verb, *args]
    encoded = _CALLBACK_PREFIX + ":".join(parts)
    # Defensive ceiling. The longest legitimate callback in this
    # module is `mem:ftd:confirmed_action` at 24 bytes, well under
    # the limit; this firing means a bug in the caller (probably an
    # unexpected long arg) rather than legitimate data. Use a real
    # if/raise rather than `assert` so the check survives `python -O`,
    # under which assertions are stripped and Telegram would silently
    # truncate the over-limit callback.
    if len(encoded.encode("utf-8")) > 64:
        raise ValueError(f"callback_data too long: {encoded!r}")
    return encoded


def _decode_callback(data: str) -> _CallbackAction | None:
    """Parse a callback string into a `_CallbackAction`.

    Returns None when the prefix does not match - callers should
    treat this as "not our callback" and ignore. Returns an action
    with an empty args list for verb-only callbacks like `mem:dash`.
    """
    if not data.startswith(_CALLBACK_PREFIX):
        return None
    body = data[len(_CALLBACK_PREFIX) :]
    if not body:
        return None
    parts = body.split(":")
    return _CallbackAction(verb=parts[0], args=parts[1:])


# ── Empty-state and error messages ──────────────────────────────────
#
# Spec §8 enumerates these. Centralized as constants so the wording
# stays consistent across the entry-point handler and any callback
# that needs to fall back to an empty-state branch.

_MSG_DISABLED = "Memory is not enabled in this install."
_MSG_UNAVAILABLE = "Memory is temporarily unavailable."
_MSG_NO_FACTS = "No memories yet. Things you tell Kai will be extracted and stored here over time."
_MSG_QUERY_FAILED = "Memory query failed. Try again in a moment."
_MSG_SESSION_EXPIRED = "Session expired, resyncing."
_MSG_NO_SEARCH_RESULTS = "No matching memories found."

# Help text - the static body shown for `/memory help`, for any
# unrecognized subcommand (spec §5), AND in the dashboard's Search
# button alert toast. The toast path passes this string to
# `answerCallbackQuery`, whose `text` field is capped at 200 chars
# by Telegram (see Bot API answerCallbackQuery docs). Keep this
# string under 200; the alignment padding the prior version used
# is invisible in Telegram's alert modal anyway since the modal
# doesn't render with a monospace font.
_HELP_TEXT = (
    "/memory - Browse by tag\n"
    "/memory search <q> - Semantic search\n"
    "/memory stats - Counts and confidence distribution\n"
    "/memory forget <tag> - Delete all memories with a tag\n"
    "/memory help - Show this help"
)


# ── Display helpers ─────────────────────────────────────────────────


def _truncate(text: str, limit: int = _LIST_TRUNCATE_LEN) -> str:
    """Trim `text` to `limit` chars, appending an ellipsis if cut.

    Used in tag and search list views (spec §6.2 / §6.5). Detail
    views render the full text - never call this from the fact
    detail builder.
    """
    # Strip newlines first; a fact text containing an embedded newline
    # would break the single-line list layout (spec §6.2 mock-up
    # assumes one row per fact). Replace with a space, not a deletion,
    # so the boundary between joined sentences stays readable.
    flat = text.replace("\n", " ").replace("\r", " ")
    if len(flat) <= limit:
        return flat
    # Reserve one character for the ellipsis so the visible width
    # stays at exactly `limit`.
    return flat[: limit - 1] + "\u2026"


def _format_date(iso_str: str, with_time: bool = False) -> str:
    """Render a Mem0 ISO timestamp as `YYYY-MM-DD` or `YYYY-MM-DD HH:MM`.

    Mem0 stores timestamps as ISO-8601 strings already, so a slice is
    sufficient. Returns the original string unchanged if shorter than
    the expected slice length - this keeps mock fixtures (which often
    pass `"2026-04-17"` directly) from blowing up.
    """
    if not iso_str:
        return ""
    if not with_time:
        return iso_str[:10]
    if len(iso_str) >= 16:
        return iso_str[:10] + " " + iso_str[11:16]
    return iso_str[:10]


# ── Builder: dashboard ──────────────────────────────────────────────


def _build_dashboard(stats: MemoryStats) -> tuple[str, InlineKeyboardMarkup | None]:
    """Render the dashboard text and keyboard from a MemoryStats.

    The dashboard restricts itself to extracted-only counts (spec
    §6.1). Tags with zero facts are hidden from the keyboard but
    appear in `/memory stats` - see spec §6.1 final paragraph for
    the asymmetry rationale.

    Empty state: when `extracted_count` is zero, returns the empty
    body text from `_MSG_NO_FACTS` and a `None` keyboard so the
    caller skips inline buttons entirely.
    """
    if stats.extracted_count == 0:
        return _MSG_NO_FACTS, None

    # Tags with at least one fact, sorted descending by count. Spec
    # §6.1 mock-up shows this ordering. Ties broken by enum order
    # (the iteration order of `_TAG_ENUM`) so output is deterministic.
    nonzero = [(tag, stats.by_tag.get(tag, 0)) for tag in _TAG_ENUM if stats.by_tag.get(tag, 0) > 0]
    nonzero.sort(key=lambda item: (-item[1], _TAG_ENUM.index(item[0])))

    if not nonzero:
        # extracted_count > 0 but no row matched a known tag. This
        # would only happen if a row's metadata.tags is empty or
        # contains an off-enum value - possible if the schema is
        # bypassed. Surface as empty rather than crashing.
        return _MSG_NO_FACTS, None

    # Per-tag counts are carried by the inline buttons below
    # (`tag (count)` labels), so the dashboard text intentionally
    # stops at the summary header and confidence footer. Per #351,
    # rendering the same counts in both surfaces was redundant - the
    # original spec §6.1 mock-up included a bar chart here, but
    # operator feedback on real data showed the bars duplicated what
    # the buttons already telegraph through the parenthesized counts.
    lines: list[str] = []
    summary = f"Memories: {stats.extracted_count} facts across {len(nonzero)} tags."
    lines.append(summary)
    lines.append("")
    # Footer line: median + min confidence. Spec §6.1 calls this the
    # "first-glance tuning signal." Median is the persisted value;
    # the spec mock-up labels it "avg" but median is more honest about
    # what is shown.
    median = stats.confidence_median
    minv = stats.confidence_min
    if median is not None and minv is not None:
        lines.append(f"Tap a tag to browse. Confidence: median {median:.2f}, min {minv:.2f}.")
    else:
        lines.append("Tap a tag to browse.")

    text = "\n".join(lines)

    # Keyboard: one row per tag (preserves ordering, lets a busy
    # dashboard scroll). A short trailing row holds Search and Stats.
    rows: list[list[InlineKeyboardButton]] = []
    for tag, count in nonzero:
        label = f"{tag} ({count})"
        rows.append([InlineKeyboardButton(label, callback_data=_encode_callback("tag", tag, "0"))])
    rows.append(
        [
            InlineKeyboardButton("Search", callback_data=_encode_callback("help")),
            InlineKeyboardButton("Stats", callback_data=_encode_callback("stats")),
        ]
    )
    return text, InlineKeyboardMarkup(rows)


# ── Builder: tag view ───────────────────────────────────────────────


def _paginate(facts: list[MemoryResult], page: int) -> tuple[list[MemoryResult], int, int]:
    """Slice `facts` into the page-th window of `_TAG_PAGE_SIZE`.

    Returns (window, clamped_page, total_pages). `clamped_page` is
    `page` clipped to [0, total_pages-1]; out-of-range page numbers
    fall back to the nearest valid page rather than rendering an
    empty screen. `total_pages` is at least 1 even on an empty list
    so the "page X of Y" footer always reads sensibly.
    """
    if not facts:
        return [], 0, 1
    total = (len(facts) + _TAG_PAGE_SIZE - 1) // _TAG_PAGE_SIZE
    clamped = max(0, min(page, total - 1))
    start = clamped * _TAG_PAGE_SIZE
    return facts[start : start + _TAG_PAGE_SIZE], clamped, total


def _build_tag_view(
    tag: str,
    facts: list[MemoryResult],
    page: int,
) -> tuple[str, InlineKeyboardMarkup, list[str], int, int]:
    """Render a paginated tag drill-down.

    Returns (text, keyboard, memory_ids, clamped_page, total_pages).
    The memory_ids list is the per-fact id for the items displayed on
    THIS page only - the caller stores it in the screen cache so
    numbered button taps can resolve back to memory ids.

    Per-fact line layout (spec §6.2):
        N.  [0.92]  <truncated text>
                    <date>
    """
    window, clamped, total = _paginate(facts, page)

    if not window:
        # Empty tag - should be unreachable from the dashboard since
        # zero-fact tags are hidden, but a delete-by-id flow can
        # leave a tag empty mid-session. Render a graceful empty
        # state with a back button.
        text = f"{tag}\n\nNo memories with this tag."
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("back", callback_data=_encode_callback("dash"))]])
        return text, kb, [], 0, 1

    # Body text
    lines = [f"{tag}  (page {clamped + 1} of {total})", ""]
    memory_ids: list[str] = []
    for idx, fact in enumerate(window, start=1):
        confidence = fact.metadata.get("confidence")
        # Defensive default: extracted rows always carry confidence,
        # but legacy or pre-#335 rows might not. Render a placeholder
        # rather than KeyError-ing the screen.
        if isinstance(confidence, (int, float)):
            conf_str = f"[{float(confidence):.2f}]"
        else:
            conf_str = "[----]"
        date_str = _format_date(fact.updated_at or fact.created_at)
        lines.append(f"{idx}.  {conf_str}  {_truncate(fact.text)}")
        lines.append(f"            {date_str}")
        lines.append("")
        memory_ids.append(fact.id)

    text = "\n".join(lines).rstrip()

    # Keyboard: numbered button row + nav row.
    number_row = [
        InlineKeyboardButton(str(i + 1), callback_data=_encode_callback("fact", str(i))) for i in range(len(window))
    ]
    nav_row: list[InlineKeyboardButton] = []
    if clamped > 0:
        nav_row.append(
            InlineKeyboardButton(
                "< prev",
                callback_data=_encode_callback("tag", tag, str(clamped - 1)),
            )
        )
    nav_row.append(InlineKeyboardButton("back", callback_data=_encode_callback("dash")))
    if clamped < total - 1:
        nav_row.append(
            InlineKeyboardButton(
                "next >",
                callback_data=_encode_callback("tag", tag, str(clamped + 1)),
            )
        )
    return text, InlineKeyboardMarkup([number_row, nav_row]), memory_ids, clamped, total


# ── Builder: fact view ──────────────────────────────────────────────


def _build_fact_view(
    fact: MemoryResult,
    return_to: tuple[str, list[str]] | None,
) -> tuple[str, InlineKeyboardMarkup]:
    """Render the fact detail screen (spec §6.3).

    `return_to` encodes where the back button should land. It can be
    None for callers (e.g., tests) that don't care; in that case the
    back button defaults to the dashboard.
    """
    md = fact.metadata or {}
    tags = md.get("tags") or []
    confidence = md.get("confidence")
    session_id = md.get("session_id") or ""
    prompt_version = md.get("prompt_version") or ""
    confirmation = md.get("confirmation_quote") or ""

    if isinstance(confidence, (int, float)):
        conf_str = f"{float(confidence):.2f}"
    else:
        conf_str = "----"

    # Confirmation row: verbatim quote on confirmed_action facts,
    # literal "n/a" elsewhere. The schema invariant ("confirmation
    # _quote present iff tags includes confirmed_action", spec §4)
    # is asserted at extraction time, so we can render confidently
    # without re-validating.
    confirmation_line = confirmation if confirmation else "n/a"

    lines = [
        "Fact",
        "",
        f'"{fact.text}"',
        "",
        f"Tags:             {', '.join(tags) if tags else '(none)'}",
        f"Confidence:       {conf_str}",
        f"Date:             {_format_date(fact.created_at, with_time=True)}",
        f"Session:          {session_id or '(none)'}",
        f"Prompt version:   {prompt_version or '(none)'}",
        "",
        f"Confirmation:     {confirmation_line}",
    ]
    text = "\n".join(lines)

    back_callback = _encode_callback(return_to[0], *return_to[1]) if return_to is not None else _encode_callback("dash")
    kb = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("back", callback_data=back_callback),
                InlineKeyboardButton("forget", callback_data=_encode_callback("ffc")),
            ]
        ]
    )
    return text, kb


# ── Builder: forget single-fact confirmation (spec §6.4) ────────────


def _build_forget_fact_confirm(fact: MemoryResult) -> tuple[str, InlineKeyboardMarkup]:
    """Render the single-fact forget confirmation."""
    text = f'Forget this fact?\n\n"{fact.text}"\n\nThis cannot be undone.'
    kb = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("confirm forget", callback_data=_encode_callback("ffd")),
                InlineKeyboardButton("cancel", callback_data=_encode_callback("fview")),
            ]
        ]
    )
    return text, kb


# ── Builder: search results (spec §6.5) ─────────────────────────────


def _build_search_results(
    query: str,
    results: list[MemoryResult],
    floor: float,
) -> tuple[str, InlineKeyboardMarkup, list[str]]:
    """Render the search-results screen.

    Score shown is the Mem0 similarity score, NOT the Haiku confidence
    (spec §6.5 explicitly calls this out). Tags + date are shown on
    the second line of each result. Returns the memory_ids list for
    cache storage - same contract as the tag view.
    """
    if not results:
        text = f'Search: "{query}"\n\n{_MSG_NO_SEARCH_RESULTS}'
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("back", callback_data=_encode_callback("dash"))]])
        return text, kb, []

    lines = [f'Search: "{query}"', ""]
    memory_ids: list[str] = []
    for idx, result in enumerate(results, start=1):
        score_str = f"[{result.score:.2f}]"
        tags = result.metadata.get("tags") or []
        tag_str = ", ".join(tags) if tags else "(no tags)"
        date_str = _format_date(result.updated_at or result.created_at)
        lines.append(f"{idx}.  {score_str}  {_truncate(result.text)}")
        lines.append(f"            {tag_str}  \u00b7  {date_str}")
        lines.append("")
        memory_ids.append(result.id)

    lines.append(f"No more results above the relevance floor ({floor:.1f}).")
    text = "\n".join(lines)

    number_row = [
        InlineKeyboardButton(str(i + 1), callback_data=_encode_callback("fact", str(i))) for i in range(len(results))
    ]
    nav_row = [InlineKeyboardButton("back", callback_data=_encode_callback("dash"))]
    return text, InlineKeyboardMarkup([number_row, nav_row]), memory_ids


# ── Builder: stats (spec §6.6) ──────────────────────────────────────


def _build_stats(stats: MemoryStats) -> tuple[str, InlineKeyboardMarkup]:
    """Render the read-only stats dashboard.

    All aggregates are extracted-only by construction (memory.py
    extends MemoryStats with that scoping). The tag list shows every
    enum value, including zero-count tags - that is the asymmetry
    documented in spec §6.1 (dashboard hides zero-count tags; stats
    surfaces them as a tuning signal).
    """
    if stats.extracted_count == 0:
        # Distinct empty-state text from the dashboard so the user
        # who navigated to /memory stats specifically knows the call
        # succeeded but the corpus is empty.
        text = "Memory stats\n\nNo extracted facts yet."
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("back", callback_data=_encode_callback("dash"))]])
        return text, kb

    lines = ["Memory stats", "", f"Total:            {stats.extracted_count} facts", "", "By tag:"]
    # Pad tag column to the longest enum name for alignment.
    label_width = max(len(t) for t in _TAG_ENUM)
    for tag in _TAG_ENUM:
        count = stats.by_tag.get(tag, 0)
        lines.append(f"  {tag.ljust(label_width)}  {count:>3}")

    # Confidence block. min/median/max are None when extracted_count
    # is zero (we returned for that case above), so under current
    # memory.py semantics they're always populated here. Render "n/a"
    # rather than 0.00 if the invariant ever breaks: a misleading
    # 0.00 looks like a real (very low) reading, while "n/a" is
    # self-describing and makes the bug visible.
    lines.append("")
    lines.append("Confidence:")
    minv = stats.confidence_min
    medv = stats.confidence_median
    maxv = stats.confidence_max
    min_str = f"{minv:.2f}" if minv is not None else "n/a"
    med_str = f"{medv:.2f}" if medv is not None else "n/a"
    max_str = f"{maxv:.2f}" if maxv is not None else "n/a"
    lines.append(f"  min               {min_str}")
    lines.append(f"  median            {med_str}")
    lines.append(f"  max               {max_str}")
    # Below-threshold counts with parenthetical percentages. When
    # confidence values are missing entirely (min/median/max all
    # None) the counts are necessarily 0, but rendering them as
    # "0 (0.0%)" reads as "all facts scored above the threshold"
    # rather than "no confidence data". Mirror the n/a fallback
    # above so the row stays honest about the underlying state.
    has_confidence = minv is not None or medv is not None or maxv is not None
    if has_confidence:
        pct_07 = (stats.confidence_below_0_7 / stats.extracted_count) * 100
        pct_06 = (stats.confidence_below_0_6 / stats.extracted_count) * 100
        below_07 = f"{stats.confidence_below_0_7:>3}  ({pct_07:.1f}%)"
        below_06 = f"{stats.confidence_below_0_6:>3}  ({pct_06:.1f}%)"
    else:
        below_07 = f"{stats.confidence_below_0_7:>3}  (n/a)"
        below_06 = f"{stats.confidence_below_0_6:>3}  (n/a)"
    lines.append(f"  below 0.7         {below_07}")
    lines.append(f"  below 0.6         {below_06}")

    lines.append("")
    lines.append(f"Confirmed actions:  {stats.confirmation_quote_count} with confirmation_quote")

    if stats.by_prompt_version:
        lines.append("")
        lines.append("Prompt versions:")
        # Sort by count desc, version asc for ties - deterministic.
        # The tiebreaker key is wrapped in str() because dict keys
        # may be mixed-type when a caller constructs MemoryStats
        # directly (bypassing the aggregation cast in memory.py);
        # Python 3 raises TypeError on int<->str comparison, so the
        # cast here must happen before the sort runs - it cannot be
        # deferred to the formatting step below.
        sorted_versions = sorted(
            stats.by_prompt_version.items(),
            key=lambda item: (-item[1], str(item[0])),
        )
        # Aggregation in memory.py already casts prompt_version to
        # str, but a caller that constructs MemoryStats by a
        # different path (tests do this directly, and a future admin
        # endpoint might) bypasses that guarantee. Wrapping len()
        # and ljust() in str() keeps the renderer resilient on its
        # own so a stray int key cannot raise TypeError here.
        version_width = max(len(str(v)) for v, _ in sorted_versions)
        for version, count in sorted_versions:
            lines.append(f"  {str(version).ljust(version_width)}  {count:>3}")

    text = "\n".join(lines)
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("back", callback_data=_encode_callback("dash"))]])
    return text, kb


# ── Builder: forget-by-tag confirmation (spec §6.7) ─────────────────


def _build_forget_tag_confirm(tag: str, count: int) -> tuple[str, InlineKeyboardMarkup]:
    """Render the bulk forget-by-tag confirmation."""
    # Singular vs plural noun. Without this, count==1 renders "1 facts"
    # / "1 memories" / "confirm forget 1 facts". A tag with one
    # surviving member is a real case (deletes whittle the count).
    fact_word = "fact" if count == 1 else "facts"
    memory_word = "memory" if count == 1 else "memories"
    text = (
        f'Forget all {count} {fact_word} tagged "{tag}"?\n\n'
        f"This will permanently remove {count} {memory_word}. Tags are "
        "independent, so a fact tagged [preference, constraint] will "
        "also be affected.\n\n"
        "This cannot be undone."
    )
    kb = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    f"confirm forget {count} {fact_word}",
                    callback_data=_encode_callback("ftd", tag),
                ),
                InlineKeyboardButton("cancel", callback_data=_encode_callback("dash")),
            ]
        ]
    )
    return text, kb


# ── Top-level command handler ───────────────────────────────────────


async def handle_memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle `/memory [subcommand ...]`.

    Dispatches on the first arg:
      (no args)      → dashboard
      help / unknown → help text
      stats          → stats screen
      search <query> → search (empty query falls through to help)
      forget <tag>   → forget-by-tag confirmation
    """
    # PTB CommandHandler guarantees a message is attached, but
    # `python -O` strips asserts; use `if/raise` for `-O` safety.
    # See `_chat_id` for the convention rationale.
    if update.message is None:
        raise ValueError("handle_memory_command: update.message is None")
    chat_id = _chat_id(update)
    config: Config = context.bot_data["config"]

    # Authorization gate. The other handlers in bot.py rely on a
    # decorator; we replicate the check inline because this handler
    # is registered without the wrapper to keep the import surface
    # small (memory_command does not import the @ _require_auth
    # decorator).
    if not _is_authorized(config, _user_id(update)):
        # Silently drop unauthorized - same as the rest of the bot.
        return

    if not memory.is_enabled():
        await update.message.reply_text(_MSG_DISABLED)
        return

    args: list[str] = list(context.args or [])
    if not args:
        await _send_dashboard(update, context, chat_id)
        return

    sub = args[0].lower()
    rest = args[1:]

    if sub == "help":
        await update.message.reply_text(_HELP_TEXT)
        return
    if sub == "stats":
        await _send_stats(update, context, chat_id)
        return
    if sub == "search":
        query = " ".join(rest).strip()
        if not query:
            # Empty query falls through to help, per spec §5.
            await update.message.reply_text(_HELP_TEXT)
            return
        await _send_search(update, context, chat_id, query)
        return
    if sub == "forget":
        if not rest:
            await update.message.reply_text("Usage: /memory forget <tag>")
            return
        tag = rest[0].lower()
        if tag not in _TAG_ENUM:
            valid = ", ".join(_TAG_ENUM)
            await update.message.reply_text(f"Unknown tag '{tag}'. Valid tags: {valid}")
            return
        await _send_forget_tag_confirm(update, context, chat_id, tag)
        return

    # Unknown subcommand - show help (spec §5).
    await update.message.reply_text(_HELP_TEXT)


# ── Top-level callback handler ──────────────────────────────────────


async def handle_memory_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Dispatch `mem:*` inline-keyboard callbacks.

    Verbs:
      dash              - re-render dashboard
      tag <name> <page> - tag drill-down at page
      fact <idx>        - open fact at index (resolved against cache)
      stats             - re-render stats
      help              - help text (Search button on dashboard)
      ffc               - forget single fact confirm
      ffd               - forget single fact: do delete
      fview             - cancel forget; return to fact view (same id)
      ftd <tag>         - forget by tag: do bulk delete
    """
    # PTB CallbackQueryHandler guarantees both callback_query and
    # query.data are present, but `python -O` strips asserts; use
    # `if/raise` for `-O` safety. See `_chat_id` for the convention
    # rationale.
    if update.callback_query is None:
        raise ValueError("handle_memory_callback: callback_query is None")
    query = update.callback_query
    config: Config = context.bot_data["config"]
    if not _is_authorized(config, _user_id(update)):
        await query.answer("Not authorized.")
        return

    chat_id = _chat_id(update)
    if query.data is None:
        raise ValueError("handle_memory_callback: query.data is None")
    action = _decode_callback(query.data)
    if action is None:
        await query.answer()
        return

    # Always check enablement before doing work - feature flag could
    # have flipped off between the original `/memory` and the click.
    if not memory.is_enabled():
        await query.answer(_MSG_DISABLED)
        return

    verb = action.verb
    args = action.args

    # Answer-after-send pattern: every branch performs its risky
    # operation (Mem0 fetch, edit_message_text, etc.) BEFORE calling
    # `query.answer()`. If the send raises, the outer `except`
    # answers with _MSG_QUERY_FAILED on a still-unanswered query;
    # answering twice would cause Telegram to reject the second call
    # with BadRequest, raising an unhandled exception inside the
    # error handler itself. The visual cost is the loading spinner
    # persists slightly longer (until the send completes) - for
    # typical Mem0 latencies this is invisible.
    #
    # The `help` verb is the one exception: it has no send, only an
    # alert toast, so answer is the response.
    try:
        if verb == "dash":
            await _send_dashboard(update, context, chat_id, edit=True)
            await query.answer()
            return
        if verb == "stats":
            await _send_stats(update, context, chat_id, edit=True)
            await query.answer()
            return
        if verb == "help":
            # Search button on dashboard - there is no inline text
            # input in Telegram, so the best we can do is show the
            # help text in a transient toast and leave the dashboard
            # visible. Use a long-form alert so the user sees the
            # full command syntax. No send to defer; the alert is
            # the response.
            await query.answer(_HELP_TEXT, show_alert=True)
            return
        if verb == "tag":
            if len(args) < 2:
                await _send_dashboard(update, context, chat_id, edit=True)
                await query.answer(_MSG_SESSION_EXPIRED)
                return
            tag = args[0]
            # Same _TAG_ENUM gate as the ftd verb and the /memory
            # forget text path. Off-enum tags are safe today (Mem0
            # returns empty), but accepting them here is inconsistent
            # with the rest of the surface and a maintenance trap.
            if tag not in _TAG_ENUM:
                await _send_dashboard(update, context, chat_id, edit=True)
                await query.answer(_MSG_SESSION_EXPIRED)
                return
            try:
                page = int(args[1])
            except ValueError:
                page = 0
            await _send_tag_view(update, context, chat_id, tag, page, edit=True)
            await query.answer()
            return
        if verb == "fact":
            cache = _get_cache(chat_id)
            if cache is None or not args:
                await _send_dashboard(update, context, chat_id, edit=True)
                await query.answer(_MSG_SESSION_EXPIRED)
                return
            try:
                idx = int(args[0])
            except ValueError:
                await _send_dashboard(update, context, chat_id, edit=True)
                await query.answer(_MSG_SESSION_EXPIRED)
                return
            if idx < 0 or idx >= len(cache.memory_ids):
                await _send_dashboard(update, context, chat_id, edit=True)
                await query.answer(_MSG_SESSION_EXPIRED)
                return
            memory_id = cache.memory_ids[idx]
            await _send_fact_view(update, context, chat_id, memory_id)
            await query.answer()
            return
        if verb == "fview":
            # Cancel button on the forget-fact confirmation: re-render
            # the fact view from cache. The cache was preserved when
            # we transitioned to the confirm screen.
            cache = _get_cache(chat_id)
            if cache is None or not cache.memory_ids:
                await _send_dashboard(update, context, chat_id, edit=True)
                await query.answer(_MSG_SESSION_EXPIRED)
                return
            # The fact id is the only one in the cache when we're on
            # a fact view. (We overwrote the page-of-ids list when
            # navigating to the fact screen; see _send_fact_view.)
            memory_id = cache.memory_ids[0]
            await _send_fact_view(update, context, chat_id, memory_id)
            await query.answer()
            return
        if verb == "ffc":
            # Forget single fact: confirm step. The memory id lives
            # in the screen cache (set by _send_fact_view).
            cache = _get_cache(chat_id)
            if cache is None or not cache.memory_ids:
                await _send_dashboard(update, context, chat_id, edit=True)
                await query.answer(_MSG_SESSION_EXPIRED)
                return
            await _send_forget_fact_confirm(update, context, chat_id, cache.memory_ids[0])
            await query.answer()
            return
        if verb == "ffd":
            # Forget single fact: execute. Read id from cache, delete,
            # then return to the screen the fact was opened from.
            cache = _get_cache(chat_id)
            if cache is None or not cache.memory_ids:
                await _send_dashboard(update, context, chat_id, edit=True)
                await query.answer(_MSG_SESSION_EXPIRED)
                return
            memory_id = cache.memory_ids[0]
            return_to = cache.return_to
            ok = memory.delete_by_id(user_id=str(chat_id), memory_id=memory_id)
            # Return to whatever screen the user was on before the
            # fact view. Tag view if known; otherwise dashboard.
            if return_to is not None and return_to[0] == "tag" and len(return_to[1]) >= 2:
                tag = return_to[1][0]
                try:
                    page = int(return_to[1][1])
                except ValueError:
                    page = 0
                await _send_tag_view(update, context, chat_id, tag, page, edit=True)
            else:
                await _send_dashboard(update, context, chat_id, edit=True)
            await query.answer("Forgotten." if ok else "Not found.")
            return
        if verb == "ftd":
            # Forget by tag: execute. Tag is in the callback args
            # (not the cache) so the action survives a stale cache.
            if not args:
                await _send_dashboard(update, context, chat_id, edit=True)
                await query.answer(_MSG_SESSION_EXPIRED)
                return
            tag = args[0]
            # Validate even though the callback is one we generated:
            # stale buttons, crafted callbacks, or version skew could
            # smuggle an arbitrary string into get_by_tag/delete loop.
            # Mirror the same _TAG_ENUM gate as the /memory forget
            # text path (see `if sub == "forget"` above).
            if tag not in _TAG_ENUM:
                await _send_dashboard(update, context, chat_id, edit=True)
                await query.answer(_MSG_SESSION_EXPIRED)
                return
            facts = memory.get_by_tag(user_id=str(chat_id), tag=tag)
            deleted = 0
            for fact in facts:
                if memory.delete_by_id(user_id=str(chat_id), memory_id=fact.id):
                    deleted += 1
            await _send_dashboard(update, context, chat_id, edit=True)
            await query.answer(f"Forgot {deleted} facts.")
            return
    except Exception as exc:
        # Spec §8: never let a Mem0 exception surface as a stack trace.
        # Safe to call query.answer() here because the
        # answer-after-send pattern above guarantees it has not been
        # answered yet on any path that can reach this except.
        log.exception("memory callback %s failed: %s", verb, exc)
        await query.answer(_MSG_QUERY_FAILED)
        return

    # Unknown verb - quietly dismiss. Should be unreachable unless a
    # version skew leaves a stale callback button in chat history.
    await query.answer()


# ── Send helpers (call builders + dispatch I/O) ─────────────────────


async def _send_dashboard(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    edit: bool = False,
) -> None:
    """Render and send/edit the dashboard."""
    try:
        stats = memory.get_stats(user_id=str(chat_id))
    except Exception as exc:
        log.exception("get_stats failed: %s", exc)
        await _send_or_edit(update, _MSG_QUERY_FAILED, None, edit=edit)
        return
    text, kb = _build_dashboard(stats)
    # Reset the cache: dashboard is the root. memory_ids cleared so
    # any stale fact button taps (from a previous tag view) trip the
    # session-expired branch instead of resolving to a wrong fact.
    _set_cache(chat_id, _ScreenCache(screen="dashboard"))
    await _send_or_edit(update, text, kb, edit=edit)


async def _send_tag_view(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    tag: str,
    page: int,
    edit: bool = False,
) -> None:
    """Render and send/edit the tag drill-down for `tag` at `page`."""
    try:
        facts = memory.get_by_tag(user_id=str(chat_id), tag=tag)
    except Exception as exc:
        log.exception("get_by_tag(%s) failed: %s", tag, exc)
        await _send_or_edit(update, _MSG_QUERY_FAILED, None, edit=edit)
        return
    text, kb, memory_ids, clamped_page, _total = _build_tag_view(tag, facts, page)
    _set_cache(
        chat_id,
        _ScreenCache(screen="tag", tag=tag, page=clamped_page, memory_ids=memory_ids),
    )
    await _send_or_edit(update, text, kb, edit=edit)


async def _send_fact_view(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    memory_id: str,
) -> None:
    """Open the fact detail view by id.

    Uses memory.get_by_id for an O(1) lookup with ownership + source
    scoping baked in. Returns None covers all four "no such fact"
    cases (missing, wrong user, non-extracted, fetch error); the UI
    treats them identically.
    """
    cache = _get_cache(chat_id)
    return_to = cache.return_to if cache is not None else None
    # If we came from a tag view, set the return target so back lands
    # the user where they started rather than at the root.
    if cache is not None and cache.screen == "tag" and cache.tag is not None:
        return_to = ("tag", [cache.tag, str(cache.page)])
    elif cache is not None and cache.screen == "search" and cache.query is not None:
        # Searches re-execute on back-nav. Encoding the full query in
        # callback data would risk the 64-byte limit; for now, back
        # from a fact opened via search returns to the dashboard.
        # See spec §7.4 - single-entry-per-chat means deep linking
        # back into a search is not a v1 requirement.
        return_to = None

    fact = memory.get_by_id(user_id=str(chat_id), memory_id=memory_id)
    if fact is None:
        # Race: fact deleted between cache write and tap, or any of the
        # other not-found conditions get_by_id collapses (wrong user,
        # non-extracted source, Mem0 fetch error).
        await _send_or_edit(update, "This memory no longer exists.", None, edit=True)
        return
    text, kb = _build_fact_view(fact, return_to)
    # Cache holds only this fact's id so the forget flow knows what to
    # delete without re-encoding the id into callback data.
    _set_cache(
        chat_id,
        _ScreenCache(
            screen="fact",
            memory_ids=[memory_id],
            return_to=return_to,
        ),
    )
    await _send_or_edit(update, text, kb, edit=True)


async def _send_forget_fact_confirm(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    memory_id: str,
) -> None:
    """Render the forget-fact confirmation screen.

    Uses memory.get_by_id for the same reason as _send_fact_view:
    O(1) lookup with ownership/source scoping enforced once.
    """
    fact = memory.get_by_id(user_id=str(chat_id), memory_id=memory_id)
    if fact is None:
        await _send_or_edit(update, "This memory no longer exists.", None, edit=True)
        return
    text, kb = _build_forget_fact_confirm(fact)
    # Preserve cache so cancel returns to the same fact, and confirm
    # knows what to delete.
    cache = _get_cache(chat_id)
    return_to = cache.return_to if cache is not None else None
    _set_cache(
        chat_id,
        _ScreenCache(
            screen="forget_fact_confirm",
            memory_ids=[memory_id],
            return_to=return_to,
        ),
    )
    await _send_or_edit(update, text, kb, edit=True)


async def _send_search(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    query: str,
) -> None:
    """Run a search and send the results screen."""
    config: Config = context.bot_data["config"]
    floor = config.memory_search_floor
    try:
        results = memory.search(query, user_id=str(chat_id), limit=_SEARCH_LIMIT)
    except Exception as exc:
        log.exception("search failed: %s", exc)
        await _send_or_edit(update, _MSG_QUERY_FAILED, None, edit=False)
        return
    # Apply the floor in the UI path the same way `format_context`
    # does (spec §7.5: one knob, two paths).
    #
    # Also enforce the source==extracted scope here. Every other read
    # path in this module (`get_by_tag`, `get_by_id`) filters at the
    # data layer, but `memory.search` is a Mem0 vector lookup that
    # spans all sources (Track 1 exchanges, legacy rows). Without this
    # post-filter, a search hit could surface a non-extracted row;
    # tapping it would call `get_by_id`, fail the source check, and
    # render "This memory no longer exists." for a row the user just
    # saw - confusing and wrong. Filtering here keeps the UI honest:
    # what the user sees in results is what they can act on.
    filtered = [r for r in results if r.score >= floor and r.metadata.get("source") == "extracted"]
    text, kb, memory_ids = _build_search_results(query, filtered, floor)
    _set_cache(
        chat_id,
        _ScreenCache(
            screen="search",
            memory_ids=memory_ids,
            query=query,
        ),
    )
    await _send_or_edit(update, text, kb, edit=False)


async def _send_stats(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    edit: bool = False,
) -> None:
    """Render the stats screen."""
    try:
        stats = memory.get_stats(user_id=str(chat_id))
    except Exception as exc:
        log.exception("get_stats failed: %s", exc)
        await _send_or_edit(update, _MSG_QUERY_FAILED, None, edit=edit)
        return
    text, kb = _build_stats(stats)
    # Stats does not need fact-id state. Drop any prior cache so a
    # stale fact-id from a previous screen can't be used.
    _set_cache(chat_id, _ScreenCache(screen="stats"))
    await _send_or_edit(update, text, kb, edit=edit)


async def _send_forget_tag_confirm(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    tag: str,
) -> None:
    """Render the forget-by-tag confirmation.

    Called only from the /memory forget text path, where
    `update.message` is guaranteed by python-telegram-bot's
    routing. The contract check uses `if/raise` rather than
    `assert` so it survives `python -O` (assertions are stripped),
    matching the same hardening applied to `_send_or_edit` and
    `_encode_callback`.
    """
    if update.message is None:
        raise ValueError("_send_forget_tag_confirm: requires update.message")
    try:
        facts = memory.get_by_tag(user_id=str(chat_id), tag=tag)
    except Exception as exc:
        log.exception("get_by_tag(%s) failed: %s", tag, exc)
        await update.message.reply_text(_MSG_QUERY_FAILED)
        return
    if not facts:
        await update.message.reply_text(f'No memories tagged "{tag}".')
        return
    text, kb = _build_forget_tag_confirm(tag, len(facts))
    await _send_or_edit(update, text, kb, edit=False)


# ── I/O dispatch (send vs edit) ─────────────────────────────────────


async def _send_or_edit(
    update: Update,
    text: str,
    keyboard: InlineKeyboardMarkup | None,
    edit: bool,
) -> None:
    """Send a fresh message or edit the most recent one.

    All `/memory` screens are sent as plain text (spec §6.0); no
    `parse_mode` is set. User-originated content (fact text, tag
    names, confirmation_quote) routinely contains Markdown-reserved
    characters and rendering them via MarkdownV2 would either corrupt
    the display or require escaping every dynamic string.

    `BadRequest: Message is not modified` is swallowed: it fires when
    a user double-taps a button and the second edit is identical to
    the first. Treating it as success keeps the UI stable.

    Contract: when `edit=True`, `update.callback_query` must be set.
    All callers today live in `handle_memory_callback` where that
    invariant holds, but a future caller from outside the callback
    path would otherwise hit a silent no-op (neither branch fires).
    Raise loudly to surface the misuse instead.
    """
    if edit and update.callback_query is None:
        raise ValueError("_send_or_edit: edit=True requires update.callback_query")
    if edit and update.callback_query is not None:
        try:
            await update.callback_query.edit_message_text(text=text, reply_markup=keyboard)
        except BadRequest as exc:
            if "not modified" in str(exc).lower():
                return
            # Fall through to a fresh send if the edit failed for
            # another reason (e.g., the original message was deleted).
            log.warning("edit_message_text failed: %s; falling back to send", exc)
            # Strip the inline keyboard from the original message
            # before sending a replacement. Otherwise the stale
            # message keeps its buttons in chat history and a tap
            # would trigger a callback against state we have moved
            # past, producing ghost navigation. Best-effort: this is
            # already an error path; the contract is that cleanup
            # failure must NOT abort the fallback send. Catch
            # `Exception` rather than just BadRequest so a transient
            # NetworkError / TimedOut from PTB cannot escape and
            # short-circuit the fresh send below; the user would
            # otherwise see "Memory query failed." for what should
            # have been a successful re-render. The original message
            # may be gone, immutable, or the network may be flaky -
            # whatever the cause, the fresh send is what matters.
            try:
                await update.callback_query.edit_message_reply_markup(reply_markup=None)
            except Exception as kb_exc:
                log.debug("clear stale keyboard failed (best-effort): %s", kb_exc)
            edit = False

    if not edit:
        # Fresh send. Either initial /memory invocation or callback
        # branch with no message to edit (e.g., search invoked from
        # text command, not callback).
        #
        # `if/raise` rather than `assert`: assert is stripped by
        # `python -O` so the contract would be silently bypassed in a
        # production launch using optimized bytecode. Every other
        # defensive check in this module follows the same pattern;
        # this branch was missed in the round-2 sweep.
        if update.effective_chat is None:
            raise ValueError("_send_or_edit: effective_chat is None on fresh send")
        await update.effective_chat.send_message(text=text, reply_markup=keyboard)


# ── Auth helpers (mirror bot.py) ────────────────────────────────────
#
# Duplicated rather than imported to keep memory_command from pulling
# in bot.py (which would create an import cycle). The bodies are tiny
# and the behavior is contractual - both sides must agree on what
# "authorized" means.


def _chat_id(update: Update) -> int:
    """Return the chat id, narrowed from Optional via runtime guard.

    `if/raise` rather than `assert`: PTB routing guarantees a chat
    is present for both CommandHandler and CallbackQueryHandler
    callbacks, but `python -O` strips assertions, so a future caller
    wiring this helper from a non-routed path under optimized
    bytecode would see `AttributeError` on the next access instead
    of a meaningful contract violation. Matches the convention used
    everywhere else in this module (_encode_callback, _send_or_edit,
    _send_forget_tag_confirm).
    """
    chat = update.effective_chat
    if chat is None:
        raise ValueError("_chat_id: effective_chat is None")
    return chat.id


def _user_id(update: Update) -> int:
    """Return the user id, narrowed from Optional via runtime guard.

    See `_chat_id` for the `if/raise`-vs-`assert` rationale.
    """
    user = update.effective_user
    if user is None:
        raise ValueError("_user_id: effective_user is None")
    return user.id


def _is_authorized(config: Config, user_id: int) -> bool:
    """Check if a Telegram user id is in the allowed list."""
    return user_id in config.allowed_user_ids
