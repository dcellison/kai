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

Source filtering has three filter-site categories:

  1. Multi-source admit list (`memory.USER_VISIBLE_SOURCES`, the
     frozenset of `extracted`, `episode`, `migration`). The data-
     layer gate is canonical: `memory.get_by_id` and
     `memory.get_by_tag` admit only those sources, and
     `memory.delete_by_id` inherits the gate via its delegation.
     The one UI-side reference is `_send_search`'s post-filter,
     which exists because `memory.search` is a Mem0 vector lookup
     that spans every source, including legacy `""`-source rows
     that must not surface in the operator-facing UI. Both sites
     read `USER_VISIBLE_SOURCES` from `memory.py` so a future
     change to the admit list lives in one place.
  2. Single-source enumeration: `memory.get_all_episodes`, scoped
     to the literal `"episode"` source for the dashboard's episode-
     list browser.
  3. Multi-source enumeration scoped to the fact bucket:
     `memory.get_all_facts`, scoped to `{"extracted", "migration"}`
     for the dashboard's facts-list browser. Narrower than
     `USER_VISIBLE_SOURCES` (excludes episode), broader than
     `get_all_episodes` (admits two sources). Episodes are
     intentionally excluded because they have their own browser.

Categories 2 and 3 do not participate in `USER_VISIBLE_SOURCES`:
their purpose is enumeration, not multi-source admission. The
duplication of source literals is deliberate so that a future
change to the shared admit list cannot silently broaden either
enumeration.

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

# Page size for paginated list views. Five fits on a single phone
# screen with the surrounding buttons; raising it forces scrolling.
_PAGE_SIZE = 5

# Default number of search hits returned by `/memory search`, before
# the relevance floor is applied (spec §5: "default N = 8"). Floor
# filtering happens after the fetch so a small N does not silently
# eliminate borderline-relevant rows that the floor would have kept.
_SEARCH_LIMIT = 8

# Maximum line length for fact text in list views (search results,
# episode list). 80 chars + ellipsis. Fact detail view shows the full
# text (no truncation).
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
    arguments needed to re-render it (page index for the episode list,
    search query, fact id list). The numbered buttons in episode and
    search views are stored as a list of memory ids indexed 0..n; the
    callback data only carries the integer index, keeping every
    callback under the Telegram 64-byte limit even when memory ids
    are 36-char UUIDs.

    `return_to` carries the encoding of the screen the back button on
    a fact view should land on. It is a tuple of (verb, args) where
    args is a list of strings that get re-encoded into a callback
    on dispatch. Storing the structured form (rather than a callback
    string) means the back-target representation can change without
    needing to reparse stale entries.
    """

    screen: str
    memory_ids: list[str] = field(default_factory=list)
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
    "/memory - Browse memories\n"
    "/memory search <q> - Semantic search\n"
    "/memory stats - Counts and confidence distribution\n"
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

    The dashboard surfaces three user-visible source counts (extracted
    facts, episode summaries, migration imports) and a single utility
    keyboard row holding two optional browse buttons (Facts when
    extracted_count + migration_count > 0; Episodes when
    episode_count > 0), plus unconditional Search and Stats buttons.
    There is no per-source filter axis: the parent decision in #388
    settled tags as row decoration only, not a primary browse axis.

    Facts vs Episodes split: extracted and migration rows fold into a
    single "facts" bucket because the extracted/imported distinction
    is internal plumbing from an operator's perspective. The fact-
    view detail screen still renders per-source headers (Fact for
    extracted, Imported for migration) so the provenance surfaces
    when the operator drills in. Episodes have a distinct semantic
    shape (outcome quality, approach, lessons) that justifies a
    separate browser.

    Empty state: only when ALL three user-visible source counts are
    zero. An operator with episode or migration rows but no extracted
    facts still has user-visible memory worth showing.
    """
    # Combined empty-state guard (issue #407). When any user-visible
    # source has rows, fall through to render the dashboard - the
    # source-count headline communicates that memory is alive even
    # for an episode-only or migration-only operator.
    if stats.extracted_count == 0 and stats.episode_count == 0 and stats.migration_count == 0:
        return _MSG_NO_FACTS, None

    # Cache the visibility predicates once. The Facts button rolls
    # extracted and migration into one count; the Episodes button is
    # episode-only. The empty-state guard above ensures at least one
    # of facts_visible or episode_count > 0 is true on every reachable
    # path through the rest of this function (otherwise all three
    # counts would be zero, which the guard already returns for).
    facts_count = stats.extracted_count + stats.migration_count
    facts_visible = facts_count > 0
    episodes_visible = stats.episode_count > 0

    lines: list[str] = []
    # Headline assembled from non-zero per-source counts so a fresh
    # extracted-only operator gets a tight "Memories: N facts."
    # while a mixed-source operator gets distinct visibility into
    # each source. Zero-valued sources are omitted from the comma
    # list rather than rendered as "0 episodes" - readability over
    # uniformity, since most operators have non-zero in only one or
    # two of the three. The headline keeps extracted and migration
    # split (rather than merging them like the Facts button label)
    # because the headline communicates "where did your memory come
    # from", which is the question the per-source split answers.
    parts: list[str] = []
    if stats.extracted_count:
        parts.append(f"{stats.extracted_count} facts")
    if stats.episode_count:
        parts.append(f"{stats.episode_count} episodes")
    if stats.migration_count:
        parts.append(f"{stats.migration_count} imported")
    summary = "Memories: " + ", ".join(parts) + "."
    lines.append(summary)
    lines.append("")
    # Footer line: three reachable branches keep the action prompt
    # accurate to what the keyboard actually offers. The fourth
    # branch (no browse buttons at all) is unreachable in production
    # because the empty-state guard above would have returned; the
    # `else` arm is retained as a safety net so a future change to
    # the empty-state guard cannot silently produce a broken footer.
    #   - Both browse buttons: name both, plus Search and Stats.
    #   - Facts only: Tap Facts to browse, then Search and Stats.
    #   - Episodes only: existing post-410 wording, unchanged.
    #   - Neither (unreachable): pre-410 fallback wording.
    if facts_visible and episodes_visible:
        lines.append("Tap Facts or Episodes to browse, Search to find specific memories, or Stats for details.")
    elif facts_visible:
        lines.append("Tap Facts to browse, Search to find specific memories, or Stats for details.")
    elif episodes_visible:
        lines.append("Tap Episodes to browse, Search to find specific memories, or Stats for details.")
    else:
        lines.append("Use Search to find specific memories, or tap Stats for details.")

    text = "\n".join(lines)

    # Keyboard: a single utility row holds the cross-corpus actions
    # in this exact left-to-right order: Facts (if any), Episodes
    # (if any), Search, Stats. Both browse buttons hide when their
    # bucket is empty so an operator does not see a button that
    # would open an empty list. Search and Stats always render.
    rows: list[list[InlineKeyboardButton]] = []
    utility_row: list[InlineKeyboardButton] = []
    if facts_visible:
        utility_row.append(
            InlineKeyboardButton(
                f"Facts ({facts_count})",
                callback_data=_encode_callback("facts", "0"),
            )
        )
    if episodes_visible:
        utility_row.append(
            InlineKeyboardButton(
                f"Episodes ({stats.episode_count})",
                callback_data=_encode_callback("eps", "0"),
            )
        )
    utility_row.append(InlineKeyboardButton("Search", callback_data=_encode_callback("help")))
    utility_row.append(InlineKeyboardButton("Stats", callback_data=_encode_callback("stats")))
    rows.append(utility_row)
    return text, InlineKeyboardMarkup(rows)


# ── Pagination helper ───────────────────────────────────────────────


def _paginate(facts: list[MemoryResult], page: int) -> tuple[list[MemoryResult], int, int]:
    """Slice `facts` into the page-th window of `_PAGE_SIZE`.

    Returns (window, clamped_page, total_pages). `clamped_page` is
    `page` clipped to [0, total_pages-1]; out-of-range page numbers
    fall back to the nearest valid page rather than rendering an
    empty screen. `total_pages` is at least 1 even on an empty list
    so the "page X of Y" footer always reads sensibly.
    """
    if not facts:
        return [], 0, 1
    total = (len(facts) + _PAGE_SIZE - 1) // _PAGE_SIZE
    clamped = max(0, min(page, total - 1))
    start = clamped * _PAGE_SIZE
    return facts[start : start + _PAGE_SIZE], clamped, total


# ── Builder: episode list view (issue #410) ────────────────────────


# The closed enum of `outcome_quality` values written by the
# memory_extraction.py episode classifier. Episodes whose metadata
# contains an off-enum string still render via the `[----]` fallback;
# the strict check defends against a schema drift where a value like
# `"good"` or `"unknown"` could leak in and look like a real category.
# Mirroring (rather than importing from `memory_extraction`) avoids
# pulling in the Anthropic SDK on the UI path. A drift between the
# two lists shows up immediately in the episode list view (every
# row would render as `[----]`). frozenset (not tuple) because the
# only operation here is `in` membership; iteration order does not
# matter.
_OUTCOME_QUALITY_ENUM: frozenset[str] = frozenset({"success", "partial", "failure"})


def _build_episode_list_view(
    facts: list[MemoryResult],
    page: int,
) -> tuple[str, InlineKeyboardMarkup, list[str], int, int]:
    """Render a paginated episode list (issue #410).

    Returns the (text, keyboard, memory_ids, clamped_page,
    total_pages) tuple consumed by `_send_episode_list`. The
    memory_ids list backs the screen cache so numbered button taps
    resolve to memory ids via integer index, keeping callback data
    well under the 64-byte Telegram ceiling.

    Per-row layout:
        N.  [<outcome_quality>]  <truncated text>
                                 <date>

    Bracket text is the literal `outcome_quality` string from
    metadata when it is one of the enumerated values
    (`success` / `partial` / `failure`); otherwise `[----]`. Episode
    rows do not carry confidence; the bracket field is the per-row
    quality signal.

    Header reads `"Episodes  (page X of Y)"` with two spaces between
    the label and the parenthetical for visual separation.
    """
    window, clamped, total = _paginate(facts, page)

    if not window:
        # Reachable only when an operator with episode_count > 0
        # deletes their last episode mid-session, since the dashboard
        # hides the Episodes button at zero. Render a graceful empty
        # state with a back button rather than an empty-pagination
        # screen.
        text = "Episodes\n\nNo episodes yet."
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("back", callback_data=_encode_callback("dash"))]])
        return text, kb, [], 0, 1

    lines = [f"Episodes  (page {clamped + 1} of {total})", ""]
    memory_ids: list[str] = []
    for idx, fact in enumerate(window, start=1):
        quality = fact.metadata.get("outcome_quality")
        # Strict enum check (not a truthy-string check). Defends
        # against schema drift where an off-enum value could leak
        # in and look like a real category in the UI.
        if quality in _OUTCOME_QUALITY_ENUM:
            quality_str = f"[{quality}]"
        else:
            quality_str = "[----]"
        date_str = _format_date(fact.updated_at or fact.created_at)
        lines.append(f"{idx}.  {quality_str}  {_truncate(fact.text)}")
        # Twelve-space indent visually nests the date under the
        # bracket label on the line above.
        lines.append(f"            {date_str}")
        lines.append("")
        memory_ids.append(fact.id)

    text = "\n".join(lines).rstrip()

    number_row = [
        InlineKeyboardButton(str(i + 1), callback_data=_encode_callback("fact", str(i))) for i in range(len(window))
    ]
    nav_row: list[InlineKeyboardButton] = []
    if clamped > 0:
        nav_row.append(
            InlineKeyboardButton(
                "< prev",
                callback_data=_encode_callback("eps", str(clamped - 1)),
            )
        )
    nav_row.append(InlineKeyboardButton("back", callback_data=_encode_callback("dash")))
    if clamped < total - 1:
        nav_row.append(
            InlineKeyboardButton(
                "next >",
                callback_data=_encode_callback("eps", str(clamped + 1)),
            )
        )
    return text, InlineKeyboardMarkup([number_row, nav_row]), memory_ids, clamped, total


# ── Builder: facts list view ────────────────────────────────────────


def _build_facts_list_view(
    facts: list[MemoryResult],
    page: int,
) -> tuple[str, InlineKeyboardMarkup, list[str], int, int]:
    """Render a paginated facts list folding extracted and migration.

    Returns the (text, keyboard, memory_ids, clamped_page,
    total_pages) tuple consumed by `_send_facts_list`. The
    memory_ids list backs the screen cache so numbered button taps
    resolve to memory ids via integer index, keeping callback data
    well under the 64-byte Telegram ceiling.

    Per-row layout:
        N.  <truncated text>
            <date>

    No bracket field. The list view is source-agnostic: extracted
    and migration rows are visually indistinguishable in this
    surface because the operator cannot meaningfully act on the
    distinction at the list level. The fact-view detail screen
    (rendered when a number is tapped) keeps the per-source
    rendering from issue #407.

    Header reads `"Facts  (page X of Y)"` with two spaces between
    the label and the parenthetical; matches the episode header
    convention so both list views look uniform side by side.
    """
    window, clamped, total = _paginate(facts, page)

    if not window:
        # Reachable only when an operator with facts_visible at
        # dashboard-fetch time deletes their last extracted/migration
        # row mid-session, since the dashboard hides the Facts button
        # at zero. Render a graceful empty state with a back button
        # rather than an empty-pagination screen.
        text = "Facts\n\nNo facts yet."
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("back", callback_data=_encode_callback("dash"))]])
        return text, kb, [], 0, 1

    lines = [f"Facts  (page {clamped + 1} of {total})", ""]
    memory_ids: list[str] = []
    for idx, fact in enumerate(window, start=1):
        date_str = _format_date(fact.updated_at or fact.created_at)
        lines.append(f"{idx}.  {_truncate(fact.text)}")
        # Four-space indent visually nests the date under the start
        # of the truncated fact text on the line above (after the
        # "N.  " prefix). Compare with the episode list, which uses
        # twelve spaces because its bracket label sits there; with
        # no bracket on this surface, four is the right alignment.
        lines.append(f"    {date_str}")
        lines.append("")
        memory_ids.append(fact.id)

    text = "\n".join(lines).rstrip()

    # Number buttons reuse the existing `fact` verb. The integer
    # index resolves against `cache.memory_ids` by position; the
    # verb does not need to know which list type populated the
    # cache. Same pattern the episode list uses today.
    number_row = [
        InlineKeyboardButton(str(i + 1), callback_data=_encode_callback("fact", str(i))) for i in range(len(window))
    ]
    nav_row: list[InlineKeyboardButton] = []
    if clamped > 0:
        nav_row.append(
            InlineKeyboardButton(
                "< prev",
                callback_data=_encode_callback("facts", str(clamped - 1)),
            )
        )
    nav_row.append(InlineKeyboardButton("back", callback_data=_encode_callback("dash")))
    if clamped < total - 1:
        nav_row.append(
            InlineKeyboardButton(
                "next >",
                callback_data=_encode_callback("facts", str(clamped + 1)),
            )
        )
    return text, InlineKeyboardMarkup([number_row, nav_row]), memory_ids, clamped, total


# ── Builder: fact view ──────────────────────────────────────────────


def _build_fact_view(
    fact: MemoryResult,
    return_to: tuple[str, list[str]] | None,
) -> tuple[str, InlineKeyboardMarkup]:
    """Render the fact detail screen (spec §6.3).

    Three render shapes (issue #407), branched on `metadata.source`:
      - extracted: `Fact` header + the existing six-row block (Tags,
        Confidence, Date, Session, Prompt version, Confirmation).
        Unchanged from the pre-#407 surface.
      - episode:   `Episode (<outcome_quality>)` header (or `Episode`
        when outcome_quality metadata is absent) + Tags + Date.
        Confidence/Session/Prompt version/Confirmation are omitted
        because episode rows do not carry those extractor-only fields;
        rendering them as placeholders ("----" / "(none)" / "n/a")
        would imply the data exists and is missing rather than
        not-applicable.
      - migration: `Imported` header + Tags + Date. Header chosen to
        distinguish operator-curated MEMORY.md content from
        Haiku-extracted facts at a glance, even though both render
        with the same `_SOURCE_SHORT="fact"` in retrieval-time prompt
        injection (see memory.py); the fact-view is operator-facing
        and benefits from the visual distinction.

    `return_to` encodes where the back button should land. It can be
    None for callers (e.g., tests) that don't care; in that case the
    back button defaults to the dashboard. The keyboard is the same
    (back / forget) across all three sources.
    """
    md = fact.metadata or {}
    tags = md.get("tags") or []
    source = md.get("source", "")

    if source == "episode" or source == "migration":
        # Episode and migration rows share a minimal body shape:
        # Tags + Date with none of the extractor-only rows. Header
        # differs (Episode (<outcome_quality>) vs Imported), and
        # episodes (issue #412) splice in the four substantive
        # Sophia fields (Approach, Outcome, Lessons-when-present,
        # Actors) between the body quote and the Tags / Date footer.
        # Migration rows skip the splice entirely - they have no
        # equivalent metadata.
        tags_line = f"Tags:  {', '.join(tags) if tags else '(none)'}"
        detail_lines: list[str] = []
        if source == "episode":
            # Episode rows surface the Sophia outcome_quality field as
            # a header parenthetical when present (e.g., "Episode
            # (good)") so the operator can tell at a glance how the
            # conversation resolved. Date renders with HH:MM precision
            # so episodes from the same day are distinguishable.
            quality = md.get("outcome_quality") or ""
            header = f"Episode ({quality})" if quality else "Episode"
            # Substantive Sophia fields written by the stage-2
            # generator's metadata-write block in memory_extraction.
            # approach / outcome / actors are schema-required and
            # always present in production; the `or ""` / `or []`
            # defensive fallbacks surface malformed-data corruption
            # rather than hiding it. lessons is optional-by-design
            # and absent (not empty) when the generator chose to
            # omit it - rendering `Lessons:  (none)` would lie about
            # that design intent.
            approach = md.get("approach") or ""
            outcome = md.get("outcome") or ""
            actors_list = md.get("actors") or []
            actors_str = ", ".join(actors_list) if actors_list else "(none)"
            lessons = md.get("lessons")
            detail_lines.append(f"Approach:  {approach}")
            detail_lines.append(f"Outcome:  {outcome}")
            # Presence-check, NOT truthy-check. The schema's
            # minLength=20 makes the two equivalent in practice today,
            # but `is not None` matches the storage contract literally
            # (absent key is the sentinel for "no lesson this time")
            # and protects against future schema relaxation.
            if lessons is not None:
                detail_lines.append(f"Lessons:  {lessons}")
            detail_lines.append(f"Actors:  {actors_str}")
        else:
            # Migration rows already include the H3 title and section
            # structure inside `fact.text` (the migration script writes
            # the chunk as `### <title>\n<body>` per #408), so no
            # additional section rendering is needed here. The header
            # diverges from the prompt-side `_SOURCE_SHORT="fact"` on
            # purpose - see docstring.
            header = "Imported"
        lines = [
            header,
            "",
            f'"{fact.text}"',
            "",
            *detail_lines,
            tags_line,
            f"Date:  {_format_date(fact.created_at, with_time=True)}",
        ]
    else:
        # Extracted (or any defensive fallback for an unknown source
        # that managed to reach this view). Renders the original
        # six-row block unchanged so existing screenshots / muscle
        # memory stay valid.
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
    """Render the single-fact forget confirmation.

    Issue #407 expanded the noun in the prompt text per source so the
    operator sees what is being forgotten (`fact` for extracted,
    `episode` for episode summaries, `imported memory` for migration
    rows). The verb (`Forget`), the warning sentence (`This cannot
    be undone.`), and the inline-button flow (`confirm forget` /
    `cancel`) are all preserved unchanged. Falls back to the generic
    label `memory` for an unknown source value (defensive: should be
    unreachable since `get_by_id` already gates on
    `USER_VISIBLE_SOURCES`, but a stale cache during a deploy
    transition could in principle slip a different source through).
    """
    source = (fact.metadata or {}).get("source", "")
    label = {"extracted": "fact", "episode": "episode", "migration": "imported memory"}.get(source, "memory")
    text = f'Forget this {label}?\n\n"{fact.text}"\n\nThis cannot be undone.'
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
    cache storage - same contract as the episode list view.
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

    Issue #407 expanded the headline to surface per-source totals for
    every user-visible source with non-zero rows (extracted facts,
    episode summaries, migration imports). The three extracted-shaped
    sections that follow (confidence block, confirmed-actions line,
    prompt-version table) are conditional on `extracted_count > 0`
    because they all read extractor-only metadata; for an episode-only
    or migration-only operator they would be all-n/a noise without
    informational value.
    """
    # Combined empty-state guard (issue #407): only when every
    # user-visible source is empty does the corpus count as "no
    # memories yet." The wording was updated from the original
    # "No extracted facts yet" because the original would lie about
    # the empty-state's scope when the post-#407 surface knows about
    # episode and migration rows too.
    if stats.extracted_count == 0 and stats.episode_count == 0 and stats.migration_count == 0:
        text = "Memory stats\n\nNo facts, episodes, or imported memory yet."
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("back", callback_data=_encode_callback("dash"))]])
        return text, kb

    # Headline block: per-source totals, one line per non-zero source.
    # Right-padding holds the "Total <source>:" labels at consistent
    # width across the three header lines so the count column lines
    # up. Lines for zero-valued counts are omitted so a fresh
    # extracted-only operator does not see two empty "Total episodes:"
    # / "Total imported:" rows below their facts total.
    lines = ["Memory stats", ""]
    if stats.extracted_count:
        lines.append(f"Total facts:      {stats.extracted_count}")
    if stats.episode_count:
        lines.append(f"Total episodes:   {stats.episode_count}")
    if stats.migration_count:
        lines.append(f"Total imported:   {stats.migration_count}")

    # The three extracted-shaped sections below all read extractor-only
    # metadata (confidence, confirmation_quote, prompt_version). For
    # an episode-only or migration-only operator (extracted_count == 0
    # but at least one other source > 0), rendering these would produce
    # an all-n/a confidence block and zero confirmed-actions noise
    # without informational value. The headline alone is the operator's
    # signal in that case.
    if stats.extracted_count > 0:
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


# ── Top-level command handler ───────────────────────────────────────


async def handle_memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle `/memory [subcommand ...]`.

    Dispatches on the first arg:
      (no args)      → dashboard
      help / unknown → help text
      stats          → stats screen
      search <query> → search (empty query falls through to help)
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

    # Unknown subcommand falls through to help (so an unrecognized
    # word, including the retired `forget` subcommand, lands the
    # operator on the syntax-reminder text).
    await update.message.reply_text(_HELP_TEXT)


# ── Top-level callback handler ──────────────────────────────────────


async def handle_memory_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Dispatch `mem:*` inline-keyboard callbacks.

    Verbs:
      dash              - re-render dashboard
      eps <page>        - episode list at page
      facts <page>      - facts list (extracted + migration) at page
      fact <idx>        - open fact at index (resolved against cache)
      stats             - re-render stats
      help              - help text (Search button on dashboard)
      ffc               - forget single fact confirm
      ffd               - forget single fact: do delete
      fview             - cancel forget; return to fact view (same id)
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
        if verb == "eps":
            # Episode list browser (issue #410). Single-arg page
            # index; invalid integer falls back to page 0 so a stale
            # callback never blocks navigation.
            try:
                page = int(args[0]) if args else 0
            except ValueError:
                page = 0
            await _send_episode_list(update, context, chat_id, page, edit=True)
            await query.answer()
            return
        if verb == "facts":
            # Facts list browser (extracted + migration). Same page-
            # arg shape as `eps`; invalid integer falls back to page
            # 0. The unknown-verb dismiss at the bottom of this
            # handler covers any future retirement of the verb,
            # which would silently no-op stale callbacks in chat
            # history (same compatibility path the retired `tag`
            # verb relies on).
            try:
                page = int(args[0]) if args else 0
            except ValueError:
                page = 0
            await _send_facts_list(update, context, chat_id, page, edit=True)
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
            # fact view. Episode list, facts list, or dashboard
            # fallback. The branches mirror the return_to encodings
            # set in _send_fact_view.
            if return_to is not None and return_to[0] == "eps" and len(return_to[1]) >= 1:
                # Issue #410: facts opened from the episode list
                # return to that list at the same page after delete.
                try:
                    page = int(return_to[1][0])
                except ValueError:
                    page = 0
                await _send_episode_list(update, context, chat_id, page, edit=True)
            elif return_to is not None and return_to[0] == "facts" and len(return_to[1]) >= 1:
                # Mirrors the eps branch above: facts opened from the
                # facts list return to that list at the same page
                # after delete, so the operator can keep deleting
                # contiguous rows without bouncing back to the
                # dashboard each time.
                try:
                    page = int(return_to[1][0])
                except ValueError:
                    page = 0
                await _send_facts_list(update, context, chat_id, page, edit=True)
            else:
                await _send_dashboard(update, context, chat_id, edit=True)
            await query.answer("Forgotten." if ok else "Not found.")
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
    # any stale fact button taps (from a previous search or episode
    # list) trip the session-expired branch instead of resolving to
    # a wrong fact.
    _set_cache(chat_id, _ScreenCache(screen="dashboard"))
    await _send_or_edit(update, text, kb, edit=edit)


async def _send_episode_list(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    page: int,
    edit: bool = False,
) -> None:
    """Render and send/edit the episode list at `page` (issue #410).

    Fetches via `memory.get_all_episodes`, builds via
    `_build_episode_list_view`, and sets the cache screen to
    `"episodes"` so a fact opened from this list can route back to
    the same page via `_send_fact_view`'s return_to logic.
    """
    try:
        episodes = memory.get_all_episodes(user_id=str(chat_id))
    except Exception as exc:
        log.exception("get_all_episodes failed: %s", exc)
        await _send_or_edit(update, _MSG_QUERY_FAILED, None, edit=edit)
        return
    text, kb, memory_ids, clamped_page, _total = _build_episode_list_view(episodes, page)
    _set_cache(
        chat_id,
        _ScreenCache(screen="episodes", page=clamped_page, memory_ids=memory_ids),
    )
    await _send_or_edit(update, text, kb, edit=edit)


async def _send_facts_list(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    page: int,
    edit: bool = False,
) -> None:
    """Render and send/edit the facts list at `page`.

    Fetches via `memory.get_all_facts` (the fact-bucket enumeration
    that folds extracted and migration), builds via
    `_build_facts_list_view`, and sets the cache screen to `"facts"`
    so a fact opened from this list can route back to the same page
    via `_send_fact_view`'s return_to logic. Distinct from the
    `"fact"` (singular) sentinel, which identifies the fact-detail
    view.
    """
    try:
        facts = memory.get_all_facts(user_id=str(chat_id))
    except Exception as exc:
        log.exception("get_all_facts failed: %s", exc)
        await _send_or_edit(update, _MSG_QUERY_FAILED, None, edit=edit)
        return
    text, kb, memory_ids, clamped_page, _total = _build_facts_list_view(facts, page)
    _set_cache(
        chat_id,
        _ScreenCache(screen="facts", page=clamped_page, memory_ids=memory_ids),
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
    if cache is not None and cache.screen == "episodes":
        # Issue #410: facts opened from the episode list return to
        # that list at the same page on back-nav. Page is the only
        # arg needed; the eps verb decodes a single-element args list.
        return_to = ("eps", [str(cache.page)])
    elif cache is not None and cache.screen == "facts":
        # Mirrors the episodes branch: facts opened from the facts
        # list return to that list at the same page on back-nav.
        # The new `"facts"` cache sentinel (set by `_send_facts_list`)
        # is what distinguishes this path from the search path; the
        # encoded back-target is `mem:facts:<page>`.
        return_to = ("facts", [str(cache.page)])
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
    # Also enforce the user-visible source admit list here. Every
    # other read path in this module (`get_by_tag`, `get_by_id`)
    # filters at the data layer, but `memory.search` is a Mem0 vector
    # lookup that spans every source, including legacy ""-source rows
    # that must not surface in the operator-facing UI. Without this
    # post-filter, a search hit could surface a non-user-visible row;
    # tapping it would call `get_by_id`, fail the admit check, and
    # render "This memory no longer exists." for a row the user just
    # saw - confusing and wrong. Filtering here keeps the UI honest:
    # what the user sees in results is what they can act on.
    # `USER_VISIBLE_SOURCES` (the {extracted, episode, migration}
    # frozenset) is read from `memory.py` so a future change to the
    # admit list lives in one place.
    filtered = [r for r in results if r.score >= floor and r.metadata.get("source") in memory.USER_VISIBLE_SOURCES]
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
    everywhere else in this module (_encode_callback, _send_or_edit).
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
