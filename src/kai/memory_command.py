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
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from kai import memory
from kai.config import ONESHOT_REASONER_BACKENDS, Config, MemoryProjectConfig
from kai.history import (
    TranscriptContext,
    TranscriptLookup,
    TranscriptTurn,
    fetch_transcript_context,
)
from kai.memory import (
    MemoryResult,
    MemoryStats,
    ResolvedMemoryScope,
    TranscriptProvenance,
    read_transcript_provenance,
)
from kai.memory_projects import ActiveMemoryProject, detect_active_memory_project, merged_registry

if TYPE_CHECKING:
    # Only used for type hints; importing at runtime would create a
    # cycle since bot.py imports this module.
    from kai.pool import SubprocessPool

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

    `scope_targets` backs the scope screen the same way `memory_ids`
    backs the list views: each entry is a (kind, project_id) tuple
    where kind is "global" (project_id None) or "project". The target
    buttons carry only the integer index, so a long project id can
    never push callback data over the Telegram 64-byte ceiling. A
    project id stored here may also collide with no reserved word:
    the tuple's kind discriminator is what distinguishes "move to
    the project named 'global'" from "make global".

    `scope_target` is the entry the user tapped, carried from the
    scope screen to its confirm step so the apply verb needs no
    callback arguments at all.
    """

    screen: str
    memory_ids: list[str] = field(default_factory=list)
    page: int = 0
    query: str | None = None
    return_to: tuple[str, list[str]] | None = None
    scope_targets: list[tuple[str, str | None]] = field(default_factory=list)
    scope_target: tuple[str, str | None] | None = None
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

# Help text - the static body shown for `/memory help` and for any
# unrecognized `/memory <subcommand>` invocation. Both code paths
# call `update.message.reply_text(_HELP_TEXT)`, which has no length
# cap, so the string can grow to whatever wording is useful without
# the prior 200-char ceiling that the deleted dashboard alert-toast
# caller imposed.
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


# ── Speaker-aware browse helpers ───────────────────────────────────


# Display labels for the speaker enum. Stored values are lower-snake
# ("user" / "assistant" / "episode_summary"); the operator-facing UI
# renders them title-cased and space-separated. Centralized here so
# the fact-detail screen and any future surface use the same labels.
_SPEAKER_DISPLAY_LABELS: dict[str, str] = {
    "user": "User",
    "assistant": "Assistant",
    "episode_summary": "Episode summary",
}


def _humanize_speaker(speaker: str) -> str:
    """Return the operator-facing display label for a speaker value.

    Falls back to a title-cased, underscore-stripped version of the
    raw value for any speaker class that lacks an explicit entry. The
    fallback exists so a future addition to the speaker enum (added
    upstream before this map is updated) renders something readable
    instead of leaking the internal snake_case identifier.
    """
    label = _SPEAKER_DISPLAY_LABELS.get(speaker)
    if label is not None:
        return label
    return speaker.replace("_", " ").capitalize()


def _browse_score(fact: MemoryResult) -> float:
    """Browse-time quality score: cosine pinned at 1.0 * speaker_weight * confidence.

    Identical formula to the retrieval ranking with cosine pinned at
    1.0; means the browse default reflects the same quality signal
    that retrieval uses. Calls into kai.memory's `_speaker_weight`
    helper rather than re-deriving the formula here, so a calibration
    sweep that retunes speaker weights or changes the multiplier
    arithmetic does not require a parallel edit in the browse path.
    """
    return memory._speaker_weight(fact)


# ── Scope helpers ───────────────────────────────────────────────────
#
# Row-level scope inspection and correction. Reads go through
# `memory.resolve_memory_scope` so legacy and corrupted rows render
# their read-time interpretation (the same one retrieval acts on)
# rather than echoing raw metadata. The retrievability verdict reuses
# `memory._scoped_memory_admission_reason` - the exact admission rule
# scoped retrieval applies - so the detail view can never disagree
# with what scoped retrieval would do. Writes go through
# `memory.build_scope_metadata` (validated shape, operator
# provenance) merged over the row's existing metadata, because
# `memory.update_metadata` replaces the metadata dict wholesale and
# a partial dict would silently destroy every unlisted field.


@dataclass(frozen=True)
class _ScopeView:
    """Pre-rendered scope strings for the detail and scope screens.

    Built once per render by `_build_scope_view` and passed into the
    pure builders, so the builders stay free of registry and
    detection I/O.

    Attributes:
        resolved: The row's read-time scope interpretation; kept so
            the scope screen can derive transition targets without
            re-resolving.
        scope_label: Operator-facing scope value, e.g. "global" or
            "project 'kai'", with an unregistered-project or
            missing-id marker when applicable.
        source_label: Scope provenance with confidence, e.g.
            "operator (confidence 1.00)". Values render raw
            (legacy_default, extraction_default, ...) so the screen
            greps against the same vocabulary the logs use.
        retrievable_label: "yes", or "no (<admission reason>)" with
            the stable exclusion-reason key retrieval logs carry.
            Always computed under scoped admission, matching the
            only live recall path.
    """

    resolved: ResolvedMemoryScope
    scope_label: str
    source_label: str
    retrievable_label: str


def _build_scope_view(
    fact: MemoryResult,
    registry: dict[str, MemoryProjectConfig],
    active: ActiveMemoryProject | None,
) -> _ScopeView:
    """Resolve a row's scope into operator-facing display strings.

    `registry` is the merged project registry (operator-pinned YAML
    over chat-registered rows) and `active` is the project detected
    for the caller's current workspace, or None. Both come from
    `_scope_inputs`; tests can pass fixtures directly.

    The retrievability verdict is always computed under scoped
    admission because scoped retrieval is the only live recall path.

    The allowed-project derivation mirrors scoped retrieval exactly:
    project authority exists only when a project is detected AND has
    memory enabled. Keeping the two predicates identical is what the
    admission-parity guarantee rests on.
    """
    resolved = memory.resolve_memory_scope(fact.metadata)

    # Scope label. Project rows render their id plus the registered
    # display name when it differs; a project id that is no longer
    # in the registry is flagged rather than hidden, because the row
    # is still movable and the operator needs to see why it stopped
    # being retrievable anywhere.
    if resolved.scope == memory.SCOPE_PROJECT:
        pid = resolved.project_id
        if pid is None:
            scope_label = "project (no project id)"
        elif pid in registry:
            display = registry[pid].display_name
            scope_label = f"project '{pid}'" if display == pid else f"project '{pid}' ({display})"
        else:
            scope_label = f"project '{pid}' (not registered)"
    else:
        scope_label = resolved.scope

    source_label = f"{resolved.scope_source} (confidence {resolved.scope_confidence:.2f})"

    # Retrievability verdict under scoped admission. Same
    # allowed-project derivation and same admission rule as the live
    # scoped retrieval path. The two enriched arms add the context an
    # operator cannot reconstruct from the bare reason key (which two
    # projects mismatched; whether "not allowed" means no project here
    # or a disabled one).
    allowed = active.project_id if active is not None and active.memory_enabled else None
    reason = memory._scoped_memory_admission_reason(resolved, allowed_project_id=allowed)
    if reason is None:
        retrievable_label = "yes"
    elif reason == memory._ADMISSION_PROJECT_ID_MISMATCH:
        retrievable_label = f"no ({reason}: row '{resolved.project_id}', here '{allowed}')"
    elif reason == memory._ADMISSION_PROJECT_SCOPE_NOT_ALLOWED:
        detail = "project memory disabled here" if active is not None else "no active project here"
        retrievable_label = f"no ({reason}: {detail})"
    else:
        retrievable_label = f"no ({reason})"

    return _ScopeView(
        resolved=resolved,
        scope_label=scope_label,
        source_label=source_label,
        retrievable_label=retrievable_label,
    )


def _scope_change_targets(
    resolved: ResolvedMemoryScope,
    registry: dict[str, MemoryProjectConfig],
) -> list[tuple[str, str | None]]:
    """Derive the scope transitions offered for a row.

    Returns (kind, project_id) tuples in render order: the global
    target first when offered, then project targets sorted by id for
    deterministic keyboards.

    Rules:
    - "Make global" is offered unless the row is already explicitly
      global with valid provenance. Legacy-default and invalid rows
      DO get the global target even though they already resolve to
      global: applying it stamps explicit operator provenance, which
      converts an auditable-debt row into a deliberate assignment.
    - Every registered project is a move target except the row's own
      project - unless the row is invalid-flagged, where re-assigning
      the same project is meaningful because it repairs the broken
      provenance while keeping the assignment.
    """
    targets: list[tuple[str, str | None]] = []
    already_explicit_global = (
        resolved.scope == memory.SCOPE_GLOBAL and not resolved.legacy_defaulted and not resolved.invalid_defaulted
    )
    if not already_explicit_global:
        targets.append(("global", None))
    for pid in sorted(registry):
        if pid == resolved.project_id and not resolved.invalid_defaulted:
            continue
        targets.append(("project", pid))
    return targets


async def _scope_inputs(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
) -> tuple[dict[str, MemoryProjectConfig], ActiveMemoryProject | None]:
    """Fetch the merged registry and the caller's active project.

    The workspace comes from the subprocess pool's effective resolver
    (the same per-user workspace scoped retrieval sees on the next
    turn). Calling the resolver here rather than the sync getter
    means the saved workspace gets restored eagerly when needed, so
    /memory's scope detection lines up with the user's settings
    instead of the home default the lazy pool would otherwise show.
    A missing pool collapses to no-workspace semantics: no path
    means no project authority, the same global-only posture scoped
    retrieval takes for a None workspace.
    """
    config: Config = context.bot_data["config"]
    registry = merged_registry(config.memory_projects)
    pool: SubprocessPool | None = context.bot_data.get("pool")
    if pool is None:
        return registry, None
    workspace = await pool.get_effective_workspace(chat_id)
    return registry, detect_active_memory_project(workspace, registry)


# ── Transcript provenance helpers ───────────────────────────────────
#
# Row-side pointer to the originating turns. Reads go through
# `read_transcript_provenance`; the transcript lookup happens at
# `source` button tap time via `fetch_transcript_context`. The
# detail view always renders a Source line (legacy fallback for
# rows without provenance), and the keyboard gains a `source`
# button only when provenance is present.


# Safe ceiling on the source-view body; Telegram's hard limit is
# 4096 characters and `_send_or_edit` does not truncate, so the
# renderer enforces a ceiling under that with a visible marker.
# 3900 leaves headroom for any future footer (currently none).
_SOURCE_VIEW_BODY_CEILING = 3900

# Per-turn cap inside the source view. Most ordinary fact lookups
# (target user + assistant + a few context turns at this length)
# stay well under the body ceiling; the marker below catches any
# pathological exchange that does not.
_SOURCE_VIEW_TURN_CAP = 400

_SOURCE_VIEW_TRUNCATION_MARKER = "\n... (output truncated)"


def _format_source_ts(ts: str) -> str:
    """Render an ISO 8601 ts as `YYYY-MM-DD HH:MM:SS UTC`.

    The stored ts is always UTC (it comes from `datetime.now(UTC)`),
    so the suffix is hardcoded rather than parsed; future display-
    timezone conversion is a separate concern. A malformed ts falls
    through unchanged so a corrupted row still renders something
    legible.
    """
    if "T" in ts and len(ts) >= 19:
        date, rest = ts.split("T", 1)
        return f"{date} {rest[:8]} UTC"
    return ts


def _build_source_label(fact: MemoryResult, provenance: TranscriptProvenance) -> str:
    """Render the Source row's right-hand value.

    Three cases keyed on the row's source field (extracted vs
    episode/migration) and on whether the user and assistant turns
    fall on the same UTC date:
    - Legacy / not-present: `not recorded (legacy)`.
    - Fact: `YYYY-MM-DD HH:MM:SS UTC` (single user ts; the assistant
      date is implicit in `source_assistant_ts`).
    - Episode: `YYYY-MM-DD HH:MM:SS to HH:MM:SS UTC` (same-day) or
      `YYYY-MM-DD HH:MM:SS to YYYY-MM-DD HH:MM:SS UTC` (midnight
      cross). The full two-date form lets the operator see the
      exchange straddled a day boundary without opening the source
      view.
    """
    if not provenance.present:
        return "not recorded (legacy)"
    source = (fact.metadata or {}).get("source", "")
    if source != "episode" or provenance.assistant_ts is None:
        # Fact and migration arms render the single user ts. Migration
        # rows do not carry provenance today (the writer does not
        # stamp `source_*`), so the not-present branch above usually
        # catches them; the defensive single-ts render here is for any
        # future caller that does stamp a migration row.
        return _format_source_ts(provenance.user_ts or "")
    user_part = _format_source_ts(provenance.user_ts or "")
    assistant_part = _format_source_ts(provenance.assistant_ts)
    # Same-day episodes drop the second date so the line reads as
    # "DATE HH:MM:SS to HH:MM:SS UTC" rather than repeating the date.
    # Midnight cross keeps both dates so the boundary is visible at a
    # glance.
    if provenance.date_end is None or provenance.date_end == provenance.date:
        # _format_source_ts produces "DATE HH:MM:SS UTC"; trim the
        # trailing "UTC" from the user part and the leading "DATE "
        # from the assistant part so the joined form reads correctly.
        # The replacement happens by string surgery rather than
        # restructuring the formatter to keep _format_source_ts the
        # single source of truth for the ts shape.
        if user_part.endswith(" UTC"):
            user_part = user_part[: -len(" UTC")]
        if " " in assistant_part:
            _, assistant_tail = assistant_part.split(" ", 1)
            assistant_part = assistant_tail
        return f"{user_part} to {assistant_part}"
    return f"{user_part} to {assistant_part}"


def _truncate_to_message_limit(body: str) -> str:
    """Trim a source-view body to fit under Telegram's 4096-char limit.

    `_send_or_edit` sends the supplied text as-is and does not
    truncate; without this clamp, an oversized body would surface
    as a Telegram `BadRequest` and the callback would collapse to
    the generic memory-query failure path. The marker is appended
    visibly so the operator knows the body was cut rather than
    rendered fully.
    """
    if len(body) <= _SOURCE_VIEW_BODY_CEILING:
        return body
    keep = _SOURCE_VIEW_BODY_CEILING - len(_SOURCE_VIEW_TRUNCATION_MARKER)
    return body[: max(keep, 0)] + _SOURCE_VIEW_TRUNCATION_MARKER


def _truncate_source_turn(text: str) -> str:
    """Cap a single turn's text inside the source view.

    Primary defence against an oversized body; the body-level clamp
    catches the residual pathological case where even the capped
    turns add up to more than the body ceiling.
    """
    flat = text.replace("\r", "")
    if len(flat) <= _SOURCE_VIEW_TURN_CAP:
        return flat
    return flat[: _SOURCE_VIEW_TURN_CAP - 1] + "…"


# ── Builder: dashboard ──────────────────────────────────────────────


def _build_dashboard(stats: MemoryStats) -> tuple[str, InlineKeyboardMarkup | None]:
    """Render the dashboard text and keyboard from a MemoryStats.

    The dashboard surfaces two user-visible counts (facts and episode
    summaries) and a single utility keyboard row holding two optional
    browse buttons (Facts when extracted_count + migration_count > 0;
    Episodes when episode_count > 0), plus an unconditional Stats
    button. There is no per-source filter axis: the parent decision
    in #388 settled tags as row decoration only, not a primary
    browse axis.

    Facts vs Episodes split: extracted and migration rows fold into a
    single "facts" bucket because the extracted/migration distinction
    is internal plumbing from an operator's perspective. The headline
    sums them; the fact-view detail screen renders both with the same
    `Fact` header. Episodes have a distinct semantic shape (outcome
    quality, approach, lessons) that justifies a separate browser.

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
    # Headline assembled from non-zero counts. The fact bucket sums
    # extracted and migration into a single "facts" count because the
    # operator does not need to be reminded that some facts arrived
    # via migration rather than extraction. The summing matches the
    # Facts button label so the headline and the keyboard agree on
    # the same number. Zero-valued counts are omitted from the comma
    # list rather than rendered as "0 episodes" - readability over
    # uniformity, since most operators have non-zero in only one or
    # both of the two count buckets.
    parts: list[str] = []
    if facts_visible:
        parts.append(f"{facts_count} facts")
    if episodes_visible:
        parts.append(f"{stats.episode_count} episodes")
    summary = "Memories: " + ", ".join(parts) + "."
    lines.append(summary)
    lines.append("")
    # Footer line: three reachable branches keep the action prompt
    # accurate to what the keyboard actually offers. The fourth
    # branch (no browse buttons at all) is unreachable in production
    # because the empty-state guard above would have returned; the
    # `else` arm is retained as a safety net so a future change to
    # the empty-state guard cannot silently produce a broken footer.
    # Two-sentence form ("Tap X to browse. Tap Stats for details.")
    # reads cleaner than the prior comma-spliced form once the Search
    # call-out is gone.
    #   - Both browse buttons: name both, plus Stats for details.
    #   - Facts only: Tap Facts to browse. Tap Stats for details.
    #   - Episodes only: same shape with Episodes.
    #   - Neither (unreachable): Tap Stats for details.
    if facts_visible and episodes_visible:
        lines.append("Tap Facts or Episodes to browse. Tap Stats for details.")
    elif facts_visible:
        lines.append("Tap Facts to browse. Tap Stats for details.")
    elif episodes_visible:
        lines.append("Tap Episodes to browse. Tap Stats for details.")
    else:
        lines.append("Tap Stats for details.")

    text = "\n".join(lines)

    # Keyboard: a single utility row holds the cross-corpus actions
    # in this exact left-to-right order: Facts (if any), Episodes
    # (if any), Stats. Both browse buttons hide when their bucket is
    # empty so an operator does not see a button that would open an
    # empty list. Stats always renders. The Search button is gone:
    # Telegram's inline keyboard cannot accept text input, so the
    # button could never actually run a query; the slash-command
    # path `/memory search <q>` is the search entry point.
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

    Per-row layout matches the facts list:
        N.  <truncated text>
            <date>

    The `outcome_quality` field is still load-bearing in the episode
    detail view (header parenthetical: `Episode (success)`) but is
    not surfaced in the list view; per-row quality bracketing was
    visual noise that did not help triage at the row level.

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
        date_str = _format_date(fact.updated_at or fact.created_at)
        lines.append(f"{idx}.  {_truncate(fact.text)}")
        # Four-space indent lines up the date under the start of the
        # truncated fact text on the line above (after the "N.  "
        # prefix). Matches the facts list convention so both list
        # views render with identical alignment.
        lines.append(f"    {date_str}")
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
    sort: Literal["quality", "recent"] = "quality",
) -> tuple[str, InlineKeyboardMarkup, list[str], int, int]:
    """Render a paginated facts list folding extracted and migration.

    Returns the (text, keyboard, memory_ids, clamped_page,
    total_pages) tuple consumed by `_send_facts_list`. The
    memory_ids list backs the screen cache so numbered button taps
    resolve to memory ids via integer index, keeping callback data
    well under the 64-byte Telegram ceiling.

    `sort` is consumed only for the toggle-button rendering: the
    builder is purely a view layer and does NOT sort the list. The
    caller (`_send_facts_list`) owns the sort decision so the same
    helper can be reused from non-browse callers (tests, future
    surfaces) without inheriting the browse-time default.

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
        # Four-space indent lines up the date under the start of the
        # truncated fact text on the line above (after the "N.  "
        # prefix). The episode list uses the same convention so both
        # list views render with identical alignment.
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
    # Sort toggle row above page nav. Each button carries its own
    # sort mode in callback_data so a tap re-renders at the same page
    # in the new mode. The active mode shows a leading checkmark; the
    # inactive mode renders unmarked. Marking via button text rather
    # than a separate state key means the rendered keyboard is
    # self-describing - a screenshot of the keyboard alone is enough
    # to tell which mode is active.
    quality_label = "✓ Sort: Quality" if sort == "quality" else "Sort: Quality"
    recent_label = "✓ Sort: Recent" if sort == "recent" else "Sort: Recent"
    sort_row = [
        InlineKeyboardButton(
            quality_label,
            callback_data=_encode_callback("facts", str(clamped), "quality"),
        ),
        InlineKeyboardButton(
            recent_label,
            callback_data=_encode_callback("facts", str(clamped), "recent"),
        ),
    ]
    nav_row: list[InlineKeyboardButton] = []
    if clamped > 0:
        nav_row.append(
            InlineKeyboardButton(
                "< prev",
                callback_data=_encode_callback("facts", str(clamped - 1), sort),
            )
        )
    nav_row.append(InlineKeyboardButton("back", callback_data=_encode_callback("dash")))
    if clamped < total - 1:
        nav_row.append(
            InlineKeyboardButton(
                "next >",
                callback_data=_encode_callback("facts", str(clamped + 1), sort),
            )
        )
    return text, InlineKeyboardMarkup([number_row, sort_row, nav_row]), memory_ids, clamped, total


# ── Builder: fact view ──────────────────────────────────────────────


def _build_fact_view(
    fact: MemoryResult,
    return_to: tuple[str, list[str]] | None,
    scope_view: _ScopeView | None = None,
    provenance: TranscriptProvenance | None = None,
) -> tuple[str, InlineKeyboardMarkup]:
    """Render the fact detail screen (spec §6.3).

    Three render shapes, branched on `metadata.source`:
      - extracted: `Fact` header + the existing six-row block (Tags,
        Confidence, Date, Session, Prompt version, Confirmation).
        Unchanged from the pre-#407 surface.
      - episode:   `Episode (<outcome_quality>)` header (or `Episode`
        when outcome_quality metadata is absent) + Approach / Outcome
        / Lessons (when present) / Actors / Tags / Date with blank
        lines between every labelled section. Confidence/Session/
        Prompt version/Confirmation are omitted because episode rows
        do not carry those extractor-only fields; rendering them as
        placeholders ("----" / "(none)" / "n/a") would imply the data
        exists and is missing rather than not-applicable.
      - migration: `Fact` header + Tags + Date. Migration rows do not
        carry extractor-only metadata (Confidence/Session/Prompt
        version/Confirmation) so the body shape stays minimal; the
        same not-applicable rationale as the episode arm applies. The
        header now matches extracted rather than calling out the
        source distinction; from the operator's perspective extracted
        and migration are both "facts" and the data-layer source
        distinction is internal plumbing.

    `return_to` encodes where the back button should land. It can be
    None for callers (e.g., tests) that don't care; in that case the
    back button defaults to the dashboard. The keyboard is the same
    (back / scope / forget) across all three sources.

    `scope_view` carries the pre-rendered scope block (scope value,
    provenance, retrievability verdict). Every production caller
    passes it; None omits the block so direct builder callers that
    are not exercising scope rendering need no registry fixture.
    All three source arms render the same three scope rows because
    scope is a cross-source axis - an episode is just as capable of
    being mis-scoped as a fact.
    """
    md = fact.metadata or {}
    tags = md.get("tags") or []
    source = md.get("source", "")

    # Speaker (and confidence, where applicable) come from the read-
    # time helper rather than directly from metadata so legacy rows
    # missing the new fields display the documented defaults instead
    # of "----" / "(none)" placeholders. The helper returns
    # (speaker, confidence); confidence is dropped on the episode
    # render path because the constant 1.0 is not informative for
    # operator-side review.
    speaker_value, confidence_value = memory._read_time_speaker(md)
    speaker_label = _humanize_speaker(speaker_value)

    if source == "episode" or source == "migration":
        # Episode and migration rows share a minimal body shape with
        # none of the extractor-only rows. The episode arm splices in
        # Approach / Outcome / Lessons (when present) / Actors between
        # the body quote and the Tags / Date footer, with blank lines
        # between every labelled section so long Approach/Outcome
        # paragraphs do not visually fuse together. Migration rows
        # skip the splice entirely (they have no equivalent metadata)
        # so detail_lines stays empty and the body collapses to the
        # quoted text plus the Tags / Date footer.
        tags_line = f"Tags:  {', '.join(tags) if tags else '(none)'}"
        speaker_line = f"Speaker:  {speaker_label}"
        # Migration rows render Confidence (always 0.9 via the
        # migration default), episode rows omit it (the constant 1.0
        # would be operator-side noise).
        confidence_line = f"Confidence:  {confidence_value:.2f}" if source == "migration" else None
        detail_lines: list[str] = []
        if source == "episode":
            # Episode rows surface the outcome_quality field as a
            # header parenthetical when present (e.g., "Episode
            # (good)") so the operator can tell at a glance how the
            # conversation resolved. Date renders with HH:MM precision
            # so episodes from the same day are distinguishable.
            quality = md.get("outcome_quality") or ""
            header = f"Episode ({quality})" if quality else "Episode"
            # Substantive fields written by the stage-2 generator's
            # metadata-write block in memory_extraction. approach /
            # outcome / actors are schema-required and always present
            # in production; the `or ""` / `or []` defensive fallbacks
            # surface malformed-data corruption rather than hiding it.
            # lessons is optional-by-design and absent (not empty) when
            # the generator chose to omit it - rendering
            # `Lessons:  (none)` would lie about that design intent.
            approach = md.get("approach") or ""
            outcome = md.get("outcome") or ""
            actors_list = md.get("actors") or []
            actors_str = ", ".join(actors_list) if actors_list else "(none)"
            lessons = md.get("lessons")
            # Trailing blank after each labelled section so the splice
            # produces a blank line between every adjacent pair, plus
            # a final blank that separates the Actors row from the
            # Tags row appended below. Lessons-absent collapses to a
            # single break between Outcome and Actors because no
            # Lessons block (and no Lessons trailing blank) is added.
            detail_lines.append(f"Approach:  {approach}")
            detail_lines.append("")
            detail_lines.append(f"Outcome:  {outcome}")
            detail_lines.append("")
            # Presence-check, NOT truthy-check. The schema's
            # minLength=20 makes the two equivalent in practice today,
            # but `is not None` matches the storage contract literally
            # (absent key is the sentinel for "no lesson this time")
            # and protects against future schema relaxation.
            if lessons is not None:
                detail_lines.append(f"Lessons:  {lessons}")
                detail_lines.append("")
            detail_lines.append(f"Actors:  {actors_str}")
            detail_lines.append("")
        else:
            # Migration rows already include the H3 title and section
            # structure inside `fact.text` (the migration script writes
            # the chunk as `### <title>\n<body>` per #408), so no
            # additional section rendering is needed here. The header
            # matches extracted (`Fact`) rather than calling out
            # `Imported` because the operator-facing UI no longer
            # surfaces the extracted/migration distinction; the
            # data-layer `source` field stays unchanged.
            header = "Fact"
        # Footer row order: detail_lines (episode-only Approach/Outcome
        # /Lessons/Actors block, empty for migration), Tags, Speaker,
        # optional Confidence (migration only), Date. Speaker sits
        # next to Tags so the two prov/identity lines render adjacent;
        # Confidence trails when present so the date is the last line
        # and matches the existing migration/episode tail convention.
        footer_lines = [tags_line, speaker_line]
        if confidence_line is not None:
            footer_lines.append(confidence_line)
        footer_lines.append(f"Date:  {_format_date(fact.created_at, with_time=True)}")
        # Scope block trails the Date row so the existing tail
        # convention (date last among the legacy fields) stays
        # recognizable while the three scope rows read as one
        # appended unit.
        if scope_view is not None:
            footer_lines.append(f"Scope:  {scope_view.scope_label}")
            footer_lines.append(f"Scope source:  {scope_view.source_label}")
            footer_lines.append(f"Retrievable here:  {scope_view.retrievable_label}")
        # Source row appended last so the existing tail (date + scope)
        # stays recognizable. Always rendered when a provenance value
        # is supplied (legacy rows produce the documented fallback);
        # production callers always pass one.
        if provenance is not None:
            footer_lines.append(f"Source:  {_build_source_label(fact, provenance)}")
        lines = [
            header,
            "",
            f'"{fact.text}"',
            "",
            *detail_lines,
            *footer_lines,
        ]
    else:
        # Extracted (or any defensive fallback for an unknown source
        # that managed to reach this view). Renders the existing
        # extractor-only block (Tags / Confidence / Date / Session /
        # Prompt version / Confirmation) plus a Speaker line inserted
        # between Tags and Confidence so the two ranking-signal rows
        # are visually grouped. Confidence comes from the read-time
        # helper rather than directly from metadata so a row missing
        # the field (extracted-legacy) renders the documented default
        # instead of "----".
        session_id = md.get("session_id") or ""
        prompt_version = md.get("prompt_version") or ""
        confirmation = md.get("confirmation_quote") or ""

        # _read_time_speaker always returns a numeric confidence (the
        # legacy default for extracted-legacy rows is 0.5), so the
        # `----` fallback the older render carried for missing-field
        # rows is no longer reachable on this branch.
        conf_str = f"{float(confidence_value):.2f}"

        # Confirmation row: verbatim quote on confirmed_action facts,
        # literal "n/a" elsewhere. The schema invariant ("confirmation
        # _quote present iff tags includes confirmed_action", spec §4)
        # is asserted at extraction time, so we can render confidently
        # without re-validating.
        confirmation_line = confirmation if confirmation else "n/a"

        # Scope block sits with the other classification rows (after
        # the extractor provenance, before the Confirmation block)
        # using the same 18-column label alignment as the rows above.
        scope_rows: list[str] = []
        if scope_view is not None:
            scope_rows = [
                f"Scope:            {scope_view.scope_label}",
                f"Scope source:     {scope_view.source_label}",
                f"Retrievable here: {scope_view.retrievable_label}",
            ]
        # Source row uses the same 18-column alignment so the new line
        # reads as part of the existing extractor block, not as an
        # appendix. Always rendered when a provenance value is supplied.
        source_rows: list[str] = []
        if provenance is not None:
            source_rows = [f"Source:           {_build_source_label(fact, provenance)}"]
        lines = [
            "Fact",
            "",
            f'"{fact.text}"',
            "",
            f"Tags:             {', '.join(tags) if tags else '(none)'}",
            f"Speaker:          {speaker_label}",
            f"Confidence:       {conf_str}",
            f"Date:             {_format_date(fact.created_at, with_time=True)}",
            f"Session:          {session_id or '(none)'}",
            f"Prompt version:   {prompt_version or '(none)'}",
            *scope_rows,
            *source_rows,
            "",
            f"Confirmation:     {confirmation_line}",
        ]
    text = "\n".join(lines)

    back_callback = _encode_callback(return_to[0], *return_to[1]) if return_to is not None else _encode_callback("dash")
    # Source button is conditional: provenance-absent rows have no
    # exchange to reveal, so the button would route to a "legacy"
    # body that adds no operator value. Position is between back
    # and scope so the navigation buttons cluster on the left and
    # the action buttons (scope, forget) cluster on the right.
    row: list[InlineKeyboardButton] = [InlineKeyboardButton("back", callback_data=back_callback)]
    if provenance is not None and provenance.present:
        row.append(InlineKeyboardButton("source", callback_data=_encode_callback("src")))
    row.append(InlineKeyboardButton("scope", callback_data=_encode_callback("scp")))
    row.append(InlineKeyboardButton("forget", callback_data=_encode_callback("ffc")))
    kb = InlineKeyboardMarkup([row])
    return text, kb


# ── Builder: forget single-fact confirmation (spec §6.4) ────────────


def _build_forget_fact_confirm(fact: MemoryResult) -> tuple[str, InlineKeyboardMarkup]:
    """Render the single-fact forget confirmation.

    The noun in the prompt text varies per source so the operator
    sees what is being forgotten (`fact` for extracted and migration,
    `episode` for episode summaries). Migration rows share the
    extracted noun because the operator-facing UI does not surface
    the extracted/migration distinction; both are facts as far as
    the prompt wording goes. The verb (`Forget`), the warning
    sentence (`This cannot be undone.`), and the inline-button flow
    (`confirm forget` / `cancel`) are all preserved unchanged. Falls
    back to the generic label `memory` for an unknown source value
    (defensive: should be unreachable since `get_by_id` already
    gates on `USER_VISIBLE_SOURCES`, but a stale cache during a
    deploy transition could in principle slip a different source
    through).
    """
    label = _source_noun(fact)
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


def _source_noun(fact: MemoryResult) -> str:
    """Operator-facing noun for a row, keyed on `metadata.source`.

    Extracted and migration share "fact" because the UI does not
    surface that distinction; the generic "memory" fallback covers
    an unknown source slipping through a stale cache during a deploy
    transition (same defensive posture as the forget flow).
    """
    source = (fact.metadata or {}).get("source", "")
    return {"extracted": "fact", "episode": "episode", "migration": "fact"}.get(source, "memory")


# ── Builder: scope screen and confirm ───────────────────────────────


def _build_scope_screen(
    fact: MemoryResult,
    scope_view: _ScopeView,
    targets: list[tuple[str, str | None]],
) -> tuple[str, InlineKeyboardMarkup]:
    """Render the scope screen: current assignment plus transitions.

    The body repeats the detail view's scope block (in the two-space
    label style) above the transition prompt so the operator decides
    with the current assignment in front of them, not from memory of
    the previous screen.

    Keyboard shape: one target button per row. Project ids are
    operator-chosen and can be long; packing several per row would
    truncate labels on a phone, and the target count is bounded by
    the registry size, which is small by construction. The nav row
    holds a single back button to the fact view.

    Targets render by cache index (`sct:<idx>`), never by id, for
    the same 64-byte-ceiling reason the list views use indexed
    `fact` callbacks.
    """
    lines = [
        "Scope",
        "",
        f'"{fact.text}"',
        "",
        f"Scope:  {scope_view.scope_label}",
        f"Scope source:  {scope_view.source_label}",
        f"Retrievable here:  {scope_view.retrievable_label}",
        "",
    ]
    if targets:
        lines.append("Choose a new scope:")
    else:
        # Reachable when the row is explicitly global and no projects
        # are registered: nothing to move to, nothing to re-stamp.
        lines.append("No scope changes available (no registered projects).")
    text = "\n".join(lines)

    rows: list[list[InlineKeyboardButton]] = []
    for idx, (kind, pid) in enumerate(targets):
        label = "Make global" if kind == "global" else f"Move to '{pid}'"
        rows.append([InlineKeyboardButton(label, callback_data=_encode_callback("sct", str(idx)))])
    rows.append([InlineKeyboardButton("back", callback_data=_encode_callback("fview"))])
    return text, InlineKeyboardMarkup(rows)


def _build_scope_confirm(
    fact: MemoryResult,
    target: tuple[str, str | None],
) -> tuple[str, InlineKeyboardMarkup]:
    """Render the scope-change confirmation.

    Mirrors the forget confirmation's shape (question, quoted row
    text, confirm/cancel row) but drops the irreversibility warning:
    a scope change can be reversed by moving the row back. The
    confirm step exists anyway because the prior assignment is not
    visible after the change lands - a fat-fingered tap would leave
    the operator unsure what the row's scope used to be.

    Cancel returns to the scope screen (the screen the tap came
    from), matching the forget flow's own convention of cancelling
    back one step rather than to a fixed screen.
    """
    kind, pid = target
    noun = _source_noun(fact)
    question = f"Make this {noun} global?" if kind == "global" else f"Move this {noun} to project '{pid}'?"
    text = f'{question}\n\n"{fact.text}"'
    kb = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("confirm", callback_data=_encode_callback("scd")),
                InlineKeyboardButton("cancel", callback_data=_encode_callback("scp")),
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


# Display labels and render order for the fixed (non-project) scope
# buckets emitted by `memory.get_stats`. Projects render between
# global_legacy and task, sorted by count descending with id as the
# tiebreaker, matching the prompt-version table's determinism rule.
_SCOPE_BUCKET_ORDER: list[tuple[str, str]] = [
    ("global", "global"),
    ("global_legacy", "global (legacy)"),
    ("task", "task"),
    ("project_missing_id", "project (no project id)"),
    ("invalid", "invalid"),
]


def _scope_distribution_lines(by_scope: dict[str, int]) -> list[str]:
    """Format the scope-distribution rows for the stats screen.

    Fixed buckets keep a stable position so repeated /memory stats
    reads scan the same way; project buckets (keys prefixed
    `project:`) slot in after the two global rows. Only non-zero
    buckets appear because `get_stats` never emits zero counts.
    """
    project_items = sorted(
        ((key.removeprefix("project:"), count) for key, count in by_scope.items() if key.startswith("project:")),
        key=lambda item: (-item[1], item[0]),
    )
    rows: list[tuple[str, int]] = []
    for key, label in _SCOPE_BUCKET_ORDER[:2]:
        if key in by_scope:
            rows.append((label, by_scope[key]))
    rows.extend((f"project '{pid}'", count) for pid, count in project_items)
    for key, label in _SCOPE_BUCKET_ORDER[2:]:
        if key in by_scope:
            rows.append((label, by_scope[key]))

    # Any bucket key this renderer does not recognize renders raw at
    # the end rather than silently vanishing: a distribution that
    # hides rows misrepresents the corpus, and the raw key is enough
    # signal that the renderer and the aggregator have drifted.
    known = {key for key, _ in _SCOPE_BUCKET_ORDER}
    rows.extend(
        (key, count) for key, count in sorted(by_scope.items()) if key not in known and not key.startswith("project:")
    )

    width = max(len(label) for label, _ in rows)
    return [f"  {label.ljust(width)}  {count:>3}" for label, count in rows]


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
    # memories yet." Wording mirrors the dashboard's two-bucket
    # surface (facts + episodes); the migration count folds into
    # "facts" so the empty-state phrasing has no third item.
    if stats.extracted_count == 0 and stats.episode_count == 0 and stats.migration_count == 0:
        text = "Memory stats\n\nNo facts or episodes yet."
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("back", callback_data=_encode_callback("dash"))]])
        return text, kb

    # Headline block: per-bucket totals, one line per non-zero bucket.
    # Right-padding holds the "Total <bucket>:" labels at consistent
    # width across the header lines so the count column lines up.
    # The fact bucket sums extracted and migration so a migration-only
    # operator sees a single "Total facts:" rather than the prior
    # split rows; the underlying source field stays unchanged.
    total_facts = stats.extracted_count + stats.migration_count
    lines = ["Memory stats", ""]
    if total_facts:
        lines.append(f"Total facts:      {total_facts}")
    if stats.episode_count:
        lines.append(f"Total episodes:   {stats.episode_count}")

    # Scope distribution over all user-visible rows. Sits right after
    # the headline because (unlike the extracted-only sections below)
    # it spans every user-visible source. The global_legacy row is
    # the operator's running measure of reclassification debt - rows
    # that retrieval treats as global only because nothing has
    # classified them yet.
    if stats.by_scope:
        lines.append("")
        lines.append("Scope:")
        lines.extend(_scope_distribution_lines(stats.by_scope))

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
      ffc               - forget single fact confirm
      ffd               - forget single fact: do delete
      fview             - return to fact view (same id); cancel target
                          of the forget confirm, back target of the
                          scope screen, back target of the source view
      scp               - scope screen for the cached fact; cancel
                          target of the scope confirm
      sct <idx>         - scope target tapped (resolved against
                          cache.scope_targets); renders confirm
      scd               - scope change confirmed: apply
      src               - source view for the cached fact; calls
                          fetch_transcript_context and renders the
                          originating exchange or a per-reason
                          failure body
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
    try:
        if verb == "dash":
            await _send_dashboard(update, context, chat_id, edit=True)
            await query.answer()
            return
        if verb == "stats":
            await _send_stats(update, context, chat_id, edit=True)
            await query.answer()
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
            # Facts list browser (extracted + migration). Args:
            #   args[0] - page integer (default 0 if missing/invalid)
            #   args[1] - sort mode "quality" or "recent" (default
            #             "quality" when missing; legacy 2-segment
            #             callbacks from chat history without a sort
            #             arg get the default behavior)
            # The unknown-verb dismiss at the bottom of this handler
            # covers any future retirement of the verb, which would
            # silently no-op stale callbacks in chat history (same
            # compatibility path the retired `tag` verb relies on).
            try:
                page = int(args[0]) if args else 0
            except ValueError:
                page = 0
            sort: Literal["quality", "recent"] = "quality"
            if len(args) >= 2 and args[1] == "recent":
                sort = "recent"
            await _send_facts_list(update, context, chat_id, page, edit=True, sort=sort)
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
        if verb == "src":
            # Source view for the fact in cache. The fact id was
            # cached when the user opened the detail view; the back
            # button on the source view uses the existing fview verb
            # to re-render the same fact view from that cache.
            cache = _get_cache(chat_id)
            if cache is None or not cache.memory_ids:
                await _send_dashboard(update, context, chat_id, edit=True)
                await query.answer(_MSG_SESSION_EXPIRED)
                return
            await _send_source_view(update, context, chat_id, cache.memory_ids[0])
            await query.answer()
            return
        if verb == "scp":
            # Scope screen for the fact in cache. Reached from the
            # fact view's scope button and from the confirm screen's
            # cancel button; both leave the fact id at memory_ids[0].
            cache = _get_cache(chat_id)
            if cache is None or not cache.memory_ids:
                await _send_dashboard(update, context, chat_id, edit=True)
                await query.answer(_MSG_SESSION_EXPIRED)
                return
            await _send_scope_screen(update, context, chat_id, cache.memory_ids[0])
            await query.answer()
            return
        if verb == "sct":
            # Scope target tapped. The integer arg indexes
            # cache.scope_targets (set by _send_scope_screen); any
            # decode or range failure routes through the standard
            # session-expired fallback like the fact verb does.
            cache = _get_cache(chat_id)
            if cache is None or not cache.memory_ids or not args:
                await _send_dashboard(update, context, chat_id, edit=True)
                await query.answer(_MSG_SESSION_EXPIRED)
                return
            try:
                idx = int(args[0])
            except ValueError:
                await _send_dashboard(update, context, chat_id, edit=True)
                await query.answer(_MSG_SESSION_EXPIRED)
                return
            if idx < 0 or idx >= len(cache.scope_targets):
                await _send_dashboard(update, context, chat_id, edit=True)
                await query.answer(_MSG_SESSION_EXPIRED)
                return
            await _send_scope_confirm(update, context, chat_id, cache.memory_ids[0], cache.scope_targets[idx])
            await query.answer()
            return
        if verb == "scd":
            # Scope change confirmed. The selected target rode the
            # cache from the confirm screen, so the callback carries
            # no arguments to validate.
            cache = _get_cache(chat_id)
            if cache is None or not cache.memory_ids or cache.scope_target is None:
                await _send_dashboard(update, context, chat_id, edit=True)
                await query.answer(_MSG_SESSION_EXPIRED)
                return
            answer = await _apply_scope_change(update, context, chat_id, cache.memory_ids[0], cache.scope_target)
            await query.answer(answer)
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
    # Backends without a OneShotReasoner never run extraction, so for
    # those users this dashboard only ever shrinks toward (or stays
    # at) empty. Without an explicit line the user's only signal is
    # silence: nothing accumulates and nothing says why. Appended
    # here rather than in _build_dashboard so the builder stays a
    # pure view over MemoryStats and the note also rides the
    # empty-state text, where the question "why is this empty?" is
    # sharpest. The per-user fall-through matches the extraction
    # gate's backend resolution in bot.py.
    config: Config = context.bot_data["config"]
    user_config = config.get_user_config(chat_id)
    effective_backend = user_config.backend if user_config and user_config.backend else config.default_backend
    if effective_backend not in ONESHOT_REASONER_BACKENDS:
        text += (
            f"\n\nNote: memory extraction is not available on the {effective_backend} "
            "backend; this memory is retrieval-only (no facts are written from "
            "conversations)."
        )
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
    sort: Literal["quality", "recent"] = "quality",
) -> None:
    """Render and send/edit the facts list at `page`.

    Fetches via `memory.get_all_facts` (the fact-bucket enumeration
    that folds extracted and migration), re-sorts when sort is
    "quality" (the default), builds via `_build_facts_list_view`,
    and sets the cache screen to `"facts"` so a fact opened from
    this list can route back to the same page via `_send_fact_view`'s
    return_to logic. Distinct from the `"fact"` (singular) sentinel,
    which identifies the fact-detail view.

    `sort` modes:
        "quality": re-sort the fetched list by browse score
            (cosine pinned at 1.0; speaker_weight * confidence)
            descending, with `updated_at` desc as the tiebreaker.
            Default mode; surfaces high-quality facts first.
        "recent": pass through unchanged. `get_all_facts` already
            returns the list in updated-at-descending order, so no
            second sort is needed.

    The sort decision lives here rather than in `_build_facts_list_view`
    so the renderer is purely a view layer; non-browse callers
    (tests, future surfaces) can pass a pre-sorted list to the
    builder without inheriting the browse-time default.
    """
    try:
        facts = memory.get_all_facts(user_id=str(chat_id))
    except Exception as exc:
        log.exception("get_all_facts failed: %s", exc)
        await _send_or_edit(update, _MSG_QUERY_FAILED, None, edit=edit)
        return
    if sort == "quality":
        # Quality score (descending) primary, updated_at (descending)
        # secondary. The compound key is a single sort call: Python's
        # sort is stable, but tuple-key descending guarantees the
        # tiebreaker direction matches the primary direction without
        # relying on stability semantics.
        facts = sorted(
            facts,
            key=lambda f: (_browse_score(f), f.updated_at or f.created_at or ""),
            reverse=True,
        )
    text, kb, memory_ids, clamped_page, _total = _build_facts_list_view(facts, page, sort=sort)
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
    registry, active = await _scope_inputs(context, chat_id)
    scope_view = _build_scope_view(fact, registry, active)
    provenance = read_transcript_provenance(fact.metadata)
    text, kb = _build_fact_view(fact, return_to, scope_view, provenance)
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


# Per-reason source-view failure messages. Strings live as module
# constants so test assertions key on them and so the lookup helper's
# stable reason vocabulary stays one-to-one with the operator-facing
# wording.
_SOURCE_FAILURE_MESSAGES: dict[str, str] = {
    "file_missing": "The history file for that date is no longer available.",
    "unreadable": "The history file could not be read.",
    "ts_not_found": "The original message was not found in the history file.",
    "hash_mismatch": "Content drift detected: the original message no longer matches its fingerprint.",
    "chat_mismatch": "This memory's source pointer does not match this chat; refusing to dereference.",
    "legacy": "This memory predates source tracking.",
}


def _build_source_view(
    fact: MemoryResult,
    lookup: TranscriptLookup,
) -> tuple[str, InlineKeyboardMarkup]:
    """Render the source view body and its back-to-fact keyboard.

    On `reason="ok"` the body shows the surrounding turns in
    chronological order with per-turn headers; on any other reason
    the body renders the message from `_SOURCE_FAILURE_MESSAGES`.
    The body-level truncation marker is appended once after assembly
    so even a pathological exchange stays under the Telegram limit.

    The keyboard is a single back button; the verb `fview` reuses
    the existing fact-view re-render flow, which reads the cached
    memory_id and presents the detail view unchanged.
    """
    if lookup.reason == "ok" and lookup.context is not None:
        body = _render_source_view_body(fact, lookup.context)
    else:
        message = _SOURCE_FAILURE_MESSAGES.get(lookup.reason, "Source unavailable.")
        body = f"Source\n\n{message}"
    body = _truncate_to_message_limit(body)
    kb = InlineKeyboardMarkup([[InlineKeyboardButton("back", callback_data=_encode_callback("fview"))]])
    return body, kb


def _render_source_view_body(fact: MemoryResult, context: TranscriptContext) -> str:
    """Compose the chronological exchange body for `reason="ok"`."""
    lines: list[str] = ["Source", "", f'For memory "{_truncate(fact.text, 60)}":', ""]
    if context.truncated:
        lines.append("(Episode window truncated.)")
        lines.append("")
    for turn in context.before:
        lines.extend(_render_source_view_turn(turn))
    lines.extend(_render_source_view_turn(context.target_user))
    if context.target_assistant is not None:
        lines.extend(_render_source_view_turn(context.target_assistant))
    for turn in context.after:
        lines.extend(_render_source_view_turn(turn))
    return "\n".join(lines).rstrip()


def _render_source_view_turn(turn: TranscriptTurn) -> list[str]:
    """One turn's lines: a `[ts UTC] direction:` header then capped text."""
    header_label = "user" if turn.direction == "user" else "assistant"
    return [
        f"[{_format_source_ts(turn.ts)}] {header_label}:",
        _truncate_source_turn(turn.text),
        "",
    ]


async def _send_source_view(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    memory_id: str,
) -> None:
    """Render the source view for a fact, edit-in-place.

    Re-fetches the row before resolving provenance so a row deleted
    between the fact view and the tap surfaces as the standard
    "no longer exists" body rather than a stale rendering.
    """
    fact = memory.get_by_id(user_id=str(chat_id), memory_id=memory_id)
    if fact is None:
        await _send_or_edit(update, "This memory no longer exists.", None, edit=True)
        return
    provenance = read_transcript_provenance(fact.metadata)
    # `expected_chat_id` enforces the ownership gate: Mem0's
    # `get_by_id` partition verifies the row belongs to this chat,
    # but the `source_chat_id` field on that row is just metadata
    # the row carries; a malformed/forged value pointing at another
    # chat would otherwise let this UI dereference an unrelated
    # chat's JSONL. The helper returns chat_mismatch before any
    # filesystem read on disagreement.
    lookup = fetch_transcript_context(provenance, memory_id=memory_id, expected_chat_id=chat_id)
    text, kb = _build_source_view(fact, lookup)
    cache = _get_cache(chat_id)
    return_to = cache.return_to if cache is not None else None
    _set_cache(
        chat_id,
        _ScreenCache(
            screen="source",
            memory_ids=[memory_id],
            return_to=return_to,
        ),
    )
    await _send_or_edit(update, text, kb, edit=True)


async def _send_scope_screen(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    memory_id: str,
) -> None:
    """Render the scope screen for a fact.

    Re-fetches the row and re-derives the targets on every render
    (including the cancel path back from the confirm screen), so the
    screen always reflects the registry and the row as they are now,
    not as they were when the fact view was first opened.
    """
    fact = memory.get_by_id(user_id=str(chat_id), memory_id=memory_id)
    if fact is None:
        await _send_or_edit(update, "This memory no longer exists.", None, edit=True)
        return
    registry, active = await _scope_inputs(context, chat_id)
    scope_view = _build_scope_view(fact, registry, active)
    targets = _scope_change_targets(scope_view.resolved, registry)
    text, kb = _build_scope_screen(fact, scope_view, targets)
    # Preserve return_to so the eventual back-nav from the fact view
    # still lands on the originating list screen after a round trip
    # through the scope flow.
    cache = _get_cache(chat_id)
    return_to = cache.return_to if cache is not None else None
    _set_cache(
        chat_id,
        _ScreenCache(
            screen="scope",
            memory_ids=[memory_id],
            return_to=return_to,
            scope_targets=targets,
        ),
    )
    await _send_or_edit(update, text, kb, edit=True)


async def _send_scope_confirm(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    memory_id: str,
    target: tuple[str, str | None],
) -> None:
    """Render the scope-change confirmation for a tapped target.

    The selected target moves into `cache.scope_target` so the apply
    verb (`scd`) needs no callback arguments. `scope_targets` is NOT
    carried forward: the confirm screen has no target buttons, so an
    empty list is the honest cache state, and a stale `sct` tap from
    an older message in chat history falls into the standard
    session-expired path instead of resolving against a list the
    user is no longer looking at.
    """
    fact = memory.get_by_id(user_id=str(chat_id), memory_id=memory_id)
    if fact is None:
        await _send_or_edit(update, "This memory no longer exists.", None, edit=True)
        return
    text, kb = _build_scope_confirm(fact, target)
    cache = _get_cache(chat_id)
    return_to = cache.return_to if cache is not None else None
    _set_cache(
        chat_id,
        _ScreenCache(
            screen="scope_confirm",
            memory_ids=[memory_id],
            return_to=return_to,
            scope_target=target,
        ),
    )
    await _send_or_edit(update, text, kb, edit=True)


async def _apply_scope_change(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    memory_id: str,
    target: tuple[str, str | None],
) -> str:
    """Apply a confirmed scope change and re-render the fact view.

    Returns the toast string for the callback answer (the caller
    answers after this helper's sends complete, preserving the
    answer-after-send pattern).

    Write shape: the row's existing metadata is copied and the five
    scope keys are overlaid from `build_scope_metadata`, because
    `memory.update_metadata` REPLACES the metadata dict wholesale -
    passing only the scope keys would destroy tags, confidence,
    episode fields, and every other stored field. Operator moves
    carry `scope_source="operator"`, full confidence, and no
    `workspace_root`: that field records write-time workspace
    provenance, which a retarget from chat does not have.
    """
    fact = memory.get_by_id(user_id=str(chat_id), memory_id=memory_id)
    if fact is None:
        await _send_or_edit(update, "This memory no longer exists.", None, edit=True)
        return "Not found."
    old = memory.resolve_memory_scope(fact.metadata)
    kind, pid = target
    scope_md = memory.build_scope_metadata(
        scope=memory.SCOPE_GLOBAL if kind == "global" else memory.SCOPE_PROJECT,
        project_id=pid,
        scope_confidence=1.0,
        scope_source=memory.SCOPE_SOURCE_OPERATOR,
    )
    merged = dict(fact.metadata or {})
    merged.update(scope_md)
    ok = memory.update_metadata(
        user_id=str(chat_id),
        memory_id=memory_id,
        data=fact.text,
        metadata=merged,
    )
    if ok:
        # Structured audit line, one per applied change. The before
        # values come from the resolver (not raw metadata) so legacy
        # rows log the same global-interpretation retrieval acted on;
        # scope_source after an operator move is always "operator",
        # so only the before value is recorded.
        log.info(
            "%s %s",
            memory.SCOPE_CHANGE_EVENT,
            json.dumps(
                {
                    "memory_id": memory_id,
                    "chat_id": chat_id,
                    "from_scope": old.scope,
                    "from_project_id": old.project_id,
                    "from_scope_source": old.scope_source,
                    "to_scope": scope_md["scope"],
                    "to_project_id": scope_md["project_id"],
                },
                separators=(",", ":"),
            ),
        )
    # Re-render the fact view either way: on success it shows the new
    # scope block; on failure it re-fetches and shows whatever state
    # the row is actually in (including "no longer exists" if the row
    # vanished mid-flow).
    await _send_fact_view(update, context, chat_id, memory_id)
    return "Scope updated." if ok else "Update failed."


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
