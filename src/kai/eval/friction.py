"""
Layer 3 longitudinal friction analysis.

Reads an operator's chat history from DATA_DIR/history/<user_id>/, runs
four pre-committed friction-detection signal families over the ordered
records, buckets events by milestone (PR #333 and PR #361 merge instants),
normalizes counts by user-turn volume, and emits a schema-versioned JSON
report plus an optional redacted markdown summary.

The harness is observational: it never writes to the live history or
memory stores. It complements Layer 1 (retrieval-only A/B; src/kai/eval/
retrieval.py) and Layer 2 (behavioral A/B; src/kai/eval/behavioral.py)
with a real-traffic signal that survives the contrived-probe critique.

Designed to run offline against a SNAPSHOT of DATA_DIR. The typical
invocation copies history/ and memory/ to /tmp/kai-eval-snapshot,
points KAI_DATA_DIR there, and invokes `python -m kai.eval.friction`.

This file is intentionally self-contained: no production code path
imports from it. Outbound dependencies are kai.config (for DATA_DIR
resolution) and kai.memory (for stage 3b known-fact overlap; optional -
the harness degrades gracefully when memory is unreachable).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

log = logging.getLogger(__name__)


# ── Schema version and milestone bucket anchors ──────────────────────

# Bumped when the JSON output shape changes incompatibly. Convention
# matches `_OUTPUT_SCHEMA_VERSION = 1` at src/kai/eval/behavioral.py:115.
# Downstream comparison scripts pin against this to detect schema drift
# without parsing the whole report shape.
_OUTPUT_SCHEMA_VERSION = 1

# Two milestone anchors carved into module-level constants because they
# define the entire bucket scheme. They are squash-merge commits on
# `main` and will not move; treat as authoritative.
#
# Original git log timestamps came back in Eastern (-04:00):
#   commit 5796286 -> 2026-04-17T13:54:27-04:00 -> 2026-04-17T17:54:27Z
#   commit e508662 -> 2026-04-22T08:43:55-04:00 -> 2026-04-22T12:43:55Z
# UTC normalization is done here so every downstream comparison is
# timezone-aware against a single canonical instant. Confirm with
# `git log --format="%H %aI" -1 <hash>` before changing either anchor.
_BUCKET_ANCHORS = (
    datetime(2026, 4, 17, 17, 54, 27, tzinfo=UTC),
    datetime(2026, 4, 22, 12, 43, 55, tzinfo=UTC),
)


# ── Volume floor and classification thresholds ───────────────────────

# Buckets with fewer user_turns than this floor cannot anchor a confident
# cross-bucket comparison. The classifier emits
# `matched_pattern="inconclusive"` when ANY of A/B/C falls below the
# floor. 30 picked on the same back-of-envelope basis as Layer 1's
# 26-probe minimum: rate estimates with single-digit denominators have
# wide enough confidence intervals that label flips are noise-driven.
_USER_TURN_FLOOR = 30

# "material drop": new-bucket rate is at most this fraction of the prior.
# The boundary at exactly 0.85x belongs to material drop (the flat band's
# lower edge is open). Widened from v1's 0.66x to 0.85x in v3 so the
# bands tile [0, +inf) without an unclassifiable gap.
_MATERIAL_DROP_RATIO = 0.85

# "up" cutoff: new-bucket rate exceeds this fraction of the prior. The
# boundary at exactly 1.15x belongs to flat. Used by both the `up` shape
# in the 9-cell grid and by the `regression` row's B->C condition.
_UP_RATIO = 1.15

# Pre-committed prediction the harness was built to test. Hardcoded at
# v1 of the harness; if the prediction itself becomes a tunable
# parameter, a future revision would expose this as a CLI flag, but
# tunability via CLI invites post-hoc threshold-fishing so the default
# stance is "compile-time constant".
_PREDICTED_PATTERN = "prediction-holds"


# ── Signal-family pattern sets ───────────────────────────────────────

# Signal 1: operator-side frustration markers. Substring match on the
# lowercased user message. Pre-committed: changes to this set require a
# changelog entry AND a test-suite update so a phrase added after
# seeing data does not constitute post-hoc pattern-hunting.
_FRUSTRATION_PHRASES = (
    "i already told you",
    "i said earlier",
    "i just said",
    "as i mentioned",
    "as i said",
    "remember i",
    "remember that i",
    "you forgot",
    "you already",
    "why are you asking",
    "why do you keep asking",
    "i told you",
)

# Signal 2: stopwords excluded from the content-word set. Filters out
# common conversational filler so the "3 distinct content-words"
# threshold reflects substantive overlap rather than incidental shared
# articles and modal verbs. Frozen so a typo in a test cannot mutate it.
_STOPWORDS = frozenset(
    {
        "this",
        "that",
        "with",
        "from",
        "have",
        "will",
        "what",
        "when",
        "where",
        "which",
        "would",
        "could",
        "should",
        "about",
        "there",
        "their",
        "your",
        "they",
        "them",
        "been",
        "were",
        "than",
        "then",
        "into",
        "some",
        "such",
        "more",
        "most",
        "also",
        "does",
    }
)

# Signal 2 + stage 3b: the "3 distinct content-words" floor is the
# false-positive cliff in informal observation - generic clarifications
# rarely share three or more substantive tokens with prior user content.
_CONTENT_WORD_FLOOR = 3

# Signal 2: lookback window for prior records. Capping at 20 preceding
# entries (user + assistant interleaved) keeps the per-record cost
# bounded; long histories would otherwise scale O(N^2) on the
# question-answer pair count.
_KAI_ASKS_BACK_WINDOW = 20

# Signal 3: stage 3a regex set. All compiled with re.IGNORECASE because
# Telegram traffic routinely capitalizes the first letter of a message
# via mobile autocorrect ("DON'T", "I Prefer"); without the flag those
# would silently miss. Word-boundary anchors keep matches honest, e.g.
# "construction" does not match "instead of" via "instead".
_CORRECTION_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bdon'?t\b",
        r"\bstop (doing|using|asking)\b",
        r"\bnot that\b",
        r"\bplease (use|don'?t)\b",
        r"\balways (use|say|start)\b",
        r"\bnever (use|say|do)\b",
        r"\binstead of\b",
        r"\bi prefer\b",
    )
)

# Signal 4: fact-shape prefixes. Compiled into a single alternation so
# the per-record cost stays linear in record length rather than scaling
# with the prefix-set size. Word-boundary anchors avoid matching the
# prefix substring inside an unrelated word.
_FACT_SHAPE_RE = re.compile(
    r"\b(?:"
    r"i am|i'm|my|we use|we're using|"
    r"i prefer|i always|i use|i don't"
    r")\b",
    re.IGNORECASE,
)

# Signal 4: 5-character shingles + Jaccard similarity threshold. Tuned
# high enough that paraphrases of the same fact register as a repeat
# without flagging unrelated user messages that happen to share a few
# word fragments. Strictly > (not >=) so a synthetic test fixture at
# exactly 0.4 verifies the open boundary.
_REPEAT_SHINGLE_K = 5
_REPEAT_JACCARD_FLOOR = 0.4

# Signal 4: cap the per-record lookback to 200 prior user-turns. Same
# rationale as Signal 2's window: bound worst-case cost on long
# histories without losing realistic-repeat detection range.
_REPEAT_LOOKBACK = 200


# ── Surface-text redaction patterns (for --include-snippets) ─────────

# Order matters when patterns overlap. URLs first because they may
# contain `@` (user:pass URL form); email next because it contains `@`;
# @handles next because they are pure `@…`; phone last because the loose
# digit-run pattern would otherwise gobble URL/email substrings. Per-
# event excerpt cap is enforced separately at _SNIPPET_MAX_CHARS.
_REDACT_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_REDACT_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_REDACT_HANDLE_RE = re.compile(r"@[A-Za-z0-9_]{3,}")
# Phone: 7+ groups of "digit + optional separator", ending in a digit.
# Wide enough to cover international and US formats; narrow enough to
# avoid matching short numeric IDs (e.g. issue #333) which would create
# false redactions in a chat that frequently references issue numbers.
_REDACT_PHONE_RE = re.compile(r"(?:\+?\d[\s.-]?){7,}\d")
_REDACT_REPLACEMENTS = (
    (_REDACT_URL_RE, "<redacted-url>"),
    (_REDACT_EMAIL_RE, "<redacted-email>"),
    (_REDACT_HANDLE_RE, "<redacted-handle>"),
    (_REDACT_PHONE_RE, "<redacted-phone>"),
)

# Cap on the surface_text the JSON report may carry per event when
# --include-snippets is set. 200 chars is plenty to identify the matching
# message without turning the report into a chat-log copy.
_SNIPPET_MAX_CHARS = 200


# ── Tokenization helper ──────────────────────────────────────────────

# Punctuation stripping uses a simple class-based regex (not the broader
# unicode category) so it is fast and predictable. Apostrophe is kept
# inside the character class so contractions like "don't" survive as a
# single token; downstream stopword filtering does not include "don't"
# so the token simply does not contribute to the content-word set.
_PUNCT_RE = re.compile(r"[^\w\s']")


def _content_words(text: str) -> set[str]:
    """Return distinct content-words for overlap comparisons.

    A "content-word" is a token of length >= 4 after lowercasing and
    stripping punctuation (other than apostrophe), excluding the small
    `_STOPWORDS` set. Used by Signal 2 (kai_asks_back) and Signal 3
    stage 3b (known-fact overlap), both of which need a substantive-
    token overlap floor that is not perturbed by common filler.

    Returns a `set` so callers can intersect via `&`; the floor check
    is `len(a & b) >= _CONTENT_WORD_FLOOR`.
    """
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    return {t for t in cleaned.split() if len(t) >= 4 and t not in _STOPWORDS}


# ── Memory-availability tri-state ────────────────────────────────────


class MemoryAvailability(Enum):
    """Tri-state result of attempting to initialize the memory store.

    DIVERGES from `_initialize_memory` at src/kai/eval/behavioral.py:1901
    in that the friction harness MUST NOT exit on a non-ENABLED result.
    A friction analysis without stage 3b is still a valid (degraded)
    report; the run completes with a `memory-unreachable-frustration-
    only` caveat instead of failing. Layer 2 cannot run without retrieval
    so it exits; Layer 3 can.

    Members:
      ENABLED  - init succeeded and is_enabled() returned True.
      DISABLED - init succeeded but is_enabled() returned False (the
                 store is configured off via MEMORY_ENABLED=false or
                 the embedding-dim validation rejected the model).
      ERROR    - init raised; the captured exception text is reported
                 in `memory_store_error_message` for debuggability.
    """

    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass(frozen=True)
class MemoryInitResult:
    """Return shape from `_initialize_memory_local`.

    `error_message` is populated only when `availability == ERROR`.
    Carrying it on the result rather than logging in the helper lets the
    renderer surface the message in the report itself, which is what an
    operator running the harness against a damaged Qdrant directory
    needs to see.
    """

    availability: MemoryAvailability
    error_message: str | None


# ── Data shapes ──────────────────────────────────────────────────────

# Sentinel used when a record's timestamp is missing or unparseable.
# Strictly before _BUCKET_ANCHORS[0] so `_classify_bucket` routes the
# record to bucket A. Module-level so callers (and tests) can compare
# against it directly rather than reconstructing the literal.
_BAD_TS_SENTINEL = datetime(1, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class HistoryRecord:
    """One line of a JSONL history file, parsed.

    Mirrors the fields written by `log_message` at src/kai/history.py:67-74.
    `ts_utc` is the parsed datetime already normalized to UTC; `ts_raw`
    keeps the original ISO string so the report can cite it verbatim
    without re-formatting.

    For lines whose `ts` is missing, empty, or unparseable, `ts_utc` is
    set to `_BAD_TS_SENTINEL` (year 1) so bucket assignment routes the
    record to bucket A (oldest, pre-semantic-memory era). The
    conservative rule is "bad timestamps land in the oldest bucket so
    the newest bucket is never inflated by data of unknown vintage."
    """

    ts_utc: datetime
    ts_raw: str
    direction: str
    chat_id: int
    text: str
    media: dict[str, Any] | None
    source_path: str
    line_no: int


@dataclass(frozen=True)
class Bucket:
    """A milestone-defined bucket with start + end boundaries.

    `start` is inclusive, `end` is exclusive. `end=None` encodes the
    open-ended current bucket (C). `start=None` encodes the open-ended
    pre-history bucket (A). Labels match the issue body: 'A', 'B', 'C'.
    """

    label: str
    start: datetime | None
    end: datetime | None


_BUCKETS = (
    Bucket(label="A", start=None, end=_BUCKET_ANCHORS[0]),
    Bucket(label="B", start=_BUCKET_ANCHORS[0], end=_BUCKET_ANCHORS[1]),
    Bucket(label="C", start=_BUCKET_ANCHORS[1], end=None),
)


@dataclass(frozen=True)
class FrictionEvent:
    """One detected friction event with enough context to audit later.

    `surface_text` is always populated at detection time so downstream
    tests can verify match correctness. The render layer strips it
    before JSON emission unless `--include-snippets` is passed.

    `metadata` carries signal-family-specific annotations. The contract
    by family is:
      - "frustration":           {} (always)
      - "kai_asks_back":         {} (always)
      - "preference_correction": {"known_fact_overlap": bool,
                                  "overlapping_fact_ids": tuple[str, ...]}
                                 where overlap=False -> overlapping_fact_ids=()
      - "repeated_fact":         {} (always)

    `metadata` MUST be wrapped in `MappingProxyType` at construction
    time so the frozen-dataclass invariant extends to the dict's
    contents. Without the wrapper, a plain dict on a frozen dataclass
    remains mutable through the field, breaking any downstream "this
    event is immutable" assumption.

    `context_message_ids` lists prior message IDs that grounded the
    detection. Empty for "frustration" and for "preference_correction"
    (which grounds via `metadata.overlapping_fact_ids` instead). Non-
    empty for "kai_asks_back" (the prior user messages whose content-
    words overlapped) and for every emitted "repeated_fact" event (the
    prior fact-shape statement). First-occurrence fact-shape records
    never reach emission per Signal 4's definition - the Jaccard
    requirement is against a PRIOR matching record - so a record with
    no prior match never produces an event.
    """

    timestamp_utc: datetime
    bucket_label: str
    signal_family: str
    message_id: str
    surface_text: str
    context_message_ids: tuple[str, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class BucketAggregate:
    """Per-bucket-per-signal count plus per-100-user-turns rate.

    `rate_per_100_user_turns` is rounded to two decimal places at
    construction time so JSON serialization emits stable strings; the
    classifier reads the same float for band assignment and re-rounds
    derived ratios to three decimals at emission time.
    """

    bucket_label: str
    signal_family: str
    count: int
    user_turns_in_bucket: int
    rate_per_100_user_turns: float


@dataclass(frozen=True)
class FrictionConfig:
    """CLI-driven configuration for one run.

    Mirrors the `BehavioralConfig` shape at src/kai/eval/behavioral.py:273.
    `run_started_at_utc` is captured exactly once in `main()` and
    threaded through every downstream call so cutoff math and the
    JSON's `generated_at_utc` derive from a single wall-clock read.
    Tests pin this by monkey-patching the module's `datetime` reference
    inside a pytest fixture (e.g. `monkeypatch.setattr("kai.eval.friction.datetime", FakeDatetime)`),
    NOT by passing a hidden CLI flag. Production users do not pass a
    reference time; introducing a flag would be a code path that runs
    only in tests.
    """

    user_id: str
    data_dir: Path
    output_path: Path
    sample_days: int | None
    include_snippets: bool
    markdown_summary_path: Path | None
    run_started_at_utc: datetime


@dataclass(frozen=True)
class TrendSummary:
    """Output of the trend classifier.

    Carries the band assignment in-memory for the markdown renderer (so
    it can label bands without re-running the classifier); the band
    field is NOT emitted at the top level of the JSON because the
    JSON's narrative already names the bands in prose.
    `per_signal_rate_breakdown` and `zero_prior_rate_caveats` feed the
    narrative and the caveats list respectively, neither surfacing as
    their own JSON fields.
    """

    predicted_pattern: str
    matched_pattern: str
    verdict_driving_sum: dict[str, float]
    ratios: dict[str, float | None]
    bands: dict[str, str | None]
    per_signal_rate_breakdown: dict[str, dict[str, float]]
    zero_prior_rate_caveats: tuple[tuple[str, str, float], ...]
    narrative: str


# ── History reading ─────────────────────────────────────────────────


def _validate_user_id(user_id: str) -> None:
    """Reject values that would escape the history root.

    Mirrors the path-traversal guard at src/kai/eval/behavioral.py:841.
    Treated as eval-only operator-run code so the guard is a defensive
    shape check (one segment, no parent refs, no slashes) rather than
    a security boundary; a determined operator running their own
    harness can still reach arbitrary paths via --data-dir. The guard
    keeps an honest mistake from silently widening the read scope.
    """
    if Path(user_id).name != user_id or user_id in ("", ".", ".."):
        raise ValueError(f"invalid user_id for history path: {user_id!r}")


def _parse_record_timestamp(ts_raw: str) -> tuple[datetime, str | None]:
    """Parse the ts field with explicit per-edge-case routing.

    Returns (parsed_datetime, parse_reason). When parse_reason is None
    the parse succeeded; otherwise it is a stable label suitable for
    inclusion in the friction.timestamp_skip debug log line. Three edge
    cases:
      1. Missing or empty `ts` -> sentinel + "ts_missing_or_empty".
      2. Non-empty but unparseable -> sentinel + "ts_unparseable".
      3. Parsed but naive -> assume UTC (log_message has always written
         tz-aware ISO; only hand-edited or legacy rows would be naive).
    """
    if not ts_raw:
        return _BAD_TS_SENTINEL, "ts_missing_or_empty"
    try:
        parsed = datetime.fromisoformat(ts_raw)
    except ValueError:
        return _BAD_TS_SENTINEL, "ts_unparseable"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    return parsed, None


def read_history(data_dir: Path, user_id: str) -> Iterator[HistoryRecord]:
    """Yield every parsed HistoryRecord under data_dir/history/<user_id>/.

    Streams file-by-file, line-by-line. The whole list at current
    volumes (dozens of MB at worst) is small enough to materialize, but
    the iterator-shaped contract keeps the harness honest against a
    future 10x growth in history size.

    JSONL parse mirrors the pattern at src/kai/history.py:136-153:
    read_text + splitlines + json.loads inside a try/except. Malformed
    lines do NOT abort the run; they log at debug level and skip,
    matching the bot's own history reader.

    Three timestamp edge cases each route to bucket A via the
    `_BAD_TS_SENTINEL` value and a stable `friction.timestamp_skip`
    debug log line so an operator can grep for them. The sentinel is
    strictly older than _BUCKET_ANCHORS[0] so bucket assignment lands
    the record in A without special-casing.
    """
    _validate_user_id(user_id)
    history_dir = data_dir / "history" / user_id
    if not history_dir.is_dir():
        return
    for path in sorted(history_dir.glob("*.jsonl")):
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            log.exception("friction.read_history_file_unreadable path=%s", path)
            continue
        for line_no, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log.debug(
                    "friction.timestamp_skip path=%s line=%d reason=json_decode_error",
                    path,
                    line_no,
                )
                continue
            if not isinstance(row, dict):
                # Defensive: a top-level non-object JSON line (e.g. an
                # array) shouldn't appear, but skipping it keeps the
                # parser from raising AttributeError on .get below.
                log.debug(
                    "friction.timestamp_skip path=%s line=%d reason=row_not_object",
                    path,
                    line_no,
                )
                continue
            ts_raw = row.get("ts") or ""
            ts_utc, parse_reason = _parse_record_timestamp(ts_raw)
            if parse_reason:
                log.debug(
                    "friction.timestamp_skip path=%s line=%d reason=%s ts=%s",
                    path,
                    line_no,
                    parse_reason,
                    ts_raw[:80] if isinstance(ts_raw, str) else "",
                )
            # Production JSONL always has integer chat_id, but a hand-
            # edited or corrupted row could carry a non-numeric value.
            # Skip the line rather than aborting the run; the docstring
            # promises malformed rows do not crash the harness.
            try:
                chat_id_value = int(row.get("chat_id") or 0)
            except (TypeError, ValueError):
                log.debug(
                    "friction.timestamp_skip path=%s line=%d reason=chat_id_not_integer",
                    path,
                    line_no,
                )
                continue
            yield HistoryRecord(
                ts_utc=ts_utc,
                ts_raw=ts_raw if isinstance(ts_raw, str) else "",
                direction=str(row.get("dir") or ""),
                chat_id=chat_id_value,
                text=str(row.get("text") or ""),
                media=row.get("media") if isinstance(row.get("media"), dict) else None,
                source_path=str(path),
                line_no=line_no,
            )


# ── Bucket assignment ───────────────────────────────────────────────


def _classify_bucket(ts_utc: datetime) -> str:
    """Map a UTC timestamp to its milestone bucket label.

    Inclusive-start / exclusive-end contract:
      - Bucket A: ts < _BUCKET_ANCHORS[0]
      - Bucket B: _BUCKET_ANCHORS[0] <= ts < _BUCKET_ANCHORS[1]
      - Bucket C: ts >= _BUCKET_ANCHORS[1]

    A record with the bad-timestamp sentinel sorts before A's end, so
    routing for malformed records is correct without a separate branch.
    """
    if ts_utc < _BUCKET_ANCHORS[0]:
        return "A"
    if ts_utc < _BUCKET_ANCHORS[1]:
        return "B"
    return "C"


def _message_id(record: HistoryRecord, data_dir: Path) -> str:
    """Stable per-record identifier suitable for inclusion in the JSON.

    Format: "<user_id>/<file>.jsonl:<line_no>". The relative path keeps
    the message_id short (no /tmp prefix) and portable across
    snapshot-vs-production data dirs. Falls back to the
    bare filename when the record's source path falls outside the
    history root (e.g. tests that construct records by hand without
    placing them under data_dir/history/).
    """
    try:
        rel = Path(record.source_path).relative_to(data_dir / "history")
        return f"{rel}:{record.line_no}"
    except ValueError:
        return f"{Path(record.source_path).name}:{record.line_no}"


# ── Signal-family detectors ─────────────────────────────────────────

_SIGNAL_FAMILIES = (
    "frustration",
    "kai_asks_back",
    "preference_correction",
    "repeated_fact",
)


def _detect_frustration(records: list[HistoryRecord], data_dir: Path) -> list[FrictionEvent]:
    """Signal 1: operator-side frustration markers.

    Lowercases the user message and tests for substring presence of any
    `_FRUSTRATION_PHRASES` entry. One event per matching record; the
    inner `break` enforces this even when multiple phrases would match
    a single message (e.g. "i told you i already told you" emits one
    event, not two).
    """
    events: list[FrictionEvent] = []
    for record in records:
        if record.direction != "user":
            continue
        normalized = record.text.lower()
        for phrase in _FRUSTRATION_PHRASES:
            if phrase in normalized:
                events.append(
                    FrictionEvent(
                        timestamp_utc=record.ts_utc,
                        bucket_label=_classify_bucket(record.ts_utc),
                        signal_family="frustration",
                        message_id=_message_id(record, data_dir),
                        surface_text=record.text,
                        context_message_ids=(),
                        metadata=MappingProxyType({}),
                    )
                )
                break
    return events


def _ends_with_question_mark(text: str) -> bool:
    """Detect "?" ending after stripping trailing whitespace and closers.

    Closers stripped: `)`, `"`, `'`. Conservative set so common quoted
    or parenthesized question forms ("...?)", "...?\"") still register
    as questions while "Hello?" does not get accidentally counted as a
    statement by aggressive trimming.
    """
    stripped = text.rstrip()
    while stripped and stripped[-1] in ")\"'":
        stripped = stripped[:-1].rstrip()
    return stripped.endswith("?")


def _detect_kai_asks_back(records: list[HistoryRecord], data_dir: Path) -> list[FrictionEvent]:
    """Signal 2: Kai asking for information previously given.

    For each assistant record ending with `?`, compute its content-word
    set and look back over the immediately-preceding `_KAI_ASKS_BACK_WINDOW`
    records. If any prior USER record's content-words overlap by at
    least `_CONTENT_WORD_FLOOR`, emit one event with all matching
    priors cited in `context_message_ids`.

    The window is `records[max(0, i-20):i]` - 20 entries immediately
    preceding the assistant record at index i, NOT counting the
    assistant record itself. The first 20 records of any history have
    a smaller effective window; do not pad. This is the noisiest of
    the four signals; the report flags this in caveats.
    """
    events: list[FrictionEvent] = []
    for i, record in enumerate(records):
        if record.direction != "assistant":
            continue
        if not _ends_with_question_mark(record.text):
            continue
        own_words = _content_words(record.text)
        if len(own_words) < _CONTENT_WORD_FLOOR:
            # The assistant question itself has fewer than 3 substantive
            # tokens; it cannot meet the overlap floor against ANY prior,
            # so skip the inner loop entirely.
            continue
        window = records[max(0, i - _KAI_ASKS_BACK_WINDOW) : i]
        matching_ids: list[str] = []
        for prior in window:
            if prior.direction != "user":
                continue
            prior_words = _content_words(prior.text)
            if len(own_words & prior_words) >= _CONTENT_WORD_FLOOR:
                matching_ids.append(_message_id(prior, data_dir))
        if not matching_ids:
            continue
        events.append(
            FrictionEvent(
                timestamp_utc=record.ts_utc,
                bucket_label=_classify_bucket(record.ts_utc),
                signal_family="kai_asks_back",
                message_id=_message_id(record, data_dir),
                surface_text=record.text,
                context_message_ids=tuple(matching_ids),
                metadata=MappingProxyType({}),
            )
        )
    return events


def _stage_3a_matches(text: str) -> bool:
    """True iff `text` matches any `_CORRECTION_PATTERNS` regex.

    Searches (not matches), so the pattern can appear anywhere in the
    text. All patterns carry re.IGNORECASE.
    """
    return any(p.search(text) for p in _CORRECTION_PATTERNS)


def _parse_fact_created_at(value: str | None) -> datetime | None:
    """Parse a fact's `created_at` field for stage-3b time-comparison.

    Returns None for empty, missing, or unparseable strings - the
    conservative "we don't know when this fact was written, so we
    cannot claim it pre-existed the correction" path. `updated_at` is
    NOT used as a fallback: a fact bumped by re-extraction must not
    silently qualify as a fact that pre-existed the correction.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    else:
        parsed = parsed.astimezone(UTC)
    return parsed


def _detect_preference_correction(
    records: list[HistoryRecord],
    data_dir: Path,
    *,
    facts: list[Any],
) -> list[FrictionEvent]:
    """Signal 3: preference corrections (raw + known-fact overlap).

    Stage 3a: any user record matching `_CORRECTION_PATTERNS` is a raw
    correction. Stage 3b: query `get_by_tag(tag='preference')` ONCE
    upstream; for each raw match, intersect the correction's content-
    words with each fact's content-words and require:
      1. Fact `created_at` parses (per `_parse_fact_created_at`).
      2. Fact `created_at` is <= correction's `timestamp_utc`.
      3. Overlap >= `_CONTENT_WORD_FLOOR`.

    Pre-computes the (created_at, fact_id, content_words) tuple per
    fact ONCE, outside the per-record loop, so the per-correction cost
    is O(F) intersections rather than re-shingling F facts F times.
    """
    fact_signatures: list[tuple[datetime, str, set[str]]] = []
    for fact in facts:
        created = _parse_fact_created_at(getattr(fact, "created_at", ""))
        if created is None:
            log.debug(
                "friction.fact_skip id=%s reason=created_at_missing_or_unparseable",
                getattr(fact, "id", "?"),
            )
            continue
        fact_signatures.append((created, getattr(fact, "id", ""), _content_words(getattr(fact, "text", "") or "")))

    events: list[FrictionEvent] = []
    for record in records:
        if record.direction != "user":
            continue
        if not _stage_3a_matches(record.text):
            continue
        own_words = _content_words(record.text)
        overlapping_ids: list[str] = []
        for created, fact_id, fact_words in fact_signatures:
            if created > record.ts_utc:
                continue
            if len(own_words & fact_words) >= _CONTENT_WORD_FLOOR:
                overlapping_ids.append(fact_id)
        meta = MappingProxyType(
            {
                "known_fact_overlap": bool(overlapping_ids),
                "overlapping_fact_ids": tuple(overlapping_ids),
            }
        )
        events.append(
            FrictionEvent(
                timestamp_utc=record.ts_utc,
                bucket_label=_classify_bucket(record.ts_utc),
                signal_family="preference_correction",
                message_id=_message_id(record, data_dir),
                surface_text=record.text,
                context_message_ids=(),
                metadata=meta,
            )
        )
    return events


def _shingles(text: str, k: int = _REPEAT_SHINGLE_K) -> set[str]:
    """Return the set of k-character shingles after lowercase + whitespace strip.

    Empty text -> empty set. Text shorter than k -> single-element set
    containing the whole (cleaned) text, so very short fact-shape
    statements still produce a comparable signature.
    """
    cleaned = re.sub(r"\s+", "", text.lower())
    if not cleaned:
        return set()
    if len(cleaned) < k:
        return {cleaned}
    return {cleaned[i : i + k] for i in range(len(cleaned) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity of two shingle sets.

    Returns 0.0 when either set is empty; division by union size is
    guarded so an empty union (both empty) returns 0.0 rather than
    raising ZeroDivisionError.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _detect_repeated_fact(records: list[HistoryRecord], data_dir: Path) -> list[FrictionEvent]:
    """Signal 4: repeated-fact statements.

    For each user record matching `_FACT_SHAPE_RE`, walk back over the
    most recent `_REPEAT_LOOKBACK` user-turns (most recent first) and
    emit one event citing the FIRST prior fact-shape match whose
    Jaccard similarity exceeds `_REPEAT_JACCARD_FLOOR`. The break-on-
    first-match keeps emission to one event per repeat AND ensures the
    cited prior is the most recent (most relevant) one.

    First-occurrence fact-shape records never emit (no prior to compare
    against), so every emitted event has a non-empty `context_message_ids`.
    """
    user_indices = [i for i, r in enumerate(records) if r.direction == "user"]
    # Cache shingles only for fact-shape user records; non-fact-shape
    # priors don't qualify as a comparison target (the prior record
    # must itself be a fact-shape match), so there's no point building
    # their shingle sets. With the eviction below running on every
    # user record (not just fact-shape ones), peak memory is bounded
    # at most _REPEAT_LOOKBACK + 1 entries regardless of total
    # history length.
    shingle_cache: dict[int, set[str]] = {}
    events: list[FrictionEvent] = []
    for pos, idx in enumerate(user_indices):
        # Evict the entry that just fell out of the lookback window;
        # nothing later in the loop can read it. This runs BEFORE the
        # fact-shape check so the eviction fires once per user record
        # rather than once per fact-shape record - otherwise sparse
        # fact-shape histories would skip the eviction step on most
        # iterations and let the cache grow unbounded.
        evict_pos = pos - _REPEAT_LOOKBACK - 1
        if evict_pos >= 0:
            shingle_cache.pop(user_indices[evict_pos], None)
        record = records[idx]
        if not _FACT_SHAPE_RE.search(record.text):
            continue
        shingle_cache[idx] = _shingles(record.text)
        # Walk priors most-recent-first; break on the first qualifying
        # match so the cited prior is the closest in time.
        start = max(0, pos - _REPEAT_LOOKBACK)
        for prior_pos in range(pos - 1, start - 1, -1):
            prior_idx = user_indices[prior_pos]
            if prior_idx not in shingle_cache:
                continue
            score = _jaccard(shingle_cache[idx], shingle_cache[prior_idx])
            if score > _REPEAT_JACCARD_FLOOR:
                events.append(
                    FrictionEvent(
                        timestamp_utc=record.ts_utc,
                        bucket_label=_classify_bucket(record.ts_utc),
                        signal_family="repeated_fact",
                        message_id=_message_id(record, data_dir),
                        surface_text=record.text,
                        context_message_ids=(_message_id(records[prior_idx], data_dir),),
                        metadata=MappingProxyType({}),
                    )
                )
                break
    return events


def detect_events(
    records: list[HistoryRecord],
    data_dir: Path,
    *,
    facts: list[Any],
) -> list[FrictionEvent]:
    """Run all four signal detectors and concatenate their event lists.

    Detectors share no state and run independently; concatenation
    preserves the per-detector emission order, and the final report
    re-sorts by `(timestamp_utc, message_id)` for byte-stable output
    (see `_build_report`).
    """
    events: list[FrictionEvent] = []
    events.extend(_detect_frustration(records, data_dir))
    events.extend(_detect_kai_asks_back(records, data_dir))
    events.extend(_detect_preference_correction(records, data_dir, facts=facts))
    events.extend(_detect_repeated_fact(records, data_dir))
    return events


# ── Aggregation and trend classification ────────────────────────────


def aggregate(
    records: list[HistoryRecord], events: list[FrictionEvent]
) -> tuple[list[BucketAggregate], dict[str, int]]:
    """Compute per-bucket-per-signal aggregates and per-bucket user-turn counts.

    Initializes the (bucket, family) matrix with zero counts so every
    cell is present in the JSON even when no events fired in that
    bucket-family combination; downstream test assertions pin the
    matrix shape and missing rows would be a regression.

    Returns (aggregates, user_turns_by_bucket). `user_turns` is the
    denominator the trend classifier uses for the volume floor check;
    aggregating it here keeps the bucket-walk single-pass.
    """
    user_turns: dict[str, int] = {label: 0 for label in ("A", "B", "C")}
    for record in records:
        if record.direction == "user":
            user_turns[_classify_bucket(record.ts_utc)] += 1
    counts: dict[tuple[str, str], int] = {
        (bucket, family): 0 for bucket in ("A", "B", "C") for family in _SIGNAL_FAMILIES
    }
    for ev in events:
        counts[(ev.bucket_label, ev.signal_family)] += 1
    aggregates: list[BucketAggregate] = []
    for bucket in ("A", "B", "C"):
        denom = user_turns[bucket]
        for family in _SIGNAL_FAMILIES:
            count = counts[(bucket, family)]
            # 0.0 (NOT NaN) when the denominator is zero. NaN would
            # break downstream JSON parsing and table renders.
            rate = round(count / denom * 100, 2) if denom > 0 else 0.0
            aggregates.append(
                BucketAggregate(
                    bucket_label=bucket,
                    signal_family=family,
                    count=count,
                    user_turns_in_bucket=denom,
                    rate_per_100_user_turns=rate,
                )
            )
    return aggregates, user_turns


def _classify_band(prior: float, new: float) -> str:
    """Return 'drop' | 'flat' | 'up' for new vs prior under the pinned bands.

    Band semantics:
      - drop: new <= 0.85 * prior (boundary inclusive on this side)
      - flat: 0.85 * prior < new <= 1.15 * prior (open lower, closed upper)
      - up:   new > 1.15 * prior

    Zero-prior is handled inline by `classify_trend`; this helper
    assumes prior > 0. Tests that exercise the `unclassified`
    catch-all monkey-patch this helper to return a sentinel value
    outside `{drop, flat, up}`, then assert the classifier emits
    `unclassified` with the correct narrative.
    """
    if new <= _MATERIAL_DROP_RATIO * prior:
        return "drop"
    if new <= _UP_RATIO * prior:
        return "flat"
    return "up"


# Map (A->B band, B->C band) -> matched_pattern identifier. Exhaustive
# over the 9-cell `{drop, flat, up}^2` grid. `regression` absorbs the
# three (any, up) cells since any post-#361 increase is the
# verdict-driving observation. Missing keys would land in `unclassified`
# via the dict.get fallback in classify_trend; under the pinned rules
# this should never happen.
_GRID_TO_PATTERN: dict[tuple[str, str], str] = {
    ("drop", "drop"): "both-milestones-helped",
    ("drop", "flat"): "prediction-holds",
    ("drop", "up"): "regression",
    ("flat", "drop"): "track-1-carried-win",
    ("flat", "flat"): "no-effect",
    ("flat", "up"): "regression",
    ("up", "drop"): "retrieval-helped-pollution-hurt",
    ("up", "flat"): "retrieval-hurt-no-recovery",
    ("up", "up"): "regression",
}


_BAND_DESCRIPTIONS = {
    "drop": f"material drop, <= {_MATERIAL_DROP_RATIO}x",
    "flat": f"flat, in ({_MATERIAL_DROP_RATIO}x, {_UP_RATIO}x]",
    "up": f"up, > {_UP_RATIO}x",
}


def classify_trend(
    *,
    aggregates: list[BucketAggregate],
    events: list[FrictionEvent],
    user_turns: dict[str, int],
) -> TrendSummary:
    """Compute the trend classification + narrative for one run.

    Order of operations (inconclusive precedence first):
      1. Compute per-signal rates per bucket + the known-fact-overlap
         subset count for preference_correction.
      2. Compute verdict_driving_sum per bucket (frustration +
         preference_correction known-fact-overlap subset). When memory
         is unreachable the overlap count is zero by construction, so
         the sum collapses to frustration alone without special-casing.
      3. Compute A->B and B->C ratios with zero-denominator handling
         (prior=0 + new=0 -> 1.0; prior=0 + new>0 -> null).
      4. Inconclusive precedence: any bucket with user_turns < 30 fires
         `inconclusive` and skips the 9-cell grid.
      5. Inconclusive precedence: any zero-prior-rate transition (which
         pushes a `null` ratio AND a fallback-to-up band) also fires
         `inconclusive`.
      6. Otherwise, classify both transitions via _classify_band and
         look up the 9-cell grid; emit `unclassified` if the lookup
         misses (defensive catch-all that should never fire under the
         pinned rules).

    `verdict_driving_sum` and `ratios` are ALWAYS computed and emitted,
    even when matched_pattern is `inconclusive`, so the JSON preserves
    diagnostic value when band classification did not run.
    """
    rates_by_bucket = {
        bucket: {agg.signal_family: agg.rate_per_100_user_turns for agg in aggregates if agg.bucket_label == bucket}
        for bucket in ("A", "B", "C")
    }
    overlap_counts = {b: 0 for b in ("A", "B", "C")}
    for ev in events:
        if ev.signal_family != "preference_correction":
            continue
        if ev.metadata.get("known_fact_overlap"):
            overlap_counts[ev.bucket_label] += 1
    overlap_rates = {
        bucket: (round(overlap_counts[bucket] / user_turns[bucket] * 100, 2) if user_turns[bucket] > 0 else 0.0)
        for bucket in ("A", "B", "C")
    }
    verdict_sum = {
        bucket: round(rates_by_bucket[bucket].get("frustration", 0.0) + overlap_rates[bucket], 2)
        for bucket in ("A", "B", "C")
    }
    # Per-signal breakdown for the narrative; the preference_correction
    # row in the breakdown is the known-fact-overlap subset rate, NOT
    # the raw stage 3a rate. Same convention as the verdict sum.
    per_signal_breakdown = {
        bucket: {
            "frustration": rates_by_bucket[bucket].get("frustration", 0.0),
            "preference_correction": overlap_rates[bucket],
            "kai_asks_back": rates_by_bucket[bucket].get("kai_asks_back", 0.0),
            "repeated_fact": rates_by_bucket[bucket].get("repeated_fact", 0.0),
        }
        for bucket in ("A", "B", "C")
    }
    # Compute ratios + bands per transition. Any zero-prior-rate
    # transition (prior=0 + new>0) triggers an inconclusive caveat;
    # we accumulate the caveats and let the inconclusive-precedence
    # check below decide whether to also flip the matched_pattern.
    zero_prior_caveats: list[tuple[str, str, float]] = []
    ratios: dict[str, float | None] = {}
    bands: dict[str, str | None] = {}
    for prior_label, new_label, key in (("A", "B", "A_to_B"), ("B", "C", "B_to_C")):
        prior, new = verdict_sum[prior_label], verdict_sum[new_label]
        if prior == 0 and new == 0:
            ratios[key] = 1.0
            bands[key] = "flat"
        elif prior == 0 and new > 0:
            ratios[key] = None
            bands[key] = "up"
            zero_prior_caveats.append((prior_label, new_label, new))
        else:
            ratios[key] = round(new / prior, 3)
            bands[key] = _classify_band(prior, new)

    def _summary(matched_pattern: str, band_pair: tuple[str | None, str | None]) -> TrendSummary:
        return TrendSummary(
            predicted_pattern=_PREDICTED_PATTERN,
            matched_pattern=matched_pattern,
            verdict_driving_sum=verdict_sum,
            ratios=ratios,
            bands=bands,
            per_signal_rate_breakdown=per_signal_breakdown,
            zero_prior_rate_caveats=tuple(zero_prior_caveats),
            narrative=_render_narrative(
                matched=matched_pattern,
                verdict_sum=verdict_sum,
                ratios=ratios,
                per_signal_breakdown=per_signal_breakdown,
                bands=band_pair,
            ),
        )

    # Inconclusive precedence: volume floor.
    if any(user_turns[b] < _USER_TURN_FLOOR for b in ("A", "B", "C")):
        return _summary("inconclusive", (bands["A_to_B"], bands["B_to_C"]))
    # Inconclusive precedence: zero-prior-rate.
    if zero_prior_caveats:
        return _summary("inconclusive", (bands["A_to_B"], bands["B_to_C"]))
    # 9-cell grid lookup; defensive catch-all for `unclassified`.
    matched = _GRID_TO_PATTERN.get((bands["A_to_B"], bands["B_to_C"]), "unclassified")
    return _summary(matched, (bands["A_to_B"], bands["B_to_C"]))


def _render_narrative(
    *,
    matched: str,
    verdict_sum: dict[str, float],
    ratios: dict[str, float | None],
    per_signal_breakdown: dict[str, dict[str, float]],
    bands: tuple[str | None, str | None],
) -> str:
    """Build the prose narrative for the trend_summary block.

    Names the verdict-driving sum per bucket, both ratios with band
    labels, the matched pattern, and the per-signal breakdown. Keeps
    everything inline so the narrative is one paste-able block; the
    markdown summary references this same text.
    """
    band_a_to_b, band_b_to_c = bands
    a_to_b = ratios.get("A_to_B")
    b_to_c = ratios.get("B_to_C")
    a_to_b_str = "undefined" if a_to_b is None else f"{a_to_b:.3f}x"
    b_to_c_str = "undefined" if b_to_c is None else f"{b_to_c:.3f}x"
    band_a_to_b_str = _BAND_DESCRIPTIONS.get(band_a_to_b or "", "see caveats")
    band_b_to_c_str = _BAND_DESCRIPTIONS.get(band_b_to_c or "", "see caveats")
    parts: list[str] = []
    parts.append(
        "Verdict-driving sum (frustration + preference_correction known-fact overlap): "
        f"A={verdict_sum['A']:.2f}, B={verdict_sum['B']:.2f}, C={verdict_sum['C']:.2f}."
    )
    parts.append(f"A->B ratio: {a_to_b_str} ({band_a_to_b_str}); B->C ratio: {b_to_c_str} ({band_b_to_c_str}).")
    parts.append(f"Classification: {matched}.")
    parts.append(
        "Per-signal breakdown: "
        f"frustration {per_signal_breakdown['A']['frustration']:.2f} -> "
        f"{per_signal_breakdown['B']['frustration']:.2f} -> "
        f"{per_signal_breakdown['C']['frustration']:.2f}; "
        "preference_correction (known-fact overlap) "
        f"{per_signal_breakdown['A']['preference_correction']:.2f} -> "
        f"{per_signal_breakdown['B']['preference_correction']:.2f} -> "
        f"{per_signal_breakdown['C']['preference_correction']:.2f}; "
        f"kai_asks_back {per_signal_breakdown['A']['kai_asks_back']:.2f} -> "
        f"{per_signal_breakdown['B']['kai_asks_back']:.2f} -> "
        f"{per_signal_breakdown['C']['kai_asks_back']:.2f} (noisy); "
        f"repeated_fact {per_signal_breakdown['A']['repeated_fact']:.2f} -> "
        f"{per_signal_breakdown['B']['repeated_fact']:.2f} -> "
        f"{per_signal_breakdown['C']['repeated_fact']:.2f} (noisy)."
    )
    return " ".join(parts)


# ── Memory-store integration ────────────────────────────────────────


def _initialize_memory_local() -> MemoryInitResult:
    """Tri-state memory init that NEVER calls sys.exit.

    Diverges from the canonical _initialize_memory at
    src/kai/eval/behavioral.py:1901 in that the friction harness must
    complete a degraded run when memory is unavailable. The caller
    interprets ENABLED as "stage 3b will run", DISABLED as "stage 3b
    is a no-op via configuration", ERROR as "stage 3b is a no-op via
    crash" (the latter additionally surfaced in the report's
    `memory_store_error_message` field).

    Imports happen lazily inside the try/except so a missing optional
    `[memory]` extra (mem0ai, sentence-transformers, qdrant-client)
    surfaces as ERROR rather than crashing the harness at module import.
    """
    try:
        from kai.config import load_config
        from kai.memory import init_memory, is_enabled

        config = load_config()
        init_memory(config)
        if not is_enabled():
            return MemoryInitResult(MemoryAvailability.DISABLED, None)
        return MemoryInitResult(MemoryAvailability.ENABLED, None)
    except Exception as e:
        return MemoryInitResult(MemoryAvailability.ERROR, str(e))


def _load_facts_for_user(user_id: str, memory_state: MemoryInitResult) -> list[Any]:
    """Cache `get_by_tag(tag='preference')` once per run; empty list otherwise.

    A non-ENABLED memory state degrades stage 3b to a no-op (every
    preference_correction event ends up with `known_fact_overlap=False`).
    Stage 3b never re-queries per event; this function is the single
    fact fetch for the whole run.
    """
    if memory_state.availability != MemoryAvailability.ENABLED:
        return []
    try:
        from kai.memory import get_by_tag

        return get_by_tag(user_id=user_id, tag="preference")
    except Exception:
        log.exception("friction.fact_query_failed user_id=%s", user_id)
        return []


# ── Render layer ────────────────────────────────────────────────────


def _redact_text(text: str) -> str:
    """Apply the redaction regex set and truncate to the snippet cap.

    Pattern order is fixed in `_REDACT_REPLACEMENTS` so URL-then-email-
    then-handle-then-phone resolves overlap cases consistently.
    Truncation happens AFTER redaction so a redacted token is never
    split mid-replacement.
    """
    out = text
    for pattern, replacement in _REDACT_REPLACEMENTS:
        out = pattern.sub(replacement, out)
    return out[:_SNIPPET_MAX_CHARS]


def _build_event_payload(event: FrictionEvent, *, include_snippets: bool) -> dict[str, Any]:
    """Produce the JSON-serializable dict for one event.

    `metadata=dict(event.metadata)` is the load-bearing conversion from
    `MappingProxyType` to a plain dict; the default `json` encoder
    rejects MappingProxyType because `isinstance(MappingProxyType({}), dict)`
    is False. Without this conversion, the immutability wrapper at
    construction time silently breaks the render path the first time
    it is exercised; the test suite has a regression for this exact case.
    """
    surface = _redact_text(event.surface_text) if include_snippets else None
    return {
        "timestamp_utc": event.timestamp_utc.isoformat(),
        "bucket_label": event.bucket_label,
        "signal_family": event.signal_family,
        "message_id": event.message_id,
        "context_message_ids": list(event.context_message_ids),
        "metadata": dict(event.metadata),
        "surface_text": surface,
    }


def _build_bucket_payload(records: list[HistoryRecord]) -> list[dict[str, Any]]:
    """Per-bucket volume rows for the JSON `buckets` block.

    `messages_total` counts every record (user + assistant + media);
    `user_turns` counts only `direction == "user"`. The classifier and
    the rate denominator both read user_turns; messages_total is
    surfaced for diagnostic value only.
    """
    user_turns: dict[str, int] = {b: 0 for b in ("A", "B", "C")}
    messages_total: dict[str, int] = {b: 0 for b in ("A", "B", "C")}
    for record in records:
        bucket = _classify_bucket(record.ts_utc)
        messages_total[bucket] += 1
        if record.direction == "user":
            user_turns[bucket] += 1
    payload: list[dict[str, Any]] = []
    for bucket in _BUCKETS:
        payload.append(
            {
                "label": bucket.label,
                "start_utc": bucket.start.isoformat() if bucket.start else None,
                "end_utc": bucket.end.isoformat() if bucket.end else None,
                "user_turns": user_turns[bucket.label],
                "messages_total": messages_total[bucket.label],
            }
        )
    return payload


def _build_caveats(
    *,
    bucket_user_turns: dict[str, int],
    memory_state: MemoryInitResult,
    sample_days: int | None,
    sample_cutoff_utc: datetime | None,
    ratios_undefined: list[tuple[str, str, float]],
) -> list[str]:
    """Build the caveats list.

    Two unconditional caveats lead the list (the noisy-signal warnings),
    followed by conditional triggers in a stable order so the JSON is
    byte-stable across runs with the same set of triggered conditions.
    """
    caveats: list[str] = [
        "signal_family 'kai_asks_back' is noisy; bucket-level trend is the unit of interpretation.",
        "signal_family 'repeated_fact' is the noisiest of the four; treat per-event matches with skepticism and rely on bucket-level trend.",
    ]
    if memory_state.availability != MemoryAvailability.ENABLED:
        # Caveat label uses uppercase {DISABLED|ERROR}; .name preserves
        # the enum's uppercase form (vs .value which is lowercase).
        status_label = memory_state.availability.name
        caveats.append(
            f"memory-unreachable-frustration-only: memory store was {status_label}; "
            "preference_correction known-fact-overlap subset is empty by construction; "
            "verdict is single-signal (frustration only)."
        )
    if memory_state.availability == MemoryAvailability.ERROR and memory_state.error_message:
        caveats.append(f"memory_store_error_message: {memory_state.error_message}")
    for bucket in ("A", "B", "C"):
        n = bucket_user_turns.get(bucket, 0)
        if n < _USER_TURN_FLOOR:
            caveats.append(
                f"inconclusive: bucket {bucket} has user_turns={n} (< {_USER_TURN_FLOOR}); "
                "cross-bucket comparison is ill-defined."
            )
    if sample_days is not None and sample_cutoff_utc is not None:
        caveats.append(
            f"--sample-days={sample_days} restricts to records with "
            f"ts_utc >= {sample_cutoff_utc.isoformat()}; older history is excluded from this report."
        )
    for prior_label, new_label, rate_new in ratios_undefined:
        caveats.append(
            f"zero-prior-rate: rate for bucket {prior_label} is 0 and bucket "
            f"{new_label} is {rate_new}; ratio is undefined and band classification "
            "falls back to up."
        )
    return caveats


def _build_report(
    *,
    config: FrictionConfig,
    records: list[HistoryRecord],
    events: list[FrictionEvent],
    aggregates: list[BucketAggregate],
    user_turns: dict[str, int],
    trend: TrendSummary,
    memory_state: MemoryInitResult,
    sample_cutoff_utc: datetime | None,
) -> dict[str, Any]:
    """Produce the top-level JSON report dict.

    Events are sorted by `(timestamp_utc, message_id)` so the JSON is
    byte-stable across runs against a fixed history snapshot, which is
    what makes regression tests possible.

    `memory_store_error_message` is included only when status is ERROR;
    omitting the field on the success path keeps the JSON shape
    minimal in the common case.
    """
    sorted_events = sorted(events, key=lambda e: (e.timestamp_utc, e.message_id))
    report: dict[str, Any] = {
        "version": _OUTPUT_SCHEMA_VERSION,
        "generated_at_utc": config.run_started_at_utc.isoformat(),
        "user_id": config.user_id,
        "data_dir": str(config.data_dir),
        "sample_days": config.sample_days,
        "memory_store_reachable": memory_state.availability == MemoryAvailability.ENABLED,
        "memory_store_status": memory_state.availability.value,
    }
    if memory_state.availability == MemoryAvailability.ERROR:
        report["memory_store_error_message"] = memory_state.error_message
    report["buckets"] = _build_bucket_payload(records)
    report["aggregates"] = [
        {
            "bucket_label": agg.bucket_label,
            "signal_family": agg.signal_family,
            "count": agg.count,
            "user_turns_in_bucket": agg.user_turns_in_bucket,
            "rate_per_100_user_turns": agg.rate_per_100_user_turns,
        }
        for agg in aggregates
    ]
    report["events"] = [_build_event_payload(ev, include_snippets=config.include_snippets) for ev in sorted_events]
    report["trend_summary"] = {
        "predicted_pattern": trend.predicted_pattern,
        "matched_pattern": trend.matched_pattern,
        "verdict_driving_sum": trend.verdict_driving_sum,
        "ratios": trend.ratios,
        "narrative": trend.narrative,
    }
    report["caveats"] = _build_caveats(
        bucket_user_turns=user_turns,
        memory_state=memory_state,
        sample_days=config.sample_days,
        sample_cutoff_utc=sample_cutoff_utc,
        ratios_undefined=list(trend.zero_prior_rate_caveats),
    )
    return report


def _json_default(obj: Any) -> Any:
    """Default encoder for any object the standard encoder rejects.

    Defensive net for `MappingProxyType` (which `_build_event_payload`
    already converts at the event layer). If this ever fires it
    indicates a NEW code path forgot the MappingProxyType -> dict
    conversion that `_build_event_payload` performs.
    """
    if isinstance(obj, Mapping):
        return dict(obj)
    raise TypeError(f"unserializable: {type(obj).__name__}")


def _render_json(report: dict[str, Any]) -> str:
    """Serialize the report dict; pretty-printed for human review."""
    return json.dumps(report, indent=2, ensure_ascii=False, default=_json_default)


def _render_markdown(report: dict[str, Any], trend: TrendSummary) -> str:
    """Build the redacted markdown summary suitable for paste-into-issue.

    Always redacts surface text regardless of `--include-snippets` (the
    flag only affects the JSON report and only for local inspection).
    The user_id is rendered as the literal "<user_id-redacted>" so
    posting the markdown publicly does not leak the chat ID.

    Bands are read from the in-memory `TrendSummary` object rather than
    re-derived from the JSON, so the markdown's band labels stay in
    sync with the classifier's view without duplicate logic.
    """
    buckets = report["buckets"]
    aggregates = report["aggregates"]
    rate_by_cell = {(agg["bucket_label"], agg["signal_family"]): agg["rate_per_100_user_turns"] for agg in aggregates}
    lines: list[str] = []
    lines.append("## Layer 3 evaluation: results")
    lines.append("")
    lines.append("Friction analysis of <user_id-redacted> chat history across three milestone buckets.")
    lines.append("")
    lines.append("### Bucket volumes")
    lines.append("")
    lines.append("| bucket | dates (UTC) | user_turns | messages_total |")
    lines.append("|---|---|---|---|")
    for bucket in buckets:
        start = bucket["start_utc"] or "(open)"
        end = bucket["end_utc"] or "(open)"
        lines.append(f"| {bucket['label']} | {start} -> {end} | {bucket['user_turns']} | {bucket['messages_total']} |")
    lines.append("")
    lines.append("### Friction rates (per 100 user-turns)")
    lines.append("")
    lines.append("| signal | bucket A | bucket B | bucket C |")
    lines.append("|---|---|---|---|")
    for family in _SIGNAL_FAMILIES:
        lines.append(
            f"| {family} | {rate_by_cell[('A', family)]:.2f} | "
            f"{rate_by_cell[('B', family)]:.2f} | {rate_by_cell[('C', family)]:.2f} |"
        )
    lines.append("")
    lines.append("### Trend")
    lines.append("")
    lines.append(f"Predicted: {trend.predicted_pattern}. Matched: {trend.matched_pattern}.")
    lines.append("")
    a_to_b = trend.ratios.get("A_to_B")
    b_to_c = trend.ratios.get("B_to_C")
    a_to_b_str = "undefined" if a_to_b is None else f"{a_to_b:.3f}x"
    b_to_c_str = "undefined" if b_to_c is None else f"{b_to_c:.3f}x"
    band_a_to_b_str = _BAND_DESCRIPTIONS.get(trend.bands.get("A_to_B") or "", "see caveats")
    band_b_to_c_str = _BAND_DESCRIPTIONS.get(trend.bands.get("B_to_C") or "", "see caveats")
    lines.append(
        "Verdict-driving sum (frustration + preference_correction known-fact overlap, "
        f"per 100 user-turns): A={trend.verdict_driving_sum['A']:.2f}, "
        f"B={trend.verdict_driving_sum['B']:.2f}, "
        f"C={trend.verdict_driving_sum['C']:.2f}. "
        f"A->B ratio: {a_to_b_str} ({band_a_to_b_str}); "
        f"B->C ratio: {b_to_c_str} ({band_b_to_c_str})."
    )
    lines.append("")
    lines.append(f"Narrative: {trend.narrative}")
    lines.append("")
    lines.append("### Caveats")
    lines.append("")
    for caveat in report["caveats"]:
        lines.append(f"- {caveat}")
    lines.append("")
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    """Write content to path atomically via temp-file + os.replace.

    A partial write (process killed mid-write, disk full mid-flush)
    would corrupt the report and silently pollute downstream comparison
    scripts. `os.replace` is atomic on both POSIX and Windows. The temp
    file is created in the SAME directory as the destination so the
    rename is intra-filesystem and the atomicity guarantee holds.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_str = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(temp_path, path)
    except Exception:
        # Best-effort cleanup of the temp file on any write failure;
        # leaving it would accumulate stale .tmp files in the output
        # directory across repeated failed runs. The os.close(fd) call
        # covers the narrow window where os.fdopen raised before taking
        # ownership of the raw fd from mkstemp; in the common case
        # (os.fdopen succeeded, the with-block already closed the fd,
        # then os.replace failed) this os.close is a no-op that raises
        # EBADF, which the inner OSError handler intentionally swallows.
        try:
            os.close(fd)
        except OSError:
            pass
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise


# ── CLI ─────────────────────────────────────────────────────────────


def _non_negative_int(value: str) -> int:
    """argparse type validator: integer >= 0.

    Mirrors `_non_negative_int` at src/kai/eval/behavioral.py:1775.
    Duplicated rather than imported to keep the two eval modules
    independently runnable (a partial install with only friction.py
    available should still parse its CLI).
    """
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected integer, got {value!r}") from exc
    if n < 0:
        raise argparse.ArgumentTypeError(f"expected non-negative integer, got {value!r}")
    return n


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser for the friction CLI surface."""
    parser = argparse.ArgumentParser(
        prog="python -m kai.eval.friction",
        description=(
            "Layer 3 longitudinal friction analysis. Reads chat history "
            "for a single user, runs four pre-committed friction-detection "
            "signal families, buckets events by milestone (PR #333 and "
            "PR #361 merge instants), and emits a schema-versioned JSON "
            "report plus an optional redacted markdown summary."
        ),
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="Chat ID subdirectory under DATA_DIR/history/. Single user per run.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Absolute path for the JSON report. Atomic temp+rename is used.",
    )
    parser.add_argument(
        "--markdown-summary",
        type=Path,
        default=None,
        help="Optional path for the redacted markdown summary suitable for paste-into-issue.",
    )
    parser.add_argument(
        "--sample-days",
        type=_non_negative_int,
        default=None,
        help=(
            "Restrict to records with ts_utc >= run_started_at_utc - timedelta(days=N). "
            "0 means 'today UTC' (midnight UTC of the run's calendar day). "
            "Omit to include all history."
        ),
    )
    parser.add_argument(
        "--include-snippets",
        action="store_true",
        help=(
            "Populate surface_text in the JSON report (still redacted: "
            "URLs, email addresses, @handles, and phone-shaped digit "
            "runs are scrubbed; per-event excerpt capped at 200 chars). "
            "The markdown summary always omits surface_text regardless."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override DATA_DIR resolution. Tests pass tmp_path; production uses KAI_DATA_DIR.",
    )
    return parser


def _resolve_sample_cutoff(run_started_at_utc: datetime, sample_days: int | None) -> datetime | None:
    """Compute the inclusive lower bound for the --sample-days filter.

    `--sample-days 0` means "today UTC" (midnight UTC of the run's
    calendar day) by convention. `None` means no filter; the caller
    treats a `None` cutoff as "include every record."
    """
    if sample_days is None:
        return None
    if sample_days == 0:
        return run_started_at_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    return run_started_at_utc - timedelta(days=sample_days)


def _resolve_data_dir(override: Path | None) -> Path:
    """Use the CLI override when provided, else the module-level DATA_DIR.

    DATA_DIR is evaluated once at import time from KAI_DATA_DIR; the
    operator's run procedure sets KAI_DATA_DIR BEFORE invoking python
    so the constant matches the snapshot. The CLI override exists for
    tests (which point at tmp_path) and for the rare ad-hoc operator
    run that wants a different path without re-launching python.
    """
    if override is not None:
        return override
    from kai.config import DATA_DIR

    return DATA_DIR


async def _run_cli(args: argparse.Namespace, run_started_at_utc: datetime) -> int:
    """CLI dispatch. Returns process exit code: 0 success, 2 on bad CLI input.

    `_run_cli` is `async` to mirror behavioral.py's convention even
    though the friction harness has no awaitable work; future
    extensions (parallel detector execution, async fact lookups) would
    sit naturally inside this function without requiring a top-level
    refactor.
    """
    try:
        _validate_user_id(args.user_id)
    except ValueError as e:
        print(f"friction: {e}", file=sys.stderr)
        return 2
    data_dir = _resolve_data_dir(args.data_dir)
    config = FrictionConfig(
        user_id=args.user_id,
        data_dir=data_dir,
        output_path=args.output,
        sample_days=args.sample_days,
        include_snippets=args.include_snippets,
        markdown_summary_path=args.markdown_summary,
        run_started_at_utc=run_started_at_utc,
    )
    sample_cutoff = _resolve_sample_cutoff(run_started_at_utc, config.sample_days)
    memory_state = _initialize_memory_local()
    facts = _load_facts_for_user(config.user_id, memory_state)
    records = list(read_history(config.data_dir, config.user_id))
    if sample_cutoff is not None:
        records = [r for r in records if r.ts_utc >= sample_cutoff]
    events = detect_events(records, config.data_dir, facts=facts)
    aggregates, user_turns = aggregate(records, events)
    trend = classify_trend(
        aggregates=aggregates,
        events=events,
        user_turns=user_turns,
    )
    report = _build_report(
        config=config,
        records=records,
        events=events,
        aggregates=aggregates,
        user_turns=user_turns,
        trend=trend,
        memory_state=memory_state,
        sample_cutoff_utc=sample_cutoff,
    )
    _atomic_write(config.output_path, _render_json(report) + "\n")
    if config.markdown_summary_path is not None:
        _atomic_write(config.markdown_summary_path, _render_markdown(report, trend))
    print(
        f"friction: {len(events)} events across {sum(user_turns.values())} user-turns "
        f"-> {config.output_path} (matched: {trend.matched_pattern})",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Process entry point.

    Captures `_run_started_at_utc` exactly once before any reading
    happens, threading it through the entire run via `FrictionConfig`.
    This is the only `datetime.now(UTC)` call the harness makes in
    production; everything downstream derives from this single
    wall-clock read so a run that straddles a day boundary while
    reading still produces consistent cutoffs (the single-clock-read
    invariant is what makes outputs deterministic across replays).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_started_at_utc = datetime.now(UTC)
    return asyncio.run(_run_cli(args, run_started_at_utc))


if __name__ == "__main__":
    sys.exit(main())
