"""
Offline transcript-provenance backfill for legacy memory rows.

Backs the `python -m kai memory backfill-provenance` subcommand.
Targets rows extracted before transcript provenance landed: every
such row has no `source_*` metadata, so `/memory` renders
"Source: not recorded (legacy)" and reclassification's optional
`said:` quote line stays empty on the entire pre-cutover corpus.

Three modes, three drivers:

1. `run_dry_run`: enumerate candidate rows, score each row's text
   against every user turn in a wide time window, accept matches with
   dominant content overlap, write a report + a proposals file. No
   store writes. Skips that do not match are surfaced as a curation
   backlog (with their top candidates) so an operator can review the
   gating.
2. `run_apply`: re-check and apply a reviewed proposals file. Drift
   gates protect against intervening writes to the store and the
   JSONL; surviving proposals get the four required `source_*` keys
   merged into the row's existing metadata. Pre-images dumped first,
   carrying `applied_source_block` so rollback can detect later
   operator corrections to the source_* fields.
3. `run_rollback`: restore rows from a pre-image file. A row whose
   current `source_*` block differs from the applied block is
   skipped: backfill writes can be rolled back, but later operator
   corrections must NOT be silently overwritten.

Design contract (the spec's load-bearing properties):

- AUTOMATIC matches require CONTENT OVERLAP, not temporal proximity.
  Memory ingestion is fire-and-forget and serialized through a
  per-user semaphore; Mem0 `created_at` is storage time, not source-
  turn time, so a later user turn can sit between the true source
  and the row's creation. Treating proximity as evidence admits
  wrong-turn matches by construction. The discriminator is 4-gram
  token-shingle overlap between the row text and a candidate user
  turn; the time window is a search prefilter, not the scoring
  signal.
- Rows that do not auto-match land in a CURATION BACKLOG, not in
  the proposals file. Interactive curation is out of scope for this
  PR; the operator addresses the backlog in a follow-up pass.
- Drift gates protect every mutating step: row presence, current
  provenance, metadata fingerprint, JSONL line fingerprint at apply;
  source-block byte-comparison at rollback.
- Apply consumes the proposals file verbatim; it never re-scores.
  Scoring-style CLI flags are rejected in mutating modes so a typo
  cannot silently change semantics.

Architectural shape mirrors `kai.memory_reclassify`: pure helpers
(selection, scoring, gating, file (de)serialization, report
rendering, header validation) do no I/O and are unit-testable
without Mem0 or the filesystem; the three async drivers own store
access, JSONL access, and file writes. The enforcement of every
safety gate (CLI flag rejection, exit-code policy, exclusive
pre-image creation) lives in the drivers and the CLI layer, never
inside a pure helper, so the gates stay visible at the orchestration
layer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from kai import memory
from kai.config import Config
from kai.history import _LOG_DIR, fetch_transcript_context
from kai.memory import (
    SOURCE_CHAT_ID_KEY,
    SOURCE_DATE_KEY,
    SOURCE_USER_TEXT_SHA256_KEY,
    SOURCE_USER_TS_KEY,
    MemoryResult,
    read_transcript_provenance,
)

log = logging.getLogger(__name__)


# ── Constants ───────────────────────────────────────────────────────


# Source values eligible for backfill. `migration` rows came from a
# one-shot data import (not real conversation turns) and have no JSONL
# to point at; `user_raw` rows are raw inputs, not extraction outputs.
# Only `extracted` and `episode` are transcript-derived.
_ELIGIBLE_SOURCES: frozenset[str] = frozenset({"extracted", "episode"})


# Default time window for the JSONL prefilter. 24 hours pre-`created_at`
# admits the slow tail of the extraction pipeline (queued tasks,
# retried backends, episodes spawned after the next message) while
# bounding the JSONL read cost per row. The window is a SEARCH FILTER,
# not the scoring discriminator; content overlap is what authorizes a
# proposal. Operators with known multi-day-delayed backlogs can widen
# via `--window-seconds`.
_DEFAULT_WINDOW_SECONDS = 86400


# Minimum overlap score for a candidate to be considered. Below this,
# no candidate passes the gate. 0.30 says "at least 30% of the shorter
# text's 4-grams must appear in both". Tunable via `--min-overlap`.
_DEFAULT_MIN_OVERLAP = 0.30


# Dominance ratio the winner must clear over the runner-up to be a
# STRONG_MATCH. 2.0 means the winner's overlap score must be at least
# twice the runner-up's. A single candidate above `--min-overlap` with
# no runner-up clearing it is treated as infinite dominance. Tunable
# via `--strong-overlap-ratio`.
_DEFAULT_STRONG_OVERLAP_RATIO = 2.0


# Shingle width for the overlap score, in tokens. 4-token shingles
# capture lifted phrases without matching stock 3-word patterns
# ("I want to", "the fact that") that appear across unrelated turns.
# Character shingles inflate overlap on short rows; token shingles
# respect word identity. Tunable via `--overlap-shingle-n`.
_DEFAULT_SHINGLE_N = 4


# Per-row candidates rendered in the curation backlog section of the
# report. Three is enough for the operator to see whether widening
# `--min-overlap` would help on this row, without ballooning the
# report for a corpus with thousands of skips.
_CURATION_REPORT_TOP_N = 3


# Stable skip-reason keys. They appear in reports and summaries, so
# renames are reader-visible; treat like log vocabulary.
SKIP_AMBIGUOUS_OVERLAP = "ambiguous_overlap"
SKIP_NO_CANDIDATE = "no_candidate"
SKIP_HISTORY_UNREADABLE = "history_unreadable"
# Apply-time re-check skips.
SKIP_ROW_GONE = "row_gone"
SKIP_DESELECTED = "deselected"
SKIP_METADATA_DRIFT = "metadata_drift"
SKIP_TRANSCRIPT_DRIFT = "transcript_drift"
# Rollback-only skip: the row's source_* block was changed by another
# writer after backfill; rollback must not silently undo that change.
SKIP_OPERATOR_CORRECTION = "operator_correction"


# Metadata keys the apply-time fingerprint covers. Selected so a drift
# means the row's classification context has actually changed (a
# concurrent reclassify or operator edit), not just a benign refresh
# of an irrelevant field. Keeping the set narrow also keeps the
# fingerprint stable across Mem0-internal metadata churn (timestamps,
# auto-preserved fields) that update_metadata can introduce.
_FINGERPRINT_KEYS: tuple[str, ...] = (
    "source",
    "session_id",
    "scope",
    "project_id",
    "workspace_root",
    "scope_confidence",
    "scope_source",
)


# Regex for normalization. Replaces every run of non-alphanumeric
# characters with a single space. Compiled once at import; the scoring
# pass calls it on every row text and every candidate.
_NON_ALPHANUM_RE = re.compile(r"[^a-z0-9]+")


# Structured log event for an applied row.
_PROVENANCE_BACKFILL_EVENT = "memory.provenance.backfill"


# ── Data shapes ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Candidate:
    """One user turn from the JSONL within a row's time window.

    Carries the per-candidate data the scoring pass and the report
    both need: the raw timestamp and text for downstream display,
    the sha256 of the text for the apply-time drift gate, the
    overlap score the gating function uses, and the gap in seconds
    between the candidate ts and the row's `created_at` (the
    operator reads this in the report when auditing why a particular
    candidate was or was not the winner).
    """

    ts: str
    text: str
    sha256: str
    overlap_score: float
    gap_seconds: float


@dataclass(frozen=True)
class Proposal:
    """One reviewable proposal from a dry run.

    The proposal carries enough context for both the apply path and
    the operator audit: the four `source_*` values to be written,
    the overlap score and runner-up score, the candidate count, the
    gap, and the metadata fingerprint apply will re-check against.
    Row text and candidate text snippets ride along so a hand audit
    of the proposals file does not require cross-referencing back
    to the store and the JSONL.
    """

    memory_id: str
    chat_id: int
    date: str
    user_ts: str
    user_text_sha256: str
    overlap_score: float
    runner_up_overlap_score: float
    candidate_count: int
    gap_seconds: float
    prior_metadata_sha256: str
    row_text_snippet: str
    candidate_text_snippet: str


@dataclass(frozen=True)
class PreImage:
    """One row's pre-apply state plus the post-image source block.

    `metadata_before` snapshots the row's metadata as it was BEFORE
    backfill stamped the four `source_*` keys. `applied_source_block`
    captures the four keys backfill is about to write; at rollback,
    the harness reads the row's current source_* values and compares
    them against this block. Without the post-image, rollback cannot
    distinguish "backfill's write is still intact" from "someone
    changed the source_* fields since".
    """

    memory_id: str
    text: str
    metadata_before: dict[str, Any]
    applied_source_block: dict[str, Any]


@dataclass
class RowMatch:
    """A row's matching outcome, for the report's narrative.

    `bucket` is one of STRONG_MATCH (a proposal was emitted),
    SKIP_AMBIGUOUS_OVERLAP, SKIP_NO_CANDIDATE, or
    SKIP_HISTORY_UNREADABLE. `top_candidates` carries the top
    `_CURATION_REPORT_TOP_N` candidates in overlap-score order so the
    report's curation backlog section can render them without re-
    scoring. STRONG_MATCH entries also carry `top_candidates`
    (winner first, runner-up second, ...) so the sample section can
    render the same context the proposals JSONL holds.
    """

    row: MemoryResult
    bucket: str
    top_candidates: list[Candidate]
    proposal: Proposal | None


# ── Pure helpers: text overlap scoring ──────────────────────────────


def _normalize_text(text: str) -> list[str]:
    """Lowercase and tokenize on whitespace after stripping non-
    alphanumerics. Returns the token list, empty when the input
    contained no alphanumeric characters at all (a corner case for
    media-only memory rows that should never have been extracted
    from a text turn anyway; the scoring pass will see zero shingles
    and produce zero overlap).
    """
    lowered = text.lower()
    cleaned = _NON_ALPHANUM_RE.sub(" ", lowered)
    return cleaned.split()


def _shingles(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    """Return the set of n-grams over `tokens`.

    Texts with fewer than `n` tokens produce a single shingle whose
    tail is empty-string padded; this lets a very short row still
    produce a non-empty shingle set so the overlap denominator is
    not zero (which would either propagate as a NaN or require a
    special-case guard at every call site). The padding is benign:
    the empty-string positions cannot coincidentally match a real
    candidate's tokens, so a short row's overlap is meaningful only
    against another short row with the same alphanumeric tokens.
    """
    if len(tokens) >= n:
        return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}
    if not tokens:
        return set()
    padded = tokens + [""] * (n - len(tokens))
    return {tuple(padded[:n])}


def _overlap_score(
    row_shingles: set[tuple[str, ...]],
    candidate_shingles: set[tuple[str, ...]],
) -> float:
    """Return the intersection-over-min overlap score.

    `|intersection| / min(|row|, |candidate|)` rather than Jaccard:
    a long candidate user turn that happens to contain a short
    row's exact phrasing should score 1.0 on overlap, not be
    penalized for the candidate's extra length. Jaccard would
    discount the match because the union is large. The "min"
    denominator captures the intuition that we are measuring how
    much of the shorter text is reflected in the longer one.
    """
    if not row_shingles or not candidate_shingles:
        return 0.0
    intersection = row_shingles & candidate_shingles
    denom = min(len(row_shingles), len(candidate_shingles))
    return len(intersection) / denom


def _score_candidates(
    row_text: str,
    candidates: list[dict],
    *,
    shingle_n: int,
    created_at_dt: datetime,
) -> list[Candidate]:
    """Score every candidate user-direction record against the row.

    Pure function over the candidate dicts (each `{ts, text}` from
    the JSONL window read). Computes overlap and gap_seconds; the
    caller sorts and gates. Returns Candidates in original order;
    the caller picks the winner.
    """
    row_tokens = _normalize_text(row_text)
    row_shingles = _shingles(row_tokens, shingle_n)
    out: list[Candidate] = []
    for cand in candidates:
        text = cand.get("text", "")
        ts = cand.get("ts", "")
        if not isinstance(text, str) or not isinstance(ts, str):
            continue
        cand_tokens = _normalize_text(text)
        cand_shingles = _shingles(cand_tokens, shingle_n)
        score = _overlap_score(row_shingles, cand_shingles)
        try:
            cand_dt = datetime.fromisoformat(ts)
        except ValueError:
            # A malformed timestamp on a JSONL record is a corruption
            # signal; skip the record rather than letting it crash
            # the row's scoring pass. The HISTORY_UNREADABLE bucket
            # catches whole-file corruption; this skips single bad
            # lines while the rest of the file still contributes.
            continue
        if cand_dt.tzinfo is None:
            cand_dt = cand_dt.replace(tzinfo=UTC)
        gap = (created_at_dt - cand_dt).total_seconds()
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        out.append(
            Candidate(
                ts=ts,
                text=text,
                sha256=sha,
                overlap_score=score,
                gap_seconds=gap,
            )
        )
    return out


def _gate_match(
    scored: list[Candidate],
    *,
    min_overlap: float,
    strong_overlap_ratio: float,
) -> tuple[Candidate | None, str, float]:
    """Decide STRONG_MATCH / AMBIGUOUS_OVERLAP / NO_CANDIDATE.

    Returns `(winner | None, bucket, runner_up_score)`. The winner is
    the highest-overlap candidate when STRONG_MATCH; None otherwise.
    `runner_up_score` is 0.0 when no runner-up exists; it is reported
    in the proposal so an audit reader can see the dominance gap.

    Decision tree:
        max_overlap < min_overlap         -> NO_CANDIDATE
        no runner-up above min_overlap    -> STRONG_MATCH (infinite dominance)
        winner / runner_up >= ratio       -> STRONG_MATCH
        else                              -> AMBIGUOUS_OVERLAP
    """
    if not scored:
        return None, SKIP_NO_CANDIDATE, 0.0
    sorted_scored = sorted(scored, key=lambda c: c.overlap_score, reverse=True)
    winner = sorted_scored[0]
    if winner.overlap_score < min_overlap:
        return None, SKIP_NO_CANDIDATE, 0.0
    # Find the highest-scoring runner-up that also clears min_overlap.
    # A runner-up below min_overlap does not count toward ambiguity:
    # the gating contract is "two candidates that BOTH look plausible,
    # neither clearly dominating", not "two candidates of any quality".
    runner_up_score = 0.0
    for c in sorted_scored[1:]:
        if c.overlap_score >= min_overlap:
            runner_up_score = c.overlap_score
            break
    if runner_up_score == 0.0:
        return winner, "STRONG_MATCH", 0.0
    if winner.overlap_score / runner_up_score >= strong_overlap_ratio:
        return winner, "STRONG_MATCH", runner_up_score
    return None, SKIP_AMBIGUOUS_OVERLAP, runner_up_score


# ── Pure helpers: selection and fingerprinting ──────────────────────


def select_rows(rows: list[MemoryResult]) -> list[MemoryResult]:
    """Filter the candidate population for backfill.

    Two conjoint criteria: the row must be transcript-derived
    (source in {extracted, episode}; `migration` and `user_raw` are
    out by design) AND the row must lack present provenance
    (`read_transcript_provenance(...).present is False`). Rows
    matching both go forward to scoring; everything else is silently
    out of the population (NOT bucketed as a skip, because they were
    never candidates).
    """
    out: list[MemoryResult] = []
    for row in rows:
        source = (row.metadata or {}).get("source")
        if source not in _ELIGIBLE_SOURCES:
            continue
        provenance = read_transcript_provenance(row.metadata)
        if provenance.present:
            continue
        out.append(row)
    return out


def _metadata_fingerprint(metadata: dict[str, Any] | None) -> str:
    """Canonical-JSON sha256 over the drift-relevant metadata keys.

    Used by the apply-time drift gate: a row's fingerprint at dry-
    run captured into the proposal, re-computed at apply, mismatched
    means the row's classification context was touched between
    dry-run and apply (most likely by a concurrent reclassify or
    operator edit). Keys outside `_FINGERPRINT_KEYS` do not
    contribute, so Mem0-internal metadata churn (auto-preserved
    timestamps and the like) does not register as drift.

    Canonical JSON is the sorted-key, separator-tight form so two
    structurally-equal metadata dicts hash identically regardless
    of insertion order.
    """
    md = metadata or {}
    subset = {k: md.get(k) for k in _FINGERPRINT_KEYS}
    blob = json.dumps(subset, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ── Pure helpers: JSONL window reads ────────────────────────────────


def _date_files_in_window(
    chat_id: int,
    window_start: datetime,
    window_end: datetime,
) -> list[Path]:
    """List every JSONL file whose UTC date intersects the window.

    Files are sharded as `<_LOG_DIR>/<chat_id>/<YYYY-MM-DD>.jsonl`;
    one file per UTC day. A window of N hours can intersect 1, 2,
    or more day files (a 24h window straddling midnight is two
    files; a 72h window is three or four; an operator widening to
    a multi-day backlog gets the corresponding number of files).
    Returns paths in chronological order.

    The list is generated by walking the date range, not by globbing
    the directory: a glob would also surface files outside the
    window when one happens to share the same prefix; the explicit
    walk is unambiguous and stays cheap (one Path per day).
    """
    start_date = window_start.date()
    end_date = window_end.date()
    out: list[Path] = []
    current = start_date
    while current <= end_date:
        out.append(_LOG_DIR / str(chat_id) / f"{current.isoformat()}.jsonl")
        current += timedelta(days=1)
    return out


def _read_jsonl_records(path: Path) -> list[dict] | None:
    """Read a JSONL file into a list of dicts.

    Returns the list on success, None when the file exists but cannot
    be read (permissions, partial corruption). A nonexistent file is
    a normal absence (no conversation on that day for this chat) and
    returns an EMPTY LIST, distinct from the None signal: the caller
    treats absence as "no candidates from this day" and unreadability
    as "we cannot trust the window's coverage for this row".

    Malformed-line policy is fail-closed: a single JSONDecodeError
    inside an otherwise-readable file collapses the whole file to
    None. The safety contract is "do not score against partial
    history": if the true source turn is the malformed line, or a
    runner-up above `--min-overlap` is, the row could be proposed
    on incomplete evidence. Silently skipping bad lines would turn
    file-level corruption into a SCORING signal, exactly what the
    HISTORY_UNREADABLE bucket exists to prevent at the file level.
    """
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    records: list[dict] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            return None
        if isinstance(rec, dict):
            records.append(rec)
    return records


def _candidates_for_row(
    chat_id: int,
    created_at_dt: datetime,
    *,
    window_seconds: int,
) -> tuple[list[dict], bool]:
    """Return user-direction records in the row's window plus an
    `any_unreadable` flag.

    Walks every JSONL file intersecting the window
    `[created_at - window_seconds, created_at]`. When ANY intersecting
    file exists but cannot be read, returns `(records_so_far, True)`:
    the caller buckets the row as HISTORY_UNREADABLE rather than
    scoring against partial history (a winner picked from partial
    data could be wrong because the true source sits inside the
    unreadable file). Absent files (no chat that day) are normal and
    do not raise the flag.

    Returns user-direction records whose `ts` falls inside the
    window, in file-then-line order. The caller scores them.
    """
    window_start = created_at_dt - timedelta(seconds=window_seconds)
    window_end = created_at_dt
    files = _date_files_in_window(chat_id, window_start, window_end)
    any_unreadable = False
    out: list[dict] = []
    for path in files:
        records = _read_jsonl_records(path)
        if records is None:
            any_unreadable = True
            continue
        for rec in records:
            if rec.get("dir") != "user":
                continue
            # User-record corruption is the same partial-history hazard
            # as a malformed JSON line: a true-source user record whose
            # `ts` is missing, the wrong type, or unparseable would be
            # silently dropped from the candidate set, leaving an
            # unrelated valid user record free to win the overlap. Mark
            # the file as unreadable for THIS row so the caller buckets
            # the whole row HISTORY_UNREADABLE rather than scoring
            # against incomplete evidence.
            ts = rec.get("ts")
            if not isinstance(ts, str):
                any_unreadable = True
                continue
            try:
                rec_dt = datetime.fromisoformat(ts)
            except ValueError:
                any_unreadable = True
                continue
            if rec_dt.tzinfo is None:
                rec_dt = rec_dt.replace(tzinfo=UTC)
            if window_start <= rec_dt <= window_end:
                out.append(rec)
    return out, any_unreadable


# ── Pure helpers: artifact (de)serialization ────────────────────────


def render_proposals(header: dict[str, Any], proposals: list[Proposal]) -> str:
    """Serialize a proposals file: header line, then proposal lines."""
    lines = [json.dumps({"type": "header", **header}, separators=(",", ":"))]
    for p in proposals:
        lines.append(
            json.dumps(
                {
                    "type": "proposal",
                    "memory_id": p.memory_id,
                    "chat_id": p.chat_id,
                    "date": p.date,
                    "user_ts": p.user_ts,
                    "user_text_sha256": p.user_text_sha256,
                    "overlap_score": p.overlap_score,
                    "runner_up_overlap_score": p.runner_up_overlap_score,
                    "candidate_count": p.candidate_count,
                    "gap_seconds": p.gap_seconds,
                    "prior_metadata_sha256": p.prior_metadata_sha256,
                    "row_text_snippet": p.row_text_snippet,
                    "candidate_text_snippet": p.candidate_text_snippet,
                },
                separators=(",", ":"),
            )
        )
    return "\n".join(lines) + "\n"


def render_preimages(header: dict[str, Any], preimages: list[PreImage]) -> str:
    """Serialize a pre-image file: header line, then row lines.

    Each row carries `metadata_before` (the row's metadata as it was
    before backfill stamped the four source_* keys) AND
    `applied_source_block` (the four keys backfill wrote). Both are
    necessary: metadata_before is what rollback writes back; the
    applied block is what rollback compares the row's current
    source_* keys against, to detect later operator corrections.
    """
    lines = [json.dumps({"type": "header", **header}, separators=(",", ":"))]
    for p in preimages:
        lines.append(
            json.dumps(
                {
                    "type": "preimage",
                    "memory_id": p.memory_id,
                    "text": p.text,
                    "metadata_before": p.metadata_before,
                    "applied_source_block": p.applied_source_block,
                },
                separators=(",", ":"),
            )
        )
    return "\n".join(lines) + "\n"


def _parse_artifact(text: str, *, row_type: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse a header-then-rows JSONL artifact.

    Shared by `parse_proposals` and `parse_preimages`. Raises
    ValueError with a readable message on structural problems
    (missing/invalid header, a line that is not JSON, a row of the
    wrong type); the CLI surfaces the message and exits non-zero.
    Strictness is deliberate: these files authorize store writes,
    so a half-parsed file must never be silently half-applied.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("artifact is empty")
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as e:
        raise ValueError(f"header line is not valid JSON: {e}") from e
    if not isinstance(header, dict) or header.get("type") != "header":
        raise ValueError("first line is not a header object")
    rows: list[dict[str, Any]] = []
    for i, ln in enumerate(lines[1:], start=2):
        try:
            row = json.loads(ln)
        except json.JSONDecodeError as e:
            raise ValueError(f"line {i} is not valid JSON: {e}") from e
        if not isinstance(row, dict) or row.get("type") != row_type:
            raise ValueError(f"line {i} is not a {row_type} row")
        rows.append(row)
    return header, rows


def validate_header(header: dict[str, Any], *, user_id: str) -> str | None:
    """Check an artifact header against the CLI invocation.

    Returns an error message, or None when valid. The user check is
    the guard that makes a wrong-user apply/rollback fail loudly up
    front instead of dwindling into per-row ownership skips that
    read as a benign empty run.
    """
    run_id = header.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        return "artifact header has no run_id"
    header_user = header.get("user_id")
    if header_user != user_id:
        return f"artifact was generated for user {header_user!r}, not {user_id!r}"
    return None


def parse_proposals(text: str) -> tuple[dict[str, Any], list[Proposal]]:
    """Parse a proposals file into fully validated Proposal objects.

    Strict by design: a proposals file is a hand-editable
    authorization artifact, and apply mutates the store row by row.
    A row that only fails AT WRITE TIME would crash the run after
    earlier rows were already updated, so every field is validated
    here, before any store access. Raises ValueError naming the
    offending line; the caller aborts the whole apply.
    """
    header, raws = _parse_artifact(text, row_type="proposal")
    proposals: list[Proposal] = []
    for i, raw in enumerate(raws, start=2):
        mid = raw.get("memory_id")
        if not isinstance(mid, str) or not mid:
            raise ValueError(f"proposal line {i}: memory_id must be a non-empty string")
        chat_id = raw.get("chat_id")
        if not isinstance(chat_id, int):
            raise ValueError(f"proposal line {i}: chat_id must be an int")
        date = raw.get("date")
        if not isinstance(date, str) or not date:
            raise ValueError(f"proposal line {i}: date must be a non-empty string")
        user_ts = raw.get("user_ts")
        if not isinstance(user_ts, str) or not user_ts:
            raise ValueError(f"proposal line {i}: user_ts must be a non-empty string")
        sha = raw.get("user_text_sha256")
        if not isinstance(sha, str) or not sha:
            raise ValueError(f"proposal line {i}: user_text_sha256 must be a non-empty string")
        overlap = raw.get("overlap_score")
        if not isinstance(overlap, int | float) or not (0.0 <= float(overlap) <= 1.0):
            raise ValueError(f"proposal line {i}: overlap_score must be a number in [0.0, 1.0]")
        runner_up = raw.get("runner_up_overlap_score", 0.0)
        if not isinstance(runner_up, int | float) or not (0.0 <= float(runner_up) <= 1.0):
            raise ValueError(f"proposal line {i}: runner_up_overlap_score must be a number in [0.0, 1.0]")
        candidate_count = raw.get("candidate_count")
        if not isinstance(candidate_count, int) or candidate_count < 1:
            raise ValueError(f"proposal line {i}: candidate_count must be a positive int")
        gap_seconds = raw.get("gap_seconds")
        if not isinstance(gap_seconds, int | float):
            raise ValueError(f"proposal line {i}: gap_seconds must be a number")
        prior_fp = raw.get("prior_metadata_sha256")
        if not isinstance(prior_fp, str) or not prior_fp:
            raise ValueError(f"proposal line {i}: prior_metadata_sha256 must be a non-empty string")
        proposals.append(
            Proposal(
                memory_id=mid,
                chat_id=chat_id,
                date=date,
                user_ts=user_ts,
                user_text_sha256=sha,
                overlap_score=float(overlap),
                runner_up_overlap_score=float(runner_up),
                candidate_count=candidate_count,
                gap_seconds=float(gap_seconds),
                prior_metadata_sha256=prior_fp,
                row_text_snippet=str(raw.get("row_text_snippet", "")),
                candidate_text_snippet=str(raw.get("candidate_text_snippet", "")),
            )
        )
    return header, proposals


def parse_preimages(text: str) -> tuple[dict[str, Any], list[PreImage]]:
    """Parse a pre-image file into fully validated PreImage objects.

    Same strictness rationale as `parse_proposals`. Each row must
    carry the post-image `applied_source_block` (the four source_*
    keys); a missing block would silently disable rollback's
    operator-correction guard, so it is REQUIRED on every line.
    """
    header, raws = _parse_artifact(text, row_type="preimage")
    preimages: list[PreImage] = []
    for i, raw in enumerate(raws, start=2):
        mid = raw.get("memory_id")
        if not isinstance(mid, str) or not mid:
            raise ValueError(f"preimage line {i}: memory_id must be a non-empty string")
        pre_text = raw.get("text")
        if not isinstance(pre_text, str) or not pre_text:
            raise ValueError(f"preimage line {i}: text must be a non-empty string")
        metadata_before = raw.get("metadata_before")
        if not isinstance(metadata_before, dict):
            raise ValueError(f"preimage line {i}: metadata_before must be an object")
        applied = raw.get("applied_source_block")
        if not isinstance(applied, dict):
            raise ValueError(f"preimage line {i}: applied_source_block must be an object")
        # Required keys check: rollback's operator-correction guard
        # compares all four. A missing key would let a partial post-
        # image silently authorize a destructive restore.
        for key in (SOURCE_CHAT_ID_KEY, SOURCE_DATE_KEY, SOURCE_USER_TS_KEY, SOURCE_USER_TEXT_SHA256_KEY):
            if key not in applied:
                raise ValueError(f"preimage line {i}: applied_source_block missing required key {key!r}")
        preimages.append(
            PreImage(
                memory_id=mid,
                text=pre_text,
                metadata_before=metadata_before,
                applied_source_block=applied,
            )
        )
    return header, preimages


# ── Pure helpers: report rendering ──────────────────────────────────


_SNIPPET_LEN = 120


def _snippet(text: str) -> str:
    """Truncate text to `_SNIPPET_LEN` chars with an ellipsis when cut.

    Used for the proposals JSONL audit fields and the report sample
    sections so an operator scanning the file does not have to open
    multi-paragraph rows to recognize them. Newlines collapse to
    spaces because the proposals file is line-oriented and a row
    text with literal newlines would otherwise break the JSONL.
    """
    flat = " ".join(text.split())
    if len(flat) <= _SNIPPET_LEN:
        return flat
    return flat[: _SNIPPET_LEN - 3] + "..."


def render_report(
    *,
    run_id: str,
    user_id: str,
    window_seconds: int,
    min_overlap: float,
    strong_overlap_ratio: float,
    shingle_n: int,
    scanned: int,
    selected: int,
    matches: list[RowMatch],
    skips: dict[str, list[str]],
    sample_size: int,
) -> str:
    """Build the human-readable dry-run report.

    Pure function (no IO). The report has four sections: parameters,
    counts, a sample of STRONG_MATCH proposals (so the operator can
    eyeball auto-matches), and a curation backlog (AMBIGUOUS_OVERLAP
    plus NO_CANDIDATE rows with their top candidates). Order is
    deterministic so two runs over the same data render byte-
    identically.
    """
    proposals = [m for m in matches if m.bucket == "STRONG_MATCH"]
    ambiguous = [m for m in matches if m.bucket == SKIP_AMBIGUOUS_OVERLAP]
    no_candidate = [m for m in matches if m.bucket == SKIP_NO_CANDIDATE]
    history_unreadable = [m for m in matches if m.bucket == SKIP_HISTORY_UNREADABLE]

    lines = [
        f"# Backfill provenance dry-run report: {run_id}",
        "",
        f"User: {user_id}",
        f"Window: {window_seconds}s ({window_seconds / 3600:.1f}h)",
        f"Min overlap: {min_overlap}",
        f"Strong overlap ratio: {strong_overlap_ratio}",
        f"Shingle n: {shingle_n}",
        "",
        "## Counts",
        "",
        f"- scanned: {scanned}",
        f"- selected: {selected}",
        f"- proposals (STRONG_MATCH): {len(proposals)}",
        f"- skipped ambiguous_overlap: {len(ambiguous)}",
        f"- skipped no_candidate: {len(no_candidate)}",
        f"- skipped history_unreadable: {len(history_unreadable)}",
    ]
    for reason, ids in sorted(skips.items()):
        if reason in {SKIP_AMBIGUOUS_OVERLAP, SKIP_NO_CANDIDATE, SKIP_HISTORY_UNREADABLE}:
            continue
        lines.append(f"- skipped {reason}: {len(ids)}")

    if proposals and sample_size > 0:
        lines.extend(["", "## Sample of STRONG_MATCH proposals"])
        for m in proposals[:sample_size]:
            assert m.proposal is not None
            lines.extend(_render_row_sample(m))

    if (ambiguous or no_candidate) and sample_size > 0:
        lines.extend(["", "## Curation backlog (operator review)"])
        for m in (ambiguous + no_candidate)[:sample_size]:
            lines.extend(_render_row_sample(m))

    return "\n".join(lines) + "\n"


def _render_row_sample(match: RowMatch) -> list[str]:
    """Render one RowMatch as a markdown sub-section."""
    out = [
        "",
        f"### {match.row.id} ({match.bucket})",
        "",
        f"Row: {_snippet(match.row.text)}",
    ]
    if not match.top_candidates:
        out.append("Top candidates: (none in window)")
        return out
    out.append("Top candidates:")
    for c in match.top_candidates:
        out.append(f"- ts={c.ts} gap={c.gap_seconds:.0f}s overlap={c.overlap_score:.3f} text={_snippet(c.text)}")
    return out


# ── Driver: dry-run ─────────────────────────────────────────────────


async def run_dry_run(
    config: Config,
    user_id: str,
    *,
    window_seconds: int,
    min_overlap: float,
    strong_overlap_ratio: float,
    shingle_n: int,
    sample: int,
    out_dir: Path,
) -> int:
    """Score every backfill candidate and write report + proposals.

    Writes nothing to the store. Returns a process exit code: 0 on a
    completed pass (even an all-skip one; the report is the
    product), non-zero only on init / IO failure (init is the
    caller's responsibility, so this driver returns 1 only on the
    artifact write).
    """
    rows = memory.get_all(user_id=user_id, limit=None)
    selected_rows = select_rows(rows)
    run_id = datetime.now(UTC).strftime("bp-%Y%m%d-%H%M%S")

    chat_id_int: int
    try:
        chat_id_int = int(user_id)
    except ValueError:
        print(f"memory admin: user_id {user_id!r} is not an integer chat id; cannot read JSONL.")
        return 1

    matches: list[RowMatch] = []
    skips: dict[str, list[str]] = {}
    for row in selected_rows:
        try:
            created_at_dt = datetime.fromisoformat(row.created_at)
        except ValueError:
            # A Mem0 row without a parseable created_at cannot be
            # window-located. This is exceptional (Mem0 stamps the
            # field at insert) and the row stays unmatched.
            skips.setdefault(SKIP_NO_CANDIDATE, []).append(row.id)
            matches.append(RowMatch(row=row, bucket=SKIP_NO_CANDIDATE, top_candidates=[], proposal=None))
            continue
        if created_at_dt.tzinfo is None:
            created_at_dt = created_at_dt.replace(tzinfo=UTC)

        records, any_unreadable = _candidates_for_row(
            chat_id_int,
            created_at_dt,
            window_seconds=window_seconds,
        )
        if any_unreadable:
            skips.setdefault(SKIP_HISTORY_UNREADABLE, []).append(row.id)
            matches.append(RowMatch(row=row, bucket=SKIP_HISTORY_UNREADABLE, top_candidates=[], proposal=None))
            continue

        scored = _score_candidates(
            row.text,
            records,
            shingle_n=shingle_n,
            created_at_dt=created_at_dt,
        )
        top = sorted(scored, key=lambda c: c.overlap_score, reverse=True)[:_CURATION_REPORT_TOP_N]
        winner, bucket, runner_up_score = _gate_match(
            scored,
            min_overlap=min_overlap,
            strong_overlap_ratio=strong_overlap_ratio,
        )
        if winner is None:
            skips.setdefault(bucket, []).append(row.id)
            matches.append(RowMatch(row=row, bucket=bucket, top_candidates=top, proposal=None))
            continue

        # STRONG_MATCH. Build the proposal with its full audit context.
        try:
            user_ts_dt = datetime.fromisoformat(winner.ts)
        except ValueError:
            # Defensive: _score_candidates already filtered malformed
            # timestamps, so this is unreachable in practice. Drop the
            # row to NO_CANDIDATE if it somehow happens; do not write
            # a proposal whose date we cannot derive.
            skips.setdefault(SKIP_NO_CANDIDATE, []).append(row.id)
            matches.append(RowMatch(row=row, bucket=SKIP_NO_CANDIDATE, top_candidates=top, proposal=None))
            continue
        if user_ts_dt.tzinfo is None:
            user_ts_dt = user_ts_dt.replace(tzinfo=UTC)
        date = user_ts_dt.strftime("%Y-%m-%d")
        proposal = Proposal(
            memory_id=row.id,
            chat_id=chat_id_int,
            date=date,
            user_ts=winner.ts,
            user_text_sha256=winner.sha256,
            overlap_score=winner.overlap_score,
            runner_up_overlap_score=runner_up_score,
            candidate_count=len(scored),
            gap_seconds=winner.gap_seconds,
            prior_metadata_sha256=_metadata_fingerprint(row.metadata),
            row_text_snippet=_snippet(row.text),
            candidate_text_snippet=_snippet(winner.text),
        )
        matches.append(RowMatch(row=row, bucket="STRONG_MATCH", top_candidates=top, proposal=proposal))

    proposals = [m.proposal for m in matches if m.proposal is not None]

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"memory admin: cannot create output directory: {e}")
        return 1
    header = {
        "run_id": run_id,
        "user_id": user_id,
        "window_seconds": window_seconds,
        "min_overlap": min_overlap,
        "strong_overlap_ratio": strong_overlap_ratio,
        "shingle_n": shingle_n,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    proposals_path = out_dir / f"backfill-{run_id}-proposals.jsonl"
    proposals_path.write_text(render_proposals(header, proposals), encoding="utf-8")
    report_path = out_dir / f"backfill-{run_id}-report.md"
    report_path.write_text(
        render_report(
            run_id=run_id,
            user_id=user_id,
            window_seconds=window_seconds,
            min_overlap=min_overlap,
            strong_overlap_ratio=strong_overlap_ratio,
            shingle_n=shingle_n,
            scanned=len(rows),
            selected=len(selected_rows),
            matches=matches,
            skips=skips,
            sample_size=sample,
        ),
        encoding="utf-8",
    )

    skip_summary = ", ".join(f"{len(ids)} {reason}" for reason, ids in sorted(skips.items())) or "none"
    print(
        f"memory admin: dry-run {run_id}: scanned {len(rows)}, selected {len(selected_rows)}, "
        f"proposals {len(proposals)}, skipped {skip_summary}."
    )
    print(f"memory admin: proposals: {proposals_path}")
    print(f"memory admin: report: {report_path}")
    return 0


# ── Driver: apply ───────────────────────────────────────────────────


def _build_applied_source_block(proposal: Proposal) -> dict[str, Any]:
    """Pack a proposal into the four-key source block apply will
    merge into the row's metadata."""
    return {
        SOURCE_CHAT_ID_KEY: proposal.chat_id,
        SOURCE_DATE_KEY: proposal.date,
        SOURCE_USER_TS_KEY: proposal.user_ts,
        SOURCE_USER_TEXT_SHA256_KEY: proposal.user_text_sha256,
    }


def _verify_jsonl_user_text(
    chat_id: int,
    date: str,
    user_ts: str,
    expected_sha256: str,
) -> bool:
    """Re-fetch the user line from JSONL at apply time, recompute sha.

    Returns True when the line exists AND its sha256 matches the
    expected fingerprint. False when the file is gone, the line is
    gone, or the text has drifted. The apply-time transcript-drift
    gate is the only safety check the harness has against a JSONL
    edit between dry-run and apply; without it, a proposal stamped
    against a since-modified turn would write an invalid pointer.
    """
    path = _LOG_DIR / str(chat_id) / f"{date}.jsonl"
    records = _read_jsonl_records(path)
    if not records:
        return False
    for rec in records:
        if rec.get("dir") != "user":
            continue
        if rec.get("ts") != user_ts:
            continue
        text = rec.get("text", "")
        if not isinstance(text, str):
            return False
        return hashlib.sha256(text.encode("utf-8")).hexdigest() == expected_sha256
    return False


async def run_apply(
    config: Config,
    user_id: str,
    *,
    proposals_path: Path,
    out_dir: Path,
) -> int:
    """Apply a reviewed proposals file with per-row re-checks.

    Never re-scores: the reviewed file IS the change set. Every
    proposal is schema-validated up front (a hand-edited row must
    abort the run BEFORE any write, not crash it midway). Pre-images
    are dumped (and fsynced) before the first store write; a dump
    failure aborts with zero changes, and an existing pre-image file
    is never truncated.
    """
    try:
        header, proposals = parse_proposals(proposals_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"memory admin: cannot read proposals: {e}")
        return 1
    error = validate_header(header, user_id=user_id)
    if error is not None:
        print(f"memory admin: {error}")
        return 1
    run_id: str = header["run_id"]

    # Per-proposal chat-id guard. The artifact-level user_id check is
    # not sufficient: a hand-edited or forged proposal row whose
    # chat_id points at a different user would otherwise pass parsing,
    # update the CLI user's memory row via get_by_id(user_id=user_id,
    # ...), verify the JSONL under the other chat, and then stamp the
    # CLI user's row with a source_chat_id that does not match. The
    # row would be durably marked present, but its source button would
    # always fail closed for the actual user via the expected_chat_id
    # ownership gate. Reject the whole apply if any proposal carries a
    # chat_id that does not match the CLI user.
    try:
        user_id_int = int(user_id)
    except ValueError:
        print(f"memory admin: user_id {user_id!r} is not an integer chat id; cannot apply.")
        return 1
    mismatched = [p.memory_id for p in proposals if p.chat_id != user_id_int]
    if mismatched:
        print(
            f"memory admin: {len(mismatched)} proposal(s) carry chat_id != user_id {user_id_int}; "
            "refusing to apply. First few: " + ", ".join(mismatched[:5]) + "."
        )
        return 1

    # Re-check phase: every proposal is verified against the live
    # store AND the live JSONL as they are NOW. The survivors carry
    # their fresh row so the pre-image dump and the write use the
    # same fetched state.
    survivors: list[tuple[Proposal, MemoryResult]] = []
    skips: dict[str, list[str]] = {}
    for proposal in proposals:
        row = memory.get_by_id(user_id=user_id, memory_id=proposal.memory_id)
        if row is None:
            skips.setdefault(SKIP_ROW_GONE, []).append(proposal.memory_id)
            continue
        # Already-present provenance is the strongest deselect signal:
        # some other path populated source_* between dry-run and apply
        # (a future re-extraction, a manual operator stamp). Backfill
        # never overwrites a present-True row.
        if read_transcript_provenance(row.metadata).present:
            skips.setdefault(SKIP_DESELECTED, []).append(proposal.memory_id)
            continue
        if _metadata_fingerprint(row.metadata) != proposal.prior_metadata_sha256:
            skips.setdefault(SKIP_METADATA_DRIFT, []).append(proposal.memory_id)
            continue
        if not _verify_jsonl_user_text(
            chat_id=proposal.chat_id,
            date=proposal.date,
            user_ts=proposal.user_ts,
            expected_sha256=proposal.user_text_sha256,
        ):
            skips.setdefault(SKIP_TRANSCRIPT_DRIFT, []).append(proposal.memory_id)
            continue
        survivors.append((proposal, row))

    if not survivors:
        skip_summary = ", ".join(f"{len(ids)} {reason}" for reason, ids in sorted(skips.items())) or "none"
        print(f"memory admin: apply {run_id}: nothing to apply; skipped: {skip_summary}.")
        return 0

    # Pre-image dump before any write; abort on failure. Exclusive
    # creation ("x") refuses to truncate an existing rollback file
    # (the path derives from the run id, so a re-run of the same
    # apply would otherwise silently overwrite the only copy of the
    # original rows). fsync so a crash mid-apply cannot leave changed
    # rows with rollback material trapped in the page cache.
    preimage_path = out_dir / f"backfill-{run_id}-preimages.jsonl"
    preimage_header = {
        "run_id": run_id,
        "user_id": user_id,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "proposal_file": proposals_path.name,
    }
    preimages = [
        PreImage(
            memory_id=row.id,
            text=row.text,
            metadata_before=dict(row.metadata or {}),
            applied_source_block=_build_applied_source_block(proposal),
        )
        for proposal, row in survivors
    ]
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(preimage_path, "x", encoding="utf-8") as f:
            f.write(render_preimages(preimage_header, preimages))
            f.flush()
            os.fsync(f.fileno())
    except FileExistsError:
        print(
            f"memory admin: pre-image file already exists for run {run_id}: {preimage_path}\n"
            "memory admin: refusing to overwrite rollback material; move it aside first "
            "(or use it with --rollback)."
        )
        return 1
    except OSError as e:
        print(f"memory admin: pre-image dump failed, aborting with no changes: {e}")
        return 1

    applied = 0
    failed = 0
    for proposal, row in survivors:
        applied_block = _build_applied_source_block(proposal)
        merged = dict(row.metadata or {})
        merged.update(applied_block)
        # update_metadata REPLACES the metadata dict wholesale; the
        # merge above preserves every existing key. Mem0 internally
        # auto-preserves a small handful of fields (data, created_at,
        # user_id, ...); see the wrapper docstring for the full list.
        ok = memory.update_metadata(user_id=user_id, memory_id=row.id, data=row.text, metadata=merged)
        if ok:
            applied += 1
            log.info(
                "%s %s",
                _PROVENANCE_BACKFILL_EVENT,
                json.dumps(
                    {
                        "memory_id": row.id,
                        "user_id": user_id,
                        "run_id": run_id,
                        "overlap_score": proposal.overlap_score,
                        "gap_seconds": proposal.gap_seconds,
                        "source_user_ts": proposal.user_ts,
                    },
                    separators=(",", ":"),
                ),
            )
            # Verify the round trip end to end: the new metadata must
            # resolve to a typed TranscriptProvenance whose `present`
            # is True, and fetch_transcript_context with the chat-id
            # ownership gate must return `ok` against the live JSONL.
            # Any non-ok reason is a warning, not a rollback: the row
            # was stamped; surfacing the failure preserves the audit
            # trail without trying to repair what we just wrote.
            provenance = read_transcript_provenance(merged)
            lookup = fetch_transcript_context(
                provenance,
                memory_id=row.id,
                expected_chat_id=proposal.chat_id,
            )
            if lookup.reason != "ok":
                log.warning(
                    "memory.provenance.backfill verification miss: row=%s reason=%s",
                    row.id,
                    lookup.reason,
                )
        else:
            failed += 1

    skip_summary = ", ".join(f"{len(ids)} {reason}" for reason, ids in sorted(skips.items())) or "none"
    print(f"memory admin: apply {run_id}: applied {applied}, failed {failed}, skipped: {skip_summary}.")
    print(f"memory admin: pre-images: {preimage_path}")
    if survivors and applied == 0:
        return 1
    return 0


# ── Driver: rollback ────────────────────────────────────────────────


async def run_rollback(
    config: Config,
    user_id: str,
    *,
    preimages_path: Path,
) -> int:
    """Restore rows from a pre-image file.

    Restores text and metadata from the pre-image. Rows whose current
    `source_*` block differs from the applied block are skipped: a
    later operator correction to any of the four keys means rollback
    would silently undo intentional work. The comparison reads the
    raw metadata keys directly (not via `read_transcript_provenance`,
    which returns a typed interpretation with coercion semantics).
    """
    try:
        header, preimages = parse_preimages(preimages_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"memory admin: cannot read pre-images: {e}")
        return 1
    error = validate_header(header, user_id=user_id)
    if error is not None:
        print(f"memory admin: {error}")
        return 1
    run_id: str = header["run_id"]

    # Per-preimage chat-id guard, symmetric with apply's. A pre-image
    # file whose applied_source_block carries a chat_id that does not
    # match the CLI user could only have come from a hand edit or a
    # forged artifact; refusing to restore in that case keeps the
    # authorization model honest at both ends of the round trip.
    try:
        user_id_int = int(user_id)
    except ValueError:
        print(f"memory admin: user_id {user_id!r} is not an integer chat id; cannot rollback.")
        return 1
    mismatched = [p.memory_id for p in preimages if p.applied_source_block.get(SOURCE_CHAT_ID_KEY) != user_id_int]
    if mismatched:
        print(
            f"memory admin: {len(mismatched)} preimage(s) carry applied chat_id != user_id "
            f"{user_id_int}; refusing to rollback. First few: " + ", ".join(mismatched[:5]) + "."
        )
        return 1

    restored = 0
    failed = 0
    skips: dict[str, list[str]] = {}
    for preimage in preimages:
        row = memory.get_by_id(user_id=user_id, memory_id=preimage.memory_id)
        if row is None:
            skips.setdefault(SKIP_ROW_GONE, []).append(preimage.memory_id)
            continue
        # Raw-key comparison: pull each source_* value from the row's
        # current metadata verbatim and compare against the recorded
        # applied block. Any divergence on any of the four keys means
        # another writer touched source_* since backfill applied; the
        # comparison must not depend on the resolver's typed coercions
        # (which could mask, e.g., a chat_id change from int 5 to
        # string "5" as "still the same value").
        current_md = row.metadata or {}
        current_block = {
            SOURCE_CHAT_ID_KEY: current_md.get(SOURCE_CHAT_ID_KEY),
            SOURCE_DATE_KEY: current_md.get(SOURCE_DATE_KEY),
            SOURCE_USER_TS_KEY: current_md.get(SOURCE_USER_TS_KEY),
            SOURCE_USER_TEXT_SHA256_KEY: current_md.get(SOURCE_USER_TEXT_SHA256_KEY),
        }
        applied_block = {
            SOURCE_CHAT_ID_KEY: preimage.applied_source_block.get(SOURCE_CHAT_ID_KEY),
            SOURCE_DATE_KEY: preimage.applied_source_block.get(SOURCE_DATE_KEY),
            SOURCE_USER_TS_KEY: preimage.applied_source_block.get(SOURCE_USER_TS_KEY),
            SOURCE_USER_TEXT_SHA256_KEY: preimage.applied_source_block.get(SOURCE_USER_TEXT_SHA256_KEY),
        }
        if current_block != applied_block:
            skips.setdefault(SKIP_OPERATOR_CORRECTION, []).append(preimage.memory_id)
            continue
        ok = memory.update_metadata(
            user_id=user_id,
            memory_id=preimage.memory_id,
            data=preimage.text,
            metadata=preimage.metadata_before,
        )
        if ok:
            restored += 1
        else:
            failed += 1

    skip_summary = ", ".join(f"{len(ids)} {reason}" for reason, ids in sorted(skips.items())) or "none"
    print(f"memory admin: rollback {run_id}: restored {restored}, failed {failed}, skipped: {skip_summary}.")
    # Exit-code policy mirrors reclassify: exit 1 ONLY when rollback
    # attempted writes and none landed. Attempted = restored + failed
    # (skips are not attempts). An all-skip run is a valid outcome
    # (every preimage's source block drifted); an attempt-with-all-
    # failures run is not, regardless of whether other preimages were
    # skipped on the same invocation.
    attempted = restored + failed
    if attempted and restored == 0:
        return 1
    return 0
