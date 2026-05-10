"""
Layer 1 retrieval evaluation harness.

Reachable as `python -m kai.eval.retrieval`.

Given a probe set of `(question, expected_fact_id)` pairs, runs each
probe through the live memory pipeline, parses the resulting
`memory.recall` log line emitted by `format_context`, and emits
precision@K, recall@K, mean reciprocal rank, and the fraction of
probes whose expected fact actually reached the injected prompt
(`fraction_in_prompt`). Optionally sweeps a parameter grid and
reports the Pareto frontier of (precision, latency).

Design rationale (the part that is not obvious from the code):

- The harness scores against `format_context`, not `search` directly.
  `format_context` is what production runs; scoring a reduced pipeline
  (search without floor filtering or budget walking) would measure a
  code path the agent never uses. `format_context` is already
  instrumented with a structured `memory.recall` log line, so the
  harness reads the log rather than wrapping retrieval. This doubles
  as a regression test: if the log emit ever stops being parseable,
  the harness fails loudly.

- Probes whose expected fact has been deleted from the store between
  authoring and evaluation are bucketed as "probe-set drift" rather
  than counted as retrieval misses. Mixing the two would conflate
  "retrieval failed" with "probe set went stale" - different operator
  actions (tune retrieval vs. refresh probes). Drift count is reported
  separately. All four metric denominators are computed against
  N_scored = N_probes - N_drift; drift probes are excluded from both
  numerator and denominator.

- The sweep mutates module-level state in `kai.memory`
  (`_SPEAKER_WEIGHTS`, `_SEARCH_OVERFETCH`) and replaces `_config` with
  a `dataclasses.replace(...)` clone for each grid point. Restoration
  in `try/finally` defends against a probe raising mid-sweep. The
  process is short-lived and has no other consumers of the memory
  module, so in-place mutation is safe.

PII posture: probe questions and expected_fact_id values are operator
data and remain in the gitignored probes file. The probe schema is
documented inline in `load_probes` below; no example fixtures ship
with the repo.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kai.config import Config

log = logging.getLogger(__name__)


# ── Constants ───────────────────────────────────────────────────────


# K values reported for precision / recall. Includes the per-probe
# `lines_used` (resolved at eval time, varies per probe) which is
# computed separately rather than pre-listed here. The fixed K values
# are the ones the eval table renders columns for: 1 (the strictest
# top-pick metric), 3 (typical mid-budget), and 5 (the default Pareto
# axis - top-K large enough to absorb tie-break noise, small enough
# to match typical budget walks).
_K_VALUES: tuple[int, ...] = (1, 3, 5)


# Default sweep grid. Three speaker-weight axes plus floor and
# overfetch. The full Cartesian product is 6 * 2 * 4 * 3 * 3 = 432
# configurations; the calibration plan in spec §5 holds floor and
# overfetch at production values, leaving 24 configurations for the
# typical sweep run. Operators can narrow each axis via CLI flags.
_DEFAULT_FLOOR_GRID: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
_DEFAULT_USER_WEIGHT_GRID: tuple[float, ...] = (0.85, 1.0)
_DEFAULT_ASSISTANT_WEIGHT_GRID: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8)
_DEFAULT_EPISODE_SUMMARY_WEIGHT_GRID: tuple[float, ...] = (0.7, 0.85, 1.0)
_DEFAULT_OVERFETCH_GRID: tuple[int, ...] = (10, 20, 30)


# Production defaults that the harness treats as the baseline
# configuration. Used both when the operator runs a single eval
# (no --sweep) without overrides and when computing the schema
# example baseline. The three speaker weights map to the entries
# in `_SPEAKER_WEIGHTS`; the legacy/extracted source-weight pair
# the older table carried no longer exists.
_PRODUCTION_FLOOR = 0.30
_PRODUCTION_USER_WEIGHT = 0.85
_PRODUCTION_ASSISTANT_WEIGHT = 0.8
_PRODUCTION_EPISODE_SUMMARY_WEIGHT = 0.85
_PRODUCTION_OVERFETCH = 20


# Greppable prefix on every memory.recall log line. Mirrored in
# `kai.memory._emit_recall_log`; if either ever changes, the parser
# in `_RecallLogCapture` must move with it. The trailing space is
# part of the prefix so empty-payload edge cases still tokenize.
_RECALL_PREFIX = "memory.recall "


# Schema version written into baseline JSON output. Bump when the
# baseline file shape changes in a way that operators reading older
# files would misinterpret. Drift detection in the harness compares
# probe-set hashes, not schema versions, so this number is purely
# informational for humans.
_BASELINE_SCHEMA_VERSION = 1


# ── Data classes ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Probe:
    """A single eval probe.

    `question` is the text fed to format_context; `expected_fact_id`
    is the Mem0 row ID the question should have surfaced. The
    optional fields (`source_turn_ts`, `notes`) are operator
    bookkeeping - they let an author trace a failing probe back to
    the conversation it was drawn from but do not feed scoring.
    """

    question: str
    expected_fact_id: str
    source_turn_ts: str = ""
    notes: str = ""


@dataclass
class ConfigOverride:
    """One point in the sweep grid.

    Each override flows into the harness's sweep loop and is the
    only mutable state per iteration: floor (replaces
    `_config.memory_search_floor`), three speaker weights (replace
    the matching entries in `_SPEAKER_WEIGHTS`), and overfetch
    (replaces `_SEARCH_OVERFETCH` on the module). The unknown-
    speaker fallback weight is intentionally not part of the grid:
    its job is to keep an unrecognized speaker class rankable, not
    to be tuned.
    """

    floor: float
    user_weight: float
    assistant_weight: float
    episode_summary_weight: float
    overfetch: int


@dataclass(frozen=True)
class _Snapshot:
    """Frozen snapshot of mutable kai.memory state captured before an
    override is applied. Used by `_restore_overrides` to put the
    module back in its pre-apply shape regardless of how the caller's
    loop exits.

    Stored as a frozen dataclass so the snapshot itself cannot be
    accidentally mutated between capture and restore (the dict
    contents are copied at capture time so future mutation of
    `_SPEAKER_WEIGHTS` does not leak through the saved reference).
    """

    speaker_weights: dict[str, float]
    overfetch: int
    config: Config


@dataclass
class ProbeResult:
    """Per-probe scoring outcome.

    `rank` is 1-indexed position of expected_fact_id in the recall
    payload's hits, or None if the expected fact was not retrieved
    at all. `in_prompt` is True iff the fact reached the slice of
    hits the agent actually saw (rank <= lines_used). `tags` carries
    the expected fact's tags for per-tag breakdowns; collected once
    during drift detection rather than re-fetched at scoring time.
    `latency_ms` mirrors the value in the recall payload, surfaced
    here so per-probe latencies can be aggregated without re-parsing.
    """

    probe: Probe
    rank: int | None
    in_prompt: bool
    lines_used: int
    latency_ms: int
    tags: tuple[str, ...]


@dataclass
class Metrics:
    """Aggregated metrics for one configuration over a probe set.

    Stored as plain dicts so JSON serialization is direct. Per-tag
    metrics are computed by re-running the same scoring math over
    each tag's subset of probes; `by_tag` is a dict of tag -> the
    same shape as the top-level metrics dict (minus `by_tag` itself).
    """

    n_probes: int
    n_scored: int
    n_drift: int
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    mrr: float
    fraction_in_prompt: float
    latency_p50_ms: float
    latency_p95_ms: float
    by_tag: dict[str, dict[str, Any]] = field(default_factory=dict)


# ── Probe file loading ─────────────────────────────────────────────


def load_probes(path: Path) -> list[Probe]:
    """Load and validate probes from a JSONL file.

    Documented format extension: lines that, after lstrip, begin with
    `#` are treated as comments and skipped. This lets operators
    annotate a probe file in-place without an external metadata file.
    Empty/whitespace-only lines are also skipped. Every other line is
    parsed as JSON and must carry `question` (non-empty string) and
    `expected_fact_id` (non-empty string); `source_turn_ts` and
    `notes` are optional.

    Validation errors include the file path and 1-indexed line number
    so an operator with a multi-hundred-line probe set can find the
    bad row immediately. Line order is preserved in the returned list
    so per-probe output rows align with the probe file.
    """
    probes: list[Probe] = []
    raw = path.read_text(encoding="utf-8").splitlines()
    for lineno, line in enumerate(raw, start=1):
        stripped = line.lstrip()
        # Skip comment and blank lines BEFORE attempting JSON parse.
        # Putting the comment check on `lstrip()` (not the raw line)
        # accepts indented annotations like `    # category notes`,
        # which operators may use when grouping probes visually.
        if not stripped or stripped.startswith("#"):
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno}: malformed JSON ({e.msg})") from e
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{lineno}: expected JSON object, got {type(obj).__name__}")
        # Required fields. Type check both presence and string-ness so
        # an int slip-through (e.g. `expected_fact_id: 42` from a
        # spreadsheet export) fails at load time rather than on the
        # first hit comparison where the type mismatch would silently
        # prevent every match.
        question = obj.get("question")
        expected = obj.get("expected_fact_id")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{path}:{lineno}: 'question' missing or not a non-empty string")
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError(f"{path}:{lineno}: 'expected_fact_id' missing or not a non-empty string")
        source_turn_ts = obj.get("source_turn_ts", "")
        notes = obj.get("notes", "")
        if not isinstance(source_turn_ts, str):
            raise ValueError(f"{path}:{lineno}: 'source_turn_ts' must be a string when present")
        if not isinstance(notes, str):
            raise ValueError(f"{path}:{lineno}: 'notes' must be a string when present")
        probes.append(
            Probe(
                question=question,
                expected_fact_id=expected,
                source_turn_ts=source_turn_ts,
                notes=notes,
            )
        )
    return probes


def probe_set_hash(probes: list[Probe]) -> str:
    """SHA-256 of the sorted (question, expected_fact_id) list.

    Locks a baseline measurement to its probe set: comparing two
    baseline files with different hashes is meaningless even if the
    metric numbers look similar. Sorted to make the hash invariant
    under reordering of the probe file. Source turn timestamps and
    notes are excluded - they are bookkeeping, not scoring inputs.
    """
    pairs = sorted((p.question, p.expected_fact_id) for p in probes)
    blob = json.dumps(pairs, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


# ── Log capture ─────────────────────────────────────────────────────


class _RecallLogCapture(logging.Handler):
    """Buffer memory.recall log records for later draining.

    Attached to the `kai.memory` logger so it sees every record the
    module emits. Filters down to records whose message starts with
    the `memory.recall ` prefix so unrelated info-level logs (init,
    delete, etc.) are ignored. The handler keeps full records, not
    just the parsed payloads, so per-record diagnostics (logger name,
    timestamp) remain accessible if a future debug hook needs them.

    Carries `_saved_level` so the attach helper can restore the
    `kai.memory` logger's effective level on detach. The harness
    forces the level to INFO at attach time because operators may
    run with LOG_LEVEL=WARNING in production - without the bump,
    `memory.recall` records would never reach any handler and the
    harness would loudly fail with "expected one record, got zero."
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self._records: list[logging.LogRecord] = []
        self._saved_level: int | None = None

    def emit(self, record: logging.LogRecord) -> None:
        # `getMessage()` is the documented stable API; `.message` is
        # set as a side effect of Formatter.format() and is not
        # guaranteed to be populated for every handler. Test #4 round-
        # trips against this exact path.
        if record.getMessage().startswith(_RECALL_PREFIX):
            self._records.append(record)

    def drain(self) -> list[dict[str, Any]]:
        """Return parsed payloads for every buffered record and clear.

        Strips the `memory.recall ` prefix and decodes the remainder
        as JSON. The harness expects exactly one record per call to
        `format_context`; the caller (`_run_one_probe`) asserts on
        the count and aborts loudly on zero or more-than-one - the
        log-shape contract is the harness's only signal that the
        retrieval path ran as expected.
        """
        parsed: list[dict[str, Any]] = []
        for r in self._records:
            blob = r.getMessage()[len(_RECALL_PREFIX) :]
            try:
                payload = json.loads(blob)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"unparseable memory.recall payload: {e.msg}") from e
            parsed.append(payload)
        self._records.clear()
        return parsed


# ── Scoring math ────────────────────────────────────────────────────


def compute_rank(hits: list[dict[str, Any]], expected_fact_id: str) -> int | None:
    """Return the 1-indexed position of expected_fact_id in hits, or None.

    Relies on the per-hit `id` field in the recall payload (carried
    from MemoryResult.id by format_context). Without it, this match
    would have to fall back to fragile snippet-substring comparison
    against the 80-char truncated text.
    """
    for i, h in enumerate(hits):
        if h.get("id") == expected_fact_id:
            return i + 1
    return None


def score_probes(results: list[ProbeResult]) -> dict[str, Any]:
    """Compute aggregate metrics over a list of ProbeResult.

    Pure function: takes scored per-probe outcomes (rank, lines_used,
    in_prompt) and returns the metrics dict. The harness's drift-
    detection step filters drifted probes out before they reach this
    function; the type signature does NOT enforce that contract (any
    list[ProbeResult] type-checks), so the caller is responsible.

    For the single-answer case (each probe has exactly one expected
    fact), recall@K equals precision@K. They are reported separately
    so a future multi-answer probe format can diverge without
    reshaping the output. fraction_in_prompt is the per-probe variant:
    each probe is checked against its own lines_used (varies with
    budget exhaustion), whereas precision@K uses a fixed K.
    """
    n = len(results)
    if n == 0:
        # Empty input is ambiguous: it can mean "all probes drifted"
        # (caller's responsibility to surface separately) or "no
        # probes loaded." Return zeroes rather than raising so the
        # caller can distinguish via n_scored=0 in the wrapping
        # Metrics object.
        return {
            "precision_at_k": {k: 0.0 for k in _K_VALUES},
            "recall_at_k": {k: 0.0 for k in _K_VALUES},
            "mrr": 0.0,
            "fraction_in_prompt": 0.0,
        }

    precision_at_k: dict[int, float] = {}
    recall_at_k: dict[int, float] = {}
    for k in _K_VALUES:
        # A hit "counts at K" iff it was retrieved (rank not None) AND
        # the rank falls within the top K. Combining the two guards
        # in a single comprehension keeps the math obviously correct
        # at a glance and avoids a None-check followed by a comparison
        # that would crash on a missed probe.
        hits_in_top_k = sum(1 for r in results if r.rank is not None and r.rank <= k)
        precision_at_k[k] = hits_in_top_k / n
        recall_at_k[k] = hits_in_top_k / n

    # fraction_in_prompt: per-probe variant where K = each probe's
    # actual lines_used (the slice the agent saw). Computed from
    # ProbeResult.in_prompt which was set in _run_one_probe as
    # `rank is not None and rank <= lines_used`. A probe whose
    # lines_used was zero (header-only output, possible if budget <
    # header tokens) cannot have an in-prompt hit, which falls out
    # naturally from the rank <= 0 check at probe-execution time.
    in_prompt_count = sum(1 for r in results if r.in_prompt)

    # MRR: mean of 1/rank, treating misses as 0. The miss-as-zero
    # convention is standard in IR literature; explicit so a reader
    # does not assume "skip misses" semantics that would inflate the
    # number on poor-coverage probe sets.
    mrr_sum = sum(1.0 / r.rank if r.rank else 0.0 for r in results)

    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "mrr": mrr_sum / n,
        "fraction_in_prompt": in_prompt_count / n,
    }


def _percentile(values: list[float], pct: int) -> float:
    """Return the requested percentile from values (1-99 integer scale).

    `pct` is `int` not `float` so a fractional percentile (e.g. 50.9)
    cannot be silently truncated to the 50th by `int(pct) - 1`. The
    only callers today are `_percentile(latencies, 50)` and `(_, 95)`,
    so narrowing the contract costs nothing and removes the footgun.

    Uses statistics.quantiles for the standard linear-interpolation
    estimate. Returns 0.0 on empty input rather than raising so the
    caller can render zero-data tables without a special case.
    statistics.quantiles requires at least n=2 points; the singleton
    case returns the lone value to avoid the library exception.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    # n=100 partitions gives 99 cut points (quantiles returns n-1
    # cuts for n partitions), so index `pct - 1` is the requested
    # 1-indexed percentile. clamp() defends against pct == 0 or
    # pct >= 100, which would index out of the cuts list.
    cuts = statistics.quantiles(values, n=100)
    idx = pct - 1
    idx = max(0, min(idx, len(cuts) - 1))
    return float(cuts[idx])


def aggregate_metrics(
    results: list[ProbeResult],
    n_probes: int,
    n_drift: int,
) -> Metrics:
    """Build the full Metrics object from per-probe results plus drift.

    `n_probes` is the original probe count; `n_drift` is the count of
    probes excluded due to drift detection; `len(results)` is
    therefore N_scored = n_probes - n_drift. The caller computes
    drift before invoking this function. Per-tag breakdowns re-run
    score_probes over each tag's subset; a probe whose expected fact
    has multiple tags contributes to each tag's slice (intentional -
    a multi-tag fact is a multi-tag probe by construction).
    """
    base = score_probes(results)
    latencies = [float(r.latency_ms) for r in results]

    # Per-tag breakdown: bucket probes by each tag the expected fact
    # carries, then score each bucket independently. A probe with
    # tags ("preferences", "tech") appears in both buckets. This
    # matches the operator's mental model: "how does retrieval look
    # for facts tagged 'preferences'?" should not exclude facts that
    # also happen to carry another tag.
    by_tag_buckets: dict[str, list[ProbeResult]] = {}
    for r in results:
        for tag in r.tags:
            by_tag_buckets.setdefault(tag, []).append(r)
    by_tag = {tag: score_probes(bucket) for tag, bucket in sorted(by_tag_buckets.items())}

    return Metrics(
        n_probes=n_probes,
        n_scored=len(results),
        n_drift=n_drift,
        precision_at_k=base["precision_at_k"],
        recall_at_k=base["recall_at_k"],
        mrr=base["mrr"],
        fraction_in_prompt=base["fraction_in_prompt"],
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        by_tag=by_tag,
    )


# ── Drift detection ────────────────────────────────────────────────


def detect_drift(probes: list[Probe], user_id: str) -> tuple[list[Probe], list[Probe], dict[str, tuple[str, ...]]]:
    """Bucket probes into (scored, drifted) and collect tag mapping.

    Calls `memory.get_by_id` for each probe's expected_fact_id. A
    None return means the fact has been deleted, the probe was
    authored against the wrong user_id (ownership mismatch), the
    fact's source is not "extracted" (filtered by get_by_id by
    design), or the recorded ID was never valid (typo at probe
    authoring). All four collapse to "drift": the probe cannot be
    scored honestly. The tag mapping is built here rather than at
    scoring time so we don't re-fetch the same fact twice; the cost
    is one get_by_id call per probe regardless.

    Returns:
        (scored_probes, drifted_probes, tags_by_probe_id) where
        tags_by_probe_id maps each scored probe's expected_fact_id
        to the tuple of tags the fact carries (empty tuple if the
        fact has none).
    """
    from kai import memory as _mem

    scored: list[Probe] = []
    drifted: list[Probe] = []
    tags_by_id: dict[str, tuple[str, ...]] = {}
    for p in probes:
        fact = _mem.get_by_id(user_id=user_id, memory_id=p.expected_fact_id)
        if fact is None:
            drifted.append(p)
            continue
        scored.append(p)
        # `tags` may be absent or None on legacy rows; defensive
        # `or []` matches the shape used elsewhere in memory.py
        # (e.g. memory.py:949). Convert to tuple for hash-friendly
        # storage in the ProbeResult.
        raw_tags = (fact.metadata.get("tags") if fact.metadata else None) or []
        tags_by_id[p.expected_fact_id] = tuple(raw_tags)
    return scored, drifted, tags_by_id


# ── Per-probe execution ────────────────────────────────────────────


async def _run_one_probe(
    probe: Probe,
    user_id: str,
    capture: _RecallLogCapture,
    tags: tuple[str, ...],
) -> ProbeResult:
    """Run format_context for one probe and score its result.

    Drains the log capture buffer immediately after the call. The
    contract that exactly one `memory.recall` line is emitted per
    `format_context` call is enforced here: zero or more than one
    indicates a regression in the memory module's logging discipline,
    which would silently break this harness's scoring math, so we
    raise RuntimeError rather than silently picking one.
    """
    from kai.memory import format_context

    await format_context(probe.question, user_id=user_id)
    payloads = capture.drain()
    if len(payloads) != 1:
        raise RuntimeError(
            f"expected exactly one memory.recall log per probe, got {len(payloads)} for question {probe.question!r}"
        )
    payload = payloads[0]
    hits = payload.get("hits", [])
    lines_used = int(payload.get("lines_used", 0))
    latency_ms = int(payload.get("latency_ms", 0))
    rank = compute_rank(hits, probe.expected_fact_id)
    in_prompt = rank is not None and rank <= lines_used
    return ProbeResult(
        probe=probe,
        rank=rank,
        in_prompt=in_prompt,
        lines_used=lines_used,
        latency_ms=latency_ms,
        tags=tags,
    )


async def _run_probes(
    probes: list[Probe],
    user_id: str,
    capture: _RecallLogCapture,
    tags_by_id: dict[str, tuple[str, ...]],
) -> list[ProbeResult]:
    """Run a probe set sequentially.

    Sequential, not parallel: format_context drives an embedding +
    Qdrant lookup that is CPU-bound on the local process; running in
    parallel would not speed it up and would make the per-probe
    latency_ms numbers meaningless (concurrent calls would inflate
    each other's wall time). The probe count is small (a few dozen)
    so total wall time is modest.
    """
    out: list[ProbeResult] = []
    for p in probes:
        tags = tags_by_id.get(p.expected_fact_id, ())
        out.append(await _run_one_probe(p, user_id, capture, tags))
    return out


# ── Single-config evaluation ──────────────────────────────────────


def _attach_capture() -> _RecallLogCapture:
    """Attach a recall-log capture handler to kai.memory's logger.

    Caller is responsible for detaching after use (see
    `_detach_capture`). The handler is additive (does NOT set
    propagate=False on the logger) so existing log destinations
    still receive the lines; the harness only intercepts a copy.

    Forces the `kai.memory` logger level to INFO if it is currently
    higher (e.g. WARNING in a quiet operator config). Without this,
    info-level `memory.recall` records would be filtered out before
    reaching any handler and the harness would abort with
    "expected one log, got zero." The original level is saved on
    the capture object so detach can restore it.
    """
    logger = logging.getLogger("kai.memory")
    capture = _RecallLogCapture()
    capture._saved_level = logger.level
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    logger.addHandler(capture)
    return capture


def _detach_capture(capture: _RecallLogCapture) -> None:
    """Symmetric removal of the handler installed by _attach_capture.

    Restores the `kai.memory` logger's pre-attach level so a
    long-lived process running multiple harness invocations does
    not slowly accumulate verbosity changes.
    """
    logger = logging.getLogger("kai.memory")
    logger.removeHandler(capture)
    if capture._saved_level is not None:
        logger.setLevel(capture._saved_level)


async def _score_against_store(
    scored: list[Probe],
    drifted: list[Probe],
    tags_by_id: dict[str, tuple[str, ...]],
    user_id: str,
) -> Metrics:
    """Run probes through the live store and aggregate metrics.

    Split out from `evaluate` so the sweep loop can pre-compute drift
    once at sweep entry and reuse it across every grid point: drift
    state (which expected_fact_ids resolve via get_by_id) does not
    depend on the floor / speaker-weight / overfetch knobs that the
    sweep mutates, so re-running detect_drift per grid point would
    issue grid_size * len(probes) get_by_id calls for no new signal.
    """
    capture = _attach_capture()
    try:
        results = await _run_probes(scored, user_id, capture, tags_by_id)
    finally:
        # Detach in finally so a probe raising mid-run still removes
        # the harness-installed handler. Otherwise repeated harness
        # runs in a long-lived process (Python REPL, future test
        # suite) would stack capture handlers.
        _detach_capture(capture)
    return aggregate_metrics(
        results,
        n_probes=len(scored) + len(drifted),
        n_drift=len(drifted),
    )


async def evaluate(probes: list[Probe], user_id: str) -> Metrics:
    """Run drift detection + scoring for one configuration.

    The configuration is whatever is currently loaded into
    `kai.memory` module state - this function does NOT mutate that
    state. The sweep loop calls `_score_against_store` directly with
    pre-computed drift to avoid redundant get_by_id calls per grid
    point; single-config callers go through this wrapper.
    """
    scored, drifted, tags_by_id = detect_drift(probes, user_id)
    return await _score_against_store(scored, drifted, tags_by_id, user_id)


# ── Sweep mode ─────────────────────────────────────────────────────


def _grid_iter(
    floors: list[float],
    user_weights: list[float],
    assistant_weights: list[float],
    episode_summary_weights: list[float],
    overfetches: list[int],
) -> list[ConfigOverride]:
    """Materialize the grid as a flat list of ConfigOverride objects.

    Materialized (not generator) so the caller can `len()` the grid
    for progress reporting and iterate it more than once if needed.
    The five-deep loop is order-stable: outer floor, then the three
    speaker-weight axes (user, assistant, episode_summary), inner
    overfetch. Operators reading the table top-to-bottom see floor
    change slowest, which makes scanning for "what does raising the
    floor do" easier than the reverse order would.
    """
    out: list[ConfigOverride] = []
    for f in floors:
        for uw in user_weights:
            for aw in assistant_weights:
                for ew in episode_summary_weights:
                    for o in overfetches:
                        out.append(
                            ConfigOverride(
                                floor=f,
                                user_weight=uw,
                                assistant_weight=aw,
                                episode_summary_weight=ew,
                                overfetch=o,
                            )
                        )
    return out


def _apply_override(override: ConfigOverride) -> _Snapshot:
    """Apply an override to kai.memory module state.

    Mutates `_SPEAKER_WEIGHTS` in place (the dict is shared module-
    state; in-place mutation propagates to live `_speaker_weight`
    callers without a module-attr swap). Reassigns `_SEARCH_OVERFETCH`
    and `_config` at module scope. Returns a frozen `_Snapshot` of
    the prior state so `_restore_overrides` can put things back
    regardless of whether the caller's loop ran to completion.

    Note on `_config` derivation: the new `_config` is built via
    `dataclasses.replace(snap.config, ...)`, where `snap.config` is
    the LIVE pre-call state. At iteration N inside a sweep loop,
    that is iteration N-1's mutated config, NOT the entry-time
    original. For the current grid (only `memory_search_floor` is
    mutated by `dataclasses.replace`), this is behaviorally
    equivalent to deriving from the entry-time config because each
    iteration overwrites the same field. If a future grid mutates
    additional Config fields, swap to a `base_config` parameter so
    the caller controls which baseline to derive from.
    """
    from kai import memory as _mem

    snap = _Snapshot(
        speaker_weights=dict(_mem._SPEAKER_WEIGHTS),
        overfetch=_mem._SEARCH_OVERFETCH,
        config=_mem._config,
    )
    _mem._SPEAKER_WEIGHTS["user"] = override.user_weight
    _mem._SPEAKER_WEIGHTS["assistant"] = override.assistant_weight
    _mem._SPEAKER_WEIGHTS["episode_summary"] = override.episode_summary_weight
    _mem._SEARCH_OVERFETCH = override.overfetch
    _mem._config = dataclasses.replace(snap.config, memory_search_floor=override.floor)
    return snap


def _restore_overrides(snap: _Snapshot) -> None:
    """Restore module state from a snapshot taken by `_apply_override`.

    Counterpart of `_apply_override`. Clears + re-populates
    `_SPEAKER_WEIGHTS` in place so any module that holds a reference
    to the dict still sees the restored mapping; reassigns
    `_SEARCH_OVERFETCH` and `_config` directly. Idempotent: calling
    twice with the same snapshot is a no-op.
    """
    from kai import memory as _mem

    _mem._SPEAKER_WEIGHTS.clear()
    _mem._SPEAKER_WEIGHTS.update(snap.speaker_weights)
    _mem._SEARCH_OVERFETCH = snap.overfetch
    _mem._config = snap.config


async def run_sweep(
    probes: list[Probe],
    user_id: str,
    grid: list[ConfigOverride],
) -> list[tuple[ConfigOverride, Metrics]]:
    """Run `evaluate` once per grid point with module-state restoration.

    Mutates three pieces of `kai.memory` module state per iteration:
    `_config` (replaced via dataclasses.replace because Config is
    frozen), `_SPEAKER_WEIGHTS` (mutated in place; dict is not
    frozen), and `_SEARCH_OVERFETCH` (replaced via setattr on the
    module). A single `try/finally` wraps the entire grid loop and
    restores all three from the entry-time snapshot in `finally`.
    Any exception from any iteration (a probe raising mid-sweep, a
    SIGINT) unwinds through the outer finally before propagating, so
    the module is left in its pre-sweep state regardless of how the
    loop exits.

    Two-snapshot pattern: the entry-time snapshot taken via
    `_apply_override(grid[0])` would be incorrect because it would
    apply the first override before snapshotting. Instead we
    construct a `_Snapshot` directly before the loop rather than
    calling `_apply_override(grid[0])` and discarding the first
    mutation. Each iteration calls `_apply_override` (which itself
    snapshots-then-mutates), and the outer `finally` calls
    `_restore_overrides(entry_snap)` to land back at the pre-sweep
    state regardless of "whatever was there one iteration ago." If
    a probe somehow mutates module state (it should not), the
    snapshot restoration still leaves the module in a consistent
    post-sweep state matching its pre-sweep state.
    """
    from kai import memory as _mem

    if _mem._config is None:
        raise RuntimeError("memory not initialized; cannot sweep")

    # Capture the entry-time state directly. The per-iteration
    # `_apply_override` calls produce their own (inner) snapshots
    # which we discard; the entry-time outer snapshot is what feeds
    # the `finally` restore.
    entry_snap = _Snapshot(
        speaker_weights=dict(_mem._SPEAKER_WEIGHTS),
        overfetch=_mem._SEARCH_OVERFETCH,
        config=_mem._config,
    )

    # Drift detection is independent of the swept knobs: get_by_id
    # consults Mem0 row presence and source filtering, neither of
    # which the floor/weight/overfetch grid touches. Computing once
    # at sweep entry collapses grid_size * len(probes) get_by_id
    # calls down to len(probes) total. The store is assumed stable
    # across the sweep (which runs in minutes, not hours).
    scored, drifted, tags_by_id = detect_drift(probes, user_id)

    out: list[tuple[ConfigOverride, Metrics]] = []
    try:
        for i, override in enumerate(grid, start=1):
            log.info("sweep %d/%d: %s", i, len(grid), override)
            # Inner per-iteration snapshot is taken-and-discarded;
            # the entry_snap above is what restores at the outer
            # `finally`. Inline `_apply_override` here so the loop
            # body reads as one operation per axis rather than
            # threading a snapshot variable that goes unused.
            _apply_override(override)
            metrics = await _score_against_store(scored, drifted, tags_by_id, user_id)
            out.append((override, metrics))
    finally:
        # Restore from entry-time snapshot regardless of whether
        # the loop ran to completion.
        _restore_overrides(entry_snap)
    return out


def pareto_frontier(
    sweep_results: list[tuple[ConfigOverride, Metrics]],
) -> list[tuple[ConfigOverride, Metrics]]:
    """Return the configurations not dominated on (precision@5, latency_p50).

    A config A dominates config B iff A has higher-or-equal precision
    AND lower-or-equal latency, with at least one strict inequality.
    The frontier is the set of configs not dominated by any other.
    Axes are fixed at (precision@5, latency_p50): operators wanting
    alternate dimensions work from the full sweep table rather than
    re-running the sweep with different flags. This keeps the harness
    output deterministic across runs and avoids a multiplicative blow-
    up of CLI options for what is a one-line jq filter.
    """
    frontier: list[tuple[ConfigOverride, Metrics]] = []
    for cand_cfg, cand_metrics in sweep_results:
        cand_p = cand_metrics.precision_at_k.get(5, 0.0)
        cand_l = cand_metrics.latency_p50_ms
        dominated = False
        for other_cfg, other_metrics in sweep_results:
            if other_cfg is cand_cfg:
                continue
            other_p = other_metrics.precision_at_k.get(5, 0.0)
            other_l = other_metrics.latency_p50_ms
            # Strict-domination check: other is at least as good on
            # both axes AND strictly better on at least one. The
            # `>` in either axis combined with `>=`/`<=` on both
            # encodes the standard Pareto definition without
            # awkward tie-breaking.
            if other_p >= cand_p and other_l <= cand_l and (other_p > cand_p or other_l < cand_l):
                dominated = True
                break
        if not dominated:
            frontier.append((cand_cfg, cand_metrics))
    # Sort frontier by precision descending so the table reads
    # "best quality" -> "worst quality" top-to-bottom; ties broken
    # by lower latency.
    frontier.sort(key=lambda t: (-t[1].precision_at_k.get(5, 0.0), t[1].latency_p50_ms))
    return frontier


# ── Output formatting ─────────────────────────────────────────────


def _metrics_to_dict(m: Metrics, *, include_by_tag: bool = True) -> dict[str, Any]:
    """Convert a Metrics object to a JSON-serializable dict.

    Used both for the baseline file and for sweep table rows. The
    `include_by_tag` flag suppresses the per-tag breakdown for sweep
    rows where it would balloon the output without adding signal
    (operators reading a 72-row sweep table want top-line metrics,
    not per-tag rollups for each row).
    """
    d: dict[str, Any] = {
        "n_probes": m.n_probes,
        "n_scored": m.n_scored,
        "n_drift": m.n_drift,
        # Stringify the int K to keep the JSON shape stable: JSON
        # object keys must be strings, and `{"1": 0.68, ...}` is
        # what jq will see anyway.
        "precision_at_k": {str(k): round(v, 4) for k, v in m.precision_at_k.items()},
        "recall_at_k": {str(k): round(v, 4) for k, v in m.recall_at_k.items()},
        "mrr": round(m.mrr, 4),
        "fraction_in_prompt": round(m.fraction_in_prompt, 4),
        "latency_p50_ms": round(m.latency_p50_ms, 1),
        "latency_p95_ms": round(m.latency_p95_ms, 1),
    }
    if include_by_tag:
        d["by_tag"] = {
            tag: {
                "precision_at_k": {str(k): round(v, 4) for k, v in vals["precision_at_k"].items()},
                "recall_at_k": {str(k): round(v, 4) for k, v in vals["recall_at_k"].items()},
                "mrr": round(vals["mrr"], 4),
                "fraction_in_prompt": round(vals["fraction_in_prompt"], 4),
            }
            for tag, vals in m.by_tag.items()
        }
    return d


def _render_single_metrics(m: Metrics) -> str:
    """Human-readable rendering of one Metrics object for stdout.

    Plain text, no color codes - the harness output is meant to be
    pasteable into a bug report or log without ANSI escapes. Per-tag
    breakdown is included when present; if no probes carry tags
    (synthetic test fixture, e.g.) the section is omitted.
    """
    lines = [
        f"Probes: {m.n_probes} total, {m.n_scored} scored, {m.n_drift} drifted",
        "",
        "Precision @K:",
        *[f"  K={k}: {m.precision_at_k.get(k, 0.0):.3f}" for k in _K_VALUES],
        "",
        "Recall @K (single-answer = precision):",
        *[f"  K={k}: {m.recall_at_k.get(k, 0.0):.3f}" for k in _K_VALUES],
        "",
        f"MRR: {m.mrr:.3f}",
        f"fraction_in_prompt: {m.fraction_in_prompt:.3f}",
        "",
        f"Latency: p50={m.latency_p50_ms:.0f}ms, p95={m.latency_p95_ms:.0f}ms",
    ]
    if m.by_tag:
        lines.append("")
        lines.append("Per-tag (precision@5 / fraction_in_prompt):")
        for tag, vals in m.by_tag.items():
            p5 = vals["precision_at_k"].get(5, 0.0)
            fip = vals["fraction_in_prompt"]
            lines.append(f"  {tag}: p@5={p5:.3f} in_prompt={fip:.3f}")
    return "\n".join(lines)


def _render_sweep_table(sweep_results: list[tuple[ConfigOverride, Metrics]]) -> str:
    """ASCII table of every sweep configuration.

    Columns: floor, user_w, asst_w, ep_w, overfetch, p@1, p@3, p@5,
    MRR, in_prompt, p50_ms, p95_ms. Sorted by precision@5 descending
    so the operator's eye lands on the strongest configs first; ties
    broken by lower latency.
    """
    sorted_results = sorted(
        sweep_results,
        key=lambda t: (-t[1].precision_at_k.get(5, 0.0), t[1].latency_p50_ms),
    )
    header = (
        f"{'floor':>6} {'usr_w':>6} {'ast_w':>6} {'ep_w':>6} {'over':>5}  "
        f"{'p@1':>6} {'p@3':>6} {'p@5':>6}  "
        f"{'MRR':>6} {'in_pr':>6}  "
        f"{'p50':>5} {'p95':>5}"
    )
    sep = "-" * len(header)
    rows = []
    for cfg, m in sorted_results:
        # `.get(k, 0.0)` defends against _K_VALUES drifting out from
        # under this renderer; matches the pattern in pareto_frontier.
        p1 = m.precision_at_k.get(1, 0.0)
        p3 = m.precision_at_k.get(3, 0.0)
        p5 = m.precision_at_k.get(5, 0.0)
        rows.append(
            f"{cfg.floor:>6.2f} {cfg.user_weight:>6.2f} "
            f"{cfg.assistant_weight:>6.2f} {cfg.episode_summary_weight:>6.2f} "
            f"{cfg.overfetch:>5d}  "
            f"{p1:>6.3f} {p3:>6.3f} {p5:>6.3f}  "
            f"{m.mrr:>6.3f} {m.fraction_in_prompt:>6.3f}  "
            f"{m.latency_p50_ms:>5.0f} {m.latency_p95_ms:>5.0f}"
        )
    return "\n".join([header, sep, *rows])


def _render_pareto(frontier: list[tuple[ConfigOverride, Metrics]]) -> str:
    """Render just the Pareto-frontier rows from a sweep."""
    return _render_sweep_table(frontier)


# ── CLI ────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Top-level argparse for `python -m kai.eval.retrieval`."""
    parser = argparse.ArgumentParser(
        prog="python -m kai.eval.retrieval",
        description="Layer 1 retrieval evaluation harness (precision/recall/MRR).",
    )
    parser.add_argument(
        "--probes",
        required=True,
        type=Path,
        help="Path to probes.jsonl (JSONL format; #-prefixed lines are comments).",
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="Telegram chat_id (string) whose memory store to evaluate against.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run the full parameter grid and report Pareto frontier + table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Write results to a JSON file. Without --sweep: a single-config "
            "baseline matching the schema documented in `load_probes`. With "
            "--sweep: a sweep envelope (version, generated_at, probe_set_hash, "
            "probe_count, drift_count, sweep[]) with one row per grid point; "
            "this shape does NOT match the documented baseline schema."
        ),
    )
    # Sweep grid override flags. Each takes one or more values; if not
    # supplied, the default grid in the constants above is used.
    # nargs="+" so an operator can pass `--floor 0.2 0.3 0.4` without
    # quoting.
    parser.add_argument(
        "--floor",
        type=float,
        nargs="+",
        help="Override the memory_search_floor sweep axis (default: 0.15..0.40).",
    )
    parser.add_argument(
        "--user-weight",
        type=float,
        nargs="+",
        help="Override the user-speaker-weight sweep axis (default: 0.85, 1.0).",
    )
    parser.add_argument(
        "--assistant-weight",
        type=float,
        nargs="+",
        help="Override the assistant-speaker-weight sweep axis (default: 0.5, 0.6, 0.7, 0.8).",
    )
    parser.add_argument(
        "--episode-summary-weight",
        type=float,
        nargs="+",
        help="Override the episode-summary-speaker-weight sweep axis (default: 0.7, 0.85, 1.0).",
    )
    parser.add_argument(
        "--overfetch",
        type=int,
        nargs="+",
        help="Override the search-overfetch sweep axis (default: 10/20/30).",
    )
    return parser


def _initialize_memory() -> bool:
    """Load config and call init_memory(); return True on success.

    Mirrors the discipline in memory_admin: the harness runs against
    the live store, so the same init path keeps embedding model,
    Qdrant directory, and history DB settings consistent with the bot
    runtime. Errors print to stderr and the caller exits with status 1.
    """
    try:
        from kai.config import load_config
        from kai.memory import init_memory, is_enabled

        config = load_config()
        init_memory(config)
        if not is_enabled():
            print(
                "eval: memory is not enabled. Set MEMORY_ENABLED=true and verify the store is readable.",
                file=sys.stderr,
            )
            return False
        return True
    except Exception as e:
        print(f"eval: init failed: {e}", file=sys.stderr)
        return False


def _flatten_score_block(
    precision_at_k: dict[int, float],
    recall_at_k: dict[int, float],
    mrr: float,
    fraction_in_prompt: float,
) -> dict[str, Any]:
    """Render a metrics block with flat keys (precision_at_1, etc.).

    Used by `_build_baseline_json` for both the top-level `metrics`
    block and each `by_tag` entry, so a jq script diffing two
    baselines can use the same metric path under both scopes
    (`.metrics.precision_at_5` and `.by_tag.<tag>.precision_at_5`).
    The sweep-mode JSON output uses the nested `precision_at_k` shape
    via `_metrics_to_dict` instead, because per-row table data is
    consumed differently (iterate K values) and never compared key-
    for-key against another baseline.
    """
    return {
        "precision_at_1": round(precision_at_k.get(1, 0.0), 4),
        "precision_at_3": round(precision_at_k.get(3, 0.0), 4),
        "precision_at_5": round(precision_at_k.get(5, 0.0), 4),
        "recall_at_1": round(recall_at_k.get(1, 0.0), 4),
        "recall_at_3": round(recall_at_k.get(3, 0.0), 4),
        "recall_at_5": round(recall_at_k.get(5, 0.0), 4),
        "mrr": round(mrr, 4),
        "fraction_in_prompt": round(fraction_in_prompt, 4),
    }


def _build_baseline_json(
    probes: list[Probe],
    metrics: Metrics,
    cfg: ConfigOverride,
) -> dict[str, Any]:
    """Construct the baseline JSON envelope around a Metrics object.

    Matches the documented schema in `load_probes` (this module).
    `probe_set_hash` makes a baseline file meaningful only against its
    specific probe set; comparing baselines across probe sets would
    silently produce noise that looks like quality drift.

    Top-level `metrics` and each `by_tag` entry share the flat key
    shape via `_flatten_score_block`, so cross-scope diffs use the
    same path for the same metric.
    """
    metrics_block = _flatten_score_block(
        metrics.precision_at_k,
        metrics.recall_at_k,
        metrics.mrr,
        metrics.fraction_in_prompt,
    )
    metrics_block["latency_p50_ms"] = round(metrics.latency_p50_ms, 1)
    metrics_block["latency_p95_ms"] = round(metrics.latency_p95_ms, 1)
    return {
        "version": _BASELINE_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "probe_set_hash": probe_set_hash(probes),
        "probe_count": metrics.n_probes,
        "drift_count": metrics.n_drift,
        "config": {
            "floor": cfg.floor,
            "user_weight": cfg.user_weight,
            "assistant_weight": cfg.assistant_weight,
            "episode_summary_weight": cfg.episode_summary_weight,
            "overfetch": cfg.overfetch,
        },
        "metrics": metrics_block,
        "by_tag": {
            tag: _flatten_score_block(
                vals["precision_at_k"],
                vals["recall_at_k"],
                vals["mrr"],
                vals["fraction_in_prompt"],
            )
            for tag, vals in metrics.by_tag.items()
        },
    }


async def _run_cli(args: argparse.Namespace) -> int:
    """CLI dispatch.

    Returns process exit code: 0 on success, 1 on init / IO failure,
    2 on probe-file validation failure (distinct so an automation
    harness can tell "your probe file is broken" apart from "the
    eval ran but something else went wrong").
    """
    try:
        probes = load_probes(args.probes)
    except (OSError, ValueError) as e:
        print(f"eval: failed to load probes: {e}", file=sys.stderr)
        return 2
    if not probes:
        print(f"eval: no probes loaded from {args.probes}", file=sys.stderr)
        return 2

    if not _initialize_memory():
        return 1

    if args.sweep:
        floors = list(args.floor) if args.floor else list(_DEFAULT_FLOOR_GRID)
        user_weights = list(args.user_weight) if args.user_weight else list(_DEFAULT_USER_WEIGHT_GRID)
        assistant_weights = (
            list(args.assistant_weight) if args.assistant_weight else list(_DEFAULT_ASSISTANT_WEIGHT_GRID)
        )
        episode_summary_weights = (
            list(args.episode_summary_weight)
            if args.episode_summary_weight
            else list(_DEFAULT_EPISODE_SUMMARY_WEIGHT_GRID)
        )
        overfetches = list(args.overfetch) if args.overfetch else list(_DEFAULT_OVERFETCH_GRID)
        grid = _grid_iter(
            floors,
            user_weights,
            assistant_weights,
            episode_summary_weights,
            overfetches,
        )
        sweep_results = await run_sweep(probes, args.user_id, grid)
        print("Pareto frontier (precision@5 / latency_p50):")
        print(_render_pareto(pareto_frontier(sweep_results)))
        print()
        print("Full sweep table:")
        print(_render_sweep_table(sweep_results))
        if args.output:
            # Sweep output mode: write a JSON file containing every
            # config + metrics tuple. The Pareto frontier is not
            # serialized separately because it is cheaply re-derivable
            # from `sweep` via `pareto_frontier()`; a downstream
            # consumer that wants the frontier view runs that one
            # function rather than parsing a redundant sub-array.
            payload = {
                "version": _BASELINE_SCHEMA_VERSION,
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "probe_set_hash": probe_set_hash(probes),
                "probe_count": len(probes),
                # Hoist drift_count to the envelope. Every per-row
                # Metrics carries the same n_drift (drift detection
                # runs once before the grid loop, not per config),
                # so the value belongs at the top alongside
                # probe_count, mirroring the single-config baseline
                # written by _build_baseline_json. A downstream
                # parser can read drift_count off either file shape
                # without descending into the per-row metrics.
                "drift_count": next((m.n_drift for _, m in sweep_results), 0),
                "sweep": [
                    {
                        # Each per-row config block carries the three
                        # speaker-weight values in addition to floor
                        # and overfetch. A downstream parser sees the
                        # same key shape across single-config and
                        # sweep baselines without a special case.
                        "config": {
                            "floor": cfg.floor,
                            "user_weight": cfg.user_weight,
                            "assistant_weight": cfg.assistant_weight,
                            "episode_summary_weight": cfg.episode_summary_weight,
                            "overfetch": cfg.overfetch,
                        },
                        "metrics": _metrics_to_dict(m, include_by_tag=False),
                    }
                    for cfg, m in sweep_results
                ],
            }
            args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"\nWrote {args.output}")
        return 0

    # Single-config mode: use whatever production defaults the live
    # config carries. The baseline scenario (floor=0.3, extracted_
    # weight=1.2, overfetch=20) matches the production defaults at
    # the time of writing; an operator who has tuned their .env will
    # see whatever values they set.
    metrics = await evaluate(probes, args.user_id)
    print(_render_single_metrics(metrics))
    if args.output:
        # Reflect whatever live config was in effect into the saved
        # baseline. Since we did not sweep, we read the values back
        # from the memory module rather than carrying them through
        # an override object.
        from kai import memory as _mem

        cfg = ConfigOverride(
            floor=_mem._config.memory_search_floor if _mem._config else _PRODUCTION_FLOOR,
            user_weight=_mem._SPEAKER_WEIGHTS.get("user", _PRODUCTION_USER_WEIGHT),
            assistant_weight=_mem._SPEAKER_WEIGHTS.get("assistant", _PRODUCTION_ASSISTANT_WEIGHT),
            episode_summary_weight=_mem._SPEAKER_WEIGHTS.get("episode_summary", _PRODUCTION_EPISODE_SUMMARY_WEIGHT),
            overfetch=_mem._SEARCH_OVERFETCH,
        )
        args.output.write_text(
            json.dumps(_build_baseline_json(probes, metrics, cfg), indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for `python -m kai.eval.retrieval`.

    `argv` defaults to `sys.argv[1:]`. Returns the exit code for
    the caller; `__main__` block below propagates it via sys.exit.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run_cli(args))


if __name__ == "__main__":
    sys.exit(main())
