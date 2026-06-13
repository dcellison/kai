"""
Scoped retrieval evaluation harness.

Reachable as `python -m kai.eval.retrieval_scoped`.

Sibling to `kai.eval.retrieval` (the legacy harness). Where the
legacy module scores `format_context`, this module scores the
production scoped read path: `retrieve_scoped_memories` for the IR
ranking family and `format_scoped_context_with_recall_payload` for
prompt placement, exclusion safety, and fail-closed observation.
Both harnesses ship in parallel until side-by-side comparisons
settle which (if either) to retire.

Design rationale (the part that is not obvious from the code):

- TWO CALLS PER PROBE, RENDERED FIRST. The harness calls the
  renderer (`format_scoped_context_with_recall_payload`) before the
  raw helper (`retrieve_scoped_memories`). The renderer is the
  production wrapper: it catches every exception from retrieval and
  rendering and collapses to `reason="scoped_error"` with an empty
  rendered context. Running the raw helper FIRST would let a
  retrieval exception escape past the wrapper's fail-closed catch
  and abort the whole eval run; the rendered-first order keeps the
  wrapper authoritative and lets the harness record a per-probe row
  for the failing probe instead of dying. The raw helper runs only
  when the rendered payload's reason is not `scoped_error`; that
  call is itself wrapped in a defensive try/except for the narrow
  case where the rendered call succeeded but the second call hits a
  transient.

- TWO SCORING SURFACES, DELIBERATELY DIFFERENT. The renderer writes
  `payload["hits"]` in PROMPT order (global section first, then
  project section), not adjusted-score order. A top-ranked project
  fact whose global section already filled the first five slots
  looks like rank 6 to the legacy IR family. So `candidate_rank`
  (used for precision@K, recall@K, MRR) is keyed on
  `ScopedRetrievalResult.hits` (adjusted-score order) from the raw
  helper, while `prompt_position` and `in_prompt` (used for
  `fraction_in_prompt`) are keyed on `payload["hits"]` from the
  renderer. The two are reported separately because they answer
  different questions: did retrieval rank the fact well, vs. did
  the agent see it in its prompt slice.

- FAIL CLOSED VISIBLE. When the rendered call returns
  `reason="scoped_error"`, the per-probe row records
  `candidate_rank=None, prompt_position=None, in_prompt=False,
  scoped_reason="scoped_error"` rather than aborting. The harness's
  fail-closed-observation contract requires that a single broken
  probe never lose data on the surviving probes.

- DRIFT IS PER POLARITY. A probe with both `expected_fact_id` and
  `expected_excluded_fact_ids` is scored on both sides independently.
  If the positive id has been deleted from the store but the
  excluded ids resolve, the probe still contributes to the exclusion
  pass/fail count; the positive side reports it as drift. The
  reverse holds when the positive is clean but an excluded id has
  been deleted. The drift counts are reported by polarity so the
  operator can tell "stale positive" from "stale negative" probes.

- REGISTRY BOOTSTRAP BEFORE SCORING. The scoped pipeline detects
  the active memory project through the merged YAML+DB registry.
  In a fresh CLI process the DB layer is empty unless we explicitly
  load it, and a probe whose `workspace` lives under a chat-
  registered root would degrade silently to global-only. The
  harness calls `kai.memory_projects.load_project_registry` (which
  inits the session DB and bulk-loads the cache) before any probe
  runs.

PII posture: probe questions and expected_fact_id values are
operator data and remain in the gitignored probes file. The output
JSON writes only an 80-char-truncated question_truncated field; the
full text never leaves the probe file.
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


# K values reported for precision / recall. Same set as the legacy
# harness so an operator running both side by side reads the same
# columns; 1 is the strictest top-pick metric, 3 is the typical mid-
# budget point, 5 is the default Pareto axis used by the legacy
# harness's sort.
_K_VALUES: tuple[int, ...] = (1, 3, 5)


# Default sweep grid. The scoped pipeline reads the same module-level
# knobs the legacy pipeline does at the relevant steps (raw-score
# floor at step 7, speaker/confidence weighting at step 8, overfetch
# inside the Mem0 fetch), so the grid is intentionally identical:
# the two harnesses cover the same axes and operators can compare
# corresponding grid points across the two output files. Project
# admission knobs (`memory_enabled` per project) are policy, not
# tuning targets; varying them would measure a configuration nobody
# runs in production, so they are excluded.
_DEFAULT_FLOOR_GRID: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
_DEFAULT_USER_WEIGHT_GRID: tuple[float, ...] = (0.85, 1.0)
_DEFAULT_ASSISTANT_WEIGHT_GRID: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8)
_DEFAULT_EPISODE_SUMMARY_WEIGHT_GRID: tuple[float, ...] = (0.7, 0.85, 1.0)
_DEFAULT_OVERFETCH_GRID: tuple[int, ...] = (10, 20, 30)


# Production defaults for the single-config baseline shape. Match the
# legacy module's constants because the scoped helper reads the same
# module-level state; an operator whose .env tunes any of these will
# see whatever they actually set in the saved baseline.
_PRODUCTION_FLOOR = 0.30
_PRODUCTION_USER_WEIGHT = 0.85
_PRODUCTION_ASSISTANT_WEIGHT = 0.8
_PRODUCTION_EPISODE_SUMMARY_WEIGHT = 0.85
_PRODUCTION_OVERFETCH = 20


# Schema version written into baseline JSON output. Separate counter
# from the legacy harness (which carries its own `_BASELINE_SCHEMA_
# VERSION`) so operator-side comparison tools can tell legacy and
# scoped baselines apart by version field alone; the two schemas are
# structurally similar but the metric blocks differ.
_BASELINE_SCHEMA_VERSION = 1


# Bucket key for the `by_active_project` distribution when a probe's
# workspace is null or matches no registered root. Picked as a sentinel
# string that cannot collide with a real project_id (which the YAML
# loader strips and project ids do not contain underscores at both
# ends; even if one did, the underscore-bracketed form makes the
# sentinel visually obvious in the stdout summary).
_NONE_PROJECT_KEY = "__none__"


# Truncation length for the question text recorded in JSON output.
# Operator data is kept in the gitignored probe file; the output JSON
# only carries enough of the question to disambiguate a row visually
# when reading a report next to the source file.
_QUESTION_TRUNC = 80


# Reason string the renderer sets when scoped retrieval or rendering
# raises. Mirrored from `kai.memory._RECALL_REASON_SCOPED_ERROR`;
# duplicated as a literal rather than imported so this module's import
# surface stays at the public scoped helpers.
_REASON_SCOPED_ERROR = "scoped_error"


# ── Data classes ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScopedProbe:
    """A single eval probe under the scoped schema (v2).

    Carries both polarities so a probe can assert that a specific
    fact reaches the prompt (`expected_fact_id`) AND that one or
    more cross-scope facts do NOT (`expected_excluded_fact_ids`).
    `workspace` parameterizes the probe against a per-project
    detection result; `None` means the probe runs as a non-project
    (global-only) probe.

    `line_number` is the 1-indexed source line in the probe file,
    threaded from the loader so the per-probe output rows include
    `probe_index` matching the file the operator opens. Source turn
    timestamp and notes are unchanged from v1, operator bookkeeping.
    """

    question: str
    expected_fact_id: str | None
    expected_excluded_fact_ids: tuple[str, ...]
    workspace: str | None
    line_number: int
    source_turn_ts: str = ""
    notes: str = ""


@dataclass
class ConfigOverride:
    """One point in the sweep grid.

    Mirrors the legacy `ConfigOverride` so the two harnesses share
    grid axes; the scoped pipeline reads the same `_SPEAKER_WEIGHTS`,
    `_SEARCH_OVERFETCH`, and `Config.memory_search_floor` state at
    its admission/ranking steps. Documented here as well so a reader
    of just this file does not have to cross-reference the legacy
    module to know what each axis controls.
    """

    floor: float
    user_weight: float
    assistant_weight: float
    episode_summary_weight: float
    overfetch: int


@dataclass(frozen=True)
class _Snapshot:
    """Frozen snapshot of mutable kai.memory state captured before an
    override is applied. Used by `_restore_overrides` to put the module
    back in its pre-apply shape regardless of how the caller's loop
    exits. The dict is copied at capture time so future mutation of
    `_SPEAKER_WEIGHTS` does not leak through the saved reference.
    """

    speaker_weights: dict[str, float]
    overfetch: int
    config: Config


@dataclass
class ScopedProbeResult:
    """Per-probe scoring outcome.

    Two scoring surfaces (see module docstring): `candidate_rank` is
    the 1-indexed position in `ScopedRetrievalResult.hits` from the
    raw helper (adjusted-score order), used for precision@K /
    recall@K / MRR; `prompt_position` is the 1-indexed position in
    the rendered payload's `hits` list (prompt order), used together
    with `lines_used` for `in_prompt` and `fraction_in_prompt`.

    `positive_drift` is True when the probe's `expected_fact_id` did
    not resolve via `get_by_id` and the positive side is therefore
    excluded from numerator AND denominator. `negative_drift_ids`
    lists the excluded ids that also did not resolve and are
    therefore excluded from the exclusion-pass denominators. Either
    bucket can be empty while the other is populated; drift is
    per-polarity, not per-probe.

    `excluded_in_prompt` and `excluded_in_candidates` are the lists
    of excluded ids the harness actually observed in the rendered
    payload, ordered by their appearance in the probe author's
    `expected_excluded_fact_ids` list so a per-probe report row
    reads in the same order as the probe file. Both axes are
    reported because they describe different safety failures:
    in-prompt means the agent saw a wrong-scope row; in-candidates
    means scope admission let one through even if the budget happened
    to drop it before rendering.

    `active_project_id` and `scoped_reason` come straight from the
    renderer's `scoped_debug` payload. `None` for `active_project_id`
    means no detected project (workspace null OR workspace path that
    did not match any registered root).
    """

    probe: ScopedProbe
    candidate_rank: int | None
    prompt_position: int | None
    in_prompt: bool
    lines_used: int
    latency_ms: int
    tags: tuple[str, ...]
    excluded_in_prompt: list[str]
    excluded_in_candidates: list[str]
    positive_drift: bool
    negative_drift_ids: list[str]
    active_project_id: str | None
    scoped_reason: str


@dataclass
class ScopedMetrics:
    """Aggregated metrics for one configuration over a probe set.

    Stored as plain dicts so JSON serialization is direct. Split into
    a positive side (IR family + prompt visibility) and a negative
    side (exclusion safety). The cross-cutting fields (latencies,
    `by_tag`, `by_scoped_reason`, `by_active_project`) apply to the
    whole probe set regardless of polarity.

    `n_scored_positive` denominates over probes with `expected_fact_
    id` set that resolved via `get_by_id`. `n_scored_negative`
    denominates over INDIVIDUAL excluded ids that resolved; a probe
    with three excluded ids and one drift contributes 2 to
    `n_scored_negative` and 1 to `n_drift_negative`.
    """

    n_probes: int
    # Positive side (probes with expected_fact_id):
    n_scored_positive: int
    n_drift_positive: int
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    mrr: float
    fraction_in_prompt: float
    # Negative side (probes with expected_excluded_fact_ids):
    n_scored_negative: int
    n_drift_negative: int
    exclusion_pass_in_prompt: float
    exclusion_pass_in_candidates: float
    # Cross-cutting:
    latency_p50_ms: float
    latency_p95_ms: float
    by_tag: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Scoped-specific:
    by_scoped_reason: dict[str, int] = field(default_factory=dict)
    by_active_project: dict[str, int] = field(default_factory=dict)


# ── Probe file loading ─────────────────────────────────────────────


def load_probes(path: Path) -> list[ScopedProbe]:
    """Load and validate probes from a JSONL file under schema v2.

    File format mirrors the legacy harness: `#`-prefixed lines are
    comments (after lstrip, so indented annotations are accepted),
    blank lines are skipped, every other line is a JSON object.

    Required field rules:
    - `question` (string, non-empty) is required on every probe.
    - At least one of `expected_fact_id` (non-empty string) or
      `expected_excluded_fact_ids` (non-empty list of non-empty
      strings) must be present; a probe with neither is unscorable
      on either polarity and is rejected at load time rather than
      silently inflating the safety metric (it would count as a pass
      against zero excluded ids).
    - `workspace` is optional (defaults to null = non-project probe).
      Probe author writes the literal path; the harness does NOT
      resolve a project_id shorthand into a path.
    - `expected_excluded_fact_ids` elements are each validated as
      non-empty strings: an int element from a hand-edited file or
      an empty-string element would be unmatchable against any real
      id and would silently inflate the safety metric.

    Validation errors include the file path and 1-indexed line number
    so an operator can find the offending row immediately. Line order
    is preserved in the returned list and each probe carries its line
    number for downstream output.
    """
    probes: list[ScopedProbe] = []
    raw = path.read_text(encoding="utf-8").splitlines()
    for lineno, line in enumerate(raw, start=1):
        stripped = line.lstrip()
        # Comment/blank lines are skipped BEFORE JSON parse so an
        # operator can annotate a probe file in-place. Matches the
        # legacy loader's behavior exactly.
        if not stripped or stripped.startswith("#"):
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno}: malformed JSON ({e.msg})") from e
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{lineno}: expected JSON object, got {type(obj).__name__}")

        question = obj.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{path}:{lineno}: 'question' missing or not a non-empty string")

        # Validate the field shape when present. An int slip-through
        # (`expected_fact_id: 42` from a spreadsheet export) would
        # never match a real string id and silently bucket as drift;
        # catching at load time is louder. The combined condition
        # checks "present AND wrong type / empty" in one branch.
        expected = obj.get("expected_fact_id", None)
        if expected is not None and (not isinstance(expected, str) or not expected.strip()):
            raise ValueError(f"{path}:{lineno}: 'expected_fact_id' must be a non-empty string when present")

        raw_excluded = obj.get("expected_excluded_fact_ids", [])
        if not isinstance(raw_excluded, list):
            raise ValueError(f"{path}:{lineno}: 'expected_excluded_fact_ids' must be a list when present")
        excluded: list[str] = []
        for elem_idx, elem in enumerate(raw_excluded):
            # Each excluded id must be a non-empty string. The element
            # type/value is named in the error so an operator with a
            # mixed-type list (e.g. a stray int from a CSV import) can
            # find and fix it without diffing the full file.
            if not isinstance(elem, str):
                raise ValueError(
                    f"{path}:{lineno}: 'expected_excluded_fact_ids[{elem_idx}]' must be a "
                    f"string, got {type(elem).__name__}"
                )
            if not elem.strip():
                raise ValueError(
                    f"{path}:{lineno}: 'expected_excluded_fact_ids[{elem_idx}]' must be non-empty (got empty string)"
                )
            excluded.append(elem)

        # Polarity gate: a probe with neither a positive id nor any
        # excluded ids is unscorable. The default-empty list lets v1-
        # shaped probes (no excluded field) work, but a probe whose
        # excluded list is present-but-empty AND has no positive id
        # is the exact authoring mistake this gate catches.
        if expected is None and not excluded:
            raise ValueError(
                f"{path}:{lineno}: probe has neither 'expected_fact_id' nor "
                f"'expected_excluded_fact_ids'; cannot be scored on either polarity"
            )

        workspace = obj.get("workspace", None)
        if workspace is not None and not isinstance(workspace, str):
            raise ValueError(f"{path}:{lineno}: 'workspace' must be a string or null when present")
        if isinstance(workspace, str) and not workspace.strip():
            # Empty-string workspace would flow into Path("") and
            # behave like "current directory", which is almost never
            # the probe author's intent. Reject explicitly so the
            # author writes null instead.
            raise ValueError(f"{path}:{lineno}: 'workspace' must be non-empty when present")

        source_turn_ts = obj.get("source_turn_ts", "")
        notes = obj.get("notes", "")
        if not isinstance(source_turn_ts, str):
            raise ValueError(f"{path}:{lineno}: 'source_turn_ts' must be a string when present")
        if not isinstance(notes, str):
            raise ValueError(f"{path}:{lineno}: 'notes' must be a string when present")

        probes.append(
            ScopedProbe(
                question=question,
                expected_fact_id=expected,
                expected_excluded_fact_ids=tuple(excluded),
                workspace=workspace,
                line_number=lineno,
                source_turn_ts=source_turn_ts,
                notes=notes,
            )
        )
    return probes


def probe_set_hash(probes: list[ScopedProbe]) -> str:
    """SHA-256 of the sorted (question, expected, sorted_excluded, ws)
    tuples. Locks a baseline measurement to its probe set: comparing
    two baselines with different hashes is meaningless. The negative
    polarity contributes to the hash so adding an excluded id changes
    the hash (otherwise a v2 probe file with new exclusions would look
    identical to its v1 ancestor). Sorted excluded ids per probe so
    the hash is reorder-invariant within a probe too.
    """
    tuples = sorted(
        (
            p.question,
            p.expected_fact_id or "",
            tuple(sorted(p.expected_excluded_fact_ids)),
            p.workspace or "",
        )
        for p in probes
    )
    blob = json.dumps(tuples, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


# ── Drift detection ────────────────────────────────────────────────


def detect_drift(
    probes: list[ScopedProbe],
    user_id: str,
) -> tuple[
    dict[int, bool],  # positive_drift_by_line: line_number -> True iff expected_fact_id drifted
    dict[int, list[str]],  # negative_drift_by_line: line_number -> list of drifted excluded ids
    dict[str, tuple[str, ...]],  # tags_by_id: expected_fact_id -> tags tuple
]:
    """Bucket drift per polarity and collect the tag mapping.

    Calls `memory.get_by_id` once per unique id across both polarities.
    Drift is per-polarity, not per-probe: a probe whose positive id is
    deleted but whose excluded ids resolve is still scored on the
    exclusion side, and vice versa. The harness reports the two drift
    counts separately so the operator can tell "stale positive" from
    "stale negative" probes apart.

    The tag mapping is built only for surviving positive ids; per-tag
    aggregates apply to the positive (IR) family and exclusion-side
    rollups have no tag concept (the excluded id is by definition
    NOT supposed to surface, so its tags are not the scoring grain).
    """
    from kai import memory as _mem

    positive_drift_by_line: dict[int, bool] = {}
    negative_drift_by_line: dict[int, list[str]] = {}
    tags_by_id: dict[str, tuple[str, ...]] = {}

    # Resolve once per unique id to avoid repeated get_by_id calls when
    # the same id appears across polarities (e.g. a probe asserts a
    # specific fact and another probe asserts that same fact is NOT
    # surfaced in a different workspace, which is a legitimate cross-
    # project collision check).
    resolved: dict[str, Any] = {}

    def _resolve(memory_id: str) -> Any:
        if memory_id not in resolved:
            resolved[memory_id] = _mem.get_by_id(user_id=user_id, memory_id=memory_id)
        return resolved[memory_id]

    for p in probes:
        if p.expected_fact_id is not None:
            fact = _resolve(p.expected_fact_id)
            if fact is None:
                positive_drift_by_line[p.line_number] = True
            else:
                positive_drift_by_line[p.line_number] = False
                # `tags` may be absent or None on legacy rows; the
                # `or []` matches the defensive shape used elsewhere
                # in memory.py. Tuple form keeps the value hashable
                # for use as a ProbeResult field.
                raw_tags = (fact.metadata.get("tags") if fact.metadata else None) or []
                tags_by_id[p.expected_fact_id] = tuple(raw_tags)
        if p.expected_excluded_fact_ids:
            drifted_negatives: list[str] = []
            for excluded_id in p.expected_excluded_fact_ids:
                if _resolve(excluded_id) is None:
                    drifted_negatives.append(excluded_id)
            negative_drift_by_line[p.line_number] = drifted_negatives

    return positive_drift_by_line, negative_drift_by_line, tags_by_id


# ── Active-project detection (pre-scoring filter) ─────────────────


def _detect_probe_active_project(
    probe: ScopedProbe,
    registry: dict,
) -> str | None:
    """Pre-detect the active project id for `probe` against the merged
    registry. Used by the `--projects` filter to drop probes BEFORE
    scoring rather than running every probe and discarding results.

    Mirrors the same detection rule the scoped helper applies at
    request time, so the filter and the scoring path agree on which
    project a workspace resolves to. A probe with no workspace, or
    a workspace path that does not resolve to a registered root,
    returns None (which the filter and the aggregator both bucket
    under `_NONE_PROJECT_KEY`).
    """
    from kai.memory_projects import detect_active_memory_project

    if probe.workspace is None:
        return None
    active = detect_active_memory_project(Path(probe.workspace), registry)
    return active.project_id if active is not None else None


# ── Per-probe execution ────────────────────────────────────────────


def _candidate_rank(hits: list, expected_fact_id: str) -> int | None:
    """Return the 1-indexed position of `expected_fact_id` in the raw
    helper's hits list. Each entry is a `ScopedMemoryHit` whose
    `.result.id` carries the Mem0 row id. Returns None when the
    expected fact is not in the list (missed retrieval or filtered
    by scope/floor admission).
    """
    for i, h in enumerate(hits):
        if h.result.id == expected_fact_id:
            return i + 1
    return None


def _prompt_position(payload_hits: list[dict[str, Any]], expected_fact_id: str) -> int | None:
    """Return the 1-indexed position of `expected_fact_id` in the
    renderer's `payload["hits"]` list. Entries are dicts whose `id`
    field carries the Mem0 row id (per `_scoped_hit_to_shadow_payload`
    in `kai.memory`). Returns None when the expected fact is not in
    the rendered list at all.
    """
    for i, h in enumerate(payload_hits):
        if h.get("id") == expected_fact_id:
            return i + 1
    return None


def _scan_excluded(
    payload_hits: list[dict[str, Any]],
    lines_used: int,
    excluded_ids: tuple[str, ...],
    drifted_ids: set[str],
) -> tuple[list[str], list[str]]:
    """Walk the rendered hits list once and bucket excluded-id matches.

    Returns `(in_prompt, in_candidates)`. An id present at a position
    `<= lines_used` lands in `in_prompt` (the agent saw it); an id
    present anywhere in the rendered list lands in `in_candidates`
    (scope admission let it through even if the budget dropped it
    before rendering). The two axes are reported because they
    describe different safety failures.

    Drifted excluded ids (their get_by_id returned None) are dropped
    from both axes: a deleted row trivially "passes" the in-prompt
    check, and counting that as a pass would inflate the safety
    metric for the wrong reason. The harness reports them under
    `negative_drift_ids` instead.

    Result order matches the probe author's
    `expected_excluded_fact_ids` order so a per-probe report row
    reads in the same order as the probe file.
    """
    # Map id -> 1-indexed position. Build once so the per-excluded-id
    # check is a dict lookup, not a list scan, which matters when the
    # probe carries many excluded ids and the rendered list is long.
    id_to_position: dict[str, int] = {}
    for i, h in enumerate(payload_hits):
        hit_id = h.get("id")
        if isinstance(hit_id, str) and hit_id not in id_to_position:
            id_to_position[hit_id] = i + 1

    in_prompt: list[str] = []
    in_candidates: list[str] = []
    for excluded_id in excluded_ids:
        if excluded_id in drifted_ids:
            continue
        pos = id_to_position.get(excluded_id)
        if pos is None:
            continue
        in_candidates.append(excluded_id)
        if pos <= lines_used:
            in_prompt.append(excluded_id)
    return in_prompt, in_candidates


async def _run_one_probe(
    probe: ScopedProbe,
    user_id: str,
    tags: tuple[str, ...],
    positive_drift: bool,
    negative_drift_ids: list[str],
) -> ScopedProbeResult:
    """Run the two-call scoring shape for one probe.

    Order: rendered call first, raw helper call second. See module
    docstring for the rationale (the renderer is the production fail-
    closed wrapper; running it first keeps it authoritative). The
    raw helper runs only when the rendered payload did not already
    report `scoped_error`; if it raises on its own (a transient that
    the renderer's first internal call ate but our second call
    exposes), we log and record `candidate_rank=None` rather than
    aborting the run.
    """
    from kai.memory import (
        ScopedRetrievalContext,
        format_scoped_context_with_recall_payload,
        retrieve_scoped_memories,
    )

    workspace_path = Path(probe.workspace) if probe.workspace else None

    # Call 1: prompt placement, exclusion safety, fail-closed
    # observation. The renderer catches every exception from
    # retrieval and rendering and collapses to reason=scoped_error.
    rendered = await format_scoped_context_with_recall_payload(
        query=probe.question,
        user_id=user_id,
        workspace=workspace_path,
        backend_name=None,
        job_type="eval",
        session_id=None,
    )
    payload = rendered.recall_payload
    payload_hits = payload.get("hits") or []
    lines_used = int(payload.get("lines_used") or 0)
    latency_ms = int(payload.get("latency_ms") or 0)
    scoped_debug = payload.get("scoped_debug") or {}
    scoped_reason = str(payload.get("reason") or "")
    active_project_id_raw = scoped_debug.get("active_project_id")
    active_project_id = active_project_id_raw if isinstance(active_project_id_raw, str) else None

    # Call 2: ranking. Skipped on `scoped_error` because the raw
    # helper does NOT carry the renderer's broad try/except; calling
    # it on a path that just failed would re-raise the same exception
    # out of the harness and lose the per-probe row the wrapper
    # already gave us. The defensive try/except around the call
    # covers the narrow transient case.
    candidate_rank: int | None = None
    if scoped_reason != _REASON_SCOPED_ERROR and probe.expected_fact_id is not None and not positive_drift:
        try:
            context = ScopedRetrievalContext(
                chat_id=user_id,
                message=probe.question,
                workspace=workspace_path,
                job_type="eval",
                backend_name=None,
                session_id=None,
            )
            scoped = await retrieve_scoped_memories(context)
            candidate_rank = _candidate_rank(scoped.hits, probe.expected_fact_id)
        except Exception:
            log.warning(
                "scoped evaluator: raw helper raised after rendered call succeeded",
                exc_info=True,
            )
            candidate_rank = None

    # Prompt position is independent of the positive-drift state: a
    # drifted positive id cannot be found in the rendered hits either,
    # so prompt_position degrades to None naturally without a guard.
    if probe.expected_fact_id is not None and not positive_drift:
        prompt_position = _prompt_position(payload_hits, probe.expected_fact_id)
    else:
        prompt_position = None
    in_prompt = prompt_position is not None and prompt_position <= lines_used

    # Negative side. Drop drifted excluded ids before scanning so a
    # deleted row does not silently inflate the safety metric.
    drifted_set = set(negative_drift_ids)
    excluded_in_prompt, excluded_in_candidates = _scan_excluded(
        payload_hits,
        lines_used,
        probe.expected_excluded_fact_ids,
        drifted_set,
    )

    return ScopedProbeResult(
        probe=probe,
        candidate_rank=candidate_rank,
        prompt_position=prompt_position,
        in_prompt=in_prompt,
        lines_used=lines_used,
        latency_ms=latency_ms,
        tags=tags,
        excluded_in_prompt=excluded_in_prompt,
        excluded_in_candidates=excluded_in_candidates,
        positive_drift=positive_drift,
        negative_drift_ids=list(negative_drift_ids),
        active_project_id=active_project_id,
        scoped_reason=scoped_reason,
    )


async def _run_probes(
    probes: list[ScopedProbe],
    user_id: str,
    tags_by_id: dict[str, tuple[str, ...]],
    positive_drift_by_line: dict[int, bool],
    negative_drift_by_line: dict[int, list[str]],
) -> list[ScopedProbeResult]:
    """Run a probe set sequentially.

    Sequential, not parallel: each scoring step drives an embedding +
    Qdrant lookup that is CPU-bound on the local process; running in
    parallel would not speed it up and would make the per-probe
    latency_ms numbers meaningless (concurrent calls would inflate
    each other's wall time). The probe count is small (dozens, not
    thousands) so total wall time is modest.
    """
    out: list[ScopedProbeResult] = []
    for p in probes:
        tags = tags_by_id.get(p.expected_fact_id, ()) if p.expected_fact_id else ()
        positive_drift = positive_drift_by_line.get(p.line_number, False)
        negative_drift_ids = negative_drift_by_line.get(p.line_number, [])
        out.append(
            await _run_one_probe(
                p,
                user_id,
                tags,
                positive_drift,
                negative_drift_ids,
            )
        )
    return out


# ── Scoring math ────────────────────────────────────────────────────


def score_positive(results: list[ScopedProbeResult]) -> dict[str, Any]:
    """Compute the positive metric block (precision, recall, MRR,
    fraction_in_prompt) over results that are not positive-drift.

    Pure function. The caller filters drifted-positive results out
    before invoking. Precision and recall are equal for single-answer
    probes; reported separately so a future multi-answer probe format
    can diverge.

    IR metrics (precision@K, recall@K, MRR) key on `candidate_rank`
    against the raw helper's adjusted-score order.
    `fraction_in_prompt` keys on `prompt_position` against the
    renderer's prompt-order list and that probe's `lines_used`. The
    two surfaces answer different questions; see module docstring.
    """
    n = len(results)
    if n == 0:
        return {
            "precision_at_k": {k: 0.0 for k in _K_VALUES},
            "recall_at_k": {k: 0.0 for k in _K_VALUES},
            "mrr": 0.0,
            "fraction_in_prompt": 0.0,
        }

    precision_at_k: dict[int, float] = {}
    recall_at_k: dict[int, float] = {}
    for k in _K_VALUES:
        # A hit "counts at K" iff candidate_rank is set AND falls
        # within the top K. Combining the two guards keeps the math
        # obviously correct and avoids a None-check that would crash
        # on a missed probe.
        hits_in_top_k = sum(1 for r in results if r.candidate_rank is not None and r.candidate_rank <= k)
        precision_at_k[k] = hits_in_top_k / n
        recall_at_k[k] = hits_in_top_k / n

    in_prompt_count = sum(1 for r in results if r.in_prompt)

    # MRR: standard "miss as zero" convention. Explicit so a reader
    # does not assume "skip misses" semantics that would inflate the
    # number on poor-coverage probe sets.
    mrr_sum = sum(1.0 / r.candidate_rank if r.candidate_rank else 0.0 for r in results)

    return {
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "mrr": mrr_sum / n,
        "fraction_in_prompt": in_prompt_count / n,
    }


def score_negative(
    results: list[ScopedProbeResult],
) -> tuple[int, int, float, float]:
    """Compute the negative-side safety metrics.

    Returns `(n_scored_negative, n_drift_negative,
    exclusion_pass_in_prompt, exclusion_pass_in_candidates)`.

    The denominator is the count of INDIVIDUAL excluded ids that
    resolved via get_by_id (not the count of probes), because each
    excluded id is an independent assertion. A probe with three
    excluded ids and one drift contributes 2 to the denominator and
    1 to `n_drift_negative`.

    `exclusion_pass_in_prompt` is the safety metric the harness
    exists to expose: a downward trend means cross-project rows are
    leaking into prompts. `exclusion_pass_in_candidates` is the
    weaker regression sentinel: a fail there is the leak that scope
    admission missed even if the budget happened to drop the row
    before rendering.

    Both metrics return 1.0 when the denominator is zero (no
    excluded ids in the probe set). 1.0 means "no failures
    observed," matching the operator's read of "all the assertions
    we made passed." This matches the legacy harness's "zero is
    safe" posture for empty inputs.
    """
    n_drift_negative = 0
    n_scored_negative = 0
    in_prompt_failures = 0
    in_candidates_failures = 0

    for r in results:
        excluded_ids = r.probe.expected_excluded_fact_ids
        drifted = set(r.negative_drift_ids)
        for excluded_id in excluded_ids:
            if excluded_id in drifted:
                n_drift_negative += 1
                continue
            n_scored_negative += 1
            if excluded_id in r.excluded_in_prompt:
                in_prompt_failures += 1
            if excluded_id in r.excluded_in_candidates:
                in_candidates_failures += 1

    if n_scored_negative == 0:
        return n_scored_negative, n_drift_negative, 1.0, 1.0

    pass_in_prompt = (n_scored_negative - in_prompt_failures) / n_scored_negative
    pass_in_candidates = (n_scored_negative - in_candidates_failures) / n_scored_negative
    return n_scored_negative, n_drift_negative, pass_in_prompt, pass_in_candidates


def _percentile(values: list[float], pct: int) -> float:
    """Return the requested percentile from values (1-99 integer scale).

    Same shape as the legacy harness's _percentile: linear-
    interpolation estimate via statistics.quantiles, with 0.0 for
    empty input and the lone value for a singleton (the library
    rejects n=1).
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    cuts = statistics.quantiles(values, n=100)
    idx = pct - 1
    idx = max(0, min(idx, len(cuts) - 1))
    return float(cuts[idx])


def aggregate_metrics(
    results: list[ScopedProbeResult],
    n_probes: int,
) -> ScopedMetrics:
    """Build the full ScopedMetrics object from per-probe results.

    Positive denominator excludes positive drift; negative denominator
    excludes drifted excluded ids. `by_scoped_reason` and
    `by_active_project` come straight from the rendered payload's
    debug fields across the whole probe set so the distribution
    surfaces in the JSON regardless of which polarity each probe
    contributes to.
    """
    # Positive side: only probes with expected_fact_id AND that did
    # not drift. Drifted-positive probes still appear in the per-
    # probe details array (with `candidate_rank=None,
    # positive_drift=True`) but are dropped from numerator and
    # denominator of the positive aggregates.
    positive_results = [r for r in results if r.probe.expected_fact_id is not None and not r.positive_drift]
    n_drift_positive = sum(1 for r in results if r.probe.expected_fact_id is not None and r.positive_drift)
    positive_block = score_positive(positive_results)

    (
        n_scored_negative,
        n_drift_negative,
        pass_in_prompt,
        pass_in_candidates,
    ) = score_negative(results)

    latencies = [float(r.latency_ms) for r in results]

    # Per-tag breakdown over the positive side only; the negative
    # axis has no tag concept (an excluded id is by definition the
    # one NOT supposed to surface, so its tags are not the scoring
    # grain). Match the legacy harness's per-tag math by re-running
    # `score_positive` over each tag's subset.
    by_tag_buckets: dict[str, list[ScopedProbeResult]] = {}
    for r in positive_results:
        for tag in r.tags:
            by_tag_buckets.setdefault(tag, []).append(r)
    by_tag = {tag: score_positive(bucket) for tag, bucket in sorted(by_tag_buckets.items())}

    # Cross-cutting distributions. by_scoped_reason and by_active_
    # project are computed over the FULL result list (both polarities
    # contribute) because they describe the pipeline's behavior on
    # every probe, not just the IR-scored ones. A probe whose only
    # job is the exclusion check still reveals "scope admission ran
    # for this workspace and reported reason X".
    by_scoped_reason: dict[str, int] = {}
    by_active_project: dict[str, int] = {}
    for r in results:
        by_scoped_reason[r.scoped_reason] = by_scoped_reason.get(r.scoped_reason, 0) + 1
        bucket_key = r.active_project_id if r.active_project_id else _NONE_PROJECT_KEY
        by_active_project[bucket_key] = by_active_project.get(bucket_key, 0) + 1

    return ScopedMetrics(
        n_probes=n_probes,
        n_scored_positive=len(positive_results),
        n_drift_positive=n_drift_positive,
        precision_at_k=positive_block["precision_at_k"],
        recall_at_k=positive_block["recall_at_k"],
        mrr=positive_block["mrr"],
        fraction_in_prompt=positive_block["fraction_in_prompt"],
        n_scored_negative=n_scored_negative,
        n_drift_negative=n_drift_negative,
        exclusion_pass_in_prompt=pass_in_prompt,
        exclusion_pass_in_candidates=pass_in_candidates,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        by_tag=by_tag,
        by_scoped_reason=by_scoped_reason,
        by_active_project=by_active_project,
    )


# ── Single-config evaluation ──────────────────────────────────────


async def _score_against_store(
    probes: list[ScopedProbe],
    tags_by_id: dict[str, tuple[str, ...]],
    positive_drift_by_line: dict[int, bool],
    negative_drift_by_line: dict[int, list[str]],
    user_id: str,
) -> tuple[list[ScopedProbeResult], ScopedMetrics]:
    """Run probes through the live scoped pipeline and aggregate.

    Returns both the raw per-probe results AND the aggregated
    metrics so the CLI can render the per-probe details array
    without re-running the probes. Split out from `evaluate` so the
    sweep loop can pre-compute drift once at sweep entry and reuse
    it across every grid point: drift state does not depend on the
    floor/weight/overfetch knobs the sweep mutates.
    """
    results = await _run_probes(
        probes,
        user_id,
        tags_by_id,
        positive_drift_by_line,
        negative_drift_by_line,
    )
    metrics = aggregate_metrics(results, n_probes=len(probes))
    return results, metrics


async def evaluate(
    probes: list[ScopedProbe],
    user_id: str,
) -> tuple[list[ScopedProbeResult], ScopedMetrics]:
    """Run drift detection + scoring for one configuration.

    Single-config callers go through this wrapper; sweep callers
    call `_score_against_store` directly with pre-computed drift to
    avoid redundant get_by_id calls per grid point.
    """
    positive_drift_by_line, negative_drift_by_line, tags_by_id = detect_drift(probes, user_id)
    return await _score_against_store(
        probes,
        tags_by_id,
        positive_drift_by_line,
        negative_drift_by_line,
        user_id,
    )


# ── Sweep mode ─────────────────────────────────────────────────────


def _grid_iter(
    floors: list[float],
    user_weights: list[float],
    assistant_weights: list[float],
    episode_summary_weights: list[float],
    overfetches: list[int],
) -> list[ConfigOverride]:
    """Materialize the grid as a flat list of ConfigOverride objects.

    Outer-to-inner: floor, user_weight, assistant_weight,
    episode_summary_weight, overfetch. Same axis ordering as the
    legacy harness so two side-by-side sweeps line up row-for-row in
    the same configuration order.
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

    Mutates `_SPEAKER_WEIGHTS` in place (the dict is shared module
    state; in-place mutation propagates to live `_speaker_weight`
    callers without a module-attr swap). Reassigns `_SEARCH_OVERFETCH`
    and `_config` at module scope. Returns a frozen `_Snapshot` of
    the prior state so `_restore_overrides` can put things back.
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

    Idempotent: clear-and-repopulate so any module holding a reference
    to the `_SPEAKER_WEIGHTS` dict still sees the restored mapping;
    reassign `_SEARCH_OVERFETCH` and `_config` directly.
    """
    from kai import memory as _mem

    _mem._SPEAKER_WEIGHTS.clear()
    _mem._SPEAKER_WEIGHTS.update(snap.speaker_weights)
    _mem._SEARCH_OVERFETCH = snap.overfetch
    _mem._config = snap.config


async def run_sweep(
    probes: list[ScopedProbe],
    user_id: str,
    grid: list[ConfigOverride],
) -> list[tuple[ConfigOverride, list[ScopedProbeResult], ScopedMetrics]]:
    """Run `evaluate` once per grid point with module-state restoration.

    Same two-snapshot pattern as the legacy harness: capture the
    entry-time state once, then per-iteration `_apply_override` takes
    its own throwaway inner snapshot. The outer `try/finally` restores
    from the entry-time snapshot regardless of how the loop exits, so
    a probe raising mid-sweep or a SIGINT still leaves the module in
    its pre-sweep state.

    Returns per-grid-point `(override, results, metrics)` so the CLI
    can produce a sweep envelope with optional per-probe details
    arrays.
    """
    from kai import memory as _mem

    if _mem._config is None:
        raise RuntimeError("memory not initialized; cannot sweep")

    entry_snap = _Snapshot(
        speaker_weights=dict(_mem._SPEAKER_WEIGHTS),
        overfetch=_mem._SEARCH_OVERFETCH,
        config=_mem._config,
    )

    # Drift is independent of the swept knobs (get_by_id consults
    # Mem0 row presence and source filtering, neither of which the
    # floor/weight/overfetch grid touches). Compute once at sweep
    # entry to collapse grid_size * len(probes) get_by_id calls down
    # to len(probes) total.
    positive_drift_by_line, negative_drift_by_line, tags_by_id = detect_drift(probes, user_id)

    out: list[tuple[ConfigOverride, list[ScopedProbeResult], ScopedMetrics]] = []
    try:
        for i, override in enumerate(grid, start=1):
            log.info("sweep %d/%d: %s", i, len(grid), override)
            _apply_override(override)
            results, metrics = await _score_against_store(
                probes,
                tags_by_id,
                positive_drift_by_line,
                negative_drift_by_line,
                user_id,
            )
            out.append((override, results, metrics))
    finally:
        _restore_overrides(entry_snap)
    return out


# ── Output formatting ─────────────────────────────────────────────


def _truncate_question(question: str) -> str:
    """Return the question text truncated to `_QUESTION_TRUNC` chars
    with an ellipsis when cut. The full text never leaves the probe
    file; the truncated form lets an operator disambiguate a row at
    a glance when reading a report next to the source file.
    """
    if len(question) <= _QUESTION_TRUNC:
        return question
    return question[: _QUESTION_TRUNC - 3] + "..."


def _probe_to_details_entry(result: ScopedProbeResult) -> dict[str, Any]:
    """Render one ScopedProbeResult as a per-probe details dict."""
    return {
        "probe_index": result.probe.line_number,
        "question_truncated": _truncate_question(result.probe.question),
        "workspace": result.probe.workspace,
        "active_project_id": result.active_project_id,
        "expected_fact_id": result.probe.expected_fact_id,
        "candidate_rank": result.candidate_rank,
        "prompt_position": result.prompt_position,
        "in_prompt": result.in_prompt,
        "positive_drift": result.positive_drift,
        "expected_excluded_fact_ids": list(result.probe.expected_excluded_fact_ids),
        "negative_drift_ids": list(result.negative_drift_ids),
        "excluded_in_prompt": list(result.excluded_in_prompt),
        "excluded_in_candidates": list(result.excluded_in_candidates),
        "scoped_reason": result.scoped_reason,
    }


def _metric_block(m: ScopedMetrics, *, include_by_tag: bool) -> dict[str, Any]:
    """Render the metric block (positive + negative + cross-cutting)
    for inclusion in either single-config or per-sweep-row output.

    Mirrors the legacy harness's `_metrics_to_dict` shape on the
    positive side so an operator-side diff tool can compare
    corresponding fields under the same key, then adds the negative
    block and the two scoped distributions. `include_by_tag` controls
    whether the per-tag breakdown rides along; suppress it for sweep
    rows where the per-tag detail would balloon the output without
    adding signal at the table-row level.
    """
    block: dict[str, Any] = {
        "n_probes": m.n_probes,
        "n_scored_positive": m.n_scored_positive,
        "n_drift_positive": m.n_drift_positive,
        # Stringify the int K so the JSON shape is stable: JSON object
        # keys must be strings, and `{"1": 0.68, ...}` is what jq will
        # see anyway.
        "precision_at_k": {str(k): round(v, 4) for k, v in m.precision_at_k.items()},
        "recall_at_k": {str(k): round(v, 4) for k, v in m.recall_at_k.items()},
        "mrr": round(m.mrr, 4),
        "fraction_in_prompt": round(m.fraction_in_prompt, 4),
        "n_scored_negative": m.n_scored_negative,
        "n_drift_negative": m.n_drift_negative,
        "exclusion_pass_in_prompt": round(m.exclusion_pass_in_prompt, 4),
        "exclusion_pass_in_candidates": round(m.exclusion_pass_in_candidates, 4),
        "latency_p50_ms": round(m.latency_p50_ms, 1),
        "latency_p95_ms": round(m.latency_p95_ms, 1),
        "by_scoped_reason": dict(m.by_scoped_reason),
        "by_active_project": dict(m.by_active_project),
    }
    if include_by_tag:
        block["by_tag"] = {
            tag: {
                "precision_at_k": {str(k): round(v, 4) for k, v in vals["precision_at_k"].items()},
                "recall_at_k": {str(k): round(v, 4) for k, v in vals["recall_at_k"].items()},
                "mrr": round(vals["mrr"], 4),
                "fraction_in_prompt": round(vals["fraction_in_prompt"], 4),
            }
            for tag, vals in m.by_tag.items()
        }
    return block


def _build_single_config_json(
    probes: list[ScopedProbe],
    results: list[ScopedProbeResult],
    metrics: ScopedMetrics,
    cfg: ConfigOverride,
) -> dict[str, Any]:
    """Build the single-config JSON envelope.

    Top-level shape mirrors the legacy harness's `_build_baseline_json`
    so operator-side tooling can navigate "config + metrics" the same
    way. The schema_version is the scoped counter (separate from the
    legacy counter), so a diff tool can tell baselines apart by
    version field alone.
    """
    return {
        "version": _BASELINE_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "probe_set_hash": probe_set_hash(probes),
        "probe_count": metrics.n_probes,
        "drift_count_positive": metrics.n_drift_positive,
        "drift_count_negative": metrics.n_drift_negative,
        "config": {
            "floor": cfg.floor,
            "user_weight": cfg.user_weight,
            "assistant_weight": cfg.assistant_weight,
            "episode_summary_weight": cfg.episode_summary_weight,
            "overfetch": cfg.overfetch,
        },
        "metrics": _metric_block(metrics, include_by_tag=True),
        "probes": [_probe_to_details_entry(r) for r in results],
    }


def _build_sweep_json(
    probes: list[ScopedProbe],
    sweep_results: list[tuple[ConfigOverride, list[ScopedProbeResult], ScopedMetrics]],
    include_details: bool,
) -> dict[str, Any]:
    """Build the sweep-mode JSON envelope.

    Mirrors the legacy harness's sweep envelope (version, generated_at,
    probe_set_hash, probe_count, drift counts, sweep[]) so operator-
    side tooling can diff legacy and scoped sweep envelopes side by
    side. The per-row block contains the override AND the metric
    block; the `probes` array is omitted by default (a 50-probe corpus
    across a 240-row grid is 12k probe records, bloating envelopes
    without informing the sweep summary) and re-included under
    `include_details` for operators chasing a specific failure mode.
    """
    # Hoist drift counts to the envelope: every per-row metric carries
    # the same n_drift_* (drift detection runs once before the grid
    # loop, not per config), so the values belong at the top alongside
    # probe_count, mirroring the single-config baseline shape.
    drift_pos = next((m.n_drift_positive for _, _, m in sweep_results), 0)
    drift_neg = next((m.n_drift_negative for _, _, m in sweep_results), 0)
    return {
        "version": _BASELINE_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "probe_set_hash": probe_set_hash(probes),
        "probe_count": len(probes),
        "drift_count_positive": drift_pos,
        "drift_count_negative": drift_neg,
        "sweep": [
            {
                "config": {
                    "floor": cfg.floor,
                    "user_weight": cfg.user_weight,
                    "assistant_weight": cfg.assistant_weight,
                    "episode_summary_weight": cfg.episode_summary_weight,
                    "overfetch": cfg.overfetch,
                },
                "metrics": _metric_block(m, include_by_tag=False),
                # Per-probe details ride along only when the operator
                # opts in. Sweep mode default omits them to keep the
                # envelope readable across 100+ grid points.
                **({"probes": [_probe_to_details_entry(r) for r in results]} if include_details else {}),
            }
            for cfg, results, m in sweep_results
        ],
    }


def _render_single_metrics(m: ScopedMetrics) -> str:
    """Human-readable rendering of one ScopedMetrics object for stdout.

    Plain text, no color codes. Two blocks (positive + negative) plus
    a latency line and the scoped distributions; per-tag breakdown is
    appended when present, omitted when no probes carry tags.
    """
    lines = [
        f"Probes: {m.n_probes} total",
        f"Positive: {m.n_scored_positive} scored, {m.n_drift_positive} drifted",
        "",
        "Precision @K (over candidate_rank, raw helper):",
        *[f"  K={k}: {m.precision_at_k.get(k, 0.0):.3f}" for k in _K_VALUES],
        "",
        "Recall @K (single-answer = precision):",
        *[f"  K={k}: {m.recall_at_k.get(k, 0.0):.3f}" for k in _K_VALUES],
        "",
        f"MRR: {m.mrr:.3f}",
        f"fraction_in_prompt (over prompt_position): {m.fraction_in_prompt:.3f}",
        "",
        f"Negative: {m.n_scored_negative} excluded ids scored, {m.n_drift_negative} drifted",
        f"exclusion_pass_in_prompt: {m.exclusion_pass_in_prompt:.3f}",
        f"exclusion_pass_in_candidates: {m.exclusion_pass_in_candidates:.3f}",
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


def _render_sweep_table(
    sweep_results: list[tuple[ConfigOverride, list[ScopedProbeResult], ScopedMetrics]],
) -> str:
    """ASCII table of every sweep configuration.

    Columns: floor, user_w, asst_w, ep_w, over, p@5, MRR, in_prompt,
    excl_pr (exclusion_pass_in_prompt), excl_cd
    (exclusion_pass_in_candidates), p50, p95. Sorted by precision@5
    descending so the operator's eye lands on the strongest configs
    first; ties broken by lower latency. Negative-side metrics are
    table columns too because a config that wins precision but leaks
    excluded ids is a regression, and the table is where the operator
    spots it.
    """
    sorted_results = sorted(
        sweep_results,
        key=lambda t: (-t[2].precision_at_k.get(5, 0.0), t[2].latency_p50_ms),
    )
    header = (
        f"{'floor':>6} {'usr_w':>6} {'ast_w':>6} {'ep_w':>6} {'over':>5}  "
        f"{'p@5':>6} {'MRR':>6} {'in_pr':>6}  "
        f"{'excl_pr':>7} {'excl_cd':>7}  "
        f"{'p50':>5} {'p95':>5}"
    )
    sep = "-" * len(header)
    rows = []
    for cfg, _results, m in sorted_results:
        p5 = m.precision_at_k.get(5, 0.0)
        rows.append(
            f"{cfg.floor:>6.2f} {cfg.user_weight:>6.2f} "
            f"{cfg.assistant_weight:>6.2f} {cfg.episode_summary_weight:>6.2f} "
            f"{cfg.overfetch:>5d}  "
            f"{p5:>6.3f} {m.mrr:>6.3f} {m.fraction_in_prompt:>6.3f}  "
            f"{m.exclusion_pass_in_prompt:>7.3f} {m.exclusion_pass_in_candidates:>7.3f}  "
            f"{m.latency_p50_ms:>5.0f} {m.latency_p95_ms:>5.0f}"
        )
    return "\n".join([header, sep, *rows])


def _print_sweep_top_config_distributions(
    sweep_results: list[tuple[ConfigOverride, list[ScopedProbeResult], ScopedMetrics]],
) -> None:
    """Emit the stdout distribution lines for the TOP-RANKED sweep
    config, with the config inline.

    A sweep produces one ScopedMetrics per grid point and each carries
    its own `by_active_project` and `by_scoped_reason`. `by_active_
    project` is invariant across the grid because workspace detection
    runs before any swept knob touches retrieval, but `by_scoped_
    reason` is NOT invariant: a probe whose rows fall below the floor
    reports `all_below_floor` at one floor and `ok` at another, so the
    distribution shifts with `--floor` sweeps.

    Picking the last grid point as a "representative" row would silently
    misreport: with `--floor 0.15 0.40` the sorted table shows the
    best row at the top, then the last grid point at floor=0.40 is the
    one whose probes fell below the floor, and the printed summary
    would say `all_below_floor=N` for a config the operator is not
    looking at. Instead, this helper sorts by the same key the sweep
    table uses (precision@5 desc, latency_p50 asc), takes the top row,
    and prints both distributions tagged with that row's config so the
    binding is explicit. Per-row distributions remain available in the
    JSON output for operators chasing a specific grid point.
    """
    if not sweep_results:
        return
    sorted_results = sorted(
        sweep_results,
        key=lambda t: (-t[2].precision_at_k.get(5, 0.0), t[2].latency_p50_ms),
    )
    top_cfg, _, top_metrics = sorted_results[0]
    print(
        "eval: top config: "
        f"floor={top_cfg.floor:.2f}, usr_w={top_cfg.user_weight:.2f}, "
        f"ast_w={top_cfg.assistant_weight:.2f}, ep_w={top_cfg.episode_summary_weight:.2f}, "
        f"over={top_cfg.overfetch}"
    )
    print(_format_distribution("by_active_project", top_metrics.by_active_project))
    print(_format_distribution("by_scoped_reason", top_metrics.by_scoped_reason))


def _format_distribution(label: str, distribution: dict[str, int]) -> str:
    """Render a sorted `key=count, ...` distribution line for stdout.

    Stable ordering for deterministic test assertions: keys sorted
    ascending. Non-zero filtering happens at the call site because
    `by_scoped_reason` and `by_active_project` should not have zero
    buckets in practice (we only insert keys we observed), but the
    filter is cheap insurance for future schema additions.
    """
    parts = [f"{k}={v}" for k, v in sorted(distribution.items()) if v > 0]
    return f"eval: {label}: {', '.join(parts) if parts else '(empty)'}"


# ── CLI ────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Top-level argparse for `python -m kai.eval.retrieval_scoped`."""
    parser = argparse.ArgumentParser(
        prog="python -m kai.eval.retrieval_scoped",
        description=("Scoped retrieval evaluation harness (precision/recall/MRR + cross-scope exclusion safety)."),
    )
    # Positional user_id to match the memory admin CLI's `purge` and
    # `reclassify-scope` shape: the three admin-style CLIs then take
    # the user as the first positional argument.
    parser.add_argument(
        "user_id",
        help="Telegram chat_id (string) whose memory store to evaluate against.",
    )
    parser.add_argument(
        "--probes",
        required=True,
        type=Path,
        help="Path to probes.jsonl (JSONL format; #-prefixed lines are comments).",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run the full parameter grid and report a sorted table per config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Write results to a JSON file. Without --sweep: a single-config "
            "envelope with per-probe details. With --sweep: a sweep envelope "
            "(per-probe details omitted by default; pass --include-details to "
            "include them per grid point)."
        ),
    )
    parser.add_argument(
        "--include-details",
        action="store_true",
        help=(
            "Include the per-probe details array in sweep-mode output. Ignored "
            "in single-config mode (details are always included there)."
        ),
    )
    # Filter flag pair (mutually exclusive). Mostly diagnostic; the
    # safety baseline always runs the whole probe set, but an operator
    # narrowing down a regression to one project does not have to edit
    # the probe file.
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--non-project-only",
        action="store_true",
        help="Score only probes whose workspace is null (non-project posture).",
    )
    filter_group.add_argument(
        "--projects",
        nargs="+",
        help=(
            "Score only probes whose detected active project id is one of "
            "the named values. Detection mirrors the scoped helper's: "
            "a probe whose workspace matches no registered root is skipped."
        ),
    )
    # Sweep grid override flags. Each takes one or more values; if not
    # supplied, the default grid is used.
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
    Qdrant directory, and history DB settings consistent with the
    bot runtime. Emits one startup line naming BOTH pipelines this
    harness exercises so an operator running both can tell the two
    log streams apart at a glance.
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
        # Identify which harness is running so a side-by-side log
        # stream (legacy + scoped) is unambiguous. The legacy module
        # emits its own "I am legacy" warning when the scoped knob is
        # on; this harness's line is the symmetric counterpart.
        print(
            "eval: scoped harness; exercises retrieve_scoped_memories (ranking) + "
            "format_scoped_context_with_recall_payload (prompt placement).",
            file=sys.stderr,
        )
        if not config.memory_scoped_recall_enabled:
            # The scoped read path is NOT live in production for this
            # install; the harness still runs the scoped pipeline
            # end-to-end (which is the point of the harness's
            # existence), but the operator should know the numbers
            # do not describe the live production behavior here.
            print(
                "eval: WARNING: MEMORY_SCOPED_RECALL_ENABLED is off; production "
                "serves the LEGACY pipeline. These results describe the scoped "
                "pipeline only and not the live production read path on this install.",
                file=sys.stderr,
            )
        return True
    except Exception as e:
        print(f"eval: init failed: {e}", file=sys.stderr)
        return False


async def _apply_cli_filters(
    probes: list[ScopedProbe],
    *,
    non_project_only: bool,
    projects: list[str] | None,
    config: Config,
) -> list[ScopedProbe]:
    """Apply the CLI filter flags before scoring.

    `--non-project-only` is a pre-scoring pre-filter on the probe
    file itself: drop every probe whose workspace is non-null. No
    detection needed.

    `--projects` requires running detection against the merged
    registry to decide which probes belong to the named projects.
    We load the registry once and re-use the cached detection result.
    Probes whose detected active project id is not in the allow-list
    (including those that detect to None / non-project) are dropped.

    Argparse already enforces the mutually-exclusive group, so at
    most one of the two filters is active per call.
    """
    if non_project_only:
        return [p for p in probes if p.workspace is None]
    if projects:
        from kai.memory_projects import load_project_registry

        registry = await load_project_registry(config)
        allowed = set(projects)
        return [p for p in probes if _detect_probe_active_project(p, registry) in allowed]
    return probes


async def _ensure_registry_loaded(config: Config) -> None:
    """Bootstrap the merged project registry once per CLI run.

    The scoped helper detects the active memory project through the
    merged YAML+DB registry. In a fresh CLI process the DB layer is
    empty unless this is called, and a probe whose `workspace` lives
    under a chat-registered root degrades silently to global-only.

    Imports the helper from `kai.memory_projects` so the project-
    registry bootstrap lives next to the registry itself rather than
    in another CLI module: no CLI has to depend on another CLI just
    for this one-line init shape.
    """
    from kai.memory_projects import load_project_registry

    await load_project_registry(config)


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

    # Registry bootstrap MUST run before scoring so detection sees
    # chat-registered projects. The CLI filter step also needs the
    # registry (for `--projects`), so loading once at the top covers
    # both consumers without a second DB init.
    from kai.config import load_config

    config = load_config()
    await _ensure_registry_loaded(config)

    probes = await _apply_cli_filters(
        probes,
        non_project_only=args.non_project_only,
        projects=args.projects,
        config=config,
    )
    if not probes:
        # The filter dropped every probe; bail with the same exit
        # code as "no probes loaded" so the operator's automation
        # treats both as the same class of authoring issue.
        print(
            "eval: no probes remain after filters; check --non-project-only / --projects.",
            file=sys.stderr,
        )
        return 2

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
        print("Full sweep table (sorted by precision@5, then latency):")
        print(_render_sweep_table(sweep_results))
        _print_sweep_top_config_distributions(sweep_results)
        if args.output:
            payload = _build_sweep_json(probes, sweep_results, args.include_details)
            args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"\nWrote {args.output}")
        return 0

    # Single-config mode: use whatever production defaults the live
    # config carries.
    results, metrics = await evaluate(probes, args.user_id)
    print(_render_single_metrics(metrics))
    print()
    print(_format_distribution("by_active_project", metrics.by_active_project))
    print(_format_distribution("by_scoped_reason", metrics.by_scoped_reason))
    if args.output:
        # Reflect the live module state into the saved baseline so
        # the file records which knobs were in effect. Falls back to
        # production defaults when memory is somehow uninitialized
        # (shouldn't happen at this point, but the fallback is
        # cheap insurance).
        from kai import memory as _mem

        cfg = ConfigOverride(
            floor=_mem._config.memory_search_floor if _mem._config else _PRODUCTION_FLOOR,
            user_weight=_mem._SPEAKER_WEIGHTS.get("user", _PRODUCTION_USER_WEIGHT),
            assistant_weight=_mem._SPEAKER_WEIGHTS.get("assistant", _PRODUCTION_ASSISTANT_WEIGHT),
            episode_summary_weight=_mem._SPEAKER_WEIGHTS.get("episode_summary", _PRODUCTION_EPISODE_SUMMARY_WEIGHT),
            overfetch=_mem._SEARCH_OVERFETCH,
        )
        args.output.write_text(
            json.dumps(_build_single_config_json(probes, results, metrics, cfg), indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for `python -m kai.eval.retrieval_scoped`.

    `argv` defaults to `sys.argv[1:]`. Returns the exit code for the
    caller; the `__main__` block below propagates it via sys.exit.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run_cli(args))


if __name__ == "__main__":
    sys.exit(main())
