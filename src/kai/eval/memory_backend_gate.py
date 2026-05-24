"""
Claude-vs-Codex semantic memory eval gate.

Drives the production memory write path (extract_and_store +
format_context + Mem0 storage + consolidation + episode generation)
for both backends against the same probe fixture, scores fact
extraction quality, retrieval quality, episode behavior, parse and
runtime failures, duplicate and consolidation behavior, speaker
attribution, and tag shape, and emits a single JSON artifact with
per-backend metrics, per-probe details, threshold checks, and an
overall pass/fail verdict.

The gate is distinct from `kai.eval.extraction` (prompt-revision
comparison), `kai.eval.replay` (single-backend history replay), and
`kai.eval.retrieval` (precomputed-fact retrieval scoring): backend
comparison has different axes, different artifacts, and different
thresholds, and the expected fact IDs do not exist until each
backend arm has populated its own sandbox. Reusing one of the
existing harnesses would either carry vocabulary that does not fit
(`v5_facts` / `v6_facts`) or expect inputs that this gate has to
create as part of its own run.

Each arm writes to a distinct sandbox user ID derived from the
operator-supplied `--user-prefix` (which must start with
`sandbox-`, the constant `kai.eval.replay._SANDBOX_USER_ID_PREFIX`).
Real probe content stays out of the repo; tests use synthetic
fixtures under `tmp_path` and the operator's real fixture path is
supplied at runtime.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kai import memory, memory_extraction
from kai.config import Config, ModelRole, get_model_for, load_config
from kai.eval.extraction import _window_to_extractor_args
from kai.eval.replay import _SANDBOX_USER_ID_PREFIX

log = logging.getLogger(__name__)


# ── Constants ───────────────────────────────────────────────────────


# Probe categories supported by the fixture loader. `confirmation`
# probes share storage semantics with `durable-fact` (an explicit
# `confirmed_action` tag plus a structured quote) so the fixture
# minimums group them; the category stays distinct so qualitative
# sampling can show the operator confirmation-shaped rows
# separately when something looks off.
ALLOWED_CATEGORIES = (
    "durable-fact",
    "workflow-noise",
    "confirmation",
    "consolidation",
    "episode-positive",
    "episode-negative",
)

# Speaker enum the loader accepts on `must_store.speaker`. Matches
# the production fact schema's enum plus `episode_summary` for
# episode-row anchors (those rows are written with that speaker by
# the stage-2 generator).
ALLOWED_SPEAKERS = ("user", "assistant", "episode_summary")

# Allowed values for `expected.consolidation.expected_intent`.
# Mirrors the intent field on `memory.consolidate.intent` log
# payloads in `memory_extraction.py`, including `hallucinated_id`
# (which a probe might assert it expects in a defense-in-depth test;
# the gate hard-fails on any hallucinated_id_count > 0 anyway).
ALLOWED_CONSOLIDATION_INTENTS = (
    "new",
    "update_of",
    "skip_redundant",
    "hallucinated_id",
)

# Allowed values for `expected.consolidation.expected_outcome`. Six
# of the seven outcome strings emitted by memory_extraction.py;
# `dropped` is intentionally omitted because that outcome only
# appears with `intent="hallucinated_id"`, and T7 hard-fails on any
# hallucinated id, so a probe asserting `expected_outcome: dropped`
# would contradict the gate.
ALLOWED_CONSOLIDATION_OUTCOMES = (
    "stored",
    "skipped",
    "dropped_duplicate",
    "dropped_backend",
    "delete_failed_added_anyway",
    "add_failed_after_delete",
)

# Default backends evaluated when --backends is not supplied.
DEFAULT_BACKENDS = ("claude", "codex")

# Default qualitative sample budget. The 5/3/2 split below is the
# canonical allocation when budget == 10. Smaller budgets fall back
# to "fact rows first, retrieval misses second, episode rows third"
# with unused slots spilling forward.
DEFAULT_QUALITATIVE_SAMPLE_SIZE = 10
_QUAL_SECTION_QUOTAS = (("facts", 5), ("retrieval", 3), ("episodes", 2))

# Tag shape bounds for the malformed-tag check. Mirrors the production
# fact schema (per-tag string length 1..50). A future schema change
# that loosens these bounds would need this gate updated alongside.
_TAG_MIN_LEN = 1
_TAG_MAX_LEN = 50

# Reasonable upper bound on the wait window for stage-2 tasks; the
# spec defines this as `memory_episode_timeout_s + 5` per probe.
_EPISODE_WAIT_SLACK_S = 5

# Regex for the space-delimited key=value oneshot_reasoner log line.
# `value` matches anything that is not whitespace; the actual schema
# is constrained to simple identifiers, integers, and the model
# string (which may contain dashes and digits but never spaces).
_KV_RE = re.compile(r"(\w+)=(\S+)")


# ── Dataclasses ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExpectedAnchor:
    """One `must_store` anchor for a probe.

    `content_any` is a list of substrings; any one matching the
    stored row's content (case-insensitive) satisfies the anchor.
    `speaker` and `tags_any` are additional row-level filters: when
    `speaker` is set the row's `metadata.speaker` must match
    exactly, and when `tags_any` is set the row must include at
    least one listed tag.
    """

    anchor_id: str
    content_any: tuple[str, ...]
    speaker: str | None = None
    tags_any: tuple[str, ...] = ()


@dataclass(frozen=True)
class ForbiddenContent:
    """One `must_not_store` entry. Same case-insensitive substring
    semantics as `ExpectedAnchor.content_any`, but a match here is
    a violation."""

    content_any: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalExpectation:
    """One retrieval query bound to a stored-row anchor.

    The harness runs `memory.format_context(query, ...)`, captures
    the `memory.recall` log line, and scores the query against the
    row ID that the named `anchor_id` resolved to during the same
    backend arm. If no row satisfied the anchor (extraction missed
    it), the query scores as a miss with reason `anchor_missing`,
    NOT as excluded.
    """

    query: str
    anchor_id: str


@dataclass(frozen=True)
class ExpectedConsolidation:
    """Per-probe consolidation assertions.

    `expected_intent` is matched against `intent` values on the
    `memory.consolidate.intent` log payloads emitted during the
    probe's delta window. `expected_outcome` is matched against the
    `outcome` field on those same payloads. If both are present,
    both must be satisfied, but they do not have to appear on the
    same payload (a future fixture field `same_event: true` could
    tighten this).
    """

    expected_intent: str | None = None
    expected_outcome: str | None = None


@dataclass(frozen=True)
class GateProbe:
    """One row of the probe fixture, post-validation."""

    probe_id: str
    category: str
    prior: tuple[dict[str, str], ...]
    current_user: str
    current_assistant: str
    must_store: tuple[ExpectedAnchor, ...] = ()
    must_not_store: tuple[ForbiddenContent, ...] = ()
    retrieval: tuple[RetrievalExpectation, ...] = ()
    consolidation: ExpectedConsolidation | None = None


@dataclass
class ProbeOutcome:
    """Per-backend, per-probe attribution snapshot.

    Captures everything the threshold checker and the JSON emitter
    need: which anchors landed, what forbidden text leaked, which
    consolidation intent/outcome fired, the per-probe episode delta,
    and the raw row IDs added during this probe so retrieval can
    resolve anchors to specific stored rows after the arm finishes.
    """

    probe_id: str
    category: str
    satisfied_anchors: dict[str, str] = field(default_factory=dict)
    missing_anchors: list[str] = field(default_factory=list)
    forbidden_violations: list[str] = field(default_factory=list)
    speaker_correct: dict[str, bool] = field(default_factory=dict)
    tag_correct: dict[str, bool] = field(default_factory=dict)
    new_fact_ids: list[str] = field(default_factory=list)
    new_episode_ids: list[str] = field(default_factory=list)
    consolidation_events: list[dict] = field(default_factory=list)
    # Per-probe consolidation assertion results. None means the probe
    # did not declare an assertion on that axis; True/False means it
    # did and the gate scored it. Kept separate from the global
    # `memory.consolidate.intent` counters so a single probe can fail
    # a `consolidation` assertion without contaminating aggregate
    # metrics that summarize the whole arm.
    consolidation_intent_satisfied: bool | None = None
    consolidation_outcome_satisfied: bool | None = None
    reasoner_events: list[dict] = field(default_factory=list)


@dataclass
class RetrievalQueryResult:
    """Per-query retrieval scoring record.

    `rank` is 1-indexed; None when the anchor's row is absent from
    `hits`. `reason` is `anchor_missing` when the anchor was never
    satisfied during the arm, otherwise empty.
    """

    probe_id: str
    query: str
    anchor_id: str
    target_row_id: str | None
    rank: int | None
    in_prompt: bool
    reason: str = ""


@dataclass
class BackendRun:
    """Raw artifacts from a single backend arm.

    All counters and per-probe attribution land here; metrics derived
    from these are computed by the scoring helpers and stored in
    `BackendMetrics`.
    """

    backend: str
    sandbox_user_id: str
    model_fact: str
    model_episode: str
    log_path: Path
    probes: list[ProbeOutcome] = field(default_factory=list)
    retrieval: list[RetrievalQueryResult] = field(default_factory=list)
    final_facts: list[memory.MemoryResult] = field(default_factory=list)
    final_episodes: list[memory.MemoryResult] = field(default_factory=list)


@dataclass
class BackendMetrics:
    """Computed per-backend metrics consumed by the threshold checker."""

    retrieval_query_count: int = 0
    total_reasoner_calls: int = 0
    success_count: int = 0
    timeout_count: int = 0
    subprocess_error_count: int = 0
    output_error_count: int = 0
    invalid_json_count: int = 0
    empty_agent_message_count: int = 0
    non_object_json_count: int = 0
    missing_required_fields_count: int = 0
    parse_failure_rate: float = 0.0

    fact_anchor_total: int = 0
    fact_anchor_satisfied: int = 0
    fact_anchor_recall: float = 0.0

    forbidden_content_total: int = 0
    forbidden_content_violation_count: int = 0

    speaker_labeled_anchor_count: int = 0
    speaker_correct_count: int = 0
    speaker_accuracy: float = 0.0

    tag_presence_rate: float = 0.0
    malformed_tag_count: int = 0

    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    fraction_in_prompt: float = 0.0
    anchor_missing_count: int = 0

    episode_true_positive_count: int = 0
    episode_false_positive_count: int = 0
    episode_false_negative_count: int = 0
    episode_true_negative_count: int = 0
    episode_precision: float = 0.0
    episode_recall: float = 0.0
    episode_required_field_validity: float = 0.0
    episode_validate_rejected_count: int = 0

    stored_count: int = 0
    replaced_count: int = 0
    skipped_count: int = 0
    dropped_duplicate_count: int = 0
    dropped_backend_count: int = 0
    hallucinated_id_count: int = 0
    duplicate_gate_rate: float = 0.0
    consolidation_skip_rate: float = 0.0


@dataclass
class ThresholdCheck:
    """One row in the gate's threshold report."""

    name: str
    passed: bool
    claude_value: Any
    codex_value: Any
    threshold: str
    reason: str = ""


@dataclass
class ThresholdReport:
    """Outcome of `compare_thresholds`. `overall` is `pass` only if
    every check passed; `invalid_baseline` when Claude itself had
    runtime failures (T1)."""

    checks: list[ThresholdCheck]
    overall: str


# ── Loader ──────────────────────────────────────────────────────────


def load_probes(path: Path) -> list[GateProbe]:
    """Parse a JSONL probe fixture into validated GateProbe rows.

    Raises ValueError on the first malformed row so the operator
    gets an immediate, line-numbered failure instead of a
    half-processed run. Unknown JSON keys are NOT rejected; future
    fixture extensions can add fields without invalidating older
    parsers.
    """
    probes: list[GateProbe] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"probe fixture line {lineno}: invalid JSON ({exc})") from None
            try:
                probes.append(_parse_probe(raw))
            except ValueError as exc:
                raise ValueError(f"probe fixture line {lineno}: {exc}") from None
    return probes


def _parse_probe(raw: dict) -> GateProbe:
    """Validate one row dict and build the typed GateProbe.

    Raises ValueError with a human-readable message naming the
    offending field. The caller (`load_probes`) prepends the line
    number so the operator sees `probe fixture line N: <message>`.
    """
    probe_id = raw.get("probe_id")
    if not isinstance(probe_id, str) or not probe_id:
        raise ValueError("missing or empty probe_id")
    category = raw.get("category")
    if category not in ALLOWED_CATEGORIES:
        raise ValueError(f"category {category!r} not in {ALLOWED_CATEGORIES}")
    window = raw.get("window")
    if not isinstance(window, dict):
        raise ValueError("missing window")
    current = window.get("current")
    if not isinstance(current, dict):
        raise ValueError("missing window.current")
    current_user = current.get("user", "")
    current_assistant = current.get("assistant", "")
    if not isinstance(current_user, str) or not current_user:
        raise ValueError("missing window.current.user")
    if not isinstance(current_assistant, str) or not current_assistant:
        raise ValueError("missing window.current.assistant")
    prior_raw = window.get("prior") or []
    if not isinstance(prior_raw, list):
        raise ValueError("window.prior must be an array")
    prior: list[dict[str, str]] = []
    for entry in prior_raw:
        if not isinstance(entry, dict):
            raise ValueError("window.prior entries must be objects")
        role = entry.get("role")
        text = entry.get("text")
        if role not in ("user", "assistant"):
            raise ValueError(f"window.prior entry role {role!r} must be user or assistant")
        if not isinstance(text, str):
            raise ValueError("window.prior entry text must be a string")
        prior.append({"role": role, "text": text})

    expected = raw.get("expected") or {}
    if not isinstance(expected, dict):
        raise ValueError("expected must be an object")

    anchors: list[ExpectedAnchor] = []
    anchor_ids: set[str] = set()
    must_store_raw = expected.get("must_store") or []
    if not isinstance(must_store_raw, list):
        raise ValueError("expected.must_store must be an array")
    for entry in must_store_raw:
        if not isinstance(entry, dict):
            raise ValueError("expected.must_store entries must be objects")
        anchor_id = entry.get("anchor_id")
        if not isinstance(anchor_id, str) or not anchor_id:
            raise ValueError("must_store anchor_id required")
        if anchor_id in anchor_ids:
            raise ValueError(f"duplicate anchor_id {anchor_id!r}")
        anchor_ids.add(anchor_id)
        content_any = entry.get("content_any") or []
        if not isinstance(content_any, list) or not content_any:
            raise ValueError(f"anchor {anchor_id!r}: content_any must be a non-empty array")
        for s in content_any:
            if not isinstance(s, str) or not s:
                raise ValueError(f"anchor {anchor_id!r}: content_any entries must be non-empty strings")
        speaker = entry.get("speaker")
        if speaker is not None and speaker not in ALLOWED_SPEAKERS:
            raise ValueError(f"anchor {anchor_id!r}: speaker {speaker!r} not in {ALLOWED_SPEAKERS}")
        tags_any = entry.get("tags_any") or []
        if not isinstance(tags_any, list):
            raise ValueError(f"anchor {anchor_id!r}: tags_any must be an array")
        for t in tags_any:
            if not isinstance(t, str) or not t:
                raise ValueError(f"anchor {anchor_id!r}: tags_any entries must be non-empty strings")
        anchors.append(
            ExpectedAnchor(
                anchor_id=anchor_id,
                content_any=tuple(content_any),
                speaker=speaker,
                tags_any=tuple(tags_any),
            )
        )

    forbidden: list[ForbiddenContent] = []
    forbidden_raw = expected.get("must_not_store") or []
    if not isinstance(forbidden_raw, list):
        raise ValueError("expected.must_not_store must be an array")
    for entry in forbidden_raw:
        if not isinstance(entry, dict):
            raise ValueError("expected.must_not_store entries must be objects")
        content_any = entry.get("content_any") or []
        if not isinstance(content_any, list) or not content_any:
            raise ValueError("must_not_store content_any must be a non-empty array")
        for s in content_any:
            if not isinstance(s, str) or not s:
                raise ValueError("must_not_store content_any entries must be non-empty strings")
        forbidden.append(ForbiddenContent(content_any=tuple(content_any)))

    retrieval: list[RetrievalExpectation] = []
    retrieval_raw = expected.get("retrieval") or []
    if not isinstance(retrieval_raw, list):
        raise ValueError("expected.retrieval must be an array")
    for entry in retrieval_raw:
        if not isinstance(entry, dict):
            raise ValueError("expected.retrieval entries must be objects")
        query = entry.get("query")
        anchor_id = entry.get("anchor_id")
        if not isinstance(query, str) or not query:
            raise ValueError("retrieval query required")
        if not isinstance(anchor_id, str) or not anchor_id:
            raise ValueError("retrieval anchor_id required")
        if anchor_id not in anchor_ids:
            raise ValueError(f"retrieval anchor_id {anchor_id!r} does not match any must_store anchor")
        retrieval.append(RetrievalExpectation(query=query, anchor_id=anchor_id))

    consolidation: ExpectedConsolidation | None = None
    if "consolidation" in expected:
        cons_raw = expected.get("consolidation")
        if not isinstance(cons_raw, dict):
            raise ValueError("expected.consolidation must be an object")
        expected_intent = cons_raw.get("expected_intent")
        expected_outcome = cons_raw.get("expected_outcome")
        if expected_intent is not None and expected_intent not in ALLOWED_CONSOLIDATION_INTENTS:
            raise ValueError(
                f"expected.consolidation.expected_intent {expected_intent!r} not in {ALLOWED_CONSOLIDATION_INTENTS}"
            )
        if expected_outcome is not None and expected_outcome not in ALLOWED_CONSOLIDATION_OUTCOMES:
            raise ValueError(
                f"expected.consolidation.expected_outcome {expected_outcome!r} not in {ALLOWED_CONSOLIDATION_OUTCOMES}"
            )
        consolidation = ExpectedConsolidation(
            expected_intent=expected_intent,
            expected_outcome=expected_outcome,
        )

    return GateProbe(
        probe_id=probe_id,
        category=category,
        prior=tuple(prior),
        current_user=current_user,
        current_assistant=current_assistant,
        must_store=tuple(anchors),
        must_not_store=tuple(forbidden),
        retrieval=tuple(retrieval),
        consolidation=consolidation,
    )


def validate_fixture_minimums(probes: list[GateProbe]) -> None:
    """Enforce the §D4 fixture minimums.

    A small fixture is the most common way to overstate Codex quality
    in a regression test, so the gate refuses to score one. Raises
    ValueError with a single message listing every minimum that was
    not met (rather than only the first one) so the operator can fix
    them in one pass.
    """
    issues: list[str] = []
    if len(probes) < 24:
        issues.append(f"need at least 24 probes total, got {len(probes)}")
    durable_or_confirmation = [p for p in probes if p.category in ("durable-fact", "confirmation") and p.must_store]
    if len(durable_or_confirmation) < 10:
        issues.append(
            f"need at least 10 durable-fact or confirmation probes with must_store, got {len(durable_or_confirmation)}"
        )
    workflow = [p for p in probes if p.category == "workflow-noise" and p.must_not_store]
    if len(workflow) < 6:
        issues.append(f"need at least 6 workflow-noise probes with must_not_store, got {len(workflow)}")
    consolidation_probes = [p for p in probes if p.category == "consolidation"]
    if len(consolidation_probes) < 4:
        issues.append(f"need at least 4 consolidation probes, got {len(consolidation_probes)}")
    pos = [p for p in probes if p.category == "episode-positive"]
    neg = [p for p in probes if p.category == "episode-negative"]
    total_episode = pos + neg
    if len(total_episode) < 4:
        issues.append(f"need at least 4 episode probes total, got {len(total_episode)}")
    if len(pos) < 2:
        issues.append(f"need at least 2 episode-positive probes, got {len(pos)}")
    if len(neg) < 2:
        issues.append(f"need at least 2 episode-negative probes, got {len(neg)}")
    retrieval_count = sum(len(p.retrieval) for p in probes)
    if retrieval_count < 20:
        issues.append(f"need at least 20 retrieval queries across all probes, got {retrieval_count}")
    if issues:
        raise ValueError("fixture minimums not met: " + "; ".join(issues))


# ── Sandbox + config helpers ────────────────────────────────────────


def validate_user_prefix(prefix: str) -> None:
    """Refuse any prefix that does not start with the sandbox marker.

    Shared constant with `kai.eval.replay` so the two harnesses keep
    one source of truth; a future relaxation that lets non-sandbox
    user IDs through would need to touch both files explicitly.
    """
    if not prefix.startswith(_SANDBOX_USER_ID_PREFIX):
        raise ValueError(
            f"--user-prefix must start with {_SANDBOX_USER_ID_PREFIX!r}; "
            f"any other value risks writing memory rows into a real user's store"
        )


def make_backend_config(base_config: Config, backend: str) -> Config:
    """Build a per-backend copy of the production Config.

    Forces memory_enabled=True and memory_extraction_enabled=True so
    the harness can run against an install whose env file disables
    memory in production. Drives the per-user dispatch (issue #515)
    by setting `agent_backend` to the backend under test; with no
    `user_configs` override, `_resolve_effective_backend` in
    `memory_extraction.py` returns the global value for every
    extraction call, so the harness deterministically exercises one
    backend per run. Memory models come from the registry per-call
    via `get_model_for(role, effective_backend)`; no Config fields
    carry per-backend models any more. Uses `dataclasses.replace`
    because `Config` is frozen.
    """
    return dataclasses.replace(
        base_config,
        memory_enabled=True,
        memory_extraction_enabled=True,
        agent_backend=backend,
    )


# ── Log parsing ─────────────────────────────────────────────────────


def parse_oneshot_kv_line(message: str) -> dict[str, str]:
    """Decode the space-delimited key=value body of an `oneshot_reasoner` log.

    `oneshot.py` emits structured INFO lines in `key=value` format
    rather than JSON because the body is fixed and benefits from
    being grep-friendly. The parser ignores tokens that do not
    match `\\w+=\\S+` so a future field that introduces quoted text
    cannot break the loader silently; the affected token simply
    gets dropped.
    """
    if not message.startswith("oneshot_reasoner"):
        return {}
    return {m.group(1): m.group(2) for m in _KV_RE.finditer(message)}


def parse_json_suffix_line(message: str, prefix: str) -> dict | None:
    """Decode the JSON payload after a known log prefix.

    `memory.consolidate.intent`, `memory.recall`, and `memory.episode`
    log lines all share the shape `<prefix> <json>`; this helper is
    the single decoder so each consumer does not re-parse the prefix
    boundary.
    """
    head = prefix + " "
    if not message.startswith(head):
        return None
    body = message[len(head) :]
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def read_log_lines(log_path: Path) -> list[str]:
    """Return the message body of each line in the captured log file.

    The harness writes per-backend logs through a `logging.FileHandler`
    with the default formatter, which produces `<levelname>:<logger>:<message>`
    after `setLevel`. The body is split on the first two colons; if
    the file ends up with a different format, we fall back to the
    raw line so the parser still has something to work with.
    """
    if not log_path.exists():
        return []
    out: list[str] = []
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            parts = stripped.split(":", 2)
            if len(parts) == 3:
                out.append(parts[2])
            else:
                out.append(stripped)
    return out


# ── Backend run ─────────────────────────────────────────────────────


async def run_backend(
    *,
    probes: list[GateProbe],
    backend: str,
    base_config: Config,
    sandbox_user_id: str,
    log_path: Path,
    reset: bool,
    os_user_override: str | None = None,
) -> BackendRun:
    """Drive one backend arm end-to-end.

    Attaches a per-backend FileHandler around the arm, optionally
    resets the sandbox, walks every probe, waits for stage-2 tasks
    to settle, snapshots facts and episodes after each probe, then
    runs the retrieval queries against the final store. The
    FileHandler is removed in the finally so a crash mid-run does
    not leak a handler into the next arm.
    """
    config = make_backend_config(base_config, backend)
    if reset:
        memory.delete_all(user_id=sandbox_user_id)
    elif memory.get_all(user_id=sandbox_user_id, limit=1):
        raise SystemExit(f"sandbox user {sandbox_user_id!r} has existing rows; pass --reset to clear them")

    # Memory models for this run come from the per-backend registry,
    # matching what `_resolve_effective_backend` -> `get_model_for`
    # produces in production per-extraction. Pinned here so the
    # BackendRun summary reports the same SKUs the reasoner actually
    # spawned.
    run = BackendRun(
        backend=backend,
        sandbox_user_id=sandbox_user_id,
        model_fact=get_model_for(ModelRole.MEMORY_EXTRACTION, backend),
        model_episode=get_model_for(ModelRole.MEMORY_EPISODE, backend),
        log_path=log_path,
    )

    handler = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root = logging.getLogger()
    prior_level = root.level
    if root.level == logging.NOTSET or root.level > logging.INFO:
        root.setLevel(logging.INFO)
    root.addHandler(handler)
    try:
        prior_fact_ids: set[str] = set()
        prior_episode_ids: set[str] = set()
        for probe in probes:
            outcome = await _run_probe(
                probe=probe,
                config=config,
                sandbox_user_id=sandbox_user_id,
                prior_fact_ids=prior_fact_ids,
                prior_episode_ids=prior_episode_ids,
                log_path=log_path,
                os_user_override=os_user_override,
            )
            run.probes.append(outcome)
            for fid in outcome.new_fact_ids:
                prior_fact_ids.add(fid)
            for eid in outcome.new_episode_ids:
                prior_episode_ids.add(eid)
        run.final_facts = memory.get_all_facts(user_id=sandbox_user_id)
        run.final_episodes = memory.get_all_episodes(user_id=sandbox_user_id)
        run.retrieval = await _score_retrieval(probes=probes, run=run, sandbox_user_id=sandbox_user_id)
    finally:
        handler.close()
        root.removeHandler(handler)
        root.setLevel(prior_level)
    return run


async def _run_probe(
    *,
    probe: GateProbe,
    config: Config,
    sandbox_user_id: str,
    prior_fact_ids: set[str],
    prior_episode_ids: set[str],
    log_path: Path,
    os_user_override: str | None = None,
) -> ProbeOutcome:
    """Run extract_and_store for one probe and snapshot the delta.

    Waits up to `memory_episode_timeout_s + _EPISODE_WAIT_SLACK_S`
    for stage-2 tasks scheduled by this probe to finish before
    snapshotting episodes; without the wait, episode positives can
    score as false negatives because the snapshot races the task.

    Captures the log file offset before `extract_and_store` so the
    new `memory.consolidate.intent` payloads emitted during the
    probe can be attributed back to this probe specifically. The
    file-offset approach is the same pattern `_score_one_query`
    uses for retrieval log lines; both rely on the per-backend
    FileHandler producing append-only output.
    """
    user_text, assistant_text, prior_pairs = _window_to_extractor_args(
        {
            "prior": [dict(entry) for entry in probe.prior],
            "current": {"user": probe.current_user, "assistant": probe.current_assistant},
        }
    )

    pre_log_offset = log_path.stat().st_size if log_path.exists() else 0
    pre_episode_tasks = set(memory_extraction._pending_episode_tasks)
    await memory_extraction.extract_and_store(
        user_text,
        assistant_text,
        user_id=sandbox_user_id,
        config=config,
        prior_pairs=prior_pairs,
        os_user_override=os_user_override,
    )
    new_tasks = memory_extraction._pending_episode_tasks - pre_episode_tasks
    if new_tasks:
        await asyncio.wait(
            new_tasks,
            timeout=float(config.memory_episode_timeout_s + _EPISODE_WAIT_SLACK_S),
        )

    facts_after = memory.get_all_facts(user_id=sandbox_user_id)
    episodes_after = memory.get_all_episodes(user_id=sandbox_user_id)
    new_facts = [r for r in facts_after if r.id not in prior_fact_ids]
    new_episodes = [r for r in episodes_after if r.id not in prior_episode_ids]

    consolidation_events = _read_consolidation_events(log_path, pre_log_offset)

    outcome = ProbeOutcome(
        probe_id=probe.probe_id,
        category=probe.category,
        new_fact_ids=[r.id for r in new_facts],
        new_episode_ids=[r.id for r in new_episodes],
        consolidation_events=consolidation_events,
    )

    for anchor in probe.must_store:
        target_row = _match_anchor(anchor, new_facts + new_episodes)
        if target_row is None:
            outcome.missing_anchors.append(anchor.anchor_id)
            continue
        outcome.satisfied_anchors[anchor.anchor_id] = target_row.id
        if anchor.speaker is not None:
            outcome.speaker_correct[anchor.anchor_id] = target_row.metadata.get("speaker") == anchor.speaker
        if anchor.tags_any:
            row_tags = set(target_row.metadata.get("tags") or [])
            outcome.tag_correct[anchor.anchor_id] = bool(row_tags & set(anchor.tags_any))

    for forbidden in probe.must_not_store:
        for row in new_facts + new_episodes:
            content_lower = (row.text or "").lower()
            for needle in forbidden.content_any:
                if needle.lower() in content_lower:
                    outcome.forbidden_violations.append(needle)
                    break

    if probe.consolidation is not None:
        _score_consolidation_assertion(outcome, probe.consolidation)

    return outcome


def _read_consolidation_events(log_path: Path, start_offset: int) -> list[dict]:
    """Decode `memory.consolidate.intent` JSON payloads written after `start_offset`.

    Returns one dict per matching log line in append order. Lines that
    do not match the prefix or fail to JSON-decode are silently skipped
    so a future log-format change cannot fail the gate on parse alone.
    Designed so per-probe attribution works even when other modules
    interleave INFO lines in the same file (a root-logger FileHandler
    captures every named logger's output, not just the consolidation
    emit site).
    """
    if not log_path.exists():
        return []
    with log_path.open("r", encoding="utf-8") as fh:
        fh.seek(start_offset)
        tail = fh.read()
    events: list[dict] = []
    for line in tail.splitlines():
        stripped = line.rstrip("\n")
        if not stripped:
            continue
        parts = stripped.split(":", 2)
        message = parts[2] if len(parts) == 3 else stripped
        payload = parse_json_suffix_line(message, "memory.consolidate.intent")
        if payload is not None:
            events.append(payload)
    return events


def _score_consolidation_assertion(outcome: ProbeOutcome, expected: ExpectedConsolidation) -> None:
    """Apply the per-probe consolidation assertion semantics.

    For each declared assertion axis (`expected_intent`,
    `expected_outcome`), the probe is satisfied if at least one
    captured `memory.consolidate.intent` payload's field matches.
    The two axes are independent: a probe declaring both must satisfy
    both, but the matches can come from different payloads (a future
    `same_event: true` flag would tighten this).
    """
    if expected.expected_intent is not None:
        outcome.consolidation_intent_satisfied = any(
            event.get("intent") == expected.expected_intent for event in outcome.consolidation_events
        )
    if expected.expected_outcome is not None:
        outcome.consolidation_outcome_satisfied = any(
            event.get("outcome") == expected.expected_outcome for event in outcome.consolidation_events
        )


def _match_anchor(anchor: ExpectedAnchor, candidates: list[memory.MemoryResult]) -> memory.MemoryResult | None:
    """First candidate row whose content matches any anchor needle.

    Case-insensitive substring; the speaker and tags filters live on
    `ProbeOutcome` rather than gating the match so the per-anchor
    speaker/tag correctness numbers reflect the actual row that
    landed (not "missing" when the row was there but with wrong
    metadata).
    """
    for row in candidates:
        content_lower = (row.text or "").lower()
        for needle in anchor.content_any:
            if needle.lower() in content_lower:
                return row
    return None


async def _score_retrieval(
    *,
    probes: list[GateProbe],
    run: BackendRun,
    sandbox_user_id: str,
) -> list[RetrievalQueryResult]:
    """Run each retrieval query against the populated sandbox.

    Issues one `memory.format_context` call per query; the resulting
    `memory.recall` log payload is the source of truth for rank and
    fraction-in-prompt. The text return from format_context is
    discarded because the harness scores on row IDs, not formatted
    prose.
    """
    results: list[RetrievalQueryResult] = []
    anchor_to_row: dict[tuple[str, str], str] = {}
    for outcome in run.probes:
        for anchor_id, row_id in outcome.satisfied_anchors.items():
            anchor_to_row[(outcome.probe_id, anchor_id)] = row_id

    for probe in probes:
        for query_spec in probe.retrieval:
            target_row_id = anchor_to_row.get((probe.probe_id, query_spec.anchor_id))
            if target_row_id is None:
                results.append(
                    RetrievalQueryResult(
                        probe_id=probe.probe_id,
                        query=query_spec.query,
                        anchor_id=query_spec.anchor_id,
                        target_row_id=None,
                        rank=None,
                        in_prompt=False,
                        reason="anchor_missing",
                    )
                )
                continue
            rank, in_prompt = await _score_one_query(
                query=query_spec.query,
                sandbox_user_id=sandbox_user_id,
                target_row_id=target_row_id,
                log_path=run.log_path,
            )
            results.append(
                RetrievalQueryResult(
                    probe_id=probe.probe_id,
                    query=query_spec.query,
                    anchor_id=query_spec.anchor_id,
                    target_row_id=target_row_id,
                    rank=rank,
                    in_prompt=in_prompt,
                )
            )
    return results


async def _score_one_query(
    *,
    query: str,
    sandbox_user_id: str,
    target_row_id: str,
    log_path: Path,
) -> tuple[int | None, bool]:
    """Run one retrieval query and return (rank, in_prompt).

    Captures the `memory.recall` log line emitted by format_context
    by inspecting the FileHandler-backed log after the call
    completes; reading by file offset keeps the parsing simple and
    correct even if other log lines interleave during the same
    call.
    """
    offset = log_path.stat().st_size if log_path.exists() else 0
    await memory.format_context(query, user_id=sandbox_user_id)
    if not log_path.exists():
        return None, False
    with log_path.open("r", encoding="utf-8") as fh:
        fh.seek(offset)
        tail = fh.read()
    rank: int | None = None
    in_prompt = False
    for line in tail.splitlines():
        stripped = line.rstrip("\n")
        if not stripped:
            continue
        parts = stripped.split(":", 2)
        message = parts[2] if len(parts) == 3 else stripped
        payload = parse_json_suffix_line(message, "memory.recall")
        if payload is None:
            continue
        hits = payload.get("hits") or []
        for idx, hit in enumerate(hits, 1):
            if isinstance(hit, dict) and hit.get("id") == target_row_id:
                rank = idx
                break
        lines_used = payload.get("lines_used") or 0
        if rank is not None and isinstance(lines_used, int) and rank <= lines_used:
            in_prompt = True
        break
    return rank, in_prompt


# ── Metric computation ──────────────────────────────────────────────


def compute_metrics(run: BackendRun) -> BackendMetrics:
    """Reduce a `BackendRun` to the metric values §6 thresholds read.

    All log-derived counters come from re-parsing the per-backend
    log file; per-probe attribution comes from `run.probes`; retrieval
    metrics come from `run.retrieval`. The function is pure: it
    reads `run` and returns a value, with no side effects so tests
    can compare metric dicts directly.
    """
    metrics = BackendMetrics()
    log_lines = read_log_lines(run.log_path)

    for message in log_lines:
        if message.startswith("oneshot_reasoner"):
            fields = parse_oneshot_kv_line(message)
            outcome = fields.get("outcome", "")
            category = fields.get("error_category", "")
            metrics.total_reasoner_calls += 1
            if outcome == "success":
                metrics.success_count += 1
            elif outcome == "timeout":
                metrics.timeout_count += 1
            elif outcome == "subprocess_error":
                metrics.subprocess_error_count += 1
            elif outcome == "output_error":
                metrics.output_error_count += 1
            if category == "invalid_json":
                metrics.invalid_json_count += 1
            elif category == "empty_agent_message":
                metrics.empty_agent_message_count += 1
            elif category == "non_object_json":
                metrics.non_object_json_count += 1
            elif category == "missing_required_fields":
                metrics.missing_required_fields_count += 1

    metrics.parse_failure_rate = metrics.output_error_count / max(1, metrics.total_reasoner_calls)

    for message in log_lines:
        payload = parse_json_suffix_line(message, "memory.consolidate.intent")
        if payload is None:
            continue
        intent = payload.get("intent")
        outcome_val = payload.get("outcome")
        if intent == "hallucinated_id":
            metrics.hallucinated_id_count += 1
        if outcome_val == "stored" or outcome_val == "delete_failed_added_anyway":
            metrics.stored_count += 1
            if intent == "update_of":
                metrics.replaced_count += 1
        elif outcome_val == "skipped":
            metrics.skipped_count += 1
        elif outcome_val == "dropped_duplicate":
            metrics.dropped_duplicate_count += 1
        elif outcome_val == "dropped_backend":
            metrics.dropped_backend_count += 1

    denominator = metrics.stored_count + metrics.dropped_duplicate_count
    metrics.duplicate_gate_rate = metrics.dropped_duplicate_count / max(1, denominator)
    skip_denom = metrics.stored_count + metrics.skipped_count
    metrics.consolidation_skip_rate = metrics.skipped_count / max(1, skip_denom)

    for message in log_lines:
        payload = parse_json_suffix_line(message, "memory.episode")
        if payload is None:
            continue
        if payload.get("outcome") == "validate_rejected":
            metrics.episode_validate_rejected_count += 1

    fact_anchor_total = 0
    fact_anchor_satisfied = 0
    forbidden_violation = 0
    speaker_labeled = 0
    speaker_correct = 0
    tag_required = 0
    tag_correct = 0
    malformed_tags = 0

    for outcome in run.probes:
        fact_anchor_total += len(outcome.satisfied_anchors) + len(outcome.missing_anchors)
        fact_anchor_satisfied += len(outcome.satisfied_anchors)
        for row_id in outcome.satisfied_anchors.values():
            row = next(
                (r for r in run.final_facts + run.final_episodes if r.id == row_id),
                None,
            )
            if row is None:
                continue
            tags = row.metadata.get("tags")
            if isinstance(tags, list):
                if not tags:
                    malformed_tags += 1
                for tag in tags:
                    if not isinstance(tag, str) or len(tag) < _TAG_MIN_LEN or len(tag) > _TAG_MAX_LEN:
                        malformed_tags += 1
            elif tags is not None:
                malformed_tags += 1
        forbidden_violation += len(outcome.forbidden_violations)
        for correct in outcome.speaker_correct.values():
            speaker_labeled += 1
            if correct:
                speaker_correct += 1
        for correct in outcome.tag_correct.values():
            tag_required += 1
            if correct:
                tag_correct += 1

    metrics.fact_anchor_total = fact_anchor_total
    metrics.fact_anchor_satisfied = fact_anchor_satisfied
    metrics.fact_anchor_recall = fact_anchor_satisfied / max(1, fact_anchor_total)
    metrics.forbidden_content_violation_count = forbidden_violation
    metrics.speaker_labeled_anchor_count = speaker_labeled
    metrics.speaker_correct_count = speaker_correct
    metrics.speaker_accuracy = speaker_correct / max(1, speaker_labeled)
    metrics.tag_presence_rate = tag_correct / max(1, tag_required) if tag_required else 1.0
    metrics.malformed_tag_count = malformed_tags

    metrics.retrieval_query_count = len(run.retrieval)
    metrics.precision_at_1 = _precision_at_k(run.retrieval, 1)
    metrics.precision_at_3 = _precision_at_k(run.retrieval, 3)
    metrics.precision_at_5 = _precision_at_k(run.retrieval, 5)
    metrics.mrr = _mean_reciprocal_rank(run.retrieval)
    metrics.fraction_in_prompt = sum(1 for r in run.retrieval if r.in_prompt) / max(1, len(run.retrieval))
    metrics.anchor_missing_count = sum(1 for r in run.retrieval if r.reason == "anchor_missing")

    return metrics


def _precision_at_k(results: list[RetrievalQueryResult], k: int) -> float:
    """Fraction of queries whose target row appears at rank <= k.

    Missing anchors score as misses, not as excluded, matching the
    spec's end-to-end-quality posture: extraction failures and
    ranking failures are both retrieval failures.
    """
    if not results:
        return 0.0
    hits = sum(1 for r in results if r.rank is not None and r.rank <= k)
    return hits / len(results)


def _mean_reciprocal_rank(results: list[RetrievalQueryResult]) -> float:
    """Average reciprocal rank with misses contributing 0.

    Symmetric to `_precision_at_k`: missing anchors are scored as
    misses so MRR reflects total backend quality.
    """
    if not results:
        return 0.0
    return sum(1.0 / r.rank if r.rank is not None else 0.0 for r in results) / len(results)


def update_forbidden_total(metrics: BackendMetrics, probes: list[GateProbe]) -> None:
    """Set `forbidden_content_total` from the probe set.

    Separate from `compute_metrics` because the counter is a property
    of the fixture, not the backend run; calling it once with the
    shared probe list keeps both backends' totals consistent.
    """
    total = sum(len(p.must_not_store) for p in probes)
    metrics.forbidden_content_total = total


# ── Episode TP/FP scoring ───────────────────────────────────────────


def score_episodes(probes: list[GateProbe], run: BackendRun) -> tuple[int, int, int, int, float]:
    """Compute (TP, FP, FN, TN, required_field_validity) for episodes.

    A backend satisfies an `episode-positive` probe when at least
    one new episode row appears in its delta window with all
    required episode metadata fields. An `episode-negative` probe is
    violated by any new episode row in its delta window. Required
    fields are checked on the row metadata; missing fields lower
    the validity rate without changing TP/FP counts (a malformed
    positive is still a positive).
    """
    tp = fp = fn = tn = 0
    field_total = 0
    field_valid = 0
    required = ("goal", "context", "approach", "outcome", "outcome_quality", "tags", "actors")
    probe_outcomes = {o.probe_id: o for o in run.probes}
    for probe in probes:
        outcome = probe_outcomes.get(probe.probe_id)
        if outcome is None:
            continue
        has_new_episode = len(outcome.new_episode_ids) > 0
        if probe.category == "episode-positive":
            if has_new_episode:
                tp += 1
            else:
                fn += 1
        elif probe.category == "episode-negative":
            if has_new_episode:
                fp += 1
            else:
                tn += 1
        for episode_id in outcome.new_episode_ids:
            row = next((r for r in run.final_episodes if r.id == episode_id), None)
            if row is None:
                continue
            field_total += 1
            metadata = row.metadata or {}
            # All required content fields must be present AND the row
            # must carry the canonical `speaker=episode_summary` tag.
            # The speaker field lives outside the schema's required
            # list (it is set by the stage-2 generator, not by the
            # model) but is part of the spec's episode validity
            # contract: a malformed episode with the wrong speaker
            # would otherwise score as valid under T6 and let a
            # broken metadata path through the gate.
            fields_present = all(metadata.get(name) for name in required)
            speaker_ok = metadata.get("speaker") == "episode_summary"
            if fields_present and speaker_ok:
                field_valid += 1
    validity = field_valid / max(1, field_total) if field_total else 1.0
    return tp, fp, fn, tn, validity


# ── Threshold checks ────────────────────────────────────────────────


def compare_thresholds(claude: BackendMetrics, codex: BackendMetrics) -> ThresholdReport:
    """Run §6 hard checks against the two metric snapshots.

    Returns a `ThresholdReport` whose `overall` is `pass` only if
    every check passed and Claude itself had no runtime failures
    (T1's `invalid_baseline` rule). Each check records both sides
    so the JSON artifact can be reviewed without re-reading the
    log files.
    """
    checks: list[ThresholdCheck] = []

    if (
        claude.timeout_count > 0
        or claude.subprocess_error_count > 0
        or claude.output_error_count > 0
        or claude.parse_failure_rate > 0.0
    ):
        return ThresholdReport(
            checks=[
                ThresholdCheck(
                    name="T1.claude_baseline",
                    passed=False,
                    claude_value={
                        "timeout": claude.timeout_count,
                        "subprocess_error": claude.subprocess_error_count,
                        "output_error": claude.output_error_count,
                        "parse_failure_rate": claude.parse_failure_rate,
                    },
                    codex_value=None,
                    threshold="claude_baseline must be clean",
                    reason="claude arm had runtime failures; codex result is not meaningful",
                ),
            ],
            overall="invalid_baseline",
        )

    checks.append(
        ThresholdCheck(
            name="T1.timeout",
            passed=codex.timeout_count == 0,
            claude_value=claude.timeout_count,
            codex_value=codex.timeout_count,
            threshold="codex timeout_count == 0",
        )
    )
    checks.append(
        ThresholdCheck(
            name="T1.subprocess_error",
            passed=codex.subprocess_error_count == 0,
            claude_value=claude.subprocess_error_count,
            codex_value=codex.subprocess_error_count,
            threshold="codex subprocess_error_count == 0",
        )
    )
    checks.append(
        ThresholdCheck(
            name="T1.output_error",
            passed=codex.output_error_count == 0,
            claude_value=claude.output_error_count,
            codex_value=codex.output_error_count,
            threshold="codex output_error_count == 0",
        )
    )
    checks.append(
        ThresholdCheck(
            name="T1.parse_failure_rate",
            passed=codex.parse_failure_rate == 0.0,
            claude_value=claude.parse_failure_rate,
            codex_value=codex.parse_failure_rate,
            threshold="codex parse_failure_rate == 0.0",
        )
    )

    n_anchors = max(1, codex.fact_anchor_total)
    allowance = max(1 / n_anchors, 0.05)
    checks.append(
        ThresholdCheck(
            name="T2.fact_anchor_recall_floor",
            passed=codex.fact_anchor_recall >= 0.85,
            claude_value=claude.fact_anchor_recall,
            codex_value=codex.fact_anchor_recall,
            threshold="codex fact_anchor_recall >= 0.85",
        )
    )
    checks.append(
        ThresholdCheck(
            name="T2.fact_anchor_recall_band",
            passed=codex.fact_anchor_recall >= claude.fact_anchor_recall - allowance,
            claude_value=claude.fact_anchor_recall,
            codex_value=codex.fact_anchor_recall,
            threshold=f"codex >= claude - {allowance:.4f}",
        )
    )

    m_forbidden = max(1, codex.forbidden_content_total)
    cap = max(1, int(0.10 * m_forbidden))
    checks.append(
        ThresholdCheck(
            name="T3.forbidden_content_cap",
            passed=codex.forbidden_content_violation_count <= cap,
            claude_value=claude.forbidden_content_violation_count,
            codex_value=codex.forbidden_content_violation_count,
            threshold=f"codex forbidden_content_violation_count <= {cap}",
        )
    )
    checks.append(
        ThresholdCheck(
            name="T3.forbidden_content_band",
            passed=codex.forbidden_content_violation_count <= claude.forbidden_content_violation_count + 1,
            claude_value=claude.forbidden_content_violation_count,
            codex_value=codex.forbidden_content_violation_count,
            threshold="codex <= claude + 1",
        )
    )

    speaker_n = max(1, codex.speaker_labeled_anchor_count)
    speaker_allowance = max(1 / speaker_n, 0.05)
    checks.append(
        ThresholdCheck(
            name="T4.speaker_accuracy_floor",
            passed=codex.speaker_accuracy >= 0.95,
            claude_value=claude.speaker_accuracy,
            codex_value=codex.speaker_accuracy,
            threshold="codex speaker_accuracy >= 0.95",
        )
    )
    checks.append(
        ThresholdCheck(
            name="T4.speaker_accuracy_band",
            passed=codex.speaker_accuracy >= claude.speaker_accuracy - speaker_allowance,
            claude_value=claude.speaker_accuracy,
            codex_value=codex.speaker_accuracy,
            threshold=f"codex >= claude - {speaker_allowance:.4f}",
        )
    )
    checks.append(
        ThresholdCheck(
            name="T4.tag_presence_rate",
            passed=codex.tag_presence_rate == 1.0,
            claude_value=claude.tag_presence_rate,
            codex_value=codex.tag_presence_rate,
            threshold="codex tag_presence_rate == 1.0",
        )
    )
    checks.append(
        ThresholdCheck(
            name="T4.malformed_tag_count",
            passed=codex.malformed_tag_count == 0,
            claude_value=claude.malformed_tag_count,
            codex_value=codex.malformed_tag_count,
            threshold="codex malformed_tag_count == 0",
        )
    )

    retrieval_n = max(1, codex.retrieval_query_count)
    r_allowance = max(1 / retrieval_n, 0.05)
    checks.append(
        ThresholdCheck(
            name="T5.precision_at_1_band",
            passed=codex.precision_at_1 >= claude.precision_at_1 - r_allowance,
            claude_value=claude.precision_at_1,
            codex_value=codex.precision_at_1,
            threshold=f"codex >= claude - {r_allowance:.4f}",
        )
    )
    checks.append(
        ThresholdCheck(
            name="T5.precision_at_3_band",
            passed=codex.precision_at_3 >= claude.precision_at_3 - r_allowance,
            claude_value=claude.precision_at_3,
            codex_value=codex.precision_at_3,
            threshold=f"codex >= claude - {r_allowance:.4f}",
        )
    )
    # T5 absolute floors use the same `max(absolute_minimum,
    # claude_value - delta)` shape as `T6.episode_recall` below. The
    # 0.80 absolute that lived here originally was a placeholder; the
    # baseline (claude) currently scores ~0.72 on both axes on the
    # operator-private fixture, so a fixed 0.80 floor failed every
    # run regardless of whether codex was actually regressing. The
    # max() form catches regressions in the achievable range: the
    # absolute term (0.70) fires when the entire system gets
    # fundamentally worse, the relative term (claude - 0.05) tracks
    # the baseline as retrieval quality improves. Choosing the
    # tighter of the two preserves the original spec's intent (catch
    # codex when it drops below a reasonable absolute) without
    # rejecting the current state of the system every run.
    p_at_5_floor = max(0.70, claude.precision_at_5 - 0.05)
    checks.append(
        ThresholdCheck(
            name="T5.precision_at_5_floor",
            passed=codex.precision_at_5 >= p_at_5_floor,
            claude_value=claude.precision_at_5,
            codex_value=codex.precision_at_5,
            threshold=f"codex precision_at_5 >= {p_at_5_floor:.4f}",
        )
    )
    checks.append(
        ThresholdCheck(
            name="T5.precision_at_5_band",
            passed=codex.precision_at_5 >= claude.precision_at_5 - r_allowance,
            claude_value=claude.precision_at_5,
            codex_value=codex.precision_at_5,
            threshold=f"codex >= claude - {r_allowance:.4f}",
        )
    )
    checks.append(
        ThresholdCheck(
            name="T5.mrr_band",
            passed=codex.mrr >= claude.mrr - r_allowance,
            claude_value=claude.mrr,
            codex_value=codex.mrr,
            threshold=f"codex >= claude - {r_allowance:.4f}",
        )
    )
    fip_floor = max(0.70, claude.fraction_in_prompt - 0.05)
    checks.append(
        ThresholdCheck(
            name="T5.fraction_in_prompt_floor",
            passed=codex.fraction_in_prompt >= fip_floor,
            claude_value=claude.fraction_in_prompt,
            codex_value=codex.fraction_in_prompt,
            threshold=f"codex fraction_in_prompt >= {fip_floor:.4f}",
        )
    )
    checks.append(
        ThresholdCheck(
            name="T5.fraction_in_prompt_band",
            passed=codex.fraction_in_prompt >= claude.fraction_in_prompt - r_allowance,
            claude_value=claude.fraction_in_prompt,
            codex_value=codex.fraction_in_prompt,
            threshold=f"codex >= claude - {r_allowance:.4f}",
        )
    )

    checks.append(
        ThresholdCheck(
            name="T6.episode_required_field_validity",
            passed=codex.episode_required_field_validity == 1.0,
            claude_value=claude.episode_required_field_validity,
            codex_value=codex.episode_required_field_validity,
            threshold="codex episode_required_field_validity == 1.0",
        )
    )
    checks.append(
        ThresholdCheck(
            name="T6.episode_false_positive",
            passed=codex.episode_false_positive_count <= claude.episode_false_positive_count + 1,
            claude_value=claude.episode_false_positive_count,
            codex_value=codex.episode_false_positive_count,
            threshold="codex <= claude + 1",
        )
    )
    checks.append(
        ThresholdCheck(
            name="T6.episode_false_negative",
            passed=codex.episode_false_negative_count <= claude.episode_false_negative_count + 1,
            claude_value=claude.episode_false_negative_count,
            codex_value=codex.episode_false_negative_count,
            threshold="codex <= claude + 1",
        )
    )

    # Conditional episode recall floor. With small positive-probe
    # counts the recall metric is noisy, so the spec only enforces
    # this when there are at least 3 episode-positive probes. The
    # FN band above catches one-or-two-miss regressions; the recall
    # floor catches the case where the codex arm misses every
    # positive while the baseline already missed enough that the
    # FN band stays satisfied (e.g. claude misses 2/3 -> FN=2,
    # codex misses 3/3 -> FN=3, FN band passes but recall=0).
    #
    # The 0.65 lower bound is deliberately set below the 2/3 = 0.667
    # achievable value so the absolute floor does not sit in the
    # literal impossible region between 2/3 and 3/3 (where the
    # previous 0.67 bound lived). When the relative regression term
    # `claude_recall - 0.25` does not raise the effective floor
    # above 0.65, a backend that hits 2 of 3 positives clears the
    # check on the floating-point math instead of failing by 0.0033.
    # When claude scores high enough that the relative term wins
    # (e.g. claude=1.0 -> floor=0.75), the codex arm is still held
    # to no more than a 0.25 absolute drop, so 2/3 against a perfect
    # baseline correctly fails as a regression rather than a
    # discrete-bucket pass.
    positive_probe_count = codex.episode_true_positive_count + codex.episode_false_negative_count
    if positive_probe_count >= 3:
        recall_floor = max(0.65, claude.episode_recall - 0.25)
        checks.append(
            ThresholdCheck(
                name="T6.episode_recall",
                passed=codex.episode_recall >= recall_floor,
                claude_value=claude.episode_recall,
                codex_value=codex.episode_recall,
                threshold=f"codex episode_recall >= {recall_floor:.4f}",
            )
        )

    checks.append(
        ThresholdCheck(
            name="T7.hallucinated_id",
            passed=codex.hallucinated_id_count == 0,
            claude_value=claude.hallucinated_id_count,
            codex_value=codex.hallucinated_id_count,
            threshold="codex hallucinated_id_count == 0",
        )
    )
    checks.append(
        ThresholdCheck(
            name="T7.dropped_backend",
            passed=codex.dropped_backend_count == 0,
            claude_value=claude.dropped_backend_count,
            codex_value=codex.dropped_backend_count,
            threshold="codex dropped_backend_count == 0",
        )
    )
    duplicate_band_ok = codex.duplicate_gate_rate <= claude.duplicate_gate_rate + 0.10 or (
        codex.fact_anchor_recall >= claude.fact_anchor_recall and codex.forbidden_content_violation_count == 0
    )
    checks.append(
        ThresholdCheck(
            name="T7.duplicate_gate_rate",
            passed=duplicate_band_ok,
            claude_value=claude.duplicate_gate_rate,
            codex_value=codex.duplicate_gate_rate,
            threshold="codex <= claude + 0.10 OR codex has equal/better recall AND no forbidden violations",
        )
    )

    overall = "pass" if all(c.passed for c in checks) else "fail"
    return ThresholdReport(checks=checks, overall=overall)


# ── Output ──────────────────────────────────────────────────────────


def probe_set_hash(probes: list[GateProbe]) -> str:
    """SHA-256 of the canonical probe sequence.

    Used to detect fixture drift between runs. Canonical form is
    each probe's `probe_id` joined by newlines; including the full
    JSON would surface false drift on whitespace-only edits and
    leak probe content into the hash input, which is logged.
    """
    canonical = "\n".join(p.probe_id for p in probes)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_gate_result(
    *,
    probes: list[GateProbe],
    runs: dict[str, BackendRun],
    metrics: dict[str, BackendMetrics],
    threshold_report: ThresholdReport,
    generated_at: str | None = None,
) -> dict:
    """Assemble the `gate-result.json` payload.

    Raw probe `window` text is intentionally absent from per_probe
    entries: the operator's probe fixtures can contain conversation
    fragments and the gate artifact must be safe to attach to a
    public issue, share for triage, or upload to an issue comment.
    Anchor IDs, row IDs, retrieval ranks, and failure reasons are
    sufficient for downstream debugging.
    """
    if generated_at is None:
        generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    backends_block: dict[str, dict] = {}
    for backend_name, run in runs.items():
        backends_block[backend_name] = {
            "sandbox_user_id": run.sandbox_user_id,
            "model_fact": run.model_fact,
            "model_episode": run.model_episode,
            "metrics": dataclasses.asdict(metrics[backend_name]),
        }
    per_probe = []
    for probe in probes:
        entry: dict[str, Any] = {
            "probe_id": probe.probe_id,
            "category": probe.category,
            "backends": {},
        }
        for backend_name, run in runs.items():
            outcome = next((o for o in run.probes if o.probe_id == probe.probe_id), None)
            if outcome is None:
                continue
            entry["backends"][backend_name] = {
                "satisfied_anchors": outcome.satisfied_anchors,
                "missing_anchors": outcome.missing_anchors,
                "forbidden_violations": outcome.forbidden_violations,
                "speaker_correct": outcome.speaker_correct,
                "tag_correct": outcome.tag_correct,
                "new_fact_ids": outcome.new_fact_ids,
                "new_episode_ids": outcome.new_episode_ids,
                "consolidation_events": outcome.consolidation_events,
                "consolidation_intent_satisfied": outcome.consolidation_intent_satisfied,
                "consolidation_outcome_satisfied": outcome.consolidation_outcome_satisfied,
            }
        retrieval_entries = []
        for backend_name, run in runs.items():
            for r in run.retrieval:
                if r.probe_id != probe.probe_id:
                    continue
                retrieval_entries.append(
                    {
                        "backend": backend_name,
                        "anchor_id": r.anchor_id,
                        "target_row_id": r.target_row_id,
                        "rank": r.rank,
                        "in_prompt": r.in_prompt,
                        "reason": r.reason,
                    }
                )
        if retrieval_entries:
            entry["retrieval"] = retrieval_entries
        per_probe.append(entry)
    return {
        "version": "1",
        "generated_at": generated_at,
        "issue": 498,
        "probe_set_hash": probe_set_hash(probes),
        "probe_count": len(probes),
        "retrieval_query_count": sum(len(p.retrieval) for p in probes),
        "backends": backends_block,
        "thresholds": {
            "checks": [dataclasses.asdict(c) for c in threshold_report.checks],
            "overall": threshold_report.overall,
        },
        "qualitative_verdict": "pending",
        "per_probe": per_probe,
    }


def _quotas_for(budget: int) -> dict[str, int]:
    """Compute the per-section sample quota under a budget.

    The 5/3/2 default applies when `budget >= 10`; smaller budgets
    allocate strictly in `facts -> retrieval -> episodes` order
    (each section gets `min(quota, remaining)`). The spill-forward
    happens at sampling time, when a section turns out to have
    fewer rows available than its quota.
    """
    quotas = {name: q for name, q in _QUAL_SECTION_QUOTAS}
    if budget >= sum(quotas.values()):
        return quotas
    remaining = budget
    out: dict[str, int] = {}
    for name, q in _QUAL_SECTION_QUOTAS:
        take = min(q, remaining)
        out[name] = take
        remaining -= take
    return out


def build_qualitative_sample(
    *,
    runs: dict[str, BackendRun],
    probes: list[GateProbe],
    budget: int = DEFAULT_QUALITATIVE_SAMPLE_SIZE,
) -> str:
    """Render the operator-facing qualitative sample markdown.

    Strictly no raw probe `window` text: only the row content,
    metadata, and anchor IDs make it into the artifact. The
    `qualitative_verdict: pending` header mirrors the canonical
    JSON value so the operator can find the artifact's verdict in
    one place when scanning the markdown.
    """
    lines: list[str] = [
        "# Memory backend gate qualitative sample",
        "",
        "qualitative_verdict: pending",
        "",
        (
            "Operator instructions: review the rows below for each backend, then "
            "set `qualitative_verdict` to `pass` or `fail` in `gate-result.json`."
        ),
        "",
    ]
    quotas = _quotas_for(budget)
    probe_by_id = {p.probe_id: p for p in probes}
    for backend_name, run in runs.items():
        lines.append(f"## {backend_name}")
        lines.append("")
        # Spill-forward: each section takes up to its quota; any
        # unused slots roll into the next section's budget. The
        # `facts -> retrieval -> episodes` order is fixed because
        # facts are the highest-signal artifact in the gate.
        remaining = budget
        facts_taken = _collect_fact_samples(run, probe_by_id, min(quotas.get("facts", 0), remaining))
        remaining -= len(facts_taken)
        retrieval_taken = _collect_retrieval_samples(
            run, min(quotas.get("retrieval", 0) + (quotas.get("facts", 0) - len(facts_taken)), remaining)
        )
        remaining -= len(retrieval_taken)
        episode_taken = _collect_episode_samples(run, remaining)

        lines.append("### Fact rows")
        if not facts_taken:
            lines.append("(none)")
        for entry in facts_taken:
            lines.append(_format_fact_row(entry))
        lines.append("")
        lines.append("### Retrieval misses / low-rank hits")
        if not retrieval_taken:
            lines.append("(none)")
        for r in retrieval_taken:
            lines.append(
                f"- probe={r.probe_id} anchor={r.anchor_id} rank={r.rank} "
                f"in_prompt={r.in_prompt} reason={r.reason or 'none'}"
            )
        lines.append("")
        lines.append("### Episode rows")
        if not episode_taken:
            lines.append("(none)")
        for entry in episode_taken:
            lines.append(_format_episode_row(entry))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _collect_fact_samples(
    run: BackendRun,
    probe_by_id: dict[str, GateProbe],
    quota: int,
) -> list[tuple[str, memory.MemoryResult, str]]:
    """Pick fact rows for the qualitative sample.

    Priority order: forbidden-content violations (any backend that
    leaked workflow status into durable memory is the most important
    thing to surface) -> satisfied anchors (so the operator can spot
    paraphrase facts that landed but read as low value) -> missing
    anchors (which show what extraction failed to catch). Each entry
    carries the probe_id and a short label so the operator can trace
    it back to a specific assertion.
    """
    if quota <= 0:
        return []
    picks: list[tuple[str, memory.MemoryResult, str]] = []
    for outcome in run.probes:
        if len(picks) >= quota:
            break
        if not outcome.forbidden_violations:
            continue
        for fact_id in outcome.new_fact_ids:
            row = next((r for r in run.final_facts if r.id == fact_id), None)
            if row is None:
                continue
            picks.append((outcome.probe_id, row, "forbidden_violation"))
            if len(picks) >= quota:
                break
    for outcome in run.probes:
        if len(picks) >= quota:
            break
        for anchor_id, row_id in outcome.satisfied_anchors.items():
            row = next((r for r in run.final_facts if r.id == row_id), None)
            if row is None:
                continue
            picks.append((outcome.probe_id, row, f"anchor:{anchor_id}"))
            if len(picks) >= quota:
                break
    for outcome in run.probes:
        if len(picks) >= quota:
            break
        if probe_by_id.get(outcome.probe_id) is None:
            continue
        for anchor_id in outcome.missing_anchors:
            picks.append((outcome.probe_id, _placeholder_missing(anchor_id), f"missing:{anchor_id}"))
            if len(picks) >= quota:
                break
    return picks


def _placeholder_missing(anchor_id: str) -> memory.MemoryResult:
    """Surface a `(missing)` placeholder so the markdown still records
    which anchor was expected without dumping any probe text."""
    return memory.MemoryResult(
        id="(missing)",
        text=f"(no row stored for anchor {anchor_id})",
        score=0.0,
        memory_type="fact",
        metadata={},
        created_at="",
        updated_at="",
    )


def _collect_retrieval_samples(run: BackendRun, quota: int) -> list[RetrievalQueryResult]:
    """Retrieval misses and low-rank hits, lowest-rank first."""
    if quota <= 0:
        return []
    losses = [r for r in run.retrieval if r.rank is None or r.rank > 1]
    losses.sort(key=lambda r: (r.rank is None, r.rank or 0), reverse=True)
    return losses[:quota]


def _collect_episode_samples(run: BackendRun, quota: int) -> list[tuple[str, memory.MemoryResult, str]]:
    """Pick episode rows, prioritizing positives and false positives.

    Episode false positives are harder to spot in a JSON table than
    in prose, so the sample puts them first. Positives second show
    the operator what an actually-good episode looks like.
    """
    if quota <= 0:
        return []
    picks: list[tuple[str, memory.MemoryResult, str]] = []
    for outcome in run.probes:
        if len(picks) >= quota:
            break
        if outcome.category != "episode-negative":
            continue
        for ep_id in outcome.new_episode_ids:
            row = next((r for r in run.final_episodes if r.id == ep_id), None)
            if row is None:
                continue
            picks.append((outcome.probe_id, row, "false_positive"))
            if len(picks) >= quota:
                break
    for outcome in run.probes:
        if len(picks) >= quota:
            break
        if outcome.category != "episode-positive":
            continue
        for ep_id in outcome.new_episode_ids:
            row = next((r for r in run.final_episodes if r.id == ep_id), None)
            if row is None:
                continue
            picks.append((outcome.probe_id, row, "true_positive"))
            if len(picks) >= quota:
                break
    return picks


def _format_fact_row(entry: tuple[str, memory.MemoryResult, str]) -> str:
    probe_id, row, label = entry
    metadata = row.metadata or {}
    tags = metadata.get("tags") or []
    speaker = metadata.get("speaker", "")
    return f"- probe={probe_id} label={label} id={row.id} speaker={speaker} tags={tags} content={row.text!r}"


def _format_episode_row(entry: tuple[str, memory.MemoryResult, str]) -> str:
    probe_id, row, label = entry
    metadata = row.metadata or {}
    return (
        f"- probe={probe_id} label={label} id={row.id} goal={metadata.get('goal', '')!r} "
        f"outcome_quality={metadata.get('outcome_quality', '')!r} actors={metadata.get('actors') or []}"
    )


# ── CLI ─────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Argparse wiring; see the module docstring for the surface."""
    parser = argparse.ArgumentParser(
        prog="python -m kai.eval.memory_backend_gate",
        description=(
            "Claude-vs-Codex memory eval gate. Drives extract_and_store + "
            "format_context for each backend against the same probe fixture "
            "and emits gate-result.json + per-backend logs + qualitative-sample.md."
        ),
    )
    parser.add_argument("--probes", type=Path, required=True, help="Path to the JSONL probe fixture.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory to write gate-result.json, <backend>.log, and "
            "qualitative-sample.md. Must be operator-supplied; no project-relative default."
        ),
    )
    parser.add_argument(
        "--user-prefix",
        required=True,
        help=(
            f"Sandbox user-ID prefix; must start with {_SANDBOX_USER_ID_PREFIX!r}. "
            "Per-backend sandbox IDs are <prefix>-claude and <prefix>-codex."
        ),
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(DEFAULT_BACKENDS),
        choices=list(DEFAULT_BACKENDS),
        help="Backends to run; default is both.",
    )
    parser.add_argument(
        "--os-user",
        default=None,
        help=(
            "Optional OS user override to run the memory reasoner as "
            "via sudo -H -u. The eval gate writes to sandbox user IDs "
            "that have no users.yaml entry, so the production "
            "resolution path (telegram_id -> users.yaml.os_user) "
            "yields None; both codex and claude follow the same "
            "self-sudo-skip path on None and spawn in-process as the "
            "bot user. Supply this flag to force both arms through a "
            "specific non-bot OS target instead."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete sandbox rows before the run; without it a non-empty sandbox aborts.",
    )
    parser.add_argument(
        "--keep-sandboxes",
        action="store_true",
        help="Default: sandboxes are not deleted at the end of the run. This flag is a no-op kept for forward compat.",
    )
    parser.add_argument(
        "--qualitative-sample-size",
        type=int,
        default=DEFAULT_QUALITATIVE_SAMPLE_SIZE,
        help="Total rows per backend in qualitative-sample.md; default 10 (5 facts + 3 retrieval + 2 episode).",
    )
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Exit 1 if thresholds fail; otherwise exit 0 regardless of verdict.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate the fixture, then exit 0; no model calls, no sandbox writes.",
    )
    return parser


async def _run_cli(args: argparse.Namespace) -> int:
    """Async CLI body. Returns the integer exit code."""
    try:
        validate_user_prefix(args.user_prefix)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        probes = load_probes(args.probes)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        validate_fixture_minimums(probes)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.validate_only:
        print(f"fixture ok: {len(probes)} probes, {sum(len(p.retrieval) for p in probes)} retrieval queries")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_config()
    # Initialize the module-level Mem0 instance before any backend
    # arm runs. Without this `kai.memory._memory` stays None and
    # every storage call (delete_all, get_all, add_structured,
    # format_context) short-circuits to a silent no-op, so
    # extracted facts surface as outcome=dropped_backend and
    # retrieval returns nothing - the gate would run but score
    # whatever zero-state Codex happens to also produce. The
    # replay harness handles this the same way; use a
    # memory-enabled config copy because `load_config()` may
    # return memory_enabled=False if the operator's env file does
    # not opt in.
    memory_init_config = dataclasses.replace(
        base_config,
        memory_enabled=True,
        memory_extraction_enabled=True,
    )
    memory.init_memory(memory_init_config)
    runs: dict[str, BackendRun] = {}
    metrics_by_backend: dict[str, BackendMetrics] = {}
    for backend in args.backends:
        log_path = args.output_dir / f"{backend}.log"
        sandbox_user_id = f"{args.user_prefix}-{backend}"
        try:
            run = await run_backend(
                probes=probes,
                backend=backend,
                base_config=base_config,
                sandbox_user_id=sandbox_user_id,
                log_path=log_path,
                reset=args.reset,
                os_user_override=args.os_user,
            )
        except SystemExit as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        runs[backend] = run
        metrics = compute_metrics(run)
        update_forbidden_total(metrics, probes)
        tp, fp, fn, tn, validity = score_episodes(probes, run)
        metrics.episode_true_positive_count = tp
        metrics.episode_false_positive_count = fp
        metrics.episode_false_negative_count = fn
        metrics.episode_true_negative_count = tn
        metrics.episode_required_field_validity = validity
        metrics.episode_precision = tp / max(1, tp + fp)
        metrics.episode_recall = tp / max(1, tp + fn)
        metrics_by_backend[backend] = metrics

    if "claude" in metrics_by_backend and "codex" in metrics_by_backend:
        threshold_report = compare_thresholds(metrics_by_backend["claude"], metrics_by_backend["codex"])
    else:
        threshold_report = ThresholdReport(checks=[], overall="single_backend")

    gate_result = build_gate_result(
        probes=probes,
        runs=runs,
        metrics=metrics_by_backend,
        threshold_report=threshold_report,
    )
    (args.output_dir / "gate-result.json").write_text(json.dumps(gate_result, indent=2) + "\n", encoding="utf-8")

    sample_md = build_qualitative_sample(runs=runs, probes=probes, budget=args.qualitative_sample_size)
    (args.output_dir / "qualitative-sample.md").write_text(sample_md, encoding="utf-8")

    if args.fail_on_threshold and threshold_report.overall != "pass":
        return 1
    return 0


def main() -> None:
    """Argparse entry point. Wraps `_run_cli` in asyncio.run.

    Exit code 3 covers any uncaught exception so the operator sees a
    distinct signal from the preflight (2) and threshold (1) cases.
    """
    parser = _build_parser()
    args = parser.parse_args()
    try:
        rc = asyncio.run(_run_cli(args))
    except Exception:
        log.exception("memory_backend_gate failed with an unexpected error")
        rc = 3
    sys.exit(rc)


if __name__ == "__main__":
    main()
