"""
Offline scope reclassification for the semantic memory store.

Backs the `python -m kai memory reclassify-scope` subcommand. The
pass targets the scope-debt backlog: rows whose resolved scope is
global only because nothing ever classified them (resolver
`legacy_default`), plus rows written global by extraction before
their project was registered (`extraction_default`). A one-shot
reasoner proposes per-row verdicts; high-confidence verdicts become a
reviewable proposals file; a separate apply invocation writes only
what the operator reviewed, with pre-images dumped first and a
rollback mode that consumes them.

Three modes, three drivers:
1. `run_dry_run`: enumerate, classify, write report + proposals.
   No store writes.
2. `run_apply`: re-check and apply a reviewed proposals file;
   pre-image dump before the first write.
3. `run_rollback`: restore rows from a pre-image file, refusing to
   overwrite later operator corrections.

Architectural shape mirrors `memory_command.py`: pure helpers
(selection, verdict gating, file (de)serialization, report
rendering, header validation) do no I/O and are unit-testable
without Mem0 or subprocesses; the three async drivers own store
access, reasoner calls, and file writes. The ENFORCEMENT of every
safety gate (registry bootstrap, acting on a header-validation
failure, exit-code policy) happens in the drivers and the CLI
layer, never inside a pure helper, so the gates stay visible at the
orchestration layer.

Conservatism contract (the spec's load-bearing properties):
- Selection admits ONLY user-visible rows resolving to global with
  legacy/extraction-default provenance. Operator rows are
  authoritative, classifier rows must not flip-flop across runs,
  and invalid rows would have their corruption masked by a
  classifier write.
- Both verdict directions are threshold-gated; everything ambiguous
  stays exactly as it is.
- Apply consumes the reviewed proposals file verbatim; it never
  re-runs classification, so what the operator eyeballed is what
  gets written.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kai import memory, sessions
from kai.config import Config, MemoryProjectConfig, ModelRole, resolve_user_model
from kai.history import fetch_transcript_context
from kai.memory import MemoryResult, ResolvedMemoryScope, read_transcript_provenance
from kai.memory_extraction import (
    _build_memory_reasoner,
    _resolve_effective_backend,
    _resolve_effective_provider,
    _resolve_os_user,
    _resolve_user_config,
)
from kai.memory_projects import load_db_registry, merged_registry
from kai.oneshot import OneShotError

log = logging.getLogger(__name__)


# ── Constants ───────────────────────────────────────────────────────

# Provenance values eligible for reclassification. Everything else is
# shielded: operator rows are authoritative, classifier rows would
# flip-flop, invalid rows need repair (not masking), and non-global
# rows are out of population by definition.
_ELIGIBLE_SCOPE_SOURCES: frozenset[str] = frozenset(
    {memory.SCOPE_SOURCE_LEGACY_DEFAULT, memory.SCOPE_SOURCE_EXTRACTION_DEFAULT}
)

# Stable skip-reason keys. They appear in reports and summaries, so
# renames are reader-visible; treat like log vocabulary.
SKIP_BELOW_THRESHOLD = "below_threshold"
SKIP_UNREGISTERED_TARGET = "unregistered_target"
SKIP_DISABLED_TARGET = "disabled_target"
SKIP_REASONER_FAILURE = "reasoner_failure"
SKIP_MALFORMED_OUTPUT = "malformed_output"
# Apply-time re-check skips.
SKIP_ROW_GONE = "row_gone"
SKIP_DESELECTED = "deselected"
SKIP_TEXT_DRIFT = "text_drift"
# Rollback-only skip: the row was corrected by the operator after the
# pre-image was taken; rollback must not overwrite that intent.
SKIP_OPERATOR_CORRECTION = "operator_correction"

# Consecutive reasoner-failure ceiling for the dry-run abort guard. A
# dead backend (wrong os-user, missing auth, stopped provider) fails
# every call; five in a row is unambiguous death, while scattered
# failures from a live backend never accumulate because any success
# resets the counter.
_CONSECUTIVE_FAILURE_ABORT = 5

# JSON Schema enforced on every classification call. Mirrors the
# extraction pattern: the reasoner passes it to the provider's schema
# mode, and the envelope parser still validates defensively because
# the root-fallback path can deliver unvalidated payloads.
_SCOPE_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scope": {"type": "string", "enum": ["global", "project"]},
        "project_id": {"type": ["string", "null"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string", "maxLength": 200},
    },
    "required": ["scope", "project_id", "confidence", "reason"],
    "additionalProperties": False,
}

# Classification prompt scaffolding. Assembled by `_render_prompt`
# with plain concatenation (never str.format) because row text is
# user-originated and routinely contains braces.
_PROMPT_HEAD = (
    "You are classifying one stored memory for scope.\n"
    "\n"
    'Scope model: a "global" memory applies to the operator\'s work everywhere. '
    'A "project" memory only matters inside one specific project.\n'
    "\n"
    "Registered projects:\n"
)
_PROMPT_RULES = (
    "\n"
    "Rules:\n"
    '- Choose "project" only when the memory is about that project\'s code, issues, design, tests, or conventions.\n'
    '- Operator identity, preferences, environment, tooling habits, people, and cross-project workflow are "global".\n'
    '- If unsure, choose "global" with low confidence.\n'
    '- "project_id" must be one of the registered ids above, or null when scope is "global".\n'
    "\n"
    'Respond with JSON only: {"scope": "global"|"project", "project_id": <id or null>, '
    '"confidence": 0.0-1.0, "reason": "<one line>"}\n'
    "\n"
    "Memory:\n"
)


# ── Data shapes ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Proposal:
    """One reviewable scope-change proposal from a dry run.

    Attributes:
        memory_id: Target row id.
        verdict: "project" (move) or "global" (provenance re-stamp).
        project_id: Move target for project verdicts; None for
            global re-stamps.
        confidence: Reasoner confidence, carried into
            `scope_confidence` at apply time so the stored row
            reflects the classifier's actual certainty.
        reason: The reasoner's one-line justification, surfaced in
            the report for eyeballing.
        prior_scope_source: Resolved provenance at classification
            time ("legacy_default" or "extraction_default");
            report-only context.
        text_sha256: SHA-256 of the row text at classification time.
            Apply skips the row when this no longer matches, because
            the verdict was rendered for content that has since
            changed.
    """

    memory_id: str
    verdict: str
    project_id: str | None
    confidence: float
    reason: str
    prior_scope_source: str
    text_sha256: str


@dataclass(frozen=True)
class PreImage:
    """One row's full pre-apply state, dumped for rollback."""

    memory_id: str
    text: str
    metadata: dict[str, Any]


def _text_sha256(text: str) -> str:
    """Content-drift fingerprint for a row's text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Pure helpers: selection ─────────────────────────────────────────


def select_rows(rows: list[MemoryResult]) -> list[tuple[MemoryResult, ResolvedMemoryScope]]:
    """Filter the corpus down to the reclassification population.

    A row is selected iff its source is user-visible, its RESOLVED
    scope is global, and its resolved provenance is legacy_default or
    extraction_default. Resolution always goes through
    `resolve_memory_scope`, never raw metadata, so corrupted rows
    land in the resolver's invalid arm and are excluded here rather
    than misread as eligible.

    Returns (row, resolved) pairs so downstream steps never
    re-resolve the same metadata.
    """
    selected: list[tuple[MemoryResult, ResolvedMemoryScope]] = []
    for row in rows:
        if (row.metadata or {}).get("source") not in memory.USER_VISIBLE_SOURCES:
            continue
        resolved = memory.resolve_memory_scope(row.metadata)
        if resolved.scope != memory.SCOPE_GLOBAL:
            continue
        # invalid_defaulted rows can resolve to global too; the
        # provenance check excludes them because their stored
        # metadata is malformed and a classifier overlay would mask
        # the corruption instead of surfacing it.
        if resolved.scope_source not in _ELIGIBLE_SCOPE_SOURCES:
            continue
        selected.append((row, resolved))
    return selected


# ── Pure helpers: verdict gating ────────────────────────────────────


def gate_verdict(
    verdict: Any,
    *,
    row: MemoryResult,
    resolved: ResolvedMemoryScope,
    registry: dict[str, MemoryProjectConfig],
    threshold: float,
) -> tuple[Proposal | None, str | None]:
    """Turn one parsed reasoner verdict into a proposal or a skip.

    Returns exactly one of (proposal, None) or (None, skip_reason).

    Validation here is defensive even though the call carries a JSON
    Schema: the envelope's root-fallback path can deliver payloads
    the provider never validated, and a malformed verdict must
    become a counted skip, not an exception that kills the run.

    Gates (conservative in both directions; threshold applies to
    both):
    - project verdict: confidence >= threshold AND the target is a
      registered, memory-enabled project. Disabled projects are not
      valid targets because their rows are unretrievable; moving a
      row there would park it invisibly.
    - global verdict: confidence >= threshold. Proposes a provenance
      re-stamp so the legacy bucket converges toward genuinely
      unreviewed rows.
    - everything else: skip with a stable reason.
    """
    if not isinstance(verdict, dict):
        return None, SKIP_MALFORMED_OUTPUT
    scope = verdict.get("scope")
    confidence = verdict.get("confidence")
    project_id = verdict.get("project_id")
    reason = verdict.get("reason")
    if scope not in (memory.SCOPE_GLOBAL, memory.SCOPE_PROJECT):
        return None, SKIP_MALFORMED_OUTPUT
    if not isinstance(confidence, int | float) or not (0.0 <= float(confidence) <= 1.0):
        return None, SKIP_MALFORMED_OUTPUT
    if not isinstance(reason, str):
        reason = ""

    if float(confidence) < threshold:
        return None, SKIP_BELOW_THRESHOLD

    if scope == memory.SCOPE_PROJECT:
        if not isinstance(project_id, str) or project_id not in registry:
            return None, SKIP_UNREGISTERED_TARGET
        if not registry[project_id].memory_enabled:
            return None, SKIP_DISABLED_TARGET
    else:
        # Global re-stamps never carry a target, whatever the model
        # emitted alongside the verdict.
        project_id = None

    return (
        Proposal(
            memory_id=row.id,
            verdict=scope,
            project_id=project_id,
            confidence=float(confidence),
            reason=reason,
            prior_scope_source=resolved.scope_source,
            text_sha256=_text_sha256(row.text),
        ),
        None,
    )


# ── Pure helpers: prompt ────────────────────────────────────────────


def _render_prompt(registry: dict[str, MemoryProjectConfig], text: str) -> str:
    """Assemble the classification prompt for one row.

    Only memory-enabled projects appear in the vocabulary: a
    disabled project is not a valid verdict target, so offering it
    to the model would invite proposals the gate must then reject.
    """
    project_lines = "".join(
        f"{pid}: {cfg.display_name}\n" for pid, cfg in sorted(registry.items()) if cfg.memory_enabled
    )
    return _PROMPT_HEAD + project_lines + _PROMPT_RULES + text


# ── Pure helpers: envelope parsing ──────────────────────────────────


def parse_verdict_envelope(raw_text: str) -> dict[str, Any] | None:
    """Parse one reasoner response into a verdict dict, or None.

    Mirrors the extraction parse block: strict `json.loads`, reject
    non-dict, reject `is_error: true`, prefer the
    `structured_output` nesting (the claude schema-mode envelope)
    and fall back to the root for providers that emit the payload
    directly. Returns None for every rejection; the caller counts it
    as a malformed-output skip.
    """
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if parsed.get("is_error") is True:
        return None
    structured = parsed.get("structured_output")
    if isinstance(structured, dict) and "scope" in structured:
        return structured
    if "scope" in parsed:
        return parsed
    return None


# ── Pure helpers: file formats ──────────────────────────────────────
#
# Both artifact formats are header-then-rows JSONL. The header makes
# every file self-describing (run id, target user) so apply and
# rollback never derive identity from filenames, and a wrong-user
# file fails validation before any store access.


def render_proposals(header: dict[str, Any], proposals: list[Proposal]) -> str:
    """Serialize a proposals file: header line, then proposal lines."""
    lines = [json.dumps({"type": "header", **header}, separators=(",", ":"))]
    for p in proposals:
        lines.append(
            json.dumps(
                {
                    "type": "proposal",
                    "memory_id": p.memory_id,
                    "verdict": p.verdict,
                    "project_id": p.project_id,
                    "confidence": p.confidence,
                    "reason": p.reason,
                    "prior_scope_source": p.prior_scope_source,
                    "text_sha256": p.text_sha256,
                },
                separators=(",", ":"),
            )
        )
    return "\n".join(lines) + "\n"


def render_preimages(header: dict[str, Any], preimages: list[PreImage]) -> str:
    """Serialize a pre-image file: header line, then row lines."""
    lines = [json.dumps({"type": "header", **header}, separators=(",", ":"))]
    for p in preimages:
        lines.append(
            json.dumps(
                {"type": "preimage", "memory_id": p.memory_id, "text": p.text, "metadata": p.metadata},
                separators=(",", ":"),
            )
        )
    return "\n".join(lines) + "\n"


def parse_artifact(text: str, *, row_type: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse a header-then-rows JSONL artifact.

    Shared by proposals (`row_type="proposal"`) and pre-images
    (`row_type="preimage"`). Raises ValueError with a readable
    message on structural problems (missing/invalid header, a line
    that is not JSON, a row of the wrong type); the CLI surfaces the
    message and exits non-zero. Strictness is deliberate: these
    files authorize store writes, so a half-parsed file must never
    be silently half-applied.
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
    header, raws = parse_artifact(text, row_type="proposal")
    proposals: list[Proposal] = []
    for i, raw in enumerate(raws, start=2):
        mid = raw.get("memory_id")
        if not isinstance(mid, str) or not mid:
            raise ValueError(f"proposal line {i}: memory_id must be a non-empty string")
        verdict = raw.get("verdict")
        if verdict not in (memory.SCOPE_GLOBAL, memory.SCOPE_PROJECT):
            raise ValueError(f"proposal line {i}: verdict must be 'global' or 'project', got {verdict!r}")
        project_id = raw.get("project_id")
        if verdict == memory.SCOPE_PROJECT:
            if not isinstance(project_id, str) or not project_id:
                raise ValueError(f"proposal line {i}: project verdict requires a non-empty project_id")
        else:
            # Normalize, matching the gate: a global re-stamp never
            # carries a target, whatever a hand edit left behind.
            project_id = None
        confidence = raw.get("confidence")
        if not isinstance(confidence, int | float) or not (0.0 <= float(confidence) <= 1.0):
            raise ValueError(f"proposal line {i}: confidence must be a number in [0.0, 1.0], got {confidence!r}")
        sha = raw.get("text_sha256")
        if not isinstance(sha, str) or not sha:
            raise ValueError(f"proposal line {i}: text_sha256 must be a non-empty string")
        reason = raw.get("reason")
        proposals.append(
            Proposal(
                memory_id=mid,
                verdict=verdict,
                project_id=project_id,
                confidence=float(confidence),
                reason=reason if isinstance(reason, str) else "",
                prior_scope_source=str(raw.get("prior_scope_source", "")),
                text_sha256=sha,
            )
        )
    return header, proposals


def parse_preimages(text: str) -> tuple[dict[str, Any], list[PreImage]]:
    """Parse a pre-image file into fully validated PreImage objects.

    Same strictness rationale as `parse_proposals`: rollback writes
    the dumped text and metadata back verbatim, so a malformed row
    (text missing, metadata not a dict) would either crash mid-run
    or silently restore an empty row. Raises ValueError naming the
    offending line; the caller aborts the whole rollback.
    """
    header, raws = parse_artifact(text, row_type="preimage")
    preimages: list[PreImage] = []
    for i, raw in enumerate(raws, start=2):
        mid = raw.get("memory_id")
        if not isinstance(mid, str) or not mid:
            raise ValueError(f"preimage line {i}: memory_id must be a non-empty string")
        pre_text = raw.get("text")
        if not isinstance(pre_text, str) or not pre_text:
            raise ValueError(f"preimage line {i}: text must be a non-empty string")
        metadata = raw.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"preimage line {i}: metadata must be an object")
        preimages.append(PreImage(memory_id=mid, text=pre_text, metadata=metadata))
    return header, preimages


# ── Pure helpers: report rendering ──────────────────────────────────


def _collect_provenance_quotes(
    selected: list[tuple[MemoryResult, ResolvedMemoryScope]],
    proposals: list[Proposal],
    *,
    user_id: str,
) -> dict[str, str]:
    """Build the memory_id → originating-user-text map for the report.

    Best-effort: a row without provenance, with a failed JSONL
    lookup, or whose helper returns any non-ok reason contributes
    nothing. The text is truncated here (80 chars) so the renderer
    stays formatter-only and the entry it pulls is already
    report-sized.

    The proposal set drives the iteration: we only look up provenance
    for rows that actually appear in the report's eyeball sample
    space, not for the larger selected pool.

    `user_id` is the CLI's `<user_id>` argument; when it parses as an
    int, it becomes `expected_chat_id` on the lookup so the helper
    refuses to dereference any row whose `source_chat_id` does not
    match. Mem0's user-id partition already scopes the row read, but
    the row's provenance pointer is independent metadata and must be
    validated separately. When the CLI was invoked against a non-
    numeric sandbox id, the gate is skipped (the helper falls back
    to its no-expected-chat-id behaviour); admin contexts that scan
    cross-chat by design can also pass `expected_chat_id=None`.
    """
    try:
        expected_chat_id: int | None = int(user_id)
    except ValueError:
        expected_chat_id = None
    quotes: dict[str, str] = {}
    proposal_ids = {p.memory_id for p in proposals}
    metadata_by_id = {row.id: row.metadata for row, _ in selected if row.id in proposal_ids}
    for memory_id, metadata in metadata_by_id.items():
        provenance = read_transcript_provenance(metadata)
        if not provenance.present:
            continue
        lookup = fetch_transcript_context(
            provenance,
            before=0,
            after=0,
            memory_id=memory_id,
            expected_chat_id=expected_chat_id,
        )
        if lookup.reason != "ok" or lookup.context is None:
            continue
        quotes[memory_id] = _truncate(lookup.context.target_user.text, 80)
    return quotes


def render_report(
    *,
    run_id: str,
    user_id: str,
    backend: str,
    threshold: float,
    scanned: int,
    selected: int,
    proposals: list[Proposal],
    skips: dict[str, list[str]],
    sample_size: int,
    texts: dict[str, str],
    provenance_user_texts: dict[str, str] | None = None,
) -> str:
    """Render the dry-run report markdown.

    `texts` maps memory_id to row text for display; row text lives
    only in the report (the proposals file carries the sha) so the
    machine-readable artifact stays small while the human-readable
    one shows what was actually classified.

    `provenance_user_texts` optionally maps memory_id to the row's
    originating user-turn text (already truncated by the caller).
    When an entry is present, the eyeball-sample line gains a `said:`
    line that quotes the originating message so the operator can
    eyeball "is this really a project fact" against the message that
    produced it. Entries are missing for legacy rows or for rows
    whose transcript lookup failed; those proposals render as before.

    The eyeball sample is drawn with `random.Random(run_id)` so
    re-rendering the same run reproduces the same sample; the
    operator's quick look and any later re-read see identical rows.
    """
    moves = sum(1 for p in proposals if p.verdict == memory.SCOPE_PROJECT)
    restamps = len(proposals) - moves
    skip_summary = ", ".join(f"{len(ids)} {reason}" for reason, ids in sorted(skips.items()) if ids) or "none"

    lines = [
        f"# Scope reclassification dry-run {run_id}",
        "",
        f"user: {user_id}  backend: {backend}  threshold: {threshold}",
        f"rows scanned: {scanned}   selected: {selected}",
        f"proposals: {moves} project moves, {restamps} global re-stamps",
        f"skipped: {skip_summary}",
        "",
    ]

    if proposals:
        sample = random.Random(run_id).sample(proposals, min(sample_size, len(proposals)))
        lines.append(f"## Eyeball sample ({len(sample)} of {len(proposals)})")
        for i, p in enumerate(sample, start=1):
            target = f"project {p.project_id}" if p.verdict == memory.SCOPE_PROJECT else "global"
            text = _truncate(texts.get(p.memory_id, ""), 80)
            lines.append(f'{i}. [{target} {p.confidence:.2f}] "{text}" (was {p.prior_scope_source})')
            lines.append(f"   reason: {p.reason}")
            # Optional originating-turn quote when transcript provenance
            # is present on the source row. Legacy rows and lookup
            # failures contribute nothing, so the line is appended
            # only when the entry actually exists. The caller has
            # already truncated the text to keep the report compact.
            if provenance_user_texts and p.memory_id in provenance_user_texts:
                lines.append(f'   said:   "{provenance_user_texts[p.memory_id]}"')
        lines.append("")
        lines.append("## All proposals")
        lines.append("| id | verdict | conf | was | text |")
        lines.append("|----|---------|------|-----|------|")
        for p in proposals:
            target = p.project_id if p.verdict == memory.SCOPE_PROJECT else "global"
            text = _truncate(texts.get(p.memory_id, ""), 60)
            lines.append(f"| {p.memory_id} | {target} | {p.confidence:.2f} | {p.prior_scope_source} | {text} |")
        lines.append("")

    lines.append("## Skips")
    any_skips = False
    for reason, ids in sorted(skips.items()):
        for mid in ids:
            lines.append(f"- {mid}: {reason}")
            any_skips = True
    if not any_skips:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def _truncate(text: str, limit: int) -> str:
    """Single-line truncation for report table cells."""
    flat = text.replace("\n", " ").replace("|", "/")
    if len(flat) <= limit:
        return flat
    return flat[: max(limit - 1, 0)] + "…"


# ── Structured logging ──────────────────────────────────────────────


def _emit_scope_change(
    *,
    memory_id: str,
    user_id: str,
    from_resolved: ResolvedMemoryScope,
    to_scope: str,
    to_project_id: str | None,
    run_id: str,
    rollback: bool = False,
) -> None:
    """Emit one scope-change audit line.

    Same event name and field names as the /memory scope tools'
    emitter, plus run_id (and rollback when applicable), so one log
    query covers every writer that reassigns scope. The chat_id
    field carries the CLI's user_id string.
    """
    payload: dict[str, Any] = {
        "memory_id": memory_id,
        "chat_id": user_id,
        "from_scope": from_resolved.scope,
        "from_project_id": from_resolved.project_id,
        "from_scope_source": from_resolved.scope_source,
        "to_scope": to_scope,
        "to_project_id": to_project_id,
        "run_id": run_id,
    }
    if rollback:
        payload["rollback"] = True
    log.info("%s %s", memory.SCOPE_CHANGE_EVENT, json.dumps(payload, separators=(",", ":")))


# ── Drivers ─────────────────────────────────────────────────────────


async def load_project_registry(config: Config) -> dict[str, MemoryProjectConfig]:
    """Bootstrap the merged project registry in a fresh CLI process.

    Mirrors daemon startup exactly: session-DB init, bulk-load the
    chat-registered rows into the detection cache, then merge under
    the operator-pinned YAML layer. Runs in BOTH dry-run and apply
    (apply re-checks target registration, so an apply without this
    load would skip every chat-registered target as unregistered,
    which is the exact failure the re-check exists to prevent).
    Rollback restores dumped state wholesale and does not need it.
    """
    await sessions.init_db(config.session_db_path)
    load_db_registry(await sessions.get_memory_project_rows())
    return merged_registry(config.memory_projects)


def resolve_classification_settings(
    config: Config,
    user_id: str,
    *,
    backend: str | None,
    os_user: str | None,
    provider: str | None,
) -> tuple[str, str | None, str] | str:
    """Resolve effective (backend, os_user, provider) for a dry run.

    Flags override; defaults follow the TARGET USER's effective
    resolution (the same cascade memory extraction walks), because
    classification quality and provider auth are per-user
    properties, not install-wide ones. Returns the resolved triple,
    or an error message string when the effective backend cannot run
    a one-shot reasoner and no explicit flag was given; the command
    must fail rather than silently substitute another backend.
    """
    from kai.config import ONESHOT_REASONER_BACKENDS

    effective_backend = backend or _resolve_effective_backend(user_id, config)
    if effective_backend not in ONESHOT_REASONER_BACKENDS:
        return (
            f"effective backend {effective_backend!r} cannot run one-shot classification; "
            f"pass --backend (one of {sorted(ONESHOT_REASONER_BACKENDS)})"
        )
    effective_os_user = os_user if os_user is not None else _resolve_os_user(user_id, config)
    effective_provider = provider or _resolve_effective_provider(user_id, config)
    return effective_backend, effective_os_user, effective_provider


async def run_dry_run(
    config: Config,
    user_id: str,
    *,
    backend: str | None,
    os_user: str | None,
    provider: str | None,
    threshold: float,
    sample: int,
    out_dir: Path,
) -> int:
    """Classify the target population and write report + proposals.

    Writes nothing to the store. Returns a process exit code: 0 on a
    completed pass (even an all-skip one; the report is the
    product), non-zero on settings errors or the abort guard.
    """
    settings = resolve_classification_settings(config, user_id, backend=backend, os_user=os_user, provider=provider)
    if isinstance(settings, str):
        print(f"memory admin: {settings}")
        return 2
    effective_backend, effective_os_user, effective_provider = settings

    registry = await load_project_registry(config)
    rows = memory.get_all(user_id=user_id, limit=None)
    selected = select_rows(rows)

    run_id = datetime.now(UTC).strftime("rs-%Y%m%d-%H%M%S")
    reasoner = _build_memory_reasoner(effective_backend, os_user=effective_os_user, provider=effective_provider)
    user_cfg = _resolve_user_config(user_id, config)
    model = resolve_user_model(
        ModelRole.MEMORY_EXTRACTION,
        user_cfg,
        config,
        backend=effective_backend,
        provider=effective_provider,
    )

    proposals: list[Proposal] = []
    skips: dict[str, list[str]] = {}
    consecutive_failures = 0
    for row, resolved in selected:
        try:
            result = await reasoner.run(
                prompt=_render_prompt(registry, row.text),
                system_prompt=None,
                model=model,
                timeout=config.memory_extraction_timeout_s,
                purpose="scope_reclassification",
                json_schema=_SCOPE_VERDICT_SCHEMA,
            )
        except OneShotError:
            # All typed reasoner failures (timeout, subprocess,
            # output) collapse to the same counted skip; the log
            # carries the detail for forensics.
            log.warning("reclassify: reasoner failed for %s", row.id, exc_info=True)
            skips.setdefault(SKIP_REASONER_FAILURE, []).append(row.id)
            consecutive_failures += 1
            if consecutive_failures >= _CONSECUTIVE_FAILURE_ABORT:
                print(
                    f"memory admin: aborting after {consecutive_failures} consecutive reasoner "
                    "failures; the backend looks dead (auth? os-user? provider?)."
                )
                return 1
            continue
        verdict = parse_verdict_envelope(result.text)
        if verdict is None:
            skips.setdefault(SKIP_MALFORMED_OUTPUT, []).append(row.id)
            # Malformed output counts toward the abort guard: a
            # backend emitting garbage every call is as dead as one
            # that times out.
            consecutive_failures += 1
            if consecutive_failures >= _CONSECUTIVE_FAILURE_ABORT:
                print(
                    f"memory admin: aborting after {consecutive_failures} consecutive reasoner "
                    "failures; the backend looks dead (auth? os-user? provider?)."
                )
                return 1
            continue
        consecutive_failures = 0
        proposal, skip_reason = gate_verdict(
            verdict, row=row, resolved=resolved, registry=registry, threshold=threshold
        )
        if proposal is not None:
            proposals.append(proposal)
        else:
            assert skip_reason is not None
            skips.setdefault(skip_reason, []).append(row.id)

    out_dir.mkdir(parents=True, exist_ok=True)
    header = {
        "run_id": run_id,
        "user_id": user_id,
        "threshold": threshold,
        "backend": effective_backend,
        "registry_ids": sorted(registry),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    proposals_path = out_dir / f"reclassify-{run_id}-proposals.jsonl"
    proposals_path.write_text(render_proposals(header, proposals), encoding="utf-8")
    # Transcript-provenance lookup is best-effort: any non-ok reason
    # (legacy, file missing, ts not found, drift) contributes nothing
    # and the proposal renders without the `said:` line. The lookup
    # itself happens here in the driver (not in the pure renderer) so
    # the renderer stays free of I/O for unit testing.
    provenance_user_texts = _collect_provenance_quotes(selected, proposals, user_id=user_id)
    report = render_report(
        run_id=run_id,
        user_id=user_id,
        backend=effective_backend,
        threshold=threshold,
        scanned=len(rows),
        selected=len(selected),
        proposals=proposals,
        skips=skips,
        sample_size=sample,
        texts={row.id: row.text for row, _ in selected},
        provenance_user_texts=provenance_user_texts,
    )
    report_path = out_dir / f"reclassify-{run_id}-report.md"
    report_path.write_text(report, encoding="utf-8")

    moves = sum(1 for p in proposals if p.verdict == memory.SCOPE_PROJECT)
    print(
        f"memory admin: dry-run {run_id}: scanned {len(rows)}, selected {len(selected)}, "
        f"proposed {moves} moves + {len(proposals) - moves} re-stamps."
    )
    print(f"memory admin: report:    {report_path}")
    print(f"memory admin: proposals: {proposals_path}")
    return 0


async def run_apply(config: Config, user_id: str, *, proposals_path: Path, out_dir: Path) -> int:
    """Apply a reviewed proposals file with per-row re-checks.

    Never re-runs classification: the reviewed file IS the change
    set. Every proposal row is schema-validated up front (a
    hand-edited row must abort the run BEFORE any write, not crash
    it midway). Pre-images are dumped (and fsynced) before the first
    store write; a dump failure aborts with zero changes, and an
    existing pre-image file is never truncated because it may be the
    only rollback material from an earlier apply of the same run.
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

    registry = await load_project_registry(config)

    # Re-check phase: every proposal is verified against the store
    # and registry as they are NOW. The pairs that survive carry
    # their fresh row so the pre-image dump and the write use the
    # same fetched state.
    survivors: list[tuple[Proposal, MemoryResult, ResolvedMemoryScope]] = []
    skips: dict[str, list[str]] = {}
    for proposal in proposals:
        row = memory.get_by_id(user_id=user_id, memory_id=proposal.memory_id)
        if row is None:
            skips.setdefault(SKIP_ROW_GONE, []).append(proposal.memory_id)
            continue
        resolved = memory.resolve_memory_scope(row.metadata)
        still_selected = (
            (row.metadata or {}).get("source") in memory.USER_VISIBLE_SOURCES
            and resolved.scope == memory.SCOPE_GLOBAL
            and resolved.scope_source in _ELIGIBLE_SCOPE_SOURCES
        )
        if not still_selected:
            # Covers the operator-moved-it-since-dry-run case: an
            # operator (or earlier classifier) write changes the
            # provenance, which deselects the row here.
            skips.setdefault(SKIP_DESELECTED, []).append(proposal.memory_id)
            continue
        if _text_sha256(row.text) != proposal.text_sha256:
            skips.setdefault(SKIP_TEXT_DRIFT, []).append(proposal.memory_id)
            continue
        if proposal.verdict == memory.SCOPE_PROJECT:
            pid = proposal.project_id
            if pid is None or pid not in registry:
                skips.setdefault(SKIP_UNREGISTERED_TARGET, []).append(proposal.memory_id)
                continue
            if not registry[pid].memory_enabled:
                skips.setdefault(SKIP_DISABLED_TARGET, []).append(proposal.memory_id)
                continue
        survivors.append((proposal, row, resolved))

    # Nothing to write means nothing to roll back: stop before the
    # pre-image step entirely. Writing a header-only file here would
    # at best be noise and at worst (same run id re-applied after a
    # successful first pass) truncate the previous run's rollback
    # material into an empty shell.
    if not survivors:
        skip_summary = ", ".join(f"{len(ids)} {reason}" for reason, ids in sorted(skips.items())) or "none"
        print(f"memory admin: apply {run_id}: nothing to apply; skipped: {skip_summary}.")
        return 0

    # Pre-image dump before any write; abort on failure. Exclusive
    # creation ("x"): the path derives from the run id, so a re-run
    # of the same apply would otherwise silently overwrite the only
    # copy of the original rows. fsync so a crash mid-apply cannot
    # leave changed rows with rollback material trapped in the page
    # cache.
    preimage_path = out_dir / f"reclassify-{run_id}-preimages.jsonl"
    preimage_header = {
        "run_id": run_id,
        "user_id": user_id,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "proposal_file": proposals_path.name,
    }
    preimages = [
        PreImage(memory_id=row.id, text=row.text, metadata=dict(row.metadata or {})) for _, row, _ in survivors
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
    for proposal, row, resolved in survivors:
        merged = dict(row.metadata or {})
        merged.update(
            memory.build_scope_metadata(
                scope=proposal.verdict,
                project_id=proposal.project_id,
                scope_confidence=proposal.confidence,
                scope_source=memory.SCOPE_SOURCE_CLASSIFIER,
            )
        )
        merged[memory.SCOPE_RUN_ID_KEY] = run_id
        ok = memory.update_metadata(user_id=user_id, memory_id=row.id, data=row.text, metadata=merged)
        if ok:
            applied += 1
            _emit_scope_change(
                memory_id=row.id,
                user_id=user_id,
                from_resolved=resolved,
                to_scope=proposal.verdict,
                to_project_id=proposal.project_id,
                run_id=run_id,
            )
        else:
            failed += 1

    skip_summary = ", ".join(f"{len(ids)} {reason}" for reason, ids in sorted(skips.items())) or "none"
    print(f"memory admin: apply {run_id}: applied {applied}, failed {failed}, skipped: {skip_summary}.")
    print(f"memory admin: pre-images: {preimage_path}")
    # Exit 1 only when the run attempted writes and none landed; an
    # all-benign-skip run is a valid outcome (everything changed or
    # was corrected since the dry run).
    if survivors and applied == 0:
        return 1
    return 0


async def run_rollback(config: Config, user_id: str, *, preimages_path: Path) -> int:
    """Restore rows from a pre-image file.

    Restores text and metadata exactly as dumped; Mem0 recomputes
    the embedding from the restored text, so the row returns to its
    prior retrieval behavior. Every pre-image row is schema-validated
    up front (same rationale as apply: a malformed row in a
    hand-editable restore artifact must abort the run before any
    write, not crash it midway or silently restore an empty row).
    Rows whose CURRENT provenance is operator are skipped: the
    pre-image file is rollback material for classifier writes, not a
    time machine over later operator intent.
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

    restored = 0
    failed = 0
    skips: dict[str, list[str]] = {}
    for pre in preimages:
        current = memory.get_by_id(user_id=user_id, memory_id=pre.memory_id)
        if current is None:
            skips.setdefault(SKIP_ROW_GONE, []).append(pre.memory_id)
            continue
        current_resolved = memory.resolve_memory_scope(current.metadata)
        if current_resolved.scope_source == memory.SCOPE_SOURCE_OPERATOR:
            skips.setdefault(SKIP_OPERATOR_CORRECTION, []).append(pre.memory_id)
            continue
        ok = memory.update_metadata(user_id=user_id, memory_id=pre.memory_id, data=pre.text, metadata=pre.metadata)
        if ok:
            restored += 1
            pre_resolved = memory.resolve_memory_scope(pre.metadata)
            _emit_scope_change(
                memory_id=pre.memory_id,
                user_id=user_id,
                from_resolved=current_resolved,
                to_scope=pre_resolved.scope,
                to_project_id=pre_resolved.project_id,
                run_id=run_id,
                rollback=True,
            )
        else:
            failed += 1

    skip_summary = ", ".join(f"{len(ids)} {reason}" for reason, ids in sorted(skips.items())) or "none"
    print(f"memory admin: rollback {run_id}: restored {restored}, failed {failed}, skipped: {skip_summary}.")
    attempted = restored + failed
    if attempted and restored == 0:
        return 1
    return 0
