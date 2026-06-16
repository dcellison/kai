"""
Automated cross-project collision probe corpus generator.

Reachable as `python -m kai.eval.gen_collision_probes`.

Produces a complete `kai.eval.retrieval_scoped` probe corpus from the
live Mem0 store with minimal operator touchpoints. Four probe kinds:

1. Collision (per target workspace project B): a project-B-framed
   question whose project-A row would surface under unscoped
   retrieval. The scoped pipeline's job is to keep project-A's row
   out of project-B's context.
2. Positive-only (per project): a direct question whose project-
   bound row should surface in that project's context.
3. Non-project exclusion (corpus-wide): a project-agnostic question
   whose project-bound row would surface under unscoped retrieval.
   The scoped pipeline's job is to keep the project-bound row out
   when no workspace is set.
4. Legacy-default (per project workspace, drawn from the resolver-
   global row pool): a project-context question whose resolver-
   legacy-default row should surface as a positive in that
   project's context.

The pipeline:

    discover -> draft -> verify (collision + non-project) ->
    assemble -> self-grade -> [promote gates] -> write

Self-grade goes through `kai.eval.retrieval_scoped.evaluate` and
returns `(results, metrics)`. A `ship` verdict is necessary but not
sufficient for promote: structural coverage and legacy-default
coverage gates layer on top and block promotion unless
`--allow-shortfalls` is passed. The canonical file is written via
`open(path, "xb")` so a fresh invocation cannot silently overwrite
the prior baseline; `--force` opts in to atomic `os.replace`.

Architecture notes:

- Drafting calls do NOT go through `run_review()`. The generator
  builds its own `OneShotReasoner` via `_build_drafting_reasoner`
  modeled on `kai.memory_extraction._build_memory_reasoner` so the
  per-user `models.pr_review` override wins via `resolve_user_model`
  and so the structured-log `purpose` tag (`"collision_probe_drafting"`)
  distinguishes drafting calls from PR reviews.
- Provider normalization: `resolve_classification_settings` returns
  the raw `llm_provider` cascade. For single-provider backends
  (claude->anthropic, codex->openai), we normalize via
  `get_effective_provider` before passing into `resolve_user_model`.
  Without that step, a codex user with inherited
  `config.llm_provider="anthropic"` would look up a non-existent
  `(codex, anthropic, pr_review)` registry entry.
- Embedding source: `kai.memory.embed_texts` is the public boundary.
  Every Mem0 internal-attribute hop lives in
  `_embed_via_configured_embedder` so a Mem0 rename updates one
  function, not every caller in this module.
- Legacy-default rows are resolver-global, not project-owned. The
  generator enumerates the whole-store pool of rows whose
  `resolve_memory_scope(metadata).scope_source` resolves to
  `legacy_default`, then walks them round-robin across projects to
  emit per-project legacy-default probes. When the pool is smaller
  than `len(projects) * --per-project-legacy`, row repetition
  across projects is allowed and the report records it.

Exit codes follow the existing eval-CLI convention:
- 0: dry-run or promote completed successfully.
- 1: init/IO failure or backend-looks-dead consecutive-failure abort.
- 2: settings error, blocked promote gate, or canonical path exists
     without `--force`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kai.config import (
    DATA_DIR,
    Config,
    MemoryProjectConfig,
    ModelRole,
    get_effective_provider,
    resolve_user_model,
)
from kai.eval._probes import compute_rank
from kai.eval._unscoped_recall_capture import legacy_retrieve_hits
from kai.eval.retrieval_scoped import evaluate, load_probes
from kai.oneshot import (
    ClaudeOneShotReasoner,
    CodexOneShotReasoner,
    GooseOneShotReasoner,
    OneShotError,
    OneShotReasoner,
    OpenCodeOneShotReasoner,
)

log = logging.getLogger(__name__)


# ── Defaults and constants ──────────────────────────────────────────


# Matches probe_corpus_check defaults so a dry-run that satisfies the
# generator's structural coverage gate also satisfies the standalone
# coverage check on the canonical file. If those defaults move in
# probe_corpus_check.py, mirror them here; the two checkers are
# intentionally in lockstep.
_DEFAULT_PER_PROJECT_COLLISIONS = 5
_DEFAULT_PER_PROJECT_POSITIVE = 2
_DEFAULT_NON_PROJECT = 3
_DEFAULT_PER_PROJECT_LEGACY = 2

# Pass-2 embedding-similarity threshold. Tunable via flag; 0.55 is
# empirically a reasonable cut between "the embeddings are clearly
# related" and "purely lexical co-occurrence." Lower the threshold
# when the live store has narrow per-project vocabularies.
_DEFAULT_SIMILARITY_THRESHOLD = 0.55

# Pass-3 fallback bound. When the token-filtered pass underdelivers
# for a project pair, the embedding-only fallback walks up to this
# many additional project-A rows ordered by descending centroid
# similarity. Bounded to keep the worst-case embedding cost
# predictable regardless of how large the project-A row set is.
_DEFAULT_FALLBACK_CAP = 200

# Acceptance gate for the unscoped verify step: a drafted exclusion
# probe is accepted only if the excluded row appears within the
# unscoped top-K. 20 is the unscoped pipeline's typical
# injected-prompt depth; outside that range the exclusion is
# theoretical, not a real retrieval the scoped pipeline must defend
# against.
_DEFAULT_VERIFY_TOP_K = 20

# Consecutive-failure abort guard. Matches the memory_reclassify
# convention exactly: after 5 typed reasoner failures in a row, the
# backend is treated as dead (auth, os-user, provider misconfig)
# rather than a transient issue. Lower the number and a single bad
# row stops the run; higher and the operator burns time on a
# definitively broken backend.
_CONSECUTIVE_FAILURE_ABORT = 5

# Output paths. The dry-run paths are stable so the operator can
# alias them in shell history; the canonical path versions the
# filename so a successor corpus does not silently overwrite the
# prior baseline.
_DRYRUN_PROBES_PATH = Path("/tmp/collision-probes-dryrun.jsonl")
_DRYRUN_REPORT_PATH = Path("/tmp/collision-corpus-report.md")
_DEFAULT_CANONICAL_PATH = DATA_DIR / "eval" / "probes" / "collision_v1.jsonl"

# Stopwords excluded from the Pass-1 Jaccard overlap. Token-overlap
# alone is a cheap pre-filter; the stopword list is short rather
# than exhaustive because the embedding pass catches semantic
# overlap the token pass misses. The point is to drop the obvious
# non-discriminative tokens, not to do real text normalization.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)

# Drafting purpose tag. Goes into structured logs at the reasoner
# subprocess boundary so an operator filtering for collision-probe
# drafting can distinguish those calls from PR-review one-shots.
_DRAFTING_PURPOSE = "collision_probe_drafting"

# Probe-kind tags appear in probe_id (D7) and in the report.
KIND_COLLISION = "collision"
KIND_POSITIVE = "positive"
KIND_NON_PROJECT = "non_project"
KIND_LEGACY_DEFAULT = "legacy_default"


# ── Exceptions ──────────────────────────────────────────────────────


class DraftingFailure(Exception):
    """
    A drafting call failed beyond the reasoner subprocess's typed
    error surface.

    Raised by `_draft_question` to translate `OneShotError` (and its
    subclasses) into a generator-local exception type so callers can
    handle drafting failures uniformly without importing the reasoner
    error hierarchy. The original exception is chained via `raise ...
    from exc` so the traceback still names the underlying typed
    failure.
    """


# ── Dataclasses ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Candidate:
    """
    One (project-A row, project-B context) pair that survived both
    discovery passes and is ready for drafting.

    `source_row` is the project-A row whose id becomes the excluded
    fact for the eventual collision probe. `source_project_id` and
    `target_project` carry the two projects' identities so the
    drafting prompt and probe_id can be constructed without
    re-looking-up registry state.

    `similarity` is the cosine similarity between the source row
    embedding and the target project's centroid; surfaced so the
    report can order or describe candidates by similarity. The
    `from_fallback` flag distinguishes Tier-1 from Pass-3
    embedding-only fallback so the report can call out which pairs
    needed the fallback to fill their quota.
    """

    source_row: Any  # MemoryResult; typed as Any to avoid the import cycle through kai.memory at module-import time
    source_project_id: str
    target_project: MemoryProjectConfig
    similarity: float
    from_fallback: bool


@dataclass(frozen=True)
class DraftedProbe:
    """
    A probe in flight between drafting and verification.

    Carries the drafted question plus enough metadata to construct
    the final `ScopedProbe` if verification accepts. For verified
    kinds (collision, non-project), `excluded_row_id` is the legacy-
    harness target; for positive kinds (positive-only, legacy-
    default), `excluded_row_id` is None and `expected_fact_id`
    carries the row id.

    `probe_id` is the stable identifier built at draft time so
    the same probe survives reject/rerun cycles.

    `workspace` follows the corpus-shape table in D6: project
    workspace path for collision/positive/legacy-default; None for
    non-project exclusion.
    """

    kind: str
    probe_id: str
    question: str
    excluded_row_id: str | None
    expected_fact_id: str | None
    workspace: str | None


@dataclass(frozen=True)
class VerifiedProbe:
    """
    A drafted probe that passed any required verification gate and
    is ready to write to JSONL.

    Verification status is implicit at this stage: a probe in this
    bucket has been accepted. The legacy rank lives on the probe so
    the report can render it in the Non-project exclusion
    verification section. None for kinds that do not run the legacy
    verify gate (positive-only, legacy-default).
    """

    kind: str
    probe_id: str
    question: str
    expected_fact_id: str | None
    expected_excluded_fact_ids: tuple[str, ...]
    workspace: str | None
    legacy_rank: int | None


@dataclass
class DroppedProbe:
    """
    A drafted probe that the unscoped verify gate rejected.

    Tracked separately so the report can name dropped probes by
    `probe_id` and explain why each was dropped (rank None vs.
    rank > verify_top_k). Counts toward the shortfall column for
    its kind.
    """

    kind: str
    probe_id: str
    question: str
    excluded_row_id: str
    reason: str  # "rank_none" | "rank_out_of_top_k"
    observed_rank: int | None


@dataclass
class GenerationConfig:
    """
    Parsed CLI flags plus resolved backend / model / paths.

    One frozen-ish snapshot of every input the orchestrator needs so
    the orchestrator does not re-read argparse Namespace fields or
    call resolver helpers at multiple points. Populated by
    `_resolve_run_config` after `init_memory` succeeds.
    """

    user_id: str
    effective_backend: str
    effective_os_user: str | None
    effective_provider: str
    model: str
    timeout_s: int
    project_filter: list[str] | None
    per_project_collisions: int
    per_project_positive: int
    non_project_quota: int
    per_project_legacy: int
    similarity_threshold: float
    fallback_cap: int
    verify_top_k: int
    output_path: Path
    promote: bool
    force: bool
    allow_shortfalls: bool
    reject_ids: set[str]


@dataclass
class GeneratorReport:
    """
    All state the report renderer needs after the run completes.

    Populated incrementally through the pipeline: discovery counts
    after Pass 1/2/3, accepted/dropped counts after verification,
    legacy-default pool size and allocation map after the legacy-
    default step, self-grade verdict and metrics after evaluate,
    promote-gate pass/block decisions, and the optional
    forced_overwrite / allow_shortfalls bookkeeping.

    The renderer (`_render_report`) only reads this dataclass; no
    cross-checking against the live store at render time, so the
    report is deterministic given the populated state.
    """

    user_id: str
    generated_at: str
    effective_backend: str
    effective_provider: str
    effective_os_user: str | None
    model: str
    timeout_s: int
    per_project_summary: dict[str, dict[str, int]] = field(default_factory=dict)
    accepted_probes: list[VerifiedProbe] = field(default_factory=list)
    dropped_probes: list[DroppedProbe] = field(default_factory=list)
    legacy_default_pool_size: int = 0
    legacy_default_allocation: dict[str, list[str]] = field(default_factory=dict)  # project_id -> list of row ids
    legacy_default_row_repeated: bool = False
    self_grade_verdict: str = "skipped"
    n_scored_negative: int = 0
    exclusion_pass_in_prompt: float = 0.0
    exclusion_pass_in_candidates: float = 0.0
    leak_records: list[dict[str, Any]] = field(default_factory=list)
    promote_outcome: str = "skipped"
    promote_gate_results: dict[str, str] = field(default_factory=dict)
    forced_overwrite: Path | None = None
    allow_shortfalls_applied: dict[str, str] = field(default_factory=dict)
    rejected_found: list[str] = field(default_factory=list)
    rejected_missing: list[str] = field(default_factory=list)
    pair_fallback_counts: dict[str, int] = field(default_factory=dict)  # target_project_id -> count from fallback


# ── CLI parser ──────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Argparse for `python -m kai.eval.gen_collision_probes`.

    Resolver-override flags (--backend / --provider / --os-user /
    --model) mirror the memory_reclassify admin-CLI pattern: each
    overrides one stage of the per-user backend/provider/model
    cascade. --timeout-s overrides the drafting subprocess timeout;
    default is config.pr_review_timeout_s because the drafting
    workload borrows the PR-review role.

    Coverage flags (--per-project-collisions, etc.) match
    probe_corpus_check's defaults so a clean dry-run also satisfies
    the standalone coverage check on the canonical file.

    Output flags split: --output overrides the canonical promote
    target only; the /tmp/ dry-run paths are not overridable because
    they are transient and the operator iterates on them. --promote
    flips the dry-run to promote attempt; --force is the explicit
    overwrite opt-in; --allow-shortfalls relaxes the two coverage
    gates but never the self-grade or exclusive-create gates.
    """
    parser = argparse.ArgumentParser(
        prog="python -m kai.eval.gen_collision_probes",
        description=(
            "Generate a cross-project collision probe corpus for kai.eval.retrieval_scoped from the live Mem0 store."
        ),
    )
    parser.add_argument(
        "user_id",
        help="Telegram chat_id (string) whose memory store to enumerate.",
    )
    # ── Resolver overrides ───────────────────────────────────────
    parser.add_argument(
        "--backend",
        choices=["claude", "codex", "goose", "opencode"],
        help=(
            "Override the target user's effective backend. Default: "
            "resolve from user config via memory_reclassify's "
            "resolver chain."
        ),
    )
    parser.add_argument(
        "--provider",
        help=(
            "Override the raw provider before normalization. Use the "
            "wire-level value the user's config would carry "
            "(e.g. 'openai', 'anthropic', 'deepseek'). The "
            "single-provider backends are normalized via "
            "get_effective_provider before resolve_user_model."
        ),
    )
    parser.add_argument(
        "--os-user",
        help="Override the sudo target user for the reasoner subprocess.",
    )
    parser.add_argument(
        "--model",
        default="",
        help=(
            "Override the resolved model directly. Bypasses "
            "resolve_user_model entirely; the value is passed to the "
            "reasoner as-is."
        ),
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        help=(
            "Drafting subprocess timeout in seconds. Default: "
            "config.pr_review_timeout_s. Tune down for one-liner "
            "drafting calls; full PR-review timeout is overkill."
        ),
    )
    # ── Discovery / corpus shape ─────────────────────────────────
    parser.add_argument(
        "--projects",
        help=(
            "Comma-separated subset of project ids to consider. "
            "Default: every project in config.memory_projects (YAML "
            "registry only; DB-registered projects are out of scope "
            "for this revision)."
        ),
    )
    parser.add_argument(
        "--per-project-collisions",
        type=int,
        default=_DEFAULT_PER_PROJECT_COLLISIONS,
        help=(
            "Verified collisions targeting each workspace project "
            "(per target project, not per source-pair). Default: 5."
        ),
    )
    parser.add_argument(
        "--per-project-positive",
        type=int,
        default=_DEFAULT_PER_PROJECT_POSITIVE,
        help="Positive-only probes per project. Default: 2.",
    )
    parser.add_argument(
        "--non-project",
        type=int,
        default=_DEFAULT_NON_PROJECT,
        help=("Non-project exclusion probes (corpus-wide, NOT per project). Default: 3."),
    )
    parser.add_argument(
        "--per-project-legacy",
        type=int,
        default=_DEFAULT_PER_PROJECT_LEGACY,
        help=("Legacy-default probes per project workspace. Rows come from the resolver-global pool. Default: 2."),
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=_DEFAULT_SIMILARITY_THRESHOLD,
        help=(
            "Pass-2 cosine similarity cutoff against the target "
            "project's centroid. Default: 0.55. Lower for narrow "
            "per-project vocabularies."
        ),
    )
    parser.add_argument(
        "--fallback-cap",
        type=int,
        default=_DEFAULT_FALLBACK_CAP,
        help=("Pass-3 embedding-only fallback cap per project pair. Default: 200."),
    )
    parser.add_argument(
        "--verify-top-k",
        type=int,
        default=_DEFAULT_VERIFY_TOP_K,
        help=(
            "Legacy harness top-K depth for exclusion verification. "
            "A drafted collision or non-project exclusion is "
            "accepted only if the excluded row appears within this "
            "many legacy hits. Default: 20."
        ),
    )
    # ── Output and promote ───────────────────────────────────────
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_CANONICAL_PATH,
        help=("Canonical promote target. Default: <data-dir>/eval/probes/collision_v1.jsonl."),
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help=(
            "Attempt to copy the dry-run corpus to the canonical "
            "path. Promotion is blocked by self-grade, structural "
            "coverage, legacy-default coverage, and exclusive-create "
            "gates. Without --promote the canonical path is never "
            "touched."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "On --promote with a pre-existing canonical file, "
            "atomically replace via os.replace and record the "
            "overwrite in the report. Without --force, a "
            "pre-existing target blocks promotion."
        ),
    )
    parser.add_argument(
        "--allow-shortfalls",
        action="store_true",
        help=(
            "Relax structural-coverage and legacy-default-coverage "
            "promote gates. Self-grade and exclusive-create gates "
            "still block. The report records every relaxed gate."
        ),
    )
    parser.add_argument(
        "--reject",
        default="",
        help=(
            "Comma-separated probe_id values to drop from the "
            "previous dry-run JSONL. The generator does NOT re-draft "
            "or re-verify; it loads the previous JSONL, removes the "
            "named ids, rebuilds the line-number mapping, and "
            "re-self-grades. Unknown ids produce a warning, not an "
            "abort."
        ),
    )
    return parser


def _parse_reject_ids(raw: str) -> set[str]:
    """Split the --reject CSV argument, ignoring empty entries.

    Whitespace around items is stripped so an operator pasting a
    comma-separated list with spaces (e.g. from a report listing)
    does not silently include " probe_id_here" with a leading space
    that would fail to match.
    """
    if not raw:
        return set()
    return {token.strip() for token in raw.split(",") if token.strip()}


def _parse_projects_filter(raw: str | None) -> list[str] | None:
    """Same CSV split as --reject but preserves order.

    Order matters for the report's per-project summary, which renders
    in the order the operator named projects. Returning None when
    no flag is passed signals "every project in the YAML registry."
    """
    if not raw:
        return None
    return [token.strip() for token in raw.split(",") if token.strip()]


# ── Init ────────────────────────────────────────────────────────────


def _initialize_memory_or_exit() -> Config | None:
    """Load config and call init_memory; return Config or None on failure.

    Mirrors the discipline from retrieval_scoped._initialize_memory:
    the generator reads the live store, so the same init path keeps
    the embedding model, Qdrant directory, and per-user state
    consistent with the bot runtime. Returns None on any failure
    after printing a diagnostic to stderr; the caller maps None to
    exit code 1.

    The single broad except is intentional: load_config and
    init_memory can fail in many ways (missing env, broken Qdrant,
    misconfigured embedder), and the generator's failure surface to
    the operator is the same in every case: "init failed: <reason>"
    is more useful than a Python traceback for a CLI invocation.
    """
    try:
        from kai.config import load_config
        from kai.memory import init_memory, is_enabled

        config = load_config()
        init_memory(config)
        if not is_enabled():
            print(
                "gen_collision_probes: memory is not enabled. "
                "Set MEMORY_ENABLED=true and verify the store is readable.",
                file=sys.stderr,
            )
            return None
        return config
    except Exception as e:
        print(f"gen_collision_probes: init failed: {e}", file=sys.stderr)
        return None


# ── Resolver chain ──────────────────────────────────────────────────


def _build_drafting_reasoner(
    effective_backend: str,
    *,
    os_user: str | None,
    provider: str,
) -> OneShotReasoner:
    """
    Build the OneShotReasoner that matches the effective backend.

    Mirrors `kai.memory_extraction._build_memory_reasoner` in shape
    but lives here so the generator does not import the extraction-
    flavored error text from that helper. The four valid backends
    match `ONESHOT_REASONER_BACKENDS`; the RuntimeError branch is a
    defensive net for a future change that widens the set without
    updating this dispatch.

    `provider` is consumed by goose only (goose carries the wire-
    name on its argv); the other backends derive provider implicitly
    or embed it in the model string.
    """
    if effective_backend == "claude":
        return ClaudeOneShotReasoner(os_user=os_user)
    if effective_backend == "codex":
        return CodexOneShotReasoner(os_user=os_user)
    if effective_backend == "opencode":
        return OpenCodeOneShotReasoner(os_user=os_user)
    if effective_backend == "goose":
        return GooseOneShotReasoner(os_user=os_user, provider=provider)
    raise RuntimeError(
        f"gen_collision_probes: unsupported effective backend "
        f"{effective_backend!r}; expected one of "
        f"claude/codex/opencode/goose"
    )


def _resolve_run_config(
    args: argparse.Namespace,
    config: Config,
) -> GenerationConfig | str:
    """
    Resolve every input the orchestrator needs from CLI flags + config.

    Returns either a populated `GenerationConfig` or an error message
    string (mapped by the caller to exit code 2). The string-return
    convention follows `resolve_classification_settings`: a misconfig
    surfaces as one human-readable line, not as a Python traceback.

    Provider normalization is the key step `resolve_classification_settings`
    does NOT do (its `_resolve_effective_provider` returns the raw
    `llm_provider` cascade). For single-provider backends, the raw
    value can be the wrong provider for the registry lookup. The
    explicit `get_effective_provider(backend, raw)` call here is the
    correct fix; without it, a codex user with inherited
    `config.llm_provider="anthropic"` fails to resolve any model.
    """
    from kai.memory_reclassify import (
        _resolve_user_config,
        resolve_classification_settings,
    )

    settings = resolve_classification_settings(
        config,
        args.user_id,
        backend=args.backend,
        os_user=args.os_user,
        provider=args.provider,
    )
    if isinstance(settings, str):
        return settings
    effective_backend, effective_os_user, raw_provider = settings

    # Normalize for single-provider backends before passing into
    # resolve_user_model. The raw cascade value can be the wrong
    # provider when the user inherits a global llm_provider set for
    # a different backend (e.g. a codex user inheriting
    # llm_provider="anthropic" would otherwise miss the
    # (codex, openai, pr_review) registry entry).
    effective_provider = get_effective_provider(effective_backend, raw_provider)

    user_cfg = _resolve_user_config(args.user_id, config)
    model = args.model or resolve_user_model(
        ModelRole.PR_REVIEW,
        user_cfg,
        config,
        backend=effective_backend,
        provider=effective_provider,
    )

    timeout_s = args.timeout_s if args.timeout_s is not None else config.pr_review_timeout_s

    return GenerationConfig(
        user_id=args.user_id,
        effective_backend=effective_backend,
        effective_os_user=effective_os_user,
        effective_provider=effective_provider,
        model=model,
        timeout_s=timeout_s,
        project_filter=_parse_projects_filter(args.projects),
        per_project_collisions=args.per_project_collisions,
        per_project_positive=args.per_project_positive,
        non_project_quota=args.non_project,
        per_project_legacy=args.per_project_legacy,
        similarity_threshold=args.similarity_threshold,
        fallback_cap=args.fallback_cap,
        verify_top_k=args.verify_top_k,
        output_path=args.output,
        promote=args.promote,
        force=args.force,
        allow_shortfalls=args.allow_shortfalls,
        reject_ids=_parse_reject_ids(args.reject),
    )


async def _draft_question(
    prompt_text: str,
    *,
    reasoner: OneShotReasoner,
    model: str,
    timeout_s: int,
) -> str:
    """
    Run one drafting prompt through the configured reasoner.

    Wraps every typed `OneShotError` (timeout, subprocess, output)
    into `DraftingFailure` so the orchestrator's consecutive-failure
    abort guard can count failures uniformly without importing the
    reasoner error hierarchy.

    `purpose="collision_probe_drafting"` lands in the reasoner's
    structured-log line at subprocess boundary so an operator
    filtering by purpose can distinguish drafting calls from PR
    reviews; this is the entire reason the generator does not
    reuse `run_review()` (which hardcodes `purpose="pr_review"`).

    The `.strip()` on the result drops the trailing newlines most
    backends append; downstream code that hashes the question for
    probe_id disambiguation does not need to special-case them.
    """
    try:
        result = await reasoner.run(
            prompt=prompt_text,
            system_prompt=None,
            model=model,
            timeout=timeout_s,
            purpose=_DRAFTING_PURPOSE,
            json_schema=None,
        )
    except OneShotError as exc:
        raise DraftingFailure(str(exc)) from exc
    return result.text.strip()


# ── Project enumeration and row partitioning ────────────────────────


def _project_workspace_root_str(project: MemoryProjectConfig) -> str:
    """
    Return the canonical workspace_root string for probe construction.

    MemoryProjectConfig.workspace_roots is a tuple because the same
    project can carry multiple roots (e.g. a symlinked clone and the
    canonical path). The first entry is the one the generator pins
    `workspace` to on emitted probes; the registry's resolver also
    treats it as canonical for display purposes. The tuple has at
    least one element by construction (load_db_registry rejects
    empty workspace_roots).
    """
    return str(project.workspace_roots[0])


def _row_matches_project(
    row_metadata: dict[str, Any] | None,
    project: MemoryProjectConfig,
) -> bool:
    """
    Test whether a row belongs to the named project.

    Primary signal is `metadata["project_id"]`. Fallback path: rows
    written before project_id was a stored field have only
    `metadata["workspace_root"]`, and the generator must still
    partition them. The fallback compares against every workspace_root
    string in the registry entry, not just the canonical one, so a
    row written under a symlinked root still matches its registered
    project.
    """
    if not row_metadata:
        return False
    pid = row_metadata.get("project_id")
    if pid:
        # project_id is the authoritative owner when present. A row
        # whose project_id explicitly names a different project must
        # NOT match here even if its workspace_root happens to point
        # at this project's root (which can happen after a workspace
        # rename if the row's workspace_root metadata is stale).
        # Falling through to the workspace_root fallback in that case
        # would contaminate this project's row pool with rows that
        # belong to a sibling.
        return pid == project.project_id
    # Fallback: match by workspace_root for rows missing project_id
    # entirely (legacy rows written before project_id became a
    # stored field). Stringify both sides because workspace_root in
    # metadata is always a string while MemoryProjectConfig holds
    # Path objects.
    workspace_root = row_metadata.get("workspace_root")
    if not workspace_root:
        return False
    return any(str(workspace_root) == str(root) for root in project.workspace_roots)


def _enumerate_project_rows(
    project: MemoryProjectConfig,
    user_id: str,
) -> list[Any]:
    """
    Return all fact and episode rows that belong to one project.

    Combines `get_all_facts` and `get_all_episodes` since both row
    types are eligible as collision sources. Episodes carry the same
    `project_id` metadata convention as facts. Rows missing
    project_id fall back to workspace_root matching via
    `_row_matches_project`. The returned list is sorted by `row.id`
    for the determinism contract in D10.

    Note the deferred import: `kai.memory` pulls in PyTorch and the
    embedder at module-import time, which is too expensive when only
    the dataclasses or argparse are needed (e.g. in a test that
    instantiates GenerationConfig directly).
    """
    from kai.memory import get_all_episodes, get_all_facts

    rows: list[Any] = []
    for row in get_all_facts(user_id=user_id):
        if _row_matches_project(row.metadata, project):
            rows.append(row)
    for row in get_all_episodes(user_id=user_id):
        if _row_matches_project(row.metadata, project):
            rows.append(row)
    rows.sort(key=lambda r: r.id)
    return rows


# ── Token-overlap pre-filter and stopword helpers ───────────────────


# Regex used to split row text and workspace-path basenames into
# tokens. Splits on whitespace, slashes, underscores, hyphens, and
# dots so a workspace path like "/Users/kai/Projects/anvil" yields
# "users kai projects anvil" rather than the single full string.
_TOKEN_SPLIT_RE = re.compile(r"[/_\-\.\s]+")


def _tokenize(text: str) -> set[str]:
    """
    Split text into lowercased tokens minus stopwords.

    Returns a set, not a list: Jaccard overlap is over unique tokens.
    Tokens shorter than 2 characters are dropped because single-
    character tokens (digits, "a", etc.) collide between any two
    projects and would inflate the Jaccard score artificially.
    """
    tokens: set[str] = set()
    for raw in _TOKEN_SPLIT_RE.split(text.lower()):
        if len(raw) < 2:
            continue
        if raw in _STOPWORDS:
            continue
        tokens.add(raw)
    return tokens


def _build_project_token_union(
    rows: list[Any],
    project: MemoryProjectConfig,
) -> set[str]:
    """
    Compute the project's distinctive token set for Pass-1 overlap.

    Includes the union of every row's tokens plus the tokens drawn
    from each registered workspace_root path. The workspace path
    tokens carry project names (e.g. "anvil", "phi") that anchor
    overlap when row text alone does not contain those terms.

    Stopwords are already excluded by `_tokenize`.
    """
    union: set[str] = set()
    for row in rows:
        union |= _tokenize(row.text or "")
    for root in project.workspace_roots:
        union |= _tokenize(str(root))
    return union


def _has_distinctive_overlap(
    source_tokens: set[str],
    target_token_union: set[str],
) -> bool:
    """
    Pass-1 gate: at least one shared non-stopword token survives.

    "Distinctive" is shorthand for "non-stopword AND length >= 2",
    which `_tokenize` already enforces on both sides. Using set
    intersection truthiness skips building a Jaccard score we do not
    need; the threshold is "any overlap at all".
    """
    return bool(source_tokens & target_token_union)


# ── Embedding helpers ───────────────────────────────────────────────


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Standard cosine similarity over two equal-length float vectors.

    Pure math, no numpy: keeps the module's import footprint small
    and avoids a numpy version-compat surface for the generator. Mem0
    embeddings are 384-dim per init validation, so 384 multiplies per
    pair is cheap enough to compute in Python directly. Returns 0.0
    for zero-magnitude vectors rather than NaN; downstream threshold
    checks then treat them as "no similarity" instead of propagating
    NaN through ordering.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b, strict=False):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / ((norm_a**0.5) * (norm_b**0.5))


def _compute_centroid(vectors: list[list[float]]) -> list[float]:
    """
    Mean per-dimension over a list of vectors.

    Used as the project-B reference for Pass-2 similarity. Empty
    input returns an empty list; downstream `_cosine_similarity`
    handles the empty case by returning 0.0, so a project with no
    facts contributes no Tier-1 collisions.
    """
    if not vectors:
        return []
    dims = len(vectors[0])
    sums = [0.0] * dims
    for vec in vectors:
        for i, v in enumerate(vec):
            sums[i] += v
    n = float(len(vectors))
    return [s / n for s in sums]


def _embed_rows_text(rows: list[Any]) -> list[list[float]]:
    """
    Batch-embed the text field of every row.

    Lifts the embedder call out of the Pass-2 loop so a project's
    full row set is embedded once per generator run rather than
    once per (project_A, project_B) pair. `kai.memory.embed_texts`
    is the public boundary; the actual Mem0 attribute hop lives
    behind `_embed_via_configured_embedder` per PR A.

    Empty input returns an empty list without touching the embedder.
    """
    from kai.memory import embed_texts

    if not rows:
        return []
    return embed_texts([row.text or "" for row in rows])


# ── TF-IDF for positive-only and non-project source selection ───────


def _project_tfidf_scores(
    project_rows: list[Any],
    all_rows_by_project: dict[str, list[Any]],
) -> dict[str, float]:
    """
    Compute a per-row TF-IDF score for one project's rows.

    Highest-scoring rows are the most distinctive within that
    project relative to the corpus of all projects. The drafter
    prefers them for positive-only and non-project source selection
    because a generic row would be answerable from many projects'
    contexts and would not exercise the scope safety polarity.

    Standard TF-IDF: term frequency normalized by row token count;
    inverse document frequency is log(N_projects / df) where
    df is the number of projects whose row union contains the
    term. Returns one score per row id; rows with zero tokens get
    score 0.0.
    """
    import math

    num_projects = len(all_rows_by_project)
    # df = number of projects whose row-union contains the term.
    df: dict[str, int] = {}
    project_token_unions: dict[str, set[str]] = {}
    for pid, rows in all_rows_by_project.items():
        union: set[str] = set()
        for row in rows:
            union |= _tokenize(row.text or "")
        project_token_unions[pid] = union
        for tok in union:
            df[tok] = df.get(tok, 0) + 1

    # Per-row TF-IDF: sum of (tf * idf) over the row's tokens.
    scores: dict[str, float] = {}
    for row in project_rows:
        tokens = _tokenize(row.text or "")
        if not tokens:
            scores[row.id] = 0.0
            continue
        score = 0.0
        # tf is uniform per unique token (set-based tokenization),
        # so this reduces to sum-of-idf over unique tokens in the row.
        for tok in tokens:
            doc_freq = df.get(tok, 0)
            if doc_freq == 0:
                continue
            idf = math.log(num_projects / doc_freq) if num_projects > 0 else 0.0
            score += idf
        scores[row.id] = score / len(tokens)
    return scores


def _top_tfidf_rows(
    rows: list[Any],
    tfidf_scores: dict[str, float],
    limit: int,
) -> list[Any]:
    """
    Pick the highest-TF-IDF rows from one project's row list.

    Ordering: descending TF-IDF score, ties broken by row.id ascending
    for determinism (D10). Returns at most `limit` rows; fewer when
    the project does not have enough rows.
    """
    sorted_rows = sorted(rows, key=lambda r: (-tfidf_scores.get(r.id, 0.0), r.id))
    return sorted_rows[:limit]


# ── Drafting prompts ────────────────────────────────────────────────


def _render_collision_prompt(
    target_project: MemoryProjectConfig,
    target_summary_rows: list[Any],
    source_row: Any,
) -> str:
    """
    Build the collision drafting prompt for one candidate pair.

    The 5-row target summary is a deterministic projection: callers
    select the highest-TF-IDF project-B rows up front and pass them
    in here. The prompt does NOT name raw row ids on either side so
    an LLM playing back the prompt as its answer cannot leak ids
    into the question text (which would later poison the probe by
    making it a trivial keyword retrieval).
    """
    # Build the inline summary from each target row's text;
    # truncate aggressively because the LLM only needs flavor, not
    # exhaustive context, and the prompt budget is small.
    summary_lines = []
    for row in target_summary_rows:
        snippet = (row.text or "").strip().replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        summary_lines.append(snippet)
    summary = ", ".join(summary_lines) if summary_lines else "(no facts available)"

    return (
        f"Project B's workspace is rooted at {_project_workspace_root_str(target_project)}. "
        f"From a sample of project-B facts: {summary}.\n\n"
        f"Project A has the following fact: {source_row.text}\n\n"
        f"Write one natural-language question that someone working in project B "
        f"might realistically ask, whose semantic embedding would land close to "
        f"project A's fact text. The question must be plausible from project B's "
        f"perspective. The answer should be specific to project B; project A's "
        f"fact should be a wrong-scope surface that the scoped retrieval pipeline "
        f"must filter out.\n\n"
        f"Return only the question. No preamble, no explanation."
    )


def _render_positive_only_prompt(source_row: Any) -> str:
    """
    Build the positive-only drafting prompt for one project-bound row.

    Used for the positive-only kind (D6 row 2) and the legacy-default
    kind (D6 row 4) which shares this template.
    """
    return (
        f"The following fact is true:\n\n"
        f"{source_row.text}\n\n"
        f"Write one natural-language question that someone might "
        f"realistically ask whose answer is this fact. Return only "
        f"the question."
    )


def _render_non_project_prompt(source_row: Any) -> str:
    """
    Build the non-project exclusion drafting prompt.

    The "must not name a specific project, workspace, or codebase"
    instruction is the structural anchor for the kind: the verifier
    accepts only when the project-bound row surfaces under the
    project-agnostic question via legacy retrieval.
    """
    return (
        f"The following fact is true:\n\n"
        f"{source_row.text}\n\n"
        f"Write a natural-language question that someone unfamiliar "
        f"with any specific project might ask whose answer is this "
        f"fact. The question must not name a specific project, "
        f"workspace, or codebase. Return only the question."
    )


# ── Candidate discovery ────────────────────────────────────────────


def _discover_tier1_candidates(
    source_project_id: str,
    source_rows: list[Any],
    source_embeddings: list[list[float]],
    target_project: MemoryProjectConfig,
    target_token_union: set[str],
    target_centroid: list[float],
    similarity_threshold: float,
) -> list[Candidate]:
    """
    Run Pass 1 (token overlap) then Pass 2 (centroid similarity).

    Returns Tier-1 candidates sorted by descending similarity, ties
    broken by source row id ascending (D10). Only candidates that
    cleared BOTH passes appear.

    `source_embeddings` is parallel to `source_rows` so the embedding
    for a given row is `source_embeddings[i]` when row is at index i.
    Built once per source project outside this helper.
    """
    candidates: list[Candidate] = []
    for i, row in enumerate(source_rows):
        # Pass 1: at least one distinctive token survives.
        if not _has_distinctive_overlap(_tokenize(row.text or ""), target_token_union):
            continue
        # Pass 2: cosine similarity vs the target centroid.
        if not target_centroid:
            continue
        sim = _cosine_similarity(source_embeddings[i], target_centroid)
        if sim < similarity_threshold:
            continue
        candidates.append(
            Candidate(
                source_row=row,
                source_project_id=source_project_id,
                target_project=target_project,
                similarity=sim,
                from_fallback=False,
            )
        )
    # Determinism: descending similarity, ties by row.id ascending.
    candidates.sort(key=lambda c: (-c.similarity, c.source_row.id))
    return candidates


def _discover_fallback_candidates(
    source_project_id: str,
    source_rows: list[Any],
    source_embeddings: list[list[float]],
    target_project: MemoryProjectConfig,
    target_centroid: list[float],
    fallback_cap: int,
    already_seen_row_ids: set[str],
) -> list[Candidate]:
    """
    Pass 3: embedding-only fallback for project-B's underfilled quota.

    Skips the token filter and considers up to `fallback_cap` rows
    ordered by descending centroid similarity. Rows already produced
    as Tier-1 candidates for this pair are excluded so the fallback
    does not redraft the same source twice. Marked
    `from_fallback=True` so the report can call out which pairs
    needed the fallback.
    """
    if not target_centroid or fallback_cap <= 0:
        return []
    # Compute similarity for every non-seen row and take the top N.
    similarities: list[tuple[float, int, Any]] = []
    for i, row in enumerate(source_rows):
        if row.id in already_seen_row_ids:
            continue
        sim = _cosine_similarity(source_embeddings[i], target_centroid)
        similarities.append((sim, i, row))
    similarities.sort(key=lambda t: (-t[0], t[2].id))
    candidates: list[Candidate] = []
    for sim, _idx, row in similarities[:fallback_cap]:
        candidates.append(
            Candidate(
                source_row=row,
                source_project_id=source_project_id,
                target_project=target_project,
                similarity=sim,
                from_fallback=True,
            )
        )
    return candidates


# ── Verification ────────────────────────────────────────────────────


async def _verify_exclusion(
    question: str,
    user_id: str,
    excluded_row_id: str,
    verify_top_k: int,
) -> tuple[bool, int | None]:
    """
    Run the unscoped recall pipeline; accept if excluded_row_id is in top-K.

    Returns (accepted, observed_rank). `observed_rank` is the
    1-indexed unscoped rank of the excluded row, or None when the
    row did not appear at all. The report uses both: a None rank
    reads as "drafted question did not retrieve the excluded row";
    a rank > verify_top_k reads as "retrieved but outside the gate."

    The unscoped pipeline runs for collision probes (excluded row =
    project-A row pinned to project-B context) and non-project
    exclusion probes (excluded row = project-bound row, workspace=
    None context). Same gate, same helper.
    """
    hits, _latency_ms = await legacy_retrieve_hits(question, user_id=user_id)
    rank = compute_rank(hits, excluded_row_id)
    accepted = rank is not None and rank <= verify_top_k
    return accepted, rank


# ── Orchestrator: drafting + verification with abort guard ──────────


@dataclass
class _AbortState:
    """
    Tracks consecutive drafting failures across heterogeneous loops.

    The same counter spans every kind's drafting calls because a
    backend that has gone dead does not magically recover between
    collision drafts and positive-only drafts. The counter resets
    on every successful draft so the abort guard catches a sustained
    failure burst rather than every transient hiccup.
    """

    consecutive_failures: int = 0


def _abort_message(state: _AbortState, gen_config: GenerationConfig) -> str:
    """Build the operator-facing 'backend looks dead' line."""
    return (
        f"gen_collision_probes: aborting after {state.consecutive_failures} "
        f"consecutive drafting failures; the backend looks dead "
        f"(backend={gen_config.effective_backend}, "
        f"provider={gen_config.effective_provider}, "
        f"os_user={gen_config.effective_os_user})."
    )


async def _draft_and_count(
    prompt: str,
    *,
    reasoner: OneShotReasoner,
    gen_config: GenerationConfig,
    state: _AbortState,
) -> str | None:
    """
    Wrap `_draft_question` with the consecutive-failure abort guard.

    Returns the drafted question on success and None on a single
    drafting failure. Raises `_AbortException` when the consecutive-
    failure threshold is reached so the orchestrator can unwind
    every nested loop without threading "did the abort fire?" back
    through every return value.
    """
    try:
        text = await _draft_question(
            prompt,
            reasoner=reasoner,
            model=gen_config.model,
            timeout_s=gen_config.timeout_s,
        )
    except DraftingFailure as exc:
        log.warning("gen_collision_probes: drafting failed: %s", exc)
        state.consecutive_failures += 1
        if state.consecutive_failures >= _CONSECUTIVE_FAILURE_ABORT:
            raise _AbortException(_abort_message(state, gen_config)) from exc
        return None
    state.consecutive_failures = 0
    return text


class _AbortException(Exception):
    """
    Signals the orchestrator to stop and return exit code 1.

    Raised by `_draft_and_count` when the consecutive-failure abort
    guard fires. The message carries the operator-facing diagnostic
    so the caller only needs to print it and propagate the code.
    """


# ── Per-target-project collision quota loop ────────────────────────


def _build_probe_id(kind: str, *parts: str) -> str:
    """
    Compose a stable probe_id from a kind tag and trailing parts.

    Format: `<kind>:<part1>:<part2>:...`. Disambiguation
    (`:dup<n>` suffix) happens in the writer, not here, so this
    function returns the canonical form regardless of whether a
    later step has to deduplicate.
    """
    return ":".join((kind, *parts))


async def _draft_and_verify_collisions(
    gen_config: GenerationConfig,
    reasoner: OneShotReasoner,
    sorted_projects: list[MemoryProjectConfig],
    rows_by_project: dict[str, list[Any]],
    embeddings_by_project: dict[str, list[list[float]]],
    token_unions_by_project: dict[str, set[str]],
    centroids_by_project: dict[str, list[float]],
    tfidf_by_project: dict[str, dict[str, float]],
    abort_state: _AbortState,
    report: GeneratorReport,
) -> tuple[list[VerifiedProbe], list[DroppedProbe]]:
    """
    Per-target-project collision quota loop.

    For each project B (sorted by project_id), accumulate accepted
    collisions until project B's quota is full or every source project
    has been visited. Tier-1 candidates take priority; the
    embedding-only fallback runs only when Tier-1 underdelivers for
    the pair. The same drafter is reused across pairs to amortize
    process startup; the abort guard accumulates across the whole
    loop because a sustained drafting failure burst is the same
    backend-dead signal regardless of which pair was being drafted.

    Tier-1 collisions are drafted in candidate order (descending
    similarity, ties by row id). Fallback collisions are drafted
    after Tier-1 is exhausted so the report can distinguish "filled
    from token-aligned candidates" from "filled from embedding-only
    fallback."
    """
    accepted: list[VerifiedProbe] = []
    dropped: list[DroppedProbe] = []

    for project_b in sorted_projects:
        accepted_for_b = 0
        # Track which source row ids have already produced a
        # candidate against this target so the fallback does not
        # redraft duplicates. The set is keyed by source row id
        # alone because a row that already collided with project B
        # via Tier-1 cannot offer a separate fallback collision.
        seen_row_ids: set[str] = set()
        fallback_count_for_b = 0

        # Source projects iterated in project_id order for the
        # determinism contract (same store state -> same candidate
        # set in the same order).
        for project_a in sorted_projects:
            if project_a.project_id == project_b.project_id:
                continue
            if accepted_for_b >= gen_config.per_project_collisions:
                break

            tier1 = _discover_tier1_candidates(
                source_project_id=project_a.project_id,
                source_rows=rows_by_project.get(project_a.project_id, []),
                source_embeddings=embeddings_by_project.get(project_a.project_id, []),
                target_project=project_b,
                target_token_union=token_unions_by_project.get(project_b.project_id, set()),
                target_centroid=centroids_by_project.get(project_b.project_id, []),
                similarity_threshold=gen_config.similarity_threshold,
            )
            for candidate in tier1:
                if accepted_for_b >= gen_config.per_project_collisions:
                    break
                seen_row_ids.add(candidate.source_row.id)
                # Project-B summary for the prompt: top-5 TF-IDF rows.
                target_summary_rows = _top_tfidf_rows(
                    rows_by_project.get(project_b.project_id, []),
                    tfidf_by_project.get(project_b.project_id, {}),
                    5,
                )
                prompt = _render_collision_prompt(project_b, target_summary_rows, candidate.source_row)
                question = await _draft_and_count(prompt, reasoner=reasoner, gen_config=gen_config, state=abort_state)
                if question is None:
                    continue
                probe_id = _build_probe_id(KIND_COLLISION, project_b.project_id, candidate.source_row.id)
                accepted_, rank = await _verify_exclusion(
                    question=question,
                    user_id=gen_config.user_id,
                    excluded_row_id=candidate.source_row.id,
                    verify_top_k=gen_config.verify_top_k,
                )
                if accepted_:
                    accepted.append(
                        VerifiedProbe(
                            kind=KIND_COLLISION,
                            probe_id=probe_id,
                            question=question,
                            expected_fact_id=None,
                            expected_excluded_fact_ids=(candidate.source_row.id,),
                            workspace=_project_workspace_root_str(project_b),
                            legacy_rank=rank,
                        )
                    )
                    accepted_for_b += 1
                else:
                    dropped.append(
                        DroppedProbe(
                            kind=KIND_COLLISION,
                            probe_id=probe_id,
                            question=question,
                            excluded_row_id=candidate.source_row.id,
                            reason="rank_none" if rank is None else "rank_out_of_top_k",
                            observed_rank=rank,
                        )
                    )

        # Fallback pass: if Tier-1 did not fill B's quota, the
        # embedding-only fallback walks the remaining rows of every
        # source project. We loop through source projects again to
        # avoid biasing the fallback toward the project_id-first
        # source: each source contributes up to (quota_remaining)
        # candidates of its own, bounded by --fallback-cap.
        for project_a in sorted_projects:
            if project_a.project_id == project_b.project_id:
                continue
            if accepted_for_b >= gen_config.per_project_collisions:
                break
            quota_remaining = gen_config.per_project_collisions - accepted_for_b
            if quota_remaining <= 0:
                break
            fallback = _discover_fallback_candidates(
                source_project_id=project_a.project_id,
                source_rows=rows_by_project.get(project_a.project_id, []),
                source_embeddings=embeddings_by_project.get(project_a.project_id, []),
                target_project=project_b,
                target_centroid=centroids_by_project.get(project_b.project_id, []),
                fallback_cap=gen_config.fallback_cap,
                already_seen_row_ids=seen_row_ids,
            )
            for candidate in fallback:
                if accepted_for_b >= gen_config.per_project_collisions:
                    break
                seen_row_ids.add(candidate.source_row.id)
                target_summary_rows = _top_tfidf_rows(
                    rows_by_project.get(project_b.project_id, []),
                    tfidf_by_project.get(project_b.project_id, {}),
                    5,
                )
                prompt = _render_collision_prompt(project_b, target_summary_rows, candidate.source_row)
                question = await _draft_and_count(prompt, reasoner=reasoner, gen_config=gen_config, state=abort_state)
                if question is None:
                    continue
                probe_id = _build_probe_id(KIND_COLLISION, project_b.project_id, candidate.source_row.id)
                accepted_, rank = await _verify_exclusion(
                    question=question,
                    user_id=gen_config.user_id,
                    excluded_row_id=candidate.source_row.id,
                    verify_top_k=gen_config.verify_top_k,
                )
                if accepted_:
                    accepted.append(
                        VerifiedProbe(
                            kind=KIND_COLLISION,
                            probe_id=probe_id,
                            question=question,
                            expected_fact_id=None,
                            expected_excluded_fact_ids=(candidate.source_row.id,),
                            workspace=_project_workspace_root_str(project_b),
                            legacy_rank=rank,
                        )
                    )
                    accepted_for_b += 1
                    fallback_count_for_b += 1
                else:
                    dropped.append(
                        DroppedProbe(
                            kind=KIND_COLLISION,
                            probe_id=probe_id,
                            question=question,
                            excluded_row_id=candidate.source_row.id,
                            reason="rank_none" if rank is None else "rank_out_of_top_k",
                            observed_rank=rank,
                        )
                    )

        # Bookkeeping for the report's per-project summary table.
        report.pair_fallback_counts[project_b.project_id] = fallback_count_for_b

    return accepted, dropped


async def _draft_positive_only_probes(
    gen_config: GenerationConfig,
    reasoner: OneShotReasoner,
    sorted_projects: list[MemoryProjectConfig],
    rows_by_project: dict[str, list[Any]],
    tfidf_by_project: dict[str, dict[str, float]],
    abort_state: _AbortState,
) -> list[VerifiedProbe]:
    """
    Emit positive-only probes per project (D6 row 2).

    Pick the top-N TF-IDF rows per project; draft a direct question
    for each. No legacy verification: positive polarity is checked
    by the scoped harness at self-grade time.
    """
    accepted: list[VerifiedProbe] = []
    for project in sorted_projects:
        rows = rows_by_project.get(project.project_id, [])
        top_rows = _top_tfidf_rows(rows, tfidf_by_project.get(project.project_id, {}), gen_config.per_project_positive)
        for row in top_rows:
            prompt = _render_positive_only_prompt(row)
            question = await _draft_and_count(prompt, reasoner=reasoner, gen_config=gen_config, state=abort_state)
            if question is None:
                continue
            probe_id = _build_probe_id(KIND_POSITIVE, project.project_id, row.id)
            accepted.append(
                VerifiedProbe(
                    kind=KIND_POSITIVE,
                    probe_id=probe_id,
                    question=question,
                    expected_fact_id=row.id,
                    expected_excluded_fact_ids=(),
                    workspace=_project_workspace_root_str(project),
                    legacy_rank=None,
                )
            )
    return accepted


async def _draft_and_verify_non_project_probes(
    gen_config: GenerationConfig,
    reasoner: OneShotReasoner,
    sorted_projects: list[MemoryProjectConfig],
    rows_by_project: dict[str, list[Any]],
    tfidf_by_project: dict[str, dict[str, float]],
    abort_state: _AbortState,
) -> tuple[list[VerifiedProbe], list[DroppedProbe]]:
    """
    Emit non-project exclusion probes (D6 row 3) with legacy verification.

    Pool the top-TF-IDF project-bound rows from every project, draft
    a project-agnostic question for each, and accept only when the
    project-bound row appears in legacy top-K. Workspace=None,
    expected_excluded_fact_ids=(row.id,). Quota is corpus-wide
    (`--non-project`), not per project.
    """
    accepted: list[VerifiedProbe] = []
    dropped: list[DroppedProbe] = []

    # Pool the most distinctive row from each project, then iterate
    # round-robin until the quota is filled. This balances the source
    # pool across projects so one large project does not monopolize
    # the non-project corpus.
    pool: list[Any] = []
    for project in sorted_projects:
        rows = rows_by_project.get(project.project_id, [])
        top_rows = _top_tfidf_rows(rows, tfidf_by_project.get(project.project_id, {}), gen_config.non_project_quota)
        pool.extend(top_rows)
    # Sort the combined pool by row.id for determinism, then truncate
    # to a working set sized for the quota. We draft until we have
    # `non_project_quota` accepted or exhaust the pool.
    pool.sort(key=lambda r: r.id)

    for row in pool:
        if len(accepted) >= gen_config.non_project_quota:
            break
        prompt = _render_non_project_prompt(row)
        question = await _draft_and_count(prompt, reasoner=reasoner, gen_config=gen_config, state=abort_state)
        if question is None:
            continue
        probe_id = _build_probe_id(KIND_NON_PROJECT, row.id)
        accepted_, rank = await _verify_exclusion(
            question=question,
            user_id=gen_config.user_id,
            excluded_row_id=row.id,
            verify_top_k=gen_config.verify_top_k,
        )
        if accepted_:
            accepted.append(
                VerifiedProbe(
                    kind=KIND_NON_PROJECT,
                    probe_id=probe_id,
                    question=question,
                    expected_fact_id=None,
                    expected_excluded_fact_ids=(row.id,),
                    workspace=None,
                    legacy_rank=rank,
                )
            )
        else:
            dropped.append(
                DroppedProbe(
                    kind=KIND_NON_PROJECT,
                    probe_id=probe_id,
                    question=question,
                    excluded_row_id=row.id,
                    reason="rank_none" if rank is None else "rank_out_of_top_k",
                    observed_rank=rank,
                )
            )

    return accepted, dropped


# ── Legacy-default enumeration and allocation ──────────────────────


def _enumerate_legacy_default_rows(user_id: str) -> list[Any]:
    """
    Return resolver-legacy-default rows from the whole-store pool.

    Legacy-default rows are resolver-global by definition: they have
    no `scope` field in raw metadata and `resolve_memory_scope`
    returns `scope_source=SCOPE_SOURCE_LEGACY_DEFAULT`. The pool is
    therefore NOT per-project. Callers walk this pool
    round-robin across projects to emit per-project legacy-default
    probes.

    Sorted by row.id so the same store state yields the same pool
    order on every run.
    """
    from kai.memory import (
        SCOPE_SOURCE_LEGACY_DEFAULT,
        get_all_episodes,
        get_all_facts,
        resolve_memory_scope,
    )

    pool: list[Any] = []
    for row in get_all_facts(user_id=user_id):
        if resolve_memory_scope(row.metadata).scope_source == SCOPE_SOURCE_LEGACY_DEFAULT:
            pool.append(row)
    for row in get_all_episodes(user_id=user_id):
        if resolve_memory_scope(row.metadata).scope_source == SCOPE_SOURCE_LEGACY_DEFAULT:
            pool.append(row)
    pool.sort(key=lambda r: r.id)
    return pool


def _allocate_legacy_default_round_robin(
    pool: list[Any],
    sorted_projects: list[MemoryProjectConfig],
    per_project: int,
) -> tuple[dict[str, list[Any]], bool]:
    """
    Walk the resolver-global pool round-robin across projects.

    Returns:
        - allocation: project_id -> ordered list of rows pinned to
          that project's workspace. Each project receives at most
          `per_project` rows.
        - repeated: True when the pool is smaller than
          `len(projects) * per_project` and rows had to be repeated
          across projects.

    Allocation algorithm: walk projects in order, walk the pool in
    order, hand out the next row to the next project. When the pool
    runs out before every project has its quota, wrap around to the
    start of the pool (so row repetition happens). This guarantees
    no project is starved while a pool row is available; the
    repetition flag tells the report whether the corpus is
    artificially fattened.
    """
    if per_project <= 0 or not sorted_projects:
        return {}, False
    allocation: dict[str, list[Any]] = {p.project_id: [] for p in sorted_projects}
    total_slots = len(sorted_projects) * per_project
    if not pool:
        return allocation, False
    repeated = total_slots > len(pool)
    pool_idx = 0
    for project in sorted_projects:
        for _ in range(per_project):
            row = pool[pool_idx % len(pool)]
            allocation[project.project_id].append(row)
            pool_idx += 1
    return allocation, repeated


async def _draft_legacy_default_probes(
    gen_config: GenerationConfig,
    reasoner: OneShotReasoner,
    sorted_projects: list[MemoryProjectConfig],
    allocation: dict[str, list[Any]],
    abort_state: _AbortState,
) -> list[VerifiedProbe]:
    """
    Emit legacy-default probes.

    Each probe pins workspace to a project's workspace_root and sets
    expected_fact_id to the allocated row's id. Uses the positive-
    only prompt template (same shape: "write a question whose answer
    is this fact"). No legacy verification; positive polarity is
    checked by the scoped harness at self-grade.

    The probe_id format `legacy_default:<project_id>:<row_id>` carries
    both the workspace dimension and the source row dimension so a
    later `--reject` invocation can drop a single probe without
    affecting other projects that share the same row.
    """
    accepted: list[VerifiedProbe] = []
    for project in sorted_projects:
        for row in allocation.get(project.project_id, []):
            prompt = _render_positive_only_prompt(row)
            question = await _draft_and_count(prompt, reasoner=reasoner, gen_config=gen_config, state=abort_state)
            if question is None:
                continue
            probe_id = _build_probe_id(KIND_LEGACY_DEFAULT, project.project_id, row.id)
            accepted.append(
                VerifiedProbe(
                    kind=KIND_LEGACY_DEFAULT,
                    probe_id=probe_id,
                    question=question,
                    expected_fact_id=row.id,
                    expected_excluded_fact_ids=(),
                    workspace=_project_workspace_root_str(project),
                    legacy_rank=None,
                )
            )
    return accepted


# ── JSONL writer and probe_id mapping ──────────────────────────────


def _sort_probes_for_output(probes: list[VerifiedProbe]) -> list[VerifiedProbe]:
    """
    Sort by (kind, workspace, probe_id) for deterministic file
    ordering across runs against the same store state.

    Workspace=None sorts last among kinds that share a sort bucket
    because Python tuples compare None as smaller-than-any-string by
    default and we want the workspace-null probes to land last among
    non_project probes (the only kind that uses None). The tuple key
    treats None as "" to push workspace=None probes to the front
    *within their kind*, which keeps the file readable when grepping.
    """
    return sorted(
        probes,
        key=lambda p: (p.kind, p.workspace or "", p.probe_id),
    )


def _disambiguate_probe_ids(probes: list[VerifiedProbe]) -> list[VerifiedProbe]:
    """
    Append `:dup<n>` to duplicate probe_ids so every emitted row
    has a unique identifier.

    Same-row collisions (a row colliding with the same partner twice
    across reruns) or legacy-default row repetition across projects
    can produce identical probe_ids. The first occurrence keeps its
    canonical id; subsequent occurrences get `:dup1`, `:dup2`, etc.
    """
    seen: dict[str, int] = {}
    out: list[VerifiedProbe] = []
    for probe in probes:
        if probe.probe_id not in seen:
            seen[probe.probe_id] = 0
            out.append(probe)
            continue
        seen[probe.probe_id] += 1
        new_id = f"{probe.probe_id}:dup{seen[probe.probe_id]}"
        # Build a new dataclass instance via dataclasses.replace
        # rather than mutating the frozen original.
        from dataclasses import replace

        out.append(replace(probe, probe_id=new_id))
    return out


def _serialize_probe_to_jsonl_line(probe: VerifiedProbe) -> str:
    """
    Render one probe as a single JSONL line carrying the probe_id extra.

    `load_probes` tolerates extra JSON keys, so `probe_id` lives on
    the JSONL row alongside the ScopedProbe-shaped fields without a
    schema change. Field order matches the dataclass field order in
    `ScopedProbe` so a diff between the dry-run and post-reject
    files is human-readable.
    """
    row = {
        "probe_id": probe.probe_id,
        "question": probe.question,
        "expected_fact_id": probe.expected_fact_id,
        "expected_excluded_fact_ids": list(probe.expected_excluded_fact_ids),
        "workspace": probe.workspace,
    }
    return json.dumps(row, ensure_ascii=False)


def _write_dryrun_jsonl(
    probes: list[VerifiedProbe],
    path: Path,
) -> None:
    """
    Write the dry-run JSONL file in deterministic order.

    Overwrites the path (dry-run output is always overwritten; the
    operator iterates on it). Each probe is one line; no trailing
    newline beyond the final line break per JSONL convention.
    """
    sorted_probes = _sort_probes_for_output(probes)
    disambiguated = _disambiguate_probe_ids(sorted_probes)
    with open(path, "w", encoding="utf-8") as fh:
        for probe in disambiguated:
            fh.write(_serialize_probe_to_jsonl_line(probe))
            fh.write("\n")


def _build_probe_id_by_line_number(path: Path) -> dict[int, str]:
    """
    Map 1-indexed source lines to their probe_id.

    `ScopedProbe.line_number` is populated by `load_probes` against
    the same file we just wrote, so this mapping survives the
    write/load round-trip. Comment lines (`#`-prefixed) and blank
    lines are skipped on both sides; the line counter advances on
    every raw line but the mapping only stores entries for parsed
    rows, mirroring the count `load_probes` increments.

    Without this side-channel mapping, the report could not name
    leaking probes by probe_id after self-grade because ScopedProbe
    and ScopedProbeResult do not carry the probe_id field. See spec
    D7-map for the full reasoning.
    """
    mapping: dict[int, str] = {}
    with open(path, encoding="utf-8") as fh:
        line_number = 0
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            line_number += 1
            row = json.loads(stripped)
            mapping[line_number] = row.get("probe_id", "")
    return mapping


# ── Self-grade ─────────────────────────────────────────────────────


@dataclass
class _SelfGradeResult:
    """
    Output of the self-grade pass.

    Bundles the verdict string, the aggregated metrics, and the
    per-probe results so the report can name leaking probes by
    `probe_id_by_line_number[r.probe.line_number]`. Verdict is one
    of `ship`, `regenerate`, `INVESTIGATE`.
    """

    verdict: str
    n_scored_negative: int
    exclusion_pass_in_prompt: float
    exclusion_pass_in_candidates: float
    results: list[Any]  # list[ScopedProbeResult]


async def _self_grade(
    probes_path: Path,
    user_id: str,
) -> _SelfGradeResult:
    """
    Run the scoped evaluator on the dry-run file and classify the verdict.

    `evaluate` returns `tuple[list[ScopedProbeResult], ScopedMetrics]`
    we destructure the (results, metrics) tuple because the
    INVESTIGATE branch needs per-probe results to name leaking
    probes by probe_id (via the line-number side-channel mapping).
    """
    probes = load_probes(probes_path)
    results, metrics = await evaluate(probes, user_id=user_id)

    # Three-outcome classification. Order matters: a
    # corpus that emits zero exclusion-polarity probes hits the
    # regenerate branch BEFORE the ship check, because
    # n_scored_negative == 0 means the safety metrics default to
    # 1.0 vacuously rather than actually passing.
    if metrics.n_scored_negative == 0:
        verdict = "regenerate"
    elif metrics.exclusion_pass_in_prompt < 1.0 or metrics.exclusion_pass_in_candidates < 1.0:
        verdict = "INVESTIGATE"
    else:
        verdict = "ship"

    return _SelfGradeResult(
        verdict=verdict,
        n_scored_negative=metrics.n_scored_negative,
        exclusion_pass_in_prompt=metrics.exclusion_pass_in_prompt,
        exclusion_pass_in_candidates=metrics.exclusion_pass_in_candidates,
        results=list(results),
    )


def _extract_leak_records(
    self_grade: _SelfGradeResult,
    probe_id_by_line_number: dict[int, str],
) -> list[dict[str, Any]]:
    """
    Build per-leak detail rows for the report.

    Selects results with non-empty `excluded_in_prompt` or
    `excluded_in_candidates` and names each leak by the probe_id
    looked up via line_number. The "what surfaced instead" field
    reports the excluded ids the scoped pipeline failed to filter
    out.
    """
    leaks: list[dict[str, Any]] = []
    for r in self_grade.results:
        if not r.excluded_in_prompt and not r.excluded_in_candidates:
            continue
        leaks.append(
            {
                "probe_id": probe_id_by_line_number.get(r.probe.line_number, ""),
                "question": r.probe.question,
                "expected_excluded_fact_ids": list(r.probe.expected_excluded_fact_ids),
                "excluded_in_prompt": list(r.excluded_in_prompt),
                "excluded_in_candidates": list(r.excluded_in_candidates),
            }
        )
    return leaks


# ── Promote gates ───────────────────────────────────────────────────


@dataclass
class _GateEvaluation:
    """
    Five-gate decision result from `_evaluate_promote_gates`.

    Each gate is one of: "pass", "block", or "relaxed". The relaxed
    state applies only to structural and legacy-default coverage
    gates and only when `--allow-shortfalls` is passed.
    `block_reason` is populated when any gate's status is "block";
    it names the gate and the shortfall so the report and stderr
    can render the same diagnostic without recomputing.
    """

    self_grade: str  # pass | block
    structural_coverage: str  # pass | block | relaxed
    legacy_default_coverage: str  # pass | block | relaxed
    parent_directory: str  # pass | block
    exclusive_create: str  # pass | block | not_attempted
    block_reason: str | None
    structural_shortfalls: dict[str, dict[str, int]]  # {project_id: {kind: shortfall_count}}
    legacy_default_shortfalls: dict[str, int]  # project_id: shortfall_count


def _compute_structural_shortfalls(
    accepted: list[VerifiedProbe],
    sorted_projects: list[MemoryProjectConfig],
    gen_config: GenerationConfig,
) -> dict[str, dict[str, int]]:
    """
    Compute per-project shortfalls for collision and positive-only kinds.

    Non-project is also returned but keyed under the sentinel
    "__corpus__" because the non-project quota is corpus-wide. The
    structure matches `probe_corpus_check`'s reporting shape so an
    operator running both checkers sees the same gap structure.

    A negative shortfall means the project overshot the minimum;
    those are clamped to 0 because overshoots are not failures.
    """
    by_kind_by_project: dict[str, dict[str, int]] = {p.project_id: {} for p in sorted_projects}
    by_kind_by_project.setdefault("__corpus__", {})

    # Per-project collision and positive-only counts.
    for project in sorted_projects:
        collision_count = sum(
            1 for p in accepted if p.kind == KIND_COLLISION and p.workspace == _project_workspace_root_str(project)
        )
        positive_count = sum(
            1 for p in accepted if p.kind == KIND_POSITIVE and p.workspace == _project_workspace_root_str(project)
        )
        by_kind_by_project[project.project_id][KIND_COLLISION] = max(
            0, gen_config.per_project_collisions - collision_count
        )
        by_kind_by_project[project.project_id][KIND_POSITIVE] = max(0, gen_config.per_project_positive - positive_count)

    # Corpus-wide non-project count.
    non_project_count = sum(1 for p in accepted if p.kind == KIND_NON_PROJECT)
    by_kind_by_project["__corpus__"][KIND_NON_PROJECT] = max(0, gen_config.non_project_quota - non_project_count)

    return by_kind_by_project


def _compute_legacy_default_shortfalls(
    accepted: list[VerifiedProbe],
    sorted_projects: list[MemoryProjectConfig],
    gen_config: GenerationConfig,
) -> dict[str, int]:
    """
    Per-project shortfalls counting EMITTED legacy-default probes.

    The gate counts probes pinned to each project's workspace, NOT
    source-pool rows. Legacy-default rows are resolver-global; the
    per-project gate is about emitted probes, not source rows.
    """
    shortfalls: dict[str, int] = {}
    for project in sorted_projects:
        ws = _project_workspace_root_str(project)
        count = sum(1 for p in accepted if p.kind == KIND_LEGACY_DEFAULT and p.workspace == ws)
        shortfalls[project.project_id] = max(0, gen_config.per_project_legacy - count)
    return shortfalls


def _evaluate_promote_gates(
    self_grade: _SelfGradeResult,
    accepted: list[VerifiedProbe],
    sorted_projects: list[MemoryProjectConfig],
    gen_config: GenerationConfig,
) -> _GateEvaluation:
    """
    Compute the five-gate evaluation for a promote attempt.

    Self-grade gate: ship verdict required. Block on regenerate or
    INVESTIGATE.

    Structural coverage gate: every per-project collision and
    positive-only count >= minimums, non-project total >= minimum.
    Block on any shortfall unless --allow-shortfalls is passed.

    Legacy-default coverage gate: every per-project EMITTED legacy-
    default probe count >= minimum. Block on any shortfall unless
    --allow-shortfalls is passed.

    Parent directory and exclusive create gates are tested at write
    time by the caller; this helper precomputes the first three
    gates and stamps the latter two as "not_attempted" placeholders.
    """
    # Gate 1: self-grade.
    if self_grade.verdict == "ship":
        sg = "pass"
        sg_block_reason: str | None = None
    else:
        sg = "block"
        sg_block_reason = f"self-grade verdict is {self_grade.verdict!r}, expected ship"

    # Gate 2: structural coverage.
    structural_shortfalls = _compute_structural_shortfalls(accepted, sorted_projects, gen_config)
    structural_has_shortfall = any(
        v > 0 for project_buckets in structural_shortfalls.values() for v in project_buckets.values()
    )
    if not structural_has_shortfall:
        struct = "pass"
        struct_block_reason: str | None = None
    elif gen_config.allow_shortfalls:
        struct = "relaxed"
        struct_block_reason = None
    else:
        struct = "block"
        struct_block_reason = "structural coverage shortfall (see Shortfalls section)"

    # Gate 3: legacy-default coverage.
    legacy_shortfalls = _compute_legacy_default_shortfalls(accepted, sorted_projects, gen_config)
    legacy_has_shortfall = any(v > 0 for v in legacy_shortfalls.values())
    if not legacy_has_shortfall:
        legacy = "pass"
        legacy_block_reason: str | None = None
    elif gen_config.allow_shortfalls:
        legacy = "relaxed"
        legacy_block_reason = None
    else:
        legacy = "block"
        legacy_block_reason = "legacy-default coverage shortfall (see Shortfalls section)"

    # Pick the first blocking gate (in evaluation order) as the
    # primary block_reason; the operator sees the most critical one
    # named on stderr and the full picture in the report.
    block_reason = sg_block_reason or struct_block_reason or legacy_block_reason

    return _GateEvaluation(
        self_grade=sg,
        structural_coverage=struct,
        legacy_default_coverage=legacy,
        parent_directory="not_attempted",
        exclusive_create="not_attempted",
        block_reason=block_reason,
        structural_shortfalls=structural_shortfalls,
        legacy_default_shortfalls=legacy_shortfalls,
    )


def _execute_promote_write(
    dryrun_path: Path,
    canonical_path: Path,
    gen_config: GenerationConfig,
    gate_eval: _GateEvaluation,
    report: GeneratorReport,
) -> int:
    """
    Run the parent-directory and exclusive-create gates and write.

    Returns the exit code for the run:
        0 if promotion succeeded (or was a no-op).
        2 if the exclusive-create gate blocked AND --force was not
          passed.
    The gate evaluation passed to this helper must already have all
    of (self_grade, structural_coverage, legacy_default_coverage)
    cleared to either "pass" or "relaxed"; the caller is responsible
    for blocking on those gates before calling here.

    Parent directory gate: `mkdir(parents=True, exist_ok=True)`
    creates `<data-dir>/eval/probes/` on first promote. exist_ok
    handles the second-and-later promote case where the directory
    already lives on disk.

    Exclusive create gate: `open(path, "xb")` is the POSIX
    no-clobber primitive (O_CREAT | O_EXCL). It removes the TOCTOU
    window a pre-check plus os.rename would have under a concurrent
    creator. `--force` opts in to atomic replacement via os.replace.
    """
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    gate_eval.parent_directory = "pass"

    dryrun_bytes = dryrun_path.read_bytes()

    try:
        with open(canonical_path, "xb") as fh:
            fh.write(dryrun_bytes)
    except FileExistsError:
        if not gen_config.force:
            print(
                f"gen_collision_probes: canonical path exists; refusing to overwrite: {canonical_path}",
                file=sys.stderr,
            )
            gate_eval.exclusive_create = "block"
            gate_eval.block_reason = f"canonical path exists at {canonical_path}; pass --force to overwrite"
            return 2
        # --force path: write to a sibling temp file, then atomically
        # rename over the target. os.replace is the POSIX-atomic
        # replacement primitive: a reader observes either the old or
        # the new file, never a half-written one.
        tmp_path = canonical_path.with_suffix(canonical_path.suffix + ".tmp")
        tmp_path.write_bytes(dryrun_bytes)
        os.replace(tmp_path, canonical_path)
        report.forced_overwrite = canonical_path

    gate_eval.exclusive_create = "pass"
    return 0


# ── --reject path ──────────────────────────────────────────────────


def _apply_rejects_to_jsonl(
    path: Path,
    reject_ids: set[str],
) -> tuple[list[str], list[str]]:
    """
    Drop rows whose probe_id is in `reject_ids`; rewrite path in place.

    Returns (found, missing): probe_ids that were dropped and
    probe_ids in `reject_ids` that did not match any row. The caller
    uses both lists in the report so an operator pasting yesterday's
    ids can see which ones still existed.

    The rewrite path reads the JSONL once, writes the filtered rows
    to a temp file, then renames atomically. A crash mid-rewrite
    leaves the original file intact rather than half-truncating.
    """
    found: list[str] = []
    raw_rows: list[tuple[str, dict[str, Any]]] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            row = json.loads(stripped)
            raw_rows.append((stripped, row))

    kept_rows: list[str] = []
    seen_ids: set[str] = set()
    for raw_line, row in raw_rows:
        pid = row.get("probe_id", "")
        if pid in reject_ids:
            found.append(pid)
            seen_ids.add(pid)
            continue
        kept_rows.append(raw_line)

    missing = [pid for pid in reject_ids if pid not in seen_ids]

    # Atomic rewrite: write to a sibling temp, then os.replace.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        for line in kept_rows:
            fh.write(line)
            fh.write("\n")
    os.replace(tmp_path, path)

    return found, missing


# ── Report renderer ────────────────────────────────────────────────


def _render_report(
    report: GeneratorReport,
    gen_config: GenerationConfig,
    sorted_projects: list[MemoryProjectConfig],
    gate_eval: _GateEvaluation,
) -> str:
    """
    Render the human-readable Markdown report.

    Pulls every line from `report` and `gate_eval`. No live-store
    reads at render time; the report is deterministic given the
    populated state. Sections appear in the order specified by
    D11 so an operator skimming the report sees verdict +
    promote-outcome at the top and the long detail sections below.
    """
    lines: list[str] = []
    lines.append("# Collision probe corpus report\n")
    lines.append(f"User: {report.user_id}")
    lines.append(f"Generated: {report.generated_at}")
    lines.append(f"Self-grade verdict: {report.self_grade_verdict}")
    lines.append(f"Promote outcome: {report.promote_outcome}")
    lines.append(
        f"Backend: {report.effective_backend} / {report.effective_provider} / {report.effective_os_user or '<none>'}"
    )
    lines.append(f"Model: {report.model}")
    lines.append(f"Drafting timeout: {report.timeout_s}s\n")

    # Summary table.
    lines.append("## Summary\n")
    lines.append(
        "| Project | Rows considered | Candidates drafted | Verified | Accepted (as target B) | Pairs hit fallback |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for project in sorted_projects:
        s = report.per_project_summary.get(project.project_id, {})
        lines.append(
            f"| {project.project_id} | {s.get('rows_considered', 0)} | "
            f"{s.get('candidates_drafted', 0)} | {s.get('verified', 0)} | "
            f"{s.get('accepted_as_target', 0)} | "
            f"{report.pair_fallback_counts.get(project.project_id, 0)} |"
        )
    lines.append("")

    # Distribution.
    accepted = report.accepted_probes
    collision_n = sum(1 for p in accepted if p.kind == KIND_COLLISION)
    positive_n = sum(1 for p in accepted if p.kind == KIND_POSITIVE)
    non_project_n = sum(1 for p in accepted if p.kind == KIND_NON_PROJECT)
    non_project_dropped_n = sum(1 for d in report.dropped_probes if d.kind == KIND_NON_PROJECT)
    legacy_default_n = sum(1 for p in accepted if p.kind == KIND_LEGACY_DEFAULT)
    lines.append("## Distribution\n")
    lines.append(f"- Collision probes (per target project B): {collision_n}")
    lines.append(f"- Positive-only probes: {positive_n}")
    lines.append(
        f"- Non-project exclusion probes: {non_project_n} (verified {non_project_n}, dropped {non_project_dropped_n})"
    )
    lines.append(
        f"- Legacy-default probes: {legacy_default_n} "
        f"(drawn from a global pool of {report.legacy_default_pool_size} legacy-default rows)\n"
    )

    # Self-grade detail.
    lines.append("## Self-grade\n")
    lines.append(f"- n_scored_negative: {report.n_scored_negative}")
    lines.append(f"- exclusion_pass_in_prompt: {report.exclusion_pass_in_prompt}")
    lines.append(f"- exclusion_pass_in_candidates: {report.exclusion_pass_in_candidates}")
    lines.append(f"- verdict: {report.self_grade_verdict}\n")

    # Promote gates.
    lines.append("## Promote gates\n")
    lines.append(f"- Self-grade gate: {gate_eval.self_grade}")
    lines.append(f"- Structural coverage gate: {gate_eval.structural_coverage}")
    lines.append(f"- Legacy-default coverage gate: {gate_eval.legacy_default_coverage}")
    lines.append(f"- Parent directory gate: {gate_eval.parent_directory}")
    lines.append(f"- Exclusive create gate: {gate_eval.exclusive_create}")
    if gate_eval.block_reason:
        lines.append(f"- Block reason: {gate_eval.block_reason}")
    lines.append("")

    # Sample probes: pick 5 deterministically by hashing the corpus.
    if accepted:
        import hashlib

        corpus_text = "\n".join(p.probe_id for p in accepted)
        seed = int(hashlib.sha256(corpus_text.encode("utf-8")).hexdigest()[:8], 16)
        # Walk the sorted accepted list at stride `len(accepted) // 5`
        # so the sample spans the whole corpus rather than clustering.
        sample_size = min(5, len(accepted))
        stride = max(1, len(accepted) // sample_size)
        sample_indices = [(seed + i * stride) % len(accepted) for i in range(sample_size)]
        # Dedupe while preserving order.
        seen_idx: set[int] = set()
        unique_indices = []
        for idx in sample_indices:
            if idx in seen_idx:
                continue
            seen_idx.add(idx)
            unique_indices.append(idx)
        lines.append("## Sample probes\n")
        for idx in unique_indices:
            probe = accepted[idx]
            lines.append(f"- `{probe.probe_id}` ({probe.kind}): {probe.question}")
        lines.append("")

    # Non-project exclusion verification.
    non_project_accepted = [p for p in accepted if p.kind == KIND_NON_PROJECT]
    non_project_dropped = [d for d in report.dropped_probes if d.kind == KIND_NON_PROJECT]
    if non_project_accepted or non_project_dropped:
        lines.append("## Non-project exclusion verification\n")
        for probe in non_project_accepted:
            excluded_id = probe.expected_excluded_fact_ids[0] if probe.expected_excluded_fact_ids else "(none)"
            lines.append(
                f"- accepted `{probe.probe_id}`: excluded row `{excluded_id}` at legacy rank "
                f"{probe.legacy_rank}; question: {probe.question}"
            )
        for d in non_project_dropped:
            rank_repr = "None" if d.observed_rank is None else str(d.observed_rank)
            lines.append(
                f"- dropped `{d.probe_id}` ({d.reason}, observed rank {rank_repr}): "
                f"excluded row `{d.excluded_row_id}`; question: {d.question}"
            )
        lines.append("")

    # Structural coverage.
    lines.append("## Structural coverage\n")
    for project in sorted_projects:
        s = gate_eval.structural_shortfalls.get(project.project_id, {})
        collision_short = s.get(KIND_COLLISION, 0)
        positive_short = s.get(KIND_POSITIVE, 0)
        lines.append(
            f"- {project.project_id}: "
            f"collision shortfall {collision_short} (min {gen_config.per_project_collisions}), "
            f"positive-only shortfall {positive_short} (min {gen_config.per_project_positive})"
        )
    non_project_short = gate_eval.structural_shortfalls.get("__corpus__", {}).get(KIND_NON_PROJECT, 0)
    lines.append(f"- corpus-wide: non-project shortfall {non_project_short} (min {gen_config.non_project_quota})\n")

    # Legacy-default coverage.
    lines.append("## Legacy-default coverage\n")
    lines.append(f"- Global pool size: {report.legacy_default_pool_size}")
    for project in sorted_projects:
        emitted_count = sum(
            1 for p in accepted if p.kind == KIND_LEGACY_DEFAULT and p.workspace == _project_workspace_root_str(project)
        )
        shortfall = gate_eval.legacy_default_shortfalls.get(project.project_id, 0)
        lines.append(
            f"- {project.project_id}: emitted {emitted_count} legacy-default probes "
            f"pinned to its workspace (min {gen_config.per_project_legacy}, shortfall {shortfall})"
        )
    if report.legacy_default_row_repeated:
        lines.append(
            "- Row repetition: the resolver-global pool was smaller than "
            "len(projects) * --per-project-legacy; some rows are pinned to multiple project workspaces."
        )
    for project in sorted_projects:
        for probe in accepted:
            if probe.kind != KIND_LEGACY_DEFAULT:
                continue
            if probe.workspace != _project_workspace_root_str(project):
                continue
            lines.append(
                f"- `{probe.probe_id}`: pinned to {project.project_id}, "
                f"expected_fact_id {probe.expected_fact_id}, "
                f"observed scope_source legacy_default (sanity)"
            )
    lines.append("")

    # Shortfalls (compiled from the gate shortfalls).
    has_any_shortfall = any(
        v > 0 for buckets in gate_eval.structural_shortfalls.values() for v in buckets.values()
    ) or any(v > 0 for v in gate_eval.legacy_default_shortfalls.values())
    if has_any_shortfall:
        lines.append("## Shortfalls\n")
        for project in sorted_projects:
            s = gate_eval.structural_shortfalls.get(project.project_id, {})
            for kind, count in s.items():
                if count > 0:
                    lines.append(f"- {project.project_id} / {kind}: shortfall {count}")
            ld_short = gate_eval.legacy_default_shortfalls.get(project.project_id, 0)
            if ld_short > 0:
                lines.append(f"- {project.project_id} / legacy_default: shortfall {ld_short}")
        corpus_buckets = gate_eval.structural_shortfalls.get("__corpus__", {})
        for kind, count in corpus_buckets.items():
            if count > 0:
                lines.append(f"- corpus-wide / {kind}: shortfall {count}")
        lines.append("")

    # Leaks (when INVESTIGATE).
    if report.leak_records:
        lines.append("## Leaks (INVESTIGATE)\n")
        for leak in report.leak_records:
            lines.append(
                f"- `{leak['probe_id']}`: question {leak['question']!r}, "
                f"expected_excluded {leak['expected_excluded_fact_ids']}, "
                f"surfaced in prompt {leak['excluded_in_prompt']}, "
                f"surfaced in candidates {leak['excluded_in_candidates']}"
            )
        lines.append("")

    # Rejected.
    if report.rejected_found or report.rejected_missing:
        lines.append("## Rejected\n")
        for pid in report.rejected_found:
            lines.append(f"- `{pid}`: found and dropped from dry-run JSONL")
        for pid in report.rejected_missing:
            lines.append(f"- `{pid}`: not found in dry-run JSONL")
        lines.append("")

    # Promoted with shortfalls.
    if report.allow_shortfalls_applied:
        lines.append("## Promoted with shortfalls (--allow-shortfalls)\n")
        for gate_name, detail in report.allow_shortfalls_applied.items():
            lines.append(f"- {gate_name}: {detail}")
        lines.append("")

    # Forced overwrite.
    if report.forced_overwrite is not None:
        lines.append("## Forced overwrite (--force)\n")
        lines.append(f"- Replaced file: {report.forced_overwrite}\n")

    return "\n".join(lines).rstrip() + "\n"


# ── Orchestrator entry ─────────────────────────────────────────────


async def _run_generate(
    gen_config: GenerationConfig,
    config: Config,
) -> int:
    """
    End-to-end orchestration: discover, draft+verify, write, self-grade, promote.

    Returns a process exit code: 0 on a completed run (dry-run OR
    successful promote), 1 on the consecutive-failure abort, 2 on a
    blocked promote gate.

    Heavily wrapped logging is intentional: a generator run can take
    several minutes on a real store; the operator needs progress
    signals on stderr at each phase boundary.
    """
    from kai.memory_projects import load_project_registry

    # Bootstrap the merged registry's DB layer so scoped retrieval (run
    # at self-grade time) can detect chat-registered projects when an
    # operator's workspace lives under one. The return value is ignored
    # because project enumeration in this run uses config.memory_projects
    # only (YAML); the registry call here exists for its
    # side effect on the scoped helper's cache.
    await load_project_registry(config)

    # Resolve project subset: --projects filter or full YAML registry.
    if gen_config.project_filter:
        allowed = set(gen_config.project_filter)
        projects_dict = {pid: p for pid, p in config.memory_projects.items() if pid in allowed}
        if not projects_dict:
            print(
                f"gen_collision_probes: --projects filter matched no entries in "
                f"config.memory_projects; available ids: "
                f"{sorted(config.memory_projects.keys())}",
                file=sys.stderr,
            )
            return 2
    else:
        projects_dict = dict(config.memory_projects)
        # DB-registered projects are out of scope: probe_corpus_check
        # uses config.memory_projects (YAML) only, and a corpus that
        # includes DB-only projects would fail the structural
        # coverage check. The merged registry is loaded above for
        # the scoped helper's cache but only the YAML half is
        # enumerated here.

    sorted_projects = sorted(projects_dict.values(), key=lambda p: p.project_id)
    print(
        f"gen_collision_probes: enumerating {len(sorted_projects)} project(s) "
        f"({', '.join(p.project_id for p in sorted_projects)})",
        file=sys.stderr,
    )

    rows_by_project = {p.project_id: _enumerate_project_rows(p, gen_config.user_id) for p in sorted_projects}
    for project in sorted_projects:
        print(
            f"gen_collision_probes:   {project.project_id}: {len(rows_by_project[project.project_id])} rows",
            file=sys.stderr,
        )

    # Pre-compute embeddings, token unions, centroids, TF-IDF.
    embeddings_by_project = {pid: _embed_rows_text(rows) for pid, rows in rows_by_project.items()}
    token_unions_by_project = {
        p.project_id: _build_project_token_union(rows_by_project[p.project_id], p) for p in sorted_projects
    }
    centroids_by_project = {pid: _compute_centroid(vecs) for pid, vecs in embeddings_by_project.items()}
    tfidf_by_project = {pid: _project_tfidf_scores(rows_by_project[pid], rows_by_project) for pid in rows_by_project}

    # Report skeleton.
    report = GeneratorReport(
        user_id=gen_config.user_id,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        effective_backend=gen_config.effective_backend,
        effective_provider=gen_config.effective_provider,
        effective_os_user=gen_config.effective_os_user,
        model=gen_config.model,
        timeout_s=gen_config.timeout_s,
    )

    reasoner = _build_drafting_reasoner(
        gen_config.effective_backend,
        os_user=gen_config.effective_os_user,
        provider=gen_config.effective_provider,
    )
    abort_state = _AbortState()

    try:
        print("gen_collision_probes: drafting collisions...", file=sys.stderr)
        collisions, dropped_collisions = await _draft_and_verify_collisions(
            gen_config,
            reasoner,
            sorted_projects,
            rows_by_project,
            embeddings_by_project,
            token_unions_by_project,
            centroids_by_project,
            tfidf_by_project,
            abort_state,
            report,
        )

        print("gen_collision_probes: drafting positive-only...", file=sys.stderr)
        positive_only = await _draft_positive_only_probes(
            gen_config, reasoner, sorted_projects, rows_by_project, tfidf_by_project, abort_state
        )

        print("gen_collision_probes: drafting non-project exclusions...", file=sys.stderr)
        non_project, dropped_non_project = await _draft_and_verify_non_project_probes(
            gen_config, reasoner, sorted_projects, rows_by_project, tfidf_by_project, abort_state
        )

        print("gen_collision_probes: enumerating legacy-default pool...", file=sys.stderr)
        legacy_pool = _enumerate_legacy_default_rows(gen_config.user_id)
        report.legacy_default_pool_size = len(legacy_pool)
        legacy_allocation, repeated = _allocate_legacy_default_round_robin(
            legacy_pool, sorted_projects, gen_config.per_project_legacy
        )
        report.legacy_default_row_repeated = repeated
        report.legacy_default_allocation = {pid: [r.id for r in rows] for pid, rows in legacy_allocation.items()}
        print(
            f"gen_collision_probes:   {len(legacy_pool)} legacy-default rows in global pool",
            file=sys.stderr,
        )

        print("gen_collision_probes: drafting legacy-default probes...", file=sys.stderr)
        legacy_default = await _draft_legacy_default_probes(
            gen_config, reasoner, sorted_projects, legacy_allocation, abort_state
        )
    except _AbortException as exc:
        print(str(exc), file=sys.stderr)
        return 1

    accepted = collisions + positive_only + non_project + legacy_default
    dropped = dropped_collisions + dropped_non_project
    report.accepted_probes = accepted
    report.dropped_probes = dropped

    # Per-project summary populated for the report's first table.
    for project in sorted_projects:
        accepted_as_target = sum(
            1 for p in accepted if p.kind == KIND_COLLISION and p.workspace == _project_workspace_root_str(project)
        )
        # "Verified" includes positive/legacy-default (no unscoped
        # verify by construction) plus exclusion probes that passed
        # the gate for this project as the workspace target.
        report.per_project_summary[project.project_id] = {
            "rows_considered": len(rows_by_project.get(project.project_id, [])),
            "candidates_drafted": accepted_as_target
            + sum(
                1
                for d in dropped
                if d.kind == KIND_COLLISION
                and any(rid == d.excluded_row_id for rid in (r.id for r in rows_by_project.get(project.project_id, [])))
            ),
            "verified": accepted_as_target,
            "accepted_as_target": accepted_as_target,
        }

    # Write the dry-run JSONL and build the probe_id mapping.
    _write_dryrun_jsonl(accepted, _DRYRUN_PROBES_PATH)
    probe_id_by_line_number = _build_probe_id_by_line_number(_DRYRUN_PROBES_PATH)

    # Self-grade.
    print("gen_collision_probes: self-grading via retrieval_scoped.evaluate...", file=sys.stderr)
    self_grade = await _self_grade(_DRYRUN_PROBES_PATH, gen_config.user_id)
    report.self_grade_verdict = self_grade.verdict
    report.n_scored_negative = self_grade.n_scored_negative
    report.exclusion_pass_in_prompt = self_grade.exclusion_pass_in_prompt
    report.exclusion_pass_in_candidates = self_grade.exclusion_pass_in_candidates
    if self_grade.verdict == "INVESTIGATE":
        report.leak_records = _extract_leak_records(self_grade, probe_id_by_line_number)

    # Promote gates.
    gate_eval = _evaluate_promote_gates(self_grade, accepted, sorted_projects, gen_config)

    exit_code = 0
    if gen_config.promote:
        # Block on self-grade or any unrelaxed coverage gate.
        if (
            gate_eval.self_grade == "block"
            or gate_eval.structural_coverage == "block"
            or gate_eval.legacy_default_coverage == "block"
        ):
            report.promote_outcome = f"blocked: {gate_eval.block_reason}"
            print(f"gen_collision_probes: promote blocked: {gate_eval.block_reason}", file=sys.stderr)
            exit_code = 2
        else:
            # Record any relaxed gates before attempting the write.
            if gate_eval.structural_coverage == "relaxed":
                report.allow_shortfalls_applied["structural"] = "relaxed via --allow-shortfalls"
            if gate_eval.legacy_default_coverage == "relaxed":
                report.allow_shortfalls_applied["legacy_default"] = "relaxed via --allow-shortfalls"
            write_code = _execute_promote_write(
                _DRYRUN_PROBES_PATH, gen_config.output_path, gen_config, gate_eval, report
            )
            if write_code == 0:
                if report.allow_shortfalls_applied:
                    report.promote_outcome = "promoted with shortfalls"
                else:
                    report.promote_outcome = "promoted"
            else:
                report.promote_outcome = f"blocked: {gate_eval.block_reason}"
                exit_code = write_code

    # Always render the report, even on a blocked promote.
    _DRYRUN_REPORT_PATH.write_text(_render_report(report, gen_config, sorted_projects, gate_eval), encoding="utf-8")
    print(f"gen_collision_probes: report written to {_DRYRUN_REPORT_PATH}", file=sys.stderr)
    print(f"gen_collision_probes: dry-run JSONL at {_DRYRUN_PROBES_PATH}", file=sys.stderr)

    return exit_code


# ── --reject orchestrator ──────────────────────────────────────────


async def _run_reject(
    gen_config: GenerationConfig,
    config: Config,
) -> int:
    """
    Re-run path: drop named probe_ids and re-self-grade without redrafting.

    The previous dry-run JSONL is required; if it does not exist
    we exit 2 with a diagnostic. After filtering and re-self-grading,
    promote gates run identically to the normal path.
    """
    if not _DRYRUN_PROBES_PATH.exists():
        print(
            f"gen_collision_probes: --reject requires a previous dry-run at {_DRYRUN_PROBES_PATH}; none found",
            file=sys.stderr,
        )
        return 2

    # Bootstrap the merged registry's DB layer before self-grade.
    # Self-grade runs the scoped evaluator, which detects each
    # probe's active project against the merged registry; without
    # this call, a probe pinned to a chat-registered project
    # workspace degrades silently to global-only and self-grade
    # measures different state than _run_generate would. Mirrors
    # the same call in _run_generate.
    from kai.memory_projects import load_project_registry

    await load_project_registry(config)

    found, missing = _apply_rejects_to_jsonl(_DRYRUN_PROBES_PATH, gen_config.reject_ids)
    print(
        f"gen_collision_probes: --reject dropped {len(found)} probe(s), {len(missing)} requested id(s) not found",
        file=sys.stderr,
    )

    # Rebuild the line-number mapping against the post-reject file.
    probe_id_by_line_number = _build_probe_id_by_line_number(_DRYRUN_PROBES_PATH)

    # Build a minimal report skeleton; the renderer reads everything
    # off the dataclass.
    report = GeneratorReport(
        user_id=gen_config.user_id,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        effective_backend=gen_config.effective_backend,
        effective_provider=gen_config.effective_provider,
        effective_os_user=gen_config.effective_os_user,
        model=gen_config.model,
        timeout_s=gen_config.timeout_s,
        rejected_found=found,
        rejected_missing=missing,
    )

    # Reconstitute accepted_probes from the JSONL so the renderer
    # can render distribution and sample sections. The reject path
    # does not have access to the in-memory pipeline state.
    accepted_from_disk: list[VerifiedProbe] = []
    with open(_DRYRUN_PROBES_PATH, encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            row = json.loads(stripped)
            kind = row["probe_id"].split(":")[0]
            accepted_from_disk.append(
                VerifiedProbe(
                    kind=kind,
                    probe_id=row["probe_id"],
                    question=row["question"],
                    expected_fact_id=row.get("expected_fact_id"),
                    expected_excluded_fact_ids=tuple(row.get("expected_excluded_fact_ids") or ()),
                    workspace=row.get("workspace"),
                    legacy_rank=None,
                )
            )
    report.accepted_probes = accepted_from_disk

    # Self-grade the filtered file.
    self_grade = await _self_grade(_DRYRUN_PROBES_PATH, gen_config.user_id)
    report.self_grade_verdict = self_grade.verdict
    report.n_scored_negative = self_grade.n_scored_negative
    report.exclusion_pass_in_prompt = self_grade.exclusion_pass_in_prompt
    report.exclusion_pass_in_candidates = self_grade.exclusion_pass_in_candidates
    if self_grade.verdict == "INVESTIGATE":
        report.leak_records = _extract_leak_records(self_grade, probe_id_by_line_number)

    # Promote gates against the post-reject corpus.
    sorted_projects = sorted(config.memory_projects.values(), key=lambda p: p.project_id)
    if gen_config.project_filter:
        allowed = set(gen_config.project_filter)
        sorted_projects = [p for p in sorted_projects if p.project_id in allowed]
    gate_eval = _evaluate_promote_gates(self_grade, accepted_from_disk, sorted_projects, gen_config)

    exit_code = 0
    if gen_config.promote:
        if (
            gate_eval.self_grade == "block"
            or gate_eval.structural_coverage == "block"
            or gate_eval.legacy_default_coverage == "block"
        ):
            report.promote_outcome = f"blocked: {gate_eval.block_reason}"
            exit_code = 2
        else:
            if gate_eval.structural_coverage == "relaxed":
                report.allow_shortfalls_applied["structural"] = "relaxed via --allow-shortfalls"
            if gate_eval.legacy_default_coverage == "relaxed":
                report.allow_shortfalls_applied["legacy_default"] = "relaxed via --allow-shortfalls"
            write_code = _execute_promote_write(
                _DRYRUN_PROBES_PATH, gen_config.output_path, gen_config, gate_eval, report
            )
            if write_code == 0:
                report.promote_outcome = "promoted with shortfalls" if report.allow_shortfalls_applied else "promoted"
            else:
                report.promote_outcome = f"blocked: {gate_eval.block_reason}"
                exit_code = write_code

    _DRYRUN_REPORT_PATH.write_text(_render_report(report, gen_config, sorted_projects, gate_eval), encoding="utf-8")
    print(f"gen_collision_probes: report written to {_DRYRUN_REPORT_PATH}", file=sys.stderr)
    return exit_code


# ── Main entry ─────────────────────────────────────────────────────


async def _run_cli(args: argparse.Namespace) -> int:
    """Async core dispatched from `main`."""
    config = _initialize_memory_or_exit()
    if config is None:
        return 1
    gen_config_or_err = _resolve_run_config(args, config)
    if isinstance(gen_config_or_err, str):
        print(f"gen_collision_probes: {gen_config_or_err}", file=sys.stderr)
        return 2
    gen_config = gen_config_or_err

    if gen_config.reject_ids:
        return await _run_reject(gen_config, config)
    return await _run_generate(gen_config, config)


def main(argv: list[str] | None = None) -> int:
    """
    Entry point. Returns a process exit code.

    Wires argparse, async dispatch, and exit code propagation.
    Tests can call `main` directly with a list of argv tokens; the
    `__main__` block at the bottom of the module passes None so
    argparse reads sys.argv as usual.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run_cli(args))


if __name__ == "__main__":
    sys.exit(main())
