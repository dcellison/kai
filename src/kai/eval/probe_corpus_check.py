"""
Probe-corpus coverage check for the scoped retrieval evaluator.

Reachable as `python -m kai.eval.probe_corpus_check <probes-path>`.

Given an operator-authored probe file under the scoped evaluator's
v2 schema, reports whether the corpus exercises the cross-scope
safety property the scoped evaluator measures. The evaluator's
`exclusion_pass_in_prompt` metric is meaningful only when probes
carry `expected_excluded_fact_ids`; without those, the safety axis
is invisible. This tool tells the operator, at corpus-authoring
time, whether the file they wrote covers every active project with
enough collision probes to make the metric trustworthy.

What this is NOT:
- A probe-file format validator. The scoped evaluator's `load_probes`
  is the structural gate; this tool reuses it and exits early on any
  load error. v2 schema fields are validated there, not here.
- A retrieval evaluator. It does not call Mem0, score anything, or
  touch the live store. The check is purely structural over the probe
  file plus the project registry.
- A check on the live legacy backlog. Coverage is measured against
  the operator's authoring intent, not against the actual /memory
  state for any user.
- A complete corpus authoring gate. The "at least N legacy-default
  probes" minimum from the original issue body cannot be enforced
  here: distinguishing a legacy-default-targeting probe from any
  other positive probe requires reading the expected_fact_id row's
  `scope_source` from the live store. The check covers STRUCTURAL
  minimums (collision count, positive-only count, non-project count,
  unregistered-workspace gate); legacy-default coverage is verified
  out of band, by inspecting each candidate expected_fact_id row's
  `scope_source` directly (via `/memory` or a get_by_id call), then
  running the scoped evaluator to confirm the chosen probes surface
  end to end. The evaluator's `scoped_reason` distribution describes
  the retrieval outcome (ok / no_results_after_scope / etc.), NOT
  the row's scope_source; the two signals are independent.

Exit codes mirror the admin-CLI convention:
- 0: corpus meets every documented minimum
- 1: IO or config-init failure
- 2: corpus loaded but one or more minimums are not met (the report
  names every shortfall by project so the operator can author the
  missing probes)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from kai.config import Config, MemoryProjectConfig
from kai.eval.retrieval_scoped import ScopedProbe, _detect_probe_active_project, load_probes

# Sentinel bucket keys for probes whose workspace does not resolve to
# a registered project. `_NONE_PROJECT_KEY` matches the scoped
# evaluator's own sentinel so the two tools render the same bucket
# name. `_UNREGISTERED_KEY` is distinct from non-project: a probe
# with a workspace path that the operator registered nowhere is an
# authoring mistake (likely a stale path), while a workspace=null
# probe is a deliberate non-project assertion.
_NONE_PROJECT_KEY = "__none__"
_UNREGISTERED_KEY = "__unregistered__"


# Default coverage minimums. Tunable via CLI; the defaults match the
# corpus shape the scoped evaluator's safety metric needs to be load-
# bearing: five collision probes per active project to give the
# negative denominator enough signal, two positive-only per project
# to keep the IR family meaningful in the same run, and three non-
# project probes to surface the `__none__` bucket.
_DEFAULT_MIN_COLLISION_PER_PROJECT = 5
_DEFAULT_MIN_POSITIVE_PER_PROJECT = 2
_DEFAULT_MIN_NON_PROJECT = 3


# ── Data shapes ────────────────────────────────────────────────────


@dataclass(frozen=True)
class CoverageMinimums:
    """The thresholds the coverage check measures against.

    Carried alongside the report so the rendered output names the
    exact minimums in effect (an operator running with non-default
    flags should not have to cross-reference what they passed).
    """

    min_collision_per_project: int
    min_positive_per_project: int
    min_non_project: int


@dataclass(frozen=True)
class ProjectCoverage:
    """Per-project coverage counts.

    `total_probes` is every probe whose workspace resolves to this
    project, regardless of polarity. `collision_probes` is the
    subset whose `expected_excluded_fact_ids` is non-empty (the
    safety signal); `positive_only_probes` is the subset with an
    `expected_fact_id` and no exclusions (the IR signal). A probe
    can be both polarities at once; it contributes to `total_probes`
    once and to `collision_probes` whenever the exclusion list is
    non-empty.
    """

    project_id: str
    total_probes: int
    collision_probes: int
    positive_only_probes: int


@dataclass(frozen=True)
class CoverageReport:
    """Structured coverage result over a probe corpus.

    `per_project` is keyed on registered project ids; an entry exists
    for every project in the registry even if its coverage is zero
    (so the report explicitly names the gap). `non_project_count` is
    workspace=null probes. `unregistered_workspace_count` is probes
    whose workspace path matches no registered project; those are
    authoring mistakes and surfaced separately so the operator can
    fix or remove them.

    `shortfalls` lists every minimum that was not met, one human-
    readable line per gap. An empty list means the corpus meets every
    minimum and the check exits 0.
    """

    per_project: dict[str, ProjectCoverage]
    non_project_count: int
    unregistered_workspace_count: int
    total_probes: int
    minimums: CoverageMinimums
    shortfalls: list[str]


# ── Pure helpers ───────────────────────────────────────────────────


def _classify_probe(probe: ScopedProbe) -> tuple[bool, bool]:
    """Return `(is_collision, is_positive_only)` for one probe.

    The two flags are independent: a probe can be a collision probe
    with a positive assertion (both `expected_fact_id` AND
    `expected_excluded_fact_ids` set), or a pure collision (exclusion
    only), or a pure positive (no exclusions), or neither (the loader
    rejects "neither" so this branch is unreachable here, but the
    classifier returns it cleanly for completeness).

    `positive_only` is True when the probe has a positive id AND no
    exclusions; the "only" qualifier reflects that this probe carries
    the IR signal but does NOT contribute to the safety signal.
    """
    has_positive = probe.expected_fact_id is not None
    has_exclusion = bool(probe.expected_excluded_fact_ids)
    return has_exclusion, (has_positive and not has_exclusion)


def check_corpus_coverage(
    probes: list[ScopedProbe],
    registry: dict[str, MemoryProjectConfig],
    minimums: CoverageMinimums,
    *,
    non_project_only: bool = False,
) -> CoverageReport:
    """Compute the coverage report for a probe corpus.

    Pure function: takes a list of probes and a project registry,
    returns the structured report. The CLI layer owns IO and exit
    codes; unit tests exercise this function directly with hand-
    constructed registries and probe lists.

    Registry filtering: only projects with `memory_enabled=True`
    contribute to per-project minimums. The scoped retrieval helper
    treats a detected disabled project as global-only at admission
    (it carries the active-project debug metadata but produces
    `allowed_scopes=("global",)`), so requiring collision probes
    against a disabled project would assert a property the scope
    filter does not enforce. Disabled projects do appear in the
    report's diagnostics so the operator sees they exist; they just
    do not block coverage.

    Non-project-only mode: when set, per-project minimums are
    suppressed and an empty memory-enabled registry stops being a
    shortfall. The mode exists for corpora that deliberately exercise
    only the global section (e.g. a smoke test corpus for the
    global-only retrieval posture). Without the mode, an empty
    memory-enabled registry is a shortfall (the tool's primary job
    is per-project safety coverage; "nothing to check" is not a
    meaningful pass).

    Unregistered workspace probes (workspace set but matching no
    registered project) are always a shortfall. Either the operator
    forgot to pin a chat-registered project in YAML, or the
    workspace path is stale. Silent counting would let the corpus
    exercise zero active projects while automation read exit 0.
    """
    enabled_registry = {pid: cfg for pid, cfg in registry.items() if cfg.memory_enabled}

    per_project_total: dict[str, int] = {pid: 0 for pid in enabled_registry}
    per_project_collision: dict[str, int] = {pid: 0 for pid in enabled_registry}
    per_project_positive: dict[str, int] = {pid: 0 for pid in enabled_registry}
    non_project_count = 0
    unregistered_workspace_count = 0

    for probe in probes:
        # Detection runs against the FULL registry (including disabled
        # projects) so a probe pinned to a disabled project's workspace
        # is bucketed correctly as "matched a registered project",
        # then dropped from the enabled-registry per-project counts.
        # Without this, a disabled-project probe would falsely land in
        # the unregistered bucket and trip the unregistered shortfall.
        bucket = _detect_probe_active_project(probe, registry)
        is_collision, is_positive_only = _classify_probe(probe)

        if bucket is None:
            if probe.workspace is None:
                non_project_count += 1
            else:
                unregistered_workspace_count += 1
            continue

        if bucket not in enabled_registry:
            # Disabled project: counted as matched-but-not-required.
            # The probe is not a corpus authoring mistake (the path is
            # registered), but it cannot contribute to the safety
            # signal because the scope filter would admit global only.
            continue

        per_project_total[bucket] = per_project_total.get(bucket, 0) + 1
        if is_collision:
            per_project_collision[bucket] = per_project_collision.get(bucket, 0) + 1
        if is_positive_only:
            per_project_positive[bucket] = per_project_positive.get(bucket, 0) + 1

    per_project: dict[str, ProjectCoverage] = {
        pid: ProjectCoverage(
            project_id=pid,
            total_probes=per_project_total[pid],
            collision_probes=per_project_collision[pid],
            positive_only_probes=per_project_positive[pid],
        )
        for pid in sorted(enabled_registry)
    }

    shortfalls: list[str] = []
    if not non_project_only:
        if not enabled_registry:
            shortfalls.append(
                "registry has no memory-enabled projects to check per-project coverage "
                "against (pass --non-project-only if the corpus is deliberately global-only)"
            )
        for pid in sorted(enabled_registry):
            cov = per_project[pid]
            if cov.collision_probes < minimums.min_collision_per_project:
                shortfalls.append(
                    f"project {pid!r}: {cov.collision_probes} collision probes "
                    f"(need {minimums.min_collision_per_project})"
                )
            if cov.positive_only_probes < minimums.min_positive_per_project:
                shortfalls.append(
                    f"project {pid!r}: {cov.positive_only_probes} positive-only probes "
                    f"(need {minimums.min_positive_per_project})"
                )
    if non_project_count < minimums.min_non_project:
        shortfalls.append(f"non-project (workspace=null): {non_project_count} probes (need {minimums.min_non_project})")
    if unregistered_workspace_count > 0:
        # Unregistered workspaces are always a shortfall: either the
        # operator forgot to pin chat-registered projects into YAML
        # (the registry layer this tool reads), or the workspace path
        # is stale. Silent counting would let the corpus exercise zero
        # active projects while the tool reported green.
        shortfalls.append(
            f"unregistered workspace: {unregistered_workspace_count} probes with workspace "
            "paths that match no registered project (pin in YAML or fix the paths)"
        )

    return CoverageReport(
        per_project=per_project,
        non_project_count=non_project_count,
        unregistered_workspace_count=unregistered_workspace_count,
        total_probes=len(probes),
        minimums=minimums,
        shortfalls=shortfalls,
    )


def render_report(report: CoverageReport) -> str:
    """Human-readable rendering of the coverage report.

    Plain text, no color codes. The output is meant to be pastable
    into a chat or commit message without ANSI escapes. Sections in
    stable order: a summary line, per-project counts (with the
    minimums next to each cell so an operator scanning the table
    sees both the actual and the target at the same column), the
    sentinel buckets, and the shortfall list.

    When the corpus meets every minimum, the shortfall section
    renders as a single "All minimums met." line; this lets a quick
    eyeball of the bottom of the report answer the operator's first
    question ("am I done authoring?") without scrolling through the
    counts.
    """
    lines = [
        "Probe corpus coverage report",
        "",
        f"Total probes: {report.total_probes}",
        f"Non-project (workspace=null): {report.non_project_count} (need {report.minimums.min_non_project})",
        f"Unregistered workspace: {report.unregistered_workspace_count}",
        "",
        "Per-project coverage:",
    ]
    if not report.per_project:
        lines.append("  (registry is empty; nothing to check)")
    for pid, cov in report.per_project.items():
        lines.append(
            f"  {pid}: total={cov.total_probes} collision={cov.collision_probes}"
            f"/{report.minimums.min_collision_per_project} "
            f"positive_only={cov.positive_only_probes}"
            f"/{report.minimums.min_positive_per_project}"
        )

    lines.append("")
    if report.shortfalls:
        lines.append("Shortfalls:")
        for sf in report.shortfalls:
            lines.append(f"  - {sf}")
    else:
        lines.append("All minimums met.")
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Top-level argparse for `python -m kai.eval.probe_corpus_check`."""
    parser = argparse.ArgumentParser(
        prog="python -m kai.eval.probe_corpus_check",
        description=(
            "Structural coverage check for scoped-evaluator probe corpora. "
            "Reports per-project collision and positive-only counts against "
            "documented minimums and exits non-zero when any shortfall is found."
        ),
    )
    parser.add_argument(
        "probes",
        type=Path,
        help="Path to the probe corpus (JSONL, scoped evaluator v2 schema).",
    )
    parser.add_argument(
        "--min-collision-per-project",
        type=int,
        default=_DEFAULT_MIN_COLLISION_PER_PROJECT,
        help=(
            "Minimum collision probes (expected_excluded_fact_ids non-empty) "
            f"per registered project. Default: {_DEFAULT_MIN_COLLISION_PER_PROJECT}."
        ),
    )
    parser.add_argument(
        "--min-positive-per-project",
        type=int,
        default=_DEFAULT_MIN_POSITIVE_PER_PROJECT,
        help=(
            "Minimum positive-only probes (expected_fact_id set, no exclusions) "
            f"per registered project. Default: {_DEFAULT_MIN_POSITIVE_PER_PROJECT}."
        ),
    )
    parser.add_argument(
        "--min-non-project",
        type=int,
        default=_DEFAULT_MIN_NON_PROJECT,
        help=(f"Minimum probes whose workspace is null (non-project assertions). Default: {_DEFAULT_MIN_NON_PROJECT}."),
    )
    parser.add_argument(
        "--non-project-only",
        action="store_true",
        help=(
            "Skip the per-project minimums entirely; only the --min-non-project "
            "threshold gates the result. Use for corpora that deliberately exercise "
            "only the global section. Without this flag, an empty memory-enabled "
            "registry is a shortfall (the tool's primary job is per-project safety "
            "coverage, and 'nothing to check' would be a meaningless pass)."
        ),
    )
    return parser


def _load_config_registry() -> tuple[Config, dict[str, MemoryProjectConfig]] | None:
    """Load config and return the YAML project registry.

    Returns `(config, registry)` on success or None on failure (the
    caller exits 1). Uses the YAML registry directly rather than
    `load_project_registry` so the check does NOT require the daemon
    to be stopped: it never touches the session DB or the Mem0 store.
    Chat-registered projects (DB layer) are invisible here; an
    operator who has only chat-registered projects must add a
    matching YAML entry for the duration of the check or run
    against the live registry by stopping the daemon and using the
    scoped evaluator's own flags.
    """
    try:
        from kai.config import load_config

        config = load_config()
    except Exception as e:
        print(f"probe-corpus-check: config init failed: {e}", file=sys.stderr)
        return None
    return config, config.memory_projects


def _run_cli(args: argparse.Namespace) -> int:
    """CLI dispatch.

    Returns process exit code: 0 on success (all minimums met), 1 on
    init or IO failure, 2 on a successful load with shortfalls.
    """
    try:
        probes = load_probes(args.probes)
    except (OSError, ValueError) as e:
        print(f"probe-corpus-check: failed to load probes: {e}", file=sys.stderr)
        return 1
    if not probes:
        print(
            f"probe-corpus-check: no probes loaded from {args.probes}",
            file=sys.stderr,
        )
        return 1

    loaded = _load_config_registry()
    if loaded is None:
        return 1
    _config, registry = loaded

    minimums = CoverageMinimums(
        min_collision_per_project=args.min_collision_per_project,
        min_positive_per_project=args.min_positive_per_project,
        min_non_project=args.min_non_project,
    )
    report = check_corpus_coverage(
        probes,
        registry,
        minimums,
        non_project_only=args.non_project_only,
    )
    print(render_report(report))
    return 2 if report.shortfalls else 0


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for `python -m kai.eval.probe_corpus_check`.

    `argv` defaults to `sys.argv[1:]`. Returns the exit code for the
    caller; the `__main__` block below propagates it via sys.exit.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _run_cli(args)


if __name__ == "__main__":
    sys.exit(main())
