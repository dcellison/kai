#!/usr/bin/env python3
"""
Periodic tag dedup script (#418, Sub D of #388).

Walks the tag distribution across `source: extracted` and `source: episode`
rows for one or more users, identifies near-duplicate tag values, and
rewrites them to a canonical form via `memory.update_metadata`.

Three near-duplicate classes ship in this version:
  - Case-insensitive equality (`Preference` vs `preference`).
  - Plural/singular pairs (`-s`, `-es` for sibilant endings, `-ies/-y`).
  - Within-row deduplication after the per-cluster rewrite collapses
    two case-variants on a single row.

Default mode is dry-run. The `--apply` flag is required to write. Per-user
scope by default; `--all-users` opts in to cross-user iteration over
`config.allowed_user_ids` (the DM-mode assumption that user_id == chat_id
holds for typical Kai installs; group-chat installs would need a
chat-id-keyed iteration source).

`confirmed_action` is structurally significant in the extractor pipeline.
Two layers of defense:
  - Cluster-skip: any cluster containing `confirmed_action` (case-insensitive)
    is dropped before any rewrite is proposed.
  - Row-level guard: if a row's tag list loses `confirmed_action` after the
    per-cluster rewrite, the row is returned unchanged. The row-level guard
    is unreachable under the cluster-skip rule but exists as defense-in-depth
    against a future rule addition that bypasses cluster-skip.

Migration rows (`source: migration`) are excluded from the dedup pass.
They carry H3-slug tags that are not part of the LLM-tag taxonomy; merging
them with extracted/episode tags would conflate two vocabulary spaces.

Usage:
    # Dry run, single user (recommended first step).
    python scripts/tag-dedup.py --chat-id 12345

    # Apply, single user, with interactive override prompt per cluster.
    python scripts/tag-dedup.py --chat-id 12345 --apply

    # Apply without prompts (accepts every proposed canonical form).
    python scripts/tag-dedup.py --chat-id 12345 --apply --no-prompt

    # Dry run across every authorised user.
    python scripts/tag-dedup.py --all-users
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kai.memory import MemoryResult


# Exit codes (documented in the script header):
#   0 - success (dry-run completed, or apply completed with all rewrites successful)
#   1 - unexpected error
#   2 - memory subsystem disabled (config.memory_enabled is False)
#   3 - one or more rewrites failed (apply mode only); audit log written for the successful subset
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_MEMORY_DISABLED = 2
EXIT_PARTIAL_FAILURE = 3


# Source filter: dedup acts on `extracted` and `episode` rows only.
# Migration rows carry H3-slug tags from a different vocabulary space
# (per the migration script's chunk-naming convention) and merging them
# with LLM-tag taxonomy values would conflate the two spaces. The
# `memory.update_metadata` wrapper itself admits all USER_VISIBLE_SOURCES
# (extracted + episode + migration); the narrower scope is enforced
# here at the script's load_rows boundary.
DEDUP_SOURCES: frozenset[str] = frozenset({"extracted", "episode"})


# Sibilant endings that take `-es` for the plural rather than `-s`. The
# plural/singular rule for these endings only fires when both the
# plural form (ending in one of these patterns) and the singular form
# (after dropping `-es`) are observed in the corpus.
#
# `zzes` (not `zes`): a singular-z word like `size` makes its plural
# via the bare `-s` rule (`size` + `s` -> `sizes`), not via `-es`.
# Matching `zes` would generate a spurious `siz` candidate for `sizes`,
# which the corpus check filters out in practice but wastes a candidate
# slot. Restricting to `zzes` keeps the rule firing for genuine
# sibilant-z plurals (`buzzes` -> `buzz`, `fizzes` -> `fizz`).
_SIBILANT_PLURAL_SUFFIXES: tuple[str, ...] = ("xes", "shes", "ches", "zzes", "sses")


# The structurally significant tag value that the extractor and validator
# pipeline keys off (Rules 4 and 4b in `_validate_facts`; the per-source
# render branch in `_build_fact_view`). The dedup pass refuses to touch
# any cluster or row that involves this string.
_CONFIRMED_ACTION_TAG: str = "confirmed_action"


# ── Cluster representation ─────────────────────────────────────────────


@dataclass
class Cluster:
    """One near-duplicate group of tag variants with a chosen canonical.

    Attributes:
        members: The variant tag values that fall into this cluster.
            Always at least 2 entries when the cluster is emitted from
            `identify_clusters` (single-tag groups are not clusters).
        canonical: The variant chosen as the cluster's target form.
            Selected by most-frequent wins, lexicographic tiebreak.
        total_occurrences: Sum of the per-variant counts across members.
            Defaults to 0 so test fixtures can construct minimal
            clusters without naming an irrelevant count value;
            production code paths build clusters via
            `identify_clusters` which always populates this from the
            distribution.
    """

    members: list[str]
    canonical: str
    total_occurrences: int = 0


# ── Pipeline helpers ──────────────────────────────────────────────────


def load_rows(user_id: int) -> list[MemoryResult]:
    """Fetch all dedup-eligible rows for `user_id` from the memory store.

    Filters client-side to `metadata.source in DEDUP_SOURCES`. The
    `limit=None` argument to `get_all` ensures a user with thousands of
    rows still gets the complete distribution. Returns an empty list
    when the user has no dedup-eligible rows.
    """
    from kai import memory

    rows = memory.get_all(user_id=str(user_id), limit=None)
    return [r for r in rows if r.metadata.get("source") in DEDUP_SOURCES]


def compute_distribution(rows: list[MemoryResult]) -> dict[str, int]:
    """Count case-preserved tag occurrences across every row's tag list.

    A row tagged `["preference", "Preference"]` contributes 1 to each
    of `"preference"` and `"Preference"` (case is preserved at this
    stage so the canonical-selection step in `identify_clusters` can
    pick the most-frequent variant). Empty or missing tag lists
    contribute nothing.
    """
    distribution: dict[str, int] = {}
    for row in rows:
        for tag in row.metadata.get("tags") or []:
            distribution[tag] = distribution.get(tag, 0) + 1
    return distribution


def _try_singular_forms(tag: str) -> list[str]:
    """Return candidate singular forms for `tag` under the dedup rules.

    Three rules in priority order:
      - `-ies/-y`: `categories` -> `category`. Applied first so the
        `-ies` ending does not get reduced to `categori` by the bare
        `-s` rule below.
      - Sibilant `-es` removal (`-xes`, `-shes`, `-ches`, `-zzes`,
        `-sses`): `boxes` -> `box`. Applied before the bare `-s` rule
        so `boxes` does not first reduce to `boxe`.
      - Bare `-s` removal: `preferences` -> `preference`.

    Returns every candidate that the rules generate; the caller checks
    against the corpus distribution to keep only those that match an
    observed tag.
    """
    # Empty and one-character tags cannot be plurals under any rule.
    if len(tag) < 2:
        return []
    candidates: list[str] = []
    if tag.endswith("ies") and len(tag) > 3:
        candidates.append(tag[:-3] + "y")
    if any(tag.endswith(suffix) for suffix in _SIBILANT_PLURAL_SUFFIXES):
        candidates.append(tag[:-2])
    if tag.endswith("s") and not tag.endswith("ss"):
        # The `not endswith("ss")` guard prevents stripping the second
        # `s` from already-singular nouns like `bus` and `mass`. The
        # bare `-s` rule only fires for genuine plural shapes.
        candidates.append(tag[:-1])
    return candidates


def _select_canonical(members: list[str], distribution: dict[str, int]) -> str:
    """Pick the cluster's canonical form from its members.

    Most-frequent wins; ties broken lexicographically. The negative
    count in the sort key turns "most" into "first" under ascending
    sort, and the secondary key (the variant string) handles ties
    deterministically.
    """
    # Build (count, variant) pairs, then sort by descending count then
    # ascending variant. Python's tuple comparison handles the dual key
    # naturally when the count is negated.
    return min(members, key=lambda m: (-distribution.get(m, 0), m))


def identify_clusters(distribution: dict[str, int]) -> list[Cluster]:
    """Group tag variants into clusters and pick a canonical per group.

    Three cluster-detection rules:
      - Case-insensitive equality: tags whose lowercase forms match
        cluster together.
      - Plural/singular pairs: case-fold representatives are joined
        when one's singular candidate is another's case-fold form.
      - Within-row deduplication is handled per-row in
        `apply_cluster_to_row`, not here.

    Implementation uses a union-find pass over case-fold keys so a
    three-way overlap like `{"Preferences", "preferences", "preference"}`
    collapses into a single cluster. A simpler "case-fold pass first,
    then plural/singular pass" sequencing would emit two disjoint
    clusters for that input because the first pass would eagerly claim
    `["Preferences", "preferences"]` and the second pass would refuse
    to link the already-assigned plural to the singular.

    Skips any cluster where any member equals `confirmed_action`
    (case-insensitive): the structurally significant tag must not
    participate in any rewrite (the extractor pipeline keys off it
    in `_validate_facts` Rules 4 and 4b plus the per-source render
    branch in `_build_fact_view`).
    """
    # Group by case-folded form first. Iterate the distribution in
    # deterministic order so the test surface is stable.
    by_case_fold: dict[str, list[str]] = {}
    for variant in sorted(distribution.keys()):
        case_fold = variant.lower()
        by_case_fold.setdefault(case_fold, []).append(variant)

    # Union-find over case-fold keys. Two keys are in the same
    # component when (a) they share a case-fold representative
    # trivially (each key is its own initial component), or (b) one
    # key's plural/singular candidate matches another key. Connected
    # components emit one cluster each.
    parent: dict[str, str] = {key: key for key in by_case_fold}

    def find(key: str) -> str:
        # Iterative path compression. The while loop walks up the
        # parent chain and rewires each node directly to the eventual
        # root so subsequent finds are O(1).
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    # Walk the case-fold keys and link any pair where the plural's
    # singular candidate matches another key. Iterate in sorted order
    # for deterministic component-roots across runs.
    for plural_key in sorted(by_case_fold.keys()):
        for singular_key in _try_singular_forms(plural_key):
            if singular_key == plural_key:
                continue
            if singular_key not in by_case_fold:
                continue
            # Cluster-skip: never link confirmed_action to anything via
            # the plural/singular rule. Trips before the union so the
            # confirmation case-fold key stays in its own component.
            if plural_key == _CONFIRMED_ACTION_TAG or singular_key == _CONFIRMED_ACTION_TAG:
                continue
            union(plural_key, singular_key)

    # Build per-component member lists. Components with only one
    # variant (single case-fold key, no plural/singular link, only one
    # variant under the case-fold) are not clusters.
    components: dict[str, list[str]] = {}
    for case_fold_key in by_case_fold:
        root = find(case_fold_key)
        components.setdefault(root, []).extend(by_case_fold[case_fold_key])

    clusters: list[Cluster] = []
    for component_members in components.values():
        if len(component_members) < 2:
            continue
        # Cluster-skip: drop any component that touches confirmed_action.
        # Catches both the all-confirmation case-fold cluster and any
        # hypothetical bypass that links confirmed_action to another
        # form despite the union-skip above.
        if any(member.lower() == _CONFIRMED_ACTION_TAG for member in component_members):
            continue
        canonical = _select_canonical(component_members, distribution)
        total = sum(distribution[m] for m in component_members)
        clusters.append(Cluster(members=list(component_members), canonical=canonical, total_occurrences=total))

    return clusters


def apply_cluster_to_row(
    row_tags: list[str],
    clusters: list[Cluster],
) -> tuple[list[str], bool]:
    """Apply every relevant cluster's canonical-form rewrite to a row's tag list.

    Walks `row_tags`, looks each tag up across the clusters, and emits
    the cluster's canonical when a match is found. Within-row duplicate
    collapse is the second pass: after all rewrites land, `dict.fromkeys`
    deduplicates while preserving order.

    Row-level `confirmed_action` guard (defense-in-depth): if the input list
    contained `confirmed_action` but the output does not, return the
    input unchanged with a False change flag. This is unreachable under
    the cluster-skip in `identify_clusters` plus the rule that the
    plural/singular rule cannot map `confirmed_action` to anything else,
    but the explicit row-level check catches a future rule addition that
    bypasses cluster-skip (e.g. a synonym-detection pass that sneaks
    `confirmed_action` into a cluster).

    Returns the new tag list and a boolean indicating whether any
    rewrite landed (or whether the within-row dedup collapsed any
    duplicates). False means the caller should skip the write entirely.
    """
    # Build a member-to-canonical lookup once per call. Multiple
    # clusters can claim multiple member tags; the dict is the cheap
    # way to fan out the per-tag rewrite without scanning every cluster
    # for every tag.
    member_to_canonical: dict[str, str] = {}
    for cluster in clusters:
        for member in cluster.members:
            member_to_canonical[member] = cluster.canonical

    # First pass: rewrite each tag if it's a cluster member.
    rewritten = [member_to_canonical.get(tag, tag) for tag in row_tags]

    # Second pass: collapse within-row duplicates while preserving order.
    # `dict.fromkeys` is the idiomatic order-preserving dedup.
    deduped = list(dict.fromkeys(rewritten))

    # Row-level confirmed_action guard. Refuse to publish a tag list
    # that drops the magic string from a row that originally had it.
    if _CONFIRMED_ACTION_TAG in row_tags and _CONFIRMED_ACTION_TAG not in deduped:
        return list(row_tags), False

    changed = deduped != row_tags
    return deduped, changed


def format_dry_run_report(distribution: dict[str, int], clusters: list[Cluster]) -> str:
    """Render the dry-run report as a multi-line string.

    Three sections: tag distribution before, proposed merges (one line
    per cluster), and tag distribution after. The "after" distribution
    is computed by simulating every cluster rewrite; this is purely
    presentational so the operator can see the post-dedup shape before
    committing to writes.
    """
    lines: list[str] = []
    lines.append("=== Tag distribution (before) ===")
    if not distribution:
        lines.append("  (no tags)")
    else:
        for variant, count in sorted(distribution.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"  {variant:<40s}  {count:>5d}")
    lines.append("")
    lines.append(f"=== Proposed merges ({len(clusters)} cluster(s)) ===")
    if not clusters:
        lines.append("  (no merges proposed)")
    else:
        for cluster in clusters:
            members_with_counts = ", ".join(
                f"{m!r} ({distribution.get(m, 0)})"
                for m in sorted(cluster.members, key=lambda v: (-distribution.get(v, 0), v))
            )
            lines.append(f"  -> {cluster.canonical!r}: [{members_with_counts}]")

    # Compute the post-dedup distribution by remapping each variant.
    member_to_canonical: dict[str, str] = {}
    for cluster in clusters:
        for member in cluster.members:
            member_to_canonical[member] = cluster.canonical
    after: dict[str, int] = {}
    for variant, count in distribution.items():
        canonical = member_to_canonical.get(variant, variant)
        after[canonical] = after.get(canonical, 0) + count

    lines.append("")
    lines.append("=== Tag distribution (after) ===")
    if not after:
        lines.append("  (no tags)")
    else:
        for variant, count in sorted(after.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"  {variant:<40s}  {count:>5d}")
    return "\n".join(lines)


def prompt_for_override(cluster: Cluster, distribution: dict[str, int]) -> str | None:
    """Interactive prompt for the operator to accept, edit, or skip a cluster.

    Returns the chosen canonical form, or None to skip this cluster.
    The `edit` branch loops until the operator picks a valid existing
    member (an arbitrary string would defeat the dedup-into-known-form
    invariant; the rewritten tag must be one the corpus already holds).
    """
    members_with_counts = ", ".join(
        f"{m!r} ({distribution.get(m, 0)})" for m in sorted(cluster.members, key=lambda v: (-distribution.get(v, 0), v))
    )
    print(f"\nCluster: [{members_with_counts}]")
    print(f"Canonical: {cluster.canonical!r}")
    while True:
        choice = input("Apply? [y/n/edit/skip]: ").strip().lower()
        if choice in ("y", "yes"):
            return cluster.canonical
        if choice in ("", "n", "no", "skip"):
            return None
        if choice in ("edit", "e"):
            new_canonical = input("New canonical (must match a cluster member): ").strip()
            if new_canonical in cluster.members:
                return new_canonical
            print(f"Invalid: {new_canonical!r} is not a member of the cluster.")
            # Fall through to re-prompt the y/n/edit/skip choice.
            continue
        print(f"Unrecognised choice {choice!r}. Use y, n, edit, or skip.")


@dataclass
class _RewritePlan:
    """Per-row plan for the apply pass.

    Holds the row, its computed new tag list, and a snapshot of the
    pre-rewrite metadata for the audit log. Built in `apply_rewrites`
    so the operator-prompt path and the no-prompt path share the same
    structure.
    """

    row: MemoryResult
    new_tags: list[str]
    old_tags: list[str]


def apply_rewrites(
    rows: list[MemoryResult],
    clusters: list[Cluster],
    user_id: int,
    audit_log_path: Path,
) -> tuple[int, int]:
    """Apply each row's tag-list rewrite via `memory.update_metadata`.

    For every row whose tag list changes, builds the new metadata dict
    by COPYING the existing metadata and overwriting only the `tags`
    field. This is the read-merge-write pattern that the wrapper's
    docstring mandates: the wrapper does NOT auto-merge; passing a
    sparse dict like `{"tags": [...]}` would destroy every other field
    on the row.

    Successful rewrites are collected in memory and flushed to the
    audit log in a single append at the end. The flush is skipped
    entirely when no rewrite landed, so the file is never created
    empty. Multiple `--apply` runs targeting the same path accumulate
    via append mode.

    Returns (success_count, fail_count). The `fail_count` includes any
    row whose `update_metadata` call returned False (Mem0 raise, row
    vanished between read and write, etc.).
    """
    from kai import memory

    success = 0
    failure = 0
    audit_entries: list[dict[str, Any]] = []
    for row in rows:
        new_tags, changed = apply_cluster_to_row(row.metadata.get("tags") or [], clusters)
        if not changed:
            continue
        # Read-merge-write: copy the existing metadata dict, set the
        # new tags, pass the merged dict. If we passed
        # `{"tags": new_tags}` directly to the wrapper, Mem0's
        # `_update_memory` would destroy every other metadata field
        # (source, confidence, prompt_version, ...) per the wrapper's
        # documented contract.
        new_metadata = dict(row.metadata)
        new_metadata["tags"] = new_tags
        ok = memory.update_metadata(
            user_id=str(user_id),
            memory_id=row.id,
            data=row.text,
            metadata=new_metadata,
        )
        if ok:
            success += 1
            audit_entries.append(
                {
                    "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "memory_id": row.id,
                    "user_id": str(user_id),
                    "old_tags": row.metadata.get("tags") or [],
                    "new_tags": new_tags,
                    "old_text": row.text,
                }
            )
        else:
            failure += 1
            print(f"WARN: update_metadata returned False for {row.id}", file=sys.stderr)

    # Flush only when there's actually something to write. Skipping
    # the open call entirely on the empty case avoids creating a
    # zero-byte audit log file when clusters were proposed but no row
    # in the corpus carried a matching tag (rare but possible if the
    # corpus changed between load_rows and apply_rewrites).
    if audit_entries:
        with audit_log_path.open("a", encoding="utf-8") as audit_log:
            for entry in audit_entries:
                audit_log.write(json.dumps(entry) + "\n")
    return success, failure


# ── CLI plumbing ──────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Periodic tag dedup script (#418, Sub D of #388).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--chat-id",
        type=int,
        help="Telegram chat_id to scan (DM-mode: chat_id == user_id).",
    )
    target.add_argument(
        "--all-users",
        action="store_true",
        help="Iterate over every authorised user from config.allowed_user_ids.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rewrite tags. Without this flag the script runs in dry-run mode.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Skip the interactive override prompt; accept every proposed canonical form.",
    )
    parser.add_argument(
        "--audit-log",
        type=Path,
        default=None,
        help="Override the default audit-log path (only used with --apply).",
    )
    return parser


def _resolve_audit_log_path(override: Path | None) -> Path:
    """Return the audit-log path, defaulting to a timestamped name in scripts/."""
    if override is not None:
        return override
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    # The script lives in scripts/; use that as the default location so
    # the .gitignore entry (`scripts/.tag-dedup-audit-*.jsonl`) catches
    # the file without an explicit `--audit-log` path.
    scripts_dir = Path(__file__).resolve().parent
    return scripts_dir / f".tag-dedup-audit-{timestamp}.jsonl"


def _run_for_user(
    user_id: int,
    *,
    apply: bool,
    no_prompt: bool,
    audit_log_path: Path,
) -> tuple[int, int]:
    """Run the dedup pipeline for one user; return (success, failure) on apply, (0, 0) on dry-run."""
    rows = load_rows(user_id)
    distribution = compute_distribution(rows)
    clusters = identify_clusters(distribution)

    print(f"\n### user_id={user_id} ###")
    print(format_dry_run_report(distribution, clusters))

    if not apply:
        return (0, 0)

    if not clusters:
        # Nothing to rewrite; skip the apply pass without opening the
        # audit log file (which would otherwise be created empty on
        # every no-op run).
        return (0, 0)

    # Operator-prompt phase: filter clusters to the operator-accepted
    # set. Each cluster's canonical may be edited.
    accepted_clusters: list[Cluster] = []
    for cluster in clusters:
        if no_prompt:
            accepted_clusters.append(cluster)
            continue
        chosen = prompt_for_override(cluster, distribution)
        if chosen is None:
            continue
        if chosen != cluster.canonical:
            cluster = Cluster(members=cluster.members, canonical=chosen, total_occurrences=cluster.total_occurrences)
        accepted_clusters.append(cluster)

    if not accepted_clusters:
        print("\nNo clusters accepted; skipping apply.")
        return (0, 0)

    return apply_rewrites(rows, accepted_clusters, user_id, audit_log_path)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Lazy imports for everything that pulls in torch/Mem0 so a `--help`
    # or argparse error exits without paying the import cost.
    from kai.config import load_config
    from kai.memory import init_memory

    config = load_config()
    if not config.memory_enabled:
        print(
            "ERROR: memory subsystem is disabled (config.memory_enabled = False).",
            file=sys.stderr,
        )
        return EXIT_MEMORY_DISABLED

    init_memory(config)

    audit_log_path = _resolve_audit_log_path(args.audit_log)

    if args.all_users:
        # Iterate config.allowed_user_ids (Telegram user IDs). Under the
        # DM-mode assumption (user_id == chat_id) these match the memory
        # store's chat_id-keyed user_id field. Group-chat installs would
        # need a chat-id-keyed iteration source instead.
        target_ids = sorted(config.allowed_user_ids)
    else:
        target_ids = [args.chat_id]

    total_success = 0
    total_failure = 0
    for target_id in target_ids:
        success, failure = _run_for_user(
            target_id,
            apply=args.apply,
            no_prompt=args.no_prompt,
            audit_log_path=audit_log_path,
        )
        total_success += success
        total_failure += failure

    if args.apply:
        print(f"\nTotal rewrites: {total_success} succeeded, {total_failure} failed.")
        # Audit log is only created when at least one rewrite succeeds
        # (apply_rewrites flushes lazily). Guard the path message on
        # total_success > 0 so an all-fail run does not direct the
        # operator to a file that was never created.
        if total_success > 0:
            stream = sys.stderr if total_failure > 0 else sys.stdout
            print(f"Audit log: {audit_log_path}", file=stream)
        if total_failure > 0:
            return EXIT_PARTIAL_FAILURE

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
