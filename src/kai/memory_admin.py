"""
Administrative CLI for the semantic memory store.

Reached via `python -m kai memory <command> ...`. Wraps the
`delete_by_source` primitive from `src/kai/memory.py` so operators can
scrub contaminated or legacy rows from the Qdrant store without
dropping into an ad-hoc Python shell.

Scope (spec §16 + Phase 4 of spec 320):
- `purge <user_id> --source <source>`: scoped deletion. Leaves rows
  whose metadata.source does not match untouched.
- `reclassify-scope <user_id> [...]`: guarded scope reclassification
  of unreviewed global rows. Dry-run by default (report + proposals
  file, no store writes); `--apply` writes a reviewed proposals file
  with pre-images dumped first; `--rollback` restores a pre-image
  file. The pass logic lives in `src/kai/memory_reclassify.py`; this
  module owns only argument parsing, the authorization gates, and
  dispatch.
- `backfill-provenance <user_id> [...]`: stamp transcript provenance
  on legacy rows via content-overlap matching against the JSONL
  history. Dry-run by default; `--apply` writes the four required
  `source_*` keys onto surviving rows with pre-images dumped first;
  `--rollback` restores rows whose source block has not drifted.
  The pass logic lives in `src/kai/memory_provenance_backfill.py`.

Commands that modify the store require an explicit `--yes` flag. When
`--yes` is absent, the command prints the action it WOULD take and
exits with status 2 so automation can detect "not authorized" distinct
from "success". Prompting interactively was considered and rejected:
ops scripts and cron invocations should either set `--yes` upfront or
fail fast on missing authorization, rather than blocking on stdin.

The broader `delete_all(user_id=...)` primitive exists in
`src/kai/memory.py` and is documented in spec §16. It is intentionally
NOT exposed here: Phase 4 is scoped to legacy cleanup, and a
per-source purge is sufficient for the advertised use case
(`--source ""` to drop rows from pre-Phase-1 installations). If a
future incident requires the nuclear option, add it as a separate
`nuke` subcommand with its own review.

This module should NOT import `kai.bot`, `kai.pool`, or other
bot-runtime code, so the CLI stays cheap to invoke. The reclassify
handler imports `kai.memory_reclassify` (which pulls the one-shot
reasoner stack) inside the function for the same reason: a plain
`purge` invocation never pays that import.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kai.config import Config

log = logging.getLogger(__name__)


# ── Known source values ─────────────────────────────────────────────
#
# Accepted values for --source. The empty-string option covers LEGACY
# rows (pre-Phase-1 entries whose metadata dict has no "source" key).
# `delete_by_source(user_id, source="")` deliberately matches both the
# key-absent and key-empty cases, per the implementation in memory.py.
#
# We enumerate explicitly so a typo ("user-raw" vs "user_raw") fails
# at arg-parse time rather than silently no-op'ing against every row.
_KNOWN_SOURCES = frozenset({"extracted", "user_raw", ""})


def _build_parser() -> argparse.ArgumentParser:
    """Return the top-level argparser for `python -m kai memory ...`.

    Split out so tests can introspect the help text and required args
    without reaching into the dispatch function.
    """
    parser = argparse.ArgumentParser(
        prog="python -m kai memory",
        description="Administrative commands for the semantic memory store.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    purge = sub.add_parser(
        "purge",
        help="Delete rows for a user filtered by metadata source",
        description=(
            "Delete every memory for <user_id> whose metadata.source "
            "equals <source>. Leaves rows with any other source "
            "untouched. Source may be 'extracted', 'user_raw', or "
            "'' (empty string; matches pre-Phase-1 legacy rows)."
        ),
    )
    purge.add_argument(
        "user_id",
        help="Telegram chat id (as a string) whose rows to purge.",
    )
    purge.add_argument(
        "--source",
        required=True,
        help="Source tag to match. Accepted: 'extracted', 'user_raw', or '' for legacy.",
    )
    purge.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion. Without this flag the command prints the planned action and exits with status 2.",
    )

    # Lazy import: the choices list is the only config dependency at
    # parser-build time. Importing it here (not at module top) keeps
    # `import kai.memory_admin` itself free of kai.config, preserving
    # the module's zero-kai-imports surface for test collection.
    from kai.config import ONESHOT_REASONER_BACKENDS

    rec = sub.add_parser(
        "reclassify-scope",
        help="Reclassify unreviewed global memory rows (dry-run by default)",
        description=(
            "Classify rows whose resolved scope is global with legacy or "
            "extraction-default provenance. Dry-run (the default) writes a "
            "report plus a proposals file and never touches the store. "
            "--apply writes a reviewed proposals file after re-checks, "
            "dumping pre-images first. --rollback restores rows from a "
            "pre-image file. The daemon must be stopped; the embedded "
            "store is single-process."
        ),
    )
    rec.add_argument(
        "user_id",
        help="Telegram chat id (as a string) whose rows to reclassify.",
    )
    # Classification flags default to None (not their documented
    # defaults) so the mutating-mode rejection below can tell "flag
    # explicitly passed" from "default in effect"; the dry-run driver
    # applies the real defaults.
    rec.add_argument(
        "--backend",
        choices=sorted(ONESHOT_REASONER_BACKENDS),
        default=None,
        help="Reasoner backend. Default: the target user's effective backend.",
    )
    rec.add_argument(
        "--os-user",
        dest="os_user",
        default=None,
        help="OS user whose provider auth runs the reasoner. Default: the target user's os_user mapping.",
    )
    rec.add_argument(
        "--provider",
        default=None,
        help="Provider wire name (goose only). Default: the target user's effective provider.",
    )
    rec.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Confidence gate for both verdict directions. Default: 0.8.",
    )
    rec.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Eyeball-sample size in the dry-run report. Default: 10.",
    )
    rec.add_argument(
        "--out-dir",
        dest="out_dir",
        default=None,
        help="Artifact directory. Default: <DATA_DIR>/home/<user_id>/docs/reclassify/.",
    )
    mode = rec.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply",
        metavar="PROPOSALS",
        default=None,
        help="Apply a reviewed proposals file (requires --yes).",
    )
    mode.add_argument(
        "--rollback",
        metavar="PREIMAGES",
        default=None,
        help="Restore rows from a pre-image file (requires --yes).",
    )
    rec.add_argument(
        "--yes",
        action="store_true",
        help="Confirm a mutating mode. Without it, --apply/--rollback print the planned change count and exit with status 2.",
    )

    bp = sub.add_parser(
        "backfill-provenance",
        help="Stamp transcript provenance on legacy rows via content-overlap matching",
        description=(
            "Backfill the four required source_* keys onto rows extracted "
            "before transcript provenance landed. Dry-run (the default) "
            "scores each row's text against user turns in a JSONL search "
            "window; auto-matches require dominant content overlap. Apply "
            "writes the four keys onto surviving proposals after re-checks. "
            "Rollback restores rows whose source_* block has not drifted "
            "since apply. The daemon must be stopped."
        ),
    )
    bp.add_argument(
        "user_id",
        help="Telegram chat id (as a string) whose rows to backfill.",
    )
    # Scoring flags default to None so the mutating-mode rejection
    # below can tell "flag explicitly passed" from "default in effect";
    # the dry-run driver applies the real defaults.
    bp.add_argument(
        "--window-seconds",
        dest="window_seconds",
        type=int,
        default=None,
        help="Time window before created_at to search the JSONL. Default: 86400 (24h).",
    )
    bp.add_argument(
        "--min-overlap",
        dest="min_overlap",
        type=float,
        default=None,
        help="Minimum overlap score for a candidate to be considered. Default: 0.30.",
    )
    bp.add_argument(
        "--strong-overlap-ratio",
        dest="strong_overlap_ratio",
        type=float,
        default=None,
        help="Dominance ratio the winner must clear over the runner-up. Default: 2.0.",
    )
    bp.add_argument(
        "--overlap-shingle-n",
        dest="overlap_shingle_n",
        type=int,
        default=None,
        help="Token-shingle width for the overlap score. Default: 4.",
    )
    bp.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size for the report's STRONG_MATCH and curation sections. Default: 10.",
    )
    bp.add_argument(
        "--out-dir",
        dest="out_dir",
        default=None,
        help="Artifact directory. Default: <DATA_DIR>/home/<user_id>/docs/backfill-provenance/.",
    )
    bp_mode = bp.add_mutually_exclusive_group()
    bp_mode.add_argument(
        "--apply",
        metavar="PROPOSALS",
        default=None,
        help="Apply a reviewed proposals file (requires --yes).",
    )
    bp_mode.add_argument(
        "--rollback",
        metavar="PREIMAGES",
        default=None,
        help="Restore rows from a pre-image file (requires --yes).",
    )
    bp.add_argument(
        "--yes",
        action="store_true",
        help="Confirm a mutating mode. Without it, --apply/--rollback print the planned change count and exit with status 2.",
    )
    return parser


def _initialize_memory() -> Config | None:
    """Load config and call `init_memory()`; return the Config on success.

    The admin CLI reuses the same init path as the bot so the same
    embedding model, Qdrant directory, and history DB settings are in
    effect. A failure here (missing optional deps, unreadable Qdrant
    dir) is reported to stderr; the caller exits with status 1.

    Returns the loaded Config (truthy) rather than a bare True so
    subcommands that need config values (reclassify-scope reads the
    session DB path, project registry, and model settings) do not
    load the environment twice; `purge`'s `if not _initialize_memory()`
    check is unaffected. None signals failure. The embedded store is
    single-process, so a failure here while the daemon is running is
    expected; stop the service first.
    """
    try:
        from kai.config import load_config
        from kai.memory import init_memory, is_enabled

        config = load_config()
        init_memory(config)
        if not is_enabled():
            # is_enabled returns False when MEMORY_ENABLED=false or when
            # init_memory hit a recoverable problem (dimension mismatch,
            # etc.). Both cases print a warning during init_memory; we
            # just stop the command from proceeding.
            print(
                "memory admin: memory is not enabled. Set MEMORY_ENABLED=true, verify the store is "
                "readable, and make sure the daemon is stopped (the embedded store is single-process).",
                file=sys.stderr,
            )
            return None
        return config
    except Exception as e:
        print(f"memory admin: init failed: {e}", file=sys.stderr)
        return None


def _cmd_purge(args: argparse.Namespace) -> int:
    """Execute the `purge` subcommand. Returns a process exit code."""
    source = args.source
    if source not in _KNOWN_SOURCES:
        # The allow-list exists precisely to catch typos before they
        # silently delete zero rows. Spell out accepted values directly
        # rather than printing the sorted list - sorted({"", ...})
        # renders as `['', ...]`, and a leading empty string in a repr
        # list looks like a rendering artifact, not a valid input.
        print(
            f"memory admin: unknown source {source!r}. "
            "Accepted values: 'extracted', 'user_raw', "
            "or '' (empty string = legacy rows with no source set).",
            file=sys.stderr,
        )
        return 2

    user_id = args.user_id
    # Display "<legacy>" for the empty-string case so the operator can
    # read back what they are about to delete without the ambiguity of
    # a quoted empty string.
    source_label = "<legacy (source absent or empty)>" if source == "" else repr(source)
    plan = f"delete_by_source(user_id={user_id!r}, source={source_label})"

    # Init happens before the dry-run branch so a misconfigured memory
    # store (MEMORY_ENABLED=false, unreadable Qdrant dir, missing deps)
    # surfaces on the dry-run too. Without this, an operator runs the
    # CLI without --yes, sees a confident "would run ..." message, then
    # re-runs with --yes and hits an init failure at exit 1. Catching
    # it up front means the dry-run answer is also a smoke test.
    if not _initialize_memory():
        return 1

    if not args.yes:
        # Dry-run path. Exit 2 (distinct from 0 and 1) so automation can
        # tell "authorization missing" apart from "success" and "error".
        print(f"memory admin: would run {plan}")
        print("memory admin: re-run with --yes to execute.")
        return 2

    from kai.memory import delete_by_source

    try:
        # delete_by_source is async (per spec §6.2: wraps sync Mem0 calls
        # in run_in_executor). Spin up a one-shot event loop here; the
        # CLI has no other async work to coordinate with.
        count = asyncio.run(delete_by_source(user_id=user_id, source=source))
    except Exception as e:
        print(f"memory admin: purge failed: {e}", file=sys.stderr)
        return 1

    print(f"memory admin: deleted {count} row(s) for {plan}")
    return 0


def _cmd_reclassify(args: argparse.Namespace) -> int:
    """Execute the `reclassify-scope` subcommand. Returns an exit code.

    Gate order, deliberately:
    1. Classification flags are rejected in mutating modes before
       anything else; a typo must not silently change apply
       semantics.
    2. Memory init runs before the --yes gate (same rationale as
       purge: the no-yes plan doubles as a smoke test).
    3. Mutating modes parse and header-validate their artifact
       BEFORE the --yes gate, so the plan shows the real change
       count and a wrong-user file fails loudly up front.
    The pass logic lives in kai.memory_reclassify; this function owns
    only the gates and dispatch, per the module-docstring split.
    """
    mutating = args.apply is not None or args.rollback is not None
    if mutating:
        # The None defaults exist precisely for this check: any
        # non-None value was explicitly passed.
        offenders = [
            name
            for name, value in (
                ("--backend", args.backend),
                ("--os-user", args.os_user),
                ("--provider", args.provider),
                ("--threshold", args.threshold),
                ("--sample", args.sample),
            )
            if value is not None
        ]
        if offenders:
            print(
                f"memory admin: {', '.join(offenders)} only apply to dry-run classification; "
                "remove them when using --apply/--rollback.",
                file=sys.stderr,
            )
            return 2

    threshold = args.threshold if args.threshold is not None else 0.8
    if not (0.0 <= threshold <= 1.0):
        print(f"memory admin: --threshold must be in [0.0, 1.0], got {threshold}", file=sys.stderr)
        return 2
    sample = args.sample if args.sample is not None else 10
    # A negative sample would survive until report rendering and kill
    # the run AFTER every reasoner call has been paid for; reject the
    # typo up front. Zero is valid (no sample section).
    if sample < 0:
        print(f"memory admin: --sample must be >= 0, got {sample}", file=sys.stderr)
        return 2

    config = _initialize_memory()
    if config is None:
        return 1

    from pathlib import Path

    from kai import memory_reclassify
    from kai.config import DATA_DIR

    out_dir = Path(args.out_dir) if args.out_dir else DATA_DIR / "home" / args.user_id / "docs" / "reclassify"

    if not mutating:
        return asyncio.run(
            memory_reclassify.run_dry_run(
                config,
                args.user_id,
                backend=args.backend,
                os_user=args.os_user,
                provider=args.provider,
                threshold=threshold,
                sample=sample,
                out_dir=out_dir,
            )
        )

    # Typed parsers, not the generic artifact reader: a hand-edited
    # row with a bad verdict or confidence fails HERE, in the plan
    # path, before --yes and before any store access. The driver
    # re-validates the same way, so both entries share one contract.
    artifact_path = Path(args.apply if args.apply is not None else args.rollback)
    row_type = "proposal" if args.apply is not None else "preimage"
    try:
        text = artifact_path.read_text(encoding="utf-8")
        if args.apply is not None:
            header, rows = memory_reclassify.parse_proposals(text)
        else:
            header, rows = memory_reclassify.parse_preimages(text)
    except (OSError, ValueError) as e:
        print(f"memory admin: cannot read {row_type} file: {e}", file=sys.stderr)
        return 1
    error = memory_reclassify.validate_header(header, user_id=args.user_id)
    if error is not None:
        print(f"memory admin: {error}", file=sys.stderr)
        return 1

    if not args.yes:
        verb = "apply" if args.apply is not None else "roll back"
        print(f"memory admin: would {verb} {len(rows)} row(s) from run {header['run_id']} for user {args.user_id}.")
        print("memory admin: re-run with --yes to execute.")
        return 2

    if args.apply is not None:
        return asyncio.run(
            memory_reclassify.run_apply(config, args.user_id, proposals_path=artifact_path, out_dir=out_dir)
        )
    return asyncio.run(memory_reclassify.run_rollback(config, args.user_id, preimages_path=artifact_path))


def _cmd_backfill_provenance(args: argparse.Namespace) -> int:
    """Execute the `backfill-provenance` subcommand. Returns an exit code.

    Mirrors `_cmd_reclassify`'s gate order: scoring-style flags are
    rejected in mutating modes BEFORE anything else (a typo on
    `--min-overlap` must not silently change apply semantics, because
    apply consumes the proposals file verbatim); memory init runs
    before the `--yes` gate so the dry-run plan doubles as a smoke
    test; mutating modes parse and header-validate their artifact
    BEFORE `--yes` so the plan shows the real change count and a
    wrong-user file fails loudly up front.
    """
    mutating = args.apply is not None or args.rollback is not None
    if mutating:
        offenders = [
            name
            for name, value in (
                ("--window-seconds", args.window_seconds),
                ("--min-overlap", args.min_overlap),
                ("--strong-overlap-ratio", args.strong_overlap_ratio),
                ("--overlap-shingle-n", args.overlap_shingle_n),
                ("--sample", args.sample),
            )
            if value is not None
        ]
        if offenders:
            print(
                f"memory admin: {', '.join(offenders)} only apply to dry-run scoring; "
                "remove them when using --apply/--rollback.",
                file=sys.stderr,
            )
            return 2

    # Apply defaults from the module's constants only after the
    # mutating-mode rejection runs, so the rejection's "explicitly
    # passed" check stays meaningful.
    from kai import memory_provenance_backfill

    window_seconds = (
        args.window_seconds if args.window_seconds is not None else memory_provenance_backfill._DEFAULT_WINDOW_SECONDS
    )
    min_overlap = args.min_overlap if args.min_overlap is not None else memory_provenance_backfill._DEFAULT_MIN_OVERLAP
    strong_overlap_ratio = (
        args.strong_overlap_ratio
        if args.strong_overlap_ratio is not None
        else memory_provenance_backfill._DEFAULT_STRONG_OVERLAP_RATIO
    )
    shingle_n = (
        args.overlap_shingle_n if args.overlap_shingle_n is not None else memory_provenance_backfill._DEFAULT_SHINGLE_N
    )
    sample = args.sample if args.sample is not None else 10

    if window_seconds < 1:
        print(f"memory admin: --window-seconds must be a positive int, got {window_seconds}", file=sys.stderr)
        return 2
    if not (0.0 <= min_overlap <= 1.0):
        print(f"memory admin: --min-overlap must be in [0.0, 1.0], got {min_overlap}", file=sys.stderr)
        return 2
    if strong_overlap_ratio < 1.0:
        print(
            f"memory admin: --strong-overlap-ratio must be >= 1.0, got {strong_overlap_ratio}",
            file=sys.stderr,
        )
        return 2
    if shingle_n < 1:
        print(f"memory admin: --overlap-shingle-n must be a positive int, got {shingle_n}", file=sys.stderr)
        return 2
    if sample < 0:
        print(f"memory admin: --sample must be >= 0, got {sample}", file=sys.stderr)
        return 2

    config = _initialize_memory()
    if config is None:
        return 1

    from pathlib import Path

    from kai.config import DATA_DIR

    out_dir = Path(args.out_dir) if args.out_dir else DATA_DIR / "home" / args.user_id / "docs" / "backfill-provenance"

    if not mutating:
        return asyncio.run(
            memory_provenance_backfill.run_dry_run(
                config,
                args.user_id,
                window_seconds=window_seconds,
                min_overlap=min_overlap,
                strong_overlap_ratio=strong_overlap_ratio,
                shingle_n=shingle_n,
                sample=sample,
                out_dir=out_dir,
            )
        )

    # Typed parsers, not the generic artifact reader: a hand-edited
    # row with a bad field fails HERE, in the plan path, before --yes
    # and before any store access. The driver re-validates the same
    # way, so both entries share one contract.
    artifact_path = Path(args.apply if args.apply is not None else args.rollback)
    row_type = "proposal" if args.apply is not None else "preimage"
    try:
        text = artifact_path.read_text(encoding="utf-8")
        if args.apply is not None:
            header, rows = memory_provenance_backfill.parse_proposals(text)
        else:
            header, rows = memory_provenance_backfill.parse_preimages(text)
    except (OSError, ValueError) as e:
        print(f"memory admin: cannot read {row_type} file: {e}", file=sys.stderr)
        return 1
    error = memory_provenance_backfill.validate_header(header, user_id=args.user_id)
    if error is not None:
        print(f"memory admin: {error}", file=sys.stderr)
        return 1

    if not args.yes:
        verb = "apply" if args.apply is not None else "roll back"
        print(f"memory admin: would {verb} {len(rows)} row(s) from run {header['run_id']} for user {args.user_id}.")
        print("memory admin: re-run with --yes to execute.")
        return 2

    if args.apply is not None:
        return asyncio.run(
            memory_provenance_backfill.run_apply(
                config,
                args.user_id,
                proposals_path=artifact_path,
                out_dir=out_dir,
            )
        )
    return asyncio.run(
        memory_provenance_backfill.run_rollback(
            config,
            args.user_id,
            preimages_path=artifact_path,
        )
    )


def cli(argv: list[str]) -> None:
    """Dispatch entry point. Called from `__main__.py` with argv[2:].

    Exits the process with the subcommand's return code. argparse
    handles `-h`/`--help` by exiting 0; unknown subcommands exit 2.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "purge":
        sys.exit(_cmd_purge(args))
    if args.command == "reclassify-scope":
        sys.exit(_cmd_reclassify(args))
    if args.command == "backfill-provenance":
        sys.exit(_cmd_backfill_provenance(args))
    # argparse's required=True on the subparsers guarantees a known
    # command reaches this point, so the else branch is unreachable
    # under normal invocation. Guarded anyway in case a future
    # subcommand is added without a dispatch entry.
    parser.error(f"unhandled command: {args.command}")
