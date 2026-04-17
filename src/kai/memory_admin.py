"""
Administrative CLI for the semantic memory store.

Reached via `python -m kai memory <command> ...`. Wraps the
`delete_by_source` primitive from `src/kai/memory.py` so operators can
scrub contaminated or legacy rows from the Qdrant store without
dropping into an ad-hoc Python shell.

Scope (spec §16 + Phase 4 of spec 320):
- `purge <user_id> --source <source>`: scoped deletion. Leaves rows
  whose metadata.source does not match untouched.

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
bot-runtime code, so the CLI stays cheap to invoke. Only the memory
and config modules are needed.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

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
    return parser


def _initialize_memory() -> bool:
    """Load config and call `init_memory()`; return True on success.

    The admin CLI reuses the same init path as the bot so the same
    embedding model, Qdrant directory, and history DB settings are in
    effect. A failure here (missing optional deps, unreadable Qdrant
    dir) is reported to stderr; the caller exits with status 1.
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
                "memory admin: memory is not enabled. Set MEMORY_ENABLED=true and verify the store is readable.",
                file=sys.stderr,
            )
            return False
        return True
    except Exception as e:
        print(f"memory admin: init failed: {e}", file=sys.stderr)
        return False


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


def cli(argv: list[str]) -> None:
    """Dispatch entry point. Called from `__main__.py` with argv[2:].

    Exits the process with the subcommand's return code. argparse
    handles `-h`/`--help` by exiting 0; unknown subcommands exit 2.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "purge":
        sys.exit(_cmd_purge(args))
    # argparse's required=True on the subparsers guarantees a known
    # command reaches this point, so the else branch is unreachable
    # under normal invocation. Guarded anyway in case a future
    # subcommand is added without a dispatch entry.
    parser.error(f"unhandled command: {args.command}")
