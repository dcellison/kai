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
- `review-legacy-scope <user_id> [...]`: export every residual
  legacy-default row for an explicit operator disposition, then apply
  the complete reviewed manifest with pre-images and rollback guards.
- `backfill-provenance <user_id> [...]`: stamp transcript provenance
  on legacy rows via content-overlap matching against the JSONL
  history. Dry-run by default; `--apply` writes the four required
  `source_*` keys onto surviving rows with pre-images dumped first;
  `--rollback` restores rows whose source block has not drifted.
  The pass logic lives in `src/kai/memory_provenance_backfill.py`.
- `purge-sandbox`: delete every eval-residue identity (user_id prefix
  `sandbox-`) from the collection. Censuses owners collection-wide,
  prints the plan, and verifies by re-census that no sandbox rows
  survive and real principals' counts are unchanged.
- `backfill-explicit`: normalize provenance on API-written rows.
  Payload-only patches (source variants to `explicit`, missing
  speaker/confidence to the server-side defaults), verified by
  re-scan; idempotent once clean.

Commands that modify the store require an explicit `--yes` flag. When
`--yes` is absent, the command prints the action it WOULD take and
exits with status 2 so automation can detect "not authorized" distinct
from "success". Prompting interactively was considered and rejected:
ops scripts and cron invocations should either set `--yes` upfront or
fail fast on missing authorization, rather than blocking on stdin.

The broader `delete_all(user_id=...)` primitive exists in
`src/kai/memory.py`. The only exposure here is the constrained one
inside `purge-sandbox`: identities are discovered by census, gated on
the `sandbox-` prefix, and verified by re-census afterwards. An
unconstrained wipe of an arbitrary user_id remains intentionally NOT
exposed; if a future incident requires that, add it as a separate
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
import sqlite3
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kai.config import Config

log = logging.getLogger(__name__)


def _default_human_report_directory(config: Config, user_id: str, report_name: str) -> Path:
    """Resolve admin artifacts under the canonical Workshop human home."""
    from kai.config import DATA_DIR

    db_path_value = getattr(config, "session_db_path", None)
    if not isinstance(db_path_value, (str, Path)):
        return DATA_DIR / "home" / user_id / "docs" / report_name
    db_path = Path(db_path_value)
    if db_path.is_symlink() or not db_path.is_file():
        return DATA_DIR / "home" / user_id / "docs" / report_name
    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        tables = {
            str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        if "external_identities" not in tables:
            return DATA_DIR / "home" / user_id / "docs" / report_name
        rows = connection.execute(
            "SELECT principal_id FROM external_identities WHERE provider = 'telegram' AND external_subject = ?",
            (user_id,),
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != 1:
        # Unmapped user (zero rows) or ambiguous mapping (several):
        # fall back to the literal-user-id path exactly like every
        # other unresolvable branch above. This helper picks a
        # DEFAULT artifact directory; failing the whole admin command
        # over a missing identity mapping punished the operator for a
        # condition --out-dir already handles.
        return DATA_DIR / "home" / user_id / "docs" / report_name
    return DATA_DIR / "home" / str(rows[0][0]) / "docs" / report_name


# ── Known source values ─────────────────────────────────────────────
#
# Accepted values for --source: every source the live system mints
# (mirrors memory.USER_VISIBLE_SOURCES; the literal copy exists
# because this module keeps kai.memory imports lazy so --help stays
# fast, and a test pins the two sets against drift) plus two
# purge-only extras: "user_raw" targets Track-1-era legacy rows that
# survive only as purge candidates, and the empty string covers
# LEGACY rows whose metadata dict has no "source" key at all
# (`delete_by_source(user_id, source="")` deliberately matches both
# the key-absent and key-empty cases, per memory.py).
#
# We enumerate explicitly so a typo ("user-raw" vs "user_raw") fails
# at arg-parse time rather than silently no-op'ing against every row.
_KNOWN_SOURCES = frozenset({"extracted", "episode", "migration", "explicit", "user_raw", ""})


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
        help="Memory owner key; protected runtime IDs resolve to their canonical principal.",
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

    ps = sub.add_parser(
        "purge-sandbox",
        help="Delete every eval-residue identity (user_id prefix 'sandbox-') from the store",
        description=(
            "Enumerate every owner identity in the collection whose "
            "user_id starts with 'sandbox-' (the eval-harness naming "
            "convention shared by the replay and backend-gate tools) "
            "and delete all of their rows. Real principals are listed "
            "in the plan and re-counted afterwards so the run verifies "
            "they were untouched."
        ),
    )
    ps.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion. Without this flag the command prints the planned deletions and exits with status 2.",
    )

    bx = sub.add_parser(
        "backfill-explicit",
        help="Normalize provenance on API-written rows (source variants, missing speaker/confidence)",
        description=(
            "Find every row whose source marks it as API-written (the "
            "canonical 'explicit' plus drifted variant strings) and "
            "patch it in place: source normalized to 'explicit', and "
            "speaker/confidence stamped with the server-side defaults "
            "('assistant', 0.9) where absent. Payload-only patches; no "
            "re-embedding, no content changes."
        ),
    )
    bx.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the patches. Without this flag the command prints the plan and exits with status 2.",
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
        help="Memory owner key; protected runtime IDs resolve to their canonical principal.",
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

    review = sub.add_parser(
        "review-legacy-scope",
        help="Census and explicitly dispose every residual legacy-default row",
        description=(
            "Export every remaining legacy-default memory row into a protected "
            "JSONL review manifest. Replace every review_required disposition "
            "with global, project, delete, or quarantine. --apply requires a "
            "fresh complete manifest and dumps pre-images before any write. "
            "--rollback restores metadata changes guarded by the review id. "
            "The daemon must be stopped."
        ),
    )
    review.add_argument(
        "user_id",
        help="Memory owner key; protected runtime IDs resolve to their canonical principal.",
    )
    review.add_argument(
        "--out-dir",
        dest="out_dir",
        default=None,
        help="Artifact directory. Default: <DATA_DIR>/home/<principal>/docs/legacy-scope-review/.",
    )
    review_mode = review.add_mutually_exclusive_group()
    review_mode.add_argument(
        "--apply",
        metavar="MANIFEST",
        default=None,
        help="Apply a complete operator-reviewed manifest (requires --yes).",
    )
    review_mode.add_argument(
        "--rollback",
        metavar="PREIMAGES",
        default=None,
        help="Restore metadata changes from a pre-image file (requires --yes).",
    )
    review.add_argument(
        "--yes",
        action="store_true",
        help="Confirm a mutating mode. Without it, --apply/--rollback print the plan and exit with status 2.",
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
        help="Memory owner key; protected runtime IDs resolve to their canonical principal.",
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
    # Pass 2 controls. `--no-assistant-pass` uses store_const + default=None
    # so the mutating-mode rejection below can distinguish "explicitly
    # passed" from "default in effect"; without it, the rejection's
    # `if value is not None` filter would always read False on a
    # boolean store_true flag and the flag would be silently
    # tolerated under --apply/--rollback.
    bp.add_argument(
        "--no-assistant-pass",
        dest="no_assistant_pass",
        action="store_const",
        const=True,
        default=None,
        help=("Disable Pass 2 (assistant-turn matching). Dry-run only; apply consumes the proposals file verbatim."),
    )
    bp.add_argument(
        "--assistant-max-user-gap-seconds",
        dest="assistant_max_user_gap_seconds",
        type=int,
        default=None,
        help=("Pass 2: maximum seconds between the paired user turn and the winning assistant turn. Default: 600."),
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


async def _run_and_close_sessions(coro):
    """Await a reclassify entrypoint, then close the session DB.

    `load_project_registry` (called by the dry-run and apply
    entrypoints) opens the session DB the way daemon startup does,
    and each aiosqlite connection runs a NON-DAEMON worker thread
    that exits only when the connection is closed. The daemon closes
    it in main.py's shutdown path; a CLI process has no shutdown
    path, so the leaked worker thread pinned interpreter exit and
    every reclassify invocation hung forever after finishing its
    work. Closing here, at the process-lifetime boundary and inside
    the same event loop that opened the connection (aiosqlite
    futures are loop-bound, so a second asyncio.run cannot close
    it), is what lets the process exit.

    close_db() no-ops when nothing was opened, so wrapping the
    rollback entrypoint (which never loads the registry) is safe
    and keeps the three dispatch sites uniform.
    """
    from kai import sessions

    try:
        return await coro
    finally:
        await sessions.close_db()


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
        from kai.memory import init_offline_memory, is_enabled

        config = load_config()
        init_offline_memory(config)
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
        print(
            f"memory admin: init failed: {e}. If canonical memory receipts are incomplete, "
            "start Kai once with memory enabled so the protected migration can finish.",
            file=sys.stderr,
        )
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


def _cmd_purge_sandbox(args: argparse.Namespace) -> int:
    """Execute the `purge-sandbox` subcommand. Returns a process exit code.

    Deletion goes through the existing `delete_all` primitive per
    identity. That primitive swallows provider exceptions by design
    (it serves the bot's forget-all path), so success is verified by
    re-running the owner census afterwards rather than by trusting
    the calls: any surviving sandbox row, or any change to a real
    principal's count, fails the run with exit 1.
    """
    if not _initialize_memory():
        return 1

    from kai.memory import count_points_by_owner, delete_all

    try:
        before = count_points_by_owner()
    except Exception as e:
        print(f"memory admin: owner census failed: {e}", file=sys.stderr)
        return 1

    sandbox = {uid: n for uid, n in sorted(before.items()) if uid.startswith("sandbox-")}
    real = {uid: n for uid, n in sorted(before.items()) if not uid.startswith("sandbox-")}

    if not sandbox:
        print("memory admin: no sandbox-prefixed identities in the store; nothing to purge.")
        return 0

    print(f"memory admin: {sum(sandbox.values())} row(s) across {len(sandbox)} sandbox identit(ies):")
    for uid, n in sandbox.items():
        print(f"  {uid}: {n}")
    print(f"memory admin: {len(real)} real principal(s) will be left untouched:")
    for uid, n in real.items():
        print(f"  {uid}: {n}")

    if not args.yes:
        # Dry-run path; exit 2 mirrors `purge` (distinct from success
        # and error so automation can tell them apart).
        print("memory admin: re-run with --yes to execute.")
        return 2

    for uid in sandbox:
        delete_all(user_id=uid)

    try:
        after = count_points_by_owner()
    except Exception as e:
        print(f"memory admin: post-purge census failed: {e}", file=sys.stderr)
        return 1

    leftover = {uid: n for uid, n in sorted(after.items()) if uid.startswith("sandbox-")}
    real_after = {uid: n for uid, n in sorted(after.items()) if not uid.startswith("sandbox-")}
    if leftover:
        print(f"memory admin: purge incomplete; sandbox rows remain: {leftover}", file=sys.stderr)
        return 1
    if real_after != real:
        print(
            f"memory admin: real principal counts changed during the purge: before={real} after={real_after}",
            file=sys.stderr,
        )
        return 1

    print(f"memory admin: deleted {sum(sandbox.values())} row(s); real principal counts unchanged.")
    return 0


def _cmd_backfill_explicit(args: argparse.Namespace) -> int:
    """Execute the `backfill-explicit` subcommand. Returns an exit code.

    Patch construction is per-row: source is normalized only when it
    differs from 'explicit', and speaker/confidence are stamped only
    when absent, so re-running is a no-op once the corpus is clean.
    Patches go through the payload-only path (no re-embed, no Mem0
    update bookkeeping) because a provenance repair is not a semantic
    edit; success is verified by re-scanning, which must find no row
    still carrying a variant source or missing either stamped field.
    """
    if not _initialize_memory():
        return 1

    from kai.memory import list_api_written_points, patch_point_payload

    try:
        points = list_api_written_points()
    except Exception as e:
        print(f"memory admin: source scan failed: {e}", file=sys.stderr)
        return 1

    def _patch_for(payload: dict[str, object]) -> dict[str, object]:
        patch: dict[str, object] = {}
        if payload.get("source") != "explicit":
            patch["source"] = "explicit"
        if "speaker" not in payload:
            patch["speaker"] = "assistant"
        if "confidence" not in payload:
            patch["confidence"] = 0.9
        return patch

    todo = [(pid, payload, _patch_for(payload)) for pid, payload in points]
    todo = [(pid, payload, patch) for pid, payload, patch in todo if patch]

    if not todo:
        print(f"memory admin: {len(points)} API-written row(s), all fully stamped; nothing to backfill.")
        return 0

    normalize = sum(1 for _, _, patch in todo if "source" in patch)
    speaker = sum(1 for _, _, patch in todo if "speaker" in patch)
    confidence = sum(1 for _, _, patch in todo if "confidence" in patch)
    print(
        f"memory admin: {len(todo)} of {len(points)} API-written row(s) need patches: "
        f"{normalize} source normalization(s), {speaker} missing speaker(s), "
        f"{confidence} missing confidence value(s)."
    )
    for pid, payload, patch in todo:
        print(f"  {pid} (owner {payload.get('user_id', '?')}, source {payload.get('source')!r}): {patch}")

    if not args.yes:
        # Dry-run path; exit 2 mirrors the sibling subcommands.
        print("memory admin: re-run with --yes to execute.")
        return 2

    for pid, _, patch in todo:
        try:
            patch_point_payload(point_id=pid, payload=patch)
        except Exception as e:
            print(f"memory admin: patch failed for {pid}: {e}", file=sys.stderr)
            return 1

    try:
        remaining = [(pid, payload) for pid, payload in list_api_written_points() if _patch_for(payload)]
    except Exception as e:
        print(f"memory admin: post-backfill scan failed: {e}", file=sys.stderr)
        return 1
    if remaining:
        print(f"memory admin: backfill incomplete; rows still need patches: {remaining}", file=sys.stderr)
        return 1

    print(f"memory admin: patched {len(todo)} row(s); re-scan clean.")
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

    from kai import memory_reclassify

    out_dir = (
        Path(args.out_dir) if args.out_dir else _default_human_report_directory(config, args.user_id, "reclassify")
    )

    if not mutating:
        return asyncio.run(
            _run_and_close_sessions(
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
            _run_and_close_sessions(
                memory_reclassify.run_apply(config, args.user_id, proposals_path=artifact_path, out_dir=out_dir)
            )
        )
    return asyncio.run(
        _run_and_close_sessions(memory_reclassify.run_rollback(config, args.user_id, preimages_path=artifact_path))
    )


def _cmd_review_legacy_scope(args: argparse.Namespace) -> int:
    """Execute the complete residual legacy-scope review workflow."""
    config = _initialize_memory()
    if config is None:
        return 1
    from kai import memory_scope_review

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else _default_human_report_directory(config, args.user_id, "legacy-scope-review")
    )
    artifact_value = args.apply if args.apply is not None else args.rollback
    if artifact_value is None:
        return asyncio.run(
            _run_and_close_sessions(memory_scope_review.run_census(config, args.user_id, out_dir=out_dir))
        )

    artifact_path = Path(artifact_value)
    row_type = "review manifest" if args.apply is not None else "pre-image file"
    try:
        text = artifact_path.read_text(encoding="utf-8")
        if args.apply is not None:
            header, rows = memory_scope_review.parse_manifest(text, user_id=args.user_id)
            pending = [row for row in rows if row.disposition == memory_scope_review.DISPOSITION_REVIEW_REQUIRED]
            if pending:
                print(
                    f"memory admin: {len(pending)} row(s) still have disposition review_required; "
                    "complete the manifest before --apply.",
                    file=sys.stderr,
                )
                return 1
        else:
            header, rows = memory_scope_review.parse_preimages(text, user_id=args.user_id)
    except (OSError, ValueError) as exc:
        print(f"memory admin: cannot read {row_type}: {exc}", file=sys.stderr)
        return 1
    if not args.yes:
        verb = "apply" if args.apply is not None else "roll back"
        print(
            f"memory admin: would {verb} {len(rows)} row(s) from review "
            f"{header.get('review_id')} for user {args.user_id}."
        )
        print("memory admin: re-run with --yes to execute.")
        return 2
    if args.apply is not None:
        return asyncio.run(
            _run_and_close_sessions(
                memory_scope_review.run_apply(
                    config,
                    args.user_id,
                    manifest_path=artifact_path,
                    out_dir=out_dir,
                )
            )
        )
    return asyncio.run(
        _run_and_close_sessions(
            memory_scope_review.run_rollback(
                config,
                args.user_id,
                preimages_path=artifact_path,
            )
        )
    )


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
                ("--no-assistant-pass", args.no_assistant_pass),
                ("--assistant-max-user-gap-seconds", args.assistant_max_user_gap_seconds),
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
    # Pass 2 settings. `assistant_pass_enabled` defaults to True; the
    # `--no-assistant-pass` flag uses store_const + default=None so an
    # explicit pass flips this to False, and the absent flag leaves
    # Pass 2 on.
    assistant_pass_enabled = args.no_assistant_pass is None
    assistant_max_user_gap_seconds = (
        args.assistant_max_user_gap_seconds
        if args.assistant_max_user_gap_seconds is not None
        else memory_provenance_backfill._DEFAULT_ASSISTANT_MAX_USER_GAP_SECONDS
    )

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
    if assistant_max_user_gap_seconds < 1:
        print(
            f"memory admin: --assistant-max-user-gap-seconds must be a positive int, "
            f"got {assistant_max_user_gap_seconds}",
            file=sys.stderr,
        )
        return 2
    if sample < 0:
        print(f"memory admin: --sample must be >= 0, got {sample}", file=sys.stderr)
        return 2

    config = _initialize_memory()
    if config is None:
        return 1

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else _default_human_report_directory(config, args.user_id, "backfill-provenance")
    )

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
                assistant_pass_enabled=assistant_pass_enabled,
                assistant_max_user_gap_seconds=assistant_max_user_gap_seconds,
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
    if args.command == "purge-sandbox":
        sys.exit(_cmd_purge_sandbox(args))
    if args.command == "backfill-explicit":
        sys.exit(_cmd_backfill_explicit(args))
    if args.command == "reclassify-scope":
        sys.exit(_cmd_reclassify(args))
    if args.command == "review-legacy-scope":
        sys.exit(_cmd_review_legacy_scope(args))
    if args.command == "backfill-provenance":
        sys.exit(_cmd_backfill_provenance(args))
    # argparse's required=True on the subparsers guarantees a known
    # command reaches this point, so the else branch is unreachable
    # under normal invocation. Guarded anyway in case a future
    # subcommand is added without a dispatch entry.
    parser.error(f"unhandled command: {args.command}")
