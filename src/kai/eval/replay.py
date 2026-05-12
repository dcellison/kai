"""
Replay extraction over a chat-history window into a sandbox `user_id`.

The bot extracts inline as each USER+ASSISTANT pair arrives via webhook;
there is no built-in path for re-extracting an older history window
under a different prompt version. This module supplies that path so a
spec-implementation cycle can capture pre-PR and post-PR sandbox fact
sets for comparison without disturbing production Qdrant state.

Production semantics this module preserves:

- USER+ASSISTANT pair as the unit of advance (matching `bot.py:3707-3727`).
- Up to `episode_classifier_context_turns` prior pairs rendered as
  PRIOR CONTEXT (default 3 from `config.py:530`).
- Prior-turn character caps `_PRIOR_USER_CHARS = 800` /
  `_PRIOR_ASSISTANT_CHARS = 1200` (`memory_extraction.py:137-138`),
  inherited automatically because the replay calls
  `extract_and_store`, which internally invokes
  `_build_extraction_payload`.
- CONSOLIDATION (intent: new / update_of / skip_redundant) against
  facts already in the sandbox `user_id`. The accumulating sandbox
  fact set rebuilds the production semantic that an existing operator
  account would carry into each new pair's extraction.

What this module deliberately does NOT do:

- Run against a real chat_id. The `--user-id` argument MUST start with
  the literal `sandbox-` prefix; any other value raises before any
  extraction or storage call. Defense in depth: a typo or misuse
  cannot write replayed facts to a real user's memory store.
- Read the prompt from a flag. The prompt version follows the source
  tree state: to replay with v8, check out the pre-swap commit; to
  replay with v9, check out the post-swap commit. Tying prompt
  version to source state matches the production semantic (no
  per-run prompt selection in the live extractor).

Pairing semantics are inherited from `history._pair_records_chronologically`
so the replay's input to `extract_and_store` is byte-equivalent to what
production fed the extractor on the same conversation, modulo
sandbox-vs-production accumulating-fact-set differences (both v8 and
v9 sandbox replays start from an empty fact set, so the
sandbox-vs-sandbox comparison controls for that on both sides).

Cleanup: `--reset` calls `memory.delete_all(user_id=<sandbox-id>)` before
the replay starts so reruns are idempotent and a stale partial run does
not contaminate the next baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# Sandbox prefix enforced on the `--user-id` flag. The whole point of
# isolation here is keeping replay facts out of real users' stores;
# enforcing the prefix structurally (rather than via documentation
# alone) closes the typo / misuse path. Mirrors the convention used in
# the spec's §3.3 sandbox example (`sandbox-464`).
_SANDBOX_USER_ID_PREFIX = "sandbox-"


def _parse_date(raw: str) -> date:
    """Parse YYYY-MM-DD; raise argparse-friendly error on bad shape."""
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as exc:
        # argparse catches this and surfaces the message verbatim.
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {raw!r}: {exc}") from exc


def _build_parser() -> argparse.ArgumentParser:
    """Argparse setup. Documented surface; see module docstring for rationale."""
    parser = argparse.ArgumentParser(
        prog="python -m kai.eval.replay",
        description=(
            "Replay extraction over a chat-history window into a sandbox "
            "user_id. See module docstring for production-semantic guarantees."
        ),
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing chat-history JSONL files for the source "
            "chat_id. Defaults to <DATA_DIR>/history/<chat_id>/."
        ),
    )
    parser.add_argument(
        "--chat-id",
        type=int,
        required=True,
        help=(
            "Source chat_id whose history is being replayed. Records "
            "belonging to a different chat_id (or no chat_id) are dropped, "
            "matching production's `get_recent_pairs` strictness."
        ),
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help=(
            "Sandbox user_id to write extracted facts to. MUST start with "
            f"{_SANDBOX_USER_ID_PREFIX!r}; any other value raises before any "
            "extraction or storage call."
        ),
    )
    parser.add_argument(
        "--start-date",
        type=_parse_date,
        default=None,
        help=("First history file to include (YYYY-MM-DD, inclusive). Defaults to the oldest file present."),
    )
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        default=None,
        help=("Last history file to include (YYYY-MM-DD, inclusive). Defaults to the newest file present."),
    )
    parser.add_argument(
        "--context-turns",
        type=int,
        default=None,
        help=(
            "Number of prior USER+ASSISTANT pairs to include as PRIOR "
            "CONTEXT. Defaults to config.episode_classifier_context_turns "
            "(currently 3)."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete the sandbox user_id's existing facts before replay "
            "starts. Idempotent reset for clean baseline reruns."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Walk history and report the pair count and a sample of "
            "payload shapes without calling the extractor or writing to "
            "the store."
        ),
    )
    return parser


def _validate_user_id(user_id: str) -> None:
    """Sandbox prefix guard. Raises ValueError; argparse-side fallthrough."""
    if not user_id.startswith(_SANDBOX_USER_ID_PREFIX):
        raise ValueError(
            f"--user-id must start with {_SANDBOX_USER_ID_PREFIX!r} to "
            f"prevent accidental writes to a real user store; got {user_id!r}"
        )


def _iter_history_files(
    history_dir: Path,
    start_date: date | None,
    end_date: date | None,
) -> list[Path]:
    """
    Return JSONL files in [start_date, end_date] inclusive, oldest first.

    Production history files are named `YYYY-MM-DD.jsonl` (one file per
    UTC day per chat_id). Files with non-matching names (e.g., a stray
    backup or rotated artifact) are skipped silently so a misplaced
    file does not abort the replay run.

    Sorting oldest-first matches the natural reading order of a replay:
    PRIOR CONTEXT for pair N is the previous N-1 pairs in chronological
    sequence, which only stays correct when files are processed in
    ascending date order.
    """
    if not history_dir.is_dir():
        return []
    selected: list[tuple[date, Path]] = []
    for path in history_dir.glob("*.jsonl"):
        try:
            file_date = datetime.strptime(path.stem, "%Y-%m-%d").date()
        except ValueError:
            # Non-date-named file; skip silently. The bot only writes
            # date-stamped files, so anything else is an operator-side
            # artifact (backup, manual edit) that the replay should not
            # touch.
            continue
        if start_date is not None and file_date < start_date:
            continue
        if end_date is not None and file_date > end_date:
            continue
        selected.append((file_date, path))
    selected.sort(key=lambda t: t[0])
    return [p for _d, p in selected]


def _load_records(
    file_paths: list[Path],
    chat_id: int,
) -> list[dict[str, Any]]:
    """
    Read JSONL records across files, filtered by chat_id and content.

    Filters mirror `history._pair_records_chronologically`'s preconditions
    so the resulting pair stream is byte-equivalent to what production
    would have built from the same records. Reusing the production
    pairing helper (imported at call site) on this filtered record list
    keeps the semantics in one place.

    Returns a flat list of records (each is a dict with `dir` and
    `text` keys) in chronological order across all files. Pairing is
    handled by the production helper; this function only does the
    filter pass.
    """
    # Import inside the function so the test suite can monkeypatch the
    # underlying constants (e.g., `_SYNTHETIC_ASSISTANT_MARKERS`) before
    # this module is loaded. Top-level imports would freeze them at
    # import time.
    from kai.history import _SYNTHETIC_ASSISTANT_MARKERS

    records: list[dict[str, Any]] = []
    for path in file_paths:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            log.warning("Skipping unreadable history file %s: %s", path, exc)
            continue
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Skip individual bad lines rather than abort the file,
                # matching `history.get_recent_pairs`'s tolerance for
                # partial corruption from interrupted writes.
                continue
            if "chat_id" not in rec or rec.get("chat_id") != chat_id:
                continue
            text = (rec.get("text") or "").strip()
            if not text:
                continue
            direction = rec.get("dir")
            if direction == "assistant" and _SYNTHETIC_ASSISTANT_MARKERS.fullmatch(text):
                # Failure-path placeholders from /stop, empty results,
                # or error paths in bot.py. Pairing them into a windowed
                # payload would feed the classifier a "botched exchange"
                # prior context that distorts results.
                continue
            records.append({"dir": direction, "text": text})
    return records


def _build_pairs(records: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Run records through the production pairing helper."""
    # Imported here (not at module top) so the test suite's pairing
    # tests can monkeypatch the helper independently of the replay
    # module's other functions. Mirrors `_load_records`'s late-import
    # for the synthetic-marker regex.
    from kai.history import _pair_records_chronologically

    return _pair_records_chronologically(records)


async def _run_replay(
    pairs: list[tuple[str, str]],
    *,
    user_id: str,
    context_turns: int,
    config: Any,
    dry_run: bool,
) -> dict[str, int]:
    """
    Walk pairs, maintain rolling prior buffer, call `extract_and_store`.

    Returns a counters dict for the summary print. Counts are: pairs
    processed, facts stored (sum of `extract_and_store` returns;
    `extract_and_store` already buckets stored / replaced / skipped
    internally but exposes only the stored count to callers, which is
    what we report).
    """
    # Late import so the test suite can monkeypatch `extract_and_store`
    # at the call site without touching the replay module's top-level
    # imports. Tests that exercise the replay's pairing logic do not
    # need a real `extract_and_store` resolved.
    from kai.memory_extraction import extract_and_store

    counters: dict[str, int] = {"pairs_processed": 0, "facts_stored": 0}
    # Rolling prior buffer. The current pair is NOT included; only the
    # previous `context_turns` pairs are passed as PRIOR CONTEXT, which
    # matches production semantics where `bot.py`'s `_ingest_memory`
    # drops the in-flight pair before threading prior_pairs into
    # `extract_and_store`.
    prior: list[tuple[str, str]] = []
    for user_text, assistant_text in pairs:
        if dry_run:
            counters["pairs_processed"] += 1
            prior.append((user_text, assistant_text))
            if len(prior) > context_turns:
                prior.pop(0)
            continue
        # Pass a SHALLOW COPY of `prior` so a future change inside
        # `extract_and_store` that mutates its `prior_pairs` argument
        # cannot corrupt the rolling buffer on the next iteration.
        # extract_and_store does not mutate today, but the contract is
        # not documented as immutable.
        stored = await extract_and_store(
            user_text,
            assistant_text,
            user_id=user_id,
            config=config,
            prior_pairs=list(prior),
        )
        counters["pairs_processed"] += 1
        counters["facts_stored"] += stored
        prior.append((user_text, assistant_text))
        if len(prior) > context_turns:
            # Drop the oldest pair to maintain the rolling window. The
            # window is bounded so older pairs cannot indirectly survive
            # into a future iteration's PRIOR CONTEXT via the buffer.
            prior.pop(0)
    return counters


async def _reset_sandbox(user_id: str) -> None:
    """Delete all facts for the sandbox `user_id`. No-op safe."""
    # Late import to keep replay-module import cheap and to let tests
    # monkeypatch the deletion path independently of the rest of the
    # module.
    from kai.memory import delete_all

    delete_all(user_id=user_id)


def _format_summary(counters: dict[str, int]) -> str:
    """Operator-readable run summary."""
    return f"replay summary: pairs_processed={counters['pairs_processed']}, facts_stored={counters['facts_stored']}"


async def _async_main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        _validate_user_id(args.user_id)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Resolve history_dir and config_turns from production config when
    # not supplied. Loaded here (not at top) so a test can run without
    # /etc/kai/env or a populated DATA_DIR.
    from kai.config import DATA_DIR, load_config

    config = load_config()
    history_dir: Path = args.history_dir or (DATA_DIR / "history" / str(args.chat_id))
    context_turns: int = (
        args.context_turns if args.context_turns is not None else config.episode_classifier_context_turns
    )

    if context_turns < 0:
        print(
            f"error: --context-turns must be non-negative, got {context_turns}",
            file=sys.stderr,
        )
        return 2

    file_paths = _iter_history_files(history_dir, args.start_date, args.end_date)
    if not file_paths:
        print(
            f"warning: no history files found in {history_dir} for the given date range; nothing to replay",
            file=sys.stderr,
        )

    records = _load_records(file_paths, args.chat_id)
    pairs = _build_pairs(records)

    if args.reset and not args.dry_run:
        # Reset only on a live run. A dry-run that destroys sandbox
        # state would be a footgun: the operator types `--dry-run` to
        # inspect, and any side effect violates the contract.
        await _reset_sandbox(args.user_id)

    counters = await _run_replay(
        pairs,
        user_id=args.user_id,
        context_turns=context_turns,
        config=config,
        dry_run=args.dry_run,
    )
    print(_format_summary(counters))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Sync entry point for `python -m kai.eval.replay`."""
    return asyncio.run(_async_main(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
