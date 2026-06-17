"""Mode-switch verification harness for the dual-mode memory subsystem.

The MEMORY_ENABLED toggle promises that exactly one fact surface is
active per mode: in disabled mode, MEMORY.md drives behavior and Qdrant
is unreached; in enabled mode, Qdrant drives behavior and MEMORY.md is
dormant. This module verifies that promise without requiring a service
restart by importing the production code paths
(`build_session_context`, `format_context`) and asserting the gating
contract directly.

Two subcommands:

    python -m kai.eval.modeswitch verify
        Black-box logic verification. Imports the production code and
        asserts every gating invariant under both flag values. Uses an
        isolated tmp-dir Qdrant store so the live memory directory at
        $DATA_DIR/memory/qdrant is never touched. Exits 0 on full
        pass, 1 on any failure.

    python -m kai.eval.modeswitch check
        Runtime sanity check against the running service. Reports the
        live effective mode plus the deployed prompt versions. Reads
        WEBHOOK_SECRET from the process environment; source
        /etc/kai/env first under sudo:

            sudo bash -c 'source /etc/kai/env && env | grep WEBHOOK_SECRET'

        Exit codes: 0 on a clean read; 2 if WEBHOOK_SECRET is not in
        the environment; 1 if /health is down or /api/memory/stats
        returns an unexpected status.

The verify subcommand is the merge gate. The check subcommand is
informational and can be re-run as production state changes.

Operator round-trip (manual; sudo-gated; not automated by this script):

    1. operator: sudo edit /etc/kai/env, set MEMORY_ENABLED=false
    2. operator restarts the service:
       macOS: sudo launchctl bootout system/com.syrinx.kai
              sudo launchctl bootstrap system /Library/LaunchDaemons/com.syrinx.kai.plist
       Linux: sudo systemctl stop kai && sudo systemctl start kai
    3. source /etc/kai/env  # pull WEBHOOK_SECRET into the shell env
       python -m kai.eval.modeswitch check    # confirms disabled
    4. python -m kai.eval.modeswitch verify   # logic invariants under both flags
    5. operator: sudo edit /etc/kai/env, set MEMORY_ENABLED=true
    6. operator restarts the service (same commands as step 2)
    7. python -m kai.eval.modeswitch check    # confirms enabled

Step 4's verify is mode-independent: it asserts both flag-value
behaviors regardless of the running service's flag, so running it
once anywhere in the procedure is sufficient. Steps 3 and 7 are the
only checks that observe the live service state.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from kai.backend import ApiContext, build_session_context
from kai.config import Config

log = logging.getLogger(__name__)


# ── Verify: invariant assertions ────────────────────────────────────


@dataclass
class _InvariantResult:
    """Outcome of one invariant check.

    `name` is the human-readable assertion label printed in the
    report. `passed` is the binary outcome. `detail` carries a
    short explanation on failure (a substring that was present
    when it should not have been, or a substring that was missing
    when it should have been). On pass, `detail` is the empty
    string.
    """

    name: str
    passed: bool
    detail: str


# Substring markers from `build_session_context` (the [Memory subsystem:
# enabled/disabled] line) and from `memory.format_context` (the
# [Relevant memories from past conversations ...] header). These are
# the load-bearing strings the harness asserts on.
#
# `_MARKER_PERSISTENT_MEMORY` matches the `[Your persistent memory
# (file: ...):]` block that build_session_context emits ONLY in
# disabled mode. The match is a leading-substring check rather than
# a full-line match so the test does not have to predict the
# tmp-path part of the format string.
#
# `_MARKER_RELEVANT_MEMORIES` matches the recall-block header that
# format_context emits when search returns non-empty after the
# floor and budget filters. Same leading-substring approach.
_MARKER_DISABLED = "[Memory subsystem: disabled]"
_MARKER_ENABLED = "[Memory subsystem: enabled]"
_MARKER_PERSISTENT_MEMORY = "[Your persistent memory (file:"
_MARKER_RELEVANT_MEMORIES = "[Relevant memories from past conversations"


# Minimal MEMORY.md fixture used by the verify subcommand. A single
# truthy line is enough to drive build_session_context's read branch
# (memory_path.read_text().strip() returns truthy), avoiding the
# "(currently empty)" / "(not yet created)" placeholder branches.
# The content is deliberately bland so it cannot be mistaken for
# operator data if the file ever leaks via test logs.
_FIXTURE_MEMORY_MD = "Test fixture content for mode-switch verification.\n"


# Test fixture chat_id used to scope the temporary MEMORY.md and the
# Qdrant user_id for the seeded fact. A small positive integer (not
# a real chat_id) is used to make it visually obvious in any
# diagnostic output that this is a test value, not production data.
_FIXTURE_CHAT_ID = 1


# Test fixture query string that is guaranteed to share enough
# semantic surface with the seeded fact to land above the relevance
# floor. The seeded fact mentions "test fixture" and "mode-switch";
# the query mentions both, so the embedder produces a high cosine.
_FIXTURE_QUERY = "test fixture for mode-switch verification"


# Seeded fact content. The recall-path invariant the harness asserts
# is the format_context OUTPUT SHAPE (empty string or starts with
# the recall-block prefix); content is not asserted programmatically.
# The real isolation guarantee is `_isolated_data_dir`'s `DATA_DIR`
# patch + the `MEM0_DIR` env redirect at the top of `_run_verify`.
# This string is deliberately distinctive so that if a future
# diagnostic wants to scan the recall output for evidence the
# isolated store is being hit (rather than the production one), the
# substring is unique enough to find without false positives.
_FIXTURE_SEED_FACT = "Mode-switch test fixture marker phrase 7f3a2b1c"


@contextmanager
def _isolated_data_dir(tmp_root: Path) -> Iterator[Path]:
    """Patch `kai.memory.DATA_DIR` to `tmp_root` for the duration of
    the with-block, reset memory module singleton state on exit.

    Mirrors the existing storage-redirect pattern in
    `tests/test_memory.py` (the per-test `with patch("kai.memory.DATA_DIR", tmp_path)`
    block in the integration tests), plus the singleton-state
    teardown in that file's `_reset_memory_module` helper
    (`_memory = None; _config = None`) so a subsequent invocation in
    the same process is not affected by the prior init_memory call.

    init_memory reads `DATA_DIR / "memory" / "qdrant"` from
    kai.memory at module-resolution time, with no parameter to
    redirect storage. Patching the module attribute is the
    supported test pattern.
    """
    from kai import memory as memory_module

    with patch.object(memory_module, "DATA_DIR", tmp_root):
        try:
            yield tmp_root
        finally:
            # Reset singleton state so a follow-up init_memory call
            # (in the next test, or the next subcommand invocation)
            # constructs a fresh client against its own DATA_DIR.
            memory_module._memory = None
            memory_module._config = None


@contextmanager
def _capture_recall_logs() -> Iterator[list[logging.LogRecord]]:
    """Capture every LogRecord emitted to the `kai.memory` logger
    during the with-block and yield the (initially empty) list to
    the caller. The list is populated as records arrive and is
    safe to inspect after the with-block exits.

    Why a manual `logging.Handler` rather than pytest's `caplog`
    fixture: the harness runs from a CLI entry point, NOT under
    pytest, so the fixture is unavailable. The pytest test pair at
    `tests/test_eval_modeswitch.py::TestRecallReasonField` uses
    `caplog` (the idiomatic pytest tool); the two surfaces capture
    the same records via different mechanisms, both correct for
    their context.

    Records are appended in the order they arrive. The harness's
    callers only inspect the most-recent record (a single
    `format_context` call is made per use of this contextmanager),
    but the list keeps prior records too in case future callers
    want to assert on multi-call sequences.

    Each fresh use creates a new `_ListHandler` instance and a new
    list, so a residual handler from a prior run cannot pollute
    THIS run's records. Removal in the `finally` block guards
    against handler leaks across calls.
    """
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler()
    handler.setLevel(logging.INFO)
    logger = logging.getLogger("kai.memory")
    # Save and lower the logger level so the `memory.recall` line
    # (emitted via `log.info(...)`) is forwarded to our handler
    # regardless of the harness's outer logging configuration. The
    # CLI's `logging.basicConfig(level=logging.WARNING, ...)` would
    # otherwise filter INFO-level records out before they reach the
    # handler.
    #
    # Disable propagation while the handler is attached so the
    # captured records do not also fan out to the root logger. The
    # CLI entry point's `basicConfig(level=WARNING)` already filters
    # them, but during pytest runs the root logger may be configured
    # at INFO and the records would appear duplicated in console
    # output. Restored in `finally` so other code paths that rely on
    # propagation are not affected after the with-block.
    prior_level = logger.level
    prior_propagate = logger.propagate
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prior_level)
        logger.propagate = prior_propagate


def _parse_recall_reason(records: list[logging.LogRecord]) -> str | None:
    """Pull the `reason` field from the most-recent `memory.recall`
    log record in `records`. Returns None if no such record exists,
    if the record's payload is malformed, or if the payload's
    `reason` field is absent.

    The payload format is fixed at the producer
    (`memory._emit_recall_log`): a compact JSON object preceded by
    the literal prefix `"memory.recall "`. Splitting on the first
    space and `json.loads`-decoding the tail is the inverse of the
    producer's `log.info("memory.recall %s", json.dumps(payload, ...))`.

    Returns the value of the `reason` field cast to `str`, or None
    on any decode failure, shape mismatch, or success-path payload
    where `reason` is omitted entirely (per `_base_recall_payload`'s
    docstring: `reason` is set only on short-circuit lines).

    Caller-side interpretation differs by mode. The disabled-mode
    caller compares `parsed == _RECALL_REASON_DISABLED` and treats
    None as a failing outcome (it neither equals nor implies the
    expected reason string). The enabled-mode caller compares
    `parsed != _RECALL_REASON_DISABLED` and treats None as a
    PASSING outcome, since the success-path payload omits `reason`
    entirely and that absence is part of the documented contract.
    """
    prefix = "memory.recall "
    for record in reversed(records):
        msg = record.getMessage()
        if not msg.startswith(prefix):
            continue
        try:
            payload = json.loads(msg[len(prefix) :])
        except ValueError:
            # `json.JSONDecodeError` is a subclass of `ValueError`
            # since Python 3.5, so the bare `ValueError` covers
            # both the well-formed-but-invalid-JSON case and the
            # malformed-input case that JSONDecodeError raises.
            #
            # `continue` rather than `return None` so a single
            # malformed trailing record does not shadow earlier
            # well-formed records. Today's callers pass a list with
            # exactly one `memory.recall` record per capture block,
            # but a future multi-record caller would silently
            # observe `None` for "valid records exist but the most
            # recent is bad," which is the wrong shape. The final
            # `return None` outside the loop still handles the
            # no-records case.
            continue
        reason = payload.get("reason") if isinstance(payload, dict) else None
        return reason if isinstance(reason, str) else None
    return None


def _build_test_configs(memory_enabled_value: bool) -> Config:
    """Build a Config with the requested memory_enabled value plus
    enough surrounding fields to satisfy the production code paths
    that will read it.

    The verify subcommand never spawns subprocesses or hits the
    network, so the Telegram and webhook fields are placeholder
    strings. allowed_user_ids contains the fixture chat_id so the
    user-id scoping in build_session_context resolves the per-user
    MEMORY.md path correctly.
    """
    return Config(
        telegram_bot_token="modeswitch-test",
        allowed_user_ids={_FIXTURE_CHAT_ID},
        webhook_secret="modeswitch-test",
        memory_enabled=memory_enabled_value,
    )


def _seed_fixture_memory_md(tmp_root: Path) -> Path:
    """Create a per-user MEMORY.md fixture under tmp_root. Returns
    the path so the caller can pass it to build_session_context's
    data_dir argument.

    Layout matches the production scoping convention:
    `<data_dir>/memory/<chat_id>/MEMORY.md`. build_session_context
    will read this file when called with memory_enabled=False.
    """
    user_dir = tmp_root / "memory" / str(_FIXTURE_CHAT_ID)
    user_dir.mkdir(parents=True, exist_ok=True)
    memory_path = user_dir / "MEMORY.md"
    memory_path.write_text(_FIXTURE_MEMORY_MD)
    return memory_path


async def _seed_fixture_fact(user_id: str) -> None:
    """Write one canned fact to the active memory store via
    `add_structured`. Caller must have already called init_memory
    inside an `_isolated_data_dir` block so the write lands in
    the tmp Qdrant.

    Source role tag matches the production extracted-fact shape so
    the row passes the source-bucket weighting at retrieval time.
    """
    from kai import memory as memory_module

    # add_structured is sync (Mem0 is sync). Wrap in
    # run_in_executor so this coroutine stays cooperative even
    # though the harness runs it serially.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        lambda: memory_module.add_structured(
            content=_FIXTURE_SEED_FACT,
            user_id=user_id,
            memory_type="fact",
            tags=["fact"],
            metadata={"source": "extracted", "confidence": 0.9},
        ),
    )


def _api_ctx_for_verify() -> ApiContext:
    """Construct an ApiContext for the verify subcommand. The verify
    path does not exercise any API endpoints; the context is needed
    only to satisfy build_session_context's signature.
    """
    return ApiContext(
        webhook_port=8080,
        webhook_secret="modeswitch-test",
        services_info=[],
    )


async def _run_verify() -> int:
    """Drive every invariant under both flag values; print PASS/FAIL
    per invariant and a final summary. Return 0 on full pass, 1 on
    any failure.

    Each invariant is computed once and recorded as an
    `_InvariantResult`. Results are batched in a list and printed
    together after all invariants run (and after the
    `MEM0_DIR`/singleton-state restoration in the `finally` block).
    If an exception is raised mid-run, the operator sees the
    traceback without partial-progress output; the env-state
    restoration still fires via `finally`. Operators wanting per-
    invariant streaming should run individual unit tests in
    `tests/test_eval_modeswitch.py::TestVerifyInvariants` instead.
    """
    results: list[_InvariantResult] = []

    with tempfile.TemporaryDirectory(prefix="modeswitch-") as tmp_str:
        tmp_root = Path(tmp_str)
        # Redirect Mem0's home directory so the per-instance
        # `migrations_qdrant` and `mem0_history.db` it auto-creates
        # land under the tmp tree rather than the operator's
        # `~/.mem0/`. The live service holds the lock on the real
        # `~/.mem0/migrations_qdrant`; without this redirect, the
        # harness's `init_memory` call would deadlock against it.
        # Set BEFORE the first import of `kai.memory` (which
        # transitively imports mem0); mem0 reads `MEM0_DIR` at
        # module-import time, so setting it later than the first
        # import has no effect.
        #
        # Invariant guard: backend.py and config.py do not transitively
        # import kai.memory today; the first kai.memory import happens
        # inside `_isolated_data_dir`'s body, AFTER MEM0_DIR is set.
        # If a future refactor pulls kai.memory into the import graph
        # of either module, mem0 would have already captured the
        # ORIGINAL MEM0_DIR (operator's `~/.mem0/`) and the redirect
        # below has no effect, leaving the harness deadlocked against
        # the live `migrations_qdrant` lock. Fail loud BEFORE
        # mutating MEM0_DIR so a failure leaves the environment
        # untouched and the try/finally below is not responsible for
        # restoring state we never changed.
        #
        # `if/raise` rather than `assert`: this guard is safety-
        # critical (silent deadlock on failure), and `assert`
        # statements are stripped under `python -O` /
        # `PYTHONOPTIMIZE=1`. The bare `RuntimeError` cannot be
        # silently disabled.
        #
        # Reentrancy note: this guard also fires on a second call
        # to `_run_verify` in the same process, because
        # `_isolated_data_dir`'s lazy `from kai import memory`
        # leaves the module in `sys.modules` after the first call.
        # The CLI invokes verify once per process, so this is fine;
        # a future caller that wants to retry verify in-process
        # needs a fresh subprocess (or to drop the guard, but the
        # guard is the only thing preventing the deadlock).
        if "kai.memory" in sys.modules:
            raise RuntimeError(
                "kai.memory was imported before _run_verify could set "
                "MEM0_DIR; mem0's home-directory constant has already been "
                "captured at the operator's ~/.mem0/ path and the tmp "
                "redirect will not take effect. A backend.py or config.py "
                "import path now pulls kai.memory transitively; restore "
                "the lazy-import contract before re-running."
            )

        # Save/restore semantics: capture the prior value (or
        # absence) and restore on the way out so a subsequent
        # process-shared invocation does not inherit the now-deleted
        # tmp path. `MEM0_DIR_SENTINEL` distinguishes "was unset"
        # from "was empty string"; the former requires `pop`, the
        # latter requires reassignment.
        _MEM0_DIR_SENTINEL: object = object()
        prior_mem0_dir: str | object = os.environ.get("MEM0_DIR", _MEM0_DIR_SENTINEL)
        os.environ["MEM0_DIR"] = str(tmp_root / "mem0")

        try:
            _seed_fixture_memory_md(tmp_root)
            api_ctx = _api_ctx_for_verify()
            # workspace and home_workspace coincide so the optional
            # identity-injection branch in build_session_context (which
            # fires only when the two differ) does not pollute the
            # output and confuse the substring assertions.
            ws = tmp_root / "workspace"
            ws.mkdir()

            # ── Disabled-mode invariants ──────────────────────────
            disabled_ctx = build_session_context(
                workspace=ws,
                home_workspace=ws,
                api=api_ctx,
                workspace_config=None,
                chat_id=_FIXTURE_CHAT_ID,
                data_dir=tmp_root,
                memory_enabled=False,
            )

            results.append(
                _InvariantResult(
                    name="disabled: subsystem marker present",
                    passed=_MARKER_DISABLED in disabled_ctx,
                    detail=f"missing {_MARKER_DISABLED!r}" if _MARKER_DISABLED not in disabled_ctx else "",
                )
            )
            results.append(
                _InvariantResult(
                    name="disabled: persistent-memory block injected",
                    passed=_MARKER_PERSISTENT_MEMORY in disabled_ctx,
                    detail=f"missing {_MARKER_PERSISTENT_MEMORY!r}"
                    if _MARKER_PERSISTENT_MEMORY not in disabled_ctx
                    else "",
                )
            )
            results.append(
                _InvariantResult(
                    name="disabled: relevant-memories block omitted",
                    passed=_MARKER_RELEVANT_MEMORIES not in disabled_ctx,
                    detail=f"unexpectedly contained {_MARKER_RELEVANT_MEMORIES!r}"
                    if _MARKER_RELEVANT_MEMORIES in disabled_ctx
                    else "",
                )
            )

            # Disabled-mode recall-log reason invariant. Drives
            # `format_context` under a config with memory_enabled=False
            # so init_memory short-circuits at its
            # `if not config.memory_enabled: return` guard, leaving
            # `_memory` and `_config` as None. format_context then
            # short-circuits via its `if not is_enabled() or _config
            # is None:` branch and emits a `memory.recall` log line
            # with `reason: "disabled"`. The invariant verifies the
            # log-line contract that downstream eval harnesses depend
            # on (a missing or wrongly-tagged record under disabled
            # mode would silently break those harnesses).
            #
            # Deferred to preserve the `if "kai.memory" in sys.modules:`
            # invariant guarded at the top of `_run_verify`. A
            # top-level `from kai.memory import _RECALL_REASON_DISABLED`
            # in this module would put `kai.memory` into `sys.modules`
            # before `_run_verify` runs, tripping the assertion. The
            # constant itself is a string literal and does not boot
            # Mem0 or touch the migrations_qdrant lock; the deadlock
            # concern that motivates `_isolated_data_dir`'s lazy
            # import does not apply directly to this constant. The
            # consistency-with-the-guard rationale is what carries
            # the deferral.
            from kai.memory import _RECALL_REASON_DISABLED

            with _isolated_data_dir(tmp_root):
                from kai import memory as memory_module

                memory_module.init_memory(_build_test_configs(memory_enabled_value=False))
                with _capture_recall_logs() as disabled_recall_records:
                    await memory_module.format_context(_FIXTURE_QUERY, user_id=str(_FIXTURE_CHAT_ID))

            disabled_reason = _parse_recall_reason(disabled_recall_records)
            disabled_reason_ok = disabled_reason == _RECALL_REASON_DISABLED
            results.append(
                _InvariantResult(
                    name="disabled: recall log reason is 'disabled'",
                    passed=disabled_reason_ok,
                    detail=(
                        ""
                        if disabled_reason_ok
                        else f"got reason={disabled_reason!r}, expected {_RECALL_REASON_DISABLED!r}"
                    ),
                )
            )

            # ── Enabled-mode invariants ───────────────────────────
            enabled_ctx = build_session_context(
                workspace=ws,
                home_workspace=ws,
                api=api_ctx,
                workspace_config=None,
                chat_id=_FIXTURE_CHAT_ID,
                data_dir=tmp_root,
                memory_enabled=True,
            )

            results.append(
                _InvariantResult(
                    name="enabled: subsystem marker present",
                    passed=_MARKER_ENABLED in enabled_ctx,
                    detail=f"missing {_MARKER_ENABLED!r}" if _MARKER_ENABLED not in enabled_ctx else "",
                )
            )
            results.append(
                _InvariantResult(
                    name="enabled: persistent-memory block omitted",
                    passed=_MARKER_PERSISTENT_MEMORY not in enabled_ctx,
                    detail=f"unexpectedly contained {_MARKER_PERSISTENT_MEMORY!r}"
                    if _MARKER_PERSISTENT_MEMORY in enabled_ctx
                    else "",
                )
            )

            # The enabled-mode format_context probe runs in an
            # isolated tmp-dir Qdrant. We seed one fact, query for it,
            # and assert the recall block either is empty or starts
            # with the expected header. Both shapes are valid; the
            # invariant is the prefix check, not a presence claim.
            #
            # The same invocation is wrapped in `_capture_recall_logs`
            # so we can additionally assert the `memory.recall` log
            # line carries a non-disabled `reason`. format_context
            # emits exactly one log line per call (multiple internal
            # short-circuit branches all funnel through
            # `_emit_recall_log` once), so the most-recent record
            # under capture is the one we want.
            recall_text = ""
            with _isolated_data_dir(tmp_root):
                from kai import memory as memory_module

                memory_module.init_memory(_build_test_configs(memory_enabled_value=True))
                await _seed_fixture_fact(user_id=str(_FIXTURE_CHAT_ID))
                with _capture_recall_logs() as enabled_recall_records:
                    recall_text = await memory_module.format_context(_FIXTURE_QUERY, user_id=str(_FIXTURE_CHAT_ID))

            recall_ok = recall_text == "" or recall_text.startswith(_MARKER_RELEVANT_MEMORIES)
            results.append(
                _InvariantResult(
                    name="enabled: format_context returns empty or recall-prefixed",
                    passed=recall_ok,
                    detail=(f"got {recall_text[:80]!r}" if not recall_ok else ""),
                )
            )

            # Enabled-mode recall-log reason invariant. Asserts the
            # negative (reason != "disabled") rather than enumerating
            # specific allowed values so a future new reason added in
            # `memory.py`'s `_RECALL_REASON_*` enum does not break
            # this test. The match-positively-on-disabled / match-
            # negatively-on-enabled split mirrors the spec contract:
            # disabled mode MUST emit reason="disabled"; enabled mode
            # MAY emit any of the non-disabled reasons depending on
            # whether the seeded fact lands above the relevance floor,
            # whether the query is empty, etc.
            #
            # `_parse_recall_reason` returns None when the payload's
            # `reason` field is missing, which is the documented
            # success-path shape (per `_base_recall_payload`'s
            # docstring: `reason` is omitted on success, present only
            # on short-circuit lines). Both None and any non-disabled
            # string satisfy the invariant; only the literal string
            # equal to `_RECALL_REASON_DISABLED` fails it.
            enabled_reason = _parse_recall_reason(enabled_recall_records)
            enabled_reason_ok = enabled_reason != _RECALL_REASON_DISABLED
            results.append(
                _InvariantResult(
                    name="enabled: recall log reason is not 'disabled'",
                    passed=enabled_reason_ok,
                    detail=(
                        ""
                        if enabled_reason_ok
                        else f"got reason={enabled_reason!r}, expected anything OTHER than {_RECALL_REASON_DISABLED!r}"
                    ),
                )
            )

            # ── Partition invariants over the combined output ─────
            # combined = build_session_context output + format_context
            # output, joined with a newline. The two substrings must
            # never coexist regardless of flag value. format_context
            # returns "" under disabled mode by contract (the
            # is_enabled() guard short-circuits before search), so the
            # disabled-mode combined output is just the session
            # context with a trailing newline.
            combined_disabled = disabled_ctx + "\n"
            combined_enabled = enabled_ctx + "\n" + recall_text

            partition_disabled_ok = (
                _MARKER_PERSISTENT_MEMORY in combined_disabled and _MARKER_RELEVANT_MEMORIES not in combined_disabled
            )
            results.append(
                _InvariantResult(
                    name="partition disabled: persistent present, relevant absent",
                    passed=partition_disabled_ok,
                    detail="" if partition_disabled_ok else "partition broken under disabled",
                )
            )

            partition_enabled_ok = _MARKER_PERSISTENT_MEMORY not in combined_enabled
            results.append(
                _InvariantResult(
                    name="partition enabled: persistent absent",
                    passed=partition_enabled_ok,
                    detail="" if partition_enabled_ok else "persistent block leaked into enabled-mode combined output",
                )
            )

            mutual_exclusion_ok = not (
                _MARKER_PERSISTENT_MEMORY in combined_enabled and _MARKER_RELEVANT_MEMORIES in combined_enabled
            ) and not (
                _MARKER_PERSISTENT_MEMORY in combined_disabled and _MARKER_RELEVANT_MEMORIES in combined_disabled
            )
            results.append(
                _InvariantResult(
                    name="partition: mutual exclusion across both modes",
                    passed=mutual_exclusion_ok,
                    detail="" if mutual_exclusion_ok else "both blocks present in one mode",
                )
            )
        finally:
            # Restore MEM0_DIR to its pre-call value (or remove it if
            # it was unset). Without this, subsequent code in the same
            # process sees MEM0_DIR pointing at the tmp path we just
            # deleted on tempfile teardown, with no diagnostic when
            # the next mem0-using subprocess hits the missing dir.
            if prior_mem0_dir is _MEM0_DIR_SENTINEL:
                os.environ.pop("MEM0_DIR", None)
            elif isinstance(prior_mem0_dir, str):
                # Pyright strict cannot narrow `str | object` through
                # an `is`-check on a non-literal sentinel; the
                # explicit isinstance check narrows the type for the
                # `os.environ` setter, which requires `str`.
                # `if isinstance` rather than `assert isinstance`
                # because this runs in a `finally` block: a hard
                # assert here would shadow any exception propagating
                # out of the `try` body. The isinstance condition
                # is structurally always true (os.environ.get returns
                # `str` or the sentinel; the elif branch only runs
                # when the sentinel comparison failed); the `if`
                # form is the defensive shape for finally blocks.
                os.environ["MEM0_DIR"] = prior_mem0_dir

    # Print results.
    failed = [r for r in results if not r.passed]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        suffix = f"  [{r.detail}]" if r.detail else ""
        print(f"  {status}  {r.name}{suffix}")
    print()
    if failed:
        print(f"FAIL: {len(failed)} of {len(results)} invariants failed.")
        return 1
    print(f"PASS: all {len(results)} invariants hold.")
    return 0


# ── Check: runtime sanity ───────────────────────────────────────────


# Pinned regexes for the prompt-version probe. Match the source-level
# constants `_EXTRACTION_PROMPT_VERSION: str = "N"` and the analogous
# episode-prompt version. Pinned shape so the probe does not have to
# re-derive the regex if the source-level wording shifts; an edit to
# the version-line format that breaks these regexes is a real
# regression that the test suite at TestPromptVersionRead will catch.
_EXTRACTION_VERSION_RE = re.compile(r'_EXTRACTION_PROMPT_VERSION:\s*str\s*=\s*"([^"]+)"')
_EPISODE_VERSION_RE = re.compile(r'_EPISODE_PROMPT_VERSION:\s*str\s*=\s*"([^"]+)"')


# Lookup order for the deployed memory_extraction.py source file.
# /opt/kai/ is the production install layout per the project
# convention; the source-tree fallback covers dev environments and
# local test invocations. Listed in priority order.
_PROMPT_VERSION_PATH_PRIMARY = Path("/opt/kai/src/kai/memory_extraction.py")
_PROMPT_VERSION_PATH_FALLBACK = Path(__file__).parent.parent / "memory_extraction.py"


def _read_prompt_versions() -> tuple[str, str]:
    """Read the deployed prompt-version constants from the source
    file. Returns (`extraction_version`, `episode_version`); each
    falls back to the literal string `"unknown"` when the file is
    not found at either lookup path or the regex does not match.

    Reading the file rather than importing the module means this
    works without sys.path manipulation and without sudo. The
    operator does not need to be the kai service user to inspect
    the deployed binary's version constants.
    """
    for path in (_PROMPT_VERSION_PATH_PRIMARY, _PROMPT_VERSION_PATH_FALLBACK):
        if path.exists():
            text = path.read_text()
            ext_match = _EXTRACTION_VERSION_RE.search(text)
            ep_match = _EPISODE_VERSION_RE.search(text)
            ext = ext_match.group(1) if ext_match else "unknown"
            ep = ep_match.group(1) if ep_match else "unknown"
            return ext, ep
    return "unknown", "unknown"


def _http_get(url: str, secret: str | None = None, timeout: float = 5.0) -> tuple[int, bytes]:
    """Issue a GET against the local kai service and return
    (status_code, body_bytes). On network error or timeout, returns
    (0, b"") so the caller can branch on a non-200/non-503 status.

    Uses urllib (stdlib) rather than requests so the harness has no
    third-party HTTP dependency. The timeout cap prevents the probe
    from hanging if the service has bound the port but is not
    responding.
    """
    req = urllib.request.Request(url)
    if secret is not None:
        req.add_header("X-Webhook-Secret", secret)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        # 4xx/5xx responses surface here; still useful to read the
        # body for the error message.
        try:
            body = exc.read()
        except Exception:
            body = b""
        return exc.code, body
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, b""


def _run_check() -> int:
    """Probe the running service and print a five-line report.

    Tri-state exit code:
      0: clean read (health=ok and stats returned a definite mode)
      1: health=down or stats returned an unexpected status
      2: WEBHOOK_SECRET not in process environment
    """
    # Distinguish "not exported" (None) from "exported but empty"
    # (the empty string). Both fail the same way for /api/memory/stats
    # auth (the request would carry no usable secret), but the
    # diagnostic differs: a missing-from-env case wants "source
    # /etc/kai/env first"; an empty-but-set case wants the operator
    # to fix the env-file value. `check` is the tool operators run
    # when debugging a config issue; a misleading diagnostic at that
    # moment is worse than verbose accuracy.
    secret = os.environ.get("WEBHOOK_SECRET")

    if secret is None:
        print("secret_found: no")
        print("  WEBHOOK_SECRET not set; source /etc/kai/env first")
        print("  e.g. sudo bash -c 'source /etc/kai/env && env | grep WEBHOOK_SECRET'")
        return 2
    if secret == "":
        print("secret_found: empty")
        print("  WEBHOOK_SECRET is exported but empty; check /etc/kai/env for a typo")
        print("  (a line like `WEBHOOK_SECRET=` with no value)")
        return 2

    print("secret_found: yes")

    # Resolve the webhook port from the environment so the harness
    # tracks a non-default deploy (dev instance on a different port,
    # port-conflict resolution). Mirrors the `WEBHOOK_PORT` env var
    # the production service reads in config.py's load_config; the
    # default of 8080 matches the Config dataclass default. A
    # non-integer value or out-of-range value falls through to 8080
    # with a warning rather than crashing the harness or producing
    # a confusing `health: down (status=0)` from a port the OS
    # cannot bind. Range check is `1 <= port <= 65535` per the
    # POSIX TCP/IP port-number contract.
    port_str = os.environ.get("WEBHOOK_PORT", "8080")
    try:
        port = int(port_str)
        if not 1 <= port <= 65535:
            raise ValueError(f"port {port} out of range")
    except ValueError as exc:
        print(f"  WEBHOOK_PORT={port_str!r} is invalid ({exc}); falling back to 8080")
        port = 8080
    base_url = f"http://localhost:{port}"

    # Health probe first; if the service is down, the stats probe
    # cannot succeed and we report up-front.
    health_status, _ = _http_get(f"{base_url}/health")
    if health_status != 200:
        print(f"health: down (status={health_status})")
        ext_v, ep_v = _read_prompt_versions()
        print("mode: unknown(service-down)")
        print(f"extraction_prompt_version: {ext_v}")
        print(f"episode_prompt_version: {ep_v}")
        return 1

    print("health: ok")

    # Stats probe: 200 = enabled, 503 = disabled, anything else =
    # unexpected. Issue the request with NO `chat_id` query
    # parameter so the server falls back to its app-default
    # (`request.app[CHAT_ID_KEY]` at webhook.py's _resolve_chat_id).
    # Earlier versions of this harness passed `chat_id=1` as a
    # placeholder, which fails: _resolve_chat_id runs after secret
    # auth but before the memory.is_enabled() branch, and rejects
    # any explicit chat_id not in `allowed_user_ids` with a 403.
    # The mode probe needs the enabled/disabled signal, not any
    # particular user's stats, so omitting the parameter routes
    # cleanly through the app-default and reports the system-wide
    # memory state.
    stats_url = f"{base_url}/api/memory/stats"
    stats_status, _ = _http_get(stats_url, secret=secret)

    ext_v, ep_v = _read_prompt_versions()

    if stats_status == 200:
        print("mode: enabled")
        rc = 0
    elif stats_status == 503:
        print("mode: disabled")
        rc = 0
    else:
        print(f"mode: unknown({stats_status})")
        rc = 1

    print(f"extraction_prompt_version: {ext_v}")
    print(f"episode_prompt_version: {ep_v}")
    return rc


# ── CLI entry point ─────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)
    sub.add_parser("verify", help="Black-box logic verification of the gating contract.")
    sub.add_parser("check", help="Runtime sanity check against the running service.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.subcommand == "verify":
        return asyncio.run(_run_verify())
    if args.subcommand == "check":
        return _run_check()
    # argparse `required=True` makes this unreachable; guard for type-checker.
    return 2


if __name__ == "__main__":
    sys.exit(main())
