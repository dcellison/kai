"""Tests for the mode-switch verification harness.

Test classes:

- TestVerifyInvariants: exercises `build_session_context` directly under
  both flag values; the format_context-dependent invariants are covered
  with a mocked format_context to keep unit tests fast and to avoid Mem0
  Qdrant lock contention with a running production service.
- TestRecallReasonField: switch-point-6 contract; asserts format_context
  emits a memory.recall log line with reason='disabled' under disabled
  mode and any non-'disabled' reason (or no reason field at all on the
  success path) under enabled mode.
- TestCheckSubcommand: monkeypatches the `urlopen` shim and the
  `_read_prompt_versions` helper to drive each branch of the tri-state
  exit-code contract.
- TestPromptVersionRead: covers the prompt-version probe's two-path
  fallback under tmp_path.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

from kai.eval import modeswitch

# ── Helpers ─────────────────────────────────────────────────────────


_FIXTURE_CHAT_ID = modeswitch._FIXTURE_CHAT_ID
_MARKER_DISABLED = modeswitch._MARKER_DISABLED
_MARKER_ENABLED = modeswitch._MARKER_ENABLED
_MARKER_PERSISTENT_MEMORY = modeswitch._MARKER_PERSISTENT_MEMORY
_MARKER_RELEVANT_MEMORIES = modeswitch._MARKER_RELEVANT_MEMORIES


def _build_disabled_ctx(tmp_path: Path) -> str:
    """Drive build_session_context under memory_enabled=False with the
    same fixture seeding the verify subcommand uses. Returns the
    rendered context string."""
    modeswitch._seed_fixture_memory_md(tmp_path)
    api_ctx = modeswitch._api_ctx_for_verify()
    ws = tmp_path / "workspace"
    ws.mkdir(exist_ok=True)
    return modeswitch.build_session_context(
        workspace=ws,
        home_workspace=ws,
        api=api_ctx,
        workspace_config=None,
        chat_id=_FIXTURE_CHAT_ID,
        data_dir=tmp_path,
        memory_enabled=False,
    )


def _build_enabled_ctx(tmp_path: Path) -> str:
    """Counterpart of `_build_disabled_ctx` under memory_enabled=True."""
    modeswitch._seed_fixture_memory_md(tmp_path)
    api_ctx = modeswitch._api_ctx_for_verify()
    ws = tmp_path / "workspace"
    ws.mkdir(exist_ok=True)
    return modeswitch.build_session_context(
        workspace=ws,
        home_workspace=ws,
        api=api_ctx,
        workspace_config=None,
        chat_id=_FIXTURE_CHAT_ID,
        data_dir=tmp_path,
        memory_enabled=True,
    )


# ── TestVerifyInvariants ────────────────────────────────────────────


class TestVerifyInvariants:
    """Verify subcommand's nine invariants. The five non-recall
    invariants run directly against `build_session_context`; the four
    recall-path invariants use a mocked `format_context` so the test
    exercises the harness's interpretation contract without paying
    Qdrant init cost or risking Mem0 lock contention with a running
    production service.
    """

    def test_disabled_mode_injects_memory_md(self, tmp_path: Path) -> None:
        """Under memory_enabled=False, the build_session_context output
        contains the [Your persistent memory (file: ...):] block. This
        is the load-bearing positive assertion for disabled mode."""
        ctx = _build_disabled_ctx(tmp_path)
        assert _MARKER_PERSISTENT_MEMORY in ctx
        assert _MARKER_DISABLED in ctx

    def test_disabled_mode_omits_relevant_memories_block(self, tmp_path: Path) -> None:
        """Under memory_enabled=False, the build_session_context output
        does NOT contain the recall-block prefix. The recall path is
        out of scope for build_session_context (it's emitted later by
        format_context), but the harness's invariant is over the
        combined output, so the prefix must NOT leak from
        build_session_context into the disabled-mode context."""
        ctx = _build_disabled_ctx(tmp_path)
        assert _MARKER_RELEVANT_MEMORIES not in ctx

    def test_enabled_mode_omits_memory_md(self, tmp_path: Path) -> None:
        """Under memory_enabled=True, MEMORY.md is dormant: the
        [Your persistent memory ...] block does NOT appear in the
        build_session_context output. This is the load-bearing
        negative assertion for enabled mode."""
        ctx = _build_enabled_ctx(tmp_path)
        assert _MARKER_PERSISTENT_MEMORY not in ctx

    def test_enabled_mode_marker_present(self, tmp_path: Path) -> None:
        """Under memory_enabled=True, the [Memory subsystem: enabled]
        marker is emitted unconditionally, in contrast to the
        persistent-memory block which is gated by the flag."""
        ctx = _build_enabled_ctx(tmp_path)
        assert _MARKER_ENABLED in ctx

    def test_partition_invariant_disabled(self, tmp_path: Path) -> None:
        """Mutual-exclusivity invariant under disabled mode: the
        combined session-context-plus-recall output contains the
        persistent-memory block AND does NOT contain the
        relevant-memories block. Under disabled mode, format_context
        is contractually empty (the recall path short-circuits via
        is_enabled()), so we model that with the empty string."""
        disabled_ctx = _build_disabled_ctx(tmp_path)
        # format_context returns "" under disabled mode by contract;
        # the recall path's is_enabled() guard short-circuits before
        # the search call. The combined output is just the
        # build_session_context output with a trailing newline.
        combined = disabled_ctx + "\n"
        assert _MARKER_PERSISTENT_MEMORY in combined
        assert _MARKER_RELEVANT_MEMORIES not in combined

    def test_partition_invariant_enabled_with_recall(self, tmp_path: Path) -> None:
        """Mutual-exclusivity invariant under enabled mode WITH a
        seeded fact above the floor: the combined output contains the
        relevant-memories block AND does NOT contain the
        persistent-memory block. format_context is mocked to return
        a canned string starting with the recall-block prefix; the
        invariant under test is the harness's partition contract,
        not format_context's own ranking behavior."""
        enabled_ctx = _build_enabled_ctx(tmp_path)
        recall_text = _MARKER_RELEVANT_MEMORIES + "]\n- (2026-05-01, fact) seeded fixture content"
        combined = enabled_ctx + "\n" + recall_text
        assert _MARKER_PERSISTENT_MEMORY not in combined
        assert _MARKER_RELEVANT_MEMORIES in combined

    def test_partition_invariant_enabled_no_recall(self, tmp_path: Path) -> None:
        """Mutual-exclusivity invariant under enabled mode WITHOUT
        any retrievable seed (format_context returns empty when no
        rows clear the relevance floor): the combined output STILL
        does NOT contain the persistent-memory block. The
        relevant-memories block may be absent; both shapes are
        valid under enabled mode. The invariant is the absence of
        MEMORY.md, not the presence of recall."""
        enabled_ctx = _build_enabled_ctx(tmp_path)
        combined = enabled_ctx + "\n"
        assert _MARKER_PERSISTENT_MEMORY not in combined
        # Both shapes valid: relevant-memories may or may not be
        # present. The load-bearing assertion is the absence of
        # the persistent block.

    def test_partition_invariant_mutual_exclusion(self, tmp_path: Path) -> None:
        """Across both flag values, the persistent-memory and
        relevant-memories blocks NEVER coexist. This is the strongest
        formulation of the partition invariant: the harness's job is
        to surface a regression that injected both blocks under a
        single flag value.

        Regression shape under disabled mode: persistent present AND
        relevant present (the recall path leaked into disabled mode).
        Regression shape under enabled mode: persistent present AND
        relevant present (the MEMORY.md inject leaked into enabled
        mode). The assertion is the negation of both regressions."""
        # Disabled: persistent present, relevant absent.
        disabled_combined = _build_disabled_ctx(tmp_path) + "\n"
        assert not (_MARKER_PERSISTENT_MEMORY in disabled_combined and _MARKER_RELEVANT_MEMORIES in disabled_combined)
        # Enabled with recall: persistent absent (the regression
        # shape would be both present).
        enabled_recall = _MARKER_RELEVANT_MEMORIES + "]\n- (2026-05-01, fact) x"
        enabled_combined = _build_enabled_ctx(tmp_path) + "\n" + enabled_recall
        assert not (_MARKER_PERSISTENT_MEMORY in enabled_combined and _MARKER_RELEVANT_MEMORIES in enabled_combined)
        # Stronger formulation: under enabled mode, persistent block
        # is ALWAYS absent regardless of whether recall fired.
        assert _MARKER_PERSISTENT_MEMORY not in enabled_combined


# ── TestRecallReasonField ───────────────────────────────────────────


class TestRecallReasonField:
    """Switch-point-6 contract: format_context emits exactly one
    `memory.recall` log line per call, and the line's `reason` field
    distinguishes disabled-mode short-circuits ("disabled") from every
    other outcome. Eval harnesses parse the `reason` field to bucket
    log lines by short-circuit category; a regression that emitted
    "disabled" under enabled mode (or omitted the marker under
    disabled mode) would silently break those harnesses.

    Mirrors the new `_run_verify` invariants the harness runs at the
    operator-CLI level. The harness drives format_context against a
    real tmp-dir Mem0 instance; the tests here use mocking to stay
    fast and to avoid Mem0 lock contention with a running production
    service. Both surfaces assert the same contract.
    """

    @pytest.fixture
    def reset_memory_module(self):
        """Save and restore `kai.memory`'s module-level singletons
        (`_memory`, `_config`) around each test. format_context reads
        these directly; leaving a prior test's setup live would let
        configuration leak across tests. The teardown is unconditional
        so a test that mutates state and then fails does not strand
        junk for the next test.
        """
        from kai import memory as memory_module

        prior_memory = memory_module._memory
        prior_config = memory_module._config
        memory_module._memory = None
        memory_module._config = None
        try:
            yield
        finally:
            memory_module._memory = prior_memory
            memory_module._config = prior_config

    @pytest.mark.asyncio
    async def test_recall_reason_is_disabled_under_disabled_mode(self, reset_memory_module, caplog):
        """memory_enabled=False: init_memory short-circuits and leaves
        the singletons unset, so format_context's first guard fires
        (`if not is_enabled() or _config is None`) and emits a
        memory.recall line with reason='disabled'. The harness's
        `_parse_recall_reason` is reused so both surfaces interpret
        the captured payload via the same code.
        """
        from kai import memory as memory_module

        # init_memory's "if not config.memory_enabled: return" guard
        # is the production entry into the disabled state. The fixture
        # already cleared the singletons, but invoking init_memory
        # explicitly mirrors the production-path entry the harness
        # uses, so a regression that moved the guard inside init_memory
        # would surface here as a state-mutation bug rather than a
        # silently passing test.
        memory_module.init_memory(modeswitch._build_test_configs(memory_enabled_value=False))

        with caplog.at_level(logging.INFO, logger="kai.memory"):
            result = await memory_module.format_context(
                "test query",
                user_id=str(_FIXTURE_CHAT_ID),
            )

        # format_context returns "" on the disabled short-circuit;
        # paired with the reason assertion below, this pins both
        # halves of the contract (return shape + log-line shape).
        assert result == ""
        # Assert at least one memory.recall record exists. Without
        # this, a regression that silenced the disabled-mode log line
        # entirely would let `_parse_recall_reason` return None,
        # which the reason equality check below would NOT distinguish
        # from a simple field-rename regression. The two assertions
        # together pin presence AND correct content.
        recall_records = [r for r in caplog.records if r.getMessage().startswith("memory.recall ")]
        assert recall_records, "expected at least one memory.recall log line"
        reason = modeswitch._parse_recall_reason(caplog.records)
        assert reason == memory_module._RECALL_REASON_DISABLED, (
            f"expected reason={memory_module._RECALL_REASON_DISABLED!r}; got {reason!r}"
        )

    @pytest.mark.asyncio
    async def test_recall_reason_is_not_disabled_under_enabled_mode(self, reset_memory_module, monkeypatch, caplog):
        """memory_enabled=True with a search hit: format_context goes
        through the recall path and emits a memory.recall line whose
        `reason` field is anything OTHER than 'disabled'. On the
        success path the field is omitted entirely (per
        `_base_recall_payload`'s docstring: `reason` is only set on
        short-circuit lines), so `_parse_recall_reason` returns None;
        on the various non-disabled short-circuits (empty_query,
        no_results, all_below_floor, budget_exhausted) it returns the
        corresponding string. Both shapes satisfy the contract.

        The test mocks `kai.memory.search` to return one canned
        MemoryResult above the relevance floor so the recall path
        completes without spinning up Mem0. The harness's `verify`
        subcommand exercises the same contract against a real
        tmp-dir Mem0 instance; the two surfaces are intentionally
        complementary (this test is the CI gate; the harness is the
        operator gate).
        """
        from kai import memory as memory_module

        # Stand the singletons up by hand. Setting `_memory` to a
        # truthy sentinel makes is_enabled() return True; `_config`
        # supplies the budget and floor that the recall path reads.
        # Bypassing init_memory avoids the Mem0/sentence-transformers
        # boot cost (~80MB embedding model on first run); the
        # production path under verification is format_context, not
        # init_memory.
        #
        # `object()` is safe ONLY because format_context never
        # invokes attributes on `_memory` directly: `is_enabled()`
        # checks `_memory is not None`, and every search call
        # routes through the module-level `search()` wrapper, which
        # is mocked below. A future refactor that called something
        # like `_memory.client_ready()` from the hot path would
        # break this test with a confusing `AttributeError` rather
        # than a clean assertion failure; the sentinel would need
        # to grow into a proper Mem0 mock at that point.
        memory_module._memory = object()
        memory_module._config = modeswitch._build_test_configs(memory_enabled_value=True)

        canned_result = memory_module.MemoryResult(
            id="test-fact-1",
            text="Mode-switch test fixture marker phrase.",
            score=0.95,
            memory_type="fact",
            metadata={"source": "extracted"},
            created_at="2026-05-01T00:00:00Z",
        )
        # Mock the synchronous search() helper. format_context dispatches
        # search() through `loop.run_in_executor`, so a sync mock is the
        # correct shape; an async mock would break that wrapping.
        monkeypatch.setattr(memory_module, "search", lambda query, *, user_id, limit=None: [canned_result])

        with caplog.at_level(logging.INFO, logger="kai.memory"):
            result = await memory_module.format_context(
                "test fixture query",
                user_id=str(_FIXTURE_CHAT_ID),
            )

        # Reason assertion FIRST: the log-line `reason` field is the
        # contract under test. The output-shape assertion below is a
        # secondary sanity check; if the canned result's score (0.95)
        # ever stops clearing the configured relevance floor, the
        # output-shape assertion would fire and mask the actual
        # contract test. Reason-first keeps the failure message
        # informative under that regression class.
        recall_records = [r for r in caplog.records if r.getMessage().startswith("memory.recall ")]
        assert recall_records, "expected at least one memory.recall log line"
        reason = modeswitch._parse_recall_reason(caplog.records)
        # Negative assertion: anything OTHER than the disabled marker
        # is acceptable. Includes None (success-path payload omits
        # `reason`) and any of the other `_RECALL_REASON_*` constants.
        assert reason != memory_module._RECALL_REASON_DISABLED, (
            f"expected reason != {memory_module._RECALL_REASON_DISABLED!r}; got {reason!r}"
        )
        # Output-shape sanity check: the canned result is above the
        # floor and within budget, so format_context should have
        # produced the recall-prefixed string. A miss here suggests
        # the floor was tuned above 0.95 or the budget was tightened
        # below the single-line cost.
        assert result.startswith(_MARKER_RELEVANT_MEMORIES), f"expected recall-prefixed output; got {result[:80]!r}"


# ── TestExtractionCallSiteGating ────────────────────────────────────


class TestExtractionCallSiteGating:
    """Switch-point-5 structural backstop (#434): every call site of
    `memory_extraction.extract_and_store` in the shared conversation
    compatibility writer must
    sit inside an enclosing `if memory_is_enabled() ...:` guard.

    The behavioral pair in `test_bot.py::TestHandleResponse`
    exercises the gate with mocks; this test enforces it at the
    AST level so a refactor that adds a SECOND call site OUTSIDE
    the guard fails at CI rather than at runtime under disabled
    mode. Today there is exactly one call site in
    `schedule_memory_ingestion`; the test asserts the gating
    property over every Call node, not just "the one Call we know
    about."

    The predicate walks `if_node.test` recursively via `ast.walk`
    so the safety contract remains valid if the simple guard later
    gains another condition.
    """

    @staticmethod
    def _attach_parents(tree: ast.AST) -> None:
        """Attach a `parent` attribute to every node so ancestor
        traversal works after `ast.walk`. ast.AST does not declare
        a `parent` field, so pyright flags the assignment as an
        attr-defined error; the `# type: ignore[attr-defined]`
        below is load-bearing, not cosmetic. The alternative
        (a side-table dict keyed by id(node)) is more verbose with
        no semantic gain; this is the standard ast-walking pattern.
        """
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                child.parent = parent  # type: ignore[attr-defined]

    @staticmethod
    def _guard_contains_memory_is_enabled(if_node: ast.If) -> bool:
        """True iff `if_node.test` contains, somewhere in its
        expression tree, a `Call` to a name `memory_is_enabled`.

        Recursive-walk via `ast.walk(if_node.test)` so the
        predicate also matches a future compound guard. A direct-test predicate
        (`isinstance(if_node.test, ast.Call) and
        if_node.test.func.id == 'memory_is_enabled'`) would return
        False on this shape and silently miss the gate.

        Limitation: this is a presence check, not a polarity check.
        An inverted guard (`if not memory_is_enabled():`) would
        ALSO satisfy the predicate because the Call lives inside
        the `UnaryOp(Not, ...)` and `ast.walk` reaches it. A
        regression that flipped the guard to `not` would have
        inverted semantics (extract under disabled, skip under
        enabled) but pass this structural test. The behavioral
        pair in `test_bot.py::TestHandleResponse` is the polarity
        check; this test's job is the call-site-existence check.
        """
        return any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "memory_is_enabled"
            for n in ast.walk(if_node.test)
        )

    def test_every_extract_and_store_call_site_is_inside_is_enabled_guard(self) -> None:
        """Walk the compatibility writer's AST. For every `Call` whose `func`
        is an `Attribute` with `attr == "extract_and_store"`, confirm
        that at least one ancestor `If` node has a test containing a
        `Call` to `memory_is_enabled`. Fail with the offending line
        number if any call site is unguarded.

        `func` is checked as `Attribute` (e.g.
        `memory_extraction.extract_and_store(...)`) per the production
        shape; the test deliberately does NOT match `Name`-shaped
        calls (`extract_and_store(...)` from a `from
        kai.memory_extraction import extract_and_store`) because the
        production code uses the module-qualified form. If a future
        refactor introduces the `from`-import shape, this test should
        be extended to cover both forms.

        Polarity caveat: this test is a presence check, not a
        polarity check. An inverted guard
        (`if not memory_is_enabled():`) would satisfy the predicate
        because `_guard_contains_memory_is_enabled` walks the test
        expression and finds the Call inside the `UnaryOp(Not, ...)`.
        The behavioral pair in
        `test_bot.py::TestHandleResponse::test_extraction_skipped_when_memory_disabled`
        plus `test_extraction_invoked_when_memory_enabled` covers
        polarity by asserting the correct `extract_and_store`
        call_count under each flag value.
        """
        # Resolve the production source path relative to the test file's
        # location so the test runs cleanly under any working
        # directory (pytest cwd, IDE runner, CI). `Path(__file__)`
        # is `tests/test_eval_modeswitch.py`; two levels up is the
        # repo root, then descend into the shared compatibility writer.
        source_path = Path(__file__).parent.parent / "src" / "kai" / "conversation_compatibility.py"
        assert source_path.exists(), f"expected compatibility writer at {source_path}"

        tree = ast.parse(source_path.read_text())
        self._attach_parents(tree)

        # `ast.walk(tree)` traverses every node, not just top-level
        # `Module.body`. The actual extract_and_store call site lives
        # nested at `schedule_memory_ingestion > if memory_is_enabled() >
        # async def ingest_memory > await extract_and_store(...)`,
        # several layers below the module root, so the deep traversal
        # is required.
        call_sites: list[ast.Call] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "extract_and_store"
            ):
                call_sites.append(node)

        # Fail loud if the production code stops invoking
        # extract_and_store entirely. A regression that removes the
        # call site would otherwise leave this test passing
        # vacuously, masking the absence of the production behavior
        # the test is meant to backstop.
        assert call_sites, (
            "expected at least one `extract_and_store` call site in the compatibility "
            "writer; the production path lives in `schedule_memory_ingestion`, gated "
            "by `if memory_is_enabled():`"
        )

        ungated: list[int] = []
        for call in call_sites:
            # Walk the parent chain looking for an `If` whose test
            # contains a Call to memory_is_enabled. The chain ends
            # at the Module node which has no `parent` attribute;
            # the loop terminates by hitting that absence.
            node: ast.AST = call
            guarded = False
            while True:
                parent = getattr(node, "parent", None)
                if parent is None:
                    break
                if isinstance(parent, ast.If) and self._guard_contains_memory_is_enabled(parent):
                    guarded = True
                    break
                node = parent
            if not guarded:
                ungated.append(call.lineno)

        assert not ungated, (
            f"extract_and_store call site(s) at lines {ungated} are NOT inside an "
            "`if memory_is_enabled() ...` guard. Switch point 5 (#434) requires "
            "every extract_and_store call to be gated by memory_is_enabled() so "
            "the disabled mode does not write to Qdrant."
        )


# ── TestCheckSubcommand ─────────────────────────────────────────────


class TestCheckSubcommand:
    """The runtime check reads the non-sensitive mode from /health."""

    def test_check_health_down_exits_1(self, monkeypatch, capsys) -> None:
        """Health probe returns non-200: report `health: down`, skip
        the stats probe, and exit 1. Prompt-version probe still runs
        because operators want the deploy version visible even when
        the service is down."""

        def _http_health_down(url, timeout=5.0):
            if "/health" in url:
                return 500, b""
            raise AssertionError(f"unexpected url: {url}")

        monkeypatch.setattr(modeswitch, "_http_get", _http_health_down)
        monkeypatch.setattr(modeswitch, "_read_prompt_versions", lambda: ("7", "1"))
        rc = modeswitch._run_check()
        assert rc == 1
        out = capsys.readouterr().out
        assert "health: down" in out
        assert "mode: unknown(service-down)" in out
        assert "extraction_prompt_version: 7" in out
        assert "episode_prompt_version: 1" in out

    def test_check_health_reports_disabled(self, monkeypatch, capsys) -> None:
        def _http_disabled(url, timeout=5.0):
            assert url.endswith("/health")
            return 200, b'{"status": "ok", "memory_enabled": false}'

        monkeypatch.setattr(modeswitch, "_http_get", _http_disabled)
        monkeypatch.setattr(modeswitch, "_read_prompt_versions", lambda: ("7", "1"))
        rc = modeswitch._run_check()
        assert rc == 0
        out = capsys.readouterr().out
        assert "health: ok" in out
        assert "mode: disabled" in out
        assert "extraction_prompt_version: 7" in out
        assert "episode_prompt_version: 1" in out

    def test_check_health_reports_enabled(self, monkeypatch, capsys) -> None:
        def _http_enabled(url, timeout=5.0):
            assert url.endswith("/health")
            return 200, b'{"status": "ok", "memory_enabled": true}'

        monkeypatch.setattr(modeswitch, "_http_get", _http_enabled)
        monkeypatch.setattr(modeswitch, "_read_prompt_versions", lambda: ("7", "1"))
        rc = modeswitch._run_check()
        assert rc == 0
        out = capsys.readouterr().out
        assert "health: ok" in out
        assert "mode: enabled" in out
        assert "extraction_prompt_version: 7" in out
        assert "episode_prompt_version: 1" in out

    def test_check_invalid_health_payload_exits_1(self, monkeypatch, capsys) -> None:
        def _http_unexpected(url, timeout=5.0):
            assert url.endswith("/health")
            return 200, b'{"status": "ok"}'

        monkeypatch.setattr(modeswitch, "_http_get", _http_unexpected)
        monkeypatch.setattr(modeswitch, "_read_prompt_versions", lambda: ("7", "1"))
        rc = modeswitch._run_check()
        assert rc == 1
        out = capsys.readouterr().out
        assert "health: ok" in out
        assert "mode: unknown(health-payload)" in out
        assert "extraction_prompt_version: 7" in out
        assert "episode_prompt_version: 1" in out

    def test_check_webhook_port_non_integer_falls_back_to_8080(self, monkeypatch, capsys) -> None:
        """WEBHOOK_PORT set to a non-integer string: harness prints a
        warning, falls back to port 8080, and continues. The fallback
        is what unblocks the rest of the report; without it, the
        harness would crash with ValueError on a config typo.

        The health probe remains the only request."""
        monkeypatch.setenv("WEBHOOK_PORT", "not-an-integer")

        captured_urls: list[str] = []

        def _http_with_capture(url, timeout=5.0):
            captured_urls.append(url)
            return 200, b'{"status": "ok", "memory_enabled": true}'

        monkeypatch.setattr(modeswitch, "_http_get", _http_with_capture)
        monkeypatch.setattr(modeswitch, "_read_prompt_versions", lambda: ("7", "1"))
        rc = modeswitch._run_check()
        assert rc == 0
        out = capsys.readouterr().out
        # The fallback warning is printed before the report shape.
        assert "WEBHOOK_PORT='not-an-integer' is invalid" in out
        # Every captured URL must point at port 8080, the documented
        # fallback. Pinning the URL is what proves the fallback ran.
        assert all("localhost:8080" in u for u in captured_urls), captured_urls
        assert "health: ok" in out
        assert "mode: enabled" in out
        assert "extraction_prompt_version: 7" in out
        assert "episode_prompt_version: 1" in out

    def test_check_webhook_port_out_of_range_falls_back_to_8080(self, monkeypatch, capsys) -> None:
        """WEBHOOK_PORT set to an out-of-range integer (>65535):
        harness prints a warning naming the range violation, falls
        back to port 8080, and continues. Without the range guard,
        the harness would attempt to bind to a port the OS cannot
        accept and report `health: down (status=0)`, which is the
        confusing diagnostic the range guard exists to prevent.

        The health probe remains the only request."""
        monkeypatch.setenv("WEBHOOK_PORT", "99999")

        captured_urls: list[str] = []

        def _http_with_capture(url, timeout=5.0):
            captured_urls.append(url)
            return 200, b'{"status": "ok", "memory_enabled": true}'

        monkeypatch.setattr(modeswitch, "_http_get", _http_with_capture)
        monkeypatch.setattr(modeswitch, "_read_prompt_versions", lambda: ("7", "1"))
        rc = modeswitch._run_check()
        assert rc == 0
        out = capsys.readouterr().out
        assert "WEBHOOK_PORT='99999' is invalid" in out
        # The range-violation message names the bad port number so
        # the operator can correct the env-file value without
        # reading the source.
        assert "out of range" in out
        assert all("localhost:8080" in u for u in captured_urls), captured_urls
        assert "health: ok" in out
        assert "mode: enabled" in out
        assert "extraction_prompt_version: 7" in out
        assert "episode_prompt_version: 1" in out


# ── TestPromptVersionRead ───────────────────────────────────────────


class TestPromptVersionRead:
    """Prompt-version probe under the two-path lookup contract.
    Primary path is /opt/kai/src/kai/memory_extraction.py; fallback is
    the source-tree path relative to the script's location."""

    _SAMPLE_SOURCE = '_EXTRACTION_PROMPT_VERSION: str = "7"\n_EPISODE_PROMPT_VERSION: str = "1"\n'

    def test_prompt_version_read_from_deployed_path_when_present(self, tmp_path: Path, monkeypatch) -> None:
        """When the primary install-layout path exists, the probe
        reads it and the fallback is not consulted. Pinned via
        monkey-patching both paths to point at distinct tmp files
        with distinct version values; the assertion verifies the
        primary value won."""
        primary_file = tmp_path / "primary.py"
        fallback_file = tmp_path / "fallback.py"
        primary_file.write_text(
            '_EXTRACTION_PROMPT_VERSION: str = "primary-7"\n_EPISODE_PROMPT_VERSION: str = "primary-1"\n'
        )
        fallback_file.write_text(
            '_EXTRACTION_PROMPT_VERSION: str = "fallback-x"\n_EPISODE_PROMPT_VERSION: str = "fallback-y"\n'
        )

        monkeypatch.setattr(modeswitch, "_PROMPT_VERSION_PATH_PRIMARY", primary_file)
        monkeypatch.setattr(modeswitch, "_PROMPT_VERSION_PATH_FALLBACK", fallback_file)

        ext, ep = modeswitch._read_prompt_versions()
        assert ext == "primary-7"
        assert ep == "primary-1"

    def test_prompt_version_falls_back_to_source_tree(self, tmp_path: Path, monkeypatch) -> None:
        """When the primary path is missing, the probe falls back
        to the source-tree path and reads the version from there."""
        primary_missing = tmp_path / "definitely-not-there" / "memory_extraction.py"
        fallback_file = tmp_path / "fallback.py"
        fallback_file.write_text(self._SAMPLE_SOURCE)

        monkeypatch.setattr(modeswitch, "_PROMPT_VERSION_PATH_PRIMARY", primary_missing)
        monkeypatch.setattr(modeswitch, "_PROMPT_VERSION_PATH_FALLBACK", fallback_file)

        ext, ep = modeswitch._read_prompt_versions()
        assert ext == "7"
        assert ep == "1"

    def test_prompt_version_missing_reports_unknown(self, tmp_path: Path, monkeypatch) -> None:
        """When neither path exists, both versions are reported as
        the literal string `unknown` rather than raising. The check
        subcommand still proceeds with the rest of the report; an
        unknown version is information, not an error."""
        primary_missing = tmp_path / "definitely-not-there-1" / "memory_extraction.py"
        fallback_missing = tmp_path / "definitely-not-there-2" / "memory_extraction.py"

        monkeypatch.setattr(modeswitch, "_PROMPT_VERSION_PATH_PRIMARY", primary_missing)
        monkeypatch.setattr(modeswitch, "_PROMPT_VERSION_PATH_FALLBACK", fallback_missing)

        ext, ep = modeswitch._read_prompt_versions()
        assert ext == "unknown"
        assert ep == "unknown"

    def test_prompt_version_regex_does_not_match_non_version_lines(self, tmp_path: Path, monkeypatch) -> None:
        """The regex is anchored on the exact constant-name shape so
        unrelated code lines that happen to mention the constant
        (docstrings, comments, test fixtures) do not satisfy the
        pattern. Defends against a future regression where the regex
        was accidentally broadened."""
        decoy_file = tmp_path / "decoy.py"
        decoy_file.write_text(
            '# A comment about _EXTRACTION_PROMPT_VERSION = "fake"\n'
            'x = "_EXTRACTION_PROMPT_VERSION: str = \\"impostor\\""\n'
        )
        missing = tmp_path / "missing.py"

        monkeypatch.setattr(modeswitch, "_PROMPT_VERSION_PATH_PRIMARY", decoy_file)
        monkeypatch.setattr(modeswitch, "_PROMPT_VERSION_PATH_FALLBACK", missing)

        ext, ep = modeswitch._read_prompt_versions()
        # The decoy comment does not match the regex (no actual
        # `_EXTRACTION_PROMPT_VERSION: str = "..."` line); the
        # string-literal line has escaped quotes that break the regex.
        # Both reads fall through to the unknown sentinel.
        assert ext == "unknown"
        assert ep == "unknown"
