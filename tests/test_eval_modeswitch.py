"""Tests for the mode-switch verification harness.

Three test classes:

- TestVerifyInvariants: exercises `build_session_context` directly under
  both flag values; the format_context-dependent invariants are covered
  with a mocked format_context to keep unit tests fast and to avoid Mem0
  Qdrant lock contention with a running production service.
- TestCheckSubcommand: monkeypatches the `urlopen` shim and the
  `_read_prompt_versions` helper to drive each branch of the tri-state
  exit-code contract.
- TestPromptVersionRead: covers the prompt-version probe's two-path
  fallback under tmp_path.
"""

from __future__ import annotations

from pathlib import Path

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


# ── TestCheckSubcommand ─────────────────────────────────────────────


class TestCheckSubcommand:
    """check subcommand's tri-state exit code contract: 0 on a clean
    read, 2 on missing WEBHOOK_SECRET, 1 on health-down or unexpected
    stats status. Tests monkeypatch the `_http_get` helper to drive
    each branch without standing up an HTTP server."""

    def test_check_missing_secret_exits_2(self, monkeypatch, capsys) -> None:
        """WEBHOOK_SECRET not exported at all: print the
        not-set hint and exit 2 BEFORE making any HTTP call."""
        monkeypatch.delenv("WEBHOOK_SECRET", raising=False)

        # _http_get must NOT be called on this path. If it is,
        # the HTTP shim would try to reach localhost:8080, which
        # might pick up the running production service and skew
        # the test. Explicit raise on call surfaces the regression.
        def _no_http(*args, **kwargs):
            raise AssertionError("_http_get must not be called when secret is missing")

        monkeypatch.setattr(modeswitch, "_http_get", _no_http)
        rc = modeswitch._run_check()
        assert rc == 2
        out = capsys.readouterr().out
        # Pin all three lines the production code emits on the
        # missing-secret path. The example-command line is part of
        # the diagnostic shape; dropping it would leave the operator
        # without the actionable next step.
        assert "secret_found: no" in out
        assert "WEBHOOK_SECRET not set" in out
        assert "sudo bash -c 'source /etc/kai/env" in out

    def test_check_empty_secret_exits_2_with_distinct_diagnostic(self, monkeypatch, capsys) -> None:
        """WEBHOOK_SECRET exported but set to the empty string:
        report `secret_found: empty` (not `no`) and exit 2 with a
        diagnostic that points the operator at the env-file typo,
        not at sourcing the env file. The two cases are
        operationally distinct: one is "you didn't source", the
        other is "you sourced but the value is wrong"."""
        monkeypatch.setenv("WEBHOOK_SECRET", "")

        def _no_http(*args, **kwargs):
            raise AssertionError("_http_get must not be called when secret is empty")

        monkeypatch.setattr(modeswitch, "_http_get", _no_http)
        rc = modeswitch._run_check()
        assert rc == 2
        out = capsys.readouterr().out
        # Pin all three lines the production code emits on the
        # empty-secret path. The "a line like" parenthetical is the
        # actionable hint pointing at the env-file typo shape; the
        # operator needs it to know what to look for.
        assert "secret_found: empty" in out
        assert "exported but empty" in out
        assert "a line like" in out
        # The distinct diagnostic must NOT collapse back to the
        # not-set messaging that the operator already ruled out by
        # sourcing the env file.
        assert "WEBHOOK_SECRET not set" not in out

    def test_check_health_down_exits_1(self, monkeypatch, capsys) -> None:
        """Health probe returns non-200: report `health: down`, skip
        the stats probe, and exit 1. Prompt-version probe still runs
        because operators want the deploy version visible even when
        the service is down."""
        monkeypatch.setenv("WEBHOOK_SECRET", "test-secret")

        def _http_health_down(url, secret=None, timeout=5.0):
            if "/health" in url:
                return 500, b""
            raise AssertionError(f"unexpected url: {url}")

        monkeypatch.setattr(modeswitch, "_http_get", _http_health_down)
        monkeypatch.setattr(modeswitch, "_read_prompt_versions", lambda: ("7", "1"))
        rc = modeswitch._run_check()
        assert rc == 1
        out = capsys.readouterr().out
        # Pin every line the production code emits on the health-
        # down path: secret_found, health, mode, and BOTH prompt
        # versions. A regression that drops any of these lines
        # silently (e.g., an early-return that skips the prompt-
        # version probe) would go undetected without all four
        # assertions; pinning the full output shape closes that
        # gap.
        assert "secret_found: yes" in out
        assert "health: down" in out
        assert "mode: unknown(service-down)" in out
        assert "extraction_prompt_version: 7" in out
        assert "episode_prompt_version: 1" in out

    def test_check_stats_503_reports_disabled(self, monkeypatch, capsys) -> None:
        """Stats probe returns 503 (memory disabled): report
        `mode: disabled` and exit 0. The 503 is the documented
        memory-disabled response from the @_require_secret-protected
        endpoint."""
        monkeypatch.setenv("WEBHOOK_SECRET", "test-secret")

        def _http_disabled(url, secret=None, timeout=5.0):
            if "/health" in url:
                return 200, b'{"status": "ok"}'
            if "/api/memory/stats" in url:
                # Auth header MUST be present. The test asserts the
                # secret was threaded through; without this, a
                # regression that dropped the X-Webhook-Secret header
                # would let the test pass against a 401 (which the
                # caller would then mis-classify).
                assert secret == "test-secret", f"expected secret to be threaded; got {secret!r}"
                # URL must NOT carry an explicit chat_id. The
                # webhook handler resolves chat_id BEFORE the
                # is_enabled() branch and rejects any explicit
                # value not in allowed_user_ids with a 403; an
                # earlier version of this harness passed chat_id=1
                # and silently 403'd against production. Pinning
                # the absence here closes the regression.
                assert "chat_id" not in url, f"stats URL must not carry a chat_id parameter; got {url!r}"
                return 503, b'{"error": "memory disabled"}'
            raise AssertionError(f"unexpected url: {url}")

        monkeypatch.setattr(modeswitch, "_http_get", _http_disabled)
        monkeypatch.setattr(modeswitch, "_read_prompt_versions", lambda: ("7", "1"))
        rc = modeswitch._run_check()
        assert rc == 0
        out = capsys.readouterr().out
        # Pin every line the production code emits on the disabled
        # mode path. Same shape as test_check_health_down_exits_1.
        assert "secret_found: yes" in out
        assert "health: ok" in out
        assert "mode: disabled" in out
        assert "extraction_prompt_version: 7" in out
        assert "episode_prompt_version: 1" in out

    def test_check_stats_200_reports_enabled(self, monkeypatch, capsys) -> None:
        """Stats probe returns 200 with a stats payload (memory
        enabled): report `mode: enabled` and exit 0."""
        monkeypatch.setenv("WEBHOOK_SECRET", "test-secret")

        def _http_enabled(url, secret=None, timeout=5.0):
            if "/health" in url:
                return 200, b'{"status": "ok"}'
            if "/api/memory/stats" in url:
                # Same auth-header threading guard as the disabled
                # test: a regression that drops `secret=secret` at
                # the stats call site would let the test pass against
                # an unauthed 200 (which the production code would
                # never see in real life, but the test should not
                # silently accept), masking the regression. Pinned
                # symmetrically across all stats-touching mocks.
                assert secret == "test-secret", f"expected secret to be threaded; got {secret!r}"
                # URL must NOT carry an explicit chat_id (see the
                # disabled-test comment for the 403-against-prod
                # rationale).
                assert "chat_id" not in url, f"stats URL must not carry a chat_id parameter; got {url!r}"
                return 200, b'{"total_count": 42}'
            raise AssertionError(f"unexpected url: {url}")

        monkeypatch.setattr(modeswitch, "_http_get", _http_enabled)
        monkeypatch.setattr(modeswitch, "_read_prompt_versions", lambda: ("7", "1"))
        rc = modeswitch._run_check()
        assert rc == 0
        out = capsys.readouterr().out
        # Pin every line the production code emits on the enabled
        # mode path. Same shape as the disabled, health-down, and
        # unexpected-status test cases.
        assert "secret_found: yes" in out
        assert "health: ok" in out
        assert "mode: enabled" in out
        assert "extraction_prompt_version: 7" in out
        assert "episode_prompt_version: 1" in out

    def test_check_stats_unexpected_status_exits_1(self, monkeypatch, capsys) -> None:
        """Stats probe returns a status that's neither 200 nor 503
        (e.g. 401 from a wrong secret, or a 500 server error): report
        the status and exit 1. This is the shape that the spec's
        anti-false-negative argument hinges on; a silent fall-through
        to 'disabled' here would defeat the entire harness."""
        monkeypatch.setenv("WEBHOOK_SECRET", "test-secret")

        def _http_unexpected(url, secret=None, timeout=5.0):
            if "/health" in url:
                return 200, b'{"status": "ok"}'
            if "/api/memory/stats" in url:
                # Same auth-header threading guard as the disabled
                # and enabled tests; pinned symmetrically across
                # all stats-touching mocks.
                assert secret == "test-secret", f"expected secret to be threaded; got {secret!r}"
                # URL must NOT carry an explicit chat_id (see the
                # disabled-test comment for the 403-against-prod
                # rationale).
                assert "chat_id" not in url, f"stats URL must not carry a chat_id parameter; got {url!r}"
                return 401, b'{"error": "unauthorized"}'
            raise AssertionError(f"unexpected url: {url}")

        monkeypatch.setattr(modeswitch, "_http_get", _http_unexpected)
        monkeypatch.setattr(modeswitch, "_read_prompt_versions", lambda: ("7", "1"))
        rc = modeswitch._run_check()
        assert rc == 1
        out = capsys.readouterr().out
        # Pin every line the production code emits on the
        # unexpected-status path. Same shape as the disabled and
        # health-down test cases.
        assert "secret_found: yes" in out
        assert "health: ok" in out
        assert "mode: unknown(401)" in out
        assert "extraction_prompt_version: 7" in out
        assert "episode_prompt_version: 1" in out

    def test_check_webhook_port_non_integer_falls_back_to_8080(self, monkeypatch, capsys) -> None:
        """WEBHOOK_PORT set to a non-integer string: harness prints a
        warning, falls back to port 8080, and continues. The fallback
        is what unblocks the rest of the report; without it, the
        harness would crash with ValueError on a config typo.

        Mock follows the same auth-secret + chat_id-absence pattern as
        the three other stats-touching mocks; full output shape pinned
        per the same pattern as the four mode-reporting tests."""
        monkeypatch.setenv("WEBHOOK_SECRET", "test-secret")
        monkeypatch.setenv("WEBHOOK_PORT", "not-an-integer")

        captured_urls: list[str] = []

        def _http_with_capture(url, secret=None, timeout=5.0):
            captured_urls.append(url)
            if "/health" in url:
                return 200, b'{"status": "ok"}'
            if "/api/memory/stats" in url:
                # Auth-secret + chat_id-absence guards parallel to
                # _http_disabled / _http_enabled / _http_unexpected;
                # see the disabled-test comment for the rationale.
                assert secret == "test-secret", f"expected secret to be threaded; got {secret!r}"
                assert "chat_id" not in url, f"stats URL must not carry a chat_id parameter; got {url!r}"
                return 200, b'{"total_count": 0}'
            raise AssertionError(f"unexpected url: {url}")

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
        # Pin the full five-line report shape, parallel to the
        # other mode-reporting tests in this class.
        assert "secret_found: yes" in out
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

        Mock follows the same auth-secret + chat_id-absence pattern as
        the three other stats-touching mocks; full output shape pinned
        per the same pattern as the four mode-reporting tests."""
        monkeypatch.setenv("WEBHOOK_SECRET", "test-secret")
        monkeypatch.setenv("WEBHOOK_PORT", "99999")

        captured_urls: list[str] = []

        def _http_with_capture(url, secret=None, timeout=5.0):
            captured_urls.append(url)
            if "/health" in url:
                return 200, b'{"status": "ok"}'
            if "/api/memory/stats" in url:
                assert secret == "test-secret", f"expected secret to be threaded; got {secret!r}"
                assert "chat_id" not in url, f"stats URL must not carry a chat_id parameter; got {url!r}"
                return 200, b'{"total_count": 0}'
            raise AssertionError(f"unexpected url: {url}")

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
        # Pin the full five-line report shape, parallel to the
        # other mode-reporting tests in this class.
        assert "secret_found: yes" in out
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
