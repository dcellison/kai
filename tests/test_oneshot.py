"""
Tests for `kai.oneshot` - the provider-agnostic one-shot reasoning
boundary used by semantic memory extraction.

Coverage focus: subprocess mechanics that the reasoner owns. The
memory-specific contracts (envelope parsing, schema validation, fact
storage) live in `tests/test_memory_extraction.py` and stay there.
"""

import asyncio
import json
import logging
import os
import pwd
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.oneshot import (
    _CODEX_ENV_ALLOWLIST,
    _GOOSE_ENV_ALLOWLIST,
    _GOOSE_PRESERVED_AUTH_VARS,
    _OPENCODE_PRESERVED_AUTH_VARS,
    _SUBPROCESS_ENV_ALLOWLIST,
    ClaudeOneShotReasoner,
    CodexOneShotReasoner,
    GooseOneShotReasoner,
    OneShotOutputError,
    OneShotResult,
    OneShotRoutingError,
    OneShotSubprocessError,
    OneShotTimeout,
    OpenCodeOneShotReasoner,
    _parse_schema_payload,
    _preserved_auth_vars_for,
    _render_goose_stdin,
    _render_opencode_session_prompt,
)


def _current_user() -> str:
    """OS username of the test runner.

    Tests that exercise the codex reasoner's direct-spawn path use
    this as `os_user` so `resolve_claude_user` self-sudo-skips back
    to None (target matches current process); the reasoner then
    spawns codex directly with no sudo wrap. Both reasoners now
    accept `os_user=None` and follow the same in-process path,
    so this helper is a parity convenience rather than a routing
    requirement.
    """
    return pwd.getpwuid(os.getuid()).pw_name


@pytest.fixture(autouse=True)
def _mock_binary_resolver():
    """Pin the binary resolver to literal backend names so existing
    argv-shape assertions stay valid across host machines. Tests that
    care about resolver behavior itself (e.g. CODEX_BIN override
    validation) patch over this fixture with their own values."""

    def fake_resolve(backend: str) -> str:
        if backend == "claude":
            return "claude"
        if backend == "codex":
            return "codex"
        if backend == "opencode":
            return "opencode"
        if backend == "goose":
            return "goose"
        raise ValueError(f"unknown backend: {backend!r}")

    with patch("kai.oneshot.resolve_oneshot_binary", side_effect=fake_resolve):
        yield


@pytest.fixture(autouse=True)
def _bypass_codex_routing_for_argv_tests(request):
    """The argv/env/schema/output tests in this file run with
    `os_user=_current_user()` so `resolve_claude_user` short-circuits
    to None (the self-sudo-skip path). Codex now treats that the
    same way claude does: spawn in-process, no sudo wrap. But the
    pre-#522 surface those tests assert against expects unwrapped
    argv even on the cross-user path, so this autouse fixture keeps
    both halves working without per-test shim:

      - makes resolve_claude_user a pass-through (so a non-None
        `os_user` survives into argv assembly rather than collapsing
        to the self-sudo-skip None)
      - makes _wrap_cmd_for_user a no-op (so the argv assertions
        that expect cmd[0] == "codex" stay valid; the real wrap
        is exercised explicitly by the tests marked `routing_test`)

    Tests that exercise real routing behavior (the routing classes
    that assert against the production wrap shape) are marked with
    `@pytest.mark.routing_test` and opt out of this bypass."""
    if request.node.get_closest_marker("routing_test"):
        yield
        return
    with (
        patch("kai.oneshot.resolve_claude_user", side_effect=lambda u: u if u else None),
        patch("kai.oneshot._wrap_cmd_for_user", side_effect=lambda cmd, *a, **kw: cmd),
    ):
        yield


def _make_proc(
    *,
    stdout: bytes = b"",
    stderr: bytes = b"",
    returncode: int = 0,
    raise_timeout: bool = False,
) -> MagicMock:
    """Build a mock subprocess for asyncio.create_subprocess_exec.

    `raise_timeout=True` makes `communicate` raise TimeoutError on the
    first await, simulating the wait_for timeout path.
    """
    proc = MagicMock()
    if raise_timeout:
        proc.communicate = AsyncMock(side_effect=TimeoutError())
    else:
        proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.kill = MagicMock()
    proc.wait = AsyncMock(return_value=returncode)
    proc.returncode = returncode
    return proc


class TestClaudeOneShotReasonerArgv:
    """The Claude reasoner's argv must match the pre-refactor memory
    extraction shape: claude --print, schema flag when provided,
    sandboxing flags, no --bare."""

    @pytest.mark.asyncio
    async def test_argv_shape_with_schema_and_system_prompt(self, tmp_path):
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path)
        proc = _make_proc(stdout=b'{"ok": true}')

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(
                prompt="hello",
                system_prompt="you are a test",
                model="claude-haiku-4-5-20251001",
                timeout=30,
                purpose="fact_extraction",
                json_schema={"type": "object", "properties": {}},
            )

        cmd = mock_exec.call_args[0]
        assert cmd[0] == "claude"
        assert cmd[1] == "--print"
        i = cmd.index("--model")
        assert cmd[i + 1] == "claude-haiku-4-5-20251001"
        i = cmd.index("--output-format")
        assert cmd[i + 1] == "json"
        assert "--json-schema" in cmd
        i = cmd.index("--system-prompt")
        assert cmd[i + 1] == "you are a test"
        i = cmd.index("--permission-mode")
        assert cmd[i + 1] == "bypassPermissions"
        i = cmd.index("--tools")
        assert cmd[i + 1] == ""
        assert "--no-session-persistence" in cmd

    @pytest.mark.asyncio
    async def test_argv_omits_bare_flag(self, tmp_path):
        """`--bare` would bypass OAuth and force ANTHROPIC_API_KEY-only auth,
        which would break Max-plan billing for memory extraction. The
        reasoner must never include it."""
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path)
        proc = _make_proc()

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(prompt="p", purpose="fact_extraction")

        cmd = mock_exec.call_args[0]
        assert "--bare" not in cmd

    @pytest.mark.asyncio
    async def test_payload_sent_on_stdin_not_argv(self, tmp_path):
        """Conversation content goes through stdin, never argv. Argv is
        visible via `ps -ef`."""
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path)
        proc = _make_proc()

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            await reasoner.run(prompt="SECRET-USER-MESSAGE", purpose="fact_extraction")

        proc.communicate.assert_awaited_once_with(input=b"SECRET-USER-MESSAGE")

    @pytest.mark.asyncio
    async def test_system_prompt_omitted_when_none(self, tmp_path):
        """A `system_prompt=None` argument means the caller does not want
        to set the flag at all. The reasoner must not emit a stray
        `--system-prompt ""` because that would silently replace the
        Claude default with the empty string (which DOES change
        behavior)."""
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path)
        proc = _make_proc()

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(prompt="p", purpose="fact_extraction", system_prompt=None)

        cmd = mock_exec.call_args[0]
        assert "--system-prompt" not in cmd

    @pytest.mark.asyncio
    async def test_json_schema_omitted_when_none(self, tmp_path):
        """When the caller passes no schema, the reasoner does not emit
        --json-schema. Stage 1 and stage 2 always pass one in
        production, but the protocol allows None and the implementation
        must honor that."""
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path)
        proc = _make_proc()

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(prompt="p", purpose="fact_extraction", json_schema=None)

        cmd = mock_exec.call_args[0]
        assert "--json-schema" not in cmd

    @pytest.mark.asyncio
    async def test_output_format_omitted_in_free_form_mode(self, tmp_path):
        """Free-form mode (json_schema=None) must not pass
        --output-format json: stdout IS the response text, and the
        review / triage callers hand it to their downstream consumers
        without any envelope parse. Emitting the envelope here would
        post raw JSON as a PR comment."""
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path)
        proc = _make_proc()

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(prompt="p", purpose="pr_review", json_schema=None)

        cmd = mock_exec.call_args[0]
        assert "--output-format" not in cmd


class TestClaudeOneShotReasonerEnv:
    """The subprocess env must be allow-listed - the parent's full env
    is NOT inherited, so a regression in --tools "" cannot leak the
    bot's secrets to the model."""

    @pytest.mark.asyncio
    async def test_env_contains_only_allowlisted_keys(self, tmp_path, monkeypatch):
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path)
        proc = _make_proc()

        # Seed the parent env with several secret-looking values and
        # one allow-listed value so the assertion can demonstrate both
        # directions.
        monkeypatch.setenv("DATABASE_URL", "postgres://leak")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_leak")
        monkeypatch.setenv("WEBHOOK_SECRET", "hmac-leak")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(prompt="p", purpose="fact_extraction")

        passed_env = mock_exec.call_args.kwargs["env"]
        # Allow-listed key present.
        assert passed_env.get("PATH") == "/usr/bin:/bin"
        # Secrets absent.
        assert "DATABASE_URL" not in passed_env
        assert "GITHUB_TOKEN" not in passed_env
        assert "WEBHOOK_SECRET" not in passed_env
        # No keys outside the allow-list reached the subprocess.
        assert set(passed_env.keys()) <= set(_SUBPROCESS_ENV_ALLOWLIST)


class TestClaudeOneShotReasonerCwd:
    """The neutral cwd must exist on call (lazy mkdir) and must be the
    path the caller supplied (or the canonical _EXTRACTOR_CWD when
    not overridden)."""

    @pytest.mark.asyncio
    async def test_cwd_is_created_lazily(self, tmp_path):
        target = tmp_path / "nested" / "extractor_cwd"
        assert not target.exists()
        reasoner = ClaudeOneShotReasoner(cwd=target)
        proc = _make_proc()

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(prompt="p", purpose="fact_extraction")

        assert target.is_dir()
        assert mock_exec.call_args.kwargs["cwd"] == str(target)


class TestClaudeOneShotReasonerTimeout:
    """On timeout, the subprocess must be killed and reaped before the
    exception propagates. The wait() prevents a subsequent kill() from
    racing on ProcessLookupError."""

    @pytest.mark.asyncio
    async def test_timeout_kills_and_awaits_then_raises(self, tmp_path):
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path)
        proc = _make_proc(raise_timeout=True)

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(OneShotTimeout),
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction", timeout=0.1)

        proc.kill.assert_called_once()
        proc.wait.assert_awaited()


class TestClaudeOneShotReasonerSubprocessError:
    """Non-zero exit must raise OneShotSubprocessError carrying
    returncode AND stderr bytes so stage 2 can preserve its
    `exit_<code>: <stderr>` failure reason."""

    @pytest.mark.asyncio
    async def test_non_zero_exit_raises_with_returncode_and_stderr(self, tmp_path):
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path)
        proc = _make_proc(stdout=b"", stderr=b"oauth refused", returncode=2)

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(OneShotSubprocessError) as excinfo,
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction")

        assert excinfo.value.returncode == 2
        assert excinfo.value.stderr == b"oauth refused"


class TestOneShotSubprocessErrorStr:
    """str() must carry the exit code and a stderr snippet: review and
    triage embed the exception in their RuntimeError messages, and the
    dataclass-generated __init__ leaves Exception.args empty, which
    would otherwise render a bare prefix with no failure detail."""

    def test_str_includes_exit_code_and_stderr(self):
        err = OneShotSubprocessError(returncode=2, stderr=b"  oauth refused\n")
        assert str(err) == "exit 2: oauth refused"

    def test_str_without_stderr_is_exit_code_only(self):
        err = OneShotSubprocessError(returncode=1, stderr=b"")
        assert str(err) == "exit 1"

    def test_str_bounds_stderr_snippet(self):
        err = OneShotSubprocessError(returncode=1, stderr=b"x" * 500)
        assert str(err) == "exit 1: " + "x" * 200


class TestClaudeOneShotReasonerSuccess:
    """A successful run returns a populated OneShotResult: decoded
    stdout, backend tag, model passthrough, raw_metadata with
    subprocess-layer fields, and a non-negative duration."""

    @pytest.mark.asyncio
    async def test_success_returns_one_shot_result(self, tmp_path):
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path)
        proc = _make_proc(stdout=b'{"text": "hi"}', stderr=b"warning: deprecated", returncode=0)

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                model="claude-haiku-4-5-20251001",
            )

        assert isinstance(result, OneShotResult)
        assert result.text == '{"text": "hi"}'
        assert result.backend == "claude"
        assert result.model == "claude-haiku-4-5-20251001"
        assert result.raw_metadata["returncode"] == 0
        assert result.raw_metadata["stderr"] == b"warning: deprecated"
        assert result.raw_metadata["cwd"] == str(tmp_path)
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_raw_metadata_excludes_parsed_envelope_fields(self, tmp_path):
        """Phase 1 contract: raw_metadata is subprocess-layer only. The
        Claude reasoner must NOT parse the JSON envelope and populate
        envelope-derived fields (is_error, total_cost_usd, etc.); that
        contract belongs to memory_extraction."""
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path)
        envelope = b'{"is_error": false, "total_cost_usd": 0.0023, "subtype": "ok", "result": "{}"}'
        proc = _make_proc(stdout=envelope, returncode=0)

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(prompt="p", purpose="fact_extraction")

        # Phase 1 must keep raw_metadata free of envelope-parsed fields.
        for forbidden in ("is_error", "total_cost_usd", "subtype", "result", "structured_output"):
            assert forbidden not in result.raw_metadata

    @pytest.mark.asyncio
    async def test_free_form_text_is_stripped(self, tmp_path):
        """Free-form mode (json_schema=None) returns the response text
        itself, stripped of surrounding whitespace - the same
        free-form contract the goose reasoner exposes, and what the
        review / triage callers post or parse directly."""
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path)
        proc = _make_proc(stdout=b"  the review body  \n")

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(prompt="p", purpose="pr_review", json_schema=None)

        assert result.text == "the review body"

    @pytest.mark.asyncio
    async def test_schema_mode_text_is_unstripped(self, tmp_path):
        """Schema mode hands raw stdout back unmodified (no strip):
        the envelope parse downstream is the caller's contract, and
        the memory path must stay byte-identical to its pre-refactor
        behavior."""
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path)
        proc = _make_proc(stdout=b'{"result": "x"}\n')

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={"type": "object", "properties": {}},
            )

        assert result.text == '{"result": "x"}\n'


class TestClaudeOneShotReasonerLogging:
    """Each call emits one structured INFO line keyed by purpose, with
    distinct outcomes for success / timeout / subprocess_error."""

    @pytest.mark.asyncio
    async def test_log_line_includes_purpose_and_outcome_success(self, tmp_path, caplog):
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path)
        proc = _make_proc(stdout=b"{}", returncode=0)

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            caplog.at_level(logging.INFO, logger="kai.oneshot"),
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction")

        # One INFO line; contains the structured fields the spec asks for.
        records = [r for r in caplog.records if r.message.startswith("oneshot_reasoner")]
        assert len(records) == 1
        msg = records[0].getMessage()
        assert "purpose=fact_extraction" in msg
        assert "backend=claude" in msg
        assert "outcome=success" in msg

    @pytest.mark.asyncio
    async def test_log_line_carries_purpose_on_timeout(self, tmp_path, caplog):
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path)
        proc = _make_proc(raise_timeout=True)

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            caplog.at_level(logging.INFO, logger="kai.oneshot"),
            pytest.raises(OneShotTimeout),
        ):
            await reasoner.run(prompt="p", purpose="episode_generation", timeout=0.1)

        records = [r for r in caplog.records if r.message.startswith("oneshot_reasoner")]
        assert len(records) == 1
        msg = records[0].getMessage()
        assert "purpose=episode_generation" in msg
        assert "outcome=timeout" in msg

    @pytest.mark.asyncio
    async def test_log_line_carries_purpose_on_subprocess_error(self, tmp_path, caplog):
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path)
        proc = _make_proc(stderr=b"refused", returncode=3)

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            caplog.at_level(logging.INFO, logger="kai.oneshot"),
            pytest.raises(OneShotSubprocessError),
        ):
            await reasoner.run(prompt="p", purpose="episode_generation")

        records = [r for r in caplog.records if r.message.startswith("oneshot_reasoner")]
        assert len(records) == 1
        msg = records[0].getMessage()
        assert "purpose=episode_generation" in msg
        assert "outcome=subprocess_error" in msg
        assert "returncode=3" in msg


# ── Codex reasoner ──────────────────────────────────────────────────


def _codex_event(item_type: str, text: str | None = None) -> str:
    """Serialize one NDJSON event the way `codex exec --json` emits.

    The reasoner extracts text by walking `item.completed` events;
    other event types in the stream are silently skipped, so the
    helper only needs to produce a well-formed agent_message item.
    """
    item: dict = {"id": "item-1", "type": item_type}
    if text is not None:
        item["text"] = text
    return json.dumps({"type": "item.completed", "item": item})


def _codex_envelope_ndjson(payload: dict) -> bytes:
    """Helper: one NDJSON line wrapping `payload` as an agent_message."""
    return _codex_event("agent_message", json.dumps(payload)).encode("utf-8") + b"\n"


class TestCodexOneShotReasonerArgv:
    """The codex argv must reflect the spec's flag set: exec mode,
    JSON event output, trusted-dir skip, ephemeral session, isolated
    config/rules, a fail-closed read-only permission profile, disabled
    tool surfaces, neutral cwd, model when provided, and --output-schema
    with a real path only when a schema is supplied."""

    @pytest.mark.asyncio
    async def test_argv_includes_required_flags(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CODEX_BIN", raising=False)
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_proc(stdout=_codex_envelope_ndjson({"ok": True}))

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(
                prompt="hello",
                system_prompt="you are a test",
                model="gpt-5.4-mini",
                timeout=30,
                purpose="fact_extraction",
                json_schema={"type": "object", "properties": {}},
            )

        cmd = mock_exec.call_args[0]
        assert cmd[0] == "codex"
        assert cmd[1] == "exec"
        assert "--json" in cmd
        assert "--skip-git-repo-check" in cmd
        assert "--ephemeral" in cmd
        assert "--ignore-user-config" in cmd
        assert "--ignore-rules" in cmd
        assert "--strict-config" in cmd
        config_values = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--config"]
        assert 'approval_policy="never"' in config_values
        assert 'default_permissions="kai-oneshot"' in config_values
        permission_config = next(value for value in config_values if value.startswith("permissions.kai-oneshot="))
        assert '":minimal"="read"' in permission_config
        assert '":workspace_roots"={"."="read"}' in permission_config
        assert "network={enabled=false}" in permission_config
        assert 'web_search="disabled"' in config_values
        assert "mcp_servers={}" in config_values

        disabled = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--disable"]
        assert {
            "shell_tool",
            "unified_exec",
            "apps",
            "plugins",
            "remote_plugin",
            "browser_use",
            "browser_use_external",
            "computer_use",
            "image_generation",
            "multi_agent",
            "hooks",
            "skill_search",
            "skill_mcp_dependency_install",
            "view_image",
        } <= set(disabled)
        # Permission profiles and the legacy --sandbox flag do not
        # compose: adding --sandbox would silently override this
        # narrower profile.
        assert "--sandbox" not in cmd
        i = cmd.index("--cd")
        assert cmd[i + 1] == str(tmp_path)
        i = cmd.index("--model")
        assert cmd[i + 1] == "gpt-5.4-mini"
        # --output-schema must point at a real file the CLI can open.
        i = cmd.index("--output-schema")
        schema_path = Path(cmd[i + 1])
        # The file is removed in the finally block AFTER the
        # subprocess exits; this assertion only proves the path
        # parameter is wired up, not lifecycle - that is covered by
        # TestCodexOneShotReasonerSchemaFile below.
        assert schema_path.parent == tmp_path

    @pytest.mark.asyncio
    async def test_argv_honors_codex_bin_env(self, tmp_path, monkeypatch):
        """CODEX_BIN must override the bare `codex` resolution so
        per-os_user homebrew installs work without PATH munging.
        Bypasses the autouse resolver mock to exercise the real
        resolver path; the test points CODEX_BIN at an executable
        file in tmp_path so the resolver's is-file + executable
        checks pass."""
        fake_codex = tmp_path / "codex-bin"
        fake_codex.write_text("#!/bin/sh\nexit 0\n")
        fake_codex.chmod(0o755)
        monkeypatch.setenv("CODEX_BIN", str(fake_codex))
        # Bypass the autouse fixture for this single test by patching
        # the real resolver back into place. The autouse fixture pins
        # to literal "codex"; here we want the real CODEX_BIN
        # honoring behavior to ground the assertion.
        from kai.oneshot_binary import resolve_oneshot_binary as real_resolver

        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_proc(stdout=_codex_envelope_ndjson({"ok": True}))

        with (
            patch("kai.oneshot.resolve_oneshot_binary", side_effect=real_resolver),
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec,
        ):
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={"type": "object"},
            )

        cmd = mock_exec.call_args[0]
        assert cmd[0] == str(fake_codex)

    @pytest.mark.asyncio
    async def test_argv_omits_output_schema_without_schema(self, tmp_path):
        """When the caller passes no schema, --output-schema must be
        absent entirely. The non-schema branch returns raw final text
        and never writes a temp file."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_proc(stdout=_codex_event("agent_message", "free-form text").encode("utf-8") + b"\n")

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(prompt="p", purpose="fact_extraction", json_schema=None)

        cmd = mock_exec.call_args[0]
        assert "--output-schema" not in cmd

    @pytest.mark.asyncio
    async def test_argv_omits_model_when_none(self, tmp_path):
        """A None model means the codex CLI's default; the reasoner
        must not emit a stray `--model` with an empty string."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_proc(stdout=_codex_envelope_ndjson({"ok": True}))

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={"type": "object"},
            )

        cmd = mock_exec.call_args[0]
        assert "--model" not in cmd

    @pytest.mark.asyncio
    async def test_payload_sent_on_stdin_not_argv(self, tmp_path):
        """User content goes through stdin only. Argv is visible via
        `ps -ef`; the rendered stdin block carries both the system
        prompt boundary and the user payload."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_proc(stdout=_codex_envelope_ndjson({"ok": True}))

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(
                prompt="SECRET-USER-MESSAGE",
                system_prompt="SYSTEM-INSTRUCTIONS",
                purpose="fact_extraction",
                json_schema={"type": "object"},
            )

        # Argv carries no conversation content.
        cmd = mock_exec.call_args[0]
        for arg in cmd:
            assert "SECRET-USER-MESSAGE" not in arg
            assert "SYSTEM-INSTRUCTIONS" not in arg
        # Stdin carries both pieces.
        stdin_bytes = proc.communicate.call_args.kwargs["input"]
        decoded = stdin_bytes.decode("utf-8")
        assert "SECRET-USER-MESSAGE" in decoded
        assert "SYSTEM-INSTRUCTIONS" in decoded


class TestCodexOneShotReasonerStdin:
    """`_render_codex_stdin` semantics: a non-empty system prompt is
    wrapped in a randomized boundary; None / empty produces the user
    payload unchanged so callers do not emit an empty SYSTEM block."""

    @pytest.mark.asyncio
    async def test_stdin_wraps_system_prompt_in_boundary_when_provided(self, tmp_path):
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_proc(stdout=_codex_envelope_ndjson({"ok": True}))

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            await reasoner.run(
                prompt="USER-PAYLOAD",
                system_prompt="SYSTEM-INSTRUCTIONS",
                purpose="fact_extraction",
                json_schema={"type": "object"},
            )

        stdin_bytes = proc.communicate.call_args.kwargs["input"]
        decoded = stdin_bytes.decode("utf-8")
        # Boundary markers wrap the system prompt; the user payload
        # follows after a blank line.
        assert "--- BEGIN SYSTEM" in decoded
        assert "--- END SYSTEM" in decoded
        assert decoded.index("SYSTEM-INSTRUCTIONS") < decoded.index("USER-PAYLOAD")

    @pytest.mark.asyncio
    async def test_stdin_omits_boundary_when_system_prompt_is_none(self, tmp_path):
        """No system prompt means no SYSTEM block at all; the user
        payload reaches codex unchanged, matching how Claude's
        omitted --system-prompt flag behaves."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_proc(stdout=_codex_envelope_ndjson({"ok": True}))

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            await reasoner.run(
                prompt="USER-PAYLOAD",
                system_prompt=None,
                purpose="fact_extraction",
                json_schema={"type": "object"},
            )

        stdin_bytes = proc.communicate.call_args.kwargs["input"]
        decoded = stdin_bytes.decode("utf-8")
        assert "--- BEGIN SYSTEM" not in decoded
        assert decoded == "USER-PAYLOAD"

    @pytest.mark.asyncio
    async def test_stdin_omits_boundary_when_system_prompt_is_empty(self, tmp_path):
        """Empty-string system prompt collapses to the same no-boundary
        behavior; a free-floating SYSTEM section with empty content
        would be visible to the model with nothing inside it."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_proc(stdout=_codex_envelope_ndjson({"ok": True}))

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            await reasoner.run(
                prompt="USER-PAYLOAD",
                system_prompt="",
                purpose="fact_extraction",
                json_schema={"type": "object"},
            )

        stdin_bytes = proc.communicate.call_args.kwargs["input"]
        decoded = stdin_bytes.decode("utf-8")
        assert "--- BEGIN SYSTEM" not in decoded
        assert decoded == "USER-PAYLOAD"


class TestCodexOneShotReasonerSchemaFile:
    """The schema temp file must exist for the duration of the
    subprocess call and be removed on every exit path (success,
    timeout, non-zero exit, output error). A leaked file under the
    cwd would accumulate across calls."""

    @pytest.mark.asyncio
    async def test_schema_file_written_with_supplied_schema(self, tmp_path):
        """The path passed via --output-schema points at a file
        whose contents are the SANITIZED form of the schema dict
        (per the OpenAI strict-mode boundary transform). The
        sanitizer adds `additionalProperties: false` and lists the
        property in `required`; the on-disk JSON reflects that."""
        from kai.oneshot import _sanitize_for_codex

        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        captured: dict = {}

        def _capture(*args, **kwargs):
            i = args.index("--output-schema")
            captured["path"] = Path(args[i + 1])
            captured["body"] = captured["path"].read_text(encoding="utf-8")
            return _make_proc(stdout=_codex_envelope_ndjson({"ok": True}))

        schema = {"type": "object", "properties": {"facts": {"type": "array"}}}
        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(side_effect=_capture)):
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema=schema,
            )

        assert json.loads(captured["body"]) == _sanitize_for_codex(schema)
        # Cleanup ran in the finally block.
        assert not captured["path"].exists()

    @pytest.mark.asyncio
    async def test_schema_file_removed_on_timeout(self, tmp_path):
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        captured: dict = {}

        def _capture(*args, **kwargs):
            # Multiple subprocess spawns may fire on the timeout
            # path: the codex run itself, and the cross-user kill
            # tree. Only the first carries --output-schema; the
            # kill subprocess's argv is the sudo+kill shape.
            if "--output-schema" in args:
                i = args.index("--output-schema")
                captured["path"] = Path(args[i + 1])
                return _make_proc(raise_timeout=True)
            # Kill subprocess: return a noop mock so the kill helper
            # awaits and returns cleanly.
            return _make_proc()

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(side_effect=_capture)),
            pytest.raises(OneShotTimeout),
        ):
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                timeout=0.1,
                json_schema={"type": "object"},
            )

        assert not captured["path"].exists()

    @pytest.mark.asyncio
    async def test_schema_file_removed_on_subprocess_error(self, tmp_path):
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        captured: dict = {}

        def _capture(*args, **kwargs):
            i = args.index("--output-schema")
            captured["path"] = Path(args[i + 1])
            return _make_proc(stdout=b"", stderr=b"refused", returncode=2)

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(side_effect=_capture)),
            pytest.raises(OneShotSubprocessError),
        ):
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={"type": "object"},
            )

        assert not captured["path"].exists()

    @pytest.mark.asyncio
    async def test_schema_file_removed_on_output_error(self, tmp_path):
        """Empty final agent_message under a schema-backed call must
        raise OneShotOutputError AND remove the temp file. Without
        the cleanup, repeat failures would leak one file per call."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        captured: dict = {}

        def _capture(*args, **kwargs):
            i = args.index("--output-schema")
            captured["path"] = Path(args[i + 1])
            # No item.completed events at all -> extract_codex_text
            # returns empty string -> OneShotOutputError fires.
            return _make_proc(stdout=b"", returncode=0)

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(side_effect=_capture)),
            pytest.raises(OneShotOutputError),
        ):
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={"type": "object"},
            )

        assert not captured["path"].exists()


class TestCodexOneShotReasonerEnv:
    """The codex env allow-list is separate from the Claude one:
    Anthropic vars are NOT forwarded, and only PATH / HOME /
    CODEX_HOME / OPENAI_API_KEY / OPENAI_BASE_URL reach the
    subprocess. A regression that started forwarding the parent's
    full env would leak the bot's secrets to the model."""

    @pytest.mark.asyncio
    async def test_env_contains_only_codex_allowlisted_keys(self, tmp_path, monkeypatch):
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_proc(stdout=_codex_envelope_ndjson({"ok": True}))

        # Allow-listed keys present in parent env.
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        monkeypatch.setenv("HOME", "/Users/test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        # Secret-looking values that MUST NOT reach the subprocess.
        monkeypatch.setenv("DATABASE_URL", "postgres://leak")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_leak")
        monkeypatch.setenv("WEBHOOK_SECRET", "hmac-leak")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-leak")
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/tmp/claude-leak")

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={"type": "object"},
            )

        passed_env = mock_exec.call_args.kwargs["env"]
        assert passed_env.get("PATH") == "/usr/bin:/bin"
        assert passed_env.get("HOME") == "/Users/test"
        assert passed_env.get("OPENAI_API_KEY") == "sk-test"
        # Claude-only and bot-internal secrets are absent.
        assert "ANTHROPIC_API_KEY" not in passed_env
        assert "CLAUDE_CONFIG_DIR" not in passed_env
        assert "DATABASE_URL" not in passed_env
        assert "GITHUB_TOKEN" not in passed_env
        assert "WEBHOOK_SECRET" not in passed_env
        assert set(passed_env.keys()) <= set(_CODEX_ENV_ALLOWLIST)


class TestCodexOneShotReasonerTimeout:
    """Timeout path mirrors Claude: kill + await + raise OneShotTimeout.
    The duration measurement is around wait_for so a slow-reap codex
    process does not inflate the logged duration."""

    @pytest.mark.asyncio
    async def test_timeout_kills_and_awaits_then_raises(self, tmp_path):
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_proc(raise_timeout=True)
        # The cross-user kill path also spawns a subprocess (sudo
        # /bin/kill -KILL -<pgid>). Returning the same timing-out
        # proc for that spawn would have proc.kill called twice.
        # Differentiate on argv: the kill subprocess starts with
        # "sudo" + "-n", the reasoner spawn starts differently.
        kill_proc = _make_proc()

        def _route(*args, **kwargs):
            if args and args[0] == "sudo" and len(args) > 1 and args[1] == "-n":
                return kill_proc
            return proc

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(side_effect=_route)),
            pytest.raises(OneShotTimeout),
        ):
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                timeout=0.1,
                json_schema={"type": "object"},
            )

        proc.kill.assert_called_once()
        proc.wait.assert_awaited()


class TestCodexOneShotReasonerSubprocessError:
    """Non-zero exit raises OneShotSubprocessError with the returncode
    AND stderr bytes so stage 2 can reconstruct its
    `exit_<code>: <stderr>` failure reason format unchanged."""

    @pytest.mark.asyncio
    async def test_non_zero_exit_raises_with_returncode_and_stderr(self, tmp_path):
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_proc(stdout=b"", stderr=b"codex auth failed", returncode=2)

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(OneShotSubprocessError) as excinfo,
        ):
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={"type": "object"},
            )

        assert excinfo.value.returncode == 2
        assert excinfo.value.stderr == b"codex auth failed"


class TestCodexOneShotReasonerOutputError:
    """Schema-backed calls require a parseable JSON final message.
    Empty output, malformed JSON, and turn.failed events all collapse
    to OneShotOutputError so memory_extraction sees a typed reasoner
    failure instead of half-shaped data reaching the validator."""

    @pytest.mark.asyncio
    async def test_empty_final_message_under_schema_raises_output_error(self, tmp_path):
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_proc(stdout=b"", returncode=0)

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(OneShotOutputError),
        ):
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={"type": "object"},
            )

    @pytest.mark.asyncio
    async def test_turn_failed_under_schema_raises_output_error(self, tmp_path):
        """A `turn.failed` event short-circuits extract_codex_text to
        empty string; the reasoner must then raise OneShotOutputError
        rather than wrap empty content in the envelope."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        stdout = json.dumps({"type": "turn.failed", "error": "x"}).encode("utf-8") + b"\n"
        proc = _make_proc(stdout=stdout, returncode=0)

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(OneShotOutputError),
        ):
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={"type": "object"},
            )

    @pytest.mark.asyncio
    async def test_malformed_final_json_under_schema_raises_output_error(self, tmp_path):
        """A non-JSON agent_message under a schema-backed call cannot
        be wrapped in `{"is_error": false, "structured_output": ...}`;
        OneShotOutputError fires so the caller sees a typed failure
        instead of a downstream parse error on the envelope."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        # agent_message text is "not json at all" - extract_codex_text
        # returns that string, then the JSON parse raises.
        stdout = _codex_event("agent_message", "not json at all").encode("utf-8") + b"\n"
        proc = _make_proc(stdout=stdout, returncode=0)

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(OneShotOutputError),
        ):
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={"type": "object"},
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "non_object_payload",
        [
            '"a plain string"',
            "[]",
            "[1, 2, 3]",
            "42",
            "true",
            "null",
        ],
    )
    async def test_non_object_json_under_schema_raises_output_error(self, tmp_path, non_object_payload):
        """A final agent_message that parses as JSON but is not an
        object must still raise OneShotOutputError. Codex's
        `--output-schema` enforcement is best-effort; without this
        guard, a string / list / scalar would wrap into a
        syntactically valid `is_error: false` envelope and the
        caller would silently store zero facts as if the model had
        legitimately found nothing. The schema-failure case must be
        distinguishable from the empty-extraction case."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        stdout = _codex_event("agent_message", non_object_payload).encode("utf-8") + b"\n"
        proc = _make_proc(stdout=stdout, returncode=0)

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(OneShotOutputError),
        ):
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={"type": "object"},
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "wrong_key_object",
        [
            {"unexpected": True},
            {"episode": {"goal": "x"}},
            {"facts": []},  # missing has_episode
            {"has_episode": False},  # missing facts
            {},
        ],
    )
    async def test_object_missing_required_fields_raises_output_error(self, tmp_path, wrong_key_object):
        """A parseable JSON object that does not satisfy the supplied
        schema's top-level `required` list must raise
        OneShotOutputError. Codex's --output-schema enforcement does
        not hard-reject these payloads, so the reasoner is the last
        boundary between a model that ignored the schema and the
        memory extractor treating wrong-shape output as `the model
        found nothing`. The check reads `required` off the supplied
        schema rather than embedding fact/episode field names so
        the reasoner stays domain-neutral."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        # Fact-extraction schema requires facts AND has_episode at root.
        fact_schema = {
            "type": "object",
            "required": ["facts", "has_episode"],
            "properties": {
                "facts": {"type": "array"},
                "has_episode": {"type": "boolean"},
            },
        }
        stdout = _codex_envelope_ndjson(wrong_key_object)
        proc = _make_proc(stdout=stdout, returncode=0)

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(OneShotOutputError),
        ):
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema=fact_schema,
            )

    @pytest.mark.asyncio
    async def test_missing_required_fields_logs_dedicated_category(self, tmp_path, caplog):
        """Operator dashboards must be able to separate `wrong-shape
        object` from `bad JSON` and from `non-object scalar`; each
        gets its own error_category so log queries can partition the
        failure modes without re-parsing the message body."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        stdout = _codex_envelope_ndjson({"unexpected": True})
        proc = _make_proc(stdout=stdout, returncode=0)
        schema = {"type": "object", "required": ["facts", "has_episode"]}

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            caplog.at_level(logging.INFO, logger="kai.oneshot"),
            pytest.raises(OneShotOutputError),
        ):
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema=schema,
            )

        records = [r for r in caplog.records if r.message.startswith("oneshot_reasoner")]
        assert len(records) == 1
        msg = records[0].getMessage()
        assert "outcome=output_error" in msg
        assert "error_category=missing_required_fields" in msg

    @pytest.mark.asyncio
    async def test_schema_without_required_field_does_not_block_success(self, tmp_path):
        """A schema that omits the top-level `required` list must not
        force a rejection; the reasoner falls back to its baseline
        contract (object payload) and accepts the payload. This keeps
        the guard from over-rejecting schemas that use other JSON
        Schema constructs (oneOf, allOf, etc.) to express constraints
        instead of a flat `required` list."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        stdout = _codex_envelope_ndjson({"any": "object", "shape": True})
        proc = _make_proc(stdout=stdout, returncode=0)
        schema_without_required = {"type": "object", "properties": {"any": {"type": "string"}}}

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema=schema_without_required,
            )

        envelope = json.loads(result.text)
        assert envelope["is_error"] is False
        assert envelope["structured_output"] == {"any": "object", "shape": True}

    @pytest.mark.asyncio
    async def test_non_object_json_logs_non_object_category(self, tmp_path, caplog):
        """The non-object-JSON path must emit its own log category so
        operator-side dashboards can separate "model returned wrong
        shape" from "model returned malformed JSON" without re-parsing
        the message body."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        stdout = _codex_event("agent_message", '"a plain string"').encode("utf-8") + b"\n"
        proc = _make_proc(stdout=stdout, returncode=0)

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            caplog.at_level(logging.INFO, logger="kai.oneshot"),
            pytest.raises(OneShotOutputError),
        ):
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={"type": "object"},
            )

        records = [r for r in caplog.records if r.message.startswith("oneshot_reasoner")]
        assert len(records) == 1
        msg = records[0].getMessage()
        assert "outcome=output_error" in msg
        assert "error_category=non_object_json" in msg


class TestCodexOneShotReasonerSuccess:
    """Valid schema-backed output: the reasoner returns the normalized
    envelope as `OneShotResult.text` so memory_extraction's existing
    `structured_output` parser path works without a codex branch."""

    @pytest.mark.asyncio
    async def test_schema_backed_success_returns_normalized_envelope(self, tmp_path):
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        payload = {"facts": [{"content": "x"}], "has_episode": False}
        stdout = _codex_envelope_ndjson(payload)
        proc = _make_proc(stdout=stdout, stderr=b"", returncode=0)

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                model="gpt-5.4-mini",
                json_schema={"type": "object"},
            )

        assert isinstance(result, OneShotResult)
        assert result.backend == "codex"
        assert result.model == "gpt-5.4-mini"
        envelope = json.loads(result.text)
        assert envelope == {"is_error": False, "structured_output": payload}
        assert result.raw_metadata["returncode"] == 0
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_multi_item_uses_last_completed(self, tmp_path):
        """Multi-item streams under schema callers must parse only the
        LAST `item.completed` agent_message. Joining a preamble with
        the JSON body (the `join_items=True` path) would produce a
        non-JSON string and trip the OneShotOutputError branch."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        preamble = _codex_event("agent_message", "preamble text, not JSON")
        payload = {"facts": []}
        final = _codex_event("agent_message", json.dumps(payload))
        stdout = (preamble + "\n" + final + "\n").encode("utf-8")
        proc = _make_proc(stdout=stdout, returncode=0)

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={"type": "object"},
            )

        envelope = json.loads(result.text)
        assert envelope == {"is_error": False, "structured_output": payload}

    @pytest.mark.asyncio
    async def test_schema_backed_recovers_fenced_final_message(self, tmp_path):
        """`--output-schema` shaping is best-effort, so a model that
        fences its final JSON must still reach the envelope; the
        codex parse runs through the same tolerant helper as
        opencode."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        payload = {"facts": [], "has_episode": False}
        fenced = "```json\n" + json.dumps(payload) + "\n```"
        stdout = (_codex_event("agent_message", fenced) + "\n").encode("utf-8")
        proc = _make_proc(stdout=stdout, returncode=0)

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={"type": "object"},
            )

        envelope = json.loads(result.text)
        assert envelope == {"is_error": False, "structured_output": payload}

    @pytest.mark.asyncio
    async def test_non_schema_success_returns_raw_text(self, tmp_path):
        """A None schema means the caller wants the free-form
        agent_message text back unchanged; no envelope wrapping
        happens. Review and triage are the production callers of
        this branch (memory extraction always supplies a schema)."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        stdout = _codex_event("agent_message", "free-form reply").encode("utf-8") + b"\n"
        proc = _make_proc(stdout=stdout, returncode=0)

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema=None,
            )

        assert result.text == "free-form reply"
        assert result.backend == "codex"

    @pytest.mark.asyncio
    async def test_non_schema_default_keeps_last_message_only(self, tmp_path):
        """Without join_items, the free-form path keeps the last-wins
        extraction: triage's downstream parser expects exactly one
        JSON object, and a preamble agent_message joined ahead of it
        would corrupt the parse."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        first = _codex_event("agent_message", "preamble text")
        second = _codex_event("agent_message", '{"labels": []}')
        proc = _make_proc(stdout=(first + "\n" + second + "\n").encode("utf-8"), returncode=0)

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(
                prompt="p",
                purpose="issue_triage",
                json_schema=None,
            )

        assert result.text == '{"labels": []}'

    @pytest.mark.asyncio
    async def test_non_schema_join_items_joins_all_messages(self, tmp_path):
        """With join_items=True, the free-form path joins every
        completed agent_message with a blank-line separator. This is
        the review contract: a turn that emits a preamble and then
        the body must surface BOTH in the posted markdown; last-wins
        would silently truncate the review to the final item."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user(), join_items=True)
        first = _codex_event("agent_message", "First finding: foo.py has a bug.")
        second = _codex_event("agent_message", "Second finding: bar.py needs a docstring.")
        proc = _make_proc(stdout=(first + "\n" + second + "\n").encode("utf-8"), returncode=0)

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(
                prompt="p",
                purpose="pr_review",
                json_schema=None,
            )

        assert result.text == "First finding: foo.py has a bug.\n\nSecond finding: bar.py needs a docstring."

    @pytest.mark.asyncio
    async def test_schema_mode_ignores_join_items(self, tmp_path):
        """Schema-backed calls pin last-wins even when the reasoner
        was constructed with join_items=True: the JSON body must
        never get a preamble glued ahead of it, regardless of how
        the free-form flag is set."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user(), join_items=True)
        preamble = _codex_event("agent_message", "preamble text, not JSON")
        payload = {"facts": []}
        final = _codex_event("agent_message", json.dumps(payload))
        proc = _make_proc(stdout=(preamble + "\n" + final + "\n").encode("utf-8"), returncode=0)

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={"type": "object"},
            )

        envelope = json.loads(result.text)
        assert envelope == {"is_error": False, "structured_output": payload}


class TestCodexOneShotReasonerLogging:
    """Each codex call emits one structured INFO line keyed by purpose,
    with `backend=codex` and distinct outcomes for success / timeout
    / subprocess_error. Returncode appears on the subprocess_error
    line so log queries can partition by exit code."""

    @pytest.mark.asyncio
    async def test_log_line_includes_backend_codex_on_success(self, tmp_path, caplog):
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_proc(stdout=_codex_envelope_ndjson({"ok": True}), returncode=0)

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            caplog.at_level(logging.INFO, logger="kai.oneshot"),
        ):
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={"type": "object"},
            )

        records = [r for r in caplog.records if r.message.startswith("oneshot_reasoner")]
        assert len(records) == 1
        msg = records[0].getMessage()
        assert "purpose=fact_extraction" in msg
        assert "backend=codex" in msg
        assert "outcome=success" in msg
        # Schema-backed success is the production memory path; the
        # routing log field must be present so operator-side log
        # review can confirm the subprocess ran under the intended
        # OS user. Under the autouse bypass, resolve_claude_user
        # passes the os_user through unchanged (no longer treated as
        # self-sudo-skip), so the field carries the actual user name.
        assert f"os_user={_current_user()}" in msg

    @pytest.mark.asyncio
    async def test_log_line_carries_target_os_user_on_wrapped_success(self, tmp_path, caplog):
        """The cross-user wrap path's schema-backed success log
        emits `os_user=<target>` (not `self`), so a production
        codex memory run can be greppable for the routing target
        per the operator smoke check. Regression test for the
        log-field gap caught in PR #504 review round 1."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user="other-user")
        proc = _make_proc(stdout=_codex_envelope_ndjson({"ok": True}), returncode=0)

        with (
            patch("kai.oneshot.resolve_claude_user", return_value="other-user"),
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            caplog.at_level(logging.INFO, logger="kai.oneshot"),
        ):
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={"type": "object"},
            )

        records = [r for r in caplog.records if r.message.startswith("oneshot_reasoner")]
        assert len(records) == 1
        msg = records[0].getMessage()
        assert "outcome=success" in msg
        assert "os_user=other-user" in msg

    @pytest.mark.asyncio
    async def test_log_line_carries_returncode_on_subprocess_error(self, tmp_path, caplog):
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_proc(stderr=b"refused", returncode=3)

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            caplog.at_level(logging.INFO, logger="kai.oneshot"),
            pytest.raises(OneShotSubprocessError),
        ):
            await reasoner.run(
                prompt="p",
                purpose="episode_generation",
                json_schema={"type": "object"},
            )

        records = [r for r in caplog.records if r.message.startswith("oneshot_reasoner")]
        assert len(records) == 1
        msg = records[0].getMessage()
        assert "backend=codex" in msg
        assert "outcome=subprocess_error" in msg
        assert "returncode=3" in msg


# ── Per-user OS routing (issue #503) ────────────────────────────────


@pytest.mark.routing_test
class TestRoutingArgvAndPreserveEnv:
    """`os_user` constructor argument controls the sudo wrap.

    Direct spawn (self-sudo-skip case): when `os_user` matches the
    current process user, `resolve_claude_user` normalizes to None
    and the reasoner spawns without a wrap. Cross-user spawn: the
    target differs from the current user, so the wrap fires with
    `--preserve-env=<auth-csv>` carrying the per-backend auth vars.
    """

    @pytest.mark.asyncio
    async def test_claude_direct_spawn_when_os_user_is_current(self, tmp_path):
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_proc(stdout=b"{}")
        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(prompt="p", purpose="fact_extraction")
        cmd = mock_exec.call_args[0]
        assert "sudo" not in cmd
        assert cmd[0] == "claude"
        assert mock_exec.call_args.kwargs["start_new_session"] is False

    @pytest.mark.asyncio
    async def test_claude_wraps_when_os_user_differs(self, tmp_path):
        """A target other than the current user produces the wrap.

        `resolve_claude_user` is patched directly so the assertion
        does not depend on which OS user the test runner happens
        to be.
        """
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path, os_user="other-user")
        proc = _make_proc(stdout=b"{}")
        with (
            patch("kai.oneshot.resolve_claude_user", return_value="other-user"),
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec,
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction")
        cmd = mock_exec.call_args[0]
        assert cmd[:6] == ("sudo", "-H", "-D", str(tmp_path), "-u", "other-user")
        # Preserve-env CSV exact contract: Claude auth vars only;
        # no HOME (covered by -H), no PATH (covered by allow-list).
        assert cmd[6] == "--preserve-env=CLAUDE_CONFIG_DIR,ANTHROPIC_API_KEY,ANTHROPIC_BASE_URL"
        assert cmd[7] == "--"
        assert cmd[8] == "claude"
        assert mock_exec.call_args.kwargs["cwd"] is None
        assert mock_exec.call_args.kwargs["start_new_session"] is True

    @pytest.mark.asyncio
    async def test_codex_runs_in_process_when_os_user_matches_bot(self, tmp_path):
        """Codex follows claude's `resolve_claude_user` symmetry
        (issue #522): when `os_user` resolves to the current process
        user, the argv stays direct - no sudo wrap. This is the
        same self-sudo-skip path claude uses. Pins the matches-bot-
        user branch so a regression that re-introduces the
        unconditional `_wrap_cmd_for_user` call fails here."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_proc(stdout=b"")
        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec,
            pytest.raises(OneShotOutputError),
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction", json_schema={"type": "object"})
        cmd = mock_exec.call_args[0]
        assert cmd[0] != "sudo"
        # start_new_session only fires on the wrap path; pin the
        # in-process default for parity with the persistent codex
        # chat backend.
        assert mock_exec.call_args.kwargs["start_new_session"] is False

    @pytest.mark.asyncio
    async def test_codex_wraps_with_preserve_env(self, tmp_path):
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user="other-user")
        proc = _make_proc(stdout=b"")
        with (
            patch("kai.oneshot.resolve_claude_user", return_value="other-user"),
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec,
            pytest.raises(OneShotOutputError),
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction", json_schema={"type": "object"})
        cmd = mock_exec.call_args[0]
        assert cmd[:6] == ("sudo", "-H", "-D", str(tmp_path), "-u", "other-user")
        assert cmd[6] == "--preserve-env=CODEX_HOME,OPENAI_API_KEY,OPENAI_BASE_URL"
        assert cmd[7] == "--"
        # PATH and bare HOME deliberately not in the preserve list
        # (CODEX_HOME is intentional; the substring check splits on
        # the CSV so a future regression that adds bare HOME still
        # fails this assertion).
        preserved = cmd[6].split("=", 1)[1].split(",")
        assert "PATH" not in preserved
        assert "HOME" not in preserved
        assert mock_exec.call_args.kwargs["cwd"] is None
        assert mock_exec.call_args.kwargs["start_new_session"] is True


@pytest.mark.routing_test
class TestRoutingSymmetry:
    """Codex and claude both spawn in-process on the self-sudo-skip
    path (issue #522). Either `os_user=None` or `os_user` matching
    the current process user produces a direct argv with no sudo
    wrap; a non-bot `os_user` wraps via `sudo -H -u <user>` on both
    backends. Pinning the symmetry guards against a future change
    that re-introduces an asymmetric refusal on either backend.
    """

    @pytest.mark.asyncio
    async def test_codex_with_no_os_user_runs_in_process(self, tmp_path):
        """`CodexOneShotReasoner(os_user=None).run()` does not raise
        and produces a direct argv (no sudo prefix). Mirror of the
        claude self-sudo-skip path."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=None)
        proc = _make_proc(stdout=b"")
        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec,
            pytest.raises(OneShotOutputError),
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction", json_schema={"type": "object"})
        cmd = mock_exec.call_args[0]
        assert cmd[0] != "sudo"
        assert mock_exec.call_args.kwargs["start_new_session"] is False

    @pytest.mark.asyncio
    async def test_claude_with_no_os_user_still_spawns(self, tmp_path):
        """Claude memory must not regress for existing single-user
        installs that have never set per-user os_user."""
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path, os_user=None)
        proc = _make_proc(stdout=b"{}")
        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(prompt="p", purpose="fact_extraction")
        cmd = mock_exec.call_args[0]
        assert "sudo" not in cmd

    @pytest.mark.asyncio
    async def test_codex_with_non_bot_user_wraps_in_sudo(self, tmp_path):
        """The cross-user codex path stays unchanged: a non-bot
        `os_user` wraps the argv in `sudo -H -u <user>`. Regression
        guard for the multi-user/service-user deployment shape."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user="other-user")
        proc = _make_proc(stdout=b"")
        with (
            patch("kai.oneshot.resolve_claude_user", return_value="other-user"),
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec,
            pytest.raises(OneShotOutputError),
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction", json_schema={"type": "object"})
        cmd = mock_exec.call_args[0]
        assert cmd[0] == "sudo"
        assert "-u" in cmd
        assert "other-user" in cmd
        assert mock_exec.call_args.kwargs["start_new_session"] is True


@pytest.mark.routing_test
class TestRoutingTimeoutCleanup:
    """On timeout from a wrapped spawn, the cross-user kill goes
    out BEFORE the wrapper reap. Negative-PGID covers the whole
    target-user descendant tree, including the npm-wrapped codex
    chain (`sudo -> node -> codex`)."""

    @pytest.mark.asyncio
    async def test_cross_user_kill_fires_before_wrapper_reap(self, tmp_path):
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path, os_user="other-user")
        proc = _make_proc(raise_timeout=True)
        proc.pid = 12345
        kill_calls: list[tuple] = []

        kill_proc = _make_proc(stdout=b"", returncode=0)

        async def _fake_exec(*args, **kwargs):
            # First call: the wrapped agent spawn. Subsequent calls
            # (the kill subprocess) are recorded so we can assert
            # the cross-user kill argv and ordering.
            if args[0] == "sudo" and len(args) > 4 and args[4] == "/bin/kill":
                kill_calls.append(args)
                return kill_proc
            return proc

        with (
            patch("kai.oneshot.resolve_claude_user", return_value="other-user"),
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(side_effect=_fake_exec)),
            pytest.raises(OneShotTimeout),
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction", timeout=0.1)

        # The cross-user kill ran exactly once with the negative-PGID
        # argv shape; the wrapper reap happened after.
        assert len(kill_calls) == 1
        kill_argv = list(kill_calls[0])
        assert kill_argv == ["sudo", "-n", "-u", "other-user", "/bin/kill", "-KILL", "-12345"]
        proc.kill.assert_called_once()
        proc.wait.assert_awaited()

    @pytest.mark.asyncio
    async def test_cross_user_kill_failure_logs_and_reaps(self, tmp_path, caplog):
        """A kill subprocess failing for a real reason (sudoers
        misconfiguration, permission denied) logs a warning, still
        reaps the wrapper, and still raises OneShotTimeout."""
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path, os_user="other-user")
        proc = _make_proc(raise_timeout=True)
        proc.pid = 999
        kill_proc = _make_proc(stdout=b"", stderr=b"sudo: a password is required", returncode=1)

        async def _fake_exec(*args, **kwargs):
            if args[0] == "sudo" and len(args) > 4 and args[4] == "/bin/kill":
                return kill_proc
            return proc

        with (
            patch("kai.oneshot.resolve_claude_user", return_value="other-user"),
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(side_effect=_fake_exec)),
            caplog.at_level(logging.WARNING, logger="kai.acp"),
            pytest.raises(OneShotTimeout),
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction", timeout=0.1)

        proc.kill.assert_called_once()
        warnings = [r for r in caplog.records if "cross-user kill returned rc=1" in r.getMessage()]
        assert warnings, "expected a warning about the failing kill subprocess"

    @pytest.mark.asyncio
    async def test_cross_user_kill_esrch_is_benign(self, tmp_path, caplog):
        """rc=1 with the POSIX ESRCH diagnostic means the agent
        already exited between TimeoutError and the kill spawn: a
        benign race that must NOT hit the WARNING stream an operator
        watches for real sudoers failures. Still reaps the wrapper
        and still raises OneShotTimeout."""
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path, os_user="other-user")
        proc = _make_proc(raise_timeout=True)
        proc.pid = 999
        kill_proc = _make_proc(stdout=b"", stderr=b"kill: -999: No such process", returncode=1)

        async def _fake_exec(*args, **kwargs):
            if args[0] == "sudo" and len(args) > 4 and args[4] == "/bin/kill":
                return kill_proc
            return proc

        with (
            patch("kai.oneshot.resolve_claude_user", return_value="other-user"),
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(side_effect=_fake_exec)),
            caplog.at_level(logging.DEBUG, logger="kai.acp"),
            pytest.raises(OneShotTimeout),
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction", timeout=0.1)

        proc.kill.assert_called_once()
        warnings = [r for r in caplog.records if "cross-user kill" in r.getMessage() and r.levelno >= logging.WARNING]
        assert not warnings, "ESRCH must demote to DEBUG, not WARNING"
        debugs = [r for r in caplog.records if "benign race" in r.getMessage()]
        assert debugs

    @pytest.mark.asyncio
    async def test_direct_spawn_does_not_run_cross_user_kill(self, tmp_path):
        """The self-sudo-skip path keeps the original kill+await
        and does NOT invoke the cross-user kill subprocess."""
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_proc(raise_timeout=True)
        exec_calls: list[tuple] = []

        async def _fake_exec(*args, **kwargs):
            exec_calls.append(args)
            return proc

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(side_effect=_fake_exec)),
            pytest.raises(OneShotTimeout),
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction", timeout=0.1)
        # One spawn only: the wrapped agent (or in this case the
        # direct claude binary). No second spawn for the cross-user
        # kill, because there is no cross-user wrap.
        assert len(exec_calls) == 1
        # And the spawn was not wrapped in sudo.
        assert exec_calls[0][0] != "sudo"


class TestCwdAndSchemaModes:
    """The neutral extractor cwd is world-traversable, and the
    codex schema temp file is world-readable, so a target `os_user`
    can enter the cwd and open the schema input."""

    @pytest.mark.asyncio
    async def test_ensure_extractor_cwd_chmods_existing_dir(self, tmp_path, monkeypatch):
        """An existing cwd at mode 0o700 self-heals to 0o755 on the
        next call. The unconditional chmod is the load-bearing line
        for cross-user routing because `mkdir(exist_ok=True)` is a
        no-op when the directory already exists."""
        cwd = tmp_path / "memory" / "extractor_cwd"
        cwd.mkdir(parents=True)
        cwd.chmod(0o700)
        monkeypatch.setattr("kai.oneshot._EXTRACTOR_CWD", cwd)

        from kai.oneshot import _ensure_extractor_cwd

        _ensure_extractor_cwd()

        # 0o7XXX mask isolates the permission bits from any sticky/
        # setuid bits Posix might add.
        assert cwd.stat().st_mode & 0o777 == 0o755

    @pytest.mark.asyncio
    async def test_ensure_extractor_cwd_creates_at_correct_mode(self, tmp_path, monkeypatch):
        """A non-existent cwd is created at 0o755 on first call."""
        cwd = tmp_path / "memory" / "extractor_cwd"
        assert not cwd.exists()
        monkeypatch.setattr("kai.oneshot._EXTRACTOR_CWD", cwd)

        from kai.oneshot import _ensure_extractor_cwd

        _ensure_extractor_cwd()

        assert cwd.is_dir()
        assert cwd.stat().st_mode & 0o777 == 0o755

    @pytest.mark.asyncio
    async def test_codex_schema_temp_file_is_world_readable(self, tmp_path):
        """`tempfile.NamedTemporaryFile` defaults to 0o600; the
        reasoner explicitly chmods to 0o644 so the target user can
        open the schema file via --output-schema."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        captured: dict = {}

        def _capture_spawn(*args, **kwargs):
            i = args.index("--output-schema")
            captured["path"] = Path(args[i + 1])
            captured["mode"] = captured["path"].stat().st_mode & 0o777
            return _make_proc(stdout=b"")

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(side_effect=_capture_spawn)),
            pytest.raises(OneShotOutputError),
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction", json_schema={"type": "object"})

        assert captured["mode"] == 0o644


@pytest.mark.routing_test
class TestRoutingLogField:
    """Every outcome line carries `os_user=<target>` or `os_user=self`."""

    @pytest.mark.asyncio
    async def test_claude_success_log_has_os_user_target(self, tmp_path, caplog):
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path, os_user="other-user")
        proc = _make_proc(stdout=b"{}")
        with (
            patch("kai.oneshot.resolve_claude_user", return_value="other-user"),
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            caplog.at_level(logging.INFO, logger="kai.oneshot"),
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction")
        msg = next(r.getMessage() for r in caplog.records if r.message.startswith("oneshot_reasoner"))
        assert "os_user=other-user" in msg

    @pytest.mark.asyncio
    async def test_claude_success_log_has_os_user_self_on_direct_spawn(self, tmp_path, caplog):
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path, os_user=None)
        proc = _make_proc(stdout=b"{}")
        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            caplog.at_level(logging.INFO, logger="kai.oneshot"),
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction")
        msg = next(r.getMessage() for r in caplog.records if r.message.startswith("oneshot_reasoner"))
        assert "os_user=self" in msg


# ── Binary resolver integration ─────────────────────────────────────


class TestBinaryResolverIntegration:
    """Both reasoners must convert leaf-module `BinaryResolutionError`
    into `OneShotRoutingError` so the existing `memory_extraction`
    catch surface (`except OneShotError`) stays unchanged. Also
    confirms `OneShotResult.raw_metadata` carries `cmd` plus
    `resolved_binary` so the smoke command can ground its output in
    the subprocess boundary rather than re-resolving the binary."""

    @pytest.mark.asyncio
    async def test_claude_raises_routing_error_on_binary_unreachable(self, tmp_path):
        """Claude argv builder catches BinaryResolutionError and
        re-raises as OneShotRoutingError with the leaf-module
        message preserved. `memory_extraction` already catches
        OneShotError; the rewrap keeps its catch surface unchanged."""
        from kai.oneshot_binary import BinaryResolutionError

        reasoner = ClaudeOneShotReasoner(cwd=tmp_path)

        def boom(_backend: str) -> str:
            raise BinaryResolutionError("could not resolve claude binary: `claude` not on PATH")

        with (
            patch("kai.oneshot.resolve_oneshot_binary", side_effect=boom),
            pytest.raises(OneShotRoutingError) as exc,
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction")
        assert "claude binary" in str(exc.value)

    @pytest.mark.asyncio
    async def test_codex_raises_routing_error_on_binary_unreachable(self, tmp_path):
        """Codex argv builder mirrors the claude branch: catch
        BinaryResolutionError, re-raise as OneShotRoutingError."""
        from kai.oneshot_binary import BinaryResolutionError

        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())

        def boom(_backend: str) -> str:
            raise BinaryResolutionError("could not resolve codex binary: CODEX_BIN unset, `codex` not on PATH")

        with (
            patch("kai.oneshot.resolve_oneshot_binary", side_effect=boom),
            pytest.raises(OneShotRoutingError) as exc,
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction")
        assert "codex binary" in str(exc.value)

    @pytest.mark.asyncio
    async def test_claude_raw_metadata_carries_cmd_and_resolved_binary(self, tmp_path):
        """OneShotResult.raw_metadata must carry `cmd` (full argv,
        including any sudo prefix on cross-user routing) and
        `resolved_binary` (pre-sudo agent path). Smoke prints
        resolved_binary so the operator-visible answer is correct
        under cross-user wrapping."""
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path)
        proc = _make_proc(stdout=b'{"ok": true}')
        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(
                prompt="hello",
                model="claude-haiku-4-5",
                timeout=30,
                purpose="fact_extraction",
            )
        assert result.raw_metadata["resolved_binary"] == "claude"  # autouse fixture pins to literal
        assert result.raw_metadata["cmd"][0] == "claude"
        assert "--print" in result.raw_metadata["cmd"]

    @pytest.mark.asyncio
    async def test_codex_raw_metadata_carries_cmd_and_resolved_binary(self, tmp_path):
        """Same shape as the claude case. Tested against the
        non-schema branch so we exercise the simpler success path."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_proc(stdout=_codex_event("agent_message", "free-form text").encode("utf-8") + b"\n")
        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(
                prompt="hello",
                timeout=30,
                purpose="fact_extraction",
                json_schema=None,
            )
        assert result.raw_metadata["resolved_binary"] == "codex"
        assert result.raw_metadata["cmd"][0] == "codex"
        assert "exec" in result.raw_metadata["cmd"]


# ── _sanitize_for_codex (issue #505) ────────────────────────────────


class TestSanitizeForCodexDisallowedKeys:
    """OpenAI strict structured-outputs rejects many JSON Schema
    keywords claude accepts (`minLength`, `maxLength`, `pattern`,
    `format`, `minimum`, `maximum`, `multipleOf`, `minItems`,
    `maxItems`, `uniqueItems`, `default`). The sanitizer strips
    them recursively before the schema reaches codex's
    `--output-schema` flag."""

    def test_strips_string_length_bounds(self):
        from kai.oneshot import _sanitize_for_codex

        schema = {"type": "string", "minLength": 1, "maxLength": 500}
        out = _sanitize_for_codex(schema)
        assert out == {"type": "string"}

    def test_strips_number_bounds(self):
        from kai.oneshot import _sanitize_for_codex

        schema = {"type": "number", "minimum": 0.5, "maximum": 1.0}
        out = _sanitize_for_codex(schema)
        assert out == {"type": "number"}

    def test_strips_array_bounds(self):
        from kai.oneshot import _sanitize_for_codex

        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
            "uniqueItems": True,
        }
        out = _sanitize_for_codex(schema)
        assert out == {"type": "array", "items": {"type": "string"}}

    def test_strips_pattern_and_format(self):
        from kai.oneshot import _sanitize_for_codex

        schema = {"type": "string", "pattern": "^x+$", "format": "email"}
        out = _sanitize_for_codex(schema)
        assert out == {"type": "string"}

    def test_strips_default(self):
        from kai.oneshot import _sanitize_for_codex

        schema = {"type": "string", "default": "foo"}
        out = _sanitize_for_codex(schema)
        assert out == {"type": "string"}

    def test_strips_not_composition(self):
        """`not` is currently rejected by OpenAI strict structured
        outputs even though plain JSON Schema accepts it. Stripping
        is forward-protection against a future schema author who
        adds a `not` branch without knowing the codex CLI would 400
        on it."""
        from kai.oneshot import _sanitize_for_codex

        schema = {"type": "string", "not": {"enum": ["forbidden"]}}
        out = _sanitize_for_codex(schema)
        assert out == {"type": "string"}

    def test_preserves_enum(self):
        from kai.oneshot import _sanitize_for_codex

        schema = {"type": "string", "enum": ["a", "b", "c"]}
        out = _sanitize_for_codex(schema)
        assert out == {"type": "string", "enum": ["a", "b", "c"]}

    def test_preserves_description(self):
        from kai.oneshot import _sanitize_for_codex

        schema = {"type": "string", "description": "a label"}
        out = _sanitize_for_codex(schema)
        assert out == {"type": "string", "description": "a label"}


class TestSanitizeForCodexRequiredExpansion:
    """Strict mode requires `required` to list every key in
    `properties`. Truly optional properties get added to required
    AND have their `type` widened to include `"null"` so the model
    can emit null for absent values."""

    def test_optional_property_added_to_required(self):
        from kai.oneshot import _sanitize_for_codex

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "nickname": {"type": "string"},
            },
            "required": ["name"],
            "additionalProperties": False,
        }
        out = _sanitize_for_codex(schema)
        assert set(out["required"]) == {"name", "nickname"}

    def test_optional_type_widened_to_nullable(self):
        from kai.oneshot import _sanitize_for_codex

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "nickname": {"type": "string"},
            },
            "required": ["name"],
            "additionalProperties": False,
        }
        out = _sanitize_for_codex(schema)
        assert out["properties"]["nickname"]["type"] == ["string", "null"]
        assert out["properties"]["name"]["type"] == "string"

    def test_optional_list_type_appends_null(self):
        from kai.oneshot import _sanitize_for_codex

        schema = {
            "type": "object",
            "properties": {
                "field": {"type": ["string", "integer"]},
            },
            "required": [],
            "additionalProperties": False,
        }
        out = _sanitize_for_codex(schema)
        assert out["properties"]["field"]["type"] == ["string", "integer", "null"]

    def test_already_nullable_type_not_double_appended(self):
        from kai.oneshot import _sanitize_for_codex

        schema = {
            "type": "object",
            "properties": {"field": {"type": ["string", "null"]}},
            "required": [],
            "additionalProperties": False,
        }
        out = _sanitize_for_codex(schema)
        # Property is added to required AND keeps its type list as-is.
        assert out["required"] == ["field"]
        assert out["properties"]["field"]["type"] == ["string", "null"]

    def test_property_without_type_left_alone(self):
        """A property whose schema is just an enum (no explicit
        `type`) is added to required but cannot be safely widened.
        Leaving the type absent is the documented posture; if a
        future schema hits this and codex 400s on it, the
        operator-side log review will surface the property name."""
        from kai.oneshot import _sanitize_for_codex

        schema = {
            "type": "object",
            "properties": {"mode": {"enum": ["a", "b"]}},
            "required": [],
            "additionalProperties": False,
        }
        out = _sanitize_for_codex(schema)
        assert out["required"] == ["mode"]
        assert "type" not in out["properties"]["mode"]


class TestSanitizeForCodexAdditionalProperties:
    """Every object node must have `additionalProperties: false`."""

    def test_adds_when_missing(self):
        from kai.oneshot import _sanitize_for_codex

        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
        out = _sanitize_for_codex(schema)
        assert out["additionalProperties"] is False

    def test_preserves_when_already_set(self):
        from kai.oneshot import _sanitize_for_codex

        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
            "additionalProperties": False,
        }
        out = _sanitize_for_codex(schema)
        assert out["additionalProperties"] is False

    def test_non_object_node_does_not_gain_additional_properties(self):
        """Arrays and scalar types must NOT get an
        `additionalProperties` injected; that would be a schema
        error of its own."""
        from kai.oneshot import _sanitize_for_codex

        schema = {"type": "array", "items": {"type": "string"}}
        out = _sanitize_for_codex(schema)
        assert "additionalProperties" not in out


class TestSanitizeForCodexRecursion:
    """The sanitizer recurses into nested objects and array items."""

    def test_recurses_into_array_items(self):
        from kai.oneshot import _sanitize_for_codex

        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "tag": {"type": "string"},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        }
        out = _sanitize_for_codex(schema)
        item = out["items"]
        assert "minLength" not in item["properties"]["name"]
        assert set(item["required"]) == {"name", "tag"}

    def test_recurses_into_nested_objects(self):
        from kai.oneshot import _sanitize_for_codex

        schema = {
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "properties": {
                        "deep": {"type": "string", "maxLength": 10},
                    },
                    "required": ["deep"],
                    "additionalProperties": False,
                },
            },
            "required": ["inner"],
            "additionalProperties": False,
        }
        out = _sanitize_for_codex(schema)
        assert "maxLength" not in out["properties"]["inner"]["properties"]["deep"]


class TestSanitizeForCodexProductionSchemas:
    """The real `_FACT_SCHEMA` and `_EPISODE_SCHEMA` must come out
    strict-mode-valid after sanitization. Regression test for the
    specific failure mode caught running the eval gate on
    2026-05-19: the API rejected the production schemas for
    `required` not covering `confirmation_quote` (FACT) and
    `lessons` (EPISODE)."""

    def _assert_strict_mode_valid(self, node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False, "object missing additionalProperties: false"
                properties = set((node.get("properties") or {}).keys())
                required = set(node.get("required") or [])
                assert properties == required or properties.issubset(required), (
                    f"required {required} does not cover properties {properties}"
                )
            for k, v in node.items():
                assert k not in (
                    "minLength",
                    "maxLength",
                    "pattern",
                    "format",
                    "minimum",
                    "maximum",
                    "multipleOf",
                    "minItems",
                    "maxItems",
                    "uniqueItems",
                    "default",
                    "not",
                ), f"disallowed key survived: {k}"
                self._assert_strict_mode_valid(v)
        elif isinstance(node, list):
            for item in node:
                self._assert_strict_mode_valid(item)

    def test_fact_schema_round_trips_strict_mode_valid(self):
        from kai.memory_extraction import _FACT_SCHEMA
        from kai.oneshot import _sanitize_for_codex

        sanitized = _sanitize_for_codex(_FACT_SCHEMA)
        self._assert_strict_mode_valid(sanitized)
        # The two known-optional fact item properties are now in
        # required AND widened to nullable.
        item = sanitized["properties"]["facts"]["items"]
        for optional in ("confirmation_quote", "existing_id"):
            assert optional in item["required"]
            assert item["properties"][optional]["type"] == ["string", "null"]

    def test_episode_schema_round_trips_strict_mode_valid(self):
        from kai.memory_extraction import _EPISODE_SCHEMA
        from kai.oneshot import _sanitize_for_codex

        sanitized = _sanitize_for_codex(_EPISODE_SCHEMA)
        self._assert_strict_mode_valid(sanitized)
        # `lessons` was the previously-optional field; should now be
        # required + nullable.
        episode = sanitized["properties"]["episode"]
        assert "lessons" in episode["required"]
        assert episode["properties"]["lessons"]["type"] == ["string", "null"]

    def test_input_schema_not_mutated(self):
        """The sanitizer must not mutate the caller's schema. Both
        production memory stages read from module-level constants;
        an in-place mutation would corrupt subsequent calls."""
        from kai.memory_extraction import _FACT_SCHEMA
        from kai.oneshot import _sanitize_for_codex

        before = json.dumps(_FACT_SCHEMA, sort_keys=True)
        _sanitize_for_codex(_FACT_SCHEMA)
        after = json.dumps(_FACT_SCHEMA, sort_keys=True)
        assert before == after


class TestCodexReasonerWritesSanitizedSchema:
    """`CodexOneShotReasoner.run()` calls the sanitizer before
    writing the temp file, so the disk contents codex actually reads
    have no strict-mode-disallowed keywords."""

    @pytest.mark.asyncio
    async def test_disk_schema_is_sanitized(self, tmp_path):
        from kai.memory_extraction import _FACT_SCHEMA

        reasoner = CodexOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        captured: dict = {}

        def _capture(*args, **kwargs):
            i = args.index("--output-schema")
            captured["path"] = Path(args[i + 1])
            captured["body"] = json.loads(captured["path"].read_text(encoding="utf-8"))
            return _make_proc(stdout=b"")

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(side_effect=_capture)),
            pytest.raises(OneShotOutputError),
        ):
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema=_FACT_SCHEMA,
            )

        # The on-disk schema reflects the sanitizer's transform, not
        # the input. Spot-check the known previously-optional fields
        # are now required with nullable types.
        disk = captured["body"]
        item = disk["properties"]["facts"]["items"]
        assert "confirmation_quote" in item["required"]
        assert item["properties"]["confirmation_quote"]["type"] == ["string", "null"]
        # And the disallowed keys are absent.
        assert "minLength" not in item["properties"]["content"]
        assert "maxItems" not in disk["properties"]["facts"]


# ── Schema-payload parsing tests ─────────────────────────────────


class TestParseSchemaPayload:
    """Contract for `_parse_schema_payload`: bare JSON passes through
    unchanged (including non-dict values, which must reach the
    caller's non_object_json check), fenced or prose-wrapped objects
    are recovered, the last candidate wins within a tier, and text
    with no recoverable object re-raises the bare-parse
    JSONDecodeError."""

    def test_bare_object_passes_through(self):
        assert _parse_schema_payload('{"facts": []}') == {"facts": []}

    @pytest.mark.parametrize("bare_non_dict", ["[1, 2]", "42", "true", "null", '"text"'])
    def test_bare_non_dict_passes_through(self, bare_non_dict):
        assert _parse_schema_payload(bare_non_dict) == json.loads(bare_non_dict)

    def test_fenced_json_block(self):
        assert _parse_schema_payload('```json\n{"a": 1}\n```') == {"a": 1}

    def test_fence_without_info_string(self):
        assert _parse_schema_payload('```\n{"a": 1}\n```') == {"a": 1}

    def test_preamble_then_fenced_payload(self):
        # The exact production failure shape: reasoning prose
        # followed by a fenced payload.
        text = (
            'Let me analyze the current exchange to extract facts.\n\n```json\n{"facts": [], "has_episode": false}\n```'
        )
        assert _parse_schema_payload(text) == {"facts": [], "has_episode": False}

    def test_last_fenced_object_wins(self):
        text = '```json\n{"example": true}\n```\nThe real answer:\n```json\n{"a": 1}\n```'
        assert _parse_schema_payload(text) == {"a": 1}

    def test_unfenced_object_in_prose(self):
        text = 'Here is the result: {"a": 1} hope that helps.'
        assert _parse_schema_payload(text) == {"a": 1}

    def test_last_unfenced_object_wins(self):
        text = 'I will use {"draft": true} as a template. Final: {"a": 1}'
        assert _parse_schema_payload(text) == {"a": 1}

    def test_nested_object_returns_outer(self):
        # The brace scan resumes after a parsed object so an inner
        # fragment cannot shadow the complete payload.
        text = 'payload {"outer": {"inner": 1}} end'
        assert _parse_schema_payload(text) == {"outer": {"inner": 1}}

    def test_fenced_payload_beats_unfenced_prose_example(self):
        text = 'The shape is {"facts": []} with values.\n```json\n{"facts": ["x"], "has_episode": true}\n```'
        assert _parse_schema_payload(text) == {"facts": ["x"], "has_episode": True}

    def test_no_json_raises_decode_error(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_schema_payload("not json at all")

    def test_fenced_non_object_raises_decode_error(self):
        # A fenced list is not a candidate (the schema demands an
        # object) and the body has no `{` for the brace scan.
        with pytest.raises(json.JSONDecodeError):
            _parse_schema_payload("```json\n[1, 2]\n```")


# ── OpenCode reasoner tests ──────────────────────────────────────


@pytest.fixture(autouse=True)
def _stub_cross_user_kill_for_opencode_tests(request):
    """Patch `_kill_target_user_tree` to a no-op so the OpenCode
    reasoner's cleanup path does not issue a second
    `asyncio.create_subprocess_exec` call (which would otherwise
    shadow the spawn call recorded on `mock_exec.call_args`). The
    routing_test marker opts out so the routing classes that
    explicitly exercise the kill escalation keep their real behavior.
    """
    if request.node.get_closest_marker("routing_test"):
        yield
        return

    async def _noop(**kwargs) -> None:
        return None

    with patch("kai.oneshot._kill_target_user_tree", side_effect=_noop):
        yield


def _make_acp_proc(lines: list[bytes], *, returncode: int | None = None) -> MagicMock:
    """Build a mock asyncio subprocess for the OpenCode ACP read loop.

    `lines` is the queue of bytes that `proc.stdout.readline` will
    yield in order. An empty bytes value at the tail simulates EOF
    (the reasoner uses an empty readline result to detect subprocess
    death). `returncode` controls `proc.returncode`: None matches a
    still-running process, an integer matches an exited process.

    `proc.stdin.write` accumulates the written bytes into
    `proc._stdin_writes` so tests can assert on the JSON-RPC payloads
    the reasoner sent. `proc.stderr.readline` returns empty bytes
    immediately so the background drain task finishes without
    consuming any test queue entries.

    The mock is intentionally minimal: only the attributes the
    reasoner reads from the subprocess are populated. Adding more
    surface would invite drift between the reasoner's actual
    interface and the test's mocked interface.
    """
    proc = MagicMock()

    # Stdout readline pops from the queue. An empty bytes element
    # (or running off the end) is treated as EOF.
    queue = list(lines)

    async def _readline() -> bytes:
        if not queue:
            return b""
        return queue.pop(0)

    proc.stdout = MagicMock()
    proc.stdout.readline = _readline

    # Stdin write accumulates payloads for later assertion.
    proc._stdin_writes = []

    def _stdin_write(data: bytes) -> None:
        proc._stdin_writes.append(data)

    async def _stdin_drain() -> None:
        return None

    proc.stdin = MagicMock()
    proc.stdin.write = _stdin_write
    proc.stdin.drain = _stdin_drain

    # Stderr never yields data; the background drain task hits EOF and
    # returns immediately.
    async def _stderr_readline() -> bytes:
        return b""

    proc.stderr = MagicMock()
    proc.stderr.readline = _stderr_readline

    proc.returncode = returncode
    proc.kill = MagicMock()
    proc.pid = 12345

    async def _wait() -> int:
        # After kill, the returncode is whatever the test passed in;
        # if None, simulate a clean exit by setting it to 0.
        if proc.returncode is None:
            proc.returncode = 0
        return proc.returncode

    proc.wait = _wait
    return proc


def _rpc_response_line(*, request_id: int, result: dict) -> bytes:
    """JSON-RPC 2.0 response with a result body, newline-terminated."""
    return (json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}) + "\n").encode()


def _rpc_error_line(*, request_id: int, message: str) -> bytes:
    """JSON-RPC 2.0 error response, newline-terminated."""
    return (json.dumps({"jsonrpc": "2.0", "id": request_id, "error": {"code": -1, "message": message}}) + "\n").encode()


def _session_update_line(text: str) -> bytes:
    """Streaming `session/update` notification carrying agent text."""
    return (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": text},
                    }
                },
            }
        )
        + "\n"
    ).encode()


def _filtered_session_update_line(kind: str) -> bytes:
    """A session/update notification with a discriminator the reasoner
    should filter out (e.g., agent_thought_chunk, tool_call). The text
    inside MUST NOT appear in the accumulated output."""
    return (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": kind,
                        "content": {"type": "text", "text": "FILTERED-MUST-NOT-APPEAR"},
                    }
                },
            }
        )
        + "\n"
    ).encode()


def _opencode_handshake_lines() -> list[bytes]:
    """The handshake responses every test must feed: initialize result
    (id=1) and session/new result (id=2 with a sessionId)."""
    return [
        _rpc_response_line(request_id=1, result={"protocolVersion": 1}),
        _rpc_response_line(request_id=2, result={"sessionId": "sess-abc-123"}),
    ]


class TestOpenCodeRenderSessionPrompt:
    """The render helper mirrors `_render_codex_stdin`: when a system
    prompt is supplied, wrap it in a randomized boundary block above
    the user prompt; when None or empty, return the user prompt
    unchanged."""

    def test_no_system_prompt_returns_user_prompt_unchanged(self):
        assert _render_opencode_session_prompt(None, "hello") == "hello"
        assert _render_opencode_session_prompt("", "hello") == "hello"

    def test_system_prompt_emits_boundary_block(self):
        rendered = _render_opencode_session_prompt("be brief", "hello")
        # The render must contain both BEGIN and END SYSTEM markers,
        # the system text between them, and the user prompt below.
        assert "--- BEGIN SYSTEM " in rendered
        assert "--- END SYSTEM " in rendered
        assert "be brief" in rendered
        assert rendered.endswith("hello")

    def test_per_call_boundary_token_randomized(self):
        """Two calls produce different boundary tokens. Collision-
        avoidance, not adversarial security; matches codex's contract."""
        a = _render_opencode_session_prompt("sys", "user")
        b = _render_opencode_session_prompt("sys", "user")
        # The system content and prompts are identical; only the
        # randomized boundary tokens should differ.
        assert a != b


class TestOpenCodePreservedAuthVars:
    """`_preserved_auth_vars_for("opencode")` returns the per-provider
    API key allow-list; the previous claude/codex-only ValueError no
    longer fires."""

    def test_returns_opencode_specific_list(self):
        assert _preserved_auth_vars_for("opencode") == _OPENCODE_PRESERVED_AUTH_VARS

    def test_config_content_survives_the_wrap(self):
        """OPENCODE_CONFIG_CONTENT is the reasoner's model-selection
        channel; without it in the preserve list, sudo's env_reset
        strips the per-call model on the cross-user path and the
        wrapped opencode silently falls back to the target user's
        config defaults (the conversational backend has always
        preserved it; the one-shot must match)."""
        assert "OPENCODE_CONFIG_CONTENT" in _OPENCODE_PRESERVED_AUTH_VARS

    def test_unknown_backend_still_raises(self):
        with pytest.raises(ValueError):
            _preserved_auth_vars_for("nonsense")


class TestGoosePreservedAuthVars:
    """`_preserved_auth_vars_for("goose")` carries the five provider
    keys plus goose's endpoint-override vars. The goose list is no
    longer an alias of the opencode one: opencode has no endpoint
    env vars (custom base URLs are opencode.json config state), while
    goose reads its custom endpoints from the environment."""

    def test_returns_goose_specific_list(self):
        assert _preserved_auth_vars_for("goose") == _GOOSE_PRESERVED_AUTH_VARS

    def test_endpoint_override_vars_survive_the_wrap(self):
        """ANTHROPIC_HOST / OPENAI_HOST / OPENAI_BASE_PATH /
        OLLAMA_HOST are how goose points a provider at a proxy or
        gateway; a custom-endpoint install loses its routing under
        env_reset without them (the claude / codex lists preserve
        their base-URL overrides the same way)."""
        for var in ("ANTHROPIC_HOST", "OPENAI_HOST", "OPENAI_BASE_PATH", "OLLAMA_HOST"):
            assert var in _GOOSE_PRESERVED_AUTH_VARS

    def test_endpoint_override_vars_forwarded_in_process(self):
        """The same endpoint vars ride the subprocess env allow-list
        so the in-process (no-wrap) spawn routes correctly too; the
        preserve list only matters cross-user."""
        for var in ("ANTHROPIC_HOST", "OPENAI_HOST", "OPENAI_BASE_PATH", "OLLAMA_HOST"):
            assert var in _GOOSE_ENV_ALLOWLIST

    def test_opencode_config_content_not_in_goose_list(self):
        """The alias split must not leak opencode's config-content
        channel into the goose list; goose delivers nothing through
        OPENCODE_CONFIG_CONTENT."""
        assert "OPENCODE_CONFIG_CONTENT" not in _GOOSE_PRESERVED_AUTH_VARS


class TestOpenCodeOneShotReasonerArgvAndEnv:
    """Argv: `opencode acp` plus the optional sudo wrap. Env:
    OPENCODE_CONFIG_CONTENT carries the model when supplied;
    OPENCODE_BIN-style env keys are NOT in the subprocess env (the
    allow-list is intentional)."""

    @pytest.mark.asyncio
    async def test_argv_is_opencode_acp(self, tmp_path):
        reasoner = OpenCodeOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_acp_proc(
            _opencode_handshake_lines() + [_rpc_response_line(request_id=3, result={"stopReason": "end_turn"})]
        )

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(prompt="hi", purpose="fact_extraction")

        cmd = mock_exec.call_args[0]
        assert cmd[0] == "opencode"
        assert cmd[1] == "acp"

    @pytest.mark.asyncio
    async def test_env_carries_model_via_opencode_config_content(self, tmp_path):
        reasoner = OpenCodeOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_acp_proc(
            _opencode_handshake_lines() + [_rpc_response_line(request_id=3, result={"stopReason": "end_turn"})]
        )

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(
                prompt="hi",
                model="anthropic/claude-sonnet-4-5",
                purpose="fact_extraction",
            )

        env = mock_exec.call_args.kwargs["env"]
        assert "OPENCODE_CONFIG_CONTENT" in env
        cfg = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        assert cfg == {"model": "anthropic/claude-sonnet-4-5"}

    @pytest.mark.asyncio
    async def test_env_omits_opencode_config_content_when_model_none(self, tmp_path):
        """`model=None` lets opencode fall back to its config-file
        defaults; emitting an empty `{"model": ""}` JSON would override
        the fallback with a broken value."""
        reasoner = OpenCodeOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_acp_proc(
            _opencode_handshake_lines() + [_rpc_response_line(request_id=3, result={"stopReason": "end_turn"})]
        )

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(prompt="hi", purpose="fact_extraction")

        env = mock_exec.call_args.kwargs["env"]
        assert "OPENCODE_CONFIG_CONTENT" not in env

    @pytest.mark.asyncio
    async def test_env_allowlist_excludes_unrelated_keys(self, tmp_path, monkeypatch):
        """Only keys from `_OPENCODE_ENV_ALLOWLIST` may flow into the
        subprocess env. A stray ANTHROPIC_BASE_URL (claude-only) must
        not appear; a webhook secret must not leak; etc."""
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://leaked.example")
        monkeypatch.setenv("KAI_WEBHOOK_SECRET", "leaked-secret")
        reasoner = OpenCodeOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_acp_proc(
            _opencode_handshake_lines() + [_rpc_response_line(request_id=3, result={"stopReason": "end_turn"})]
        )

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(prompt="hi", purpose="fact_extraction")

        env = mock_exec.call_args.kwargs["env"]
        assert "ANTHROPIC_BASE_URL" not in env
        assert "KAI_WEBHOOK_SECRET" not in env


class TestOpenCodeOneShotReasonerFlow:
    """End-to-end ACP flow: handshake, prompt, accumulate text from
    `agent_message_chunk` notifications, return on prompt response."""

    @pytest.mark.asyncio
    async def test_accumulates_agent_message_chunk_text(self, tmp_path):
        reasoner = OpenCodeOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_acp_proc(
            _opencode_handshake_lines()
            + [
                _session_update_line("Hello, "),
                _session_update_line("world!"),
                _rpc_response_line(request_id=3, result={"stopReason": "end_turn"}),
            ]
        )

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(prompt="hi", purpose="fact_extraction")

        assert result.text == "Hello, world!"
        assert result.backend == "opencode"

    @pytest.mark.asyncio
    async def test_filters_non_agent_message_chunk_shapes(self, tmp_path):
        """`agent_thought_chunk`, `tool_call`, `tool_call_update`,
        `usage_update`, `available_commands_update` must NOT accumulate
        as text. Only `agent_message_chunk` is user-visible."""
        reasoner = OpenCodeOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_acp_proc(
            _opencode_handshake_lines()
            + [
                _filtered_session_update_line("agent_thought_chunk"),
                _filtered_session_update_line("tool_call"),
                _filtered_session_update_line("usage_update"),
                _session_update_line("REAL"),
                _rpc_response_line(request_id=3, result={"stopReason": "end_turn"}),
            ]
        )

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(prompt="hi", purpose="fact_extraction")

        assert result.text == "REAL"
        assert "FILTERED" not in result.text

    @pytest.mark.asyncio
    async def test_drains_text_chunk_arriving_after_prompt_response(self, tmp_path):
        """A text chunk that flushes to stdout after the session/prompt
        response still accumulates. OpenCode resolves the prompt when
        the session goes idle while its event forwarding can still
        have the final agent_message_chunk in flight; without the
        post-response drain the tail is lost, and for JSON consumers
        (triage, extraction) a missing tail makes the entire response
        unparseable."""
        reasoner = OpenCodeOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_acp_proc(
            _opencode_handshake_lines()
            + [
                _session_update_line('{\n  "priority": "'),
                _rpc_response_line(request_id=3, result={"stopReason": "end_turn"}),
                _session_update_line('low"\n}'),
            ]
        )

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(prompt="hi", purpose="fact_extraction")

        assert json.loads(result.text) == {"priority": "low"}

    @pytest.mark.asyncio
    async def test_drain_filters_non_text_shapes_and_quiet_window_closes(self, tmp_path, monkeypatch):
        """The drain applies the same notification filter as the main
        loop (thought chunks do not accumulate), and when the pipe
        stays open with nothing arriving inside the drain window the
        accumulated text is returned as-is."""
        monkeypatch.setattr("kai.acp.COMPLETION_DRAIN_WINDOW_S", 0.05)
        reasoner = OpenCodeOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_acp_proc([])
        queue = _opencode_handshake_lines() + [
            _session_update_line("REAL"),
            _rpc_response_line(request_id=3, result={"stopReason": "end_turn"}),
            _filtered_session_update_line("agent_thought_chunk"),
            _session_update_line("-TAIL"),
        ]

        async def _readline() -> bytes:
            if queue:
                return queue.pop(0)
            # Pipe open, no data: block until the drain window
            # cancels the read.
            await asyncio.Event().wait()
            return b""

        proc.stdout.readline = _readline

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(prompt="hi", purpose="fact_extraction")

        assert result.text == "REAL-TAIL"
        assert "FILTERED" not in result.text

    @pytest.mark.asyncio
    async def test_session_prompt_carries_rendered_text_with_boundary(self, tmp_path):
        """The session/prompt request's content[0].text must be the
        boundary-wrapped render when a system_prompt is supplied."""
        reasoner = OpenCodeOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_acp_proc(
            _opencode_handshake_lines() + [_rpc_response_line(request_id=3, result={"stopReason": "end_turn"})]
        )

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            await reasoner.run(
                prompt="USER",
                system_prompt="SYS",
                purpose="fact_extraction",
            )

        # The third write is the session/prompt payload (the first
        # two are initialize and session/new). The text content
        # carries both SYS and USER inside the boundary block.
        prompt_msg = json.loads(proc._stdin_writes[2].decode().strip())
        assert prompt_msg["method"] == "session/prompt"
        content = prompt_msg["params"]["prompt"][0]
        assert content["type"] == "text"
        text = content["text"]
        assert "--- BEGIN SYSTEM " in text
        assert "SYS" in text
        assert text.rstrip().endswith("USER")


class TestOpenCodeOneShotReasonerPermissionRejection:
    """The one-shot reasoner must reject every `session/request_permission`
    using the `reject_once` policy; memory extraction, review, and
    triage prompts MUST NOT execute tools."""

    @pytest.mark.asyncio
    async def test_responds_to_permission_request_with_reject_once(self, tmp_path):
        reasoner = OpenCodeOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        permission_req = (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "session/request_permission",
                    "id": 99,
                    "params": {
                        "options": [
                            {"optionId": "opt-a", "kind": "allow_always"},
                            {"optionId": "opt-r", "kind": "reject_once"},
                        ]
                    },
                }
            )
            + "\n"
        ).encode()
        proc = _make_acp_proc(
            _opencode_handshake_lines()
            + [
                permission_req,
                _session_update_line("blocked-but-text-still-flows"),
                _rpc_response_line(request_id=3, result={"stopReason": "end_turn"}),
            ]
        )

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            await reasoner.run(prompt="p", purpose="fact_extraction")

        # The 4th stdin write (after initialize, session/new,
        # session/prompt) is the permission response. The response
        # body must select the reject_once option.
        response = json.loads(proc._stdin_writes[3].decode().strip())
        assert response["id"] == 99
        assert response["result"]["outcome"]["outcome"] == "selected"
        assert response["result"]["outcome"]["optionId"] == "opt-r"


class TestOpenCodeOneShotReasonerFailures:
    """Failure surface: JSON-RPC error on prompt id, subprocess EOF,
    timeout, binary resolution failure. Each maps to a specific
    typed exception so the memory extraction caller can branch."""

    @pytest.mark.asyncio
    async def test_jsonrpc_error_raises_output_error(self, tmp_path):
        reasoner = OpenCodeOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_acp_proc(_opencode_handshake_lines() + [_rpc_error_line(request_id=3, message="model refused")])

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(OneShotOutputError) as excinfo,
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction")
        assert "model refused" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_eof_with_returncode_raises_subprocess_error(self, tmp_path):
        """Subprocess dies mid-stream: empty readline + non-None
        returncode -> OneShotSubprocessError."""
        reasoner = OpenCodeOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_acp_proc(
            _opencode_handshake_lines() + [b""],
            returncode=137,
        )

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(OneShotSubprocessError) as excinfo,
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction")
        assert excinfo.value.returncode == 137

    @pytest.mark.asyncio
    async def test_binary_resolution_failure_raises_routing_error(self, tmp_path):
        from kai.oneshot_binary import BinaryResolutionError

        reasoner = OpenCodeOneShotReasoner(cwd=tmp_path, os_user=_current_user())

        with (
            patch("kai.oneshot.resolve_oneshot_binary", side_effect=BinaryResolutionError("no opencode")),
            pytest.raises(OneShotRoutingError) as excinfo,
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction")
        assert "no opencode" in str(excinfo.value)


class TestOpenCodeOneShotReasonerSchema:
    """When `json_schema` is supplied, the accumulated text must carry
    a JSON object containing the schema's `required` fields; fence /
    prose wrapping around the object is tolerated. The reasoner wraps
    the parsed payload in the same envelope claude / codex emit, so
    the memory extraction caller does not need an opencode branch."""

    @pytest.mark.asyncio
    async def test_schema_backed_returns_envelope(self, tmp_path):
        reasoner = OpenCodeOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        body = json.dumps({"facts": [], "has_episode": False})
        proc = _make_acp_proc(
            _opencode_handshake_lines()
            + [
                _session_update_line(body),
                _rpc_response_line(request_id=3, result={"stopReason": "end_turn"}),
            ]
        )

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={
                    "type": "object",
                    "required": ["facts", "has_episode"],
                },
            )

        envelope = json.loads(result.text)
        assert envelope == {
            "is_error": False,
            "structured_output": {"facts": [], "has_episode": False},
        }

    @pytest.mark.asyncio
    async def test_schema_backed_recovers_fenced_payload_with_preamble(self, tmp_path):
        """Some models narrate before answering and fence the JSON
        (deepseek-v4-flash does both); the payload must still reach
        the envelope instead of collapsing to invalid_json."""
        reasoner = OpenCodeOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        body = (
            'Let me analyze the current exchange to extract facts.\n\n```json\n{"facts": [], "has_episode": false}\n```'
        )
        proc = _make_acp_proc(
            _opencode_handshake_lines()
            + [
                _session_update_line(body),
                _rpc_response_line(request_id=3, result={"stopReason": "end_turn"}),
            ]
        )

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={
                    "type": "object",
                    "required": ["facts", "has_episode"],
                },
            )

        envelope = json.loads(result.text)
        assert envelope == {
            "is_error": False,
            "structured_output": {"facts": [], "has_episode": False},
        }

    @pytest.mark.asyncio
    async def test_schema_backed_invalid_json_raises_output_error(self, tmp_path):
        reasoner = OpenCodeOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_acp_proc(
            _opencode_handshake_lines()
            + [
                _session_update_line("not-json-at-all"),
                _rpc_response_line(request_id=3, result={"stopReason": "end_turn"}),
            ]
        )

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(OneShotOutputError),
        ):
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={"type": "object", "required": ["facts"]},
            )

    @pytest.mark.asyncio
    async def test_schema_backed_missing_required_raises_output_error(self, tmp_path):
        reasoner = OpenCodeOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        body = json.dumps({"facts": []})  # missing has_episode
        proc = _make_acp_proc(
            _opencode_handshake_lines()
            + [
                _session_update_line(body),
                _rpc_response_line(request_id=3, result={"stopReason": "end_turn"}),
            ]
        )

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(OneShotOutputError) as excinfo,
        ):
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={
                    "type": "object",
                    "required": ["facts", "has_episode"],
                },
            )
        assert "has_episode" in str(excinfo.value)


class TestOpenCodeOneShotReasonerCleanup:
    """The subprocess must be killed (or reaped) on every exit path:
    success, error, EOF. Prevents orphan `opencode acp` processes."""

    @pytest.mark.asyncio
    async def test_subprocess_kill_called_after_success_when_still_running(self, tmp_path):
        """proc.returncode is None on the happy path (the subprocess
        is still up when the prompt response arrives), so the cleanup
        path issues a kill and reaps it."""
        reasoner = OpenCodeOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_acp_proc(
            _opencode_handshake_lines()
            + [
                _session_update_line("ok"),
                _rpc_response_line(request_id=3, result={"stopReason": "end_turn"}),
            ]
        )

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            await reasoner.run(prompt="p", purpose="fact_extraction")

        proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_subprocess_kill_called_on_jsonrpc_error(self, tmp_path):
        reasoner = OpenCodeOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_acp_proc(_opencode_handshake_lines() + [_rpc_error_line(request_id=3, message="err")])

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(OneShotOutputError),
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction")

        proc.kill.assert_called_once()


@pytest.mark.routing_test
class TestOpenCodeOneShotReasonerSudoWrap:
    """Cross-user routing: when `os_user` resolves to a non-bot user
    via `resolve_claude_user`, the argv MUST be wrapped in
    `sudo -H -u <user> --preserve-env=<csv> --` with the opencode
    preserve list. The `routing_test` marker opts out of the autouse
    bypass fixture so the production routing path is exercised in
    full."""

    @pytest.mark.asyncio
    async def test_sudo_wrap_argv_shape(self, tmp_path):
        """Argv starts with `sudo -H -u <target>`, carries the
        opencode preserve-env CSV, then the resolved opencode binary
        and `acp`. The exact ordering matches `_wrap_cmd_for_user`."""
        reasoner = OpenCodeOneShotReasoner(cwd=tmp_path, os_user="some_other_user")
        proc = _make_acp_proc(
            _opencode_handshake_lines() + [_rpc_response_line(request_id=3, result={"stopReason": "end_turn"})]
        )

        async def _noop_kill(**kwargs) -> None:
            return None

        with (
            # Force `resolve_claude_user` to return the target as a
            # non-bot user so the wrap fires; the routing_test marker
            # already opts out of the autouse bypass.
            patch("kai.oneshot.resolve_claude_user", side_effect=lambda u: u),
            patch("kai.oneshot._kill_target_user_tree", side_effect=_noop_kill),
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec,
        ):
            await reasoner.run(prompt="hi", purpose="fact_extraction")

        cmd = mock_exec.call_args_list[0][0]
        assert cmd[:6] == ("sudo", "-H", "-D", str(tmp_path), "-u", "some_other_user")
        # preserve-env follows the sudo identity options; the value carries every
        # _OPENCODE_PRESERVED_AUTH_VARS entry in a comma-separated CSV.
        assert cmd[6].startswith("--preserve-env=")
        csv = cmd[6][len("--preserve-env=") :]
        preserved = set(csv.split(","))
        assert preserved == set(_OPENCODE_PRESERVED_AUTH_VARS)
        # `--` ends the sudo options; the wrapped argv (opencode acp)
        # follows.
        assert cmd[7] == "--"
        assert cmd[8] == "opencode"
        assert cmd[9] == "acp"
        assert mock_exec.call_args.kwargs["cwd"] is None

    @pytest.mark.asyncio
    async def test_sudo_wrap_uses_start_new_session_true(self, tmp_path):
        """The wrap path passes `start_new_session=True` so a future
        cross-user kill has a process group to target. Direct path
        keeps the default; the wrap path needs the new session because
        signal permissions prevent the bot user from signaling the
        target user's descendants directly."""
        reasoner = OpenCodeOneShotReasoner(cwd=tmp_path, os_user="some_other_user")
        proc = _make_acp_proc(
            _opencode_handshake_lines() + [_rpc_response_line(request_id=3, result={"stopReason": "end_turn"})]
        )

        async def _noop_kill(**kwargs) -> None:
            return None

        with (
            patch("kai.oneshot.resolve_claude_user", side_effect=lambda u: u),
            patch("kai.oneshot._kill_target_user_tree", side_effect=_noop_kill),
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec,
        ):
            await reasoner.run(prompt="hi", purpose="fact_extraction")

        spawn_kwargs = mock_exec.call_args_list[0].kwargs
        assert spawn_kwargs["start_new_session"] is True


# ── Goose reasoner tests ─────────────────────────────────────────────


class TestGooseOneShotReasonerArgvAndEnv:
    """Argv shape, provider wire-name translation, stdin payload, and
    the env allow-list of the goose one-shot."""

    @pytest.mark.asyncio
    async def test_argv_shape_with_provider_translation(self, tmp_path):
        """The full `goose run` argv, with the Kai provider key
        translated to goose's wire name (deepseek is custom_deepseek
        on the goose side) and the registry model untranslated."""
        reasoner = GooseOneShotReasoner(cwd=tmp_path, os_user=_current_user(), provider="deepseek")
        proc = _make_proc(stdout=b'{"facts": [], "has_episode": false}')
        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                model="deepseek-v4-flash",
                json_schema={"type": "object"},
            )
        cmd = list(mock_exec.call_args[0])
        assert cmd == [
            "goose",
            "run",
            "-i",
            "-",
            "--provider",
            "custom_deepseek",
            "--model",
            "deepseek-v4-flash",
            "-q",
            "--no-session",
            "--no-profile",
            "--max-turns",
            "1",
        ]

    @pytest.mark.asyncio
    async def test_argv_omits_provider_when_empty(self, tmp_path):
        reasoner = GooseOneShotReasoner(cwd=tmp_path, os_user=_current_user())
        proc = _make_proc(stdout=b"{}")
        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(prompt="p", purpose="fact_extraction", model="m", json_schema={"type": "object"})
        cmd = list(mock_exec.call_args[0])
        assert "--provider" not in cmd

    @pytest.mark.asyncio
    async def test_argv_omits_model_when_none(self, tmp_path):
        reasoner = GooseOneShotReasoner(cwd=tmp_path, os_user=_current_user(), provider="anthropic")
        proc = _make_proc(stdout=b"{}")
        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(prompt="p", purpose="fact_extraction", json_schema={"type": "object"})
        cmd = list(mock_exec.call_args[0])
        assert "--model" not in cmd
        assert cmd[4] == "--provider"
        assert cmd[5] == "anthropic"

    @pytest.mark.asyncio
    async def test_stdin_wraps_system_prompt_in_boundary(self, tmp_path):
        reasoner = GooseOneShotReasoner(cwd=tmp_path, os_user=_current_user(), provider="anthropic")
        proc = _make_proc(stdout=b"{}")
        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            await reasoner.run(
                prompt="USER TEXT",
                system_prompt="SYSTEM TEXT",
                purpose="fact_extraction",
                json_schema={"type": "object"},
            )
        payload = proc.communicate.call_args.kwargs["input"].decode("utf-8")
        assert "--- BEGIN SYSTEM " in payload
        assert "SYSTEM TEXT" in payload
        assert payload.endswith("USER TEXT")

    @pytest.mark.asyncio
    async def test_stdin_omits_boundary_when_no_system_prompt(self, tmp_path):
        reasoner = GooseOneShotReasoner(cwd=tmp_path, os_user=_current_user(), provider="anthropic")
        proc = _make_proc(stdout=b"{}")
        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            await reasoner.run(prompt="USER TEXT", purpose="fact_extraction", json_schema={"type": "object"})
        payload = proc.communicate.call_args.kwargs["input"].decode("utf-8")
        assert payload == "USER TEXT"

    @pytest.mark.asyncio
    async def test_env_contains_only_allowlisted_keys(self, tmp_path, monkeypatch):
        """Secrets outside the allow-list must not reach the goose
        subprocess; GOOSE_MODEL / GOOSE_PROVIDER stay out too because
        the one-shot passes both as argv flags."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
        monkeypatch.setenv("GOOSE_MODEL", "leak-model")
        monkeypatch.setenv("GOOSE_PROVIDER", "leak-provider")
        monkeypatch.setenv("KAI_WEBHOOK_SECRET", "leak-secret")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "leak-token")
        reasoner = GooseOneShotReasoner(cwd=tmp_path, os_user=_current_user(), provider="anthropic")
        proc = _make_proc(stdout=b"{}")
        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(prompt="p", purpose="fact_extraction", json_schema={"type": "object"})
        env = mock_exec.call_args.kwargs["env"]
        assert env["ANTHROPIC_API_KEY"] == "key"
        assert set(env) <= set(_GOOSE_ENV_ALLOWLIST)
        assert "GOOSE_MODEL" not in env
        assert "GOOSE_PROVIDER" not in env
        assert "KAI_WEBHOOK_SECRET" not in env
        assert "TELEGRAM_BOT_TOKEN" not in env


class TestGooseOneShotReasonerOutput:
    """Free-form and schema-backed output contracts, mirroring the
    codex / opencode reasoner suites."""

    @pytest.mark.asyncio
    async def test_non_schema_returns_stripped_text(self, tmp_path):
        reasoner = GooseOneShotReasoner(cwd=tmp_path, os_user=_current_user(), provider="anthropic")
        proc = _make_proc(stdout=b"  free-form review text  \n")
        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(prompt="p", purpose="pr_review", json_schema=None)
        assert isinstance(result, OneShotResult)
        assert result.backend == "goose"
        assert result.text == "free-form review text"

    @pytest.mark.asyncio
    async def test_schema_backed_success_returns_normalized_envelope(self, tmp_path):
        reasoner = GooseOneShotReasoner(cwd=tmp_path, os_user=_current_user(), provider="deepseek")
        payload = {"facts": [{"content": "x"}], "has_episode": False}
        proc = _make_proc(stdout=json.dumps(payload).encode())
        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                model="deepseek-v4-flash",
                json_schema={"type": "object"},
            )
        envelope = json.loads(result.text)
        assert envelope == {"is_error": False, "structured_output": payload}
        assert result.raw_metadata["resolved_binary"] == "goose"

    @pytest.mark.asyncio
    async def test_schema_backed_recovers_fenced_payload_with_preamble(self, tmp_path):
        """Goose has no structured-output channel, so models that
        narrate and fence (deepseek-v4-flash does both) must still
        reach the envelope via the tolerant parser."""
        reasoner = GooseOneShotReasoner(cwd=tmp_path, os_user=_current_user(), provider="deepseek")
        body = 'Let me analyze the exchange.\n\n```json\n{"facts": [], "has_episode": false}\n```'
        proc = _make_proc(stdout=body.encode())
        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            result = await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={"type": "object", "required": ["facts", "has_episode"]},
            )
        envelope = json.loads(result.text)
        assert envelope == {
            "is_error": False,
            "structured_output": {"facts": [], "has_episode": False},
        }

    @pytest.mark.asyncio
    async def test_empty_response_raises_output_error(self, tmp_path):
        reasoner = GooseOneShotReasoner(cwd=tmp_path, os_user=_current_user(), provider="anthropic")
        proc = _make_proc(stdout=b"   \n")
        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(OneShotOutputError),
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction", json_schema={"type": "object"})

    @pytest.mark.asyncio
    async def test_invalid_json_raises_output_error(self, tmp_path):
        reasoner = GooseOneShotReasoner(cwd=tmp_path, os_user=_current_user(), provider="anthropic")
        proc = _make_proc(stdout=b"not json at all")
        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(OneShotOutputError),
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction", json_schema={"type": "object"})

    @pytest.mark.asyncio
    async def test_invalid_json_error_carries_response_snippet(self, tmp_path):
        """The invalid_json error message carries the start of the
        offending text so non-JSON output is self-diagnosing in the
        operator log. This is the goose auth-failure surface: goose
        prints auth errors to stdout and exits 0, so without the
        snippet an expired credential reads as a bare position-only
        parse error."""
        reasoner = GooseOneShotReasoner(cwd=tmp_path, os_user=_current_user(), provider="anthropic")
        proc = _make_proc(stdout=b"Error: no credentials configured for provider anthropic")
        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(OneShotOutputError, match=r"no credentials configured"),
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction", json_schema={"type": "object"})

    @pytest.mark.asyncio
    @pytest.mark.parametrize("non_object_payload", ['"a plain string"', "[]", "42", "true", "null"])
    async def test_non_object_json_raises_output_error(self, tmp_path, non_object_payload):
        reasoner = GooseOneShotReasoner(cwd=tmp_path, os_user=_current_user(), provider="anthropic")
        proc = _make_proc(stdout=non_object_payload.encode())
        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(OneShotOutputError),
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction", json_schema={"type": "object"})

    @pytest.mark.asyncio
    async def test_missing_required_fields_raises_output_error(self, tmp_path):
        reasoner = GooseOneShotReasoner(cwd=tmp_path, os_user=_current_user(), provider="anthropic")
        proc = _make_proc(stdout=b'{"facts": []}')
        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(OneShotOutputError) as excinfo,
        ):
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={"type": "object", "required": ["facts", "has_episode"]},
            )
        assert "has_episode" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_non_zero_exit_raises_subprocess_error(self, tmp_path):
        reasoner = GooseOneShotReasoner(cwd=tmp_path, os_user=_current_user(), provider="anthropic")
        proc = _make_proc(stdout=b"", stderr=b"boom", returncode=3)
        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(OneShotSubprocessError) as excinfo,
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction", json_schema={"type": "object"})
        assert excinfo.value.returncode == 3
        assert excinfo.value.stderr == b"boom"

    @pytest.mark.asyncio
    async def test_timeout_kills_and_raises(self, tmp_path):
        reasoner = GooseOneShotReasoner(cwd=tmp_path, os_user=_current_user(), provider="anthropic")
        proc = _make_proc(raise_timeout=True)
        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(OneShotTimeout),
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction", timeout=0.01, json_schema={"type": "object"})
        proc.kill.assert_called_once()
        proc.wait.assert_awaited()

    @pytest.mark.asyncio
    async def test_binary_resolution_failure_raises_routing_error(self, tmp_path):
        from kai.oneshot_binary import BinaryResolutionError

        reasoner = GooseOneShotReasoner(cwd=tmp_path, os_user=_current_user(), provider="anthropic")
        with (
            patch(
                "kai.oneshot.resolve_oneshot_binary",
                side_effect=BinaryResolutionError("could not resolve goose binary"),
            ),
            pytest.raises(OneShotRoutingError),
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction", json_schema={"type": "object"})


class TestGooseStdinRender:
    """`_render_goose_stdin` boundary contract."""

    def test_none_system_prompt_returns_prompt_unchanged(self):
        assert _render_goose_stdin(None, "hello") == "hello"

    def test_empty_system_prompt_returns_prompt_unchanged(self):
        assert _render_goose_stdin("", "hello") == "hello"

    def test_system_prompt_rides_boundary_block(self):
        rendered = _render_goose_stdin("SYS", "USER")
        assert rendered.index("SYS") < rendered.index("USER")
        assert "--- BEGIN SYSTEM " in rendered
        assert "--- END SYSTEM " in rendered


@pytest.mark.routing_test
class TestGooseRouting:
    """Sudo wrap and preserve-env contract for the goose one-shot,
    mirroring the codex / opencode routing classes."""

    @pytest.mark.asyncio
    async def test_goose_runs_in_process_when_os_user_matches_bot(self, tmp_path):
        reasoner = GooseOneShotReasoner(cwd=tmp_path, os_user=_current_user(), provider="anthropic")
        proc = _make_proc(stdout=b"{}")
        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(prompt="p", purpose="fact_extraction", json_schema={"type": "object"})
        cmd = mock_exec.call_args[0]
        assert cmd[0] != "sudo"
        assert mock_exec.call_args.kwargs["start_new_session"] is False

    @pytest.mark.asyncio
    async def test_goose_wraps_with_preserve_env(self, tmp_path):
        reasoner = GooseOneShotReasoner(cwd=tmp_path, os_user="other-user", provider="deepseek")
        proc = _make_proc(stdout=b"{}")
        with (
            patch("kai.oneshot.resolve_claude_user", return_value="other-user"),
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec,
        ):
            await reasoner.run(prompt="p", purpose="fact_extraction", json_schema={"type": "object"})
        cmd = list(mock_exec.call_args[0])
        assert cmd[:6] == ["sudo", "-H", "-D", str(tmp_path), "-u", "other-user"]
        assert cmd[6] == (
            "--preserve-env=ANTHROPIC_API_KEY,OPENAI_API_KEY,GOOGLE_API_KEY,OPENROUTER_API_KEY,DEEPSEEK_API_KEY,"
            "ANTHROPIC_HOST,OPENAI_HOST,OPENAI_BASE_PATH,OLLAMA_HOST"
        )
        assert cmd[7] == "--"
        assert cmd[8] == "goose"
        assert mock_exec.call_args.kwargs["cwd"] is None
        assert mock_exec.call_args.kwargs["start_new_session"] is True
