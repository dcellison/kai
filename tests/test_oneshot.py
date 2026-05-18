"""
Tests for `kai.oneshot` - the provider-agnostic one-shot reasoning
boundary used by semantic memory extraction.

Coverage focus: subprocess mechanics that the reasoner owns. The
memory-specific contracts (envelope parsing, schema validation, fact
storage) live in `tests/test_memory_extraction.py` and stay there.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.oneshot import (
    _SUBPROCESS_ENV_ALLOWLIST,
    ClaudeOneShotReasoner,
    OneShotResult,
    OneShotSubprocessError,
    OneShotTimeout,
)


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
    sandboxing flags, no --bare, no --max-budget-usd."""

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
    async def test_argv_omits_max_budget_usd_flag(self, tmp_path):
        """Max-plan OAuth makes --max-budget-usd a phantom signal; runaway
        protection is the wait_for timeout. The reasoner must not emit
        the flag regardless of how a future config knob looks."""
        reasoner = ClaudeOneShotReasoner(cwd=tmp_path)
        proc = _make_proc()

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(prompt="p", purpose="fact_extraction")

        cmd = mock_exec.call_args[0]
        assert "--max-budget-usd" not in cmd

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
