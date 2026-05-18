"""
Tests for `kai.oneshot` - the provider-agnostic one-shot reasoning
boundary used by semantic memory extraction.

Coverage focus: subprocess mechanics that the reasoner owns. The
memory-specific contracts (envelope parsing, schema validation, fact
storage) live in `tests/test_memory_extraction.py` and stay there.
"""

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.oneshot import (
    _CODEX_ENV_ALLOWLIST,
    _SUBPROCESS_ENV_ALLOWLIST,
    ClaudeOneShotReasoner,
    CodexOneShotReasoner,
    OneShotOutputError,
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
    JSON event output, trusted-dir skip, ephemeral session, ignore
    user/project rules, neutral cwd, model when provided, and
    --output-schema with a real path only when a schema is supplied."""

    @pytest.mark.asyncio
    async def test_argv_includes_required_flags(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CODEX_BIN", raising=False)
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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
        assert "--ignore-rules" in cmd
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
        per-os_user homebrew installs work without PATH munging."""
        monkeypatch.setenv("CODEX_BIN", "/custom/path/to/codex")
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
        proc = _make_proc(stdout=_codex_envelope_ndjson({"ok": True}))

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(
                prompt="p",
                purpose="fact_extraction",
                json_schema={"type": "object"},
            )

        cmd = mock_exec.call_args[0]
        assert cmd[0] == "/custom/path/to/codex"

    @pytest.mark.asyncio
    async def test_argv_omits_output_schema_without_schema(self, tmp_path):
        """When the caller passes no schema, --output-schema must be
        absent entirely. The non-schema branch returns raw final text
        and never writes a temp file."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
        proc = _make_proc(stdout=_codex_event("agent_message", "free-form text").encode("utf-8") + b"\n")

        with patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await reasoner.run(prompt="p", purpose="fact_extraction", json_schema=None)

        cmd = mock_exec.call_args[0]
        assert "--output-schema" not in cmd

    @pytest.mark.asyncio
    async def test_argv_omits_model_when_none(self, tmp_path):
        """A None model means the codex CLI's default; the reasoner
        must not emit a stray `--model` with an empty string."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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
        """The path passed via --output-schema must point at a file
        whose contents are the JSON-serialized schema dict."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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

        assert json.loads(captured["body"]) == schema
        # Cleanup ran in the finally block.
        assert not captured["path"].exists()

    @pytest.mark.asyncio
    async def test_schema_file_removed_on_timeout(self, tmp_path):
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
        captured: dict = {}

        def _capture(*args, **kwargs):
            i = args.index("--output-schema")
            captured["path"] = Path(args[i + 1])
            return _make_proc(raise_timeout=True)

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
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
        proc = _make_proc(raise_timeout=True)

        with (
            patch("kai.oneshot.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
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
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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
    async def test_non_schema_success_returns_raw_text(self, tmp_path):
        """A None schema means the caller wants the free-form
        agent_message text back unchanged; no envelope wrapping
        happens. Memory extraction never takes this branch, but the
        protocol permits it for future callers."""
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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


class TestCodexOneShotReasonerLogging:
    """Each codex call emits one structured INFO line keyed by purpose,
    with `backend=codex` and distinct outcomes for success / timeout
    / subprocess_error. Returncode appears on the subprocess_error
    line so log queries can partition by exit code."""

    @pytest.mark.asyncio
    async def test_log_line_includes_backend_codex_on_success(self, tmp_path, caplog):
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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

    @pytest.mark.asyncio
    async def test_log_line_carries_returncode_on_subprocess_error(self, tmp_path, caplog):
        reasoner = CodexOneShotReasoner(cwd=tmp_path)
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
