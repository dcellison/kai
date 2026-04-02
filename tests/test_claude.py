"""
Tests for claude.py persistent subprocess manager.

Covers:
1. Command construction and sudo wrapping (existing tests)
2. Signal handling and process group dispatch (existing tests)
3. Properties: is_alive, session_id
4. _ensure_started() subprocess launch
5. _drain_stderr() background reader
6. _send_locked() stream parsing, error handling, context injection
7. send() lock acquisition
8. _kill(), shutdown(), change_workspace(), restart()
"""

import asyncio
import json
import os
import pwd
import signal
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.claude import PersistentClaude, StreamEvent
from kai.config import WorkspaceConfig

# ── Shared helpers ───────────────────────────────────────────────────


def _make_claude(**kwargs) -> PersistentClaude:
    """Create a PersistentClaude with sensible defaults for testing."""
    defaults = {
        "model": "sonnet",
        "workspace": Path("/tmp/test-workspace"),
        "max_budget_usd": 1.0,
        "timeout_seconds": 30,
    }
    defaults.update(kwargs)
    return PersistentClaude(**defaults)


def _json_line(obj: dict) -> bytes:
    """Encode a dict as a JSON line (bytes with trailing newline)."""
    return json.dumps(obj).encode() + b"\n"


def _make_mock_proc(stdout_lines: list[bytes]) -> MagicMock:
    """
    Build a mock subprocess that yields predefined stdout lines.

    stdout_lines should be a list of bytes, each ending with b"\\n".
    The final entry should be b"" to signal EOF.
    """
    proc = MagicMock()
    proc.returncode = None
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.stdout = MagicMock()
    proc.stdout.readline = AsyncMock(side_effect=stdout_lines)
    proc.stderr = MagicMock()
    proc.stderr.readline = AsyncMock(return_value=b"")
    proc.wait = AsyncMock()
    proc.send_signal = MagicMock()
    return proc


async def _collect_events(claude: PersistentClaude, prompt: str | list = "test") -> list[StreamEvent]:
    """Send a prompt and collect all yielded StreamEvents."""
    events = []
    async for event in claude._send_locked(prompt):
        events.append(event)
    return events


def _system_event(session_id: str = "sess-123") -> bytes:
    """Build a system event JSON line."""
    return _json_line({"type": "system", "session_id": session_id})


def _assistant_event(text: str) -> bytes:
    """Build an assistant event JSON line with a single text block."""
    return _json_line(
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": text}]},
        }
    )


def _result_event(
    text: str = "Final",
    is_error: bool = False,
    session_id: str = "sess-123",
    cost: float = 0.05,
    duration: int = 1500,
) -> bytes:
    """Build a result event JSON line."""
    return _json_line(
        {
            "type": "result",
            "result": text,
            "is_error": is_error,
            "session_id": session_id,
            "total_cost_usd": cost,
            "duration_ms": duration,
        }
    )


# ── Command construction ─────────────────────────────────────────────


class TestCommandConstruction:
    """Verify _ensure_started() builds the right command depending on claude_user."""

    @pytest.mark.asyncio
    async def test_cmd_without_claude_user(self):
        """Without claude_user, command starts with 'claude' directly."""
        claude = _make_claude()

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = AsyncMock()
            mock_exec.return_value = mock_proc

            await claude._ensure_started()

            args = mock_exec.call_args
            cmd = args[0]
            assert cmd[0] == "claude"
            assert "sudo" not in cmd
            # Should NOT use start_new_session when running as same user
            assert args[1].get("start_new_session") is False

    @pytest.mark.asyncio
    async def test_cmd_with_claude_user(self):
        """With claude_user set, command is prefixed with 'sudo -u <user> --'."""
        claude = _make_claude(claude_user="daniel")

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = AsyncMock()
            mock_exec.return_value = mock_proc

            await claude._ensure_started()

            args = mock_exec.call_args
            cmd = args[0]
            assert cmd[0] == "sudo"
            assert cmd[1] == "-u"
            assert cmd[2] == "daniel"
            assert cmd[3] == "--preserve-env=KAI_WEBHOOK_SECRET"
            assert cmd[4] == "--"
            assert cmd[5] == "claude"

    @pytest.mark.asyncio
    async def test_start_new_session_true_with_claude_user(self):
        """start_new_session=True when claude_user is set (process group isolation)."""
        claude = _make_claude(claude_user="daniel")

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = AsyncMock()
            mock_exec.return_value = mock_proc

            await claude._ensure_started()

            args = mock_exec.call_args
            assert args[1].get("start_new_session") is True

    @pytest.mark.asyncio
    async def test_self_sudo_skipped(self, caplog):
        """When claude_user matches the current process user, sudo is skipped."""
        try:
            current_user = pwd.getpwuid(os.getuid()).pw_name
        except KeyError:
            pytest.skip("UID has no passwd entry; self-sudo-skip path not reachable")
        claude = _make_claude(claude_user=current_user)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = AsyncMock()
            mock_exec.return_value = mock_proc

            with caplog.at_level("WARNING", logger="kai.config"):
                await claude._ensure_started()

            args = mock_exec.call_args
            cmd = args[0]
            # Command should NOT start with sudo
            assert cmd[0] == "claude"
            assert "sudo" not in cmd
            # No process group isolation
            assert args[1].get("start_new_session") is False
            assert claude._pgid is None
            # Warning was logged by resolve_claude_user()
            assert "skipping sudo" in caplog.text

    @pytest.mark.asyncio
    async def test_different_user_still_uses_sudo(self):
        """When claude_user is a different user, sudo is still used."""
        claude = _make_claude(claude_user="some_other_user")

        # Patch pwd.getpwuid in config.py (where resolve_claude_user lives)
        # to return a fixed value so the test doesn't depend on the real
        # system user being different from "some_other_user".
        mock_pw = MagicMock(pw_name="kai")
        with (
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("kai.config.pwd.getpwuid", return_value=mock_pw),
        ):
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = AsyncMock()
            mock_exec.return_value = mock_proc

            await claude._ensure_started()

            args = mock_exec.call_args
            cmd = args[0]
            assert cmd[0] == "sudo"
            assert cmd[1] == "-u"
            assert cmd[2] == "some_other_user"
            assert cmd[3] == "--preserve-env=KAI_WEBHOOK_SECRET"
            assert cmd[4] == "--"
            assert args[1].get("start_new_session") is True

    @pytest.mark.asyncio
    async def test_pwd_keyerror_falls_through_to_sudo(self):
        """When pwd.getpwuid raises KeyError (unmapped UID), sudo is used."""
        claude = _make_claude(claude_user="container_user")

        # Simulate a container environment where the UID has no passwd entry.
        # Patch in config.py where resolve_claude_user() calls pwd.getpwuid.
        with (
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("kai.config.pwd.getpwuid", side_effect=KeyError("uid not found")),
        ):
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = AsyncMock()
            mock_exec.return_value = mock_proc

            await claude._ensure_started()

            args = mock_exec.call_args
            cmd = args[0]
            # Should still wrap with sudo since we can't determine the user.
            assert cmd[0] == "sudo"
            assert cmd[2] == "container_user"
            assert args[1].get("start_new_session") is True

    @pytest.mark.asyncio
    async def test_sudo_preserves_webhook_secret_flag(self):
        """Sudo invocation includes --preserve-env for the webhook secret."""
        claude = _make_claude(claude_user="some_other_user", webhook_secret="s3cret")

        mock_pw = MagicMock(pw_name="kai")
        with (
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("kai.config.pwd.getpwuid", return_value=mock_pw),
        ):
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = AsyncMock()
            mock_exec.return_value = mock_proc

            await claude._ensure_started()

            cmd = mock_exec.call_args[0]
            assert "--preserve-env=KAI_WEBHOOK_SECRET" in cmd
            # Secret is also in the env dict (unchanged behavior).
            env = mock_exec.call_args[1]["env"]
            assert env["KAI_WEBHOOK_SECRET"] == "s3cret"

    @pytest.mark.asyncio
    async def test_no_preserve_env_without_sudo(self):
        """No --preserve-env flag when not using sudo."""
        claude = _make_claude(webhook_secret="s3cret")

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = AsyncMock()
            mock_exec.return_value = mock_proc

            await claude._ensure_started()

            cmd = mock_exec.call_args[0]
            assert "sudo" not in cmd
            assert "--preserve-env=KAI_WEBHOOK_SECRET" not in cmd

    @pytest.mark.asyncio
    async def test_max_context_window_in_cmd(self):
        """--settings with maxContextWindow is added when max_context_window > 0."""
        claude = _make_claude(max_context_window=200000)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = AsyncMock()
            mock_exec.return_value = mock_proc

            await claude._ensure_started()

            cmd = mock_exec.call_args[0]
            assert "--settings" in cmd
            idx = cmd.index("--settings")
            settings = json.loads(cmd[idx + 1])
            assert settings["preferences"]["maxContextWindow"] == 200000

    @pytest.mark.asyncio
    async def test_no_context_window_flag_when_zero(self):
        """--settings for context window is omitted when max_context_window is 0."""
        claude = _make_claude(max_context_window=0)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = AsyncMock()
            mock_exec.return_value = mock_proc

            await claude._ensure_started()

            cmd = mock_exec.call_args[0]
            assert "--settings" not in cmd

    @pytest.mark.asyncio
    async def test_autocompact_pct_in_env(self):
        """CLAUDE_AUTOCOMPACT_PCT_OVERRIDE is set in subprocess env when configured."""
        claude = _make_claude(autocompact_pct=80)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = AsyncMock()
            mock_exec.return_value = mock_proc

            await claude._ensure_started()

            env = mock_exec.call_args[1]["env"]
            assert env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "80"

    @pytest.mark.asyncio
    async def test_no_autocompact_env_when_zero(self):
        """CLAUDE_AUTOCOMPACT_PCT_OVERRIDE is not set when autocompact_pct is 0."""
        claude = _make_claude(autocompact_pct=0)

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = AsyncMock()
            mock_exec.return_value = mock_proc

            await claude._ensure_started()

            env = mock_exec.call_args[1]["env"]
            assert "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE" not in env


# ── Process signal handling ──────────────────────────────────────────


class TestProcessSignals:
    """Verify _send_signal() and force_kill() use the right signal strategy."""

    def test_force_kill_same_user(self):
        """Without claude_user, force_kill sends SIGKILL via proc.send_signal()."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.pid = 12345
        claude._proc = mock_proc

        claude.force_kill()

        mock_proc.send_signal.assert_called_once_with(signal.SIGKILL)

    def test_force_kill_different_user(self):
        """With claude_user, force_kill sends SIGKILL via saved PGID."""
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        claude._proc = mock_proc
        claude._pgid = 12345  # Saved at spawn time

        with patch("os.killpg") as mock_killpg:
            claude.force_kill()

            # Uses saved PGID, not os.getpgid()
            mock_killpg.assert_called_once_with(12345, signal.SIGKILL)

    def test_send_signal_noop_when_no_process(self):
        """_send_signal() is a no-op when there's no subprocess or PGID."""
        claude = _make_claude()
        # _proc is None, _pgid is None by default; should not raise
        claude._send_signal(signal.SIGKILL)

    def test_send_signal_ignores_returncode(self):
        """_send_signal() sends signal even when returncode is set.

        This is the key behavioral change from the old _kill_proc(). When
        claude_user is set, self._proc tracks sudo - if sudo exits before
        claude, we still need to signal the process group via saved PGID.
        """
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.returncode = 0  # sudo already exited
        mock_proc.pid = 12345
        claude._proc = mock_proc
        claude._pgid = 12345

        with patch("os.killpg") as mock_killpg:
            claude._send_signal(signal.SIGKILL)

            # Signal sent despite returncode being set
            mock_killpg.assert_called_once_with(12345, signal.SIGKILL)

    def test_send_signal_handles_process_lookup_error(self):
        """_send_signal() swallows ProcessLookupError (process already dead)."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.send_signal.side_effect = ProcessLookupError
        claude._proc = mock_proc

        # Should not raise
        claude._send_signal(signal.SIGKILL)

    def test_send_signal_handles_permission_error_with_pgid(self):
        """_send_signal() swallows PermissionError on killpg()."""
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        claude._proc = mock_proc
        claude._pgid = 12345

        with patch("os.killpg", side_effect=PermissionError):
            # Should not raise
            claude._send_signal(signal.SIGKILL)

    def test_force_kill_cancels_stderr_task(self):
        """force_kill cancels the stderr drain task."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.pid = 12345
        claude._proc = mock_proc

        mock_task = MagicMock()
        claude._stderr_task = mock_task

        claude.force_kill()

        mock_task.cancel.assert_called_once()
        assert claude._stderr_task is None

    def test_force_kill_no_stderr_task(self):
        """force_kill works when _stderr_task is None."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.pid = 12345
        claude._proc = mock_proc
        claude._stderr_task = None

        # Should not raise
        claude.force_kill()


# ── Properties ───────────────────────────────────────────────────────


class TestProperties:
    def test_is_alive_no_process(self):
        """is_alive returns False when _proc is None (initial state)."""
        claude = _make_claude()
        assert claude.is_alive is False

    def test_is_alive_running(self):
        """is_alive returns True when process exists and returncode is None."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        claude._proc = mock_proc
        assert claude.is_alive is True

    def test_is_alive_exited(self):
        """is_alive returns False when process has exited (returncode set)."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        claude._proc = mock_proc
        assert claude.is_alive is False

    def test_session_id_initial(self):
        """session_id returns None before any interaction."""
        claude = _make_claude()
        assert claude.session_id is None

    def test_session_id_after_set(self):
        """session_id returns the value after it's been set."""
        claude = _make_claude()
        claude._session_id = "sess-abc"
        assert claude.session_id == "sess-abc"


# ── Session age limit ────────────────────────────────────────────────


class TestSessionAgeLimit:
    def test_session_age_hours_no_session(self):
        """Returns 0.0 when no session is active."""
        claude = _make_claude()
        assert claude._session_age_hours() == 0.0

    def test_session_age_hours_running(self):
        """Returns elapsed hours since session started."""
        claude = _make_claude()
        # Simulate a session started 2 hours ago
        claude._session_started_at = time.monotonic() - 7200
        age = claude._session_age_hours()
        assert 1.9 < age < 2.1

    def test_should_recycle_disabled(self):
        """Returns False when max_session_hours is 0 (disabled)."""
        claude = _make_claude(max_session_hours=0)
        mock_proc = MagicMock()
        mock_proc.returncode = None
        claude._proc = mock_proc
        claude._session_started_at = time.monotonic() - 99999
        assert claude._should_recycle() is False

    def test_should_recycle_young_session(self):
        """Returns False when session is younger than the limit."""
        claude = _make_claude(max_session_hours=4)
        mock_proc = MagicMock()
        mock_proc.returncode = None
        claude._proc = mock_proc
        claude._session_started_at = time.monotonic() - 3600  # 1 hour
        assert claude._should_recycle() is False

    def test_should_recycle_expired_session(self):
        """Returns True when session exceeds the age limit."""
        claude = _make_claude(max_session_hours=4)
        mock_proc = MagicMock()
        mock_proc.returncode = None
        claude._proc = mock_proc
        claude._session_started_at = time.monotonic() - 18000  # 5 hours
        assert claude._should_recycle() is True

    def test_should_recycle_dead_process(self):
        """Returns False when the process is not alive, even if expired."""
        claude = _make_claude(max_session_hours=4)
        mock_proc = MagicMock()
        mock_proc.returncode = 0  # Already exited
        claude._proc = mock_proc
        claude._session_started_at = time.monotonic() - 18000
        assert claude._should_recycle() is False

    @pytest.mark.asyncio
    async def test_recycle_before_ensure_started(self):
        """_send_locked() kills the process before _ensure_started() when expired."""
        claude = _make_claude(max_session_hours=1)

        # _ensure_started must set up _proc so the rest of _send_locked works.
        # We make it set up a mock process that immediately returns EOF so the
        # streaming loop exits cleanly.
        async def fake_ensure_started():
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stdin = MagicMock()
            mock_proc.stdin.write = MagicMock()
            mock_proc.stdin.drain = AsyncMock()
            mock_proc.stdout = MagicMock()
            mock_proc.stdout.readline = AsyncMock(return_value=b"")  # EOF
            claude._proc = mock_proc
            claude._fresh_session = False

        with (
            patch.object(claude, "_should_recycle", return_value=True),
            patch.object(claude, "_kill", new_callable=AsyncMock) as mock_kill,
            patch.object(claude, "_ensure_started", side_effect=fake_ensure_started),
            patch.object(claude, "_session_age_hours", return_value=2.5),
        ):
            events = []
            async for event in claude._send_locked("test"):
                events.append(event)

            # _kill is called at least once for the recycle (and again from
            # the streaming loop's EOF handler, which is expected)
            assert mock_kill.call_count >= 1


# ── _ensure_started ──────────────────────────────────────────────────


class TestEnsureStarted:
    @pytest.mark.asyncio
    async def test_noop_when_alive(self):
        """No-op when process is already alive."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        claude._proc = mock_proc

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            await claude._ensure_started()
        mock_exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_passes_webhook_secret_in_env(self):
        """Webhook secret is passed via KAI_WEBHOOK_SECRET env var."""
        claude = _make_claude(webhook_secret="my-secret")

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = AsyncMock()
            mock_exec.return_value = mock_proc

            await claude._ensure_started()

            call_kwargs = mock_exec.call_args[1]
            assert call_kwargs["env"]["KAI_WEBHOOK_SECRET"] == "my-secret"

    @pytest.mark.asyncio
    async def test_sets_fresh_session(self):
        """_fresh_session is True after starting a new process."""
        claude = _make_claude()
        claude._fresh_session = False  # Simulate a prior session

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = AsyncMock()
            mock_exec.return_value = mock_proc

            await claude._ensure_started()

        assert claude._fresh_session is True

    @pytest.mark.asyncio
    async def test_sets_session_started_at(self):
        """_ensure_started records the session start time via time.monotonic()."""
        claude = _make_claude()
        assert claude._session_started_at is None

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = AsyncMock()
            mock_exec.return_value = mock_proc

            before = time.monotonic()
            await claude._ensure_started()
            after = time.monotonic()

        assert claude._session_started_at is not None
        assert before <= claude._session_started_at <= after

    @pytest.mark.asyncio
    async def test_starts_stderr_drain(self):
        """_ensure_started creates a background task for stderr draining."""
        claude = _make_claude()

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = MagicMock()
            mock_proc.stderr.readline = AsyncMock(return_value=b"")
            mock_exec.return_value = mock_proc

            await claude._ensure_started()

        assert claude._stderr_task is not None


# ── _drain_stderr ────────────────────────────────────────────────────


class TestDrainStderr:
    @pytest.mark.asyncio
    async def test_logs_stderr_at_debug(self, caplog):
        """Stderr lines are logged at DEBUG level."""
        caplog.set_level("DEBUG", logger="kai.claude")
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.readline = AsyncMock(side_effect=[b"some warning\n", b""])
        claude._proc = mock_proc

        await claude._drain_stderr()

        assert "some warning" in caplog.text

    @pytest.mark.asyncio
    async def test_truncates_long_lines(self, caplog):
        """Long stderr lines are truncated to 200 chars in the log."""
        caplog.set_level("DEBUG", logger="kai.claude")
        claude = _make_claude()
        long_line = "x" * 300 + "\n"
        mock_proc = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.readline = AsyncMock(side_effect=[long_line.encode(), b""])
        claude._proc = mock_proc

        await claude._drain_stderr()

        # The log message should contain the truncated text (200 chars max)
        for record in caplog.records:
            if "stderr" in record.message.lower():
                # %s formatting inserts the truncated value
                assert len(record.args[0]) <= 200

    @pytest.mark.asyncio
    async def test_stops_on_eof(self):
        """Stops reading when readline returns empty bytes (EOF)."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.readline = AsyncMock(side_effect=[b"line1\n", b""])
        claude._proc = mock_proc

        await claude._drain_stderr()

        # readline should have been called exactly twice (one line + EOF)
        assert mock_proc.stderr.readline.call_count == 2

    @pytest.mark.asyncio
    async def test_breaks_on_exception(self):
        """Catches exceptions and stops rather than crashing."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.readline = AsyncMock(side_effect=RuntimeError("pipe broken"))
        claude._proc = mock_proc

        # Should not raise
        await claude._drain_stderr()


# ── _send_locked: basic stream parsing ───────────────────────────────


class TestSendLockedBasic:
    """Tests for _send_locked stream parsing and event dispatch."""

    @pytest.fixture(autouse=True)
    def _patch_kill(self, monkeypatch):
        """Prevent _kill from interacting with mock processes after test scenarios."""
        monkeypatch.setattr(PersistentClaude, "_kill", AsyncMock())

    @pytest.mark.asyncio
    async def test_writes_json_to_stdin(self):
        """Sends a JSON-formatted message to the process stdin."""
        proc = _make_mock_proc([_system_event(), _result_event(), b""])
        claude = _make_claude()
        claude._proc = proc
        claude._fresh_session = False

        await _collect_events(claude)

        # Verify stdin.write was called with valid JSON
        written = proc.stdin.write.call_args[0][0]
        msg = json.loads(written.decode())
        assert msg["type"] == "user"
        assert msg["message"]["role"] == "user"

    @pytest.mark.asyncio
    async def test_yields_text_from_assistant_events(self):
        """Assistant events yield StreamEvents with accumulated text."""
        proc = _make_mock_proc(
            [
                _system_event(),
                _assistant_event("Hello"),
                _result_event("Hello"),
                b"",
            ]
        )
        claude = _make_claude()
        claude._proc = proc
        claude._fresh_session = False

        events = await _collect_events(claude)

        # Should have at least a text event and a done event
        text_events = [e for e in events if not e.done]
        assert len(text_events) >= 1
        assert text_events[0].text_so_far == "Hello"

    @pytest.mark.asyncio
    async def test_final_event_has_claude_response(self):
        """Final event has done=True with a complete ClaudeResponse."""
        proc = _make_mock_proc(
            [
                _system_event(),
                _assistant_event("Hello"),
                _result_event("Hello", cost=0.05, duration=1500),
                b"",
            ]
        )
        claude = _make_claude()
        claude._proc = proc
        claude._fresh_session = False

        events = await _collect_events(claude)

        done_event = events[-1]
        assert done_event.done is True
        assert done_event.response is not None
        assert done_event.response.success is True
        assert done_event.response.cost_usd == 0.05
        assert done_event.response.duration_ms == 1500
        assert done_event.response.session_id == "sess-123"

    @pytest.mark.asyncio
    async def test_system_event_sets_session_id(self):
        """System events update the instance's session_id."""
        proc = _make_mock_proc(
            [
                _system_event("new-session-42"),
                _result_event(),
                b"",
            ]
        )
        claude = _make_claude()
        claude._proc = proc
        claude._fresh_session = False

        await _collect_events(claude)

        assert claude._session_id == "new-session-42"

    @pytest.mark.asyncio
    async def test_multiple_assistant_events_accumulate(self):
        """Multiple assistant events accumulate text with \\n\\n separator."""
        proc = _make_mock_proc(
            [
                _system_event(),
                _assistant_event("First"),
                _assistant_event("Second"),
                _result_event(),
                b"",
            ]
        )
        claude = _make_claude()
        claude._proc = proc
        claude._fresh_session = False

        events = await _collect_events(claude)

        # The last text event before done should have both chunks
        text_events = [e for e in events if not e.done]
        last_text = text_events[-1].text_so_far
        assert "First" in last_text
        assert "Second" in last_text
        assert "\n\n" in last_text

    @pytest.mark.asyncio
    async def test_non_text_content_blocks_ignored(self):
        """Content blocks that aren't type=text are skipped."""
        proc = _make_mock_proc(
            [
                _system_event(),
                _json_line(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {"type": "tool_use", "id": "tool-1"},
                                {"type": "text", "text": "Result"},
                            ]
                        },
                    }
                ),
                _result_event(),
                b"",
            ]
        )
        claude = _make_claude()
        claude._proc = proc
        claude._fresh_session = False

        events = await _collect_events(claude)

        text_events = [e for e in events if not e.done]
        assert len(text_events) == 1
        assert text_events[0].text_so_far == "Result"

    @pytest.mark.asyncio
    async def test_non_json_lines_skipped(self):
        """Non-JSON stdout lines are skipped without breaking the stream."""
        proc = _make_mock_proc(
            [
                b"Some startup banner\n",
                _system_event(),
                _result_event(text="Done"),
                b"",
            ]
        )
        claude = _make_claude()
        claude._proc = proc
        claude._fresh_session = False

        events = await _collect_events(claude)

        done_event = events[-1]
        assert done_event.done is True
        assert done_event.response.success is True


# ── _send_locked: error handling ─────────────────────────────────────


class TestSendLockedErrors:
    @pytest.fixture(autouse=True)
    def _patch_kill(self, monkeypatch):
        """Prevent _kill from interacting with mock processes after test scenarios."""
        monkeypatch.setattr(PersistentClaude, "_kill", AsyncMock())

    @pytest.mark.asyncio
    async def test_cli_not_found(self):
        """FileNotFoundError from _ensure_started yields a done event with error."""
        claude = _make_claude()
        claude._proc = None
        claude._fresh_session = False

        with patch.object(claude, "_ensure_started", side_effect=FileNotFoundError):
            events = await _collect_events(claude)

        assert len(events) == 1
        assert events[0].done is True
        assert "not found" in events[0].response.error

    @pytest.mark.asyncio
    async def test_stdin_write_failure(self):
        """OSError on stdin.write kills the process and yields an error event."""
        proc = _make_mock_proc([])
        proc.stdin.write = MagicMock(side_effect=OSError("broken pipe"))
        claude = _make_claude()
        claude._proc = proc
        claude._fresh_session = False

        events = await _collect_events(claude)

        assert len(events) == 1
        assert events[0].done is True
        assert events[0].response.success is False
        assert "died" in events[0].response.error

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Timeout on stdout.readline yields a 'timed out' error event."""
        proc = _make_mock_proc([])
        # Override readline to simulate timeout
        proc.stdout.readline = AsyncMock(side_effect=asyncio.TimeoutError)
        claude = _make_claude()
        claude._proc = proc
        claude._fresh_session = False

        # Patch wait_for to propagate the TimeoutError
        with patch("asyncio.wait_for", side_effect=TimeoutError):
            events = await _collect_events(claude)

        assert events[-1].done is True
        assert "timed out" in events[-1].response.error.lower()

    @pytest.mark.asyncio
    async def test_idle_timeout(self):
        """Interaction killed when process goes silent longer than idle limit.

        The idle timeout (timeout_seconds * 5) is a secondary safety net
        behind the per-readline timeout (timeout_seconds * 3). Because the
        readline timeout is shorter, it normally fires first for a truly
        silent process. This test uses a mocked time source to force the
        idle check to trigger, verifying the error path works correctly.
        """
        claude = _make_claude(timeout_seconds=1)  # idle limit = 5s

        # Control time progression in kai.claude without affecting asyncio.
        # The streaming loop calls time.monotonic() in a fixed pattern:
        #   1. Init: last_activity = time.monotonic()
        #   2. Idle check (iter 1): time.monotonic() - last_activity
        #   3. Reset after readline 1: last_activity = time.monotonic()
        #   4. Idle check (iter 2): time.monotonic() - last_activity
        #   5. Reset after readline 2: last_activity = time.monotonic()
        #   6. Idle check (iter 3): time.monotonic() - last_activity <- jump here
        # Calls 1-5 return small values; call 6+ returns 100.0 so the
        # idle check sees (100.0 - 0.5) > 5s and fires.
        mono_call = [0]

        def fake_monotonic():
            mono_call[0] += 1
            if mono_call[0] <= 5:
                return mono_call[0] * 0.1
            return 100.0

        readline_count = [0]

        async def readline_with_output():
            readline_count[0] += 1
            if readline_count[0] <= 2:
                return _assistant_event(f"Output {readline_count[0]}")
            # If we get here, the idle timeout didn't fire as expected
            pytest.fail("readline reached call 3 - idle timeout did not fire")

        proc = _make_mock_proc([])
        proc.stdout.readline = readline_with_output
        claude._proc = proc
        claude._fresh_session = False

        with patch("kai.claude.time.monotonic", side_effect=fake_monotonic):
            events = await _collect_events(claude)

        # Should get the idle timeout error
        done_event = events[-1]
        assert done_event.done is True
        assert done_event.response.success is False
        assert "no output" in done_event.response.error.lower()

    @pytest.mark.asyncio
    async def test_idle_timeout_normal_completion_unaffected(self):
        """Normal interactions that complete quickly are not affected by idle timeout."""
        proc = _make_mock_proc(
            [
                _system_event(),
                _assistant_event("Hello"),
                _result_event("Done"),
            ]
        )
        claude = _make_claude(timeout_seconds=1)  # idle limit = 5s
        claude._proc = proc
        claude._fresh_session = False

        events = await _collect_events(claude)

        done_event = events[-1]
        assert done_event.done is True
        assert done_event.response.success is True
        assert done_event.response.error is None

    @pytest.mark.asyncio
    async def test_active_process_survives_past_old_wall_clock(self):
        """A process that keeps producing output is not killed by the idle timer.

        This is the core behavioral change: under the old wall-clock limit,
        any interaction exceeding timeout_seconds * 5 was killed regardless
        of activity. The idle timer only fires after prolonged silence.
        """
        claude = _make_claude(timeout_seconds=1)  # idle limit = 5s, old wall-clock = 5s

        # Simulate a process that takes longer than timeout_seconds * 5 total
        # but keeps producing output (resetting the idle timer each time).
        # Uses mocked time so the test runs instantly. Each readline advances
        # time by 0.5s; after 20 calls that is 10s total, well past the old
        # 5s wall-clock limit. The idle timer never fires because each
        # readline resets it to within 0.5s.
        sim_time = [0.0]

        def advancing_monotonic():
            return sim_time[0]

        call_count = [0]

        async def active_readline():
            call_count[0] += 1
            if call_count[0] <= 20:
                sim_time[0] += 0.5  # advance 0.5s per event
                return _assistant_event(f"Working... step {call_count[0]}")
            return _result_event("All done")

        proc = _make_mock_proc([])
        proc.stdout.readline = active_readline
        claude._proc = proc
        claude._fresh_session = False

        with patch("kai.claude.time.monotonic", side_effect=advancing_monotonic):
            events = await _collect_events(claude)

        # Should complete successfully despite total time > timeout_seconds * 5
        done_event = events[-1]
        assert done_event.done is True
        assert done_event.response.success is True
        assert done_event.response.text  # Has content

    @pytest.mark.asyncio
    async def test_eof_with_text(self):
        """EOF with accumulated text yields success=True, error=None."""
        proc = _make_mock_proc(
            [
                _system_event(),
                _assistant_event("Partial response"),
                b"",  # EOF
            ]
        )
        claude = _make_claude()
        claude._proc = proc
        claude._fresh_session = False

        events = await _collect_events(claude)

        done_event = events[-1]
        assert done_event.done is True
        assert done_event.response.success is True
        assert done_event.response.error is None
        assert "Partial response" in done_event.response.text

    @pytest.mark.asyncio
    async def test_eof_without_text(self):
        """EOF without accumulated text yields success=False with error."""
        proc = _make_mock_proc([b""])  # Immediate EOF
        claude = _make_claude()
        claude._proc = proc
        claude._fresh_session = False

        events = await _collect_events(claude)

        done_event = events[-1]
        assert done_event.done is True
        assert done_event.response.success is False
        assert done_event.response.error is not None


# ── _send_locked: result event parsing ───────────────────────────────


class TestSendLockedResult:
    @pytest.fixture(autouse=True)
    def _patch_kill(self, monkeypatch):
        monkeypatch.setattr(PersistentClaude, "_kill", AsyncMock())

    @pytest.mark.asyncio
    async def test_is_error_sets_failure(self):
        """is_error=True in result event sets success=False and error to result text."""
        proc = _make_mock_proc(
            [
                _system_event(),
                _result_event(text="Something went wrong", is_error=True),
                b"",
            ]
        )
        claude = _make_claude()
        claude._proc = proc
        claude._fresh_session = False

        events = await _collect_events(claude)

        done = events[-1]
        assert done.response.success is False
        assert done.response.error == "Something went wrong"

    @pytest.mark.asyncio
    async def test_accumulated_text_preferred_over_result(self):
        """When text has been accumulated, it's used instead of result event text."""
        proc = _make_mock_proc(
            [
                _system_event(),
                _assistant_event("Accumulated text here"),
                _result_event(text="Result text"),
                b"",
            ]
        )
        claude = _make_claude()
        claude._proc = proc
        claude._fresh_session = False

        events = await _collect_events(claude)

        done = events[-1]
        assert done.response.text == "Accumulated text here"

    @pytest.mark.asyncio
    async def test_falls_back_to_result_text(self):
        """When no text accumulated, falls back to result event text."""
        proc = _make_mock_proc(
            [
                _system_event(),
                _result_event(text="Only in result"),
                b"",
            ]
        )
        claude = _make_claude()
        claude._proc = proc
        claude._fresh_session = False

        events = await _collect_events(claude)

        done = events[-1]
        assert done.response.text == "Only in result"


# ── _send_locked: context injection ──────────────────────────────────


class TestContextInjection:
    """Tests for first-message context injection in _send_locked."""

    @pytest.fixture(autouse=True)
    def _patch_kill(self, monkeypatch):
        monkeypatch.setattr(PersistentClaude, "_kill", AsyncMock())

    @pytest.fixture()
    def home_workspace(self, tmp_path, monkeypatch):
        """Create a home workspace with identity file and DATA_DIR memory."""
        home = tmp_path / "home"
        claude_dir = home / ".claude"
        claude_dir.mkdir(parents=True)
        (claude_dir / "CLAUDE.md").write_text("You are Kai.")

        # Personal memory now lives under DATA_DIR, not the workspace
        data_dir = tmp_path / "data"
        memory_dir = data_dir / "memory"
        memory_dir.mkdir(parents=True)
        (memory_dir / "MEMORY.md").write_text("User likes concise responses.")
        monkeypatch.setattr("kai.claude.DATA_DIR", data_dir)

        return home

    @pytest.fixture()
    def foreign_workspace(self, tmp_path):
        """Create a foreign workspace (different from home)."""
        foreign = tmp_path / "foreign"
        claude_dir = foreign / ".claude"
        claude_dir.mkdir(parents=True)
        (claude_dir / "MEMORY.md").write_text("Foreign workspace memory.")
        return foreign

    def _extract_prompt(self, proc: MagicMock) -> str:
        """Extract the prompt text from what was written to stdin."""
        written = proc.stdin.write.call_args[0][0]
        msg = json.loads(written.decode())
        content = msg["message"]["content"]
        # Content is a list of blocks; concatenate all text blocks
        return "\n".join(block["text"] for block in content if block.get("type") == "text")

    @pytest.mark.asyncio
    async def test_first_message_injects_context(self, home_workspace):
        """First message in a session prepends identity, memory, and history."""
        proc = _make_mock_proc([_system_event(), _result_event(), b""])
        claude = _make_claude(
            workspace=home_workspace,
            home_workspace=home_workspace,
            webhook_secret="secret",
        )
        claude._proc = proc
        claude._fresh_session = True

        with patch("kai.claude.get_recent_history", return_value="User: hello\nKai: hi"):
            await _collect_events(claude, "What's up?")

        prompt = self._extract_prompt(proc)
        # Memory should be injected (home workspace memory)
        assert "User likes concise responses" in prompt
        # History should be injected
        assert "Recent conversations" in prompt
        # The actual user prompt should be at the end
        assert "What's up?" in prompt

    @pytest.mark.asyncio
    async def test_second_message_no_injection(self, home_workspace):
        """Second message does NOT re-inject context (fresh_session is False)."""
        proc = _make_mock_proc([_system_event(), _result_event(), b""])
        claude = _make_claude(workspace=home_workspace, home_workspace=home_workspace)
        claude._proc = proc
        claude._fresh_session = False

        await _collect_events(claude, "Follow-up question")

        prompt = self._extract_prompt(proc)
        # Should just be the raw prompt, no injected sections
        assert "persistent memory" not in prompt.lower()
        assert "Follow-up question" in prompt

    @pytest.mark.asyncio
    async def test_foreign_workspace_injects_identity(self, home_workspace, foreign_workspace):
        """Foreign workspace injects identity from home and per-message reminder."""
        proc = _make_mock_proc([_system_event(), _result_event(), b""])
        claude = _make_claude(
            workspace=foreign_workspace,
            home_workspace=home_workspace,
        )
        claude._proc = proc
        claude._fresh_session = True

        with patch("kai.claude.get_recent_history", return_value=""):
            await _collect_events(claude, "Help me")

        prompt = self._extract_prompt(proc)
        # Identity from home workspace should be injected
        assert "You are Kai" in prompt
        # Foreign workspace memory should NOT be injected (Claude Code reads
        # it natively from cwd; bot-side reads risk PermissionError on Linux)
        assert "Foreign workspace memory" not in prompt
        # Per-message reminder should be present
        assert "IMPORTANT" in prompt
        assert "Respond ONLY" in prompt

    @pytest.mark.asyncio
    async def test_home_workspace_no_identity_injection(self, home_workspace):
        """Home workspace does NOT inject identity (Claude Code reads it natively)."""
        proc = _make_mock_proc([_system_event(), _result_event(), b""])
        claude = _make_claude(
            workspace=home_workspace,
            home_workspace=home_workspace,
        )
        claude._proc = proc
        claude._fresh_session = True

        with patch("kai.claude.get_recent_history", return_value=""):
            await _collect_events(claude, "Test")

        prompt = self._extract_prompt(proc)
        # Identity text should NOT be injected when in home workspace
        assert "core identity" not in prompt.lower()
        # No per-message reminder either
        assert "Respond ONLY" not in prompt

    @pytest.mark.asyncio
    async def test_webhook_secret_injects_api_info(self, home_workspace):
        """When webhook secret is set, scheduling and file API info are injected."""
        proc = _make_mock_proc([_system_event(), _result_event(), b""])
        claude = _make_claude(
            workspace=home_workspace,
            home_workspace=home_workspace,
            webhook_secret="my-secret",
            webhook_port=8080,
        )
        claude._proc = proc
        claude._fresh_session = True

        with patch("kai.claude.get_recent_history", return_value=""):
            await _collect_events(claude, "Test")

        prompt = self._extract_prompt(proc)
        assert "Scheduling API" in prompt
        assert "File API" in prompt
        assert "8080" in prompt

    @pytest.mark.asyncio
    async def test_no_webhook_secret_no_api_info(self, home_workspace):
        """Without webhook secret, no API info is injected."""
        proc = _make_mock_proc([_system_event(), _result_event(), b""])
        claude = _make_claude(
            workspace=home_workspace,
            home_workspace=home_workspace,
            webhook_secret="",
        )
        claude._proc = proc
        claude._fresh_session = True

        with patch("kai.claude.get_recent_history", return_value=""):
            await _collect_events(claude, "Test")

        prompt = self._extract_prompt(proc)
        assert "Scheduling API" not in prompt
        assert "File API" not in prompt

    @pytest.mark.asyncio
    async def test_services_info_injected(self, home_workspace):
        """Available services info is injected when services are configured."""
        proc = _make_mock_proc([_system_event(), _result_event(), b""])
        claude = _make_claude(
            workspace=home_workspace,
            home_workspace=home_workspace,
            webhook_secret="secret",
            services_info=[
                {
                    "name": "perplexity",
                    "method": "POST",
                    "description": "Web search",
                    "notes": "Use sonar model",
                }
            ],
        )
        claude._proc = proc
        claude._fresh_session = True

        with patch("kai.claude.get_recent_history", return_value=""):
            await _collect_events(claude, "Test")

        prompt = self._extract_prompt(proc)
        assert "perplexity" in prompt
        assert "Web search" in prompt
        assert "sonar" in prompt

    @pytest.mark.asyncio
    async def test_api_prompts_mandate_curl(self, home_workspace):
        """API prompt sections explicitly mandate curl over WebFetch."""
        proc = _make_mock_proc([_system_event(), _result_event(), b""])
        claude = _make_claude(
            workspace=home_workspace,
            home_workspace=home_workspace,
            webhook_secret="test-secret",
            # Services info needed to trigger the External Services section
            services_info=[
                {"name": "test-svc", "method": "POST", "description": "Test", "notes": ""},
            ],
        )
        claude._proc = proc
        claude._fresh_session = True

        with patch("kai.claude.get_recent_history", return_value=""):
            await _collect_events(claude, "test message")

        prompt = self._extract_prompt(proc)
        # All four API sections should mandate curl.
        assert prompt.count("use curl (NEVER WebFetch)") >= 4


# ── _send_locked: multi-modal prompt ─────────────────────────────────


class TestMultiModalPrompt:
    @pytest.fixture(autouse=True)
    def _patch_kill(self, monkeypatch):
        monkeypatch.setattr(PersistentClaude, "_kill", AsyncMock())

    @pytest.mark.asyncio
    async def test_list_prompt_with_context_injection(self, tmp_path, monkeypatch):
        """List prompts get context prepended as text blocks, content sent as-is."""
        home = tmp_path / "home"
        claude_dir = home / ".claude"
        claude_dir.mkdir(parents=True)

        # Personal memory lives under DATA_DIR, not the workspace
        data_dir = tmp_path / "data"
        memory_dir = data_dir / "memory"
        memory_dir.mkdir(parents=True)
        (memory_dir / "MEMORY.md").write_text("Some memory")
        monkeypatch.setattr("kai.claude.DATA_DIR", data_dir)

        proc = _make_mock_proc([_system_event(), _result_event(), b""])
        claude = _make_claude(
            workspace=home,
            home_workspace=home,
            webhook_secret="secret",
        )
        claude._proc = proc
        claude._fresh_session = True

        # Multi-modal prompt (e.g., image + text)
        prompt_list = [
            {"type": "image", "source": {"data": "base64data"}},
            {"type": "text", "text": "What's in this image?"},
        ]

        with patch("kai.claude.get_recent_history", return_value=""):
            await _collect_events(claude, prompt_list)

        written = proc.stdin.write.call_args[0][0]
        msg = json.loads(written.decode())
        content = msg["message"]["content"]

        # First block should be injected context (text type)
        assert content[0]["type"] == "text"
        assert "Some memory" in content[0]["text"]

        # Original content blocks should follow
        assert content[-1]["type"] == "text"
        assert content[-1]["text"] == "What's in this image?"


# ── send() lock acquisition ──────────────────────────────────────────


class TestSendLock:
    @pytest.mark.asyncio
    async def test_acquires_lock(self):
        """send() acquires the internal lock before calling _send_locked."""
        proc = _make_mock_proc([_system_event(), _result_event(), b""])
        claude = _make_claude()
        claude._proc = proc
        claude._fresh_session = False

        # Patch _kill to prevent cleanup issues
        with patch.object(claude, "_kill", new_callable=AsyncMock):
            lock_was_held = False

            # Wrap _send_locked to check if the lock is held when it runs
            original = claude._send_locked

            async def checking_send(prompt, chat_id=None):
                nonlocal lock_was_held
                lock_was_held = claude._lock.locked()
                async for event in original(prompt, chat_id=chat_id):
                    yield event

            with patch.object(claude, "_send_locked", checking_send):
                async for _ in claude.send("test"):
                    pass

            assert lock_was_held is True


# ── _kill ────────────────────────────────────────────────────────────


class TestKill:
    @pytest.mark.asyncio
    async def test_kills_and_clears_state(self):
        """_kill sends SIGKILL, waits, and clears all process state."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.send_signal = MagicMock()
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc
        claude._session_id = "sess-123"
        claude._session_started_at = 12345.0

        await claude._kill()

        mock_proc.send_signal.assert_called_with(signal.SIGKILL)
        assert claude._proc is None
        assert claude._pgid is None
        assert claude._session_id is None
        assert claude._session_started_at is None

    @pytest.mark.asyncio
    async def test_clears_pgid_with_claude_user(self):
        """_kill clears _pgid when claude_user is set."""
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc
        claude._pgid = 12345

        with patch("os.killpg"):
            await claude._kill()

        assert claude._pgid is None
        assert claude._proc is None

    @pytest.mark.asyncio
    async def test_cancels_stderr_task(self):
        """_kill cancels the stderr drain task before clearing proc."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.send_signal = MagicMock()
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc

        mock_task = MagicMock()
        claude._stderr_task = mock_task

        await claude._kill()

        mock_task.cancel.assert_called_once()
        assert claude._stderr_task is None

    @pytest.mark.asyncio
    async def test_stderr_cancelled_before_proc_cleared(self):
        """_kill cancels stderr task while self._proc is still set."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.send_signal = MagicMock()
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc

        proc_at_cancel_time: list[object] = []

        def tracking_cancel():
            # Record whether self._proc is still set when cancel is called
            proc_at_cancel_time.append(claude._proc)

        mock_task = MagicMock()
        mock_task.cancel = MagicMock(side_effect=tracking_cancel)
        claude._stderr_task = mock_task

        await claude._kill()

        # stderr task was cancelled while proc was still set (not None)
        assert len(proc_at_cancel_time) == 1
        assert proc_at_cancel_time[0] is mock_proc
        # After _kill completes, proc is cleared
        assert claude._proc is None

    @pytest.mark.asyncio
    async def test_idempotent_no_process(self):
        """_kill is idempotent when _proc is already None."""
        claude = _make_claude()
        # Should not raise
        await claude._kill()

    @pytest.mark.asyncio
    async def test_wait_timeout_does_not_hang(self):
        """_kill does not hang if wait() times out after SIGKILL."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.send_signal = MagicMock()
        # wait() never completes - simulates a zombie process
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc

        async def timeout_wait(coro, timeout):
            coro.close()
            raise TimeoutError

        with patch("asyncio.wait_for", side_effect=timeout_wait):
            await claude._kill()

        # State is cleaned up even when wait times out
        assert claude._proc is None
        assert claude._session_id is None

    @pytest.mark.asyncio
    async def test_kill_signals_saved_pgid_after_clearing(self):
        """_kill sends a final SIGKILL to the saved pgid after clearing state.

        This is the core fix for the orphan race: even after self._pgid is
        cleared (making subsequent _kill() calls no-ops), the saved pgid
        gets one final signal to catch any claude process that survived
        the initial SIGKILL (e.g., reparented to init after sudo died).
        """
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc
        claude._pgid = 12345

        with patch("os.killpg") as mock_killpg:
            await claude._kill()

        # killpg called at least twice: once from _send_signal (initial SIGKILL)
        # and once from the final cleanup after state is cleared
        killpg_calls = [call.args for call in mock_killpg.call_args_list]
        assert len(killpg_calls) >= 2
        assert (12345, signal.SIGKILL) in killpg_calls

    @pytest.mark.asyncio
    async def test_kill_no_final_signal_without_pgid(self):
        """Final cleanup is skipped when _pgid is None (non-claude_user mode).

        Without claude_user, _proc IS the claude process and send_signal()
        on the proc is sufficient. No process group signal needed.
        """
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.send_signal = MagicMock()
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc
        # _pgid is None (no claude_user)

        with patch("os.killpg") as mock_killpg:
            await claude._kill()

        # killpg should never be called in non-claude_user mode
        mock_killpg.assert_not_called()

    @pytest.mark.asyncio
    async def test_second_kill_is_noop_but_orphan_already_handled(self):
        """Simulates the race: first _kill() handles the orphan, second is a no-op.

        change_workspace() calls _kill() while the streaming loop is active.
        The streaming loop sees EOF and calls _kill() again. The second call
        is a no-op (self._proc is None), but the first call already sent the
        final signal to the saved pgid, so the orphan is handled.
        """
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc
        claude._pgid = 12345

        with patch("os.killpg") as mock_killpg:
            # First call: from change_workspace() - signals and clears state
            await claude._kill()
            first_call_count = mock_killpg.call_count

            # Verify state is cleared
            assert claude._proc is None
            assert claude._pgid is None

            # Second call: from EOF handler - no-op since _proc is None
            await claude._kill()

            # No additional killpg calls from the second _kill()
            assert mock_killpg.call_count == first_call_count


# ── shutdown ─────────────────────────────────────────────────────────


class TestShutdown:
    @pytest.mark.asyncio
    async def test_sigterm_then_wait(self):
        """shutdown sends SIGTERM first and waits for clean exit."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.send_signal = MagicMock()
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc
        claude._session_started_at = 12345.0

        await claude.shutdown()

        mock_proc.send_signal.assert_called_with(signal.SIGTERM)
        assert claude._proc is None
        assert claude._pgid is None
        assert claude._session_started_at is None

    @pytest.mark.asyncio
    async def test_falls_back_to_sigkill_on_timeout(self):
        """When SIGTERM times out, falls back to SIGKILL."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.send_signal = MagicMock()
        # First wait (SIGTERM) times out, second wait (SIGKILL) succeeds
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc

        # Simulate SIGTERM timeout on the first wait_for call, success on second
        call_count = 0
        original_wait_for = asyncio.wait_for

        async def mock_wait_for(coro, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Cancel the coroutine to avoid "was never awaited" warning
                coro.close()
                raise TimeoutError
            return await original_wait_for(coro, timeout=timeout)

        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            await claude.shutdown()

        # Should have sent SIGTERM then SIGKILL
        signals_sent = [call.args[0] for call in mock_proc.send_signal.call_args_list]
        assert signal.SIGTERM in signals_sent
        assert signal.SIGKILL in signals_sent
        assert claude._proc is None

    @pytest.mark.asyncio
    async def test_sigkill_fallback_when_sudo_exits_before_claude(self):
        """SIGKILL fallback fires even when sudo has already exited.

        This is the core bug fix: the old _kill_proc() checked returncode
        before sending signals. When sudo exited from SIGTERM (setting
        returncode), the SIGKILL fallback was skipped - orphaning claude.
        Now _send_signal() uses the saved PGID and ignores returncode.
        """
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc
        claude._pgid = 12345

        call_count = 0
        original_wait_for = asyncio.wait_for

        async def sudo_exits_then_timeout(coro, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # sudo exits from SIGTERM (returncode becomes set)
                mock_proc.returncode = 0
                coro.close()
                raise TimeoutError
            return await original_wait_for(coro, timeout=timeout)

        with (
            patch("asyncio.wait_for", side_effect=sudo_exits_then_timeout),
            patch("os.killpg") as mock_killpg,
        ):
            await claude.shutdown()

        # SIGKILL was sent to the process group despite returncode being set
        killpg_calls = [call.args[1] for call in mock_killpg.call_args_list]
        assert signal.SIGTERM in killpg_calls
        assert signal.SIGKILL in killpg_calls
        assert claude._proc is None
        assert claude._pgid is None

    @pytest.mark.asyncio
    async def test_still_sends_sigterm_when_already_exited(self):
        """shutdown sends SIGTERM even when returncode is already set.

        When the process is already dead, _send_signal() catches the OSError
        and wait() returns immediately. State is still cleaned up.
        """
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = 0  # Already exited
        mock_proc.send_signal = MagicMock(side_effect=ProcessLookupError)
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc

        await claude.shutdown()

        # Signal attempted (caught by _send_signal), state cleaned up
        mock_proc.send_signal.assert_called_with(signal.SIGTERM)
        assert claude._proc is None

    @pytest.mark.asyncio
    async def test_clears_stderr_task(self):
        """shutdown cancels the stderr drain task."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.send_signal = MagicMock()
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc
        mock_task = MagicMock()
        claude._stderr_task = mock_task

        await claude.shutdown()

        mock_task.cancel.assert_called_once()
        assert claude._stderr_task is None

    @pytest.mark.asyncio
    async def test_zombie_process_logs_warning(self, caplog):
        """When both SIGTERM and SIGKILL timeout, logs a warning and clears state."""
        caplog.set_level("WARNING", logger="kai.claude")
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.send_signal = MagicMock()
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc

        # Both wait_for calls time out
        async def always_timeout(coro, timeout):
            coro.close()
            raise TimeoutError

        with patch("asyncio.wait_for", side_effect=always_timeout):
            await claude.shutdown()

        assert "did not exit" in caplog.text.lower()
        assert claude._proc is None

    @pytest.mark.asyncio
    async def test_shutdown_signals_saved_pgid_after_clearing(self):
        """shutdown sends a final SIGKILL to the saved pgid after state cleanup.

        Same belt-and-suspenders pattern as _kill(): saves the pgid before
        clearing state, then signals the process group one final time to
        catch any orphaned claude process that survived the initial signals.
        """
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc
        claude._pgid = 12345

        with patch("os.killpg") as mock_killpg:
            await claude.shutdown()

        # Final killpg call should be SIGKILL to the saved pgid
        killpg_calls = [call.args for call in mock_killpg.call_args_list]
        # At least: SIGTERM from _send_signal, and final SIGKILL cleanup
        assert (12345, signal.SIGKILL) in killpg_calls
        assert claude._proc is None
        assert claude._pgid is None


# ── _save_prompt ─────────────────────────────────────────────────────


class TestSavePrompt:
    @pytest.mark.asyncio
    async def test_sends_save_message_to_stdin(self):
        """_save_prompt writes a stream-json message to stdin."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()
        # Simulate a result event response
        result_event = json.dumps({"type": "result"}).encode() + b"\n"
        mock_proc.stdout = AsyncMock()
        mock_proc.stdout.readline = AsyncMock(return_value=result_event)
        claude._proc = mock_proc
        claude._fresh_session = False

        await claude._save_prompt()

        # Verify stdin was written to with a stream-json user message
        written = mock_proc.stdin.write.call_args[0][0]
        msg = json.loads(written.decode())
        assert msg["type"] == "user"
        assert msg["message"]["role"] == "user"
        assert "shut down" in msg["message"]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_skips_fresh_session(self):
        """_save_prompt returns immediately for fresh sessions."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        claude._proc = mock_proc
        claude._fresh_session = True

        await claude._save_prompt()

        # No stdin write should have occurred
        mock_proc.stdin.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_dead_process(self):
        """_save_prompt returns immediately when process is not alive."""
        claude = _make_claude()
        claude._proc = None
        claude._fresh_session = False

        # Should not raise
        await claude._save_prompt()

    @pytest.mark.asyncio
    async def test_handles_stdin_write_failure(self):
        """_save_prompt handles a broken pipe gracefully."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock(side_effect=OSError("broken pipe"))
        mock_proc.stdout = AsyncMock()
        claude._proc = mock_proc
        claude._fresh_session = False

        # Should not raise
        await claude._save_prompt()

    @pytest.mark.asyncio
    async def test_handles_drain_runtime_error(self):
        """_save_prompt handles RuntimeError from drain() gracefully."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock(side_effect=RuntimeError("transport closed"))
        mock_proc.stdout = AsyncMock()
        claude._proc = mock_proc
        claude._fresh_session = False

        # Should not raise
        await claude._save_prompt()

    @pytest.mark.asyncio
    async def test_stops_on_result_event(self):
        """_save_prompt stops reading when it sees a result event."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()

        # Send a system event, then a result event
        responses = [
            json.dumps({"type": "system"}).encode() + b"\n",
            json.dumps({"type": "result"}).encode() + b"\n",
        ]
        mock_proc.stdout = AsyncMock()
        mock_proc.stdout.readline = AsyncMock(side_effect=responses)
        claude._proc = mock_proc
        claude._fresh_session = False

        await claude._save_prompt()

        # readline was called twice (system event + result event)
        assert mock_proc.stdout.readline.call_count == 2

    @pytest.mark.asyncio
    async def test_stops_on_eof(self):
        """_save_prompt stops reading on EOF (process died)."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.stdout = AsyncMock()
        mock_proc.stdout.readline = AsyncMock(return_value=b"")
        claude._proc = mock_proc
        claude._fresh_session = False

        await claude._save_prompt()

    @pytest.mark.asyncio
    async def test_stops_on_deadline_expired(self):
        """_save_prompt exits when the deadline has elapsed."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.stdout = AsyncMock()
        # readline should never be called if the deadline is already past
        mock_proc.stdout.readline = AsyncMock(side_effect=AssertionError("should not be called"))
        claude._proc = mock_proc
        claude._fresh_session = False

        # Use timeout=0 so deadline expires immediately after write
        await claude._save_prompt(timeout=0)

    @pytest.mark.asyncio
    async def test_stops_on_readline_timeout(self):
        """_save_prompt handles TimeoutError from readline."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.stdout = AsyncMock()
        mock_proc.stdout.readline = AsyncMock(side_effect=TimeoutError())
        claude._proc = mock_proc
        claude._fresh_session = False

        await claude._save_prompt(timeout=5)

    @pytest.mark.asyncio
    async def test_handles_json_parse_error(self):
        """_save_prompt handles non-JSON response lines."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()

        # Non-JSON line followed by result event
        responses = [
            b"not json at all\n",
            json.dumps({"type": "result"}).encode() + b"\n",
        ]
        mock_proc.stdout = AsyncMock()
        mock_proc.stdout.readline = AsyncMock(side_effect=responses)
        claude._proc = mock_proc
        claude._fresh_session = False

        await claude._save_prompt()

        # Both lines were read (non-JSON skipped, result stopped the loop)
        assert mock_proc.stdout.readline.call_count == 2

    @pytest.mark.asyncio
    async def test_handles_unexpected_exception(self):
        """_save_prompt handles unexpected exceptions during read."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.stdout = AsyncMock()
        mock_proc.stdout.readline = AsyncMock(side_effect=RuntimeError("unexpected"))
        claude._proc = mock_proc
        claude._fresh_session = False

        # Should not raise
        await claude._save_prompt()

    @pytest.mark.asyncio
    async def test_shutdown_calls_save_prompt(self):
        """shutdown() calls _save_prompt() before SIGTERM."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.send_signal = MagicMock()
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc

        call_order: list[str] = []

        async def tracking_save():
            call_order.append("save_prompt")

        with (
            patch.object(claude, "_save_prompt", side_effect=tracking_save),
            patch.object(claude, "_send_signal", side_effect=lambda s: call_order.append(f"signal_{s}")),
        ):
            await claude.shutdown()

        assert call_order[0] == "save_prompt"
        assert "signal_15" in call_order  # SIGTERM = 15

    @pytest.mark.asyncio
    async def test_force_kill_does_not_call_save_prompt(self):
        """force_kill() does NOT call _save_prompt()."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.pid = 12345
        claude._proc = mock_proc

        with patch.object(claude, "_save_prompt", new_callable=AsyncMock) as mock_save:
            claude.force_kill()

        mock_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_recycle_calls_save_prompt_before_kill(self):
        """Session recycle calls _save_prompt(timeout=10) before _kill()."""
        claude = _make_claude()
        claude.max_session_hours = 0.001  # Tiny value so _should_recycle is True
        claude._session_started_at = 0  # Long ago

        # Mock a running process
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.stdout = AsyncMock()
        mock_proc.stdout.readline = AsyncMock(return_value=json.dumps({"type": "result"}).encode() + b"\n")
        claude._proc = mock_proc
        claude._fresh_session = False

        call_order: list[str] = []

        async def tracking_save(timeout=30):
            call_order.append(f"save_prompt_timeout={timeout}")
            # Don't actually send a save prompt
            return

        async def tracking_kill():
            call_order.append("kill")

        with (
            patch.object(claude, "_save_prompt", side_effect=tracking_save),
            patch.object(claude, "_kill", side_effect=tracking_kill),
        ):
            # Only need to verify the recycle path fires and calls
            # save_prompt then kill in order. We don't need to run
            # the full _send_locked - just trigger the recycle check.
            # Since _should_recycle returns True, _send_locked will
            # call save_prompt then kill before _ensure_started.
            # Catch the StopAsyncIteration when the generator can't
            # proceed without a real process.
            try:
                async for _ in claude._send_locked("test", chat_id=123):
                    pass
            except (StopAsyncIteration, AttributeError, TypeError):
                pass  # Expected - no real process after kill

        assert call_order[0] == "save_prompt_timeout=10"
        assert call_order[1] == "kill"


# ── change_workspace ─────────────────────────────────────────────────


class TestChangeWorkspace:
    @pytest.mark.asyncio
    async def test_updates_workspace_and_kills(self):
        """change_workspace updates the path and kills the current process."""
        claude = _make_claude()
        new_path = Path("/tmp/other-workspace")

        with patch.object(claude, "_kill", new_callable=AsyncMock) as mock_kill:
            await claude.change_workspace(new_path)

        assert claude.workspace == new_path
        mock_kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_kill_before_state_mutation(self):
        """_kill() runs before attributes are mutated, not after."""
        claude = _make_claude()
        original_workspace = claude.workspace
        kill_order: list[tuple[str, Path]] = []

        async def tracking_kill():
            # Record what workspace was set when _kill was called
            kill_order.append(("kill", claude.workspace))

        with patch.object(claude, "_kill", side_effect=tracking_kill):
            await claude.change_workspace(Path("/tmp/new-workspace"))

        # _kill should have seen the ORIGINAL workspace, not the new one
        assert kill_order == [("kill", original_workspace)]
        # Final state should still be the new workspace
        assert claude.workspace == Path("/tmp/new-workspace")


# ── restart ──────────────────────────────────────────────────────────


class TestRestart:
    @pytest.mark.asyncio
    async def test_calls_kill(self):
        """restart() kills the current process so the next send() starts fresh."""
        claude = _make_claude()

        with patch.object(claude, "_kill", new_callable=AsyncMock) as mock_kill:
            await claude.restart()

        mock_kill.assert_called_once()


# ── Workspace config ────────────────────────────────────────────────


class TestWorkspaceConfig:
    def test_constructor_with_workspace_config(self):
        """WorkspaceConfig overrides model, budget, and timeout."""
        ws_config = WorkspaceConfig(
            path=Path("/tmp/ws"),
            model="opus",
            budget=15.0,
            timeout=300,
        )
        claude = _make_claude(workspace_config=ws_config)
        assert claude.model == "opus"
        assert claude.max_budget_usd == 15.0
        assert claude.timeout_seconds == 300

    def test_constructor_without_workspace_config(self):
        """Without config, global defaults are used."""
        claude = _make_claude()
        assert claude.model == "sonnet"
        assert claude.max_budget_usd == 1.0
        assert claude.timeout_seconds == 30

    def test_constructor_partial_workspace_config(self):
        """Config with only model set leaves budget and timeout at defaults."""
        ws_config = WorkspaceConfig(path=Path("/tmp/ws"), model="haiku")
        claude = _make_claude(workspace_config=ws_config)
        assert claude.model == "haiku"
        assert claude.max_budget_usd == 1.0  # unchanged
        assert claude.timeout_seconds == 30  # unchanged

    def test_defaults_preserved(self):
        """Constructor stores the original global defaults."""
        ws_config = WorkspaceConfig(path=Path("/tmp/ws"), model="opus", budget=20.0)
        claude = _make_claude(workspace_config=ws_config)
        assert claude._default_model == "sonnet"
        assert claude._default_budget == 1.0
        assert claude._default_timeout == 30

    @pytest.mark.asyncio
    async def test_change_workspace_with_config(self):
        """Switching to a configured workspace applies overrides."""
        claude = _make_claude()
        ws_config = WorkspaceConfig(path=Path("/tmp/ws2"), model="opus", budget=20.0)

        with patch.object(claude, "_kill", new_callable=AsyncMock):
            await claude.change_workspace(Path("/tmp/ws2"), workspace_config=ws_config)

        assert claude.model == "opus"
        assert claude.max_budget_usd == 20.0

    @pytest.mark.asyncio
    async def test_change_workspace_to_unconfigured(self):
        """Switching from configured to unconfigured reverts to global defaults."""
        ws_config = WorkspaceConfig(path=Path("/tmp/ws"), model="opus", budget=20.0)
        claude = _make_claude(workspace_config=ws_config)
        assert claude.model == "opus"

        with patch.object(claude, "_kill", new_callable=AsyncMock):
            await claude.change_workspace(Path("/tmp/other"))

        assert claude.model == "sonnet"  # reverted to default
        assert claude.max_budget_usd == 1.0  # reverted to default

    @pytest.mark.asyncio
    async def test_change_workspace_no_stale_values(self):
        """Partial config doesn't carry over values from previous workspace.

        Scenario: workspace A has budget=20.0. Switch to workspace B
        which only sets model. Budget must revert to the global default,
        not carry over workspace A's 20.0.
        """
        ws_a = WorkspaceConfig(path=Path("/tmp/a"), model="opus", budget=20.0)
        ws_b = WorkspaceConfig(path=Path("/tmp/b"), model="haiku")
        claude = _make_claude(workspace_config=ws_a)

        with patch.object(claude, "_kill", new_callable=AsyncMock):
            await claude.change_workspace(Path("/tmp/b"), workspace_config=ws_b)

        assert claude.model == "haiku"
        assert claude.max_budget_usd == 1.0  # global default, not 20.0

    @pytest.mark.asyncio
    async def test_change_workspace_model_override_cycle(self):
        """Config model restored after /model override and workspace switch.

        Scenario: configured workspace (opus), user does /model haiku,
        switches away, switches back. The workspace config model (opus)
        should be restored, not the /model override (haiku).
        """
        ws_config = WorkspaceConfig(path=Path("/tmp/ws"), model="opus")
        claude = _make_claude(workspace_config=ws_config)
        assert claude.model == "opus"

        # User overrides model via /model command
        claude.model = "haiku"
        assert claude.model == "haiku"

        # Switch away (unconfigured workspace)
        with patch.object(claude, "_kill", new_callable=AsyncMock):
            await claude.change_workspace(Path("/tmp/other"))
        assert claude.model == "sonnet"  # global default

        # Switch back to configured workspace
        with patch.object(claude, "_kill", new_callable=AsyncMock):
            await claude.change_workspace(Path("/tmp/ws"), workspace_config=ws_config)
        assert claude.model == "opus"  # config model, not haiku

    def test_system_prompt_inline(self):
        """_get_workspace_system_prompt returns inline prompt."""
        ws_config = WorkspaceConfig(path=Path("/tmp/ws"), system_prompt="Be concise.")
        claude = _make_claude(workspace_config=ws_config)
        assert claude._get_workspace_system_prompt() == "Be concise."

    def test_system_prompt_from_file(self, tmp_path):
        """_get_workspace_system_prompt reads from file."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Use pytest.")
        ws_config = WorkspaceConfig(path=Path("/tmp/ws"), system_prompt_file=prompt_file)
        claude = _make_claude(workspace_config=ws_config)
        assert claude._get_workspace_system_prompt() == "Use pytest."

    def test_system_prompt_file_deleted(self, tmp_path):
        """Returns None if system_prompt_file is deleted after load."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("hello")
        ws_config = WorkspaceConfig(path=Path("/tmp/ws"), system_prompt_file=prompt_file)
        claude = _make_claude(workspace_config=ws_config)
        prompt_file.unlink()
        assert claude._get_workspace_system_prompt() is None

    def test_system_prompt_none_without_config(self):
        """Returns None when no workspace config is set."""
        claude = _make_claude()
        assert claude._get_workspace_system_prompt() is None

    @pytest.mark.asyncio
    async def test_env_merge_in_ensure_started(self):
        """Per-workspace env vars are merged into the subprocess environment."""
        ws_config = WorkspaceConfig(
            path=Path("/tmp/ws"),
            env={"MY_VAR": "my_value"},
        )
        claude = _make_claude(workspace_config=ws_config)
        claude._fresh_session = False

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = MagicMock()
            mock_proc.stderr.readline = AsyncMock(return_value=b"")
            mock_exec.return_value = mock_proc

            await claude._ensure_started()

            # Check the env kwarg passed to create_subprocess_exec
            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs["env"]["MY_VAR"] == "my_value"

    @pytest.mark.asyncio
    async def test_env_merge_webhook_secret_preserved(self):
        """Workspace env can't override the webhook secret."""
        ws_config = WorkspaceConfig(
            path=Path("/tmp/ws"),
            env={"KAI_WEBHOOK_SECRET": "evil"},
        )
        claude = _make_claude(workspace_config=ws_config)
        claude.webhook_secret = "real_secret"
        claude._fresh_session = False

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = MagicMock()
            mock_proc.stderr.readline = AsyncMock(return_value=b"")
            mock_exec.return_value = mock_proc

            await claude._ensure_started()

            call_kwargs = mock_exec.call_args.kwargs
            # Webhook secret is set LAST and overrides workspace env
            assert call_kwargs["env"]["KAI_WEBHOOK_SECRET"] == "real_secret"

    @pytest.mark.asyncio
    async def test_env_file_loading(self, tmp_path):
        """Per-workspace env_file values are merged into subprocess env."""
        env_file = tmp_path / ".env.kai"
        env_file.write_text("FROM_FILE=hello\n")
        ws_config = WorkspaceConfig(path=Path("/tmp/ws"), env_file=env_file)
        claude = _make_claude(workspace_config=ws_config)
        claude._fresh_session = False

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = MagicMock()
            mock_proc.stderr.readline = AsyncMock(return_value=b"")
            mock_exec.return_value = mock_proc

            await claude._ensure_started()

            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs["env"]["FROM_FILE"] == "hello"

    @pytest.mark.asyncio
    async def test_env_file_overridden_by_inline(self, tmp_path):
        """Inline env overrides env_file values for the same key."""
        env_file = tmp_path / ".env.kai"
        env_file.write_text("SHARED=from_file\n")
        ws_config = WorkspaceConfig(
            path=Path("/tmp/ws"),
            env_file=env_file,
            env={"SHARED": "from_inline"},
        )
        claude = _make_claude(workspace_config=ws_config)
        claude._fresh_session = False

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = MagicMock()
            mock_proc.stderr.readline = AsyncMock(return_value=b"")
            mock_exec.return_value = mock_proc

            await claude._ensure_started()

            call_kwargs = mock_exec.call_args.kwargs
            assert call_kwargs["env"]["SHARED"] == "from_inline"
