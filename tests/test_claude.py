"""Tests for claude.py user separation (sudo spawning and process group signals)."""

import signal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.claude import PersistentClaude


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
            assert cmd[3] == "--"
            assert cmd[4] == "claude"

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


# ── Process signal handling ──────────────────────────────────────────


class TestProcessSignals:
    """Verify _kill_proc() and force_kill() use the right signal strategy."""

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
        """With claude_user, force_kill sends SIGKILL to the entire process group."""
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.pid = 12345
        claude._proc = mock_proc

        with patch("os.getpgid", return_value=12345) as mock_getpgid, patch("os.killpg") as mock_killpg:
            claude.force_kill()

            mock_getpgid.assert_called_once_with(12345)
            mock_killpg.assert_called_once_with(12345, signal.SIGKILL)

    def test_kill_proc_noop_when_no_process(self):
        """_kill_proc() is a no-op when there's no subprocess."""
        claude = _make_claude()
        # _proc is None by default; should not raise
        claude._kill_proc(signal.SIGKILL)

    def test_kill_proc_noop_when_already_exited(self):
        """_kill_proc() is a no-op when the process has already exited."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = 0  # Already exited
        claude._proc = mock_proc

        claude._kill_proc(signal.SIGKILL)

        mock_proc.send_signal.assert_not_called()

    def test_kill_proc_handles_process_lookup_error(self):
        """_kill_proc() swallows ProcessLookupError (race between check and kill)."""
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.send_signal.side_effect = ProcessLookupError
        claude._proc = mock_proc

        # Should not raise
        claude._kill_proc(signal.SIGKILL)

    def test_kill_proc_handles_permission_error(self):
        """_kill_proc() swallows PermissionError (process owned by another user)."""
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.pid = 12345
        claude._proc = mock_proc

        with patch("os.getpgid", return_value=12345), patch("os.killpg", side_effect=PermissionError):
            # Should not raise
            claude._kill_proc(signal.SIGKILL)
