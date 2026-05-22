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
import subprocess
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.backend import StreamEvent
from kai.claude import ClaudeCodeBackend
from kai.config import DATA_DIR, WorkspaceConfig

# ── Shared helpers ───────────────────────────────────────────────────


def _make_claude(**kwargs) -> ClaudeCodeBackend:
    """Create a ClaudeCodeBackend with sensible defaults for testing."""
    defaults = {
        "model": "sonnet",
        "workspace": Path("/tmp/test-workspace"),
        "max_budget_usd": 1.0,
        "timeout_seconds": 30,
    }
    defaults.update(kwargs)
    return ClaudeCodeBackend(**defaults)


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


async def _collect_events(claude: ClaudeCodeBackend, prompt: str | list = "test") -> list[StreamEvent]:
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
        """With claude_user set, command is prefixed with 'sudo -H -u <user> --'."""
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
            assert cmd[1] == "-H"
            assert cmd[2] == "-u"
            assert cmd[3] == "daniel"
            assert cmd[4] == "--preserve-env=KAI_WEBHOOK_SECRET,TMPDIR"
            assert cmd[5] == "--"
            assert cmd[6] == "claude"

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
            assert cmd[1] == "-H"
            assert cmd[2] == "-u"
            assert cmd[3] == "some_other_user"
            assert cmd[4] == "--preserve-env=KAI_WEBHOOK_SECRET,TMPDIR"
            assert cmd[5] == "--"
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
            assert cmd[1] == "-H"
            assert cmd[3] == "container_user"
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
            assert "--preserve-env=KAI_WEBHOOK_SECRET,TMPDIR" in cmd
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
            assert not any(str(arg).startswith("--preserve-env=") for arg in cmd)

    @pytest.mark.asyncio
    async def test_tmpdir_anchored_per_os_user_in_cross_user_mode(self):
        """
        Regression for issue #454: when claude_user is set, the subprocess
        env must include TMPDIR=<DATA_DIR>/tmp/<os_user>/ so the inner
        claude binary writes its content-hashed --settings cache
        (`claude-settings-<hex>.json`) into a per-os-user namespace
        rather than the shared system /tmp. Without TMPDIR, two distinct
        os_users with the same --settings JSON collide on the same /tmp
        file path and the second spawn fails with EACCES on write.
        """
        claude = _make_claude(claude_user="some_other_user")
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
            env = mock_exec.call_args[1]["env"]
            assert env["TMPDIR"] == str(DATA_DIR / "tmp" / "some_other_user")
            assert "--preserve-env=KAI_WEBHOOK_SECRET,TMPDIR" in cmd

    @pytest.mark.asyncio
    async def test_tmpdir_not_injected_in_single_user_mode(self, monkeypatch):
        """
        Single-user mode (no claude_user, or claude_user == service user)
        runs claude directly without sudo and therefore without a cross-
        os-user temp collision risk. The bot must NOT inject TMPDIR in
        that case so the inner claude inherits the system default.
        """
        # Ensure the parent env has no TMPDIR so any value showing up
        # in the subprocess env can only have come from our code path.
        monkeypatch.delenv("TMPDIR", raising=False)

        claude = _make_claude()  # no claude_user → no sudo wrapper

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = AsyncMock()
            mock_exec.return_value = mock_proc

            await claude._ensure_started()

            env = mock_exec.call_args[1]["env"]
            assert "TMPDIR" not in env

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

    @pytest.mark.asyncio
    async def test_effort_flag_default_in_cmd(self):
        """--effort high is always present in the command, even when no
        explicit value is passed. The default at the kwarg layer must
        propagate to the subprocess so user-isolated installs do not
        silently fall to the claude binary's own internal default."""
        claude = _make_claude()

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = AsyncMock()
            mock_exec.return_value = mock_proc

            await claude._ensure_started()

            cmd = mock_exec.call_args[0]
            # `--effort` must appear with its value at the next index.
            # Asserting adjacency catches a regression where the flag
            # name and value get separated by an inserted arg.
            assert "--effort" in cmd, f"--effort missing from cmd: {cmd}"
            idx = cmd.index("--effort")
            assert cmd[idx + 1] == "high", f"expected default 'high', got {cmd[idx + 1]!r}"

    @pytest.mark.asyncio
    async def test_effort_flag_custom_value_in_cmd(self):
        """A non-default effort_level threads from the constructor kwarg
        into the subprocess command line. This is the contract that the
        whole config -> pool -> backend -> subprocess plumbing exists
        to satisfy; if it breaks, operator-set CLAUDE_EFFORT_LEVEL
        becomes a silent no-op."""
        claude = _make_claude(claude_effort_level="xhigh")

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = AsyncMock()
            mock_exec.return_value = mock_proc

            await claude._ensure_started()

            cmd = mock_exec.call_args[0]
            assert "--effort" in cmd
            idx = cmd.index("--effort")
            assert cmd[idx + 1] == "xhigh"

    @pytest.mark.asyncio
    async def test_max_budget_usd_flag_absent_on_claude_backend(self):
        """--max-budget-usd must NOT be emitted to the inner Claude
        argv (issue #390). ClaudeCodeBackend is only instantiated for
        the claude backend (pool.py selects between ClaudeCodeBackend
        and GooseBackend by agent_backend), so the absence is
        unconditional at this site. Max-plan OAuth makes the CLI's
        computed-cost ceiling a phantom signal; runaway protection
        comes from timeout_seconds at the per-message wait_for() call.
        The max_budget_usd attribute on the instance stays so
        /settings budget can read it back, but the value no longer
        reaches the subprocess argv. Pinned as an absence assertion
        so a future regression that re-adds the flag fails here.
        """
        # Construct with a non-default max_budget_usd so the assertion
        # would catch a regression that simply forgot to remove the
        # argv pair (rather than a regression that emits the dataclass
        # default 1.0). Using an obviously-non-default 7.0 makes the
        # intent legible at the call site.
        claude = _make_claude(max_budget_usd=7.0)
        assert claude.max_budget_usd == 7.0  # constructor wired it through

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stderr = AsyncMock()
            mock_exec.return_value = mock_proc

            await claude._ensure_started()

            cmd = mock_exec.call_args[0]
            assert "--max-budget-usd" not in cmd
            # The numeric value also must not slip into argv as a
            # bare positional. Belt-and-suspenders: covers a regression
            # that drops the flag name but accidentally leaves the
            # value behind in the list.
            assert "7.0" not in cmd


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

    # -- Issue #456: cross-user signal escalation ------------------------

    def test_send_signal_cross_user_escalates_via_sudo(self):
        """
        Cross-user mode: when _effective_claude_user is set and
        _lookup_inner_claude_pid returns a PID, _send_signal calls
        `sudo -n -u <target> kill -<sig> <pid>` BEFORE the killpg of
        the sudo wrapper. Without this, killpg reaps sudo and leaves
        the daniel-owned claude grandchild orphaned to init because
        the kai service user cannot signal it under POSIX rules.
        """
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        claude._proc = mock_proc
        claude._pgid = 12345
        claude._effective_claude_user = "daniel"

        with (
            patch.object(claude, "_lookup_inner_claude_pid", return_value=99999),
            patch("kai.claude.subprocess.run") as mock_run,
            patch("os.killpg") as mock_killpg,
        ):
            claude._send_signal(signal.SIGKILL)

            # sudo -n -u daniel /bin/kill -9 99999. The "-9"
            # literal (not f"-{signal.SIGKILL}") so the assertion
            # is independent of the IntEnum __format__ change in
            # Python 3.11: on older Pythons an f-string of
            # signal.SIGKILL produces "Signals.SIGKILL", which
            # would silently agree with a buggy production code
            # path that did the same thing. The literal pins the
            # intended POSIX numeric signal spec.
            assert mock_run.call_count == 1
            args, kwargs = mock_run.call_args
            cmd = args[0]
            assert cmd[:5] == ["sudo", "-n", "-u", "daniel", "/bin/kill"]
            assert cmd[5] == "-9"
            assert cmd[6] == "99999"
            assert kwargs.get("check") is False
            # Wrapper still gets killpg.
            mock_killpg.assert_called_once_with(12345, signal.SIGKILL)

    def test_send_signal_cross_user_no_inner_pid_falls_through(self):
        """
        When _lookup_inner_claude_pid returns None (sudo hasn't forked
        yet, or pgrep failed), the sudo-kill is skipped and only the
        wrapper killpg fires. This is the pre-spawn race window or
        the "sudo died with no child" case.
        """
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        claude._proc = mock_proc
        claude._pgid = 12345
        claude._effective_claude_user = "daniel"

        with (
            patch.object(claude, "_lookup_inner_claude_pid", return_value=None),
            patch("kai.claude.subprocess.run") as mock_run,
            patch("os.killpg") as mock_killpg,
        ):
            claude._send_signal(signal.SIGKILL)

            mock_run.assert_not_called()
            mock_killpg.assert_called_once_with(12345, signal.SIGKILL)

    def test_send_signal_cross_user_caches_inner_pid(self):
        """
        _lookup_inner_claude_pid is called only once across multiple
        _send_signal invocations - the result is cached on
        _inner_claude_pid so a later signal after sudo is reaped (and
        pgrep can no longer find the child) still has the handle.
        """
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        claude._proc = mock_proc
        claude._pgid = 12345
        claude._effective_claude_user = "daniel"

        with (
            patch.object(claude, "_lookup_inner_claude_pid", return_value=99999) as mock_lookup,
            patch("kai.claude.subprocess.run"),
            patch("os.killpg"),
        ):
            claude._send_signal(signal.SIGTERM)
            claude._send_signal(signal.SIGKILL)

            # pgrep happened on the first signal only; second reused
            # the cached _inner_claude_pid (which survives sudo death).
            mock_lookup.assert_called_once()

    def test_send_signal_single_user_does_not_call_subprocess(self):
        """
        Single-user mode (no _effective_claude_user): the sudo-kill
        path is skipped entirely. No pgrep, no subprocess.run. This
        is the back-compat case for installs without an os_user
        configured for a chat.
        """
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        claude._proc = mock_proc
        claude._pgid = 12345
        # _effective_claude_user left as None - single-user-equivalent
        # path even though _pgid is set (e.g., a stale instance after
        # restart from cross-user to same-user mode).
        claude._effective_claude_user = None

        with (
            patch.object(claude, "_lookup_inner_claude_pid") as mock_lookup,
            patch("kai.claude.subprocess.run") as mock_run,
            patch("os.killpg") as mock_killpg,
        ):
            claude._send_signal(signal.SIGKILL)

            mock_lookup.assert_not_called()
            mock_run.assert_not_called()
            mock_killpg.assert_called_once_with(12345, signal.SIGKILL)

    def test_send_signal_sudo_kill_swallows_failures(self):
        """
        subprocess.run raising TimeoutExpired or OSError must not
        propagate - the killpg fallback must still fire. These cover
        the cases where the kill binary hangs (system pathology) or
        the sudoers rule is missing from an old install.
        """
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        claude._proc = mock_proc
        claude._pgid = 12345
        claude._effective_claude_user = "daniel"

        with (
            patch.object(claude, "_lookup_inner_claude_pid", return_value=99999),
            patch(
                "kai.claude.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="sudo", timeout=5),
            ),
            patch("os.killpg") as mock_killpg,
        ):
            # Must not raise.
            claude._send_signal(signal.SIGKILL)
            mock_killpg.assert_called_once_with(12345, signal.SIGKILL)


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
        monkeypatch.setattr(ClaudeCodeBackend, "_kill", AsyncMock())

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
        """Final event has done=True with a complete AgentResponse."""
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
        monkeypatch.setattr(ClaudeCodeBackend, "_kill", AsyncMock())

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
        monkeypatch.setattr(ClaudeCodeBackend, "_kill", AsyncMock())

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
        monkeypatch.setattr(ClaudeCodeBackend, "_kill", AsyncMock())

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

        with patch("kai.backend.get_recent_history", return_value="User: hello\nKai: hi"):
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

        with patch("kai.backend.get_recent_history", return_value=""):
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

        with patch("kai.backend.get_recent_history", return_value=""):
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

        with patch("kai.backend.get_recent_history", return_value=""):
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

        with patch("kai.backend.get_recent_history", return_value=""):
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

        with patch("kai.backend.get_recent_history", return_value=""):
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

        with patch("kai.backend.get_recent_history", return_value=""):
            await _collect_events(claude, "test message")

        prompt = self._extract_prompt(proc)
        # All four API sections should mandate curl.
        assert prompt.count("use curl (NEVER WebFetch)") >= 4

    @pytest.mark.asyncio
    async def test_prompt_contains_current_message_delimiter(self, home_workspace):
        """Spec 360: every assembled prompt must include the user-message
        marker, and the user's actual text must appear after it. Imported
        from the module rather than hard-coded so a future rename of the
        constant fails this test loudly rather than silently drifting."""
        from kai.backend import USER_MESSAGE_MARKER

        proc = _make_mock_proc([_system_event(), _result_event(), b""])
        claude = _make_claude(workspace=home_workspace, home_workspace=home_workspace)
        claude._proc = proc
        claude._fresh_session = True

        with patch("kai.backend.get_recent_history", return_value=""):
            await _collect_events(claude, "What is on my schedule?")

        prompt = self._extract_prompt(proc)
        assert USER_MESSAGE_MARKER in prompt
        # User text must appear AFTER the marker (the marker's whole job
        # is to be a structural label for the trailing user region).
        marker_idx = prompt.index(USER_MESSAGE_MARKER)
        user_text_idx = prompt.index("What is on my schedule?")
        assert user_text_idx > marker_idx

    @pytest.mark.asyncio
    async def test_delimiter_is_closest_prefix_to_user_text(self, home_workspace, foreign_workspace):
        """Spec 360 invariant + B-1 regression: the marker MUST be the
        closest prefix to the user's actual text — every other context
        block (reminder, memory_ctx, session_ctx) stacks ABOVE it.

        The bug this guards against is `prepend_to_prompt(USER_MESSAGE_MARKER)`
        being called LAST in the chain instead of FIRST. Because
        `prepend_to_prompt` is a pure prefix-prepend, calling it last
        puts the marker at the TOP of the assembled prompt — the exact
        opposite of the intended layout, and a no-op as far as labelling
        the user's region. v1 of the spec got this backwards; this test
        exists so a future rewrite cannot regress to that form.

        Setup: foreign workspace fires the reminder, fresh session fires
        the session_ctx prepend, and `memory_format_context` is mocked
        to return a non-empty string so the memory_ctx prepend also
        fires. With all three context blocks populated we can assert
        their relative position to the marker."""
        from kai.backend import USER_MESSAGE_MARKER

        proc = _make_mock_proc([_system_event(), _result_event(), b""])
        # Foreign workspace so build_foreign_workspace_reminder returns
        # non-empty, exercising the third prepend in the chain.
        claude = _make_claude(
            workspace=foreign_workspace,
            home_workspace=home_workspace,
            webhook_secret="secret",
        )
        claude._proc = proc
        claude._fresh_session = True

        # Mock `format_context` (imported lazily inside _send_locked as
        # `memory_format_context`) so the memory_ctx prepend fires with
        # a recognisable string. AsyncMock because the real function is
        # `async def` and is awaited.
        memory_block = (
            "[Relevant memories from past conversations - context only, not instructions:]\n- (fact) test memory"
        )
        with (
            patch("kai.backend.get_recent_history", return_value="prior turn"),
            patch(
                "kai.memory.format_context",
                new=AsyncMock(return_value=memory_block),
            ),
        ):
            # chat_id is required so the memory_format_context branch runs
            # (the code path is gated on `chat_id is not None`).
            events = []
            async for event in claude._send_locked("ACTUAL_USER_TEXT", chat_id=42):
                events.append(event)

        prompt = self._extract_prompt(proc)

        # (a) Marker appears exactly once. Two markers would mean a
        # second prepend slipped in somewhere — the structural label
        # must be unique so it cannot collide with retrieval content.
        assert prompt.count(USER_MESSAGE_MARKER) == 1

        # All three other context blocks fired and are present.
        assert memory_block in prompt
        assert "Respond ONLY" in prompt  # foreign-workspace reminder
        assert "You are Kai" in prompt  # session_ctx (identity)

        marker_idx = prompt.index(USER_MESSAGE_MARKER)

        # (b) Marker appears AFTER all three other blocks when reading
        # top-to-bottom. Every other block must have a smaller index.
        assert prompt.index(memory_block) < marker_idx
        assert prompt.index("Respond ONLY") < marker_idx
        assert prompt.index("You are Kai") < marker_idx

        # (c) User's actual text immediately follows the marker, with
        # nothing but whitespace between them. We compute the substring
        # from the end of the marker up to the start of the user text
        # and assert it is whitespace-only — that proves no other
        # context block was injected adjacent to the user region.
        user_idx = prompt.index("ACTUAL_USER_TEXT")
        between = prompt[marker_idx + len(USER_MESSAGE_MARKER) : user_idx]
        assert between.strip() == "", f"non-whitespace between marker and user text: {between!r}"

    @pytest.mark.asyncio
    async def test_memory_query_captured_before_session_context_pollution(self, home_workspace, foreign_workspace):
        """Pin the read-path invariant for the integration: when Claude
        is on a fresh session (so build_session_context fires and the
        prompt grows to include CLAUDE.md, recent history, API docs,
        etc.), format_context must still receive the RAW user text as
        the embedding query, not the post-prepend prompt. Pre-fix
        shape was protected inside claude._send_locked by capturing
        search_query before any prepend; post-extraction the same
        guarantee comes from `assemble_turn_context` calling
        `extract_text_query` as its first operation. This test exists
        at the integration boundary so a future regression where
        Claude bypasses the helper would be caught here, not only in
        the helper-level unit tests.
        """
        proc = _make_mock_proc([_system_event(), _result_event(), b""])
        claude = _make_claude(
            workspace=foreign_workspace,
            home_workspace=home_workspace,
            webhook_secret="secret",
        )
        claude._proc = proc
        claude._fresh_session = True

        captured: dict = {}

        async def fake_format_context(query, *, user_id, **kwargs):
            captured["query"] = query
            captured["user_id"] = user_id
            return ""

        with (
            patch("kai.backend.get_recent_history", return_value="prior turn"),
            patch("kai.memory.format_context", new=fake_format_context),
        ):
            events = []
            async for event in claude._send_locked("What do I prefer?", chat_id=42):
                events.append(event)

        assert captured["query"] == "What do I prefer?"
        assert captured["user_id"] == "42"


# ── _send_locked: multi-modal prompt ─────────────────────────────────


class TestMultiModalPrompt:
    @pytest.fixture(autouse=True)
    def _patch_kill(self, monkeypatch):
        monkeypatch.setattr(ClaudeCodeBackend, "_kill", AsyncMock())

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

        with patch("kai.backend.get_recent_history", return_value=""):
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

    # -- Issue #456: cross-user kill escalation ----------------------

    @pytest.mark.asyncio
    async def test_kill_primes_inner_pid_cache_before_signaling(self):
        """
        _kill must look up the inner claude PID BEFORE the first
        signal. Once killpg reaps the sudo wrapper, pgrep on the
        dead sudo PID returns nothing - the cache primed here is
        the only handle the final-cleanup pass has on the orphaned
        grandchild.
        """
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc
        claude._pgid = 12345
        claude._effective_claude_user = "daniel"

        # Track the order: lookup must come before sudo_kill/killpg.
        # #459 routed _kill through the async helpers, so we patch
        # the async versions (and _async_sudo_kill so the final-
        # cleanup pass is recorded too).
        call_order: list[str] = []

        async def lookup_records():
            call_order.append("lookup")
            return 99999

        async def sudo_kill_records(*_a, **_k):
            call_order.append("sudo_kill")

        with (
            patch.object(claude, "_async_lookup_inner_claude_pid", side_effect=lookup_records),
            patch.object(claude, "_async_sudo_kill", side_effect=sudo_kill_records),
            patch("os.killpg") as mock_killpg,
        ):
            mock_killpg.side_effect = lambda *_a, **_k: call_order.append("killpg")
            await claude._kill()

        # First event must be the lookup; killpg must come after.
        assert call_order[0] == "lookup"
        assert "killpg" in call_order
        assert call_order.index("lookup") < call_order.index("killpg")

    @pytest.mark.asyncio
    async def test_kill_final_cleanup_escalates_via_sudo(self):
        """
        After the main signal/wait/killpg sequence finishes, _kill
        issues a final `sudo -n -u <target> /bin/kill -9 <pid>` for
        the cached inner claude PID. This is the belt-and-
        suspenders pass that catches a daniel-owned grandchild
        that survived the killpg above (POSIX signal-permission
        gap). #459 moved this off the event loop; we assert the
        async helper is invoked with the right args.
        """
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc
        claude._pgid = 12345
        claude._effective_claude_user = "daniel"

        mock_sudo_kill = AsyncMock()
        with (
            patch.object(claude, "_async_lookup_inner_claude_pid", new=AsyncMock(return_value=99999)),
            patch.object(claude, "_async_sudo_kill", new=mock_sudo_kill),
            patch("os.killpg"),
        ):
            await claude._kill()

        # _async_sudo_kill called at least twice: once during the
        # per-signal escalation (in _async_send_signal_for_close)
        # and once at the final cleanup. Both go to daniel with
        # SIGKILL and PID 99999.
        calls = mock_sudo_kill.await_args_list
        assert len(calls) >= 2, f"expected >= 2 sudo-kill calls, got {len(calls)}: {calls}"
        for call in calls:
            args = call.args
            assert args[0] == "daniel"
            assert args[1] == 99999
            assert args[2] == int(signal.SIGKILL)

    @pytest.mark.asyncio
    async def test_kill_skips_sudo_when_inner_pid_unknown(self):
        """
        If _async_lookup_inner_claude_pid never returns a PID (sudo
        died before forking, or pgrep failed every time), neither
        the per-signal escalation nor the final cleanup call sudo.
        killpg still fires as a best-effort wrapper reap.
        """
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc
        claude._pgid = 12345
        claude._effective_claude_user = "daniel"

        mock_sudo_kill = AsyncMock()
        with (
            patch.object(claude, "_async_lookup_inner_claude_pid", new=AsyncMock(return_value=None)),
            patch.object(claude, "_async_sudo_kill", new=mock_sudo_kill),
            patch("os.killpg") as mock_killpg,
        ):
            await claude._kill()

        mock_sudo_kill.assert_not_called()
        # killpg still fires (initial + final) so any signalable
        # process in the group (the sudo wrapper) gets reaped.
        assert mock_killpg.called

    @pytest.mark.asyncio
    async def test_kill_does_not_block_event_loop_on_slow_pgrep(self):
        """
        Acceptance test for #459: _kill awaits the async pgrep
        helper so the event loop is not stalled while pgrep runs.
        We verify by running a parallel coroutine that increments
        a counter every 10ms; if _kill held the loop for the full
        pgrep duration, the counter would not advance.

        Pre-#459, _lookup_inner_claude_pid used synchronous
        subprocess.run which DID block the event loop. This test
        would have failed on that implementation - the counter
        would be 0 after _kill completed.
        """
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc
        claude._pgid = 12345
        claude._effective_claude_user = "daniel"

        # Slow pgrep mock: takes ~150ms to return, simulating a
        # pathological host where pgrep is sluggish but not hung.
        async def slow_lookup():
            await asyncio.sleep(0.15)
            return 99999

        counter = 0

        async def ticker():
            nonlocal counter
            while True:
                counter += 1
                await asyncio.sleep(0.01)

        ticker_task = asyncio.create_task(ticker())
        try:
            with (
                patch.object(claude, "_async_lookup_inner_claude_pid", side_effect=slow_lookup),
                patch.object(claude, "_async_sudo_kill", new=AsyncMock()),
                patch("os.killpg"),
            ):
                await claude._kill()
        finally:
            ticker_task.cancel()
            try:
                await ticker_task
            except asyncio.CancelledError:
                pass

        # In 150ms with a 10ms tick interval, the ticker should
        # have run ~15 times. Use a conservative lower bound (8)
        # to absorb scheduler jitter on a loaded test host. A
        # blocking implementation would yield 0 - 1.
        assert counter >= 8, f"event loop appears blocked: ticker only reached {counter}"

    @pytest.mark.asyncio
    async def test_kill_skips_sudo_in_single_user_mode(self):
        """
        Single-user mode (no _effective_claude_user) takes the
        non-sudo path entirely. The whole cross-user branch is
        bypassed - no pgrep lookup, no sudo kill.
        """
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc
        claude._pgid = 12345
        # _effective_claude_user stays None - this models a stale
        # instance where the spawn flow never set it (legacy code
        # path), or single-user mode where resolve_claude_user()
        # short-circuited the sudo wrapper.
        claude._effective_claude_user = None

        mock_lookup = AsyncMock()
        mock_sudo_kill = AsyncMock()
        with (
            patch.object(claude, "_async_lookup_inner_claude_pid", new=mock_lookup),
            patch.object(claude, "_async_sudo_kill", new=mock_sudo_kill),
            patch("os.killpg"),
        ):
            await claude._kill()

        mock_lookup.assert_not_called()
        mock_sudo_kill.assert_not_called()


# ── _lookup_inner_claude_pid ─────────────────────────────────────────


class TestLookupInnerClaudePid:
    """
    pgrep-based PID discovery for the cross-user kill escalation
    (#456). The bot spawns `sudo -u <target> -- claude`; sudo
    fork+execs claude as its sole child, so pgrep -P <sudo_pid>
    returns the claude PID.
    """

    def test_returns_none_when_proc_is_none(self):
        """No subprocess -> no PID to look up."""
        claude = _make_claude(claude_user="daniel")
        assert claude._lookup_inner_claude_pid() is None

    def test_parses_pgrep_stdout(self):
        """A single PID line from pgrep is parsed to an int."""
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        claude._proc = mock_proc

        completed = MagicMock(returncode=0, stdout="99999\n")
        with patch("kai.claude.subprocess.run", return_value=completed):
            assert claude._lookup_inner_claude_pid() == 99999

    def test_returns_none_on_pgrep_nonzero_exit(self):
        """pgrep returns 1 when no child matches; we treat as 'no PID'."""
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        claude._proc = mock_proc

        completed = MagicMock(returncode=1, stdout="")
        with patch("kai.claude.subprocess.run", return_value=completed):
            assert claude._lookup_inner_claude_pid() is None

    def test_returns_none_on_empty_stdout(self):
        """pgrep success but no output (race: sudo not yet forked)."""
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        claude._proc = mock_proc

        completed = MagicMock(returncode=0, stdout="   \n")
        with patch("kai.claude.subprocess.run", return_value=completed):
            assert claude._lookup_inner_claude_pid() is None

    def test_handles_pgrep_timeout(self):
        """A hung pgrep must not propagate."""
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        claude._proc = mock_proc

        with patch(
            "kai.claude.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="pgrep", timeout=2),
        ):
            assert claude._lookup_inner_claude_pid() is None

    def test_handles_pgrep_oserror(self):
        """OSError (e.g., pgrep binary missing) returns None silently."""
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        claude._proc = mock_proc

        with patch("kai.claude.subprocess.run", side_effect=OSError("no pgrep")):
            assert claude._lookup_inner_claude_pid() is None

    def test_first_line_when_pgrep_returns_multiple(self):
        """
        pgrep should return one child for sudo (it execs a single
        target), but be defensive: take the first line if multiple
        come back.
        """
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        claude._proc = mock_proc

        completed = MagicMock(returncode=0, stdout="99999\n11111\n")
        with patch("kai.claude.subprocess.run", return_value=completed):
            assert claude._lookup_inner_claude_pid() == 99999


# ── async helpers introduced by #459 ─────────────────────────────────


def _fake_async_proc(returncode: int, stdout: bytes = b"", stderr: bytes = b""):
    """
    Build a fake asyncio.subprocess.Process for tests that patch
    asyncio.create_subprocess_exec. Returns a MagicMock that surfaces
    the communicate(), wait(), and kill() interface that
    _async_lookup_inner_claude_pid and _async_sudo_kill actually
    invoke.
    """
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.wait = AsyncMock(return_value=returncode)
    proc.kill = MagicMock()
    return proc


class TestAsyncLookupInnerClaudePid:
    """
    Async pgrep equivalent of TestLookupInnerClaudePid. Verifies
    `_async_lookup_inner_claude_pid` (#459) returns the same set
    of values as its sync sibling across the same edge cases,
    without blocking the event loop on the underlying pgrep call.
    """

    @pytest.mark.asyncio
    async def test_returns_none_when_proc_is_none(self):
        claude = _make_claude(claude_user="daniel")
        assert await claude._async_lookup_inner_claude_pid() is None

    @pytest.mark.asyncio
    async def test_parses_pgrep_stdout(self):
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        claude._proc = mock_proc

        fake = _fake_async_proc(returncode=0, stdout=b"99999\n")
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=fake)):
            assert await claude._async_lookup_inner_claude_pid() == 99999

    @pytest.mark.asyncio
    async def test_returns_none_on_nonzero_exit(self):
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        claude._proc = mock_proc

        fake = _fake_async_proc(returncode=1, stdout=b"")
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=fake)):
            assert await claude._async_lookup_inner_claude_pid() is None

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_stdout(self):
        """pgrep success but no output (race: sudo not yet forked)."""
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        claude._proc = mock_proc

        fake = _fake_async_proc(returncode=0, stdout=b"   \n")
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=fake)):
            assert await claude._async_lookup_inner_claude_pid() is None

    @pytest.mark.asyncio
    async def test_handles_pgrep_timeout(self):
        """A hung pgrep must not propagate; reaped via proc.kill()."""
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        claude._proc = mock_proc

        fake = MagicMock()
        fake.communicate = AsyncMock(side_effect=TimeoutError)
        fake.kill = MagicMock()
        fake.wait = AsyncMock()
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=fake)):
            assert await claude._async_lookup_inner_claude_pid() is None
        # Hung pgrep must be reaped, not leaked.
        fake.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_handler_swallows_process_lookup_error_on_kill(self):
        """
        Regression for PR #461 review M-1: if pgrep exits between
        wait_for timing out and the proc.kill() call,
        asyncio.subprocess.Process.kill() raises ProcessLookupError
        (via os.kill on a dead PID). The handler must swallow
        because the outcome we wanted (process gone) is already
        true; otherwise the exception propagates out of _kill /
        shutdown and crashes the caller.
        """
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        claude._proc = mock_proc

        fake = MagicMock()
        fake.communicate = AsyncMock(side_effect=TimeoutError)
        fake.kill = MagicMock(side_effect=ProcessLookupError)
        fake.wait = AsyncMock()
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=fake)):
            # Must NOT raise. Returns None like any other failure mode.
            assert await claude._async_lookup_inner_claude_pid() is None

    @pytest.mark.asyncio
    async def test_handles_oserror_on_spawn(self):
        """OSError (e.g., pgrep binary missing) returns None silently."""
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        claude._proc = mock_proc

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=OSError("no pgrep"))):
            assert await claude._async_lookup_inner_claude_pid() is None

    @pytest.mark.asyncio
    async def test_first_line_when_pgrep_returns_multiple(self):
        """Defensive: take the first line if multiple come back."""
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        claude._proc = mock_proc

        fake = _fake_async_proc(returncode=0, stdout=b"99999\n11111\n")
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=fake)):
            assert await claude._async_lookup_inner_claude_pid() == 99999


class TestAsyncSudoKill:
    """
    Async sudo-kill helper (#459). Verifies command shape,
    diagnostic-log behavior on non-zero exit, and timeout/OSError
    handling parallel to the sync subprocess.run-based equivalent.
    """

    @pytest.mark.asyncio
    async def test_calls_sudo_with_bin_kill_and_int_sig(self):
        """Argument order, /bin/kill anchor, and int(sig) all correct."""
        claude = _make_claude(claude_user="daniel")
        fake = _fake_async_proc(returncode=0)
        spawn = AsyncMock(return_value=fake)
        with patch("asyncio.create_subprocess_exec", new=spawn):
            await claude._async_sudo_kill("daniel", 99999, int(signal.SIGKILL))

        args = spawn.call_args.args
        assert args[:5] == ("sudo", "-n", "-u", "daniel", "/bin/kill")
        assert args[5] == f"-{int(signal.SIGKILL)}"
        assert args[6] == "99999"

    @pytest.mark.asyncio
    async def test_logs_warning_on_nonzero_exit(self, caplog):
        """ESRCH, missing sudoers rule, etc. surface at WARNING."""
        claude = _make_claude(claude_user="daniel")
        fake = _fake_async_proc(returncode=1, stderr=b"sudo: a password is required")
        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=fake)),
            caplog.at_level("WARNING", logger="kai.claude"),
        ):
            await claude._async_sudo_kill("daniel", 99999, int(signal.SIGKILL))

        assert "sudo kill escalation failed" in caplog.text
        assert "99999" in caplog.text

    @pytest.mark.asyncio
    async def test_silent_on_zero_exit(self, caplog):
        """Happy path: no log noise."""
        claude = _make_claude(claude_user="daniel")
        fake = _fake_async_proc(returncode=0)
        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=fake)),
            caplog.at_level("WARNING", logger="kai.claude"),
        ):
            await claude._async_sudo_kill("daniel", 99999, int(signal.SIGKILL))

        assert "escalation failed" not in caplog.text

    @pytest.mark.asyncio
    async def test_logs_warning_on_timeout(self, caplog):
        """Hung sudo logs a timeout warning and reaps the subprocess."""
        claude = _make_claude(claude_user="daniel")
        fake = MagicMock()
        fake.communicate = AsyncMock(side_effect=TimeoutError)
        fake.kill = MagicMock()
        fake.wait = AsyncMock()
        with (
            patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=fake)),
            caplog.at_level("WARNING", logger="kai.claude"),
        ):
            await claude._async_sudo_kill("daniel", 99999, int(signal.SIGKILL))

        assert "timed out" in caplog.text
        fake.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_handler_swallows_process_lookup_error_on_kill(self):
        """
        Regression for PR #461 review M-1: if sudo exits between
        wait_for timing out and proc.kill(), kill() raises
        ProcessLookupError. The handler must swallow because the
        outcome we wanted (process gone) is already true; otherwise
        the exception propagates out of _kill / shutdown and crashes
        the caller.
        """
        claude = _make_claude(claude_user="daniel")
        fake = MagicMock()
        fake.communicate = AsyncMock(side_effect=TimeoutError)
        fake.kill = MagicMock(side_effect=ProcessLookupError)
        fake.wait = AsyncMock()
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=fake)):
            # Must NOT raise.
            await claude._async_sudo_kill("daniel", 99999, int(signal.SIGKILL))

    @pytest.mark.asyncio
    async def test_swallows_oserror_on_spawn(self):
        """Missing sudo binary -> silent return; no propagation."""
        claude = _make_claude(claude_user="daniel")
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(side_effect=OSError("no sudo"))):
            # Must not raise.
            await claude._async_sudo_kill("daniel", 99999, int(signal.SIGKILL))


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

    # -- Issue #456: cross-user kill escalation in shutdown ----------

    @pytest.mark.asyncio
    async def test_shutdown_primes_inner_pid_cache_before_signaling(self):
        """
        shutdown must look up the inner claude PID BEFORE the first
        signal, same as _kill. Once killpg reaps the sudo wrapper,
        pgrep on the dead sudo PID returns nothing; the cache primed
        here is the only handle the final-cleanup pass has on the
        orphaned grandchild.

        Parallel to _kill's test of the same invariant - the two
        async kill paths share the same orphan-leak failure mode,
        and a regression in either would silently leak processes
        on cross-user setups.
        """
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc
        claude._pgid = 12345
        claude._effective_claude_user = "daniel"

        # #459: shutdown now routes through the async helpers, so the
        # patches target the async variants. Same ordering invariant
        # as the _kill version of this test.
        call_order: list[str] = []

        async def lookup_records():
            call_order.append("lookup")
            return 99999

        async def sudo_kill_records(*_a, **_k):
            call_order.append("sudo_kill")

        with (
            patch.object(claude, "_async_lookup_inner_claude_pid", side_effect=lookup_records),
            patch.object(claude, "_async_sudo_kill", side_effect=sudo_kill_records),
            patch("os.killpg") as mock_killpg,
        ):
            mock_killpg.side_effect = lambda *_a, **_k: call_order.append("killpg")
            await claude.shutdown()

        assert call_order[0] == "lookup"
        assert "killpg" in call_order
        assert call_order.index("lookup") < call_order.index("killpg")

    @pytest.mark.asyncio
    async def test_shutdown_final_cleanup_escalates_via_sudo(self):
        """
        After the main shutdown signal/wait sequence, the final
        cleanup invokes the async sudo-kill helper with SIGKILL
        for the cached inner claude PID. This catches the daniel-
        owned grandchild that survived the killpg (POSIX signal-
        permission gap on cross-user spawn). #459 moved this off
        the event loop; we assert the async helper is invoked with
        the right args.
        """
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc
        claude._pgid = 12345
        claude._effective_claude_user = "daniel"

        mock_sudo_kill = AsyncMock()
        with (
            patch.object(claude, "_async_lookup_inner_claude_pid", new=AsyncMock(return_value=99999)),
            patch.object(claude, "_async_sudo_kill", new=mock_sudo_kill),
            patch("os.killpg"),
        ):
            await claude.shutdown()

        # _async_sudo_kill called at least twice: once during the
        # per-signal escalation (in _async_send_signal_for_close)
        # and once at the final cleanup. The final-cleanup pass
        # uses SIGKILL even when the initial signal was SIGTERM.
        calls = mock_sudo_kill.await_args_list
        assert len(calls) >= 2, f"expected >= 2 sudo-kill calls, got {len(calls)}: {calls}"
        sigs_seen = {call.args[2] for call in calls}
        assert int(signal.SIGKILL) in sigs_seen, f"no SIGKILL final cleanup in: {calls}"
        for call in calls:
            args = call.args
            assert args[0] == "daniel"
            assert args[1] == 99999

    @pytest.mark.asyncio
    async def test_shutdown_skips_sudo_when_inner_pid_unknown(self):
        """
        If _async_lookup_inner_claude_pid never returns a PID (sudo
        died before forking, or pgrep failed every time), neither
        the per-signal escalation nor the final cleanup calls sudo.
        killpg still fires as a best-effort wrapper reap.
        """
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc
        claude._pgid = 12345
        claude._effective_claude_user = "daniel"

        mock_sudo_kill = AsyncMock()
        with (
            patch.object(claude, "_async_lookup_inner_claude_pid", new=AsyncMock(return_value=None)),
            patch.object(claude, "_async_sudo_kill", new=mock_sudo_kill),
            patch("os.killpg") as mock_killpg,
        ):
            await claude.shutdown()

        mock_sudo_kill.assert_not_called()
        assert mock_killpg.called

    @pytest.mark.asyncio
    async def test_shutdown_skips_sudo_in_single_user_mode(self):
        """
        Single-user mode (no _effective_claude_user) takes the
        non-sudo path entirely. No pgrep lookup, no sudo kill.
        """
        claude = _make_claude(claude_user="daniel")
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc
        claude._pgid = 12345
        # _effective_claude_user stays None - mirrors single-user
        # mode where resolve_claude_user() short-circuits sudo.
        claude._effective_claude_user = None

        mock_lookup = AsyncMock()
        mock_sudo_kill = AsyncMock()
        with (
            patch.object(claude, "_async_lookup_inner_claude_pid", new=mock_lookup),
            patch.object(claude, "_async_sudo_kill", new=mock_sudo_kill),
            patch("os.killpg"),
        ):
            await claude.shutdown()

        mock_lookup.assert_not_called()
        mock_sudo_kill.assert_not_called()


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

        async def tracking_send(sig):
            call_order.append(f"signal_{int(sig)}")

        # #459 routes shutdown's per-signal dispatch through the
        # async helper rather than the sync _send_signal; patch the
        # async variant to record SIGTERM/SIGKILL order.
        with (
            patch.object(claude, "_save_prompt", side_effect=tracking_save),
            patch.object(claude, "_async_send_signal_for_close", side_effect=tracking_send),
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

    @pytest.mark.asyncio
    async def test_shutdown_waits_for_lock_then_saves(self):
        """shutdown() waits for an active stream to finish, then saves.

        If _send_locked() is mid-stream, shutdown() blocks on lock
        acquisition until the stream completes, then runs _save_prompt()
        and terminates. This prevents the TOCTOU race where the old
        _lock.locked() check could be stale by the time _save_prompt()
        started reading stdout.
        """
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc

        # Track ordering: lock release happens before save
        call_order: list[str] = []
        original_save = AsyncMock(side_effect=lambda **kw: call_order.append("save"))

        # Hold the lock to simulate an in-flight stream
        await claude._lock.acquire()

        with (
            patch.object(claude, "_save_prompt", new=original_save) as mock_save,
            patch.object(claude, "_send_signal"),
        ):
            # Start shutdown in a task (it will block on the lock)
            shutdown_task = asyncio.create_task(claude.shutdown())
            # Let the event loop run - shutdown should be blocked
            await asyncio.sleep(0)
            assert not shutdown_task.done(), "shutdown should be waiting for the lock"

            # Release the lock as if the stream finished
            call_order.append("lock_released")
            claude._lock.release()

            # Let shutdown proceed
            await asyncio.wait_for(shutdown_task, timeout=2)

        # Save was called AFTER the lock was released
        mock_save.assert_called_once()
        assert call_order == ["lock_released", "save"]
        assert claude._proc is None

    @pytest.mark.asyncio
    async def test_shutdown_calls_save_prompt_when_lock_free(self):
        """shutdown() calls _save_prompt() when no interaction is in flight.

        Regression test: the lock guard must not break the normal idle
        shutdown path where _save_prompt() should still run.
        """
        claude = _make_claude()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.wait = AsyncMock()
        claude._proc = mock_proc

        # Lock is free (no active stream)
        assert not claude._lock.locked()

        with (
            patch.object(claude, "_save_prompt", new_callable=AsyncMock) as mock_save,
            patch.object(claude, "_send_signal"),
        ):
            await claude.shutdown()

        mock_save.assert_called_once()
        assert claude._proc is None

    @pytest.mark.asyncio
    async def test_shutdown_during_active_stream(self):
        """Concurrent shutdown during an active stream does not raise readuntil errors.

        Simulates the real race: send() holds the lock and _send_locked()
        reads stdout, then shutdown() fires from another task. shutdown()
        blocks on the lock until the stream finishes. Since EOF triggers
        _kill() inside _send_locked() (cleaning up _proc), shutdown()
        finds _proc=None when it acquires the lock and skips save/terminate
        (nothing left to do). The key assertion: no concurrent stdout
        reads, no RuntimeError.
        """
        claude = _make_claude()

        # Build a mock process whose stdout blocks, then returns EOF.
        eof_event = asyncio.Event()

        async def slow_readline():
            """Block until signalled, then return EOF."""
            await eof_event.wait()
            return b""

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.pid = 99999
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline = slow_readline
        mock_proc.wait = AsyncMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.readline = AsyncMock(return_value=b"")

        claude._proc = mock_proc
        claude._fresh_session = False
        claude._session_started_at = time.monotonic()

        with (
            patch.object(claude, "_ensure_started", new_callable=AsyncMock),
            patch.object(claude, "_save_prompt", new_callable=AsyncMock) as mock_save,
            patch.object(claude, "_send_signal"),
        ):
            # Start streaming via send() (which acquires the lock)
            async def do_stream():
                async for _ in claude.send("hello", chat_id=123):
                    pass

            stream_task = asyncio.create_task(do_stream())
            for _ in range(5):
                await asyncio.sleep(0)

            assert claude._lock.locked(), "Lock should be held by the stream task"

            # Start shutdown - blocks on lock
            shutdown_task = asyncio.create_task(claude.shutdown())
            await asyncio.sleep(0)
            assert not shutdown_task.done(), "shutdown should be waiting for the lock"

            # End the stream by signalling EOF. _send_locked() will
            # call _kill() which sets _proc=None, then release the lock.
            eof_event.set()

            await asyncio.wait_for(stream_task, timeout=2)
            await asyncio.wait_for(shutdown_task, timeout=2)

        # _save_prompt NOT called: the stream's EOF handler calls
        # _kill() which sets _proc=None synchronously before the
        # coroutine yields, so shutdown() always finds _proc=None
        # after acquiring the lock.
        mock_save.assert_not_called()
        assert claude._proc is None


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

    # System prompt tests moved to tests/test_backend.py
    # (TestGetWorkspaceSystemPrompt class)

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
        claude = _make_claude(workspace_config=ws_config, webhook_secret="real_secret")
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
