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
import signal
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.claude import PersistentClaude, StreamEvent

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
    def home_workspace(self, tmp_path):
        """Create a home workspace with identity and memory files."""
        home = tmp_path / "home"
        claude_dir = home / ".claude"
        claude_dir.mkdir(parents=True)
        (claude_dir / "CLAUDE.md").write_text("You are Kai.")
        (claude_dir / "MEMORY.md").write_text("User likes concise responses.")
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


# ── _send_locked: multi-modal prompt ─────────────────────────────────


class TestMultiModalPrompt:
    @pytest.fixture(autouse=True)
    def _patch_kill(self, monkeypatch):
        monkeypatch.setattr(PersistentClaude, "_kill", AsyncMock())

    @pytest.mark.asyncio
    async def test_list_prompt_with_context_injection(self, tmp_path):
        """List prompts get context prepended as text blocks, content sent as-is."""
        home = tmp_path / "home"
        claude_dir = home / ".claude"
        claude_dir.mkdir(parents=True)
        (claude_dir / "MEMORY.md").write_text("Some memory")

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

            async def checking_send(prompt):
                nonlocal lock_was_held
                lock_was_held = claude._lock.locked()
                async for event in original(prompt):
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
        """_kill cancels the stderr drain task."""
        claude = _make_claude()
        mock_task = MagicMock()
        claude._stderr_task = mock_task
        claude._proc = None  # No process to kill

        await claude._kill()

        mock_task.cancel.assert_called_once()
        assert claude._stderr_task is None

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


# ── restart ──────────────────────────────────────────────────────────


class TestRestart:
    @pytest.mark.asyncio
    async def test_calls_kill(self):
        """restart() kills the current process so the next send() starts fresh."""
        claude = _make_claude()

        with patch.object(claude, "_kill", new_callable=AsyncMock) as mock_kill:
            await claude.restart()

        mock_kill.assert_called_once()
