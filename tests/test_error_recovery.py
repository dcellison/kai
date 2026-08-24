"""Backend protocol error recovery tests retained from issue #326."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.backend import StreamEvent
from kai.claude import ClaudeCodeBackend

# ── claude.py error-event handling ──────────────────────────────────


def _make_claude(**kwargs) -> ClaudeCodeBackend:
    """Mirror of test_claude.py's helper. Local copy so this file is
    self-contained and a future move of the cross-file helper does
    not break this suite silently."""
    defaults = {
        "model": "sonnet",
        "workspace": Path("/tmp/test-workspace"),
        "timeout_seconds": 30,
    }
    defaults.update(kwargs)
    return ClaudeCodeBackend(**defaults)


def _json_line(obj: dict) -> bytes:
    return json.dumps(obj).encode() + b"\n"


def _make_mock_proc(stdout_lines: list[bytes]) -> MagicMock:
    """Mock subprocess that yields the given stdout lines and then
    EOF (b'' as the last entry). Mirrors test_claude.py's helper."""
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


async def _collect_events(claude: ClaudeCodeBackend, prompt: str = "test") -> list[StreamEvent]:
    return [event async for event in claude._send_locked(prompt)]


def _system_event(session_id: str = "sess-326") -> bytes:
    return _json_line({"type": "system", "session_id": session_id})


class TestClaudeErrorPopulation:
    """The CLI's `is_error=true` events come in two shapes (per #326):
    (a) `result` populated with a human-readable reason, (b) `result`
    empty BUT `errors` populated with a list of strings. Pre-#326
    only shape (a) produced a non-None error string; shape (b)
    silently fell through to None and rendered as the literal
    "Error: None" in chat. The fix reads `errors` when `result` is
    empty, with a sentinel fallback for the pathological case where
    both fields are absent."""

    @pytest.mark.asyncio
    async def test_errors_field_populates_response_error_when_result_empty(self):
        """Shape (b): result is empty, errors carries the reason.
        AgentResponse.error must reflect the errors-field content,
        not None."""
        result_event = _json_line(
            {
                "type": "result",
                "result": "",
                "is_error": True,
                "errors": ["API connection lost"],
                "session_id": "sess-326",
                "duration_ms": 90000,
            }
        )
        claude = _make_claude()
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = _make_mock_proc([_system_event(), result_event, b""])
            events = await _collect_events(claude)

        done = [e for e in events if e.done]
        assert len(done) == 1
        response = done[0].response
        assert response is not None
        assert response.success is False
        assert response.error == "API connection lost"

    @pytest.mark.asyncio
    async def test_empty_result_and_empty_errors_falls_back_to_sentinel(self):
        """Pathological case: is_error=true with neither result nor
        errors populated. Must still produce a non-None error string
        so the "Error: None" rendering can't recur. Sentinel value is
        a documented placeholder, not a real error reason."""
        result_event = _json_line(
            {
                "type": "result",
                "result": "",
                "is_error": True,
                "session_id": "sess-326",
                "duration_ms": 100,
            }
        )
        claude = _make_claude()
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = _make_mock_proc([_system_event(), result_event, b""])
            events = await _collect_events(claude)

        done = [e for e in events if e.done]
        response = done[0].response
        assert response is not None
        assert response.success is False
        assert response.error is not None
        assert response.error != ""
        # Sentinel string is "no error detail provided" per the spec;
        # asserted exactly so a future change to the sentinel surfaces
        # here as a test failure rather than silently slipping by.
        assert response.error == "no error detail provided"

    @pytest.mark.asyncio
    async def test_nonempty_result_takes_precedence_over_errors_field(self):
        """Classic happy-failure path: is_error=true with a populated
        `result` field (e.g., the model returned an error message it
        composed itself). The result field wins; errors is ignored.
        Pinned so the new branch logic doesn't accidentally regress
        the original code path."""
        result_event = _json_line(
            {
                "type": "result",
                "result": "Model self-reported error",
                "is_error": True,
                "errors": ["should-be-ignored"],
                "session_id": "sess-326",
                "duration_ms": 1500,
            }
        )
        claude = _make_claude()
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = _make_mock_proc([_system_event(), result_event, b""])
            events = await _collect_events(claude)

        response = next(e for e in events if e.done).response
        assert response is not None
        assert response.error == "Model self-reported error"
