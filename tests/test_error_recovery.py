"""
Tests for error-recovery UX (issue #326).

Two layers of behavior:

1. claude.py: when the CLI emits an `is_error=true` event with an
   empty `result` field, the actual reason comes from the `errors`
   field. Falls back to a non-None sentinel string when both are
   empty so the downstream "Error: None" surface can never recur.

2. bot.py: error rendering APPENDS a follow-up message instead of
   OVERWRITING the live streamed message. Pre-#326 the error edit
   erased any tool-use, partial reasoning, and intermediate output
   the user was watching - on long sessions, minutes of visible
   work disappearing into a single error line.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai import bot
from kai.backend import AgentResponse, StreamEvent
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


# ── bot.py error message lifecycle ──────────────────────────────────


class TestErrorMessageLifecycle:
    """End-to-end behavior of `_handle_response` when the stream ends
    in an error: append (don't overwrite), no "Error: None".

    All tests stream a text event BEFORE the terminal error so
    `_handle_response` actually creates a live_msg via the streaming
    loop. Without that, the live_msg branch in the error path is
    never exercised and the overwrite-not-called assertion is
    trivially true regardless of the fix. Assertions on the error
    path target the patched `_reply_safe` directly rather than its
    internal `reply_text` call, so a future refactor of `_reply_safe`
    cannot silently void these tests."""

    def _make_pool_text_then_error(self, error_text: str | None = "API connection lost"):
        """Mock pool whose .send() yields a text event followed by a
        done StreamEvent with the given error string. The text event
        triggers live_msg creation in the streaming loop, so the
        error path's append-not-overwrite contract is exercised
        against an actually-existing live_msg."""

        async def _fake_stream(*args, **kwargs):
            # First a text event so the streaming loop creates live_msg.
            yield StreamEvent(text_so_far="streamed work", done=False, response=None)
            # Then the terminal error.
            yield StreamEvent(
                text_so_far="streamed work",
                done=True,
                response=AgentResponse(
                    text="streamed work",
                    success=False,
                    error=error_text,
                    duration_ms=90000,
                    session_id="sess-326",
                ),
            )

        pool = MagicMock()
        pool.send = MagicMock(side_effect=_fake_stream)
        return pool

    def _make_update_with_live_msg(self):
        """Build an update whose `update.message.reply_text` returns a
        mock `live_msg` (with its own AsyncMock `edit_text`). The
        streaming loop calls reply_text once to create live_msg; that
        first call returns the mock and live_msg becomes truthy in the
        error branch. Subsequent reply_text traffic on update.message
        (e.g., from `_reply_safe` calling reply_text internally) keeps
        going through the same AsyncMock and is observable separately."""
        live_msg = MagicMock()
        live_msg.edit_text = AsyncMock()
        update = MagicMock()
        update.message = MagicMock()
        update.message.reply_text = AsyncMock(return_value=live_msg)
        update.effective_chat.id = 12345
        return update, live_msg

    def _make_context(self):
        from kai.config import Config

        ctx = MagicMock()
        ctx.bot_data = {
            "config": Config(
                telegram_bot_token="t",
                allowed_user_ids={1},
                webhook_secret="s",
                tts_enabled=False,  # voice-mode lookup is skipped
            ),
        }
        ctx.bot.send_chat_action = AsyncMock()
        return ctx

    @pytest.mark.asyncio
    async def test_error_does_not_overwrite_live_msg(self):
        """The streamed message stays untouched; the error is sent as
        a follow-up via _reply_safe. Pre-fix the live_msg.edit_text
        overwrite (via _edit_message_safe) erased visible tool-use
        context; the new behavior preserves it. Asserted by exercising
        the live_msg path (text event triggers creation) and pinning
        that _edit_message_safe is never invoked for the error
        rendering AND _reply_safe IS invoked with an error notice."""
        update, live_msg = self._make_update_with_live_msg()
        ctx = self._make_context()
        pool = self._make_pool_text_then_error("API connection lost")

        with (
            patch("kai.bot.log_message"),
            patch("kai.bot.sessions"),
            patch("kai.bot._edit_message_safe", new_callable=AsyncMock) as mock_edit_safe,
            patch("kai.bot._reply_safe", new_callable=AsyncMock) as mock_reply_safe,
        ):
            await bot._handle_response(update, ctx, chat_id=12345, prompt="hi", pool=pool, model="sonnet")

        # The error path never overwrites live_msg via the safe
        # editor. _edit_message_safe may have been called by the
        # streaming loop's chunk-by-chunk update path (depending on
        # debouncing/edit cadence), but it must NOT have been called
        # with the error string.
        for call in mock_edit_safe.await_args_list:
            text_arg = call.args[1] if len(call.args) > 1 else ""
            assert "Error" not in text_arg, (
                f"_edit_message_safe was called with an error string ({text_arg!r}), "
                f"violating the append-not-overwrite contract"
            )
        # The error notice WAS sent via _reply_safe (asserts the
        # follow-up message path is taken). Routes through the
        # `_error_path_calls` helper so the args-length guard is
        # consistent with the other tests in this class - a future
        # call site that omits the text arg won't raise IndexError
        # here, just produce a meaningful "no error notice" failure.
        error_calls = self._error_path_calls(mock_reply_safe)
        assert len(error_calls) >= 1, "_reply_safe was not called with an error notice"
        # live_msg.edit_text directly should not carry the error
        # either (defensive against bypassing the wrapper).
        for call in live_msg.edit_text.await_args_list:
            text_arg = call.args[0] if call.args else ""
            assert "Error" not in text_arg

    @staticmethod
    def _error_path_calls(mock_reply_safe) -> list:
        """Filter `_reply_safe` calls to those carrying an error
        notice, separating them from the streaming-loop's
        live_msg-creation call (which uses the same wrapper to send
        the initial text chunk). Error-path calls are identified by
        the "Error: " prefix, which the streaming text would never
        legitimately contain."""
        out = []
        for call in mock_reply_safe.await_args_list:
            text = call.args[1] if len(call.args) > 1 else ""
            if text.startswith("Error: "):
                out.append(call)
        return out

    @pytest.mark.asyncio
    async def test_error_sends_exactly_one_message(self):
        """An error produces exactly one error notice (in addition to
        the streaming loop's live_msg-creation call); no extra
        directive messages follow it."""
        update, _live_msg = self._make_update_with_live_msg()
        ctx = self._make_context()
        pool = self._make_pool_text_then_error("Authentication failed")

        with (
            patch("kai.bot.log_message"),
            patch("kai.bot.sessions"),
            patch("kai.bot._edit_message_safe", new_callable=AsyncMock),
            patch("kai.bot._reply_safe", new_callable=AsyncMock) as mock_reply_safe,
        ):
            await bot._handle_response(update, ctx, chat_id=12345, prompt="hi", pool=pool, model="sonnet")

        error_calls = self._error_path_calls(mock_reply_safe)
        assert len(error_calls) == 1
        assert error_calls[0].args[1] == "Error: Authentication failed"

    @pytest.mark.asyncio
    async def test_no_error_none_literal_in_user_facing_output(self):
        """Belt-and-suspenders: even if AgentResponse.error
        regresses to None (despite claude.py's defensive sentinel),
        bot.py's `or "no error detail provided"` fallback prevents
        the literal "Error: None" string from appearing. Pin the
        full chain so a future change at either layer can't
        re-introduce it."""
        update, _live_msg = self._make_update_with_live_msg()
        ctx = self._make_context()
        pool = self._make_pool_text_then_error(None)  # forces fallback path

        captured_log: list[str] = []

        def _capture_log(*, direction, chat_id, text):
            captured_log.append(text)

        with (
            patch("kai.bot.log_message", side_effect=_capture_log),
            patch("kai.bot.sessions"),
            patch("kai.bot._edit_message_safe", new_callable=AsyncMock),
            patch("kai.bot._reply_safe", new_callable=AsyncMock) as mock_reply_safe,
        ):
            await bot._handle_response(update, ctx, chat_id=12345, prompt="hi", pool=pool, model="sonnet")

        # Iterate over ALL _reply_safe calls (not just error-path
        # ones) because the streaming text is also user-facing
        # output that must not contain the bug string. The
        # `_error_path_calls` helper would over-filter here.
        # Length guard mirrors the helper's pattern - protects
        # against a future call site that omits the text arg
        # raising IndexError instead of producing the meaningful
        # "Error: None re-introduced" failure.
        for call in mock_reply_safe.await_args_list:
            text = call.args[1] if len(call.args) > 1 else ""
            assert "Error: None" not in text, f"chat surface re-introduced the bug: {text!r}"
        for text in captured_log:
            assert "[error: None]" not in text, f"history log re-introduced the bug: {text!r}"

    @pytest.mark.asyncio
    async def test_chat_string_matches_history_log(self):
        """The synthetic history entry written by log_message and the
        chat-rendered error notice must carry the same error string,
        so a post-hoc grep of the log lands on the same text the
        operator could see in chat. Divergence would break
        debuggability."""
        update, _live_msg = self._make_update_with_live_msg()
        ctx = self._make_context()
        pool = self._make_pool_text_then_error("API connection lost")

        captured_log: list[str] = []

        def _capture_log(*, direction, chat_id, text):
            captured_log.append(text)

        with (
            patch("kai.bot.log_message", side_effect=_capture_log),
            patch("kai.bot.sessions"),
            patch("kai.bot._edit_message_safe", new_callable=AsyncMock),
            patch("kai.bot._reply_safe", new_callable=AsyncMock) as mock_reply_safe,
        ):
            await bot._handle_response(update, ctx, chat_id=12345, prompt="hi", pool=pool, model="sonnet")

        # The error-path _reply_safe call (filtered from the
        # streaming-loop's live_msg-creation call) carries the error
        # notice; the first log entry carries the same reason inside
        # the [error: <reason>] format.
        error_calls = self._error_path_calls(mock_reply_safe)
        assert len(error_calls) >= 1
        chat_error = error_calls[0].args[1]
        assert "API connection lost" in chat_error
        # Synthetic history entry uses the same reason string.
        assert any("API connection lost" in t for t in captured_log)
