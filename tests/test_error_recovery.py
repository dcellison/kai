"""
Tests for error-recovery UX (issue #326).

Three layers of behavior change:

1. claude.py: when the CLI emits an `is_error=true` event with an
   empty `result` field, the actual reason now comes from the
   `errors` field (where BUDGET_CEILING exhaustion places it). Falls
   back to a non-None sentinel string when both are empty so the
   downstream "Error: None" surface can never recur.

2. bot.py: error rendering APPENDS a follow-up message instead of
   OVERWRITING the live streamed message. Pre-#326 the error edit
   erased any tool-use, partial reasoning, and intermediate output
   the user was watching - on long sessions, minutes of visible
   work disappearing into a single error line.

3. bot.py: budget-exhaustion errors send a recovery directive as a
   second follow-up message ("Type /new to start fresh, or ask your
   operator to raise BUDGET_CEILING"). Other error types fall
   through with no extra guidance for v1.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai import bot
from kai.backend import AgentResponse, StreamEvent
from kai.bot import _budget_recovery_hint, _is_budget_exhaustion
from kai.claude import ClaudeCodeBackend

# ── §9.1 claude.py error-event handling ──────────────────────────────


def _make_claude(**kwargs) -> ClaudeCodeBackend:
    """Mirror of test_claude.py's helper. Local copy so this file is
    self-contained and a future move of the cross-file helper does
    not break this suite silently."""
    defaults = {
        "model": "sonnet",
        "workspace": Path("/tmp/test-workspace"),
        "max_budget_usd": 1.0,
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
    empty BUT `errors` populated with a list of strings (the
    BUDGET_CEILING variant). Pre-#326 only shape (a) produced a
    non-None error string; shape (b) silently fell through to None
    and rendered as the literal "Error: None" in chat. The fix reads
    `errors` when `result` is empty, with a sentinel fallback for the
    pathological case where both fields are absent."""

    @pytest.mark.asyncio
    async def test_errors_field_populates_response_error_when_result_empty(self):
        """The BUDGET_CEILING variant: result is empty, errors carries
        the reason. AgentResponse.error must reflect the errors-field
        content, not None."""
        result_event = _json_line(
            {
                "type": "result",
                "result": "",
                "is_error": True,
                "errors": ["Reached maximum budget ($10)"],
                "session_id": "sess-326",
                "total_cost_usd": 10.103919,
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
        assert response.error == "Reached maximum budget ($10)"

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
                "total_cost_usd": 0.0,
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
                "total_cost_usd": 0.05,
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


# ── §9.3 budget detection + recovery directive ───────────────────────


class TestBudgetDetection:
    """`_is_budget_exhaustion` and `_budget_recovery_hint` are pure
    helpers, easiest to verify directly. The bot.py message-lifecycle
    integration exercises them through the response handler in the
    next class."""

    def test_is_budget_exhaustion_matches_canonical_phrasing(self):
        """The CLI's documented error text includes 'maximum budget'.
        Substring match is case-insensitive and tolerant of dollar
        amount variation."""
        assert _is_budget_exhaustion("Reached maximum budget ($10)") is True
        assert _is_budget_exhaustion("Reached maximum budget ($25.50)") is True
        # Case-insensitive: a future CLI capitalization tweak still matches.
        assert _is_budget_exhaustion("REACHED MAXIMUM BUDGET ($10)") is True

    def test_is_budget_exhaustion_rejects_non_matches(self):
        """Auth failures, network errors, generic strings, and
        None/empty all return False - the directive should NOT fire
        for these."""
        assert _is_budget_exhaustion("Authentication failed") is False
        assert _is_budget_exhaustion("Connection refused") is False
        assert _is_budget_exhaustion("no error detail provided") is False
        assert _is_budget_exhaustion(None) is False
        assert _is_budget_exhaustion("") is False

    def test_budget_recovery_hint_includes_dollar_amount_when_extractable(self):
        """The hint inlines the actual ceiling so the user sees the
        number they hit, not a generic placeholder."""
        hint = _budget_recovery_hint("Reached maximum budget ($10)")
        assert "$10" in hint
        assert "/new" in hint
        assert "BUDGET_CEILING" in hint

        hint_2 = _budget_recovery_hint("Reached maximum budget ($25.50)")
        assert "$25.50" in hint_2

    def test_budget_recovery_hint_falls_back_when_amount_unparseable(self):
        """If the CLI ever emits the budget phrasing without a dollar
        amount in parens (or with a different format), the hint
        gracefully drops the amount rather than failing or rendering
        a malformed string."""
        hint = _budget_recovery_hint("Reached maximum budget")
        # No $ amount in the output, but the directive is still useful
        assert "/new" in hint
        assert "BUDGET_CEILING" in hint
        # No "$N" leakage from the template
        assert "{amount}" not in hint
        assert "$" not in hint

    def test_budget_recovery_hint_tolerates_none(self):
        """Defensive: hint should not crash on None input, since the
        caller's contract relies on this being safe even on
        unexpected error-string shapes."""
        hint = _budget_recovery_hint(None)
        assert "/new" in hint
        assert "BUDGET_CEILING" in hint


# ── §9.2 + §9.3 bot.py message lifecycle on error ────────────────────


class TestErrorMessageLifecycle:
    """End-to-end behavior of `_handle_response` when the stream ends
    in an error: append (don't overwrite), no "Error: None", budget
    errors get the directive, non-budget errors don't."""

    def _make_pool_yielding_error(self, error_text: str | None = "Reached maximum budget ($10)"):
        """Mock pool whose .send() yields one done StreamEvent with
        success=False and the given error string."""

        async def _fake_stream(*args, **kwargs):
            yield StreamEvent(
                text_so_far="",
                done=True,
                response=AgentResponse(
                    text="",
                    success=False,
                    error=error_text,
                    cost_usd=10.10,
                    duration_ms=90000,
                    session_id="sess-326",
                ),
            )

        pool = MagicMock()
        pool.send = MagicMock(side_effect=_fake_stream)
        return pool

    def _make_update(self):
        update = MagicMock()
        update.message = MagicMock()
        update.message.reply_text = AsyncMock()
        update.effective_chat.id = 12345
        return update

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
        a reply_text. Pre-#326 the live_msg.edit_text overwrite
        erased visible tool-use context; the new behavior preserves
        it. Asserted by patching _edit_message_safe and asserting it
        was never called for the error path."""
        update = self._make_update()
        ctx = self._make_context()
        pool = self._make_pool_yielding_error("Reached maximum budget ($10)")

        with (
            patch("kai.bot.log_message"),
            patch("kai.bot.sessions"),
            patch("kai.bot._edit_message_safe", new_callable=AsyncMock) as mock_edit,
        ):
            await bot._handle_response(update, ctx, chat_id=12345, prompt="hi", pool=pool, model="sonnet")

        # The error path never touches live_msg via edit_text or
        # _edit_message_safe; it sends new messages instead.
        mock_edit.assert_not_called()
        # At least one reply_text call (the error notice). Budget
        # variant adds a second; tested separately below.
        assert update.message.reply_text.await_count >= 1

    @pytest.mark.asyncio
    async def test_budget_error_sends_two_messages(self):
        """Budget-exhaustion: error notice + recovery directive,
        sent as two separate reply_text calls."""
        update = self._make_update()
        ctx = self._make_context()
        pool = self._make_pool_yielding_error("Reached maximum budget ($10)")

        with (
            patch("kai.bot.log_message"),
            patch("kai.bot.sessions"),
            patch("kai.bot._edit_message_safe", new_callable=AsyncMock),
        ):
            await bot._handle_response(update, ctx, chat_id=12345, prompt="hi", pool=pool, model="sonnet")

        assert update.message.reply_text.await_count == 2
        first_call = update.message.reply_text.await_args_list[0].args[0]
        second_call = update.message.reply_text.await_args_list[1].args[0]
        assert first_call == "Error: Reached maximum budget ($10)"
        # The directive carries the dollar amount and the recovery hint.
        assert "$10" in second_call
        assert "/new" in second_call
        assert "BUDGET_CEILING" in second_call

    @pytest.mark.asyncio
    async def test_non_budget_error_sends_one_message(self):
        """Generic errors (auth failures, transport errors, etc.)
        get just the error notice; no directive. Structure leaves
        room for additional error-class directives if recurring
        patterns emerge."""
        update = self._make_update()
        ctx = self._make_context()
        pool = self._make_pool_yielding_error("Authentication failed")

        with (
            patch("kai.bot.log_message"),
            patch("kai.bot.sessions"),
            patch("kai.bot._edit_message_safe", new_callable=AsyncMock),
        ):
            await bot._handle_response(update, ctx, chat_id=12345, prompt="hi", pool=pool, model="sonnet")

        assert update.message.reply_text.await_count == 1
        assert update.message.reply_text.await_args.args[0] == "Error: Authentication failed"

    @pytest.mark.asyncio
    async def test_no_error_none_literal_in_user_facing_output(self):
        """Belt-and-suspenders: even if AgentResponse.error
        regresses to None (despite claude.py's defensive sentinel),
        bot.py's `or "no error detail provided"` fallback prevents
        the literal "Error: None" string from appearing. Pin the
        full chain so a future change at either layer can't
        re-introduce it."""
        update = self._make_update()
        ctx = self._make_context()
        pool = self._make_pool_yielding_error(None)  # forces fallback path

        captured_chat: list[str] = []
        captured_log: list[str] = []

        def _capture_log(*, direction, chat_id, text):
            captured_log.append(text)

        async def _capture_reply(text, *args, **kwargs):
            captured_chat.append(text)
            return MagicMock()

        update.message.reply_text = AsyncMock(side_effect=_capture_reply)

        with (
            patch("kai.bot.log_message", side_effect=_capture_log),
            patch("kai.bot.sessions"),
            patch("kai.bot._edit_message_safe", new_callable=AsyncMock),
        ):
            await bot._handle_response(update, ctx, chat_id=12345, prompt="hi", pool=pool, model="sonnet")

        for text in captured_chat:
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
        update = self._make_update()
        ctx = self._make_context()
        pool = self._make_pool_yielding_error("Reached maximum budget ($10)")

        captured_log: list[str] = []

        def _capture_log(*, direction, chat_id, text):
            captured_log.append(text)

        with (
            patch("kai.bot.log_message", side_effect=_capture_log),
            patch("kai.bot.sessions"),
            patch("kai.bot._edit_message_safe", new_callable=AsyncMock),
        ):
            await bot._handle_response(update, ctx, chat_id=12345, prompt="hi", pool=pool, model="sonnet")

        # First reply is the error notice; first log entry should
        # carry the same reason inside the [error: <reason>] format.
        chat_error = update.message.reply_text.await_args_list[0].args[0]
        assert "Reached maximum budget ($10)" in chat_error
        # Synthetic history entry uses the same reason string.
        assert any("Reached maximum budget ($10)" in t for t in captured_log)
