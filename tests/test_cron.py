"""
Tests for cron.py scheduled job registration and execution.

Covers:
1. _ensure_utc() - timezone normalization (existing tests)
2. _register_job() - APScheduler trigger dispatch
3. register_job_by_id() - DB lookup + registration
4. _register_new_jobs() - startup bulk registration
5. init_jobs() - entry point delegation
6. _job_callback() - reminder and Claude job execution branches
"""

import json
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import Forbidden

from kai.cron import (
    _ensure_utc,
    _job_callback,
    _register_job,
    _register_new_jobs,
    init_jobs,
    register_job_by_id,
)

# ── Shared fixtures ─────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def mock_lock(monkeypatch):
    """Stub out the async lock so Claude job tests don't need real asyncio locks."""
    lock = AsyncMock()
    lock.__aenter__ = AsyncMock()
    lock.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr("kai.cron.get_lock", lambda _: lock)


@pytest.fixture()
def mock_app():
    """Build a mock Telegram Application with a stubbed JobQueue."""
    app = MagicMock()
    app.job_queue = MagicMock()
    app.job_queue.jobs.return_value = []
    app.job_queue.run_once = MagicMock()
    app.job_queue.run_daily = MagicMock()
    app.job_queue.run_repeating = MagicMock()
    return app


def _make_job(
    *,
    job_id=1,
    chat_id=12345,
    name="Test Job",
    job_type="reminder",
    prompt="Test prompt",
    auto_remove=False,
    schedule_type="daily",
    schedule_data='{"times": ["09:00"]}',
    notify_on_check=False,
):
    """Build a job dict matching the shape returned by sessions.get_job_by_id."""
    return {
        "id": job_id,
        "chat_id": chat_id,
        "name": name,
        "job_type": job_type,
        "prompt": prompt,
        "auto_remove": auto_remove,
        "schedule_type": schedule_type,
        "schedule_data": schedule_data,
        "notify_on_check": notify_on_check,
    }


@pytest.fixture()
def mock_context():
    """Build a mock Telegram callback context with job data for _job_callback."""
    ctx = AsyncMock()
    ctx.bot = AsyncMock()
    ctx.bot.send_message = AsyncMock()
    ctx.bot.send_chat_action = AsyncMock()
    ctx.job = MagicMock()
    ctx.job.data = {
        "job_id": 1,
        "chat_id": 12345,
        "job_type": "reminder",
        "prompt": "Test prompt",
        "auto_remove": False,
        "name": "Test Job",
        "schedule_type": "daily",
    }
    ctx.job.schedule_removal = MagicMock()
    # Default: no Claude process (reminder tests don't need it)
    ctx.bot_data = {}
    return ctx


def _make_claude_mock(text="The response", success=True, error=None):
    """
    Build a mock Claude process whose send() yields a single done event.

    The async generator protocol matches what _job_callback expects:
    iterate events until event.done is True, then read event.response.
    """
    mock_claude = MagicMock()

    async def fake_send(prompt, chat_id=None):
        event = MagicMock()
        event.done = True
        event.response = MagicMock()
        event.response.success = success
        event.response.text = text
        event.response.error = error
        yield event

    mock_claude.send = fake_send
    return mock_claude


# ── _ensure_utc ──────────────────────────────────────────────────────


class TestEnsureUtc:
    def test_naive_gets_utc(self):
        naive = datetime(2026, 1, 15, 12, 0, 0)
        assert naive.tzinfo is None
        result = _ensure_utc(naive)
        assert result.tzinfo is UTC
        assert result.year == 2026
        assert result.hour == 12

    def test_utc_aware_returned_as_is(self):
        aware = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
        result = _ensure_utc(aware)
        assert result is aware

    def test_other_timezone_returned_as_is(self):
        eastern = timezone(timedelta(hours=-5))
        dt = datetime(2026, 1, 15, 12, 0, 0, tzinfo=eastern)
        result = _ensure_utc(dt)
        assert result is dt
        assert result.tzinfo is eastern


# ── _register_job ────────────────────────────────────────────────────


class TestRegisterJob:
    def test_once_schedule(self, mock_app):
        """run_once is called with the parsed run_at datetime."""
        job = _make_job(
            schedule_type="once",
            schedule_data='{"run_at": "2026-06-01T12:00:00+00:00"}',
        )
        _register_job(mock_app, job)
        mock_app.job_queue.run_once.assert_called_once()
        call_kwargs = mock_app.job_queue.run_once.call_args
        assert call_kwargs.kwargs["name"] == "cron_1"
        assert call_kwargs.kwargs["when"].hour == 12

    def test_interval_schedule(self, mock_app):
        """run_repeating is called with the interval in seconds."""
        job = _make_job(
            schedule_type="interval",
            schedule_data='{"seconds": 3600}',
        )
        _register_job(mock_app, job)
        mock_app.job_queue.run_repeating.assert_called_once()
        call_kwargs = mock_app.job_queue.run_repeating.call_args
        assert call_kwargs.kwargs["interval"] == 3600
        assert call_kwargs.kwargs["name"] == "cron_1"

    def test_daily_single_time(self, mock_app):
        """Single-time daily job uses unsuffixed name (cron_1, not cron_1_0)."""
        job = _make_job(schedule_data='{"times": ["14:30"]}')
        _register_job(mock_app, job)
        mock_app.job_queue.run_daily.assert_called_once()
        call_kwargs = mock_app.job_queue.run_daily.call_args
        assert call_kwargs.kwargs["name"] == "cron_1"
        # Verify the parsed time
        t = call_kwargs.kwargs["time"]
        assert t.hour == 14
        assert t.minute == 30

    def test_daily_multiple_times(self, mock_app):
        """Multi-time daily jobs get suffixed names to avoid APScheduler collisions."""
        job = _make_job(schedule_data='{"times": ["08:00", "20:00"]}')
        _register_job(mock_app, job)
        assert mock_app.job_queue.run_daily.call_count == 2
        names = [c.kwargs["name"] for c in mock_app.job_queue.run_daily.call_args_list]
        assert names == ["cron_1_0", "cron_1_1"]

    def test_daily_invalid_time_string_skipped(self, mock_app, caplog):
        """Unparseable time strings are skipped with an error log."""
        job = _make_job(schedule_data='{"times": ["abc"]}')
        _register_job(mock_app, job)
        mock_app.job_queue.run_daily.assert_not_called()
        assert "Invalid time" in caplog.text

    def test_daily_out_of_range_skipped(self, mock_app, caplog):
        """Out-of-range hour/minute values (e.g. 25:00) are skipped."""
        job = _make_job(schedule_data='{"times": ["25:00"]}')
        _register_job(mock_app, job)
        mock_app.job_queue.run_daily.assert_not_called()
        assert "Invalid time" in caplog.text

    def test_unknown_schedule_type_logs_warning(self, mock_app, caplog):
        """Unknown schedule types are ignored with a warning."""
        job = _make_job(schedule_type="weekly", schedule_data="{}")
        _register_job(mock_app, job)
        assert "Unknown schedule type" in caplog.text
        # Nothing was scheduled
        mock_app.job_queue.run_once.assert_not_called()
        mock_app.job_queue.run_daily.assert_not_called()
        mock_app.job_queue.run_repeating.assert_not_called()


# ── register_job_by_id ───────────────────────────────────────────────


class TestRegisterJobById:
    @pytest.mark.asyncio()
    async def test_job_exists(self, mock_app):
        """Loads the job from the DB and delegates to _register_job."""
        job = _make_job()
        with (
            patch("kai.cron.sessions.get_job_by_id", new_callable=AsyncMock, return_value=job),
            patch("kai.cron._register_job") as mock_register,
        ):
            result = await register_job_by_id(mock_app, 1)
        assert result is True
        mock_register.assert_called_once_with(mock_app, job)

    @pytest.mark.asyncio()
    async def test_job_not_found(self, mock_app, caplog):
        """Returns False and logs an error when the job ID doesn't exist."""
        with patch("kai.cron.sessions.get_job_by_id", new_callable=AsyncMock, return_value=None):
            result = await register_job_by_id(mock_app, 999)
        assert result is False
        assert "not found" in caplog.text


# ── _register_new_jobs ───────────────────────────────────────────────


class TestRegisterNewJobs:
    @pytest.mark.asyncio()
    async def test_registers_unregistered_jobs(self, mock_app):
        """Active jobs not in the scheduler get registered."""
        jobs = [_make_job(job_id=1), _make_job(job_id=2)]
        with (
            patch("kai.cron.sessions.get_all_active_jobs", new_callable=AsyncMock, return_value=jobs),
            patch("kai.cron._register_job") as mock_register,
        ):
            count = await _register_new_jobs(mock_app)
        assert count == 2
        assert mock_register.call_count == 2

    @pytest.mark.asyncio()
    async def test_skips_already_registered(self, mock_app):
        """Jobs whose name is already in the scheduler are skipped."""
        jobs = [_make_job(job_id=5)]
        # Simulate cron_5 already in the job queue
        existing_job = MagicMock()
        existing_job.name = "cron_5"
        mock_app.job_queue.jobs.return_value = [existing_job]
        with (
            patch("kai.cron.sessions.get_all_active_jobs", new_callable=AsyncMock, return_value=jobs),
            patch("kai.cron._register_job") as mock_register,
        ):
            count = await _register_new_jobs(mock_app)
        assert count == 0
        mock_register.assert_not_called()

    @pytest.mark.asyncio()
    async def test_skips_already_registered_daily_multi_time(self, mock_app):
        """Daily multi-time jobs with suffixed names (cron_5_0) are recognized as registered."""
        jobs = [_make_job(job_id=5)]
        existing_job = MagicMock()
        existing_job.name = "cron_5_0"
        mock_app.job_queue.jobs.return_value = [existing_job]
        with (
            patch("kai.cron.sessions.get_all_active_jobs", new_callable=AsyncMock, return_value=jobs),
            patch("kai.cron._register_job") as mock_register,
        ):
            count = await _register_new_jobs(mock_app)
        assert count == 0
        mock_register.assert_not_called()

    @pytest.mark.asyncio()
    async def test_deactivates_expired_one_shot(self, mock_app, caplog):
        """One-shot jobs with run_at in the past get deactivated instead of registered."""
        caplog.set_level("INFO", logger="kai.cron")
        past = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        jobs = [_make_job(job_id=3, schedule_type="once", schedule_data=json.dumps({"run_at": past}))]
        with (
            patch("kai.cron.sessions.get_all_active_jobs", new_callable=AsyncMock, return_value=jobs),
            patch("kai.cron.sessions.deactivate_job", new_callable=AsyncMock) as mock_deactivate,
            patch("kai.cron._register_job") as mock_register,
        ):
            count = await _register_new_jobs(mock_app)
        assert count == 0
        mock_deactivate.assert_called_once_with(3)
        mock_register.assert_not_called()
        assert "expired" in caplog.text.lower()

    @pytest.mark.asyncio()
    async def test_returns_correct_count(self, mock_app):
        """Count reflects only newly registered jobs, not skipped ones."""
        jobs = [_make_job(job_id=1), _make_job(job_id=2), _make_job(job_id=3)]
        # Job 2 is already registered
        existing = MagicMock()
        existing.name = "cron_2"
        mock_app.job_queue.jobs.return_value = [existing]
        with (
            patch("kai.cron.sessions.get_all_active_jobs", new_callable=AsyncMock, return_value=jobs),
            patch("kai.cron._register_job"),
        ):
            count = await _register_new_jobs(mock_app)
        assert count == 2


# ── init_jobs ────────────────────────────────────────────────────────


class TestInitJobs:
    @pytest.mark.asyncio()
    async def test_delegates_to_register_new_jobs(self, mock_app):
        """init_jobs is a thin wrapper that calls _register_new_jobs."""
        with patch("kai.cron._register_new_jobs", new_callable=AsyncMock) as mock_register:
            await init_jobs(mock_app)
        mock_register.assert_called_once_with(mock_app)


# ── _job_callback: reminder jobs ─────────────────────────────────────


class TestJobCallbackReminder:
    @pytest.mark.asyncio()
    async def test_sends_prompt_to_telegram(self, mock_context):
        """Reminder jobs send the prompt text as a Telegram message."""
        with patch("kai.cron.log_message"):
            await _job_callback(mock_context)
        mock_context.bot.send_message.assert_called_once_with(chat_id=12345, text="Test prompt")

    @pytest.mark.asyncio()
    async def test_strips_backslash_escapes(self, mock_context):
        """Stray backslash escapes from bash double-quoting in curl are cleaned."""
        mock_context.job.data["prompt"] = "Don\\!t forget\\. Really\\?"
        with patch("kai.cron.log_message"):
            await _job_callback(mock_context)
        mock_context.bot.send_message.assert_called_once_with(chat_id=12345, text="Don!t forget. Really?")

    @pytest.mark.asyncio()
    async def test_logs_message_to_history(self, mock_context):
        """Reminder delivery is recorded in message history."""
        with patch("kai.cron.log_message") as mock_log:
            await _job_callback(mock_context)
        mock_log.assert_called_once()
        call_kwargs = mock_log.call_args.kwargs
        assert call_kwargs["direction"] == "assistant"
        assert call_kwargs["chat_id"] == 12345

    @pytest.mark.asyncio()
    async def test_forbidden_deactivates_and_removes(self, mock_context):
        """When the chat is gone (Forbidden), the job is deactivated and removed."""
        mock_context.bot.send_message.side_effect = Forbidden("bot was blocked")
        with (
            patch("kai.cron.log_message"),
            patch("kai.cron.sessions.deactivate_job", new_callable=AsyncMock) as mock_deactivate,
        ):
            await _job_callback(mock_context)
        mock_deactivate.assert_called_once_with(1)
        mock_context.job.schedule_removal.assert_called_once()

    @pytest.mark.asyncio()
    async def test_other_exception_does_not_deactivate(self, mock_context):
        """Non-Forbidden exceptions log the error but don't deactivate the job."""
        mock_context.bot.send_message.side_effect = RuntimeError("network error")
        with (
            patch("kai.cron.log_message"),
            patch("kai.cron.sessions.deactivate_job", new_callable=AsyncMock) as mock_deactivate,
        ):
            await _job_callback(mock_context)
        mock_deactivate.assert_not_called()

    @pytest.mark.asyncio()
    async def test_one_shot_deactivates_after_sending(self, mock_context):
        """One-shot reminders (schedule_type=once) auto-deactivate after delivery."""
        mock_context.job.data["schedule_type"] = "once"
        with (
            patch("kai.cron.log_message"),
            patch("kai.cron.sessions.deactivate_job", new_callable=AsyncMock) as mock_deactivate,
        ):
            await _job_callback(mock_context)
        mock_deactivate.assert_called_once_with(1)
        # schedule_removal is NOT called - APScheduler handles that for run_once
        mock_context.job.schedule_removal.assert_not_called()


# ── _job_callback: Claude jobs (no auto-remove) ─────────────────────


class TestJobCallbackClaude:
    @pytest.fixture(autouse=True)
    def _setup_claude_context(self, mock_context):
        """Configure mock_context for Claude job tests."""
        mock_context.job.data["job_type"] = "claude"
        self.ctx = mock_context

    @pytest.mark.asyncio()
    async def test_sends_claude_response_to_telegram(self):
        """Claude response is delivered with a [Job: name] prefix."""
        self.ctx.bot_data = {"pool": _make_claude_mock("Hello world")}
        with patch("kai.cron.log_message"):
            await _job_callback(self.ctx)
        self.ctx.bot.send_message.assert_called_once_with(chat_id=12345, text="[Job: Test Job]\nHello world")

    @pytest.mark.asyncio()
    async def test_shows_typing_indicator(self):
        """A typing indicator is sent before the Claude request."""
        self.ctx.bot_data = {"pool": _make_claude_mock()}
        with patch("kai.cron.log_message"):
            await _job_callback(self.ctx)
        self.ctx.bot.send_chat_action.assert_called_once()

    @pytest.mark.asyncio()
    async def test_no_claude_process_returns_early(self, caplog):
        """When claude isn't in bot_data, logs error and returns without crashing."""
        self.ctx.bot_data = {}
        await _job_callback(self.ctx)
        assert "No subprocess pool" in caplog.text
        self.ctx.bot.send_message.assert_not_called()

    @pytest.mark.asyncio()
    async def test_claude_stream_exception_returns_early(self, caplog):
        """Exceptions during Claude interaction are caught and logged."""
        mock_claude = MagicMock()

        async def exploding_send(prompt):
            raise RuntimeError("process died")
            yield

        mock_claude.send = exploding_send
        self.ctx.bot_data = {"pool": mock_claude}
        await _job_callback(self.ctx)
        assert "crashed" in caplog.text
        self.ctx.bot.send_message.assert_not_called()

    @pytest.mark.asyncio()
    async def test_no_done_event_returns_early(self, caplog):
        """When the stream ends without a done event, logs warning and returns."""
        mock_claude = MagicMock()

        async def empty_send(prompt, chat_id=None):
            # Yield a non-done event, then end
            event = MagicMock()
            event.done = False
            yield event

        mock_claude.send = empty_send
        self.ctx.bot_data = {"pool": mock_claude}
        await _job_callback(self.ctx)
        assert "without a done event" in caplog.text
        self.ctx.bot.send_message.assert_not_called()

    @pytest.mark.asyncio()
    async def test_claude_error_response_returns_early(self, caplog):
        """When Claude returns success=False, logs the error and returns."""
        self.ctx.bot_data = {"pool": _make_claude_mock(success=False, error="rate limited")}
        await _job_callback(self.ctx)
        assert "rate limited" in caplog.text
        self.ctx.bot.send_message.assert_not_called()

    @pytest.mark.asyncio()
    async def test_forbidden_on_send_deactivates(self):
        """Forbidden when sending the result deactivates and removes the job."""
        self.ctx.bot_data = {"pool": _make_claude_mock()}
        self.ctx.bot.send_message.side_effect = Forbidden("blocked")
        with (
            patch("kai.cron.log_message"),
            patch("kai.cron.sessions.deactivate_job", new_callable=AsyncMock) as mock_deactivate,
        ):
            await _job_callback(self.ctx)
        mock_deactivate.assert_called_once_with(1)
        self.ctx.job.schedule_removal.assert_called_once()

    @pytest.mark.asyncio()
    async def test_other_send_exception_does_not_deactivate(self):
        """Non-Forbidden send exceptions log the error but keep the job active."""
        self.ctx.bot_data = {"pool": _make_claude_mock()}
        self.ctx.bot.send_message.side_effect = RuntimeError("timeout")
        with (
            patch("kai.cron.log_message"),
            patch("kai.cron.sessions.deactivate_job", new_callable=AsyncMock) as mock_deactivate,
        ):
            await _job_callback(self.ctx)
        mock_deactivate.assert_not_called()


# ── _job_callback: CONDITION_MET ─────────────────────────────────────


class TestJobCallbackConditionMet:
    @pytest.fixture(autouse=True)
    def _setup_auto_remove_context(self, mock_context):
        """Configure mock_context for auto-remove Claude job tests."""
        mock_context.job.data["job_type"] = "claude"
        mock_context.job.data["auto_remove"] = True
        self.ctx = mock_context

    @pytest.mark.asyncio()
    async def test_detects_condition_met_prefix(self):
        """Case-insensitive CONDITION_MET: prefix triggers the met branch."""
        self.ctx.bot_data = {"pool": _make_claude_mock("condition_met: Package arrived!")}
        with (
            patch("kai.cron.log_message"),
            patch("kai.cron.sessions.deactivate_job", new_callable=AsyncMock) as mock_deactivate,
        ):
            await _job_callback(self.ctx)
        # Job was deactivated and removed
        mock_deactivate.assert_called_once_with(1)
        self.ctx.job.schedule_removal.assert_called_once()

    @pytest.mark.asyncio()
    async def test_extracts_message_after_marker(self):
        """The message text after CONDITION_MET: is delivered to the user."""
        self.ctx.bot_data = {"pool": _make_claude_mock("CONDITION_MET: Package arrived!")}
        with (
            patch("kai.cron.log_message"),
            patch("kai.cron.sessions.deactivate_job", new_callable=AsyncMock),
        ):
            await _job_callback(self.ctx)
        sent_text = self.ctx.bot.send_message.call_args.kwargs["text"]
        assert sent_text == "[Job: Test Job]\nPackage arrived!"

    @pytest.mark.asyncio()
    async def test_multi_line_response(self):
        """Text on lines after the marker line is included in the message."""
        self.ctx.bot_data = {"pool": _make_claude_mock("CONDITION_MET: Done\nHere are details.")}
        with (
            patch("kai.cron.log_message"),
            patch("kai.cron.sessions.deactivate_job", new_callable=AsyncMock),
        ):
            await _job_callback(self.ctx)
        sent_text = self.ctx.bot.send_message.call_args.kwargs["text"]
        assert "Done" in sent_text
        assert "Here are details." in sent_text

    @pytest.mark.asyncio()
    async def test_no_message_after_marker(self):
        """When nothing follows CONDITION_MET:, a default message is sent."""
        self.ctx.bot_data = {"pool": _make_claude_mock("CONDITION_MET:")}
        with (
            patch("kai.cron.log_message"),
            patch("kai.cron.sessions.deactivate_job", new_callable=AsyncMock),
        ):
            await _job_callback(self.ctx)
        sent_text = self.ctx.bot.send_message.call_args.kwargs["text"]
        assert sent_text == "[Job: Test Job] Condition met."

    @pytest.mark.asyncio()
    async def test_forbidden_still_deactivates(self):
        """Even if the chat is gone (Forbidden), the job is still deactivated."""
        self.ctx.bot_data = {"pool": _make_claude_mock("CONDITION_MET: done")}
        self.ctx.bot.send_message.side_effect = Forbidden("blocked")
        with (
            patch("kai.cron.log_message"),
            patch("kai.cron.sessions.deactivate_job", new_callable=AsyncMock) as mock_deactivate,
        ):
            await _job_callback(self.ctx)
        mock_deactivate.assert_called_once_with(1)
        self.ctx.job.schedule_removal.assert_called_once()


# ── _job_callback: CONDITION_NOT_MET ─────────────────────────────────


class TestJobCallbackConditionNotMet:
    @pytest.fixture(autouse=True)
    def _setup_auto_remove_context(self, mock_context):
        """Configure mock_context for auto-remove Claude job tests."""
        mock_context.job.data["job_type"] = "claude"
        mock_context.job.data["auto_remove"] = True
        mock_context.job.data["notify_on_check"] = False
        self.ctx = mock_context

    @pytest.mark.asyncio()
    async def test_detects_condition_not_met(self):
        """CONDITION_NOT_MET is recognized (case-insensitive, no colon required)."""
        self.ctx.bot_data = {"pool": _make_claude_mock("CONDITION_NOT_MET")}
        with patch("kai.cron.sessions.deactivate_job", new_callable=AsyncMock) as mock_deactivate:
            await _job_callback(self.ctx)
        # Job stays active - not deactivated
        mock_deactivate.assert_not_called()
        self.ctx.job.schedule_removal.assert_not_called()

    @pytest.mark.asyncio()
    async def test_silent_when_notify_disabled(self):
        """With notify_on_check=False, no message is sent to the user."""
        self.ctx.bot_data = {"pool": _make_claude_mock("CONDITION_NOT_MET: still waiting")}
        await _job_callback(self.ctx)
        self.ctx.bot.send_message.assert_not_called()

    @pytest.mark.asyncio()
    async def test_sends_message_when_notify_enabled(self):
        """With notify_on_check=True, the status message is sent to the user."""
        self.ctx.job.data["notify_on_check"] = True
        self.ctx.bot_data = {"pool": _make_claude_mock("CONDITION_NOT_MET: Package in transit")}
        with patch("kai.cron.log_message"):
            await _job_callback(self.ctx)
        sent_text = self.ctx.bot.send_message.call_args.kwargs["text"]
        assert sent_text == "[Job: Test Job]\nPackage in transit"

    @pytest.mark.asyncio()
    async def test_handles_optional_colon(self):
        """Both 'CONDITION_NOT_MET: msg' and 'CONDITION_NOT_MET msg' work."""
        self.ctx.job.data["notify_on_check"] = True
        # Without colon
        self.ctx.bot_data = {"pool": _make_claude_mock("CONDITION_NOT_MET Still checking")}
        with patch("kai.cron.log_message"):
            await _job_callback(self.ctx)
        # The lstrip(":") handles the optional colon; without it, the space
        # separates "CONDITION_NOT_MET" from "Still checking"
        sent_text = self.ctx.bot.send_message.call_args.kwargs["text"]
        assert "Still checking" in sent_text

    @pytest.mark.asyncio()
    async def test_no_message_after_marker_with_notify(self):
        """When nothing follows CONDITION_NOT_MET with notify=True, sends default text."""
        self.ctx.job.data["notify_on_check"] = True
        self.ctx.bot_data = {"pool": _make_claude_mock("CONDITION_NOT_MET")}
        with patch("kai.cron.log_message"):
            await _job_callback(self.ctx)
        sent_text = self.ctx.bot.send_message.call_args.kwargs["text"]
        assert sent_text == "[Job: Test Job] Still checking..."

    @pytest.mark.asyncio()
    async def test_forbidden_with_notify_deactivates(self):
        """Forbidden when sending a notify update deactivates and removes the job."""
        self.ctx.job.data["notify_on_check"] = True
        self.ctx.bot_data = {"pool": _make_claude_mock("CONDITION_NOT_MET: status")}
        self.ctx.bot.send_message.side_effect = Forbidden("blocked")
        with (
            patch("kai.cron.log_message"),
            patch("kai.cron.sessions.deactivate_job", new_callable=AsyncMock) as mock_deactivate,
        ):
            await _job_callback(self.ctx)
        mock_deactivate.assert_called_once_with(1)
        self.ctx.job.schedule_removal.assert_called_once()
