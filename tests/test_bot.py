"""
Tests for bot.py - pure functions and handler coverage.

The first section tests pure/synchronous helpers (resolve_workspace_path,
chunk_text, etc.) with no mocking needed. The second section tests command
handlers, callback handlers, media handlers, and the streaming response
handler using mock Telegram Update/Context objects.
"""

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import BadRequest

from kai.bot import (
    _QUEUED_MESSAGE_MARKER,
    _chunk_text,
    _clear_responding,
    _edit_message_safe,
    _is_authorized,
    _is_workspace_allowed,
    _models_keyboard,
    _notify_if_queued,
    _prepend_queue_marker,
    _reply_safe,
    _require_auth,
    _resolve_workspace_path,
    _save_to_workspace,
    _set_responding,
    _short_workspace_name,
    _truncate_for_telegram,
    _voices_keyboard,
    _workspaces_keyboard,
    create_bot,
    handle_canceljob,
    handle_document,
    handle_help,
    handle_jobs,
    handle_message,
    handle_model,
    handle_model_callback,
    handle_models,
    handle_new,
    handle_photo,
    handle_start,
    handle_stats,
    handle_stop,
    handle_unknown_command,
    handle_voice,
    handle_voice_callback,
    handle_voice_command,
    handle_voices,
    handle_webhooks,
    handle_workspace,
    handle_workspace_callback,
    handle_workspaces,
)
from kai.claude import ClaudeResponse, StreamEvent
from kai.config import Config
from kai.tts import DEFAULT_VOICE, VOICES

# ── _resolve_workspace_path ──────────────────────────────────────────


class TestResolveWorkspacePath:
    def test_valid_name(self, tmp_path):
        result = _resolve_workspace_path("myproject", tmp_path)
        assert result == (tmp_path / "myproject").resolve()

    def test_returns_none_when_no_base(self):
        assert _resolve_workspace_path("anything", None) is None

    def test_rejects_traversal(self, tmp_path):
        assert _resolve_workspace_path("../escape", tmp_path) is None

    def test_resolves_to_base_itself(self, tmp_path):
        result = _resolve_workspace_path(".", tmp_path)
        assert result == tmp_path

    def test_nested_path(self, tmp_path):
        result = _resolve_workspace_path("sub/project", tmp_path)
        assert result == (tmp_path / "sub" / "project").resolve()


# ── _short_workspace_name ────────────────────────────────────────────


class TestShortWorkspaceName:
    def test_path_under_base(self):
        assert _short_workspace_name("/base/myproject", Path("/base")) == "myproject"

    def test_path_not_under_base(self):
        assert _short_workspace_name("/other/myproject", Path("/base")) == "myproject"

    def test_base_is_none(self):
        assert _short_workspace_name("/some/path/project", None) == "project"


# ── _chunk_text ──────────────────────────────────────────────────────


class TestChunkText:
    def test_short_text_single_chunk(self):
        assert _chunk_text("hello", 100) == ["hello"]

    def test_splits_at_double_newline(self):
        text = "a" * 50 + "\n\n" + "b" * 50
        chunks = _chunk_text(text, 60)
        assert len(chunks) == 2
        assert chunks[0] == "a" * 50
        assert chunks[1] == "b" * 50

    def test_splits_at_single_newline_if_no_double(self):
        text = "a" * 50 + "\n" + "b" * 50
        chunks = _chunk_text(text, 60)
        assert len(chunks) == 2
        assert chunks[0] == "a" * 50
        assert chunks[1] == "b" * 50

    def test_splits_at_max_len_if_no_newlines(self):
        text = "a" * 100
        chunks = _chunk_text(text, 50)
        assert chunks == ["a" * 50, "a" * 50]

    def test_empty_string(self):
        assert _chunk_text("") == []


# ── _truncate_for_telegram ───────────────────────────────────────────


class TestTruncateForTelegram:
    def test_short_text_unchanged(self):
        assert _truncate_for_telegram("hello", 100) == "hello"

    def test_long_text_truncated_with_suffix(self):
        result = _truncate_for_telegram("a" * 100, 50)
        assert len(result) == 50
        assert result.endswith("\n...")
        assert result == "a" * 46 + "\n..."

    def test_exact_length_not_truncated(self):
        text = "a" * 50
        assert _truncate_for_telegram(text, 50) == text


# ── _save_to_workspace ──────────────────────────────────────────────


class TestSaveToWorkspace:
    def test_creates_files_directory(self, tmp_path):
        """Automatically creates the files/ subdirectory if missing."""
        _save_to_workspace(b"hello", "test.txt", tmp_path)
        assert (tmp_path / "files").is_dir()

    def test_saves_content_correctly(self, tmp_path):
        """Written bytes match the input exactly."""
        data = b"binary content here"
        result = _save_to_workspace(data, "doc.pdf", tmp_path)
        assert result.read_bytes() == data

    def test_filename_contains_original_name(self, tmp_path):
        """Saved filename preserves the original name after the timestamp."""
        result = _save_to_workspace(b"x", "report.pdf", tmp_path)
        assert "report.pdf" in result.name

    def test_timestamp_prefix_format(self, tmp_path):
        """Filename starts with YYYYMMDD_HHMMSS_ffffff timestamp."""
        result = _save_to_workspace(b"x", "file.txt", tmp_path)
        # Format: YYYYMMDD_HHMMSS_ffffff_file.txt
        parts = result.name.split("_", 3)
        assert len(parts[0]) == 8  # date
        assert len(parts[1]) == 6  # time
        assert len(parts[2]) == 6  # microseconds

    def test_sanitizes_slashes_and_spaces(self, tmp_path):
        """Slashes and spaces in filenames are replaced with underscores."""
        result = _save_to_workspace(b"x", "my file/name.txt", tmp_path)
        assert "/" not in result.name
        assert " " not in result.name

    def test_returns_absolute_path(self, tmp_path):
        """Returned path is absolute and points to an existing file."""
        result = _save_to_workspace(b"x", "test.txt", tmp_path)
        assert result.is_absolute()
        assert result.is_file()


# ── _workspaces_keyboard ────────────────────────────────────────────


def _button_labels(markup) -> list[str]:
    """Flatten InlineKeyboardMarkup into a list of button labels."""
    return [btn.text for row in markup.inline_keyboard for btn in row]


def _button_callbacks(markup) -> list[str]:
    """Flatten InlineKeyboardMarkup into a list of callback data strings."""
    return [btn.callback_data for row in markup.inline_keyboard for btn in row]


class TestWorkspacesKeyboard:
    @pytest.mark.asyncio
    async def test_home_always_first(self, tmp_path):
        """Home button appears first regardless of history or allowed workspaces."""
        markup = await _workspaces_keyboard([], "/home", "/home", None, [])
        assert _button_labels(markup)[0] == "\U0001f3e0 Home \U0001f7e2"

    @pytest.mark.asyncio
    async def test_allowed_workspaces_appear_before_history(self, tmp_path):
        """Pinned workspaces appear between Home and history entries."""
        pinned = tmp_path / "pinned"
        pinned.mkdir()
        history = [{"path": "/other/project"}]
        markup = await _workspaces_keyboard(history, "/other/project", "/home", None, [pinned])
        labels = _button_labels(markup)
        # Home, then pinned, then history
        assert labels[0].startswith("\U0001f3e0 Home")
        assert labels[1] == "pinned"
        assert labels[2].endswith("\U0001f7e2")  # history entry marked as current

    @pytest.mark.asyncio
    async def test_allowed_workspace_callback_data(self, tmp_path):
        """Pinned workspaces use ws:allowed:<index> callback data."""
        pinned = tmp_path / "project-a"
        pinned.mkdir()
        markup = await _workspaces_keyboard([], "/home", "/home", None, [pinned])
        callbacks = _button_callbacks(markup)
        assert "ws:allowed:0" in callbacks

    @pytest.mark.asyncio
    async def test_history_deduplicated_against_allowed(self, tmp_path):
        """A path in both allowed and history appears only once (in allowed section)."""
        pinned = tmp_path / "shared"
        pinned.mkdir()
        history = [{"path": str(pinned)}]
        markup = await _workspaces_keyboard(history, "/home", "/home", None, [pinned])
        labels = _button_labels(markup)
        # Should be: Home + one "shared" entry — not two "shared" entries
        assert labels.count("shared") == 1
        callbacks = _button_callbacks(markup)
        # The single entry should be the allowed version, not a bare history index
        assert "ws:allowed:0" in callbacks
        assert not any(c == "ws:0" for c in callbacks)

    @pytest.mark.asyncio
    async def test_current_workspace_marked_in_allowed(self, tmp_path):
        """Green dot appears on the pinned workspace button when it is current."""
        pinned = tmp_path / "active"
        pinned.mkdir()
        markup = await _workspaces_keyboard([], str(pinned), "/home", None, [pinned])
        labels = _button_labels(markup)
        assert any("active" in lbl and "\U0001f7e2" in lbl for lbl in labels)

    @pytest.mark.asyncio
    async def test_no_allowed_no_history_shows_only_home(self):
        """With no allowed workspaces and no history, only the Home button appears."""
        markup = await _workspaces_keyboard([], "/home", "/home", None, [])
        assert len(_button_labels(markup)) == 1

    @pytest.mark.asyncio
    async def test_disambiguates_duplicate_names(self, tmp_path):
        """Two allowed workspaces with the same directory name get parent/name labels."""
        foo_a = tmp_path / "projects" / "foo"
        foo_b = tmp_path / "clients" / "foo"
        foo_a.mkdir(parents=True)
        foo_b.mkdir(parents=True)
        markup = await _workspaces_keyboard([], "/home", "/home", None, [foo_a, foo_b])
        labels = _button_labels(markup)
        assert "projects/foo" in labels
        assert "clients/foo" in labels
        # Neither bare "foo" label should appear
        assert "foo" not in labels

    @pytest.mark.asyncio
    async def test_unique_names_not_disambiguated(self, tmp_path):
        """Allowed workspaces with unique names keep their short labels."""
        bar = tmp_path / "bar"
        baz = tmp_path / "baz"
        bar.mkdir()
        baz.mkdir()
        markup = await _workspaces_keyboard([], "/home", "/home", None, [bar, baz])
        labels = _button_labels(markup)
        assert "bar" in labels
        assert "baz" in labels


# ── _is_workspace_allowed ────────────────────────────────────────────


def _make_config(**overrides) -> Config:
    """
    Create a Config for tests with sensible defaults.

    Accepts any Config field as a keyword override. Used by both the pure
    function tests (workspace_base, allowed_workspaces) and the handler
    tests (tts_enabled, voice_enabled, webhook_secret, etc.).
    """
    defaults: dict = {
        "telegram_bot_token": "test-token",
        "allowed_user_ids": {1},
        "claude_workspace": Path("/home/workspace"),
        "workspace_base": None,
        "allowed_workspaces": [],
        "webhook_secret": "test-secret",
        "webhook_port": 8080,
        "tts_enabled": False,
        "voice_enabled": False,
    }
    defaults.update(overrides)
    return Config(**defaults)


class TestIsWorkspaceAllowed:
    def test_no_sources_allows_anything(self, tmp_path):
        """With no WORKSPACE_BASE and no ALLOWED_WORKSPACES, all paths are allowed."""
        config = _make_config()
        assert _is_workspace_allowed(tmp_path / "anything", config) is True

    def test_path_under_base_is_allowed(self, tmp_path):
        """Paths under WORKSPACE_BASE are allowed."""
        config = _make_config(workspace_base=tmp_path)
        assert _is_workspace_allowed(tmp_path / "myproject", config) is True

    def test_path_in_allowed_workspaces_is_allowed(self, tmp_path):
        """Paths listed in ALLOWED_WORKSPACES are allowed."""
        project = tmp_path / "project"
        project.mkdir()
        config = _make_config(allowed_workspaces=[project])
        assert _is_workspace_allowed(project, config) is True

    def test_path_outside_both_is_rejected(self, tmp_path):
        """Paths not in WORKSPACE_BASE or ALLOWED_WORKSPACES are rejected."""
        base = tmp_path / "base"
        base.mkdir()
        outside = tmp_path / "outside"
        config = _make_config(workspace_base=base)
        assert _is_workspace_allowed(outside, config) is False

    def test_base_set_allowed_workspaces_empty_rejects_outside(self, tmp_path):
        """With WORKSPACE_BASE set but no allowed workspaces, outside paths are rejected."""
        base = tmp_path / "base"
        base.mkdir()
        config = _make_config(workspace_base=base, allowed_workspaces=[])
        assert _is_workspace_allowed(tmp_path / "other", config) is False

    def test_only_allowed_workspaces_set_rejects_unlisted(self, tmp_path):
        """With only ALLOWED_WORKSPACES set, unlisted paths are rejected."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        unlisted = tmp_path / "unlisted"
        config = _make_config(allowed_workspaces=[allowed])
        assert _is_workspace_allowed(unlisted, config) is False

    def test_resolves_symlinks_for_comparison(self, tmp_path):
        """Path resolution handles non-canonical paths correctly."""
        project = tmp_path / "project"
        project.mkdir()
        config = _make_config(allowed_workspaces=[project])
        # Pass the resolved canonical path — should still match
        assert _is_workspace_allowed(project.resolve(), config) is True


# ── create_bot transport mode ──────────────────────────────────────


class TestCreateBotTransportMode:
    @pytest.fixture(autouse=True)
    def _init_services(self, tmp_path):
        """Initialize services before create_bot() (normally done in main.py).

        create_bot() calls services.get_available_services(), which requires
        load_services() to have been called first. Use a nonexistent file so
        it loads an empty service registry (graceful degradation).
        """
        from kai import services

        services.load_services(tmp_path / "nonexistent.yaml")

    def test_webhook_mode_suppresses_updater(self):
        """In webhook mode, the Updater is suppressed (None)."""
        config = _make_config()
        app = create_bot(config, use_webhook=True)
        assert app.updater is None

    def test_polling_mode_keeps_updater(self):
        """In polling mode, the Updater is present for start_polling()."""
        config = _make_config()
        app = create_bot(config, use_webhook=False)
        assert app.updater is not None


# ══════════════════════════════════════════════════════════════════════
# Handler tests - mock Telegram Update/Context objects
# ══════════════════════════════════════════════════════════════════════


# ── Test helpers ─────────────────────────────────────────────────────


def _make_update(text="hello", chat_id=12345, user_id=1):
    """Create a mock Telegram Update for handler tests."""
    update = MagicMock()
    update.message.text = text
    update.message.reply_text = AsyncMock()
    update.message.delete = AsyncMock()
    update.message.caption = None
    update.message.photo = None
    update.message.document = None
    update.message.voice = None
    update.effective_chat.id = chat_id
    update.effective_chat.send_message = AsyncMock()
    update.effective_user.id = user_id
    return update


def _make_callback_update(data="model:opus", chat_id=12345, user_id=1):
    """Create a mock Update for callback query handlers."""
    update = MagicMock()
    update.callback_query.data = data
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    update.callback_query.edit_message_reply_markup = AsyncMock()
    update.effective_chat.id = chat_id
    update.effective_user.id = user_id
    return update


def _make_mock_claude(model="sonnet", workspace=None, is_alive=True):
    """Create a mock PersistentClaude."""
    claude = MagicMock()
    claude.model = model
    claude.is_alive = is_alive
    claude.workspace = workspace or Path("/home/workspace")
    claude.restart = AsyncMock()
    claude.change_workspace = AsyncMock()
    claude.force_kill = MagicMock()
    claude.send = MagicMock()  # configured per test
    return claude


def _make_context(config=None, claude=None, args=None, user_data=None, job_queue=None):
    """Create a mock PTB context with bot_data, args, and user_data."""
    ctx = MagicMock()
    ctx.bot_data = {
        "config": config or _make_config(),
        "claude": claude or _make_mock_claude(),
    }
    ctx.args = args or []
    ctx.user_data = user_data if user_data is not None else {}
    ctx.bot.send_chat_action = AsyncMock()
    ctx.bot.get_file = AsyncMock()
    ctx.bot.send_voice = AsyncMock()
    if job_queue is not None:
        ctx.application.job_queue = job_queue
    return ctx


def _fake_lock(*_args, **_kwargs):
    """Return a real asyncio.Lock to stand in for the per-chat lock.

    Uses a real Lock instead of a bare async context manager so that both
    async-with and .locked() work (the latter is needed by _notify_if_queued).
    The lock starts unlocked, so _notify_if_queued correctly skips notification.
    """
    return asyncio.Lock()


def _text_event(text: str) -> StreamEvent:
    """Non-final streaming event with accumulated text."""
    return StreamEvent(text_so_far=text, done=False, response=None)


def _done_event(text="Final response", cost=0.01, session_id="sess-1", success=True, error=None) -> StreamEvent:
    """Final streaming event with a ClaudeResponse."""
    return StreamEvent(
        text_so_far=text,
        done=True,
        response=ClaudeResponse(
            text=text,
            success=success,
            error=error,
            cost_usd=cost,
            duration_ms=1000,
            session_id=session_id,
        ),
    )


async def _fake_stream(*events):
    """Async generator that yields StreamEvents."""
    for e in events:
        yield e


# ── Crash recovery flag ──────────────────────────────────────────────


class TestCrashRecoveryFlag:
    def test_set_responding_writes_chat_id(self, tmp_path):
        """Flag file contains the chat ID as text."""
        flag = tmp_path / ".responding_to"
        with patch("kai.bot._RESPONDING_FLAG", flag):
            _set_responding(12345)
        assert flag.read_text() == "12345"

    def test_clear_responding_removes_flag(self, tmp_path):
        """Flag file is deleted after clearing."""
        flag = tmp_path / ".responding_to"
        flag.write_text("12345")
        with patch("kai.bot._RESPONDING_FLAG", flag):
            _clear_responding()
        assert not flag.exists()

    def test_clear_responding_noop_if_missing(self, tmp_path):
        """No error when flag file doesn't exist."""
        flag = tmp_path / ".responding_to"
        with patch("kai.bot._RESPONDING_FLAG", flag):
            _clear_responding()  # should not raise


# ── Authorization ────────────────────────────────────────────────────


class TestAuthorization:
    def test_authorized_user(self):
        config = _make_config(allowed_user_ids={1, 2})
        assert _is_authorized(config, 1) is True

    def test_unauthorized_user(self):
        config = _make_config(allowed_user_ids={1})
        assert _is_authorized(config, 99) is False

    @pytest.mark.asyncio
    async def test_require_auth_calls_wrapped(self):
        """Authorized user: the wrapped function is called."""
        inner = AsyncMock()
        wrapped = _require_auth(inner)
        update = _make_update(user_id=1)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        await wrapped(update, ctx)
        inner.assert_called_once()

    @pytest.mark.asyncio
    async def test_require_auth_blocks_unauthorized(self):
        """Unauthorized user: the wrapped function is NOT called."""
        inner = AsyncMock()
        wrapped = _require_auth(inner)
        update = _make_update(user_id=99)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        await wrapped(update, ctx)
        inner.assert_not_called()


# ── _reply_safe ──────────────────────────────────────────────────────


class TestReplySafe:
    @pytest.mark.asyncio
    async def test_markdown_success(self):
        """Successful Markdown send returns the message."""
        msg = MagicMock()
        msg.reply_text = AsyncMock(return_value="sent")
        result = await _reply_safe(msg, "hello")
        assert result == "sent"

    @pytest.mark.asyncio
    async def test_markdown_fails_retries_plain(self):
        """Markdown failure falls back to plain text."""
        msg = MagicMock()
        msg.reply_text = AsyncMock(side_effect=[BadRequest("bad markup"), "sent-plain"])
        result = await _reply_safe(msg, "*bad*")
        assert result == "sent-plain"
        assert msg.reply_text.call_count == 2


# ── _edit_message_safe ───────────────────────────────────────────────


class TestEditMessageSafe:
    @pytest.mark.asyncio
    async def test_markdown_edit_success(self):
        msg = MagicMock()
        msg.edit_text = AsyncMock()
        await _edit_message_safe(msg, "hello")
        msg.edit_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_markdown_fails_retries_plain(self):
        """BadRequest on Markdown triggers plain text retry."""
        msg = MagicMock()
        msg.edit_text = AsyncMock(side_effect=[BadRequest("bad"), None])
        await _edit_message_safe(msg, "text")
        assert msg.edit_text.call_count == 2

    @pytest.mark.asyncio
    async def test_both_fail_no_exception(self, caplog):
        """Both Markdown and plain text fail: logs debug, no exception raised."""
        msg = MagicMock()
        msg.edit_text = AsyncMock(side_effect=[BadRequest("bad"), RuntimeError("fail")])
        with caplog.at_level(logging.DEBUG, logger="kai.bot"):
            await _edit_message_safe(msg, "text")
        assert "Failed to edit message" in caplog.text

    @pytest.mark.asyncio
    async def test_non_badrequest_exception(self, caplog):
        """Non-BadRequest exception is caught and logged."""
        msg = MagicMock()
        msg.edit_text = AsyncMock(side_effect=RuntimeError("network"))
        with caplog.at_level(logging.DEBUG, logger="kai.bot"):
            await _edit_message_safe(msg, "text")
        assert "Failed to edit message" in caplog.text

    @pytest.mark.asyncio
    async def test_long_text_truncated(self):
        """Text exceeding 4096 chars is truncated before editing."""
        msg = MagicMock()
        msg.edit_text = AsyncMock()
        await _edit_message_safe(msg, "a" * 5000)
        sent = msg.edit_text.call_args[0][0]
        assert len(sent) <= 4096


# ── _models_keyboard ─────────────────────────────────────────────────


class TestModelsKeyboard:
    def test_current_model_gets_green_dot(self):
        kb = _models_keyboard("sonnet")
        labels = _button_labels(kb)
        assert any("\U0001f7e2" in lbl and "Sonnet" in lbl for lbl in labels)

    def test_all_models_present(self):
        kb = _models_keyboard("sonnet")
        callbacks = _button_callbacks(kb)
        assert "model:opus" in callbacks
        assert "model:sonnet" in callbacks
        assert "model:haiku" in callbacks

    def test_callback_data_format(self):
        kb = _models_keyboard("opus")
        callbacks = _button_callbacks(kb)
        assert all(c.startswith("model:") for c in callbacks)


# ── _voices_keyboard ─────────────────────────────────────────────────


class TestVoicesKeyboard:
    def test_current_voice_gets_green_dot(self):
        kb = _voices_keyboard(DEFAULT_VOICE)
        labels = _button_labels(kb)
        assert any("\U0001f7e2" in lbl for lbl in labels)

    def test_all_voices_present(self):
        kb = _voices_keyboard(DEFAULT_VOICE)
        callbacks = _button_callbacks(kb)
        for key in VOICES:
            assert f"voice:{key}" in callbacks

    def test_callback_data_format(self):
        kb = _voices_keyboard("jenny")
        callbacks = _button_callbacks(kb)
        assert all(c.startswith("voice:") for c in callbacks)


# ── Simple command handlers ──────────────────────────────────────────


class TestHandleStart:
    @pytest.mark.asyncio
    async def test_sends_greeting(self):
        update = _make_update()
        ctx = _make_context()
        await handle_start(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "ready" in reply.lower()


class TestHandleNew:
    @pytest.mark.asyncio
    async def test_clears_session_and_restarts(self):
        claude = _make_mock_claude()
        update = _make_update()
        ctx = _make_context(claude=claude)
        with patch("kai.bot.sessions.clear_session", new_callable=AsyncMock) as mock_clear:
            await handle_new(update, ctx)
        claude.restart.assert_called_once()
        mock_clear.assert_called_once_with(12345)
        reply = update.message.reply_text.call_args[0][0]
        assert "cleared" in reply.lower()


class TestHandleHelp:
    @pytest.mark.asyncio
    async def test_sends_help_text(self):
        update = _make_update()
        ctx = _make_context()
        await handle_help(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "/stop" in reply
        assert "/new" in reply
        assert "/workspace" in reply


class TestHandleUnknownCommand:
    @pytest.mark.asyncio
    async def test_echoes_unknown_command(self):
        update = _make_update(text="/foo")
        ctx = _make_context()
        await handle_unknown_command(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "/foo" in reply
        assert "Unknown command" in reply


class TestHandleStop:
    @pytest.mark.asyncio
    async def test_sets_stop_event_and_kills(self):
        """Sets the stop event, kills Claude, and sends confirmation."""
        claude = _make_mock_claude()
        update = _make_update()
        ctx = _make_context(claude=claude)
        stop_event = asyncio.Event()
        with patch("kai.bot.get_stop_event", return_value=stop_event):
            await handle_stop(update, ctx)
        assert stop_event.is_set()
        claude.force_kill.assert_called_once()
        reply = update.message.reply_text.call_args[0][0]
        assert "stopping" in reply.lower()


# ── handle_stats ─────────────────────────────────────────────────────


class TestHandleStats:
    @pytest.mark.asyncio
    async def test_no_active_session(self):
        update = _make_update()
        ctx = _make_context()
        with patch("kai.bot.sessions.get_stats", new_callable=AsyncMock, return_value=None):
            await handle_stats(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "No active session" in reply

    @pytest.mark.asyncio
    async def test_active_session(self):
        update = _make_update()
        ctx = _make_context()
        stats = {
            "session_id": "abcd1234efgh",
            "model": "sonnet",
            "created_at": "2026-01-01",
            "last_used_at": "2026-01-02",
            "total_cost_usd": 0.1234,
        }
        with patch("kai.bot.sessions.get_stats", new_callable=AsyncMock, return_value=stats):
            await handle_stats(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "abcd1234" in reply
        assert "sonnet" in reply
        assert "0.1234" in reply


# ── handle_jobs ──────────────────────────────────────────────────────


class TestHandleJobs:
    @pytest.mark.asyncio
    async def test_no_jobs(self):
        update = _make_update()
        ctx = _make_context()
        with patch("kai.bot.sessions.get_jobs", new_callable=AsyncMock, return_value=[]):
            await handle_jobs(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "No active" in reply

    @pytest.mark.asyncio
    async def test_formats_interval_hours(self):
        """Interval >= 3600s displays as hours."""
        update = _make_update()
        ctx = _make_context()
        jobs = [
            {
                "id": 1,
                "name": "Check",
                "job_type": "claude",
                "schedule_type": "interval",
                "schedule_data": json.dumps({"seconds": 7200}),
            }
        ]
        with patch("kai.bot.sessions.get_jobs", new_callable=AsyncMock, return_value=jobs):
            await handle_jobs(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "2h" in reply
        assert "\U0001f916" in reply  # robot emoji for claude type

    @pytest.mark.asyncio
    async def test_formats_interval_minutes(self):
        """Interval >= 60s but < 3600s displays as minutes."""
        update = _make_update()
        ctx = _make_context()
        jobs = [
            {
                "id": 2,
                "name": "Ping",
                "job_type": "reminder",
                "schedule_type": "interval",
                "schedule_data": json.dumps({"seconds": 300}),
            }
        ]
        with patch("kai.bot.sessions.get_jobs", new_callable=AsyncMock, return_value=jobs):
            await handle_jobs(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "5m" in reply
        assert "\U0001f514" in reply  # bell emoji for reminder type

    @pytest.mark.asyncio
    async def test_formats_daily(self):
        update = _make_update()
        ctx = _make_context()
        jobs = [
            {
                "id": 3,
                "name": "Standup",
                "job_type": "reminder",
                "schedule_type": "daily",
                "schedule_data": json.dumps({"times": ["14:00"]}),
            }
        ]
        with patch("kai.bot.sessions.get_jobs", new_callable=AsyncMock, return_value=jobs):
            await handle_jobs(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "14:00" in reply


# ── handle_canceljob ─────────────────────────────────────────────────


class TestHandleCancelJob:
    @pytest.mark.asyncio
    async def test_no_args(self):
        update = _make_update()
        ctx = _make_context()
        await handle_canceljob(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Usage" in reply

    @pytest.mark.asyncio
    async def test_non_numeric_arg(self):
        update = _make_update()
        ctx = _make_context(args=["abc"])
        await handle_canceljob(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "number" in reply.lower()

    @pytest.mark.asyncio
    async def test_job_not_found(self):
        update = _make_update()
        ctx = _make_context(args=["99"])
        with patch("kai.bot.sessions.delete_job", new_callable=AsyncMock, return_value=False):
            await handle_canceljob(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "not found" in reply.lower()

    @pytest.mark.asyncio
    async def test_successful_deletion(self):
        """Deletes from DB and removes APScheduler jobs."""
        update = _make_update()
        # Mock the job queue with matching jobs
        mock_job = MagicMock()
        mock_job.name = "cron_5"
        mock_job.schedule_removal = MagicMock()
        jq = MagicMock()
        jq.jobs.return_value = [mock_job]
        ctx = _make_context(args=["5"], job_queue=jq)
        with patch("kai.bot.sessions.delete_job", new_callable=AsyncMock, return_value=True):
            await handle_canceljob(update, ctx)
        mock_job.schedule_removal.assert_called_once()
        reply = update.message.reply_text.call_args[0][0]
        assert "cancelled" in reply.lower()


# ── handle_models ────────────────────────────────────────────────────


class TestHandleModels:
    @pytest.mark.asyncio
    async def test_sends_keyboard(self):
        update = _make_update()
        ctx = _make_context()
        await handle_models(update, ctx)
        call = update.message.reply_text.call_args
        assert "Choose a model" in call[0][0]
        assert call[1]["reply_markup"] is not None


# ── handle_model ─────────────────────────────────────────────────────


class TestHandleModel:
    @pytest.mark.asyncio
    async def test_no_args(self):
        update = _make_update()
        ctx = _make_context()
        await handle_model(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Usage" in reply

    @pytest.mark.asyncio
    async def test_invalid_model(self):
        update = _make_update()
        ctx = _make_context(args=["gpt4"])
        await handle_model(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "opus" in reply.lower() or "sonnet" in reply.lower()

    @pytest.mark.asyncio
    async def test_valid_model(self):
        claude = _make_mock_claude(model="sonnet")
        update = _make_update()
        ctx = _make_context(claude=claude, args=["opus"])
        with patch("kai.bot.sessions.clear_session", new_callable=AsyncMock):
            await handle_model(update, ctx)
        assert claude.model == "opus"
        claude.restart.assert_called_once()


# ── handle_model_callback ────────────────────────────────────────────


class TestHandleModelCallback:
    @pytest.mark.asyncio
    async def test_unauthorized(self):
        update = _make_callback_update(data="model:opus", user_id=99)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        await handle_model_callback(update, ctx)
        update.callback_query.answer.assert_called_once_with("Not authorized.")

    @pytest.mark.asyncio
    async def test_invalid_model(self):
        update = _make_callback_update(data="model:gpt4")
        ctx = _make_context()
        await handle_model_callback(update, ctx)
        update.callback_query.answer.assert_called_once_with("Invalid model.")

    @pytest.mark.asyncio
    async def test_same_model_no_change(self):
        """Selecting the current model shows 'No change.'"""
        claude = _make_mock_claude(model="opus")
        update = _make_callback_update(data="model:opus")
        ctx = _make_context(claude=claude)
        await handle_model_callback(update, ctx)
        edit_text = update.callback_query.edit_message_text.call_args[0][0]
        assert "No change" in edit_text

    @pytest.mark.asyncio
    async def test_switch_model(self):
        claude = _make_mock_claude(model="sonnet")
        update = _make_callback_update(data="model:opus")
        ctx = _make_context(claude=claude)
        with patch("kai.bot.sessions.clear_session", new_callable=AsyncMock):
            await handle_model_callback(update, ctx)
        assert claude.model == "opus"
        edit_text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Switched" in edit_text


# ── handle_voice_command ─────────────────────────────────────────────


class TestHandleVoiceCommand:
    @pytest.mark.asyncio
    async def test_tts_disabled(self):
        update = _make_update()
        ctx = _make_context(config=_make_config(tts_enabled=False))
        await handle_voice_command(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "not enabled" in reply.lower()

    @pytest.mark.asyncio
    async def test_toggle_off_to_only(self):
        """No args when mode is off: toggles to 'only'."""
        update = _make_update()
        ctx = _make_context(config=_make_config(tts_enabled=True))
        with (
            patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value=None),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock) as mock_set,
        ):
            await handle_voice_command(update, ctx)
        # Should set to "only" (toggling from default "off")
        mock_set.assert_called_once_with("voice_mode:12345", "only")

    @pytest.mark.asyncio
    async def test_toggle_only_to_off(self):
        """No args when mode is 'only': toggles to 'off'."""
        update = _make_update()
        ctx = _make_context(config=_make_config(tts_enabled=True))

        async def _get(key):
            if "voice_mode" in key:
                return "only"
            return None

        with (
            patch("kai.bot.sessions.get_setting", side_effect=_get),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock) as mock_set,
        ):
            await handle_voice_command(update, ctx)
        mock_set.assert_called_once_with("voice_mode:12345", "off")

    @pytest.mark.asyncio
    async def test_set_mode_on(self):
        update = _make_update()
        ctx = _make_context(config=_make_config(tts_enabled=True), args=["on"])
        with (
            patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value=None),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock) as mock_set,
        ):
            await handle_voice_command(update, ctx)
        mock_set.assert_called_once_with("voice_mode:12345", "on")

    @pytest.mark.asyncio
    async def test_set_voice_name_enables_if_off(self):
        """Setting a voice name auto-enables voice mode when off."""
        update = _make_update()
        # Use a real voice key from the VOICES dict
        voice_key = next(iter(VOICES.keys()))
        ctx = _make_context(config=_make_config(tts_enabled=True), args=[voice_key])

        async def _get(key):
            if "voice_mode" in key:
                return "off"
            return DEFAULT_VOICE

        with (
            patch("kai.bot.sessions.get_setting", side_effect=_get),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock) as mock_set,
        ):
            await handle_voice_command(update, ctx)
        # Should set both voice name and mode
        calls = {c[0] for c in mock_set.call_args_list}
        assert ("voice_name:12345", voice_key) in calls
        assert ("voice_mode:12345", "only") in calls

    @pytest.mark.asyncio
    async def test_invalid_voice_name(self):
        update = _make_update()
        ctx = _make_context(config=_make_config(tts_enabled=True), args=["badname"])
        with patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value=None):
            await handle_voice_command(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Unknown" in reply or "Usage" in reply


# ── handle_voices ────────────────────────────────────────────────────


class TestHandleVoices:
    @pytest.mark.asyncio
    async def test_tts_disabled(self):
        update = _make_update()
        ctx = _make_context(config=_make_config(tts_enabled=False))
        await handle_voices(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "not enabled" in reply.lower()

    @pytest.mark.asyncio
    async def test_sends_keyboard(self):
        update = _make_update()
        ctx = _make_context(config=_make_config(tts_enabled=True))
        with patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value=None):
            await handle_voices(update, ctx)
        call = update.message.reply_text.call_args
        assert call[1]["reply_markup"] is not None


# ── handle_voice_callback ────────────────────────────────────────────


class TestHandleVoiceCallback:
    @pytest.mark.asyncio
    async def test_unauthorized(self):
        update = _make_callback_update(data="voice:jenny", user_id=99)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        await handle_voice_callback(update, ctx)
        update.callback_query.answer.assert_called_once_with("Not authorized.")

    @pytest.mark.asyncio
    async def test_invalid_voice(self):
        update = _make_callback_update(data="voice:nonexistent")
        ctx = _make_context()
        await handle_voice_callback(update, ctx)
        update.callback_query.answer.assert_called_once_with("Invalid voice.")

    @pytest.mark.asyncio
    async def test_same_voice_no_change(self):
        update = _make_callback_update(data=f"voice:{DEFAULT_VOICE}")
        ctx = _make_context()
        with patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value=None):
            await handle_voice_callback(update, ctx)
        edit_text = update.callback_query.edit_message_text.call_args[0][0]
        assert "No change" in edit_text

    @pytest.mark.asyncio
    async def test_switch_voice(self):
        """Switching voice sets the new name and confirms."""
        new_voice = "jenny" if DEFAULT_VOICE != "jenny" else "alan"
        update = _make_callback_update(data=f"voice:{new_voice}")
        ctx = _make_context()
        with (
            patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value=None),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
        ):
            await handle_voice_callback(update, ctx)
        edit_text = update.callback_query.edit_message_text.call_args[0][0]
        assert VOICES[new_voice] in edit_text

    @pytest.mark.asyncio
    async def test_auto_enables_when_off(self):
        """Switching voice auto-enables mode to 'only' when off."""
        new_voice = "jenny" if DEFAULT_VOICE != "jenny" else "alan"
        update = _make_callback_update(data=f"voice:{new_voice}")
        ctx = _make_context()

        async def _get(key):
            if "voice_mode" in key:
                return "off"
            return None  # default voice

        with (
            patch("kai.bot.sessions.get_setting", side_effect=_get),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock) as mock_set,
        ):
            await handle_voice_callback(update, ctx)
        calls = {c[0] for c in mock_set.call_args_list}
        assert ("voice_mode:12345", "only") in calls


# ── handle_webhooks ──────────────────────────────────────────────────


class TestHandleWebhooks:
    @pytest.mark.asyncio
    async def test_running_with_secret(self):
        update = _make_update()
        ctx = _make_context(config=_make_config(webhook_secret="s3cret"))
        with patch("kai.bot.webhook.is_running", return_value=True):
            await handle_webhooks(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "running" in reply
        assert "GitHub setup" in reply

    @pytest.mark.asyncio
    async def test_not_running(self):
        update = _make_update()
        ctx = _make_context(config=_make_config(webhook_secret="s3cret"))
        with patch("kai.bot.webhook.is_running", return_value=False):
            await handle_webhooks(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "not running" in reply

    @pytest.mark.asyncio
    async def test_no_secret(self):
        update = _make_update()
        ctx = _make_context(config=_make_config(webhook_secret=""))
        with patch("kai.bot.webhook.is_running", return_value=True):
            await handle_webhooks(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "WEBHOOK_SECRET not set" in reply


# ── handle_workspace ─────────────────────────────────────────────────


class TestHandleWorkspace:
    @pytest.mark.asyncio
    async def test_no_args_shows_current(self):
        """No args: shows the current workspace."""
        claude = _make_mock_claude(workspace=Path("/home/workspace"))
        update = _make_update()
        ctx = _make_context(claude=claude)
        await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Home" in reply or "workspace" in reply.lower()

    @pytest.mark.asyncio
    async def test_home_switches(self):
        """'home' keyword switches to home workspace."""
        claude = _make_mock_claude(workspace=Path("/other"))
        config = _make_config(claude_workspace=Path("/home/workspace"))
        update = _make_update()
        ctx = _make_context(claude=claude, config=config, args=["home"])
        with (
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.delete_setting", new_callable=AsyncMock),
            patch("kai.bot.webhook.update_workspace"),
        ):
            await handle_workspace(update, ctx)
        claude.change_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_absolute_path_rejected(self):
        update = _make_update()
        ctx = _make_context(args=["/tmp/evil"])
        await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Absolute paths" in reply

    @pytest.mark.asyncio
    async def test_tilde_path_rejected(self):
        update = _make_update()
        ctx = _make_context(args=["~/foo"])
        await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Absolute paths" in reply

    @pytest.mark.asyncio
    async def test_new_without_name(self):
        update = _make_update()
        ctx = _make_context(args=["new"])
        await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Usage" in reply

    @pytest.mark.asyncio
    async def test_new_without_base(self):
        update = _make_update()
        ctx = _make_context(config=_make_config(workspace_base=None), args=["new", "myproj"])
        await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "WORKSPACE_BASE" in reply

    @pytest.mark.asyncio
    async def test_new_already_exists(self, tmp_path):
        existing = tmp_path / "myproj"
        existing.mkdir()
        update = _make_update()
        ctx = _make_context(
            config=_make_config(workspace_base=tmp_path, claude_workspace=Path("/home")),
            args=["new", "myproj"],
        )
        await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Already exists" in reply

    @pytest.mark.asyncio
    async def test_new_creates_and_switches(self, tmp_path):
        """'new <name>' creates the directory, runs git init, and switches."""
        update = _make_update()
        claude = _make_mock_claude(workspace=Path("/home"))
        config = _make_config(workspace_base=tmp_path, claude_workspace=Path("/home"))
        ctx = _make_context(config=config, claude=claude, args=["new", "fresh"])
        mock_proc = MagicMock()
        mock_proc.wait = AsyncMock()
        with (
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.upsert_workspace_history", new_callable=AsyncMock),
            patch("kai.bot.webhook.update_workspace"),
        ):
            await handle_workspace(update, ctx)
        # Directory should have been created
        assert (tmp_path / "fresh").is_dir()
        claude.change_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_name_found_in_base(self, tmp_path):
        """Name resolved via WORKSPACE_BASE."""
        project = tmp_path / "myproj"
        project.mkdir()
        claude = _make_mock_claude(workspace=Path("/other"))
        config = _make_config(workspace_base=tmp_path, claude_workspace=Path("/home"))
        update = _make_update()
        ctx = _make_context(config=config, claude=claude, args=["myproj"])
        with (
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.upsert_workspace_history", new_callable=AsyncMock),
            patch("kai.bot.webhook.update_workspace"),
        ):
            await handle_workspace(update, ctx)
        claude.change_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_name_found_in_allowed(self, tmp_path):
        """Name resolved via ALLOWED_WORKSPACES directory name match."""
        project = tmp_path / "myproj"
        project.mkdir()
        claude = _make_mock_claude(workspace=Path("/other"))
        config = _make_config(
            allowed_workspaces=[project],
            claude_workspace=Path("/home"),
        )
        update = _make_update()
        ctx = _make_context(config=config, claude=claude, args=["myproj"])
        with (
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.upsert_workspace_history", new_callable=AsyncMock),
            patch("kai.bot.webhook.update_workspace"),
        ):
            await handle_workspace(update, ctx)
        claude.change_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_matches_in_allowed(self, tmp_path):
        """Multiple ALLOWED_WORKSPACES with the same name: shows disambiguation message."""
        proj_a = tmp_path / "a" / "proj"
        proj_b = tmp_path / "b" / "proj"
        proj_a.mkdir(parents=True)
        proj_b.mkdir(parents=True)
        config = _make_config(allowed_workspaces=[proj_a, proj_b], claude_workspace=Path("/home"))
        update = _make_update()
        ctx = _make_context(config=config, args=["proj"])
        await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Multiple workspaces" in reply

    @pytest.mark.asyncio
    async def test_not_found_with_sources(self, tmp_path):
        config = _make_config(workspace_base=tmp_path, claude_workspace=Path("/home"))
        update = _make_update()
        ctx = _make_context(config=config, args=["nonexistent"])
        await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "not found" in reply.lower()

    @pytest.mark.asyncio
    async def test_not_found_no_sources(self):
        config = _make_config(workspace_base=None, allowed_workspaces=[], claude_workspace=Path("/home"))
        update = _make_update()
        ctx = _make_context(config=config, args=["anything"])
        await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "WORKSPACE_BASE" in reply


# ── handle_workspaces ────────────────────────────────────────────────


class TestHandleWorkspaces:
    @pytest.mark.asyncio
    async def test_no_history_at_home(self):
        update = _make_update()
        claude = _make_mock_claude(workspace=Path("/home/workspace"))
        config = _make_config(claude_workspace=Path("/home/workspace"))
        ctx = _make_context(config=config, claude=claude)
        with patch("kai.bot.sessions.get_workspace_history", new_callable=AsyncMock, return_value=[]):
            await handle_workspaces(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "No workspace history" in reply

    @pytest.mark.asyncio
    async def test_has_history_shows_keyboard(self):
        update = _make_update()
        claude = _make_mock_claude(workspace=Path("/home/workspace"))
        config = _make_config(claude_workspace=Path("/home/workspace"))
        ctx = _make_context(config=config, claude=claude)
        history = [{"path": "/some/project"}]
        with patch("kai.bot.sessions.get_workspace_history", new_callable=AsyncMock, return_value=history):
            await handle_workspaces(update, ctx)
        call = update.message.reply_text.call_args
        assert call[1]["reply_markup"] is not None


# ── handle_workspace_callback ────────────────────────────────────────


class TestHandleWorkspaceCallback:
    @pytest.mark.asyncio
    async def test_unauthorized(self):
        update = _make_callback_update(data="ws:home", user_id=99)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        await handle_workspace_callback(update, ctx)
        update.callback_query.answer.assert_called_once_with("Not authorized.")

    @pytest.mark.asyncio
    async def test_home(self):
        """ws:home switches to home workspace."""
        claude = _make_mock_claude(workspace=Path("/other"))
        config = _make_config(claude_workspace=Path("/home/workspace"))
        update = _make_callback_update(data="ws:home")
        ctx = _make_context(config=config, claude=claude)
        with (
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.delete_setting", new_callable=AsyncMock),
            patch("kai.bot.webhook.update_workspace"),
        ):
            await handle_workspace_callback(update, ctx)
        claude.change_workspace.assert_called_once()
        edit_text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Home" in edit_text

    @pytest.mark.asyncio
    async def test_allowed_workspace(self, tmp_path):
        """ws:allowed:<idx> switches to the indexed allowed workspace."""
        project = tmp_path / "proj"
        project.mkdir()
        claude = _make_mock_claude(workspace=Path("/other"))
        config = _make_config(
            allowed_workspaces=[project],
            claude_workspace=Path("/home/workspace"),
        )
        update = _make_callback_update(data="ws:allowed:0")
        ctx = _make_context(config=config, claude=claude)
        with (
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.upsert_workspace_history", new_callable=AsyncMock),
            patch("kai.bot.webhook.update_workspace"),
        ):
            await handle_workspace_callback(update, ctx)
        claude.change_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_allowed_nonexistent_dir(self, tmp_path):
        """ws:allowed:<idx> where directory was deleted."""
        gone = tmp_path / "gone"  # not created
        config = _make_config(allowed_workspaces=[gone], claude_workspace=Path("/home"))
        update = _make_callback_update(data="ws:allowed:0")
        ctx = _make_context(config=config)
        await handle_workspace_callback(update, ctx)
        update.callback_query.answer.assert_called_once_with("That workspace no longer exists.")

    @pytest.mark.asyncio
    async def test_allowed_bad_index(self):
        update = _make_callback_update(data="ws:allowed:bad")
        ctx = _make_context()
        await handle_workspace_callback(update, ctx)
        update.callback_query.answer.assert_called_once_with("Invalid selection.")

    @pytest.mark.asyncio
    async def test_allowed_out_of_range(self):
        config = _make_config(allowed_workspaces=[], claude_workspace=Path("/home"))
        update = _make_callback_update(data="ws:allowed:99")
        ctx = _make_context(config=config)
        await handle_workspace_callback(update, ctx)
        update.callback_query.answer.assert_called_once_with("Workspace no longer available.")

    @pytest.mark.asyncio
    async def test_history_entry(self, tmp_path):
        """ws:<idx> switches to a history entry."""
        project = tmp_path / "proj"
        project.mkdir()
        claude = _make_mock_claude(workspace=Path("/other"))
        config = _make_config(claude_workspace=Path("/home/workspace"))
        update = _make_callback_update(data="ws:0")
        ctx = _make_context(config=config, claude=claude)
        history = [{"path": str(project)}]
        with (
            patch("kai.bot.sessions.get_workspace_history", new_callable=AsyncMock, return_value=history),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.upsert_workspace_history", new_callable=AsyncMock),
            patch("kai.bot.webhook.update_workspace"),
        ):
            await handle_workspace_callback(update, ctx)
        claude.change_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_history_no_longer_allowed(self, tmp_path):
        """History entry disallowed: deleted from history, keyboard refreshed."""
        project = tmp_path / "proj"
        project.mkdir()
        # Configure with a base that doesn't contain the project
        other_base = tmp_path / "base"
        other_base.mkdir()
        config = _make_config(
            workspace_base=other_base,
            claude_workspace=Path("/home/workspace"),
        )
        claude = _make_mock_claude(workspace=Path("/home/workspace"))
        update = _make_callback_update(data="ws:0")
        ctx = _make_context(config=config, claude=claude)
        history = [{"path": str(project)}]
        with (
            patch("kai.bot.sessions.get_workspace_history", new_callable=AsyncMock, return_value=history),
            patch("kai.bot.sessions.delete_workspace_history", new_callable=AsyncMock) as mock_del,
        ):
            await handle_workspace_callback(update, ctx)
        mock_del.assert_called_once_with(str(project))
        update.callback_query.answer.assert_called_once_with("That workspace is no longer allowed.")

    @pytest.mark.asyncio
    async def test_history_dir_deleted(self, tmp_path):
        """History entry whose directory no longer exists: removed from history."""
        gone = tmp_path / "gone"  # not created
        config = _make_config(claude_workspace=Path("/home/workspace"))
        claude = _make_mock_claude(workspace=Path("/home/workspace"))
        update = _make_callback_update(data="ws:0")
        ctx = _make_context(config=config, claude=claude)
        history = [{"path": str(gone)}]
        with (
            patch("kai.bot.sessions.get_workspace_history", new_callable=AsyncMock, return_value=history),
            patch("kai.bot.sessions.delete_workspace_history", new_callable=AsyncMock) as mock_del,
        ):
            await handle_workspace_callback(update, ctx)
        mock_del.assert_called_once_with(str(gone))

    @pytest.mark.asyncio
    async def test_already_in_workspace(self, tmp_path):
        """Selecting the current workspace shows 'No change.'"""
        home = Path("/home/workspace")
        claude = _make_mock_claude(workspace=home)
        config = _make_config(claude_workspace=home)
        update = _make_callback_update(data="ws:home")
        ctx = _make_context(config=config, claude=claude)
        await handle_workspace_callback(update, ctx)
        edit_text = update.callback_query.edit_message_text.call_args[0][0]
        assert "No change" in edit_text


# ── handle_message (non-TOTP) ────────────────────────────────────────


class TestHandleMessage:
    @pytest.mark.asyncio
    async def test_normal_message(self):
        """Normal message: logs, acquires lock, calls _handle_response."""
        update = _make_update(text="hello world")
        ctx = _make_context()
        with (
            patch("kai.bot.is_totp_configured", return_value=False),
            patch("kai.bot._handle_response", new_callable=AsyncMock) as mock_resp,
            patch("kai.bot.log_message") as mock_log,
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_message(update, ctx)
        mock_resp.assert_called_once()
        mock_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_message(self):
        """Empty text: returns early without processing."""
        update = _make_update()
        update.message.text = None
        ctx = _make_context()
        with (
            patch("kai.bot.is_totp_configured", return_value=False),
            patch("kai.bot._handle_response", new_callable=AsyncMock) as mock_resp,
        ):
            await handle_message(update, ctx)
        mock_resp.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_and_clear_responding(self):
        """_set_responding called before and _clear_responding after, even on error."""
        update = _make_update()
        ctx = _make_context()
        with (
            patch("kai.bot.is_totp_configured", return_value=False),
            patch("kai.bot._handle_response", new_callable=AsyncMock, side_effect=RuntimeError("boom")),
            patch("kai.bot.log_message"),
            patch("kai.bot._set_responding") as mock_set,
            patch("kai.bot._clear_responding") as mock_clear,
            patch("kai.bot.get_lock", return_value=_fake_lock()),
            pytest.raises(RuntimeError),
        ):
            await handle_message(update, ctx)
        mock_set.assert_called_once()
        mock_clear.assert_called_once()


# ── handle_photo ─────────────────────────────────────────────────────


class TestHandlePhoto:
    @pytest.mark.asyncio
    async def test_downloads_and_sends_multimodal(self, tmp_path):
        """Downloads photo, base64-encodes, and calls _handle_response with list content."""
        update = _make_update()
        photo = MagicMock()
        photo.file_id = "file123"
        photo.file_unique_id = "uniq123"
        update.message.photo = [MagicMock(), photo]  # last is highest res

        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"image-data"))
        claude = _make_mock_claude(workspace=tmp_path)
        ctx = _make_context(claude=claude)
        ctx.bot.get_file = AsyncMock(return_value=mock_file)

        with (
            patch("kai.bot._handle_response", new_callable=AsyncMock) as mock_resp,
            patch("kai.bot.log_message"),
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_photo(update, ctx)
        # The content arg should be a list (multi-modal)
        content = mock_resp.call_args[0][3]
        assert isinstance(content, list)
        assert content[1]["type"] == "image"

    @pytest.mark.asyncio
    async def test_uses_caption(self, tmp_path):
        """Uses caption if provided instead of default question."""
        update = _make_update()
        photo = MagicMock()
        photo.file_id = "file123"
        photo.file_unique_id = "uniq123"
        update.message.photo = [photo]
        update.message.caption = "Describe this logo"

        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"img"))
        claude = _make_mock_claude(workspace=tmp_path)
        ctx = _make_context(claude=claude)
        ctx.bot.get_file = AsyncMock(return_value=mock_file)

        with (
            patch("kai.bot._handle_response", new_callable=AsyncMock) as mock_resp,
            patch("kai.bot.log_message"),
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_photo(update, ctx)
        content = mock_resp.call_args[0][3]
        assert "Describe this logo" in content[0]["text"]


# ── handle_document ──────────────────────────────────────────────────


class TestHandleDocument:
    def _setup_doc(self, update, file_name, mime_type=None, data=b"content"):
        """Attach a mock document to the update."""
        update.message.document = MagicMock()
        update.message.document.file_name = file_name
        update.message.document.file_id = "doc_file_id"
        update.message.document.mime_type = mime_type
        return data

    @pytest.mark.asyncio
    async def test_image_document(self, tmp_path):
        """Image file extension: sent as multi-modal content."""
        update = _make_update()
        self._setup_doc(update, "logo.png", "image/png")
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"png-data"))
        claude = _make_mock_claude(workspace=tmp_path)
        ctx = _make_context(claude=claude)
        ctx.bot.get_file = AsyncMock(return_value=mock_file)

        with (
            patch("kai.bot._handle_response", new_callable=AsyncMock) as mock_resp,
            patch("kai.bot.log_message"),
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_document(update, ctx)
        content = mock_resp.call_args[0][3]
        assert isinstance(content, list)
        assert content[1]["type"] == "image"

    @pytest.mark.asyncio
    async def test_text_document(self, tmp_path):
        """Text file: decoded as UTF-8 and sent as code block string."""
        update = _make_update()
        self._setup_doc(update, "script.py", "text/python")
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"print('hi')"))
        claude = _make_mock_claude(workspace=tmp_path)
        ctx = _make_context(claude=claude)
        ctx.bot.get_file = AsyncMock(return_value=mock_file)

        with (
            patch("kai.bot._handle_response", new_callable=AsyncMock) as mock_resp,
            patch("kai.bot.log_message"),
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_document(update, ctx)
        content = mock_resp.call_args[0][3]
        assert isinstance(content, str)
        assert "```" in content

    @pytest.mark.asyncio
    async def test_text_decode_error(self, tmp_path):
        """UTF-8 decode failure: sends error reply, no _handle_response call."""
        update = _make_update()
        self._setup_doc(update, "data.txt", "text/plain")
        mock_file = MagicMock()
        # Invalid UTF-8 bytes
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"\xff\xfe"))
        claude = _make_mock_claude(workspace=tmp_path)
        ctx = _make_context(claude=claude)
        ctx.bot.get_file = AsyncMock(return_value=mock_file)

        with (
            patch("kai.bot._handle_response", new_callable=AsyncMock) as mock_resp,
            patch("kai.bot.log_message"),
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_document(update, ctx)
        mock_resp.assert_not_called()
        reply = update.message.reply_text.call_args[0][0]
        assert "decode" in reply.lower() or "text" in reply.lower()

    @pytest.mark.asyncio
    async def test_other_file_type(self, tmp_path):
        """Non-image, non-text file: saved to disk, path sent in prompt."""
        update = _make_update()
        self._setup_doc(update, "archive.zip", "application/zip")
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"PK..."))
        claude = _make_mock_claude(workspace=tmp_path)
        ctx = _make_context(claude=claude)
        ctx.bot.get_file = AsyncMock(return_value=mock_file)

        with (
            patch("kai.bot._handle_response", new_callable=AsyncMock) as mock_resp,
            patch("kai.bot.log_message"),
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_document(update, ctx)
        content = mock_resp.call_args[0][3]
        assert isinstance(content, str)
        assert "archive.zip" in content


# ── handle_voice ─────────────────────────────────────────────────────


class TestHandleVoice:
    @pytest.mark.asyncio
    async def test_voice_not_enabled(self):
        update = _make_update()
        update.message.voice = MagicMock()
        ctx = _make_context(config=_make_config(voice_enabled=False))
        await handle_voice(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "not enabled" in reply.lower()

    @pytest.mark.asyncio
    async def test_missing_dependencies(self, tmp_path):
        """Lists missing deps when ffmpeg/whisper/model aren't available."""
        update = _make_update()
        update.message.voice = MagicMock()
        config = _make_config(voice_enabled=True, whisper_model_path=tmp_path / "nomodel")
        ctx = _make_context(config=config)
        with patch("shutil.which", return_value=None):
            await handle_voice(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "ffmpeg" in reply

    @pytest.mark.asyncio
    async def test_transcription_error(self, tmp_path):
        """TranscriptionError: sends error reply."""
        from kai.transcribe import TranscriptionError

        model_file = tmp_path / "model.bin"
        model_file.touch()
        update = _make_update()
        voice_msg = MagicMock()
        voice_msg.file_id = "v1"
        voice_msg.duration = 5
        update.message.voice = voice_msg
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"audio"))
        config = _make_config(voice_enabled=True, whisper_model_path=model_file)
        ctx = _make_context(config=config)
        ctx.bot.get_file = AsyncMock(return_value=mock_file)

        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("kai.bot.transcribe_voice", new_callable=AsyncMock, side_effect=TranscriptionError("fail")),
            patch("kai.bot.log_message"),
        ):
            await handle_voice(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Transcription failed" in reply

    @pytest.mark.asyncio
    async def test_empty_transcript(self, tmp_path):
        model_file = tmp_path / "model.bin"
        model_file.touch()
        update = _make_update()
        voice_msg = MagicMock()
        voice_msg.file_id = "v1"
        voice_msg.duration = 5
        update.message.voice = voice_msg
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"audio"))
        config = _make_config(voice_enabled=True, whisper_model_path=model_file)
        ctx = _make_context(config=config)
        ctx.bot.get_file = AsyncMock(return_value=mock_file)

        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("kai.bot.transcribe_voice", new_callable=AsyncMock, return_value=""),
            patch("kai.bot.log_message"),
        ):
            await handle_voice(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "speech" in reply.lower()

    @pytest.mark.asyncio
    async def test_successful_transcription(self, tmp_path):
        """Echoes transcript, then sends to Claude."""
        model_file = tmp_path / "model.bin"
        model_file.touch()
        update = _make_update()
        voice_msg = MagicMock()
        voice_msg.file_id = "v1"
        voice_msg.duration = 5
        update.message.voice = voice_msg
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"audio"))
        claude = _make_mock_claude(workspace=tmp_path)
        config = _make_config(voice_enabled=True, whisper_model_path=model_file)
        ctx = _make_context(config=config, claude=claude)
        ctx.bot.get_file = AsyncMock(return_value=mock_file)

        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("kai.bot.transcribe_voice", new_callable=AsyncMock, return_value="Hello there"),
            patch("kai.bot.log_message"),
            patch("kai.bot._handle_response", new_callable=AsyncMock) as mock_resp,
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_voice(update, ctx)
        # Echo the transcript
        echo_call = update.message.reply_text.call_args
        assert "Hello there" in echo_call[0][0]
        # Then send to Claude
        prompt = mock_resp.call_args[0][3]
        assert "Hello there" in prompt


# ── _handle_response ─────────────────────────────────────────────────


class TestHandleResponse:
    """Tests for the streaming response handler."""

    def _base_patches(self):
        """
        Common patches for _handle_response tests.

        Returns a dict suitable for patch.multiple("kai.bot", ...).
        Voice mode defaults to "off" (normal text mode).
        """
        return {
            "sessions": MagicMock(
                get_setting=AsyncMock(return_value="off"),
                save_session=AsyncMock(),
            ),
            "log_message": MagicMock(),
        }

    @pytest.mark.asyncio
    async def test_normal_flow(self):
        """Streams text, creates live message, delivers final text."""
        from kai.bot import _handle_response

        update = _make_update()
        claude = _make_mock_claude()
        claude.send = MagicMock(return_value=_fake_stream(_text_event("Hello"), _done_event("Hello world")))
        ctx = _make_context(claude=claude)

        with patch.multiple("kai.bot", **self._base_patches()):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        # Live message should have been created via reply_text
        assert update.message.reply_text.called

    @pytest.mark.asyncio
    async def test_final_matches_last_edit_no_redundant(self):
        """When final text matches last edit, no extra edit is made."""
        from kai.bot import _handle_response

        update = _make_update()
        # The live message mock - track edit_text calls
        live_msg = MagicMock()
        live_msg.edit_text = AsyncMock()
        update.message.reply_text = AsyncMock(return_value=live_msg)

        claude = _make_mock_claude()
        claude.send = MagicMock(return_value=_fake_stream(_text_event("Done"), _done_event("Done")))
        ctx = _make_context(claude=claude)

        with patch.multiple("kai.bot", **self._base_patches()):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        # The important thing is no exception was raised and
        # the response completed successfully

    @pytest.mark.asyncio
    async def test_stop_interruption(self):
        """Stop event during streaming: edits '(stopped)', returns without error."""
        from kai.bot import _handle_response

        update = _make_update()
        live_msg = MagicMock()
        live_msg.edit_text = AsyncMock()
        update.message.reply_text = AsyncMock(return_value=live_msg)

        stop_event = asyncio.Event()

        async def _streaming(*args):
            yield _text_event("Partial")
            # Simulate /stop during streaming
            stop_event.set()
            yield _text_event("More text")
            yield _done_event("Should not reach")

        claude = _make_mock_claude()
        claude.send = MagicMock(return_value=_streaming())
        ctx = _make_context(claude=claude)

        with (
            patch.multiple("kai.bot", **self._base_patches()),
            patch("kai.bot.get_stop_event", return_value=stop_event),
        ):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        # Should NOT send the "No response" error
        replies = [c[0][0] for c in update.message.reply_text.call_args_list]
        assert not any("Error" in r for r in replies)

    @pytest.mark.asyncio
    async def test_no_done_event_error(self):
        """No done event: sends 'No response from Claude' error."""
        from kai.bot import _handle_response

        update = _make_update()
        claude = _make_mock_claude()
        # Stream that ends without a done event
        claude.send = MagicMock(return_value=_fake_stream(_text_event("Partial")))
        ctx = _make_context(claude=claude)

        with patch.multiple("kai.bot", **self._base_patches()):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        replies = [c[0][0] for c in update.message.reply_text.call_args_list]
        assert any("No response from Claude" in r for r in replies)

    @pytest.mark.asyncio
    async def test_error_response_with_live_msg(self):
        """success=False with existing live message: edits error into live message."""
        from kai.bot import _handle_response

        update = _make_update()
        live_msg = MagicMock()
        live_msg.edit_text = AsyncMock()
        update.message.reply_text = AsyncMock(return_value=live_msg)

        claude = _make_mock_claude()
        claude.send = MagicMock(
            return_value=_fake_stream(
                _text_event("Partial"),
                _done_event("Something broke", success=False, error="Something broke"),
            )
        )
        ctx = _make_context(claude=claude)

        with patch.multiple("kai.bot", **self._base_patches()):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        # Error should be edited into the live message
        last_edit = live_msg.edit_text.call_args_list[-1]
        assert "Error" in last_edit[0][0]

    @pytest.mark.asyncio
    async def test_error_response_no_live_msg(self):
        """success=False with no live message: sends error as new reply."""
        from kai.bot import _handle_response

        update = _make_update()
        claude = _make_mock_claude()
        # Done event immediately (no text events to create a live message)
        claude.send = MagicMock(
            return_value=_fake_stream(
                _done_event("Broke", success=False, error="Broke"),
            )
        )
        ctx = _make_context(claude=claude)

        with patch.multiple("kai.bot", **self._base_patches()):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        replies = [c[0][0] for c in update.message.reply_text.call_args_list]
        assert any("Error" in r for r in replies)

    @pytest.mark.asyncio
    async def test_long_response_chunked(self):
        """Response > 4096 chars: first chunk edits live message, rest sent as new messages."""
        from kai.bot import _handle_response

        update = _make_update()
        live_msg = MagicMock()
        live_msg.edit_text = AsyncMock()
        update.message.reply_text = AsyncMock(return_value=live_msg)

        long_text = "a" * 5000
        claude = _make_mock_claude()
        claude.send = MagicMock(return_value=_fake_stream(_text_event("start"), _done_event(long_text)))
        ctx = _make_context(claude=claude)

        with patch.multiple("kai.bot", **self._base_patches()):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        # Multiple messages should have been sent (chunked)
        assert update.message.reply_text.call_count >= 2

    @pytest.mark.asyncio
    async def test_session_saved_with_id(self):
        """Saves session when session_id is present."""
        from kai.bot import _handle_response

        update = _make_update()
        claude = _make_mock_claude()
        claude.send = MagicMock(return_value=_fake_stream(_done_event("Ok", session_id="sess-abc")))
        ctx = _make_context(claude=claude)
        mock_sessions = MagicMock(
            get_setting=AsyncMock(return_value="off"),
            save_session=AsyncMock(),
        )

        with patch("kai.bot.sessions", mock_sessions), patch("kai.bot.log_message"):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        mock_sessions.save_session.assert_called_once()
        args = mock_sessions.save_session.call_args[0]
        assert args[0] == 12345
        assert args[1] == "sess-abc"

    @pytest.mark.asyncio
    async def test_session_not_saved_without_id(self):
        """Does NOT save session when session_id is None."""
        from kai.bot import _handle_response

        update = _make_update()
        claude = _make_mock_claude()
        claude.send = MagicMock(return_value=_fake_stream(_done_event("Ok", session_id=None)))
        ctx = _make_context(claude=claude)
        mock_sessions = MagicMock(
            get_setting=AsyncMock(return_value="off"),
            save_session=AsyncMock(),
        )

        with patch("kai.bot.sessions", mock_sessions), patch("kai.bot.log_message"):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        mock_sessions.save_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_voice_only_mode(self):
        """Voice-only: no live text message, synthesizes and sends voice."""
        from kai.bot import _handle_response

        update = _make_update()
        claude = _make_mock_claude()
        claude.send = MagicMock(return_value=_fake_stream(_done_event("Hello voice")))
        config = _make_config(tts_enabled=True, piper_model_dir=Path("/models"))
        ctx = _make_context(config=config, claude=claude)
        mock_sessions = MagicMock(
            get_setting=AsyncMock(return_value="only"),
            save_session=AsyncMock(),
        )

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.log_message"),
            patch("kai.bot.synthesize_speech", new_callable=AsyncMock, return_value=b"audio-bytes"),
        ):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        ctx.bot.send_voice.assert_called_once()

    @pytest.mark.asyncio
    async def test_voice_only_tts_failure_falls_back(self):
        """Voice-only TTS failure: falls back to text delivery."""
        from kai.bot import _handle_response
        from kai.tts import TTSError

        update = _make_update()
        claude = _make_mock_claude()
        claude.send = MagicMock(return_value=_fake_stream(_done_event("Fallback text")))
        config = _make_config(tts_enabled=True, piper_model_dir=Path("/models"))
        ctx = _make_context(config=config, claude=claude)
        mock_sessions = MagicMock(
            get_setting=AsyncMock(return_value="only"),
            save_session=AsyncMock(),
        )

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.log_message"),
            patch("kai.bot.synthesize_speech", new_callable=AsyncMock, side_effect=TTSError("fail")),
        ):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        # Should fall back to text
        assert update.message.reply_text.called

    @pytest.mark.asyncio
    async def test_text_plus_voice_mode(self):
        """Text+voice mode: sends text normally, then sends voice note."""
        from kai.bot import _handle_response

        update = _make_update()
        live_msg = MagicMock()
        live_msg.edit_text = AsyncMock()
        update.message.reply_text = AsyncMock(return_value=live_msg)

        claude = _make_mock_claude()
        claude.send = MagicMock(return_value=_fake_stream(_text_event("Hi"), _done_event("Hi there")))
        config = _make_config(tts_enabled=True, piper_model_dir=Path("/models"))
        ctx = _make_context(config=config, claude=claude)

        async def _get_setting(key):
            if "voice_mode" in key:
                return "on"
            return DEFAULT_VOICE

        mock_sessions = MagicMock(
            get_setting=AsyncMock(side_effect=_get_setting),
            save_session=AsyncMock(),
        )

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.log_message"),
            patch("kai.bot.synthesize_speech", new_callable=AsyncMock, return_value=b"audio"),
        ):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        # Both text (reply_text) and voice (send_voice) should be sent
        assert update.message.reply_text.called
        ctx.bot.send_voice.assert_called_once()

    @pytest.mark.asyncio
    async def test_typing_task_cancelled(self):
        """Typing indicator task is cancelled in finally block."""
        from kai.bot import _handle_response

        update = _make_update()
        claude = _make_mock_claude()
        claude.send = MagicMock(return_value=_fake_stream(_done_event("Ok")))
        ctx = _make_context(claude=claude)

        with patch.multiple("kai.bot", **self._base_patches()):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        # If we got here without hanging, the typing task was properly cancelled


# ── _notify_if_queued ────────────────────────────────────────────────


class TestNotifyIfQueued:
    """Tests for the pre-lock queue notification and context-switch marker."""

    @pytest.mark.asyncio
    async def test_sends_when_locked(self):
        """Sends a notification and returns True when the lock is already held."""
        update = _make_update()
        chat_id = 12345

        # Acquire the lock to simulate Kai being busy
        from kai.locks import get_lock

        lock = get_lock(chat_id)
        await lock.acquire()
        try:
            result = await _notify_if_queued(update, chat_id)
            assert result is True
            # The notification goes via reply_text (called by _reply_safe)
            update.message.reply_text.assert_called()
            call_text = update.message.reply_text.call_args[0][0]
            assert "Got your message" in call_text
            assert "/stop" in call_text
        finally:
            lock.release()

    @pytest.mark.asyncio
    async def test_silent_when_unlocked(self):
        """Does nothing and returns False when the lock is free."""
        update = _make_update()
        # Use a unique chat_id to avoid lock state from other tests
        chat_id = 99999

        result = await _notify_if_queued(update, chat_id)

        assert result is False
        update.message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_notification_not_sent_to_claude(self):
        """The notification goes directly to Telegram, not through Claude."""
        update = _make_update()
        chat_id = 12345

        from kai.locks import get_lock

        lock = get_lock(chat_id)
        await lock.acquire()
        try:
            # Mock Claude's send to verify it's never called
            mock_claude = _make_mock_claude()
            mock_claude.send = MagicMock()

            await _notify_if_queued(update, chat_id)

            # Claude.send was never called; the notification uses reply_text
            mock_claude.send.assert_not_called()
            update.message.reply_text.assert_called()
        finally:
            lock.release()

    @pytest.mark.asyncio
    async def test_queued_message_gets_notification_then_processes(self):
        """Integration test: message B gets a notification while A holds the lock.

        Simulates two concurrent messages: A acquires the lock and processes,
        B arrives while A is busy, gets a notification, then processes after
        A releases. Proves the full flow works end to end.
        """
        update_b = _make_update(text="second message")
        chat_id = 77777

        from kai.locks import get_lock

        lock = get_lock(chat_id)

        # Track ordering of events
        events: list[str] = []

        async def handler_a():
            """Simulate message A holding the lock."""
            async with lock:
                events.append("a_acquired")
                # Simulate processing time so B's handler runs
                await asyncio.sleep(0.05)
                events.append("a_released")

        async def handler_b():
            """Simulate message B arriving while A is busy."""
            # Small delay so A grabs the lock first
            await asyncio.sleep(0.01)
            # This is the pre-lock notification
            result = await _notify_if_queued(update_b, chat_id)
            assert result is True
            events.append("b_notified")
            async with lock:
                events.append("b_acquired")

        await asyncio.gather(handler_a(), handler_b())

        # B's notification happened while A held the lock
        assert events.index("b_notified") > events.index("a_acquired")
        assert events.index("b_notified") < events.index("a_released")
        # B acquired the lock after A released
        assert events.index("b_acquired") > events.index("a_released")
        # B got the notification
        update_b.message.reply_text.assert_called()
        call_text = update_b.message.reply_text.call_args[0][0]
        assert "Got your message" in call_text


class TestPrependQueueMarker:
    """Tests for _prepend_queue_marker(), the context-switch prompt helper."""

    def test_string_prompt(self):
        """Prepends marker to a plain string prompt."""
        result = _prepend_queue_marker("hello world")
        assert result.startswith(_QUEUED_MESSAGE_MARKER)
        assert result.endswith("hello world")

    def test_multimodal_prompt(self):
        """Prepends marker to the first text block of a multimodal list."""
        content = [
            {"type": "text", "text": "Photo caption"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc"}},
        ]
        result = _prepend_queue_marker(content)
        assert isinstance(result, list)
        assert len(result) == 2
        # First block has marker prepended
        assert result[0]["text"].startswith(_QUEUED_MESSAGE_MARKER)
        assert result[0]["text"].endswith("Photo caption")
        # Second block (image) is unchanged
        assert result[1] == content[1]

    def test_does_not_mutate_original(self):
        """Returns a new list; does not mutate the original content."""
        content = [
            {"type": "text", "text": "original"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc"}},
        ]
        _prepend_queue_marker(content)
        # Original is untouched
        assert content[0]["text"] == "original"
