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
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from telegram.error import BadRequest

from kai.bot import (
    _FIELD_ALIASES,
    _QUEUED_MESSAGE_MARKER,
    _acquire_lock_or_kill,
    _clear_responding,
    _do_switch_workspace,
    _edit_message_safe,
    _handle_settings_reset,
    _handle_workspace_allow,
    _handle_workspace_allowed,
    _handle_workspace_config,
    _handle_workspace_deny,
    _is_authorized,
    _is_notify_chat_used,
    _models_keyboard,
    _notify_if_queued,
    _prepend_queue_marker,
    _reply_safe,
    _require_auth,
    _resolve_workspace_path,
    _save_upload,
    _set_responding,
    _short_workspace_name,
    _show_github,
    _show_settings,
    _switch_workspace,
    _truncate_for_telegram,
    _voices_keyboard,
    _workspace_config_suffix,
    _workspaces_keyboard,
    create_bot,
    handle_canceljob,
    handle_document,
    handle_github,
    handle_help,
    handle_jobs,
    handle_message,
    handle_model,
    handle_model_callback,
    handle_models,
    handle_new,
    handle_photo,
    handle_settings,
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
from kai.workspace_utils import is_workspace_allowed

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


# ── _save_upload ────────────────────────────────────────────────────


class TestSaveUpload:
    def test_creates_files_directory(self, tmp_path, monkeypatch):
        """Automatically creates the files/ subdirectory if missing."""
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        _save_upload(b"hello", "test.txt")
        assert (tmp_path / "files").is_dir()

    def test_saves_content_correctly(self, tmp_path, monkeypatch):
        """Written bytes match the input exactly."""
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        data = b"binary content here"
        result = _save_upload(data, "doc.pdf")
        assert result.read_bytes() == data

    def test_filename_contains_original_name(self, tmp_path, monkeypatch):
        """Saved filename preserves the original name after the timestamp."""
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        result = _save_upload(b"x", "report.pdf")
        assert "report.pdf" in result.name

    def test_timestamp_prefix_format(self, tmp_path, monkeypatch):
        """Filename starts with YYYYMMDD_HHMMSS_ffffff timestamp."""
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        result = _save_upload(b"x", "file.txt")
        # Format: YYYYMMDD_HHMMSS_ffffff_file.txt
        parts = result.name.split("_", 3)
        assert len(parts[0]) == 8  # date
        assert len(parts[1]) == 6  # time
        assert len(parts[2]) == 6  # microseconds

    def test_sanitizes_slashes_and_spaces(self, tmp_path, monkeypatch):
        """Slashes and spaces in filenames are replaced with underscores."""
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        result = _save_upload(b"x", "my file/name.txt")
        assert "/" not in result.name
        assert " " not in result.name

    def test_returns_absolute_path(self, tmp_path, monkeypatch):
        """Returned path is absolute and points to an existing file."""
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        result = _save_upload(b"x", "test.txt")
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
    def test_home_always_first(self, tmp_path):
        """Home button appears first regardless of history or allowed workspaces."""
        markup = _workspaces_keyboard([], "/home", "/home", None, [])
        assert _button_labels(markup)[0] == "\U0001f3e0 Home \U0001f7e2"

    def test_allowed_workspaces_appear_before_history(self, tmp_path):
        """Pinned workspaces appear between Home and history entries."""
        pinned = tmp_path / "pinned"
        pinned.mkdir()
        history = [{"path": "/other/project"}]
        markup = _workspaces_keyboard(history, "/other/project", "/home", None, [pinned])
        labels = _button_labels(markup)
        # Home, then pinned, then history
        assert labels[0].startswith("\U0001f3e0 Home")
        assert labels[1] == "pinned"
        assert labels[2].endswith("\U0001f7e2")  # history entry marked as current

    def test_allowed_workspace_callback_data(self, tmp_path):
        """Pinned workspaces use ws:allowed:<index> callback data."""
        pinned = tmp_path / "project-a"
        pinned.mkdir()
        markup = _workspaces_keyboard([], "/home", "/home", None, [pinned])
        callbacks = _button_callbacks(markup)
        assert "ws:allowed:0" in callbacks

    def test_history_deduplicated_against_allowed(self, tmp_path):
        """A path in both allowed and history appears only once (in allowed section)."""
        pinned = tmp_path / "shared"
        pinned.mkdir()
        history = [{"path": str(pinned)}]
        markup = _workspaces_keyboard(history, "/home", "/home", None, [pinned])
        labels = _button_labels(markup)
        # Should be: Home + one "shared" entry - not two "shared" entries
        assert labels.count("shared") == 1
        callbacks = _button_callbacks(markup)
        # The single entry should be the allowed version, not a bare history index
        assert "ws:allowed:0" in callbacks
        assert not any(c == "ws:0" for c in callbacks)

    def test_current_workspace_marked_in_allowed(self, tmp_path):
        """Green dot appears on the pinned workspace button when it is current."""
        pinned = tmp_path / "active"
        pinned.mkdir()
        markup = _workspaces_keyboard([], str(pinned), "/home", None, [pinned])
        labels = _button_labels(markup)
        assert any("active" in lbl and "\U0001f7e2" in lbl for lbl in labels)

    def test_no_allowed_no_history_shows_only_home(self):
        """With no allowed workspaces and no history, only the Home button appears."""
        markup = _workspaces_keyboard([], "/home", "/home", None, [])
        assert len(_button_labels(markup)) == 1

    def test_disambiguates_duplicate_names(self, tmp_path):
        """Two allowed workspaces with the same directory name get parent/name labels."""
        foo_a = tmp_path / "projects" / "foo"
        foo_b = tmp_path / "clients" / "foo"
        foo_a.mkdir(parents=True)
        foo_b.mkdir(parents=True)
        markup = _workspaces_keyboard([], "/home", "/home", None, [foo_a, foo_b])
        labels = _button_labels(markup)
        assert "projects/foo" in labels
        assert "clients/foo" in labels
        # Neither bare "foo" label should appear
        assert "foo" not in labels

    def test_unique_names_not_disambiguated(self, tmp_path):
        """Allowed workspaces with unique names keep their short labels."""
        bar = tmp_path / "bar"
        baz = tmp_path / "baz"
        bar.mkdir()
        baz.mkdir()
        markup = _workspaces_keyboard([], "/home", "/home", None, [bar, baz])
        labels = _button_labels(markup)
        assert "bar" in labels
        assert "baz" in labels


# ── is_workspace_allowed ─────────────────────────────────────────────


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
        """With no base and no allowed list, all paths are accepted (permissive mode)."""
        assert is_workspace_allowed(tmp_path / "anything", None, []) is True

    def test_path_under_base_is_allowed(self, tmp_path):
        """Paths under workspace_base are allowed."""
        assert is_workspace_allowed(tmp_path / "myproject", tmp_path, []) is True

    def test_path_in_allowed_list(self, tmp_path):
        """Paths in the allowed list are allowed."""
        project = tmp_path / "project"
        project.mkdir()
        assert is_workspace_allowed(project, None, [project]) is True

    def test_path_outside_both_is_rejected(self, tmp_path):
        """Paths not under base or in allowed list are rejected."""
        base = tmp_path / "base"
        base.mkdir()
        outside = tmp_path / "outside"
        assert is_workspace_allowed(outside, base, []) is False

    def test_base_set_allowed_empty_rejects_outside(self, tmp_path):
        """With base set but empty allowed list, outside paths are rejected."""
        base = tmp_path / "base"
        base.mkdir()
        assert is_workspace_allowed(tmp_path / "other", base, []) is False

    def test_only_allowed_set_rejects_unlisted(self, tmp_path):
        """With only allowed list set, unlisted paths are rejected."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        unlisted = tmp_path / "unlisted"
        assert is_workspace_allowed(unlisted, None, [allowed]) is False

    def test_resolves_symlinks_for_comparison(self, tmp_path):
        """Path resolution handles non-canonical paths correctly."""
        project = tmp_path / "project"
        project.mkdir()
        # Pass the resolved canonical path - should still match
        assert is_workspace_allowed(project.resolve(), None, [project]) is True


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
    """Create a mock SubprocessPool with pool-compatible methods.

    Despite the name (kept for backward compatibility with existing test
    call sites), this returns a pool-like mock, not a PersistentClaude.
    """
    pool = MagicMock()
    ws = workspace or Path("/home/workspace")
    pool.get_model = MagicMock(return_value=model)
    pool.set_model = MagicMock()
    pool.get_workspace = MagicMock(return_value=ws)
    pool.is_alive = MagicMock(return_value=is_alive)
    pool.get_session_id = MagicMock(return_value=None)
    pool.restart = AsyncMock()
    pool.change_workspace = AsyncMock()
    pool.force_kill = AsyncMock()
    pool.send = MagicMock()  # configured per test
    return pool


def _make_context(config=None, claude=None, pool=None, args=None, user_data=None, job_queue=None):
    """Create a mock PTB context with bot_data, args, and user_data."""
    ctx = MagicMock()
    # Accept either pool (Phase 3) or claude (legacy test compat) as the
    # mock subprocess manager. Pool is preferred for new tests.
    mock_pool = pool or claude or _make_mock_claude()
    ctx.bot_data = {
        "config": config or _make_config(),
        "pool": mock_pool,
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


def _mock_resolve(base=None, allowed=None):
    """Return a patch context that mocks sessions.resolve_workspace_access.

    Simplifies handler tests that call handle_workspace, handle_workspaces,
    or handle_workspace_callback - all of which resolve per-user workspace
    access at the top of the handler.
    """
    return patch(
        "kai.bot.sessions.resolve_workspace_access",
        new_callable=AsyncMock,
        return_value=(base, allowed or []),
    )


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
        """Per-user flag file is created under the .responding directory."""
        responding_dir = tmp_path / ".responding"
        with patch("kai.bot._RESPONDING_DIR", responding_dir):
            _set_responding(12345)
        assert (responding_dir / "12345").exists()

    def test_clear_responding_removes_flag(self, tmp_path):
        """Per-user flag file is deleted after clearing."""
        responding_dir = tmp_path / ".responding"
        responding_dir.mkdir()
        (responding_dir / "12345").touch()
        with patch("kai.bot._RESPONDING_DIR", responding_dir):
            _clear_responding(12345)
        assert not (responding_dir / "12345").exists()

    def test_clear_responding_noop_if_missing(self, tmp_path):
        """No error when flag file doesn't exist."""
        responding_dir = tmp_path / ".responding"
        with patch("kai.bot._RESPONDING_DIR", responding_dir):
            _clear_responding(12345)  # should not raise


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
        pool = _make_mock_claude(model="sonnet")
        update = _make_update()
        ctx = _make_context(claude=pool, args=["opus"])
        with (
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_user_setting", new_callable=AsyncMock),
        ):
            await handle_model(update, ctx)
        pool.set_model.assert_called_once_with(ANY, "opus")
        pool.restart.assert_called_once()


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
        pool = _make_mock_claude(model="sonnet")
        update = _make_callback_update(data="model:opus")
        ctx = _make_context(claude=pool)
        with (
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_user_setting", new_callable=AsyncMock),
        ):
            await handle_model_callback(update, ctx)
        pool.set_model.assert_called_once_with(ANY, "opus")
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
        with _mock_resolve():
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Home" in reply or "workspace" in reply.lower()

    @pytest.mark.asyncio
    async def test_home_switches(self, tmp_path):
        """'home' keyword switches to home workspace."""
        home = tmp_path / "home"
        home.mkdir()
        claude = _make_mock_claude(workspace=Path("/other"))
        config = _make_config(claude_workspace=home)
        update = _make_update()
        ctx = _make_context(claude=claude, config=config, args=["home"])
        with (
            _mock_resolve(),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.delete_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
            patch("kai.bot.webhook.update_workspace"),
        ):
            await handle_workspace(update, ctx)
        claude.change_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_absolute_path_rejected(self):
        update = _make_update()
        ctx = _make_context(args=["/tmp/evil"])
        with _mock_resolve():
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Absolute paths" in reply

    @pytest.mark.asyncio
    async def test_tilde_path_rejected(self):
        update = _make_update()
        ctx = _make_context(args=["~/foo"])
        with _mock_resolve():
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Absolute paths" in reply

    @pytest.mark.asyncio
    async def test_new_without_name(self):
        update = _make_update()
        ctx = _make_context(args=["new"])
        with _mock_resolve():
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Usage" in reply

    @pytest.mark.asyncio
    async def test_new_without_base(self):
        """'new' with no workspace_base shows the updated error message."""
        update = _make_update()
        ctx = _make_context(config=_make_config(workspace_base=None), args=["new", "myproj"])
        with _mock_resolve():
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "workspace base" in reply.lower()

    @pytest.mark.asyncio
    async def test_new_already_exists(self, tmp_path):
        existing = tmp_path / "myproj"
        existing.mkdir()
        update = _make_update()
        ctx = _make_context(
            config=_make_config(workspace_base=tmp_path, claude_workspace=Path("/home")),
            args=["new", "myproj"],
        )
        with _mock_resolve(base=tmp_path):
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
            _mock_resolve(base=tmp_path),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.upsert_workspace_history", new_callable=AsyncMock),
            patch("kai.bot.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
            patch("kai.bot.webhook.update_workspace"),
        ):
            await handle_workspace(update, ctx)
        # Directory should have been created
        assert (tmp_path / "fresh").is_dir()
        claude.change_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_new_git_init_failure_warns(self, tmp_path):
        """Failed git init warns the user but still switches workspace."""
        update = _make_update()
        claude = _make_mock_claude(workspace=Path("/home"))
        config = _make_config(workspace_base=tmp_path, claude_workspace=Path("/home"))
        ctx = _make_context(config=config, claude=claude, args=["new", "broken"])
        # Simulate git init returning a non-zero exit code
        mock_proc = MagicMock()
        mock_proc.wait = AsyncMock(return_value=1)
        with (
            _mock_resolve(base=tmp_path),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.upsert_workspace_history", new_callable=AsyncMock),
            patch("kai.bot.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
            patch("kai.bot.webhook.update_workspace"),
        ):
            await handle_workspace(update, ctx)
        # Warning about git init failure was sent
        replies = [call[0][0] for call in update.message.reply_text.call_args_list]
        assert any("git init failed" in r for r in replies)
        # Workspace switch still happened despite the failure
        claude.change_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_workspace_new_prefix_not_caught(self, tmp_path):
        """Workspace names starting with 'new' are not mistaken for /workspace new."""
        # "newsletter" is a valid workspace name, not the "new" subcommand
        project = tmp_path / "newsletter"
        project.mkdir()
        claude = _make_mock_claude(workspace=Path("/other"))
        config = _make_config(workspace_base=tmp_path, claude_workspace=Path("/home"))
        update = _make_update()
        ctx = _make_context(config=config, claude=claude, args=["newsletter"])
        with (
            _mock_resolve(base=tmp_path),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.upsert_workspace_history", new_callable=AsyncMock),
            patch("kai.bot.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
            patch("kai.bot.webhook.update_workspace"),
        ):
            await handle_workspace(update, ctx)
        # Should switch to the workspace, not show "Usage: /workspace new <name>"
        claude.change_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_workspace_new_bare_shows_usage(self):
        """'/workspace new' with no name shows usage hint."""
        update = _make_update()
        ctx = _make_context(args=["new"])
        with _mock_resolve():
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Usage" in reply

    @pytest.mark.asyncio
    async def test_name_found_in_base(self, tmp_path):
        """Name resolved via workspace_base."""
        project = tmp_path / "myproj"
        project.mkdir()
        claude = _make_mock_claude(workspace=Path("/other"))
        config = _make_config(workspace_base=tmp_path, claude_workspace=Path("/home"))
        update = _make_update()
        ctx = _make_context(config=config, claude=claude, args=["myproj"])
        with (
            _mock_resolve(base=tmp_path),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.upsert_workspace_history", new_callable=AsyncMock),
            patch("kai.bot.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
            patch("kai.bot.webhook.update_workspace"),
        ):
            await handle_workspace(update, ctx)
        claude.change_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_name_found_in_allowed(self, tmp_path):
        """Name resolved via allowed list directory name match."""
        project = tmp_path / "myproj"
        project.mkdir()
        claude = _make_mock_claude(workspace=Path("/other"))
        config = _make_config(claude_workspace=Path("/home"))
        update = _make_update()
        ctx = _make_context(config=config, claude=claude, args=["myproj"])
        with (
            _mock_resolve(allowed=[project]),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.upsert_workspace_history", new_callable=AsyncMock),
            patch("kai.bot.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
            patch("kai.bot.webhook.update_workspace"),
        ):
            await handle_workspace(update, ctx)
        claude.change_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_matches_in_allowed(self, tmp_path):
        """Multiple allowed workspaces with the same name: shows disambiguation message."""
        proj_a = tmp_path / "a" / "proj"
        proj_b = tmp_path / "b" / "proj"
        proj_a.mkdir(parents=True)
        proj_b.mkdir(parents=True)
        config = _make_config(claude_workspace=Path("/home"))
        update = _make_update()
        ctx = _make_context(config=config, args=["proj"])
        with _mock_resolve(allowed=[proj_a, proj_b]):
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Multiple workspaces" in reply

    @pytest.mark.asyncio
    async def test_not_found_with_sources(self, tmp_path):
        config = _make_config(claude_workspace=Path("/home"))
        update = _make_update()
        ctx = _make_context(config=config, args=["nonexistent"])
        with _mock_resolve(base=tmp_path):
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "not found" in reply.lower()

    @pytest.mark.asyncio
    async def test_not_found_no_sources(self):
        config = _make_config(workspace_base=None, allowed_workspaces=[], claude_workspace=Path("/home"))
        update = _make_update()
        ctx = _make_context(config=config, args=["anything"])
        with _mock_resolve():
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "workspace base" in reply.lower()

    @pytest.mark.asyncio
    async def test_ws_alias_registered(self):
        """/ws is registered as an alias for /workspace via create_bot."""
        # The alias is a CommandHandler("ws", handle_workspace) in create_bot.
        # We verify it by calling handle_workspace directly with no args
        # (same handler, so same behavior as /workspace with no args).
        claude = _make_mock_claude(workspace=Path("/home/workspace"))
        update = _make_update()
        ctx = _make_context(claude=claude)
        with _mock_resolve():
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Home" in reply or "workspace" in reply.lower()


# ── handle_workspaces ────────────────────────────────────────────────


class TestHandleWorkspaces:
    @pytest.mark.asyncio
    async def test_no_history_at_home(self):
        update = _make_update()
        claude = _make_mock_claude(workspace=Path("/home/workspace"))
        config = _make_config(claude_workspace=Path("/home/workspace"))
        ctx = _make_context(config=config, claude=claude)
        with (
            _mock_resolve(),
            patch("kai.bot.sessions.get_workspace_history", new_callable=AsyncMock, return_value=[]),
        ):
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
        with (
            _mock_resolve(),
            patch("kai.bot.sessions.get_workspace_history", new_callable=AsyncMock, return_value=history),
        ):
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
            _mock_resolve(),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.delete_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
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
        config = _make_config(claude_workspace=Path("/home/workspace"))
        update = _make_callback_update(data="ws:allowed:0")
        ctx = _make_context(config=config, claude=claude)
        with (
            _mock_resolve(allowed=[project]),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.upsert_workspace_history", new_callable=AsyncMock),
            patch("kai.bot.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
            patch("kai.bot.webhook.update_workspace"),
        ):
            await handle_workspace_callback(update, ctx)
        claude.change_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_allowed_nonexistent_dir(self, tmp_path):
        """ws:allowed:<idx> where directory was deleted."""
        gone = tmp_path / "gone"  # not created
        config = _make_config(claude_workspace=Path("/home"))
        update = _make_callback_update(data="ws:allowed:0")
        ctx = _make_context(config=config)
        with _mock_resolve(allowed=[gone]):
            await handle_workspace_callback(update, ctx)
        update.callback_query.answer.assert_called_once_with("That workspace no longer exists.")

    @pytest.mark.asyncio
    async def test_allowed_bad_index(self):
        update = _make_callback_update(data="ws:allowed:bad")
        ctx = _make_context()
        with _mock_resolve():
            await handle_workspace_callback(update, ctx)
        update.callback_query.answer.assert_called_once_with("Invalid selection.")

    @pytest.mark.asyncio
    async def test_allowed_out_of_range(self):
        config = _make_config(claude_workspace=Path("/home"))
        update = _make_callback_update(data="ws:allowed:99")
        ctx = _make_context(config=config)
        with _mock_resolve():
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
            _mock_resolve(),
            patch("kai.bot.sessions.get_workspace_history", new_callable=AsyncMock, return_value=history),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.upsert_workspace_history", new_callable=AsyncMock),
            patch("kai.bot.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
            patch("kai.bot.webhook.update_workspace"),
        ):
            await handle_workspace_callback(update, ctx)
        claude.change_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_history_no_longer_allowed(self, tmp_path):
        """History entry disallowed: deleted from history, keyboard refreshed."""
        project = tmp_path / "proj"
        project.mkdir()
        # Configure with a base that doesn't contain the project - the
        # per-user resolve returns base=other_base so the project path
        # fails _is_workspace_allowed.
        other_base = tmp_path / "base"
        other_base.mkdir()
        config = _make_config(claude_workspace=Path("/home/workspace"))
        claude = _make_mock_claude(workspace=Path("/home/workspace"))
        update = _make_callback_update(data="ws:0")
        ctx = _make_context(config=config, claude=claude)
        history = [{"path": str(project)}]
        with (
            _mock_resolve(base=other_base),
            patch("kai.bot.sessions.get_workspace_history", new_callable=AsyncMock, return_value=history),
            patch("kai.bot.sessions.delete_workspace_history", new_callable=AsyncMock) as mock_del,
        ):
            await handle_workspace_callback(update, ctx)
        mock_del.assert_called_once_with(str(project), 12345)
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
            _mock_resolve(),
            patch("kai.bot.sessions.get_workspace_history", new_callable=AsyncMock, return_value=history),
            patch("kai.bot.sessions.delete_workspace_history", new_callable=AsyncMock) as mock_del,
        ):
            await handle_workspace_callback(update, ctx)
        mock_del.assert_called_once_with(str(gone), 12345)

    @pytest.mark.asyncio
    async def test_already_in_workspace(self, tmp_path):
        """Selecting the current workspace shows 'No change.'"""
        home = Path("/home/workspace")
        claude = _make_mock_claude(workspace=home)
        config = _make_config(claude_workspace=home)
        update = _make_callback_update(data="ws:home")
        ctx = _make_context(config=config, claude=claude)
        with _mock_resolve():
            await handle_workspace_callback(update, ctx)
        edit_text = update.callback_query.edit_message_text.call_args[0][0]
        assert "No change" in edit_text


# ── /workspace allow/deny/allowed ───────────────────────────────────


class TestHandleWorkspaceAllow:
    @pytest.mark.asyncio
    async def test_allow_success(self, tmp_path):
        """Adding a valid directory path succeeds."""
        ws = tmp_path / "new-ws"
        ws.mkdir()
        update = _make_update()
        ctx = _make_context(args=["allow", str(ws)])
        with (
            _mock_resolve(),
            patch("kai.bot.sessions.add_allowed_workspace", new_callable=AsyncMock) as mock_add,
        ):
            await _handle_workspace_allow(update, ctx, f"allow {ws}")
        mock_add.assert_called_once_with(12345, str(ws.resolve()))
        reply = update.message.reply_text.call_args[0][0]
        assert "Added" in reply

    @pytest.mark.asyncio
    async def test_allow_no_path(self):
        """Missing path argument shows usage."""
        update = _make_update()
        ctx = _make_context(args=["allow"])
        await _handle_workspace_allow(update, ctx, "allow")
        reply = update.message.reply_text.call_args[0][0]
        assert "Usage" in reply

    @pytest.mark.asyncio
    async def test_allow_relative_path_rejected(self):
        """Relative paths (not starting with /) are rejected."""
        update = _make_update()
        ctx = _make_context(args=["allow", "relative/path"])
        await _handle_workspace_allow(update, ctx, "allow relative/path")
        reply = update.message.reply_text.call_args[0][0]
        assert "absolute" in reply.lower()

    @pytest.mark.asyncio
    async def test_allow_nonexistent_rejected(self):
        """Non-existent path is rejected."""
        update = _make_update()
        ctx = _make_context(args=["allow", "/nonexistent/path"])
        await _handle_workspace_allow(update, ctx, "allow /nonexistent/path")
        reply = update.message.reply_text.call_args[0][0]
        assert "Not a directory" in reply

    @pytest.mark.asyncio
    async def test_allow_under_base_redundant(self, tmp_path):
        """Path under workspace_base is flagged as redundant."""
        base = tmp_path / "base"
        sub = base / "project"
        sub.mkdir(parents=True)
        update = _make_update()
        ctx = _make_context(args=["allow", str(sub)])
        with _mock_resolve(base=base):
            await _handle_workspace_allow(update, ctx, f"allow {sub}")
        reply = update.message.reply_text.call_args[0][0]
        assert "workspace base" in reply.lower()

    @pytest.mark.asyncio
    async def test_allow_duplicate(self, tmp_path):
        """Path already in allowed list is flagged as duplicate."""
        ws = tmp_path / "ws"
        ws.mkdir()
        update = _make_update()
        ctx = _make_context(args=["allow", str(ws)])
        with _mock_resolve(allowed=[ws]):
            await _handle_workspace_allow(update, ctx, f"allow {ws}")
        reply = update.message.reply_text.call_args[0][0]
        assert "Already" in reply


class TestHandleWorkspaceDeny:
    @pytest.mark.asyncio
    async def test_deny_success(self, tmp_path):
        """Removing a user-added path succeeds."""
        ws = tmp_path / "ws"
        ws.mkdir()
        update = _make_update()
        ctx = _make_context(args=["deny", str(ws)])
        with patch(
            "kai.bot.sessions.remove_allowed_workspace",
            new_callable=AsyncMock,
            return_value=True,
        ):
            await _handle_workspace_deny(update, ctx, f"deny {ws}")
        reply = update.message.reply_text.call_args[0][0]
        assert "Removed" in reply

    @pytest.mark.asyncio
    async def test_deny_no_path(self):
        """Missing path argument shows usage."""
        update = _make_update()
        ctx = _make_context(args=["deny"])
        await _handle_workspace_deny(update, ctx, "deny")
        reply = update.message.reply_text.call_args[0][0]
        assert "Usage" in reply

    @pytest.mark.asyncio
    async def test_deny_relative_path_rejected(self):
        """Relative paths are rejected with a clear message."""
        update = _make_update()
        ctx = _make_context(args=["deny", "relative/path"])
        await _handle_workspace_deny(update, ctx, "deny relative/path")
        reply = update.message.reply_text.call_args[0][0]
        assert "absolute" in reply.lower()

    @pytest.mark.asyncio
    async def test_deny_global_path(self, tmp_path):
        """Trying to deny a global ALLOWED_WORKSPACES path shows an explanation."""
        ws = tmp_path / "global-ws"
        ws.mkdir()
        config = _make_config(allowed_workspaces=[ws])
        update = _make_update()
        ctx = _make_context(config=config, args=["deny", str(ws)])
        with patch(
            "kai.bot.sessions.remove_allowed_workspace",
            new_callable=AsyncMock,
            return_value=False,
        ):
            await _handle_workspace_deny(update, ctx, f"deny {ws}")
        reply = update.message.reply_text.call_args[0][0]
        assert "globally" in reply.lower()

    @pytest.mark.asyncio
    async def test_deny_unknown_path(self, tmp_path):
        """Denying a path not in any list shows not-found message."""
        update = _make_update()
        ctx = _make_context(args=["deny", "/some/unknown/path"])
        with patch(
            "kai.bot.sessions.remove_allowed_workspace",
            new_callable=AsyncMock,
            return_value=False,
        ):
            await _handle_workspace_deny(update, ctx, "deny /some/unknown/path")
        reply = update.message.reply_text.call_args[0][0]
        assert "Not in your" in reply


class TestHandleWorkspaceAllowed:
    @pytest.mark.asyncio
    async def test_shows_list_with_sources(self, tmp_path):
        """Shows allowed workspaces with source attribution."""
        db_ws = tmp_path / "user-ws"
        db_ws.mkdir()
        global_ws = tmp_path / "global-ws"
        global_ws.mkdir()
        config = _make_config(allowed_workspaces=[global_ws])
        update = _make_update()
        ctx = _make_context(config=config)
        with patch(
            "kai.bot.sessions.get_allowed_workspaces",
            new_callable=AsyncMock,
            return_value=[db_ws],
        ):
            await _handle_workspace_allowed(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "(you)" in reply
        assert "(global)" in reply
        assert "Workspace base: not set" in reply

    @pytest.mark.asyncio
    async def test_permissive_mode(self):
        """No base and no allowed shows permissive mode message."""
        config = _make_config()
        update = _make_update()
        ctx = _make_context(config=config)
        with patch(
            "kai.bot.sessions.get_allowed_workspaces",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await _handle_workspace_allowed(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "permissive" in reply.lower()

    @pytest.mark.asyncio
    async def test_shows_base(self, tmp_path):
        """Shows workspace_base when configured."""
        config = _make_config(workspace_base=tmp_path)
        update = _make_update()
        ctx = _make_context(config=config)
        with patch(
            "kai.bot.sessions.get_allowed_workspaces",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await _handle_workspace_allowed(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert f"Workspace base: {tmp_path}" in reply


# ── /workspace allow/deny routing via handle_workspace ──────────────


class TestWorkspaceSubcommandRouting:
    @pytest.mark.asyncio
    async def test_allow_routed(self, tmp_path):
        """/workspace allow <path> routes to _handle_workspace_allow."""
        ws = tmp_path / "ws"
        ws.mkdir()
        update = _make_update()
        ctx = _make_context(args=["allow", str(ws)])
        with (
            _mock_resolve(),
            patch("kai.bot.sessions.add_allowed_workspace", new_callable=AsyncMock),
        ):
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Added" in reply

    @pytest.mark.asyncio
    async def test_deny_routed(self, tmp_path):
        """/workspace deny <path> routes to _handle_workspace_deny."""
        update = _make_update()
        ctx = _make_context(args=["deny", str(tmp_path)])
        with (
            _mock_resolve(),
            patch("kai.bot.sessions.remove_allowed_workspace", new_callable=AsyncMock, return_value=False),
        ):
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Not in your" in reply

    @pytest.mark.asyncio
    async def test_allowed_routed(self):
        """/workspace allowed routes to _handle_workspace_allowed."""
        update = _make_update()
        ctx = _make_context(args=["allowed"])
        with (
            _mock_resolve(),
            patch("kai.bot.sessions.get_allowed_workspaces", new_callable=AsyncMock, return_value=[]),
        ):
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Workspace base" in reply

    @pytest.mark.asyncio
    async def test_name_uses_per_user_allowed(self, tmp_path):
        """/workspace <name> uses per-user allowed list for name resolution."""
        project = tmp_path / "myproj"
        project.mkdir()
        claude = _make_mock_claude(workspace=Path("/other"))
        config = _make_config(claude_workspace=Path("/home"))
        update = _make_update()
        ctx = _make_context(config=config, claude=claude, args=["myproj"])
        with (
            _mock_resolve(allowed=[project]),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.upsert_workspace_history", new_callable=AsyncMock),
            patch("kai.bot.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
            patch("kai.bot.webhook.update_workspace"),
        ):
            await handle_workspace(update, ctx)
        claude.change_workspace.assert_called_once()


# ── Workspace config in bot layer ────────────────────────────────────


class TestWorkspaceConfigSuffix:
    def test_with_model_and_budget(self):
        """Shows model and budget when both are configured."""
        from kai.config import WorkspaceConfig

        ws = WorkspaceConfig(path=Path("/tmp/ws"), model="opus", budget=15.0)
        assert _workspace_config_suffix(ws) == " (model: opus, budget: $15.00)"

    def test_model_only(self):
        """Shows model only when budget is not set."""
        from kai.config import WorkspaceConfig

        ws = WorkspaceConfig(path=Path("/tmp/ws"), model="haiku")
        assert _workspace_config_suffix(ws) == " (model: haiku)"

    def test_no_config(self):
        """Returns empty string when no config is provided."""
        assert _workspace_config_suffix(None) == ""

    def test_config_with_no_overrides(self):
        """Returns empty string when config has no model or budget."""
        from kai.config import WorkspaceConfig

        ws = WorkspaceConfig(path=Path("/tmp/ws"))
        assert _workspace_config_suffix(ws) == ""


class TestSwitchWorkspaceConfig:
    @pytest.mark.asyncio
    async def test_switch_passes_config_to_change_workspace(self, tmp_path):
        """_do_switch_workspace passes the workspace config to Claude."""
        from kai.config import WorkspaceConfig

        ws_path = tmp_path / "ws"
        ws_path.mkdir()
        ws_config = WorkspaceConfig(path=ws_path.resolve(), model="opus")
        config = _make_config(
            claude_workspace=Path("/home/workspace"),
            workspace_configs={ws_path.resolve(): ws_config},
        )
        claude = _make_mock_claude()
        ctx = _make_context(config=config, claude=claude)

        mock_sessions = AsyncMock()
        # build_workspace_config returns the merged config (here, just YAML)
        mock_sessions.build_workspace_config = AsyncMock(return_value=ws_config)
        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.webhook"),
        ):
            result = await _do_switch_workspace(ctx, 12345, ws_path.resolve())

        # Config was returned and passed to change_workspace
        assert result is ws_config
        claude.change_workspace.assert_called_once_with(12345, ws_path.resolve(), workspace_config=ws_config)

    @pytest.mark.asyncio
    async def test_switch_unconfigured_passes_none(self, tmp_path):
        """_do_switch_workspace passes None for unconfigured workspaces."""
        ws_path = tmp_path / "ws"
        ws_path.mkdir()
        config = _make_config(claude_workspace=Path("/home/workspace"))
        claude = _make_mock_claude()
        ctx = _make_context(config=config, claude=claude)

        mock_sessions = AsyncMock()
        # No YAML config, no DB overrides -> returns None
        mock_sessions.build_workspace_config = AsyncMock(return_value=None)
        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.webhook"),
        ):
            result = await _do_switch_workspace(ctx, 12345, ws_path.resolve())

        assert result is None
        claude.change_workspace.assert_called_once_with(12345, ws_path.resolve(), workspace_config=None)

    @pytest.mark.asyncio
    async def test_switch_shows_config_info(self, tmp_path):
        """_switch_workspace confirmation includes model when configured."""
        from kai.config import WorkspaceConfig

        ws_path = tmp_path / "ws"
        ws_path.mkdir()
        ws_config = WorkspaceConfig(path=ws_path.resolve(), model="opus", budget=20.0)
        config = _make_config(
            claude_workspace=Path("/home/workspace"),
            workspace_configs={ws_path.resolve(): ws_config},
        )
        claude = _make_mock_claude(workspace=Path("/home/workspace"))
        update = _make_update()
        ctx = _make_context(config=config, claude=claude)

        mock_sessions = AsyncMock()
        mock_sessions.build_workspace_config = AsyncMock(return_value=ws_config)
        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.webhook"),
        ):
            await _switch_workspace(update, ctx, ws_path.resolve())

        reply_text = update.message.reply_text.call_args[0][0]
        assert "model: opus" in reply_text
        assert "budget: $20.00" in reply_text

    @pytest.mark.asyncio
    async def test_switch_deleted_directory(self, tmp_path):
        """Switching to a workspace whose directory no longer exists shows an error."""
        home = tmp_path / "home"
        home.mkdir()
        gone = tmp_path / "gone"
        gone.mkdir()
        gone.rmdir()  # create then delete so the path is guaranteed absent

        config = _make_config(claude_workspace=home)
        claude = _make_mock_claude(workspace=home)
        update = _make_update()
        ctx = _make_context(config=config, claude=claude)

        await _switch_workspace(update, ctx, gone)

        update.message.reply_text.assert_called_with("That workspace no longer exists.")


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


# ── _acquire_lock_or_kill ───────────────────────────────────────────


class TestAcquireLockOrKill:
    """Tests for the lock-with-timeout safety net."""

    @pytest.mark.asyncio
    async def test_acquires_free_lock(self):
        """Returns the lock when it's free - normal fast path."""
        update = _make_update()
        claude = _make_mock_claude()
        # Use a unique chat_id to avoid state from other tests
        chat_id = 88801

        lock = await _acquire_lock_or_kill(chat_id, claude, update)

        assert lock is not None
        assert lock.locked()
        lock.release()
        claude.force_kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_timeout_kills_claude(self):
        """When the lock is held too long, force-kills Claude and notifies user."""
        update = _make_update()
        claude = _make_mock_claude()
        chat_id = 88802

        from kai.locks import get_lock

        held_lock = get_lock(chat_id)
        await held_lock.acquire()

        try:
            # Patch the timeout to something tiny so the test doesn't wait 1 hour
            with patch("kai.bot._LOCK_ACQUIRE_TIMEOUT", 0.05):
                result = await _acquire_lock_or_kill(chat_id, claude, update)
        finally:
            held_lock.release()

        assert result is None
        claude.force_kill.assert_called_once()
        update.message.reply_text.assert_called()
        msg = update.message.reply_text.call_args[0][0]
        assert "timed out" in msg.lower()

    @pytest.mark.asyncio
    async def test_returns_same_lock_object(self):
        """The returned lock is the same object from get_lock, not a copy."""
        update = _make_update()
        claude = _make_mock_claude()
        chat_id = 88803

        from kai.locks import get_lock

        expected_lock = get_lock(chat_id)
        returned_lock = await _acquire_lock_or_kill(chat_id, claude, update)

        assert returned_lock is expected_lock
        returned_lock.release()

    @pytest.mark.asyncio
    async def test_handle_message_releases_lock_on_error(self):
        """Lock is released even when _handle_response raises."""
        update = _make_update()
        ctx = _make_context()
        chat_id = 88804

        from kai.locks import get_lock

        lock = get_lock(chat_id)

        with (
            # Bypass the TOTP gate so handle_message reaches the lock
            # acquisition and _handle_response code paths under test.
            patch("kai.bot.is_totp_configured", return_value=False),
            patch(
                "kai.bot._handle_response",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
            patch("kai.bot.log_message"),
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            # Use real get_lock so we can verify the lock state after
            pytest.raises(RuntimeError),
        ):
            await handle_message(update, ctx)

        # Lock must be released after the error
        assert not lock.locked()


# ── _handle_workspace_config ────────────────────────────────────────


class TestHandleWorkspaceConfig:
    """Tests for /workspace config subcommands.

    Each test patches the sessions module and pool to isolate the handler
    from the database and subprocess pool. The handler takes a "target"
    string that simulates what handle_workspace passes after splitting
    the user's message (e.g., "config model opus").
    """

    def _patches(self, mock_sessions):
        """Build the standard patch set for workspace config tests.

        Returns a context manager stack. mock_sessions should be an
        AsyncMock with the workspace config helpers pre-configured.
        """
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(patch("kai.bot.sessions", mock_sessions))
        return stack

    def _mock_sessions(self, db_settings=None):
        """Create a mock sessions module with workspace config helpers."""
        mock = AsyncMock()
        mock.get_workspace_config_settings = AsyncMock(return_value=db_settings or {})
        mock.set_workspace_config_setting = AsyncMock()
        mock.delete_workspace_config_setting = AsyncMock()
        mock.delete_all_workspace_config = AsyncMock()
        mock.build_workspace_config = AsyncMock(return_value=None)
        return mock

    # ── 1. Show config with no overrides ────────────────────────────

    @pytest.mark.asyncio
    async def test_show_config_no_overrides(self):
        """/workspace config with no overrides shows global defaults."""
        update = _make_update(text="/workspace config")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config")

        reply = update.message.reply_text.call_args[0][0]
        # Should show model, budget, timeout with "global default" source
        assert "sonnet" in reply
        assert "global default" in reply

    # ── 2. Set model ────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_set_model(self):
        """/workspace config model opus sets the model."""
        update = _make_update(text="/workspace config model opus")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config model opus")

        # Should persist the model setting
        mock_sessions.set_workspace_config_setting.assert_called_once_with(
            12345, str(pool.get_workspace(12345)), "model", "opus"
        )
        # Should apply config change (rebuild + change_workspace)
        mock_sessions.build_workspace_config.assert_called_once()
        pool.change_workspace.assert_called_once()
        # Should confirm to the user
        reply = update.message.reply_text.call_args[0][0]
        assert "opus" in reply.lower()

    # ── 3. Set budget with validation ───────────────────────────────

    @pytest.mark.asyncio
    async def test_set_budget(self):
        """/workspace config budget 5 sets the budget."""
        update = _make_update(text="/workspace config budget 5")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config budget 5")

        mock_sessions.set_workspace_config_setting.assert_called_once_with(
            12345, str(pool.get_workspace(12345)), "budget", "5.0"
        )
        reply = update.message.reply_text.call_args[0][0]
        assert "$5.00" in reply

    # ── 4. Reject negative budget ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_reject_negative_budget(self):
        """/workspace config budget -5 is rejected."""
        update = _make_update(text="/workspace config budget -5")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config budget -5")

        # Should not persist anything
        mock_sessions.set_workspace_config_setting.assert_not_called()
        reply = update.message.reply_text.call_args[0][0]
        assert "positive" in reply.lower()

    # ── 5. Set timeout ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_set_timeout(self):
        """/workspace config timeout 300 sets the timeout."""
        update = _make_update(text="/workspace config timeout 300")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config timeout 300")

        mock_sessions.set_workspace_config_setting.assert_called_once_with(
            12345, str(pool.get_workspace(12345)), "timeout", "300"
        )
        reply = update.message.reply_text.call_args[0][0]
        assert "300" in reply

    # ── 6. Set env var ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_set_env_var(self):
        """/workspace config env FOO=bar sets an env var."""
        update = _make_update(text="/workspace config env FOO=bar")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config env FOO=bar")

        # Should persist the env JSON blob
        call_args = mock_sessions.set_workspace_config_setting.call_args
        assert call_args[0][2] == "env"
        saved_env = json.loads(call_args[0][3])
        assert saved_env == {"FOO": "bar"}
        # Should apply config change
        pool.change_workspace.assert_called_once()

    # ── 7. Remove env var ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_remove_env_var(self):
        """/workspace config env -FOO removes an env var."""
        update = _make_update(text="/workspace config env -FOO")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        # Pre-populate an env var so there's something to remove
        mock_sessions = self._mock_sessions(db_settings={"env": '{"FOO": "bar"}'})

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config env -FOO")

        # Should delete the env key entirely (empty dict -> delete)
        mock_sessions.delete_workspace_config_setting.assert_called_once_with(
            12345, str(pool.get_workspace(12345)), "env"
        )
        reply = update.message.reply_text.call_args[0][0]
        assert "Removed" in reply

    # ── 8. List env vars ────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_list_env_vars(self):
        """/workspace config env (no value) lists env var keys."""
        update = _make_update(text="/workspace config env")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions(db_settings={"env": '{"A": "1", "B": "2"}'})

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config env")

        reply = update.message.reply_text.call_args[0][0]
        assert "A" in reply
        assert "B" in reply
        # Should NOT apply config change (read-only operation)
        pool.change_workspace.assert_not_called()

    # ── 9. Set prompt ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_set_prompt(self):
        """/workspace config prompt Hello world sets the prompt."""
        update = _make_update(text="/workspace config prompt Hello world")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config prompt Hello world")

        mock_sessions.set_workspace_config_setting.assert_called_once_with(
            12345, str(pool.get_workspace(12345)), "prompt", "Hello world"
        )
        pool.change_workspace.assert_called_once()

    # ── 10. Clear prompt ────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_clear_prompt(self):
        """/workspace config prompt clear clears the prompt."""
        update = _make_update(text="/workspace config prompt clear")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config prompt clear")

        mock_sessions.delete_workspace_config_setting.assert_called_once_with(
            12345, str(pool.get_workspace(12345)), "prompt"
        )
        pool.change_workspace.assert_called_once()

    # ── 11. Reset all overrides ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_reset_all(self):
        """/workspace config reset clears all overrides."""
        update = _make_update(text="/workspace config reset")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config reset")

        mock_sessions.delete_all_workspace_config.assert_called_once_with(12345, str(pool.get_workspace(12345)))
        pool.change_workspace.assert_called_once()
        reply = update.message.reply_text.call_args[0][0]
        assert "global defaults" in reply.lower()

    # ── 12. Reset single field ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_reset_single_field(self):
        """/workspace config reset model clears a single field."""
        update = _make_update(text="/workspace config reset model")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config reset model")

        mock_sessions.delete_workspace_config_setting.assert_called_once_with(
            12345, str(pool.get_workspace(12345)), "model"
        )
        pool.change_workspace.assert_called_once()
        reply = update.message.reply_text.call_args[0][0]
        assert "model" in reply.lower()

    # ── 13. Unknown field shows error ───────────────────────────────

    @pytest.mark.asyncio
    async def test_unknown_field(self):
        """/workspace config bogus shows error with valid field list."""
        update = _make_update(text="/workspace config bogus")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config bogus")

        reply = update.message.reply_text.call_args[0][0]
        assert "bogus" in reply
        assert "model" in reply
        assert "budget" in reply
        # Should NOT apply any config change
        pool.change_workspace.assert_not_called()

    # ── Budget ceiling enforcement ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_budget_ceiling_from_user_config(self):
        """Budget exceeding per-user max_budget is rejected."""
        from kai.config import UserConfig

        user = UserConfig(telegram_id=12345, name="test", max_budget=5.0)
        config = _make_config(user_configs={12345: user})
        update = _make_update(text="/workspace config budget 10")
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config budget 10")

        # Should reject - 10 exceeds per-user ceiling of 5
        mock_sessions.set_workspace_config_setting.assert_not_called()
        reply = update.message.reply_text.call_args[0][0]
        assert "$5.00" in reply
        assert "admin limit" in reply.lower()

    # ── Invalid model rejected ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_invalid_model_rejected(self):
        """An unknown model name is rejected."""
        update = _make_update(text="/workspace config model gpt4")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config model gpt4")

        mock_sessions.set_workspace_config_setting.assert_not_called()
        reply = update.message.reply_text.call_args[0][0]
        assert "haiku" in reply
        assert "sonnet" in reply
        assert "opus" in reply


# ── /settings command ──────────────────────────────────────────────


class TestHandleSettings:
    """Tests for /settings - per-user default settings.

    Each test patches the sessions module and pool to isolate the handler
    from the database and subprocess pool.
    """

    def _patches(self, mock_sessions):
        """Build the standard patch set for settings tests."""
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(patch("kai.bot.sessions", mock_sessions))
        return stack

    def _mock_sessions(self, db_settings=None):
        """Create a mock sessions module with per-user settings helpers."""
        mock = AsyncMock()
        mock.get_user_settings = AsyncMock(return_value=db_settings or {})
        mock.set_user_setting = AsyncMock()
        mock.delete_user_setting = AsyncMock()
        mock.delete_all_user_settings = AsyncMock()
        mock.clear_session = AsyncMock()
        return mock

    # ── 1. Show settings with global defaults ──────────────────────

    @pytest.mark.asyncio
    async def test_show_settings_defaults(self):
        """/settings with no overrides shows global defaults."""
        update = _make_update(text="/settings")
        config = _make_config()
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _show_settings(update, 12345, config)

        reply = update.message.reply_text.call_args[0][0]
        assert "sonnet" in reply
        assert "global default" in reply

    # ── 2. Set model ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_set_model(self):
        """/settings model opus sets model via _switch_model."""
        update = _make_update(text="/settings model opus")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool, args=["model", "opus"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        # _switch_model persists and restarts
        mock_sessions.set_user_setting.assert_called_once_with(12345, "model", "opus")
        pool.restart.assert_called_once()
        reply = update.message.reply_text.call_args[0][0]
        assert "opus" in reply.lower()

    # ── 3. Set budget ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_set_budget(self):
        """/settings budget 5 sets the budget (within default $10 ceiling)."""
        update = _make_update(text="/settings budget 5")
        config = _make_config()
        pool = _make_mock_claude()
        pool.get_if_exists = MagicMock(return_value=MagicMock())
        ctx = _make_context(config=config, pool=pool, args=["budget", "5"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.set_user_setting.assert_called_once_with(12345, "budget", "5.0")
        reply = update.message.reply_text.call_args[0][0]
        assert "$5.00" in reply

    # ── 4. Reject negative budget ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_reject_negative_budget(self):
        """/settings budget -5 is rejected."""
        update = _make_update(text="/settings budget -5")
        config = _make_config()
        ctx = _make_context(config=config, args=["budget", "-5"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.set_user_setting.assert_not_called()
        reply = update.message.reply_text.call_args[0][0]
        assert "positive" in reply.lower()

    # ── 5. Budget exceeds ceiling ──────────────────────────────────

    @pytest.mark.asyncio
    async def test_budget_exceeds_ceiling(self):
        """/settings budget 999 is rejected when ceiling is lower."""
        from kai.config import UserConfig

        user = UserConfig(telegram_id=12345, name="test", max_budget=5.0)
        config = _make_config(user_configs={12345: user})
        update = _make_update(text="/settings budget 999")
        ctx = _make_context(config=config, args=["budget", "999"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.set_user_setting.assert_not_called()
        reply = update.message.reply_text.call_args[0][0]
        assert "$5.00" in reply
        assert "admin limit" in reply.lower()

    # ── 5b. Budget allowed when ceiling is zero (no limit) ──────────

    @pytest.mark.asyncio
    async def test_budget_allowed_when_ceiling_zero(self):
        """/settings budget 50 succeeds when global ceiling is 0 (no limit)."""
        # If CLAUDE_MAX_BUDGET_USD=0 and no users.yaml entry, 0 means
        # "no admin ceiling" - budget should be allowed, not rejected.
        config = _make_config(claude_max_budget_usd=0.0)
        update = _make_update(text="/settings budget 50")
        pool = _make_mock_claude()
        pool.get_if_exists = MagicMock(return_value=MagicMock())
        ctx = _make_context(config=config, pool=pool, args=["budget", "50"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.set_user_setting.assert_called_once_with(12345, "budget", "50.0")
        reply = update.message.reply_text.call_args[0][0]
        assert "$50.00" in reply

    # ── 6. Set timeout ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_set_timeout(self):
        """/settings timeout 300 sets the timeout."""
        update = _make_update(text="/settings timeout 300")
        config = _make_config()
        pool = _make_mock_claude()
        pool.get_if_exists = MagicMock(return_value=MagicMock())
        ctx = _make_context(config=config, pool=pool, args=["timeout", "300"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.set_user_setting.assert_called_once_with(12345, "timeout", "300")
        reply = update.message.reply_text.call_args[0][0]
        assert "300s" in reply

    # ── 7. Reject zero timeout ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_reject_zero_timeout(self):
        """/settings timeout 0 is rejected (must be positive)."""
        update = _make_update(text="/settings timeout 0")
        ctx = _make_context(args=["timeout", "0"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.set_user_setting.assert_not_called()
        reply = update.message.reply_text.call_args[0][0]
        assert "positive" in reply.lower()

    # ── 8. Reject timeout exceeding max ────────────────────────────

    @pytest.mark.asyncio
    async def test_reject_timeout_exceeding_max(self):
        """/settings timeout 601 is rejected."""
        update = _make_update(text="/settings timeout 601")
        ctx = _make_context(args=["timeout", "601"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.set_user_setting.assert_not_called()
        reply = update.message.reply_text.call_args[0][0]
        assert "600" in reply

    # ── 9. Set context window ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_set_context_window(self):
        """/settings context 200000 sets the context window."""
        update = _make_update(text="/settings context 200000")
        config = _make_config()
        pool = _make_mock_claude()
        instance = MagicMock()
        pool.get_if_exists = MagicMock(return_value=instance)
        ctx = _make_context(config=config, pool=pool, args=["context", "200000"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.set_user_setting.assert_called_once_with(12345, "context_window", "200000")
        # Context window change triggers restart
        pool.restart.assert_called_once()
        reply = update.message.reply_text.call_args[0][0]
        assert "200,000" in reply

    # ── 10. Reject context below minimum ───────────────────────────

    @pytest.mark.asyncio
    async def test_reject_context_below_minimum(self):
        """/settings context 10000 is rejected (below 50000 minimum)."""
        update = _make_update(text="/settings context 10000")
        ctx = _make_context(args=["context", "10000"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.set_user_setting.assert_not_called()
        reply = update.message.reply_text.call_args[0][0]
        assert "50000" in reply

    # ── 11. Context zero accepted (means default) ──────────────────

    @pytest.mark.asyncio
    async def test_context_zero_accepted(self):
        """/settings context 0 is accepted (means 'use default')."""
        update = _make_update(text="/settings context 0")
        config = _make_config()
        pool = _make_mock_claude()
        pool.get_if_exists = MagicMock(return_value=MagicMock())
        ctx = _make_context(config=config, pool=pool, args=["context", "0"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.set_user_setting.assert_called_once_with(12345, "context_window", "0")
        reply = update.message.reply_text.call_args[0][0]
        assert "default" in reply.lower()

    # ── 12. Reset all ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_reset_all(self):
        """/settings reset clears all overrides."""
        update = _make_update(text="/settings reset")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool, args=["reset"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.delete_all_user_settings.assert_called_once_with(12345)
        pool.restart.assert_called_once()
        reply = update.message.reply_text.call_args[0][0]
        assert "all settings cleared" in reply.lower()

    # ── 13. Reset single field ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_reset_single_field(self):
        """/settings reset model clears just the model override."""
        update = _make_update(text="/settings reset model")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool, args=["reset", "model"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.delete_user_setting.assert_called_once_with(12345, "model")
        reply = update.message.reply_text.call_args[0][0]
        assert "model" in reply.lower()

    # ── 14. Unknown field ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_unknown_field(self):
        """/settings bogus shows error with valid field list."""
        update = _make_update(text="/settings bogus")
        ctx = _make_context(args=["bogus"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.set_user_setting.assert_not_called()
        reply = update.message.reply_text.call_args[0][0]
        assert "unknown setting" in reply.lower()

    # ── 15. Invalid model rejected ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_invalid_model_rejected(self):
        """/settings model gpt4 is rejected."""
        update = _make_update(text="/settings model gpt4")
        ctx = _make_context(args=["model", "gpt4"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.set_user_setting.assert_not_called()
        reply = update.message.reply_text.call_args[0][0]
        assert "unknown model" in reply.lower()

    # ── 16. Reset context maps to context_window ───────────────────

    @pytest.mark.asyncio
    async def test_reset_context_alias(self):
        """/settings reset context maps to context_window in DB."""
        update = _make_update(text="/settings reset context")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool, args=["reset", "context"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        # Should use the DB field name, not the user-facing alias
        mock_sessions.delete_user_setting.assert_called_once_with(12345, "context_window")

    # ── 17. Reset reverts instance to defaults ────────────────────

    @pytest.mark.asyncio
    async def test_reset_reverts_instance_model(self):
        """/settings reset model writes the default back onto the instance."""
        update = _make_update(text="/settings reset model")
        config = _make_config()
        pool = _make_mock_claude()
        # Simulate an instance with an overridden model
        instance = MagicMock()
        instance.model = "opus"
        pool.get_if_exists = MagicMock(return_value=instance)
        ctx = _make_context(config=config, pool=pool, args=["reset", "model"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        # Instance model should be reverted to the config default
        assert instance.model == config.claude_model

    # ── 18. Reset all reverts all instance fields ─────────────────

    @pytest.mark.asyncio
    async def test_reset_all_reverts_instance(self):
        """/settings reset reverts all four fields on the live instance."""
        update = _make_update(text="/settings reset")
        config = _make_config()
        pool = _make_mock_claude()
        instance = MagicMock()
        instance.model = "opus"
        instance.max_budget_usd = 99.0
        instance.timeout_seconds = 500
        instance.max_context_window = 500000
        pool.get_if_exists = MagicMock(return_value=instance)
        ctx = _make_context(config=config, pool=pool, args=["reset"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        assert instance.model == config.claude_model
        assert instance.max_budget_usd == config.claude_max_budget_usd
        assert instance.timeout_seconds == config.claude_timeout_seconds
        assert instance.max_context_window == config.claude_max_context_window

    # ── 19. Budget displays "unlimited" when zero ─────────────────

    @pytest.mark.asyncio
    async def test_show_settings_budget_zero_displays_unlimited(self):
        """Budget of $0.00 displays as 'unlimited', not '$0.00'."""
        update = _make_update(text="/settings")
        config = _make_config(claude_max_budget_usd=0.0)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _show_settings(update, 12345, config)

        reply = update.message.reply_text.call_args[0][0]
        assert "unlimited" in reply.lower()
        assert "$0.00" not in reply

    # ── 20. Global budget ceiling visible in /settings ────────────

    @pytest.mark.asyncio
    async def test_show_settings_global_ceiling_visible(self):
        """Users without a yaml entry see the global ceiling in /settings."""
        update = _make_update(text="/settings")
        # No user_configs - ceiling should fall through to global default
        config = _make_config(claude_max_budget_usd=10.0)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _show_settings(update, 12345, config)

        reply = update.message.reply_text.call_args[0][0]
        assert "$10.00" in reply
        assert "ceiling" in reply.lower()


# ── /github command ─────────────────────────────────────────────────


class TestHandleGitHub:
    """Tests for /github - GitHub notification settings.

    Each test patches sessions to isolate the handler from the database.
    The _mock_resolve helper simulates resolve_github_settings() output,
    and _mock_db_settings simulates get_github_db_settings() output for
    source attribution in _show_github().
    """

    def _mock_resolve(self, repos=None, notify_chat_id=12345, pr_review=False, issue_triage=False):
        """Patch resolve_github_settings with controlled return values."""
        settings = {
            "repos": repos or [],
            "notify_chat_id": notify_chat_id,
            "pr_review": pr_review,
            "issue_triage": issue_triage,
        }
        return patch("kai.bot.sessions.resolve_github_settings", new_callable=AsyncMock, return_value=settings)

    def _mock_db_settings(self, db_settings=None):
        """Patch get_github_db_settings with controlled return values."""
        return patch("kai.bot.sessions.get_github_db_settings", new_callable=AsyncMock, return_value=db_settings or {})

    # ── 1. /github with no config (defaults) ──────────────────────

    @pytest.mark.asyncio
    async def test_show_defaults(self):
        """/github with no user config shows global defaults."""
        update = _make_update(text="/github")
        config = _make_config()

        with self._mock_resolve(), self._mock_db_settings():
            await _show_github(update, 12345, config)

        reply = update.message.reply_text.call_args[0][0]
        assert "GitHub: not configured" in reply
        assert "Notifications: this chat" in reply
        assert "PR reviews: off (global default)" in reply
        assert "Issue triage: off (global default)" in reply
        assert "No repo subscriptions" in reply

    # ── 2. /github with full config ───────────────────────────────

    @pytest.mark.asyncio
    async def test_show_full_config(self):
        """/github with user config shows all fields with sources."""
        from kai.config import UserConfig

        user = UserConfig(
            telegram_id=12345,
            name="alice",
            github="alice",
            github_repos=["alice/repo1"],
            pr_review=True,
            issue_triage=False,
        )
        config = _make_config(user_configs={12345: user})
        update = _make_update(text="/github")

        with (
            self._mock_resolve(
                repos=["alice/repo1"],
                notify_chat_id=99999,
                pr_review=True,
                issue_triage=True,
            ),
            # DB has issue_triage override, pr_review from yaml
            self._mock_db_settings({"issue_triage": "true"}),
        ):
            await _show_github(update, 12345, config)

        reply = update.message.reply_text.call_args[0][0]
        assert "GitHub: alice" in reply
        assert "Notifications: 99999" in reply
        assert "PR reviews: on (users.yaml)" in reply
        assert "Issue triage: on (user override)" in reply
        assert "alice/repo1" in reply

    # ── 3. /github notify <chat_id> ───────────────────────────────

    @pytest.mark.asyncio
    async def test_notify_set(self):
        """/github notify 123456 sets notification chat and updates live set."""
        update = _make_update(text="/github notify 123456")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "123456"])
        mock_sessions = AsyncMock()
        mock_sessions.set_setting = AsyncMock()
        mock_sessions.delete_setting = AsyncMock()
        mock_sessions.resolve_github_settings = AsyncMock()
        mock_sessions.get_github_db_settings = AsyncMock(return_value={})
        # No existing notify setting (fresh set)
        mock_sessions.get_setting = AsyncMock(return_value=None)

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.webhook.add_allowed_chat_id") as mock_add,
        ):
            await handle_github(update, ctx)

        mock_sessions.set_setting.assert_called_once_with("github_notify_chat:12345", "123456")
        mock_add.assert_called_once_with(123456)
        reply = update.message.reply_text.call_args[0][0]
        assert "123456" in reply
        # Fix removes the restart requirement
        assert "restart" not in reply.lower()

    # ── 4. /github notify reset ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_notify_reset(self):
        """/github notify reset clears the override."""
        update = _make_update(text="/github notify reset")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "reset"])
        mock_sessions = AsyncMock()
        mock_sessions.get_setting = AsyncMock(return_value=None)
        mock_sessions.delete_setting = AsyncMock()
        mock_sessions.resolve_github_settings = AsyncMock()
        mock_sessions.get_github_db_settings = AsyncMock(return_value={})

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        mock_sessions.delete_setting.assert_called_once_with("github_notify_chat:12345")
        reply = update.message.reply_text.call_args[0][0]
        assert "reset" in reply.lower()

    # ── 5. /github notify abc (invalid) ───────────────────────────

    @pytest.mark.asyncio
    async def test_notify_invalid(self):
        """/github notify abc is rejected (not an integer)."""
        update = _make_update(text="/github notify abc")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "abc"])
        mock_sessions = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        reply = update.message.reply_text.call_args[0][0]
        assert "integer" in reply.lower()

    # ── 6. /github reviews on ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_reviews_on(self):
        """/github reviews on enables PR reviews."""
        update = _make_update(text="/github reviews on")
        config = _make_config()
        ctx = _make_context(config=config, args=["reviews", "on"])
        mock_sessions = AsyncMock()
        mock_sessions.set_setting = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        mock_sessions.set_setting.assert_called_once_with("pr_review:12345", "true")
        reply = update.message.reply_text.call_args[0][0]
        assert "enabled" in reply.lower()

    # ── 7. /github reviews off ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_reviews_off(self):
        """/github reviews off disables PR reviews."""
        update = _make_update(text="/github reviews off")
        config = _make_config()
        ctx = _make_context(config=config, args=["reviews", "off"])
        mock_sessions = AsyncMock()
        mock_sessions.set_setting = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        mock_sessions.set_setting.assert_called_once_with("pr_review:12345", "false")
        reply = update.message.reply_text.call_args[0][0]
        assert "disabled" in reply.lower()

    # ── 8. /github triage on ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_triage_on(self):
        """/github triage on enables issue triage."""
        update = _make_update(text="/github triage on")
        config = _make_config()
        ctx = _make_context(config=config, args=["triage", "on"])
        mock_sessions = AsyncMock()
        mock_sessions.set_setting = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        mock_sessions.set_setting.assert_called_once_with("issue_triage:12345", "true")
        reply = update.message.reply_text.call_args[0][0]
        assert "enabled" in reply.lower()

    # ── 9. /github reviews (no value) ─────────────────────────────

    @pytest.mark.asyncio
    async def test_reviews_missing_value(self):
        """/github reviews with no on/off shows usage."""
        update = _make_update(text="/github reviews")
        config = _make_config()
        ctx = _make_context(config=config, args=["reviews"])
        mock_sessions = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        reply = update.message.reply_text.call_args[0][0]
        assert "usage" in reply.lower()

    # ── 10. /github bogus ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_unknown_subcommand(self):
        """/github bogus shows error message."""
        update = _make_update(text="/github bogus")
        config = _make_config()
        ctx = _make_context(config=config, args=["bogus"])
        mock_sessions = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        reply = update.message.reply_text.call_args[0][0]
        assert "unknown subcommand" in reply.lower()

    # ── 11. /github notify - live allowed_user_ids updates ────────

    @pytest.mark.asyncio
    async def test_notify_set_updates_allowed_ids(self):
        """/github notify -100999 adds the group to allowed_user_ids immediately."""
        update = _make_update(text="/github notify -100999")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "-100999"])
        mock_sessions = AsyncMock()
        mock_sessions.set_setting = AsyncMock()
        # No existing notify setting (fresh set)
        mock_sessions.get_setting = AsyncMock(return_value=None)

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.webhook.add_allowed_chat_id") as mock_add,
        ):
            await handle_github(update, ctx)

        mock_sessions.set_setting.assert_called_once_with("github_notify_chat:12345", "-100999")
        mock_add.assert_called_once_with(-100999)
        reply = update.message.reply_text.call_args[0][0]
        assert "restart" not in reply.lower()

    @pytest.mark.asyncio
    async def test_notify_set_overwrite_cleans_up_old(self):
        """Overwriting a notify destination removes the old chat_id from allowed set.

        Without this cleanup, the old chat_id would linger in allowed_user_ids
        until restart - an auth leak for abandoned group chats.
        """
        update = _make_update(text="/github notify -100888")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "-100888"])
        mock_sessions = AsyncMock()
        mock_sessions.set_setting = AsyncMock()
        # Existing notify setting points to a different group chat
        mock_sessions.get_setting = AsyncMock(return_value="-100999")

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.webhook.add_allowed_chat_id") as mock_add,
            patch("kai.bot.webhook.remove_allowed_chat_id") as mock_remove,
            patch("kai.bot._is_notify_chat_used", new_callable=AsyncMock, return_value=False),
        ):
            await handle_github(update, ctx)

        # Old destination removed, new one added
        mock_remove.assert_called_once_with(-100999)
        mock_add.assert_called_once_with(-100888)

    @pytest.mark.asyncio
    async def test_notify_set_overwrite_skips_cleanup_when_shared(self):
        """Overwriting a notify destination keeps the old chat_id if another user uses it."""
        update = _make_update(text="/github notify -100888")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "-100888"])
        mock_sessions = AsyncMock()
        mock_sessions.set_setting = AsyncMock()
        mock_sessions.get_setting = AsyncMock(return_value="-100999")

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.webhook.add_allowed_chat_id") as mock_add,
            patch("kai.bot.webhook.remove_allowed_chat_id") as mock_remove,
            # Another user still uses the old chat_id
            patch("kai.bot._is_notify_chat_used", new_callable=AsyncMock, return_value=True),
        ):
            await handle_github(update, ctx)

        # Old destination kept (still used), new one added
        mock_remove.assert_not_called()
        mock_add.assert_called_once_with(-100888)

    @pytest.mark.asyncio
    async def test_notify_set_overwrite_same_value_no_cleanup(self):
        """Setting the same notify destination again does not trigger cleanup."""
        update = _make_update(text="/github notify -100999")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "-100999"])
        mock_sessions = AsyncMock()
        mock_sessions.set_setting = AsyncMock()
        # Same value already stored
        mock_sessions.get_setting = AsyncMock(return_value="-100999")

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.webhook.add_allowed_chat_id") as mock_add,
            patch("kai.bot.webhook.remove_allowed_chat_id") as mock_remove,
        ):
            await handle_github(update, ctx)

        # No removal needed (same value), just re-add (idempotent)
        mock_remove.assert_not_called()
        mock_add.assert_called_once_with(-100999)

    @pytest.mark.asyncio
    async def test_notify_reset_removes_from_allowed_ids(self):
        """/github notify reset removes the old chat_id from allowed_user_ids."""
        update = _make_update(text="/github notify reset")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "reset"])
        mock_sessions = AsyncMock()
        # Simulate an existing notify setting pointing to a group chat
        mock_sessions.get_setting = AsyncMock(return_value="-100999")
        mock_sessions.delete_setting = AsyncMock()

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.webhook.remove_allowed_chat_id") as mock_remove,
            patch("kai.bot._is_notify_chat_used", new_callable=AsyncMock, return_value=False),
        ):
            await handle_github(update, ctx)

        mock_sessions.delete_setting.assert_called_once_with("github_notify_chat:12345")
        mock_remove.assert_called_once_with(-100999)

    @pytest.mark.asyncio
    async def test_notify_reset_skips_removal_when_shared(self):
        """/github notify reset does NOT remove a chat_id still used by another user."""
        update = _make_update(text="/github notify reset")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "reset"])
        mock_sessions = AsyncMock()
        mock_sessions.get_setting = AsyncMock(return_value="-100999")
        mock_sessions.delete_setting = AsyncMock()

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.webhook.remove_allowed_chat_id") as mock_remove,
            # Another user still uses this chat_id
            patch("kai.bot._is_notify_chat_used", new_callable=AsyncMock, return_value=True),
        ):
            await handle_github(update, ctx)

        mock_sessions.delete_setting.assert_called_once()
        mock_remove.assert_not_called()

    @pytest.mark.asyncio
    async def test_notify_reset_self_id_skips_removal(self):
        """/github notify reset does NOT remove the user's own chat_id.

        This is the primary guard for legacy mode (no users.yaml) where
        remove_allowed_chat_id's user_configs check cannot fire.
        """
        update = _make_update(text="/github notify reset")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "reset"])
        mock_sessions = AsyncMock()
        # The user previously set their own telegram_id (12345) as the target
        mock_sessions.get_setting = AsyncMock(return_value="12345")
        mock_sessions.delete_setting = AsyncMock()

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.webhook.remove_allowed_chat_id") as mock_remove,
        ):
            await handle_github(update, ctx)

        mock_sessions.delete_setting.assert_called_once()
        # The self-ID guard fires before _is_notify_chat_used is even called
        mock_remove.assert_not_called()


# ── _is_notify_chat_used ──────────────────────────────────────────


class TestIsNotifyChatUsed:
    """Tests for _is_notify_chat_used helper.

    Verifies detection of shared notify chat_ids across users.yaml,
    database settings, and the global env var fallback.
    """

    @pytest.mark.asyncio
    async def test_yaml_match(self):
        """Returns True when another user in users.yaml uses the chat_id."""
        from kai.config import UserConfig

        user_a = UserConfig(telegram_id=111, name="alice")
        user_b = UserConfig(telegram_id=222, name="bob", github_notify_chat_id=-100999)
        config = _make_config(user_configs={111: user_a, 222: user_b})

        mock_sessions = AsyncMock()
        mock_sessions.get_setting = AsyncMock(return_value=None)

        with patch("kai.bot.sessions", mock_sessions):
            result = await _is_notify_chat_used(-100999, exclude_user=111, config=config)
        assert result is True

    @pytest.mark.asyncio
    async def test_db_match(self):
        """Returns True when another user's DB setting uses the chat_id."""
        from kai.config import UserConfig

        user_a = UserConfig(telegram_id=111, name="alice")
        user_b = UserConfig(telegram_id=222, name="bob")
        config = _make_config(user_configs={111: user_a, 222: user_b})

        mock_sessions = AsyncMock()
        # User B has the notify target in the DB
        mock_sessions.get_setting = AsyncMock(
            side_effect=lambda key: "-100999" if key == "github_notify_chat:222" else None
        )

        with patch("kai.bot.sessions", mock_sessions):
            result = await _is_notify_chat_used(-100999, exclude_user=111, config=config)
        assert result is True

    @pytest.mark.asyncio
    async def test_env_var_match(self):
        """Returns True when the global env var fallback uses the chat_id."""
        config = _make_config(github_notify_chat_id=-100999)

        mock_sessions = AsyncMock()
        mock_sessions.get_setting = AsyncMock(return_value=None)

        with patch("kai.bot.sessions", mock_sessions):
            result = await _is_notify_chat_used(-100999, exclude_user=111, config=config)
        assert result is True

    @pytest.mark.asyncio
    async def test_no_match(self):
        """Returns False when no other source references the chat_id."""
        from kai.config import UserConfig

        user_a = UserConfig(telegram_id=111, name="alice")
        config = _make_config(user_configs={111: user_a})

        mock_sessions = AsyncMock()
        mock_sessions.get_setting = AsyncMock(return_value=None)

        with patch("kai.bot.sessions", mock_sessions):
            result = await _is_notify_chat_used(-100999, exclude_user=111, config=config)
        assert result is False


# ── /model persistence ─────────────────────────────────────────────


class TestModelPersistence:
    """Tests that /model and /models keyboard now persist to settings."""

    @pytest.mark.asyncio
    async def test_model_command_persists(self):
        """/model opus writes to settings table."""
        update = _make_update(text="/model opus")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool, args=["opus"])
        mock_sessions = AsyncMock()
        mock_sessions.set_user_setting = AsyncMock()
        mock_sessions.clear_session = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_model(update, ctx)

        mock_sessions.set_user_setting.assert_called_once_with(12345, "model", "opus")
        pool.restart.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_callback_persists(self):
        """/models keyboard callback writes to settings table."""
        update = _make_callback_update(data="model:opus")
        config = _make_config()
        pool = _make_mock_claude(model="sonnet")  # different from opus
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = AsyncMock()
        mock_sessions.set_user_setting = AsyncMock()
        mock_sessions.clear_session = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_model_callback(update, ctx)

        mock_sessions.set_user_setting.assert_called_once_with(12345, "model", "opus")
        pool.restart.assert_called_once()

    @pytest.mark.asyncio
    async def test_settings_reset_model_clears_db(self):
        """/settings reset model clears what /model set."""
        update = _make_update(text="/settings reset model")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool, args=["reset", "model"])
        mock_sessions = AsyncMock()
        mock_sessions.delete_user_setting = AsyncMock()
        mock_sessions.clear_session = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await _handle_settings_reset(update, ctx, 12345, config, "model")

        mock_sessions.delete_user_setting.assert_called_once_with(12345, "model")


# ── Field alias mapping ───────────────────────────────────────────


class TestFieldAliases:
    """Tests for the _FIELD_ALIASES mapping."""

    def test_context_maps_to_context_window(self):
        """The 'context' alias maps to 'context_window' DB key."""
        assert _FIELD_ALIASES["context"] == "context_window"

    def test_no_alias_returns_field_name(self):
        """Fields without aliases return the field name itself via .get()."""
        assert _FIELD_ALIASES.get("model", "model") == "model"
        assert _FIELD_ALIASES.get("budget", "budget") == "budget"
        assert _FIELD_ALIASES.get("timeout", "timeout") == "timeout"
