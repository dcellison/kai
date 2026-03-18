"""
Tests for multi-user isolation boundaries.

Verifies that user A's data, state, and processes remain completely separate
from user B's across all subsystems: ClaudeManager instances, SQLite sessions
and settings, per-user locks, crash recovery flags, JSONL history, handler
routing, webhook user extraction, and cron job dispatch.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai import history, locks, sessions
from kai.bot import (
    _clear_responding,
    _set_responding,
    handle_document,
    handle_jobs,
    handle_message,
    handle_model_callback,
    handle_models,
    handle_new,
    handle_photo,
    handle_stats,
    handle_stop,
)
from kai.claude import ClaudeManager
from kai.logging import reset_session_id
from kai.webhook import _extract_user_id, update_workspace

# Re-use test helpers from test_bot
from tests.test_bot import (
    _done_event,
    _fake_lock,
    _make_callback_update,
    _make_config,
    _make_context,
    _make_mock_claude,
    _make_update,
)

USER_A = 100
USER_B = 200


@pytest.fixture()
async def db(tmp_path):
    await sessions.init_db(tmp_path / "test.db")
    yield
    await sessions.close_db()


# ── 1. ClaudeManager isolation ───────────────────────────────────────


class TestClaudeManagerIsolation:
    @pytest.fixture()
    def manager(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        config = _make_config(
            allowed_user_ids={USER_A, USER_B},
            claude_workspace=ws,
        )
        with patch("kai.claude.DATA_DIR", tmp_path):
            mgr = ClaudeManager(config=config)
            yield mgr

    async def test_separate_instances_per_user(self, manager):
        a = await manager.get_or_create(USER_A)
        b = await manager.get_or_create(USER_B)
        assert a is not b
        assert a.user_id == USER_A
        assert b.user_id == USER_B

    async def test_same_instance_on_repeat_call(self, manager):
        first = await manager.get_or_create(USER_A)
        second = await manager.get_or_create(USER_A)
        assert first is second

    async def test_user_home_dirs_created(self, tmp_path, manager):
        await manager.get_or_create(USER_A)
        await manager.get_or_create(USER_B)
        assert (tmp_path / "users" / "100" / "home" / ".claude" / "history").is_dir()
        assert (tmp_path / "users" / "200" / "home" / "files").is_dir()

    async def test_template_claude_md_copied(self, tmp_path, manager):
        # Create template CLAUDE.md in the workspace
        ws = tmp_path / "workspace"
        template = ws / ".claude" / "CLAUDE.md"
        template.parent.mkdir(parents=True, exist_ok=True)
        template.write_text("# Template Instructions")

        await manager.get_or_create(USER_A)
        await manager.get_or_create(USER_B)

        a_md = tmp_path / "users" / "100" / "home" / ".claude" / "CLAUDE.md"
        b_md = tmp_path / "users" / "200" / "home" / ".claude" / "CLAUDE.md"
        assert a_md.read_text() == "# Template Instructions"
        assert b_md.read_text() == "# Template Instructions"

        # Independent copies — modifying one doesn't affect the other
        a_md.write_text("# Modified")
        assert b_md.read_text() == "# Template Instructions"

    async def test_shutdown_user_preserves_other(self, manager):
        await manager.get_or_create(USER_A)
        await manager.get_or_create(USER_B)
        with patch.object(manager._instances[USER_A], "shutdown", new_callable=AsyncMock):
            await manager.shutdown_user(USER_A)
        assert USER_A not in manager._instances
        assert USER_B in manager._instances

    async def test_get_returns_existing_instance(self, manager):
        created = await manager.get_or_create(USER_A)
        assert manager.get(USER_A) is created

    def test_get_returns_none_for_missing(self, manager):
        assert manager.get(999) is None

    async def test_shutdown_all_clears_everything(self, manager):
        await manager.get_or_create(USER_A)
        await manager.get_or_create(USER_B)
        with (
            patch.object(manager._instances[USER_A], "shutdown", new_callable=AsyncMock),
            patch.object(manager._instances[USER_B], "shutdown", new_callable=AsyncMock),
        ):
            await manager.shutdown_all()
        assert len(manager._instances) == 0


# ── 2. Session isolation ─────────────────────────────────────────────


class TestSessionIsolation:
    async def test_sessions_isolated(self, db):
        await sessions.save_session(USER_A, USER_A, "sess-A", "sonnet", 0.1)
        await sessions.save_session(USER_B, USER_B, "sess-B", "opus", 0.2)
        assert await sessions.get_session(USER_A) == "sess-A"
        assert await sessions.get_session(USER_B) == "sess-B"

    async def test_clear_session_only_affects_target(self, db):
        await sessions.save_session(USER_A, USER_A, "sess-A", "sonnet", 0.1)
        await sessions.save_session(USER_B, USER_B, "sess-B", "opus", 0.2)
        await sessions.clear_session(USER_A)
        assert await sessions.get_session(USER_A) is None
        assert await sessions.get_session(USER_B) == "sess-B"

    async def test_stats_isolated(self, db):
        await sessions.save_session(USER_A, USER_A, "sess-A", "sonnet", 0.5)
        await sessions.save_session(USER_B, USER_B, "sess-B", "opus", 0.3)
        stats_a = await sessions.get_stats(USER_A)
        stats_b = await sessions.get_stats(USER_B)
        assert stats_a["model"] == "sonnet"
        assert stats_b["model"] == "opus"

    async def test_cost_accumulates_independently(self, db):
        await sessions.save_session(USER_A, USER_A, "sess-A", "sonnet", 0.5)
        await sessions.save_session(USER_A, USER_A, "sess-A", "sonnet", 0.5)
        await sessions.save_session(USER_B, USER_B, "sess-B", "opus", 0.3)
        stats_a = await sessions.get_stats(USER_A)
        stats_b = await sessions.get_stats(USER_B)
        assert abs(stats_a["total_cost_usd"] - 1.0) < 0.001
        assert abs(stats_b["total_cost_usd"] - 0.3) < 0.001

    async def test_jobs_isolated(self, db):
        await sessions.create_job(USER_A, USER_A, "job-a", "reminder", "hi", "interval", '{"seconds":60}')
        await sessions.create_job(USER_B, USER_B, "job-b", "reminder", "bye", "interval", '{"seconds":60}')
        jobs_a = await sessions.get_jobs(USER_A)
        jobs_b = await sessions.get_jobs(USER_B)
        assert len(jobs_a) == 1 and jobs_a[0]["name"] == "job-a"
        assert len(jobs_b) == 1 and jobs_b[0]["name"] == "job-b"

    async def test_canceljob_cannot_delete_other_users_job(self, db):
        """User B cannot cancel User A's job via /canceljob."""
        await sessions.create_job(USER_A, USER_A, "secret", "reminder", "hi", "interval", '{"seconds":60}')
        jobs_a = await sessions.get_jobs(USER_A)
        job_id = jobs_a[0]["id"]
        # User B tries to delete User A's job
        deleted = await sessions.delete_job(job_id, user_id=USER_B)
        assert not deleted
        # Job still exists for User A
        assert len(await sessions.get_jobs(USER_A)) == 1


# ── 3. Settings isolation ────────────────────────────────────────────


class TestSettingsIsolation:
    async def test_same_key_different_values(self, db):
        await sessions.set_setting(USER_A, "voice_mode", "on")
        await sessions.set_setting(USER_B, "voice_mode", "only")
        assert await sessions.get_setting(USER_A, "voice_mode") == "on"
        assert await sessions.get_setting(USER_B, "voice_mode") == "only"

    async def test_delete_leaves_other_user(self, db):
        await sessions.set_setting(USER_A, "workspace", "/alpha")
        await sessions.set_setting(USER_B, "workspace", "/beta")
        await sessions.delete_setting(USER_A, "workspace")
        assert await sessions.get_setting(USER_A, "workspace") is None
        assert await sessions.get_setting(USER_B, "workspace") == "/beta"

    async def test_workspace_history_isolated(self, db):
        await sessions.upsert_workspace_history(USER_A, "/alpha")
        await sessions.upsert_workspace_history(USER_B, "/beta")
        history_a = await sessions.get_workspace_history(USER_A)
        history_b = await sessions.get_workspace_history(USER_B)
        assert [h["path"] for h in history_a] == ["/alpha"]
        assert [h["path"] for h in history_b] == ["/beta"]


# ── 4. Lock isolation ────────────────────────────────────────────────


class TestLockIsolation:
    @pytest.fixture(autouse=True)
    def _clear_locks(self):
        locks._user_locks.clear()
        locks._stop_events.clear()
        yield
        locks._user_locks.clear()
        locks._stop_events.clear()

    def test_separate_lock_objects(self):
        assert locks.get_lock(USER_A) is not locks.get_lock(USER_B)

    async def test_user_a_locked_user_b_free(self):
        lock_a = locks.get_lock(USER_A)
        lock_b = locks.get_lock(USER_B)
        await lock_a.acquire()
        try:
            assert lock_a.locked()
            assert not lock_b.locked()
        finally:
            lock_a.release()

    def test_stop_events_independent(self):
        event_a = locks.get_stop_event(USER_A)
        event_b = locks.get_stop_event(USER_B)
        event_a.set()
        assert event_a.is_set()
        assert not event_b.is_set()

    def test_same_lock_on_repeat_call(self):
        first = locks.get_lock(USER_A)
        second = locks.get_lock(USER_A)
        assert first is second

    def test_same_stop_event_on_repeat_call(self):
        first = locks.get_stop_event(USER_A)
        second = locks.get_stop_event(USER_A)
        assert first is second


# ── 5. Crash flag isolation ──────────────────────────────────────────


class TestCrashFlagIsolation:
    async def test_flags_per_user(self, tmp_path):
        with patch("kai.bot.DATA_DIR", tmp_path):
            await _set_responding(USER_A, 111)
            await _set_responding(USER_B, 222)
        flag_a = tmp_path / "users" / "100" / ".responding_to"
        flag_b = tmp_path / "users" / "200" / ".responding_to"
        assert flag_a.read_text() == "111"
        assert flag_b.read_text() == "222"

    async def test_clear_only_affects_target(self, tmp_path):
        with patch("kai.bot.DATA_DIR", tmp_path):
            await _set_responding(USER_A, 111)
            await _set_responding(USER_B, 222)
            await _clear_responding(USER_A)
        flag_a = tmp_path / "users" / "100" / ".responding_to"
        flag_b = tmp_path / "users" / "200" / ".responding_to"
        assert not flag_a.exists()
        assert flag_b.read_text() == "222"


# ── 6. History isolation ─────────────────────────────────────────────


class TestHistoryIsolation:
    async def test_messages_in_separate_dirs(self, tmp_path):
        with patch("kai.history.DATA_DIR", tmp_path):
            await history.log_message(direction="user", user_id=USER_A, chat_id=USER_A, text="hello from A")
            await history.log_message(direction="user", user_id=USER_B, chat_id=USER_B, text="hello from B")

        dir_a = tmp_path / "users" / "100" / "home" / ".claude" / "history"
        dir_b = tmp_path / "users" / "200" / "home" / ".claude" / "history"
        assert dir_a.is_dir()
        assert dir_b.is_dir()

        files_a = list(dir_a.glob("*.jsonl"))
        files_b = list(dir_b.glob("*.jsonl"))
        assert len(files_a) == 1
        assert len(files_b) == 1

        content_a = files_a[0].read_text()
        content_b = files_b[0].read_text()
        assert "hello from A" in content_a
        assert "hello from B" not in content_a
        assert "hello from B" in content_b
        assert "hello from A" not in content_b

    async def test_get_recent_history_isolated(self, tmp_path):
        with patch("kai.history.DATA_DIR", tmp_path):
            await history.log_message(direction="user", user_id=USER_A, chat_id=USER_A, text="hello from A")
            await history.log_message(direction="user", user_id=USER_B, chat_id=USER_B, text="hello from B")
            recent_a = history.get_recent_history(USER_A)
            recent_b = history.get_recent_history(USER_B)

        assert "hello from A" in recent_a
        assert "hello from B" not in recent_a
        assert "hello from B" in recent_b
        assert "hello from A" not in recent_b


# ── 7. Handler user routing ──────────────────────────────────────────


def _two_user_setup():
    """Create two mock Claude instances and a manager that routes by user_id."""
    claude_a = _make_mock_claude()
    claude_b = _make_mock_claude()
    instances = {USER_A: claude_a, USER_B: claude_b}
    manager = MagicMock()
    manager.get_or_create = AsyncMock(side_effect=lambda uid: instances[uid])
    config = _make_config(allowed_user_ids={USER_A, USER_B})
    return claude_a, claude_b, manager, config


def _make_two_user_context(manager, config, **kwargs):
    """Create a context wired to the two-user manager."""
    ctx = _make_context(config=config, **kwargs)
    ctx.bot_data["claude_manager"] = manager
    return ctx


class TestHandlerUserRouting:
    async def test_handle_new_routes_to_user(self):
        claude_a, claude_b, manager, config = _two_user_setup()
        update = _make_update(user_id=USER_A)
        ctx = _make_two_user_context(manager, config)
        with patch("kai.bot.sessions.clear_session", new_callable=AsyncMock):
            await handle_new(update, ctx)
        claude_a.restart.assert_called_once()
        claude_b.restart.assert_not_called()

    async def test_handle_stop_routes_to_user(self):
        claude_a, claude_b, manager, config = _two_user_setup()
        instances = {USER_A: claude_a, USER_B: claude_b}
        manager.get = MagicMock(side_effect=lambda uid: instances.get(uid))
        update = _make_update(user_id=USER_B)
        ctx = _make_two_user_context(manager, config)
        stop_event = asyncio.Event()
        with patch("kai.bot.get_stop_event", return_value=stop_event):
            await handle_stop(update, ctx)
        claude_b.force_kill.assert_called_once()
        claude_a.force_kill.assert_not_called()

    async def test_handle_message_sends_to_correct_claude(self):
        claude_a, claude_b, manager, config = _two_user_setup()
        update = _make_update(text="hello from A", user_id=USER_A)
        ctx = _make_two_user_context(manager, config)
        with (
            patch("kai.bot.is_totp_configured", return_value=False),
            patch("kai.bot._handle_response", new_callable=AsyncMock) as mock_resp,
            patch("kai.bot.log_message", new_callable=AsyncMock),
            patch("kai.bot._set_responding", new_callable=AsyncMock),
            patch("kai.bot._clear_responding", new_callable=AsyncMock),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_message(update, ctx)
        # _handle_response receives claude_a (6th positional arg, index 5)
        called_claude = mock_resp.call_args[0][5]
        assert called_claude is claude_a
        assert called_claude is not claude_b  # Explicit isolation check

    async def test_handle_photo_saves_to_user_session_dir(self, tmp_path):
        _claude_a, _claude_b, manager, config = _two_user_setup()

        update = _make_update(user_id=USER_A)
        photo = MagicMock()
        photo.file_id = "file123"
        photo.file_unique_id = "uniq123"
        update.message.photo = [MagicMock(), photo]

        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"img"))
        ctx = _make_two_user_context(manager, config)
        ctx.bot.get_file = AsyncMock(return_value=mock_file)

        reset_session_id(USER_A)
        with (
            patch("kai.bot.DATA_DIR", tmp_path),
            patch("kai.bot._handle_response", new_callable=AsyncMock),
            patch("kai.bot.log_message", new_callable=AsyncMock),
            patch("kai.bot._set_responding", new_callable=AsyncMock),
            patch("kai.bot._clear_responding", new_callable=AsyncMock),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_photo(update, ctx)
        # Photo should be saved under user A's home/files/ dir
        user_a_files = tmp_path / "users" / str(USER_A) / "home" / "files"
        assert user_a_files.is_dir()
        assert len(list(user_a_files.iterdir())) == 1
        # User B should have no files
        user_b_files = tmp_path / "users" / str(USER_B) / "home" / "files"
        assert not user_b_files.exists()

    async def test_models_shows_correct_user_model(self):
        claude_a, claude_b, manager, config = _two_user_setup()
        claude_a.model = "opus"
        claude_b.model = "sonnet"
        update = _make_update(user_id=USER_B)
        ctx = _make_two_user_context(manager, config)
        await handle_models(update, ctx)
        # The reply should reflect user B's model (sonnet), not A's (opus)
        call = update.message.reply_text.call_args
        reply_markup = call[1]["reply_markup"]
        # The keyboard buttons should have the current model marked
        button_texts = [btn.text for row in reply_markup.inline_keyboard for btn in row]
        # At least one button should contain "sonnet" with the green circle marker
        has_sonnet_marker = any("sonnet" in t.lower() and "\U0001f7e2" in t for t in button_texts)
        has_opus_marker = any("opus" in t.lower() and "\U0001f7e2" in t for t in button_texts)
        assert has_sonnet_marker
        assert not has_opus_marker

    async def test_handle_document_saves_to_user_files_dir(self, tmp_path):
        _claude_a, _claude_b, manager, config = _two_user_setup()

        update = _make_update(user_id=USER_A)
        doc = MagicMock()
        doc.file_id = "doc123"
        doc.file_name = "notes.txt"
        update.message.document = doc
        update.message.caption = "my notes"

        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"hello"))
        ctx = _make_two_user_context(manager, config)
        ctx.bot.get_file = AsyncMock(return_value=mock_file)

        reset_session_id(USER_A)
        with (
            patch("kai.bot.DATA_DIR", tmp_path),
            patch("kai.bot._handle_response", new_callable=AsyncMock),
            patch("kai.bot.log_message", new_callable=AsyncMock),
            patch("kai.bot._set_responding", new_callable=AsyncMock),
            patch("kai.bot._clear_responding", new_callable=AsyncMock),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_document(update, ctx)
        # Document should be saved under user A's home/files/ dir
        user_a_files = tmp_path / "users" / str(USER_A) / "home" / "files"
        assert user_a_files.is_dir()
        assert len(list(user_a_files.iterdir())) == 1
        # User B should have no files
        user_b_files = tmp_path / "users" / str(USER_B) / "home" / "files"
        assert not user_b_files.exists()

    async def test_handle_model_callback_isolated(self):
        claude_a, claude_b, manager, config = _two_user_setup()
        claude_a.model = "sonnet"
        claude_b.model = "sonnet"
        update = _make_callback_update(data="model:opus", user_id=USER_A)
        ctx = _make_two_user_context(manager, config)
        with patch("kai.bot.sessions.clear_session", new_callable=AsyncMock):
            await handle_model_callback(update, ctx)
        assert claude_a.model == "opus"
        assert claude_b.model == "sonnet"  # Unchanged

    async def test_workspace_switch_isolated(self):
        claude_a, claude_b, manager, config = _two_user_setup()
        # Simulate a workspace switch for user A via _do_switch_workspace
        from kai.bot import _do_switch_workspace

        target = Path("/tmp/test_ws")
        with (
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.upsert_workspace_history", new_callable=AsyncMock),
            patch("kai.bot.webhook.update_workspace"),
        ):
            ctx = _make_two_user_context(manager, config)
            await _do_switch_workspace(ctx, USER_A, target)
        claude_a.change_workspace.assert_called_once()
        claude_b.change_workspace.assert_not_called()

    async def test_stats_shows_correct_user(self):
        _claude_a, _claude_b, manager, config = _two_user_setup()
        stats_data = {
            USER_A: {
                "session_id": "a1",
                "model": "opus",
                "created_at": "2026-01-01",
                "last_used_at": "2026-01-02",
                "total_cost_usd": 5.0,
            },
            USER_B: {
                "session_id": "b1",
                "model": "sonnet",
                "created_at": "2026-01-01",
                "last_used_at": "2026-01-02",
                "total_cost_usd": 1.0,
            },
        }

        async def mock_get_stats(uid):
            return stats_data.get(uid)

        update = _make_update(user_id=USER_B)
        ctx = _make_two_user_context(manager, config)
        with patch("kai.bot.sessions.get_stats", new_callable=AsyncMock, side_effect=mock_get_stats):
            await handle_stats(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "sonnet" in reply
        assert "1.0" in reply

    async def test_jobs_shows_correct_user(self):
        _claude_a, _claude_b, manager, config = _two_user_setup()
        jobs_data = {
            USER_A: [
                {
                    "id": 1,
                    "name": "job-A",
                    "job_type": "reminder",
                    "prompt": "hi",
                    "schedule_type": "interval",
                    "schedule_data": '{"seconds": 3600}',
                    "auto_remove": False,
                    "notify_on_check": False,
                    "created_at": "2026-01-01",
                }
            ],
            USER_B: [],
        }

        async def mock_get_jobs(uid):
            return jobs_data.get(uid, [])

        update = _make_update(user_id=USER_A)
        ctx = _make_two_user_context(manager, config)
        with patch("kai.bot.sessions.get_jobs", new_callable=AsyncMock, side_effect=mock_get_jobs):
            await handle_jobs(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "job-A" in reply


# ── 8. Webhook user routing ──────────────────────────────────────────


class TestWebhookUserRouting:
    def _make_request(self, headers=None):
        req = MagicMock()
        req.headers = headers or {}
        req.app = {"allowed_user_ids": {USER_A, USER_B}}
        return req

    def test_extract_from_header(self):
        req = self._make_request(headers={"X-User-Id": "100"})
        assert _extract_user_id(req) == USER_A

    def test_extract_from_body_chat_id(self):
        req = self._make_request()
        assert _extract_user_id(req, body={"chat_id": 200}) == USER_B

    def test_unknown_user_returns_none(self):
        req = self._make_request(headers={"X-User-Id": "999"})
        result = _extract_user_id(req)
        assert result is None

    def test_update_workspace_per_user(self):
        import kai.webhook as wh

        # Set up a mock app
        mock_app = {"user_workspaces": {}}
        with patch.object(wh, "_app", mock_app):
            update_workspace(USER_A, "/ws_a")
            update_workspace(USER_B, "/ws_b")
        assert mock_app["user_workspaces"][USER_A] == "/ws_a"
        assert mock_app["user_workspaces"][USER_B] == "/ws_b"


# ── 9. Cron job user routing ─────────────────────────────────────────


class TestCronJobUserRouting:
    async def test_job_uses_correct_user_claude(self):
        from kai.cron import _job_callback

        claude_a = _make_mock_claude()
        claude_b = _make_mock_claude()
        instances = {USER_A: claude_a, USER_B: claude_b}
        manager = MagicMock()
        manager.get_or_create = AsyncMock(side_effect=lambda uid: instances[uid])

        # Set up a done event for the stream
        done = _done_event(text="reminder sent")

        async def fake_stream(prompt):
            yield done

        claude_a.send = MagicMock(side_effect=fake_stream)

        context = MagicMock()
        context.job = MagicMock()
        context.job.data = {
            "job_id": 1,
            "user_id": USER_A,
            "chat_id": USER_A,
            "job_type": "claude",
            "prompt": "check something",
            "auto_remove": False,
            "notify_on_check": False,
            "name": "test-job",
            "schedule_type": "interval",
        }
        context.bot_data = {"claude_manager": manager}
        context.bot.send_chat_action = AsyncMock()
        context.bot.send_message = AsyncMock()

        with (
            patch("kai.cron.log_message", new_callable=AsyncMock),
            patch("kai.cron.get_lock", return_value=_fake_lock()),
            patch("kai.cron.sessions.save_session", new_callable=AsyncMock),
        ):
            await _job_callback(context)

        manager.get_or_create.assert_called_with(USER_A)
        claude_a.send.assert_called_once()
        claude_b.send.assert_not_called()


# ── File cleanup ──────────────────────────────────────────────────


class TestCleanupOldFiles:
    """Tests for ClaudeManager.cleanup_old_files."""

    @pytest.fixture
    def manager_with_files(self, tmp_path):
        """Create a ClaudeManager with tmp DATA_DIR and sample files."""
        config = MagicMock()
        config.claude_workspace = tmp_path / "workspace"
        config.claude_workspace.mkdir()
        mgr = ClaudeManager(config=config)
        return mgr, tmp_path

    @pytest.mark.asyncio
    async def test_removes_old_files(self, manager_with_files):
        """Files older than max_age_days are removed."""
        mgr, tmp_path = manager_with_files
        import os
        import time

        files_dir = tmp_path / "users" / "42" / "home" / "files"
        files_dir.mkdir(parents=True)
        old_file = files_dir / "old_photo.jpg"
        old_file.write_bytes(b"old")
        # Set mtime to 10 days ago
        old_time = time.time() - (10 * 86400)
        os.utime(old_file, (old_time, old_time))

        with patch("kai.claude.DATA_DIR", tmp_path):
            removed = await mgr.cleanup_old_files(max_age_days=7)

        assert removed == 1
        assert not old_file.exists()

    @pytest.mark.asyncio
    async def test_keeps_recent_files(self, manager_with_files):
        """Files newer than max_age_days are kept."""
        mgr, tmp_path = manager_with_files
        files_dir = tmp_path / "users" / "42" / "home" / "files"
        files_dir.mkdir(parents=True)
        new_file = files_dir / "new_photo.jpg"
        new_file.write_bytes(b"new")

        with patch("kai.claude.DATA_DIR", tmp_path):
            removed = await mgr.cleanup_old_files(max_age_days=7)

        assert removed == 0
        assert new_file.exists()

    @pytest.mark.asyncio
    async def test_handles_missing_users_dir(self, manager_with_files):
        """Returns 0 when users/ directory doesn't exist."""
        mgr, tmp_path = manager_with_files
        with patch("kai.claude.DATA_DIR", tmp_path):
            removed = await mgr.cleanup_old_files()
        assert removed == 0
