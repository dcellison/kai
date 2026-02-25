"""Integration tests for webhook HTTP API endpoints (jobs CRUD, file exchange)."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web

import kai.webhook as webhook_mod
from kai import sessions
from kai.webhook import (
    _handle_delete_job,
    _handle_schedule,
    _handle_send_file,
    _handle_telegram_update,
    _handle_update_job,
    update_workspace,
)


@pytest.fixture
async def db(tmp_path):
    """Initialize a fresh database for each test."""
    await sessions.init_db(tmp_path / "test.db")
    yield
    await sessions.close_db()


@pytest.fixture
def mock_request():
    """Create a minimal mock request with app dict and helpers."""
    request = MagicMock(spec=web.Request)
    request.app = {
        "webhook_secret": "test-secret",
        "telegram_app": MagicMock(),
    }
    # Mock the job_queue on the telegram app
    job_queue = MagicMock()
    job_queue.jobs = MagicMock(return_value=[])
    request.app["telegram_app"].job_queue = job_queue
    request.headers = {}
    request.match_info = {}
    return request


# ── POST /api/schedule ────────────────────────────────────────────────


class TestScheduleJobType:
    async def test_invalid_job_type_returns_400(self, db, mock_request):
        """Schedule endpoint rejects unrecognized job_type values."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123
        mock_request.json = AsyncMock(
            return_value={
                "name": "test",
                "prompt": "test",
                "schedule_type": "once",
                "schedule_data": {"run_at": "2026-02-20T10:00:00+00:00"},
                "job_type": "invalid",
            }
        )

        resp = await _handle_schedule(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "error" in body
        assert "job_type" in body["error"]

    async def test_valid_job_type_accepted(self, db, mock_request):
        """Schedule endpoint accepts valid job_type values without error."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123
        mock_request.app["telegram_app"].job_queue = MagicMock()

        # Mock register_job_by_id so we don't need a full APScheduler setup
        import kai.cron as cron_mod

        cron_mod.register_job_by_id = AsyncMock()

        mock_request.json = AsyncMock(
            return_value={
                "name": "test claude job",
                "prompt": "test",
                "schedule_type": "once",
                "schedule_data": {"run_at": "2026-02-20T10:00:00+00:00"},
                "job_type": "claude",
            }
        )

        resp = await _handle_schedule(mock_request)

        assert resp.status == 200
        body = json.loads(resp.body.decode())
        assert "job_id" in body


# ── DELETE /api/jobs/{id} ────────────────────────────────────────────


class TestDeleteJob:
    async def test_delete_existing_job(self, db, mock_request):
        """DELETE handler removes a job and returns 200."""
        job_id = await sessions.create_job(
            chat_id=123,
            name="test job",
            job_type="reminder",
            prompt="test prompt",
            schedule_type="once",
            schedule_data='{"run_at": "2026-02-20T10:00:00+00:00"}',
        )

        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.match_info = {"id": str(job_id)}

        resp = await _handle_delete_job(mock_request)

        assert resp.status == 200
        assert resp.content_type == "application/json"
        # Parse the JSON from the response body
        body = json.loads(resp.body.decode())
        assert body == {"deleted": job_id}

        # Verify job was actually deleted from database
        job = await sessions.get_job_by_id(job_id)
        assert job is None

    async def test_delete_nonexistent_job_returns_404(self, db, mock_request):
        """DELETE handler returns 404 for nonexistent job."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.match_info = {"id": "999"}

        resp = await _handle_delete_job(mock_request)

        assert resp.status == 404
        body = json.loads(resp.body.decode())
        assert "error" in body
        assert "not found" in body["error"].lower()

    async def test_delete_invalid_job_id_returns_400(self, db, mock_request):
        """DELETE handler returns 400 for non-numeric ID."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.match_info = {"id": "not-a-number"}

        resp = await _handle_delete_job(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "error" in body
        assert "invalid" in body["error"].lower()

    async def test_delete_missing_secret_returns_401(self, db, mock_request):
        """DELETE handler returns 401 without webhook secret."""
        mock_request.headers = {}
        mock_request.match_info = {"id": "1"}

        resp = await _handle_delete_job(mock_request)

        assert resp.status == 401


# ── PATCH /api/jobs/{id} ─────────────────────────────────────────────


class TestUpdateJob:
    async def test_update_name_only(self, db, mock_request):
        """PATCH handler updates only the name field."""
        job_id = await sessions.create_job(
            chat_id=123,
            name="original name",
            job_type="reminder",
            prompt="original prompt",
            schedule_type="once",
            schedule_data='{"run_at": "2026-02-20T10:00:00+00:00"}',
        )

        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.match_info = {"id": str(job_id)}
        # Mock the json() method to return the payload
        mock_request.json = AsyncMock(return_value={"name": "updated name"})

        resp = await _handle_update_job(mock_request)

        assert resp.status == 200
        body = json.loads(resp.body.decode())
        assert body == {"updated": job_id}

        # Verify only name changed
        job = await sessions.get_job_by_id(job_id)
        assert job is not None
        assert job["name"] == "updated name"
        assert job["prompt"] == "original prompt"

    async def test_update_multiple_fields(self, db, mock_request):
        """PATCH handler updates multiple fields at once."""
        job_id = await sessions.create_job(
            chat_id=123,
            name="old name",
            job_type="claude",
            prompt="old prompt",
            schedule_type="interval",
            schedule_data='{"seconds": 3600}',
            auto_remove=False,
        )

        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.match_info = {"id": str(job_id)}
        mock_request.json = AsyncMock(
            return_value={
                "name": "new name",
                "prompt": "new prompt",
                "auto_remove": True,
            }
        )

        resp = await _handle_update_job(mock_request)

        assert resp.status == 200
        job = await sessions.get_job_by_id(job_id)
        assert job is not None
        assert job["name"] == "new name"
        assert job["prompt"] == "new prompt"
        assert job["auto_remove"] is True

    async def test_update_nonexistent_job_returns_404(self, db, mock_request):
        """PATCH handler returns 404 for nonexistent job."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.match_info = {"id": "999"}
        mock_request.json = AsyncMock(return_value={"name": "new name"})

        resp = await _handle_update_job(mock_request)

        assert resp.status == 404

    async def test_update_invalid_schedule_type_returns_400(self, db, mock_request):
        """PATCH handler returns 400 for invalid schedule_type."""
        job_id = await sessions.create_job(
            chat_id=123,
            name="test job",
            job_type="reminder",
            prompt="test",
            schedule_type="once",
            schedule_data="{}",
        )

        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.match_info = {"id": str(job_id)}
        mock_request.json = AsyncMock(return_value={"schedule_type": "invalid"})

        resp = await _handle_update_job(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "error" in body
        assert "schedule_type" in body["error"]

    async def test_update_empty_body_returns_404(self, db, mock_request):
        """PATCH handler with empty body returns 404 (no fields to update)."""
        job_id = await sessions.create_job(
            chat_id=123,
            name="test job",
            job_type="reminder",
            prompt="test",
            schedule_type="once",
            schedule_data="{}",
        )

        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.match_info = {"id": str(job_id)}
        mock_request.json = AsyncMock(return_value={})

        resp = await _handle_update_job(mock_request)

        # Empty update returns 404 because update_job returns False
        assert resp.status == 404

    async def test_update_missing_secret_returns_401(self, db, mock_request):
        """PATCH handler returns 401 without webhook secret."""
        mock_request.headers = {}
        mock_request.match_info = {"id": "1"}
        mock_request.json = AsyncMock(return_value={"name": "new"})

        resp = await _handle_update_job(mock_request)

        assert resp.status == 401

    async def test_update_invalid_json_returns_400(self, db, mock_request):
        """PATCH handler returns 400 for malformed JSON."""
        from json import JSONDecodeError

        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.match_info = {"id": "1"}
        mock_request.json = AsyncMock(side_effect=JSONDecodeError("test", "doc", 0))

        resp = await _handle_update_job(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "error" in body
        assert "json" in body["error"].lower()


# ── POST /api/send-file ─────────────────────────────────────────────


@pytest.fixture
def send_file_request(tmp_path):
    """Create a mock request for the send-file endpoint with workspace confinement."""
    request = MagicMock(spec=web.Request)
    request.app = {
        "webhook_secret": "test-secret",
        "telegram_bot": AsyncMock(),
        "chat_id": 123,
        "workspace": str(tmp_path),
    }
    request.headers = {"X-Webhook-Secret": "test-secret"}
    return request


class TestSendFile:
    async def test_send_image_as_photo(self, tmp_path, send_file_request):
        """Image files are sent via send_photo (rendered inline in Telegram)."""
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")

        send_file_request.json = AsyncMock(return_value={"path": str(img)})
        resp = await _handle_send_file(send_file_request)

        assert resp.status == 200
        body = json.loads(resp.body)
        assert body["status"] == "sent"
        assert body["file"] == "photo.jpg"
        send_file_request.app["telegram_bot"].send_photo.assert_called_once()

    async def test_send_document(self, tmp_path, send_file_request):
        """Non-image files are sent via send_document (as attachments)."""
        doc = tmp_path / "report.pdf"
        doc.write_bytes(b"%PDF-1.4 fake")

        send_file_request.json = AsyncMock(return_value={"path": str(doc)})
        resp = await _handle_send_file(send_file_request)

        assert resp.status == 200
        body = json.loads(resp.body)
        assert body["status"] == "sent"
        send_file_request.app["telegram_bot"].send_document.assert_called_once()

    async def test_caption_forwarded_to_telegram(self, tmp_path, send_file_request):
        """Optional caption is passed through to the Telegram send call."""
        f = tmp_path / "pic.png"
        f.write_bytes(b"fake-png")

        send_file_request.json = AsyncMock(return_value={"path": str(f), "caption": "Here you go"})
        resp = await _handle_send_file(send_file_request)

        assert resp.status == 200
        call_kwargs = send_file_request.app["telegram_bot"].send_photo.call_args
        assert call_kwargs[1].get("caption") == "Here you go"

    async def test_missing_path_returns_400(self, send_file_request):
        """Returns 400 when the required path field is absent."""
        send_file_request.json = AsyncMock(return_value={})
        resp = await _handle_send_file(send_file_request)
        assert resp.status == 400

    async def test_file_not_found_returns_404(self, tmp_path, send_file_request):
        """Returns 404 when the file doesn't exist on disk."""
        send_file_request.json = AsyncMock(return_value={"path": str(tmp_path / "nonexistent.txt")})
        resp = await _handle_send_file(send_file_request)
        assert resp.status == 404

    async def test_path_outside_workspace_returns_403(self, send_file_request):
        """Returns 403 for paths that escape the workspace via traversal."""
        send_file_request.json = AsyncMock(return_value={"path": "/etc/passwd"})
        resp = await _handle_send_file(send_file_request)
        assert resp.status == 403

    async def test_invalid_json_returns_400(self, send_file_request):
        """Returns 400 for malformed JSON body."""
        send_file_request.json = AsyncMock(side_effect=json.JSONDecodeError("test", "doc", 0))
        resp = await _handle_send_file(send_file_request)
        assert resp.status == 400

    async def test_missing_secret_returns_401(self, send_file_request):
        """Returns 401 without a valid webhook secret."""
        send_file_request.headers = {}
        send_file_request.json = AsyncMock(return_value={"path": "/any"})
        resp = await _handle_send_file(send_file_request)
        assert resp.status == 401


# ── update_workspace() ──────────────────────────────────────────────


class TestUpdateWorkspace:
    def test_updates_app_workspace(self, monkeypatch):
        """update_workspace() changes the path stored in the live aiohttp app dict."""
        # Simulate a running server by setting _app to a real Application instance
        app = web.Application()
        app["workspace"] = "/original/workspace"
        monkeypatch.setattr(webhook_mod, "_app", app)

        update_workspace("/switched/workspace")

        assert app["workspace"] == "/switched/workspace"

    def test_no_op_when_server_not_running(self, monkeypatch):
        """update_workspace() is a no-op (no exception) when the server hasn't started."""
        monkeypatch.setattr(webhook_mod, "_app", None)
        # Should not raise
        update_workspace("/any/path")


# ── POST /webhook/telegram ─────────────────────────────────────────


@pytest.fixture
def telegram_request():
    """Create a mock request for the Telegram webhook endpoint."""
    request = MagicMock(spec=web.Request)
    request.app = {
        "telegram_webhook_secret": "tg-secret",
        "telegram_app": MagicMock(),
        "telegram_bot": MagicMock(),
    }
    request.app["telegram_app"].process_update = AsyncMock()
    request.headers = {"X-Telegram-Bot-Api-Secret-Token": "tg-secret"}
    return request


class TestTelegramUpdate:
    async def test_valid_secret_dispatches_update(self, telegram_request, monkeypatch):
        """Valid secret and JSON body dispatches to process_update."""
        fake_update = MagicMock()
        monkeypatch.setattr("kai.webhook.Update.de_json", MagicMock(return_value=fake_update))
        telegram_request.json = AsyncMock(return_value={"update_id": 123})

        resp = await _handle_telegram_update(telegram_request)

        assert resp.status == 200
        # process_update runs as a background task (fire-and-forget to avoid
        # Telegram's webhook timeout). Yield to the event loop so the task
        # actually executes before we assert.
        await asyncio.sleep(0)
        telegram_request.app["telegram_app"].process_update.assert_called_once_with(fake_update)

    async def test_wrong_secret_returns_401(self, telegram_request):
        """Wrong secret token returns 401 without dispatching."""
        telegram_request.headers = {"X-Telegram-Bot-Api-Secret-Token": "wrong"}

        resp = await _handle_telegram_update(telegram_request)

        assert resp.status == 401
        telegram_request.app["telegram_app"].process_update.assert_not_called()

    async def test_missing_secret_returns_401(self, telegram_request):
        """Missing secret header returns 401."""
        telegram_request.headers = {}

        resp = await _handle_telegram_update(telegram_request)

        assert resp.status == 401

    async def test_malformed_json_returns_200(self, telegram_request):
        """Malformed JSON returns 200 (swallowed to prevent Telegram retries)."""
        telegram_request.json = AsyncMock(side_effect=json.JSONDecodeError("test", "doc", 0))

        resp = await _handle_telegram_update(telegram_request)

        assert resp.status == 200
        telegram_request.app["telegram_app"].process_update.assert_not_called()

    async def test_null_update_skips_dispatch(self, telegram_request, monkeypatch):
        """If Update.de_json returns None, process_update is not called."""
        monkeypatch.setattr("kai.webhook.Update.de_json", MagicMock(return_value=None))
        telegram_request.json = AsyncMock(return_value={"update_id": 999})

        resp = await _handle_telegram_update(telegram_request)

        assert resp.status == 200
        telegram_request.app["telegram_app"].process_update.assert_not_called()
