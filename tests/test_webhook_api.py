"""Integration tests for webhook HTTP API endpoints (jobs CRUD, file exchange)."""

import asyncio
import hashlib
import hmac as hmac_mod
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

import kai.webhook as webhook_mod
from kai import sessions
from kai.services import ServiceResponse
from kai.webhook import (
    _handle_delete_job,
    _handle_generic,
    _handle_get_job,
    _handle_get_jobs,
    _handle_github,
    _handle_schedule,
    _handle_send_file,
    _handle_send_message,
    _handle_service_call,
    _handle_telegram_update,
    _handle_update_job,
    _resolve_chat_id,
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
        "telegram_bot": AsyncMock(),
        "chat_id": 123,
        "allowed_user_ids": {123, 456},
    }
    # Mock the job_queue on the telegram app
    job_queue = MagicMock()
    job_queue.jobs = MagicMock(return_value=[])
    request.app["telegram_app"].job_queue = job_queue
    request.headers = {}
    request.match_info = {}
    # Multidict-like query object for GET parameter access
    request.query = {}
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


# ── POST /api/send-message ────────────────────────────────────────────


@pytest.fixture
def send_message_request():
    """Create a mock request for the send-message endpoint."""
    request = MagicMock(spec=web.Request)
    request.app = {
        "webhook_secret": "test-secret",
        "telegram_bot": AsyncMock(),
        "chat_id": 123,
    }
    request.headers = {"X-Webhook-Secret": "test-secret"}
    return request


class TestSendMessage:
    async def test_sends_short_message(self, send_message_request):
        """Short messages are sent as a single Telegram message."""
        send_message_request.json = AsyncMock(return_value={"text": "Hello!"})
        resp = await _handle_send_message(send_message_request)

        assert resp.status == 200
        body = json.loads(resp.body)
        assert body["status"] == "sent"
        send_message_request.app["telegram_bot"].send_message.assert_called_once_with(123, "Hello!")

    async def test_splits_long_message(self, send_message_request):
        """Messages exceeding 4096 chars are split into multiple sends."""
        # Create a message with two paragraphs, each over 2048 chars
        long_text = ("A" * 2100) + "\n\n" + ("B" * 2100)
        send_message_request.json = AsyncMock(return_value={"text": long_text})
        resp = await _handle_send_message(send_message_request)

        assert resp.status == 200
        bot = send_message_request.app["telegram_bot"]
        assert bot.send_message.call_count == 2

    async def test_missing_text_returns_400(self, send_message_request):
        """Returns 400 when the required text field is absent."""
        send_message_request.json = AsyncMock(return_value={})
        resp = await _handle_send_message(send_message_request)
        assert resp.status == 400

    async def test_empty_text_returns_400(self, send_message_request):
        """Returns 400 when text is an empty string."""
        send_message_request.json = AsyncMock(return_value={"text": "   "})
        resp = await _handle_send_message(send_message_request)
        assert resp.status == 400

    async def test_invalid_json_returns_400(self, send_message_request):
        """Returns 400 for malformed JSON body."""
        send_message_request.json = AsyncMock(side_effect=json.JSONDecodeError("test", "doc", 0))
        resp = await _handle_send_message(send_message_request)
        assert resp.status == 400

    async def test_missing_secret_returns_401(self, send_message_request):
        """Returns 401 without a valid webhook secret."""
        send_message_request.headers = {}
        send_message_request.json = AsyncMock(return_value={"text": "Hello"})
        resp = await _handle_send_message(send_message_request)
        assert resp.status == 401

    async def test_telegram_error_returns_500(self, send_message_request):
        """Returns 500 when the Telegram send fails."""
        send_message_request.json = AsyncMock(return_value={"text": "Hello"})
        send_message_request.app["telegram_bot"].send_message = AsyncMock(side_effect=RuntimeError("Boom"))
        resp = await _handle_send_message(send_message_request)
        assert resp.status == 500


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


# ── GitHub webhook helpers ─────────────────────────────────────────


def _sign_body(secret: str, body: bytes) -> str:
    """Compute a valid GitHub HMAC-SHA256 signature for test payloads."""
    digest = hmac_mod.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


@pytest.fixture()
def github_request():
    """Create a mock request for the GitHub webhook endpoint.

    Includes a mock config with user_configs=None (no per-user routing)
    and mocks resolve_github_settings to return defaults. This simulates
    fallback routing where events go to the admin chat_id.
    """
    mock_config = MagicMock()
    mock_config.user_configs = None
    request = MagicMock(spec=web.Request)
    request.app = {
        "webhook_secret": "test-secret",
        "telegram_bot": AsyncMock(),
        "chat_id": 12345,
        "config": mock_config,
        "pr_review_cooldown": 300,
    }
    request.headers = {}
    return request


def _github_push_payload() -> dict:
    """Minimal GitHub push event payload for testing."""
    return {
        "pusher": {"name": "testuser"},
        "ref": "refs/heads/main",
        "commits": [{"id": "abc1234def5678", "message": "Fix bug"}],
        "repository": {"full_name": "testuser/repo"},
        "compare": "https://github.com/testuser/repo/compare/abc...def",
    }


# ── POST /webhook/github ──────────────────────────────────────────


class TestGitHubWebhook:
    @pytest.fixture(autouse=True)
    def _mock_github_settings(self):
        """Mock resolve_github_settings for all GitHub webhook tests.

        Returns default settings (no review, no triage, admin chat_id)
        so the standard notification path fires for push/issues events.
        """
        settings = {
            "repos": [],
            "notify_chat_id": 12345,
            "pr_review": False,
            "issue_triage": False,
        }
        with patch(
            "kai.webhook.sessions.resolve_github_settings",
            new_callable=AsyncMock,
            return_value=settings,
        ):
            yield

    async def test_valid_push_sends_markdown(self, github_request):
        """Valid signature + push event sends a Markdown-formatted message."""
        payload = _github_push_payload()
        body = json.dumps(payload).encode()
        github_request.read = AsyncMock(return_value=body)
        github_request.headers = {
            "X-Hub-Signature-256": _sign_body("test-secret", body),
            "X-GitHub-Event": "push",
        }

        resp = await _handle_github(github_request)

        assert resp.status == 200
        bot = github_request.app["telegram_bot"]
        bot.send_message.assert_called_once()
        call_kwargs = bot.send_message.call_args
        assert call_kwargs.kwargs.get("parse_mode") == "Markdown" or call_kwargs[2] == "Markdown"

    async def test_markdown_failure_falls_back_to_plain(self, github_request):
        """When Markdown parse fails, resends as stripped plain text."""
        payload = _github_push_payload()
        body = json.dumps(payload).encode()
        github_request.read = AsyncMock(return_value=body)
        github_request.headers = {
            "X-Hub-Signature-256": _sign_body("test-secret", body),
            "X-GitHub-Event": "push",
        }
        bot = github_request.app["telegram_bot"]
        # First call (Markdown) fails, second call (plain) succeeds
        bot.send_message = AsyncMock(side_effect=[Exception("parse error"), None])

        resp = await _handle_github(github_request)

        assert resp.status == 200
        assert bot.send_message.call_count == 2

    async def test_both_sends_fail_logs_error(self, github_request):
        """When both Markdown and plain text fail, error is logged but HTTP returns ok.

        Per-user routing handles send failures per-user (logged, not
        surfaced in HTTP response) since GitHub doesn't retry based on
        response codes anyway.
        """
        payload = _github_push_payload()
        body = json.dumps(payload).encode()
        github_request.read = AsyncMock(return_value=body)
        github_request.headers = {
            "X-Hub-Signature-256": _sign_body("test-secret", body),
            "X-GitHub-Event": "push",
        }
        bot = github_request.app["telegram_bot"]
        bot.send_message = AsyncMock(side_effect=Exception("always fails"))

        resp = await _handle_github(github_request)

        # Both sends failed, but HTTP response is still ok (error logged)
        body_json = json.loads(resp.body.decode())
        assert body_json["status"] == "ok"

    async def test_invalid_signature_returns_401(self, github_request):
        """Requests with an invalid HMAC signature are rejected."""
        body = b'{"any": "payload"}'
        github_request.read = AsyncMock(return_value=body)
        github_request.headers = {
            "X-Hub-Signature-256": "sha256=invalid",
            "X-GitHub-Event": "push",
        }

        resp = await _handle_github(github_request)

        assert resp.status == 401
        github_request.app["telegram_bot"].send_message.assert_not_called()

    async def test_ping_event_returns_pong(self, github_request):
        """GitHub ping events are acknowledged without sending to Telegram."""
        body = b'{"zen": "testing"}'
        github_request.read = AsyncMock(return_value=body)
        github_request.headers = {
            "X-Hub-Signature-256": _sign_body("test-secret", body),
            "X-GitHub-Event": "ping",
        }

        resp = await _handle_github(github_request)

        body_json = json.loads(resp.body.decode())
        assert body_json["msg"] == "pong"
        github_request.app["telegram_bot"].send_message.assert_not_called()

    async def test_unknown_event_type_ignored(self, github_request):
        """Unsupported event types (e.g. 'star') are silently acknowledged."""
        payload = {"action": "created"}
        body = json.dumps(payload).encode()
        github_request.read = AsyncMock(return_value=body)
        github_request.headers = {
            "X-Hub-Signature-256": _sign_body("test-secret", body),
            "X-GitHub-Event": "star",
        }

        resp = await _handle_github(github_request)

        body_json = json.loads(resp.body.decode())
        # Per-user routing always returns "ok" - the event is still
        # silently dropped (no formatter, no notification sent)
        assert body_json["status"] == "ok"
        github_request.app["telegram_bot"].send_message.assert_not_called()

    async def test_filtered_action_ignored(self, github_request):
        """Known event type with filtered action (e.g. PR 'edited') is ignored."""
        # PR "edited" is not in the formatter's accepted actions
        payload = {"action": "edited", "pull_request": {"title": "test"}}
        body = json.dumps(payload).encode()
        github_request.read = AsyncMock(return_value=body)
        github_request.headers = {
            "X-Hub-Signature-256": _sign_body("test-secret", body),
            "X-GitHub-Event": "pull_request",
        }

        resp = await _handle_github(github_request)

        body_json = json.loads(resp.body.decode())
        # Per-user routing always returns "ok" - the formatter returns
        # None for "edited" so no notification is sent
        assert body_json["status"] == "ok"
        github_request.app["telegram_bot"].send_message.assert_not_called()

    async def test_invalid_json_after_valid_signature_returns_400(self, github_request):
        """Valid signature over malformed JSON body returns 400."""
        body = b"not valid json"
        github_request.read = AsyncMock(return_value=body)
        github_request.headers = {
            "X-Hub-Signature-256": _sign_body("test-secret", body),
            # Use a known event type so JSON parsing is attempted
            "X-GitHub-Event": "push",
        }

        resp = await _handle_github(github_request)

        assert resp.status == 400


# ── POST /webhook (generic) ───────────────────────────────────────


@pytest.fixture()
def generic_request():
    """Create a mock request for the generic webhook endpoint."""
    request = MagicMock(spec=web.Request)
    request.app = {
        "webhook_secret": "test-secret",
        "telegram_bot": AsyncMock(),
        "chat_id": 12345,
    }
    request.headers = {"X-Webhook-Secret": "test-secret"}
    return request


class TestGenericWebhook:
    async def test_sends_message_field(self, generic_request):
        """Payload with a 'message' field sends that string to Telegram."""
        generic_request.json = AsyncMock(return_value={"message": "Alert: disk full"})

        resp = await _handle_generic(generic_request)

        assert resp.status == 200
        generic_request.app["telegram_bot"].send_message.assert_called_once_with(12345, "Alert: disk full")

    async def test_dumps_full_payload_when_no_message(self, generic_request):
        """Payload without 'message' sends the full JSON dump to Telegram."""
        payload = {"key": "value", "count": 42}
        generic_request.json = AsyncMock(return_value=payload)

        resp = await _handle_generic(generic_request)

        assert resp.status == 200
        sent_text = generic_request.app["telegram_bot"].send_message.call_args[0][1]
        # Should be a pretty-printed JSON dump
        assert '"key": "value"' in sent_text
        assert '"count": 42' in sent_text

    async def test_empty_message_field_sends_empty_string(self, generic_request):
        """Empty string 'message' is sent as-is (not treated as missing)."""
        generic_request.json = AsyncMock(return_value={"message": ""})

        resp = await _handle_generic(generic_request)

        assert resp.status == 200
        sent_text = generic_request.app["telegram_bot"].send_message.call_args[0][1]
        assert sent_text == ""

    async def test_long_message_truncated(self, generic_request):
        """Messages over 4096 chars are truncated with '...' suffix."""
        long_msg = "x" * 5000
        generic_request.json = AsyncMock(return_value={"message": long_msg})

        resp = await _handle_generic(generic_request)

        assert resp.status == 200
        sent_text = generic_request.app["telegram_bot"].send_message.call_args[0][1]
        assert len(sent_text) == 4096
        assert sent_text.endswith("...")

    async def test_invalid_json_returns_400(self, generic_request):
        """Malformed JSON body returns 400."""
        generic_request.json = AsyncMock(side_effect=json.JSONDecodeError("test", "doc", 0))

        resp = await _handle_generic(generic_request)

        assert resp.status == 400

    async def test_send_failure_still_returns_ok(self, generic_request):
        """Telegram send failures are logged but the response is still 200/ok."""
        generic_request.json = AsyncMock(return_value={"message": "test"})
        generic_request.app["telegram_bot"].send_message = AsyncMock(side_effect=RuntimeError("network error"))

        resp = await _handle_generic(generic_request)

        assert resp.status == 200
        body_json = json.loads(resp.body.decode())
        assert body_json["status"] == "ok"

    async def test_missing_secret_returns_401(self, generic_request):
        """Missing webhook secret header returns 401."""
        generic_request.headers = {}
        generic_request.json = AsyncMock(return_value={"message": "test"})

        resp = await _handle_generic(generic_request)

        assert resp.status == 401


# ── GET /api/jobs ──────────────────────────────────────────────────


class TestGetJobs:
    async def test_returns_active_jobs(self, db, mock_request):
        """Returns a list of active jobs for the configured chat."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123

        await sessions.create_job(
            chat_id=123,
            name="Job A",
            job_type="reminder",
            prompt="hello",
            schedule_type="daily",
            schedule_data='{"times": ["09:00"]}',
        )
        await sessions.create_job(
            chat_id=123,
            name="Job B",
            job_type="claude",
            prompt="check",
            schedule_type="interval",
            schedule_data='{"seconds": 3600}',
        )

        resp = await _handle_get_jobs(mock_request)

        assert resp.status == 200
        body = json.loads(resp.body.decode())
        assert len(body) == 2
        names = {j["name"] for j in body}
        assert names == {"Job A", "Job B"}

    async def test_returns_empty_list_when_no_jobs(self, db, mock_request):
        """Returns an empty list when no jobs exist."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123

        resp = await _handle_get_jobs(mock_request)

        assert resp.status == 200
        body = json.loads(resp.body.decode())
        assert body == []

    async def test_missing_secret_returns_401(self, db, mock_request):
        """Missing webhook secret returns 401."""
        mock_request.headers = {}
        mock_request.app["chat_id"] = 123

        resp = await _handle_get_jobs(mock_request)

        assert resp.status == 401


# ── GET /api/jobs/{id} ─────────────────────────────────────────────


class TestGetJob:
    async def test_returns_existing_job(self, db, mock_request):
        """Returns the full job record for a valid ID."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        job_id = await sessions.create_job(
            chat_id=123,
            name="My Job",
            job_type="reminder",
            prompt="test prompt",
            schedule_type="once",
            schedule_data='{"run_at": "2026-06-01T12:00:00+00:00"}',
        )
        mock_request.match_info = {"id": str(job_id)}

        resp = await _handle_get_job(mock_request)

        assert resp.status == 200
        body = json.loads(resp.body.decode())
        assert body["name"] == "My Job"
        assert body["id"] == job_id

    async def test_nonexistent_job_returns_404(self, db, mock_request):
        """Returns 404 for a job ID that doesn't exist."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.match_info = {"id": "999"}

        resp = await _handle_get_job(mock_request)

        assert resp.status == 404

    async def test_invalid_id_returns_400(self, db, mock_request):
        """Returns 400 for a non-numeric job ID."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.match_info = {"id": "abc"}

        resp = await _handle_get_job(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "invalid" in body["error"].lower()

    async def test_missing_secret_returns_401(self, db, mock_request):
        """Missing webhook secret returns 401."""
        mock_request.headers = {}
        mock_request.match_info = {"id": "1"}

        resp = await _handle_get_job(mock_request)

        assert resp.status == 401


# ── POST /api/schedule (additional coverage) ───────────────────────


class TestScheduleValidation:
    async def test_missing_required_fields_returns_400(self, db, mock_request):
        """Returns 400 when required fields are missing."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123
        # Missing prompt, schedule_type, and schedule_data
        mock_request.json = AsyncMock(return_value={"name": "incomplete"})

        resp = await _handle_schedule(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "required" in body["error"].lower()

    async def test_invalid_schedule_type_returns_400(self, db, mock_request):
        """Returns 400 for unrecognized schedule_type."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123
        mock_request.json = AsyncMock(
            return_value={
                "name": "test",
                "prompt": "test",
                "schedule_type": "weekly",
                "schedule_data": {},
            }
        )

        resp = await _handle_schedule(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "schedule_type" in body["error"]

    async def test_dict_schedule_data_serialized_to_json(self, db, mock_request):
        """schedule_data as a dict is serialized to a JSON string for DB storage."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123
        mock_request.json = AsyncMock(
            return_value={
                "name": "dict test",
                "prompt": "test",
                "schedule_type": "interval",
                "schedule_data": {"seconds": 600},
            }
        )
        with patch("kai.cron.register_job_by_id", new_callable=AsyncMock):
            resp = await _handle_schedule(mock_request)

        assert resp.status == 200
        body = json.loads(resp.body.decode())
        # Verify the stored data is valid JSON
        job = await sessions.get_job_by_id(body["job_id"])
        assert job is not None
        stored = json.loads(job["schedule_data"])
        assert stored["seconds"] == 600

    async def test_string_schedule_data_passed_through(self, db, mock_request):
        """schedule_data as a pre-serialized string is stored as-is."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123
        mock_request.json = AsyncMock(
            return_value={
                "name": "string test",
                "prompt": "test",
                "schedule_type": "interval",
                "schedule_data": '{"seconds": 900}',
            }
        )
        with patch("kai.cron.register_job_by_id", new_callable=AsyncMock):
            resp = await _handle_schedule(mock_request)

        assert resp.status == 200
        body = json.loads(resp.body.decode())
        job = await sessions.get_job_by_id(body["job_id"])
        assert job is not None
        assert job["schedule_data"] == '{"seconds": 900}'

    async def test_invalid_string_schedule_data_returns_400(self, db, mock_request):
        """schedule_data as a non-JSON string is rejected with 400."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123
        mock_request.json = AsyncMock(
            return_value={
                "name": "bad data",
                "prompt": "test",
                "schedule_type": "interval",
                "schedule_data": "not json at all",
            }
        )

        resp = await _handle_schedule(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "not valid JSON" in body["error"]

    async def test_defaults_for_optional_fields(self, db, mock_request):
        """auto_remove defaults to False when omitted. job_type defaults to 'reminder'."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123
        mock_request.json = AsyncMock(
            return_value={
                "name": "defaults test",
                "prompt": "test",
                "schedule_type": "once",
                "schedule_data": {"run_at": "2026-06-01T12:00:00+00:00"},
            }
        )
        with patch("kai.cron.register_job_by_id", new_callable=AsyncMock):
            resp = await _handle_schedule(mock_request)

        assert resp.status == 200
        body = json.loads(resp.body.decode())
        job = await sessions.get_job_by_id(body["job_id"])
        assert job is not None
        assert job["auto_remove"] is False
        assert job["job_type"] == "reminder"

    async def test_db_failure_returns_500(self, db, mock_request):
        """Database create failure returns 500 with an error message."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123
        mock_request.json = AsyncMock(
            return_value={
                "name": "fail test",
                "prompt": "test",
                "schedule_type": "daily",
                "schedule_data": {"times": ["09:00"]},
            }
        )
        with patch(
            "kai.webhook.sessions.create_job",
            new_callable=AsyncMock,
            side_effect=RuntimeError("DB locked"),
        ):
            resp = await _handle_schedule(mock_request)

        assert resp.status == 500
        body = json.loads(resp.body.decode())
        assert "error" in body

    async def test_successful_creation_registers_with_scheduler(self, db, mock_request):
        """Successful job creation calls register_job_by_id with the new ID."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123
        mock_request.json = AsyncMock(
            return_value={
                "name": "scheduler test",
                "prompt": "test",
                "schedule_type": "interval",
                "schedule_data": {"seconds": 300},
            }
        )
        with patch("kai.cron.register_job_by_id", new_callable=AsyncMock) as mock_register:
            resp = await _handle_schedule(mock_request)

        assert resp.status == 200
        body = json.loads(resp.body.decode())
        mock_register.assert_called_once_with(mock_request.app["telegram_app"], body["job_id"])

    async def test_invalid_json_returns_400(self, db, mock_request):
        """Malformed JSON body returns 400."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123
        mock_request.json = AsyncMock(side_effect=json.JSONDecodeError("test", "doc", 0))

        resp = await _handle_schedule(mock_request)

        assert resp.status == 400

    async def test_once_missing_run_at_returns_400(self, db, mock_request):
        """schedule_data for 'once' without 'run_at' is rejected."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123
        mock_request.json = AsyncMock(
            return_value={
                "name": "bad once",
                "prompt": "test",
                "schedule_type": "once",
                "schedule_data": {"times": ["08:00"]},
            }
        )

        resp = await _handle_schedule(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "run_at" in body["error"]

    async def test_daily_missing_times_returns_400(self, db, mock_request):
        """schedule_data for 'daily' without 'times' is rejected."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123
        mock_request.json = AsyncMock(
            return_value={
                "name": "bad daily",
                "prompt": "test",
                "schedule_type": "daily",
                "schedule_data": {"seconds": 60},
            }
        )

        resp = await _handle_schedule(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "times" in body["error"]

    async def test_interval_missing_seconds_returns_400(self, db, mock_request):
        """schedule_data for 'interval' without 'seconds' is rejected."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123
        mock_request.json = AsyncMock(
            return_value={
                "name": "bad interval",
                "prompt": "test",
                "schedule_type": "interval",
                "schedule_data": {"run_at": "2026-01-01T00:00:00Z"},
            }
        )

        resp = await _handle_schedule(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "seconds" in body["error"]

    async def test_interval_negative_seconds_returns_400(self, db, mock_request):
        """schedule_data with non-positive seconds is rejected."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123
        mock_request.json = AsyncMock(
            return_value={
                "name": "bad interval",
                "prompt": "test",
                "schedule_type": "interval",
                "schedule_data": {"seconds": -10},
            }
        )

        resp = await _handle_schedule(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "positive" in body["error"]

    async def test_daily_invalid_time_format_returns_400(self, db, mock_request):
        """schedule_data with a malformed time string is rejected."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123
        mock_request.json = AsyncMock(
            return_value={
                "name": "bad time",
                "prompt": "test",
                "schedule_type": "daily",
                "schedule_data": {"times": ["25:00"]},
            }
        )

        resp = await _handle_schedule(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "HH:MM" in body["error"]

    async def test_once_invalid_datetime_returns_400(self, db, mock_request):
        """schedule_data with an unparseable run_at is rejected."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123
        mock_request.json = AsyncMock(
            return_value={
                "name": "bad datetime",
                "prompt": "test",
                "schedule_type": "once",
                "schedule_data": {"run_at": "not-a-date"},
            }
        )

        resp = await _handle_schedule(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "ISO datetime" in body["error"]


# ── PATCH schedule_data validation ───────────────────────────────


class TestUpdateJobScheduleDataValidation:
    async def test_update_invalid_schedule_data_returns_400(self, db, mock_request):
        """PATCH with malformed schedule_data is rejected."""
        # Create a valid job first
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123
        mock_request.json = AsyncMock(
            return_value={
                "name": "to update",
                "prompt": "test",
                "schedule_type": "interval",
                "schedule_data": {"seconds": 600},
            }
        )
        with patch("kai.cron.register_job_by_id", new_callable=AsyncMock):
            resp = await _handle_schedule(mock_request)
        job_id = json.loads(resp.body.decode())["job_id"]

        # Now PATCH with invalid schedule_data
        mock_request.match_info = {"id": str(job_id)}
        mock_request.json = AsyncMock(return_value={"schedule_data": "not json"})

        resp = await _handle_update_job(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "not valid JSON" in body["error"]


# ── POST /api/services/{name} ─────────────────────────────────────


@pytest.fixture()
def service_request():
    """Create a mock request for the service proxy endpoint."""
    request = MagicMock(spec=web.Request)
    request.app = {
        "webhook_secret": "test-secret",
    }
    request.headers = {"X-Webhook-Secret": "test-secret"}
    request.match_info = {"name": "perplexity"}
    return request


class TestServiceCall:
    async def test_successful_call_returns_status_and_body(self, service_request):
        """Successful service call returns the status code and response body."""
        service_request.json = AsyncMock(return_value={"body": {"model": "sonar", "messages": []}})
        mock_result = ServiceResponse(success=True, status=200, body='{"answer": "42"}')
        with patch("kai.services.call_service", new_callable=AsyncMock, return_value=mock_result):
            resp = await _handle_service_call(service_request)

        assert resp.status == 200
        body = json.loads(resp.body.decode())
        assert body["status"] == 200
        assert body["body"] == '{"answer": "42"}'

    async def test_failed_call_returns_502(self, service_request):
        """Failed service call (success=False) returns 502 with error message."""
        service_request.json = AsyncMock(return_value={"body": {}})
        mock_result = ServiceResponse(success=False, error="Connection refused")
        with patch("kai.services.call_service", new_callable=AsyncMock, return_value=mock_result):
            resp = await _handle_service_call(service_request)

        assert resp.status == 502
        body = json.loads(resp.body.decode())
        assert "Connection refused" in body["error"]

    async def test_forwards_body_params_and_path_suffix(self, service_request):
        """All request fields (body, params, path_suffix) are forwarded to call_service."""
        service_request.json = AsyncMock(
            return_value={
                "body": {"query": "test"},
                "params": {"limit": "10"},
                "path_suffix": "/search",
            }
        )
        mock_result = ServiceResponse(success=True, status=200, body="ok")
        with patch("kai.services.call_service", new_callable=AsyncMock, return_value=mock_result) as mock_call:
            await _handle_service_call(service_request)

        mock_call.assert_called_once_with(
            "perplexity",
            body={"query": "test"},
            params={"limit": "10"},
            path_suffix="/search",
        )

    async def test_no_json_body_passes_defaults(self, service_request):
        """Request with no JSON body passes None/defaults to call_service."""
        service_request.json = AsyncMock(side_effect=json.JSONDecodeError("test", "doc", 0))
        mock_result = ServiceResponse(success=True, status=200, body="ok")
        with patch("kai.services.call_service", new_callable=AsyncMock, return_value=mock_result) as mock_call:
            await _handle_service_call(service_request)

        mock_call.assert_called_once_with(
            "perplexity",
            body=None,
            params=None,
            path_suffix="",
        )

    async def test_invalid_json_treated_as_no_body(self, service_request):
        """Invalid JSON is silently ignored (all fields are optional)."""
        service_request.json = AsyncMock(side_effect=json.JSONDecodeError("test", "doc", 0))
        mock_result = ServiceResponse(success=True, status=200, body="ok")
        with patch("kai.services.call_service", new_callable=AsyncMock, return_value=mock_result):
            resp = await _handle_service_call(service_request)

        # Should NOT return 400 - invalid JSON is fine for this endpoint
        assert resp.status == 200

    async def test_missing_secret_returns_401(self, service_request):
        """Missing webhook secret returns 401."""
        service_request.headers = {}

        resp = await _handle_service_call(service_request)

        assert resp.status == 401


# ── _resolve_chat_id ────────────────────────────────────────────────


class TestResolveChatId:
    def _make_request(self, app_chat_id=12345):
        """Create a minimal mock request with an app-level chat_id."""
        request = MagicMock()
        request.app = {"chat_id": app_chat_id}
        return request

    def test_explicit_chat_id(self):
        """Uses chat_id from payload when present."""
        request = self._make_request(app_chat_id=99999)
        assert _resolve_chat_id(request, {"chat_id": 42}) == 42

    def test_fallback_to_app_default(self):
        """Falls back to app-level chat_id when payload has no chat_id."""
        request = self._make_request(app_chat_id=12345)
        assert _resolve_chat_id(request, {}) == 12345

    def test_invalid_non_numeric(self):
        """Raises ValueError for non-numeric chat_id."""
        request = self._make_request()
        with pytest.raises(ValueError, match="must be an integer"):
            _resolve_chat_id(request, {"chat_id": "abc"})

    def test_invalid_float(self):
        """Raises ValueError for non-integer float chat_id."""
        request = self._make_request()
        with pytest.raises(ValueError, match="must be an integer"):
            _resolve_chat_id(request, {"chat_id": 12345.6})

    def test_invalid_bool(self):
        """Raises ValueError for boolean chat_id."""
        request = self._make_request()
        with pytest.raises(ValueError, match="must be an integer"):
            _resolve_chat_id(request, {"chat_id": True})

    def test_integer_like_float_accepted(self):
        """Integer-like float (e.g. 42.0) is accepted."""
        request = self._make_request()
        assert _resolve_chat_id(request, {"chat_id": 42.0}) == 42

    def test_string_integer_accepted(self):
        """String-encoded integer (e.g. from JSON) is accepted."""
        request = self._make_request()
        assert _resolve_chat_id(request, {"chat_id": "42"}) == 42


class TestGetJobsChatIdRouting:
    @pytest.mark.asyncio
    async def test_query_param_routes_to_user(self, db, mock_request):
        """GET /api/jobs?chat_id=456 returns jobs for that user."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123
        mock_request.query = {"chat_id": "456"}

        # Create a job for user 456
        await sessions.create_job(
            chat_id=456,
            name="User 456 Job",
            job_type="reminder",
            prompt="hello",
            schedule_type="daily",
            schedule_data='{"times": ["09:00"]}',
        )

        resp = await _handle_get_jobs(mock_request)
        body = json.loads(resp.body.decode())
        assert len(body) == 1
        assert body[0]["name"] == "User 456 Job"

    @pytest.mark.asyncio
    async def test_invalid_query_param_returns_400(self, db, mock_request):
        """GET /api/jobs?chat_id=abc returns 400."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123
        mock_request.query = {"chat_id": "abc"}

        resp = await _handle_get_jobs(mock_request)
        assert resp.status == 400


class TestScheduleChatIdRouting:
    @pytest.mark.asyncio
    async def test_explicit_chat_id_in_body(self, db, mock_request):
        """POST /api/schedule with chat_id routes job to that user."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["chat_id"] = 123
        mock_request.app["telegram_app"].job_queue = MagicMock()
        mock_request.app["telegram_app"].job_queue.jobs.return_value = []

        mock_request.json = AsyncMock(
            return_value={
                "name": "Routed Job",
                "prompt": "test",
                "schedule_type": "daily",
                "schedule_data": {"times": ["09:00"]},
                "chat_id": 456,
            }
        )

        resp = await _handle_schedule(mock_request)
        assert resp.status == 200

        # Verify the job was created for user 456, not 123
        jobs = await sessions.get_jobs(456)
        assert len(jobs) == 1
        assert jobs[0]["name"] == "Routed Job"

        # User 123 should have no jobs
        jobs_123 = await sessions.get_jobs(123)
        assert len(jobs_123) == 0


# ── chat_id authorization ──────────────────────────────────────────


class TestChatIdAuthorization:
    @pytest.mark.asyncio
    async def test_unauthorized_chat_id_returns_403(self, db, mock_request):
        """POST /api/schedule with chat_id not in allowed_user_ids returns 403."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["telegram_app"].job_queue = MagicMock()
        mock_request.app["telegram_app"].job_queue.jobs.return_value = []

        mock_request.json = AsyncMock(
            return_value={
                "name": "Evil Job",
                "prompt": "test",
                "schedule_type": "daily",
                "schedule_data": {"times": ["09:00"]},
                "chat_id": 999999,  # not in allowed_user_ids
            }
        )

        resp = await _handle_schedule(mock_request)
        assert resp.status == 403

    @pytest.mark.asyncio
    async def test_authorized_chat_id_accepted(self, db, mock_request):
        """POST /api/schedule with chat_id in allowed_user_ids succeeds."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app["telegram_app"].job_queue = MagicMock()
        mock_request.app["telegram_app"].job_queue.jobs.return_value = []

        mock_request.json = AsyncMock(
            return_value={
                "name": "Good Job",
                "prompt": "test",
                "schedule_type": "daily",
                "schedule_data": {"times": ["09:00"]},
                "chat_id": 456,  # in allowed_user_ids
            }
        )

        resp = await _handle_schedule(mock_request)
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_send_message_unauthorized_returns_403(self, db, mock_request):
        """POST /api/send-message with unauthorized chat_id returns 403."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"text": "hello", "chat_id": 999999})

        resp = await _handle_send_message(mock_request)
        assert resp.status == 403

    @pytest.mark.asyncio
    async def test_send_file_unauthorized_returns_403(self, db, mock_request):
        """POST /api/send-file with unauthorized chat_id returns 403."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"path": "/tmp/test.txt", "chat_id": 999999})

        resp = await _handle_send_file(mock_request)
        assert resp.status == 403

    def test_resolve_chat_id_unauthorized(self):
        """_resolve_chat_id raises UnauthorizedChatIdError for unknown users."""
        from kai.webhook import UnauthorizedChatIdError

        request = MagicMock()
        request.app = {"chat_id": 123, "allowed_user_ids": {123, 456}}

        with pytest.raises(UnauthorizedChatIdError):
            _resolve_chat_id(request, {"chat_id": 999999})

    def test_resolve_chat_id_no_allowed_list(self):
        """_resolve_chat_id skips validation when allowed_user_ids is not set."""
        request = MagicMock()
        request.app = {"chat_id": 123}

        # Should not raise even though 999 isn't in any allowed list
        result = _resolve_chat_id(request, {"chat_id": 999})
        assert result == 999


# ── Job ownership ──────────────────────────────────────────────────


class TestJobOwnership:
    @pytest.fixture(autouse=True)
    async def db(self, tmp_path):
        await sessions.init_db(tmp_path / "test.db")
        yield
        await sessions.close_db()

    @pytest.mark.asyncio
    async def test_delete_wrong_owner_returns_false(self):
        """delete_job with wrong chat_id returns False."""
        job_id = await sessions.create_job(
            chat_id=111,
            name="A",
            job_type="reminder",
            prompt="x",
            schedule_type="daily",
            schedule_data='{"times":["09:00"]}',
        )
        # User 222 tries to delete user 111's job
        assert await sessions.delete_job(job_id, chat_id=222) is False
        # Job still exists
        jobs = await sessions.get_jobs(111)
        assert len(jobs) == 1

    @pytest.mark.asyncio
    async def test_delete_correct_owner(self):
        """delete_job with correct chat_id succeeds."""
        job_id = await sessions.create_job(
            chat_id=111,
            name="A",
            job_type="reminder",
            prompt="x",
            schedule_type="daily",
            schedule_data='{"times":["09:00"]}',
        )
        assert await sessions.delete_job(job_id, chat_id=111) is True

    @pytest.mark.asyncio
    async def test_delete_no_chat_id_backward_compatible(self):
        """delete_job without chat_id deletes unconditionally."""
        job_id = await sessions.create_job(
            chat_id=111,
            name="A",
            job_type="reminder",
            prompt="x",
            schedule_type="daily",
            schedule_data='{"times":["09:00"]}',
        )
        assert await sessions.delete_job(job_id) is True

    @pytest.mark.asyncio
    async def test_deactivate_wrong_owner(self):
        """deactivate_job with wrong chat_id does not deactivate."""
        job_id = await sessions.create_job(
            chat_id=111,
            name="A",
            job_type="reminder",
            prompt="x",
            schedule_type="daily",
            schedule_data='{"times":["09:00"]}',
        )
        result = await sessions.deactivate_job(job_id, chat_id=222)
        assert result is False
        # Job should still be active
        jobs = await sessions.get_jobs(111)
        assert len(jobs) == 1

    @pytest.mark.asyncio
    async def test_update_wrong_owner(self):
        """update_job with wrong chat_id does not update."""
        job_id = await sessions.create_job(
            chat_id=111,
            name="Original",
            job_type="reminder",
            prompt="x",
            schedule_type="daily",
            schedule_data='{"times":["09:00"]}',
        )
        result = await sessions.update_job(job_id, chat_id=222, name="Hacked")
        assert result is False
        # Name should be unchanged
        jobs = await sessions.get_jobs(111)
        assert jobs[0]["name"] == "Original"


# ── Jobs API multi-user authorization ─────────────────────────────


class TestGetJobsAuth:
    """Authorization tests for GET /api/jobs."""

    @pytest.mark.asyncio
    async def test_omitted_chat_id_defaults_to_admin(self, db, mock_request):
        """GET /api/jobs without chat_id falls back to admin, preserving backward compat."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        # No query param set; should fall back to app["chat_id"] = 123
        await sessions.create_job(
            chat_id=123,
            name="Admin Job",
            job_type="reminder",
            prompt="test",
            schedule_type="daily",
            schedule_data='{"times": ["09:00"]}',
        )

        resp = await _handle_get_jobs(mock_request)
        assert resp.status == 200
        body = json.loads(resp.body.decode())
        assert len(body) == 1
        assert body[0]["name"] == "Admin Job"

    @pytest.mark.asyncio
    async def test_unauthorized_chat_id_returns_403(self, db, mock_request):
        """GET /api/jobs?chat_id=999 returns 403 for unauthorized users."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.query = {"chat_id": "999"}

        resp = await _handle_get_jobs(mock_request)
        assert resp.status == 403

    @pytest.mark.asyncio
    async def test_authorized_chat_id_returns_only_their_jobs(self, db, mock_request):
        """GET /api/jobs?chat_id=456 returns only user 456's jobs."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.query = {"chat_id": "456"}

        await sessions.create_job(
            chat_id=123,
            name="Admin Job",
            job_type="reminder",
            prompt="admin",
            schedule_type="daily",
            schedule_data='{"times": ["09:00"]}',
        )
        await sessions.create_job(
            chat_id=456,
            name="User Job",
            job_type="reminder",
            prompt="user",
            schedule_type="daily",
            schedule_data='{"times": ["10:00"]}',
        )

        resp = await _handle_get_jobs(mock_request)
        body = json.loads(resp.body.decode())
        assert len(body) == 1
        assert body[0]["name"] == "User Job"


class TestGetJobAuth:
    """Authorization and ownership tests for GET /api/jobs/{id}."""

    @pytest.mark.asyncio
    async def test_omitted_chat_id_defaults_to_admin(self, db, mock_request):
        """GET /api/jobs/{id} without chat_id falls back to admin, preserving backward compat."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        # No query param set; should fall back to app["chat_id"] = 123
        job_id = await sessions.create_job(
            chat_id=123,
            name="Admin Job",
            job_type="reminder",
            prompt="test",
            schedule_type="once",
            schedule_data='{"run_at": "2026-06-01T12:00:00+00:00"}',
        )
        mock_request.match_info = {"id": str(job_id)}

        resp = await _handle_get_job(mock_request)
        assert resp.status == 200
        body = json.loads(resp.body.decode())
        assert body["name"] == "Admin Job"

    @pytest.mark.asyncio
    async def test_wrong_owner_returns_404(self, db, mock_request):
        """GET /api/jobs/{id} returns 404 when job belongs to another user."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        # Caller is user 456, job belongs to user 123
        mock_request.query = {"chat_id": "456"}
        job_id = await sessions.create_job(
            chat_id=123,
            name="Admin Job",
            job_type="reminder",
            prompt="test",
            schedule_type="once",
            schedule_data='{"run_at": "2026-06-01T12:00:00+00:00"}',
        )
        mock_request.match_info = {"id": str(job_id)}

        resp = await _handle_get_job(mock_request)
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_owner_can_view_own_job(self, db, mock_request):
        """GET /api/jobs/{id}?chat_id=456 returns the job when owned by 456."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.query = {"chat_id": "456"}
        job_id = await sessions.create_job(
            chat_id=456,
            name="User Job",
            job_type="reminder",
            prompt="test",
            schedule_type="once",
            schedule_data='{"run_at": "2026-06-01T12:00:00+00:00"}',
        )
        mock_request.match_info = {"id": str(job_id)}

        resp = await _handle_get_job(mock_request)
        assert resp.status == 200
        body = json.loads(resp.body.decode())
        assert body["name"] == "User Job"

    @pytest.mark.asyncio
    async def test_unauthorized_chat_id_returns_403(self, db, mock_request):
        """GET /api/jobs/{id}?chat_id=999 returns 403 for unauthorized user."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.query = {"chat_id": "999"}
        mock_request.match_info = {"id": "1"}

        resp = await _handle_get_job(mock_request)
        assert resp.status == 403


class TestDeleteJobAuth:
    """Authorization tests for DELETE /api/jobs/{id}."""

    @pytest.mark.asyncio
    async def test_omitted_chat_id_defaults_to_admin(self, db, mock_request):
        """DELETE without chat_id falls back to admin, preserving backward compat."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        # No query param set; should fall back to app["chat_id"] = 123
        job_id = await sessions.create_job(
            chat_id=123,
            name="Admin Job",
            job_type="reminder",
            prompt="test",
            schedule_type="once",
            schedule_data='{"run_at": "2026-06-01T12:00:00+00:00"}',
        )
        mock_request.match_info = {"id": str(job_id)}

        resp = await _handle_delete_job(mock_request)
        assert resp.status == 200
        assert await sessions.get_job_by_id(job_id) is None

    @pytest.mark.asyncio
    async def test_user_can_delete_own_job(self, db, mock_request):
        """DELETE /api/jobs/{id}?chat_id=456 deletes user 456's job."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.query = {"chat_id": "456"}
        job_id = await sessions.create_job(
            chat_id=456,
            name="User Job",
            job_type="reminder",
            prompt="test",
            schedule_type="once",
            schedule_data='{"run_at": "2026-06-01T12:00:00+00:00"}',
        )
        mock_request.match_info = {"id": str(job_id)}

        resp = await _handle_delete_job(mock_request)
        assert resp.status == 200
        assert await sessions.get_job_by_id(job_id) is None

    @pytest.mark.asyncio
    async def test_cannot_delete_other_users_job(self, db, mock_request):
        """DELETE /api/jobs/{id}?chat_id=456 returns 404 for admin's job."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.query = {"chat_id": "456"}
        job_id = await sessions.create_job(
            chat_id=123,
            name="Admin Job",
            job_type="reminder",
            prompt="test",
            schedule_type="once",
            schedule_data='{"run_at": "2026-06-01T12:00:00+00:00"}',
        )
        mock_request.match_info = {"id": str(job_id)}

        resp = await _handle_delete_job(mock_request)
        assert resp.status == 404
        # Job should still exist
        assert await sessions.get_job_by_id(job_id) is not None

    @pytest.mark.asyncio
    async def test_unauthorized_chat_id_returns_403(self, db, mock_request):
        """DELETE /api/jobs/{id}?chat_id=999 returns 403."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.query = {"chat_id": "999"}
        mock_request.match_info = {"id": "1"}

        resp = await _handle_delete_job(mock_request)
        assert resp.status == 403


class TestUpdateJobAuth:
    """Authorization tests for PATCH /api/jobs/{id}."""

    @pytest.mark.asyncio
    async def test_omitted_chat_id_defaults_to_admin(self, db, mock_request):
        """PATCH without chat_id in body falls back to admin, preserving backward compat."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        job_id = await sessions.create_job(
            chat_id=123,
            name="Original",
            job_type="reminder",
            prompt="test",
            schedule_type="once",
            schedule_data='{"run_at": "2026-06-01T12:00:00+00:00"}',
        )
        mock_request.match_info = {"id": str(job_id)}
        # No chat_id in body; falls back to app["chat_id"] = 123
        mock_request.json = AsyncMock(return_value={"name": "Updated"})

        resp = await _handle_update_job(mock_request)
        assert resp.status == 200
        job = await sessions.get_job_by_id(job_id)
        assert job is not None
        assert job["name"] == "Updated"

    @pytest.mark.asyncio
    async def test_user_can_update_own_job(self, db, mock_request):
        """PATCH with chat_id in body updates the user's own job."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        job_id = await sessions.create_job(
            chat_id=456,
            name="Original",
            job_type="reminder",
            prompt="test",
            schedule_type="once",
            schedule_data='{"run_at": "2026-06-01T12:00:00+00:00"}',
        )
        mock_request.match_info = {"id": str(job_id)}
        mock_request.json = AsyncMock(return_value={"chat_id": 456, "name": "Updated"})

        resp = await _handle_update_job(mock_request)
        assert resp.status == 200
        job = await sessions.get_job_by_id(job_id)
        assert job is not None
        assert job["name"] == "Updated"

    @pytest.mark.asyncio
    async def test_cannot_update_other_users_job(self, db, mock_request):
        """PATCH with chat_id=456 returns 404 for admin's job."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        job_id = await sessions.create_job(
            chat_id=123,
            name="Admin Job",
            job_type="reminder",
            prompt="test",
            schedule_type="once",
            schedule_data='{"run_at": "2026-06-01T12:00:00+00:00"}',
        )
        mock_request.match_info = {"id": str(job_id)}
        mock_request.json = AsyncMock(return_value={"chat_id": 456, "name": "Hacked"})

        resp = await _handle_update_job(mock_request)
        assert resp.status == 404
        # Job should be unchanged
        job = await sessions.get_job_by_id(job_id)
        assert job is not None
        assert job["name"] == "Admin Job"

    @pytest.mark.asyncio
    async def test_unauthorized_chat_id_returns_403(self, db, mock_request):
        """PATCH with unauthorized chat_id returns 403."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.match_info = {"id": "1"}
        mock_request.json = AsyncMock(return_value={"chat_id": 999, "name": "Nope"})

        resp = await _handle_update_job(mock_request)
        assert resp.status == 403


# ── Filename sanitization ──────────────────────────────────────────


class TestFilenameSanitization:
    def test_path_traversal(self, tmp_path, monkeypatch):
        """../../etc/passwd becomes 'passwd' inside the files directory."""
        from kai.bot import _save_upload

        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        saved = _save_upload(b"test", "../../etc/passwd")
        assert saved.parent == tmp_path / "files"
        assert "passwd" in saved.name
        assert ".." not in str(saved)

    def test_empty_filename(self, tmp_path, monkeypatch):
        """Empty filename produces unnamed_file."""
        from kai.bot import _save_upload

        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        saved = _save_upload(b"test", "")
        assert "unnamed_file" in saved.name

    def test_slash_only(self, tmp_path, monkeypatch):
        """Slash-only filename produces unnamed_file."""
        from kai.bot import _save_upload

        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        saved = _save_upload(b"test", "/")
        assert "unnamed_file" in saved.name


# ── WAL mode ────────────────────────────────────────────────────────


class TestWALMode:
    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self, tmp_path):
        """init_db enables WAL journal mode."""
        await sessions.init_db(tmp_path / "test.db")
        try:
            async with sessions._get_db().execute("PRAGMA journal_mode") as cursor:
                row = await cursor.fetchone()
            assert row[0] == "wal"
        finally:
            await sessions.close_db()
