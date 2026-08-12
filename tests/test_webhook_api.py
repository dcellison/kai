"""Integration tests for webhook HTTP API endpoints (jobs CRUD, file exchange)."""

import asyncio
import hashlib
import hmac as hmac_mod
import json
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from aiohttp import web

import kai.webhook as webhook_mod
from kai import sessions
from kai.internal_api_auth import InternalAPIAuth, InternalAPIPrincipal, InternalAPIScope
from kai.services import ServiceResponse
from kai.webhook import (
    ALLOWED_USER_IDS_KEY,
    CHAT_ID_KEY,
    CONFIG_KEY,
    GENERIC_WEBHOOK_SECRET_KEY,
    GITHUB_WEBHOOK_SECRET_KEY,
    INTERNAL_API_AUTH_KEY,
    POOL_KEY,
    PR_REVIEW_COOLDOWN_KEY,
    TELEGRAM_APP_KEY,
    TELEGRAM_BOT_KEY,
    TELEGRAM_WEBHOOK_SECRET_KEY,
    UnauthorizedChatIdError,
    _handle_delete_job,
    _handle_generic,
    _handle_get_job,
    _handle_get_jobs,
    _handle_github,
    _handle_memory_add,
    _handle_memory_delete_all,
    _handle_memory_search,
    _handle_memory_stats,
    _handle_schedule,
    _handle_send_file,
    _handle_send_message,
    _handle_service_call,
    _handle_telegram_update,
    _handle_update_job,
    _resolve_chat_id,
    stop,
)


def _make_internal_api_auth() -> InternalAPIAuth:
    """Create deterministic credentials for the two API test principals."""
    return InternalAPIAuth(
        {123: "test-secret", 456: "other-secret"},
        allowed_services_by_user={123: {"perplexity"}},
    )


async def _call_memory_delete_all_as_authorized(request, *, chat_id: int = 123):
    """Exercise delete-all handler behavior with an explicitly privileged principal."""
    principal = InternalAPIPrincipal(
        chat_id=chat_id,
        scopes=frozenset({InternalAPIScope.MEMORY_DELETE_ALL}),
    )
    return await _handle_memory_delete_all.__wrapped__(request, principal)


@pytest.fixture
async def db(tmp_path):
    """Initialize a fresh database for each test."""
    await sessions.init_db(tmp_path / "test.db")
    yield
    await sessions.close_db()


@pytest.fixture(autouse=True)
def _default_github_token_lookup():
    """Webhook API tests default to no stored per-user GitHub token."""
    with patch("kai.webhook.sessions.get_setting", new_callable=AsyncMock, return_value=None):
        yield


@pytest.fixture
def mock_request():
    """Create a minimal mock request with app dict and helpers."""
    request = MagicMock(spec=web.Request)
    request.app = {
        INTERNAL_API_AUTH_KEY: _make_internal_api_auth(),
        GENERIC_WEBHOOK_SECRET_KEY: "signing-secret",
        TELEGRAM_APP_KEY: MagicMock(),
        TELEGRAM_BOT_KEY: AsyncMock(),
        CHAT_ID_KEY: 123,
        ALLOWED_USER_IDS_KEY: {123, 456},
    }
    # Mock the job_queue on the telegram app
    job_queue = MagicMock()
    job_queue.jobs = MagicMock(return_value=[])
    request.app[TELEGRAM_APP_KEY].job_queue = job_queue
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
        mock_request.app[CHAT_ID_KEY] = 123
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

    async def test_agent_job_type_accepted(self, db, mock_request):
        """Schedule endpoint accepts the canonical agent job type."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app[CHAT_ID_KEY] = 123
        mock_request.app[TELEGRAM_APP_KEY].job_queue = MagicMock()

        # Mock register_job_by_id so we don't need a full APScheduler setup
        import kai.cron as cron_mod

        cron_mod.register_job_by_id = AsyncMock()

        mock_request.json = AsyncMock(
            return_value={
                "name": "test agent job",
                "prompt": "test",
                "schedule_type": "once",
                "schedule_data": {"run_at": "2026-02-20T10:00:00+00:00"},
                "job_type": "agent",
            }
        )

        resp = await _handle_schedule(mock_request)

        assert resp.status == 200
        body = json.loads(resp.body.decode())
        assert "job_id" in body

    async def test_legacy_agent_job_type_is_stored_canonically(self, db, mock_request):
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app[CHAT_ID_KEY] = 123
        mock_request.app[TELEGRAM_APP_KEY].job_queue = MagicMock()

        import kai.cron as cron_mod

        cron_mod.register_job_by_id = AsyncMock(return_value=True)
        mock_request.json = AsyncMock(
            return_value={
                "name": "legacy agent job",
                "prompt": "test",
                "schedule_type": "once",
                "schedule_data": {"run_at": "2026-02-20T10:00:00+00:00"},
                "job_type": "claude",
            }
        )

        resp = await _handle_schedule(mock_request)

        assert resp.status == 200
        job = await sessions.get_job_by_id(json.loads(resp.body.decode())["job_id"])
        assert job is not None
        assert job["job_type"] == "agent"


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
            job_type="agent",
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

    async def test_update_schedule_registration_false_rolls_back_job(self, db, mock_request):
        """PATCH rolls back DB changes if scheduler registration returns False."""
        job_id = await sessions.create_job(
            chat_id=123,
            name="old name",
            job_type="reminder",
            prompt="old prompt",
            schedule_type="interval",
            schedule_data='{"seconds": 300}',
            auto_remove=False,
            notify_on_check=True,
        )
        queued_job = MagicMock()
        queued_job.name = f"cron_{job_id}"
        mock_request.app[TELEGRAM_APP_KEY].job_queue.jobs.return_value = [queued_job]
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.match_info = {"id": str(job_id)}
        mock_request.json = AsyncMock(
            return_value={
                "name": "new name",
                "prompt": "new prompt",
                "schedule_data": {"seconds": 600},
                "auto_remove": True,
                "notify_on_check": False,
            }
        )

        with patch(
            "kai.cron.register_job_by_id",
            new_callable=AsyncMock,
            side_effect=[False, True],
        ) as mock_register:
            resp = await _handle_update_job(mock_request)

        assert resp.status == 500
        body = json.loads(resp.body.decode())
        assert body["error"] == "Failed to register job"
        queued_job.schedule_removal.assert_called_once()
        mock_register.assert_has_awaits(
            [
                call(mock_request.app[TELEGRAM_APP_KEY], job_id),
                call(mock_request.app[TELEGRAM_APP_KEY], job_id),
            ]
        )
        job = await sessions.get_job_by_id(job_id)
        assert job is not None
        assert job["name"] == "old name"
        assert job["prompt"] == "old prompt"
        assert job["schedule_type"] == "interval"
        assert job["schedule_data"] == '{"seconds": 300}'
        assert job["auto_remove"] is False
        assert job["notify_on_check"] is True

    async def test_update_schedule_registration_exception_rolls_back_job(self, db, mock_request):
        """PATCH rolls back DB changes if scheduler registration raises."""
        job_id = await sessions.create_job(
            chat_id=123,
            name="old name",
            job_type="reminder",
            prompt="old prompt",
            schedule_type="daily",
            schedule_data='{"times": ["09:00"]}',
        )
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.match_info = {"id": str(job_id)}
        mock_request.json = AsyncMock(return_value={"schedule_data": {"times": ["10:00"]}})

        with patch(
            "kai.cron.register_job_by_id",
            new_callable=AsyncMock,
            side_effect=[RuntimeError("scheduler unavailable"), True],
        ):
            resp = await _handle_update_job(mock_request)

        assert resp.status == 500
        body = json.loads(resp.body.decode())
        assert body["error"] == "Failed to register job"
        job = await sessions.get_job_by_id(job_id)
        assert job is not None
        assert job["schedule_type"] == "daily"
        assert job["schedule_data"] == '{"times": ["09:00"]}'


# ── POST /api/send-file ─────────────────────────────────────────────


@pytest.fixture
def send_file_request(tmp_path):
    """Create a mock request for the send-file endpoint with workspace confinement."""
    # Send-file resolves the workspace via pool.get_effective_workspace(chat_id)
    # rather than a global app["workspace"] slot. The fixture supplies a mock
    # pool whose get_effective_workspace() returns tmp_path so path confinement
    # passes.
    mock_pool = MagicMock()
    mock_pool.get_effective_workspace = AsyncMock(return_value=tmp_path)
    request = MagicMock(spec=web.Request)
    request.app = {
        INTERNAL_API_AUTH_KEY: _make_internal_api_auth(),
        GENERIC_WEBHOOK_SECRET_KEY: "signing-secret",
        TELEGRAM_BOT_KEY: AsyncMock(),
        CHAT_ID_KEY: 123,
        POOL_KEY: mock_pool,
    }
    request.headers = {"X-Webhook-Secret": "test-secret"}
    return request


@pytest.fixture
def isolated_send_file_roots(tmp_path, send_file_request, monkeypatch):
    """Give send-file distinct workspace and per-principal upload roots."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_dir = tmp_path / "data"
    monkeypatch.setattr("kai.webhook.DATA_DIR", data_dir)
    send_file_request.app[POOL_KEY].get_effective_workspace.return_value = workspace
    return send_file_request, workspace, data_dir


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
        send_file_request.app[TELEGRAM_BOT_KEY].send_photo.assert_called_once()

    async def test_send_document(self, tmp_path, send_file_request):
        """Non-image files are sent via send_document (as attachments)."""
        doc = tmp_path / "report.pdf"
        doc.write_bytes(b"%PDF-1.4 fake")

        send_file_request.json = AsyncMock(return_value={"path": str(doc)})
        resp = await _handle_send_file(send_file_request)

        assert resp.status == 200
        body = json.loads(resp.body)
        assert body["status"] == "sent"
        send_file_request.app[TELEGRAM_BOT_KEY].send_document.assert_called_once()

    async def test_caption_forwarded_to_telegram(self, tmp_path, send_file_request):
        """Optional caption is passed through to the Telegram send call."""
        f = tmp_path / "pic.png"
        f.write_bytes(b"fake-png")

        send_file_request.json = AsyncMock(return_value={"path": str(f), "caption": "Here you go"})
        resp = await _handle_send_file(send_file_request)

        assert resp.status == 200
        call_kwargs = send_file_request.app[TELEGRAM_BOT_KEY].send_photo.call_args
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

    async def test_authenticated_principal_can_send_own_uploaded_file(self, isolated_send_file_roots):
        """The principal's DATA_DIR/files/<chat_id> directory is allowed."""
        request, _workspace, data_dir = isolated_send_file_roots
        uploaded = data_dir / "files" / "123" / "report.txt"
        uploaded.parent.mkdir(parents=True)
        uploaded.write_text("principal-owned")
        request.json = AsyncMock(return_value={"path": str(uploaded)})

        resp = await _handle_send_file(request)

        assert resp.status == 200
        request.app[TELEGRAM_BOT_KEY].send_document.assert_awaited_once()

    async def test_file_scope_follows_authenticated_credential(self, isolated_send_file_roots):
        """A second credential receives its own scope, independent of app defaults."""
        request, _workspace, data_dir = isolated_send_file_roots
        uploaded = data_dir / "files" / "456" / "report.txt"
        uploaded.parent.mkdir(parents=True)
        uploaded.write_text("second principal")
        request.headers = {"X-Webhook-Secret": "other-secret"}
        request.json = AsyncMock(return_value={"path": str(uploaded)})

        resp = await _handle_send_file(request)

        assert resp.status == 200
        request.app[TELEGRAM_BOT_KEY].send_document.assert_awaited_once()
        assert request.app[TELEGRAM_BOT_KEY].send_document.await_args.args[0] == 456

    async def test_authenticated_principal_cannot_send_sibling_uploaded_file(self, isolated_send_file_roots):
        """A FILES_SEND credential cannot select another principal's upload."""
        request, _workspace, data_dir = isolated_send_file_roots
        sibling_file = data_dir / "files" / "456" / "secret.txt"
        sibling_file.parent.mkdir(parents=True)
        sibling_file.write_text("other principal")
        request.json = AsyncMock(return_value={"path": str(sibling_file)})

        resp = await _handle_send_file(request)

        assert resp.status == 403
        request.app[TELEGRAM_BOT_KEY].send_document.assert_not_awaited()

    async def test_authenticated_principal_cannot_send_legacy_shared_file(self, isolated_send_file_roots):
        """Ambiguous files in the legacy shared root are not exposed by the API."""
        request, _workspace, data_dir = isolated_send_file_roots
        shared_file = data_dir / "files" / "legacy.txt"
        shared_file.parent.mkdir(parents=True)
        shared_file.write_text("unattributed")
        request.json = AsyncMock(return_value={"path": str(shared_file)})

        resp = await _handle_send_file(request)

        assert resp.status == 403
        request.app[TELEGRAM_BOT_KEY].send_document.assert_not_awaited()

    async def test_symlink_cannot_escape_principal_upload_directory(self, isolated_send_file_roots):
        """Resolving a symlink into a sibling principal's directory is denied."""
        request, _workspace, data_dir = isolated_send_file_roots
        sibling_file = data_dir / "files" / "456" / "secret.txt"
        sibling_file.parent.mkdir(parents=True)
        sibling_file.write_text("other principal")
        link = data_dir / "files" / "123" / "link.txt"
        link.parent.mkdir(parents=True)
        link.symlink_to(sibling_file)
        request.json = AsyncMock(return_value={"path": str(link)})

        resp = await _handle_send_file(request)

        assert resp.status == 403
        request.app[TELEGRAM_BOT_KEY].send_document.assert_not_awaited()

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
        INTERNAL_API_AUTH_KEY: _make_internal_api_auth(),
        GENERIC_WEBHOOK_SECRET_KEY: "signing-secret",
        TELEGRAM_BOT_KEY: AsyncMock(),
        CHAT_ID_KEY: 123,
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
        send_message_request.app[TELEGRAM_BOT_KEY].send_message.assert_called_once_with(123, "Hello!")

    async def test_splits_long_message(self, send_message_request):
        """Messages exceeding 4096 chars are split into multiple sends."""
        # Create a message with two paragraphs, each over 2048 chars
        long_text = ("A" * 2100) + "\n\n" + ("B" * 2100)
        send_message_request.json = AsyncMock(return_value={"text": long_text})
        resp = await _handle_send_message(send_message_request)

        assert resp.status == 200
        bot = send_message_request.app[TELEGRAM_BOT_KEY]
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
        send_message_request.app[TELEGRAM_BOT_KEY].send_message = AsyncMock(side_effect=RuntimeError("Boom"))
        resp = await _handle_send_message(send_message_request)
        assert resp.status == 500


# ── POST /webhook/telegram ─────────────────────────────────────────


@pytest.fixture
def telegram_request():
    """Create a mock request for the Telegram webhook endpoint."""
    request = MagicMock(spec=web.Request)
    request.app = {
        TELEGRAM_WEBHOOK_SECRET_KEY: "tg-secret",
        TELEGRAM_APP_KEY: MagicMock(),
        TELEGRAM_BOT_KEY: MagicMock(),
    }
    request.app[TELEGRAM_APP_KEY].process_update = AsyncMock()
    request.headers = {"X-Telegram-Bot-Api-Secret-Token": "tg-secret"}
    return request


class TestTelegramUpdate:
    def test_priority_classifier_matches_only_stop_commands(self):
        assert webhook_mod._is_telegram_stop_update({"message": {"text": "/stop"}}) is True
        assert webhook_mod._is_telegram_stop_update({"message": {"text": "/stop@kai_bot"}}) is True
        assert webhook_mod._is_telegram_stop_update({"message": {"text": "/stop now"}}) is True
        assert webhook_mod._is_telegram_stop_update({"message": {"text": "/stopping"}}) is False
        assert webhook_mod._is_telegram_stop_update({"message": {"caption": "/stop"}}) is False

    async def test_valid_secret_dispatches_update(self, db, telegram_request, monkeypatch):
        """Valid secret and JSON body dispatches to process_update."""
        fake_update = MagicMock()
        monkeypatch.setattr("kai.webhook.Update.de_json", MagicMock(return_value=fake_update))
        telegram_request.json = AsyncMock(return_value={"update_id": 123})

        resp = await _handle_telegram_update(telegram_request)

        assert resp.status == 200
        # process_update runs from the durable queue worker. Wait for that
        # worker before the DB fixture closes.
        assert webhook_mod._telegram_queue_worker_task is not None
        await webhook_mod._telegram_queue_worker_task
        telegram_request.app[TELEGRAM_APP_KEY].process_update.assert_called_once_with(fake_update)

    async def test_wrong_secret_returns_401(self, telegram_request):
        """Wrong secret token returns 401 without dispatching."""
        telegram_request.headers = {"X-Telegram-Bot-Api-Secret-Token": "wrong"}

        resp = await _handle_telegram_update(telegram_request)

        assert resp.status == 401
        telegram_request.app[TELEGRAM_APP_KEY].process_update.assert_not_called()

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
        telegram_request.app[TELEGRAM_APP_KEY].process_update.assert_not_called()

    async def test_missing_update_id_skips_dispatch(self, telegram_request):
        """Updates without a durable dedupe key are swallowed to avoid permanent retries."""
        telegram_request.json = AsyncMock(return_value={"message": {"text": "missing id"}})

        resp = await _handle_telegram_update(telegram_request)

        assert resp.status == 200
        telegram_request.app[TELEGRAM_APP_KEY].process_update.assert_not_called()

    async def test_persisted_stop_interrupts_busy_fifo_worker(self, db, telegram_request, monkeypatch):
        """A queued /stop dispatches while an earlier webhook update is still running."""
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        stop_processed = asyncio.Event()

        def deserialize(data, _bot):
            return MagicMock(update_id=data["update_id"])

        async def process(update):
            if update.update_id == 2001:
                first_started.set()
                await release_first.wait()
            elif update.update_id == 2002:
                stop_processed.set()

        monkeypatch.setattr("kai.webhook.Update.de_json", deserialize)
        telegram_request.app[TELEGRAM_APP_KEY].process_update = AsyncMock(side_effect=process)

        telegram_request.json = AsyncMock(return_value={"update_id": 2001, "message": {"text": "work"}})
        assert (await _handle_telegram_update(telegram_request)).status == 200
        await asyncio.wait_for(first_started.wait(), timeout=1)

        telegram_request.json = AsyncMock(return_value={"update_id": 2002, "message": {"text": "/stop"}})
        assert (await _handle_telegram_update(telegram_request)).status == 200
        await asyncio.wait_for(stop_processed.wait(), timeout=1)

        assert release_first.is_set() is False
        release_first.set()
        assert webhook_mod._telegram_queue_worker_task is not None
        await webhook_mod._telegram_queue_worker_task
        await asyncio.gather(*webhook_mod._background_tasks)

        async with sessions._get_db().execute(
            "SELECT status, attempt_count FROM telegram_update_queue ORDER BY id"
        ) as cursor:
            rows = await cursor.fetchall()
        assert [(row["status"], row["attempt_count"]) for row in rows] == [
            ("done", 1),
            ("done", 1),
        ]

    async def test_ordinary_update_remains_fifo_while_worker_is_busy(self, db, telegram_request, monkeypatch):
        """The interrupt path does not broadly parallelize Telegram updates."""
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        second_processed = asyncio.Event()

        def deserialize(data, _bot):
            return MagicMock(update_id=data["update_id"])

        async def process(update):
            if update.update_id == 2011:
                first_started.set()
                await release_first.wait()
            elif update.update_id == 2012:
                second_processed.set()

        monkeypatch.setattr("kai.webhook.Update.de_json", deserialize)
        telegram_request.app[TELEGRAM_APP_KEY].process_update = AsyncMock(side_effect=process)

        telegram_request.json = AsyncMock(return_value={"update_id": 2011, "message": {"text": "one"}})
        assert (await _handle_telegram_update(telegram_request)).status == 200
        await asyncio.wait_for(first_started.wait(), timeout=1)

        telegram_request.json = AsyncMock(return_value={"update_id": 2012, "message": {"text": "two"}})
        assert (await _handle_telegram_update(telegram_request)).status == 200
        await asyncio.sleep(0)
        assert second_processed.is_set() is False

        release_first.set()
        assert webhook_mod._telegram_queue_worker_task is not None
        await webhook_mod._telegram_queue_worker_task
        assert second_processed.is_set() is True

    async def test_persistence_failure_returns_500(self, telegram_request, monkeypatch):
        """If the durable queue write fails, do not acknowledge the update as accepted."""
        monkeypatch.setattr(
            "kai.webhook.sessions.enqueue_telegram_update",
            AsyncMock(side_effect=RuntimeError("database locked")),
        )
        telegram_request.json = AsyncMock(return_value={"update_id": 123})
        webhook_mod._background_tasks.clear()

        resp = await _handle_telegram_update(telegram_request)

        assert resp.status == 500
        telegram_request.app[TELEGRAM_APP_KEY].process_update.assert_not_called()
        assert webhook_mod._background_tasks == set()

    async def test_worker_retries_then_completes_transient_failure(self, db, telegram_request, monkeypatch):
        """A process_update exception requeues the durable row for another attempt."""
        fake_update = MagicMock()
        monkeypatch.setattr("kai.webhook.Update.de_json", MagicMock(return_value=fake_update))
        telegram_request.app[TELEGRAM_APP_KEY].process_update = AsyncMock(side_effect=[RuntimeError("boom"), None])

        row_id, _ = await sessions.enqueue_telegram_update(124, '{"update_id":124}')
        webhook_mod._ensure_telegram_update_queue_worker(
            telegram_request.app[TELEGRAM_APP_KEY],
            telegram_request.app[TELEGRAM_BOT_KEY],
        )
        assert webhook_mod._telegram_queue_worker_task is not None
        await webhook_mod._telegram_queue_worker_task

        telegram_request.app[TELEGRAM_APP_KEY].process_update.assert_awaited_with(fake_update)
        assert telegram_request.app[TELEGRAM_APP_KEY].process_update.await_count == 2
        async with sessions._get_db().execute(
            "SELECT status, attempt_count, last_error FROM telegram_update_queue WHERE id = ?",
            (row_id,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row["status"] == "done"
        assert row["attempt_count"] == 2
        assert "RuntimeError: boom" in row["last_error"]

    async def test_worker_discards_poison_update_after_max_attempts(self, db, telegram_request, monkeypatch):
        """A permanently failing queued update is discarded after bounded retries."""
        fake_update = MagicMock()
        monkeypatch.setattr("kai.webhook.Update.de_json", MagicMock(return_value=fake_update))
        monkeypatch.setattr(webhook_mod, "_TELEGRAM_UPDATE_MAX_ATTEMPTS", 2)
        telegram_request.app[TELEGRAM_APP_KEY].process_update = AsyncMock(side_effect=RuntimeError("always bad"))

        row_id, _ = await sessions.enqueue_telegram_update(125, '{"update_id":125}')
        webhook_mod._ensure_telegram_update_queue_worker(
            telegram_request.app[TELEGRAM_APP_KEY],
            telegram_request.app[TELEGRAM_BOT_KEY],
        )
        assert webhook_mod._telegram_queue_worker_task is not None
        await webhook_mod._telegram_queue_worker_task

        assert telegram_request.app[TELEGRAM_APP_KEY].process_update.await_count == 2
        async with sessions._get_db().execute(
            "SELECT status, attempt_count, last_error FROM telegram_update_queue WHERE id = ?",
            (row_id,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row["status"] == "done"
        assert row["attempt_count"] == 2
        assert "RuntimeError: always bad" in row["last_error"]


# ── webhook shutdown background task drain ─────────────────────────


class TestWebhookStopBackgroundTasks:
    async def test_stop_waits_for_background_tasks_before_runner_cleanup(self, monkeypatch):
        """Clean shutdown drains in-flight webhook work before tearing down aiohttp."""
        events: list[str] = []

        async def background_work():
            await asyncio.sleep(0)
            events.append("task")

        async def cleanup():
            events.append("cleanup")

        task = asyncio.create_task(background_work())
        webhook_mod._background_tasks.add(task)
        task.add_done_callback(webhook_mod._background_tasks.discard)

        runner = MagicMock()
        runner.cleanup = AsyncMock(side_effect=cleanup)
        monkeypatch.setattr(webhook_mod, "_runner", runner)
        monkeypatch.setattr(webhook_mod, "_app", None)
        monkeypatch.setattr(webhook_mod, "_webhook_registered", False)
        monkeypatch.setattr(webhook_mod, "_health_monitor_task", None)

        await stop()

        assert events == ["task", "cleanup"]
        assert webhook_mod._background_tasks == set()

    async def test_stop_cancels_background_tasks_after_timeout(self, monkeypatch):
        """Shutdown remains bounded if webhook background work does not finish."""
        started = asyncio.Event()

        async def background_work():
            started.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(background_work())
        await started.wait()
        webhook_mod._background_tasks.add(task)
        task.add_done_callback(webhook_mod._background_tasks.discard)

        runner = MagicMock()
        runner.cleanup = AsyncMock()
        monkeypatch.setattr(webhook_mod, "_runner", runner)
        monkeypatch.setattr(webhook_mod, "_app", None)
        monkeypatch.setattr(webhook_mod, "_webhook_registered", False)
        monkeypatch.setattr(webhook_mod, "_health_monitor_task", None)
        monkeypatch.setattr(webhook_mod, "_BACKGROUND_TASK_DRAIN_TIMEOUT", 0.001)

        await stop()

        assert task.cancelled()
        assert webhook_mod._background_tasks == set()
        runner.cleanup.assert_awaited_once()


# ── GitHub webhook helpers ─────────────────────────────────────────


def _sign_body(secret: str, body: bytes) -> str:
    """Compute a valid GitHub HMAC-SHA256 signature for test payloads."""
    digest = hmac_mod.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


@pytest.fixture()
def github_request():
    """Create a mock request for the GitHub webhook endpoint.

    Includes a mock config with user_configs={} (no per-user routing)
    and mocks resolve_github_settings to return defaults. This simulates
    fallback routing where events go to the admin chat_id.
    """
    mock_config = MagicMock()
    mock_config.user_configs = {}
    request = MagicMock(spec=web.Request)
    request.app = {
        INTERNAL_API_AUTH_KEY: _make_internal_api_auth(),
        GITHUB_WEBHOOK_SECRET_KEY: "test-secret",
        TELEGRAM_BOT_KEY: AsyncMock(),
        CHAT_ID_KEY: 12345,
        CONFIG_KEY: mock_config,
        PR_REVIEW_COOLDOWN_KEY: 300,
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
        bot = github_request.app[TELEGRAM_BOT_KEY]
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
        bot = github_request.app[TELEGRAM_BOT_KEY]
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
        bot = github_request.app[TELEGRAM_BOT_KEY]
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
        github_request.app[TELEGRAM_BOT_KEY].send_message.assert_not_called()

    async def test_legacy_signature_is_rejected(self, github_request):
        """WEBHOOK_SECRET no longer authenticates the GitHub ingress domain."""
        body = b'{"zen": "migration"}'
        github_request.read = AsyncMock(return_value=body)
        github_request.headers = {
            "X-Hub-Signature-256": _sign_body("legacy-secret", body),
            "X-GitHub-Event": "ping",
        }

        resp = await _handle_github(github_request)

        assert resp.status == 401
        github_request.app[TELEGRAM_BOT_KEY].send_message.assert_not_called()

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
        github_request.app[TELEGRAM_BOT_KEY].send_message.assert_not_called()

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
        github_request.app[TELEGRAM_BOT_KEY].send_message.assert_not_called()

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
        github_request.app[TELEGRAM_BOT_KEY].send_message.assert_not_called()

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
        INTERNAL_API_AUTH_KEY: _make_internal_api_auth(),
        GENERIC_WEBHOOK_SECRET_KEY: "test-secret",
        TELEGRAM_BOT_KEY: AsyncMock(),
        CHAT_ID_KEY: 12345,
    }
    request.headers = {"X-Webhook-Secret": "test-secret"}
    return request


class TestGenericWebhook:
    async def test_sends_message_field(self, generic_request):
        """Payload with a 'message' field sends that string to Telegram."""
        generic_request.json = AsyncMock(return_value={"message": "Alert: disk full"})

        resp = await _handle_generic(generic_request)

        assert resp.status == 200
        generic_request.app[TELEGRAM_BOT_KEY].send_message.assert_called_once_with(12345, "Alert: disk full")

    async def test_dumps_full_payload_when_no_message(self, generic_request):
        """Payload without 'message' sends the full JSON dump to Telegram."""
        payload = {"key": "value", "count": 42}
        generic_request.json = AsyncMock(return_value=payload)

        resp = await _handle_generic(generic_request)

        assert resp.status == 200
        sent_text = generic_request.app[TELEGRAM_BOT_KEY].send_message.call_args[0][1]
        # Should be a pretty-printed JSON dump
        assert '"key": "value"' in sent_text
        assert '"count": 42' in sent_text

    async def test_empty_message_field_sends_empty_string(self, generic_request):
        """Empty string 'message' is sent as-is (not treated as missing)."""
        generic_request.json = AsyncMock(return_value={"message": ""})

        resp = await _handle_generic(generic_request)

        assert resp.status == 200
        sent_text = generic_request.app[TELEGRAM_BOT_KEY].send_message.call_args[0][1]
        assert sent_text == ""

    async def test_long_message_truncated(self, generic_request):
        """Messages over 4096 chars are truncated with '...' suffix."""
        long_msg = "x" * 5000
        generic_request.json = AsyncMock(return_value={"message": long_msg})

        resp = await _handle_generic(generic_request)

        assert resp.status == 200
        sent_text = generic_request.app[TELEGRAM_BOT_KEY].send_message.call_args[0][1]
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
        generic_request.app[TELEGRAM_BOT_KEY].send_message = AsyncMock(side_effect=RuntimeError("network error"))

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

    async def test_legacy_secret_is_rejected(self, generic_request):
        """WEBHOOK_SECRET no longer authenticates the generic ingress domain."""
        generic_request.headers = {"X-Webhook-Secret": "legacy-secret"}
        generic_request.json = AsyncMock(return_value={"message": "legacy caller"})

        resp = await _handle_generic(generic_request)

        assert resp.status == 401
        generic_request.app[TELEGRAM_BOT_KEY].send_message.assert_not_called()


# ── GET /api/jobs ──────────────────────────────────────────────────


class TestGetJobs:
    async def test_returns_active_jobs(self, db, mock_request):
        """Returns a list of active jobs for the configured chat."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app[CHAT_ID_KEY] = 123

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
            job_type="agent",
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
        mock_request.app[CHAT_ID_KEY] = 123

        resp = await _handle_get_jobs(mock_request)

        assert resp.status == 200
        body = json.loads(resp.body.decode())
        assert body == []

    async def test_missing_secret_returns_401(self, db, mock_request):
        """Missing webhook secret returns 401."""
        mock_request.headers = {}
        mock_request.app[CHAT_ID_KEY] = 123

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
        mock_request.app[CHAT_ID_KEY] = 123
        # Missing prompt, schedule_type, and schedule_data
        mock_request.json = AsyncMock(return_value={"name": "incomplete"})

        resp = await _handle_schedule(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "required" in body["error"].lower()

    async def test_invalid_schedule_type_returns_400(self, db, mock_request):
        """Returns 400 for unrecognized schedule_type."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app[CHAT_ID_KEY] = 123
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
        mock_request.app[CHAT_ID_KEY] = 123
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
        mock_request.app[CHAT_ID_KEY] = 123
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
        mock_request.app[CHAT_ID_KEY] = 123
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
        mock_request.app[CHAT_ID_KEY] = 123
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
        mock_request.app[CHAT_ID_KEY] = 123
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
        mock_request.app[CHAT_ID_KEY] = 123
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
        mock_register.assert_called_once_with(mock_request.app[TELEGRAM_APP_KEY], body["job_id"])

    async def test_scheduler_false_deactivates_created_job(self, db, mock_request):
        """A registration miss returns 500 and does not leave an active duplicateable job."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app[CHAT_ID_KEY] = 123
        mock_request.json = AsyncMock(
            return_value={
                "name": "scheduler false",
                "prompt": "test",
                "schedule_type": "interval",
                "schedule_data": {"seconds": 300},
            }
        )
        with patch("kai.cron.register_job_by_id", new_callable=AsyncMock, return_value=False):
            resp = await _handle_schedule(mock_request)

        assert resp.status == 500
        body = json.loads(resp.body.decode())
        assert body["error"] == "Failed to register job"
        assert await sessions.get_jobs(123) == []

    async def test_scheduler_exception_deactivates_created_job(self, db, mock_request):
        """A registration exception returns 500 and compensates the committed row."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app[CHAT_ID_KEY] = 123
        mock_request.json = AsyncMock(
            return_value={
                "name": "scheduler raise",
                "prompt": "test",
                "schedule_type": "interval",
                "schedule_data": {"seconds": 300},
            }
        )
        with patch(
            "kai.cron.register_job_by_id",
            new_callable=AsyncMock,
            side_effect=RuntimeError("scheduler unavailable"),
        ):
            resp = await _handle_schedule(mock_request)

        assert resp.status == 500
        body = json.loads(resp.body.decode())
        assert body["error"] == "Failed to register job"
        assert await sessions.get_jobs(123) == []

    async def test_invalid_json_returns_400(self, db, mock_request):
        """Malformed JSON body returns 400."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app[CHAT_ID_KEY] = 123
        mock_request.json = AsyncMock(side_effect=json.JSONDecodeError("test", "doc", 0))

        resp = await _handle_schedule(mock_request)

        assert resp.status == 400

    async def test_once_missing_run_at_returns_400(self, db, mock_request):
        """schedule_data for 'once' without 'run_at' is rejected."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app[CHAT_ID_KEY] = 123
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
        mock_request.app[CHAT_ID_KEY] = 123
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
        mock_request.app[CHAT_ID_KEY] = 123
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
        mock_request.app[CHAT_ID_KEY] = 123
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
        mock_request.app[CHAT_ID_KEY] = 123
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
        mock_request.app[CHAT_ID_KEY] = 123
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
        mock_request.app[CHAT_ID_KEY] = 123
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
        INTERNAL_API_AUTH_KEY: _make_internal_api_auth(),
        GENERIC_WEBHOOK_SECRET_KEY: "signing-secret",
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

    async def test_unlisted_service_returns_403_before_proxy_call(self, service_request):
        """A service-scoped principal cannot select a different service name."""
        service_request.match_info = {"name": "billing"}
        service_request.json = AsyncMock(return_value={"body": {}})

        with patch("kai.services.call_service", new_callable=AsyncMock) as mock_call:
            resp = await _handle_service_call(service_request)

        assert resp.status == 403
        assert json.loads(resp.body.decode()) == {"error": "Service is not authorized for this principal"}
        mock_call.assert_not_called()

    async def test_user_without_service_grants_lacks_scope(self, service_request):
        """An empty allowlist omits the broad service-call scope entirely."""
        service_request.headers = {"X-Webhook-Secret": "other-secret"}
        service_request.json = AsyncMock(return_value={"body": {}})

        with patch("kai.services.call_service", new_callable=AsyncMock) as mock_call:
            resp = await _handle_service_call(service_request)

        assert resp.status == 403
        assert resp.text == "Credential is not authorized for this operation"
        mock_call.assert_not_called()


# ── _resolve_chat_id ────────────────────────────────────────────────


class TestResolveChatId:
    def _principal(self, chat_id: int = 123) -> InternalAPIPrincipal:
        """Resolve a deterministic test principal from its credential."""
        auth = _make_internal_api_auth()
        credential = "test-secret" if chat_id == 123 else "other-secret"
        principal = auth.authenticate(credential)
        assert principal is not None
        return principal

    def test_mismatched_explicit_chat_id_rejected(self):
        """Rejects a request-supplied identity that differs from the principal."""
        principal = self._principal()
        with pytest.raises(UnauthorizedChatIdError, match="does not match authenticated principal"):
            _resolve_chat_id(principal, {"chat_id": 42})

    def test_omitted_chat_id_uses_principal(self):
        """Omitted chat_id resolves to the authenticated principal."""
        principal = self._principal()
        assert _resolve_chat_id(principal, {}) == 123

    def test_invalid_non_numeric(self):
        """Raises ValueError for non-numeric chat_id."""
        principal = self._principal()
        with pytest.raises(ValueError, match="must be an integer"):
            _resolve_chat_id(principal, {"chat_id": "abc"})

    def test_invalid_float(self):
        """Raises ValueError for non-integer float chat_id."""
        principal = self._principal()
        with pytest.raises(ValueError, match="must be an integer"):
            _resolve_chat_id(principal, {"chat_id": 12345.6})

    def test_invalid_bool(self):
        """Raises ValueError for boolean chat_id."""
        principal = self._principal()
        with pytest.raises(ValueError, match="must be an integer"):
            _resolve_chat_id(principal, {"chat_id": True})

    def test_integer_like_float_accepted(self):
        """Integer-like float (e.g. 42.0) is accepted."""
        principal = self._principal(chat_id=456)
        assert _resolve_chat_id(principal, {"chat_id": 456.0}) == 456

    def test_string_integer_accepted(self):
        """String-encoded integer (e.g. from JSON) is accepted."""
        principal = self._principal(chat_id=456)
        assert _resolve_chat_id(principal, {"chat_id": "456"}) == 456


class TestGetJobsChatIdRouting:
    @pytest.mark.asyncio
    async def test_query_param_routes_to_user(self, db, mock_request):
        """GET /api/jobs?chat_id=456 returns jobs for that user."""
        mock_request.headers = {"X-Webhook-Secret": "other-secret"}
        mock_request.app[CHAT_ID_KEY] = 123
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
        mock_request.app[CHAT_ID_KEY] = 123
        mock_request.query = {"chat_id": "abc"}

        resp = await _handle_get_jobs(mock_request)
        assert resp.status == 400


class TestScheduleChatIdRouting:
    @pytest.mark.asyncio
    async def test_explicit_chat_id_in_body(self, db, mock_request):
        """POST /api/schedule with chat_id routes job to that user."""
        mock_request.headers = {"X-Webhook-Secret": "other-secret"}
        mock_request.app[CHAT_ID_KEY] = 123
        mock_request.app[TELEGRAM_APP_KEY].job_queue = MagicMock()
        mock_request.app[TELEGRAM_APP_KEY].job_queue.jobs.return_value = []

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
    async def test_external_signing_secret_is_not_an_api_credential(self, mock_request):
        """Possession of the GitHub/generic signing secret does not authorize API calls."""
        mock_request.headers = {"X-Webhook-Secret": "signing-secret"}
        mock_request.json = AsyncMock(return_value={"text": "hello"})

        resp = await _handle_send_message(mock_request)

        assert resp.status == 401
        mock_request.app[TELEGRAM_BOT_KEY].send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_notification_credential_cannot_manage_jobs(self, mock_request):
        """A review/triage notification credential is limited to sending messages."""
        auth = mock_request.app[INTERNAL_API_AUTH_KEY]
        credential = auth.notification_credential_for(123)
        mock_request.headers = {"X-Webhook-Secret": credential}

        mock_request.json = AsyncMock(return_value={"text": "review complete"})
        send_resp = await _handle_send_message(mock_request)
        assert send_resp.status == 200

        schedule_resp = await _handle_schedule(mock_request)
        assert schedule_resp.status == 403

    @pytest.mark.asyncio
    async def test_unauthorized_chat_id_returns_403(self, db, mock_request):
        """POST /api/schedule with chat_id differing from the principal returns 403."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app[TELEGRAM_APP_KEY].job_queue = MagicMock()
        mock_request.app[TELEGRAM_APP_KEY].job_queue.jobs.return_value = []

        mock_request.json = AsyncMock(
            return_value={
                "name": "Evil Job",
                "prompt": "test",
                "schedule_type": "daily",
                "schedule_data": {"times": ["09:00"]},
                "chat_id": 999999,  # differs from authenticated principal 123
            }
        )

        resp = await _handle_schedule(mock_request)
        assert resp.status == 403

    @pytest.mark.asyncio
    async def test_other_authorized_chat_id_rejected(self, db, mock_request):
        """A credential for user 123 cannot schedule work as user 456."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app[TELEGRAM_APP_KEY].job_queue = MagicMock()
        mock_request.app[TELEGRAM_APP_KEY].job_queue.jobs.return_value = []

        mock_request.json = AsyncMock(
            return_value={
                "name": "Cross-user Job",
                "prompt": "test",
                "schedule_type": "daily",
                "schedule_data": {"times": ["09:00"]},
                "chat_id": 456,  # authorized user, but not this caller's principal
            }
        )

        resp = await _handle_schedule(mock_request)
        assert resp.status == 403
        assert await sessions.get_jobs(456) == []

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
        """_resolve_chat_id rejects any identity other than the principal."""
        principal = _make_internal_api_auth().authenticate("test-secret")
        assert principal is not None

        with pytest.raises(UnauthorizedChatIdError):
            _resolve_chat_id(principal, {"chat_id": 999999})

    def test_resolve_chat_id_accepts_matching_principal(self):
        """_resolve_chat_id accepts an explicit repetition of the principal."""
        principal = _make_internal_api_auth().authenticate("test-secret")
        assert principal is not None
        assert _resolve_chat_id(principal, {"chat_id": 123}) == 123


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
    async def test_omitted_chat_id_uses_authenticated_principal(self, db, mock_request):
        """Omitted identity resolves to the credential owner, never the app default."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        # A deliberately different app default proves it is not an auth fallback.
        mock_request.app[CHAT_ID_KEY] = 456
        await sessions.create_job(
            chat_id=123,
            name="Principal Job",
            job_type="reminder",
            prompt="test",
            schedule_type="daily",
            schedule_data='{"times": ["09:00"]}',
        )

        resp = await _handle_get_jobs(mock_request)
        assert resp.status == 200
        body = json.loads(resp.body.decode())
        assert len(body) == 1
        assert body[0]["name"] == "Principal Job"

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
        mock_request.headers = {"X-Webhook-Secret": "other-secret"}
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
    async def test_omitted_chat_id_uses_principal(self, db, mock_request):
        """GET /api/jobs/{id} without chat_id uses the credential owner."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app[CHAT_ID_KEY] = 456
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
        mock_request.headers = {"X-Webhook-Secret": "other-secret"}
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
        mock_request.headers = {"X-Webhook-Secret": "other-secret"}
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
    async def test_omitted_chat_id_uses_principal(self, db, mock_request):
        """DELETE without chat_id uses the credential owner."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app[CHAT_ID_KEY] = 456
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
        mock_request.headers = {"X-Webhook-Secret": "other-secret"}
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
        mock_request.headers = {"X-Webhook-Secret": "other-secret"}
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
    async def test_omitted_chat_id_uses_principal(self, db, mock_request):
        """PATCH without chat_id in the body uses the credential owner."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app[CHAT_ID_KEY] = 456
        job_id = await sessions.create_job(
            chat_id=123,
            name="Original",
            job_type="reminder",
            prompt="test",
            schedule_type="once",
            schedule_data='{"run_at": "2026-06-01T12:00:00+00:00"}',
        )
        mock_request.match_info = {"id": str(job_id)}
        # No chat_id in body; the token still resolves to user 123.
        mock_request.json = AsyncMock(return_value={"name": "Updated"})

        resp = await _handle_update_job(mock_request)
        assert resp.status == 200
        job = await sessions.get_job_by_id(job_id)
        assert job is not None
        assert job["name"] == "Updated"

    @pytest.mark.asyncio
    async def test_user_can_update_own_job(self, db, mock_request):
        """PATCH with chat_id in body updates the user's own job."""
        mock_request.headers = {"X-Webhook-Secret": "other-secret"}
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
        mock_request.headers = {"X-Webhook-Secret": "other-secret"}
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


# ── /api/memory/* ────────────────────────────────────────────────────
# Tests for the four memory REST endpoints. The underlying primitives
# (memory.add_structured, memory.search, memory.get_stats,
# memory.delete_all, memory.is_enabled) are independently tested in
# tests/test_memory.py - these tests cover only handler-level behavior
# (auth, validation, JSON serialization, status-code mapping, and the
# symmetric is_enabled() precheck).
#
# Patch targets are kai.memory.<func> because webhook.py imports memory
# as a module (`from kai import memory`) and calls it via the module
# attribute. Patching `kai.webhook.add_structured` would miss because
# that name does not exist on webhook.


class TestMemoryAdd:
    """POST /api/memory/add"""

    async def test_returns_200_with_id_on_success(self, mock_request):
        """Happy path: enabled memory + valid body -> 200 with the new id."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"content": "User likes Earl Grey"})

        with (
            patch("kai.memory.is_enabled", return_value=True),
            patch("kai.memory.add_structured", return_value="mem-uuid-123"),
        ):
            resp = await _handle_memory_add(mock_request)

        assert resp.status == 200
        body = json.loads(resp.body.decode())
        assert body == {"id": "mem-uuid-123"}

    async def test_returns_503_when_memory_disabled(self, mock_request):
        """Precheck path: is_enabled() False -> 503; primitive NOT called."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"content": "anything"})

        with patch("kai.memory.is_enabled", return_value=False), patch("kai.memory.add_structured") as mock_add:
            resp = await _handle_memory_add(mock_request)

        assert resp.status == 503
        body = json.loads(resp.body.decode())
        assert body == {"error": "Memory system disabled"}
        # The whole point of the precheck is to avoid invoking the
        # primitive when memory is off; assert that contract explicitly.
        mock_add.assert_not_called()

    async def test_returns_500_when_storage_fails(self, mock_request):
        """Storage-failure path: enabled but add returns None -> 500."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"content": "anything"})

        with patch("kai.memory.is_enabled", return_value=True), patch("kai.memory.add_structured", return_value=None):
            resp = await _handle_memory_add(mock_request)

        assert resp.status == 500
        body = json.loads(resp.body.decode())
        assert body == {"error": "Memory storage failed"}

    async def test_returns_400_on_missing_content(self, mock_request):
        """Required-field validation: missing `content` -> 400."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"memory_type": "fact"})

        resp = await _handle_memory_add(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "content" in body["error"]

    async def test_returns_400_on_whitespace_only_content(self, mock_request):
        """Empty-after-strip check: whitespace-only content -> 400."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"content": "   \n\t  "})

        resp = await _handle_memory_add(mock_request)

        assert resp.status == 400

    async def test_returns_400_on_invalid_json(self, mock_request):
        """Malformed JSON body -> 400 with a clear error."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(side_effect=json.JSONDecodeError("expecting value", "", 0))

        resp = await _handle_memory_add(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert body == {"error": "Invalid JSON"}

    async def test_returns_401_on_missing_secret(self, mock_request):
        """No X-Webhook-Secret header -> 401 from the decorator."""
        mock_request.headers = {}
        mock_request.json = AsyncMock(return_value={"content": "x"})

        resp = await _handle_memory_add(mock_request)

        assert resp.status == 401

    async def test_returns_401_on_wrong_secret(self, mock_request):
        """Wrong X-Webhook-Secret -> 401 from the decorator."""
        mock_request.headers = {"X-Webhook-Secret": "nope"}
        mock_request.json = AsyncMock(return_value={"content": "x"})

        resp = await _handle_memory_add(mock_request)

        assert resp.status == 401

    async def test_passes_optional_fields_through(self, mock_request):
        """memory_type, tags, metadata are forwarded to add_structured verbatim."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(
            return_value={
                "content": "User prefers dark mode",
                "memory_type": "preference",
                "tags": ["ui", "preference"],
                "metadata": {"source": "explicit"},
            }
        )

        with (
            patch("kai.memory.is_enabled", return_value=True),
            patch("kai.memory.add_structured", return_value="id-1") as mock_add,
        ):
            await _handle_memory_add(mock_request)

        # Verify each optional field reached the primitive intact - if any
        # got dropped or renamed, this catches the regression at the
        # handler boundary rather than letting it surface as missing data
        # in the vector store.
        kwargs = mock_add.call_args.kwargs
        assert kwargs["memory_type"] == "preference"
        assert kwargs["tags"] == ["ui", "preference"]
        assert kwargs["metadata"] == {"source": "explicit"}

    async def test_stringifies_chat_id_for_user_id(self, mock_request):
        """Handler converts int chat_id -> str user_id at the memory boundary.

        This is load-bearing: Mem0 keys memories by the string form of
        user_id. A missed cast would isolate API-stored memories from the
        existing facts under the same chat_id (silently, since both writes
        succeed).
        """
        mock_request.headers = {"X-Webhook-Secret": "other-secret"}
        mock_request.json = AsyncMock(return_value={"content": "x", "chat_id": 456})

        with (
            patch("kai.memory.is_enabled", return_value=True),
            patch("kai.memory.add_structured", return_value="id-1") as mock_add,
        ):
            await _handle_memory_add(mock_request)

        kwargs = mock_add.call_args.kwargs
        assert kwargs["user_id"] == "456"
        assert isinstance(kwargs["user_id"], str)

    async def test_returns_403_on_unauthorized_chat_id(self, mock_request):
        """chat_id differing from the principal -> 403, primitive not called."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        # The credential resolves to principal 123, not the requested 999.
        mock_request.json = AsyncMock(return_value={"content": "x", "chat_id": 999})

        with patch("kai.memory.is_enabled", return_value=True), patch("kai.memory.add_structured") as mock_add:
            resp = await _handle_memory_add(mock_request)

        assert resp.status == 403
        mock_add.assert_not_called()

    async def test_returns_500_when_add_structured_raises(self, mock_request):
        """Unexpected exception in add_structured() -> clean 500 JSON.

        Defense-in-depth: add_structured catches its own exceptions and
        returns None (handled by the None-check 500 path), but the
        handler also wraps in case the call raises BEFORE reaching that
        internal try (init failure, bad argument shape, etc.). Pins the
        contract that BOTH failure modes produce the same 500 JSON body.
        """
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"content": "x"})

        with (
            patch("kai.memory.is_enabled", return_value=True),
            patch("kai.memory.add_structured", side_effect=RuntimeError("boom")),
        ):
            resp = await _handle_memory_add(mock_request)

        assert resp.status == 500
        body = json.loads(resp.body.decode())
        assert body == {"error": "Memory storage failed"}

    async def test_returns_400_on_non_string_content(self, mock_request):
        """Non-string content (number, bool, list, etc.) -> 400, not 500.

        Without the isinstance guard at the API boundary, a JSON number
        or null reaching .strip() would raise AttributeError and escape
        to aiohttp as an unstyled 500. This test pins the contract that
        the handler always returns a clean 400 with a JSON error body.
        """
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"content": 123})

        resp = await _handle_memory_add(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "content" in body["error"]

    async def test_returns_400_on_non_string_memory_type(self, mock_request):
        """memory_type must be a string when provided."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"content": "x", "memory_type": 42})

        resp = await _handle_memory_add(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "memory_type" in body["error"]

    async def test_treats_null_memory_type_as_default(self, mock_request):
        """memory_type: null falls back to "fact" rather than 400.

        Matches the lenient None handling already used for tags and
        metadata (None is equivalent to omitting the field). Pinning
        the contract so a future "simplify" that switches back to
        `payload.get("memory_type", "fact")` and rejects explicit
        nulls would fail this test.
        """
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"content": "x", "memory_type": None})

        with (
            patch("kai.memory.is_enabled", return_value=True),
            patch("kai.memory.add_structured", return_value="id-1") as mock_add,
        ):
            resp = await _handle_memory_add(mock_request)

        assert resp.status == 200
        assert mock_add.call_args.kwargs["memory_type"] == "fact"

    async def test_returns_400_on_non_list_tags(self, mock_request):
        """tags must be a list when provided (not a string or dict)."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"content": "x", "tags": "single-tag"})

        resp = await _handle_memory_add(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "tags" in body["error"]

    async def test_returns_400_on_tags_with_non_string_element(self, mock_request):
        """All tag elements must be strings; mixed-type list -> 400."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"content": "x", "tags": ["ok", 42]})

        resp = await _handle_memory_add(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "tags" in body["error"]

    async def test_returns_400_on_non_dict_metadata(self, mock_request):
        """metadata must be a JSON object when provided."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"content": "x", "metadata": ["a", "b"]})

        resp = await _handle_memory_add(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "metadata" in body["error"]


class TestMemorySearch:
    """POST /api/memory/search"""

    async def test_returns_200_with_results(self, mock_request):
        """Happy path: results from search() are serialized into {"results": [...]}."""
        from kai.memory import MemoryResult

        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"query": "tea preferences"})

        fake_hit = MemoryResult(
            id="m1",
            text="User likes Earl Grey",
            score=0.85,
            memory_type="fact",
            metadata={"source": "extracted", "tags": ["preference"]},
            created_at="2026-04-23T10:00:00Z",
        )
        with patch("kai.memory.is_enabled", return_value=True), patch("kai.memory.search", return_value=[fake_hit]):
            resp = await _handle_memory_search(mock_request)

        assert resp.status == 200
        body = json.loads(resp.body.decode())
        # Verify the result structure preserves the dataclass shape via
        # asdict; metadata round-trips as a nested dict including its
        # tags list.
        assert len(body["results"]) == 1
        assert body["results"][0]["id"] == "m1"
        assert body["results"][0]["text"] == "User likes Earl Grey"
        assert body["results"][0]["metadata"]["tags"] == ["preference"]

    async def test_returns_200_with_empty_results(self, mock_request):
        """No matches: still 200, just an empty list - 503 is reserved for disabled."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"query": "no matches"})

        with patch("kai.memory.is_enabled", return_value=True), patch("kai.memory.search", return_value=[]):
            resp = await _handle_memory_search(mock_request)

        assert resp.status == 200
        body = json.loads(resp.body.decode())
        assert body == {"results": []}

    async def test_returns_503_when_memory_disabled(self, mock_request):
        """Precheck overrides search()'s graceful-degrade [] return."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"query": "anything"})

        with patch("kai.memory.is_enabled", return_value=False), patch("kai.memory.search") as mock_search:
            resp = await _handle_memory_search(mock_request)

        assert resp.status == 503
        # Without the precheck, search() would have returned [] and the
        # API would have returned 200 - indistinguishable from "no
        # matches". Asserting not_called makes the M-8 contract a test
        # invariant, not just a code comment.
        mock_search.assert_not_called()

    async def test_returns_400_on_missing_query(self, mock_request):
        """Required-field validation."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={})

        resp = await _handle_memory_search(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "query" in body["error"]

    async def test_returns_400_on_empty_query(self, mock_request):
        """Whitespace-only query is rejected like missing."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"query": "   "})

        resp = await _handle_memory_search(mock_request)

        assert resp.status == 400

    async def test_returns_401_on_bad_secret(self, mock_request):
        mock_request.headers = {"X-Webhook-Secret": "nope"}
        mock_request.json = AsyncMock(return_value={"query": "x"})

        resp = await _handle_memory_search(mock_request)

        assert resp.status == 401

    async def test_passes_limit_through(self, mock_request):
        """Caller-supplied `limit` is forwarded as-is to search()."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"query": "x", "limit": 5})

        with (
            patch("kai.memory.is_enabled", return_value=True),
            patch("kai.memory.search", return_value=[]) as mock_search,
        ):
            await _handle_memory_search(mock_request)

        assert mock_search.call_args.kwargs["limit"] == 5

    async def test_passes_none_limit_when_omitted(self, mock_request):
        """When `limit` is omitted, the handler passes None so search()
        falls back to its config default rather than a hardcoded number."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"query": "x"})

        with (
            patch("kai.memory.is_enabled", return_value=True),
            patch("kai.memory.search", return_value=[]) as mock_search,
        ):
            await _handle_memory_search(mock_request)

        assert mock_search.call_args.kwargs["limit"] is None

    async def test_returns_400_on_non_string_query(self, mock_request):
        """Non-string query (number, bool, list, etc.) -> 400.

        Same isinstance-before-.strip() guard as content in the add
        handler. Pinning the contract so a future refactor can't
        regress to the AttributeError-escapes-as-500 failure mode.
        """
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"query": 42})

        resp = await _handle_memory_search(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "query" in body["error"]

    async def test_returns_400_on_non_int_limit(self, mock_request):
        """`limit` must be an int when provided; string -> 400."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"query": "x", "limit": "five"})

        resp = await _handle_memory_search(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        assert "limit" in body["error"]

    async def test_returns_400_on_zero_limit(self, mock_request):
        """`limit` must be positive; 0 -> 400."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"query": "x", "limit": 0})

        resp = await _handle_memory_search(mock_request)

        assert resp.status == 400

    async def test_returns_400_on_negative_limit(self, mock_request):
        """Negative `limit` -> 400."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"query": "x", "limit": -3})

        resp = await _handle_memory_search(mock_request)

        assert resp.status == 400

    async def test_returns_400_on_bool_limit(self, mock_request):
        """`limit: true` rejected: bool is an int subclass in Python.

        Without the explicit isinstance(limit, bool) short-circuit,
        `{"limit": true}` would pass `isinstance(limit, int)` and be
        treated as `limit=1`, silently truncating results. This test
        is the contract that this Python-specific footgun stays closed.
        """
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"query": "x", "limit": True})

        resp = await _handle_memory_search(mock_request)

        assert resp.status == 400

    async def test_returns_500_when_search_raises(self, mock_request):
        """Unexpected exception in search() -> clean 500 JSON, not aiohttp 500 page.

        Defense-in-depth: search() catches its own exceptions today, but
        the handler still wraps to ensure the 500 contract is structural
        rather than dependent on the primitive's internal behavior.
        """
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"query": "x"})

        with (
            patch("kai.memory.is_enabled", return_value=True),
            patch("kai.memory.search", side_effect=RuntimeError("boom")),
        ):
            resp = await _handle_memory_search(mock_request)

        assert resp.status == 500
        body = json.loads(resp.body.decode())
        assert body == {"error": "Memory search failed"}

    async def test_returns_500_when_serialization_raises(self, mock_request):
        """asdict failure on a non-dataclass return -> wrapped into 500.

        Pins the contract that the try/except guard covers BOTH the
        primitive call and the asdict serialization, not just the
        primitive. Without the wider guard, asdict raising TypeError
        would escape to aiohttp as an HTML 500.
        """
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"query": "x"})

        # Returning a non-dataclass forces asdict() to TypeError.
        with (
            patch("kai.memory.is_enabled", return_value=True),
            patch("kai.memory.search", return_value=["not a dataclass"]),
        ):
            resp = await _handle_memory_search(mock_request)

        assert resp.status == 500
        body = json.loads(resp.body.decode())
        assert body == {"error": "Memory search failed"}

    async def test_returns_403_on_unauthorized_chat_id(self, mock_request):
        """chat_id differing from the principal -> 403, primitive not called."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"query": "x", "chat_id": 999})

        with (
            patch("kai.memory.is_enabled", return_value=True),
            patch("kai.memory.search") as mock_search,
        ):
            resp = await _handle_memory_search(mock_request)

        assert resp.status == 403
        mock_search.assert_not_called()


class TestMemoryStats:
    """GET /api/memory/stats"""

    async def test_returns_200_with_stats_at_top_level(self, mock_request):
        """Stats are returned bare, not wrapped in {"results": ...}."""
        from kai.memory import MemoryStats

        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.query = {}

        fake_stats = MemoryStats(
            total_count=5,
            by_type={"fact": 4, "preference": 1},
            extracted_count=4,
            by_tag={"food": 2, "ui": 1},
            confidence_min=0.7,
            confidence_median=0.8,
            confidence_max=0.95,
            confidence_below_0_7=0,
            confidence_below_0_6=0,
            confirmation_quote_count=1,
            by_prompt_version={"v2": 4},
        )
        with patch("kai.memory.is_enabled", return_value=True), patch("kai.memory.get_stats", return_value=fake_stats):
            resp = await _handle_memory_stats(mock_request)

        assert resp.status == 200
        body = json.loads(resp.body.decode())
        # Top-level fields means body["total_count"] not body["stats"]["total_count"].
        # Catches a regression that wraps the stats dict accidentally.
        assert body["total_count"] == 5
        assert body["extracted_count"] == 4
        assert body["by_type"] == {"fact": 4, "preference": 1}

    async def test_serializes_null_confidence_fields(self, mock_request):
        """Fresh user (extracted_count == 0): confidence_* fields ship as JSON null.

        The handler MUST preserve None as JSON null, not coerce to 0 or
        omit the field, because CLAUDE.md tells inner Claude that null
        means "no extracted facts to summarize" - changing the encoding
        would silently break that contract.
        """
        from kai.memory import MemoryStats

        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.query = {}

        empty_stats = MemoryStats(total_count=0, by_type={})
        with patch("kai.memory.is_enabled", return_value=True), patch("kai.memory.get_stats", return_value=empty_stats):
            resp = await _handle_memory_stats(mock_request)

        body = json.loads(resp.body.decode())
        assert body["confidence_min"] is None
        assert body["confidence_median"] is None
        assert body["confidence_max"] is None

    async def test_returns_503_when_memory_disabled(self, mock_request):
        """Precheck overrides get_stats()'s zeroed-stats degrade."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.query = {}

        with patch("kai.memory.is_enabled", return_value=False), patch("kai.memory.get_stats") as mock_get_stats:
            resp = await _handle_memory_stats(mock_request)

        assert resp.status == 503
        mock_get_stats.assert_not_called()

    async def test_reads_chat_id_from_query_string(self, mock_request):
        """GET endpoint: chat_id comes from query params, mirroring _handle_get_jobs."""
        from kai.memory import MemoryStats

        mock_request.headers = {"X-Webhook-Secret": "other-secret"}
        # The user-456 credential matches the string query parameter.
        mock_request.query = {"chat_id": "456"}

        with (
            patch("kai.memory.is_enabled", return_value=True),
            patch("kai.memory.get_stats", return_value=MemoryStats(total_count=0, by_type={})) as mock_get_stats,
        ):
            resp = await _handle_memory_stats(mock_request)

        assert resp.status == 200
        # Verify the int->str cast happened at the memory boundary.
        assert mock_get_stats.call_args.kwargs["user_id"] == "456"

    async def test_returns_401_on_bad_secret(self, mock_request):
        mock_request.headers = {"X-Webhook-Secret": "nope"}
        mock_request.query = {}

        resp = await _handle_memory_stats(mock_request)

        assert resp.status == 401

    async def test_returns_403_on_unauthorized_chat_id(self, mock_request):
        """Query-string chat_id differing from the principal -> 403."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.query = {"chat_id": "999"}

        with patch("kai.memory.is_enabled", return_value=True), patch("kai.memory.get_stats") as mock_get_stats:
            resp = await _handle_memory_stats(mock_request)

        assert resp.status == 403
        mock_get_stats.assert_not_called()

    async def test_returns_500_when_get_stats_raises(self, mock_request):
        """Unexpected exception in get_stats() -> clean 500 JSON.

        get_stats() doesn't catch its own exceptions today (the
        aggregation could raise on a malformed row), so this guard is
        the actual error contract, not just defense-in-depth.
        """
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.query = {}

        with (
            patch("kai.memory.is_enabled", return_value=True),
            patch("kai.memory.get_stats", side_effect=RuntimeError("boom")),
        ):
            resp = await _handle_memory_stats(mock_request)

        assert resp.status == 500
        body = json.loads(resp.body.decode())
        assert body == {"error": "Memory stats failed"}

    async def test_returns_500_when_serialization_raises(self, mock_request):
        """asdict failure on a non-dataclass return -> wrapped into 500.

        Same wider-guard contract as the search handler: try/except
        covers both get_stats() and the asdict step. A future refactor
        that returns a dict (or anything non-dataclass) instead of
        MemoryStats would otherwise leak as HTML 500.
        """
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.query = {}

        # asdict() raises TypeError on a plain dict.
        with (
            patch("kai.memory.is_enabled", return_value=True),
            patch("kai.memory.get_stats", return_value={"not": "a dataclass"}),
        ):
            resp = await _handle_memory_stats(mock_request)

        assert resp.status == 500
        body = json.loads(resp.body.decode())
        assert body == {"error": "Memory stats failed"}


class TestMemoryDeleteAll:
    """DELETE /api/memory/all"""

    _CONFIRM = "delete-all-memories"

    async def test_returns_200_on_correct_confirm(self, mock_request):
        """Happy path: correct confirm token -> 200 with deletion ack."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"confirm": self._CONFIRM})

        with patch("kai.memory.is_enabled", return_value=True), patch("kai.memory.delete_all"):
            resp = await _call_memory_delete_all_as_authorized(mock_request)

        assert resp.status == 200
        body = json.loads(resp.body.decode())
        assert body == {"status": "deleted"}

    async def test_returns_400_on_missing_confirm(self, mock_request):
        """No confirm field -> 400; delete_all is never called."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={})

        with patch("kai.memory.delete_all") as mock_del:
            resp = await _call_memory_delete_all_as_authorized(mock_request)

        assert resp.status == 400
        body = json.loads(resp.body.decode())
        # Error body must echo the expected token so a stray curl gets
        # a directly fixable error message.
        assert self._CONFIRM in body["error"]
        mock_del.assert_not_called()

    async def test_returns_400_on_wrong_confirm(self, mock_request):
        """Wrong confirm value -> 400; delete_all is never called."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"confirm": "yes-please"})

        with patch("kai.memory.delete_all") as mock_del:
            resp = await _call_memory_delete_all_as_authorized(mock_request)

        assert resp.status == 400
        mock_del.assert_not_called()

    async def test_returns_503_when_memory_disabled(self, mock_request):
        """Precheck blocks delete even when confirm is correct."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"confirm": self._CONFIRM})

        with patch("kai.memory.is_enabled", return_value=False), patch("kai.memory.delete_all") as mock_del:
            resp = await _call_memory_delete_all_as_authorized(mock_request)

        assert resp.status == 503
        # The user explicitly asked for a delete; if memory is off, we
        # tell them the delete didn't actually run rather than silently
        # acking a no-op.
        mock_del.assert_not_called()

    async def test_calls_delete_all_with_stringified_user_id(self, mock_request):
        """int -> str cast at the memory boundary, same as add."""
        mock_request.headers = {"X-Webhook-Secret": "other-secret"}
        mock_request.json = AsyncMock(return_value={"confirm": self._CONFIRM, "chat_id": 456})

        with patch("kai.memory.is_enabled", return_value=True), patch("kai.memory.delete_all") as mock_del:
            await _call_memory_delete_all_as_authorized(mock_request, chat_id=456)

        assert mock_del.call_args.kwargs["user_id"] == "456"
        assert isinstance(mock_del.call_args.kwargs["user_id"], str)

    async def test_returns_401_on_bad_secret(self, mock_request):
        mock_request.headers = {"X-Webhook-Secret": "nope"}
        mock_request.json = AsyncMock(return_value={"confirm": self._CONFIRM})

        resp = await _handle_memory_delete_all(mock_request)

        assert resp.status == 401

    async def test_returns_500_when_delete_all_raises(self, mock_request):
        """Unexpected exception in delete_all() -> clean 500 JSON.

        Same defense-in-depth pattern as the search and stats handlers
        (added in the prior review round). delete_all() catches its own
        internal errors today but the handler-level guard makes the 500
        contract structural rather than primitive-dependent.
        """
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"confirm": self._CONFIRM})

        with (
            patch("kai.memory.is_enabled", return_value=True),
            patch("kai.memory.delete_all", side_effect=RuntimeError("boom")),
        ):
            resp = await _call_memory_delete_all_as_authorized(mock_request)

        assert resp.status == 500
        body = json.loads(resp.body.decode())
        assert body == {"error": "Memory delete failed"}

    async def test_returns_403_on_unauthorized_chat_id(self, mock_request):
        """chat_id differing from the principal -> 403, primitive not called.

        Verifies the 403 path runs even when the confirm token is
        correct: an unauthorized caller with the right token should
        still be blocked at the chat_id resolution step.
        """
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"confirm": self._CONFIRM, "chat_id": 999})

        with (
            patch("kai.memory.is_enabled", return_value=True),
            patch("kai.memory.delete_all") as mock_del,
        ):
            resp = await _call_memory_delete_all_as_authorized(mock_request)

        assert resp.status == 403
        mock_del.assert_not_called()

    async def test_persistent_agent_scope_cannot_delete_all(self, mock_request):
        """A normal persistent-agent credential is denied before deletion."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"confirm": self._CONFIRM})

        with patch("kai.memory.delete_all") as mock_del:
            resp = await _handle_memory_delete_all(mock_request)

        assert resp.status == 403
        assert resp.text == "Credential is not authorized for this operation"
        mock_del.assert_not_called()


# ── Cross-handler payload-shape rejection (issue #377) ────────────────


# Every JSON-body handler must reject non-object JSON shapes (null,
# lists, scalars, strings) with a clean 400 instead of crashing on
# AttributeError when the next payload.get() is called. The existing
# `try: payload = await request.json() except json.JSONDecodeError`
# guard catches malformed JSON only - valid JSON that happens to be
# `null`, `[]`, `42`, or `"string"` parses successfully and then
# escapes as an unstyled HTML 500 traceback the moment a `.get()` runs.
#
# The fix is one isinstance check after the existing JSONDecodeError
# guard, applied uniformly across all 9 handlers. These tests pin the
# rejection at every handler so a future refactor that drops the check
# at any one site fails loudly here.

# (handler_func, fixture_overrides) tuples for each in-scope handler.
# fixture_overrides is a callable that takes the mock_request and
# applies any handler-specific setup needed to reach the isinstance
# check (e.g., match_info["name"] for service_call). Keep the override
# minimal - the goal is to reach the isinstance check, not to exercise
# the rest of the handler.
_NON_OBJECT_HANDLERS = [
    pytest.param(
        _handle_generic,
        lambda r: setattr(r, "headers", {"X-Webhook-Secret": "signing-secret"}),
        id="generic",
    ),
    pytest.param(_handle_schedule, lambda r: None, id="schedule"),
    pytest.param(
        _handle_update_job,
        lambda r: r.match_info.update({"id": "1"}),
        id="update_job",
    ),
    pytest.param(
        _handle_service_call,
        lambda r: r.match_info.update({"name": "perplexity"}),
        id="service_call",
    ),
    pytest.param(_handle_send_message, lambda r: None, id="send_message"),
    pytest.param(_handle_send_file, lambda r: None, id="send_file"),
    pytest.param(_handle_memory_add, lambda r: None, id="memory_add"),
    pytest.param(_handle_memory_search, lambda r: None, id="memory_search"),
    pytest.param(_call_memory_delete_all_as_authorized, lambda r: None, id="memory_delete_all"),
]

# Non-object JSON shapes the handlers must reject. `True` is included
# because Python's `bool` is a subclass of `int`, so a True payload
# would pass an `isinstance(payload, int)` check that someone might
# someday be tempted to write - covers the bool/int subclass footgun.
_NON_OBJECT_PAYLOADS = [
    pytest.param(None, id="null"),
    pytest.param([], id="empty_list"),
    pytest.param([1, 2, 3], id="non_empty_list"),
    pytest.param(42, id="int"),
    pytest.param("a string", id="string"),
    pytest.param(True, id="bool"),
]


class TestNonObjectPayloadRejection:
    """Every JSON-body handler must reject non-object JSON shapes with
    400 instead of crashing on AttributeError."""

    @pytest.mark.parametrize("payload", _NON_OBJECT_PAYLOADS)
    @pytest.mark.parametrize("handler, override", _NON_OBJECT_HANDLERS)
    async def test_non_object_payload_returns_400(self, mock_request, handler, override, payload):
        # Apply the handler-specific override (e.g., match_info) to reach
        # the isinstance check. The mock_request fixture already supplies
        # webhook_secret, telegram_app/bot, chat_id, and match_info as
        # an empty dict ready for in-place updates.
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value=payload)
        override(mock_request)

        resp = await handler(mock_request)

        # The isinstance check fires before any handler-specific work
        # (storage, scheduling, sending), so no kai.memory / sessions /
        # bot patches are needed. A 400 response confirms the check
        # caught the bad shape; anything else (200, 500, exception)
        # means the check is missing or misordered for this handler.
        assert resp.status == 400, (
            f"{handler.__name__} accepted non-object payload {payload!r} (status={resp.status}); expected 400"
        )
