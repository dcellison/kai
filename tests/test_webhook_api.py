"""Integration tests for webhook HTTP API endpoints (jobs CRUD, file exchange)."""

import asyncio
import hashlib
import hmac as hmac_mod
import inspect
import json
from types import SimpleNamespace
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
    CORE_HOST_KEY,
    GENERIC_WEBHOOK_SECRET_KEY,
    GITHUB_WEBHOOK_SECRET_KEY,
    INTERNAL_API_AUTH_KEY,
    TELEGRAM_APP_KEY,
    TELEGRAM_BOT_KEY,
    TELEGRAM_WEBHOOK_SECRET_KEY,
    WORKSHOP_GITHUB_AUTOMATION_KEY,
    WORKSHOP_INTEGRATION_NOTIFICATIONS_KEY,
    WORKSHOP_PRINCIPAL_STORAGE_KEY,
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
    stop,
)
from kai.workshop.domain import AgentId, ChannelId, MessageId, PrincipalId
from kai.workshop.internal_api_contexts import WorkshopInternalAPIExecutionContext
from kai.workshop.proactive_publication import ProactivePublicationResult
from kai.workshop.scheduler import WorkshopScheduledJobRegistrationError
from kai.workshop.storage_namespaces import (
    WorkshopPrincipalStorageNamespace,
    WorkshopPrincipalStorageRegistry,
)
from tests.workshop_profiles import profile_id


def _make_internal_api_auth() -> InternalAPIAuth:
    """Create deterministic credentials for the two API test principals."""
    context_123 = _internal_api_context(123)
    context_456 = _internal_api_context(456)
    return InternalAPIAuth(
        {context_123: "test-secret", context_456: "other-secret"},
        allowed_services_by_profile={context_123.runtime_profile_id: {"perplexity"}},
    )


def _internal_api_context(runtime_config_id: int) -> WorkshopInternalAPIExecutionContext:
    return WorkshopInternalAPIExecutionContext(
        principal_id=PrincipalId(f"prn_{runtime_config_id:032x}"),
        channel_id=ChannelId(f"chn_{runtime_config_id:032x}"),
        agent_id=AgentId(f"agt_{runtime_config_id:032x}"),
        runtime_profile_id=profile_id(runtime_config_id),
        _runtime_config_id=runtime_config_id,
    )


def _principal_storage_registry() -> WorkshopPrincipalStorageRegistry:
    return WorkshopPrincipalStorageRegistry(
        (
            WorkshopPrincipalStorageNamespace(
                _internal_api_context(123).principal_id,
                profile_id(123),
                123,
            ),
            WorkshopPrincipalStorageNamespace(
                _internal_api_context(456).principal_id,
                profile_id(456),
                456,
            ),
        )
    )


def _runtime_config_id_for_authority(authority) -> int:
    return int(str(authority.principal_id).removeprefix("prn_"), 16)


class _CanonicalSchedulerDouble:
    """Exercise handler contracts while legacy session tests remain isolated."""

    def __init__(self) -> None:
        self.register_job = AsyncMock(return_value=True)
        self.remove_job = AsyncMock()

    async def create_job(self, authority, **fields) -> int:
        runtime_config_id = _runtime_config_id_for_authority(authority)
        job_id = await sessions.create_job(chat_id=runtime_config_id, **fields)
        try:
            if not await self.register_job(job_id):
                raise RuntimeError("registration failed")
        except Exception as exc:
            await sessions.deactivate_job(job_id, chat_id=runtime_config_id)
            raise WorkshopScheduledJobRegistrationError("registration failed") from exc
        return job_id

    async def list_jobs(self, authority) -> list[dict]:
        jobs = await sessions.get_jobs(_runtime_config_id_for_authority(authority))
        return [{key: value for key, value in job.items() if key != "chat_id"} for job in jobs]

    async def get_job(self, job_id: int, authority) -> dict | None:
        job = await sessions.get_job_by_id(job_id)
        if job is None or job["chat_id"] != _runtime_config_id_for_authority(authority):
            return None
        return {key: value for key, value in job.items() if key != "chat_id"}

    async def delete_job(self, job_id: int, authority) -> bool:
        deleted = await sessions.delete_job(
            job_id,
            chat_id=_runtime_config_id_for_authority(authority),
        )
        if deleted:
            await self.remove_job(job_id)
        return deleted

    async def update_job(self, job_id: int, authority, update) -> bool:
        runtime_config_id = _runtime_config_id_for_authority(authority)
        previous = await sessions.get_job_by_id(job_id)
        if previous is None or previous["chat_id"] != runtime_config_id:
            return False
        updated = await sessions.update_job(
            job_id,
            chat_id=runtime_config_id,
            name=update.name,
            prompt=update.prompt,
            schedule_type=update.schedule_type,
            schedule_data=update.schedule_data,
            auto_remove=update.auto_remove,
            notify_on_check=update.notify_on_check,
        )
        if not updated or (update.schedule_type is None and update.schedule_data is None):
            return updated
        try:
            if not await self.register_job(job_id):
                raise RuntimeError("registration failed")
        except Exception:
            await sessions.update_job(
                job_id,
                chat_id=runtime_config_id,
                name=previous["name"],
                prompt=previous["prompt"],
                schedule_type=previous["schedule_type"],
                schedule_data=previous["schedule_data"],
                auto_remove=previous["auto_remove"],
                notify_on_check=previous["notify_on_check"],
            )
            await self.register_job(job_id)
            raise
        return True


async def _call_memory_delete_all_as_authorized(request, *, chat_id: int = 123):
    """Exercise delete-all handler behavior with an explicitly privileged principal."""
    context = _internal_api_context(chat_id)
    principal = InternalAPIPrincipal(
        principal_id=context.principal_id,
        channel_id=context.channel_id,
        agent_id=context.agent_id,
        runtime_profile_id=context.runtime_profile_id,
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
def mock_request(tmp_path):
    """Create a minimal mock request with app dict and helpers."""
    request = MagicMock(spec=web.Request)
    request.app = {
        INTERNAL_API_AUTH_KEY: _make_internal_api_auth(),
        GENERIC_WEBHOOK_SECRET_KEY: "signing-secret",
        TELEGRAM_APP_KEY: MagicMock(),
        TELEGRAM_BOT_KEY: AsyncMock(),
        CHAT_ID_KEY: 123,
        ALLOWED_USER_IDS_KEY: {123, 456},
        CONFIG_KEY: SimpleNamespace(memory_projects={}),
        CORE_HOST_KEY: SimpleNamespace(
            services=SimpleNamespace(
                scheduler=_CanonicalSchedulerDouble(),
                runtime_pool=SimpleNamespace(
                    get_effective_workspace=AsyncMock(return_value=tmp_path),
                ),
                proactive_publication=SimpleNamespace(
                    publish_text=AsyncMock(return_value=ProactivePublicationResult(MessageId.new(), True, ())),
                    publish_file=AsyncMock(return_value=ProactivePublicationResult(MessageId.new(), True, ())),
                ),
            )
        ),
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

        scheduler = mock_request.app[CORE_HOST_KEY].services.scheduler
        scheduler.register_job.side_effect = [False, True]
        resp = await _handle_update_job(mock_request)

        assert resp.status == 500
        body = json.loads(resp.body.decode())
        assert body["error"] == "Failed to register job"
        scheduler.register_job.assert_has_awaits(
            [
                call(job_id),
                call(job_id),
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

        scheduler = mock_request.app[CORE_HOST_KEY].services.scheduler
        scheduler.register_job.side_effect = [RuntimeError("scheduler unavailable"), True]
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
    # Send-file resolves workspace through the credential's protected runtime
    # profile. The fixture supplies that canonical facade directly.
    runtime_pool = SimpleNamespace(
        get_effective_workspace=AsyncMock(return_value=tmp_path),
    )
    request = MagicMock(spec=web.Request)
    request.app = {
        INTERNAL_API_AUTH_KEY: _make_internal_api_auth(),
        GENERIC_WEBHOOK_SECRET_KEY: "signing-secret",
        CHAT_ID_KEY: 123,
        CORE_HOST_KEY: SimpleNamespace(
            services=SimpleNamespace(
                runtime_pool=runtime_pool,
                proactive_publication=SimpleNamespace(
                    publish_file=AsyncMock(return_value=ProactivePublicationResult(MessageId.new(), True, ()))
                ),
            ),
        ),
        WORKSHOP_PRINCIPAL_STORAGE_KEY: _principal_storage_registry(),
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
    send_file_request.app[CORE_HOST_KEY].services.runtime_pool.get_effective_workspace.return_value = workspace
    return send_file_request, workspace, data_dir


class TestSendFile:
    async def test_records_image_as_canonical_artifact(self, tmp_path, send_file_request):
        """Image files enter canonical publication before adapter delivery."""
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")

        send_file_request.json = AsyncMock(return_value={"path": str(img)})
        resp = await _handle_send_file(send_file_request)

        assert resp.status == 200
        body = json.loads(resp.body)
        assert body["status"] == "recorded"
        assert body["delivery"] == "not_configured"
        assert body["file"] == "photo.jpg"
        publish = send_file_request.app[CORE_HOST_KEY].services.proactive_publication.publish_file
        publish.assert_awaited_once()
        assert publish.await_args.kwargs["path"] == img

    async def test_records_document_as_canonical_artifact(self, tmp_path, send_file_request):
        """Documents use the same canonical publication service."""
        doc = tmp_path / "report.pdf"
        doc.write_bytes(b"%PDF-1.4 fake")

        send_file_request.json = AsyncMock(return_value={"path": str(doc)})
        resp = await _handle_send_file(send_file_request)

        assert resp.status == 200
        body = json.loads(resp.body)
        assert body["status"] == "recorded"
        send_file_request.app[CORE_HOST_KEY].services.proactive_publication.publish_file.assert_awaited_once()
        send_file_request.app[CORE_HOST_KEY].services.runtime_pool.get_effective_workspace.assert_awaited_once_with(
            profile_id(123)
        )

    async def test_caption_forwarded_to_canonical_publication(self, tmp_path, send_file_request):
        """Optional caption is passed to canonical publication."""
        f = tmp_path / "pic.png"
        f.write_bytes(b"fake-png")

        send_file_request.json = AsyncMock(return_value={"path": str(f), "caption": "Here you go"})
        resp = await _handle_send_file(send_file_request)

        assert resp.status == 200
        publish = send_file_request.app[CORE_HOST_KEY].services.proactive_publication.publish_file
        assert publish.await_args.kwargs["caption"] == "Here you go"

    async def test_idempotency_key_is_forwarded_to_file_publication(self, tmp_path, send_file_request):
        f = tmp_path / "report.txt"
        f.write_text("report")
        send_file_request.json = AsyncMock(return_value={"path": str(f), "idempotency_key": "caller-file-1"})

        assert (await _handle_send_file(send_file_request)).status == 200

        publish = send_file_request.app[CORE_HOST_KEY].services.proactive_publication.publish_file
        assert publish.await_args.kwargs["request_id"] == "caller-file-1"

    async def test_missing_path_returns_400(self, send_file_request):
        """Returns 400 when the required path field is absent."""
        send_file_request.json = AsyncMock(return_value={})
        resp = await _handle_send_file(send_file_request)
        assert resp.status == 400

    async def test_non_string_path_returns_400(self, send_file_request):
        send_file_request.json = AsyncMock(return_value={"path": 42})

        resp = await _handle_send_file(send_file_request)

        assert resp.status == 400
        assert "path must be a string" in resp.text

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
        """The principal's canonical opaque upload directory is allowed."""
        request, _workspace, data_dir = isolated_send_file_roots
        namespace = request.app[WORKSHOP_PRINCIPAL_STORAGE_KEY].for_runtime_profile(profile_id(123))
        uploaded = namespace.files_directory(data_dir) / "report.txt"
        uploaded.parent.mkdir(parents=True)
        uploaded.write_text("principal-owned")
        request.json = AsyncMock(return_value={"path": str(uploaded)})

        resp = await _handle_send_file(request)

        assert resp.status == 200
        request.app[CORE_HOST_KEY].services.proactive_publication.publish_file.assert_awaited_once()

    async def test_file_scope_follows_authenticated_credential(self, isolated_send_file_roots):
        """A second credential receives its own scope, independent of app defaults."""
        request, _workspace, data_dir = isolated_send_file_roots
        namespace = request.app[WORKSHOP_PRINCIPAL_STORAGE_KEY].for_runtime_profile(profile_id(456))
        uploaded = namespace.files_directory(data_dir) / "report.txt"
        uploaded.parent.mkdir(parents=True)
        uploaded.write_text("second principal")
        request.headers = {"X-Webhook-Secret": "other-secret"}
        request.json = AsyncMock(return_value={"path": str(uploaded)})

        resp = await _handle_send_file(request)

        assert resp.status == 200
        publish = request.app[CORE_HOST_KEY].services.proactive_publication.publish_file
        assert publish.await_args.args[0].principal_id == _internal_api_context(456).principal_id

    async def test_authenticated_principal_cannot_send_sibling_uploaded_file(self, isolated_send_file_roots):
        """A FILES_SEND credential cannot select another principal's upload."""
        request, _workspace, data_dir = isolated_send_file_roots
        namespace = request.app[WORKSHOP_PRINCIPAL_STORAGE_KEY].for_runtime_profile(profile_id(456))
        sibling_file = namespace.files_directory(data_dir) / "secret.txt"
        sibling_file.parent.mkdir(parents=True)
        sibling_file.write_text("other principal")
        request.json = AsyncMock(return_value={"path": str(sibling_file)})

        resp = await _handle_send_file(request)

        assert resp.status == 403
        request.app[CORE_HOST_KEY].services.proactive_publication.publish_file.assert_not_awaited()

    async def test_runtime_profile_storage_must_belong_to_authenticated_principal(
        self,
        isolated_send_file_roots,
    ):
        """A mismatched protected profile/storage mapping fails closed."""
        request, _workspace, data_dir = isolated_send_file_roots
        wrong_owner = _internal_api_context(456)
        request.app[WORKSHOP_PRINCIPAL_STORAGE_KEY] = WorkshopPrincipalStorageRegistry(
            (
                WorkshopPrincipalStorageNamespace(
                    wrong_owner.principal_id,
                    profile_id(123),
                    123,
                ),
            )
        )
        uploaded = data_dir / "files" / str(wrong_owner.principal_id) / "secret.txt"
        uploaded.parent.mkdir(parents=True)
        uploaded.write_text("wrong owner")
        request.json = AsyncMock(return_value={"path": str(uploaded)})

        resp = await _handle_send_file(request)

        assert resp.status == 403
        request.app[CORE_HOST_KEY].services.proactive_publication.publish_file.assert_not_awaited()

    async def test_authenticated_principal_cannot_send_legacy_shared_file(self, isolated_send_file_roots):
        """Ambiguous files in the legacy shared root are not exposed by the API."""
        request, _workspace, data_dir = isolated_send_file_roots
        shared_file = data_dir / "files" / "legacy.txt"
        shared_file.parent.mkdir(parents=True)
        shared_file.write_text("unattributed")
        request.json = AsyncMock(return_value={"path": str(shared_file)})

        resp = await _handle_send_file(request)

        assert resp.status == 403
        request.app[CORE_HOST_KEY].services.proactive_publication.publish_file.assert_not_awaited()

    async def test_symlink_cannot_escape_principal_upload_directory(self, isolated_send_file_roots):
        """Resolving a symlink into a sibling principal's directory is denied."""
        request, _workspace, data_dir = isolated_send_file_roots
        registry = request.app[WORKSHOP_PRINCIPAL_STORAGE_KEY]
        sibling_file = registry.for_runtime_profile(profile_id(456)).files_directory(data_dir) / "secret.txt"
        sibling_file.parent.mkdir(parents=True)
        sibling_file.write_text("other principal")
        link = registry.for_runtime_profile(profile_id(123)).files_directory(data_dir) / "link.txt"
        link.parent.mkdir(parents=True)
        link.symlink_to(sibling_file)
        request.json = AsyncMock(return_value={"path": str(link)})

        resp = await _handle_send_file(request)

        assert resp.status == 403
        request.app[CORE_HOST_KEY].services.proactive_publication.publish_file.assert_not_awaited()

    async def test_authenticated_principal_cannot_send_from_numeric_archive(
        self,
        isolated_send_file_roots,
    ):
        """Numeric upload archives are not protected runtime read roots."""
        request, _workspace, data_dir = isolated_send_file_roots
        legacy = data_dir / "files" / "123" / "legacy.txt"
        legacy.parent.mkdir(parents=True)
        legacy.write_text("pre-migration")
        request.json = AsyncMock(return_value={"path": str(legacy)})

        resp = await _handle_send_file(request)

        assert resp.status == 403
        request.app[CORE_HOST_KEY].services.proactive_publication.publish_file.assert_not_awaited()

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
        CHAT_ID_KEY: 123,
        CORE_HOST_KEY: SimpleNamespace(
            services=SimpleNamespace(
                proactive_publication=SimpleNamespace(
                    publish_text=AsyncMock(return_value=ProactivePublicationResult(MessageId.new(), True, ()))
                )
            )
        ),
    }
    request.headers = {"X-Webhook-Secret": "test-secret"}
    return request


class TestSendMessage:
    async def test_records_short_message(self, send_message_request):
        """Short messages are recorded through canonical publication."""
        send_message_request.json = AsyncMock(return_value={"text": "Hello!"})
        resp = await _handle_send_message(send_message_request)

        assert resp.status == 200
        body = json.loads(resp.body)
        assert body == {"status": "recorded", "delivery": "not_configured", "deliveries": 0}
        publish = send_message_request.app[CORE_HOST_KEY].services.proactive_publication.publish_text
        publish.assert_awaited_once()
        assert publish.await_args.kwargs["body"] == "Hello!"

    async def test_records_long_message_once(self, send_message_request):
        """Transport chunking occurs in adapters, after one canonical record."""
        long_text = ("A" * 2100) + "\n\n" + ("B" * 2100)
        send_message_request.json = AsyncMock(return_value={"text": long_text})
        resp = await _handle_send_message(send_message_request)

        assert resp.status == 200
        publish = send_message_request.app[CORE_HOST_KEY].services.proactive_publication.publish_text
        publish.assert_awaited_once()
        assert publish.await_args.kwargs["body"] == long_text

    async def test_idempotency_key_is_forwarded_to_text_publication(self, send_message_request):
        send_message_request.json = AsyncMock(return_value={"text": "Hello", "idempotency_key": "caller-text-1"})

        assert (await _handle_send_message(send_message_request)).status == 200

        publish = send_message_request.app[CORE_HOST_KEY].services.proactive_publication.publish_text
        assert publish.await_args.kwargs["request_id"] == "caller-text-1"

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

    async def test_non_string_text_returns_400(self, send_message_request):
        send_message_request.json = AsyncMock(return_value={"text": ["not", "text"]})

        resp = await _handle_send_message(send_message_request)

        assert resp.status == 400
        assert "text must be a string" in resp.text

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

    async def test_publication_error_returns_500(self, send_message_request):
        """Returns 500 when canonical publication fails."""
        send_message_request.json = AsyncMock(return_value={"text": "Hello"})
        send_message_request.app[CORE_HOST_KEY].services.proactive_publication.publish_text.side_effect = RuntimeError(
            "Boom"
        )
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
    """Create a mock request for canonical GitHub webhook routing."""
    mock_config = MagicMock()
    mock_config.user_configs = {}
    automation = MagicMock()
    automation.routes_for_repository = AsyncMock(return_value=())
    notifications = MagicMock()
    notifications.record_for_default_admin = AsyncMock(
        return_value=SimpleNamespace(
            message_id="msg_" + "1" * 32,
            inserted=True,
            deliveries=(),
        )
    )
    request = MagicMock(spec=web.Request)
    request.app = {
        INTERNAL_API_AUTH_KEY: _make_internal_api_auth(),
        GITHUB_WEBHOOK_SECRET_KEY: "test-secret",
        TELEGRAM_BOT_KEY: AsyncMock(),
        CHAT_ID_KEY: 12345,
        CONFIG_KEY: mock_config,
        WORKSHOP_GITHUB_AUTOMATION_KEY: automation,
        WORKSHOP_INTEGRATION_NOTIFICATIONS_KEY: notifications,
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
    async def test_valid_push_records_canonical_default_notification(self, github_request):
        """A subscribed-less push is recorded for the canonical default admin."""
        payload = _github_push_payload()
        body = json.dumps(payload).encode()
        github_request.read = AsyncMock(return_value=body)
        github_request.headers = {
            "X-Hub-Signature-256": _sign_body("test-secret", body),
            "X-GitHub-Event": "push",
            "X-GitHub-Delivery": "push-default-admin-1",
        }

        resp = await _handle_github(github_request)

        assert resp.status == 200
        github_request.app[TELEGRAM_BOT_KEY].send_message.assert_not_called()
        notification_service = github_request.app[WORKSHOP_INTEGRATION_NOTIFICATIONS_KEY]
        notification_service.record_for_default_admin.assert_awaited_once()
        notification = notification_service.record_for_default_admin.await_args.args[0]
        assert notification.delivery_id == "push-default-admin-1"
        assert notification.repository == "testuser/repo"

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
            "X-GitHub-Delivery": "unknown-event-1",
        }

        resp = await _handle_github(github_request)

        body_json = json.loads(resp.body.decode())
        # The canonical router acknowledges but does not persist unsupported events.
        assert body_json["msg"] == "ignored"
        github_request.app[TELEGRAM_BOT_KEY].send_message.assert_not_called()
        github_request.app[WORKSHOP_INTEGRATION_NOTIFICATIONS_KEY].record_for_default_admin.assert_not_awaited()

    async def test_filtered_action_ignored(self, github_request):
        """Known event type with filtered action (e.g. PR 'edited') is ignored."""
        # PR "edited" is not in the formatter's accepted actions
        payload = {"action": "edited", "pull_request": {"title": "test"}}
        body = json.dumps(payload).encode()
        github_request.read = AsyncMock(return_value=body)
        github_request.headers = {
            "X-Hub-Signature-256": _sign_body("test-secret", body),
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "filtered-action-1",
        }

        resp = await _handle_github(github_request)

        body_json = json.loads(resp.body.decode())
        # The formatter returns None for "edited", so no notification is stored.
        assert body_json["msg"] == "ignored"
        github_request.app[TELEGRAM_BOT_KEY].send_message.assert_not_called()
        github_request.app[WORKSHOP_INTEGRATION_NOTIFICATIONS_KEY].record_for_default_admin.assert_not_awaited()

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
    notification_service = MagicMock()
    notification_service.record_for_route = AsyncMock(
        return_value=MagicMock(message_id="msg_" + "2" * 32, inserted=True)
    )
    request.app = {
        INTERNAL_API_AUTH_KEY: _make_internal_api_auth(),
        GENERIC_WEBHOOK_SECRET_KEY: "test-secret",
        WORKSHOP_INTEGRATION_NOTIFICATIONS_KEY: notification_service,
    }
    request.headers = {"X-Webhook-Secret": "test-secret"}
    return request


class TestGenericWebhook:
    async def test_records_message_field_canonically(self, generic_request):
        generic_request.json = AsyncMock(return_value={"message": "Alert: disk full"})

        resp = await _handle_generic(generic_request)

        assert resp.status == 200
        service = generic_request.app[WORKSHOP_INTEGRATION_NOTIFICATIONS_KEY]
        service.record_for_route.assert_awaited_once()
        assert service.record_for_route.await_args.kwargs == {"route_name": "default"}
        assert service.record_for_route.await_args.args[0].body == "Alert: disk full"

    async def test_ignores_legacy_telegram_chat_destination(self, generic_request):
        generic_request.app[CHAT_ID_KEY] = 12345
        service = generic_request.app[WORKSHOP_INTEGRATION_NOTIFICATIONS_KEY]
        service.record_for_binding = AsyncMock()
        generic_request.json = AsyncMock(return_value={"message": "Canonical only"})

        resp = await _handle_generic(generic_request)

        assert resp.status == 200
        service.record_for_binding.assert_not_awaited()
        service.record_for_route.assert_awaited_once()

    async def test_dumps_full_payload_when_no_message(self, generic_request):
        payload = {"key": "value", "count": 42}
        generic_request.json = AsyncMock(return_value=payload)

        resp = await _handle_generic(generic_request)

        assert resp.status == 200
        service = generic_request.app[WORKSHOP_INTEGRATION_NOTIFICATIONS_KEY]
        sent_text = service.record_for_route.await_args.args[0].body
        # Should be a pretty-printed JSON dump
        assert '"key": "value"' in sent_text
        assert '"count": 42' in sent_text

    async def test_empty_message_field_is_rejected(self, generic_request):
        generic_request.json = AsyncMock(return_value={"message": ""})

        resp = await _handle_generic(generic_request)

        assert resp.status == 400

    async def test_long_message_is_preserved_for_transport_specific_fragmentation(self, generic_request):
        long_msg = "x" * 5000
        generic_request.json = AsyncMock(return_value={"message": long_msg})

        resp = await _handle_generic(generic_request)

        assert resp.status == 200
        service = generic_request.app[WORKSHOP_INTEGRATION_NOTIFICATIONS_KEY]
        sent_text = service.record_for_route.await_args.args[0].body
        assert sent_text == long_msg

    async def test_invalid_json_returns_400(self, generic_request):
        """Malformed JSON body returns 400."""
        generic_request.json = AsyncMock(side_effect=json.JSONDecodeError("test", "doc", 0))

        resp = await _handle_generic(generic_request)

        assert resp.status == 400

    async def test_record_failure_returns_retryable_unavailable(self, generic_request):
        generic_request.json = AsyncMock(return_value={"message": "test"})
        service = generic_request.app[WORKSHOP_INTEGRATION_NOTIFICATIONS_KEY]
        service.record_for_route = AsyncMock(side_effect=RuntimeError("database error"))

        resp = await _handle_generic(generic_request)

        assert resp.status == 503
        body_json = json.loads(resp.body.decode())
        assert body_json["status"] == "unavailable"

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
        service = generic_request.app[WORKSHOP_INTEGRATION_NOTIFICATIONS_KEY]
        service.record_for_route.assert_not_awaited()


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
        scheduler = mock_request.app[CORE_HOST_KEY].services.scheduler
        resp = await _handle_schedule(mock_request)

        assert resp.status == 200
        body = json.loads(resp.body.decode())
        scheduler.register_job.assert_awaited_once_with(body["job_id"])

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
        mock_request.app[CORE_HOST_KEY].services.scheduler.register_job.return_value = False
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
        mock_request.app[CORE_HOST_KEY].services.scheduler.register_job.side_effect = RuntimeError(
            "scheduler unavailable"
        )
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


class TestInternalPublicationRetirementGuard:
    def test_handlers_have_no_compatibility_identity_or_direct_telegram_send(self):
        """Protected publication cannot regress to direct transport effects."""
        source = "\n".join(
            (
                inspect.getsource(webhook_mod._handle_send_message),
                inspect.getsource(webhook_mod._handle_send_file),
            )
        )
        assert "compatibility_runtime_config_id" not in source
        assert "TELEGRAM_BOT_KEY" not in source
        assert ".send_message(" not in source
        assert ".send_photo(" not in source
        assert ".send_document(" not in source


class TestGetJobsCredentialRouting:
    @pytest.mark.asyncio
    async def test_query_identity_selector_is_rejected(self, db, mock_request):
        """GET /api/jobs rejects the retired query identity selector."""
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
        assert resp.status == 400
        assert "Identity selector chat_id is not accepted" in resp.text

    @pytest.mark.asyncio
    async def test_invalid_query_param_returns_400(self, db, mock_request):
        """Any query identity selector returns 400."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.app[CHAT_ID_KEY] = 123
        mock_request.query = {"chat_id": "abc"}

        resp = await _handle_get_jobs(mock_request)
        assert resp.status == 400


class TestScheduleCredentialRouting:
    @pytest.mark.asyncio
    async def test_explicit_chat_id_in_body_is_rejected(self, db, mock_request):
        """POST /api/schedule rejects caller-selected identity."""
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
        assert resp.status == 400
        assert await sessions.get_jobs(456) == []


# ── chat_id authorization ──────────────────────────────────────────


class TestChatIdAuthorization:
    @pytest.mark.asyncio
    async def test_external_signing_secret_is_not_an_api_credential(self, mock_request):
        """Possession of the GitHub/generic signing secret does not authorize API calls."""
        mock_request.headers = {"X-Webhook-Secret": "signing-secret"}
        mock_request.json = AsyncMock(return_value={"text": "hello"})

        resp = await _handle_send_message(mock_request)

        assert resp.status == 401
        mock_request.app[CORE_HOST_KEY].services.proactive_publication.publish_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_notification_credential_cannot_manage_jobs(self, mock_request):
        """A review/triage notification credential is limited to sending messages."""
        auth = mock_request.app[INTERNAL_API_AUTH_KEY]
        credential = auth.notification_credential_for(_internal_api_context(123))
        mock_request.headers = {"X-Webhook-Secret": credential}

        mock_request.json = AsyncMock(return_value={"text": "review complete"})
        send_resp = await _handle_send_message(mock_request)
        assert send_resp.status == 200

        schedule_resp = await _handle_schedule(mock_request)
        assert schedule_resp.status == 403

    @pytest.mark.asyncio
    async def test_caller_selected_chat_id_returns_400(self, db, mock_request):
        """POST /api/schedule rejects the retired identity selector."""
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
        assert resp.status == 400

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
        assert resp.status == 400
        assert await sessions.get_jobs(456) == []

    @pytest.mark.asyncio
    async def test_send_message_identity_selector_returns_400(self, db, mock_request):
        """POST /api/send-message rejects caller-selected identity."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"text": "hello", "chat_id": 999999})

        resp = await _handle_send_message(mock_request)
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_send_file_identity_selector_returns_400(self, db, mock_request):
        """POST /api/send-file rejects caller-selected identity."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"path": "/tmp/test.txt", "chat_id": 999999})

        resp = await _handle_send_file(mock_request)
        assert resp.status == 400


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
    async def test_identity_selector_returns_400(self, db, mock_request):
        """GET /api/jobs rejects caller-supplied identity."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.query = {"chat_id": "999"}

        resp = await _handle_get_jobs(mock_request)
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_credential_returns_only_its_jobs(self, db, mock_request):
        """GET /api/jobs uses only the authenticated canonical context."""
        mock_request.headers = {"X-Webhook-Secret": "other-secret"}
        mock_request.query = {}

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
        mock_request.query = {}
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
        """The credential owner can view its own job."""
        mock_request.headers = {"X-Webhook-Secret": "other-secret"}
        mock_request.query = {}
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
    async def test_identity_selector_returns_400(self, db, mock_request):
        """GET /api/jobs/{id} rejects caller-supplied identity."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.query = {"chat_id": "999"}
        mock_request.match_info = {"id": "1"}

        resp = await _handle_get_job(mock_request)
        assert resp.status == 400


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
        """A credential can delete its own job."""
        mock_request.headers = {"X-Webhook-Secret": "other-secret"}
        mock_request.query = {}
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
        """A credential cannot delete another context's job."""
        mock_request.headers = {"X-Webhook-Secret": "other-secret"}
        mock_request.query = {}
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
    async def test_identity_selector_returns_400(self, db, mock_request):
        """DELETE /api/jobs/{id} rejects caller-supplied identity."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.query = {"chat_id": "999"}
        mock_request.match_info = {"id": "1"}

        resp = await _handle_delete_job(mock_request)
        assert resp.status == 400


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
        """PATCH uses the authenticated context to update its own job."""
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
        mock_request.json = AsyncMock(return_value={"name": "Updated"})

        resp = await _handle_update_job(mock_request)
        assert resp.status == 200
        job = await sessions.get_job_by_id(job_id)
        assert job is not None
        assert job["name"] == "Updated"

    @pytest.mark.asyncio
    async def test_cannot_update_other_users_job(self, db, mock_request):
        """PATCH returns 404 for another context's job."""
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
        mock_request.json = AsyncMock(return_value={"name": "Hacked"})

        resp = await _handle_update_job(mock_request)
        assert resp.status == 404
        # Job should be unchanged
        job = await sessions.get_job_by_id(job_id)
        assert job is not None
        assert job["name"] == "Admin Job"

    @pytest.mark.asyncio
    async def test_identity_selector_returns_400(self, db, mock_request):
        """PATCH rejects caller-supplied identity."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.match_info = {"id": "1"}
        mock_request.json = AsyncMock(return_value={"chat_id": 999, "name": "Nope"})

        resp = await _handle_update_job(mock_request)
        assert resp.status == 400


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
        mock_request.app[CORE_HOST_KEY].services.runtime_pool.get_effective_workspace.assert_awaited_once_with(
            profile_id(123)
        )

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
        """memory_type and tags are forwarded verbatim; metadata is
        forwarded with the server-side provenance stamp on top."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(
            return_value={
                "content": "User prefers dark mode",
                "memory_type": "preference",
                "tags": ["ui", "preference"],
                "metadata": {"project_note": "kept"},
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
        assert kwargs["metadata"]["project_note"] == "kept"

    async def test_server_stamp_overrides_caller_source(self, mock_request):
        """Caller-supplied source cannot override the server stamp;
        speaker is a caller-overridable default; confidence defaults
        when absent; scope is routed server-side (global here, since
        the test app has no pool and therefore no detected project)."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(
            return_value={
                "content": "User said they prefer tabs",
                "metadata": {"source": "extracted", "speaker": "user"},
            }
        )

        with (
            patch("kai.memory.is_enabled", return_value=True),
            patch("kai.memory.add_structured", return_value="id-1") as mock_add,
        ):
            await _handle_memory_add(mock_request)

        stamped = mock_add.call_args.kwargs["metadata"]
        assert stamped["source"] == "explicit"
        assert stamped["speaker"] == "user"
        assert stamped["confidence"] == 0.9
        assert stamped["scope"] == "global"
        assert stamped["scope_source"] == "extraction_default"

    async def test_returns_400_on_bad_speaker_type(self, mock_request):
        """The caller-overridable speaker key gets the same 400-boundary
        treatment as every other field: a non-string speaker would
        surface later as a broken detail view, not a clean error."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"content": "x", "metadata": {"speaker": 42}})
        with patch("kai.memory.add_structured") as mock_add:
            resp = await _handle_memory_add(mock_request)
        assert resp.status == 400
        mock_add.assert_not_called()

    @pytest.mark.parametrize("bad_confidence", ["high", True, 1.5, -0.1])
    async def test_returns_400_on_bad_confidence(self, mock_request, bad_confidence):
        """Confidence feeds ranking arithmetic and formatted rendering;
        non-numeric, boolean, and out-of-range values are rejected at
        the boundary (bool passes isinstance(int) and would silently
        rank as 1/0, so it is rejected explicitly)."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"content": "x", "metadata": {"confidence": bad_confidence}})
        with patch("kai.memory.add_structured") as mock_add:
            resp = await _handle_memory_add(mock_request)
        assert resp.status == 400
        mock_add.assert_not_called()

    async def test_returns_500_when_workspace_resolution_fails(self, mock_request):
        """Scope routing hits the settings DB via the runtime facade; a transient
        failure there must produce the handler's clean JSON 500, not
        mis-scope the write to global silently or escape as a framework
        HTML 500."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"content": "x"})
        mock_request.app[CORE_HOST_KEY].services.runtime_pool.get_effective_workspace.side_effect = RuntimeError(
            "settings DB down"
        )

        with (
            patch("kai.memory.is_enabled", return_value=True),
            patch("kai.memory.add_structured") as mock_add,
        ):
            resp = await _handle_memory_add(mock_request)

        assert resp.status == 500
        assert json.loads(resp.body.decode()) == {"error": "Memory storage failed"}
        mock_add.assert_not_called()

    async def test_stamped_write_is_visible_to_the_memory_ui(self, mock_request):
        """Integration seam: what the HTTP add endpoint writes must
        pass the exact gates the /memory UI reads through.
        The stamped metadata from a real handler call is stored in a
        fake Mem0 and must come back out of get_all_facts (fact list)
        and get_by_id (detail view / delete gate)."""
        from types import SimpleNamespace

        from kai import memory

        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"content": "User prefers Earl Grey"})

        with (
            patch("kai.memory.is_enabled", return_value=True),
            patch("kai.memory.add_structured", return_value="id-1") as mock_add,
        ):
            await _handle_memory_add(mock_request)
        stamped = mock_add.call_args.kwargs["metadata"]

        row = {
            "id": "mem-1",
            "memory": "User prefers Earl Grey",
            "metadata": stamped,
            "user_id": str(_internal_api_context(123).principal_id),
            "created_at": "2026-08-19T00:00:00+00:00",
            "updated_at": "2026-08-19T00:00:00+00:00",
        }
        fake_mem0 = SimpleNamespace(
            get_all=lambda filters, top_k: {"results": [row]},
            get=lambda memory_id: row,
        )
        with patch("kai.memory._memory", fake_mem0):
            canonical_user_id = str(_internal_api_context(123).principal_id)
            facts = memory.get_all_facts(user_id=canonical_user_id)
            assert [f.id for f in facts] == ["mem-1"]
            detail = memory.get_by_id(user_id=canonical_user_id, memory_id="mem-1")
            assert detail is not None and detail.metadata["source"] == "explicit"

    async def test_uses_credential_bound_principal_for_user_id(self, mock_request):
        """Memory ownership comes directly from the credential's principal."""
        mock_request.headers = {"X-Webhook-Secret": "other-secret"}
        mock_request.json = AsyncMock(return_value={"content": "x"})

        with (
            patch("kai.memory.is_enabled", return_value=True),
            patch("kai.memory.add_structured", return_value="id-1") as mock_add,
        ):
            await _handle_memory_add(mock_request)

        kwargs = mock_add.call_args.kwargs
        assert kwargs["user_id"] == str(_internal_api_context(456).principal_id)
        assert kwargs["runtime_profile_id"] == str(profile_id(456))
        assert isinstance(kwargs["user_id"], str)

    async def test_returns_400_on_identity_selector(self, mock_request):
        """Caller-supplied identity is rejected before the primitive."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        # The credential resolves to principal 123, not the requested 999.
        mock_request.json = AsyncMock(return_value={"content": "x", "chat_id": 999})

        with patch("kai.memory.is_enabled", return_value=True), patch("kai.memory.add_structured") as mock_add:
            resp = await _handle_memory_add(mock_request)

        assert resp.status == 400
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
        assert mock_search.call_args.kwargs["user_id"] == str(_internal_api_context(123).principal_id)
        assert mock_search.call_args.kwargs["runtime_profile_id"] == str(profile_id(123))

    async def test_search_credentials_keep_principals_isolated(self, mock_request):
        """A second credential searches its own canonical namespace."""
        mock_request.headers = {"X-Webhook-Secret": "other-secret"}
        mock_request.json = AsyncMock(return_value={"query": "x"})

        with (
            patch("kai.memory.is_enabled", return_value=True),
            patch("kai.memory.search", return_value=[]) as mock_search,
        ):
            await _handle_memory_search(mock_request)

        assert mock_search.call_args.kwargs["user_id"] == str(_internal_api_context(456).principal_id)
        assert mock_search.call_args.kwargs["runtime_profile_id"] == str(profile_id(456))

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

    async def test_returns_400_on_identity_selector(self, mock_request):
        """Caller-supplied identity is rejected before the primitive."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.json = AsyncMock(return_value={"query": "x", "chat_id": 999})

        with (
            patch("kai.memory.is_enabled", return_value=True),
            patch("kai.memory.search") as mock_search,
        ):
            resp = await _handle_memory_search(mock_request)

        assert resp.status == 400
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

    async def test_uses_credential_bound_principal_for_stats(self, mock_request):
        """GET stats derives canonical ownership from the credential."""
        from kai.memory import MemoryStats

        mock_request.headers = {"X-Webhook-Secret": "other-secret"}
        mock_request.query = {}

        with (
            patch("kai.memory.is_enabled", return_value=True),
            patch("kai.memory.get_stats", return_value=MemoryStats(total_count=0, by_type={})) as mock_get_stats,
        ):
            resp = await _handle_memory_stats(mock_request)

        assert resp.status == 200
        assert mock_get_stats.call_args.kwargs["user_id"] == str(_internal_api_context(456).principal_id)
        assert mock_get_stats.call_args.kwargs["runtime_profile_id"] == str(profile_id(456))

    async def test_returns_401_on_bad_secret(self, mock_request):
        mock_request.headers = {"X-Webhook-Secret": "nope"}
        mock_request.query = {}

        resp = await _handle_memory_stats(mock_request)

        assert resp.status == 401

    async def test_returns_400_on_identity_selector(self, mock_request):
        """Query-string identity selectors are rejected."""
        mock_request.headers = {"X-Webhook-Secret": "test-secret"}
        mock_request.query = {"chat_id": "999"}

        with patch("kai.memory.is_enabled", return_value=True), patch("kai.memory.get_stats") as mock_get_stats:
            resp = await _handle_memory_stats(mock_request)

        assert resp.status == 400
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

    async def test_delete_all_uses_credential_bound_principal(self, mock_request):
        """Delete-all receives the canonical principal string."""
        mock_request.headers = {"X-Webhook-Secret": "other-secret"}
        mock_request.json = AsyncMock(return_value={"confirm": self._CONFIRM})

        with patch("kai.memory.is_enabled", return_value=True), patch("kai.memory.delete_all") as mock_del:
            await _call_memory_delete_all_as_authorized(mock_request, chat_id=456)

        assert mock_del.call_args.kwargs["user_id"] == str(_internal_api_context(456).principal_id)
        assert mock_del.call_args.kwargs["runtime_profile_id"] == str(profile_id(456))
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

    async def test_returns_400_on_identity_selector(self, mock_request):
        """Caller-supplied identity is rejected before the primitive.

        Verifies the 400 path runs even when the confirm token is
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

        assert resp.status == 400
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
