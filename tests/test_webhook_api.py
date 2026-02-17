"""Integration tests for webhook HTTP API endpoints (jobs CRUD)."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web

from kai import sessions
from kai.webhook import _handle_delete_job, _handle_update_job


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
