"""HTTP contract for Workshop client enrollment redemption."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.client_api import WorkshopEnrollmentRateLimiter, register_workshop_enrollment_routes
from kai.workshop.client_sessions import WorkshopClientEnrollmentManager, WorkshopClientSessionManager
from kai.workshop.domain import PrincipalId
from kai.workshop.store import WorkshopEventStore

_NOW = datetime(2026, 8, 11, 20, 0, tzinfo=UTC)
_PATH = "/v1/client/enrollment/redeem"


@dataclass
class _Clock:
    value: datetime = _NOW

    def __call__(self) -> datetime:
        return self.value


async def _open_store(path: Path) -> tuple[WorkshopEventStore, PrincipalId]:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (BootstrapHuman("Alice", "admin", "telegram", "101", "101"),),
    )
    async with store.connection.execute(
        "SELECT principal_id FROM external_identities WHERE provider = 'telegram' AND external_subject = '101'"
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return store, PrincipalId(str(row[0]))


def _manager(store: WorkshopEventStore, clock: _Clock) -> WorkshopClientEnrollmentManager:
    grant_secrets = iter(("g" * 43, "h" * 43, "i" * 43))
    session_secrets = iter(("s" * 43, "t" * 43, "u" * 43))
    return WorkshopClientEnrollmentManager(
        store,
        clock=clock,
        grant_secret_factory=lambda: next(grant_secrets),
        session_secret_factory=lambda: next(session_secrets),
    )


async def _open_client(
    manager: WorkshopClientEnrollmentManager,
    *,
    rate_limiter: WorkshopEnrollmentRateLimiter | None = None,
) -> TestClient:
    app = web.Application()
    register_workshop_enrollment_routes(
        app,
        enrollment_manager=manager,
        rate_limiter=rate_limiter or WorkshopEnrollmentRateLimiter(),
        request_lock=asyncio.Lock(),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


class TestWorkshopEnrollmentHTTPContract:
    async def test_valid_grant_creates_device_and_returns_one_session_secret(self, tmp_path: Path):
        store, alice_id = await _open_store(tmp_path / "kai.db")
        clock = _Clock()
        manager = _manager(store, clock)
        grant = await manager.issue_grant(alice_id)
        client = await _open_client(manager)
        try:
            response = await client.post(
                _PATH,
                json={
                    "enrollment_token": grant.token,
                    "device_display_name": "  Alice's Mac mini  ",
                },
            )
            payload = await response.json()

            assert response.status == 201
            assert response.headers["Cache-Control"] == "private, no-store"
            assert response.headers["Content-Security-Policy"] == "default-src 'none'"
            assert response.headers["Referrer-Policy"] == "no-referrer"
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert payload == {
                "version": 1,
                "device": {
                    "device_id": payload["device"]["device_id"],
                    "display_name": "Alice's Mac mini",
                },
                "session": {
                    "session_id": payload["session"]["session_id"],
                    "token": payload["session"]["token"],
                    "expires_at": "2026-09-10T20:00:00Z",
                },
            }
            assert payload["device"]["device_id"].startswith("dev_")
            assert payload["session"]["session_id"].startswith("cse_")
            assert payload["session"]["token"].startswith(f"kai_ws_v1.{payload['session']['session_id']}.")
            assert "principal" not in str(payload)

            authenticated = await WorkshopClientSessionManager(store, clock=clock).authenticate_token(
                payload["session"]["token"]
            )
            assert authenticated is not None
            assert authenticated.principal_id == alice_id
        finally:
            await client.close()
            await store.close()

    @pytest.mark.parametrize(
        ("body", "content_type"),
        [
            (b"{", "application/json"),
            (b"[]", "application/json"),
            (b"{}", "application/json"),
            (b'{"enrollment_token":"token","device_display_name":7}', "application/json"),
            (b"token", "text/plain"),
        ],
    )
    async def test_malformed_or_wrongly_typed_requests_are_bounded(
        self,
        tmp_path: Path,
        body: bytes,
        content_type: str,
    ):
        store, _ = await _open_store(tmp_path / "kai.db")
        client = await _open_client(_manager(store, _Clock()))
        try:
            response = await client.post(_PATH, data=body, headers={"Content-Type": content_type})

            expected_status = 415 if content_type != "application/json" else 400
            assert response.status == expected_status
            assert await response.json() == {
                "error": {
                    "code": "unsupported_media_type" if expected_status == 415 else "invalid_request",
                    "message": "Content-Type must be application/json"
                    if expected_status == 415
                    else "Invalid enrollment request",
                }
            }
        finally:
            await client.close()
            await store.close()

    async def test_client_cannot_claim_a_principal_or_control_session_lifetime(self, tmp_path: Path):
        store, alice_id = await _open_store(tmp_path / "kai.db")
        manager = _manager(store, _Clock())
        grant = await manager.issue_grant(alice_id)
        client = await _open_client(manager)
        try:
            for forbidden in (
                {"principal_id": str(alice_id)},
                {"session_lifetime_seconds": 86400},
            ):
                response = await client.post(
                    _PATH,
                    json={
                        "enrollment_token": grant.token,
                        "device_display_name": "Alice laptop",
                        **forbidden,
                    },
                )
                assert response.status == 400

            # Schema rejection happens before redemption and does not consume
            # the one-time credential.
            valid = await client.post(
                _PATH,
                json={
                    "enrollment_token": grant.token,
                    "device_display_name": "Alice laptop",
                },
            )
            assert valid.status == 201
        finally:
            await client.close()
            await store.close()

    async def test_invalid_expired_and_reused_grants_are_indistinguishable(self, tmp_path: Path):
        store, alice_id = await _open_store(tmp_path / "kai.db")
        clock = _Clock()
        manager = _manager(store, clock)
        expired = await manager.issue_grant(alice_id, lifetime=timedelta(minutes=1))
        reusable = await manager.issue_grant(alice_id)
        client = await _open_client(manager)
        try:
            first = await client.post(
                _PATH,
                json={"enrollment_token": reusable.token, "device_display_name": "Alice laptop"},
            )
            assert first.status == 201
            clock.value = expired.expires_at

            responses = [
                await client.post(
                    _PATH,
                    json={"enrollment_token": "not-a-grant", "device_display_name": "Client"},
                ),
                await client.post(
                    _PATH,
                    json={"enrollment_token": expired.token, "device_display_name": "Client"},
                ),
                await client.post(
                    _PATH,
                    json={"enrollment_token": reusable.token, "device_display_name": "Client"},
                ),
            ]

            assert [response.status for response in responses] == [401, 401, 401]
            payloads = [await response.json() for response in responses]
            assert payloads == [{"error": {"code": "enrollment_unavailable", "message": "Enrollment unavailable"}}] * 3
        finally:
            await client.close()
            await store.close()

    async def test_contract_registers_only_post_and_is_not_a_grant_issuance_surface(self, tmp_path: Path):
        store, _ = await _open_store(tmp_path / "kai.db")
        client = await _open_client(_manager(store, _Clock()))
        try:
            assert (await client.get(_PATH)).status == 405
            assert (await client.put(_PATH, json={})).status == 405
            assert (await client.post("/v1/client/enrollment", json={})).status == 404
        finally:
            await client.close()
            await store.close()

    async def test_redemption_attempts_are_rate_limited_before_grant_lookup(self, tmp_path: Path):
        store, _ = await _open_store(tmp_path / "kai.db")
        now = [100.0]
        limiter = WorkshopEnrollmentRateLimiter(
            per_source_limit=2,
            global_limit=4,
            window_seconds=60,
            clock=lambda: now[0],
        )
        client = await _open_client(_manager(store, _Clock()), rate_limiter=limiter)
        try:
            request = {"enrollment_token": "not-a-grant", "device_display_name": "Client"}
            first = await client.post(_PATH, json=request, headers={"CF-Connecting-IP": "203.0.113.7"})
            second = await client.post(_PATH, json=request, headers={"CF-Connecting-IP": "203.0.113.7"})
            blocked = await client.post(_PATH, json=request, headers={"CF-Connecting-IP": "203.0.113.7"})

            assert first.status == second.status == 401
            assert blocked.status == 429
            assert blocked.headers["Retry-After"] == "60"
            assert await blocked.json() == {
                "error": {"code": "rate_limited", "message": "Too many enrollment attempts"}
            }

            now[0] += 60
            allowed_again = await client.post(
                _PATH,
                json=request,
                headers={"CF-Connecting-IP": "203.0.113.7"},
            )
            assert allowed_again.status == 401
        finally:
            await client.close()
            await store.close()
