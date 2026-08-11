"""Security contracts for production-unused Workshop human-client sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from aiohttp.test_utils import make_mocked_request
from multidict import CIMultiDict

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.client_sessions import (
    ClientDeviceUnavailableError,
    WorkshopBearerSessionAuthenticator,
    WorkshopClientSessionManager,
)
from kai.workshop.domain import ClientSessionId, DeviceId, PrincipalId
from kai.workshop.store import WorkshopEventStore

_NOW = datetime(2026, 8, 11, 15, 0, tzinfo=UTC)


@dataclass
class _Clock:
    value: datetime = _NOW

    def __call__(self) -> datetime:
        return self.value


async def _open_store(path: Path) -> tuple[WorkshopEventStore, PrincipalId, PrincipalId]:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman("Alice", "admin", "telegram", "101", "101"),
            BootstrapHuman("Bob", "member", "telegram", "202", "202"),
        ),
    )
    principal_ids: list[PrincipalId] = []
    for subject in ("101", "202"):
        async with store.connection.execute(
            "SELECT principal_id FROM external_identities WHERE provider = 'telegram' AND external_subject = ?",
            (subject,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        principal_ids.append(PrincipalId(str(row[0])))
    return store, principal_ids[0], principal_ids[1]


def _manager(
    store: WorkshopEventStore,
    clock: _Clock,
    secrets: list[str] | None = None,
) -> WorkshopClientSessionManager:
    token_secrets = iter(secrets or ["a" * 43, "b" * 43, "c" * 43])
    return WorkshopClientSessionManager(
        store,
        clock=clock,
        token_secret_factory=lambda: next(token_secrets),
    )


class TestWorkshopClientSessionIssuance:
    async def test_tracks_device_and_stores_only_token_hash(self, tmp_path: Path):
        store, alice_id, _ = await _open_store(tmp_path / "kai.db")
        manager = _manager(store, _Clock())
        try:
            device = await manager.register_device(alice_id, "Alice's MacBook")
            issued = await manager.issue_session(alice_id, device.device_id)

            assert isinstance(device.device_id, DeviceId)
            assert device.principal_id == alice_id
            assert device.display_name == "Alice's MacBook"
            assert isinstance(issued.session_id, ClientSessionId)
            assert issued.device_id == device.device_id
            assert issued.principal_id == alice_id
            assert issued.expires_at == _NOW + timedelta(days=30)
            assert issued.token.startswith(f"kai_ws_v1.{issued.session_id}.")
            assert issued.token not in repr(issued)

            async with store.connection.execute(
                "SELECT token_hash, created_at, expires_at FROM workshop_client_sessions WHERE id = ?",
                (issued.session_id,),
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            assert len(str(row[0])) == 64
            assert issued.token not in " ".join(str(value) for value in row)
        finally:
            await store.close()

    async def test_only_human_principals_can_register_devices(self, tmp_path: Path):
        store, _, _ = await _open_store(tmp_path / "kai.db")
        manager = _manager(store, _Clock())
        try:
            async with store.connection.execute("SELECT principal_id FROM agents LIMIT 1") as cursor:
                row = await cursor.fetchone()
            assert row is not None

            with pytest.raises(ClientDeviceUnavailableError):
                await manager.register_device(PrincipalId(str(row[0])), "Agent device")
            with pytest.raises(ValueError, match="display_name"):
                await manager.register_device(PrincipalId.new(), "")
        finally:
            await store.close()

    async def test_session_issuance_requires_active_owned_device(self, tmp_path: Path):
        store, alice_id, bob_id = await _open_store(tmp_path / "kai.db")
        manager = _manager(store, _Clock())
        try:
            device = await manager.register_device(alice_id, "Alice laptop")

            with pytest.raises(ClientDeviceUnavailableError):
                await manager.issue_session(bob_id, device.device_id)
            with pytest.raises(ClientDeviceUnavailableError):
                await manager.issue_session(alice_id, DeviceId.new())

            assert await manager.revoke_device(alice_id, device.device_id) is True
            with pytest.raises(ClientDeviceUnavailableError):
                await manager.issue_session(alice_id, device.device_id)
        finally:
            await store.close()

    @pytest.mark.parametrize(
        "lifetime",
        (timedelta(0), timedelta(seconds=-1), timedelta(days=90, microseconds=1)),
    )
    async def test_session_lifetime_is_bounded(self, tmp_path: Path, lifetime: timedelta):
        store, alice_id, _ = await _open_store(tmp_path / "kai.db")
        manager = _manager(store, _Clock())
        try:
            device = await manager.register_device(alice_id, "Alice laptop")

            with pytest.raises(ValueError, match="lifetime"):
                await manager.issue_session(alice_id, device.device_id, lifetime=lifetime)
        finally:
            await store.close()


class TestWorkshopClientSessionAuthentication:
    async def test_valid_token_resolves_canonical_session_and_updates_device_activity(self, tmp_path: Path):
        store, alice_id, _ = await _open_store(tmp_path / "kai.db")
        clock = _Clock()
        manager = _manager(store, clock)
        try:
            device = await manager.register_device(alice_id, "Alice laptop")
            issued = await manager.issue_session(alice_id, device.device_id)
            clock.value += timedelta(minutes=5)

            authenticated = await manager.authenticate_token(issued.token)

            assert authenticated is not None
            assert authenticated.principal_id == alice_id
            assert authenticated.session_id == issued.session_id
            assert authenticated.device_id == device.device_id
            async with store.connection.execute(
                "SELECT s.last_seen_at, d.last_seen_at FROM workshop_client_sessions s "
                "JOIN workshop_client_devices d ON d.id = s.device_id WHERE s.id = ?",
                (issued.session_id,),
            ) as cursor:
                row = await cursor.fetchone()
            assert tuple(row) == ("2026-08-11T15:05:00.000000Z", "2026-08-11T15:05:00.000000Z")
        finally:
            await store.close()

    async def test_malformed_token_is_rejected_before_storage_access(self, tmp_path: Path):
        store, _, _ = await _open_store(tmp_path / "kai.db")
        manager = _manager(store, _Clock())
        await store.close()

        assert await manager.authenticate_token("") is None
        assert await manager.authenticate_token("not-a-workshop-session") is None
        assert await manager.authenticate_token("x" * 513) is None

    async def test_wrong_secret_and_expired_session_are_rejected(self, tmp_path: Path):
        store, alice_id, _ = await _open_store(tmp_path / "kai.db")
        clock = _Clock()
        manager = _manager(store, clock)
        try:
            device = await manager.register_device(alice_id, "Alice laptop")
            issued = await manager.issue_session(alice_id, device.device_id, lifetime=timedelta(minutes=5))
            wrong_token = issued.token.rsplit(".", 1)[0] + "." + "z" * 43

            assert await manager.authenticate_token(wrong_token) is None
            async with store.connection.execute(
                "SELECT last_seen_at FROM workshop_client_sessions WHERE id = ?",
                (issued.session_id,),
            ) as cursor:
                assert (await cursor.fetchone())[0] is None
            clock.value = issued.expires_at
            assert await manager.authenticate_token(issued.token) is None
        finally:
            await store.close()

    async def test_bearer_adapter_rejects_ambiguous_headers_and_survives_restart(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        store, alice_id, _ = await _open_store(db_path)
        clock = _Clock()
        manager = _manager(store, clock)
        device = await manager.register_device(alice_id, "Alice laptop")
        issued = await manager.issue_session(alice_id, device.device_id)
        await store.close()

        restarted = await WorkshopEventStore.open(db_path)
        authenticator = WorkshopBearerSessionAuthenticator(_manager(restarted, clock))
        try:
            valid = make_mocked_request("GET", "/", headers={"Authorization": f"Bearer {issued.token}"})
            duplicate = make_mocked_request(
                "GET",
                "/",
                headers=CIMultiDict(
                    (
                        ("Authorization", f"Bearer {issued.token}"),
                        ("Authorization", f"Bearer {issued.token}"),
                    )
                ),
            )
            wrong_scheme = make_mocked_request("GET", "/", headers={"Authorization": f"Basic {issued.token}"})

            assert await authenticator.authenticate(valid) == alice_id
            assert await authenticator.authenticate(duplicate) is None
            assert await authenticator.authenticate(wrong_scheme) is None
        finally:
            await restarted.close()


class TestWorkshopClientSessionRevocation:
    async def test_session_revocation_is_scoped_idempotent_and_durable(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        store, alice_id, bob_id = await _open_store(db_path)
        clock = _Clock()
        manager = _manager(store, clock)
        device = await manager.register_device(alice_id, "Alice laptop")
        first = await manager.issue_session(alice_id, device.device_id)
        second = await manager.issue_session(alice_id, device.device_id)

        assert await manager.revoke_session(bob_id, first.session_id) is False
        assert await manager.revoke_session(alice_id, first.session_id) is True
        assert await manager.revoke_session(alice_id, first.session_id) is False
        assert await manager.authenticate_token(first.token) is None
        assert await manager.authenticate_token(second.token) is not None
        await store.close()

        restarted = await WorkshopEventStore.open(db_path)
        restarted_manager = _manager(restarted, clock)
        try:
            assert await restarted_manager.authenticate_token(first.token) is None
            assert await restarted_manager.authenticate_token(second.token) is not None
        finally:
            await restarted.close()

    async def test_device_revocation_invalidates_every_session(self, tmp_path: Path):
        store, alice_id, bob_id = await _open_store(tmp_path / "kai.db")
        manager = _manager(store, _Clock())
        try:
            device = await manager.register_device(alice_id, "Alice laptop")
            first = await manager.issue_session(alice_id, device.device_id)
            second = await manager.issue_session(alice_id, device.device_id)

            assert await manager.revoke_device(bob_id, device.device_id) is False
            assert await manager.revoke_device(alice_id, device.device_id) is True
            assert await manager.revoke_device(alice_id, device.device_id) is False
            assert await manager.authenticate_token(first.token) is None
            assert await manager.authenticate_token(second.token) is None
        finally:
            await store.close()
