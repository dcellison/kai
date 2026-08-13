"""Security contracts for production-unused Workshop client enrollment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite
import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.client_sessions import (
    ClientDeviceUnavailableError,
    EnrollmentGrantUnavailableError,
    WorkshopClientEnrollmentManager,
    WorkshopClientSessionManager,
)
from kai.workshop.domain import EnrollmentGrantId, PrincipalId
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore

_NOW = datetime(2026, 8, 11, 17, 0, tzinfo=UTC)


@dataclass
class _Clock:
    value: datetime = _NOW

    def __call__(self) -> datetime:
        return self.value


async def _open_store(path: Path) -> tuple[WorkshopEventStore, PrincipalId, PrincipalId, PrincipalId]:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman("Alice", "admin", "telegram", "101", "101"),
            BootstrapHuman("Bob", "member", "telegram", "202", "202"),
        ),
    )
    human_ids: list[PrincipalId] = []
    for subject in ("101", "202"):
        async with store.connection.execute(
            "SELECT principal_id FROM external_identities WHERE provider = 'telegram' AND external_subject = ?",
            (subject,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        human_ids.append(PrincipalId(str(row[0])))
    async with store.connection.execute("SELECT principal_id FROM agents LIMIT 1") as cursor:
        agent_row = await cursor.fetchone()
    assert agent_row is not None
    return store, human_ids[0], human_ids[1], PrincipalId(str(agent_row[0]))


def _manager(
    store: WorkshopEventStore,
    clock: _Clock,
    *,
    grant_secrets: list[str] | None = None,
    session_secrets: list[str] | None = None,
) -> WorkshopClientEnrollmentManager:
    grants = iter(grant_secrets or ["g" * 43, "h" * 43, "i" * 43])
    sessions = iter(session_secrets or ["s" * 43, "t" * 43, "u" * 43])
    return WorkshopClientEnrollmentManager(
        store,
        clock=clock,
        grant_secret_factory=lambda: next(grants),
        session_secret_factory=lambda: next(sessions),
    )


class TestWorkshopEnrollmentGrantIssuance:
    async def test_trusted_issuance_binds_a_human_and_stores_only_a_hash(self, tmp_path: Path):
        store, alice_id, _, _ = await _open_store(tmp_path / "kai.db")
        manager = _manager(store, _Clock())
        try:
            issued = await manager.issue_grant(alice_id)

            assert isinstance(issued.grant_id, EnrollmentGrantId)
            assert issued.principal_id == alice_id
            assert issued.expires_at == _NOW + timedelta(minutes=10)
            assert issued.token.startswith(f"kai_ws_enroll_v1.{issued.grant_id}.")
            assert issued.token not in repr(issued)
            async with store.connection.execute(
                "SELECT principal_id, token_hash, created_at, expires_at FROM workshop_client_enrollment_grants "
                "WHERE id = ?",
                (issued.grant_id,),
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            assert row[0] == alice_id
            assert row[1] == hashlib.sha256(issued.token.encode()).hexdigest()
            assert issued.token not in " ".join(str(value) for value in row)
        finally:
            await store.close()

    async def test_grants_can_only_target_existing_human_principals(self, tmp_path: Path):
        store, _, _, agent_id = await _open_store(tmp_path / "kai.db")
        manager = _manager(store, _Clock())
        try:
            with pytest.raises(ClientDeviceUnavailableError, match="Human principal"):
                await manager.issue_grant(agent_id)
            with pytest.raises(ClientDeviceUnavailableError, match="Human principal"):
                await manager.issue_grant(PrincipalId.new())
        finally:
            await store.close()

    @pytest.mark.parametrize(
        "lifetime",
        (timedelta(0), timedelta(seconds=-1), timedelta(hours=1, microseconds=1)),
    )
    async def test_grant_lifetime_is_positive_and_short_bounded(self, tmp_path: Path, lifetime: timedelta):
        store, alice_id, _, _ = await _open_store(tmp_path / "kai.db")
        manager = _manager(store, _Clock())
        try:
            with pytest.raises(ValueError, match="lifetime"):
                await manager.issue_grant(alice_id, lifetime=lifetime)
        finally:
            await store.close()


class TestWorkshopEnrollmentRedemption:
    async def test_redeems_to_one_device_and_authenticated_session_without_identity_input(self, tmp_path: Path):
        store, alice_id, _, _ = await _open_store(tmp_path / "kai.db")
        clock = _Clock()
        manager = _manager(store, clock)
        try:
            grant = await manager.issue_grant(alice_id)
            redeemed = await manager.redeem_grant(grant.token, "  Alice's Mac mini  ")

            assert redeemed.device.principal_id == alice_id
            assert redeemed.device.display_name == "Alice's Mac mini"
            assert redeemed.session.principal_id == alice_id
            assert redeemed.session.device_id == redeemed.device.device_id
            assert redeemed.session.token not in repr(redeemed)
            authenticated = await WorkshopClientSessionManager(store, clock=clock).authenticate_token(
                redeemed.session.token
            )
            assert authenticated is not None
            assert authenticated.principal_id == alice_id
            assert authenticated.device_id == redeemed.device.device_id

            async with store.connection.execute(
                "SELECT redeemed_at, device_id, session_id FROM workshop_client_enrollment_grants WHERE id = ?",
                (grant.grant_id,),
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            assert row[0] == "2026-08-11T17:00:00.000000Z"
            assert row[1] == redeemed.device.device_id
            assert row[2] == redeemed.session.session_id
        finally:
            await store.close()

    async def test_wrong_secret_does_not_consume_the_grant_but_reuse_is_rejected(self, tmp_path: Path):
        store, alice_id, _, _ = await _open_store(tmp_path / "kai.db")
        manager = _manager(store, _Clock())
        try:
            grant = await manager.issue_grant(alice_id)
            wrong = grant.token.rsplit(".", 1)[0] + "." + "z" * 43

            with pytest.raises(EnrollmentGrantUnavailableError):
                await manager.redeem_grant(wrong, "Attacker device")
            redeemed = await manager.redeem_grant(grant.token, "Alice laptop")
            with pytest.raises(EnrollmentGrantUnavailableError):
                await manager.redeem_grant(grant.token, "Second device")

            async with store.connection.execute(
                "SELECT COUNT(*) FROM workshop_client_devices WHERE principal_id = ?",
                (alice_id,),
            ) as cursor:
                assert (await cursor.fetchone())[0] == 1
            assert redeemed.device.principal_id == alice_id
        finally:
            await store.close()

    @pytest.mark.parametrize("token", (None, "", "not-an-enrollment-token", "x" * 513))
    async def test_malformed_tokens_receive_the_same_unavailable_result(self, tmp_path: Path, token: object):
        store, _, _, _ = await _open_store(tmp_path / "kai.db")
        manager = _manager(store, _Clock())
        try:
            with pytest.raises(EnrollmentGrantUnavailableError, match="unavailable"):
                await manager.redeem_grant(token, "Client")
        finally:
            await store.close()

    async def test_expiry_boundary_rejects_without_creating_a_device(self, tmp_path: Path):
        store, alice_id, _, _ = await _open_store(tmp_path / "kai.db")
        clock = _Clock()
        manager = _manager(store, clock)
        try:
            grant = await manager.issue_grant(alice_id, lifetime=timedelta(minutes=1))
            clock.value = grant.expires_at

            with pytest.raises(EnrollmentGrantUnavailableError):
                await manager.redeem_grant(grant.token, "Late client")
            async with store.connection.execute(
                "SELECT COUNT(*) FROM workshop_client_devices WHERE principal_id = ?",
                (alice_id,),
            ) as cursor:
                assert (await cursor.fetchone())[0] == 0
        finally:
            await store.close()

    async def test_device_session_and_redemption_are_one_transaction(self, tmp_path: Path):
        store, alice_id, _, _ = await _open_store(tmp_path / "kai.db")
        manager = _manager(store, _Clock(), session_secrets=["s" * 43, "t" * 43])
        grant = await manager.issue_grant(alice_id)
        await store.connection.execute(
            "CREATE TRIGGER reject_enrollment_session BEFORE INSERT ON workshop_client_sessions "
            "BEGIN SELECT RAISE(ABORT, 'injected session failure'); END"
        )
        await store.connection.commit()
        try:
            with pytest.raises(aiosqlite.IntegrityError, match="injected session failure"):
                await manager.redeem_grant(grant.token, "Alice laptop")
            async with store.connection.execute(
                "SELECT redeemed_at FROM workshop_client_enrollment_grants WHERE id = ?",
                (grant.grant_id,),
            ) as cursor:
                assert (await cursor.fetchone())[0] is None
            async with store.connection.execute(
                "SELECT COUNT(*) FROM workshop_client_devices WHERE principal_id = ?",
                (alice_id,),
            ) as cursor:
                assert (await cursor.fetchone())[0] == 0

            await store.connection.execute("DROP TRIGGER reject_enrollment_session")
            await store.connection.commit()
            redeemed = await manager.redeem_grant(grant.token, "Alice laptop")
            assert redeemed.device.principal_id == alice_id
        finally:
            await store.close()


class TestWorkshopEnrollmentRevocation:
    async def test_revoke_is_principal_scoped_idempotent_and_durable(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        store, alice_id, bob_id, _ = await _open_store(db_path)
        clock = _Clock()
        manager = _manager(store, clock)
        grant = await manager.issue_grant(alice_id)

        assert await manager.revoke_grant(bob_id, grant.grant_id) is False
        assert await manager.revoke_grant(alice_id, grant.grant_id) is True
        assert await manager.revoke_grant(alice_id, grant.grant_id) is False
        await store.close()

        restarted = await WorkshopEventStore.open(db_path)
        try:
            with pytest.raises(EnrollmentGrantUnavailableError):
                await _manager(restarted, clock).redeem_grant(grant.token, "Alice laptop")
        finally:
            await restarted.close()


class TestWorkshopClientSecurityStateIsolation:
    async def test_version_sixteen_upgrade_and_projection_rebuild_preserve_security_state(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        from kai.workshop import schema

        db_path = tmp_path / "kai.db"
        with monkeypatch.context() as migration_context:
            migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 16)
            migration_context.setattr(schema, "_MIGRATIONS", schema._MIGRATIONS[:16])
            store, alice_id, _, _ = await _open_store(db_path)
            clock = _Clock()
            enrollment = _manager(store, clock)
            sessions = WorkshopClientSessionManager(
                store,
                clock=clock,
                token_secret_factory=iter(("x" * 43, "y" * 43, "z" * 43)).__next__,
            )

            active_device = await sessions.register_device(alice_id, "Active device")
            active_session = await sessions.issue_session(alice_id, active_device.device_id)
            revoked_session = await sessions.issue_session(alice_id, active_device.device_id)
            assert await sessions.revoke_session(alice_id, revoked_session.session_id) is True

            revoked_device = await sessions.register_device(alice_id, "Revoked device")
            revoked_device_session = await sessions.issue_session(alice_id, revoked_device.device_id)
            assert await sessions.revoke_device(alice_id, revoked_device.device_id) is True

            active_grant = await enrollment.issue_grant(alice_id)
            revoked_grant = await enrollment.issue_grant(alice_id)
            assert await enrollment.revoke_grant(alice_id, revoked_grant.grant_id) is True
            redeemed_grant = await enrollment.issue_grant(alice_id)
            redeemed = await enrollment.redeem_grant(redeemed_grant.token, "Redeemed device")
            await store.close()

        upgraded = await WorkshopEventStore.open(db_path)
        try:

            async def security_rows(table: str) -> list[tuple[object, ...]]:
                async with upgraded.connection.execute(f"SELECT * FROM {table} ORDER BY id") as cursor:
                    return [tuple(row) for row in await cursor.fetchall()]

            before = {
                table: await security_rows(table)
                for table in (
                    "workshop_client_devices",
                    "workshop_client_sessions",
                    "workshop_client_enrollment_grants",
                )
            }
            await upgraded.rebuild_projection(CanonicalConversationProjection())
            after = {table: await security_rows(table) for table in before}

            assert await upgraded.schema_version() == 18
            assert after == before
            async with upgraded.connection.execute("PRAGMA foreign_key_check") as cursor:
                assert await cursor.fetchall() == []
            for table in ("workshop_client_devices", "workshop_client_enrollment_grants"):
                async with upgraded.connection.execute(f"PRAGMA foreign_key_list({table})") as cursor:
                    assert "principals" not in {str(row[2]) for row in await cursor.fetchall()}

            session_manager = WorkshopClientSessionManager(upgraded, clock=clock)
            assert await session_manager.authenticate_token(active_session.token) is not None
            assert await session_manager.authenticate_token(revoked_session.token) is None
            assert await session_manager.authenticate_token(revoked_device_session.token) is None
            assert await session_manager.authenticate_token(redeemed.session.token) is not None

            enrollment_manager = WorkshopClientEnrollmentManager(
                upgraded,
                clock=clock,
                session_secret_factory=lambda: "w" * 43,
            )
            with pytest.raises(EnrollmentGrantUnavailableError):
                await enrollment_manager.redeem_grant(revoked_grant.token, "Rejected device")
            new_client = await enrollment_manager.redeem_grant(active_grant.token, "New device")
            assert new_client.device.principal_id == alice_id
        finally:
            await upgraded.close()
