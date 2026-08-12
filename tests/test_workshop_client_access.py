"""Operator and production-registration tests for Workshop client access."""

from __future__ import annotations

from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kai import webhook, workshop_cli
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.client_access import WorkshopClientAccess, WorkshopClientAccessError
from kai.workshop.client_sessions import (
    EnrollmentGrantUnavailableError,
    WorkshopClientEnrollmentManager,
    WorkshopClientSessionManager,
)
from kai.workshop.domain import DeviceId, EnrollmentGrantId
from kai.workshop.store import WorkshopEventStore


async def _store(path: Path) -> WorkshopEventStore:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman("Alice", "admin", "telegram", "101", "101"),
            BootstrapHuman("Bob", "member", "telegram", "202", "202"),
        ),
    )
    return store


class TestWorkshopClientAccess:
    async def test_issue_resolves_server_selected_human_and_direct_channel(self, tmp_path: Path):
        store = await _store(tmp_path / "kai.db")
        try:
            issued = await WorkshopClientAccess(store).issue_enrollment(101)

            assert issued.grant.token.startswith(f"kai_ws_enroll_v1.{issued.grant.grant_id}.")
            assert str(issued.channel_id).startswith("chn_")
            async with store.connection.execute(
                "SELECT ei.external_subject FROM workshop_client_enrollment_grants g "
                "JOIN external_identities ei ON ei.principal_id = g.principal_id "
                "WHERE g.id = ? AND ei.provider = 'telegram'",
                (issued.grant.grant_id,),
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None and row[0] == "101"
        finally:
            await store.close()

    async def test_unknown_telegram_identity_cannot_receive_a_grant(self, tmp_path: Path):
        store = await _store(tmp_path / "kai.db")
        try:
            with pytest.raises(WorkshopClientAccessError, match="exactly one canonical human"):
                await WorkshopClientAccess(store).issue_enrollment(999)
            async with store.connection.execute("SELECT COUNT(*) FROM workshop_client_enrollment_grants") as cursor:
                assert (await cursor.fetchone())[0] == 0
        finally:
            await store.close()

    @pytest.mark.parametrize("telegram_user_id", [0, -1, True, 2**63])
    async def test_invalid_telegram_identity_is_bounded(self, tmp_path: Path, telegram_user_id):
        store = await _store(tmp_path / "kai.db")
        try:
            with pytest.raises(WorkshopClientAccessError, match="positive signed 64-bit"):
                await WorkshopClientAccess(store).issue_enrollment(telegram_user_id)
        finally:
            await store.close()

    async def test_device_revocation_survives_store_restart(self, tmp_path: Path):
        database = tmp_path / "kai.db"
        store = await _store(database)
        access = WorkshopClientAccess(store)
        issued = await access.issue_enrollment(101)
        redeemed = await WorkshopClientEnrollmentManager(store).redeem_grant(issued.grant.token, "Alice laptop")
        token = redeemed.session.token
        await access.revoke_device(101, redeemed.device.device_id)
        await store.close()

        restarted = await WorkshopEventStore.open(database)
        try:
            assert await WorkshopClientSessionManager(restarted).authenticate_token(token) is None
            with pytest.raises(WorkshopClientAccessError, match="unavailable"):
                await WorkshopClientAccess(restarted).revoke_device(101, redeemed.device.device_id)
        finally:
            await restarted.close()

    async def test_grant_revocation_is_scoped_and_durable(self, tmp_path: Path):
        database = tmp_path / "kai.db"
        store = await _store(database)
        access = WorkshopClientAccess(store)
        issued = await access.issue_enrollment(101)
        await access.revoke_enrollment(101, issued.grant.grant_id)
        await store.close()

        restarted = await WorkshopEventStore.open(database)
        try:
            with pytest.raises(EnrollmentGrantUnavailableError, match="Enrollment grant is unavailable"):
                await WorkshopClientEnrollmentManager(restarted).redeem_grant(issued.grant.token, "Alice laptop")
        finally:
            await restarted.close()


class TestWorkshopClientAccessCLI:
    def test_actions_require_explicit_identifiers(self):
        with pytest.raises(SystemExit):
            workshop_cli._parser().parse_args(["client-access", "issue-enrollment"])
        with pytest.raises(SystemExit):
            workshop_cli._parser().parse_args(["client-access", "revoke-device", "--telegram-user-id", "101"])
        with pytest.raises(SystemExit):
            workshop_cli._parser().parse_args(["client-access", "revoke-enrollment", "--telegram-user-id", "101"])

    def test_opaque_identifiers_are_type_checked(self):
        with pytest.raises(WorkshopClientAccessError, match="Invalid device ID"):
            workshop_cli._device_id("101")
        with pytest.raises(WorkshopClientAccessError, match="Invalid enrollment grant ID"):
            workshop_cli._enrollment_grant_id("101")

        assert workshop_cli._device_id("dev_" + "a" * 32) == DeviceId("dev_" + "a" * 32)
        assert workshop_cli._enrollment_grant_id("enr_" + "b" * 32) == EnrollmentGrantId("enr_" + "b" * 32)

    async def test_issue_command_prints_the_token_once_with_qualification_coordinates(
        self,
        tmp_path: Path,
        monkeypatch,
        capsys,
    ):
        store = await _store(tmp_path / "kai.db")
        await store.close()
        monkeypatch.setattr(workshop_cli, "DATA_DIR", tmp_path)
        args = workshop_cli._parser().parse_args(["client-access", "issue-enrollment", "--telegram-user-id", "101"])

        assert await workshop_cli._run(args) == 0
        output = capsys.readouterr().out
        assert "Enrollment: enr_" in output
        assert "Channel: chn_" in output
        assert "Expires:" in output
        assert output.count("kai_ws_enroll_v1.") == 1
        assert "stores only its hash" in output


class TestWorkshopClientProductionRegistration:
    async def test_runtime_wires_operator_grant_to_authenticated_timeline_read(self, tmp_path: Path):
        database = tmp_path / "kai.db"
        store = await _store(database)
        await store.close()
        app = web.Application()
        client: TestClient | None = None
        try:
            await webhook._register_workshop_client_api(app, database)
            routes = {(route.method, route.resource.canonical) for route in app.router.routes()}

            assert ("POST", "/v1/client/enrollment/redeem") in routes
            assert ("GET", "/v1/channels/{channel_id}/timeline") in routes
            assert ("POST", "/v1/channels/{channel_id}/timeline") not in routes
            assert ("POST", "/v1/client/enrollment") not in routes

            operator_store = await WorkshopEventStore.open(database)
            try:
                issued = await WorkshopClientAccess(operator_store).issue_enrollment(101)
            finally:
                await operator_store.close()

            client = TestClient(TestServer(app))
            await client.start_server()
            redemption = await client.post(
                "/v1/client/enrollment/redeem",
                json={"enrollment_token": issued.grant.token, "device_display_name": "Alice laptop"},
            )
            redeemed = await redemption.json()
            assert redemption.status == 201

            timeline = await client.get(
                f"/v1/channels/{issued.channel_id}/timeline",
                headers={"Authorization": f"Bearer {redeemed['session']['token']}"},
            )
            assert timeline.status == 200
            assert (await timeline.json())["channel_id"] == issued.channel_id
        finally:
            if client is not None:
                await client.close()
            await webhook.stop()
