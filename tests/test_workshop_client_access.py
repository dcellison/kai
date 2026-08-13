"""Operator and production-registration tests for Workshop client access."""

from __future__ import annotations

from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kai import webhook, workshop_cli
from kai.config import Config, UserConfig
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.client_access import WorkshopClientAccess, WorkshopClientAccessError
from kai.workshop.client_sessions import (
    EnrollmentGrantUnavailableError,
    WorkshopClientEnrollmentManager,
    WorkshopClientSessionManager,
)
from kai.workshop.domain import ChannelId, DeviceId, EnrollmentGrantId, PrincipalId
from kai.workshop.human_provisioning import WorkshopHumanProvisioningError
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id


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


async def _canonical_access_ids(
    store: WorkshopEventStore,
    external_subject: str = "101",
) -> tuple[PrincipalId, ChannelId]:
    async with store.connection.execute(
        "SELECT e.principal_id, c.id FROM external_identities e "
        "JOIN channel_memberships cm ON cm.principal_id = e.principal_id AND cm.role = 'owner' "
        "JOIN channels c ON c.id = cm.channel_id AND c.kind = 'direct' "
        "WHERE e.provider = 'telegram' AND e.external_subject = ?",
        (external_subject,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return PrincipalId(str(row[0])), ChannelId(str(row[1]))


class TestWorkshopClientAccess:
    async def test_issue_resolves_server_selected_human_and_direct_channel(self, tmp_path: Path):
        store = await _store(tmp_path / "kai.db")
        try:
            principal_id, channel_id = await _canonical_access_ids(store)
            issued = await WorkshopClientAccess(store).issue_enrollment(principal_id, channel_id)

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

    async def test_canonical_enrollment_needs_no_external_identity_or_transport_binding(
        self,
        tmp_path: Path,
    ):
        store = await _store(tmp_path / "kai.db")
        try:
            principal_id, channel_id = await _canonical_access_ids(store)
            await store.connection.execute(
                "DELETE FROM external_identities WHERE principal_id = ?",
                (principal_id,),
            )
            await store.connection.execute(
                "DELETE FROM channel_bindings WHERE channel_id = ?",
                (channel_id,),
            )
            await store.connection.commit()

            access = WorkshopClientAccess(store)
            humans = await access.list_humans()
            issued = await access.issue_enrollment(principal_id, channel_id)

            assert any(human.principal_id == principal_id and channel_id in human.direct_channels for human in humans)
            assert issued.grant.principal_id == principal_id
            assert issued.channel_id == channel_id
        finally:
            await store.close()

    async def test_canonical_enrollment_requires_owned_direct_channel(self, tmp_path: Path):
        store = await _store(tmp_path / "kai.db")
        try:
            alice_id, _ = await _canonical_access_ids(store, "101")
            _, bob_channel_id = await _canonical_access_ids(store, "202")

            with pytest.raises(WorkshopClientAccessError, match="does not own"):
                await WorkshopClientAccess(store).issue_enrollment(alice_id, bob_channel_id)
            async with store.connection.execute("SELECT COUNT(*) FROM workshop_client_enrollment_grants") as cursor:
                assert (await cursor.fetchone())[0] == 0
        finally:
            await store.close()

    async def test_unknown_telegram_identity_cannot_receive_a_grant(self, tmp_path: Path):
        store = await _store(tmp_path / "kai.db")
        try:
            with pytest.raises(WorkshopClientAccessError, match="exactly one canonical human"):
                await WorkshopClientAccess(store).issue_enrollment_for_telegram(999)
            async with store.connection.execute("SELECT COUNT(*) FROM workshop_client_enrollment_grants") as cursor:
                assert (await cursor.fetchone())[0] == 0
        finally:
            await store.close()

    @pytest.mark.parametrize("telegram_user_id", [0, -1, True, 2**63])
    async def test_invalid_telegram_identity_is_bounded(self, tmp_path: Path, telegram_user_id):
        store = await _store(tmp_path / "kai.db")
        try:
            with pytest.raises(WorkshopClientAccessError, match="positive signed 64-bit"):
                await WorkshopClientAccess(store).issue_enrollment_for_telegram(telegram_user_id)
        finally:
            await store.close()

    async def test_device_revocation_survives_store_restart(self, tmp_path: Path):
        database = tmp_path / "kai.db"
        store = await _store(database)
        access = WorkshopClientAccess(store)
        principal_id, channel_id = await _canonical_access_ids(store)
        issued = await access.issue_enrollment(principal_id, channel_id)
        redeemed = await WorkshopClientEnrollmentManager(store).redeem_grant(issued.grant.token, "Alice laptop")
        token = redeemed.session.token
        await access.revoke_device(principal_id, redeemed.device.device_id)
        await store.close()

        restarted = await WorkshopEventStore.open(database)
        try:
            assert await WorkshopClientSessionManager(restarted).authenticate_token(token) is None
            with pytest.raises(WorkshopClientAccessError, match="unavailable"):
                await WorkshopClientAccess(restarted).revoke_device(
                    principal_id,
                    redeemed.device.device_id,
                )
        finally:
            await restarted.close()

    async def test_grant_revocation_is_scoped_and_durable(self, tmp_path: Path):
        database = tmp_path / "kai.db"
        store = await _store(database)
        access = WorkshopClientAccess(store)
        principal_id, channel_id = await _canonical_access_ids(store)
        issued = await access.issue_enrollment(principal_id, channel_id)
        await access.revoke_enrollment(principal_id, issued.grant.grant_id)
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
        with pytest.raises(SystemExit):
            workshop_cli._parser().parse_args(["client-access", "assign-runtime"])

    def test_opaque_identifiers_are_type_checked(self):
        with pytest.raises(WorkshopClientAccessError, match="Invalid device ID"):
            workshop_cli._device_id("101")
        with pytest.raises(WorkshopClientAccessError, match="Invalid enrollment grant ID"):
            workshop_cli._enrollment_grant_id("101")
        with pytest.raises(WorkshopClientAccessError, match="Invalid principal ID"):
            workshop_cli._principal_id("101")
        with pytest.raises(WorkshopClientAccessError, match="Invalid channel ID"):
            workshop_cli._channel_id("101")
        with pytest.raises(WorkshopHumanProvisioningError, match="Invalid Workshop ID"):
            workshop_cli._workshop_id("101")

        assert workshop_cli._device_id("dev_" + "a" * 32) == DeviceId("dev_" + "a" * 32)
        assert workshop_cli._enrollment_grant_id("enr_" + "b" * 32) == EnrollmentGrantId("enr_" + "b" * 32)

    async def test_list_humans_prints_canonical_enrollment_coordinates(
        self,
        tmp_path: Path,
        monkeypatch,
        capsys,
    ):
        store = await _store(tmp_path / "kai.db")
        principal_id, channel_id = await _canonical_access_ids(store)
        await store.close()
        monkeypatch.setattr(workshop_cli, "DATA_DIR", tmp_path)
        args = workshop_cli._parser().parse_args(["client-access", "list-humans"])

        assert await workshop_cli._run(args) == 0
        output = capsys.readouterr().out
        assert "Human: Alice" in output
        assert f"Principal: {principal_id}" in output
        assert f"Direct channel: {channel_id}" in output

    async def test_provision_command_reports_canonical_identity_without_implied_access(
        self,
        tmp_path: Path,
        monkeypatch,
        capsys,
    ):
        store = await _store(tmp_path / "kai.db")
        await store.close()
        monkeypatch.setattr(workshop_cli, "DATA_DIR", tmp_path)
        args = workshop_cli._parser().parse_args(
            [
                "client-access",
                "provision-human",
                "--provisioning-key",
                "charlie",
                "--display-name",
                "Charlie",
                "--role",
                "member",
            ]
        )

        assert await workshop_cli._run(args) == 0
        output = capsys.readouterr().out
        assert "Human: Charlie" in output
        assert "Principal: prn_" in output
        assert "Workshop: wsp_" in output
        assert "Direct channel: chn_" in output
        assert "Role: member" in output
        assert "Provisioning key: charlie" in output
        assert "Status: created" in output
        assert "Transport access: not assigned" in output
        assert "Runtime access: not assigned" in output

    async def test_issue_command_prints_the_token_once_with_qualification_coordinates(
        self,
        tmp_path: Path,
        monkeypatch,
        capsys,
    ):
        store = await _store(tmp_path / "kai.db")
        await store.close()
        monkeypatch.setattr(workshop_cli, "DATA_DIR", tmp_path)
        store = await WorkshopEventStore.open(tmp_path / "kai.db")
        principal_id, channel_id = await _canonical_access_ids(store)
        await store.close()
        args = workshop_cli._parser().parse_args(
            [
                "client-access",
                "issue-enrollment",
                "--principal-id",
                principal_id,
                "--channel-id",
                channel_id,
            ]
        )

        assert await workshop_cli._run(args) == 0
        output = capsys.readouterr().out
        assert "Enrollment: enr_" in output
        assert "Channel: chn_" in output
        assert "Expires:" in output
        assert output.count("kai_ws_enroll_v1.") == 1
        assert "stores only its hash" in output

    async def test_assign_runtime_command_reports_explicit_channel_agent_policy(
        self,
        tmp_path: Path,
        monkeypatch,
        capsys,
    ):
        store = await _store(tmp_path / "kai.db")
        principal_id, channel_id = await _canonical_access_ids(store)
        await store.close()
        monkeypatch.setattr(workshop_cli, "DATA_DIR", tmp_path)
        monkeypatch.setattr(
            workshop_cli,
            "load_config",
            lambda: Config(
                telegram_bot_token="test",
                allowed_user_ids={101},
                default_backend="codex",
                default_provider="openai",
                default_model="gpt-5.6-sol",
                user_configs={101: UserConfig(telegram_id=101, name="Alice")},
            ),
        )
        args = workshop_cli._parser().parse_args(
            [
                "client-access",
                "assign-runtime",
                "--principal-id",
                principal_id,
                "--channel-id",
                channel_id,
                "--runtime-profile-id",
                profile_id(101),
            ]
        )

        assert await workshop_cli._run(args) == 0
        output = capsys.readouterr().out
        assert f"Principal: {principal_id}" in output
        assert f"Direct channel: {channel_id}" in output
        assert "Agent: agt_" in output
        assert f"Runtime profile: {profile_id(101)}" in output
        assert "Status: assigned" in output
        assert "not human or transport identity" in output


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
            assert ("GET", "/v1/channels/{channel_id}/events") in routes
            assert ("GET", "/workshop") in routes
            assert ("GET", "/workshop/") in routes
            assert ("GET", "/workshop/app.css") in routes
            assert ("GET", "/workshop/app.js") in routes
            assert ("POST", "/v1/channels/{channel_id}/timeline") not in routes
            assert ("POST", "/v1/channels/{channel_id}/events") not in routes
            assert ("POST", "/v1/client/enrollment") not in routes

            operator_store = await WorkshopEventStore.open(database)
            try:
                principal_id, channel_id = await _canonical_access_ids(operator_store)
                issued = await WorkshopClientAccess(operator_store).issue_enrollment(
                    principal_id,
                    channel_id,
                )
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

            events = await client.get(
                f"/v1/channels/{issued.channel_id}/events",
                headers={"Authorization": f"Bearer {redeemed['session']['token']}"},
            )
            assert events.status == 200
            assert events.content_type == "text/event-stream"
            events.close()
        finally:
            if client is not None:
                await client.close()
            await webhook.stop()
