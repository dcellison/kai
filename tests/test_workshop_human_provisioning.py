"""Canonical Workshop human provisioning contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from kai.workshop.authorization import CanonicalChannelAuthorizer
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.client_access import WorkshopClientAccess
from kai.workshop.client_sessions import WorkshopClientEnrollmentManager
from kai.workshop.conversation_commands import (
    ConversationCommandStateConflictError,
    WorkshopConversationCommandService,
)
from kai.workshop.human_provisioning import (
    WorkshopHumanProvisioner,
    WorkshopHumanProvisioningError,
)
from kai.workshop.inbound import ClientInboundMessage
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id


async def _store(path: Path) -> WorkshopEventStore:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (BootstrapHuman("Alice", "admin", "telegram", "101", "101", profile_id(101)),),
    )
    return store


class TestWorkshopHumanProvisioner:
    async def test_provisions_complete_collaboration_identity_without_external_authority(
        self,
        tmp_path: Path,
    ):
        store = await _store(tmp_path / "kai.db")
        try:
            provisioned = await WorkshopHumanProvisioner(store).provision(
                "charlie",
                "  Charlie  ",
                "member",
            )

            async with store.connection.execute(
                "SELECT p.display_name, wm.role, c.kind, cm.role "
                "FROM principals p "
                "JOIN workshop_memberships wm ON wm.principal_id = p.id "
                "JOIN channel_memberships cm ON cm.principal_id = p.id "
                "JOIN channels c ON c.id = cm.channel_id AND c.workshop_id = wm.workshop_id "
                "WHERE p.id = ? AND c.id = ?",
                (provisioned.principal_id, provisioned.channel_id),
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            assert tuple(row) == ("Charlie", "member", "direct", "owner")

            async with store.connection.execute(
                "SELECT COUNT(*) FROM external_identities WHERE principal_id = ?",
                (provisioned.principal_id,),
            ) as cursor:
                assert (await cursor.fetchone())[0] == 0
            async with store.connection.execute(
                "SELECT COUNT(*) FROM channel_bindings WHERE channel_id = ?",
                (provisioned.channel_id,),
            ) as cursor:
                assert (await cursor.fetchone())[0] == 0
            async with store.connection.execute(
                "SELECT COUNT(*) FROM channel_agents ca "
                "JOIN agents a ON a.id = ca.agent_id AND a.name = 'Kai' "
                "JOIN channel_memberships cm ON cm.channel_id = ca.channel_id "
                "AND cm.principal_id = a.principal_id AND cm.role = 'participant' "
                "WHERE ca.channel_id = ?",
                (provisioned.channel_id,),
            ) as cursor:
                assert (await cursor.fetchone())[0] == 1
            async with store.connection.execute(
                "SELECT COUNT(*) FROM event_log WHERE idempotency_key LIKE ? "
                "AND json_extract(metadata_json, '$.source') = 'operator_cli'",
                (f"operator:human-provisioning:{provisioned.principal_id}:%",),
            ) as cursor:
                assert (await cursor.fetchone())[0] == 6
        finally:
            await store.close()

    async def test_provisioned_human_can_enroll_and_read_but_has_no_runtime_authority(
        self,
        tmp_path: Path,
    ):
        store = await _store(tmp_path / "kai.db")
        try:
            provisioned = await WorkshopHumanProvisioner(store).provision(
                "charlie",
                "Charlie",
                "member",
            )
            access = WorkshopClientAccess(store)
            issued = await access.issue_enrollment(
                provisioned.principal_id,
                provisioned.channel_id,
            )
            redeemed = await WorkshopClientEnrollmentManager(store).redeem_grant(
                issued.grant.token,
                "Charlie's laptop",
            )

            assert redeemed.device.principal_id == provisioned.principal_id
            assert await CanonicalChannelAuthorizer(store).can_read_channel(
                provisioned.principal_id,
                provisioned.channel_id,
            )
            with pytest.raises(
                ConversationCommandStateConflictError,
                match="explicit runtime profile assignment",
            ):
                await WorkshopConversationCommandService(store).accept_client(
                    ClientInboundMessage(
                        principal_id=provisioned.principal_id,
                        channel_id=provisioned.channel_id,
                        client_message_id="provisioned-human-no-runtime",
                        body="This must not inherit another user's runtime",
                        occurred_at=datetime.now(UTC),
                    )
                )
        finally:
            await store.close()

    async def test_same_provisioning_key_is_a_safe_semantic_retry(self, tmp_path: Path):
        store = await _store(tmp_path / "kai.db")
        try:
            provisioner = WorkshopHumanProvisioner(store)
            first = await provisioner.provision("charlie", "Charlie", "member")
            async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
                after_first = (await cursor.fetchone())[0]

            retried = await provisioner.provision("charlie", "Charlie", "member")

            assert first.created is True
            assert retried.created is False
            assert retried.principal_id == first.principal_id
            assert retried.channel_id == first.channel_id
            async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
                assert (await cursor.fetchone())[0] == after_first

            with pytest.raises(WorkshopHumanProvisioningError, match="different human"):
                await provisioner.provision("charlie", "Charles", "member")
            async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
                assert (await cursor.fetchone())[0] == after_first
        finally:
            await store.close()

    async def test_missing_canonical_kai_agent_rolls_back_without_partial_human(
        self,
        tmp_path: Path,
    ):
        store = await _store(tmp_path / "kai.db")
        try:
            await store.connection.execute("DELETE FROM channel_agents")
            await store.connection.execute("DELETE FROM agents")
            await store.connection.commit()
            async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
                before_events = (await cursor.fetchone())[0]
            async with store.connection.execute("SELECT COUNT(*) FROM principals") as cursor:
                before_principals = (await cursor.fetchone())[0]

            with pytest.raises(WorkshopHumanProvisioningError, match="canonical Kai agent"):
                await WorkshopHumanProvisioner(store).provision(
                    "charlie",
                    "Charlie",
                    "member",
                )

            async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
                assert (await cursor.fetchone())[0] == before_events
            async with store.connection.execute("SELECT COUNT(*) FROM principals") as cursor:
                assert (await cursor.fetchone())[0] == before_principals
        finally:
            await store.close()

    @pytest.mark.parametrize(
        ("display_name", "role", "match"),
        [
            ("", "member", "Display name"),
            ("x" * 201, "member", "Display name"),
            ("Charlie", "owner", "Role"),
        ],
    )
    async def test_invalid_operator_input_creates_no_events(
        self,
        tmp_path: Path,
        display_name: str,
        role: str,
        match: str,
    ):
        store = await _store(tmp_path / "kai.db")
        try:
            async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
                before = (await cursor.fetchone())[0]
            with pytest.raises(WorkshopHumanProvisioningError, match=match):
                await WorkshopHumanProvisioner(store).provision(
                    "charlie",
                    display_name,
                    role,
                )
            async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
                assert (await cursor.fetchone())[0] == before
        finally:
            await store.close()

    @pytest.mark.parametrize("provisioning_key", ["", "Charlie", "two words", "x" * 65])
    async def test_invalid_provisioning_key_creates_no_events(
        self,
        tmp_path: Path,
        provisioning_key: str,
    ):
        store = await _store(tmp_path / "kai.db")
        try:
            async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
                before = (await cursor.fetchone())[0]
            with pytest.raises(WorkshopHumanProvisioningError, match="Provisioning key"):
                await WorkshopHumanProvisioner(store).provision(
                    provisioning_key,
                    "Charlie",
                    "member",
                )
            async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
                assert (await cursor.fetchone())[0] == before
        finally:
            await store.close()
