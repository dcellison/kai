"""Explicit Workshop channel-agent runtime authority contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.conversation_commands import WorkshopConversationCommandService
from kai.workshop.conversation_runs import resolve_canonical_conversation_run
from kai.workshop.domain import MessageId
from kai.workshop.human_provisioning import WorkshopHumanProvisioner
from kai.workshop.inbound import ClientInboundMessage
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.runtime_assignments import (
    WorkshopRuntimeAssignmentError,
    WorkshopRuntimeAssignmentService,
    compatibility_user_id,
    resolve_channel_runtime_profile,
)
from kai.workshop.store import WorkshopEventStore


async def _store(path: Path) -> WorkshopEventStore:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (BootstrapHuman("Alice", "admin", "telegram", "101", "101", "101"),),
    )
    return store


class TestRuntimeAssignmentPolicy:
    async def test_provisioned_human_gains_runtime_only_after_explicit_assignment(
        self,
        tmp_path: Path,
    ):
        store = await _store(tmp_path / "kai.db")
        try:
            human = await WorkshopHumanProvisioner(store).provision(
                "charlie",
                "Charlie",
                "member",
            )
            service = WorkshopRuntimeAssignmentService(store)

            assigned = await service.assign(
                human.principal_id,
                human.channel_id,
                "202",
            )
            retried = await service.assign(
                human.principal_id,
                human.channel_id,
                "202",
            )

            assert assigned.created is True
            assert retried.created is False
            assert retried.assignment_id == assigned.assignment_id
            assert await resolve_channel_runtime_profile(store, human.channel_id) == (
                assigned.agent_id,
                "202",
            )
            assert compatibility_user_id("202") == 202
            async with store.connection.execute(
                "SELECT COUNT(*) FROM external_identities WHERE principal_id = ?",
                (human.principal_id,),
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 0

            accepted = await WorkshopConversationCommandService(store).accept_client(
                ClientInboundMessage(
                    principal_id=human.principal_id,
                    channel_id=human.channel_id,
                    client_message_id="charlie-command-1",
                    body="Use only my explicitly assigned runtime",
                    occurred_at=datetime.now(UTC),
                )
            )
            resolution = await resolve_canonical_conversation_run(
                store,
                MessageId(str(accepted.command.message.event.envelope.aggregate_id)),
            )

            assert accepted.runtime_profile_id == "202"
            assert resolution._legacy_pool_key == 202
        finally:
            await store.close()

    async def test_assignment_rejects_cross_human_authority_and_profile_reuse(
        self,
        tmp_path: Path,
    ):
        store = await _store(tmp_path / "kai.db")
        try:
            charlie = await WorkshopHumanProvisioner(store).provision(
                "charlie",
                "Charlie",
                "member",
            )
            dana = await WorkshopHumanProvisioner(store).provision(
                "dana",
                "Dana",
                "member",
            )
            service = WorkshopRuntimeAssignmentService(store)

            with pytest.raises(WorkshopRuntimeAssignmentError, match="must own"):
                await service.assign(charlie.principal_id, dana.channel_id, "202")

            await service.assign(charlie.principal_id, charlie.channel_id, "202")
            with pytest.raises(WorkshopRuntimeAssignmentError, match="already assigned"):
                await service.assign(dana.principal_id, dana.channel_id, "202")
        finally:
            await store.close()

    async def test_projection_rebuild_restores_runtime_assignment(self, tmp_path: Path):
        store = await _store(tmp_path / "kai.db")
        try:
            human = await WorkshopHumanProvisioner(store).provision(
                "charlie",
                "Charlie",
                "member",
            )
            assigned = await WorkshopRuntimeAssignmentService(store).assign(
                human.principal_id,
                human.channel_id,
                "202",
            )
            await store.connection.execute("DELETE FROM channel_agent_runtime_assignments")
            await store.connection.commit()

            checkpoint = await store.rebuild_projection(CanonicalConversationProjection())

            assert checkpoint.version == 7
            assert await resolve_channel_runtime_profile(store, human.channel_id) == (
                assigned.agent_id,
                "202",
            )
        finally:
            await store.close()


class TestRuntimeProfileCompatibilityBoundary:
    @pytest.mark.parametrize("value", ("profile-daniel", "0", "01"))
    def test_current_host_runtime_rejects_non_integer_profile_keys(self, value: str):
        with pytest.raises(WorkshopRuntimeAssignmentError, match="integer-keyed"):
            compatibility_user_id(value)
