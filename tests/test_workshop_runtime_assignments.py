"""Explicit Workshop channel-agent runtime authority contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.conversation_commands import WorkshopConversationCommandService
from kai.workshop.conversation_runs import resolve_canonical_conversation_run
from kai.workshop.domain import MessageId, RuntimeProfileId
from kai.workshop.human_provisioning import WorkshopHumanProvisioner
from kai.workshop.inbound import ClientInboundMessage
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.runtime_assignments import (
    WorkshopRuntimeAssignmentError,
    WorkshopRuntimeAssignmentService,
    resolve_channel_runtime_profile,
)
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileError, WorkshopRuntimeProfileRegistry
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id, profile_registry


async def _store(path: Path) -> WorkshopEventStore:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (BootstrapHuman("Alice", "admin", "telegram", "101", "101", profile_id(101)),),
    )
    return store


class TestRuntimeAssignmentPolicy:
    async def test_non_telegram_policy_profile_can_be_assigned_to_browser_only_human(self, tmp_path: Path):
        store = await _store(tmp_path / "kai.db")
        runtime_profile_id = RuntimeProfileId("rtp_99999999999999999999999999999999")
        profiles = WorkshopRuntimeProfileRegistry.from_document(
            {
                "version": 1,
                "runtime_profiles": {
                    str(runtime_profile_id): {
                        "display_name": "Browser coding",
                        "backend": "codex",
                        "provider": "openai",
                        "model": "gpt-5.5",
                        "timeout_seconds": 120,
                    }
                },
            },
            backend_registry={"codex": {}},
        )
        try:
            human = await WorkshopHumanProvisioner(store).provision(
                "browser-human",
                "Browser human",
                "member",
            )

            assigned = await WorkshopRuntimeAssignmentService(store, profiles).assign(
                human.principal_id,
                human.channel_id,
                runtime_profile_id,
            )

            assert assigned.runtime_profile_id == runtime_profile_id
            async with store.connection.execute(
                "SELECT COUNT(*) FROM external_identities WHERE principal_id = ?",
                (human.principal_id,),
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 0
        finally:
            await store.close()

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
            profiles = profile_registry(101, 202)
            service = WorkshopRuntimeAssignmentService(store, profiles)

            assigned = await service.assign(
                human.principal_id,
                human.channel_id,
                profile_id(202),
            )
            retried = await service.assign(
                human.principal_id,
                human.channel_id,
                profile_id(202),
            )

            assert assigned.created is True
            assert retried.created is False
            assert retried.assignment_id == assigned.assignment_id
            assert await resolve_channel_runtime_profile(store, human.channel_id) == (
                assigned.agent_id,
                profile_id(202),
            )
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

            assert accepted.runtime_profile_id == profile_id(202)
            assert resolution.runtime_profile_id == profile_id(202)
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
            service = WorkshopRuntimeAssignmentService(store, profile_registry(101, 202))

            with pytest.raises(WorkshopRuntimeAssignmentError, match="must own"):
                await service.assign(charlie.principal_id, dana.channel_id, profile_id(202))

            await service.assign(charlie.principal_id, charlie.channel_id, profile_id(202))
            with pytest.raises(WorkshopRuntimeAssignmentError, match="already assigned"):
                await service.assign(dana.principal_id, dana.channel_id, profile_id(202))
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
            assigned = await WorkshopRuntimeAssignmentService(store, profile_registry(101, 202)).assign(
                human.principal_id,
                human.channel_id,
                profile_id(202),
            )
            await store.connection.execute("DELETE FROM channel_agent_runtime_assignments")
            await store.connection.commit()

            checkpoint = await store.rebuild_projection(CanonicalConversationProjection())

            assert checkpoint.version == 8
            assert await resolve_channel_runtime_profile(store, human.channel_id) == (
                assigned.agent_id,
                profile_id(202),
            )
        finally:
            await store.close()


class TestRuntimeProfileCompatibilityBoundary:
    @pytest.mark.parametrize("value", ("profile-daniel", "0", "01"))
    def test_protected_registry_rejects_non_profile_ids(self, value: str):
        with pytest.raises(WorkshopRuntimeProfileError, match="invalid"):
            profile_registry(101).resolve(value)

    def test_opaque_profile_does_not_encode_runtime_configuration_key(self):
        profile = profile_registry(202).resolve(profile_id(202))

        assert profile.profile_id != str(profile.runtime_config_id)
        assert "202" not in profile.profile_id
