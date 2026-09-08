"""Fresh transport-independent Workshop provisioning contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from kai.workshop.agent_authority import reconcile_single_owner_agent_authority
from kai.workshop.agent_enablement import enable_initial_workshop_agent
from kai.workshop.bootstrap import bootstrap_default_workshop
from kai.workshop.client_access import WorkshopClientAccess
from kai.workshop.domain import RuntimeProfileId
from kai.workshop.execution_state import WorkshopExecutionStateRegistry
from kai.workshop.human_provisioning import WorkshopHumanProvisioner
from kai.workshop.initial_provisioning import (
    WorkshopInitialProvisioning,
    WorkshopInitialProvisioningError,
    parse_initial_provisioning,
)
from kai.workshop.internal_api_contexts import WorkshopInternalAPIContextRegistry
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileRegistry
from kai.workshop.store import WorkshopEventStore
from kai.workshop.transport_linking import (
    WorkshopTransportLinker,
    WorkshopTransportLinkError,
)


def _profiles(profile_id: RuntimeProfileId) -> WorkshopRuntimeProfileRegistry:
    return WorkshopRuntimeProfileRegistry.from_document(
        {
            "version": 2,
            "runtime_profiles": {
                str(profile_id): {
                    "display_name": "Daniel runtime",
                    "backend": "codex",
                    "provider": "openai",
                    "model": "gpt-5.5",
                    "timeout_seconds": 120,
                    "allowed_services": [],
                    "allowed_workspaces": [],
                }
            },
        },
        backend_registry={"codex": {}},
    )


def test_initial_policy_round_trips_without_transport_identity() -> None:
    plan = WorkshopInitialProvisioning.create("Daniel")

    restored = parse_initial_provisioning(plan.to_json())

    assert restored == plan
    assert "telegram" not in plan.to_json().lower()
    with pytest.raises(WorkshopInitialProvisioningError):
        parse_initial_provisioning("v1.not-base64")


async def test_fresh_workshop_only_boot_enables_kai_through_canonical_authority(
    tmp_path: Path,
) -> None:
    plan = WorkshopInitialProvisioning.create("Daniel")
    profiles = _profiles(plan.runtime_profile_id)
    store = await WorkshopEventStore.open(tmp_path / "kai.db")
    try:
        await bootstrap_default_workshop(store, (), workshop_id=plan.workshop_id)
        human = await WorkshopHumanProvisioner(store).provision(
            plan.provisioning_key,
            plan.display_name,
            plan.role,
            workshop_id=plan.workshop_id,
        )
        enabled = await enable_initial_workshop_agent(
            store,
            profiles,
            human.principal_id,
            plan.runtime_profile_id,
        )
        async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
            event_count = int((await cursor.fetchone())[0])
        replayed = await enable_initial_workshop_agent(
            store,
            profiles,
            human.principal_id,
            plan.runtime_profile_id,
        )
        async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
            assert int((await cursor.fetchone())[0]) == event_count
        authority = await reconcile_single_owner_agent_authority(store, profiles)

        assert enabled.created is True
        assert replayed.created is False
        assert replayed.direct_channel_id == enabled.direct_channel_id
        async with store.connection.execute(
            "SELECT lifecycle_state, conversation_started_at FROM principal_agent_enablements WHERE principal_id = ?",
            (human.principal_id,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert str(row[0]) == "enabled"
        assert row[1] is not None
        async with store.connection.execute(
            "SELECT COUNT(*) FROM channels c JOIN channel_memberships cm "
            "ON cm.channel_id = c.id AND cm.principal_id = ? "
            "WHERE c.kind = 'direct'",
            (human.principal_id,),
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 1
        async with store.connection.execute(
            "SELECT COUNT(*) FROM channel_agents ca JOIN channel_memberships cm "
            "ON cm.channel_id = ca.channel_id AND cm.principal_id = ?",
            (human.principal_id,),
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 1
        assert authority.definitions == 1
        assert len((await WorkshopExecutionStateRegistry.from_store(store, profiles)).lanes) == 1
        assert len((await WorkshopInternalAPIContextRegistry.from_store(store, profiles)).contexts) == 1
        enrollment = await WorkshopClientAccess(store).issue_enrollment(
            human.principal_id,
            enabled.direct_channel_id,
        )
        assert enrollment.channel_id == enabled.direct_channel_id
    finally:
        await store.close()


class TestWorkshopTransportLinking:
    async def _provision(self, path: Path):
        plan = WorkshopInitialProvisioning.create("Daniel")
        store = await WorkshopEventStore.open(path)
        await bootstrap_default_workshop(store, (), workshop_id=plan.workshop_id)
        human = await WorkshopHumanProvisioner(store).provision(
            plan.provisioning_key,
            plan.display_name,
            plan.role,
            workshop_id=plan.workshop_id,
        )
        profiles = _profiles(plan.runtime_profile_id)
        enablement = await enable_initial_workshop_agent(
            store,
            profiles,
            human.principal_id,
            plan.runtime_profile_id,
        )
        return store, plan, human, profiles, enablement

    async def test_later_telegram_link_reuses_human_channel_and_assignment(
        self,
        tmp_path: Path,
    ) -> None:
        store, plan, human, profiles, enablement = await self._provision(tmp_path / "kai.db")
        try:
            linker = WorkshopTransportLinker(store, profiles)
            linked = await linker.link_runtime_profile(
                plan.runtime_profile_id,
                transport="telegram",
                external_subject="2114582497",
                external_channel_id="2114582497",
            )
            retried = await linker.link_runtime_profile(
                plan.runtime_profile_id,
                transport="telegram",
                external_subject="2114582497",
                external_channel_id="2114582497",
            )

            assert linked.principal_id == human.principal_id
            assert linked.channel_id == enablement.direct_channel_id
            assert linked.created_events == 2
            assert retried.created_events == 0
            async with store.connection.execute("SELECT COUNT(*) FROM principals WHERE kind = 'human'") as cursor:
                assert int((await cursor.fetchone())[0]) == 1
            async with store.connection.execute(
                "SELECT runtime_profile_id FROM channel_agent_runtime_assignments WHERE channel_id = ?",
                (enablement.direct_channel_id,),
            ) as cursor:
                assert str((await cursor.fetchone())[0]) == str(plan.runtime_profile_id)
        finally:
            await store.close()

    async def test_transport_identity_cannot_be_rebound_to_another_human(
        self,
        tmp_path: Path,
    ) -> None:
        store, plan, _human, profiles, _assignment = await self._provision(tmp_path / "kai.db")
        try:
            await WorkshopTransportLinker(store, profiles).link_runtime_profile(
                plan.runtime_profile_id,
                transport="telegram",
                external_subject="2114582497",
                external_channel_id="2114582497",
            )
            other_profile = RuntimeProfileId("rtp_99999999999999999999999999999999")
            other_profiles = WorkshopRuntimeProfileRegistry(
                (*profiles.profiles, _profiles(other_profile).resolve(other_profile))
            )
            other = await WorkshopHumanProvisioner(store).provision(
                "other",
                "Other",
                "admin",
                workshop_id=plan.workshop_id,
            )
            await enable_initial_workshop_agent(
                store,
                other_profiles,
                other.principal_id,
                other_profile,
            )
            with pytest.raises(
                WorkshopTransportLinkError,
                match="different canonical principal",
            ):
                await WorkshopTransportLinker(
                    store,
                    other_profiles,
                ).link_runtime_profile(
                    other_profile,
                    transport="telegram",
                    external_subject="2114582497",
                    external_channel_id="999",
                )
        finally:
            await store.close()
