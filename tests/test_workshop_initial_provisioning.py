"""Fresh transport-independent Workshop provisioning contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from kai.workshop.bootstrap import bootstrap_default_workshop
from kai.workshop.domain import RuntimeProfileId
from kai.workshop.human_provisioning import WorkshopHumanProvisioner
from kai.workshop.initial_provisioning import (
    WorkshopInitialProvisioning,
    WorkshopInitialProvisioningError,
    parse_initial_provisioning,
)
from kai.workshop.runtime_assignments import WorkshopRuntimeAssignmentService
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
        assignment = await WorkshopRuntimeAssignmentService(store, profiles).assign(
            human.principal_id,
            human.channel_id,
            plan.runtime_profile_id,
        )
        return store, plan, human, profiles, assignment

    async def test_later_telegram_link_reuses_human_channel_and_assignment(
        self,
        tmp_path: Path,
    ) -> None:
        store, plan, human, profiles, assignment = await self._provision(tmp_path / "kai.db")
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
            assert linked.channel_id == human.channel_id
            assert linked.created_events == 2
            assert retried.created_events == 0
            async with store.connection.execute("SELECT COUNT(*) FROM principals WHERE kind = 'human'") as cursor:
                assert int((await cursor.fetchone())[0]) == 1
            async with store.connection.execute(
                "SELECT runtime_profile_id FROM channel_agent_runtime_assignments WHERE id = ?",
                (assignment.assignment_id,),
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
                "member",
                workshop_id=plan.workshop_id,
            )
            await WorkshopRuntimeAssignmentService(store, other_profiles).assign(
                other.principal_id,
                other.channel_id,
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
