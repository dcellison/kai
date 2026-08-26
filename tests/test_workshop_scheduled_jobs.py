"""Canonical scheduled-job persistence and authorization contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from kai import sessions
from kai.workshop.bootstrap import BootstrapHuman
from kai.workshop.domain import AgentId
from kai.workshop.scheduled_jobs import (
    WorkshopScheduledJobAuthority,
    WorkshopScheduledJobAuthorityError,
    WorkshopScheduledJobStore,
    WorkshopScheduledJobUpdate,
)
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id, profile_registry


@pytest.fixture
async def scheduled_jobs(tmp_path: Path):
    database = tmp_path / "kai.db"
    await sessions.init_db(database)
    await sessions.bootstrap_workshop_foundation(
        (
            BootstrapHuman(
                display_name="Daniel",
                role="admin",
                transport="telegram",
                external_subject="101",
                external_channel_id="101",
                runtime_profile_id=profile_id(101),
            ),
            BootstrapHuman(
                display_name="Scott",
                role="member",
                transport="telegram",
                external_subject="202",
                external_channel_id="202",
                runtime_profile_id=profile_id(202),
            ),
        )
    )
    registry, _ = await sessions.initialize_workshop_execution_state(profile_registry(101, 202))
    store = WorkshopEventStore.from_initialized_connection(sessions._get_db())
    authorities = {
        namespace.require_legacy_runtime_key(): WorkshopScheduledJobAuthority(
            namespace.principal_id,
            namespace.channel_id,
            namespace.agent_id,
            namespace.runtime_profile_id,
        )
        for namespace in registry.namespaces
    }
    yield WorkshopScheduledJobStore(store), authorities
    await sessions.close_db()


class TestWorkshopScheduledJobStore:
    async def test_crud_is_scoped_to_one_canonical_execution_lane(self, scheduled_jobs):
        store, authorities = scheduled_jobs
        daniel = authorities[101]
        scott = authorities[202]

        created = await store.create(
            daniel,
            name="Daily status",
            job_type="claude",
            prompt="Report status",
            schedule_type="daily",
            schedule_data='{"times":["09:00"]}',
            auto_remove=True,
        )

        assert created.job_type == "agent"
        assert created.as_dict()["id"] == created.job_id
        assert "principal_id" not in created.as_dict()
        assert [job.job_id for job in await store.list_active(daniel)] == [created.job_id]
        assert await store.get(created.job_id, scott) is None

        assert await store.update(
            created.job_id,
            daniel,
            WorkshopScheduledJobUpdate(
                name="Updated status",
                schedule_data='{"times":["10:00"]}',
                notify_on_check=True,
            ),
        )
        updated = await store.get(created.job_id, daniel)
        assert updated is not None
        assert updated.name == "Updated status"
        assert updated.schedule_data == '{"times":["10:00"]}'
        assert updated.notify_on_check is True

        assert not await store.delete(created.job_id, scott)
        assert await store.delete(created.job_id, daniel)
        assert await store.get(created.job_id, daniel, active_only=False) is None

    async def test_invalid_authority_fails_closed_without_mutation(self, scheduled_jobs):
        store, authorities = scheduled_jobs
        valid = authorities[101]
        invalid = WorkshopScheduledJobAuthority(
            valid.principal_id,
            valid.channel_id,
            AgentId("agt_" + "f" * 32),
            valid.runtime_profile_id,
        )

        with pytest.raises(WorkshopScheduledJobAuthorityError, match="canonical execution lane"):
            await store.create(
                invalid,
                name="Forbidden",
                job_type="reminder",
                prompt="Do not create",
                schedule_type="once",
                schedule_data='{"run_at":"2036-01-01T00:00:00Z"}',
            )

        assert await store.active_ids() == set()

    async def test_deactivation_retains_definition_but_hides_active_job(self, scheduled_jobs):
        store, authorities = scheduled_jobs
        authority = authorities[101]
        created = await store.create(
            authority,
            name="One shot",
            job_type="reminder",
            prompt="Remember",
            schedule_type="once",
            schedule_data='{"run_at":"2036-01-01T00:00:00Z"}',
        )

        assert await store.deactivate(created.job_id, authority)
        assert await store.get(created.job_id, authority) is None
        retained = await store.get(created.job_id, authority, active_only=False)
        assert retained is not None
        assert retained.active is False
