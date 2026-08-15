"""Canonical scheduled-job and GitHub subscription authority tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kai import sessions
from kai.main import _warn_if_compatibility_jobs_are_dormant
from kai.workshop.bootstrap import BootstrapHuman
from kai.workshop.diagnostics import workshop_operational_state_status
from kai.workshop.execution_state import (
    WorkshopExecutionStateNamespace,
    WorkshopExecutionStateRegistry,
)
from kai.workshop.operational_state import WorkshopOperationalStateError
from tests.workshop_profiles import profile_id, profile_registry


class _Config:
    def __init__(self, users: dict[int, SimpleNamespace]) -> None:
        self._users = users

    def get_user_config(self, runtime_config_id: int):
        return self._users.get(runtime_config_id)


def _config(*runtime_ids: int) -> _Config:
    return _Config(
        {
            runtime_id: SimpleNamespace(
                github_repos=[f"owner/repo-{runtime_id}"],
                pr_review=True,
                issue_triage=False,
                github_notify_chat_id=None,
            )
            for runtime_id in runtime_ids
        }
    )


async def _bootstrap(*runtime_ids: int):
    await sessions.bootstrap_workshop_foundation(
        tuple(
            BootstrapHuman(
                display_name=f"Human {runtime_id}",
                role="admin" if index == 0 else "member",
                transport="telegram",
                external_subject=str(runtime_id),
                external_channel_id=str(runtime_id),
                runtime_profile_id=profile_id(runtime_id),
            )
            for index, runtime_id in enumerate(runtime_ids)
        )
    )
    registry, _ = await sessions.initialize_workshop_execution_state(profile_registry(*runtime_ids))
    return registry


@pytest.fixture
async def database(tmp_path: Path):
    path = tmp_path / "kai.db"
    await sessions.init_db(path)
    yield path
    await sessions.close_db()


class TestCanonicalOperationalStateMigration:
    async def test_backfills_jobs_and_github_policy_once(self, database: Path):
        await sessions.set_setting("github_repos_added:101", '["owner/added"]')
        await sessions.set_setting("github_repos_removed:101", '["owner/repo-101"]')
        await sessions.set_setting("pr_review:101", "false")
        job_id = await sessions.create_job(
            101,
            "daily",
            "agent",
            "Check status",
            "daily",
            '{"times":["09:00"]}',
        )
        registry = await _bootstrap(101)

        first = await sessions.initialize_workshop_operational_state(
            registry,
            _config(101),
        )
        second = await sessions.initialize_workshop_operational_state(
            registry,
            _config(101),
        )

        assert first.newly_migrated == 1
        assert first.jobs == 1
        assert first.github_subscriptions == 1
        assert second.newly_migrated == 0
        assert [job["id"] for job in await sessions.get_jobs(101)] == [job_id]
        assert await sessions.get_effective_repos(101, ["ignored/after-cutover"]) == ["owner/added"]
        settings = await sessions.resolve_github_settings(101, _config(101))
        assert settings["pr_review"] is False
        assert settings["issue_triage"] is False

    async def test_restart_does_not_reimport_legacy_tampering(self, database: Path):
        registry = await _bootstrap(101)
        await sessions.initialize_workshop_operational_state(registry, _config(101))
        await sessions.close_db()

        await sessions.init_db(database)
        await sessions.set_setting("github_repos_added:101", '["tampered/repo"]')
        registry = await _bootstrap(101)
        migration = await sessions.initialize_workshop_operational_state(
            registry,
            _config(101),
        )

        assert migration.newly_migrated == 0
        assert await sessions.get_effective_repos(101, []) == ["owner/repo-101"]

    async def test_restart_self_heals_job_created_during_rollback_window(self, database: Path):
        registry = await _bootstrap(101)
        await sessions.initialize_workshop_operational_state(registry, _config(101))
        cursor = await sessions._get_db().execute(
            "INSERT INTO jobs (chat_id, name, job_type, prompt, schedule_type, schedule_data) "
            "VALUES (101, 'rollback', 'reminder', 'Check safely', 'once', '{}')"
        )
        await sessions._get_db().commit()
        job_id = int(cursor.lastrowid)

        await sessions.close_db()
        await sessions.init_db(database)
        registry = await _bootstrap(101)
        migration = await sessions.initialize_workshop_operational_state(registry, _config(101))

        assert migration.newly_migrated == 0
        assert migration.jobs == 1
        assert [job["id"] for job in await sessions.get_jobs(101)] == [job_id]
        assert await _warn_if_compatibility_jobs_are_dormant(SimpleNamespace(telegram_enabled=False)) == 1

    async def test_restart_rejects_conflicting_job_owner_with_remediation(self, database: Path):
        registry = await _bootstrap(101)
        await sessions.initialize_workshop_operational_state(registry, _config(101))
        cursor = await sessions._get_db().execute(
            "INSERT INTO jobs (chat_id, name, job_type, prompt, schedule_type, schedule_data) "
            "VALUES (101, 'conflict', 'reminder', 'Do not expose', 'once', '{}')"
        )
        job_id = int(cursor.lastrowid)
        namespace = registry.namespaces[0]
        await sessions._get_db().execute(
            "INSERT INTO workshop_job_owners "
            "(job_id, principal_id, channel_id, agent_id, runtime_profile_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                job_id,
                "prn_00000000000000000000000000000001",
                namespace.channel_id,
                namespace.agent_id,
                namespace.runtime_profile_id,
            ),
        )
        await sessions._get_db().commit()

        with pytest.raises(
            WorkshopOperationalStateError,
            match=rf"job ids: {job_id}.*restore workshop_job_owners",
        ):
            await sessions.initialize_workshop_operational_state(registry, _config(101))

    async def test_conflicting_receipt_owner_fails_closed(self, database: Path):
        registry = await _bootstrap(101)
        await sessions.initialize_workshop_operational_state(registry, _config(101))
        await sessions._get_db().execute(
            "UPDATE workshop_operational_state_migrations SET channel_id = ?",
            ("chn_00000000000000000000000000000001",),
        )
        await sessions._get_db().commit()

        with pytest.raises(WorkshopOperationalStateError, match="conflicts with current"):
            await sessions.initialize_workshop_operational_state(registry, _config(101))

    async def test_empty_sibling_profile_reuses_principal_policy(self, database: Path):
        registry = await _bootstrap(101)
        first = registry.namespaces[0]
        sibling = WorkshopExecutionStateNamespace(
            principal_id=first.principal_id,
            channel_id=first.channel_id,
            agent_id=first.agent_id,
            runtime_profile_id=profile_id(202),
            runtime_config_id=202,
        )

        migration = await sessions.initialize_workshop_operational_state(
            WorkshopExecutionStateRegistry((first, sibling)),
            _config(101),
        )

        assert migration.newly_migrated == 2
        assert migration.github_subscriptions == 1
        assert await sessions.get_effective_repos(101, []) == ["owner/repo-101"]


class TestCanonicalOperationalStateWrites:
    async def test_jobs_are_owned_by_canonical_lane_and_isolated(self, database: Path):
        registry = await _bootstrap(101, 202)
        await sessions.initialize_workshop_operational_state(registry, _config(101, 202))

        job_id = await sessions.create_job(
            101,
            "once",
            "reminder",
            "Canonical reminder",
            "once",
            '{"run_at":"2026-08-16T00:00:00Z"}',
        )

        assert sessions.execution_lane_key(101) != 101
        assert sessions.execution_lane_key(101) != sessions.execution_lane_key(202)
        assert [job["id"] for job in await sessions.get_jobs(101)] == [job_id]
        assert await sessions.get_jobs(202) == []
        assert await sessions.delete_job(job_id, 202) is False
        assert await sessions.delete_job(job_id, 101) is True

    async def test_github_mutations_use_canonical_rows_and_dual_write(self, database: Path):
        registry = await _bootstrap(101)
        await sessions.initialize_workshop_operational_state(registry, _config(101))

        await sessions.set_github_added_repos(101, ["OWNER/ADDED"])
        await sessions.set_github_removed_repos(101, ["owner/repo-101"])
        await sessions.set_github_toggle(101, "issue_triage", True)

        assert await sessions.get_effective_repos(101, []) == ["owner/added"]
        assert (await sessions.resolve_github_settings(101, _config(101)))["issue_triage"] is True
        assert await sessions.get_setting("github_repos_added:101") == '["owner/added"]'
        assert await sessions.get_setting("issue_triage:101") == "true"

    async def test_operator_policy_resync_preserves_user_toggle_override(self, database: Path):
        registry = await _bootstrap(101)
        await sessions.initialize_workshop_operational_state(registry, _config(101))
        await sessions.set_github_toggle(101, "pr_review", True)
        changed = _Config(
            {
                101: SimpleNamespace(
                    github_repos=["owner/replacement"],
                    pr_review=False,
                    issue_triage=True,
                    github_notify_chat_id=None,
                )
            }
        )

        await sessions.initialize_workshop_operational_state(registry, changed)

        assert await sessions.get_effective_repos(101, ["ignored/legacy"]) == ["owner/replacement"]
        settings = await sessions.resolve_github_settings(101, changed)
        assert settings["pr_review"] is True
        assert settings["issue_triage"] is True

    async def test_admin_wildcard_tracks_current_workshop_role(self, database: Path):
        registry = await _bootstrap(101)
        await sessions.initialize_workshop_operational_state(registry, _config(101))
        namespace = registry.namespaces[0]

        assert await sessions.github_admin_wildcard(101, legacy_admin=False) is True
        await sessions._get_db().execute(
            "UPDATE workshop_memberships SET role = 'member' "
            "WHERE workshop_id = (SELECT workshop_id FROM channels WHERE id = ?) "
            "AND principal_id = ?",
            (namespace.channel_id, namespace.principal_id),
        )
        await sessions._get_db().commit()

        assert await sessions.github_admin_wildcard(101, legacy_admin=True) is False

    @pytest.mark.parametrize(
        ("column", "reader", "message"),
        (
            ("added_repos_json", sessions.get_github_added_repos, "added repository policy"),
            ("removed_repos_json", sessions.get_github_removed_repos, "removed repository policy"),
        ),
    )
    async def test_corrupt_canonical_repo_policy_fails_closed(
        self,
        database: Path,
        column: str,
        reader,
        message: str,
    ):
        registry = await _bootstrap(101)
        await sessions.initialize_workshop_operational_state(registry, _config(101))
        await sessions._get_db().execute(f"UPDATE principal_github_subscriptions SET {column} = '{{}}'")
        await sessions._get_db().commit()

        with pytest.raises(RuntimeError, match=message):
            await reader(101)


class TestCanonicalOperationalStateDiagnostic:
    async def test_reports_clean_authority_and_detects_unowned_job(self, database: Path):
        registry = await _bootstrap(101)
        await sessions.initialize_workshop_operational_state(registry, _config(101))

        clean = workshop_operational_state_status(database)
        assert clean.startswith("Workshop operational state: active;")
        assert "profiles=1, migrated=1, missing=0, stale=0" in clean
        assert "protected legacy ownership reads=disabled" in clean

        await sessions._get_db().execute(
            "INSERT INTO jobs (chat_id, name, job_type, prompt, schedule_type, schedule_data) "
            "VALUES (101, 'unowned', 'reminder', 'secret', 'once', '{}')"
        )
        await sessions._get_db().commit()

        drifted = workshop_operational_state_status(database)
        assert drifted.startswith("Workshop operational state: INCOMPLETE;")
        assert "unowned jobs=1" in drifted
        assert "secret" not in drifted
