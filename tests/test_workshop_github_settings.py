"""Canonical personal GitHub-settings authority tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.execution_state import WorkshopExecutionStateRegistry
from kai.workshop.github_settings import (
    GitHubSettingsAuthority,
    WorkshopGitHubSettingsAccessDenied,
    WorkshopGitHubSettingsConflict,
    WorkshopGitHubSettingsService,
    WorkshopGitHubSettingsStorageError,
    WorkshopGitHubSettingsValidationError,
)
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileRegistry
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id, profile_registry


async def _seed(path: Path):
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman("Alice", "admin", "telegram", "101", "101", profile_id(101)),
            BootstrapHuman("Bob", "member", "telegram", "202", "202", profile_id(202)),
        ),
    )
    base_profiles = profile_registry(101, 202)
    profiles = WorkshopRuntimeProfileRegistry(
        tuple(
            replace(
                profile,
                github_login="alice" if profile.profile_id == profile_id(101) else "bob",
                github_repos=("owner/repo",),
                pr_review=True,
                issue_triage=False,
            )
            for profile in base_profiles.profiles
        ),
        legacy_runtime_keys={profile_id(value): value for value in (101, 202)},
    )
    registry = await WorkshopExecutionStateRegistry.from_store(store, profiles)
    for namespace in registry.namespaces:
        await store.connection.execute(
            "INSERT INTO principal_github_subscriptions ("
            "principal_id, baseline_repos_json, added_repos_json, removed_repos_json, "
            "pr_review_enabled, issue_triage_enabled, pr_review_source, issue_triage_source) "
            "VALUES (?, '[\"owner/repo\"]', '[]', '[]', 1, 0, 'operator', 'operator')",
            (namespace.principal_id,),
        )
    await store.connection.commit()
    service = await WorkshopGitHubSettingsService.open(path, registry, profiles)
    return store, service, registry


@pytest.mark.asyncio
async def test_snapshot_is_canonical_and_never_returns_token(tmp_path: Path) -> None:
    store, service, registry = await _seed(tmp_path / "kai.db")
    try:
        authority = service.authority_for_principal(registry.namespaces[0].principal_id)
        await store.connection.execute(
            "UPDATE principal_github_subscriptions SET github_token = 'secret-value' WHERE principal_id = ?",
            (authority.principal_id,),
        )
        await store.connection.commit()

        snapshot = await service.inspect(authority)

        assert snapshot.token_stored is True
        assert snapshot.github_login == "alice"
        assert "secret-value" not in repr(snapshot)
        assert [(item.repository, item.source, item.automation_authorized) for item in snapshot.repositories] == [
            ("owner/repo", "operator", True)
        ]
        assert snapshot.pr_review.enabled is True
        assert snapshot.pr_review.source == "operator"
        assert snapshot.issue_triage.enabled is False
    finally:
        await service.close()
        await store.close()


@pytest.mark.asyncio
async def test_repository_changes_preserve_operator_automation_boundary(tmp_path: Path) -> None:
    store, service, registry = await _seed(tmp_path / "kai.db")
    try:
        authority = service.authority_for_principal(registry.namespaces[0].principal_id)
        initial = await service.inspect(authority)

        added = await service.set_repository_subscription(
            authority,
            "Personal/Notifications",
            subscribed=True,
            expected_revision=initial.revision,
        )
        assert [(item.repository, item.automation_authorized) for item in added.repositories] == [
            ("owner/repo", True),
            ("personal/notifications", False),
        ]

        removed = await service.set_repository_subscription(
            authority,
            "owner/repo",
            subscribed=False,
            expected_revision=added.revision,
        )
        assert [item.repository for item in removed.repositories] == ["personal/notifications"]

        restored = await service.set_repository_subscription(
            authority,
            "owner/repo",
            subscribed=True,
            expected_revision=removed.revision,
        )
        owner = next(item for item in restored.repositories if item.repository == "owner/repo")
        assert owner.source == "operator"
        assert owner.automation_authorized is True

        reset = await service.reset_repository_subscriptions(
            authority,
            expected_revision=restored.revision,
        )
        assert [item.repository for item in reset.repositories] == ["owner/repo"]
        assert reset.repositories_resettable is False
    finally:
        await service.close()
        await store.close()


@pytest.mark.asyncio
async def test_toggle_override_can_reset_to_protected_policy(tmp_path: Path) -> None:
    store, service, registry = await _seed(tmp_path / "kai.db")
    try:
        authority = service.authority_for_principal(registry.namespaces[0].principal_id)
        initial = await service.inspect(authority)

        disabled = await service.set_toggle(
            authority,
            "pr_review",
            False,
            expected_revision=initial.revision,
        )
        assert disabled.pr_review.enabled is False
        assert disabled.pr_review.source == "user"
        assert disabled.pr_review.resettable is True

        reset = await service.set_toggle(
            authority,
            "pr_review",
            None,
            expected_revision=disabled.revision,
        )
        assert reset.pr_review.enabled is True
        assert reset.pr_review.source == "operator"
        assert reset.pr_review.resettable is False
    finally:
        await service.close()
        await store.close()


@pytest.mark.asyncio
async def test_token_is_write_only_replaceable_and_removable(tmp_path: Path) -> None:
    store, service, registry = await _seed(tmp_path / "kai.db")
    try:
        authority = service.authority_for_principal(registry.namespaces[0].principal_id)
        initial = await service.inspect(authority)

        stored = await service.set_token(authority, " token-one ", expected_revision=initial.revision)
        assert stored.token_stored is True
        replaced = await service.set_token(authority, "token-two", expected_revision=stored.revision)
        assert replaced.token_stored is True
        assert replaced.mutation is not None and replaced.mutation.changed is True
        async with store.connection.execute(
            "SELECT github_token FROM principal_github_subscriptions WHERE principal_id = ?",
            (authority.principal_id,),
        ) as cursor:
            assert (await cursor.fetchone())[0] == "token-two"

        cleared = await service.set_token(authority, None, expected_revision=replaced.revision)
        assert cleared.token_stored is False
    finally:
        await service.close()
        await store.close()


@pytest.mark.asyncio
async def test_stale_revision_and_cross_principal_authority_fail_closed(tmp_path: Path) -> None:
    store, service, registry = await _seed(tmp_path / "kai.db")
    try:
        first = registry.namespaces[0]
        second = registry.namespaces[1]
        authority = service.authority_for_principal(first.principal_id)
        initial = await service.inspect(authority)
        changed = await service.set_toggle(authority, "issue_triage", True)
        assert changed.revision != initial.revision

        with pytest.raises(WorkshopGitHubSettingsConflict):
            await service.set_toggle(
                authority,
                "issue_triage",
                False,
                expected_revision=initial.revision,
            )
        with pytest.raises(WorkshopGitHubSettingsAccessDenied):
            service.authority_for_principal_profile(first.principal_id, second.runtime_profile_id)
        forged = GitHubSettingsAuthority(first.principal_id, second.runtime_profile_id)
        with pytest.raises(WorkshopGitHubSettingsAccessDenied):
            await service.inspect(forged)
    finally:
        await service.close()
        await store.close()


@pytest.mark.asyncio
async def test_invalid_and_corrupt_settings_fail_closed(tmp_path: Path) -> None:
    store, service, registry = await _seed(tmp_path / "kai.db")
    try:
        authority = service.authority_for_principal(registry.namespaces[0].principal_id)
        with pytest.raises(WorkshopGitHubSettingsValidationError):
            await service.set_repository_subscription(authority, "not-a-repository", subscribed=True)
        with pytest.raises(WorkshopGitHubSettingsValidationError):
            await service.set_token(authority, " ")

        await store.connection.execute(
            "UPDATE principal_github_subscriptions SET added_repos_json = ? WHERE principal_id = ?",
            (json.dumps(["owner/repo", "OWNER/REPO"]), authority.principal_id),
        )
        await store.connection.commit()
        with pytest.raises(WorkshopGitHubSettingsStorageError):
            await service.inspect(authority)
    finally:
        await service.close()
        await store.close()
