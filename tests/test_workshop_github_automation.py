"""Contracts for canonical GitHub subscription routing and durable automation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.workshop.bootstrap import (
    BootstrapHuman,
    BootstrapNotificationChannel,
    bootstrap_default_workshop,
)
from kai.workshop.domain import MessageId
from kai.workshop.execution_state import WorkshopExecutionStateRegistry
from kai.workshop.github_automation import (
    GitHubAutomationRoutingError,
    WorkshopGitHubAutomationService,
)
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id, profile_registry


async def _seed(path: Path, *, role: str = "member"):
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                "Daniel",
                role,
                "telegram",
                "101",
                "101",
                runtime_profile_id=profile_id(101),
            ),
        ),
        notification_channels=(BootstrapNotificationChannel("telegram", "-100123", ("101",)),),
    )
    registry = await WorkshopExecutionStateRegistry.from_store(store, profile_registry(101))
    namespace = registry.namespaces[0]
    await store.connection.execute(
        "INSERT INTO principal_github_subscriptions ("
        "principal_id, baseline_repos_json, added_repos_json, removed_repos_json, "
        "pr_review_enabled, issue_triage_enabled, pr_review_source, issue_triage_source) "
        "VALUES (?, ?, '[]', '[]', 1, 1, 'operator', 'operator')",
        (namespace.principal_id, '["owner/repo"]'),
    )
    await store.connection.commit()
    return store, registry


def _service(store, registry):
    return WorkshopGitHubAutomationService(
        store,
        MagicMock(),
        registry,
        MagicMock(),
        MagicMock(),
        spec_dir="specs",
        review_timeout_seconds=900,
    )


def _payload(kind: str = "pr_review") -> dict:
    return {
        "action": "opened",
        "repository": {"full_name": "owner/repo"},
        "pull_request" if kind == "pr_review" else "issue": {
            "number": 42,
            "title": "Canonical work",
        },
    }


class TestCanonicalSubscriptionRouting:
    async def test_resolves_principal_runtime_and_notification_channel(self, tmp_path: Path):
        store, registry = await _seed(tmp_path / "kai.db")
        try:
            routes = await _service(store, registry).routes_for_repository("OWNER/REPO")

            assert len(routes) == 1
            route = routes[0]
            assert route.principal_id == registry.namespaces[0].principal_id
            assert route.execution_channel_id == registry.namespaces[0].channel_id
            assert route.runtime_profile_id == profile_id(101)
            assert route.notification_channel_id != route.execution_channel_id
            assert route.pr_review_enabled is True
            assert route.issue_triage_enabled is True
            assert route.operations_authorized is True
        finally:
            await store.close()

    async def test_added_subscription_routes_notifications_but_not_mutations(self, tmp_path: Path):
        store, registry = await _seed(tmp_path / "kai.db")
        try:
            await store.connection.execute(
                "UPDATE principal_github_subscriptions SET added_repos_json = '[\"other/repo\"]'"
            )
            await store.connection.commit()

            route = (await _service(store, registry).routes_for_repository("other/repo"))[0]

            assert route.operations_authorized is False
        finally:
            await store.close()

    async def test_admin_empty_policy_retains_notification_wildcard_without_mutation_authority(
        self,
        tmp_path: Path,
    ):
        store, registry = await _seed(tmp_path / "kai.db", role="admin")
        try:
            await store.connection.execute("UPDATE principal_github_subscriptions SET baseline_repos_json = '[]'")
            await store.connection.commit()

            route = (await _service(store, registry).routes_for_repository("any/repo"))[0]

            assert route.operations_authorized is False
        finally:
            await store.close()

    async def test_corrupt_policy_fails_closed(self, tmp_path: Path):
        store, registry = await _seed(tmp_path / "kai.db")
        try:
            await store.connection.execute("UPDATE principal_github_subscriptions SET baseline_repos_json = 'not-json'")
            await store.connection.commit()

            with pytest.raises(GitHubAutomationRoutingError, match="baseline"):
                await _service(store, registry).routes_for_repository("owner/repo")
        finally:
            await store.close()


class TestDurableGitHubAutomation:
    async def test_stop_bounds_active_work_and_closes_store(self, monkeypatch):
        store = MagicMock()
        store.close = AsyncMock()
        service = WorkshopGitHubAutomationService(
            store,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            spec_dir="specs",
            review_timeout_seconds=900,
        )

        async def never_finishes():
            await asyncio.Event().wait()

        active = asyncio.create_task(never_finishes())

        async def worker():
            await active

        service._active = active
        service._task = asyncio.create_task(worker())
        monkeypatch.setattr("kai.workshop.github_automation._DRAIN_SECONDS", 0.001)

        await service.stop()

        assert active.cancelled()
        assert service._task is None
        store.close.assert_awaited_once()

    async def test_enqueue_is_idempotent_and_rejects_changed_delivery(self, tmp_path: Path):
        store, registry = await _seed(tmp_path / "kai.db")
        try:
            service = _service(store, registry)
            route = (await service.routes_for_repository("owner/repo"))[0]

            first = await service.enqueue(
                delivery_id="delivery-1",
                kind="pr_review",
                event_type="pull_request",
                payload=_payload(),
                route=route,
            )
            replay = await service.enqueue(
                delivery_id="delivery-1",
                kind="pr_review",
                event_type="pull_request",
                payload=_payload(),
                route=route,
            )

            assert first.inserted is True
            assert replay == type(first)(first.work_id, False)
            changed = _payload()
            changed["pull_request"]["title"] = "Different"
            with pytest.raises(GitHubAutomationRoutingError, match="reused"):
                await service.enqueue(
                    delivery_id="delivery-1",
                    kind="pr_review",
                    event_type="pull_request",
                    payload=changed,
                    route=route,
                )
        finally:
            await store.close()

    async def test_worker_executes_with_protected_policy_and_canonical_sink(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        store, registry = await _seed(path)
        route = (await _service(store, registry).routes_for_repository("owner/repo"))[0]
        await store.close()

        runtime_pool = MagicMock()
        runtime_pool.runtime_profile.return_value = SimpleNamespace(
            backend="codex",
            provider="openai",
            os_user="daniel",
        )
        runtime_pool.get_role_model.return_value = "gpt-5.6-sol"
        profile_state = MagicMock()
        profile_state.github_token = AsyncMock(return_value="ghp_secret")
        profile_state.allowed_triage_projects = ("Kai",)
        compatibility = MagicMock()
        compatibility.for_profile.return_value = profile_state
        notifications = MagicMock()
        notifications.record_for_channel = AsyncMock(return_value=SimpleNamespace(message_id=MessageId.new()))

        service = await WorkshopGitHubAutomationService.open_and_start(
            path,
            runtime_pool,
            registry,
            compatibility,
            notifications,
            spec_dir="specs",
            review_timeout_seconds=777,
        )
        try:

            async def fake_review(*_args, **kwargs):
                await kwargs["notification_sink"]("Reviewed")
                return True

            with patch("kai.workshop.github_automation.review.review_pr", side_effect=fake_review) as run:
                result = await service.enqueue(
                    delivery_id="delivery-2",
                    kind="pr_review",
                    event_type="pull_request",
                    payload=_payload(),
                    route=route,
                    local_repo_path="/srv/kai",
                )
                for _ in range(100):
                    async with service._store.connection.execute(
                        "SELECT status FROM workshop_github_automation_work WHERE work_id = ?",
                        (result.work_id,),
                    ) as cursor:
                        status = str((await cursor.fetchone())[0])
                    if status == "succeeded":
                        break
                    await asyncio.sleep(0.01)

            assert status == "succeeded"
            assert run.call_args.kwargs["agent_backend"] == "codex"
            assert run.call_args.kwargs["claude_user"] == "daniel"
            assert run.call_args.kwargs["model_override"] == "gpt-5.6-sol"
            assert run.call_args.kwargs["github_token"] == "ghp_secret"
            assert run.call_args.kwargs["local_repo_path"] == "/srv/kai"
            notifications.record_for_channel.assert_awaited_once()
        finally:
            await service.stop()

    async def test_restart_marks_in_flight_work_uncertain_instead_of_retrying(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        store, registry = await _seed(path)
        route = (await _service(store, registry).routes_for_repository("owner/repo"))[0]
        service = _service(store, registry)
        queued = await service.enqueue(
            delivery_id="delivery-3",
            kind="pr_review",
            event_type="pull_request",
            payload=_payload(),
            route=route,
        )
        await store.connection.execute(
            "UPDATE workshop_github_automation_work SET status = 'executing' WHERE work_id = ?",
            (queued.work_id,),
        )
        await store.connection.commit()
        await store.close()

        reopened = await WorkshopEventStore.open(path)
        recovery = _service(reopened, registry)
        try:
            await recovery._recover_interrupted()
            async with reopened.connection.execute(
                "SELECT status, last_error_code FROM workshop_github_automation_work WHERE work_id = ?",
                (queued.work_id,),
            ) as cursor:
                assert tuple(await cursor.fetchone()) == (
                    "uncertain",
                    "process_restarted_during_execution",
                )
        finally:
            await reopened.close()
