"""Security and routing contracts for canonical notification preferences."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kai.workshop.bootstrap import (
    BootstrapHuman,
    BootstrapNotificationChannel,
    bootstrap_default_workshop,
)
from kai.workshop.delivery_policy import WorkshopDeliveryBindingPolicy
from kai.workshop.execution_state import WorkshopExecutionStateRegistry
from kai.workshop.integration_notifications import (
    IntegrationNotification,
    WorkshopIntegrationNotificationService,
)
from kai.workshop.notification_preferences import (
    WorkshopNotificationPreferenceAccessDenied,
    WorkshopNotificationPreferenceConflict,
    WorkshopNotificationPreferenceService,
)
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id, profile_registry


async def _seed(path: Path):
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman("Daniel", "admin", "telegram", "101", "101", profile_id(101)),
            BootstrapHuman("Scott", "member", "telegram", "202", "202", profile_id(202)),
        ),
        notification_channels=(BootstrapNotificationChannel("telegram", "-100123", ("101", "202")),),
    )
    registry = await WorkshopExecutionStateRegistry.from_store(
        store,
        profile_registry(101, 202),
    )
    preferences = await WorkshopNotificationPreferenceService.open(path, registry)
    notifications = await WorkshopIntegrationNotificationService.open(
        path,
        WorkshopDeliveryBindingPolicy.disabled(),
        preferences,
    )
    return store, preferences, notifications, registry


def _authority(service, registry, index: int):
    return service.authority_for_principal(registry.namespaces[index].principal_id)


@pytest.mark.asyncio
async def test_principals_see_only_authorized_opaque_destinations(tmp_path: Path) -> None:
    store, service, notifications, registry = await _seed(tmp_path / "kai.db")
    try:
        daniel = await service.inspect(_authority(service, registry, 0))
        scott = await service.inspect(_authority(service, registry, 1))

        assert {item.kind for item in daniel.destinations} == {"direct", "notification"}
        assert {item.kind for item in scott.destinations} == {"direct", "notification"}
        assert all(item.choice_id.startswith("ndst_") for item in daniel.destinations)
        assert not ({item.choice_id for item in daniel.destinations} & {item.choice_id for item in scott.destinations})
        assert [item.integration_class for item in daniel.preferences] == ["github", "generic"]
        assert [item.integration_class for item in scott.preferences] == ["github"]
        assert next(item for item in daniel.preferences if item.integration_class == "github").source == (
            "operator policy"
        )
        assert next(item for item in daniel.preferences if item.integration_class == "generic").source == (
            "operator policy"
        )
        assert "telegram" not in repr(daniel)
        assert "-100123" not in repr(daniel)
    finally:
        await notifications.close()
        await service.close()
        await store.close()


@pytest.mark.asyncio
async def test_selection_reset_and_restart_preserve_canonical_routing(tmp_path: Path) -> None:
    path = tmp_path / "kai.db"
    store, service, notifications, registry = await _seed(path)
    authority = _authority(service, registry, 0)
    initial = await service.inspect(authority)
    direct = next(item for item in initial.destinations if item.kind == "direct")
    selected = await service.select(
        authority,
        "github",
        direct.choice_id,
        expected_revision=initial.revision,
    )
    assert next(item for item in selected.preferences if item.integration_class == "github").source == (
        "personal override"
    )
    await notifications.close()
    await service.close()

    reopened = await WorkshopNotificationPreferenceService.open(path, registry)
    try:
        restarted = await reopened.inspect(reopened.authority_for_principal(authority.principal_id))
        github = next(item for item in restarted.preferences if item.integration_class == "github")
        assert github.destination_choice_id == direct.choice_id
        reset = await reopened.reset(
            reopened.authority_for_principal(authority.principal_id),
            "github",
            expected_revision=restarted.revision,
        )
        restored = next(item for item in reset.preferences if item.integration_class == "github")
        assert restored.destination_kind == "notification"
        assert restored.source == "operator policy"
    finally:
        await reopened.close()
        await store.close()


@pytest.mark.asyncio
async def test_forged_cross_principal_and_stale_choices_fail_without_mutation(tmp_path: Path) -> None:
    store, service, notifications, registry = await _seed(tmp_path / "kai.db")
    try:
        daniel_authority = _authority(service, registry, 0)
        daniel = await service.inspect(daniel_authority)
        scott = await service.inspect(_authority(service, registry, 1))
        scott_direct = next(item for item in scott.destinations if item.kind == "direct")
        with pytest.raises(WorkshopNotificationPreferenceAccessDenied):
            await service.select(
                daniel_authority,
                "github",
                scott_direct.choice_id,
                expected_revision=daniel.revision,
            )

        direct = next(item for item in daniel.destinations if item.kind == "direct")
        changed = await service.select(
            daniel_authority,
            "github",
            direct.choice_id,
            expected_revision=daniel.revision,
        )
        with pytest.raises(WorkshopNotificationPreferenceConflict):
            await service.reset(
                daniel_authority,
                "github",
                expected_revision=daniel.revision,
            )
        current = await service.inspect(daniel_authority)
        assert current.revision == changed.revision
    finally:
        await notifications.close()
        await service.close()
        await store.close()


@pytest.mark.asyncio
async def test_revoked_destination_uses_explicit_direct_fallback(tmp_path: Path) -> None:
    store, service, notifications, registry = await _seed(tmp_path / "kai.db")
    try:
        authority = _authority(service, registry, 0)
        initial = await service.inspect(authority)
        destination = next(item for item in initial.destinations if item.kind == "notification")
        selected = await service.select(
            authority,
            "github",
            destination.choice_id,
            expected_revision=initial.revision,
        )
        async with store.connection.execute(
            "SELECT channel_id FROM channel_memberships WHERE principal_id = ? AND channel_id != ?",
            (authority.principal_id, registry.namespaces[0].channel_id),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        await store.connection.execute(
            "DELETE FROM channel_memberships WHERE principal_id = ? AND channel_id = ?",
            (authority.principal_id, str(row[0])),
        )
        await store.connection.commit()

        fallback = await service.inspect(authority)
        github = next(item for item in fallback.preferences if item.integration_class == "github")
        assert github.destination_kind == "direct"
        assert github.source == "canonical direct fallback"
        assert github.resettable is True
        assert fallback.revision != selected.revision
    finally:
        await notifications.close()
        await service.close()
        await store.close()


@pytest.mark.asyncio
async def test_generic_route_uses_its_owner_preference_and_rejects_other_principal(
    tmp_path: Path,
) -> None:
    store, service, notifications, registry = await _seed(tmp_path / "kai.db")
    try:
        daniel_authority = _authority(service, registry, 0)
        daniel = await service.inspect(daniel_authority)
        direct = next(item for item in daniel.destinations if item.kind == "direct")
        await service.select(
            daniel_authority,
            "generic",
            direct.choice_id,
            expected_revision=daniel.revision,
        )
        assert (
            await service.effective_channel_for_route(
                source="generic",
                route_name="default",
            )
            == registry.namespaces[0].channel_id
        )

        recorded = await notifications.record_for_route(
            IntegrationNotification(
                delivery_id="generic-preference-qualification",
                source="generic",
                event_type="notification",
                body="Canonical personal destination",
                occurred_at=datetime(2026, 8, 27, tzinfo=UTC),
            ),
            route_name="default",
        )
        async with store.connection.execute(
            "SELECT channel_id FROM messages WHERE id = ?",
            (recorded.message_id,),
        ) as cursor:
            assert str((await cursor.fetchone())[0]) == str(registry.namespaces[0].channel_id)

        scott_authority = _authority(service, registry, 1)
        scott = await service.inspect(scott_authority)
        scott_direct = next(item for item in scott.destinations if item.kind == "direct")
        with pytest.raises(WorkshopNotificationPreferenceAccessDenied):
            await service.select(
                scott_authority,
                "generic",
                scott_direct.choice_id,
                expected_revision=scott.revision,
            )
    finally:
        await notifications.close()
        await service.close()
        await store.close()


@pytest.mark.asyncio
async def test_concurrent_revision_checked_writes_have_one_winner(tmp_path: Path) -> None:
    store, service, notifications, registry = await _seed(tmp_path / "kai.db")
    try:
        authority = _authority(service, registry, 0)
        initial = await service.inspect(authority)
        direct = next(item for item in initial.destinations if item.kind == "direct")
        notification = next(item for item in initial.destinations if item.kind == "notification")
        results = await asyncio.gather(
            service.select(
                authority,
                "github",
                direct.choice_id,
                expected_revision=initial.revision,
            ),
            service.select(
                authority,
                "github",
                notification.choice_id,
                expected_revision=initial.revision,
            ),
            return_exceptions=True,
        )
        assert sum(not isinstance(item, Exception) for item in results) == 1
        assert sum(isinstance(item, WorkshopNotificationPreferenceConflict) for item in results) == 1
    finally:
        await notifications.close()
        await service.close()
        await store.close()
