"""Canonical Workshop appearance preference contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from kai.workshop.appearance_preferences import (
    DEFAULT_WORKSHOP_THEME,
    WORKSHOP_APPEARANCE_THEMES,
    WorkshopAppearancePreferenceAccessDenied,
    WorkshopAppearancePreferenceConflict,
    WorkshopAppearancePreferenceService,
    WorkshopAppearancePreferenceValidationError,
)
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.diagnostics import workshop_appearance_preference_status
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id


async def _seed(path: Path) -> tuple[WorkshopEventStore, str, str]:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman("Daniel", "admin", "telegram", "101", "101", profile_id(101)),
            BootstrapHuman("Scott", "member", "telegram", "202", "202", profile_id(202)),
        ),
    )
    async with store.connection.execute(
        "SELECT external_subject, principal_id FROM external_identities "
        "WHERE provider = 'telegram' ORDER BY external_subject"
    ) as cursor:
        principals = {str(row[0]): str(row[1]) for row in await cursor.fetchall()}
    return store, principals["101"], principals["202"]


@pytest.mark.asyncio
async def test_version_thirty_seven_preferences_upgrade_without_data_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kai.workshop import schema

    path = tmp_path / "kai.db"
    principal_id = "prn_00000000000000000000000000000001"
    with monkeypatch.context() as migration_context:
        migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 37)
        migration_context.setattr(schema, "_MIGRATIONS", schema._MIGRATIONS[:37])
        version_thirty_seven = await WorkshopEventStore.open(path)
        await version_thirty_seven.connection.execute(
            "INSERT INTO principals (id, kind, display_name, created_at) "
            "VALUES (?, 'human', 'Daniel', '2026-08-27T12:00:00Z')",
            (principal_id,),
        )
        await version_thirty_seven.connection.execute(
            "INSERT INTO principal_appearance_preferences "
            "(principal_id, theme_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (
                principal_id,
                "github-dark-dimmed",
                "2026-08-27T12:01:00Z",
                "2026-08-27T12:02:00Z",
            ),
        )
        await version_thirty_seven.connection.commit()
        await version_thirty_seven.close()

    upgraded = await WorkshopEventStore.open(path)
    try:
        assert await upgraded.schema_version() == 52
        async with upgraded.connection.execute(
            "SELECT principal_id, theme_id, created_at, updated_at FROM principal_appearance_preferences"
        ) as cursor:
            assert tuple(await cursor.fetchone()) == (
                principal_id,
                "github-dark-dimmed",
                "2026-08-27T12:01:00Z",
                "2026-08-27T12:02:00Z",
            )
        async with upgraded.connection.execute("PRAGMA foreign_key_list(principal_appearance_preferences)") as cursor:
            assert await cursor.fetchall() == []
    finally:
        await upgraded.close()


@pytest.mark.asyncio
async def test_defaults_are_principal_scoped_and_survive_restart(tmp_path: Path) -> None:
    path = tmp_path / "kai.db"
    store, daniel_id, scott_id = await _seed(path)
    service = WorkshopAppearancePreferenceService(store.connection)
    daniel = service.authority_for_principal(daniel_id)
    scott = service.authority_for_principal(scott_id)

    daniel_snapshot = await service.inspect(daniel)
    scott_snapshot = await service.inspect(scott)
    assert daniel_snapshot.theme_id == DEFAULT_WORKSHOP_THEME
    assert scott_snapshot.theme_id == DEFAULT_WORKSHOP_THEME
    assert daniel_snapshot.revision != scott_snapshot.revision
    assert [item.theme_id for item in daniel_snapshot.themes] == [
        "atom-one-dark",
        "atom-one-light",
        "dracula",
        "nord",
        "solarized-dark",
        "solarized-light",
        "catppuccin-mocha",
        "catppuccin-latte",
        "github-light-default",
        "github-dark-default",
        "github-dark-dimmed",
    ]
    assert workshop_appearance_preference_status(path) == (
        "Workshop appearance preferences: active; principals=2, explicit=0, defaulted=2, invalid=0; authority=canonical"
    )

    unchanged = await service.set_theme(
        daniel,
        DEFAULT_WORKSHOP_THEME,
        expected_revision=daniel_snapshot.revision,
    )
    assert unchanged.mutation is not None and unchanged.mutation.changed is False
    with pytest.raises(WorkshopAppearancePreferenceConflict):
        await service.set_theme(
            scott,
            DEFAULT_WORKSHOP_THEME,
            expected_revision=daniel_snapshot.revision,
        )
    await store.close()

    reopened = await WorkshopAppearancePreferenceService.open(path)
    try:
        assert (await reopened.inspect(reopened.authority_for_principal(daniel_id))).theme_id == DEFAULT_WORKSHOP_THEME
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_non_default_light_and_dark_themes_are_isolated_and_survive_restart(tmp_path: Path) -> None:
    path = tmp_path / "kai.db"
    store, daniel_id, scott_id = await _seed(path)
    service = WorkshopAppearancePreferenceService(store.connection)
    daniel = service.authority_for_principal(daniel_id)
    scott = service.authority_for_principal(scott_id)

    daniel_before = await service.inspect(daniel)
    scott_before = await service.inspect(scott)
    daniel_changed = await service.set_theme(
        daniel,
        "github-light-default",
        expected_revision=daniel_before.revision,
    )
    scott_changed = await service.set_theme(
        scott,
        "catppuccin-mocha",
        expected_revision=scott_before.revision,
    )
    assert daniel_changed.theme_id == "github-light-default"
    assert scott_changed.theme_id == "catppuccin-mocha"
    assert len(WORKSHOP_APPEARANCE_THEMES) == 11
    await store.close()

    # Application startup bootstraps canonical conversations by deleting and
    # replaying the collaboration projection. Appearance state must survive
    # that full lifecycle, not merely a database close/reopen.
    restarted_store, restarted_daniel_id, restarted_scott_id = await _seed(path)
    assert restarted_daniel_id == daniel_id
    assert restarted_scott_id == scott_id
    reopened = WorkshopAppearancePreferenceService(restarted_store.connection)
    try:
        assert (await reopened.inspect(reopened.authority_for_principal(daniel_id))).theme_id == "github-light-default"
        assert (await reopened.inspect(reopened.authority_for_principal(scott_id))).theme_id == "catppuccin-mocha"
        assert workshop_appearance_preference_status(path) == (
            "Workshop appearance preferences: active; principals=2, explicit=2, "
            "defaulted=0, invalid=0; authority=canonical"
        )
    finally:
        await restarted_store.close()


@pytest.mark.asyncio
async def test_unknown_and_corrupt_themes_fail_safe_to_atom_one_dark(tmp_path: Path) -> None:
    store, daniel_id, _ = await _seed(tmp_path / "kai.db")
    service = WorkshopAppearancePreferenceService(store.connection)
    authority = service.authority_for_principal(daniel_id)
    initial = await service.inspect(authority)

    with pytest.raises(WorkshopAppearancePreferenceValidationError):
        await service.set_theme(authority, "../../custom.css", expected_revision=initial.revision)

    await store.connection.execute(
        "INSERT INTO principal_appearance_preferences (principal_id, theme_id) VALUES (?, ?)",
        (daniel_id, "removed-theme"),
    )
    await store.connection.commit()
    corrupt = await service.inspect(authority)
    assert corrupt.theme_id == DEFAULT_WORKSHOP_THEME
    repaired = await service.set_theme(
        authority,
        DEFAULT_WORKSHOP_THEME,
        expected_revision=corrupt.revision,
    )
    assert repaired.theme_id == DEFAULT_WORKSHOP_THEME
    assert repaired.mutation is not None and repaired.mutation.changed is True
    await store.close()


@pytest.mark.asyncio
async def test_non_human_principals_cannot_own_appearance_preferences(tmp_path: Path) -> None:
    store, _, _ = await _seed(tmp_path / "kai.db")
    async with store.connection.execute("SELECT id FROM principals WHERE kind = 'agent' LIMIT 1") as cursor:
        agent_id = str((await cursor.fetchone())[0])
    service = WorkshopAppearancePreferenceService(store.connection)
    with pytest.raises(WorkshopAppearancePreferenceAccessDenied):
        await service.inspect(service.authority_for_principal(agent_id))
    await store.close()
