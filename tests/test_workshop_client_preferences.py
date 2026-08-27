"""Canonical client-binding voice preference contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.client_preferences import (
    VOICE_MODE_OFF,
    VOICE_MODE_TEXT_AND_VOICE,
    VOICE_MODE_VOICE_ONLY,
    ClientVoiceCapability,
    WorkshopClientPreferenceAccessDenied,
    WorkshopClientPreferenceConflict,
    WorkshopClientPreferenceService,
    WorkshopClientPreferenceValidationError,
)
from kai.workshop.diagnostics import workshop_client_preference_status
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id

_CAPABILITY = ClientVoiceCapability(
    transport="telegram",
    display_name="Telegram",
    enabled=True,
    voices=(("alan", "Alan"), ("jenny", "Jenny")),
    default_voice="alan",
)


async def _seed(path: Path):
    store = await WorkshopEventStore.open(path)
    await store.connection.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman("Daniel", "admin", "telegram", "101", "101", profile_id(101)),
            BootstrapHuman("Scott", "member", "telegram", "202", "202", profile_id(202)),
        ),
    )
    return store


@pytest.mark.asyncio
async def test_migrates_legacy_voice_state_without_exposing_transport_identity(tmp_path: Path) -> None:
    store = await _seed(tmp_path / "kai.db")
    await store.connection.executemany(
        "INSERT INTO settings (key, value) VALUES (?, ?)",
        (("voice_mode:101", "on"), ("voice_name:101", "jenny")),
    )
    await store.connection.commit()
    service = WorkshopClientPreferenceService(store.connection, (_CAPABILITY,))
    try:
        await service.reconcile_legacy_preferences()
        authority = await service.authority_for_transport_binding(
            transport="telegram",
            external_subject="101",
            external_channel_id="101",
        )
        preference = await service.inspect_binding(authority)
        snapshot = await service.inspect(service.authority_for_principal(authority.principal_id))

        assert preference.mode == VOICE_MODE_TEXT_AND_VOICE
        assert preference.voice == "jenny"
        assert preference.choice_id.startswith("cbd_")
        assert "101" not in repr(snapshot)
        assert workshop_client_preference_status(tmp_path / "kai.db").startswith(
            "Workshop client preferences: active; eligible bindings=2, preferences=2, migrations=2"
        )
        async with store.connection.execute(
            "SELECT mode_migrated, voice_migrated, legacy_reads_disabled, rollback_dual_writes "
            "FROM client_binding_voice_migrations WHERE channel_binding_id = ?",
            (authority.binding_id,),
        ) as cursor:
            assert tuple(await cursor.fetchone()) == (1, 1, 1, 1)

        await store.connection.executemany(
            "UPDATE settings SET value = ? WHERE key = ?",
            (("off", "voice_mode:101"), ("alan", "voice_name:101")),
        )
        await store.connection.commit()
        await service.reconcile_legacy_preferences()
        unchanged = await service.inspect_binding(authority)
        assert unchanged.mode == VOICE_MODE_TEXT_AND_VOICE
        assert unchanged.voice == "jenny"
        async with store.connection.execute("SELECT COUNT(*) FROM client_binding_voice_migrations") as cursor:
            assert int((await cursor.fetchone())[0]) == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_mutation_is_live_revision_checked_and_dual_writes_for_rollback(tmp_path: Path) -> None:
    path = tmp_path / "kai.db"
    store = await _seed(path)
    service = WorkshopClientPreferenceService(store.connection, (_CAPABILITY,))
    await service.reconcile_legacy_preferences()
    authority = await service.authority_for_transport_binding(
        transport="telegram",
        external_subject="101",
        external_channel_id="101",
    )
    principal = service.authority_for_principal(authority.principal_id)
    initial = await service.inspect(principal)
    changed = await service.set_mode(
        authority,
        VOICE_MODE_VOICE_ONLY,
        expected_revision=initial.revision,
    )
    assert changed.bindings[0].mode == VOICE_MODE_VOICE_ONLY
    assert changed.mutation is not None and changed.mutation.changed is True
    with pytest.raises(WorkshopClientPreferenceConflict):
        await service.set_voice(
            authority,
            "jenny",
            expected_revision=initial.revision,
        )
    changed = await service.set_voice(
        authority,
        "jenny",
        expected_revision=changed.revision,
    )
    assert changed.bindings[0].voice == "jenny"
    with pytest.raises(WorkshopClientPreferenceValidationError):
        await service.set_mode(authority, "telegram_only")
    with pytest.raises(WorkshopClientPreferenceValidationError):
        await service.set_voice(authority, "../../voice-model.onnx")
    async with store.connection.execute(
        "SELECT key, value FROM settings WHERE key IN ('voice_mode:101', 'voice_name:101') ORDER BY key"
    ) as cursor:
        assert [tuple(row) for row in await cursor.fetchall()] == [
            ("voice_mode:101", "only"),
            ("voice_name:101", "jenny"),
        ]
    await store.close()

    reopened = await WorkshopClientPreferenceService.open(path, (_CAPABILITY,))
    try:
        restarted = await reopened.inspect(reopened.authority_for_principal(authority.principal_id))
        assert restarted.bindings[0].mode == VOICE_MODE_VOICE_ONLY
        assert restarted.bindings[0].voice == "jenny"
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_cross_principal_and_forged_binding_choices_fail_closed(tmp_path: Path) -> None:
    store = await _seed(tmp_path / "kai.db")
    service = WorkshopClientPreferenceService(store.connection, (_CAPABILITY,))
    try:
        await service.reconcile_legacy_preferences()
        daniel = await service.authority_for_transport_binding(
            transport="telegram",
            external_subject="101",
            external_channel_id="101",
        )
        scott = await service.authority_for_transport_binding(
            transport="telegram",
            external_subject="202",
            external_channel_id="202",
        )
        daniel_snapshot = await service.inspect(service.authority_for_principal(daniel.principal_id))
        scott_snapshot = await service.inspect(service.authority_for_principal(scott.principal_id))

        with pytest.raises(WorkshopClientPreferenceAccessDenied):
            await service.set_choice_mode(
                service.authority_for_principal(daniel.principal_id),
                scott_snapshot.bindings[0].choice_id,
                VOICE_MODE_OFF,
                expected_revision=daniel_snapshot.revision,
            )
        with pytest.raises(WorkshopClientPreferenceAccessDenied):
            await service.authority_for_transport_binding(
                transport="telegram",
                external_subject="101",
                external_channel_id="202",
            )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_disabled_capability_is_visible_but_not_editable(tmp_path: Path) -> None:
    store = await _seed(tmp_path / "kai.db")
    disabled = ClientVoiceCapability(
        transport="telegram",
        display_name="Telegram",
        enabled=False,
        voices=_CAPABILITY.voices,
        default_voice=_CAPABILITY.default_voice,
    )
    service = WorkshopClientPreferenceService(store.connection, (disabled,))
    try:
        await service.reconcile_legacy_preferences()
        async with store.connection.execute(
            "SELECT principal_id FROM external_identities WHERE provider = 'telegram' AND external_subject = '101'"
        ) as cursor:
            principal_id = str((await cursor.fetchone())[0])
        snapshot = await service.inspect(service.authority_for_principal(principal_id))
        assert snapshot.available is False
        assert snapshot.unavailable_reason is not None
        assert snapshot.bindings[0].editable is False
        assert workshop_client_preference_status(
            tmp_path / "kai.db",
            telegram_enabled=True,
            tts_enabled=False,
        ).startswith("Workshop client preferences: active;")
        assert workshop_client_preference_status(
            tmp_path / "kai.db",
            telegram_enabled=False,
            tts_enabled=False,
        ).startswith("Workshop client preferences: inactive;")
    finally:
        await store.close()
