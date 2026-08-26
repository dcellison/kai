"""Immutable receipts for retiring transport-shaped protected runtime keys."""

from __future__ import annotations

from pathlib import Path

import pytest

from kai import sessions
from kai.config import Config
from kai.workshop.bootstrap import BootstrapHuman
from kai.workshop.runtime_key_cutover import WorkshopRuntimeKeyCutoverError
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileRegistry
from tests.workshop_profiles import profile_id, profile_registry


@pytest.fixture
async def database(tmp_path: Path):
    path = tmp_path / "kai.db"
    await sessions.init_db(path)
    yield path
    await sessions.close_db()


async def _prepare(
    profiles: WorkshopRuntimeProfileRegistry,
    *,
    transport: str,
    external_subject: str,
):
    profile = profiles.profiles[0]
    await sessions.bootstrap_workshop_foundation(
        (
            BootstrapHuman(
                display_name="Runtime-key human",
                role="admin",
                transport=transport,
                external_subject=external_subject,
                external_channel_id=external_subject,
                runtime_profile_id=profile.profile_id,
            ),
        )
    )
    registry, _migration = await sessions.initialize_workshop_execution_state(profiles)
    await sessions.initialize_workshop_operational_state(
        registry,
        Config(telegram_bot_token="test", allowed_user_ids=set()),
        profiles,
    )
    return registry


async def test_archived_key_inventory_is_recorded_once(database: Path):
    profiles = profile_registry(101)
    await sessions.set_setting("model:101", "gpt-5.6-sol")
    await sessions._get_db().execute(
        "INSERT INTO sessions (chat_id, session_id, model) VALUES (101, 'legacy-session', 'gpt-5.6-sol')"
    )
    await sessions._get_db().commit()
    registry = await _prepare(profiles, transport="telegram", external_subject="101")

    first = await sessions.initialize_workshop_runtime_key_cutover(
        registry,
        memory_enabled=False,
    )
    second = await sessions.initialize_workshop_runtime_key_cutover(
        registry,
        memory_enabled=False,
    )

    assert (first.profiles, first.newly_recorded, first.archived_keys) == (1, 1, 1)
    assert (second.profiles, second.newly_recorded, second.archived_keys) == (1, 0, 1)
    async with sessions._get_db().execute(
        "SELECT legacy_runtime_key, settings_rows, session_rows, legacy_reads_disabled "
        "FROM workshop_runtime_key_cutovers WHERE runtime_profile_id = ?",
        (profile_id(101),),
    ) as cursor:
        assert tuple(await cursor.fetchone()) == (101, 1, 1, 1)


async def test_canonical_only_profile_records_no_archived_key(database: Path):
    migrated = profile_registry(101).resolve(profile_id(101))
    profiles = WorkshopRuntimeProfileRegistry((migrated,))
    registry = await _prepare(
        profiles,
        transport="workshop",
        external_subject="browser-only-human",
    )

    result = await sessions.initialize_workshop_runtime_key_cutover(
        registry,
        memory_enabled=False,
    )

    assert (result.profiles, result.newly_recorded, result.archived_keys) == (1, 1, 0)
    async with sessions._get_db().execute(
        "SELECT legacy_runtime_key, legacy_reads_disabled "
        "FROM workshop_runtime_key_cutovers WHERE runtime_profile_id = ?",
        (profile_id(101),),
    ) as cursor:
        assert tuple(await cursor.fetchone()) == (None, 1)


async def test_changed_canonical_owner_conflicts_with_receipt(database: Path):
    profiles = profile_registry(101)
    registry = await _prepare(profiles, transport="telegram", external_subject="101")
    await sessions.initialize_workshop_runtime_key_cutover(
        registry,
        memory_enabled=False,
    )
    namespace = registry.namespaces[0]
    changed = type(namespace)(
        principal_id=namespace.principal_id,
        channel_id=namespace.channel_id,
        agent_id=type(namespace.agent_id).new(),
        runtime_profile_id=namespace.runtime_profile_id,
        legacy_runtime_key=namespace.legacy_runtime_key,
    )
    changed_registry = type(registry)((changed,))

    with pytest.raises(
        WorkshopRuntimeKeyCutoverError,
        match="conflicts with current canonical ownership",
    ):
        await sessions.initialize_workshop_runtime_key_cutover(
            changed_registry,
            memory_enabled=False,
        )
