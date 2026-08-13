"""Canonical principal ownership for transitional Workshop storage."""

from __future__ import annotations

from pathlib import Path

import pytest

from kai.install import _canonical_storage_reader_users
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.domain import ChannelId, PrincipalId
from kai.workshop.storage_namespaces import (
    WorkshopChannelHistoryNamespace,
    WorkshopChannelHistoryRegistry,
    WorkshopPrincipalStorageNamespace,
    WorkshopPrincipalStorageRegistry,
    WorkshopStorageNamespaceError,
)
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id, profile_registry


async def _store(path: Path) -> WorkshopEventStore:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                "Alice",
                "admin",
                "telegram",
                "101",
                "101",
                profile_id(101),
            ),
            BootstrapHuman(
                "Bob",
                "member",
                "telegram",
                "202",
                "202",
                profile_id(202),
            ),
        ),
    )
    return store


async def _principal_for_telegram_subject(
    store: WorkshopEventStore,
    subject: str,
) -> PrincipalId:
    async with store.connection.execute(
        "SELECT principal_id FROM external_identities WHERE provider = 'telegram' AND external_subject = ?",
        (subject,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return PrincipalId(str(row[0]))


async def _channel_for_telegram_subject(
    store: WorkshopEventStore,
    subject: str,
) -> ChannelId:
    async with store.connection.execute(
        "SELECT channel_id FROM channel_bindings WHERE transport = 'telegram' AND external_channel_id = ?",
        (subject,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return ChannelId(str(row[0]))


class TestWorkshopPrincipalStorageRegistry:
    async def test_resolves_profiles_to_canonical_human_principals(
        self,
        tmp_path: Path,
    ):
        store = await _store(tmp_path / "kai.db")
        try:
            registry = await WorkshopPrincipalStorageRegistry.from_store(
                store,
                profile_registry(101, 202),
            )
            alice = registry.for_runtime_config_id(101)

            assert alice.principal_id == await _principal_for_telegram_subject(
                store,
                "101",
            )
            assert registry.for_runtime_profile(profile_id(101)) is alice
            assert alice.files_directory(tmp_path) == (tmp_path / "files" / str(alice.principal_id))
            assert alice.legacy_files_directory(tmp_path) == (tmp_path / "files" / "101")
            assert alice.files_directory(tmp_path).name != "101"
            assert alice.home_directory(tmp_path) == (tmp_path / "home" / str(alice.principal_id))
            assert alice.memory_directory(tmp_path) == (tmp_path / "memory" / str(alice.principal_id))
            assert alice.preferences_directory(tmp_path) == (tmp_path / "preferences" / str(alice.principal_id))
        finally:
            await store.close()

    async def test_missing_canonical_owner_fails_closed(self, tmp_path: Path):
        store = await _store(tmp_path / "kai.db")
        try:
            with pytest.raises(
                WorkshopStorageNamespaceError,
                match="exactly one canonical human",
            ):
                await WorkshopPrincipalStorageRegistry.from_store(
                    store,
                    profile_registry(101, 202, 303),
                )
        finally:
            await store.close()

    def test_duplicate_runtime_configuration_is_rejected(self):
        with pytest.raises(
            WorkshopStorageNamespaceError,
            match="Duplicate runtime configuration",
        ):
            WorkshopPrincipalStorageRegistry(
                (
                    WorkshopPrincipalStorageNamespace(
                        PrincipalId("prn_" + "1" * 32),
                        profile_id(101),
                        101,
                    ),
                    WorkshopPrincipalStorageNamespace(
                        PrincipalId("prn_" + "2" * 32),
                        profile_id(202),
                        101,
                    ),
                )
            )

    def test_unknown_runtime_configuration_is_rejected(self):
        registry = WorkshopPrincipalStorageRegistry(
            (
                WorkshopPrincipalStorageNamespace(
                    PrincipalId("prn_" + "1" * 32),
                    profile_id(101),
                    101,
                ),
            )
        )

        with pytest.raises(
            WorkshopStorageNamespaceError,
            match="no canonical principal",
        ):
            registry.for_runtime_config_id(202)


class TestWorkshopChannelHistoryRegistry:
    async def test_resolves_transport_and_runtime_keys_to_canonical_channel(
        self,
        tmp_path: Path,
    ):
        store = await _store(tmp_path / "kai.db")
        try:
            registry = await WorkshopChannelHistoryRegistry.from_store(
                store,
                profile_registry(101, 202),
            )
            expected_channel = await _channel_for_telegram_subject(store, "101")
            namespace = registry.for_compatibility_chat_id(101)

            assert namespace.channel_id == expected_channel
            assert namespace.history_directory(tmp_path) == (tmp_path / "history" / str(expected_channel))
            assert namespace.legacy_history_directory(tmp_path) == (tmp_path / "history" / "101")
        finally:
            await store.close()

    def test_unknown_compatibility_chat_fails_closed(self):
        registry = WorkshopChannelHistoryRegistry(
            (
                WorkshopChannelHistoryNamespace(
                    ChannelId("chn_" + "1" * 32),
                    101,
                ),
            )
        )

        with pytest.raises(
            WorkshopStorageNamespaceError,
            match="no canonical history channel",
        ):
            registry.for_compatibility_chat_id(202)

    async def test_installer_maps_canonical_principal_and_channel_to_os_reader(
        self,
        tmp_path: Path,
    ):
        data_path = tmp_path / "data"
        data_path.mkdir()
        store = await _store(data_path / "kai.db")
        principal_id = await _principal_for_telegram_subject(store, "101")
        channel_id = await _channel_for_telegram_subject(store, "101")
        await store.close()

        readers = _canonical_storage_reader_users(
            data_path,
            {"101": "daniel", "202": "scott"},
        )

        assert readers[str(principal_id)] == "daniel"
        assert readers[str(channel_id)] == "daniel"
