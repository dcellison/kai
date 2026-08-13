"""Canonical principal ownership for transitional Workshop storage."""

from __future__ import annotations

from pathlib import Path

import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.domain import PrincipalId
from kai.workshop.storage_namespaces import (
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
