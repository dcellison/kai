"""Canonical Workshop human provisioning contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.human_provisioning import (
    WorkshopHumanProvisioner,
    WorkshopHumanProvisioningError,
)
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id


async def _store(path: Path) -> WorkshopEventStore:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (BootstrapHuman("Alice", "admin", "telegram", "101", "101", profile_id(101)),),
    )
    return store


class TestWorkshopHumanProvisioner:
    async def test_provisions_complete_collaboration_identity_without_external_authority(
        self,
        tmp_path: Path,
    ):
        store = await _store(tmp_path / "kai.db")
        try:
            provisioned = await WorkshopHumanProvisioner(store).provision(
                "charlie",
                "  Charlie  ",
                "member",
                handle="charlie_ops",
            )

            assert provisioned.handle == "charlie_ops"

            async with store.connection.execute(
                "SELECT p.display_name, hh.handle, wm.role "
                "FROM principals p "
                "JOIN workshop_memberships wm ON wm.principal_id = p.id "
                "JOIN human_handles hh ON hh.workshop_id = wm.workshop_id "
                "AND hh.principal_id = p.id "
                "WHERE p.id = ?",
                (provisioned.principal_id,),
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            assert tuple(row) == ("Charlie", "charlie_ops", "member")

            async with store.connection.execute(
                "SELECT COUNT(*) FROM external_identities WHERE principal_id = ?",
                (provisioned.principal_id,),
            ) as cursor:
                assert (await cursor.fetchone())[0] == 0
            async with store.connection.execute(
                "SELECT COUNT(*) FROM channel_memberships WHERE principal_id = ?",
                (provisioned.principal_id,),
            ) as cursor:
                assert (await cursor.fetchone())[0] == 0
            async with store.connection.execute(
                "SELECT COUNT(*) FROM principal_agent_enablements WHERE principal_id = ?",
                (provisioned.principal_id,),
            ) as cursor:
                assert (await cursor.fetchone())[0] == 0
            async with store.connection.execute(
                "SELECT COUNT(*) FROM channel_agents ca JOIN channel_memberships cm "
                "ON cm.channel_id = ca.channel_id WHERE cm.principal_id = ?",
                (provisioned.principal_id,),
            ) as cursor:
                assert (await cursor.fetchone())[0] == 0
            async with store.connection.execute(
                "SELECT COUNT(*) FROM event_log WHERE idempotency_key LIKE ? "
                "AND json_extract(metadata_json, '$.source') = 'operator_cli'",
                (f"operator:human-provisioning:{provisioned.principal_id}:%",),
            ) as cursor:
                assert (await cursor.fetchone())[0] == 2
        finally:
            await store.close()

    async def test_provisioned_human_has_no_implicit_agent_or_runtime_authority(
        self,
        tmp_path: Path,
    ):
        store = await _store(tmp_path / "kai.db")
        try:
            provisioned = await WorkshopHumanProvisioner(store).provision(
                "charlie",
                "Charlie",
                "member",
            )
            async with store.connection.execute(
                "SELECT COUNT(*) FROM channels c JOIN channel_memberships cm "
                "ON cm.channel_id = c.id WHERE cm.principal_id = ?",
                (provisioned.principal_id,),
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 0
            async with store.connection.execute(
                "SELECT COUNT(*) FROM channel_agent_runtime_assignments ra "
                "JOIN channel_memberships cm ON cm.channel_id = ra.channel_id "
                "WHERE cm.principal_id = ?",
                (provisioned.principal_id,),
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 0
        finally:
            await store.close()

    async def test_same_provisioning_key_is_a_safe_semantic_retry(self, tmp_path: Path):
        store = await _store(tmp_path / "kai.db")
        try:
            provisioner = WorkshopHumanProvisioner(store)
            first = await provisioner.provision("charlie", "Charlie", "member")
            async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
                after_first = (await cursor.fetchone())[0]

            retried = await provisioner.provision("charlie", "Charlie", "member")

            assert first.created is True
            assert retried.created is False
            assert retried.principal_id == first.principal_id
            async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
                assert (await cursor.fetchone())[0] == after_first

            with pytest.raises(WorkshopHumanProvisioningError, match="different human"):
                await provisioner.provision("charlie", "Charles", "member")
            async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
                assert (await cursor.fetchone())[0] == after_first
        finally:
            await store.close()

    async def test_shared_human_and_agent_handle_namespace_fails_closed(self, tmp_path: Path):
        store = await _store(tmp_path / "kai.db")
        try:
            async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
                before = int((await cursor.fetchone())[0])
            with pytest.raises(WorkshopHumanProvisioningError, match="already used"):
                await WorkshopHumanProvisioner(store).provision(
                    "another-kai",
                    "Another Kai",
                    "member",
                    handle="kai",
                )
            with pytest.raises(WorkshopHumanProvisioningError, match="already used"):
                await WorkshopHumanProvisioner(store).provision(
                    "another-alice",
                    "Another Alice",
                    "member",
                    handle="alice",
                )
            async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
                assert int((await cursor.fetchone())[0]) == before
        finally:
            await store.close()

    async def test_identity_provisioning_does_not_depend_on_a_kai_agent(
        self,
        tmp_path: Path,
    ):
        store = await _store(tmp_path / "kai.db")
        try:
            await store.connection.execute("DELETE FROM channel_agents")
            await store.connection.execute("DELETE FROM agents")
            await store.connection.commit()
            provisioned = await WorkshopHumanProvisioner(store).provision(
                "charlie",
                "Charlie",
                "member",
            )
            async with store.connection.execute(
                "SELECT display_name FROM principals WHERE id = ?",
                (provisioned.principal_id,),
            ) as cursor:
                assert tuple(await cursor.fetchone()) == ("Charlie",)
        finally:
            await store.close()

    @pytest.mark.parametrize(
        ("display_name", "role", "match"),
        [
            ("", "member", "Display name"),
            ("x" * 201, "member", "Display name"),
            ("Charlie", "owner", "Role"),
        ],
    )
    async def test_invalid_operator_input_creates_no_events(
        self,
        tmp_path: Path,
        display_name: str,
        role: str,
        match: str,
    ):
        store = await _store(tmp_path / "kai.db")
        try:
            async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
                before = (await cursor.fetchone())[0]
            with pytest.raises(WorkshopHumanProvisioningError, match=match):
                await WorkshopHumanProvisioner(store).provision(
                    "charlie",
                    display_name,
                    role,
                )
            async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
                assert (await cursor.fetchone())[0] == before
        finally:
            await store.close()

    @pytest.mark.parametrize("provisioning_key", ["", "Charlie", "two words", "x" * 65])
    async def test_invalid_provisioning_key_creates_no_events(
        self,
        tmp_path: Path,
        provisioning_key: str,
    ):
        store = await _store(tmp_path / "kai.db")
        try:
            async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
                before = (await cursor.fetchone())[0]
            with pytest.raises(WorkshopHumanProvisioningError, match="Provisioning key"):
                await WorkshopHumanProvisioner(store).provision(
                    provisioning_key,
                    "Charlie",
                    "member",
                )
            async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
                assert (await cursor.fetchone())[0] == before
        finally:
            await store.close()
