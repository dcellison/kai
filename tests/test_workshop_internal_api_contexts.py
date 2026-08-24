"""Canonical execution contexts for protected internal API credentials."""

from pathlib import Path

import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.internal_api_contexts import (
    WorkshopInternalAPIContextError,
    WorkshopInternalAPIContextRegistry,
    WorkshopInternalAPIExecutionContext,
)
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id, profile_registry


async def _store(path: Path) -> WorkshopEventStore:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman("Alice", "admin", "telegram", "101", "101", profile_id(101)),
            BootstrapHuman("Bob", "member", "telegram", "202", "202", profile_id(202)),
        ),
    )
    return store


async def test_resolves_complete_canonical_context_for_every_profile(tmp_path: Path) -> None:
    store = await _store(tmp_path / "kai.db")
    try:
        registry = await WorkshopInternalAPIContextRegistry.from_store(
            store,
            profile_registry(101, 202),
        )
        context = registry.for_runtime_profile(profile_id(101))

        assert context.principal_id.startswith("prn_")
        assert context.channel_id.startswith("chn_")
        assert context.agent_id.startswith("agt_")
        assert context.runtime_profile_id == profile_id(101)
        assert not hasattr(context, "compatibility_runtime_config_id")
        assert not hasattr(context, "_runtime_config_id")
        assert not hasattr(registry, "for_runtime_config_id")
        assert registry.for_runtime_profile(profile_id(101)) is context
    finally:
        await store.close()


async def test_missing_profile_assignment_fails_closed(tmp_path: Path) -> None:
    store = await _store(tmp_path / "kai.db")
    try:
        with pytest.raises(
            WorkshopInternalAPIContextError,
            match="exactly one canonical internal API context",
        ):
            await WorkshopInternalAPIContextRegistry.from_store(
                store,
                profile_registry(101, 202, 303),
            )
    finally:
        await store.close()


async def test_missing_channel_agent_attachment_fails_closed(tmp_path: Path) -> None:
    store = await _store(tmp_path / "kai.db")
    try:
        await store.connection.execute(
            "DELETE FROM channel_agents WHERE agent_id = ("
            "SELECT agent_id FROM channel_agent_runtime_assignments WHERE runtime_profile_id = ?)",
            (profile_id(101),),
        )
        await store.connection.commit()
        with pytest.raises(
            WorkshopInternalAPIContextError,
            match="exactly one canonical internal API context",
        ):
            await WorkshopInternalAPIContextRegistry.from_store(
                store,
                profile_registry(101, 202),
            )
    finally:
        await store.close()


async def test_ambiguous_direct_channel_ownership_fails_closed(tmp_path: Path) -> None:
    store = await _store(tmp_path / "kai.db")
    try:
        channel_id, workshop_id = await (
            await store.connection.execute(
                "SELECT channel_id, workshop_id FROM channel_agent_runtime_assignments ra "
                "JOIN channels c ON c.id = ra.channel_id WHERE ra.runtime_profile_id = ?",
                (profile_id(101),),
            )
        ).fetchone()
        second_principal = "prn_" + "f" * 32
        await store.connection.execute(
            "INSERT INTO principals (id, kind, display_name, created_at) VALUES (?, 'human', 'Other', ?)",
            (second_principal, "2026-01-01T00:00:00Z"),
        )
        await store.connection.execute(
            "INSERT INTO workshop_memberships "
            "(id, workshop_id, principal_id, role, created_at) VALUES (?, ?, ?, 'member', ?)",
            ("wsm_" + "f" * 32, workshop_id, second_principal, "2026-01-01T00:00:00Z"),
        )
        await store.connection.execute(
            "INSERT INTO channel_memberships "
            "(id, channel_id, principal_id, role, created_at) VALUES (?, ?, ?, 'owner', ?)",
            ("cmm_" + "f" * 32, channel_id, second_principal, "2026-01-01T00:00:00Z"),
        )
        await store.connection.commit()

        with pytest.raises(
            WorkshopInternalAPIContextError,
            match="exactly one canonical internal API context",
        ):
            await WorkshopInternalAPIContextRegistry.from_store(
                store,
                profile_registry(101, 202),
            )
    finally:
        await store.close()


async def test_cross_workshop_agent_assignment_fails_closed(tmp_path: Path) -> None:
    store = await _store(tmp_path / "kai.db")
    try:
        other_workshop = "wsp_" + "f" * 32
        await store.connection.execute(
            "INSERT INTO workshops (id, name, created_at) VALUES (?, 'Other', ?)",
            (other_workshop, "2026-01-01T00:00:00Z"),
        )
        await store.connection.execute(
            "UPDATE agents SET workshop_id = ? WHERE id = ("
            "SELECT agent_id FROM channel_agent_runtime_assignments WHERE runtime_profile_id = ?)",
            (other_workshop, profile_id(101)),
        )
        await store.connection.commit()

        with pytest.raises(
            WorkshopInternalAPIContextError,
            match="exactly one canonical internal API context",
        ):
            await WorkshopInternalAPIContextRegistry.from_store(
                store,
                profile_registry(101, 202),
            )
    finally:
        await store.close()


def test_duplicate_runtime_profile_contexts_are_rejected() -> None:
    context = WorkshopInternalAPIExecutionContext.for_unprotected_runtime(
        101,
        profile_id(101),
    )
    duplicate_assignment = WorkshopInternalAPIExecutionContext.for_unprotected_runtime(
        202,
        profile_id(101),
    )

    with pytest.raises(
        WorkshopInternalAPIContextError,
        match="Duplicate internal API runtime profile",
    ):
        WorkshopInternalAPIContextRegistry((context, duplicate_assignment))
