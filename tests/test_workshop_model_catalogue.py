from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from kai.backend_registry import BackendRegistryEntry
from kai.workshop.domain import AgentId, ChannelId, PrincipalId, RuntimeProfileId
from kai.workshop.execution_state import (
    WorkshopExecutionStateNamespace,
    WorkshopExecutionStateRegistry,
)
from kai.workshop.model_catalogue import (
    ModelCatalogueAccessDenied,
    ModelCatalogueEntryStatus,
    ModelCatalogueOperatorAuthority,
    ModelCatalogueRefreshStatus,
    ModelCatalogueValidationError,
    ModelDiscoveryBatch,
    ModelDiscoveryCandidate,
    ModelDiscoveryUnsupported,
    WorkshopModelCatalogueService,
)
from kai.workshop.model_discovery_inventory import WorkshopModelDiscoveryInventoryService
from kai.workshop.runtime_profiles import (
    ProtectedRuntimeBackend,
    ProtectedRuntimeProfile,
    WorkshopRuntimeProfileRegistry,
)


def _id(identifier_type, value: int):
    return identifier_type(f"{identifier_type.prefix}_{value:032x}")


def _profile(value: int, *, allowed_models: tuple[str, ...] | None = None) -> ProtectedRuntimeProfile:
    option = ProtectedRuntimeBackend(
        "claude",
        "anthropic",
        "default-model",
        allowed_models=allowed_models,
    )
    return ProtectedRuntimeProfile(
        profile_id=_id(RuntimeProfileId, value),
        display_name=f"Profile {value}",
        os_user="daniel",
        backend=option.backend,
        provider=option.provider,
        model=option.model,
        timeout_seconds=300,
        allowed_services=(),
        home_workspace=None,
        workspace_base=None,
        allowed_workspaces=(),
        allowed_models=allowed_models,
        backend_options=(option,),
    )


def _namespace(value: int) -> WorkshopExecutionStateNamespace:
    return WorkshopExecutionStateNamespace(
        principal_id=_id(PrincipalId, value),
        channel_id=_id(ChannelId, value),
        agent_id=_id(AgentId, value),
        runtime_profile_id=_id(RuntimeProfileId, value),
        legacy_runtime_key=None,
    )


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o700)
    return path


@dataclass
class _Fixture:
    database: Path
    inventory: WorkshopModelDiscoveryInventoryService
    selected_models: dict[RuntimeProfileId, str]
    executable: Path

    async def selected_model(self, profile_id: RuntimeProfileId) -> str:
        return self.selected_models[profile_id]


def _fixture(tmp_path: Path, *, count: int = 1, allowed_models: tuple[str, ...] | None = None) -> _Fixture:
    profiles = tuple(_profile(value, allowed_models=allowed_models) for value in range(1, count + 1))
    namespaces = tuple(_namespace(value) for value in range(1, count + 1))
    executable = _executable(tmp_path / "claude")
    inventory = WorkshopModelDiscoveryInventoryService(
        config=SimpleNamespace(codex_auth_mode="subscription"),  # type: ignore[arg-type]
        runtime_profiles=WorkshopRuntimeProfileRegistry(profiles),
        execution_state=WorkshopExecutionStateRegistry(namespaces),
        backend_registry={
            "claude": BackendRegistryEntry(
                id="claude",
                driver="claude",
                runtime="local_process",
                command=str(executable),
            )
        },
        selected_backend=lambda _profile_id: ("claude", "anthropic"),
        service_os_user="kai",
        environment={},
    )
    return _Fixture(
        tmp_path / "kai.db",
        inventory,
        {profile.profile_id: "selected-model" for profile in profiles},
        executable,
    )


class _QueueAdapter:
    def __init__(self, results: list[object]) -> None:
        self.results = results
        self.calls = 0
        self.active = 0
        self.maximum_active = 0

    async def discover(self, _lane) -> ModelDiscoveryBatch:
        self.calls += 1
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        await asyncio.sleep(0.01)
        try:
            result = self.results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result  # type: ignore[return-value]
        finally:
            self.active -= 1


class _BlockingAdapter:
    def __init__(self, batch: ModelDiscoveryBatch) -> None:
        self.batch = batch
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def discover(self, _lane) -> ModelDiscoveryBatch:
        self.started.set()
        await self.release.wait()
        return self.batch


class _SleepingAdapter:
    async def discover(self, _lane) -> ModelDiscoveryBatch:
        await asyncio.sleep(60)
        raise AssertionError("unreachable")


def _batch(*models: str, source: str = "fixture") -> ModelDiscoveryBatch:
    return ModelDiscoveryBatch(
        source,
        tuple(ModelDiscoveryCandidate(model, model.replace("-", " ").title(), {"tools": True}) for model in models),
        ttl_seconds=3600,
    )


async def _open(
    fixture: _Fixture,
    *,
    adapter=None,
    curated: Mapping[str, str] | None = None,
) -> WorkshopModelCatalogueService:
    return await WorkshopModelCatalogueService.open(
        fixture.database,
        fixture.inventory,
        selected_model=fixture.selected_model,
        curated_models=lambda _lane: curated,
        adapters={"claude": adapter} if adapter is not None else None,
    )


def _entry(snapshot, model_id: str):
    return next(item for item in snapshot.entries if item.model_id == model_id)


async def test_refresh_tracks_additions_removals_and_retained_models(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    adapter = _QueueAdapter([_batch("alpha", "selected-model"), _batch("beta")])
    service = await _open(fixture, adapter=adapter, curated={"curated": "Curated Model"})
    authority = service.authority_for_principal(_id(PrincipalId, 1))
    try:
        first = await service.refresh(authority, _id(RuntimeProfileId, 1), "claude:anthropic")
        first_snapshot = await service.inspect(authority, _id(RuntimeProfileId, 1), "claude:anthropic")
        second = await service.refresh(authority, _id(RuntimeProfileId, 1), "claude:anthropic")
        second_snapshot = await service.inspect(authority, _id(RuntimeProfileId, 1), "claude:anthropic")
    finally:
        await service.close()

    assert first.status == ModelCatalogueRefreshStatus.SUCCEEDED
    assert first.discovered_models == 2
    assert _entry(first_snapshot, "alpha").status == ModelCatalogueEntryStatus.AVAILABLE
    assert _entry(first_snapshot, "curated").status == ModelCatalogueEntryStatus.AVAILABLE
    assert _entry(first_snapshot, "default-model").status == ModelCatalogueEntryStatus.NOT_ADVERTISED
    assert second.status == ModelCatalogueRefreshStatus.SUCCEEDED
    assert _entry(second_snapshot, "beta").status == ModelCatalogueEntryStatus.AVAILABLE
    assert _entry(second_snapshot, "alpha").status == ModelCatalogueEntryStatus.NOT_ADVERTISED
    retained = _entry(second_snapshot, "selected-model")
    assert retained.retained is True
    assert retained.status == ModelCatalogueEntryStatus.NOT_ADVERTISED
    assert retained.selectable is False


async def test_failures_preserve_last_known_good_and_sanitize_diagnostics(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    adapter = _QueueAdapter(
        [
            _batch("known-good"),
            object(),
            RuntimeError("secret-bearing raw response"),
            ModelDiscoveryUnsupported(),
        ]
    )
    service = await _open(fixture, adapter=adapter)
    authority = service.authority_for_principal(_id(PrincipalId, 1))
    try:
        assert (
            await service.refresh(authority, _id(RuntimeProfileId, 1), "claude:anthropic")
        ).status == ModelCatalogueRefreshStatus.SUCCEEDED
        malformed = await service.refresh(authority, _id(RuntimeProfileId, 1), "claude:anthropic")
        failed = await service.refresh(authority, _id(RuntimeProfileId, 1), "claude:anthropic")
        unsupported = await service.refresh(authority, _id(RuntimeProfileId, 1), "claude:anthropic")
        snapshot = await service.inspect(authority, _id(RuntimeProfileId, 1), "claude:anthropic")
    finally:
        await service.close()

    assert malformed.status == ModelCatalogueRefreshStatus.MALFORMED
    assert failed.status == ModelCatalogueRefreshStatus.FAILED
    assert unsupported.status == ModelCatalogueRefreshStatus.UNSUPPORTED
    assert all(result.preserved_last_known_good for result in (malformed, failed, unsupported))
    assert _entry(snapshot, "known-good").status == ModelCatalogueEntryStatus.AVAILABLE
    assert snapshot.stale is True
    assert snapshot.refresh is not None
    assert snapshot.refresh.error_detail == "This backend context does not support model enumeration"
    assert "secret-bearing" not in repr(snapshot)
    raw_database = fixture.database.read_bytes()
    assert b"secret-bearing" not in raw_database


async def test_timeout_and_missing_adapter_preserve_catalogue(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    service = await _open(fixture, adapter=_QueueAdapter([_batch("known-good")]))
    authority = service.authority_for_principal(_id(PrincipalId, 1))
    try:
        await service.refresh(authority, _id(RuntimeProfileId, 1), "claude:anthropic")
        service.register_adapter("claude", _SleepingAdapter())
        timed_out = await service.refresh(
            authority,
            _id(RuntimeProfileId, 1),
            "claude:anthropic",
            timeout_seconds=0.01,
        )
    finally:
        await service.close()
    unsupported_service = await _open(fixture)
    unsupported_authority = unsupported_service.authority_for_principal(_id(PrincipalId, 1))
    try:
        unsupported = await unsupported_service.refresh(
            unsupported_authority,
            _id(RuntimeProfileId, 1),
            "claude:anthropic",
        )
        snapshot = await unsupported_service.inspect(
            unsupported_authority,
            _id(RuntimeProfileId, 1),
            "claude:anthropic",
        )
    finally:
        await unsupported_service.close()

    assert timed_out.status == ModelCatalogueRefreshStatus.TIMED_OUT
    assert timed_out.preserved_last_known_good is True
    assert unsupported.status == ModelCatalogueRefreshStatus.UNSUPPORTED
    assert _entry(snapshot, "known-good").status == ModelCatalogueEntryStatus.AVAILABLE


async def test_same_process_concurrent_refreshes_are_serialized(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    adapter = _QueueAdapter([_batch("first"), _batch("second")])
    service = await _open(fixture, adapter=adapter)
    authority = service.authority_for_principal(_id(PrincipalId, 1))
    try:
        results = await asyncio.gather(
            service.refresh(authority, _id(RuntimeProfileId, 1), "claude:anthropic"),
            service.refresh(authority, _id(RuntimeProfileId, 1), "claude:anthropic"),
        )
        snapshot = await service.inspect(authority, _id(RuntimeProfileId, 1), "claude:anthropic")
    finally:
        await service.close()

    assert adapter.maximum_active == 1
    assert [result.generation for result in results] == [1, 2]
    assert _entry(snapshot, "second").status == ModelCatalogueEntryStatus.AVAILABLE
    assert _entry(snapshot, "first").status == ModelCatalogueEntryStatus.NOT_ADVERTISED


async def test_stale_cross_process_result_cannot_overwrite_newer_refresh(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    slow_adapter = _BlockingAdapter(_batch("stale"))
    slow = await _open(fixture, adapter=slow_adapter)
    fast = await _open(fixture, adapter=_QueueAdapter([_batch("fresh")]))
    authority = slow.authority_for_principal(_id(PrincipalId, 1))
    fast_authority = fast.authority_for_principal(_id(PrincipalId, 1))
    try:
        slow_task = asyncio.create_task(slow.refresh(authority, _id(RuntimeProfileId, 1), "claude:anthropic"))
        await slow_adapter.started.wait()
        fast_result = await fast.refresh(
            fast_authority,
            _id(RuntimeProfileId, 1),
            "claude:anthropic",
        )
        slow_adapter.release.set()
        slow_result = await slow_task
        snapshot = await fast.inspect(fast_authority, _id(RuntimeProfileId, 1), "claude:anthropic")
    finally:
        await slow.close()
        await fast.close()

    assert fast_result.status == ModelCatalogueRefreshStatus.SUCCEEDED
    assert slow_result.status == ModelCatalogueRefreshStatus.SUPERSEDED
    assert _entry(snapshot, "fresh").status == ModelCatalogueEntryStatus.AVAILABLE
    assert all(entry.model_id != "stale" for entry in snapshot.entries)


async def test_inventory_change_invalidates_context_and_uses_stale_fallback(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    service = await _open(fixture, adapter=_QueueAdapter([_batch("last-good")]))
    authority = service.authority_for_principal(_id(PrincipalId, 1))
    try:
        old = await service.refresh(authority, _id(RuntimeProfileId, 1), "claude:anthropic")
        fixture.executable.write_text("#!/bin/sh\nexit 123\n", encoding="utf-8")
        invalidation = await service.synchronize_inventory()
        snapshot = await service.inspect(authority, _id(RuntimeProfileId, 1), "claude:anthropic")
    finally:
        await service.close()

    assert invalidation.invalidated == 1
    assert snapshot.cache_key != old.cache_key
    assert snapshot.last_known_good is True
    assert snapshot.stale is True
    assert _entry(snapshot, "last-good").status == ModelCatalogueEntryStatus.AVAILABLE


async def test_operator_entries_and_refreshes_never_mutate_runtime_settings(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    service = await _open(fixture, adapter=_QueueAdapter([_batch("discovered")]))
    operator = service.operator_authority()
    authority = service.authority_for_principal(_id(PrincipalId, 1))
    connection = sqlite3.connect(fixture.database)
    try:
        connection.execute(
            "INSERT INTO channel_agent_execution_settings "
            "(channel_id, agent_id, runtime_profile_id, field, value) VALUES (?, ?, ?, 'model', ?)",
            (_id(ChannelId, 1), _id(AgentId, 1), _id(RuntimeProfileId, 1), "selected-model"),
        )
        connection.commit()
        before = connection.execute(
            "SELECT channel_id, agent_id, runtime_profile_id, field, value, updated_at "
            "FROM channel_agent_execution_settings"
        ).fetchall()
        await service.refresh(authority, _id(RuntimeProfileId, 1), "claude:anthropic")
        service.register_adapter("claude", _QueueAdapter([RuntimeError("failed refresh")]))
        failed = await service.refresh(authority, _id(RuntimeProfileId, 1), "claude:anthropic")
        await service.upsert_operator_entry(
            operator,
            _id(RuntimeProfileId, 1),
            "claude:anthropic",
            model_id="operator-model",
            display_label="Operator Model",
            capabilities={"vision": True},
        )
        active = await service.inspect(authority, _id(RuntimeProfileId, 1), "claude:anthropic")
        await service.upsert_operator_entry(
            operator,
            _id(RuntimeProfileId, 1),
            "claude:anthropic",
            model_id="operator-model",
            display_label="Updated Operator Model",
            capabilities={"vision": False},
        )
        updated = await service.inspect(authority, _id(RuntimeProfileId, 1), "claude:anthropic")
        assert await service.deactivate_operator_entry(
            operator,
            _id(RuntimeProfileId, 1),
            "claude:anthropic",
            model_id="operator-model",
        )
        inactive = await service.inspect(authority, _id(RuntimeProfileId, 1), "claude:anthropic")
        after = connection.execute(
            "SELECT channel_id, agent_id, runtime_profile_id, field, value, updated_at "
            "FROM channel_agent_execution_settings"
        ).fetchall()
    finally:
        connection.close()
        await service.close()

    assert _entry(active, "operator-model").status == ModelCatalogueEntryStatus.AVAILABLE
    assert _entry(updated, "operator-model").display_label == "Updated Operator Model"
    assert _entry(inactive, "operator-model").status == ModelCatalogueEntryStatus.UNAVAILABLE
    assert failed.status == ModelCatalogueRefreshStatus.FAILED
    assert before == after


async def test_operator_entry_respects_protected_model_ceiling(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, allowed_models=("allowed", "default-model"))
    service = await _open(fixture)
    try:
        with pytest.raises(ModelCatalogueValidationError, match="outside protected"):
            await service.upsert_operator_entry(
                service.operator_authority(),
                _id(RuntimeProfileId, 1),
                "claude:anthropic",
                model_id="forbidden",
                display_label="Forbidden",
            )
    finally:
        await service.close()


async def test_catalogues_are_isolated_by_canonical_principal(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, count=2)
    service = await _open(fixture, adapter=_QueueAdapter([_batch("one"), _batch("two")]))
    first = service.authority_for_principal(_id(PrincipalId, 1))
    second = service.authority_for_principal(_id(PrincipalId, 2))
    try:
        await service.refresh(first, _id(RuntimeProfileId, 1), "claude:anthropic")
        await service.refresh(second, _id(RuntimeProfileId, 2), "claude:anthropic")
        first_snapshot = await service.inspect(first, _id(RuntimeProfileId, 1), "claude:anthropic")
        second_snapshot = await service.inspect(second, _id(RuntimeProfileId, 2), "claude:anthropic")
        with pytest.raises(ModelCatalogueAccessDenied):
            await service.inspect(first, _id(RuntimeProfileId, 2), "claude:anthropic")
    finally:
        await service.close()

    assert _entry(first_snapshot, "one").status == ModelCatalogueEntryStatus.AVAILABLE
    assert all(entry.model_id != "two" for entry in first_snapshot.entries)
    assert _entry(second_snapshot, "two").status == ModelCatalogueEntryStatus.AVAILABLE
    assert all(entry.model_id != "one" for entry in second_snapshot.entries)


async def test_operator_can_refresh_all_contexts_with_unforgeable_authority(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, count=2)
    service = await _open(fixture, adapter=_QueueAdapter([_batch("one"), _batch("two")]))
    try:
        results = await service.refresh_all(service.operator_authority())
        with pytest.raises(ModelCatalogueAccessDenied):
            await service.refresh_all(ModelCatalogueOperatorAuthority(object()))
    finally:
        await service.close()

    assert len(results) == 2
    assert all(result.status == ModelCatalogueRefreshStatus.SUCCEEDED for result in results)
