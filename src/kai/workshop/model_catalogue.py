"""Durable, canonical model catalogue and refresh coordination."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Protocol

import aiosqlite

from kai.workshop.domain import PrincipalId, RuntimeProfileId
from kai.workshop.model_discovery_inventory import (
    ModelDiscoveryBackendInventory,
    WorkshopModelDiscoveryInventoryService,
)
from kai.workshop.store import WorkshopEventStore

_SOURCE_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_DEFAULT_TTL_SECONDS = 86_400
_MAX_TTL_SECONDS = 604_800


class ModelCatalogueError(RuntimeError):
    """Base failure for canonical model-catalogue operations."""


class ModelCatalogueAccessDenied(ModelCatalogueError):
    """The caller does not own the requested catalogue lane."""


class ModelCatalogueValidationError(ModelCatalogueError):
    """Discovery or operator catalogue content is malformed."""


class ModelDiscoveryUnsupported(ModelCatalogueError):
    """A backend/auth context does not support model enumeration."""


class ModelDiscoveryAuthenticationError(ModelCatalogueError):
    """The backend account cannot authenticate metadata discovery."""


class ModelCatalogueEntryStatus(StrEnum):
    AVAILABLE = "available"
    NOT_ADVERTISED = "not_advertised"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class ModelCatalogueRefreshStatus(StrEnum):
    REFRESHING = "refreshing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    MALFORMED = "malformed"
    TIMED_OUT = "timed_out"
    INVALIDATED = "invalidated"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True)
class ModelDiscoveryCandidate:
    """One normalized candidate returned by a metadata-only adapter."""

    model_id: str
    display_label: str
    capabilities: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class ModelDiscoveryBatch:
    """One complete, authoritative observation for a discovery lane."""

    source: str
    models: tuple[ModelDiscoveryCandidate, ...]
    ttl_seconds: int = _DEFAULT_TTL_SECONDS


class ModelDiscoveryAdapter(Protocol):
    """Metadata-only backend discovery contract; generation is forbidden."""

    async def discover(self, lane: ModelDiscoveryBackendInventory) -> ModelDiscoveryBatch: ...


@dataclass(frozen=True, slots=True)
class ModelCatalogueAuthority:
    principal_id: PrincipalId


@dataclass(frozen=True, slots=True)
class ModelCatalogueOperatorAuthority:
    _token: object


@dataclass(frozen=True, slots=True)
class ModelCatalogueProvenance:
    source: str
    status: ModelCatalogueEntryStatus
    display_label: str
    capabilities: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ModelCatalogueEntry:
    model_id: str
    display_label: str
    status: ModelCatalogueEntryStatus
    selectable: bool
    retained: bool
    provenances: tuple[ModelCatalogueProvenance, ...]


@dataclass(frozen=True, slots=True)
class ModelCatalogueRefreshState:
    status: ModelCatalogueRefreshStatus
    generation: int
    last_attempt_at: datetime
    last_successful_refresh_at: datetime | None
    expires_at: datetime | None
    error_code: str | None
    error_detail: str | None


@dataclass(frozen=True, slots=True)
class ModelCatalogueSnapshot:
    principal_id: PrincipalId
    runtime_profile_id: RuntimeProfileId
    option_id: str
    cache_key: str
    entries: tuple[ModelCatalogueEntry, ...]
    refresh: ModelCatalogueRefreshState | None
    stale: bool
    last_known_good: bool


@dataclass(frozen=True, slots=True)
class ModelCatalogueRefreshResult:
    runtime_profile_id: RuntimeProfileId
    option_id: str
    cache_key: str
    status: ModelCatalogueRefreshStatus
    generation: int
    discovered_models: int
    preserved_last_known_good: bool


@dataclass(frozen=True, slots=True)
class ModelCatalogueInvalidationResult:
    invalidated: int


SelectedModelResolver = Callable[[RuntimeProfileId], Awaitable[str]]
CuratedModelResolver = Callable[[ModelDiscoveryBackendInventory], Mapping[str, str] | None]
Clock = Callable[[], datetime]


class WorkshopModelCatalogueService:
    """Persist observations without owning or mutating runtime selection."""

    def __init__(
        self,
        store: WorkshopEventStore,
        inventory: WorkshopModelDiscoveryInventoryService,
        *,
        selected_model: SelectedModelResolver,
        curated_models: CuratedModelResolver,
        adapters: Mapping[str, ModelDiscoveryAdapter] | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._store = store
        self._inventory = inventory
        self._selected_model = selected_model
        self._curated_models = curated_models
        self._adapters = dict(adapters or {})
        self._clock = clock or (lambda: datetime.now(UTC))
        self._locks: dict[tuple[RuntimeProfileId, str, str], asyncio.Lock] = {}
        self._operator_token = object()
        self._periodic_stop = asyncio.Event()
        self._periodic_task: asyncio.Task[None] | None = None

    @classmethod
    async def open(
        cls,
        database_path: Path,
        inventory: WorkshopModelDiscoveryInventoryService,
        *,
        selected_model: SelectedModelResolver,
        curated_models: CuratedModelResolver,
        adapters: Mapping[str, ModelDiscoveryAdapter] | None = None,
        clock: Clock | None = None,
    ) -> WorkshopModelCatalogueService:
        service = cls(
            await WorkshopEventStore.open(database_path),
            inventory,
            selected_model=selected_model,
            curated_models=curated_models,
            adapters=adapters,
            clock=clock,
        )
        await service.synchronize_inventory()
        return service

    async def close(self) -> None:
        await self.stop_periodic_refresh()
        await self._store.close()

    def authority_for_principal(self, principal_id: str | PrincipalId) -> ModelCatalogueAuthority:
        try:
            canonical = principal_id if isinstance(principal_id, PrincipalId) else PrincipalId(principal_id)
        except (TypeError, ValueError) as exc:
            raise ModelCatalogueAccessDenied("Model catalogue access denied") from exc
        return ModelCatalogueAuthority(canonical)

    def operator_authority(self) -> ModelCatalogueOperatorAuthority:
        """Return the capability used only by trusted operator surfaces."""
        return ModelCatalogueOperatorAuthority(self._operator_token)

    def register_adapter(self, backend: str, adapter: ModelDiscoveryAdapter) -> None:
        normalized = backend.strip().lower()
        if not normalized:
            raise ValueError("backend must be non-empty")
        self._adapters[normalized] = adapter

    async def inspect(
        self,
        authority: ModelCatalogueAuthority,
        runtime_profile_id: str | RuntimeProfileId,
        option_id: str,
    ) -> ModelCatalogueSnapshot:
        lane = self._principal_lane(authority, runtime_profile_id, option_id)
        return await self._snapshot(lane)

    async def inspect_as_operator(
        self,
        authority: ModelCatalogueOperatorAuthority,
        runtime_profile_id: str | RuntimeProfileId,
        option_id: str,
    ) -> ModelCatalogueSnapshot:
        """Inspect one protected lane without weakening principal APIs."""
        self._require_operator(authority)
        return await self._snapshot(self._operator_lane(runtime_profile_id, option_id))

    async def inspect_all_as_operator(
        self,
        authority: ModelCatalogueOperatorAuthority,
    ) -> tuple[ModelCatalogueSnapshot, ...]:
        """Return every protected lane to a trusted local operator surface."""
        self._require_operator(authority)
        snapshots: list[ModelCatalogueSnapshot] = []
        for profile in self._inventory.inventories:
            for lane in profile.backends:
                snapshots.append(await self._snapshot(lane))
        return tuple(snapshots)

    async def refresh(
        self,
        authority: ModelCatalogueAuthority,
        runtime_profile_id: str | RuntimeProfileId,
        option_id: str,
        *,
        timeout_seconds: float = 30.0,
    ) -> ModelCatalogueRefreshResult:
        lane = self._principal_lane(authority, runtime_profile_id, option_id)
        return await self._refresh_lane(lane, timeout_seconds=timeout_seconds)

    async def refresh_as_operator(
        self,
        authority: ModelCatalogueOperatorAuthority,
        runtime_profile_id: str | RuntimeProfileId,
        option_id: str,
        *,
        timeout_seconds: float = 30.0,
    ) -> ModelCatalogueRefreshResult:
        self._require_operator(authority)
        lane = self._operator_lane(runtime_profile_id, option_id)
        return await self._refresh_lane(lane, timeout_seconds=timeout_seconds)

    async def refresh_all(
        self,
        authority: ModelCatalogueOperatorAuthority,
        *,
        timeout_seconds: float = 30.0,
    ) -> tuple[ModelCatalogueRefreshResult, ...]:
        self._require_operator(authority)
        results: list[ModelCatalogueRefreshResult] = []
        for profile in self._inventory.inventories:
            for lane in profile.backends:
                results.append(await self._refresh_lane(lane, timeout_seconds=timeout_seconds))
        return tuple(results)

    async def start_periodic_refresh(
        self,
        authority: ModelCatalogueOperatorAuthority,
        *,
        interval_seconds: float,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._require_operator(authority)
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if self._periodic_task is not None:
            raise RuntimeError("Periodic model refresh is already active")
        self._periodic_stop.clear()
        self._periodic_task = asyncio.create_task(
            self._periodic_refresh_loop(
                authority,
                interval_seconds=interval_seconds,
                timeout_seconds=timeout_seconds,
            ),
            name="kai-workshop-model-catalogue-refresh",
        )

    async def stop_periodic_refresh(self) -> None:
        task = self._periodic_task
        if task is None:
            return
        self._periodic_stop.set()
        await task
        self._periodic_task = None

    async def synchronize_inventory(self) -> ModelCatalogueInvalidationResult:
        """Invalidate cache contexts whose protected lane inputs changed."""
        current = {
            (str(profile.runtime_profile_id), lane.backend, lane.provider): lane.cache_key
            for profile in self._inventory.inventories
            for lane in profile.backends
        }
        now = _format_timestamp(self._now())
        try:
            await self._store.connection.execute("BEGIN IMMEDIATE")
            async with self._store.connection.execute(
                "SELECT cache_key, runtime_profile_id, backend, provider "
                "FROM workshop_model_catalogue_refreshes WHERE active = 1"
            ) as cursor:
                rows = tuple(await cursor.fetchall())
            invalidated = 0
            for row in rows:
                lane_key = (str(row[1]), str(row[2]), str(row[3]))
                if current.get(lane_key) == str(row[0]):
                    continue
                cursor = await self._store.connection.execute(
                    "UPDATE workshop_model_catalogue_refreshes SET "
                    "active = 0, status = 'invalidated', updated_at = ? "
                    "WHERE cache_key = ? AND active = 1",
                    (now, str(row[0])),
                )
                invalidated += cursor.rowcount
            await self._store.connection.commit()
            return ModelCatalogueInvalidationResult(invalidated)
        except Exception:
            await self._store.connection.rollback()
            raise

    async def invalidate(
        self,
        authority: ModelCatalogueOperatorAuthority,
        runtime_profile_id: str | RuntimeProfileId,
        option_id: str,
    ) -> ModelCatalogueInvalidationResult:
        self._require_operator(authority)
        lane = self._operator_lane(runtime_profile_id, option_id)
        now = _format_timestamp(self._now())
        cursor = await self._store.connection.execute(
            "UPDATE workshop_model_catalogue_refreshes SET "
            "active = 0, status = 'invalidated', updated_at = ? "
            "WHERE cache_key = ? AND active = 1",
            (now, lane.cache_key),
        )
        await self._store.connection.commit()
        return ModelCatalogueInvalidationResult(cursor.rowcount)

    async def upsert_operator_entry(
        self,
        authority: ModelCatalogueOperatorAuthority,
        runtime_profile_id: str | RuntimeProfileId,
        option_id: str,
        *,
        model_id: str,
        display_label: str,
        capabilities: Mapping[str, object] | None = None,
    ) -> None:
        self._require_operator(authority)
        lane = self._operator_lane(runtime_profile_id, option_id)
        model, label, encoded_capabilities = _normalize_model(
            ModelDiscoveryCandidate(model_id, display_label, capabilities)
        )
        if not _policy_allows_model(model, lane.allowed_models):
            raise ModelCatalogueValidationError("Operator model is outside protected runtime policy")
        now = _format_timestamp(self._now())
        await self._store.connection.execute(
            "INSERT INTO workshop_model_catalogue_operator_entries ("
            "runtime_profile_id, backend, provider, model_id, display_label, "
            "capabilities_json, active, created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?) "
            "ON CONFLICT(runtime_profile_id, backend, provider, model_id) DO UPDATE SET "
            "display_label = excluded.display_label, "
            "capabilities_json = excluded.capabilities_json, active = 1, "
            "updated_at = excluded.updated_at",
            (
                lane.cache_inputs.runtime_profile_id,
                lane.backend,
                lane.provider,
                model,
                label,
                encoded_capabilities,
                now,
                now,
            ),
        )
        await self._store.connection.commit()

    async def deactivate_operator_entry(
        self,
        authority: ModelCatalogueOperatorAuthority,
        runtime_profile_id: str | RuntimeProfileId,
        option_id: str,
        *,
        model_id: str,
    ) -> bool:
        self._require_operator(authority)
        lane = self._operator_lane(runtime_profile_id, option_id)
        model = _clean_text(model_id, "model_id", maximum=512)
        cursor = await self._store.connection.execute(
            "UPDATE workshop_model_catalogue_operator_entries SET active = 0, updated_at = ? "
            "WHERE runtime_profile_id = ? AND backend = ? AND provider = ? "
            "AND model_id = ? AND active = 1",
            (
                _format_timestamp(self._now()),
                lane.cache_inputs.runtime_profile_id,
                lane.backend,
                lane.provider,
                model,
            ),
        )
        await self._store.connection.commit()
        return cursor.rowcount == 1

    async def _refresh_lane(
        self,
        initial_lane: ModelDiscoveryBackendInventory,
        *,
        timeout_seconds: float,
    ) -> ModelCatalogueRefreshResult:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        lock = self._locks.setdefault(
            (
                initial_lane.cache_inputs.runtime_profile_id,
                initial_lane.backend,
                initial_lane.provider,
            ),
            asyncio.Lock(),
        )
        async with lock:
            lane = self._operator_lane(
                initial_lane.cache_inputs.runtime_profile_id,
                initial_lane.option_id,
            )
            generation = await self._begin_refresh(lane)
            adapter = self._adapters.get(lane.backend)
            if adapter is None:
                return await self._complete_failure(
                    lane,
                    generation,
                    ModelCatalogueRefreshStatus.UNSUPPORTED,
                    "enumeration_unsupported",
                    "This backend has no model-discovery adapter",
                )
            try:
                async with asyncio.timeout(timeout_seconds):
                    batch = await adapter.discover(lane)
                normalized = _normalize_batch(batch)
            except TimeoutError:
                return await self._complete_failure(
                    lane,
                    generation,
                    ModelCatalogueRefreshStatus.TIMED_OUT,
                    "discovery_timed_out",
                    "Model discovery timed out",
                )
            except ModelDiscoveryUnsupported:
                return await self._complete_failure(
                    lane,
                    generation,
                    ModelCatalogueRefreshStatus.UNSUPPORTED,
                    "enumeration_unsupported",
                    "This backend context does not support model enumeration",
                )
            except ModelDiscoveryAuthenticationError:
                return await self._complete_failure(
                    lane,
                    generation,
                    ModelCatalogueRefreshStatus.FAILED,
                    "authentication_required",
                    f"{lane.backend.title()} authentication is unavailable; "
                    "log in again or verify the configured API key",
                )
            except ModelCatalogueValidationError:
                return await self._complete_failure(
                    lane,
                    generation,
                    ModelCatalogueRefreshStatus.MALFORMED,
                    "malformed_discovery_result",
                    "The discovery adapter returned malformed model metadata",
                )
            except Exception:
                return await self._complete_failure(
                    lane,
                    generation,
                    ModelCatalogueRefreshStatus.FAILED,
                    "discovery_failed",
                    "Model discovery failed",
                )
            current = self._operator_lane(
                lane.cache_inputs.runtime_profile_id,
                lane.option_id,
            )
            if current.cache_key != lane.cache_key:
                await self._invalidate_generation(lane, generation)
                return ModelCatalogueRefreshResult(
                    lane.cache_inputs.runtime_profile_id,
                    lane.option_id,
                    lane.cache_key,
                    ModelCatalogueRefreshStatus.SUPERSEDED,
                    generation,
                    0,
                    await self._has_last_known_good(lane),
                )
            return await self._complete_success(lane, generation, normalized)

    async def _begin_refresh(self, lane: ModelDiscoveryBackendInventory) -> int:
        now = _format_timestamp(self._now())
        try:
            await self._store.connection.execute("BEGIN IMMEDIATE")
            await self._store.connection.execute(
                "UPDATE workshop_model_catalogue_refreshes SET "
                "active = 0, status = 'invalidated', updated_at = ? "
                "WHERE runtime_profile_id = ? AND backend = ? AND provider = ? "
                "AND cache_key <> ? AND active = 1",
                (
                    now,
                    lane.cache_inputs.runtime_profile_id,
                    lane.backend,
                    lane.provider,
                    lane.cache_key,
                ),
            )
            async with self._store.connection.execute(
                "SELECT generation, created_at FROM workshop_model_catalogue_refreshes WHERE cache_key = ?",
                (lane.cache_key,),
            ) as cursor:
                row = await cursor.fetchone()
            generation = int(row[0]) + 1 if row is not None else 1
            created_at = str(row[1]) if row is not None else now
            await self._store.connection.execute(
                "INSERT INTO workshop_model_catalogue_refreshes ("
                "cache_key, principal_id, runtime_profile_id, backend, provider, "
                "auth_fingerprint, executable_fingerprint, status, generation, active, "
                "refresh_started_at, last_attempt_at, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, 'refreshing', ?, 1, ?, ?, ?, ?) "
                "ON CONFLICT(cache_key) DO UPDATE SET "
                "principal_id = excluded.principal_id, runtime_profile_id = excluded.runtime_profile_id, "
                "backend = excluded.backend, provider = excluded.provider, "
                "auth_fingerprint = excluded.auth_fingerprint, "
                "executable_fingerprint = excluded.executable_fingerprint, "
                "status = 'refreshing', generation = excluded.generation, active = 1, "
                "last_error_code = NULL, last_error_detail = NULL, "
                "refresh_started_at = excluded.refresh_started_at, "
                "last_attempt_at = excluded.last_attempt_at, updated_at = excluded.updated_at",
                (
                    lane.cache_key,
                    lane.cache_inputs.principal_id,
                    lane.cache_inputs.runtime_profile_id,
                    lane.backend,
                    lane.provider,
                    lane.auth.fingerprint,
                    lane.executable.fingerprint,
                    generation,
                    now,
                    now,
                    created_at,
                    now,
                ),
            )
            await self._store.connection.commit()
            return generation
        except Exception:
            await self._store.connection.rollback()
            raise

    async def _complete_success(
        self,
        lane: ModelDiscoveryBackendInventory,
        generation: int,
        batch: ModelDiscoveryBatch,
    ) -> ModelCatalogueRefreshResult:
        now_value = self._now()
        now = _format_timestamp(now_value)
        expires = _format_timestamp(now_value + timedelta(seconds=batch.ttl_seconds))
        try:
            await self._store.connection.execute("BEGIN IMMEDIATE")
            if not await self._generation_is_current(lane.cache_key, generation):
                await self._store.connection.rollback()
                return ModelCatalogueRefreshResult(
                    lane.cache_inputs.runtime_profile_id,
                    lane.option_id,
                    lane.cache_key,
                    ModelCatalogueRefreshStatus.SUPERSEDED,
                    generation,
                    0,
                    await self._has_last_known_good(lane),
                )
            await self._store.connection.execute(
                "UPDATE workshop_model_catalogue_discovered_entries SET "
                "status = 'not_advertised', last_successful_refresh_at = ?, expires_at = ? "
                "WHERE cache_key = ?",
                (now, expires, lane.cache_key),
            )
            for candidate in batch.models:
                model, label, capabilities = _normalize_model(candidate)
                await self._store.connection.execute(
                    "INSERT INTO workshop_model_catalogue_discovered_entries ("
                    "cache_key, model_id, display_label, discovery_source, capabilities_json, "
                    "status, first_seen_at, last_seen_at, last_successful_refresh_at, expires_at"
                    ") VALUES (?, ?, ?, ?, ?, 'available', ?, ?, ?, ?) "
                    "ON CONFLICT(cache_key, model_id) DO UPDATE SET "
                    "display_label = excluded.display_label, "
                    "discovery_source = excluded.discovery_source, "
                    "capabilities_json = excluded.capabilities_json, status = 'available', "
                    "last_seen_at = excluded.last_seen_at, "
                    "last_successful_refresh_at = excluded.last_successful_refresh_at, "
                    "expires_at = excluded.expires_at",
                    (
                        lane.cache_key,
                        model,
                        label,
                        batch.source,
                        capabilities,
                        now,
                        now,
                        now,
                        expires,
                    ),
                )
            cursor = await self._store.connection.execute(
                "UPDATE workshop_model_catalogue_refreshes SET "
                "status = 'succeeded', discovery_source = ?, last_error_code = NULL, "
                "last_error_detail = NULL, last_successful_refresh_at = ?, expires_at = ?, "
                "updated_at = ? WHERE cache_key = ? AND generation = ? AND active = 1",
                (batch.source, now, expires, now, lane.cache_key, generation),
            )
            if cursor.rowcount != 1:
                await self._store.connection.rollback()
                return ModelCatalogueRefreshResult(
                    lane.cache_inputs.runtime_profile_id,
                    lane.option_id,
                    lane.cache_key,
                    ModelCatalogueRefreshStatus.SUPERSEDED,
                    generation,
                    0,
                    await self._has_last_known_good(lane),
                )
            await self._store.connection.commit()
        except Exception:
            await self._store.connection.rollback()
            raise
        return ModelCatalogueRefreshResult(
            lane.cache_inputs.runtime_profile_id,
            lane.option_id,
            lane.cache_key,
            ModelCatalogueRefreshStatus.SUCCEEDED,
            generation,
            len(batch.models),
            False,
        )

    async def _complete_failure(
        self,
        lane: ModelDiscoveryBackendInventory,
        generation: int,
        status: ModelCatalogueRefreshStatus,
        error_code: str,
        error_detail: str,
    ) -> ModelCatalogueRefreshResult:
        now = _format_timestamp(self._now())
        cursor = await self._store.connection.execute(
            "UPDATE workshop_model_catalogue_refreshes SET status = ?, "
            "last_error_code = ?, last_error_detail = ?, updated_at = ? "
            "WHERE cache_key = ? AND generation = ? AND active = 1",
            (status.value, error_code, error_detail, now, lane.cache_key, generation),
        )
        await self._store.connection.commit()
        effective_status = status if cursor.rowcount == 1 else ModelCatalogueRefreshStatus.SUPERSEDED
        return ModelCatalogueRefreshResult(
            lane.cache_inputs.runtime_profile_id,
            lane.option_id,
            lane.cache_key,
            effective_status,
            generation,
            0,
            await self._has_last_known_good(lane),
        )

    async def _snapshot(self, lane: ModelDiscoveryBackendInventory) -> ModelCatalogueSnapshot:
        refresh = await self._refresh_state(lane.cache_key)
        rows = await self._discovered_rows(lane.cache_key)
        last_known_good = False
        if not rows:
            fallback_key = await self._last_known_good_cache_key(lane)
            if fallback_key is not None and fallback_key != lane.cache_key:
                rows = await self._discovered_rows(fallback_key)
                last_known_good = bool(rows)
        merged: dict[str, list[ModelCatalogueProvenance]] = {}
        for row in rows:
            provenance = ModelCatalogueProvenance(
                source=f"discovered:{row['discovery_source']}",
                status=ModelCatalogueEntryStatus(str(row["status"])),
                display_label=str(row["display_label"]),
                capabilities=_decode_capabilities(str(row["capabilities_json"])),
            )
            merged.setdefault(str(row["model_id"]), []).append(provenance)
        async with self._store.connection.execute(
            "SELECT model_id, display_label, capabilities_json, active "
            "FROM workshop_model_catalogue_operator_entries "
            "WHERE runtime_profile_id = ? AND backend = ? AND provider = ? "
            "ORDER BY model_id",
            (lane.cache_inputs.runtime_profile_id, lane.backend, lane.provider),
        ) as cursor:
            operator_rows = tuple(await cursor.fetchall())
        for row in operator_rows:
            provenance = ModelCatalogueProvenance(
                source="operator",
                status=(ModelCatalogueEntryStatus.AVAILABLE if bool(row[3]) else ModelCatalogueEntryStatus.UNAVAILABLE),
                display_label=str(row[1]),
                capabilities=_decode_capabilities(str(row[2])),
            )
            merged.setdefault(str(row[0]), []).append(provenance)
        curated = self._curated_models(lane) or {}
        for model_id, label in sorted(curated.items()):
            model = _clean_text(model_id, "curated model id", maximum=512)
            display = _clean_text(label, "curated display label", maximum=512)
            merged.setdefault(model, []).append(
                ModelCatalogueProvenance(
                    source="curated",
                    status=ModelCatalogueEntryStatus.AVAILABLE,
                    display_label=display,
                    capabilities={},
                )
            )
        retained = {lane.default_model}
        if lane.selected:
            retained.add(await self._selected_model(lane.cache_inputs.runtime_profile_id))
        for model_id in retained:
            model = _clean_text(model_id, "retained model id", maximum=512)
            if model not in merged:
                merged[model] = [
                    ModelCatalogueProvenance(
                        source="retained_selection",
                        status=ModelCatalogueEntryStatus.NOT_ADVERTISED,
                        display_label=model,
                        capabilities={},
                    )
                ]
        entries = tuple(
            _merge_entry(model_id, tuple(provenances), retained=model_id in retained, lane=lane)
            for model_id, provenances in sorted(merged.items())
        )
        stale = last_known_good or refresh is None or refresh.status != ModelCatalogueRefreshStatus.SUCCEEDED
        if refresh is not None and refresh.expires_at is not None and refresh.expires_at <= self._now():
            stale = True
        return ModelCatalogueSnapshot(
            principal_id=lane.cache_inputs.principal_id,
            runtime_profile_id=lane.cache_inputs.runtime_profile_id,
            option_id=lane.option_id,
            cache_key=lane.cache_key,
            entries=entries,
            refresh=refresh,
            stale=stale,
            last_known_good=last_known_good,
        )

    async def _refresh_state(self, cache_key: str) -> ModelCatalogueRefreshState | None:
        async with self._store.connection.execute(
            "SELECT status, generation, last_attempt_at, last_successful_refresh_at, "
            "expires_at, last_error_code, last_error_detail "
            "FROM workshop_model_catalogue_refreshes WHERE cache_key = ?",
            (cache_key,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return ModelCatalogueRefreshState(
            ModelCatalogueRefreshStatus(str(row[0])),
            int(row[1]),
            _parse_timestamp(str(row[2])),
            _parse_timestamp(str(row[3])) if row[3] is not None else None,
            _parse_timestamp(str(row[4])) if row[4] is not None else None,
            str(row[5]) if row[5] is not None else None,
            str(row[6]) if row[6] is not None else None,
        )

    async def _discovered_rows(self, cache_key: str) -> tuple[aiosqlite.Row, ...]:
        async with self._store.connection.execute(
            "SELECT model_id, display_label, discovery_source, capabilities_json, status "
            "FROM workshop_model_catalogue_discovered_entries "
            "WHERE cache_key = ? ORDER BY model_id",
            (cache_key,),
        ) as cursor:
            return tuple(await cursor.fetchall())

    async def _last_known_good_cache_key(self, lane: ModelDiscoveryBackendInventory) -> str | None:
        async with self._store.connection.execute(
            "SELECT cache_key FROM workshop_model_catalogue_refreshes "
            "WHERE runtime_profile_id = ? AND backend = ? AND provider = ? "
            "AND last_successful_refresh_at IS NOT NULL "
            "ORDER BY last_successful_refresh_at DESC, generation DESC LIMIT 1",
            (lane.cache_inputs.runtime_profile_id, lane.backend, lane.provider),
        ) as cursor:
            row = await cursor.fetchone()
        return str(row[0]) if row is not None else None

    async def _has_last_known_good(self, lane: ModelDiscoveryBackendInventory) -> bool:
        return await self._last_known_good_cache_key(lane) is not None

    async def _generation_is_current(self, cache_key: str, generation: int) -> bool:
        async with self._store.connection.execute(
            "SELECT 1 FROM workshop_model_catalogue_refreshes "
            "WHERE cache_key = ? AND generation = ? AND active = 1 AND status = 'refreshing'",
            (cache_key, generation),
        ) as cursor:
            return await cursor.fetchone() is not None

    async def _invalidate_generation(self, lane: ModelDiscoveryBackendInventory, generation: int) -> None:
        await self._store.connection.execute(
            "UPDATE workshop_model_catalogue_refreshes SET "
            "active = 0, status = 'invalidated', updated_at = ? "
            "WHERE cache_key = ? AND generation = ? AND active = 1",
            (_format_timestamp(self._now()), lane.cache_key, generation),
        )
        await self._store.connection.commit()

    def _principal_lane(
        self,
        authority: ModelCatalogueAuthority,
        runtime_profile_id: str | RuntimeProfileId,
        option_id: str,
    ) -> ModelDiscoveryBackendInventory:
        try:
            profile_id = (
                runtime_profile_id
                if isinstance(runtime_profile_id, RuntimeProfileId)
                else RuntimeProfileId(runtime_profile_id)
            )
        except (TypeError, ValueError) as exc:
            raise ModelCatalogueAccessDenied("Model catalogue access denied") from exc
        profile = next(
            (
                item
                for item in self._inventory.for_principal(authority.principal_id)
                if item.runtime_profile_id == profile_id
            ),
            None,
        )
        if profile is None:
            raise ModelCatalogueAccessDenied("Model catalogue access denied")
        lane = next((item for item in profile.backends if item.option_id == option_id.strip().lower()), None)
        if lane is None:
            raise ModelCatalogueAccessDenied("Model catalogue access denied")
        return lane

    def _operator_lane(
        self,
        runtime_profile_id: str | RuntimeProfileId,
        option_id: str,
    ) -> ModelDiscoveryBackendInventory:
        try:
            profile_id = (
                runtime_profile_id
                if isinstance(runtime_profile_id, RuntimeProfileId)
                else RuntimeProfileId(runtime_profile_id)
            )
        except (TypeError, ValueError) as exc:
            raise ModelCatalogueAccessDenied("Model catalogue lane does not exist") from exc
        for profile in self._inventory.inventories:
            if profile.runtime_profile_id != profile_id:
                continue
            lane = next((item for item in profile.backends if item.option_id == option_id.strip().lower()), None)
            if lane is not None:
                return lane
        raise ModelCatalogueAccessDenied("Model catalogue lane does not exist")

    def _require_operator(self, authority: ModelCatalogueOperatorAuthority) -> None:
        if authority._token is not self._operator_token:
            raise ModelCatalogueAccessDenied("Model catalogue operator access denied")

    async def _periodic_refresh_loop(
        self,
        authority: ModelCatalogueOperatorAuthority,
        *,
        interval_seconds: float,
        timeout_seconds: float,
    ) -> None:
        while not self._periodic_stop.is_set():
            try:
                await asyncio.wait_for(self._periodic_stop.wait(), timeout=interval_seconds)
            except TimeoutError:
                await self.refresh_all(authority, timeout_seconds=timeout_seconds)

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ModelCatalogueError("Model catalogue clock must return a timezone-aware datetime")
        return value.astimezone(UTC)


def _normalize_batch(batch: object) -> ModelDiscoveryBatch:
    if not isinstance(batch, ModelDiscoveryBatch):
        raise ModelCatalogueValidationError("Discovery result has the wrong type")
    source = batch.source.strip().lower()
    if not _SOURCE_PATTERN.fullmatch(source):
        raise ModelCatalogueValidationError("Discovery source is invalid")
    if isinstance(batch.ttl_seconds, bool) or not 1 <= batch.ttl_seconds <= _MAX_TTL_SECONDS:
        raise ModelCatalogueValidationError("Discovery TTL is invalid")
    seen: dict[str, tuple[str, str]] = {}
    normalized: list[ModelDiscoveryCandidate] = []
    for candidate in batch.models:
        if not isinstance(candidate, ModelDiscoveryCandidate):
            raise ModelCatalogueValidationError("Discovery candidates have the wrong type")
        model, label, capabilities = _normalize_model(candidate)
        prior = seen.get(model)
        identity = (label, capabilities)
        if prior is not None and prior != identity:
            raise ModelCatalogueValidationError("Discovery contains conflicting duplicate models")
        if prior is None:
            seen[model] = identity
            normalized.append(ModelDiscoveryCandidate(model, label, _decode_capabilities(capabilities)))
    return ModelDiscoveryBatch(source, tuple(normalized), batch.ttl_seconds)


def _normalize_model(candidate: ModelDiscoveryCandidate) -> tuple[str, str, str]:
    model = _clean_text(candidate.model_id, "model_id", maximum=512)
    label = _clean_text(candidate.display_label, "display_label", maximum=512)
    capabilities = candidate.capabilities or {}
    try:
        encoded = json.dumps(capabilities, sort_keys=True, separators=(",", ":"), allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ModelCatalogueValidationError("Model capabilities are not valid JSON metadata") from exc
    if not isinstance(decoded, dict) or len(encoded) > 16_384:
        raise ModelCatalogueValidationError("Model capabilities must be a bounded JSON object")
    return model, label, encoded


def _clean_text(value: object, field: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise ModelCatalogueValidationError(f"{field} must be text")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > maximum or any(ord(character) < 32 for character in cleaned):
        raise ModelCatalogueValidationError(f"{field} is invalid")
    return cleaned


def _decode_capabilities(value: str) -> Mapping[str, object]:
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ModelCatalogueError("Stored model capability metadata is invalid")
    return decoded


def _merge_entry(
    model_id: str,
    provenances: tuple[ModelCatalogueProvenance, ...],
    *,
    retained: bool,
    lane: ModelDiscoveryBackendInventory,
) -> ModelCatalogueEntry:
    status_values = {item.status for item in provenances}
    policy_allowed = _policy_allows_model(model_id, lane.allowed_models)
    if not policy_allowed:
        status = ModelCatalogueEntryStatus.UNAVAILABLE
    elif ModelCatalogueEntryStatus.AVAILABLE in status_values:
        status = ModelCatalogueEntryStatus.AVAILABLE
    elif retained or ModelCatalogueEntryStatus.NOT_ADVERTISED in status_values:
        status = ModelCatalogueEntryStatus.NOT_ADVERTISED
    elif ModelCatalogueEntryStatus.UNAVAILABLE in status_values:
        status = ModelCatalogueEntryStatus.UNAVAILABLE
    else:
        status = ModelCatalogueEntryStatus.UNKNOWN
    preferred = sorted(
        provenances,
        key=lambda item: (
            {"operator": 0, "discovered": 1, "curated": 2, "retained_selection": 3}.get(
                item.source.partition(":")[0],
                4,
            ),
            item.source,
        ),
    )[0]
    return ModelCatalogueEntry(
        model_id=model_id,
        display_label=preferred.display_label,
        status=status,
        selectable=status == ModelCatalogueEntryStatus.AVAILABLE,
        retained=retained,
        provenances=tuple(sorted(provenances, key=lambda item: item.source)),
    )


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _policy_allows_model(model_id: str, allowed_models: tuple[str, ...] | None) -> bool:
    """Apply only the protected operator ceiling to observed catalogue IDs.

    Static curated lists are deliberately not validators here: discovery must
    be able to advertise a newly released model before Kai itself is updated.
    """
    if allowed_models is None:
        return True
    return any(fnmatchcase(model_id, pattern) for pattern in allowed_models)
