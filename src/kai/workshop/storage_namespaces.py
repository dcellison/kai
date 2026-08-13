"""Canonical principal ownership for compatibility storage namespaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kai.workshop.domain import PrincipalId, RuntimeProfileId
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileRegistry
from kai.workshop.store import WorkshopEventStore


class WorkshopStorageNamespaceError(RuntimeError):
    """Protected runtime policy cannot resolve one canonical storage owner."""


@dataclass(frozen=True, slots=True)
class WorkshopPrincipalStorageNamespace:
    """One canonical human principal's transitional storage authority."""

    principal_id: PrincipalId
    runtime_profile_id: RuntimeProfileId
    _runtime_config_id: int = field(repr=False)

    def files_directory(self, data_dir: Path) -> Path:
        """Return the transport-independent upload directory."""
        return data_dir / "files" / str(self.principal_id)

    def legacy_files_directory(self, data_dir: Path) -> Path:
        """Return the prior configured-user directory during migration."""
        return data_dir / "files" / str(self._runtime_config_id)


class WorkshopPrincipalStorageRegistry:
    """Resolve protected runtimes to canonical human-owned storage."""

    def __init__(
        self,
        namespaces: tuple[WorkshopPrincipalStorageNamespace, ...],
    ) -> None:
        by_profile: dict[RuntimeProfileId, WorkshopPrincipalStorageNamespace] = {}
        by_config_id: dict[int, WorkshopPrincipalStorageNamespace] = {}
        for namespace in namespaces:
            if not isinstance(namespace, WorkshopPrincipalStorageNamespace):
                raise TypeError("namespaces must contain WorkshopPrincipalStorageNamespace values")
            if namespace.runtime_profile_id in by_profile:
                raise WorkshopStorageNamespaceError("Duplicate runtime profile storage namespace")
            if namespace._runtime_config_id in by_config_id:
                raise WorkshopStorageNamespaceError("Duplicate runtime configuration storage namespace")
            by_profile[namespace.runtime_profile_id] = namespace
            by_config_id[namespace._runtime_config_id] = namespace
        if not by_profile:
            raise WorkshopStorageNamespaceError("At least one principal storage namespace is required")
        self._by_profile = by_profile
        self._by_config_id = by_config_id

    @classmethod
    async def from_store(
        cls,
        store: WorkshopEventStore,
        runtime_profiles: WorkshopRuntimeProfileRegistry,
    ) -> WorkshopPrincipalStorageRegistry:
        """Resolve every protected runtime through canonical direct ownership."""
        async with store.connection.execute(
            "SELECT ra.runtime_profile_id, cm.principal_id "
            "FROM channel_agent_runtime_assignments ra "
            "JOIN channels c ON c.id = ra.channel_id AND c.kind = 'direct' "
            "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.role = 'owner' "
            "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
            "ORDER BY ra.runtime_profile_id, cm.principal_id"
        ) as cursor:
            rows = list(await cursor.fetchall())

        owners_by_profile: dict[RuntimeProfileId, set[PrincipalId]] = {}
        for row in rows:
            try:
                runtime_profile_id = RuntimeProfileId(str(row[0]))
                principal_id = PrincipalId(str(row[1]))
            except (TypeError, ValueError) as exc:
                raise WorkshopStorageNamespaceError(
                    "Canonical storage ownership contains an invalid opaque identifier"
                ) from exc
            owners_by_profile.setdefault(runtime_profile_id, set()).add(principal_id)

        namespaces: list[WorkshopPrincipalStorageNamespace] = []
        for profile in runtime_profiles.profiles:
            owners = owners_by_profile.get(profile.profile_id, set())
            if len(owners) != 1:
                raise WorkshopStorageNamespaceError(
                    "Protected runtime profile must resolve to exactly one canonical human storage owner"
                )
            namespaces.append(
                WorkshopPrincipalStorageNamespace(
                    next(iter(owners)),
                    profile.profile_id,
                    profile.runtime_config_id,
                )
            )
        return cls(tuple(namespaces))

    @property
    def namespaces(self) -> tuple[WorkshopPrincipalStorageNamespace, ...]:
        return tuple(
            sorted(
                self._by_profile.values(),
                key=lambda namespace: namespace.runtime_profile_id,
            )
        )

    def for_runtime_profile(
        self,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> WorkshopPrincipalStorageNamespace:
        try:
            normalized = (
                runtime_profile_id
                if isinstance(runtime_profile_id, RuntimeProfileId)
                else RuntimeProfileId(runtime_profile_id)
            )
        except (TypeError, ValueError) as exc:
            raise WorkshopStorageNamespaceError("Runtime profile ID is invalid") from exc
        namespace = self._by_profile.get(normalized)
        if namespace is None:
            raise WorkshopStorageNamespaceError("Runtime profile has no canonical principal storage namespace")
        return namespace

    def for_runtime_config_id(
        self,
        runtime_config_id: int,
    ) -> WorkshopPrincipalStorageNamespace:
        if isinstance(runtime_config_id, bool) or not isinstance(runtime_config_id, int) or runtime_config_id <= 0:
            raise WorkshopStorageNamespaceError("Runtime configuration ID must be a positive integer")
        namespace = self._by_config_id.get(runtime_config_id)
        if namespace is None:
            raise WorkshopStorageNamespaceError("Runtime configuration has no canonical principal storage namespace")
        return namespace
