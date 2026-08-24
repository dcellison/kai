"""Canonical execution authority for Kai's protected internal HTTP API."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from kai.workshop.domain import AgentId, ChannelId, PrincipalId, RuntimeProfileId
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileRegistry
from kai.workshop.store import WorkshopEventStore


class WorkshopInternalAPIContextError(RuntimeError):
    """Protected runtime policy cannot resolve one canonical API context."""


@dataclass(frozen=True, slots=True)
class WorkshopInternalAPIExecutionContext:
    """Canonical authority attached to one protected agent runtime."""

    principal_id: PrincipalId
    channel_id: ChannelId
    agent_id: AgentId
    runtime_profile_id: RuntimeProfileId
    _runtime_config_id: int = field(repr=False)

    @classmethod
    def for_unprotected_runtime(
        cls,
        runtime_config_id: int,
        runtime_profile_id: RuntimeProfileId,
    ) -> WorkshopInternalAPIExecutionContext:
        """Build deterministic development-only identity outside protected installs."""
        digest = hashlib.sha256(f"kai-unprotected:{runtime_config_id}".encode()).hexdigest()
        return cls(
            principal_id=PrincipalId(f"prn_{digest[:32]}"),
            channel_id=ChannelId(f"chn_{digest[1:33]}"),
            agent_id=AgentId(f"agt_{digest[2:34]}"),
            runtime_profile_id=runtime_profile_id,
            _runtime_config_id=runtime_config_id,
        )


class WorkshopInternalAPIContextRegistry:
    """Resolve every protected runtime to one canonical execution context."""

    def __init__(
        self,
        contexts: tuple[WorkshopInternalAPIExecutionContext, ...],
    ) -> None:
        by_profile: dict[RuntimeProfileId, WorkshopInternalAPIExecutionContext] = {}
        by_config_id: dict[int, WorkshopInternalAPIExecutionContext] = {}
        for context in contexts:
            if not isinstance(context, WorkshopInternalAPIExecutionContext):
                raise TypeError("contexts must contain WorkshopInternalAPIExecutionContext values")
            if context.runtime_profile_id in by_profile:
                raise WorkshopInternalAPIContextError("Duplicate internal API runtime profile")
            if context._runtime_config_id in by_config_id:
                raise WorkshopInternalAPIContextError("Duplicate internal API runtime configuration")
            by_profile[context.runtime_profile_id] = context
            by_config_id[context._runtime_config_id] = context
        if not by_profile:
            raise WorkshopInternalAPIContextError("At least one internal API execution context is required")
        self._by_profile = by_profile
        self._by_config_id = by_config_id

    @classmethod
    async def from_store(
        cls,
        store: WorkshopEventStore,
        runtime_profiles: WorkshopRuntimeProfileRegistry,
    ) -> WorkshopInternalAPIContextRegistry:
        """Load complete, same-workshop direct-channel execution contexts."""
        async with store.connection.execute(
            "SELECT ra.runtime_profile_id, cm.principal_id, ra.channel_id, ra.agent_id "
            "FROM channel_agent_runtime_assignments ra "
            "JOIN channels c ON c.id = ra.channel_id AND c.kind = 'direct' "
            "JOIN agents a ON a.id = ra.agent_id AND a.workshop_id = c.workshop_id "
            "JOIN channel_agents ca ON ca.channel_id = c.id AND ca.agent_id = a.id "
            "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.role = 'owner' "
            "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
            "JOIN workshop_memberships wm ON wm.principal_id = p.id "
            "AND wm.workshop_id = c.workshop_id "
            "ORDER BY ra.runtime_profile_id, cm.principal_id"
        ) as cursor:
            rows = list(await cursor.fetchall())

        rows_by_profile: dict[
            RuntimeProfileId,
            set[tuple[PrincipalId, ChannelId, AgentId]],
        ] = {}
        for row in rows:
            try:
                profile_id = RuntimeProfileId(str(row[0]))
                identity = (
                    PrincipalId(str(row[1])),
                    ChannelId(str(row[2])),
                    AgentId(str(row[3])),
                )
            except (TypeError, ValueError) as exc:
                raise WorkshopInternalAPIContextError(
                    "Canonical internal API context contains an invalid opaque identifier"
                ) from exc
            rows_by_profile.setdefault(profile_id, set()).add(identity)

        contexts: list[WorkshopInternalAPIExecutionContext] = []
        for profile in runtime_profiles.profiles:
            identities = rows_by_profile.get(profile.profile_id, set())
            if len(identities) != 1:
                raise WorkshopInternalAPIContextError(
                    "Protected runtime profile must resolve to exactly one canonical internal API context"
                )
            principal_id, channel_id, agent_id = next(iter(identities))
            contexts.append(
                WorkshopInternalAPIExecutionContext(
                    principal_id=principal_id,
                    channel_id=channel_id,
                    agent_id=agent_id,
                    runtime_profile_id=profile.profile_id,
                    _runtime_config_id=profile.runtime_config_id,
                )
            )
        return cls(tuple(contexts))

    @property
    def contexts(self) -> tuple[WorkshopInternalAPIExecutionContext, ...]:
        return tuple(sorted(self._by_profile.values(), key=lambda context: context.runtime_profile_id))

    def for_runtime_profile(
        self,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> WorkshopInternalAPIExecutionContext:
        try:
            normalized = (
                runtime_profile_id
                if isinstance(runtime_profile_id, RuntimeProfileId)
                else RuntimeProfileId(runtime_profile_id)
            )
        except (TypeError, ValueError) as exc:
            raise WorkshopInternalAPIContextError("Runtime profile ID is invalid") from exc
        context = self._by_profile.get(normalized)
        if context is None:
            raise WorkshopInternalAPIContextError("Runtime profile has no canonical internal API context")
        return context

    def for_runtime_config_id(self, runtime_config_id: int) -> WorkshopInternalAPIExecutionContext:
        if isinstance(runtime_config_id, bool) or not isinstance(runtime_config_id, int):
            raise WorkshopInternalAPIContextError("Runtime configuration ID must be an integer")
        context = self._by_config_id.get(runtime_config_id)
        if context is None:
            raise WorkshopInternalAPIContextError("Runtime configuration has no canonical internal API context")
        return context
