"""Canonical execution authority for Kai's protected internal HTTP API."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

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
    private_context: bool = True
    sponsor_principal_id: PrincipalId | None = None
    settings_channel_id: ChannelId | None = None

    @property
    def runtime_owner_principal_id(self) -> PrincipalId:
        return self.sponsor_principal_id or self.principal_id

    @property
    def effective_settings_channel_id(self) -> ChannelId:
        return self.settings_channel_id or self.channel_id

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
        )


class WorkshopInternalAPIContextRegistry:
    """Resolve every protected runtime to one canonical execution context."""

    def __init__(
        self,
        contexts: tuple[WorkshopInternalAPIExecutionContext, ...],
    ) -> None:
        by_profile: dict[RuntimeProfileId, WorkshopInternalAPIExecutionContext] = {}
        by_lane: dict[tuple[PrincipalId, ChannelId, AgentId], WorkshopInternalAPIExecutionContext] = {}
        ordered: list[WorkshopInternalAPIExecutionContext] = []
        for context in contexts:
            if not isinstance(context, WorkshopInternalAPIExecutionContext):
                raise TypeError("contexts must contain WorkshopInternalAPIExecutionContext values")
            key = (context.principal_id, context.channel_id, context.agent_id)
            if key in by_lane:
                raise WorkshopInternalAPIContextError("Duplicate internal API execution lane")
            primary = by_profile.get(context.runtime_profile_id)
            if primary is not None and (primary.runtime_owner_principal_id != context.runtime_owner_principal_id):
                raise WorkshopInternalAPIContextError("Runtime profile cannot cross internal API principals")
            by_profile.setdefault(context.runtime_profile_id, context)
            by_lane[key] = context
            ordered.append(context)
        if not by_profile:
            raise WorkshopInternalAPIContextError("At least one internal API execution context is required")
        self._by_profile = by_profile
        self._by_lane = by_lane
        self._contexts = ordered

    @classmethod
    async def from_store(
        cls,
        store: WorkshopEventStore,
        runtime_profiles: WorkshopRuntimeProfileRegistry,
    ) -> WorkshopInternalAPIContextRegistry:
        """Load complete, same-workshop direct-channel execution contexts."""
        async with store.connection.execute(
            "SELECT ra.runtime_profile_id, cm.principal_id, ra.channel_id, ra.agent_id, "
            "ra.created_event_position "
            "FROM channel_agent_runtime_assignments ra "
            "JOIN channels c ON c.id = ra.channel_id AND c.kind = 'direct' "
            "JOIN agents a ON a.id = ra.agent_id AND a.workshop_id = c.workshop_id "
            "JOIN channel_agents ca ON ca.channel_id = c.id AND ca.agent_id = a.id "
            "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.role = 'owner' "
            "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
            "JOIN workshop_memberships wm ON wm.principal_id = p.id "
            "AND wm.workshop_id = c.workshop_id "
            "LEFT JOIN principal_agent_enablements pae ON pae.direct_channel_id = c.id "
            "AND pae.agent_id = ra.agent_id "
            "WHERE pae.id IS NULL OR pae.lifecycle_state = 'enabled' "
            "ORDER BY ra.runtime_profile_id, ra.created_event_position, cm.principal_id"
        ) as cursor:
            rows = list(await cursor.fetchall())

        rows_by_profile: dict[
            RuntimeProfileId,
            list[tuple[PrincipalId, ChannelId, AgentId]],
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
            if identity not in rows_by_profile.setdefault(profile_id, []):
                rows_by_profile[profile_id].append(identity)

        contexts: list[WorkshopInternalAPIExecutionContext] = []
        for profile in runtime_profiles.profiles:
            identities = rows_by_profile.get(profile.profile_id, [])
            if not identities or len({item[0] for item in identities}) != 1:
                raise WorkshopInternalAPIContextError(
                    "Protected runtime profile must resolve to one canonical human principal"
                )
            for principal_id, channel_id, agent_id in identities:
                contexts.append(
                    WorkshopInternalAPIExecutionContext(
                        principal_id=principal_id,
                        channel_id=channel_id,
                        agent_id=agent_id,
                        runtime_profile_id=profile.profile_id,
                    )
                )
        return cls(tuple(contexts))

    @property
    def contexts(self) -> tuple[WorkshopInternalAPIExecutionContext, ...]:
        return tuple(self._contexts)

    def register_context(self, context: WorkshopInternalAPIExecutionContext) -> None:
        """Register a live principal-agent lane while keeping the profile primary stable."""
        key = (context.principal_id, context.channel_id, context.agent_id)
        existing = self._by_lane.get(key)
        if existing is not None:
            if existing != context:
                raise WorkshopInternalAPIContextError("Internal API execution lane conflicts")
            return
        primary = self._by_profile.get(context.runtime_profile_id)
        if primary is None or (primary.runtime_owner_principal_id != context.runtime_owner_principal_id):
            raise WorkshopInternalAPIContextError("Runtime profile is not owned by this internal API principal")
        self._by_lane[key] = context
        self._contexts.append(context)

    def replace_context(
        self,
        prior: WorkshopInternalAPIExecutionContext,
        replacement: WorkshopInternalAPIExecutionContext,
    ) -> None:
        """Rebind one live lane to another profile owned by the same human."""
        key = (prior.principal_id, prior.channel_id, prior.agent_id)
        if self._by_lane.get(key) != prior or replacement.principal_id != prior.principal_id:
            raise WorkshopInternalAPIContextError("Internal API context replacement conflicts")
        if (replacement.principal_id, replacement.channel_id, replacement.agent_id) != key:
            raise WorkshopInternalAPIContextError("Internal API context identity cannot change")
        primary = self._by_profile.get(replacement.runtime_profile_id)
        if primary is None or (primary.runtime_owner_principal_id != replacement.runtime_owner_principal_id):
            raise WorkshopInternalAPIContextError("Replacement runtime profile is not owned by the principal")
        self._by_lane[key] = replacement
        self._contexts[self._contexts.index(prior)] = replacement

    def unregister_context(self, context: WorkshopInternalAPIExecutionContext) -> None:
        """Remove one non-primary internal API lane after authority retirement."""
        key = (context.principal_id, context.channel_id, context.agent_id)
        existing = self._by_lane.get(key)
        if existing is None:
            return
        if existing != context or self._by_profile.get(context.runtime_profile_id) == context:
            raise WorkshopInternalAPIContextError("Canonical internal API context cannot be retired")
        self._by_lane.pop(key)
        self._contexts.remove(context)

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
