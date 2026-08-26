"""Bind optional client identities to existing canonical Workshop humans."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from kai.workshop.domain import (
    ChannelBindingId,
    ChannelId,
    EventEnvelope,
    ExternalIdentityId,
    PrincipalId,
    RuntimeProfileId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileRegistry
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore

_TRANSPORT_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class WorkshopTransportLinkError(RuntimeError):
    """A transport identity cannot be linked without changing authority."""


@dataclass(frozen=True, slots=True)
class WorkshopTransportLink:
    principal_id: PrincipalId
    channel_id: ChannelId
    transport: str
    external_subject: str
    external_channel_id: str
    created_events: int


def _token(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


class WorkshopTransportLinker:
    """Attach adapter routing to a pre-existing runtime-owned direct channel."""

    def __init__(
        self,
        store: WorkshopEventStore,
        runtime_profiles: WorkshopRuntimeProfileRegistry,
    ) -> None:
        self._store = store
        self._runtime_profiles = runtime_profiles

    async def link_runtime_profile(
        self,
        runtime_profile_id: RuntimeProfileId,
        *,
        transport: str,
        external_subject: str,
        external_channel_id: str,
    ) -> WorkshopTransportLink:
        self._runtime_profiles.resolve(runtime_profile_id)
        if not _TRANSPORT_PATTERN.fullmatch(transport):
            raise WorkshopTransportLinkError("Transport must be a lowercase identifier")
        subject = external_subject.strip()
        external_channel = external_channel_id.strip()
        if not subject or not external_channel:
            raise WorkshopTransportLinkError("External subject and channel must be non-empty")

        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            workshop_id, principal_id, channel_id = await self._resolve_runtime_owner(runtime_profile_id)
            await self._reject_conflicting_identity(transport, subject, principal_id)
            await self._reject_conflicting_channel(transport, external_channel, channel_id)
            stable = _token(transport, subject, external_channel)
            identity_key = f"operator:transport-link:{stable}:identity"
            channel_key = f"operator:transport-link:{stable}:channel"
            prior_identity = await self._store.event_by_idempotency_key(identity_key)
            prior_channel = await self._store.event_by_idempotency_key(channel_key)
            if prior_identity is not None or prior_channel is not None:
                if prior_identity is None or prior_channel is None:
                    raise WorkshopTransportLinkError("Transport link has an incomplete canonical event pair")
                await connection.commit()
                return WorkshopTransportLink(
                    principal_id,
                    channel_id,
                    transport,
                    subject,
                    external_channel,
                    0,
                )
            identity_id = ExternalIdentityId.derived(
                workshop_id,
                f"transport-link:identity:{stable}",
            )
            binding_id = ChannelBindingId.derived(
                workshop_id,
                f"transport-link:channel:{stable}",
            )
            events = (
                EventEnvelope.create(
                    event_type=WorkshopEventType.EXTERNAL_IDENTITY_BOUND,
                    event_version=1,
                    workshop_id=workshop_id,
                    aggregate_type="external_identity",
                    aggregate_id=identity_id,
                    occurred_at=datetime.now(UTC),
                    idempotency_key=identity_key,
                    payload={
                        "principal_id": principal_id,
                        "provider": transport,
                        "external_subject": subject,
                    },
                    metadata={"source": "adapter_policy"},
                ),
                EventEnvelope.create(
                    event_type=WorkshopEventType.TRANSPORT_CHANNEL_BOUND,
                    event_version=1,
                    workshop_id=workshop_id,
                    aggregate_type="channel_binding",
                    aggregate_id=binding_id,
                    occurred_at=datetime.now(UTC),
                    idempotency_key=channel_key,
                    payload={
                        "channel_id": channel_id,
                        "transport": transport,
                        "external_channel_id": external_channel,
                    },
                    metadata={"source": "adapter_policy"},
                ),
            )
            inserted = 0
            for event in events:
                result = await self._store.append_in_transaction(event)
                inserted += int(result.inserted)
            await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            await connection.commit()
        except IdempotencyConflictError as exc:
            await connection.rollback()
            raise WorkshopTransportLinkError("Transport link conflicts with an existing canonical event") from exc
        except Exception:
            await connection.rollback()
            raise
        return WorkshopTransportLink(
            principal_id,
            channel_id,
            transport,
            subject,
            external_channel,
            inserted,
        )

    async def _resolve_runtime_owner(
        self,
        runtime_profile_id: RuntimeProfileId,
    ) -> tuple[WorkshopId, PrincipalId, ChannelId]:
        async with self._store.connection.execute(
            "SELECT c.workshop_id, cm.principal_id, c.id "
            "FROM channel_agent_runtime_assignments ra "
            "JOIN channels c ON c.id = ra.channel_id AND c.kind = 'direct' "
            "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.role = 'owner' "
            "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
            "WHERE ra.runtime_profile_id = ?",
            (runtime_profile_id,),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if len(rows) != 1:
            raise WorkshopTransportLinkError(
                "Runtime profile must resolve to exactly one canonical human direct channel"
            )
        return (
            WorkshopId(str(rows[0][0])),
            PrincipalId(str(rows[0][1])),
            ChannelId(str(rows[0][2])),
        )

    async def _reject_conflicting_identity(
        self,
        transport: str,
        subject: str,
        principal_id: PrincipalId,
    ) -> None:
        async with self._store.connection.execute(
            "SELECT principal_id FROM external_identities WHERE provider = ? AND external_subject = ?",
            (transport, subject),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if rows and {str(row[0]) for row in rows} != {str(principal_id)}:
            raise WorkshopTransportLinkError("External identity is already bound to a different canonical principal")

    async def _reject_conflicting_channel(
        self,
        transport: str,
        external_channel_id: str,
        channel_id: ChannelId,
    ) -> None:
        async with self._store.connection.execute(
            "SELECT channel_id FROM channel_bindings WHERE transport = ? AND external_channel_id = ?",
            (transport, external_channel_id),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if rows and {str(row[0]) for row in rows} != {str(channel_id)}:
            raise WorkshopTransportLinkError("External channel is already bound to a different canonical channel")
