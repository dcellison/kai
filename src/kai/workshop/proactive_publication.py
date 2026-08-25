"""Canonical proactive messages and artifacts from protected agent runtimes."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kai.workshop.artifacts import (
    StagedArtifact,
    WorkshopArtifactService,
    record_published_artifact_in_transaction,
)
from kai.workshop.delivery_outbox import NOTIFICATION_PURPOSE, DeliveryRequestResult
from kai.workshop.delivery_planning import CanonicalDeliveryIntent, WorkshopDeliveryPlanner
from kai.workshop.delivery_policy import WorkshopDeliveryBindingPolicy
from kai.workshop.domain import (
    AgentId,
    ChannelId,
    EventEnvelope,
    EventId,
    MessageId,
    PrincipalId,
    RuntimeProfileId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore

PROACTIVE_ARTIFACT_MODE = "artifact"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class ProactivePublicationError(RuntimeError):
    """A proactive effect could not be recorded under canonical authority."""


@dataclass(frozen=True, slots=True)
class ProactivePublicationAuthority:
    principal_id: PrincipalId
    channel_id: ChannelId
    agent_id: AgentId
    runtime_profile_id: RuntimeProfileId


@dataclass(frozen=True, slots=True)
class ProactivePublicationResult:
    message_id: MessageId
    inserted: bool
    deliveries: tuple[DeliveryRequestResult, ...]

    @property
    def delivery_status(self) -> str:
        if not self.deliveries:
            return "not_configured"
        states = {result.delivery.status for result in self.deliveries}
        if states == {"succeeded"}:
            return "delivered"
        if "failed" in states:
            return "failed"
        return "queued"


@dataclass(frozen=True, slots=True)
class _ResolvedAuthority:
    workshop_id: WorkshopId
    agent_principal_id: PrincipalId


class WorkshopProactivePublicationService:
    """Record proactive effects before requesting optional adapter delivery."""

    def __init__(
        self,
        store: WorkshopEventStore,
        artifacts: WorkshopArtifactService,
        *,
        artifact_storage_root: Path,
        delivery_policy: WorkshopDeliveryBindingPolicy,
    ) -> None:
        self._store = store
        self._artifacts = artifacts
        self._artifact_storage_root = artifact_storage_root.resolve()
        self._delivery_planner = WorkshopDeliveryPlanner(store, delivery_policy)
        self._lock = asyncio.Lock()

    async def validate_authority(self, authority: ProactivePublicationAuthority) -> None:
        await self._resolve_authority(authority)

    async def publish_text(
        self,
        authority: ProactivePublicationAuthority,
        *,
        request_id: str,
        body: str,
        occurred_at: datetime,
    ) -> ProactivePublicationResult:
        if not body:
            raise ValueError("body must be non-empty")
        async with self._lock:
            return await self._record(
                authority,
                request_id=request_id,
                body=body,
                occurred_at=occurred_at,
                kind="text",
                mode="text",
                caption=None,
                artifact=None,
            )

    async def publish_file(
        self,
        authority: ProactivePublicationAuthority,
        *,
        request_id: str,
        path: Path,
        caption: str,
        occurred_at: datetime,
    ) -> ProactivePublicationResult:
        self._validate_request(request_id, occurred_at)
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("path must be absolute")

        async def chunks() -> AsyncIterator[bytes]:
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    yield chunk

        async with self._lock:
            staged = await self._artifacts.stage_upload(
                principal_id=authority.principal_id,
                runtime_profile_id=authority.runtime_profile_id,
                filename=path.name,
                claimed_media_type=None,
                chunks=chunks(),
                source_transport="internal_api",
                source_unique_id=request_id,
                occurred_at=occurred_at,
                original_filename=path.name,
            )
            try:
                return await self._record(
                    authority,
                    request_id=request_id,
                    body=caption or f"File: {path.name}",
                    occurred_at=occurred_at,
                    kind="file",
                    mode=PROACTIVE_ARTIFACT_MODE,
                    caption=caption,
                    artifact=staged,
                )
            except Exception:
                staged.discard()
                raise

    async def _record(
        self,
        authority: ProactivePublicationAuthority,
        *,
        request_id: str,
        body: str,
        occurred_at: datetime,
        kind: str,
        mode: str,
        caption: str | None,
        artifact: StagedArtifact | None,
    ) -> ProactivePublicationResult:
        self._validate_request(request_id, occurred_at)
        resolved = await self._resolve_authority(authority)
        stable_name = f"internal-api-publication:v1:{authority.runtime_profile_id}:{kind}:{request_id}"
        message_id = MessageId.derived(resolved.workshop_id, stable_name)
        payload = {
            "channel_id": authority.channel_id,
            "author_principal_id": resolved.agent_principal_id,
            "body": body,
        }
        metadata: dict[str, object] = {
            "source": "internal_api",
            "publication_kind": kind,
            "request_id": request_id,
            "runtime_profile_id": authority.runtime_profile_id,
        }
        if caption is not None:
            metadata["caption"] = caption
        envelope = EventEnvelope.create(
            event_id=EventId.derived(resolved.workshop_id, f"{stable_name}:event"),
            event_type=WorkshopEventType.MESSAGE_CREATED,
            event_version=1,
            workshop_id=resolved.workshop_id,
            aggregate_type="message",
            aggregate_id=message_id,
            actor_principal_id=resolved.agent_principal_id,
            occurred_at=occurred_at.astimezone(UTC),
            idempotency_key=stable_name,
            payload=payload,
            metadata=metadata,
        )
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            existing = await self._store.event_by_idempotency_key(stable_name)
            inserted = existing is None
            if existing is None:
                await self._store.append_in_transaction(envelope)
            elif existing.envelope.content_hash != envelope.content_hash:
                raise IdempotencyConflictError(f"Event identity {stable_name!r} was reused with different content")
            projection = CanonicalConversationProjection()
            await self._store.project_pending_in_transaction(projection)
            artifact_result = None
            if artifact is not None:
                artifact_result = await record_published_artifact_in_transaction(
                    self._store,
                    artifact.for_message(message_id),
                    storage_root=self._artifact_storage_root,
                )
            planning = await self._delivery_planner.plan_in_transaction(
                CanonicalDeliveryIntent(
                    message_id=message_id,
                    channel_id=authority.channel_id,
                    mode=mode,
                    purpose=NOTIFICATION_PURPOSE,
                    occurred_at=occurred_at,
                    content_kind="attachment" if artifact is not None else "text",
                )
            )
            deliveries = planning.deliveries
            prior_states = {inserted, *(delivery.inserted for delivery in deliveries)}
            if artifact_result is not None:
                prior_states.add(artifact_result.inserted)
            if len(prior_states) != 1:
                raise ProactivePublicationError(
                    "Canonical publication, artifact, and delivery requests do not share one prior state"
                )
            await self._store.project_pending_in_transaction(projection)
            await connection.commit()
            return ProactivePublicationResult(message_id, inserted, deliveries)
        except Exception:
            await connection.rollback()
            raise

    async def _resolve_authority(
        self,
        authority: ProactivePublicationAuthority,
    ) -> _ResolvedAuthority:
        if not isinstance(authority, ProactivePublicationAuthority):
            raise ValueError("authority must be canonical")
        async with self._store.connection.execute(
            "SELECT c.workshop_id, a.principal_id FROM channels c "
            "JOIN channel_memberships cm ON cm.channel_id = c.id "
            "AND cm.principal_id = ? AND cm.role = 'owner' "
            "JOIN principals human ON human.id = cm.principal_id AND human.kind = 'human' "
            "JOIN channel_agents ca ON ca.channel_id = c.id AND ca.agent_id = ? "
            "JOIN agents a ON a.id = ca.agent_id AND a.workshop_id = c.workshop_id "
            "JOIN principals agent ON agent.id = a.principal_id AND agent.kind = 'agent' "
            "JOIN channel_agent_runtime_assignments ra ON ra.channel_id = c.id "
            "AND ra.agent_id = a.id AND ra.runtime_profile_id = ? "
            "WHERE c.id = ? AND c.kind = 'direct'",
            (
                authority.principal_id,
                authority.agent_id,
                authority.runtime_profile_id,
                authority.channel_id,
            ),
        ) as cursor:
            rows = tuple(await cursor.fetchall())
        if len(rows) != 1:
            raise ProactivePublicationError("Canonical proactive publication authority is missing or ambiguous")
        return _ResolvedAuthority(
            workshop_id=WorkshopId(str(rows[0][0])),
            agent_principal_id=PrincipalId(str(rows[0][1])),
        )

    @staticmethod
    def _validate_request(request_id: str, occurred_at: datetime) -> None:
        if not isinstance(request_id, str) or not _REQUEST_ID_PATTERN.fullmatch(request_id):
            raise ValueError("request_id must be a bounded delivery identifier")
        if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
