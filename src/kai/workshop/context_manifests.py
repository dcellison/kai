"""Redacted, replay-stable context manifests for canonical run attempts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from kai.principal_documents import PrincipalDocument, PrincipalDocumentState
from kai.workshop.domain import (
    AgentId,
    ChannelId,
    EventEnvelope,
    EventId,
    PrincipalId,
    RunAttemptId,
    RunId,
    RuntimeProfileId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.run_execution_authority import RunExecutionClaim, RunExecutionSelection
from kai.workshop.store import WorkshopEventStore

if TYPE_CHECKING:
    from kai.backend import ContextAssemblyObservation

_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ContextSourceKind(StrEnum):
    HOST_POLICY = "host_policy"
    PRINCIPAL_POLICY = "principal_policy"
    AGENT_DEFINITION = "agent_definition"
    WORKSPACE_POLICY = "workspace_policy"
    PERSONAL_PREFERENCES = "personal_preferences"
    FILE_MEMORY = "file_memory"
    SEMANTIC_RECALL = "semantic_recall"
    CANONICAL_CONVERSATION = "canonical_conversation"
    CAPABILITY_GUIDANCE = "capability_guidance"
    ATTEMPT_AUTHORITY = "attempt_authority"
    CURRENT_INPUT = "current_input"
    PROVIDER_NATIVE = "provider_native"


CONTEXT_SOURCE_ORDER = tuple(ContextSourceKind)


class ContextOwnerKind(StrEnum):
    HOST = "host"
    PRINCIPAL = "principal"
    AGENT = "agent"
    WORKSPACE = "workspace"
    CHANNEL = "channel"
    ATTEMPT = "attempt"
    PROVIDER = "provider"


class ContextTrustClass(StrEnum):
    HOST_POLICY = "host_policy"
    PRINCIPAL_POLICY = "principal_policy"
    AGENT_DEFINITION = "agent_definition"
    WORKSPACE_POLICY = "workspace_policy"
    UNTRUSTED_DATA = "untrusted_data"
    CAPABILITY_METADATA = "capability_metadata"
    ATTEMPT_AUTHORITY = "attempt_authority"
    CURRENT_INPUT = "current_input"
    PROVIDER_CONTROLLED = "provider_controlled"


class ContextAuthorityClass(StrEnum):
    HOST = "host"
    PRINCIPAL = "principal"
    AGENT_OWNER = "agent_owner"
    WORKSPACE_OWNER = "workspace_owner"
    CANONICAL_DATA = "canonical_data"
    CAPABILITY_CONTRACT = "capability_contract"
    ATTEMPT = "attempt"
    REQUESTER = "requester"
    PROVIDER = "provider"


class ContextRefreshClass(StrEnum):
    DEPLOYMENT = "deployment"
    SOURCE_REVISION = "source_revision"
    PROVIDER_SESSION = "provider_session"
    PER_ATTEMPT = "per_attempt"
    PER_TURN = "per_turn"
    PROVIDER_CONTROLLED = "provider_controlled"


class ContextDeliveryRole(StrEnum):
    NATIVE_INSTRUCTION = "native_instruction"
    SESSION_CONTEXT = "session_context"
    TURN_CONTEXT = "turn_context"
    UNTRUSTED_CONTEXT = "untrusted_context"
    ATTEMPT_CONTEXT = "attempt_context"
    CURRENT_INPUT = "current_input"
    PROVIDER_NATIVE = "provider_native"


class ContextSourceState(StrEnum):
    NEWLY_DELIVERED = "newly_delivered"
    RETAINED = "retained"
    OMITTED = "omitted"
    UNAVAILABLE = "unavailable"
    PROVIDER_CONTROLLED = "provider_controlled"


@dataclass(frozen=True, slots=True)
class ContextSourceDescriptor:
    """One content-free description of a logical context source."""

    kind: ContextSourceKind
    owner_kind: ContextOwnerKind
    owner_id: str | None
    scope: str
    trust_class: ContextTrustClass
    authority_class: ContextAuthorityClass
    refresh_class: ContextRefreshClass
    delivery_role: ContextDeliveryRole
    state: ContextSourceState
    reason: str
    revision: str | None = None
    history_boundary: int | None = None
    delivery_shape: str = "metadata"

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.scope, "scope"),
            (self.reason, "reason"),
            (self.delivery_shape, "delivery_shape"),
        ):
            if not isinstance(value, str) or _CODE.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a bounded lowercase identifier")
        if self.owner_id is not None and (
            not isinstance(self.owner_id, str) or not self.owner_id or len(self.owner_id) > 128
        ):
            raise ValueError("owner_id must be a bounded identifier")
        if self.revision is not None and _REVISION.fullmatch(self.revision) is None:
            raise ValueError("revision must be a bounded revision identifier")
        if self.history_boundary is not None and (
            not isinstance(self.history_boundary, int)
            or isinstance(self.history_boundary, bool)
            or self.history_boundary < 0
        ):
            raise ValueError("history_boundary must be a non-negative integer")

    def payload(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "owner_kind": self.owner_kind.value,
            "owner_id": self.owner_id,
            "scope": self.scope,
            "trust_class": self.trust_class.value,
            "authority_class": self.authority_class.value,
            "refresh_class": self.refresh_class.value,
            "delivery_role": self.delivery_role.value,
            "state": self.state.value,
            "reason": self.reason,
            "revision": self.revision,
            "history_boundary": self.history_boundary,
            "delivery_shape": self.delivery_shape,
        }

    @classmethod
    def from_payload(cls, value: object) -> ContextSourceDescriptor:
        if not isinstance(value, dict) or set(value) != {
            "kind",
            "owner_kind",
            "owner_id",
            "scope",
            "trust_class",
            "authority_class",
            "refresh_class",
            "delivery_role",
            "state",
            "reason",
            "revision",
            "history_boundary",
            "delivery_shape",
        }:
            raise ValueError("Context source descriptor has an invalid shape")
        owner_id = value["owner_id"]
        revision = value["revision"]
        boundary = value["history_boundary"]
        return cls(
            kind=ContextSourceKind(str(value["kind"])),
            owner_kind=ContextOwnerKind(str(value["owner_kind"])),
            owner_id=None if owner_id is None else str(owner_id),
            scope=str(value["scope"]),
            trust_class=ContextTrustClass(str(value["trust_class"])),
            authority_class=ContextAuthorityClass(str(value["authority_class"])),
            refresh_class=ContextRefreshClass(str(value["refresh_class"])),
            delivery_role=ContextDeliveryRole(str(value["delivery_role"])),
            state=ContextSourceState(str(value["state"])),
            reason=str(value["reason"]),
            revision=None if revision is None else str(revision),
            history_boundary=None if boundary is None else int(boundary),
            delivery_shape=str(value["delivery_shape"]),
        )


@dataclass(frozen=True, slots=True)
class ContextManifestDraft:
    runtime_profile_id: RuntimeProfileId
    selection: RunExecutionSelection
    workspace_kind: str
    workspace_digest: str | None
    provider_session_revision: str | None
    sources: tuple[ContextSourceDescriptor, ...]

    def __post_init__(self) -> None:
        if self.workspace_kind not in {"home", "foreign", "unknown"}:
            raise ValueError("workspace_kind is invalid")
        for value, field_name in (
            (self.workspace_digest, "workspace_digest"),
            (self.provider_session_revision, "provider_session_revision"),
        ):
            if value is not None and _SHA256.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a SHA-256 digest")
        if tuple(source.kind for source in self.sources) != CONTEXT_SOURCE_ORDER:
            raise ValueError("sources must contain every context kind exactly once in canonical order")


@dataclass(frozen=True, slots=True)
class RunContextManifest:
    attempt_id: RunAttemptId
    run_id: RunId
    workshop_id: WorkshopId
    channel_id: ChannelId
    requested_by_principal_id: PrincipalId
    agent_id: AgentId
    draft: ContextManifestDraft
    manifest_sha256: str
    created_at: datetime
    created_event_position: int


def content_digest(value: object) -> str:
    """Return a deterministic digest without retaining the source material."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_digest(payload: dict[str, object]) -> str:
    """Digest the redacted manifest body in its canonical JSON form."""
    return content_digest(payload)


def _timestamp(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("occurred_at must be timezone-aware")
    return value.astimezone(UTC)


def _draft_payload(draft: ContextManifestDraft) -> dict[str, object]:
    return {
        "runtime_profile_id": str(draft.runtime_profile_id),
        "backend": draft.selection.backend,
        "provider": draft.selection.provider,
        "model": draft.selection.model,
        "workspace_kind": draft.workspace_kind,
        "workspace_digest": draft.workspace_digest,
        "provider_session_revision": draft.provider_session_revision,
        "sources": [source.payload() for source in draft.sources],
    }


def _manifest_body(
    *,
    run_id: RunId,
    channel_id: ChannelId,
    requested_by_principal_id: PrincipalId,
    agent_id: AgentId,
    draft: ContextManifestDraft,
) -> dict[str, object]:
    return {
        "run_id": str(run_id),
        "channel_id": str(channel_id),
        "requested_by_principal_id": str(requested_by_principal_id),
        "agent_id": str(agent_id),
        **_draft_payload(draft),
    }


def validate_manifest_sources(value: object) -> tuple[ContextSourceDescriptor, ...]:
    if not isinstance(value, list):
        raise ValueError("Context manifest sources must be a list")
    sources = tuple(ContextSourceDescriptor.from_payload(item) for item in value)
    if tuple(source.kind for source in sources) != CONTEXT_SOURCE_ORDER:
        raise ValueError("Context manifest sources are incomplete or out of canonical order")
    return sources


def validate_manifest_event_payload(payload: dict[str, Any]) -> tuple[ContextManifestDraft, str]:
    expected = {
        "run_id",
        "channel_id",
        "requested_by_principal_id",
        "agent_id",
        "runtime_profile_id",
        "backend",
        "provider",
        "model",
        "workspace_kind",
        "workspace_digest",
        "provider_session_revision",
        "sources",
        "manifest_sha256",
    }
    if set(payload) != expected:
        raise ValueError("Context manifest event payload has an invalid shape")
    provider = payload["provider"]
    selection = RunExecutionSelection(
        backend=str(payload["backend"]),
        provider=None if provider is None else str(provider),
        model=str(payload["model"]),
    )
    draft = ContextManifestDraft(
        runtime_profile_id=RuntimeProfileId(str(payload["runtime_profile_id"])),
        selection=selection,
        workspace_kind=str(payload["workspace_kind"]),
        workspace_digest=None if payload["workspace_digest"] is None else str(payload["workspace_digest"]),
        provider_session_revision=(
            None if payload["provider_session_revision"] is None else str(payload["provider_session_revision"])
        ),
        sources=validate_manifest_sources(payload["sources"]),
    )
    digest = str(payload["manifest_sha256"])
    if _SHA256.fullmatch(digest) is None:
        raise ValueError("Context manifest digest is invalid")
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if manifest_digest(body) != digest:
        raise ValueError("Context manifest digest does not match its redacted body")
    return draft, digest


def build_context_manifest_draft(
    *,
    runtime_profile_id: RuntimeProfileId,
    selection: RunExecutionSelection,
    workspace_kind: str,
    workspace_digest: str | None,
    provider_session_revision: str | None,
    principal_id: PrincipalId,
    agent_id: AgentId,
    agent_revision: str,
    agent_context_digest: str,
    channel_id: ChannelId,
    history_boundary: int,
    history_digest: str,
    current_input_digest: str,
    attempt_id: RunAttemptId,
    attempt_authority_revision: str | None,
    observation: ContextAssemblyObservation,
) -> ContextManifestDraft:
    """Build the fixed source vocabulary from facts observed at dispatch."""
    dispatch_reached = observation.provider_dispatch_reached is True
    dispatch_reason = (
        "provider_dispatch_reached"
        if dispatch_reached
        else "dispatch_not_reached"
        if observation.provider_dispatch_reached is False
        else "dispatch_state_unknown"
    )
    session_state = (
        ContextSourceState.UNAVAILABLE
        if not dispatch_reached
        else ContextSourceState.NEWLY_DELIVERED
        if observation.session_context_delivered
        else ContextSourceState.RETAINED
    )
    session_reason = (
        dispatch_reason
        if not dispatch_reached
        else "fresh_provider_session"
        if observation.session_context_delivered
        else "live_provider_session"
    )
    granular_session_state = ContextSourceState.UNAVAILABLE if observation.session_context_delivered else session_state
    granular_session_reason = (
        "bootstrap_source_not_individually_observable" if observation.session_context_delivered else session_reason
    )
    workspace_policy_state = (
        ContextSourceState.NEWLY_DELIVERED
        if dispatch_reached and observation.workspace_reminder_delivered
        else ContextSourceState.UNAVAILABLE
        if not dispatch_reached or observation.session_context_delivered
        else ContextSourceState.RETAINED
    )
    workspace_policy_reason = (
        dispatch_reason
        if not dispatch_reached
        else "foreign_workspace_reminder"
        if observation.workspace_reminder_delivered
        else granular_session_reason
    )
    semantic_state = (
        ContextSourceState.NEWLY_DELIVERED if observation.semantic_recall_delivered else ContextSourceState.OMITTED
    )
    semantic_reason = _safe_code(observation.semantic_recall_reason, fallback="no_matches")
    authority_state = (
        ContextSourceState.NEWLY_DELIVERED
        if dispatch_reached and attempt_authority_revision is not None
        else ContextSourceState.UNAVAILABLE
        if not dispatch_reached and attempt_authority_revision is not None
        else ContextSourceState.OMITTED
    )
    authority_reason = (
        "grant_delivered"
        if dispatch_reached and attempt_authority_revision is not None
        else dispatch_reason
        if attempt_authority_revision is not None
        else "no_grant"
    )
    effective_provider_session_revision = (
        None
        if not dispatch_reached
        else content_digest(
            {
                "attempt_id": str(attempt_id),
                "bootstrap_revision": observation.session_context_revision,
            }
        )
        if observation.session_context_delivered
        else provider_session_revision
    )
    session_revision = observation.session_context_revision or effective_provider_session_revision

    def principal_document_facts(
        document: PrincipalDocument | None,
        *,
        absent_reason: str,
    ) -> tuple[ContextSourceState, str, str | None]:
        if observation.principal_documents is None:
            return granular_session_state, granular_session_reason, session_revision
        if document is None:
            return ContextSourceState.OMITTED, absent_reason, None
        reason = _safe_code(document.reason, fallback=document.state.value)
        if document.state is PrincipalDocumentState.PRESENT:
            return (
                ContextSourceState.NEWLY_DELIVERED if dispatch_reached else ContextSourceState.UNAVAILABLE,
                reason if dispatch_reached else "dispatch_not_reached",
                document.revision,
            )
        if document.state is PrincipalDocumentState.MISSING and dispatch_reached:
            return ContextSourceState.OMITTED, reason, None
        return ContextSourceState.UNAVAILABLE, reason, None

    document_report = observation.principal_documents
    principal_policy_facts = principal_document_facts(
        document_report.policy if document_report is not None else None,
        absent_reason="private_document_not_applicable",
    )
    preferences_facts = principal_document_facts(
        document_report.preferences if document_report is not None else None,
        absent_reason="private_document_not_applicable",
    )
    file_memory_facts = principal_document_facts(
        document_report.file_memory if document_report is not None else None,
        absent_reason="semantic_memory_enabled_or_shared",
    )
    sources = (
        ContextSourceDescriptor(
            ContextSourceKind.HOST_POLICY,
            ContextOwnerKind.HOST,
            None,
            "service",
            ContextTrustClass.HOST_POLICY,
            ContextAuthorityClass.HOST,
            ContextRefreshClass.PER_TURN,
            ContextDeliveryRole.TURN_CONTEXT,
            ContextSourceState.NEWLY_DELIVERED if dispatch_reached else ContextSourceState.UNAVAILABLE,
            "host_turn_contract" if dispatch_reached else dispatch_reason,
            revision="context_contract_v1",
            delivery_shape="inline_host_marker",
        ),
        ContextSourceDescriptor(
            ContextSourceKind.PRINCIPAL_POLICY,
            ContextOwnerKind.PRINCIPAL,
            str(principal_id),
            "principal",
            ContextTrustClass.PRINCIPAL_POLICY,
            ContextAuthorityClass.PRINCIPAL,
            ContextRefreshClass.PROVIDER_SESSION,
            ContextDeliveryRole.SESSION_CONTEXT,
            principal_policy_facts[0],
            principal_policy_facts[1],
            revision=principal_policy_facts[2],
            delivery_shape="inline_verified_document",
        ),
        ContextSourceDescriptor(
            ContextSourceKind.AGENT_DEFINITION,
            ContextOwnerKind.AGENT,
            str(agent_id),
            "agent",
            ContextTrustClass.AGENT_DEFINITION,
            ContextAuthorityClass.AGENT_OWNER,
            ContextRefreshClass.SOURCE_REVISION,
            ContextDeliveryRole.TURN_CONTEXT,
            ContextSourceState.NEWLY_DELIVERED if dispatch_reached else ContextSourceState.OMITTED,
            "run_bound_revision" if dispatch_reached else dispatch_reason,
            revision=f"{agent_revision}:{agent_context_digest}",
            delivery_shape="inline_redacted_block",
        ),
        ContextSourceDescriptor(
            ContextSourceKind.WORKSPACE_POLICY,
            ContextOwnerKind.WORKSPACE,
            workspace_digest,
            "workspace",
            ContextTrustClass.WORKSPACE_POLICY,
            ContextAuthorityClass.WORKSPACE_OWNER,
            ContextRefreshClass.PER_TURN,
            ContextDeliveryRole.SESSION_CONTEXT,
            workspace_policy_state,
            workspace_policy_reason,
            revision=observation.workspace_reminder_revision or session_revision,
            delivery_shape="bootstrap_or_reminder",
        ),
        ContextSourceDescriptor(
            ContextSourceKind.PERSONAL_PREFERENCES,
            ContextOwnerKind.PRINCIPAL,
            str(principal_id),
            "principal",
            ContextTrustClass.UNTRUSTED_DATA,
            ContextAuthorityClass.PRINCIPAL,
            ContextRefreshClass.PROVIDER_SESSION,
            ContextDeliveryRole.UNTRUSTED_CONTEXT,
            preferences_facts[0],
            preferences_facts[1],
            revision=preferences_facts[2],
            delivery_shape="inline_verified_document",
        ),
        ContextSourceDescriptor(
            ContextSourceKind.FILE_MEMORY,
            ContextOwnerKind.PRINCIPAL,
            str(principal_id),
            "principal",
            ContextTrustClass.UNTRUSTED_DATA,
            ContextAuthorityClass.PRINCIPAL,
            ContextRefreshClass.PROVIDER_SESSION,
            ContextDeliveryRole.UNTRUSTED_CONTEXT,
            file_memory_facts[0],
            file_memory_facts[1],
            revision=file_memory_facts[2],
            delivery_shape="randomized_untrusted_block",
        ),
        ContextSourceDescriptor(
            ContextSourceKind.SEMANTIC_RECALL,
            ContextOwnerKind.PRINCIPAL,
            str(principal_id),
            "principal",
            ContextTrustClass.UNTRUSTED_DATA,
            ContextAuthorityClass.CANONICAL_DATA,
            ContextRefreshClass.PER_TURN,
            ContextDeliveryRole.UNTRUSTED_CONTEXT,
            semantic_state,
            semantic_reason,
            revision=observation.semantic_recall_revision,
            delivery_shape="scoped_recall_block",
        ),
        ContextSourceDescriptor(
            ContextSourceKind.CANONICAL_CONVERSATION,
            ContextOwnerKind.CHANNEL,
            str(channel_id),
            "channel",
            ContextTrustClass.UNTRUSTED_DATA,
            ContextAuthorityClass.CANONICAL_DATA,
            ContextRefreshClass.PROVIDER_SESSION,
            ContextDeliveryRole.UNTRUSTED_CONTEXT,
            session_state,
            session_reason,
            revision=history_digest,
            history_boundary=history_boundary,
            delivery_shape="bounded_canonical_timeline",
        ),
        ContextSourceDescriptor(
            ContextSourceKind.CAPABILITY_GUIDANCE,
            ContextOwnerKind.HOST,
            None,
            "service",
            ContextTrustClass.CAPABILITY_METADATA,
            ContextAuthorityClass.CAPABILITY_CONTRACT,
            ContextRefreshClass.PROVIDER_SESSION,
            ContextDeliveryRole.SESSION_CONTEXT,
            granular_session_state,
            granular_session_reason,
            revision=session_revision or "capability_guidance_v1",
            delivery_shape="generated_guidance",
        ),
        ContextSourceDescriptor(
            ContextSourceKind.ATTEMPT_AUTHORITY,
            ContextOwnerKind.ATTEMPT,
            str(attempt_id),
            "attempt",
            ContextTrustClass.ATTEMPT_AUTHORITY,
            ContextAuthorityClass.ATTEMPT,
            ContextRefreshClass.PER_ATTEMPT,
            ContextDeliveryRole.ATTEMPT_CONTEXT,
            authority_state,
            authority_reason,
            revision=attempt_authority_revision,
            delivery_shape="proof_excluded",
        ),
        ContextSourceDescriptor(
            ContextSourceKind.CURRENT_INPUT,
            ContextOwnerKind.PRINCIPAL,
            str(principal_id),
            "attempt",
            ContextTrustClass.CURRENT_INPUT,
            ContextAuthorityClass.REQUESTER,
            ContextRefreshClass.PER_TURN,
            ContextDeliveryRole.CURRENT_INPUT,
            ContextSourceState.NEWLY_DELIVERED if dispatch_reached else ContextSourceState.OMITTED,
            "accepted_input" if dispatch_reached else dispatch_reason,
            revision=current_input_digest,
            delivery_shape="native_user_input",
        ),
        ContextSourceDescriptor(
            ContextSourceKind.PROVIDER_NATIVE,
            ContextOwnerKind.PROVIDER,
            selection.provider,
            "provider",
            ContextTrustClass.PROVIDER_CONTROLLED,
            ContextAuthorityClass.PROVIDER,
            ContextRefreshClass.PROVIDER_CONTROLLED,
            ContextDeliveryRole.PROVIDER_NATIVE,
            (
                ContextSourceState.UNAVAILABLE
                if observation.provider_dispatch_reached is False
                else ContextSourceState.PROVIDER_CONTROLLED
            ),
            (
                "dispatch_not_reached"
                if observation.provider_dispatch_reached is False
                else "ambient_discovery_disabled"
                if observation.ambient_context_discovery_enabled is False
                else "ambient_discovery_enabled"
                if observation.ambient_context_discovery_enabled is True
                else "not_observable"
            ),
            delivery_shape=(
                "provider_managed_ambient_disabled"
                if observation.ambient_context_discovery_enabled is False
                else "provider_managed_unknown"
            ),
        ),
    )
    return ContextManifestDraft(
        runtime_profile_id=runtime_profile_id,
        selection=selection,
        workspace_kind=workspace_kind,
        workspace_digest=workspace_digest,
        provider_session_revision=effective_provider_session_revision,
        sources=sources,
    )


def _safe_code(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")[:64]
    return normalized if normalized and _CODE.fullmatch(normalized) else fallback


class WorkshopContextManifestService:
    """Persist and read immutable manifests at the run-attempt boundary."""

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store

    async def available(self) -> bool:
        async with self._store.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'run_context_manifests'"
        ) as cursor:
            return await cursor.fetchone() is not None

    async def record(
        self,
        claim: RunExecutionClaim,
        draft: ContextManifestDraft,
        *,
        occurred_at: datetime,
    ) -> RunContextManifest:
        if not isinstance(claim, RunExecutionClaim) or not isinstance(draft, ContextManifestDraft):
            raise TypeError("claim and draft must be canonical context-manifest values")
        occurred_at = _timestamp(occurred_at)
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            projection = CanonicalConversationProjection()
            await self._store.project_pending_in_transaction(projection)
            async with connection.execute(
                "SELECT ra.run_id, ra.owner_id, ra.fence_token, ra.backend, ra.provider, ra.model, "
                "r.workshop_id, r.channel_id, r.requested_by_principal_id, r.agent_id, "
                "r.runtime_profile_id, a.principal_id FROM run_attempts ra "
                "JOIN runs r ON r.id = ra.run_id JOIN agents a ON a.id = r.agent_id "
                "WHERE ra.id = ?",
                (claim.attempt_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None or (
                str(row[0]) != str(claim.run_id)
                or str(row[1]) != str(claim.owner_id)
                or int(row[2]) != claim.fence_token
                or str(row[3]) != draft.selection.backend
                or (None if row[4] is None else str(row[4])) != draft.selection.provider
                or str(row[5]) != draft.selection.model
                or str(row[10]) != str(draft.runtime_profile_id)
            ):
                raise ValueError("Context manifest does not match its fenced run attempt")
            workshop_id = WorkshopId(str(row[6]))
            channel_id = ChannelId(str(row[7]))
            requester = PrincipalId(str(row[8]))
            agent_id = AgentId(str(row[9]))
            body = _manifest_body(
                run_id=claim.run_id,
                channel_id=channel_id,
                requested_by_principal_id=requester,
                agent_id=agent_id,
                draft=draft,
            )
            digest = manifest_digest(body)
            existing = await self.load_attempt(claim.attempt_id)
            if existing is not None:
                if existing.manifest_sha256 != digest:
                    raise ValueError("Context manifest attempt already has different immutable facts")
                await connection.commit()
                return existing
            event = EventEnvelope.create(
                event_id=EventId.derived(claim.attempt_id, "context-manifest"),
                event_type=WorkshopEventType.RUN_ATTEMPT_CONTEXT_MANIFEST_RECORDED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="run_attempt",
                aggregate_id=claim.attempt_id,
                actor_principal_id=PrincipalId(str(row[11])),
                occurred_at=occurred_at,
                idempotency_key=f"workshop-context-manifest:v1:{claim.attempt_id}",
                payload={**body, "manifest_sha256": digest},
                metadata={"source": "workshop_context_manifest"},
            )
            await self._store.append_in_transaction(event)
            await self._store.project_pending_in_transaction(projection)
            manifest = await self.load_attempt(claim.attempt_id)
            if manifest is None:
                raise RuntimeError("Context manifest projection did not persist the recorded event")
            await connection.commit()
            return manifest
        except Exception:
            await connection.rollback()
            raise

    async def load_attempt(self, attempt_id: RunAttemptId) -> RunContextManifest | None:
        async with self._store.connection.execute(
            "SELECT attempt_id, run_id, workshop_id, channel_id, requested_by_principal_id, agent_id, "
            "runtime_profile_id, backend, provider, model, workspace_kind, workspace_digest, "
            "provider_session_revision, sources_json, manifest_sha256, created_at, created_event_position "
            "FROM run_context_manifests WHERE attempt_id = ?",
            (attempt_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return None if row is None else _manifest_from_row(row)

    async def load_run(self, run_id: RunId) -> tuple[RunContextManifest, ...]:
        async with self._store.connection.execute(
            "SELECT attempt_id, run_id, workshop_id, channel_id, requested_by_principal_id, agent_id, "
            "runtime_profile_id, backend, provider, model, workspace_kind, workspace_digest, "
            "provider_session_revision, sources_json, manifest_sha256, created_at, created_event_position "
            "FROM run_context_manifests WHERE run_id = ? ORDER BY created_event_position, attempt_id",
            (run_id,),
        ) as cursor:
            rows = list(await cursor.fetchall())
        return tuple(_manifest_from_row(row) for row in rows)


def _manifest_from_row(row: Sequence[Any]) -> RunContextManifest:
    values = tuple(row)
    provider = values[8]
    sources = validate_manifest_sources(json.loads(str(values[13])))
    draft = ContextManifestDraft(
        runtime_profile_id=RuntimeProfileId(str(values[6])),
        selection=RunExecutionSelection(
            backend=str(values[7]),
            provider=None if provider is None else str(provider),
            model=str(values[9]),
        ),
        workspace_kind=str(values[10]),
        workspace_digest=None if values[11] is None else str(values[11]),
        provider_session_revision=None if values[12] is None else str(values[12]),
        sources=sources,
    )
    return RunContextManifest(
        attempt_id=RunAttemptId(str(values[0])),
        run_id=RunId(str(values[1])),
        workshop_id=WorkshopId(str(values[2])),
        channel_id=ChannelId(str(values[3])),
        requested_by_principal_id=PrincipalId(str(values[4])),
        agent_id=AgentId(str(values[5])),
        draft=draft,
        manifest_sha256=str(values[14]),
        created_at=datetime.fromisoformat(str(values[15]).replace("Z", "+00:00")).astimezone(UTC),
        created_event_position=int(values[16]),
    )
