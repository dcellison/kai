"""Canonical metadata and provenance for Workshop inbound artifacts."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from kai.workshop.domain import (
    ArtifactId,
    ChannelId,
    EventEnvelope,
    EventId,
    MessageId,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import AppendResult, WorkshopEventStore

_ARTIFACT_KINDS = frozenset({"photo", "document", "voice"})
_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_MEDIA_TYPE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")
_MAX_SOURCE_UNIQUE_ID_LENGTH = 512
_MAX_FILENAME_LENGTH = 255


class ArtifactMessageNotFoundError(LookupError):
    """The target is not one canonical inbound human message."""


class ArtifactStorageBoundaryError(ValueError):
    """The artifact path is absent or escapes the operator-selected root."""


@dataclass(frozen=True, slots=True)
class InboundArtifact:
    message_id: MessageId
    kind: str
    media_type: str
    storage_path: Path
    source_transport: str
    # Stable provider identifier that cannot retrieve content (for Telegram,
    # this is file_unique_id, never the downloadable file_id capability).
    source_unique_id: str
    occurred_at: datetime
    original_filename: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.message_id, MessageId):
            raise ValueError("message_id must be a MessageId")
        if self.kind not in _ARTIFACT_KINDS:
            raise ValueError("kind must be photo, document, or voice")
        if not isinstance(self.media_type, str) or not _MEDIA_TYPE_PATTERN.fullmatch(self.media_type):
            raise ValueError("media_type must be a lowercase MIME type")
        if not isinstance(self.storage_path, Path):
            raise ValueError("storage_path must be a Path")
        if not self.storage_path.is_absolute():
            raise ValueError("storage_path must be absolute")
        if not _IDENTIFIER_PATTERN.fullmatch(self.source_transport):
            raise ValueError("source_transport must be a lowercase identifier")
        if (
            not isinstance(self.source_unique_id, str)
            or not self.source_unique_id
            or self.source_unique_id != self.source_unique_id.strip()
            or len(self.source_unique_id) > _MAX_SOURCE_UNIQUE_ID_LENGTH
        ):
            raise ValueError("source_unique_id must be a bounded non-empty string")
        if self.original_filename is not None:
            filename = self.original_filename
            if (
                not isinstance(filename, str)
                or not filename
                or filename != filename.strip()
                or len(filename) > _MAX_FILENAME_LENGTH
                or filename in {".", ".."}
                or "/" in filename
                or "\\" in filename
                or "\0" in filename
            ):
                raise ValueError("original_filename must be a bounded basename or None")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class _ResolvedArtifactMessage:
    workshop_id: WorkshopId
    channel_id: ChannelId
    created_by_principal_id: PrincipalId


async def _resolve_inbound_message(
    store: WorkshopEventStore,
    message_id: MessageId,
) -> _ResolvedArtifactMessage:
    async with store.connection.execute(
        "SELECT c.workshop_id, m.channel_id, m.author_principal_id "
        "FROM messages m "
        "JOIN channels c ON c.id = m.channel_id "
        "JOIN principals p ON p.id = m.author_principal_id AND p.kind = 'human' "
        "WHERE m.id = ?",
        (message_id,),
    ) as cursor:
        rows = list(await cursor.fetchall())
    if len(rows) != 1:
        raise ArtifactMessageNotFoundError("Inbound artifact message was not found")
    return _ResolvedArtifactMessage(
        workshop_id=WorkshopId(str(rows[0][0])),
        channel_id=ChannelId(str(rows[0][1])),
        created_by_principal_id=PrincipalId(str(rows[0][2])),
    )


def _resolve_storage_path(storage_path: Path, storage_root: Path) -> Path:
    try:
        root = storage_root.resolve(strict=True)
        resolved = storage_path.resolve(strict=True)
    except OSError as exc:
        raise ArtifactStorageBoundaryError("Artifact storage path is unavailable") from exc
    if not root.is_dir() or not resolved.is_file() or not resolved.is_relative_to(root):
        raise ArtifactStorageBoundaryError("Artifact storage path is outside its allowed root")
    return resolved


def _file_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_size = 0
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                byte_size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactStorageBoundaryError("Artifact content is unavailable") from exc
    return byte_size, digest.hexdigest()


def _stable_source_token(artifact: InboundArtifact) -> str:
    identity = "\0".join(
        (
            artifact.message_id,
            artifact.source_transport,
            artifact.source_unique_id,
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


async def record_inbound_artifact(
    store: WorkshopEventStore,
    artifact: InboundArtifact,
    *,
    storage_root: Path,
) -> AppendResult:
    """Append and project metadata for one durable inbound media object."""
    if not isinstance(storage_root, Path):
        raise ValueError("storage_root must be a Path")
    resolved_message = await _resolve_inbound_message(store, artifact.message_id)
    resolved_path = _resolve_storage_path(artifact.storage_path, storage_root)
    byte_size, content_sha256 = _file_identity(resolved_path)
    token = _stable_source_token(artifact)

    result = await store.append(
        EventEnvelope.create(
            event_id=EventId.derived(resolved_message.workshop_id, f"artifact-event:{token}"),
            event_type=WorkshopEventType.ARTIFACT_CREATED,
            event_version=1,
            workshop_id=resolved_message.workshop_id,
            aggregate_type="artifact",
            aggregate_id=ArtifactId.derived(resolved_message.workshop_id, f"artifact:{token}"),
            actor_principal_id=resolved_message.created_by_principal_id,
            occurred_at=artifact.occurred_at,
            idempotency_key=f"workshop-artifact:v1:{token}",
            payload={
                "channel_id": resolved_message.channel_id,
                "message_id": artifact.message_id,
                "created_by_principal_id": resolved_message.created_by_principal_id,
                "kind": artifact.kind,
                "media_type": artifact.media_type,
                "byte_size": byte_size,
                "content_sha256": content_sha256,
                "original_filename": artifact.original_filename,
                "storage_path": str(resolved_path),
                "source_transport": artifact.source_transport,
                "source_unique_id": artifact.source_unique_id,
            },
            metadata={"source": "artifact_shadow"},
        )
    )
    await store.project_pending(CanonicalConversationProjection())
    return result
