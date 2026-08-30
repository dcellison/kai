"""Canonical metadata and provenance for Workshop inbound artifacts."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
from collections.abc import AsyncIterable
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
    RuntimeProfileId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.runtime_assignments import resolve_channel_runtime_profile
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileRegistry
from kai.workshop.storage_namespaces import WorkshopPrincipalStorageRegistry
from kai.workshop.store import AppendResult, WorkshopEventStore

_ARTIFACT_KINDS = frozenset({"photo", "document", "voice"})
_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_MEDIA_TYPE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")
_MAX_SOURCE_UNIQUE_ID_LENGTH = 512
_MAX_FILENAME_LENGTH = 255
MAX_ARTIFACT_BYTES = 20 * 1024 * 1024
MAX_INLINE_TEXT_BYTES = 200_000
_UPLOAD_ROOT_MODE = 0o711
_UPLOAD_FILE_MODE = 0o600
_TEXT_EXTENSIONS = frozenset(
    {
        ".bash",
        ".c",
        ".cfg",
        ".conf",
        ".cpp",
        ".css",
        ".csv",
        ".dockerfile",
        ".env",
        ".erl",
        ".ex",
        ".exs",
        ".fish",
        ".gitignore",
        ".go",
        ".h",
        ".hpp",
        ".htm",
        ".html",
        ".ini",
        ".java",
        ".js",
        ".json",
        ".jsx",
        ".kt",
        ".log",
        ".lua",
        ".makefile",
        ".md",
        ".php",
        ".pl",
        ".py",
        ".r",
        ".rb",
        ".rs",
        ".rst",
        ".sh",
        ".sql",
        ".swift",
        ".toml",
        ".ts",
        ".tsv",
        ".tsx",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
        ".zsh",
    }
)
_IMAGE_MEDIA_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
_AUDIO_MEDIA_TYPES = {
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".oga": "audio/ogg",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
}


class ArtifactMessageNotFoundError(LookupError):
    """The target is not one canonical inbound human message."""


class ArtifactStorageBoundaryError(ValueError):
    """The artifact path is absent or escapes the operator-selected root."""


class ArtifactTooLargeError(ValueError):
    """An inbound artifact exceeded the configured hard limit."""


class ArtifactAccessDeniedError(PermissionError):
    """A principal cannot resolve the requested canonical artifact."""


@dataclass(frozen=True, slots=True)
class ArtifactSummary:
    artifact_id: ArtifactId
    message_id: MessageId
    channel_id: ChannelId
    kind: str
    media_type: str
    byte_size: int
    content_sha256: str
    original_filename: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    summary: ArtifactSummary
    storage_path: Path


@dataclass(frozen=True, slots=True)
class StagedArtifact:
    kind: str
    media_type: str
    storage_path: Path
    source_transport: str
    source_unique_id: str
    occurred_at: datetime
    original_filename: str | None
    created_for_attempt: bool = False

    def discard(self) -> None:
        """Remove content created by this attempt before canonical acceptance."""
        if self.created_for_attempt:
            self.storage_path.unlink(missing_ok=True)

    def for_message(self, message_id: MessageId) -> InboundArtifact:
        return InboundArtifact(
            message_id=message_id,
            kind=self.kind,
            media_type=self.media_type,
            storage_path=self.storage_path,
            source_transport=self.source_transport,
            source_unique_id=self.source_unique_id,
            occurred_at=self.occurred_at,
            original_filename=self.original_filename,
        )


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


async def _resolve_artifact_message(
    store: WorkshopEventStore,
    message_id: MessageId,
    *,
    author_kind: str,
) -> _ResolvedArtifactMessage:
    if author_kind not in {"human", "agent"}:
        raise ValueError("author_kind must be human or agent")
    async with store.connection.execute(
        "SELECT c.workshop_id, m.channel_id, m.author_principal_id "
        "FROM messages m "
        "JOIN channels c ON c.id = m.channel_id AND c.archived_at IS NULL "
        "JOIN principals p ON p.id = m.author_principal_id AND p.kind = ? "
        "WHERE m.id = ?",
        (author_kind, message_id),
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


def canonical_artifact_filename(filename: str | None) -> str | None:
    """Return a bounded cross-platform basename suitable for provenance."""
    if not isinstance(filename, str):
        return None
    normalized = Path(filename.replace("\\", "/")).name.strip()
    if (
        not normalized
        or normalized in {".", ".."}
        or len(normalized) > _MAX_FILENAME_LENGTH
        or any(ord(character) < 32 or ord(character) == 127 for character in normalized)
    ):
        return None
    return normalized


def canonical_artifact_media_type(filename: str | None, claimed_media_type: str | None) -> str:
    """Derive stable MIME metadata without trusting a client claim alone."""
    safe_filename = canonical_artifact_filename(filename)
    suffix = Path(safe_filename or "").suffix.lower()
    if suffix in _IMAGE_MEDIA_TYPES:
        return _IMAGE_MEDIA_TYPES[suffix]
    if suffix in _AUDIO_MEDIA_TYPES:
        return _AUDIO_MEDIA_TYPES[suffix]
    normalized_claim: str | None = None
    if isinstance(claimed_media_type, str):
        normalized = claimed_media_type.split(";", 1)[0].strip().lower()
        if _MEDIA_TYPE_PATTERN.fullmatch(normalized):
            normalized_claim = normalized
    if normalized_claim is not None and normalized_claim != "application/octet-stream":
        # Active browser content is still forced to attachment by the
        # download endpoint. Retaining the type here is provenance, not
        # permission to render it inline.
        return normalized_claim
    if suffix in _TEXT_EXTENSIONS:
        return "text/plain"
    if normalized_claim is not None:
        return normalized_claim
    return "application/octet-stream"


def canonical_artifact_kind(media_type: str) -> str:
    if media_type.startswith("image/"):
        return "photo"
    if media_type.startswith("audio/"):
        return "voice"
    return "document"


def _grant_read_access(path: Path, reader_user: str) -> None:
    if sys.platform == "darwin":
        command = [
            "/bin/chmod",
            "+a",
            f"user:{reader_user} allow read,readattr,readextattr,readsecurity",
            str(path),
        ]
    elif sys.platform.startswith("linux"):
        setfacl = shutil.which("setfacl")
        if setfacl is None:
            raise OSError("setfacl is required for isolated uploaded-file handoff on Linux")
        command = [setfacl, "-m", f"u:{reader_user}:r--", str(path)]
    else:
        raise OSError(f"isolated uploaded-file handoff is unsupported on {sys.platform}")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        raise OSError(f"could not grant upload access to {reader_user}: {detail}")


async def _write_bounded_file(
    destination: Path,
    chunks: AsyncIterable[bytes],
    *,
    max_bytes: int,
) -> tuple[int, str, bool]:
    if max_bytes < 1 or max_bytes > MAX_ARTIFACT_BYTES:
        raise ValueError("max_bytes exceeds the artifact service limit")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{secrets.token_hex(8)}.part")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor: int | None = None
    digest = hashlib.sha256()
    byte_size = 0
    try:
        descriptor = os.open(temporary, flags, _UPLOAD_FILE_MODE)
        async for chunk in chunks:
            if not isinstance(chunk, bytes):
                raise TypeError("artifact chunks must be bytes")
            byte_size += len(chunk)
            if byte_size > max_bytes:
                raise ArtifactTooLargeError(f"Artifact must be no larger than {max_bytes} bytes")
            digest.update(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        created = False
        try:
            os.link(temporary, destination)
            created = True
        except FileExistsError:
            existing_size, existing_hash = _file_identity(destination)
            if existing_size != byte_size or existing_hash != digest.hexdigest():
                raise ValueError("Artifact identity conflicts with different content") from None
        temporary.unlink()
        destination.chmod(_UPLOAD_FILE_MODE)
        return byte_size, digest.hexdigest(), created
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise


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
    if store.connection.in_transaction:
        raise RuntimeError("record_inbound_artifact cannot join a caller-owned transaction")
    try:
        await store.connection.execute("BEGIN IMMEDIATE")
        result = await record_inbound_artifact_in_transaction(
            store,
            artifact,
            storage_root=storage_root,
        )
        await store.connection.commit()
        return result
    except Exception:
        await store.connection.rollback()
        raise


async def record_inbound_artifact_in_transaction(
    store: WorkshopEventStore,
    artifact: InboundArtifact,
    *,
    storage_root: Path,
) -> AppendResult:
    """Append and project artifact metadata inside a caller transaction."""
    if not store.connection.in_transaction:
        raise RuntimeError("record_inbound_artifact_in_transaction requires an active transaction")
    if not isinstance(storage_root, Path):
        raise ValueError("storage_root must be a Path")
    return await _record_artifact_in_transaction(
        store,
        artifact,
        storage_root=storage_root,
        author_kind="human",
        event_version=1,
        metadata_source="artifact_shadow",
    )


async def record_published_artifact_in_transaction(
    store: WorkshopEventStore,
    artifact: InboundArtifact,
    *,
    storage_root: Path,
) -> AppendResult:
    """Append agent-authored artifact metadata inside a publication transaction."""
    return await _record_artifact_in_transaction(
        store,
        artifact,
        storage_root=storage_root,
        author_kind="agent",
        event_version=2,
        metadata_source="internal_api",
    )


async def _record_artifact_in_transaction(
    store: WorkshopEventStore,
    artifact: InboundArtifact,
    *,
    storage_root: Path,
    author_kind: str,
    event_version: int,
    metadata_source: str,
) -> AppendResult:
    if not store.connection.in_transaction:
        raise RuntimeError("artifact recording requires an active transaction")
    if not isinstance(storage_root, Path):
        raise ValueError("storage_root must be a Path")
    resolved_message = await _resolve_artifact_message(
        store,
        artifact.message_id,
        author_kind=author_kind,
    )
    resolved_path = _resolve_storage_path(artifact.storage_path, storage_root)
    byte_size, content_sha256 = _file_identity(resolved_path)
    token = _stable_source_token(artifact)
    idempotency_key = f"workshop-artifact:v1:{token}"

    def create_envelope(storage_path: str) -> EventEnvelope:
        return EventEnvelope.create(
            event_id=EventId.derived(resolved_message.workshop_id, f"artifact-event:{token}"),
            event_type=WorkshopEventType.ARTIFACT_CREATED,
            event_version=event_version,
            workshop_id=resolved_message.workshop_id,
            aggregate_type="artifact",
            aggregate_id=ArtifactId.derived(resolved_message.workshop_id, f"artifact:{token}"),
            actor_principal_id=resolved_message.created_by_principal_id,
            occurred_at=artifact.occurred_at,
            idempotency_key=idempotency_key,
            payload={
                "channel_id": resolved_message.channel_id,
                "message_id": artifact.message_id,
                "created_by_principal_id": resolved_message.created_by_principal_id,
                "kind": artifact.kind,
                "media_type": artifact.media_type,
                "byte_size": byte_size,
                "content_sha256": content_sha256,
                "original_filename": artifact.original_filename,
                "storage_path": storage_path,
                "source_transport": artifact.source_transport,
                "source_unique_id": artifact.source_unique_id,
            },
            metadata={"source": metadata_source},
        )

    envelope = create_envelope(str(resolved_path))
    existing = await store.event_by_idempotency_key(idempotency_key)
    if existing is not None:
        existing_path = existing.envelope.payload.get("storage_path")
        if (
            isinstance(existing_path, str)
            and create_envelope(existing_path).content_hash == existing.envelope.content_hash
        ):
            result = AppendResult(event=existing, inserted=False)
        else:
            # Preserve the event store's uniform conflict behavior and error
            # type for retries that differ in any semantic property other
            # than the timestamped compatibility storage path.
            result = await store.append_in_transaction(envelope)
    else:
        result = await store.append_in_transaction(envelope)
    await store.project_pending_in_transaction(CanonicalConversationProjection())
    return result


async def artifact_for_delivery(
    store: WorkshopEventStore,
    message_id: MessageId,
    *,
    storage_root: Path,
) -> tuple[StoredArtifact, str]:
    """Resolve one agent-published artifact and its transport caption."""
    async with store.connection.execute(
        "SELECT a.id, a.message_id, a.channel_id, a.kind, a.media_type, "
        "a.byte_size, a.content_sha256, a.original_filename, a.created_at, "
        "a.storage_path, e.metadata_json "
        "FROM artifacts a JOIN messages m ON m.id = a.message_id "
        "JOIN principals p ON p.id = m.author_principal_id AND p.kind = 'agent' "
        "JOIN event_log e ON e.position = m.created_event_position "
        "WHERE a.message_id = ? ORDER BY a.created_event_position, a.id",
        (message_id,),
    ) as cursor:
        rows = list(await cursor.fetchall())
    if len(rows) != 1:
        raise ArtifactMessageNotFoundError("Published artifact message does not resolve uniquely")
    row = rows[0]
    path = _resolve_storage_path(Path(str(row[9])), storage_root)
    summary = ArtifactSummary(
        artifact_id=ArtifactId(str(row[0])),
        message_id=MessageId(str(row[1])),
        channel_id=ChannelId(str(row[2])),
        kind=str(row[3]),
        media_type=str(row[4]),
        byte_size=int(row[5]),
        content_sha256=str(row[6]),
        original_filename=str(row[7]) if row[7] is not None else None,
        created_at=datetime.fromisoformat(str(row[8]).replace("Z", "+00:00")),
    )
    actual_size, actual_hash = _file_identity(path)
    if actual_size != summary.byte_size or actual_hash != summary.content_sha256:
        raise ArtifactStorageBoundaryError("Artifact content no longer matches canonical provenance")
    metadata = json.loads(str(row[10]))
    if (
        not isinstance(metadata, dict)
        or metadata.get("source") != "internal_api"
        or metadata.get("publication_kind") != "file"
    ):
        raise ArtifactStorageBoundaryError("Published artifact provenance is invalid")
    caption = metadata.get("caption", "")
    if not isinstance(caption, str):
        raise ArtifactStorageBoundaryError("Published artifact caption is invalid")
    return StoredArtifact(summary=summary, storage_path=path), caption


async def artifacts_for_messages(
    store: WorkshopEventStore,
    message_ids: tuple[MessageId, ...],
) -> dict[MessageId, tuple[ArtifactSummary, ...]]:
    if not message_ids:
        return {}
    placeholders = ", ".join("?" for _ in message_ids)
    async with store.connection.execute(
        "SELECT id, message_id, channel_id, kind, media_type, byte_size, "
        "content_sha256, original_filename, created_at FROM artifacts "
        f"WHERE message_id IN ({placeholders}) ORDER BY created_event_position, id",
        message_ids,
    ) as cursor:
        rows = list(await cursor.fetchall())
    grouped: dict[MessageId, list[ArtifactSummary]] = {}
    for row in rows:
        message_id = MessageId(str(row[1]))
        grouped.setdefault(message_id, []).append(
            ArtifactSummary(
                artifact_id=ArtifactId(str(row[0])),
                message_id=message_id,
                channel_id=ChannelId(str(row[2])),
                kind=str(row[3]),
                media_type=str(row[4]),
                byte_size=int(row[5]),
                content_sha256=str(row[6]),
                original_filename=str(row[7]) if row[7] is not None else None,
                created_at=datetime.fromisoformat(str(row[8]).replace("Z", "+00:00")),
            )
        )
    return {message_id: tuple(artifacts) for message_id, artifacts in grouped.items()}


async def build_agent_prompt_for_message(
    store: WorkshopEventStore,
    message_id: MessageId,
    *,
    storage_root: Path,
) -> str | list:
    """Build one backend prompt from canonical text and attached artifacts."""
    async with store.connection.execute(
        "SELECT body FROM messages WHERE id = ?",
        (message_id,),
    ) as cursor:
        message_row = await cursor.fetchone()
    if message_row is None:
        raise ArtifactMessageNotFoundError("Canonical prompt message was not found")
    body = str(message_row[0])
    async with store.connection.execute(
        "SELECT kind, media_type, byte_size, content_sha256, original_filename, storage_path "
        "FROM artifacts WHERE message_id = ? ORDER BY created_event_position, id",
        (message_id,),
    ) as cursor:
        rows = list(await cursor.fetchall())
    if not rows:
        return body

    text_sections = [body]
    image_blocks: list[dict] = []
    for row in rows:
        kind = str(row[0])
        media_type = str(row[1])
        expected_size = int(row[2])
        expected_hash = str(row[3])
        filename = str(row[4]) if row[4] is not None else "unnamed artifact"
        path = _resolve_storage_path(Path(str(row[5])), storage_root)
        actual_size, actual_hash = _file_identity(path)
        if actual_size != expected_size or actual_hash != expected_hash:
            raise ArtifactStorageBoundaryError("Artifact content no longer matches canonical provenance")
        descriptor = f"User-provided {kind}: {filename} ({media_type}, {actual_size} bytes)"
        if media_type in _IMAGE_MEDIA_TYPES.values():
            image_blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                    },
                }
            )
            text_sections.append(f"[{descriptor}; private local path: {path}]")
        elif media_type.startswith("text/") and actual_size <= MAX_INLINE_TEXT_BYTES:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text_sections.append(f"[{descriptor}; private local path: {path}]")
            else:
                text_sections.append(
                    f"[{descriptor}; private local path: {path}]\n"
                    "<untrusted_user_file>\n"
                    f"{content}\n"
                    "</untrusted_user_file>"
                )
        else:
            text_sections.append(f"[{descriptor}; private local path: {path}]")
    text = "\n\n".join(text_sections)
    return [{"type": "text", "text": text}, *image_blocks] if image_blocks else text


class WorkshopArtifactService:
    """Core-owned storage and authorized access for canonical artifacts."""

    def __init__(
        self,
        store: WorkshopEventStore,
        *,
        data_dir: Path,
        principal_storage: WorkshopPrincipalStorageRegistry,
        runtime_profiles: WorkshopRuntimeProfileRegistry,
    ) -> None:
        self._store = store
        self._data_dir = data_dir.resolve()
        self._storage = principal_storage
        self._runtime_profiles = runtime_profiles

    async def stage_client_upload(
        self,
        *,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        client_message_id: str,
        filename: str | None,
        claimed_media_type: str | None,
        chunks: AsyncIterable[bytes],
        occurred_at: datetime,
        max_bytes: int = MAX_ARTIFACT_BYTES,
    ) -> StagedArtifact:
        if not isinstance(principal_id, PrincipalId) or not isinstance(channel_id, ChannelId):
            raise ValueError("principal_id and channel_id must be canonical identifiers")
        if not isinstance(client_message_id, str) or not client_message_id:
            raise ValueError("client_message_id must be non-empty")
        async with self._store.connection.execute(
            "SELECT 1 FROM channels WHERE id = ? AND archived_at IS NULL",
            (channel_id,),
        ) as cursor:
            if await cursor.fetchone() is None:
                raise ArtifactAccessDeniedError("Archived channels are read-only")
        _, profile_id = await resolve_channel_runtime_profile(self._store, channel_id)
        namespace = self._storage.for_runtime_profile(profile_id)
        if namespace.principal_id != principal_id:
            raise ArtifactAccessDeniedError("Artifact storage owner does not match the authenticated principal")
        return await self.stage_upload(
            principal_id=principal_id,
            runtime_profile_id=profile_id,
            filename=filename,
            claimed_media_type=claimed_media_type,
            chunks=chunks,
            source_transport="workshop_client",
            source_unique_id=client_message_id,
            occurred_at=occurred_at,
            kind=None,
            original_filename=filename,
            max_bytes=max_bytes,
        )

    async def stage_upload(
        self,
        *,
        principal_id: PrincipalId,
        runtime_profile_id: RuntimeProfileId,
        filename: str | None,
        claimed_media_type: str | None,
        chunks: AsyncIterable[bytes],
        source_transport: str,
        source_unique_id: str,
        occurred_at: datetime,
        kind: str | None = None,
        original_filename: str | None = None,
        max_bytes: int = MAX_ARTIFACT_BYTES,
    ) -> StagedArtifact:
        if not _IDENTIFIER_PATTERN.fullmatch(source_transport):
            raise ValueError("source_transport must be a lowercase identifier")
        if not source_unique_id or len(source_unique_id) > _MAX_SOURCE_UNIQUE_ID_LENGTH:
            raise ValueError("source_unique_id must be bounded and non-empty")
        namespace = self._storage.for_runtime_profile(runtime_profile_id)
        if namespace.principal_id != principal_id:
            raise ArtifactAccessDeniedError("Artifact storage owner does not match the authenticated principal")
        safe_filename = canonical_artifact_filename(filename) or "upload.bin"
        media_type = canonical_artifact_media_type(safe_filename, claimed_media_type)
        artifact_kind = canonical_artifact_kind(media_type) if kind is None else kind
        if artifact_kind not in _ARTIFACT_KINDS:
            raise ValueError("kind must be photo, document, or voice")
        root = self._data_dir / "files"
        directory = namespace.files_directory(self._data_dir)
        root.mkdir(parents=True, exist_ok=True)
        directory.mkdir(parents=True, exist_ok=True)
        root.chmod(_UPLOAD_ROOT_MODE)
        directory.chmod(_UPLOAD_ROOT_MODE)
        token = hashlib.sha256(f"{principal_id}\0{source_transport}\0{source_unique_id}".encode()).hexdigest()
        destination = directory / f"{token}_{safe_filename.replace(' ', '_')}"
        byte_size, _, created = await _write_bounded_file(
            destination,
            chunks,
            max_bytes=max_bytes,
        )
        if byte_size == 0:
            if created:
                destination.unlink(missing_ok=True)
            raise ValueError("Artifact content must not be empty")
        profile = self._runtime_profiles.resolve(namespace.runtime_profile_id)
        if profile.os_user:
            try:
                _grant_read_access(destination, profile.os_user)
            except Exception:
                if created:
                    destination.unlink(missing_ok=True)
                raise
        return StagedArtifact(
            kind=artifact_kind,
            media_type=media_type,
            storage_path=destination,
            source_transport=source_transport,
            source_unique_id=source_unique_id,
            occurred_at=occurred_at,
            original_filename=canonical_artifact_filename(original_filename),
            created_for_attempt=created,
        )

    async def authorized_artifact(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        artifact_id: ArtifactId,
    ) -> StoredArtifact:
        async with self._store.connection.execute(
            "SELECT a.id, a.message_id, a.channel_id, a.kind, a.media_type, "
            "a.byte_size, a.content_sha256, a.original_filename, a.created_at, "
            "a.storage_path FROM artifacts a "
            "JOIN channel_memberships cm ON cm.channel_id = a.channel_id "
            "AND cm.principal_id = ? WHERE a.id = ? AND a.channel_id = ?",
            (principal_id, artifact_id, channel_id),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if len(rows) != 1:
            raise ArtifactAccessDeniedError("Artifact access denied")
        row = rows[0]
        path = _resolve_storage_path(Path(str(row[9])), self._data_dir / "files")
        summary = ArtifactSummary(
            artifact_id=ArtifactId(str(row[0])),
            message_id=MessageId(str(row[1])),
            channel_id=ChannelId(str(row[2])),
            kind=str(row[3]),
            media_type=str(row[4]),
            byte_size=int(row[5]),
            content_sha256=str(row[6]),
            original_filename=str(row[7]) if row[7] is not None else None,
            created_at=datetime.fromisoformat(str(row[8]).replace("Z", "+00:00")),
        )
        actual_size, actual_hash = _file_identity(path)
        if actual_size != summary.byte_size or actual_hash != summary.content_sha256:
            raise ArtifactStorageBoundaryError("Artifact content no longer matches canonical provenance")
        return StoredArtifact(summary=summary, storage_path=path)
