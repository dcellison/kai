"""Canonical, self-owned human avatars for Kai Workshop."""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import tempfile
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from kai.workshop.domain import EventEnvelope, PrincipalId, WorkshopEventType, WorkshopId
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import StoredEvent, WorkshopEventStore

MAX_AVATAR_UPLOAD_BYTES = 256 * 1024
MAX_AVATAR_OUTPUT_BYTES = 512 * 1024
MAX_AVATAR_SOURCE_DIMENSION = 4096
MAX_AVATAR_SOURCE_PIXELS = 16_000_000
MAX_AVATAR_DIMENSION = 256
_ACCEPTED_FORMATS = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/gif": "GIF",
    "image/webp": "WEBP",
}


class WorkshopHumanAvatarError(RuntimeError):
    """A canonical human-avatar operation failed."""


class WorkshopHumanAvatarAccessDenied(WorkshopHumanAvatarError):
    """The principal cannot access the requested avatar."""


class WorkshopHumanAvatarValidationError(WorkshopHumanAvatarError):
    """The supplied avatar request is invalid."""


class WorkshopHumanAvatarUnsupportedType(WorkshopHumanAvatarValidationError):
    """The supplied file is not an accepted image type."""


class WorkshopHumanAvatarTooLarge(WorkshopHumanAvatarValidationError):
    """The supplied image exceeds an avatar safety limit."""


class WorkshopHumanAvatarConflict(WorkshopHumanAvatarError):
    """The avatar state changed after the caller loaded it."""


class WorkshopHumanAvatarStorageError(WorkshopHumanAvatarError):
    """Canonical avatar state or bytes are unavailable."""


@dataclass(frozen=True, slots=True)
class HumanAvatarSnapshot:
    principal_id: PrincipalId
    state_version: int
    active: bool
    media_type: str | None = None
    byte_size: int | None = None
    width: int | None = None
    height: int | None = None
    sha256: str | None = None
    mutation_changed: bool | None = None
    replayed: bool = False


@dataclass(frozen=True, slots=True)
class HumanAvatarBytes:
    snapshot: HumanAvatarSnapshot
    path: Path


@dataclass(frozen=True, slots=True)
class _NormalizedAvatar:
    content: bytes
    width: int
    height: int
    sha256: str


def _validate_operation(expected_state_version: object, client_operation_id: object) -> tuple[int, str]:
    if (
        not isinstance(expected_state_version, int)
        or isinstance(expected_state_version, bool)
        or expected_state_version < 0
    ):
        raise WorkshopHumanAvatarValidationError("expected_state_version must be a non-negative integer")
    if not isinstance(client_operation_id, str) or not client_operation_id.strip() or len(client_operation_id) > 200:
        raise WorkshopHumanAvatarValidationError("client_operation_id must be a non-empty string")
    return expected_state_version, client_operation_id.strip()


def _normalize_avatar(raw: bytes, claimed_media_type: str) -> _NormalizedAvatar:
    if claimed_media_type not in _ACCEPTED_FORMATS:
        raise WorkshopHumanAvatarUnsupportedType("Avatar must be a PNG, JPEG, GIF, or WebP image")
    if not raw:
        raise WorkshopHumanAvatarValidationError("Avatar image is empty")
    if len(raw) > MAX_AVATAR_UPLOAD_BYTES:
        raise WorkshopHumanAvatarTooLarge("Avatar upload exceeds 256 KiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                if source.format != _ACCEPTED_FORMATS[claimed_media_type]:
                    raise WorkshopHumanAvatarUnsupportedType(
                        "Avatar Content-Type does not match the decoded image format"
                    )
                width, height = source.size
                if (
                    width <= 0
                    or height <= 0
                    or width > MAX_AVATAR_SOURCE_DIMENSION
                    or height > MAX_AVATAR_SOURCE_DIMENSION
                    or width * height > MAX_AVATAR_SOURCE_PIXELS
                ):
                    raise WorkshopHumanAvatarTooLarge("Avatar image dimensions are too large")
                source.seek(0)
                source.load()
                normalized = ImageOps.exif_transpose(source).convert("RGBA")
                normalized.thumbnail(
                    (MAX_AVATAR_DIMENSION, MAX_AVATAR_DIMENSION),
                    Image.Resampling.LANCZOS,
                )
                output = io.BytesIO()
                normalized.save(output, format="PNG", optimize=True)
    except WorkshopHumanAvatarError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise WorkshopHumanAvatarTooLarge("Avatar image dimensions are too large") from None
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise WorkshopHumanAvatarValidationError("Avatar image could not be decoded") from exc
    content = output.getvalue()
    if len(content) > MAX_AVATAR_OUTPUT_BYTES:
        raise WorkshopHumanAvatarTooLarge("Normalized avatar is too large")
    return _NormalizedAvatar(
        content=content,
        width=normalized.width,
        height=normalized.height,
        sha256=hashlib.sha256(content).hexdigest(),
    )


class WorkshopHumanAvatarService:
    """Own canonical avatar state and normalized image storage."""

    def __init__(self, store: WorkshopEventStore, *, data_dir: Path) -> None:
        self._store = store
        self._root = data_dir.resolve() / "files" / "avatars"

    @property
    def storage_root(self) -> Path:
        return self._root

    async def inspect(self, principal_id: PrincipalId) -> HumanAvatarSnapshot:
        await self._require_human(principal_id)
        return await self._snapshot(principal_id)

    async def upload(
        self,
        principal_id: PrincipalId,
        raw: bytes,
        claimed_media_type: str,
        *,
        expected_state_version: object,
        client_operation_id: object,
    ) -> HumanAvatarSnapshot:
        expected, operation_id = _validate_operation(expected_state_version, client_operation_id)
        normalized = await asyncio.to_thread(_normalize_avatar, raw, claimed_media_type)
        await self._require_human(principal_id)
        payload: dict[str, object] = {
            "expected_state_version": expected,
            "media_type": "image/png",
            "byte_size": len(normalized.content),
            "width": normalized.width,
            "height": normalized.height,
            "sha256": normalized.sha256,
        }
        return await self._change(
            principal_id,
            payload=payload,
            expected_state_version=expected,
            operation_id=operation_id,
            normalized=normalized,
        )

    async def clear(
        self,
        principal_id: PrincipalId,
        *,
        expected_state_version: object,
        client_operation_id: object,
    ) -> HumanAvatarSnapshot:
        expected, operation_id = _validate_operation(expected_state_version, client_operation_id)
        await self._require_human(principal_id)
        payload: dict[str, object] = {"expected_state_version": expected, "cleared": True}
        existing = await self._store.event_by_idempotency_key(
            f"workshop-client:human-avatar:{principal_id}:{operation_id}"
        )
        if existing is not None:
            return await self._change(
                principal_id,
                payload=payload,
                expected_state_version=expected,
                operation_id=operation_id,
                normalized=None,
            )
        before = await self._snapshot(principal_id)
        if before.state_version != expected:
            raise WorkshopHumanAvatarConflict("Human avatar changed since it was loaded")
        if not before.active:
            return HumanAvatarSnapshot(
                before.principal_id,
                before.state_version,
                False,
                mutation_changed=False,
            )
        return await self._change(
            principal_id,
            payload=payload,
            expected_state_version=expected,
            operation_id=operation_id,
            normalized=None,
        )

    async def retrieve(
        self,
        requester_principal_id: PrincipalId,
        target_principal_id: PrincipalId,
        state_version: int,
    ) -> HumanAvatarBytes:
        if state_version <= 0:
            raise WorkshopHumanAvatarAccessDenied("Avatar access denied")
        await self._require_shared_workshop(requester_principal_id, target_principal_id)
        snapshot = await self._snapshot(target_principal_id)
        if not snapshot.active or snapshot.state_version != state_version:
            raise WorkshopHumanAvatarAccessDenied("Avatar access denied")
        path = self._avatar_path(target_principal_id, state_version)
        try:
            if path.parent.is_symlink():
                raise WorkshopHumanAvatarStorageError("Avatar bytes are unavailable")
            stat = path.lstat()
            if not path.is_file() or path.is_symlink() or stat.st_size != snapshot.byte_size:
                raise WorkshopHumanAvatarStorageError("Avatar bytes are unavailable")
            digest = await asyncio.to_thread(_sha256_file, path)
        except FileNotFoundError as exc:
            raise WorkshopHumanAvatarStorageError("Avatar bytes are unavailable") from exc
        if digest != snapshot.sha256:
            raise WorkshopHumanAvatarStorageError("Avatar bytes failed integrity validation")
        return HumanAvatarBytes(snapshot=snapshot, path=path)

    async def _change(
        self,
        principal_id: PrincipalId,
        *,
        payload: dict[str, object],
        expected_state_version: int,
        operation_id: str,
        normalized: _NormalizedAvatar | None,
    ) -> HumanAvatarSnapshot:
        connection = self._store.connection
        created_path: Path | None = None
        try:
            await connection.execute("BEGIN IMMEDIATE")
            idempotency_key = f"workshop-client:human-avatar:{principal_id}:{operation_id}"
            existing = await self._store.event_by_idempotency_key(idempotency_key)
            if existing is not None:
                self._validate_replay(existing, principal_id, payload)
                current = await self._snapshot(principal_id)
                if (
                    normalized is not None
                    and current.active
                    and current.state_version == expected_state_version + 1
                    and current.sha256 == normalized.sha256
                ):
                    replay_path = self._avatar_path(principal_id, current.state_version)
                    if not replay_path.exists():
                        await asyncio.to_thread(self._write_version_file, replay_path, normalized.content)
                await connection.rollback()
                return _with_mutation(current, changed=True, replayed=True)
            before = await self._snapshot(principal_id)
            if before.state_version != expected_state_version:
                raise WorkshopHumanAvatarConflict("Human avatar changed since it was loaded")
            if normalized is not None:
                created_path = self._avatar_path(principal_id, expected_state_version + 1)
                await asyncio.to_thread(self._write_version_file, created_path, normalized.content)
            workshop_id = await self._workshop_id(principal_id)
            event = EventEnvelope.create(
                event_type=WorkshopEventType.PRINCIPAL_AVATAR_CHANGED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="principal_profile",
                aggregate_id=principal_id,
                actor_principal_id=principal_id,
                occurred_at=datetime.now(UTC),
                idempotency_key=idempotency_key,
                payload=payload,
                metadata={"source": "workshop_client"},
            )
            result = await self._store.append_in_transaction(event)
            if not result.inserted:
                raise WorkshopHumanAvatarConflict("New human-avatar event already exists")
            await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            after = await self._snapshot(principal_id)
            await connection.commit()
        except WorkshopHumanAvatarError:
            await connection.rollback()
            if created_path is not None:
                await asyncio.to_thread(_unlink_if_exists, created_path)
            raise
        except Exception as exc:
            await connection.rollback()
            if created_path is not None:
                await asyncio.to_thread(_unlink_if_exists, created_path)
            raise WorkshopHumanAvatarStorageError("Human avatar could not be saved") from exc
        try:
            await asyncio.to_thread(
                self._remove_other_versions,
                principal_id,
                after.state_version if after.active else None,
            )
        except OSError:
            # The canonical mutation succeeded. Retained obsolete bytes are
            # non-authoritative and surfaced by install diagnostics.
            pass
        return _with_mutation(after, changed=True)

    async def _snapshot(self, principal_id: PrincipalId) -> HumanAvatarSnapshot:
        async with self._store.connection.execute(
            "SELECT state_version, active, media_type, byte_size, width, height, sha256 "
            "FROM principal_avatars WHERE principal_id = ?",
            (principal_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return HumanAvatarSnapshot(principal_id, 0, False)
        return HumanAvatarSnapshot(
            principal_id,
            int(row[0]),
            bool(row[1]),
            str(row[2]) if row[2] is not None else None,
            int(row[3]) if row[3] is not None else None,
            int(row[4]) if row[4] is not None else None,
            int(row[5]) if row[5] is not None else None,
            str(row[6]) if row[6] is not None else None,
        )

    async def _require_human(self, principal_id: PrincipalId) -> None:
        if not isinstance(principal_id, PrincipalId):
            raise WorkshopHumanAvatarAccessDenied("Avatar access denied")
        async with self._store.connection.execute(
            "SELECT 1 FROM principals p JOIN workshop_memberships wm ON wm.principal_id = p.id "
            "WHERE p.id = ? AND p.kind = 'human' LIMIT 1",
            (principal_id,),
        ) as cursor:
            if await cursor.fetchone() is None:
                raise WorkshopHumanAvatarAccessDenied("Avatar access denied")

    async def _require_shared_workshop(self, requester: PrincipalId, target: PrincipalId) -> None:
        if not isinstance(requester, PrincipalId) or not isinstance(target, PrincipalId):
            raise WorkshopHumanAvatarAccessDenied("Avatar access denied")
        async with self._store.connection.execute(
            "SELECT 1 FROM workshop_memberships requester "
            "JOIN workshop_memberships target ON target.workshop_id = requester.workshop_id "
            "JOIN principals requester_principal "
            "ON requester_principal.id = requester.principal_id AND requester_principal.kind = 'human' "
            "JOIN principals target_principal "
            "ON target_principal.id = target.principal_id AND target_principal.kind = 'human' "
            "WHERE requester.principal_id = ? AND target.principal_id = ? LIMIT 1",
            (requester, target),
        ) as cursor:
            if await cursor.fetchone() is None:
                raise WorkshopHumanAvatarAccessDenied("Avatar access denied")

    async def _workshop_id(self, principal_id: PrincipalId) -> WorkshopId:
        async with self._store.connection.execute(
            "SELECT workshop_id FROM workshop_memberships WHERE principal_id = ? ORDER BY created_at LIMIT 1",
            (principal_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise WorkshopHumanAvatarAccessDenied("Avatar access denied")
        return WorkshopId(str(row[0]))

    def _avatar_path(self, principal_id: PrincipalId, state_version: int) -> Path:
        path = self._root / str(principal_id) / f"{state_version}.png"
        if not path.resolve().is_relative_to(self._root):
            raise WorkshopHumanAvatarStorageError("Avatar storage boundary rejected the path")
        return path

    def _write_version_file(self, path: Path, content: bytes) -> None:
        self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self._root.is_symlink() or not self._root.is_dir():
            raise WorkshopHumanAvatarStorageError("Avatar storage root is invalid")
        os.chmod(self._root, 0o700)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if path.parent.is_symlink() or not path.parent.is_dir() or path.parent.resolve().parent != self._root:
            raise WorkshopHumanAvatarStorageError("Avatar storage boundary rejected the path")
        os.chmod(path.parent, 0o700)
        if path.exists():
            if path.is_symlink() or not path.is_file() or _sha256_file(path) != hashlib.sha256(content).hexdigest():
                raise WorkshopHumanAvatarStorageError("Avatar version path already contains different bytes")
            return
        descriptor, temporary_name = tempfile.mkstemp(prefix=".avatar-", dir=path.parent)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary_path, 0o600)
            os.link(temporary_path, path)
            os.chmod(path, 0o600)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _remove_other_versions(self, principal_id: PrincipalId, keep_version: int | None) -> None:
        directory = self._root / str(principal_id)
        if directory.is_symlink():
            raise OSError("Avatar principal storage directory is a symlink")
        try:
            entries = tuple(directory.iterdir())
        except FileNotFoundError:
            return
        keep_name = f"{keep_version}.png" if keep_version is not None else None
        for path in entries:
            if path.name != keep_name and path.is_file() and not path.is_symlink():
                path.unlink(missing_ok=True)

    @staticmethod
    def _validate_replay(stored: StoredEvent, principal_id: PrincipalId, payload: dict[str, object]) -> None:
        envelope = stored.envelope
        if (
            envelope.event_type != WorkshopEventType.PRINCIPAL_AVATAR_CHANGED
            or envelope.event_version != 1
            or envelope.aggregate_type != "principal_profile"
            or envelope.aggregate_id != principal_id
            or envelope.actor_principal_id != principal_id
            or envelope.payload != payload
        ):
            raise WorkshopHumanAvatarConflict(
                "client_operation_id is already bound to a different human-avatar operation"
            )


def _with_mutation(
    snapshot: HumanAvatarSnapshot,
    *,
    changed: bool,
    replayed: bool = False,
) -> HumanAvatarSnapshot:
    return HumanAvatarSnapshot(
        snapshot.principal_id,
        snapshot.state_version,
        snapshot.active,
        snapshot.media_type,
        snapshot.byte_size,
        snapshot.width,
        snapshot.height,
        snapshot.sha256,
        mutation_changed=changed,
        replayed=replayed,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unlink_if_exists(path: Path) -> None:
    path.unlink(missing_ok=True)
