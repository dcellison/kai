"""Canonical principal-owned AGENTS.md policy authority."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pwd
import re
import secrets
import stat
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from kai.principal_documents import (
    MAX_PRINCIPAL_DOCUMENT_BYTES,
    PRINCIPAL_DOCUMENT_READER,
    PrincipalDocumentKind,
    PrincipalDocumentState,
    read_local_principal_document,
    read_protected_principal_document,
)
from kai.principal_policy import PrincipalPolicyMigrationConflict, plan_principal_policy_migration
from kai.workshop.domain import PrincipalId
from kai.workshop.storage_namespaces import (
    WorkshopPrincipalStorageNamespace,
    WorkshopPrincipalStorageRegistry,
    WorkshopStorageNamespaceError,
)

PRINCIPAL_POLICY_MANAGER = Path("/etc/kai/manage-principal-policy")
_REVISION = re.compile(r"^[0-9a-f]{64}$")
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)


class WorkshopPrincipalPolicyError(RuntimeError):
    """Base failure for principal-policy operations."""


class WorkshopPrincipalPolicyAccessDenied(WorkshopPrincipalPolicyError):
    """The authenticated principal has no policy authority."""


class WorkshopPrincipalPolicyValidationError(WorkshopPrincipalPolicyError):
    """Policy content or revision is invalid."""


class WorkshopPrincipalPolicyConflict(WorkshopPrincipalPolicyError):
    """The expected policy revision is no longer current."""

    def __init__(self, current_revision: str) -> None:
        super().__init__("Principal policy changed since it was opened")
        self.current_revision = current_revision


class WorkshopPrincipalPolicyStorageError(WorkshopPrincipalPolicyError):
    """Principal-policy storage failed closed."""


@dataclass(frozen=True, slots=True)
class PrincipalPolicyAuthority:
    principal_id: PrincipalId
    namespace: WorkshopPrincipalStorageNamespace


@dataclass(frozen=True, slots=True)
class PrincipalPolicyDocument:
    content: str
    revision: str
    size_bytes: int
    max_bytes: int = MAX_PRINCIPAL_DOCUMENT_BYTES
    editable: bool = True


@dataclass(frozen=True, slots=True)
class PrincipalPolicyContextInvalidation:
    state: str
    applied: int
    pending: int


def _normalize(content: str) -> tuple[str, bytes]:
    if not isinstance(content, str):
        raise WorkshopPrincipalPolicyValidationError("Principal policy must be text")
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    if "\x00" in normalized:
        raise WorkshopPrincipalPolicyValidationError("Principal policy contains an invalid character")
    encoded = normalized.encode("utf-8")
    if len(encoded) > MAX_PRINCIPAL_DOCUMENT_BYTES:
        raise WorkshopPrincipalPolicyValidationError(
            f"Principal policy exceeds the {MAX_PRINCIPAL_DOCUMENT_BYTES}-byte limit"
        )
    try:
        migration = plan_principal_policy_migration(normalized)
    except PrincipalPolicyMigrationConflict as exc:
        raise WorkshopPrincipalPolicyValidationError(str(exc)) from exc
    if migration.changed:
        raise WorkshopPrincipalPolicyValidationError(
            "Agent identity belongs in the owned agent definition, not principal policy"
        )
    return normalized, encoded


def _validate_revision(revision: str) -> str:
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise WorkshopPrincipalPolicyValidationError("Principal-policy revision is invalid")
    return revision


def _document(content: str, revision: str | None) -> PrincipalPolicyDocument:
    if revision is None:
        raise WorkshopPrincipalPolicyStorageError("Principal policy has no canonical revision")
    return PrincipalPolicyDocument(content, revision, len(content.encode("utf-8")))


def _local_write(path: Path, expected_revision: str, content: bytes) -> None:
    parent_fd = os.open(path.parent, os.O_RDONLY | _NOFOLLOW)
    try:
        parent_info = os.fstat(parent_fd)
        if not stat.S_ISDIR(parent_info.st_mode) or parent_info.st_uid == 0 or parent_info.st_mode & 0o077:
            raise WorkshopPrincipalPolicyStorageError("Principal home is not a private user directory")
        fd = os.open(path.name, os.O_RDONLY | _NOFOLLOW | _NONBLOCK, dir_fd=parent_fd)
        try:
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_uid != parent_info.st_uid
                or info.st_mode & 0o077
            ):
                raise WorkshopPrincipalPolicyStorageError("Principal policy is not a private owner file")
            current = b""
            while True:
                chunk = os.read(fd, 64 * 1024)
                if not chunk:
                    break
                current += chunk
                if len(current) > MAX_PRINCIPAL_DOCUMENT_BYTES:
                    raise WorkshopPrincipalPolicyStorageError("Principal policy is oversized")
        finally:
            os.close(fd)
        current_revision = hashlib.sha256(current).hexdigest()
        if current_revision != expected_revision:
            raise WorkshopPrincipalPolicyConflict(current_revision)
        temporary = f".AGENTS.md.{secrets.token_hex(12)}.tmp"
        out = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW, 0o600, dir_fd=parent_fd)
        try:
            offset = 0
            while offset < len(content):
                offset += os.write(out, content[offset:])
            os.fchmod(out, 0o600)
            os.fchown(out, parent_info.st_uid, parent_info.st_gid)
            os.fsync(out)
        finally:
            os.close(out)
        try:
            os.replace(temporary, path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            os.fsync(parent_fd)
        except BaseException:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except OSError:
                pass
            raise
    finally:
        os.close(parent_fd)


class WorkshopPrincipalPolicyService:
    """Owner-scoped read/write authority for canonical principal policy."""

    def __init__(
        self,
        data_dir: Path,
        principal_storage: WorkshopPrincipalStorageRegistry,
        *,
        manager: Path = PRINCIPAL_POLICY_MANAGER,
        reader: Path = PRINCIPAL_DOCUMENT_READER,
        on_content_changed: Callable[[PrincipalId], Awaitable[tuple[int, int]]] | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._principal_storage = principal_storage
        self._manager = manager
        self._reader = reader
        self._on_content_changed = on_content_changed
        self._locks: dict[PrincipalId, asyncio.Lock] = {}
        self._last_invalidation: dict[PrincipalId, tuple[str, PrincipalPolicyContextInvalidation]] = {}

    def authority_for_principal(self, principal_id: str | PrincipalId) -> PrincipalPolicyAuthority:
        try:
            namespace = self._principal_storage.for_principal(principal_id)
        except WorkshopStorageNamespaceError as exc:
            raise WorkshopPrincipalPolicyAccessDenied("Principal has no policy authority") from exc
        return PrincipalPolicyAuthority(namespace.principal_id, namespace)

    def _validate_authority(self, authority: PrincipalPolicyAuthority) -> Path:
        try:
            current = self._principal_storage.for_principal(authority.principal_id)
        except WorkshopStorageNamespaceError as exc:
            raise WorkshopPrincipalPolicyAccessDenied("Principal policy authority changed") from exc
        if current != authority.namespace:
            raise WorkshopPrincipalPolicyAccessDenied("Principal policy authority changed")
        return current.home_directory(self._data_dir) / "AGENTS.md"

    def _protected(self) -> bool:
        try:
            info = self._manager.lstat()
        except OSError:
            return False
        return (
            stat.S_ISREG(info.st_mode) and info.st_uid == 0 and not info.st_mode & 0o022 and bool(info.st_mode & 0o111)
        )

    async def read(self, authority: PrincipalPolicyAuthority) -> PrincipalPolicyDocument:
        path = self._validate_authority(authority)
        if self._protected():
            document = await asyncio.to_thread(
                read_protected_principal_document,
                str(authority.principal_id),
                PrincipalDocumentKind.POLICY,
                helper_path=self._reader,
            )
        else:
            document = await asyncio.to_thread(
                read_local_principal_document,
                path,
                PrincipalDocumentKind.POLICY,
            )
        if document.state is not PrincipalDocumentState.PRESENT or document.content is None:
            raise WorkshopPrincipalPolicyStorageError(f"Principal policy is unavailable: {document.reason}")
        return _document(document.content, document.revision)

    async def save(
        self,
        authority: PrincipalPolicyAuthority,
        *,
        expected_revision: str,
        content: str,
    ) -> PrincipalPolicyDocument:
        path = self._validate_authority(authority)
        expected = _validate_revision(expected_revision)
        _normalized, encoded = _normalize(content)
        async with self._locks.setdefault(authority.principal_id, asyncio.Lock()):
            before = await self.read(authority)
            if before.revision != expected:
                raise WorkshopPrincipalPolicyConflict(before.revision)
            if self._protected():
                await self._privileged_write(str(authority.principal_id), expected, encoded)
            else:
                try:
                    await asyncio.to_thread(_local_write, path, expected, encoded)
                except WorkshopPrincipalPolicyError:
                    raise
                except OSError as exc:
                    raise WorkshopPrincipalPolicyStorageError("Principal-policy storage is unavailable") from exc
            after = await self.read(authority)
            if after.content != before.content and self._on_content_changed is not None:
                try:
                    applied, pending = await self._on_content_changed(authority.principal_id)
                    invalidation = PrincipalPolicyContextInvalidation(
                        "pending" if pending else "applied", applied, pending
                    )
                except Exception:
                    invalidation = PrincipalPolicyContextInvalidation("failed", 0, 0)
            else:
                invalidation = PrincipalPolicyContextInvalidation("unchanged", 0, 0)
            self._last_invalidation[authority.principal_id] = (after.revision, invalidation)
            return after

    def context_invalidation(
        self,
        authority: PrincipalPolicyAuthority,
        revision: str,
    ) -> PrincipalPolicyContextInvalidation | None:
        self._validate_authority(authority)
        latest = self._last_invalidation.get(authority.principal_id)
        return latest[1] if latest is not None and latest[0] == revision else None

    async def _privileged_write(self, principal: str, revision: str, content: bytes) -> None:
        process = await asyncio.create_subprocess_exec(
            "sudo",
            "-n",
            str(self._manager),
            "write",
            principal,
            revision,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        output, _ = await process.communicate(content)
        response = output.decode("utf-8", errors="replace").strip()
        if process.returncode == 0 and response == "ok":
            return
        if response.startswith("conflict:") and _REVISION.fullmatch(response[9:]):
            raise WorkshopPrincipalPolicyConflict(response[9:])
        if response == "validation":
            raise WorkshopPrincipalPolicyValidationError("Principal-policy request is invalid")
        if response == "access_denied":
            raise WorkshopPrincipalPolicyAccessDenied("Principal-policy access denied")
        raise WorkshopPrincipalPolicyStorageError("Principal-policy storage is unavailable")


def _helper_main(arguments: list[str]) -> int:
    """Execute one fixed-map write after dropping to the document owner."""
    if len(arguments) != 5 or arguments[0] != "--helper" or arguments[2] != "write":
        sys.stdout.write("validation")
        return 1
    try:
        mapping = json.loads(arguments[1])
    except (TypeError, json.JSONDecodeError):
        sys.stdout.write("validation")
        return 1
    principal, expected = arguments[3], arguments[4]
    entry = mapping.get(principal) if isinstance(mapping, dict) else None
    if (
        not isinstance(entry, dict)
        or not isinstance(entry.get("path"), str)
        or not isinstance(entry.get("os_user"), str)
        or _REVISION.fullmatch(expected) is None
    ):
        sys.stdout.write("access_denied")
        return 1
    raw = sys.stdin.buffer.read(MAX_PRINCIPAL_DOCUMENT_BYTES + 1)
    if len(raw) > MAX_PRINCIPAL_DOCUMENT_BYTES:
        sys.stdout.write("validation")
        return 1
    try:
        content = raw.decode("utf-8")
        _normalize(content)
        account = pwd.getpwnam(entry["os_user"])
        os.initgroups(entry["os_user"], account.pw_gid)
        os.setgid(account.pw_gid)
        os.setuid(account.pw_uid)
        _local_write(Path(entry["path"]), expected, raw)
    except WorkshopPrincipalPolicyConflict as exc:
        sys.stdout.write(f"conflict:{exc.current_revision}")
        return 1
    except WorkshopPrincipalPolicyValidationError:
        sys.stdout.write("validation")
        return 1
    except (KeyError, OSError, WorkshopPrincipalPolicyStorageError):
        sys.stdout.write("storage")
        return 1
    sys.stdout.write("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(_helper_main(sys.argv[1:]))
