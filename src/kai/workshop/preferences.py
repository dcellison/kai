"""Canonical principal-owned preference document authority."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import stat
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from kai.workshop.domain import PrincipalId
from kai.workshop.storage_namespaces import (
    WorkshopPrincipalStorageNamespace,
    WorkshopPrincipalStorageRegistry,
    WorkshopStorageNamespaceError,
)

MAX_PREFERENCE_BYTES = 32 * 1024
MAX_PREFERENCE_REVISIONS = 20
PREFERENCE_MANAGER = Path("/etc/kai/manage-principal-preferences")
_PREFERENCE_FILE = "PREFERENCES.md"
_HISTORY_ROOT = "preference-revisions"
_PRINCIPAL_PATTERN = re.compile(r"^prn_[0-9a-f]{32}$")
_REVISION_PATTERN = re.compile(r"^pref_v1_([0-9a-f]{32})_([0-9a-f]{32})$")
_MISSING_REVISION = f"pref_v1_{hashlib.sha256(b'').hexdigest()[:32]}_{'0' * 32}"
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)


class WorkshopPreferenceError(RuntimeError):
    """Base failure for canonical preference operations."""


class WorkshopPreferenceAccessDenied(WorkshopPreferenceError):
    """The authenticated principal has no preference authority."""


class WorkshopPreferenceValidationError(WorkshopPreferenceError):
    """Preference content or a revision identifier is invalid."""


class WorkshopPreferenceConflict(WorkshopPreferenceError):
    """The expected preference revision is no longer current."""

    def __init__(self, current_revision: str) -> None:
        super().__init__("Preferences changed since they were opened")
        self.current_revision = current_revision


class WorkshopPreferenceRevisionNotFound(WorkshopPreferenceError):
    """A private preference revision does not exist."""


class WorkshopPreferenceStorageError(WorkshopPreferenceError):
    """Private preference storage failed closed."""


@dataclass(frozen=True, slots=True)
class PreferenceAuthority:
    principal_id: PrincipalId
    namespace: WorkshopPrincipalStorageNamespace


@dataclass(frozen=True, slots=True)
class PreferenceDocument:
    content: str
    revision: str
    updated_at: str | None
    size_bytes: int
    max_bytes: int = MAX_PREFERENCE_BYTES
    editable: bool = True


@dataclass(frozen=True, slots=True)
class PreferenceRevision:
    revision: str
    updated_at: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class PreferenceRevisionHistory:
    revisions: tuple[PreferenceRevision, ...]
    limit: int = MAX_PREFERENCE_REVISIONS


@dataclass(frozen=True, slots=True)
class _OpenedDocument:
    content: bytes
    info: os.stat_result | None


def _timestamp(info: os.stat_result) -> str:
    return datetime.fromtimestamp(info.st_mtime, tz=UTC).isoformat().replace("+00:00", "Z")


def _revision(content: bytes, info: os.stat_result | None) -> str:
    if info is None:
        return _MISSING_REVISION
    content_hash = hashlib.sha256(content).hexdigest()[:32]
    state = f"{info.st_dev}:{info.st_ino}:{info.st_mtime_ns}:{info.st_size}".encode("ascii")
    state_hash = hashlib.sha256(state).hexdigest()[:32]
    return f"pref_v1_{content_hash}_{state_hash}"


def _normalize_content(content: str) -> tuple[str, bytes]:
    if not isinstance(content, str):
        raise WorkshopPreferenceValidationError("Preference content must be text")
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    if "\x00" in normalized:
        raise WorkshopPreferenceValidationError("Preference content contains an invalid character")
    try:
        encoded = normalized.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise WorkshopPreferenceValidationError("Preference content is not valid UTF-8") from exc
    if len(encoded) > MAX_PREFERENCE_BYTES:
        raise WorkshopPreferenceValidationError(f"Preference content exceeds the {MAX_PREFERENCE_BYTES}-byte limit")
    return normalized, encoded


def _validate_revision(value: str) -> str:
    if not isinstance(value, str) or _REVISION_PATTERN.fullmatch(value) is None:
        raise WorkshopPreferenceValidationError("Preference revision is invalid")
    return value


def _validate_principal(value: str) -> str:
    if _PRINCIPAL_PATTERN.fullmatch(value) is None:
        raise WorkshopPreferenceAccessDenied("Preference authority is invalid")
    return value


def _read_fd(fd: int) -> bytes:
    chunks: list[bytes] = []
    remaining = MAX_PREFERENCE_BYTES + 1
    while remaining > 0:
        chunk = os.read(fd, min(remaining, 64 * 1024))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    content = b"".join(chunks)
    if len(content) > MAX_PREFERENCE_BYTES:
        raise WorkshopPreferenceValidationError("Stored preference document exceeds the size limit")
    try:
        content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise WorkshopPreferenceValidationError("Stored preference document is not valid UTF-8") from exc
    return content


def _validate_private_directory(info: os.stat_result, *, owner_uid: int | None = None) -> None:
    if not stat.S_ISDIR(info.st_mode) or info.st_uid == 0 or info.st_mode & 0o077:
        raise WorkshopPreferenceStorageError("Preference directory is not a private user directory")
    if owner_uid is not None and info.st_uid != owner_uid:
        raise WorkshopPreferenceStorageError("Preference directory ownership changed")


def _validate_private_file(info: os.stat_result, *, owner_uid: int) -> None:
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != owner_uid or info.st_mode & 0o077:
        raise WorkshopPreferenceStorageError("Preference document is not a private regular file")


class _FilesystemPreferenceStore:
    """Fd-anchored preference operations used directly and by the root helper."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def _open_data_root(self) -> tuple[int, os.stat_result]:
        data_fd = os.open(self._data_dir, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
        data_info = os.fstat(data_fd)
        if not stat.S_ISDIR(data_info.st_mode) or data_info.st_uid == 0 or data_info.st_mode & 0o022:
            os.close(data_fd)
            raise WorkshopPreferenceStorageError("Data directory ownership or mode is invalid")
        return data_fd, data_info

    def _open_principal(self, principal: str) -> tuple[int, os.stat_result]:
        _validate_principal(principal)
        data_fd, data_info = self._open_data_root()
        try:
            root_fd = os.open(
                "preferences",
                os.O_RDONLY | _DIRECTORY | _NOFOLLOW,
                dir_fd=data_fd,
            )
        finally:
            os.close(data_fd)
        root_info = os.fstat(root_fd)
        if not stat.S_ISDIR(root_info.st_mode) or root_info.st_uid != data_info.st_uid or root_info.st_mode & 0o022:
            os.close(root_fd)
            raise WorkshopPreferenceStorageError("Preference root is not a protected directory")
        try:
            principal_fd = os.open(
                principal,
                os.O_RDONLY | _DIRECTORY | _NOFOLLOW,
                dir_fd=root_fd,
            )
        finally:
            os.close(root_fd)
        info = os.fstat(principal_fd)
        try:
            _validate_private_directory(info)
        except BaseException:
            os.close(principal_fd)
            raise
        return principal_fd, info

    def _read_current(self, principal_fd: int, owner_uid: int) -> _OpenedDocument:
        try:
            fd = os.open(
                _PREFERENCE_FILE,
                os.O_RDONLY | _NOFOLLOW | _NONBLOCK,
                dir_fd=principal_fd,
            )
        except FileNotFoundError:
            return _OpenedDocument(b"", None)
        try:
            info = os.fstat(fd)
            _validate_private_file(info, owner_uid=owner_uid)
            return _OpenedDocument(_read_fd(fd), info)
        finally:
            os.close(fd)

    @staticmethod
    def _document(opened: _OpenedDocument) -> PreferenceDocument:
        return PreferenceDocument(
            content=opened.content.decode("utf-8"),
            revision=_revision(opened.content, opened.info),
            updated_at=_timestamp(opened.info) if opened.info is not None else None,
            size_bytes=len(opened.content),
        )

    def snapshot(self, principal: str) -> PreferenceDocument:
        principal_fd, principal_info = self._open_principal(principal)
        try:
            return self._document(self._read_current(principal_fd, principal_info.st_uid))
        finally:
            os.close(principal_fd)

    def _open_history(self, principal: str, *, create: bool) -> tuple[int, os.stat_result]:
        data_fd, data_info = self._open_data_root()
        try:
            if create:
                try:
                    os.mkdir(_HISTORY_ROOT, 0o700, dir_fd=data_fd)
                    os.chown(_HISTORY_ROOT, data_info.st_uid, data_info.st_gid, dir_fd=data_fd)
                except FileExistsError:
                    pass
            history_root_fd = os.open(
                _HISTORY_ROOT,
                os.O_RDONLY | _DIRECTORY | _NOFOLLOW,
                dir_fd=data_fd,
            )
        finally:
            os.close(data_fd)
        history_root_info = os.fstat(history_root_fd)
        try:
            _validate_private_directory(history_root_info, owner_uid=data_info.st_uid)
            if create:
                try:
                    os.mkdir(principal, 0o700, dir_fd=history_root_fd)
                    os.chown(principal, data_info.st_uid, data_info.st_gid, dir_fd=history_root_fd)
                except FileExistsError:
                    pass
            principal_fd = os.open(
                principal,
                os.O_RDONLY | _DIRECTORY | _NOFOLLOW,
                dir_fd=history_root_fd,
            )
        finally:
            os.close(history_root_fd)
        principal_info = os.fstat(principal_fd)
        try:
            _validate_private_directory(principal_info, owner_uid=data_info.st_uid)
        except BaseException:
            os.close(principal_fd)
            raise
        return principal_fd, principal_info

    @staticmethod
    def _atomic_write(
        directory_fd: int,
        name: str,
        content: bytes,
        *,
        owner_uid: int,
        owner_gid: int,
    ) -> None:
        temporary = f".{name}.{secrets.token_hex(12)}.tmp"
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            offset = 0
            while offset < len(content):
                offset += os.write(fd, content[offset:])
            os.fchmod(fd, 0o600)
            os.fchown(fd, owner_uid, owner_gid)
            os.fsync(fd)
        except BaseException:
            os.close(fd)
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except OSError:
                pass
            raise
        else:
            os.close(fd)
        try:
            os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            os.fsync(directory_fd)
        except BaseException:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except OSError:
                pass
            raise

    def _archive(
        self,
        principal: str,
        opened: _OpenedDocument,
    ) -> None:
        if opened.info is None:
            return
        revision = _revision(opened.content, opened.info)
        history_fd, history_info = self._open_history(principal, create=True)
        try:
            try:
                existing_fd = os.open(
                    revision,
                    os.O_RDONLY | _NOFOLLOW | _NONBLOCK,
                    dir_fd=history_fd,
                )
            except FileNotFoundError:
                self._atomic_write(
                    history_fd,
                    revision,
                    opened.content,
                    owner_uid=history_info.st_uid,
                    owner_gid=history_info.st_gid,
                )
                os.utime(
                    revision,
                    ns=(opened.info.st_mtime_ns, opened.info.st_mtime_ns),
                    dir_fd=history_fd,
                    follow_symlinks=False,
                )
            else:
                try:
                    info = os.fstat(existing_fd)
                    _validate_private_file(info, owner_uid=history_info.st_uid)
                    if _read_fd(existing_fd) != opened.content:
                        raise WorkshopPreferenceStorageError("Preference revision content is inconsistent")
                finally:
                    os.close(existing_fd)
            self._prune_history(history_fd, history_info.st_uid)
        finally:
            os.close(history_fd)

    @staticmethod
    def _prune_history(history_fd: int, owner_uid: int) -> None:
        entries: list[tuple[int, str]] = []
        for name in os.listdir(history_fd):
            if _REVISION_PATTERN.fullmatch(name) is None:
                raise WorkshopPreferenceStorageError("Preference revision directory contains an invalid entry")
            info = os.stat(name, dir_fd=history_fd, follow_symlinks=False)
            _validate_private_file(info, owner_uid=owner_uid)
            entries.append((info.st_mtime_ns, name))
        entries.sort(reverse=True)
        for _mtime, name in entries[MAX_PREFERENCE_REVISIONS:]:
            os.unlink(name, dir_fd=history_fd)

    def write(self, principal: str, expected_revision: str, content: str) -> PreferenceDocument:
        expected = _validate_revision(expected_revision)
        _normalized, encoded = _normalize_content(content)
        principal_fd, principal_info = self._open_principal(principal)
        try:
            opened = self._read_current(principal_fd, principal_info.st_uid)
            current_revision = _revision(opened.content, opened.info)
            if current_revision != expected:
                raise WorkshopPreferenceConflict(current_revision)
            self._archive(principal, opened)
            # Re-read immediately before replacement so a concurrent direct
            # editor is detected before the atomic name swap.
            rechecked = self._read_current(principal_fd, principal_info.st_uid)
            rechecked_revision = _revision(rechecked.content, rechecked.info)
            if rechecked_revision != expected:
                raise WorkshopPreferenceConflict(rechecked_revision)
            self._atomic_write(
                principal_fd,
                _PREFERENCE_FILE,
                encoded,
                owner_uid=principal_info.st_uid,
                owner_gid=principal_info.st_gid,
            )
            return self._document(self._read_current(principal_fd, principal_info.st_uid))
        finally:
            os.close(principal_fd)

    def history(self, principal: str) -> PreferenceRevisionHistory:
        _validate_principal(principal)
        try:
            history_fd, history_info = self._open_history(principal, create=False)
        except FileNotFoundError:
            return PreferenceRevisionHistory(())
        try:
            revisions: list[PreferenceRevision] = []
            for name in os.listdir(history_fd):
                match = _REVISION_PATTERN.fullmatch(name)
                if match is None:
                    raise WorkshopPreferenceStorageError("Preference revision directory contains an invalid entry")
                fd = os.open(name, os.O_RDONLY | _NOFOLLOW | _NONBLOCK, dir_fd=history_fd)
                try:
                    info = os.fstat(fd)
                    _validate_private_file(info, owner_uid=history_info.st_uid)
                    content = _read_fd(fd)
                finally:
                    os.close(fd)
                if hashlib.sha256(content).hexdigest()[:32] != match.group(1):
                    raise WorkshopPreferenceStorageError("Preference revision content is inconsistent")
                revisions.append(PreferenceRevision(name, _timestamp(info), len(content)))
            revisions.sort(key=lambda item: item.updated_at, reverse=True)
            return PreferenceRevisionHistory(tuple(revisions))
        finally:
            os.close(history_fd)

    def restore(
        self,
        principal: str,
        target_revision: str,
        expected_revision: str,
    ) -> PreferenceDocument:
        target = _validate_revision(target_revision)
        try:
            history_fd, history_info = self._open_history(principal, create=False)
        except FileNotFoundError as exc:
            raise WorkshopPreferenceRevisionNotFound("Preference revision was not found") from exc
        try:
            try:
                fd = os.open(target, os.O_RDONLY | _NOFOLLOW | _NONBLOCK, dir_fd=history_fd)
            except FileNotFoundError as exc:
                raise WorkshopPreferenceRevisionNotFound("Preference revision was not found") from exc
            try:
                info = os.fstat(fd)
                _validate_private_file(info, owner_uid=history_info.st_uid)
                content = _read_fd(fd)
            finally:
                os.close(fd)
            match = _REVISION_PATTERN.fullmatch(target)
            assert match is not None
            if hashlib.sha256(content).hexdigest()[:32] != match.group(1):
                raise WorkshopPreferenceStorageError("Preference revision content is inconsistent")
        finally:
            os.close(history_fd)
        return self.write(principal, expected_revision, content.decode("utf-8"))


class WorkshopPreferenceService:
    """Principal-scoped preference authority above local or privileged storage."""

    def __init__(
        self,
        data_dir: Path,
        principal_storage: WorkshopPrincipalStorageRegistry,
        *,
        privileged_helper: Path = PREFERENCE_MANAGER,
    ) -> None:
        self._data_dir = data_dir
        self._principal_storage = principal_storage
        self._local = _FilesystemPreferenceStore(data_dir)
        self._privileged_helper = privileged_helper
        self._locks: dict[PrincipalId, asyncio.Lock] = {}

    def authority_for_principal(self, principal_id: str | PrincipalId) -> PreferenceAuthority:
        try:
            namespace = self._principal_storage.for_principal(principal_id)
        except WorkshopStorageNamespaceError as exc:
            raise WorkshopPreferenceAccessDenied("Principal has no preference authority") from exc
        return PreferenceAuthority(namespace.principal_id, namespace)

    def _validate_authority(self, authority: PreferenceAuthority) -> str:
        try:
            namespace = self._principal_storage.for_principal(authority.principal_id)
        except WorkshopStorageNamespaceError as exc:
            raise WorkshopPreferenceAccessDenied("Preference authority changed") from exc
        if namespace.principal_id != authority.namespace.principal_id:
            raise WorkshopPreferenceAccessDenied("Preference authority changed")
        return _validate_principal(str(namespace.principal_id))

    def _lock(self, principal_id: PrincipalId) -> asyncio.Lock:
        return self._locks.setdefault(principal_id, asyncio.Lock())

    async def read(self, authority: PreferenceAuthority) -> PreferenceDocument:
        principal = self._validate_authority(authority)
        return await self._operation("snapshot", principal)

    async def save(
        self,
        authority: PreferenceAuthority,
        *,
        expected_revision: str,
        content: str,
    ) -> PreferenceDocument:
        principal = self._validate_authority(authority)
        _normalize_content(content)
        _validate_revision(expected_revision)
        async with self._lock(authority.principal_id):
            return await self._operation(
                "write",
                principal,
                expected_revision,
                input_text=content,
            )

    async def history(self, authority: PreferenceAuthority) -> PreferenceRevisionHistory:
        principal = self._validate_authority(authority)
        return await self._operation("history", principal)

    async def restore(
        self,
        authority: PreferenceAuthority,
        *,
        target_revision: str,
        expected_revision: str,
    ) -> PreferenceDocument:
        principal = self._validate_authority(authority)
        _validate_revision(target_revision)
        _validate_revision(expected_revision)
        async with self._lock(authority.principal_id):
            return await self._operation(
                "restore",
                principal,
                target_revision,
                expected_revision,
            )

    async def _operation(
        self,
        operation: Literal["snapshot", "write", "history", "restore"],
        principal: str,
        *arguments: str,
        input_text: str | None = None,
    ) -> Any:
        local_method = getattr(self._local, operation)
        try:
            return await asyncio.to_thread(
                local_method, principal, *arguments, **({"content": input_text} if operation == "write" else {})
            )
        except PermissionError:
            return await self._privileged_operation(
                operation,
                principal,
                *arguments,
                input_text=input_text,
            )
        except WorkshopPreferenceError:
            raise
        except OSError as exc:
            raise WorkshopPreferenceStorageError("Preference storage is unavailable") from exc

    async def _privileged_operation(
        self,
        operation: str,
        principal: str,
        *arguments: str,
        input_text: str | None,
    ) -> Any:
        try:
            helper_info = self._privileged_helper.lstat()
        except OSError as exc:
            raise WorkshopPreferenceStorageError("Preference storage is unavailable") from exc
        if (
            not stat.S_ISREG(helper_info.st_mode)
            or helper_info.st_uid != 0
            or helper_info.st_mode & 0o022
            or not helper_info.st_mode & 0o111
        ):
            raise WorkshopPreferenceStorageError("Preference storage is unavailable")
        process = await asyncio.create_subprocess_exec(
            "sudo",
            "-n",
            str(self._privileged_helper),
            operation,
            principal,
            *arguments,
            stdin=asyncio.subprocess.PIPE if input_text is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        output, _ = await process.communicate(input_text.encode("utf-8") if input_text is not None else None)
        try:
            payload = json.loads(output)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkshopPreferenceStorageError("Preference storage is unavailable") from exc
        if process.returncode != 0:
            self._raise_helper_error(payload)
        return _decode_helper_result(operation, payload)

    @staticmethod
    def _raise_helper_error(payload: object) -> None:
        if not isinstance(payload, dict):
            raise WorkshopPreferenceStorageError("Preference storage is unavailable")
        code = payload.get("error")
        if code == "conflict" and isinstance(payload.get("current_revision"), str):
            raise WorkshopPreferenceConflict(str(payload["current_revision"]))
        if code == "not_found":
            raise WorkshopPreferenceRevisionNotFound("Preference revision was not found")
        if code == "validation":
            raise WorkshopPreferenceValidationError("Preference request is invalid")
        if code == "access_denied":
            raise WorkshopPreferenceAccessDenied("Preference access denied")
        raise WorkshopPreferenceStorageError("Preference storage is unavailable")


def _decode_helper_result(operation: str, payload: object) -> Any:
    if not isinstance(payload, dict):
        raise WorkshopPreferenceStorageError("Preference storage returned an invalid response")
    try:
        if operation in {"snapshot", "write", "restore"}:
            return PreferenceDocument(**payload)
        if operation == "history":
            raw_revisions = payload["revisions"]
            if not isinstance(raw_revisions, list):
                raise TypeError
            return PreferenceRevisionHistory(
                tuple(PreferenceRevision(**item) for item in raw_revisions),
                int(payload["limit"]),
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkshopPreferenceStorageError("Preference storage returned an invalid response") from exc
    raise WorkshopPreferenceStorageError("Preference storage returned an invalid response")


def _helper_error(error: str, **details: object) -> int:
    sys.stdout.write(json.dumps({"error": error, **details}, separators=(",", ":")))
    return 1


def _helper_main(arguments: list[str]) -> int:
    if len(arguments) < 3 or arguments[0] != "--helper":
        return _helper_error("validation")
    data_dir = Path(arguments[1])
    operation = arguments[2]
    operation_arguments = arguments[3:]
    store = _FilesystemPreferenceStore(data_dir)
    try:
        if operation == "snapshot" and len(operation_arguments) == 1:
            result: object = store.snapshot(operation_arguments[0])
        elif operation == "write" and len(operation_arguments) == 2:
            raw = sys.stdin.buffer.read(MAX_PREFERENCE_BYTES + 1)
            if len(raw) > MAX_PREFERENCE_BYTES:
                raise WorkshopPreferenceValidationError("Preference content is too large")
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise WorkshopPreferenceValidationError("Preference content is not UTF-8") from exc
            result = store.write(operation_arguments[0], operation_arguments[1], content)
        elif operation == "history" and len(operation_arguments) == 1:
            result = store.history(operation_arguments[0])
        elif operation == "restore" and len(operation_arguments) == 3:
            result = store.restore(
                operation_arguments[0],
                operation_arguments[1],
                operation_arguments[2],
            )
        else:
            raise WorkshopPreferenceValidationError("Preference operation is invalid")
    except WorkshopPreferenceConflict as exc:
        return _helper_error("conflict", current_revision=exc.current_revision)
    except WorkshopPreferenceRevisionNotFound:
        return _helper_error("not_found")
    except WorkshopPreferenceValidationError:
        return _helper_error("validation")
    except WorkshopPreferenceAccessDenied:
        return _helper_error("access_denied")
    except (WorkshopPreferenceStorageError, OSError):
        return _helper_error("storage")
    sys.stdout.write(json.dumps(asdict(result), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(_helper_main(sys.argv[1:]))
