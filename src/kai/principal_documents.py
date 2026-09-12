"""Bounded, owner-authorized loading of principal context documents."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final

PRINCIPAL_DOCUMENT_READER: Final = Path("/etc/kai/read-principal-document")
MAX_PRINCIPAL_DOCUMENT_BYTES: Final = 128 * 1024


class PrincipalDocumentKind(StrEnum):
    POLICY = "principal_policy"
    PREFERENCES = "personal_preferences"
    FILE_MEMORY = "file_memory"


class PrincipalDocumentState(StrEnum):
    PRESENT = "present"
    MISSING = "missing"
    UNREADABLE = "unreadable"
    UNSAFE = "unsafe"
    OVERSIZED = "oversized"
    MALFORMED = "malformed"
    READ_RACE = "read_race"
    WRONG_OWNER = "wrong_owner"


@dataclass(frozen=True, slots=True)
class PrincipalDocument:
    """One allowlisted document result, with no caller-controlled path."""

    kind: PrincipalDocumentKind
    state: PrincipalDocumentState
    content: str | None = field(repr=False)
    revision: str | None
    reason: str

    @property
    def delivered(self) -> bool:
        return self.state is PrincipalDocumentState.PRESENT


@dataclass(frozen=True, slots=True)
class PrincipalDocumentReport:
    """Loaded documents whose metadata can feed durable context manifests.

    Content is intentionally excluded from representations and manifest code
    must copy only state, reason, and revision.
    """

    policy: PrincipalDocument | None = None
    preferences: PrincipalDocument | None = None
    file_memory: PrincipalDocument | None = None


class PrincipalPolicyUnavailable(RuntimeError):
    """The principal's authoritative policy could not be read safely."""

    def __init__(self, document: PrincipalDocument) -> None:
        super().__init__(f"Principal policy is unavailable: {document.reason}")
        self.document = document


def _result(
    kind: PrincipalDocumentKind,
    state: PrincipalDocumentState,
    *,
    content: str | None = None,
    reason: str | None = None,
) -> PrincipalDocument:
    revision = hashlib.sha256(content.encode("utf-8")).hexdigest() if content is not None else None
    return PrincipalDocument(kind, state, content, revision, reason or state.value)


def _decode_helper_result(kind: PrincipalDocumentKind, stdout: str) -> PrincipalDocument:
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Principal document helper returned malformed JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"version", "kind", "state", "reason", "content", "sha256"}:
        raise ValueError("Principal document helper returned an invalid result shape")
    if type(payload["version"]) is not int or payload["version"] != 1 or payload["kind"] != kind.value:
        raise ValueError("Principal document helper returned mismatched authority")
    if not isinstance(payload["state"], str) or not isinstance(payload["reason"], str) or not payload["reason"]:
        raise ValueError("Principal document helper returned invalid state metadata")
    state = PrincipalDocumentState(payload["state"])
    reason = payload["reason"]
    content = payload["content"]
    digest = payload["sha256"]
    if state is PrincipalDocumentState.PRESENT:
        if not isinstance(content, str) or not isinstance(digest, str):
            raise ValueError("Principal document helper omitted delivered content metadata")
        actual = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if digest != actual:
            raise ValueError("Principal document helper returned a mismatched digest")
        return PrincipalDocument(kind, state, content, actual, reason)
    if content is not None or digest is not None:
        raise ValueError("Unavailable principal document unexpectedly included content")
    return PrincipalDocument(kind, state, None, None, reason)


def read_protected_principal_document(
    principal_id: str,
    kind: PrincipalDocumentKind,
    *,
    helper_path: Path = PRINCIPAL_DOCUMENT_READER,
) -> PrincipalDocument:
    """Read one document through the installed root-owned allowlist helper."""
    try:
        result = subprocess.run(
            ["sudo", "-n", str(helper_path), principal_id, kind.value],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return _result(kind, PrincipalDocumentState.UNREADABLE, reason="helper_unavailable")
    if result.returncode != 0:
        return _result(kind, PrincipalDocumentState.UNREADABLE, reason="helper_rejected")
    try:
        return _decode_helper_result(kind, result.stdout)
    except (ValueError, KeyError):
        return _result(kind, PrincipalDocumentState.MALFORMED, reason="helper_malformed")


def read_local_principal_document(path: Path, kind: PrincipalDocumentKind) -> PrincipalDocument:
    """Development-mode implementation of the same bounded typed contract."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return _result(kind, PrincipalDocumentState.MISSING)
    except OSError as exc:
        state = PrincipalDocumentState.UNSAFE if exc.errno == errno.ELOOP else PrincipalDocumentState.UNREADABLE
        return _result(kind, state)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            return _result(kind, PrincipalDocumentState.UNSAFE)
        if before.st_size > MAX_PRINCIPAL_DOCUMENT_BYTES:
            return _result(kind, PrincipalDocumentState.OVERSIZED)
        chunks: list[bytes] = []
        remaining = MAX_PRINCIPAL_DOCUMENT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(raw) > MAX_PRINCIPAL_DOCUMENT_BYTES:
        return _result(kind, PrincipalDocumentState.OVERSIZED)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
        return _result(kind, PrincipalDocumentState.READ_RACE)
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        return _result(kind, PrincipalDocumentState.MALFORMED, reason="invalid_utf8")
    return _result(kind, PrincipalDocumentState.PRESENT, content=content, reason="verified_owner_read")


def load_principal_document(
    *,
    principal_id: str,
    kind: PrincipalDocumentKind,
    path: Path,
    protected: bool,
) -> PrincipalDocument:
    """Resolve one document without exposing a model- or caller-selected path."""
    if protected:
        return read_protected_principal_document(principal_id, kind)
    return read_local_principal_document(path, kind)
