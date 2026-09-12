"""Tests for bounded principal-document loading."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from kai.principal_documents import (
    MAX_PRINCIPAL_DOCUMENT_BYTES,
    PrincipalDocumentKind,
    PrincipalDocumentState,
    read_local_principal_document,
    read_protected_principal_document,
)


def test_local_reader_returns_content_and_revision(tmp_path):
    path = tmp_path / "AGENTS.md"
    path.write_text("Owner policy")

    result = read_local_principal_document(path, PrincipalDocumentKind.POLICY)

    assert result.state is PrincipalDocumentState.PRESENT
    assert result.content == "Owner policy"
    assert result.revision == hashlib.sha256(b"Owner policy").hexdigest()


def test_local_reader_rejects_symlink(tmp_path):
    target = tmp_path / "target"
    target.write_text("secret")
    link = tmp_path / "MEMORY.md"
    link.symlink_to(target)

    result = read_local_principal_document(link, PrincipalDocumentKind.FILE_MEMORY)

    assert result.state is PrincipalDocumentState.UNSAFE
    assert result.content is None


def test_local_reader_reports_missing_file(tmp_path):
    result = read_local_principal_document(tmp_path / "AGENTS.md", PrincipalDocumentKind.POLICY)

    assert result.state is PrincipalDocumentState.MISSING
    assert result.content is None


def test_local_reader_rejects_non_utf8_content(tmp_path):
    path = tmp_path / "PREFERENCES.md"
    path.write_bytes(b"\xff")

    result = read_local_principal_document(path, PrincipalDocumentKind.PREFERENCES)

    assert result.state is PrincipalDocumentState.MALFORMED
    assert result.reason == "invalid_utf8"


def test_local_reader_rejects_changed_file_metadata(tmp_path):
    path = tmp_path / "AGENTS.md"
    path.write_text("policy")
    actual = os.stat(path)
    changed = SimpleNamespace(
        st_dev=actual.st_dev,
        st_ino=actual.st_ino,
        st_size=actual.st_size,
        st_mtime_ns=actual.st_mtime_ns + 1,
        st_ctime_ns=actual.st_ctime_ns,
    )

    with patch("kai.principal_documents.os.fstat", side_effect=[actual, changed]):
        result = read_local_principal_document(path, PrincipalDocumentKind.POLICY)

    assert result.state is PrincipalDocumentState.READ_RACE
    assert result.content is None


def test_local_reader_rejects_oversized_file(tmp_path):
    path = tmp_path / "PREFERENCES.md"
    path.write_bytes(b"x" * (MAX_PRINCIPAL_DOCUMENT_BYTES + 1))

    result = read_local_principal_document(path, PrincipalDocumentKind.PREFERENCES)

    assert result.state is PrincipalDocumentState.OVERSIZED
    assert result.content is None


def test_protected_reader_passes_no_path_and_validates_digest():
    content = "private owner content"
    completed = MagicMock(
        returncode=0,
        stdout=(
            '{"version":1,"kind":"personal_preferences","state":"present",'
            '"reason":"verified_owner_read","content":"private owner content",'
            f'"sha256":"{hashlib.sha256(content.encode()).hexdigest()}"}}'
        ),
        stderr="",
    )
    helper = Path("/etc/kai/read-principal-document")
    with patch("kai.principal_documents.subprocess.run", return_value=completed) as run:
        result = read_protected_principal_document(
            "prn_" + "a" * 32,
            PrincipalDocumentKind.PREFERENCES,
            helper_path=helper,
        )

    assert result.content == content
    command = run.call_args.args[0]
    assert command == ["sudo", "-n", str(helper), "prn_" + "a" * 32, "personal_preferences"]
    assert all("PREFERENCES.md" not in argument for argument in command)


def test_protected_reader_rejects_mismatched_digest():
    completed = MagicMock(
        returncode=0,
        stdout=(
            '{"version":1,"kind":"principal_policy","state":"present",'
            '"reason":"verified_owner_read","content":"policy","sha256":"bad"}'
        ),
        stderr="",
    )
    with patch("kai.principal_documents.subprocess.run", return_value=completed):
        result = read_protected_principal_document(
            "prn_" + "a" * 32,
            PrincipalDocumentKind.POLICY,
        )

    assert result.state is PrincipalDocumentState.MALFORMED
    assert result.content is None


def test_protected_reader_rejects_nonzero_helper_exit():
    completed = MagicMock(returncode=65, stdout="", stderr="rejected")
    with patch("kai.principal_documents.subprocess.run", return_value=completed):
        result = read_protected_principal_document(
            "prn_" + "a" * 32,
            PrincipalDocumentKind.POLICY,
        )

    assert result.state is PrincipalDocumentState.UNREADABLE
    assert result.reason == "helper_rejected"


def test_protected_reader_rejects_invalid_result_shape():
    completed = MagicMock(returncode=0, stdout='{"version":1}', stderr="")
    with patch("kai.principal_documents.subprocess.run", return_value=completed):
        result = read_protected_principal_document(
            "prn_" + "a" * 32,
            PrincipalDocumentKind.POLICY,
        )

    assert result.state is PrincipalDocumentState.MALFORMED
    assert result.reason == "helper_malformed"


def test_protected_reader_preserves_typed_unavailable_state():
    completed = MagicMock(
        returncode=0,
        stdout=(
            '{"version":1,"kind":"principal_policy","state":"wrong_owner",'
            '"reason":"file_owner_mismatch","content":null,"sha256":null}'
        ),
        stderr="",
    )
    with patch("kai.principal_documents.subprocess.run", return_value=completed):
        result = read_protected_principal_document(
            "prn_" + "a" * 32,
            PrincipalDocumentKind.POLICY,
        )

    assert result.state is PrincipalDocumentState.WRONG_OWNER
    assert result.reason == "file_owner_mismatch"
