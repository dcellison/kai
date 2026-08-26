"""Canonical Workshop preference authority and API security tests."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kai.backend import (
    ApiContext,
    PrincipalStorageNamespaceResolver,
    build_session_context,
    configure_principal_storage_namespaces,
)
from kai.workshop.client_api import register_workshop_read_routes
from kai.workshop.domain import PrincipalId, RuntimeProfileId
from kai.workshop.preferences import (
    MAX_PREFERENCE_BYTES,
    MAX_PREFERENCE_REVISIONS,
    WorkshopPreferenceAccessDenied,
    WorkshopPreferenceConflict,
    WorkshopPreferenceRevisionNotFound,
    WorkshopPreferenceService,
    WorkshopPreferenceStorageError,
    WorkshopPreferenceValidationError,
    _OpenedDocument,
)
from kai.workshop.storage_namespaces import (
    WorkshopPrincipalStorageNamespace,
    WorkshopPrincipalStorageRegistry,
)
from kai.workshop.store import WorkshopEventStore


@dataclass(frozen=True)
class _PreferenceFixture:
    data_dir: Path
    alice: PrincipalId
    bob: PrincipalId
    registry: WorkshopPrincipalStorageRegistry
    service: WorkshopPreferenceService


@pytest.fixture
def preferences(tmp_path: Path) -> _PreferenceFixture:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    preferences_root = data_dir / "preferences"
    preferences_root.mkdir(mode=0o711)
    os.chmod(preferences_root, 0o711)
    alice = PrincipalId.new()
    bob = PrincipalId.new()
    namespaces = (
        WorkshopPrincipalStorageNamespace(alice, RuntimeProfileId.new(), 101),
        WorkshopPrincipalStorageNamespace(bob, RuntimeProfileId.new(), 202),
    )
    for principal, content in ((alice, "# Preferences\n\nAlice original.\n"), (bob, "Bob private.\n")):
        directory = preferences_root / str(principal)
        directory.mkdir(mode=0o700)
        os.chmod(directory, 0o700)
        document = directory / "PREFERENCES.md"
        document.write_text(content)
        os.chmod(document, 0o600)
    registry = WorkshopPrincipalStorageRegistry(namespaces)
    return _PreferenceFixture(
        data_dir,
        alice,
        bob,
        registry,
        WorkshopPreferenceService(
            data_dir,
            registry,
            privileged_helper=tmp_path / "missing-preference-helper",
        ),
    )


@pytest.mark.asyncio
async def test_save_is_revision_checked_atomic_private_and_recoverable(
    preferences: _PreferenceFixture,
) -> None:
    authority = preferences.service.authority_for_principal(preferences.alice)
    original = await preferences.service.read(authority)

    saved = await preferences.service.save(
        authority,
        expected_revision=original.revision,
        content="# Preferences\r\n\r\nAlice updated.\r\n",
    )

    path = preferences.data_dir / "preferences" / str(preferences.alice) / "PREFERENCES.md"
    assert saved.content == "# Preferences\n\nAlice updated.\n"
    assert saved.revision != original.revision
    assert path.read_text() == saved.content
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(path.parent.glob(".*.tmp"))

    history = await preferences.service.history(authority)
    assert history.limit == MAX_PREFERENCE_REVISIONS
    assert [item.revision for item in history.revisions] == [original.revision]

    restored = await preferences.service.restore(
        authority,
        target_revision=original.revision,
        expected_revision=saved.revision,
    )
    assert restored.content == original.content
    assert restored.revision not in {original.revision, saved.revision}


@pytest.mark.asyncio
async def test_stale_save_and_restore_fail_without_mutation(preferences: _PreferenceFixture) -> None:
    authority = preferences.service.authority_for_principal(preferences.alice)
    original = await preferences.service.read(authority)
    current = await preferences.service.save(
        authority,
        expected_revision=original.revision,
        content="Current preferences.\n",
    )

    with pytest.raises(WorkshopPreferenceConflict) as save_conflict:
        await preferences.service.save(
            authority,
            expected_revision=original.revision,
            content="Stale overwrite.\n",
        )
    assert save_conflict.value.current_revision == current.revision

    with pytest.raises(WorkshopPreferenceConflict) as restore_conflict:
        await preferences.service.restore(
            authority,
            target_revision=original.revision,
            expected_revision=original.revision,
        )
    assert restore_conflict.value.current_revision == current.revision
    assert (await preferences.service.read(authority)).content == "Current preferences.\n"


@pytest.mark.asyncio
async def test_concurrent_saves_from_one_revision_allow_exactly_one_writer(
    preferences: _PreferenceFixture,
) -> None:
    authority = preferences.service.authority_for_principal(preferences.alice)
    original = await preferences.service.read(authority)

    results = await asyncio.gather(
        preferences.service.save(
            authority,
            expected_revision=original.revision,
            content="First concurrent edit.\n",
        ),
        preferences.service.save(
            authority,
            expected_revision=original.revision,
            content="Second concurrent edit.\n",
        ),
        return_exceptions=True,
    )

    assert sum(not isinstance(result, BaseException) for result in results) == 1
    assert sum(isinstance(result, WorkshopPreferenceConflict) for result in results) == 1
    current = await preferences.service.read(authority)
    assert current.content in {"First concurrent edit.\n", "Second concurrent edit.\n"}


@pytest.mark.asyncio
async def test_atomic_replace_failure_preserves_current_document(
    preferences: _PreferenceFixture,
) -> None:
    authority = preferences.service.authority_for_principal(preferences.alice)
    original = await preferences.service.read(authority)
    real_replace = os.replace

    def fail_current_replace(
        src: str,
        dst: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        if dst == "PREFERENCES.md":
            raise OSError("injected replace failure")
        real_replace(
            src,
            dst,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    with (
        patch("kai.workshop.preferences.os.replace", side_effect=fail_current_replace),
        pytest.raises(WorkshopPreferenceStorageError),
    ):
        await preferences.service.save(
            authority,
            expected_revision=original.revision,
            content="Must not become current.\n",
        )

    current = await preferences.service.read(authority)
    assert current.content == original.content
    assert current.revision == original.revision
    directory = preferences.data_dir / "preferences" / str(preferences.alice)
    assert not list(directory.glob(".*.tmp"))


@pytest.mark.asyncio
async def test_direct_editor_race_is_detected_before_atomic_replace(
    preferences: _PreferenceFixture,
) -> None:
    authority = preferences.service.authority_for_principal(preferences.alice)
    original = await preferences.service.read(authority)
    path = preferences.data_dir / "preferences" / str(preferences.alice) / "PREFERENCES.md"
    real_read = preferences.service._local._read_current
    read_count = 0

    def race_before_recheck(principal_fd: int, owner_uid: int) -> _OpenedDocument:
        nonlocal read_count
        read_count += 1
        if read_count == 2:
            path.write_text("External direct edit.\n")
            os.chmod(path, 0o600)
        return real_read(principal_fd, owner_uid)

    with (
        patch.object(
            preferences.service._local,
            "_read_current",
            side_effect=race_before_recheck,
        ),
        pytest.raises(WorkshopPreferenceConflict),
    ):
        await preferences.service.save(
            authority,
            expected_revision=original.revision,
            content="Stale Workshop edit.\n",
        )

    assert path.read_text() == "External direct edit.\n"


@pytest.mark.asyncio
async def test_missing_document_has_stable_bootstrap_revision(preferences: _PreferenceFixture) -> None:
    path = preferences.data_dir / "preferences" / str(preferences.alice) / "PREFERENCES.md"
    path.unlink()
    authority = preferences.service.authority_for_principal(preferences.alice)

    missing = await preferences.service.read(authority)
    assert missing.content == ""
    assert missing.updated_at is None
    created = await preferences.service.save(
        authority,
        expected_revision=missing.revision,
        content="# Preferences\n",
    )
    assert created.content == "# Preferences\n"
    assert created.updated_at is not None


@pytest.mark.asyncio
async def test_validation_and_unknown_revisions_fail_closed(preferences: _PreferenceFixture) -> None:
    authority = preferences.service.authority_for_principal(preferences.alice)
    current = await preferences.service.read(authority)

    with pytest.raises(WorkshopPreferenceValidationError):
        await preferences.service.save(
            authority,
            expected_revision=current.revision,
            content="x" * (MAX_PREFERENCE_BYTES + 1),
        )
    with pytest.raises(WorkshopPreferenceValidationError):
        await preferences.service.save(
            authority,
            expected_revision=current.revision,
            content="invalid\x00content",
        )
    with pytest.raises(WorkshopPreferenceRevisionNotFound):
        await preferences.service.restore(
            authority,
            target_revision="pref_v1_" + "1" * 32 + "_" + "2" * 32,
            expected_revision=current.revision,
        )
    assert (await preferences.service.read(authority)).revision == current.revision


@pytest.mark.asyncio
async def test_principal_authority_and_documents_are_isolated(preferences: _PreferenceFixture) -> None:
    alice = preferences.service.authority_for_principal(preferences.alice)
    bob = preferences.service.authority_for_principal(preferences.bob)
    alice_document = await preferences.service.read(alice)
    bob_document = await preferences.service.read(bob)
    assert "Alice" in alice_document.content
    assert "Bob" in bob_document.content

    with pytest.raises(WorkshopPreferenceAccessDenied):
        preferences.service.authority_for_principal(PrincipalId.new())


@pytest.mark.asyncio
async def test_untrusted_privileged_helper_is_never_executed(
    preferences: _PreferenceFixture,
    tmp_path: Path,
) -> None:
    helper = tmp_path / "helper"
    helper.write_text("#!/bin/sh\nexit 0\n")
    os.chmod(helper, 0o755)
    service = WorkshopPreferenceService(
        preferences.data_dir,
        preferences.registry,
        privileged_helper=helper,
    )
    authority = service.authority_for_principal(preferences.alice)
    with (
        patch.object(service._local, "snapshot", side_effect=PermissionError),
        patch("asyncio.create_subprocess_exec") as spawn,
        pytest.raises(WorkshopPreferenceStorageError),
    ):
        await service.read(authority)
    spawn.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("attack", ["symlink", "hardlink"])
async def test_unsafe_preference_files_are_refused(
    preferences: _PreferenceFixture,
    tmp_path: Path,
    attack: str,
) -> None:
    path = preferences.data_dir / "preferences" / str(preferences.alice) / "PREFERENCES.md"
    path.unlink()
    target = tmp_path / "outside-secret"
    target.write_text("outside")
    os.chmod(target, 0o600)
    if attack == "symlink":
        path.symlink_to(target)
    else:
        os.link(target, path)

    authority = preferences.service.authority_for_principal(preferences.alice)
    with pytest.raises(WorkshopPreferenceStorageError):
        await preferences.service.read(authority)
    assert target.read_text() == "outside"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attack",
    ["data_writable", "root_symlink", "principal_symlink", "writable_root"],
)
async def test_unsafe_preference_directories_are_refused(
    preferences: _PreferenceFixture,
    tmp_path: Path,
    attack: str,
) -> None:
    preferences_root = preferences.data_dir / "preferences"
    principal_directory = preferences_root / str(preferences.alice)
    if attack == "data_writable":
        os.chmod(preferences.data_dir, 0o733)
    elif attack == "root_symlink":
        moved_root = tmp_path / "moved-preferences"
        preferences_root.rename(moved_root)
        preferences_root.symlink_to(moved_root, target_is_directory=True)
    elif attack == "principal_symlink":
        moved_principal = tmp_path / "moved-principal"
        principal_directory.rename(moved_principal)
        principal_directory.symlink_to(moved_principal, target_is_directory=True)
    else:
        os.chmod(preferences_root, 0o733)

    authority = preferences.service.authority_for_principal(preferences.alice)
    with pytest.raises(WorkshopPreferenceStorageError):
        await preferences.service.read(authority)


@pytest.mark.asyncio
async def test_history_is_private_bounded_and_tamper_checked(preferences: _PreferenceFixture) -> None:
    authority = preferences.service.authority_for_principal(preferences.alice)
    current = await preferences.service.read(authority)
    for index in range(MAX_PREFERENCE_REVISIONS + 3):
        current = await preferences.service.save(
            authority,
            expected_revision=current.revision,
            content=f"Preference revision {index}.\n",
        )

    history = await preferences.service.history(authority)
    assert len(history.revisions) == MAX_PREFERENCE_REVISIONS
    history_dir = preferences.data_dir / "preference-revisions" / str(preferences.alice)
    assert stat.S_IMODE(history_dir.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in history_dir.iterdir())

    newest = history_dir / history.revisions[0].revision
    newest.write_text("tampered")
    with pytest.raises(WorkshopPreferenceStorageError):
        await preferences.service.history(authority)


@pytest.mark.asyncio
async def test_saved_preferences_are_the_exact_next_session_context(
    preferences: _PreferenceFixture,
    tmp_path: Path,
) -> None:
    authority = preferences.service.authority_for_principal(preferences.alice)
    current = await preferences.service.read(authority)
    saved = await preferences.service.save(
        authority,
        expected_revision=current.revision,
        content="Use Celsius.\nAvoid decorative headings.\n",
    )
    configure_principal_storage_namespaces(cast(PrincipalStorageNamespaceResolver, preferences.registry))
    try:
        with patch("kai.backend.get_recent_history", return_value=""):
            context = build_session_context(
                workspace=tmp_path,
                home_workspace=tmp_path,
                api=ApiContext(webhook_port=8080, webhook_secret=""),
                workspace_config=None,
                chat_id=101,
                data_dir=preferences.data_dir,
                memory_enabled=True,
            )
    finally:
        configure_principal_storage_namespaces(None)
    assert saved.content.strip() in context


def test_fixed_helper_protocol_returns_no_paths(preferences: _PreferenceFixture) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kai.workshop.preferences",
            "--helper",
            str(preferences.data_dir),
            "snapshot",
            str(preferences.alice),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["content"].startswith("# Preferences")
    assert str(preferences.data_dir) not in result.stdout


@pytest.mark.parametrize("principal", ["101", "../preferences", "prn_not_hex"])
def test_fixed_helper_rejects_noncanonical_principal_names(
    preferences: _PreferenceFixture,
    principal: str,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kai.workshop.preferences",
            "--helper",
            str(preferences.data_dir),
            "snapshot",
            principal,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout) == {"error": "access_denied"}
    assert str(preferences.data_dir) not in result.stdout


class _BearerAuthenticator:
    def __init__(self, principals: dict[str, PrincipalId]) -> None:
        self._principals = principals

    async def authenticate(self, request: web.Request) -> PrincipalId | None:
        header = request.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        return self._principals.get(token) if scheme == "Bearer" else None

    async def authenticate_token(self, token: str) -> PrincipalId | None:
        return self._principals.get(token)


async def _preference_client(
    preferences: _PreferenceFixture,
) -> tuple[TestClient, WorkshopEventStore]:
    store = await WorkshopEventStore.open(preferences.data_dir / "kai.db")
    app = web.Application()
    register_workshop_read_routes(
        app,
        store=store,
        authenticator=_BearerAuthenticator({"alice-token": preferences.alice, "bob-token": preferences.bob}),
        request_lock=__import__("asyncio").Lock(),
        preference_documents=preferences.service,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, store


@pytest.mark.asyncio
async def test_preference_api_is_authenticated_revision_checked_and_principal_scoped(
    preferences: _PreferenceFixture,
) -> None:
    client, store = await _preference_client(preferences)
    try:
        unauthorized = await client.get("/v1/preferences")
        alice_response = await client.get(
            "/v1/preferences",
            headers={"Authorization": "Bearer alice-token"},
        )
        bob_response = await client.get(
            "/v1/preferences",
            headers={"Authorization": "Bearer bob-token"},
        )
        assert unauthorized.status == 401
        assert unauthorized.headers["WWW-Authenticate"] == "Bearer"
        assert (await alice_response.json())["document"]["content"].startswith("# Preferences")
        assert (await bob_response.json())["document"]["content"] == "Bob private.\n"

        alice_payload = await alice_response.json()
        updated = await client.put(
            "/v1/preferences",
            headers={"Authorization": "Bearer alice-token"},
            json={
                "content": "Alice API update.\n",
                "revision": alice_payload["document"]["revision"],
            },
        )
        assert updated.status == 200
        updated_payload = await updated.json()
        stale = await client.put(
            "/v1/preferences",
            headers={"Authorization": "Bearer alice-token"},
            json={
                "content": "Stale.\n",
                "revision": alice_payload["document"]["revision"],
            },
        )
        assert stale.status == 409
        assert (await stale.json())["error"]["current_revision"] == updated_payload["document"]["revision"]

        history = await client.get(
            "/v1/preferences/revisions",
            headers={"Authorization": "Bearer alice-token"},
        )
        history_payload = await history.json()
        assert history.status == 200
        assert len(history_payload["revisions"]) == 1
        restored = await client.post(
            f"/v1/preferences/revisions/{history_payload['revisions'][0]['revision']}/restore",
            headers={"Authorization": "Bearer alice-token"},
            json={"revision": updated_payload["document"]["revision"]},
        )
        assert restored.status == 200
        assert (await restored.json())["document"]["content"].startswith("# Preferences")
    finally:
        await client.close()
        await store.close()


@pytest.mark.asyncio
async def test_preference_api_rejects_untyped_requests_without_mutation(
    preferences: _PreferenceFixture,
) -> None:
    client, store = await _preference_client(preferences)
    try:
        before = await client.get(
            "/v1/preferences",
            headers={"Authorization": "Bearer alice-token"},
        )
        revision = (await before.json())["document"]["revision"]
        foreign_selector = await client.put(
            "/v1/preferences",
            headers={"Authorization": "Bearer alice-token"},
            json={
                "content": "Attack.\n",
                "revision": revision,
                "principal_id": str(preferences.bob),
            },
        )
        wrong_content_type = await client.put(
            "/v1/preferences",
            headers={
                "Authorization": "Bearer alice-token",
                "Content-Type": "text/plain",
            },
            data="Attack",
        )
        oversized = await client.put(
            "/v1/preferences",
            headers={"Authorization": "Bearer alice-token"},
            json={
                "content": "x" * (MAX_PREFERENCE_BYTES + 1),
                "revision": revision,
            },
        )
        assert foreign_selector.status == 400
        assert wrong_content_type.status == 400
        assert oversized.status == 400
        after = await client.get(
            "/v1/preferences",
            headers={"Authorization": "Bearer alice-token"},
        )
        assert (await after.json())["document"]["revision"] == revision
    finally:
        await client.close()
        await store.close()
