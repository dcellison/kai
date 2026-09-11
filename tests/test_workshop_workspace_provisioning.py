"""Bounded self-service workspace filesystem provisioning tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kai.workshop.domain import RuntimeProfileId
from kai.workshop.workspace_provisioning import (
    WorkspaceProvisioningError,
    WorkspaceProvisioningResult,
    _helper_main,
    provision_via_helper,
    provision_workspace,
)


def test_local_provisioning_creates_direct_child_and_git_repository(tmp_path: Path) -> None:
    base = tmp_path / "configured-projects"
    base.mkdir()

    created = provision_workspace(base, "research")
    retried = provision_workspace(base, "research")

    target = base / "research"
    assert created == WorkspaceProvisioningResult(str(target), True, True)
    assert retried == WorkspaceProvisioningResult(str(target), False, True)
    assert target.stat().st_mode & 0o777 == 0o755
    assert (target / ".git").is_dir()


@pytest.mark.parametrize("name", ["", ".", "..", "../escape", "nested/name", "nested\\name", "bad\x00name"])
def test_local_provisioning_rejects_unsafe_names(tmp_path: Path, name: str) -> None:
    with pytest.raises(WorkspaceProvisioningError, match="Invalid workspace name"):
        provision_workspace(tmp_path, name)


def test_local_provisioning_rejects_symlinked_base_and_target(tmp_path: Path) -> None:
    real_base = tmp_path / "real"
    real_base.mkdir()
    linked_base = tmp_path / "linked"
    linked_base.symlink_to(real_base, target_is_directory=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (real_base / "linked-target").symlink_to(outside, target_is_directory=True)

    with pytest.raises(WorkspaceProvisioningError, match="workspace base"):
        provision_workspace(linked_base, "project")
    with pytest.raises(WorkspaceProvisioningError, match="symbolic link"):
        provision_workspace(real_base, "linked-target")


def test_helper_invocation_accepts_only_bounded_json_result(monkeypatch) -> None:
    profile_id = RuntimeProfileId("rtp_" + "1" * 32)
    monkeypatch.setattr(
        "kai.workshop.workspace_provisioning.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "result": {
                        "path": "/configured/projects/research",
                        "directory_created": True,
                        "git_ready": True,
                    },
                }
            ),
        ),
    )

    assert provision_via_helper(profile_id, "research") == WorkspaceProvisioningResult(
        "/configured/projects/research",
        True,
        True,
    )


def test_helper_invocation_surfaces_bounded_error(monkeypatch) -> None:
    profile_id = RuntimeProfileId("rtp_" + "1" * 32)
    monkeypatch.setattr(
        "kai.workshop.workspace_provisioning.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout=json.dumps({"ok": False, "error": "No workspace base is configured for this runtime"}),
        ),
    )

    with pytest.raises(WorkspaceProvisioningError, match="No workspace base"):
        provision_via_helper(profile_id, "research")


def test_root_helper_derives_base_and_owner_from_runtime_profile(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    profile_id = RuntimeProfileId("rtp_" + "1" * 32)
    configured_base = tmp_path / "configured-projects"
    configured_base.mkdir()
    policy = tmp_path / "runtime-profiles.yaml"
    policy.write_text("version: 2\nruntime_profiles: {}\n")
    profile = SimpleNamespace(workspace_base=configured_base, os_user="daniel")
    registry = SimpleNamespace(resolve=lambda selected: profile if selected == profile_id else None)

    def provisioner(base: Path, name: str, *, os_user: str) -> WorkspaceProvisioningResult:
        assert base == configured_base
        assert name == "research"
        assert os_user == "daniel"
        return WorkspaceProvisioningResult(str(base / name), True, True)

    monkeypatch.setattr("kai.workshop.workspace_provisioning.RUNTIME_PROFILES_POLICY", policy)
    monkeypatch.setattr("kai.workshop.workspace_provisioning.os.geteuid", lambda: 0)
    monkeypatch.setattr(
        "kai.workshop.workspace_provisioning.WorkshopRuntimeProfileRegistry.from_yaml",
        lambda _content: registry,
    )
    monkeypatch.setattr("kai.workshop.workspace_provisioning.provision_workspace", provisioner)

    assert _helper_main([str(profile_id), "research"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "ok": True,
        "result": {
            "path": str(configured_base / "research"),
            "directory_created": True,
            "git_ready": True,
        },
    }
