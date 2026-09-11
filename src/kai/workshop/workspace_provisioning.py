"""Bounded filesystem provisioning for self-service workspaces."""

from __future__ import annotations

import json
import os
import pwd
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from kai.workshop.domain import RuntimeProfileId
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileRegistry

WORKSPACE_PROVISIONER = Path("/etc/kai/provision-principal-workspace")
RUNTIME_PROFILES_POLICY = Path("/etc/kai/runtime-profiles.yaml")
GIT_COMMAND = Path("/usr/bin/git")


class WorkspaceProvisioningError(RuntimeError):
    """A bounded workspace filesystem operation could not be completed."""


@dataclass(frozen=True, slots=True)
class WorkspaceProvisioningResult:
    """Filesystem facts returned by the local or privileged provisioner."""

    path: str
    directory_created: bool
    git_ready: bool


def _validate_flat_name(name: str) -> str:
    if (
        not name
        or name != name.strip()
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise WorkspaceProvisioningError("Invalid workspace name")
    return name


def _target_identity(os_user: str | None) -> tuple[int, int, tuple[int, ...], str] | None:
    if os_user is None:
        return None
    try:
        account = pwd.getpwnam(os_user)
    except KeyError as exc:
        raise WorkspaceProvisioningError("The runtime OS identity is unavailable") from exc
    groups = tuple(os.getgrouplist(account.pw_name, account.pw_gid))
    return account.pw_uid, account.pw_gid, groups, account.pw_dir


def provision_workspace(
    workspace_base: Path,
    name: str,
    *,
    os_user: str | None = None,
    git_command: Path = GIT_COMMAND,
) -> WorkspaceProvisioningResult:
    """Create one direct child of an operator-authorized workspace base."""
    checked_name = _validate_flat_name(name)
    if not workspace_base.is_absolute() or workspace_base.is_symlink() or not workspace_base.is_dir():
        raise WorkspaceProvisioningError("The configured workspace base is unavailable or invalid")
    try:
        base = workspace_base.resolve(strict=True)
        base_fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise WorkspaceProvisioningError("The configured workspace base is unavailable or invalid") from exc

    identity = _target_identity(os_user)
    created = False
    target_fd = -1
    try:
        try:
            os.mkdir(checked_name, mode=0o755, dir_fd=base_fd)
            created = True
        except FileExistsError:
            pass
        try:
            target_fd = os.open(
                checked_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=base_fd,
            )
        except OSError as exc:
            raise WorkspaceProvisioningError(
                "A non-directory or symbolic link already uses this workspace name"
            ) from exc
        info = os.fstat(target_fd)
        if not stat.S_ISDIR(info.st_mode):
            raise WorkspaceProvisioningError("A non-directory entry already uses this workspace name")
        if identity is not None:
            uid, gid, _groups, _home = identity
            if created:
                if os.geteuid() != 0 and info.st_uid != uid:
                    raise WorkspaceProvisioningError("Protected workspace provisioning requires root authority")
                if os.geteuid() == 0:
                    os.fchown(target_fd, uid, gid)
            elif info.st_uid != uid:
                raise WorkspaceProvisioningError("An existing workspace with this name has a different owner")
        if created:
            os.fchmod(target_fd, 0o755)
    finally:
        if target_fd >= 0:
            os.close(target_fd)
        os.close(base_fd)

    target = base / checked_name
    git_ready = (target / ".git").is_dir()
    if not git_ready:
        command: dict[str, object] = {
            "cwd": str(target),
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "check": False,
            "env": {"LANG": "C.UTF-8", "PATH": "/usr/bin:/bin"},
        }
        if identity is not None:
            uid, gid, groups, home = identity
            command.update(user=uid, group=gid, extra_groups=groups)
            environment = command["env"]
            assert isinstance(environment, dict)
            environment["HOME"] = home
        try:
            completed = subprocess.run([str(git_command), "init"], **command)  # type: ignore[arg-type]
        except OSError:
            git_ready = False
        else:
            git_ready = completed.returncode == 0 and (target / ".git").is_dir()
    return WorkspaceProvisioningResult(str(target), created, git_ready)


def provision_via_helper(profile_id: RuntimeProfileId, name: str) -> WorkspaceProvisioningResult:
    """Invoke the fixed root helper without giving the daemon a path argument."""
    completed = subprocess.run(
        ["sudo", "-n", str(WORKSPACE_PROVISIONER), str(profile_id), name],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise WorkspaceProvisioningError("Protected workspace provisioning failed") from exc
    if completed.returncode != 0 or not isinstance(payload, dict) or payload.get("ok") is not True:
        message = payload.get("error") if isinstance(payload, dict) else None
        raise WorkspaceProvisioningError(
            str(message) if isinstance(message, str) and message else "Protected workspace provisioning failed"
        )
    result = payload.get("result")
    if not isinstance(result, dict):
        raise WorkspaceProvisioningError("Protected workspace provisioning returned an invalid result")
    path = result.get("path")
    directory_created = result.get("directory_created")
    git_ready = result.get("git_ready")
    if not isinstance(path, str) or not isinstance(directory_created, bool) or not isinstance(git_ready, bool):
        raise WorkspaceProvisioningError("Protected workspace provisioning returned an invalid result")
    return WorkspaceProvisioningResult(path, directory_created, git_ready)


def _helper_main(argv: list[str]) -> int:
    if len(argv) != 2 or os.geteuid() != 0:
        print(json.dumps({"ok": False, "error": "Invalid workspace provisioning request"}))
        return 64
    try:
        profile_id = RuntimeProfileId(argv[0])
        content = RUNTIME_PROFILES_POLICY.read_text(encoding="utf-8")
        profile = WorkshopRuntimeProfileRegistry.from_yaml(content).resolve(profile_id)
        if profile.workspace_base is None:
            raise WorkspaceProvisioningError("No workspace base is configured for this runtime")
        if profile.os_user is None:
            raise WorkspaceProvisioningError("The protected runtime has no OS identity")
        result = provision_workspace(
            profile.workspace_base,
            argv[1],
            os_user=profile.os_user,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    print(json.dumps({"ok": True, "result": asdict(result)}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(_helper_main(sys.argv[1:]))
