"""Neutral filesystem workspaces for cross-owner private agent lanes."""

from __future__ import annotations

import json
import os
import pwd
import sqlite3
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from kai.workshop.domain import AgentId, ChannelId, PrincipalId, RuntimeProfileId
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileRegistry

SPONSORED_WORKSPACE_PROVISIONER = Path("/etc/kai/provision-sponsored-workspace")
RUNTIME_PROFILES_POLICY = Path("/etc/kai/runtime-profiles.yaml")


class SponsoredWorkspaceError(RuntimeError):
    """A neutral sponsored workspace could not be resolved safely."""


@dataclass(frozen=True, slots=True)
class SponsoredWorkspaceResult:
    """Canonical path and ownership returned by the bounded provisioner."""

    path: str
    created: bool


def _mkdir_component(parent_fd: int, name: str, *, mode: int, uid: int, gid: int) -> tuple[int, bool]:
    created = False
    try:
        os.mkdir(name, mode=mode, dir_fd=parent_fd)
        created = True
    except FileExistsError:
        pass
    try:
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    except OSError as exc:
        raise SponsoredWorkspaceError("Sponsored workspace contains an unsafe path component") from exc
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode):
        os.close(fd)
        raise SponsoredWorkspaceError("Sponsored workspace contains a non-directory component")
    if created:
        os.fchown(fd, uid, gid)
        os.fchmod(fd, mode)
    elif info.st_uid != uid or info.st_gid != gid or stat.S_IMODE(info.st_mode) != mode:
        os.close(fd)
        raise SponsoredWorkspaceError("Sponsored workspace ownership or mode is invalid")
    return fd, created


def _canonical_lane_exists(
    database: Path,
    *,
    requester_principal_id: PrincipalId,
    channel_id: ChannelId,
    agent_id: AgentId,
    runtime_profile_id: RuntimeProfileId,
) -> bool:
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise SponsoredWorkspaceError("Canonical Workshop storage is unavailable") from exc
    try:
        row = connection.execute(
            "SELECT 1 FROM channels c "
            "JOIN channel_memberships cm ON cm.channel_id = c.id "
            "AND cm.principal_id = ? AND cm.role = 'owner' "
            "JOIN channel_agents ca ON ca.channel_id = c.id AND ca.agent_id = ? "
            "AND ca.detached_at IS NULL AND ca.sponsored_runtime_profile_id = ? "
            "JOIN agent_definitions d ON d.agent_id = ca.agent_id "
            "AND d.lifecycle_state = 'active' AND d.owner_runtime_profile_id = ? "
            "WHERE c.id = ? AND c.kind = 'direct' AND c.archived_at IS NULL "
            "AND d.owner_principal_id != ? LIMIT 1",
            (
                str(requester_principal_id),
                str(agent_id),
                str(runtime_profile_id),
                str(runtime_profile_id),
                str(channel_id),
                str(requester_principal_id),
            ),
        ).fetchone()
    except sqlite3.Error as exc:
        raise SponsoredWorkspaceError("Canonical sponsored lane could not be verified") from exc
    finally:
        connection.close()
    return row is not None


def provision_sponsored_workspace(
    data_dir: Path,
    *,
    requester_principal_id: PrincipalId,
    channel_id: ChannelId,
    agent_id: AgentId,
    runtime_profile_id: RuntimeProfileId,
    os_user: str | None,
) -> SponsoredWorkspaceResult:
    """Create one empty lane workspace without exposing either principal's home."""
    if not data_dir.is_absolute() or data_dir.is_symlink() or not data_dir.is_dir():
        raise SponsoredWorkspaceError("Kai data storage is unavailable or invalid")
    if os_user is None:
        uid, gid = os.geteuid(), os.getegid()
    else:
        try:
            account = pwd.getpwnam(os_user)
        except KeyError as exc:
            raise SponsoredWorkspaceError("The sponsored runtime OS identity is unavailable") from exc
        uid, gid = account.pw_uid, account.pw_gid
    if os.geteuid() != 0 and uid != os.geteuid():
        raise SponsoredWorkspaceError("Protected sponsored workspace provisioning requires root authority")

    root = data_dir / "sponsored-workspaces"
    if root.is_symlink():
        raise SponsoredWorkspaceError("Sponsored workspace root must not be a symbolic link")
    root_created = False
    try:
        root.mkdir(mode=0o711)
        root_created = True
    except FileExistsError:
        pass
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    root_info = os.fstat(root_fd)
    expected_root_uid = 0 if os.geteuid() == 0 else os.geteuid()
    expected_root_gid = 0 if os.geteuid() == 0 else os.getegid()
    if root_created:
        os.fchown(root_fd, expected_root_uid, expected_root_gid)
        os.fchmod(root_fd, 0o711)
        root_info = os.fstat(root_fd)
    if (
        root_info.st_uid != expected_root_uid
        or root_info.st_gid != expected_root_gid
        or stat.S_IMODE(root_info.st_mode) != 0o711
    ):
        os.close(root_fd)
        raise SponsoredWorkspaceError("Sponsored workspace root ownership or mode is invalid")
    descriptors = [root_fd]
    created_leaf = False
    parent_uid = expected_root_uid
    parent_gid = expected_root_gid
    try:
        current_fd = root_fd
        components = (
            str(runtime_profile_id),
            str(requester_principal_id),
            str(channel_id),
            str(agent_id),
        )
        for index, component in enumerate(components):
            leaf = index == len(components) - 1
            next_fd, created = _mkdir_component(
                current_fd,
                component,
                mode=0o700 if leaf else 0o711,
                uid=uid if leaf else parent_uid,
                gid=gid if leaf else parent_gid,
            )
            descriptors.append(next_fd)
            current_fd = next_fd
            if leaf:
                created_leaf = created
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    path = root.joinpath(*components)
    return SponsoredWorkspaceResult(path=str(path), created=created_leaf)


def provision_via_helper(
    *,
    requester_principal_id: PrincipalId,
    channel_id: ChannelId,
    agent_id: AgentId,
    runtime_profile_id: RuntimeProfileId,
) -> SponsoredWorkspaceResult:
    """Invoke the fixed helper without accepting a path or OS username."""
    try:
        completed = subprocess.run(
            [
                "sudo",
                "-n",
                str(SPONSORED_WORKSPACE_PROVISIONER),
                str(requester_principal_id),
                str(channel_id),
                str(agent_id),
                str(runtime_profile_id),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SponsoredWorkspaceError("Protected sponsored workspace provisioning failed") from exc
    try:
        payload = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise SponsoredWorkspaceError("Protected sponsored workspace provisioning failed") from exc
    if completed.returncode != 0 or not isinstance(payload, dict) or payload.get("ok") is not True:
        message = payload.get("error") if isinstance(payload, dict) else None
        raise SponsoredWorkspaceError(
            str(message)
            if isinstance(message, str) and message
            else "Protected sponsored workspace provisioning failed"
        )
    result = payload.get("result")
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("path"), str)
        or not isinstance(result.get("created"), bool)
    ):
        raise SponsoredWorkspaceError("Protected sponsored workspace provisioning returned an invalid result")
    return SponsoredWorkspaceResult(path=result["path"], created=result["created"])


def _helper_main(argv: list[str]) -> int:
    if len(argv) != 5 or os.geteuid() != 0:
        print(json.dumps({"ok": False, "error": "Invalid sponsored workspace request"}))
        return 64
    try:
        data_dir = Path(argv[0])
        requester_principal_id = PrincipalId(argv[1])
        channel_id = ChannelId(argv[2])
        agent_id = AgentId(argv[3])
        runtime_profile_id = RuntimeProfileId(argv[4])
        profile = WorkshopRuntimeProfileRegistry.from_yaml(RUNTIME_PROFILES_POLICY.read_text(encoding="utf-8")).resolve(
            runtime_profile_id
        )
        if not _canonical_lane_exists(
            data_dir / "kai.db",
            requester_principal_id=requester_principal_id,
            channel_id=channel_id,
            agent_id=agent_id,
            runtime_profile_id=runtime_profile_id,
        ):
            raise SponsoredWorkspaceError("Canonical cross-owner agent lane is unavailable")
        result = provision_sponsored_workspace(
            data_dir,
            requester_principal_id=requester_principal_id,
            channel_id=channel_id,
            agent_id=agent_id,
            runtime_profile_id=runtime_profile_id,
            os_user=profile.os_user,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    print(json.dumps({"ok": True, "result": asdict(result)}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(_helper_main(sys.argv[1:]))
