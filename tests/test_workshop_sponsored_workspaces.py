"""Security contract for neutral cross-owner agent workspaces."""

from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kai.workshop.domain import AgentId, ChannelId, PrincipalId, RuntimeProfileId
from kai.workshop.sponsored_workspaces import (
    SponsoredWorkspaceError,
    _canonical_lane_exists,
    provision_sponsored_workspace,
    provision_via_helper,
)


def _principal(character: str) -> PrincipalId:
    return PrincipalId("prn_" + character * 32)


def _channel(character: str) -> ChannelId:
    return ChannelId("chn_" + character * 32)


def _agent(character: str) -> AgentId:
    return AgentId("agt_" + character * 32)


def _profile(character: str) -> RuntimeProfileId:
    return RuntimeProfileId("rtp_" + character * 32)


def test_provisions_one_private_opaque_lane_without_principal_homes(tmp_path: Path) -> None:
    result = provision_sponsored_workspace(
        tmp_path,
        requester_principal_id=_principal("1"),
        channel_id=_channel("2"),
        agent_id=_agent("3"),
        runtime_profile_id=_profile("4"),
        os_user=None,
    )
    path = Path(result.path)

    assert result.created is True
    assert path == (
        tmp_path
        / "sponsored-workspaces"
        / str(_profile("4"))
        / str(_principal("1"))
        / str(_channel("2"))
        / str(_agent("3"))
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o700
    assert list(path.iterdir()) == []
    assert "home" not in path.parts

    replay = provision_sponsored_workspace(
        tmp_path,
        requester_principal_id=_principal("1"),
        channel_id=_channel("2"),
        agent_id=_agent("3"),
        runtime_profile_id=_profile("4"),
        os_user=None,
    )
    assert replay.path == result.path
    assert replay.created is False


def test_rejects_symlinked_lane_component(tmp_path: Path) -> None:
    provision_sponsored_workspace(
        tmp_path,
        requester_principal_id=_principal("1"),
        channel_id=_channel("2"),
        agent_id=_agent("3"),
        runtime_profile_id=_profile("5"),
        os_user=None,
    )
    root = tmp_path / "sponsored-workspaces"
    target = tmp_path / "elsewhere"
    target.mkdir()
    (root / str(_profile("4"))).symlink_to(target, target_is_directory=True)

    with pytest.raises(SponsoredWorkspaceError, match="unsafe path component"):
        provision_sponsored_workspace(
            tmp_path,
            requester_principal_id=_principal("1"),
            channel_id=_channel("2"),
            agent_id=_agent("3"),
            runtime_profile_id=_profile("4"),
            os_user=None,
        )


def test_rejects_tampered_leaf_mode(tmp_path: Path) -> None:
    request = dict(
        requester_principal_id=_principal("1"),
        channel_id=_channel("2"),
        agent_id=_agent("3"),
        runtime_profile_id=_profile("4"),
        os_user=None,
    )
    result = provision_sponsored_workspace(tmp_path, **request)
    os.chmod(result.path, 0o755)

    with pytest.raises(SponsoredWorkspaceError, match="ownership or mode"):
        provision_sponsored_workspace(tmp_path, **request)


def test_helper_call_accepts_only_typed_result() -> None:
    completed = MagicMock(
        returncode=0,
        stdout='{"ok":true,"result":{"path":"/var/lib/kai/sponsored-workspaces/lane","created":true}}',
    )
    with patch("kai.workshop.sponsored_workspaces.subprocess.run", return_value=completed) as run:
        result = provision_via_helper(
            requester_principal_id=_principal("1"),
            channel_id=_channel("2"),
            agent_id=_agent("3"),
            runtime_profile_id=_profile("4"),
        )

    assert result.created is True
    argv = run.call_args.args[0]
    assert argv[:3] == ["sudo", "-n", "/etc/kai/provision-sponsored-workspace"]
    assert len(argv) == 7
    assert all("/" not in value for value in argv[3:])
    assert run.call_args.kwargs["timeout"] == 10


def test_canonical_lane_verification_rejects_owner_and_detached_lanes(tmp_path: Path) -> None:
    database = tmp_path / "kai.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        "CREATE TABLE channels (id TEXT, kind TEXT, archived_at TEXT);"
        "CREATE TABLE channel_memberships (channel_id TEXT, principal_id TEXT, role TEXT);"
        "CREATE TABLE channel_agents (channel_id TEXT, agent_id TEXT, detached_at TEXT, "
        "sponsored_runtime_profile_id TEXT);"
        "CREATE TABLE agent_definitions (agent_id TEXT, lifecycle_state TEXT, "
        "owner_runtime_profile_id TEXT, owner_principal_id TEXT);"
    )
    requester = _principal("1")
    owner = _principal("9")
    channel = _channel("2")
    agent = _agent("3")
    profile = _profile("4")
    connection.execute("INSERT INTO channels VALUES (?, 'direct', NULL)", (str(channel),))
    connection.execute(
        "INSERT INTO channel_memberships VALUES (?, ?, 'owner')",
        (str(channel), str(requester)),
    )
    connection.execute(
        "INSERT INTO channel_agents VALUES (?, ?, NULL, ?)",
        (str(channel), str(agent), str(profile)),
    )
    connection.execute(
        "INSERT INTO agent_definitions VALUES (?, 'active', ?, ?)",
        (str(agent), str(profile), str(owner)),
    )
    connection.commit()

    assert _canonical_lane_exists(
        database,
        requester_principal_id=requester,
        channel_id=channel,
        agent_id=agent,
        runtime_profile_id=profile,
    )
    connection.execute(
        "UPDATE agent_definitions SET owner_principal_id = ?",
        (str(requester),),
    )
    connection.commit()
    assert not _canonical_lane_exists(
        database,
        requester_principal_id=requester,
        channel_id=channel,
        agent_id=agent,
        runtime_profile_id=profile,
    )
    connection.close()
