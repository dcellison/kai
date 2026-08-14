from pathlib import Path

from kai.subprocess_identity import subprocess_spawn_cwd, wrap_command_for_target_user


def test_cross_user_wrapper_enters_workspace_after_identity_change():
    workspace = Path("/var/lib/kai/home/prn_test")

    command = wrap_command_for_target_user(
        ["/usr/local/bin/codex", "app-server"],
        target_user="daniel",
        working_directory=workspace,
        preserve_env=("CODEX_HOME", "OPENAI_API_KEY"),
    )

    assert command == [
        "sudo",
        "-H",
        "-D",
        str(workspace),
        "-u",
        "daniel",
        "--preserve-env=CODEX_HOME,OPENAI_API_KEY",
        "--",
        "/usr/local/bin/codex",
        "app-server",
    ]


def test_cross_user_wrapper_omits_empty_preserve_option():
    command = wrap_command_for_target_user(
        ["agent"],
        target_user="alice",
        working_directory=Path("/private/workspace"),
    )

    assert command == [
        "sudo",
        "-H",
        "-D",
        "/private/workspace",
        "-u",
        "alice",
        "--",
        "agent",
    ]


def test_cross_user_spawn_does_not_pre_enter_private_workspace():
    workspace = Path("/var/lib/kai/home/prn_test")

    assert subprocess_spawn_cwd(workspace, target_user="daniel") is None


def test_direct_spawn_keeps_requested_workspace():
    workspace = Path("/workspace")

    assert subprocess_spawn_cwd(workspace, target_user=None) == str(workspace)
