from unittest.mock import MagicMock

import pytest

from kai import named_access


def _completed() -> MagicMock:
    result = MagicMock()
    result.returncode = 0
    result.stderr = ""
    return result


def test_macos_grants_directory_traversal_and_listing(monkeypatch, tmp_path):
    run = MagicMock(return_value=_completed())
    monkeypatch.setattr(named_access.sys, "platform", "darwin")
    monkeypatch.setattr(named_access.subprocess, "run", run)

    path = tmp_path / "history"
    named_access.grant_named_read_access(path, "daniel", directory=True)

    run.assert_called_once_with(
        [
            "/bin/chmod",
            "+a",
            "user:daniel allow list,search,readattr,readextattr,readsecurity",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_macos_replacement_clears_stale_acl_before_grant(monkeypatch, tmp_path):
    run = MagicMock(return_value=_completed())
    monkeypatch.setattr(named_access.sys, "platform", "darwin")
    monkeypatch.setattr(named_access.subprocess, "run", run)
    path = tmp_path / "messages.jsonl"

    named_access.replace_named_read_access(path, "daniel", directory=False)

    assert [call.args[0] for call in run.call_args_list] == [
        ["/bin/chmod", "-N", str(path)],
        [
            "/bin/chmod",
            "+a",
            "user:daniel allow read,readattr,readextattr,readsecurity",
            str(path),
        ],
    ]


def test_linux_replacement_uses_setfacl(monkeypatch, tmp_path):
    run = MagicMock(return_value=_completed())
    monkeypatch.setattr(named_access.sys, "platform", "linux")
    monkeypatch.setattr(named_access.shutil, "which", lambda _name: "/usr/bin/setfacl")
    monkeypatch.setattr(named_access.subprocess, "run", run)
    path = tmp_path / "messages.jsonl"

    named_access.replace_named_read_access(path, "alice", directory=False)

    assert [call.args[0] for call in run.call_args_list] == [
        ["/usr/bin/setfacl", "-b", str(path)],
        ["/usr/bin/setfacl", "-m", "u:alice:r--", str(path)],
    ]


def test_linux_directory_replacement_removes_default_acl(monkeypatch, tmp_path):
    run = MagicMock(return_value=_completed())
    monkeypatch.setattr(named_access.sys, "platform", "linux")
    monkeypatch.setattr(named_access.shutil, "which", lambda _name: "/usr/bin/setfacl")
    monkeypatch.setattr(named_access.subprocess, "run", run)
    path = tmp_path / "history"

    named_access.replace_named_read_access(path, "alice", directory=True)

    assert [call.args[0] for call in run.call_args_list] == [
        ["/usr/bin/setfacl", "-b", "-k", str(path)],
        ["/usr/bin/setfacl", "-m", "u:alice:r-x", str(path)],
    ]


def test_replacement_without_reader_only_clears_acl(monkeypatch, tmp_path):
    run = MagicMock(return_value=_completed())
    monkeypatch.setattr(named_access.sys, "platform", "darwin")
    monkeypatch.setattr(named_access.subprocess, "run", run)
    path = tmp_path / "group-history"

    named_access.replace_named_read_access(path, None, directory=True)

    run.assert_called_once()
    assert run.call_args.args[0] == ["/bin/chmod", "-N", str(path)]


def test_missing_linux_setfacl_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(named_access.sys, "platform", "linux")
    monkeypatch.setattr(named_access.shutil, "which", lambda _name: None)

    with pytest.raises(OSError, match="setfacl is required"):
        named_access.replace_named_read_access(tmp_path / "history", "alice", directory=True)


def test_acl_command_failure_is_reported(monkeypatch, tmp_path):
    result = _completed()
    result.returncode = 1
    result.stderr = "permission denied"
    monkeypatch.setattr(named_access.sys, "platform", "darwin")
    monkeypatch.setattr(named_access.subprocess, "run", MagicMock(return_value=result))

    with pytest.raises(OSError, match="permission denied"):
        named_access.grant_named_read_access(tmp_path / "history", "alice", directory=True)
