from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kai.protected_config import (
    ProtectedConfigError,
    max_mode_for_protected_file,
    validate_protected_file_metadata,
)


def test_missing_ok_returns_false(tmp_path):
    assert validate_protected_file_metadata(tmp_path / "missing", missing_ok=True) is False


def test_default_max_mode_is_0600():
    assert max_mode_for_protected_file("/etc/kai/env") == 0o600


def test_backends_yaml_max_mode_is_0644():
    assert max_mode_for_protected_file("/etc/kai/backends.yaml") == 0o644


def test_rejects_permissions_more_permissive_than_max(tmp_path):
    path = tmp_path / "env"
    path.write_text("KEY=value\n")
    path.chmod(0o644)

    with pytest.raises(ProtectedConfigError, match="unsafe permissions"):
        validate_protected_file_metadata(path, max_mode=0o600, require_root_owner=False)


def test_rejects_non_regular_file(tmp_path):
    path = tmp_path / "env"
    path.mkdir()
    path.chmod(0o600)

    with pytest.raises(ProtectedConfigError, match="regular file"):
        validate_protected_file_metadata(path, max_mode=0o600, require_root_owner=False)


def test_accepts_subset_of_max_permissions(tmp_path):
    path = tmp_path / "env"
    path.write_text("KEY=value\n")
    path.chmod(0o400)

    assert validate_protected_file_metadata(path, max_mode=0o600, require_root_owner=False) is True


def test_rejects_non_root_owner_when_required(tmp_path, monkeypatch):
    path = tmp_path / "env"
    path.write_text("KEY=value\n")
    path.chmod(0o600)
    real_stat = Path.stat

    def fake_stat(self, *args, **kwargs):
        result = real_stat(self, *args, **kwargs)
        if self == path:
            return SimpleNamespace(st_mode=result.st_mode, st_uid=12345)
        return result

    monkeypatch.setattr("kai.protected_config.Path.stat", fake_stat)

    with pytest.raises(ProtectedConfigError, match="root ownership"):
        validate_protected_file_metadata(path, require_root_owner=True)
