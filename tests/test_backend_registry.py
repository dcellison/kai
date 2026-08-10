from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from kai.backend_registry import (
    BackendRegistryError,
    backend_registry_is_authoritative,
    render_backend_registry,
    resolve_backend_command,
)


def _exe(path):
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def test_registry_command_wins_over_legacy_env(tmp_path, monkeypatch):
    codex = _exe(tmp_path / "codex")
    registry = tmp_path / "backends.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "backends": {
                    "codex": {
                        "driver": "codex",
                        "runtime": "local_process",
                        "command": str(codex),
                    }
                },
            }
        )
    )
    monkeypatch.setenv("KAI_BACKENDS_YAML", str(registry))
    monkeypatch.setenv("CODEX_BIN", str(tmp_path / "stale-codex"))

    assert resolve_backend_command("codex") == str(codex)


def test_present_registry_rejects_unknown_backend(tmp_path, monkeypatch):
    registry = tmp_path / "backends.yaml"
    registry.write_text("version: 1\nbackends: {}\n")
    monkeypatch.setenv("KAI_BACKENDS_YAML", str(registry))

    with pytest.raises(BackendRegistryError, match="no entry"):
        resolve_backend_command("codex")


def test_registry_command_must_be_absolute(tmp_path, monkeypatch):
    registry = tmp_path / "backends.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "backends": {
                    "claude": {
                        "driver": "claude",
                        "runtime": "local_process",
                        "command": "claude",
                    }
                },
            }
        )
    )
    monkeypatch.setenv("KAI_BACKENDS_YAML", str(registry))

    with pytest.raises(BackendRegistryError, match="not absolute"):
        resolve_backend_command("claude")


@pytest.mark.parametrize("mode", [0o664, 0o646])
def test_registry_file_must_not_be_group_or_world_writable(tmp_path, monkeypatch, mode):
    """The installed backend registry is a machine-capability boundary.

    If the file is writable by group/other users, a non-admin account
    could swap backend commands and bypass the curated registry.
    """
    claude = _exe(tmp_path / "claude")
    registry = tmp_path / "backends.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "backends": {
                    "claude": {
                        "driver": "claude",
                        "runtime": "local_process",
                        "command": str(claude),
                    }
                },
            }
        )
    )
    registry.chmod(mode)
    monkeypatch.setenv("KAI_BACKENDS_YAML", str(registry))

    with pytest.raises(BackendRegistryError, match="unsafe permissions"):
        resolve_backend_command("claude")


def test_default_missing_registry_uses_legacy_path_resolution(monkeypatch):
    monkeypatch.delenv("KAI_BACKENDS_YAML", raising=False)
    monkeypatch.delenv("KAI_INSTALL_DIR", raising=False)
    monkeypatch.delenv("CLAUDE_BIN", raising=False)
    monkeypatch.setattr("kai.backend_registry.DEFAULT_BACKENDS_YAML", Path("/does/not/exist"))
    monkeypatch.setattr("kai.backend_registry.shutil.which", lambda name: f"/fake/{name}")

    assert resolve_backend_command("claude") == "/fake/claude"


def test_installed_mode_missing_registry_fails(monkeypatch):
    monkeypatch.delenv("KAI_BACKENDS_YAML", raising=False)
    monkeypatch.setenv("KAI_INSTALL_DIR", "/opt/kai")
    monkeypatch.setenv("CLAUDE_BIN", "/tmp/legacy-claude")
    monkeypatch.setattr("kai.backend_registry.DEFAULT_BACKENDS_YAML", Path("/does/not/exist"))

    assert backend_registry_is_authoritative()
    with pytest.raises(BackendRegistryError, match="does not exist"):
        resolve_backend_command("claude")


def test_explicit_missing_registry_fails(monkeypatch):
    monkeypatch.setenv("KAI_BACKENDS_YAML", "/does/not/exist")

    with pytest.raises(BackendRegistryError, match="does not exist"):
        resolve_backend_command("claude")


def test_render_backend_registry_is_stable():
    rendered = render_backend_registry(
        {
            "codex": {"driver": "codex", "runtime": "local_process", "command": "/usr/local/bin/codex"},
            "claude": {"driver": "claude", "runtime": "local_process", "command": "/opt/homebrew/bin/claude"},
        }
    )
    data = yaml.safe_load(rendered)

    assert list(data["backends"]) == ["claude", "codex"]
    assert data["version"] == 1
