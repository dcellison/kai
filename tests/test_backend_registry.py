from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from kai.backend_registry import (
    BackendRegistryError,
    backend_registry_is_authoritative,
    load_backend_registry,
    load_backend_registry_default_backend,
    render_backend_registry,
    resolve_backend_command,
    resolve_default_backend,
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


def test_registry_rejects_unsupported_backend_id(tmp_path, monkeypatch):
    custom = _exe(tmp_path / "custom")
    registry = tmp_path / "backends.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "backends": {
                    "custom": {
                        "driver": "custom",
                        "runtime": "local_process",
                        "command": str(custom),
                    }
                },
            }
        )
    )
    monkeypatch.setenv("KAI_BACKENDS_YAML", str(registry))

    with pytest.raises(BackendRegistryError, match="unsupported backend 'custom'"):
        load_backend_registry()


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


def test_registry_driver_must_match_backend_id(tmp_path, monkeypatch):
    codex = _exe(tmp_path / "codex")
    registry = tmp_path / "backends.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "backends": {
                    "codex": {
                        "driver": "claude",
                        "runtime": "local_process",
                        "command": str(codex),
                    }
                },
            }
        )
    )
    monkeypatch.setenv("KAI_BACKENDS_YAML", str(registry))

    with pytest.raises(BackendRegistryError, match="driver 'claude' must match backend id"):
        load_backend_registry()


def test_registry_runtime_must_be_local_process(tmp_path, monkeypatch):
    codex = _exe(tmp_path / "codex")
    registry = tmp_path / "backends.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "backends": {
                    "codex": {
                        "driver": "codex",
                        "runtime": "container",
                        "command": str(codex),
                    }
                },
            }
        )
    )
    monkeypatch.setenv("KAI_BACKENDS_YAML", str(registry))

    with pytest.raises(BackendRegistryError, match="runtime 'container' is not supported"):
        load_backend_registry()


def test_registry_defaults_driver_and_runtime_to_backend_local_process(tmp_path, monkeypatch):
    codex = _exe(tmp_path / "codex")
    registry = tmp_path / "backends.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "backends": {
                    "codex": {
                        "command": str(codex),
                    }
                },
            }
        )
    )
    monkeypatch.setenv("KAI_BACKENDS_YAML", str(registry))

    entry = load_backend_registry()["codex"]
    assert entry.driver == "codex"
    assert entry.runtime == "local_process"


def test_registry_allowed_models_entries_must_be_strings(tmp_path, monkeypatch):
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
                        "allowed_models": ["gpt-5.6-sol", 56],
                    }
                },
            }
        )
    )
    monkeypatch.setenv("KAI_BACKENDS_YAML", str(registry))

    with pytest.raises(BackendRegistryError, match="allowed_models entries must be strings"):
        load_backend_registry()


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


def test_installed_registry_file_must_be_root_owned(tmp_path, monkeypatch):
    """Installed mode treats the registry as an admin-owned capability map."""
    registry = tmp_path / "backends.yaml"
    registry.write_text("version: 1\nbackends: {}\n")
    monkeypatch.setenv("KAI_INSTALL_DIR", "/opt/kai")
    monkeypatch.setattr("kai.backend_registry.DEFAULT_BACKENDS_YAML", registry)
    real_stat = Path.stat

    def fake_stat(self, *args, **kwargs):
        result = real_stat(self, *args, **kwargs)
        if self == registry:
            return SimpleNamespace(st_mode=result.st_mode, st_uid=12345)
        return result

    monkeypatch.setattr("kai.backend_registry.Path.stat", fake_stat)

    with pytest.raises(BackendRegistryError, match="root ownership"):
        load_backend_registry()


def test_explicit_dev_registry_does_not_require_root_owner(tmp_path, monkeypatch):
    """A test/dev registry can be user-owned when installed mode is not active."""
    registry = tmp_path / "backends.yaml"
    registry.write_text("version: 1\nbackends: {}\n")
    registry.chmod(0o644)
    monkeypatch.setenv("KAI_BACKENDS_YAML", str(registry))
    monkeypatch.delenv("KAI_INSTALL_DIR", raising=False)

    assert load_backend_registry() == {}


@pytest.mark.parametrize("mode", [0o775, 0o757])
def test_registry_command_must_not_be_group_or_world_writable(tmp_path, monkeypatch, mode):
    """The registry file is not sufficient if the registered executable
    can be replaced in place by a non-admin account.
    """
    codex = _exe(tmp_path / "codex")
    codex.chmod(mode)
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

    with pytest.raises(BackendRegistryError, match="unsafe permissions"):
        resolve_backend_command("codex")


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


def test_render_backend_registry_includes_valid_default_backend():
    rendered = render_backend_registry(
        {
            "codex": {"driver": "codex", "runtime": "local_process", "command": "/usr/local/bin/codex"},
            "goose": {"driver": "goose", "runtime": "local_process", "command": "/opt/homebrew/bin/goose"},
        },
        default_backend="codex",
    )
    data = yaml.safe_load(rendered)

    assert data["default_backend"] == "codex"
    assert list(data["backends"]) == ["codex", "goose"]


def test_render_backend_registry_rejects_unknown_default_backend():
    with pytest.raises(BackendRegistryError, match="default_backend 'opencode' has no backend entry"):
        render_backend_registry(
            {"codex": {"driver": "codex", "runtime": "local_process", "command": "/usr/local/bin/codex"}},
            default_backend="opencode",
        )


def test_resolve_default_backend_uses_registry_default(tmp_path, monkeypatch):
    codex = _exe(tmp_path / "codex")
    goose = _exe(tmp_path / "goose")
    registry = tmp_path / "backends.yaml"
    registry.write_text(
        render_backend_registry(
            {
                "codex": {"driver": "codex", "runtime": "local_process", "command": str(codex)},
                "goose": {"driver": "goose", "runtime": "local_process", "command": str(goose)},
            },
            default_backend="goose",
        )
    )
    monkeypatch.setenv("KAI_BACKENDS_YAML", str(registry))

    assert load_backend_registry_default_backend() == "goose"
    assert resolve_default_backend("") == "goose"


def test_resolve_default_backend_uses_sole_registry_backend(tmp_path, monkeypatch):
    codex = _exe(tmp_path / "codex")
    registry = tmp_path / "backends.yaml"
    registry.write_text(
        render_backend_registry({"codex": {"driver": "codex", "runtime": "local_process", "command": str(codex)}})
    )
    monkeypatch.setenv("KAI_BACKENDS_YAML", str(registry))

    assert resolve_default_backend("") == "codex"


def test_resolve_default_backend_requires_selection_for_ambiguous_registry(tmp_path, monkeypatch):
    codex = _exe(tmp_path / "codex")
    goose = _exe(tmp_path / "goose")
    registry = tmp_path / "backends.yaml"
    registry.write_text(
        render_backend_registry(
            {
                "codex": {"driver": "codex", "runtime": "local_process", "command": str(codex)},
                "goose": {"driver": "goose", "runtime": "local_process", "command": str(goose)},
            }
        )
    )
    monkeypatch.setenv("KAI_BACKENDS_YAML", str(registry))

    with pytest.raises(BackendRegistryError, match="does not define default_backend"):
        resolve_default_backend("")


def test_resolve_default_backend_rejects_configured_backend_missing_from_registry(tmp_path, monkeypatch):
    codex = _exe(tmp_path / "codex")
    registry = tmp_path / "backends.yaml"
    registry.write_text(
        render_backend_registry({"codex": {"driver": "codex", "runtime": "local_process", "command": str(codex)}})
    )
    monkeypatch.setenv("KAI_BACKENDS_YAML", str(registry))

    with pytest.raises(BackendRegistryError, match="not installed"):
        resolve_default_backend("goose")
