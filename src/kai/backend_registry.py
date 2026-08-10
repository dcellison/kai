"""
Installed backend registry.

This module is the machine-capability boundary for local agent
backends. User-facing configuration names backend IDs ("codex",
"claude", etc.); protected installs resolve executable paths from
an admin-owned registry. Single-user/dev runs keep the legacy
``*_BIN`` / PATH fallback unless an explicit registry override is set.
"""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kai.protected_config import ProtectedConfigError, validate_protected_file_metadata

DEFAULT_BACKENDS_YAML = Path("/etc/kai/backends.yaml")
BACKENDS_YAML_ENV = "KAI_BACKENDS_YAML"
INSTALL_DIR_ENV = "KAI_INSTALL_DIR"

_BACKEND_ENV_VARS: dict[str, str] = {
    "claude": "CLAUDE_BIN",
    "codex": "CODEX_BIN",
    "opencode": "OPENCODE_BIN",
    "goose": "GOOSE_BIN",
}
_SUPPORTED_RUNTIME = "local_process"


class BackendRegistryError(Exception):
    """Raised when the installed backend registry is malformed or unusable."""


@dataclass(frozen=True)
class BackendRegistryEntry:
    """One installed backend capability."""

    id: str
    driver: str
    runtime: str
    command: str
    allowed_models: tuple[str, ...] = ()


def backend_registry_path() -> Path:
    """Return the configured backend registry path."""
    override = os.environ.get(BACKENDS_YAML_ENV, "").strip()
    if override:
        return Path(override)
    return DEFAULT_BACKENDS_YAML


def backend_registry_exists(path: Path | None = None) -> bool:
    """Return whether an installed backend registry is present."""
    if path is not None:
        return path.is_file()
    if os.environ.get(BACKENDS_YAML_ENV, "").strip():
        return backend_registry_path().is_file()
    if not os.environ.get(INSTALL_DIR_ENV, "").strip():
        return False
    return backend_registry_path().is_file()


def backend_registry_required() -> bool:
    """Return whether this process must use the backend registry."""
    return bool(os.environ.get(BACKENDS_YAML_ENV, "").strip() or os.environ.get(INSTALL_DIR_ENV, "").strip())


def backend_registry_is_authoritative() -> bool:
    """Return whether backend command resolution must use the registry."""
    return backend_registry_required() or backend_registry_exists()


def load_backend_registry(path: Path | None = None) -> dict[str, BackendRegistryEntry]:
    """
    Load the admin-owned backend registry.

    Returns an empty dict only for single-user/dev mode when no registry
    exists. A missing required registry, or a present but malformed
    registry, is a configuration error because runtime and sudoers path
    decisions must not silently fall back to unrelated binaries.
    """
    explicit_path = path is not None or backend_registry_required()
    registry_path = path or backend_registry_path()
    if not registry_path.is_file():
        if explicit_path:
            raise BackendRegistryError(f"backend registry {registry_path} does not exist")
        return {}
    try:
        validate_protected_file_metadata(
            registry_path,
            max_mode=0o644,
            require_root_owner=bool(os.environ.get(INSTALL_DIR_ENV, "").strip()),
        )
    except ProtectedConfigError as e:
        raise BackendRegistryError(str(e)) from e
    try:
        raw = yaml.safe_load(registry_path.read_text()) or {}
    except OSError as e:
        raise BackendRegistryError(f"could not read backend registry {registry_path}: {e}") from e
    except yaml.YAMLError as e:
        raise BackendRegistryError(f"backend registry {registry_path} is not valid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise BackendRegistryError(f"backend registry {registry_path}: expected a YAML mapping")
    backends = raw.get("backends", {})
    if not isinstance(backends, dict):
        raise BackendRegistryError(f"backend registry {registry_path}: 'backends' must be a mapping")

    entries: dict[str, BackendRegistryEntry] = {}
    for backend_id, value in backends.items():
        backend = str(backend_id).strip().lower()
        if not backend:
            raise BackendRegistryError(f"backend registry {registry_path}: empty backend id")
        if backend not in _BACKEND_ENV_VARS:
            valid = ", ".join(sorted(_BACKEND_ENV_VARS))
            raise BackendRegistryError(
                f"backend registry {registry_path}: unsupported backend {backend!r}; valid backends: {valid}"
            )
        if not isinstance(value, dict):
            raise BackendRegistryError(f"backend registry {registry_path}: backend {backend!r} must be a mapping")
        command = str(value.get("command") or "").strip()
        if not command:
            raise BackendRegistryError(f"backend registry {registry_path}: backend {backend!r} missing command")
        driver = str(value.get("driver") or backend).strip().lower() or backend
        if driver != backend:
            raise BackendRegistryError(
                f"backend registry {registry_path}: backend {backend!r} driver {driver!r} must match backend id"
            )
        runtime = str(value.get("runtime") or _SUPPORTED_RUNTIME).strip().lower() or _SUPPORTED_RUNTIME
        if runtime != _SUPPORTED_RUNTIME:
            raise BackendRegistryError(
                f"backend registry {registry_path}: backend {backend!r} runtime {runtime!r} is not supported"
            )
        models = value.get("allowed_models", [])
        if models is None:
            model_tuple: tuple[str, ...] = ()
        elif isinstance(models, list):
            checked_models: list[str] = []
            for item in models:
                if not isinstance(item, str):
                    raise BackendRegistryError(
                        f"backend registry {registry_path}: backend {backend!r} allowed_models entries must be strings"
                    )
                model = item.strip()
                if model:
                    checked_models.append(model)
            model_tuple = tuple(checked_models)
        else:
            raise BackendRegistryError(
                f"backend registry {registry_path}: backend {backend!r} allowed_models must be a list"
            )
        entries[backend] = BackendRegistryEntry(
            id=backend,
            driver=driver,
            runtime=runtime,
            command=command,
            allowed_models=model_tuple,
        )
    return entries


def load_backend_registry_default_backend(path: Path | None = None) -> str:
    """Return the registry's configured default backend, if present."""
    entries = load_backend_registry(path)
    registry_path = path or backend_registry_path()
    try:
        raw = yaml.safe_load(registry_path.read_text()) or {}
    except OSError as e:
        raise BackendRegistryError(f"could not read backend registry {registry_path}: {e}") from e
    except yaml.YAMLError as e:
        raise BackendRegistryError(f"backend registry {registry_path} is not valid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise BackendRegistryError(f"backend registry {registry_path}: expected a YAML mapping")
    default_backend = str(raw.get("default_backend") or "").strip().lower()
    if not default_backend:
        return ""
    if default_backend not in entries:
        raise BackendRegistryError(
            f"backend registry {registry_path}: default_backend {default_backend!r} has no backend entry"
        )
    return default_backend


def resolve_default_backend(configured_backend: str = "") -> str:
    """
    Resolve the effective default backend without provider-specific fallback.

    A configured backend wins, but installed/explicit-registry mode also
    requires it to exist in the backend registry. If no backend is
    configured, an authoritative registry may select one via
    `default_backend`, or by containing exactly one backend entry.
    """
    configured = configured_backend.strip().lower()
    registry: dict[str, BackendRegistryEntry] | None = None
    if backend_registry_is_authoritative():
        registry = load_backend_registry()

    if configured:
        if configured not in _BACKEND_ENV_VARS:
            valid = ", ".join(sorted(_BACKEND_ENV_VARS))
            raise BackendRegistryError(
                f"configured default backend {configured!r} is not valid; valid backends: {valid}"
            )
        if registry is not None and configured not in registry:
            installed = ", ".join(sorted(registry)) or "<none>"
            raise BackendRegistryError(
                f"configured default backend {configured!r} is not installed; installed backends: {installed}"
            )
        return configured

    if registry is not None:
        registry_default = load_backend_registry_default_backend()
        if registry_default:
            return registry_default
        if len(registry) == 1:
            return next(iter(registry))
        installed = ", ".join(sorted(registry)) or "<none>"
        raise BackendRegistryError(
            "DEFAULT_BACKEND is not set and backend registry does not define default_backend; "
            f"installed backends: {installed}"
        )

    raise BackendRegistryError("DEFAULT_BACKEND is not set and no backend registry is available")


def _validate_absolute_executable(backend: str, command: str, source: str) -> str:
    path = Path(command)
    if not path.is_absolute():
        raise BackendRegistryError(f"{source} for backend {backend!r} is not absolute: {command!r}")
    if not path.is_file():
        raise BackendRegistryError(f"{source} for backend {backend!r} is not a file: {command!r}")
    if not os.access(str(path), os.X_OK):
        raise BackendRegistryError(f"{source} for backend {backend!r} is not executable: {command!r}")
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError as e:
        raise BackendRegistryError(f"could not stat {source} for backend {backend!r}: {e}") from e
    if mode & 0o022:
        raise BackendRegistryError(
            f"{source} for backend {backend!r} has unsafe permissions {mode:#04o}; remove group/other write access"
        )
    return str(path)


def _legacy_env_or_path_command(backend: str, *, allow_bare_fallback: bool) -> str:
    env_var = _BACKEND_ENV_VARS.get(backend)
    if env_var is None:
        raise ValueError(f"unknown backend for command resolution: {backend!r}")
    override = os.environ.get(env_var)
    if override:
        if allow_bare_fallback:
            return override
        return _validate_absolute_executable(backend, override, env_var)
    if allow_bare_fallback:
        return backend
    resolved = shutil.which(backend)
    if resolved is not None:
        return resolved
    raise BackendRegistryError(f"{env_var} unset, `{backend}` not on PATH")


def resolve_backend_command(backend: str, *, allow_bare_fallback: bool = False) -> str:
    """
    Resolve the command for an installed backend.

    Precedence:
      1. admin-owned backend registry for protected installs or explicit
         registry overrides;
      2. legacy ``*_BIN`` environment variable for single-user/dev mode;
      3. PATH lookup for single-user/dev mode;
      4. optional bare command fallback for persistent/dev spawns.
    """
    backend = backend.strip().lower()
    if backend_registry_is_authoritative():
        registry = load_backend_registry()
        entry = registry.get(backend)
        if entry is None:
            raise BackendRegistryError(f"backend registry has no entry for backend {backend!r}")
        return _validate_absolute_executable(backend, entry.command, "backend registry command")
    return _legacy_env_or_path_command(backend, allow_bare_fallback=allow_bare_fallback)


def render_backend_registry(entries: dict[str, dict[str, Any]], *, default_backend: str = "") -> str:
    """Render backend registry YAML with stable ordering."""
    ordered: dict[str, Any] = {"version": 1}
    normalized_default = default_backend.strip().lower()
    if normalized_default:
        if normalized_default not in entries:
            raise BackendRegistryError(
                f"default_backend {normalized_default!r} has no backend entry; "
                f"available backends: {', '.join(sorted(entries))}"
            )
        ordered["default_backend"] = normalized_default
    ordered["backends"] = {key: entries[key] for key in sorted(entries)}
    return yaml.safe_dump(ordered, sort_keys=False)
