"""Canonical runtime inventory for backend model discovery.

The inventory is deliberately read-only.  It describes which discovery lanes
exist and which non-secret inputs make their caches distinct; later discovery
workers own provider calls and durable catalogue refreshes.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from kai.backend_registry import BackendRegistryEntry
from kai.config import PROVIDER_KEY_VARS, Config
from kai.workshop.domain import PrincipalId, RuntimeAssignmentId, RuntimeProfileId
from kai.workshop.execution_state import (
    WorkshopExecutionStateNamespace,
    WorkshopExecutionStateRegistry,
)
from kai.workshop.runtime_profiles import (
    ProtectedRuntimeBackend,
    ProtectedRuntimeProfile,
    WorkshopRuntimeProfileRegistry,
)

_CACHE_KEY_DOMAIN = b"kai-workshop-model-discovery-cache:v1\0"
_IDENTITY_DOMAIN = b"kai-workshop-model-discovery-identity:v1\0"


class ModelDiscoveryInventoryError(RuntimeError):
    """Canonical model-discovery inventory cannot be resolved safely."""


class ModelDiscoverySelectionStatus(StrEnum):
    """Relationship between an authorized option and the current selection."""

    SELECTED = "selected"
    SELECTABLE = "selectable"
    UNAVAILABLE = "unavailable"
    MISCONFIGURED = "misconfigured"


class ModelDiscoveryReadiness(StrEnum):
    """Whether a discovery lane can be used without probing its backend."""

    READY = "ready"
    UNVERIFIED = "unverified"
    UNAVAILABLE = "unavailable"
    MISCONFIGURED = "misconfigured"


class ModelDiscoveryAuthMode(StrEnum):
    """Non-secret authentication class used by one backend/provider lane."""

    API_KEY = "api_key"
    SUBSCRIPTION = "subscription"
    BACKEND_MANAGED = "backend_managed"
    LOCAL = "local"


@dataclass(frozen=True, slots=True)
class ModelDiscoveryAuthContext:
    """Sanitized authentication inputs relevant to catalogue discovery."""

    mode: ModelDiscoveryAuthMode
    source: str
    configured: bool | None
    fingerprint: str


@dataclass(frozen=True, slots=True)
class ModelDiscoveryExecutableIdentity:
    """Installed executable identity without executing the backend."""

    configured_path: str | None
    resolved_path: str | None
    fingerprint: str
    readiness: ModelDiscoveryReadiness
    diagnostic: str | None


@dataclass(frozen=True, slots=True)
class ModelDiscoveryCacheInputs:
    """Stable, non-secret inputs that invalidate one discovery cache lane."""

    version: int
    principal_id: PrincipalId
    runtime_profile_id: RuntimeProfileId
    backend: str
    provider: str
    os_user: str
    executable_fingerprint: str
    auth_fingerprint: str
    default_model: str
    allowed_models: tuple[str, ...] | None
    role_models: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ModelDiscoveryBackendInventory:
    """One operator-authorized backend/provider discovery lane."""

    option_id: str
    backend: str
    provider: str
    selected: bool
    status: ModelDiscoverySelectionStatus
    readiness: ModelDiscoveryReadiness
    effective_os_user: str
    executable: ModelDiscoveryExecutableIdentity
    auth: ModelDiscoveryAuthContext
    default_model: str
    allowed_models: tuple[str, ...] | None
    role_models: tuple[tuple[str, str], ...]
    cache_inputs: ModelDiscoveryCacheInputs
    cache_key: str
    diagnostic: str | None


@dataclass(frozen=True, slots=True)
class ModelDiscoveryProfileInventory:
    """Canonical owner and discovery lanes for one protected runtime profile."""

    principal_id: PrincipalId
    channel_id: str
    agent_id: str
    runtime_profile_id: RuntimeProfileId
    runtime_assignment_id: RuntimeAssignmentId
    display_name: str
    selected_option_id: str
    backends: tuple[ModelDiscoveryBackendInventory, ...]


@dataclass(frozen=True, slots=True)
class ModelDiscoveryOperatorDiagnostics:
    """Sanitized aggregate diagnostics suitable for installed status output."""

    profiles: int
    options: int
    selected: int
    selectable: int
    unavailable: int
    misconfigured: int


SelectedBackendResolver = Callable[[RuntimeProfileId], tuple[str, str]]


class WorkshopModelDiscoveryInventoryService:
    """Resolve model-discovery inventory from canonical protected authorities."""

    def __init__(
        self,
        *,
        config: Config,
        runtime_profiles: WorkshopRuntimeProfileRegistry,
        execution_state: WorkshopExecutionStateRegistry,
        backend_registry: Mapping[str, BackendRegistryEntry],
        selected_backend: SelectedBackendResolver,
        service_os_user: str,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if not service_os_user.strip():
            raise ModelDiscoveryInventoryError("Kai service OS user is required")
        self._config = config
        self._runtime_profiles = runtime_profiles
        self._execution_state = execution_state
        self._backend_registry = dict(backend_registry)
        self._selected_backend = selected_backend
        self._service_os_user = service_os_user.strip()
        self._environment = os.environ if environment is None else environment

    @property
    def inventories(self) -> tuple[ModelDiscoveryProfileInventory, ...]:
        """Return a fresh snapshot so canonical selection changes are visible."""
        return tuple(self._inventory(profile) for profile in self._runtime_profiles.profiles)

    def for_principal(
        self,
        principal_id: str | PrincipalId,
    ) -> tuple[ModelDiscoveryProfileInventory, ...]:
        """Return only the caller principal's discovery inventory."""
        try:
            canonical_id = principal_id if isinstance(principal_id, PrincipalId) else PrincipalId(principal_id)
        except (TypeError, ValueError):
            return ()
        return tuple(item for item in self.inventories if item.principal_id == canonical_id)

    @property
    def operator_diagnostics(self) -> ModelDiscoveryOperatorDiagnostics:
        """Return aggregate, credential-free readiness counts."""
        inventories = self.inventories
        options = tuple(option for inventory in inventories for option in inventory.backends)
        counts = {status: 0 for status in ModelDiscoverySelectionStatus}
        for option in options:
            counts[option.status] += 1
        return ModelDiscoveryOperatorDiagnostics(
            profiles=len(inventories),
            options=len(options),
            selected=counts[ModelDiscoverySelectionStatus.SELECTED],
            selectable=counts[ModelDiscoverySelectionStatus.SELECTABLE],
            unavailable=counts[ModelDiscoverySelectionStatus.UNAVAILABLE],
            misconfigured=counts[ModelDiscoverySelectionStatus.MISCONFIGURED],
        )

    def _inventory(self, profile: ProtectedRuntimeProfile) -> ModelDiscoveryProfileInventory:
        namespace = self._execution_state.resolve_profile(profile.profile_id)
        backend, provider = self._selected_backend(profile.profile_id)
        selected_option_id = f"{backend}:{provider}"
        if selected_option_id not in {option.option_id for option in profile.backend_options}:
            raise ModelDiscoveryInventoryError(
                f"Runtime profile {profile.profile_id} selected an option outside protected policy"
            )
        effective_os_user = profile.os_user or self._service_os_user
        backends = tuple(
            self._backend_inventory(
                profile,
                namespace,
                option,
                effective_os_user=effective_os_user,
                selected=option.option_id == selected_option_id,
            )
            for option in sorted(profile.backend_options, key=lambda item: item.option_id)
        )
        return ModelDiscoveryProfileInventory(
            principal_id=namespace.principal_id,
            channel_id=str(namespace.channel_id),
            agent_id=str(namespace.agent_id),
            runtime_profile_id=profile.profile_id,
            runtime_assignment_id=RuntimeAssignmentId.derived(
                namespace.channel_id,
                f"runtime-profile:{namespace.agent_id}",
            ),
            display_name=profile.display_name,
            selected_option_id=selected_option_id,
            backends=backends,
        )

    def _backend_inventory(
        self,
        profile: ProtectedRuntimeProfile,
        namespace: WorkshopExecutionStateNamespace,
        option: ProtectedRuntimeBackend,
        *,
        effective_os_user: str,
        selected: bool,
    ) -> ModelDiscoveryBackendInventory:
        executable = _executable_identity(option.backend, self._backend_registry.get(option.backend))
        auth = _auth_context(
            option.backend,
            option.provider,
            effective_os_user,
            codex_auth_mode=self._config.codex_auth_mode,
            environment=self._environment,
        )
        readiness, diagnostic = _combined_readiness(executable, auth)
        if selected:
            status = ModelDiscoverySelectionStatus.SELECTED
        elif readiness == ModelDiscoveryReadiness.MISCONFIGURED:
            status = ModelDiscoverySelectionStatus.MISCONFIGURED
        elif readiness == ModelDiscoveryReadiness.UNAVAILABLE:
            status = ModelDiscoverySelectionStatus.UNAVAILABLE
        else:
            status = ModelDiscoverySelectionStatus.SELECTABLE
        cache_inputs = ModelDiscoveryCacheInputs(
            version=1,
            principal_id=namespace.principal_id,
            runtime_profile_id=profile.profile_id,
            backend=option.backend,
            provider=option.provider,
            os_user=effective_os_user,
            executable_fingerprint=executable.fingerprint,
            auth_fingerprint=auth.fingerprint,
            default_model=option.model,
            allowed_models=option.allowed_models,
            role_models=option.role_models,
        )
        return ModelDiscoveryBackendInventory(
            option_id=option.option_id,
            backend=option.backend,
            provider=option.provider,
            selected=selected,
            status=status,
            readiness=readiness,
            effective_os_user=effective_os_user,
            executable=executable,
            auth=auth,
            default_model=option.model,
            allowed_models=option.allowed_models,
            role_models=option.role_models,
            cache_inputs=cache_inputs,
            cache_key=_fingerprint(
                _CACHE_KEY_DOMAIN,
                {
                    "version": cache_inputs.version,
                    "principal_id": str(cache_inputs.principal_id),
                    "runtime_profile_id": str(cache_inputs.runtime_profile_id),
                    "backend": cache_inputs.backend,
                    "provider": cache_inputs.provider,
                    "os_user": cache_inputs.os_user,
                    "executable": cache_inputs.executable_fingerprint,
                    "auth": cache_inputs.auth_fingerprint,
                    "default_model": cache_inputs.default_model,
                    "allowed_models": cache_inputs.allowed_models,
                    "role_models": cache_inputs.role_models,
                },
            ),
            diagnostic=diagnostic,
        )


def _auth_context(
    backend: str,
    provider: str,
    os_user: str,
    *,
    codex_auth_mode: str,
    environment: Mapping[str, str],
) -> ModelDiscoveryAuthContext:
    key_var = PROVIDER_KEY_VARS.get(provider)
    key_configured = bool(key_var and environment.get(key_var, "").strip())
    if provider == "ollama":
        mode = ModelDiscoveryAuthMode.LOCAL
        source = "local provider"
        configured: bool | None = True
    elif backend == "codex":
        mode = ModelDiscoveryAuthMode.API_KEY if codex_auth_mode == "api_key" else ModelDiscoveryAuthMode.SUBSCRIPTION
        source = key_var or "backend login"
        configured = key_configured if mode == ModelDiscoveryAuthMode.API_KEY else None
    elif backend == "claude":
        mode = ModelDiscoveryAuthMode.API_KEY if key_configured else ModelDiscoveryAuthMode.SUBSCRIPTION
        source = key_var if key_configured and key_var is not None else "backend login"
        configured = True if key_configured else None
    elif backend == "pi" and provider in {"openai-codex", "github-copilot"}:
        mode = ModelDiscoveryAuthMode.SUBSCRIPTION
        source = "backend login"
        configured = None
    elif key_configured:
        mode = ModelDiscoveryAuthMode.API_KEY
        source = key_var or "environment credential"
        configured = True
    else:
        mode = ModelDiscoveryAuthMode.BACKEND_MANAGED
        source = "backend configuration"
        configured = None
    fingerprint = _fingerprint(
        _IDENTITY_DOMAIN,
        {
            "kind": "auth",
            "backend": backend,
            "provider": provider,
            "os_user": os_user,
            "mode": mode.value,
            "source": source,
            "configured": configured,
        },
    )
    return ModelDiscoveryAuthContext(mode, source, configured, fingerprint)


def _executable_identity(
    backend: str,
    entry: BackendRegistryEntry | None,
) -> ModelDiscoveryExecutableIdentity:
    if entry is None:
        return ModelDiscoveryExecutableIdentity(
            configured_path=None,
            resolved_path=None,
            fingerprint=_fingerprint(
                _IDENTITY_DOMAIN,
                {"kind": "executable", "backend": backend, "state": "missing_registry_entry"},
            ),
            readiness=ModelDiscoveryReadiness.MISCONFIGURED,
            diagnostic="Backend has no installed registry entry",
        )
    configured = Path(entry.command)
    if (
        entry.id != backend
        or entry.driver != backend
        or entry.runtime != "local_process"
        or not configured.is_absolute()
    ):
        return ModelDiscoveryExecutableIdentity(
            configured_path=entry.command,
            resolved_path=None,
            fingerprint=_fingerprint(
                _IDENTITY_DOMAIN,
                {"kind": "executable", "backend": backend, "state": "invalid_registry_entry"},
            ),
            readiness=ModelDiscoveryReadiness.MISCONFIGURED,
            diagnostic="Backend registry entry is invalid",
        )
    try:
        resolved = configured.resolve(strict=True)
        metadata = resolved.stat()
    except OSError:
        return ModelDiscoveryExecutableIdentity(
            configured_path=str(configured),
            resolved_path=None,
            fingerprint=_fingerprint(
                _IDENTITY_DOMAIN,
                {"kind": "executable", "backend": backend, "path": str(configured), "state": "unavailable"},
            ),
            readiness=ModelDiscoveryReadiness.UNAVAILABLE,
            diagnostic="Backend executable is unavailable",
        )
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        readiness = ModelDiscoveryReadiness.UNAVAILABLE
        diagnostic = "Backend command is not an executable file"
    elif metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        readiness = ModelDiscoveryReadiness.MISCONFIGURED
        diagnostic = "Backend executable has unsafe write permissions"
    else:
        readiness = ModelDiscoveryReadiness.READY
        diagnostic = None
    fingerprint = _fingerprint(
        _IDENTITY_DOMAIN,
        {
            "kind": "executable",
            "backend": backend,
            "configured": str(configured),
            "resolved": str(resolved),
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "size": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
            "mode": stat.S_IMODE(metadata.st_mode),
        },
    )
    return ModelDiscoveryExecutableIdentity(
        configured_path=str(configured),
        resolved_path=str(resolved),
        fingerprint=fingerprint,
        readiness=readiness,
        diagnostic=diagnostic,
    )


def _combined_readiness(
    executable: ModelDiscoveryExecutableIdentity,
    auth: ModelDiscoveryAuthContext,
) -> tuple[ModelDiscoveryReadiness, str | None]:
    if executable.readiness != ModelDiscoveryReadiness.READY:
        return executable.readiness, executable.diagnostic
    if auth.mode == ModelDiscoveryAuthMode.API_KEY and auth.configured is not True:
        return ModelDiscoveryReadiness.UNAVAILABLE, "Required environment credential is not configured"
    if auth.configured is None:
        return ModelDiscoveryReadiness.UNVERIFIED, "Backend-managed authentication is not probed at startup"
    return ModelDiscoveryReadiness.READY, None


def _fingerprint(domain: bytes, value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(domain + payload).hexdigest()
