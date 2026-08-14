"""Protected, transport-neutral Workshop runtime-profile policy."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kai.backend_registry import BackendRegistryEntry, load_backend_registry
from kai.config import (
    BACKEND_PROVIDERS,
    VALID_BACKENDS,
    Config,
    UserConfig,
    _read_protected_file,
    canonicalize_model_for_backend,
    get_default_model_for_backend,
    get_effective_provider,
    get_user_backend_and_provider,
    validate_model_for_backend_policy,
)
from kai.workshop.domain import RuntimeProfileId

_PROFILE_DERIVATION_DOMAIN = b"kai-workshop-runtime-profile:v1\0"
_COMPATIBILITY_KEY_DERIVATION_DOMAIN = b"kai-workshop-runtime-compatibility-key:v1\0"

DEFAULT_RUNTIME_PROFILES_YAML = Path("/etc/kai/runtime-profiles.yaml")
RUNTIME_PROFILES_YAML_ENV = "KAI_RUNTIME_PROFILES_YAML"
INSTALL_DIR_ENV = "KAI_INSTALL_DIR"
_OS_USER_RE = re.compile(r"^[a-zA-Z0-9._-]+$")


class WorkshopRuntimeProfileError(RuntimeError):
    """A canonical runtime profile is missing or conflicts with policy."""


@dataclass(frozen=True, slots=True)
class ProtectedRuntimeProfile:
    """One operator-configured execution identity behind an opaque ID.

    ``runtime_config_id`` is deliberately private compatibility state. The
    current host runtime still stores settings, files, memory, and subprocess
    instances under an integer key. It is not a transport identity and new
    profiles derive it from the opaque profile ID. Canonical Workshop callers
    see only ``profile_id``.
    """

    profile_id: RuntimeProfileId
    runtime_config_id: int
    display_name: str
    os_user: str | None
    backend: str
    provider: str
    model: str
    timeout_seconds: int


def _compatibility_model(
    user: UserConfig | None,
    config: Config,
    *,
    backend: str,
    provider: str,
) -> str:
    """Mirror the conversational pool's pre-cutover model cascade."""
    global_provider = get_effective_provider(config.default_backend, config.default_provider)
    if user is not None and user.model:
        model = user.model
    elif backend == config.default_backend and provider == global_provider:
        model = config.default_model
    else:
        model = get_default_model_for_backend(backend, provider)
    return canonicalize_model_for_backend(model, backend)


def _registry_model_ceiling(
    backend: str,
    installed: Mapping[str, BackendRegistryEntry | object] | None,
    *,
    profile_id: RuntimeProfileId,
) -> tuple[str, ...] | None:
    """Read one already-loaded backend entry's optional model ceiling."""
    if installed is None:
        return None
    entry = installed[backend]
    if isinstance(entry, BackendRegistryEntry):
        return entry.allowed_models or None
    if isinstance(entry, Mapping):
        raw = entry.get("allowed_models")
        if raw is None:
            return None
        if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
            raise WorkshopRuntimeProfileError(
                f"Runtime profile {profile_id}: backend registry allowed_models is invalid"
            )
        checked = tuple(item.strip() for item in raw if item.strip())
        return checked or None
    raise WorkshopRuntimeProfileError(
        f"Runtime profile {profile_id}: backend registry entry for {backend!r} is invalid"
    )


def runtime_profile_id_for_config_id(runtime_config_id: int) -> RuntimeProfileId:
    """Derive the preserved profile ID for one migrated configured user."""
    if isinstance(runtime_config_id, bool) or not isinstance(runtime_config_id, int) or runtime_config_id <= 0:
        raise WorkshopRuntimeProfileError("Runtime configuration ID must be a positive integer")
    digest = hashlib.sha256(_PROFILE_DERIVATION_DOMAIN + str(runtime_config_id).encode("ascii")).hexdigest()[:32]
    return RuntimeProfileId(f"rtp_{digest}")


def compatibility_runtime_config_id_for_profile_id(profile_id: RuntimeProfileId) -> int:
    """Derive non-transport compatibility storage for a new profile.

    The high bit remains clear so the result fits SQLite's signed INTEGER.
    Existing migrated profiles carry their former positive key explicitly;
    newly authored profiles need no Telegram-shaped value.
    """
    digest = hashlib.sha256(_COMPATIBILITY_KEY_DERIVATION_DOMAIN + str(profile_id).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1) or 1


def runtime_profiles_path() -> Path:
    """Return the configured protected runtime-policy path."""
    override = os.environ.get(RUNTIME_PROFILES_YAML_ENV, "").strip()
    return Path(override) if override else DEFAULT_RUNTIME_PROFILES_YAML


def _positive_compatibility_key(value: object, *, profile_id: RuntimeProfileId) -> int:
    if value is None:
        return compatibility_runtime_config_id_for_profile_id(profile_id)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WorkshopRuntimeProfileError(
            f"Runtime profile {profile_id}: compatibility_runtime_config_id must be a positive integer"
        )
    return value


def _policy_text(path: Path) -> str:
    """Read an explicit/dev file directly or the canonical protected file via sudo."""
    canonical = path == DEFAULT_RUNTIME_PROFILES_YAML and not os.environ.get(RUNTIME_PROFILES_YAML_ENV, "").strip()
    if canonical:
        content = _read_protected_file(str(path))
        if content is None:
            raise WorkshopRuntimeProfileError(f"Protected runtime policy {path} is missing or unreadable")
        return content
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkshopRuntimeProfileError(f"Could not read runtime policy {path}: {exc}") from exc


class WorkshopRuntimeProfileRegistry:
    """Resolve canonical profiles only through protected operator policy."""

    def __init__(self, profiles: tuple[ProtectedRuntimeProfile, ...]) -> None:
        by_id: dict[RuntimeProfileId, ProtectedRuntimeProfile] = {}
        by_config_id: dict[int, ProtectedRuntimeProfile] = {}
        for profile in profiles:
            if not isinstance(profile, ProtectedRuntimeProfile):
                raise TypeError("profiles must contain ProtectedRuntimeProfile values")
            if profile.profile_id in by_id:
                raise WorkshopRuntimeProfileError("Duplicate runtime profile ID")
            if profile.runtime_config_id in by_config_id:
                raise WorkshopRuntimeProfileError("Duplicate runtime compatibility configuration ID")
            by_id[profile.profile_id] = profile
            by_config_id[profile.runtime_config_id] = profile
        if not by_id:
            raise WorkshopRuntimeProfileError("At least one protected runtime profile is required")
        self._by_id = by_id
        self._by_config_id = by_config_id

    @classmethod
    def from_config(cls, config: Config) -> WorkshopRuntimeProfileRegistry:
        """Build the development-only compatibility projection.

        Protected startup uses :meth:`load`; this constructor remains useful
        for direct/dev runs and focused tests with no installed policy file.
        """
        profiles: list[ProtectedRuntimeProfile] = []
        for runtime_config_id, user in sorted(config.user_configs.items()):
            if user.telegram_id != runtime_config_id:
                raise WorkshopRuntimeProfileError("Configured-user key does not match its protected user record")
            backend, provider = get_user_backend_and_provider(user, config)
            profiles.append(
                ProtectedRuntimeProfile(
                    profile_id=runtime_profile_id_for_config_id(runtime_config_id),
                    runtime_config_id=runtime_config_id,
                    display_name=user.name,
                    os_user=user.os_user,
                    backend=backend,
                    provider=provider,
                    model=_compatibility_model(user, config, backend=backend, provider=provider),
                    timeout_seconds=user.timeout if user.timeout is not None else config.default_timeout,
                )
            )
        return cls(tuple(profiles))

    @classmethod
    def from_document(
        cls,
        document: object,
        *,
        backend_registry: Mapping[str, BackendRegistryEntry | object] | None = None,
    ) -> WorkshopRuntimeProfileRegistry:
        """Parse and validate one versioned runtime-policy document."""
        if not isinstance(document, dict):
            raise WorkshopRuntimeProfileError("Runtime policy must be a YAML mapping")
        if document.get("version") != 1:
            raise WorkshopRuntimeProfileError("Runtime policy version must be 1")
        raw_profiles = document.get("runtime_profiles")
        if not isinstance(raw_profiles, dict) or not raw_profiles:
            raise WorkshopRuntimeProfileError("Runtime policy must contain a non-empty runtime_profiles mapping")

        installed = backend_registry
        profiles: list[ProtectedRuntimeProfile] = []
        for raw_profile_id, raw_profile in raw_profiles.items():
            try:
                profile_id = RuntimeProfileId(str(raw_profile_id))
            except (TypeError, ValueError) as exc:
                raise WorkshopRuntimeProfileError(f"Invalid runtime profile ID {raw_profile_id!r}") from exc
            if not isinstance(raw_profile, dict):
                raise WorkshopRuntimeProfileError(f"Runtime profile {profile_id} must be a mapping")
            display_name = str(raw_profile.get("display_name") or "").strip()
            if not display_name:
                raise WorkshopRuntimeProfileError(f"Runtime profile {profile_id}: display_name is required")
            backend = str(raw_profile.get("backend") or "").strip().lower()
            if backend not in VALID_BACKENDS:
                raise WorkshopRuntimeProfileError(f"Runtime profile {profile_id}: unsupported backend {backend!r}")
            if installed is not None and backend not in installed:
                raise WorkshopRuntimeProfileError(
                    f"Runtime profile {profile_id}: backend {backend!r} is not present in the backend registry"
                )
            provider = str(raw_profile.get("provider") or "").strip().lower()
            allowed_providers = BACKEND_PROVIDERS[backend]
            if not provider and len(allowed_providers) == 1:
                provider = allowed_providers[0]
            elif not provider:
                raise WorkshopRuntimeProfileError(
                    f"Runtime profile {profile_id}: provider is required for backend {backend!r}"
                )
            if provider not in allowed_providers:
                allowed = ", ".join(allowed_providers)
                raise WorkshopRuntimeProfileError(
                    f"Runtime profile {profile_id}: provider {provider!r} is not valid for backend {backend!r}; "
                    f"expected one of: {allowed}"
                )
            model = canonicalize_model_for_backend(str(raw_profile.get("model") or "").strip(), backend)
            if not model:
                raise WorkshopRuntimeProfileError(f"Runtime profile {profile_id}: model is required")
            allowed_models = _registry_model_ceiling(backend, installed, profile_id=profile_id)
            if not validate_model_for_backend_policy(
                model,
                backend,
                provider,
                allowed_models=allowed_models,
            ):
                raise WorkshopRuntimeProfileError(
                    f"Runtime profile {profile_id}: model {model!r} is not valid for "
                    f"backend {backend!r} and provider {provider!r}"
                )
            raw_timeout = raw_profile.get("timeout_seconds")
            if isinstance(raw_timeout, bool) or not isinstance(raw_timeout, int) or raw_timeout <= 0:
                raise WorkshopRuntimeProfileError(
                    f"Runtime profile {profile_id}: timeout_seconds must be a positive integer"
                )
            raw_os_user = raw_profile.get("os_user")
            os_user = None if raw_os_user is None else str(raw_os_user).strip() or None
            if os_user is not None and _OS_USER_RE.fullmatch(os_user) is None:
                raise WorkshopRuntimeProfileError(f"Runtime profile {profile_id}: os_user is invalid")
            profiles.append(
                ProtectedRuntimeProfile(
                    profile_id=profile_id,
                    runtime_config_id=_positive_compatibility_key(
                        raw_profile.get("compatibility_runtime_config_id"),
                        profile_id=profile_id,
                    ),
                    display_name=display_name,
                    os_user=os_user,
                    backend=backend,
                    provider=provider,
                    model=model,
                    timeout_seconds=raw_timeout,
                )
            )
        return cls(tuple(profiles))

    @classmethod
    def from_yaml(
        cls,
        content: str,
        *,
        backend_registry: Mapping[str, BackendRegistryEntry | object] | None = None,
    ) -> WorkshopRuntimeProfileRegistry:
        try:
            document: Any = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            raise WorkshopRuntimeProfileError(f"Runtime policy is not valid YAML: {exc}") from exc
        return cls.from_document(document, backend_registry=backend_registry)

    @classmethod
    def load(
        cls,
        config: Config,
        *,
        path: Path | None = None,
    ) -> WorkshopRuntimeProfileRegistry:
        """Load installed policy, failing closed when protected policy is absent."""
        policy_path = path or runtime_profiles_path()
        explicit = path is not None or bool(os.environ.get(RUNTIME_PROFILES_YAML_ENV, "").strip())
        protected = bool(os.environ.get(INSTALL_DIR_ENV, "").strip())
        if not explicit and not protected:
            return cls.from_config(config)
        registry = cls.from_yaml(_policy_text(policy_path), backend_registry=load_backend_registry())
        registry._validate_compatibility_projection(config)
        return registry

    def _validate_compatibility_projection(self, config: Config) -> None:
        """Fail closed while users.yaml still provisions existing OS identities.

        Backend selection is owned by this registry. While users.yaml still
        provisions the corresponding OS account, models, workspaces, and
        service grants for migrated profiles, the duplicated backend/provider/
        OS-user/model/timeout fields must agree.
        """
        for runtime_config_id, user in config.user_configs.items():
            profile = self.for_config_id(runtime_config_id)
            backend, provider = get_user_backend_and_provider(user, config)
            expected_model = _compatibility_model(user, config, backend=backend, provider=provider)
            expected_timeout = user.timeout if user.timeout is not None else config.default_timeout
            if (
                profile.backend,
                profile.provider,
                profile.os_user,
                profile.model,
                profile.timeout_seconds,
            ) != (backend, provider, user.os_user, expected_model, expected_timeout):
                raise WorkshopRuntimeProfileError(
                    f"Runtime profile {profile.profile_id} conflicts with the migrated users.yaml execution policy"
                )

    @property
    def profiles(self) -> tuple[ProtectedRuntimeProfile, ...]:
        return tuple(sorted(self._by_id.values(), key=lambda profile: profile.profile_id))

    def resolve(self, profile_id: str | RuntimeProfileId) -> ProtectedRuntimeProfile:
        try:
            normalized = profile_id if isinstance(profile_id, RuntimeProfileId) else RuntimeProfileId(profile_id)
        except (TypeError, ValueError) as exc:
            raise WorkshopRuntimeProfileError("Runtime profile ID is invalid") from exc
        profile = self._by_id.get(normalized)
        if profile is None:
            raise WorkshopRuntimeProfileError("Runtime profile is not present in protected operator policy")
        return profile

    def for_config_id(self, runtime_config_id: int) -> ProtectedRuntimeProfile:
        profile = self._by_config_id.get(runtime_config_id)
        if profile is None:
            raise WorkshopRuntimeProfileError("Configured user has no protected runtime profile")
        return profile
