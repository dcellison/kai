"""Protected, transport-neutral Workshop runtime-profile policy."""

from __future__ import annotations

import hashlib
import os
import pwd
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kai.backend_registry import BackendRegistryEntry, load_backend_registry
from kai.config import (
    BACKEND_PROVIDERS,
    VALID_BACKENDS,
    Config,
    ModelRole,
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
_V1_LEGACY_KEY_DERIVATION_DOMAIN = b"kai-workshop-runtime-compatibility-key:v1\0"
DEFAULT_RUNTIME_PROFILES_YAML = Path("/etc/kai/runtime-profiles.yaml")
RUNTIME_PROFILES_YAML_ENV = "KAI_RUNTIME_PROFILES_YAML"
INSTALL_DIR_ENV = "KAI_INSTALL_DIR"
_OS_USER_RE = re.compile(r"^[a-zA-Z0-9._-]+$")
RUNTIME_KEY_ARCHIVE_REMOVAL_GATE = "canonical_runtime_state_v1"
DEFAULT_MAXIMUM_TIMEOUT_SECONDS = 600


class WorkshopRuntimeProfileError(RuntimeError):
    """A canonical runtime profile is missing or conflicts with policy."""


@dataclass(frozen=True, slots=True)
class ProtectedRuntimeProfile:
    """One operator-configured execution identity behind an opaque ID.

    Live execution policy contains no transport-shaped identity.  Integer keys
    retained solely for one-time installed-state migration live in the
    registry's explicit legacy archive, never on this object.
    """

    profile_id: RuntimeProfileId
    display_name: str
    os_user: str | None
    backend: str
    provider: str
    model: str
    timeout_seconds: int
    allowed_services: tuple[str, ...]
    home_workspace: Path | None
    workspace_base: Path | None
    allowed_workspaces: tuple[Path, ...]
    maximum_timeout_seconds: int = 0
    allowed_models: tuple[str, ...] | None = None
    role_models: tuple[tuple[str, str], ...] = ()
    github_repos: tuple[str, ...] = ()
    pr_review: bool | None = None
    issue_triage: bool | None = None
    allowed_triage_projects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        maximum_timeout = self.maximum_timeout_seconds
        if maximum_timeout == 0:
            maximum_timeout = max(DEFAULT_MAXIMUM_TIMEOUT_SECONDS, self.timeout_seconds)
        if isinstance(maximum_timeout, bool) or maximum_timeout < self.timeout_seconds:
            raise WorkshopRuntimeProfileError(
                "maximum_timeout_seconds must be an integer greater than or equal to timeout_seconds"
            )
        object.__setattr__(self, "maximum_timeout_seconds", maximum_timeout)


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


def _service_scopes(value: object, *, profile_id: RuntimeProfileId) -> tuple[str, ...]:
    """Validate one explicit ordered service-scope list."""
    if value is None:
        raise WorkshopRuntimeProfileError(f"Runtime profile {profile_id}: allowed_services is required")
    if not isinstance(value, list):
        raise WorkshopRuntimeProfileError(f"Runtime profile {profile_id}: allowed_services must be a list")
    checked: list[str] = []
    for raw_service in value:
        if not isinstance(raw_service, str):
            raise WorkshopRuntimeProfileError(
                f"Runtime profile {profile_id}: allowed_services entries must be service names"
            )
        service = raw_service.strip()
        if not service or service == "*" or "/" in service:
            raise WorkshopRuntimeProfileError(
                f"Runtime profile {profile_id}: allowed service {raw_service!r} is invalid"
            )
        if service in checked:
            raise WorkshopRuntimeProfileError(
                f"Runtime profile {profile_id}: allowed service {service!r} is duplicated"
            )
        checked.append(service)
    return tuple(checked)


def _workspace_directory(
    value: object,
    *,
    field: str,
    profile_id: RuntimeProfileId,
) -> Path | None:
    """Validate one optional protected workspace path.

    Availability is deliberately a use-time concern. Retaining an absolute
    path while its volume is unmounted keeps access restrictive and lets the
    runtime recover when the path becomes available again.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise WorkshopRuntimeProfileError(f"Runtime profile {profile_id}: {field} must be a directory path or null")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        raise WorkshopRuntimeProfileError(f"Runtime profile {profile_id}: {field} must be an absolute path")
    return path.resolve()


def _workspace_directories(value: object, *, profile_id: RuntimeProfileId) -> tuple[Path, ...]:
    """Validate one explicit unique protected workspace allowlist."""
    if not isinstance(value, list):
        raise WorkshopRuntimeProfileError(f"Runtime profile {profile_id}: allowed_workspaces must be a list")
    checked: list[Path] = []
    for raw_path in value:
        path = _workspace_directory(
            raw_path,
            field="allowed_workspaces entry",
            profile_id=profile_id,
        )
        assert path is not None
        if path in checked:
            raise WorkshopRuntimeProfileError(f"Runtime profile {profile_id}: allowed workspace {path} is duplicated")
        checked.append(path)
    return tuple(checked)


def runtime_profile_id_for_config_id(runtime_config_id: int) -> RuntimeProfileId:
    """Derive the preserved profile ID for one migrated configured user."""
    if isinstance(runtime_config_id, bool) or not isinstance(runtime_config_id, int) or runtime_config_id <= 0:
        raise WorkshopRuntimeProfileError("Runtime configuration ID must be a positive integer")
    digest = hashlib.sha256(_PROFILE_DERIVATION_DOMAIN + str(runtime_config_id).encode("ascii")).hexdigest()[:32]
    return RuntimeProfileId(f"rtp_{digest}")


def legacy_runtime_key_for_v1_profile_id(profile_id: RuntimeProfileId) -> int:
    """Derive the state key used by version-1 policies that omitted one.

    This exists only so the installer can preserve access to already-written
    state while upgrading that historical policy shape into the explicit
    legacy archive.  Live version-2 policy never derives or consumes it.
    """
    digest = hashlib.sha256(_V1_LEGACY_KEY_DERIVATION_DOMAIN + str(profile_id).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1) or 1


def runtime_profiles_path() -> Path:
    """Return the configured protected runtime-policy path."""
    override = os.environ.get(RUNTIME_PROFILES_YAML_ENV, "").strip()
    return Path(override) if override else DEFAULT_RUNTIME_PROFILES_YAML


def _positive_legacy_key(value: object, *, profile_id: RuntimeProfileId) -> int:
    if value is None:
        return legacy_runtime_key_for_v1_profile_id(profile_id)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WorkshopRuntimeProfileError(
            f"Runtime profile {profile_id}: archived runtime key must be a positive integer"
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

    def __init__(
        self,
        profiles: tuple[ProtectedRuntimeProfile, ...],
        *,
        legacy_runtime_keys: Mapping[RuntimeProfileId, int] | None = None,
        legacy_archive_removal_gate: str | None = None,
    ) -> None:
        by_id: dict[RuntimeProfileId, ProtectedRuntimeProfile] = {}
        for profile in profiles:
            if not isinstance(profile, ProtectedRuntimeProfile):
                raise TypeError("profiles must contain ProtectedRuntimeProfile values")
            if profile.profile_id in by_id:
                raise WorkshopRuntimeProfileError("Duplicate runtime profile ID")
            by_id[profile.profile_id] = profile
        if not by_id:
            raise WorkshopRuntimeProfileError("At least one protected runtime profile is required")
        archived_by_profile: dict[RuntimeProfileId, int] = {}
        archived_by_key: dict[int, RuntimeProfileId] = {}
        for profile_id, legacy_key in (legacy_runtime_keys or {}).items():
            if profile_id not in by_id:
                raise WorkshopRuntimeProfileError("Legacy runtime archive references an unknown profile")
            checked = _positive_legacy_key(legacy_key, profile_id=profile_id)
            if checked in archived_by_key:
                raise WorkshopRuntimeProfileError("Legacy runtime archive contains a duplicate integer key")
            archived_by_profile[profile_id] = checked
            archived_by_key[checked] = profile_id
        self._by_id = by_id
        self._legacy_runtime_keys = archived_by_profile
        self._profiles_by_legacy_key = archived_by_key
        self._legacy_archive_removal_gate = legacy_archive_removal_gate

    @classmethod
    def from_config(cls, config: Config) -> WorkshopRuntimeProfileRegistry:
        """Build the development-only compatibility projection.

        Protected startup uses :meth:`load`; this constructor remains useful
        for direct/dev runs and focused tests with no installed policy file.
        """
        profiles: list[ProtectedRuntimeProfile] = []
        legacy_runtime_keys: dict[RuntimeProfileId, int] = {}
        for runtime_config_id, user in sorted(config.user_configs.items()):
            if user.telegram_id != runtime_config_id:
                raise WorkshopRuntimeProfileError("Configured-user key does not match its protected user record")
            backend, provider = get_user_backend_and_provider(user, config)
            profile_id = runtime_profile_id_for_config_id(runtime_config_id)
            timeout_seconds = user.timeout if user.timeout is not None else config.default_timeout
            profiles.append(
                ProtectedRuntimeProfile(
                    profile_id=profile_id,
                    display_name=user.name,
                    os_user=user.os_user,
                    backend=backend,
                    provider=provider,
                    model=_compatibility_model(user, config, backend=backend, provider=provider),
                    timeout_seconds=timeout_seconds,
                    allowed_services=tuple(user.allowed_services),
                    home_workspace=user.home_workspace,
                    workspace_base=user.workspace_base,
                    allowed_workspaces=tuple(user.allowed_workspaces),
                    maximum_timeout_seconds=max(
                        DEFAULT_MAXIMUM_TIMEOUT_SECONDS,
                        timeout_seconds,
                    ),
                    role_models=tuple(sorted((user.models or {}).items())),
                    github_repos=tuple(user.github_repos),
                    pr_review=user.pr_review,
                    issue_triage=user.issue_triage,
                    allowed_triage_projects=tuple(user.allowed_triage_projects),
                )
            )
            legacy_runtime_keys[profile_id] = runtime_config_id
        return cls(tuple(profiles), legacy_runtime_keys=legacy_runtime_keys)

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
        version = document.get("version")
        if version not in {1, 2}:
            raise WorkshopRuntimeProfileError("Runtime policy version must be 1 or 2")
        raw_profiles = document.get("runtime_profiles")
        if not isinstance(raw_profiles, dict) or not raw_profiles:
            raise WorkshopRuntimeProfileError("Runtime policy must contain a non-empty runtime_profiles mapping")

        installed = backend_registry
        profiles: list[ProtectedRuntimeProfile] = []
        legacy_runtime_keys: dict[RuntimeProfileId, int] = {}
        archive_removal_gate: str | None = None
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
            raw_maximum_timeout = raw_profile.get(
                "maximum_timeout_seconds",
                max(DEFAULT_MAXIMUM_TIMEOUT_SECONDS, raw_timeout),
            )
            if (
                isinstance(raw_maximum_timeout, bool)
                or not isinstance(raw_maximum_timeout, int)
                or raw_maximum_timeout < raw_timeout
            ):
                raise WorkshopRuntimeProfileError(
                    f"Runtime profile {profile_id}: maximum_timeout_seconds must be an integer "
                    "greater than or equal to timeout_seconds"
                )
            raw_os_user = raw_profile.get("os_user")
            os_user = None if raw_os_user is None else str(raw_os_user).strip() or None
            if os_user is not None and _OS_USER_RE.fullmatch(os_user) is None:
                raise WorkshopRuntimeProfileError(f"Runtime profile {profile_id}: os_user is invalid")
            raw_role_models = raw_profile.get("models", {})
            if not isinstance(raw_role_models, dict):
                raise WorkshopRuntimeProfileError(f"Runtime profile {profile_id}: models must be a mapping")
            role_models: list[tuple[str, str]] = []
            for raw_role, raw_role_model in raw_role_models.items():
                role = str(raw_role).strip().lower()
                if role not in {item.value for item in ModelRole}:
                    raise WorkshopRuntimeProfileError(
                        f"Runtime profile {profile_id}: unsupported model role {raw_role!r}"
                    )
                role_model = canonicalize_model_for_backend(str(raw_role_model).strip(), backend)
                if not validate_model_for_backend_policy(
                    role_model,
                    backend,
                    provider,
                    allowed_models=allowed_models,
                ):
                    raise WorkshopRuntimeProfileError(
                        f"Runtime profile {profile_id}: role model {role_model!r} is invalid for {backend}/{provider}"
                    )
                role_models.append((role, role_model))
            raw_triage_projects = raw_profile.get("allowed_triage_projects", [])
            if not isinstance(raw_triage_projects, list):
                raise WorkshopRuntimeProfileError(
                    f"Runtime profile {profile_id}: allowed_triage_projects must be a list"
                )
            triage_projects: list[str] = []
            for raw_project in raw_triage_projects:
                project = str(raw_project).strip()
                if not project or project == "*":
                    raise WorkshopRuntimeProfileError(
                        f"Runtime profile {profile_id}: allowed triage project {raw_project!r} is invalid"
                    )
                if project not in triage_projects:
                    triage_projects.append(project)
            raw_github_repos = raw_profile.get("github_repos", [])
            if not isinstance(raw_github_repos, list) or any(
                not isinstance(item, str) or not item.strip() for item in raw_github_repos
            ):
                raise WorkshopRuntimeProfileError(
                    f"Runtime profile {profile_id}: github_repos must be a list of repositories"
                )
            raw_pr_review = raw_profile.get("pr_review")
            raw_issue_triage = raw_profile.get("issue_triage")
            if raw_pr_review is not None and not isinstance(raw_pr_review, bool):
                raise WorkshopRuntimeProfileError(
                    f"Runtime profile {profile_id}: pr_review must be true, false, or null"
                )
            if raw_issue_triage is not None and not isinstance(raw_issue_triage, bool):
                raise WorkshopRuntimeProfileError(
                    f"Runtime profile {profile_id}: issue_triage must be true, false, or null"
                )
            profiles.append(
                ProtectedRuntimeProfile(
                    profile_id=profile_id,
                    display_name=display_name,
                    os_user=os_user,
                    backend=backend,
                    provider=provider,
                    model=model,
                    timeout_seconds=raw_timeout,
                    allowed_services=_service_scopes(
                        raw_profile.get("allowed_services"),
                        profile_id=profile_id,
                    ),
                    home_workspace=_workspace_directory(
                        raw_profile.get("home_workspace"),
                        field="home_workspace",
                        profile_id=profile_id,
                    ),
                    workspace_base=_workspace_directory(
                        raw_profile.get("workspace_base"),
                        field="workspace_base",
                        profile_id=profile_id,
                    ),
                    allowed_workspaces=_workspace_directories(
                        raw_profile.get("allowed_workspaces"),
                        profile_id=profile_id,
                    ),
                    maximum_timeout_seconds=raw_maximum_timeout,
                    allowed_models=allowed_models,
                    role_models=tuple(sorted(role_models)),
                    github_repos=tuple(sorted({item.strip().lower() for item in raw_github_repos})),
                    pr_review=raw_pr_review,
                    issue_triage=raw_issue_triage,
                    allowed_triage_projects=tuple(triage_projects),
                )
            )
            if version == 1:
                legacy_runtime_keys[profile_id] = _positive_legacy_key(
                    raw_profile.get("compatibility_runtime_config_id"),
                    profile_id=profile_id,
                )

        if version == 2:
            raw_archive = document.get("legacy_runtime_archive")
            if raw_archive is not None:
                if not isinstance(raw_archive, dict):
                    raise WorkshopRuntimeProfileError("legacy_runtime_archive must be a mapping")
                if raw_archive.get("version") != 1:
                    raise WorkshopRuntimeProfileError("legacy_runtime_archive version must be 1")
                archive_removal_gate = str(raw_archive.get("removal_gate") or "").strip()
                if archive_removal_gate != RUNTIME_KEY_ARCHIVE_REMOVAL_GATE:
                    raise WorkshopRuntimeProfileError(
                        "legacy_runtime_archive removal_gate must be canonical_runtime_state_v1"
                    )
                raw_keys = raw_archive.get("runtime_keys")
                if not isinstance(raw_keys, dict):
                    raise WorkshopRuntimeProfileError("legacy_runtime_archive.runtime_keys must be a mapping")
                for raw_profile_id, raw_key in raw_keys.items():
                    try:
                        archived_profile_id = RuntimeProfileId(str(raw_profile_id))
                    except (TypeError, ValueError) as exc:
                        raise WorkshopRuntimeProfileError(
                            f"Invalid archived runtime profile ID {raw_profile_id!r}"
                        ) from exc
                    legacy_runtime_keys[archived_profile_id] = _positive_legacy_key(
                        raw_key,
                        profile_id=archived_profile_id,
                    )
        return cls(
            tuple(profiles),
            legacy_runtime_keys=legacy_runtime_keys,
            legacy_archive_removal_gate=archive_removal_gate,
        )

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
        if protected:
            service_uid = os.geteuid()
            try:
                service_user = pwd.getpwuid(service_uid).pw_name
            except KeyError:
                raise WorkshopRuntimeProfileError(
                    f"Protected installation could not resolve the Kai service account for effective uid {service_uid}"
                ) from None
            registry.validate_protected_os_users(
                service_user,
                account_uid=lambda name: pwd.getpwnam(name).pw_uid,
                service_uid=service_uid,
            )
        return registry

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

    @property
    def legacy_runtime_archive(self) -> tuple[tuple[RuntimeProfileId, int], ...]:
        """Return explicit rollback/migration keys, never live runtime policy."""
        return tuple(sorted(self._legacy_runtime_keys.items(), key=lambda item: item[0]))

    @property
    def legacy_archive_removal_gate(self) -> str | None:
        """Return the explicit operator gate protecting archive removal."""
        return self._legacy_archive_removal_gate

    def legacy_runtime_key(self, profile_id: str | RuntimeProfileId) -> int | None:
        """Return one archived pre-canonical key for migration-only callers."""
        profile = self.resolve(profile_id)
        return self._legacy_runtime_keys.get(profile.profile_id)

    def profile_for_legacy_runtime_key(self, legacy_key: int) -> ProtectedRuntimeProfile:
        """Translate an adapter/migration key at the canonical boundary."""
        profile_id = self._profiles_by_legacy_key.get(legacy_key)
        if profile_id is None:
            raise WorkshopRuntimeProfileError("Legacy runtime key has no protected runtime profile")
        return self._by_id[profile_id]

    def validate_protected_os_users(
        self,
        service_user: str,
        *,
        account_uid: Callable[[str], int],
        service_uid: int,
    ) -> None:
        """Keep protected backend processes off the privileged service account."""
        errors: list[str] = []
        normalized_service_user = service_user.strip()
        for profile in self.profiles:
            os_user = profile.os_user.strip() if profile.os_user else ""
            label = f"runtime profile {profile.profile_id} ({profile.display_name!r})"
            if not os_user:
                errors.append(f"{label} is missing required os_user")
                continue
            if os_user == normalized_service_user:
                errors.append(f"{label} maps to service account {normalized_service_user!r}")
                continue
            try:
                uid = account_uid(os_user)
            except KeyError:
                errors.append(f"{label} maps to nonexistent OS account {os_user!r}")
                continue
            if uid == service_uid:
                errors.append(f"{label} maps to OS account {os_user!r}, which resolves to service uid {service_uid}")
        if errors:
            detail = "\n  - ".join(errors)
            raise WorkshopRuntimeProfileError(
                "Protected runtime profiles must use existing non-service OS accounts:\n"
                f"  - {detail}\n"
                f"Assign every profile an OS account distinct from service account {normalized_service_user!r}."
            )
