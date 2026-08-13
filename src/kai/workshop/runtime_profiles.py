"""Protected, transport-neutral Workshop runtime-profile registry."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from kai.config import Config, get_user_backend_and_provider
from kai.workshop.domain import RuntimeProfileId

_PROFILE_DERIVATION_DOMAIN = b"kai-workshop-runtime-profile:v1\0"


class WorkshopRuntimeProfileError(RuntimeError):
    """A canonical runtime profile is missing or conflicts with policy."""


@dataclass(frozen=True, slots=True)
class ProtectedRuntimeProfile:
    """One operator-configured execution identity behind an opaque ID.

    ``runtime_config_id`` is deliberately private compatibility state. The
    current host runtime still stores settings, files, memory, and subprocess
    instances under the configured-user integer key. Canonical Workshop
    assignments and callers see only ``profile_id``.
    """

    profile_id: RuntimeProfileId
    runtime_config_id: int
    display_name: str
    os_user: str | None
    backend: str
    provider: str


def runtime_profile_id_for_config_id(runtime_config_id: int) -> RuntimeProfileId:
    """Derive a stable opaque profile ID from protected configured-user state."""
    if isinstance(runtime_config_id, bool) or not isinstance(runtime_config_id, int) or runtime_config_id <= 0:
        raise WorkshopRuntimeProfileError("Runtime configuration ID must be a positive integer")
    digest = hashlib.sha256(_PROFILE_DERIVATION_DOMAIN + str(runtime_config_id).encode("ascii")).hexdigest()[:32]
    return RuntimeProfileId(f"rtp_{digest}")


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
                raise WorkshopRuntimeProfileError("Duplicate runtime configuration ID")
            by_id[profile.profile_id] = profile
            by_config_id[profile.runtime_config_id] = profile
        if not by_id:
            raise WorkshopRuntimeProfileError("At least one protected runtime profile is required")
        self._by_id = by_id
        self._by_config_id = by_config_id

    @classmethod
    def from_config(cls, config: Config) -> WorkshopRuntimeProfileRegistry:
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
                )
            )
        return cls(tuple(profiles))

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
