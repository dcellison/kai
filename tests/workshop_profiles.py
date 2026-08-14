"""Shared protected runtime-profile fixtures for Workshop tests."""

from __future__ import annotations

from kai.workshop.runtime_profiles import (
    ProtectedRuntimeProfile,
    WorkshopRuntimeProfileRegistry,
    runtime_profile_id_for_config_id,
)


def profile_id(runtime_config_id: int):
    return runtime_profile_id_for_config_id(runtime_config_id)


def profile_registry(*runtime_config_ids: int) -> WorkshopRuntimeProfileRegistry:
    return WorkshopRuntimeProfileRegistry(
        tuple(
            ProtectedRuntimeProfile(
                profile_id=profile_id(runtime_config_id),
                runtime_config_id=runtime_config_id,
                display_name=f"User {runtime_config_id}",
                os_user=None,
                backend="codex",
                provider="openai",
                model="gpt-5.6-sol",
                timeout_seconds=120,
                allowed_services=(),
            )
            for runtime_config_id in runtime_config_ids
        )
    )
