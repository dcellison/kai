"""Transport-neutral ownership metadata for adapter capability boundaries.

The capability registry names stable operation families. This module binds
those descriptive service names to the Python modules that own canonical
authorization and state transitions, and gives adapters a checked way to name
their entry points. It deliberately imports no adapter framework.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class CanonicalServiceBoundary:
    """One named canonical service and the modules allowed to implement it."""

    modules: tuple[str, ...]


CANONICAL_SERVICE_BOUNDARIES: Mapping[str, CanonicalServiceBoundary] = MappingProxyType(
    {
        "activity_service": CanonicalServiceBoundary(
            (
                "kai.workshop.channel_unread",
                "kai.workshop.human_notifications",
                "kai.workshop.thread_unread",
            )
        ),
        "agent_definitions": CanonicalServiceBoundary(
            ("kai.workshop.agent_definitions", "kai.workshop.agent_lifecycle")
        ),
        "appearance_preferences": CanonicalServiceBoundary(("kai.workshop.appearance_preferences",)),
        "artifact_service": CanonicalServiceBoundary(("kai.workshop.artifacts",)),
        "canonical_scheduler": CanonicalServiceBoundary(("kai.workshop.scheduled_jobs", "kai.workshop.scheduler")),
        "capability_registry": CanonicalServiceBoundary(("kai.capability_registry",)),
        "channel_lifecycle": CanonicalServiceBoundary(("kai.workshop.channel_lifecycle",)),
        "channel_membership": CanonicalServiceBoundary(("kai.workshop.channel_lifecycle",)),
        "client_preferences": CanonicalServiceBoundary(("kai.workshop.client_preferences",)),
        "context_manifests": CanonicalServiceBoundary(("kai.workshop.context_manifests",)),
        "conversation_commands": CanonicalServiceBoundary(("kai.workshop.conversation_commands",)),
        "direct_message_lifecycle": CanonicalServiceBoundary(
            ("kai.workshop.direct_message_archives", "kai.workshop.human_direct_messages")
        ),
        "enrollment": CanonicalServiceBoundary(("kai.workshop.initial_provisioning",)),
        "github_settings": CanonicalServiceBoundary(("kai.workshop.github_settings",)),
        "human_profile": CanonicalServiceBoundary(("kai.workshop.human_avatars", "kai.workshop.human_profiles")),
        "memory_queries": CanonicalServiceBoundary(("kai.workshop.memory_queries",)),
        "message_reactions": CanonicalServiceBoundary(("kai.workshop.message_reactions",)),
        "notification_preferences": CanonicalServiceBoundary(
            ("kai.workshop.channel_notification_policy", "kai.workshop.notification_preferences")
        ),
        "preference_documents": CanonicalServiceBoundary(("kai.workshop.preferences",)),
        "principal_policies": CanonicalServiceBoundary(("kai.workshop.principal_policies",)),
        "private_text_execution": CanonicalServiceBoundary(("kai.workshop.private_text_execution",)),
        "review_jobs": CanonicalServiceBoundary(("kai.workshop.review_jobs",)),
        "runtime_lane_status": CanonicalServiceBoundary(("kai.workshop.runtime_lane_status",)),
        "routing_policy": CanonicalServiceBoundary(("kai.workshop.routing_policy",)),
        "settings_workspaces": CanonicalServiceBoundary(("kai.workshop.settings_workspaces",)),
        "standing_participation": CanonicalServiceBoundary(
            ("kai.workshop.standing_observation", "kai.workshop.standing_participation")
        ),
        "thread_service": CanonicalServiceBoundary(("kai.workshop.thread_unread", "kai.workshop.timeline")),
        "webhook_diagnostics": CanonicalServiceBoundary(("kai.workshop.diagnostics",)),
    }
)
