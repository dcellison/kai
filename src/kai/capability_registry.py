"""Canonical inventory of user-facing Kai capabilities and adapter dispositions.

The registry is descriptive, not authoritative.  Runtime services remain the
only source of authorization and mutation policy; this module gives adapters a
shared vocabulary for discovery, parity checks, and safe availability hints.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class AdapterId(StrEnum):
    """User-facing adapters covered by the parity contract."""

    TELEGRAM = "telegram"
    WORKSHOP = "workshop"


class AdapterDisposition(StrEnum):
    """How one adapter presents, or deliberately does not present, a capability."""

    NATIVE_SURFACE = "native_surface"
    ACTION_PALETTE = "action_palette"
    ADMINISTRATOR_ONLY = "administrator_only"
    PRESENTATION_ALIAS = "presentation_alias"
    COMPATIBILITY_ONLY = "compatibility_only"
    UNSUPPORTED_BY_DESIGN = "unsupported_by_design"


class AuthorityScope(StrEnum):
    """Canonical authority boundary required by an operation family."""

    PUBLIC = "public"
    PRINCIPAL = "principal"
    CONVERSATION = "conversation"
    PRINCIPAL_AGENT = "principal_agent"
    AGENT_OWNER = "agent_owner"
    CHANNEL_MEMBER = "channel_member"
    CHANNEL_OWNER = "channel_owner"
    ADMINISTRATOR = "administrator"


class ConfirmationPolicy(StrEnum):
    """Whether a native presentation must confirm an operation."""

    NONE = "none"
    CONTEXT_DEPENDENT = "context_dependent"
    REQUIRED = "required"


class IdempotencyPolicy(StrEnum):
    """Idempotency requirement for mutation requests."""

    NONE = "none"
    CONTEXT_DEPENDENT = "context_dependent"
    REQUIRED = "required"


class ContinuityEffect(StrEnum):
    """Effect a capability can have on live provider continuity."""

    NONE = "none"
    CANCEL_ACTIVE_RUN = "cancel_active_run"
    RESET_PROVIDER_SESSION = "reset_provider_session"
    MAY_RESTART_RUNTIME = "may_restart_runtime"


class ImplementationState(StrEnum):
    """Current relationship between adapters and the named canonical service."""

    CANONICAL = "canonical"
    MIXED_COMPATIBILITY = "mixed_compatibility"
    TELEGRAM_COMPATIBILITY = "telegram_compatibility"
    PLANNED = "planned"


class CompatibilityRemovalGate(StrEnum):
    """Named condition that permits a temporary adapter compatibility path."""

    SINGLE_USER_DEPLOYMENT_RETIRED = "single_user_deployment_retired"
    TELEGRAM_MEMORY_CONVERGED = "telegram_memory_converged"


class ContextRequirement(StrEnum):
    """Non-secret context predicates used only for discovery hints."""

    AUTHENTICATED = "authenticated"
    CONVERSATION = "conversation"
    AGENT = "agent"
    AGENT_OWNER = "agent_owner"
    WORKSPACE = "workspace"
    CHANNEL_OWNER = "channel_owner"
    ADMINISTRATOR = "administrator"


class CapabilityInputShape(StrEnum):
    """Typed interaction shape a native adapter should present."""

    NONE = "none"
    OPTIONAL_AGENT = "optional_agent"
    MODEL_SELECTION = "model_selection"
    BACKEND_SELECTION = "backend_selection"
    RUNTIME_SETTINGS = "runtime_settings"
    WORKSPACE_OPERATION = "workspace_operation"
    MEMORY_PROJECT_OPERATION = "memory_project_operation"
    GITHUB_SETTINGS = "github_settings"
    NOTIFICATION_SETTINGS = "notification_settings"
    PULL_REQUEST_REFERENCE = "pull_request_reference"
    MEMORY_OPERATION = "memory_operation"
    PREFERENCE_OPERATION = "preference_operation"
    VOICE_PREFERENCE = "voice_preference"
    SCHEDULED_JOB_OPERATION = "scheduled_job_operation"
    MESSAGE = "message"
    CHANNEL_OPERATION = "channel_operation"
    DIRECT_MESSAGE_PEER = "direct_message_peer"
    AGENT_DEFINITION = "agent_definition"
    CHANNEL_PARTICIPANT = "channel_participant"
    STANDING_PARTICIPATION_POLICY = "standing_participation_policy"
    THREAD_OPERATION = "thread_operation"
    REACTION = "reaction"
    ACTIVITY_STATE = "activity_state"
    PROFILE = "profile"
    APPEARANCE = "appearance"
    ARTIFACT_REFERENCE = "artifact_reference"


class WorkshopSurface(StrEnum):
    """Stable identifiers for current or deliberately planned Workshop surfaces."""

    ENROLLMENT = "enrollment"
    CONVERSATION = "conversation"
    CONVERSATION_CONTEXT = "conversation_context"
    AGENT_SETTINGS = "agent_settings"
    WORKSPACES = "workspaces"
    MEMORY = "memory"
    SCHEDULED_JOBS = "scheduled_jobs"
    GITHUB_SETTINGS = "github_settings"
    NOTIFICATION_SETTINGS = "notification_settings"
    PERSONAL_SETTINGS = "personal_settings"
    CLIENT_SETTINGS = "client_settings"
    ADMINISTRATION = "administration"
    ACTION_PALETTE = "action_palette"
    CHANNELS = "channels"
    DIRECT_MESSAGES = "direct_messages"
    AGENTS = "agents"
    THREADS = "threads"
    ACTIVITY = "activity"
    PROFILE = "profile"
    APPEARANCE = "appearance"
    RUN_INSPECTOR = "run_inspector"
    ARTIFACTS = "artifacts"


@dataclass(frozen=True, slots=True)
class TelegramHelpEntry:
    """One adapter-native help line backed by a registered operation."""

    usage: str
    description: str
    commands: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CapabilityContext:
    """Redacted caller context used to calculate discovery availability."""

    authenticated: bool = False
    conversation: bool = False
    agent: bool = False
    agent_owner: bool = False
    workspace: bool = False
    channel_owner: bool = False
    administrator: bool = False

    def satisfies(self, requirement: ContextRequirement) -> bool:
        return {
            ContextRequirement.AUTHENTICATED: self.authenticated,
            ContextRequirement.CONVERSATION: self.conversation,
            ContextRequirement.AGENT: self.agent,
            ContextRequirement.AGENT_OWNER: self.agent_owner,
            ContextRequirement.WORKSPACE: self.workspace,
            ContextRequirement.CHANNEL_OWNER: self.channel_owner,
            ContextRequirement.ADMINISTRATOR: self.administrator,
        }[requirement]


@dataclass(frozen=True, slots=True)
class AdapterPresentation:
    """Adapter-specific placement without adapter-specific business logic."""

    disposition: AdapterDisposition
    surface: str | None = None
    primary_command: str | None = None
    aliases: tuple[str, ...] = ()
    palette_entry: bool = False
    input_shape: CapabilityInputShape = CapabilityInputShape.NONE
    implemented: bool = True
    help_group: int | None = None
    help_entries: tuple[TelegramHelpEntry, ...] = ()
    help_discoverable: bool = False

    @property
    def commands(self) -> tuple[str, ...]:
        if self.primary_command is None:
            return ()
        return (self.primary_command, *self.aliases)


@dataclass(frozen=True, slots=True)
class CapabilityDefinition:
    """One transport-neutral user-facing operation family."""

    operation_id: str
    label: str
    description: str
    canonical_service: str
    implementation_state: ImplementationState
    authority_scope: AuthorityScope
    mutates_state: bool
    confirmation: ConfirmationPolicy
    idempotency: IdempotencyPolicy
    revision_check: bool
    continuity_effect: ContinuityEffect
    compatibility_removal_gate: CompatibilityRemovalGate | None
    requirements: frozenset[ContextRequirement]
    presentations: Mapping[AdapterId, AdapterPresentation]


@dataclass(frozen=True, slots=True)
class CapabilityAvailability:
    """Safe discovery result with no principal, resource, path, or secret data."""

    operation_id: str
    label: str
    description: str
    disposition: AdapterDisposition
    surface: str | None
    palette_entry: bool
    authority_scope: AuthorityScope
    input_shape: CapabilityInputShape
    mutates_state: bool
    confirmation: ConfirmationPolicy
    available: bool
    unavailable_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "operation_id": self.operation_id,
            "label": self.label,
            "description": self.description,
            "disposition": self.disposition.value,
            "surface": self.surface,
            "palette_entry": self.palette_entry,
            "scope": self.authority_scope.value,
            "input_shape": self.input_shape.value,
            "mutates_state": self.mutates_state,
            "confirmation": self.confirmation.value,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
        }


def _presentations(
    telegram: AdapterPresentation,
    workshop: AdapterPresentation,
) -> Mapping[AdapterId, AdapterPresentation]:
    return MappingProxyType({AdapterId.TELEGRAM: telegram, AdapterId.WORKSHOP: workshop})


def _telegram(
    command: str | None = None,
    *,
    aliases: tuple[str, ...] = (),
    disposition: AdapterDisposition = AdapterDisposition.NATIVE_SURFACE,
    input_shape: CapabilityInputShape = CapabilityInputShape.NONE,
    help_group: int | None = None,
    help_entries: tuple[TelegramHelpEntry, ...] = (),
    help_discoverable: bool | None = None,
) -> AdapterPresentation:
    return AdapterPresentation(
        disposition,
        primary_command=command,
        aliases=aliases,
        input_shape=input_shape,
        implemented=disposition != AdapterDisposition.UNSUPPORTED_BY_DESIGN,
        help_group=help_group,
        help_entries=help_entries,
        help_discoverable=(command is not None if help_discoverable is None else help_discoverable),
    )


def _workshop(
    surface: WorkshopSurface | None,
    *,
    disposition: AdapterDisposition = AdapterDisposition.NATIVE_SURFACE,
    palette: bool = False,
    input_shape: CapabilityInputShape = CapabilityInputShape.NONE,
    implemented: bool | None = None,
) -> AdapterPresentation:
    return AdapterPresentation(
        disposition,
        surface=None if surface is None else surface.value,
        palette_entry=palette,
        input_shape=input_shape,
        implemented=(disposition != AdapterDisposition.UNSUPPORTED_BY_DESIGN if implemented is None else implemented),
        help_discoverable=False,
    )


def _help(usage: str, description: str, *commands: str) -> TelegramHelpEntry:
    return TelegramHelpEntry(usage, description, commands)


def _capability(
    operation_id: str,
    label: str,
    description: str,
    canonical_service: str,
    authority_scope: AuthorityScope,
    *,
    telegram: AdapterPresentation,
    workshop: AdapterPresentation,
    implementation_state: ImplementationState = ImplementationState.CANONICAL,
    mutates_state: bool = False,
    confirmation: ConfirmationPolicy = ConfirmationPolicy.NONE,
    idempotency: IdempotencyPolicy = IdempotencyPolicy.NONE,
    revision_check: bool = False,
    continuity_effect: ContinuityEffect = ContinuityEffect.NONE,
    compatibility_removal_gate: CompatibilityRemovalGate | None = None,
    requirements: tuple[ContextRequirement, ...] = (),
) -> CapabilityDefinition:
    return CapabilityDefinition(
        operation_id=operation_id,
        label=label,
        description=description,
        canonical_service=canonical_service,
        implementation_state=implementation_state,
        authority_scope=authority_scope,
        mutates_state=mutates_state,
        confirmation=confirmation,
        idempotency=idempotency,
        revision_check=revision_check,
        continuity_effect=continuity_effect,
        compatibility_removal_gate=compatibility_removal_gate,
        requirements=frozenset(requirements),
        presentations=_presentations(telegram, workshop),
    )


_AUTHENTICATED = (ContextRequirement.AUTHENTICATED,)
_CONVERSATION = (*_AUTHENTICATED, ContextRequirement.CONVERSATION)
_AGENT_CONVERSATION = (*_CONVERSATION, ContextRequirement.AGENT)


CAPABILITY_REGISTRY: tuple[CapabilityDefinition, ...] = (
    _capability(
        "onboarding.start",
        "Start Kai",
        "Begin or recover access to Kai.",
        "enrollment",
        AuthorityScope.PUBLIC,
        telegram=_telegram("start", help_discoverable=False),
        workshop=_workshop(WorkshopSurface.ENROLLMENT),
    ),
    _capability(
        "conversation.run.cancel",
        "Stop response",
        "Interrupt the active agent response in this conversation.",
        "private_text_execution",
        AuthorityScope.CONVERSATION,
        telegram=_telegram(
            "stop",
            help_group=1,
            help_entries=(_help("/stop", "Interrupt current response", "stop"),),
        ),
        workshop=_workshop(WorkshopSurface.CONVERSATION),
        mutates_state=True,
        idempotency=IdempotencyPolicy.REQUIRED,
        continuity_effect=ContinuityEffect.CANCEL_ACTIVE_RUN,
        implementation_state=ImplementationState.MIXED_COMPATIBILITY,
        compatibility_removal_gate=CompatibilityRemovalGate.SINGLE_USER_DEPLOYMENT_RETIRED,
        requirements=_AGENT_CONVERSATION,
    ),
    _capability(
        "conversation.session.reset",
        "Start fresh provider session",
        "Replace one agent's provider session without removing conversation history or memory.",
        "runtime_lane_status",
        AuthorityScope.PRINCIPAL_AGENT,
        telegram=_telegram(
            "new",
            input_shape=CapabilityInputShape.OPTIONAL_AGENT,
            help_group=1,
            help_entries=(_help("/new [@agent]", "Start a fresh provider session", "new"),),
        ),
        workshop=_workshop(
            WorkshopSurface.CONVERSATION_CONTEXT,
            palette=True,
            input_shape=CapabilityInputShape.OPTIONAL_AGENT,
        ),
        mutates_state=True,
        confirmation=ConfirmationPolicy.REQUIRED,
        idempotency=IdempotencyPolicy.REQUIRED,
        continuity_effect=ContinuityEffect.RESET_PROVIDER_SESSION,
        requirements=_AGENT_CONVERSATION,
    ),
    _capability(
        "runtime.model.manage",
        "Change model",
        "Inspect, refresh, and select models for an authorized runtime.",
        "settings_workspaces",
        AuthorityScope.PRINCIPAL_AGENT,
        telegram=_telegram(
            "model",
            aliases=("models",),
            input_shape=CapabilityInputShape.MODEL_SELECTION,
            help_group=2,
            help_entries=(
                _help("/models [refresh]", "Choose or refresh models", "models"),
                _help("/model <name>", "Switch model directly", "model"),
            ),
        ),
        workshop=_workshop(
            WorkshopSurface.AGENT_SETTINGS,
            palette=True,
            input_shape=CapabilityInputShape.MODEL_SELECTION,
        ),
        mutates_state=True,
        idempotency=IdempotencyPolicy.CONTEXT_DEPENDENT,
        continuity_effect=ContinuityEffect.MAY_RESTART_RUNTIME,
        implementation_state=ImplementationState.MIXED_COMPATIBILITY,
        compatibility_removal_gate=CompatibilityRemovalGate.SINGLE_USER_DEPLOYMENT_RETIRED,
        requirements=(*_AUTHENTICATED, ContextRequirement.AGENT),
    ),
    _capability(
        "runtime.backend.manage",
        "Change backend",
        "Inspect and select an authorized backend for an agent runtime.",
        "settings_workspaces",
        AuthorityScope.PRINCIPAL_AGENT,
        telegram=_telegram(
            "backend",
            aliases=("backends",),
            input_shape=CapabilityInputShape.BACKEND_SELECTION,
            help_group=2,
            help_entries=(
                _help("/backends", "Choose a backend", "backends"),
                _help("/backend [backend:provider]", "Show or switch backend", "backend"),
            ),
        ),
        workshop=_workshop(
            WorkshopSurface.AGENT_SETTINGS,
            palette=True,
            input_shape=CapabilityInputShape.BACKEND_SELECTION,
        ),
        mutates_state=True,
        idempotency=IdempotencyPolicy.CONTEXT_DEPENDENT,
        continuity_effect=ContinuityEffect.MAY_RESTART_RUNTIME,
        implementation_state=ImplementationState.MIXED_COMPATIBILITY,
        compatibility_removal_gate=CompatibilityRemovalGate.SINGLE_USER_DEPLOYMENT_RETIRED,
        requirements=(*_AUTHENTICATED, ContextRequirement.AGENT),
    ),
    _capability(
        "runtime.settings.manage",
        "Runtime settings",
        "Inspect or change policy-bounded runtime and workspace settings.",
        "settings_workspaces",
        AuthorityScope.PRINCIPAL_AGENT,
        telegram=_telegram(
            "settings",
            input_shape=CapabilityInputShape.RUNTIME_SETTINGS,
            help_group=3,
            help_entries=(
                _help("/settings", "Show your settings", "settings"),
                _help("/settings model <name>", "Set the default model", "settings"),
                _help("/settings timeout <n>", "Set response timeout in seconds", "settings"),
                _help("/settings reset [field]", "Clear overrides", "settings"),
            ),
        ),
        workshop=_workshop(
            WorkshopSurface.AGENT_SETTINGS,
            input_shape=CapabilityInputShape.RUNTIME_SETTINGS,
        ),
        mutates_state=True,
        idempotency=IdempotencyPolicy.CONTEXT_DEPENDENT,
        revision_check=True,
        continuity_effect=ContinuityEffect.MAY_RESTART_RUNTIME,
        implementation_state=ImplementationState.MIXED_COMPATIBILITY,
        compatibility_removal_gate=CompatibilityRemovalGate.SINGLE_USER_DEPLOYMENT_RETIRED,
        requirements=(*_AUTHENTICATED, ContextRequirement.AGENT),
    ),
    _capability(
        "runtime.status.read",
        "Show runtime status",
        "Inspect the current agent, backend, model, workspace, and continuity state.",
        "runtime_lane_status",
        AuthorityScope.PRINCIPAL_AGENT,
        telegram=_telegram(
            "stats",
            input_shape=CapabilityInputShape.OPTIONAL_AGENT,
            help_group=8,
            help_entries=(_help("/stats [@agent]", "Show canonical runtime status", "stats"),),
        ),
        workshop=_workshop(WorkshopSurface.RUN_INSPECTOR, palette=True),
        requirements=(*_AUTHENTICATED, ContextRequirement.AGENT),
    ),
    _capability(
        "workspace.catalogue.manage",
        "Open workspaces",
        "Inspect, authorize, create, select, remove, or delete eligible workspaces.",
        "settings_workspaces",
        AuthorityScope.PRINCIPAL,
        telegram=_telegram(
            "workspace",
            aliases=("ws", "workspaces"),
            input_shape=CapabilityInputShape.WORKSPACE_OPERATION,
            help_group=4,
            help_entries=(
                _help("/workspace (or /ws)", "Show current workspace", "workspace", "ws"),
                _help("/workspace <name>", "Switch by name", "workspace"),
                _help("/workspace home", "Return to default", "workspace"),
                _help("/workspace new <name>", "Create, initialize Git, and switch", "workspace"),
                _help(
                    "/workspace delete <name> confirm <name>",
                    "Permanently delete an eligible workspace",
                    "workspace",
                ),
                _help("/workspace allow <path>", "Add an allowed workspace", "workspace"),
                _help("/workspace deny <path>", "Remove an allowed workspace", "workspace"),
                _help("/workspace allowed", "List your workspaces", "workspace"),
                _help("/workspace config", "Show workspace settings", "workspace"),
                _help(
                    "/workspace config <field> <value>",
                    "Override a workspace setting",
                    "workspace",
                ),
                _help("/workspace config env KEY=VALUE", "Set an environment variable", "workspace"),
                _help("/workspace config prompt <text>", "Set system prompt", "workspace"),
                _help("/workspace config reset [field]", "Clear workspace overrides", "workspace"),
                _help("/workspaces", "Switch workspace with inline buttons", "workspaces"),
            ),
        ),
        workshop=_workshop(
            WorkshopSurface.WORKSPACES,
            palette=True,
            input_shape=CapabilityInputShape.WORKSPACE_OPERATION,
        ),
        mutates_state=True,
        confirmation=ConfirmationPolicy.CONTEXT_DEPENDENT,
        idempotency=IdempotencyPolicy.CONTEXT_DEPENDENT,
        implementation_state=ImplementationState.MIXED_COMPATIBILITY,
        compatibility_removal_gate=CompatibilityRemovalGate.SINGLE_USER_DEPLOYMENT_RETIRED,
        requirements=_AUTHENTICATED,
    ),
    _capability(
        "workspace.memory_project.manage",
        "Memory projects",
        "Inspect or change the memory-project registration for an authorized workspace.",
        "settings_workspaces",
        AuthorityScope.PRINCIPAL,
        telegram=_telegram(
            "project",
            input_shape=CapabilityInputShape.MEMORY_PROJECT_OPERATION,
            help_group=4,
            help_entries=(
                _help("/project", "List memory projects", "project"),
                _help("/project register [name]", "Register the current workspace", "project"),
                _help("/project unregister <name>", "Remove a chat-registered project", "project"),
            ),
        ),
        workshop=_workshop(
            WorkshopSurface.WORKSPACES,
            input_shape=CapabilityInputShape.MEMORY_PROJECT_OPERATION,
        ),
        mutates_state=True,
        confirmation=ConfirmationPolicy.CONTEXT_DEPENDENT,
        idempotency=IdempotencyPolicy.CONTEXT_DEPENDENT,
        requirements=(*_AUTHENTICATED, ContextRequirement.WORKSPACE),
    ),
    _capability(
        "integration.github.manage",
        "GitHub settings",
        "Inspect or change personal GitHub automation and notification settings.",
        "github_settings",
        AuthorityScope.PRINCIPAL,
        telegram=_telegram(
            "github",
            input_shape=CapabilityInputShape.GITHUB_SETTINGS,
            help_group=5,
            help_entries=(
                _help("/github", "Show GitHub settings", "github"),
                _help(
                    "/github notify [number|reset]",
                    "View, route, or reset notifications",
                    "github",
                ),
                _help("/github reviews [on|off]", "Toggle pull-request reviews", "github"),
                _help("/github triage [on|off]", "Toggle issue triage", "github"),
                _help("/github token [<token>]", "Manage access token", "github"),
                _help("/github add <repo>", "Watch a repository", "github"),
                _help("/github remove <repo>", "Unwatch a repository", "github"),
            ),
        ),
        workshop=_workshop(
            WorkshopSurface.GITHUB_SETTINGS,
            palette=True,
            input_shape=CapabilityInputShape.GITHUB_SETTINGS,
        ),
        mutates_state=True,
        idempotency=IdempotencyPolicy.CONTEXT_DEPENDENT,
        revision_check=True,
        implementation_state=ImplementationState.MIXED_COMPATIBILITY,
        compatibility_removal_gate=CompatibilityRemovalGate.SINGLE_USER_DEPLOYMENT_RETIRED,
        requirements=_AUTHENTICATED,
    ),
    _capability(
        "notification.delivery.manage",
        "Notification delivery",
        "Inspect or change personal notification destinations and delivery preferences.",
        "notification_preferences",
        AuthorityScope.PRINCIPAL,
        telegram=_telegram(
            "notifications",
            input_shape=CapabilityInputShape.NOTIFICATION_SETTINGS,
            help_group=5,
            help_entries=(
                _help("/notifications", "Show personal notification destinations", "notifications"),
                _help(
                    "/notifications <github|generic> [number|reset]",
                    "Route notifications",
                    "notifications",
                ),
            ),
        ),
        workshop=_workshop(
            WorkshopSurface.NOTIFICATION_SETTINGS,
            input_shape=CapabilityInputShape.NOTIFICATION_SETTINGS,
        ),
        mutates_state=True,
        idempotency=IdempotencyPolicy.CONTEXT_DEPENDENT,
        revision_check=True,
        requirements=_AUTHENTICATED,
    ),
    _capability(
        "integration.github.review",
        "Review pull request",
        "Start a durable review of an authorized GitHub pull request.",
        "review_jobs",
        AuthorityScope.PRINCIPAL,
        telegram=_telegram(
            "review",
            disposition=AdapterDisposition.COMPATIBILITY_ONLY,
            input_shape=CapabilityInputShape.PULL_REQUEST_REFERENCE,
            help_group=5,
            help_entries=(
                _help("/review <pr-number>", "Review a pull request on the inferred repository", "review"),
                _help(
                    "/review <owner/repo> <pr-number>",
                    "Review an explicit pull request",
                    "review",
                ),
            ),
        ),
        workshop=_workshop(
            WorkshopSurface.ACTION_PALETTE,
            disposition=AdapterDisposition.ACTION_PALETTE,
            palette=True,
            input_shape=CapabilityInputShape.PULL_REQUEST_REFERENCE,
            implemented=False,
        ),
        implementation_state=ImplementationState.TELEGRAM_COMPATIBILITY,
        mutates_state=True,
        confirmation=ConfirmationPolicy.CONTEXT_DEPENDENT,
        idempotency=IdempotencyPolicy.REQUIRED,
        requirements=(*_AUTHENTICATED, ContextRequirement.WORKSPACE),
    ),
    _capability(
        "memory.manage",
        "Open memory",
        "Browse, search, add, edit, re-scope, or remove personal memory records.",
        "memory_queries",
        AuthorityScope.PRINCIPAL,
        telegram=_telegram(
            "memory",
            disposition=AdapterDisposition.COMPATIBILITY_ONLY,
            input_shape=CapabilityInputShape.MEMORY_OPERATION,
            help_group=6,
            help_entries=(
                _help("/memory", "Browse remembered facts and episodes", "memory"),
                _help("/memory search <q>", "Semantic search over memories", "memory"),
                _help("/memory stats", "Show counts and confidence distribution", "memory"),
                _help("/memory help", "Show memory subcommand reference", "memory"),
            ),
        ),
        workshop=_workshop(
            WorkshopSurface.MEMORY,
            palette=True,
            input_shape=CapabilityInputShape.MEMORY_OPERATION,
        ),
        implementation_state=ImplementationState.MIXED_COMPATIBILITY,
        compatibility_removal_gate=CompatibilityRemovalGate.TELEGRAM_MEMORY_CONVERGED,
        mutates_state=True,
        confirmation=ConfirmationPolicy.CONTEXT_DEPENDENT,
        idempotency=IdempotencyPolicy.CONTEXT_DEPENDENT,
        requirements=_AUTHENTICATED,
    ),
    _capability(
        "preferences.manage",
        "Personal preferences",
        "Inspect, replace, or restore the principal's preference document.",
        "preference_documents",
        AuthorityScope.PRINCIPAL,
        telegram=_telegram(
            "preferences",
            input_shape=CapabilityInputShape.PREFERENCE_OPERATION,
            help_group=6,
            help_entries=(
                _help("/preferences", "Show your preference document", "preferences"),
                _help("/preferences set <text>", "Replace your preference document", "preferences"),
                _help("/preferences history", "List previous preference revisions", "preferences"),
                _help(
                    "/preferences restore <number>",
                    "Restore a displayed revision",
                    "preferences",
                ),
            ),
        ),
        workshop=_workshop(
            WorkshopSurface.PERSONAL_SETTINGS,
            input_shape=CapabilityInputShape.PREFERENCE_OPERATION,
        ),
        mutates_state=True,
        idempotency=IdempotencyPolicy.CONTEXT_DEPENDENT,
        revision_check=True,
        requirements=_AUTHENTICATED,
    ),
    _capability(
        "voice.manage",
        "Voice preferences",
        "Inspect or change personal voice-output preferences.",
        "client_preferences",
        AuthorityScope.PRINCIPAL,
        telegram=_telegram(
            "voice",
            aliases=("voices",),
            input_shape=CapabilityInputShape.VOICE_PREFERENCE,
            help_group=7,
            help_entries=(
                _help("/voice", "Toggle voice off / voice-only", "voice"),
                _help("/voice only", "Use voice only without text", "voice"),
                _help("/voice on", "Use text and voice", "voice"),
                _help("/voice off", "Use text only", "voice"),
                _help("/voice <name>", "Set voice", "voice"),
                _help("/voices", "Choose a voice with inline buttons", "voices"),
            ),
        ),
        workshop=_workshop(
            WorkshopSurface.CLIENT_SETTINGS,
            input_shape=CapabilityInputShape.VOICE_PREFERENCE,
        ),
        mutates_state=True,
        idempotency=IdempotencyPolicy.CONTEXT_DEPENDENT,
        revision_check=True,
        requirements=_AUTHENTICATED,
    ),
    _capability(
        "scheduled_jobs.manage",
        "Open scheduled jobs",
        "List, inspect, and cancel active scheduled jobs.",
        "canonical_scheduler",
        AuthorityScope.PRINCIPAL,
        telegram=_telegram(
            "job",
            aliases=("jobs",),
            input_shape=CapabilityInputShape.SCHEDULED_JOB_OPERATION,
            help_group=8,
            help_entries=(
                _help("/job", "List scheduled jobs", "job", "jobs"),
                _help("/job info <id>", "Show job details", "job"),
                _help("/job cancel <id>", "Cancel a job", "job"),
            ),
        ),
        workshop=_workshop(
            WorkshopSurface.SCHEDULED_JOBS,
            palette=True,
            input_shape=CapabilityInputShape.SCHEDULED_JOB_OPERATION,
        ),
        mutates_state=True,
        confirmation=ConfirmationPolicy.CONTEXT_DEPENDENT,
        idempotency=IdempotencyPolicy.REQUIRED,
        requirements=_AUTHENTICATED,
    ),
    _capability(
        "administration.webhook_status.read",
        "Webhook diagnostics",
        "Inspect redacted host webhook and integration readiness.",
        "webhook_diagnostics",
        AuthorityScope.ADMINISTRATOR,
        telegram=_telegram(
            "webhooks",
            disposition=AdapterDisposition.COMPATIBILITY_ONLY,
            help_group=8,
            help_entries=(_help("/webhooks", "Show webhook server status", "webhooks"),),
        ),
        workshop=_workshop(
            WorkshopSurface.ADMINISTRATION,
            disposition=AdapterDisposition.ADMINISTRATOR_ONLY,
            implemented=False,
        ),
        implementation_state=ImplementationState.TELEGRAM_COMPATIBILITY,
        requirements=(*_AUTHENTICATED, ContextRequirement.ADMINISTRATOR),
    ),
    _capability(
        "capabilities.discover",
        "Help and actions",
        "Discover operations available in the current adapter and context.",
        "capability_registry",
        AuthorityScope.PUBLIC,
        telegram=_telegram(
            "help",
            disposition=AdapterDisposition.COMPATIBILITY_ONLY,
            help_group=8,
            help_entries=(_help("/help", "Show this message", "help"),),
        ),
        workshop=_workshop(
            WorkshopSurface.ACTION_PALETTE,
            disposition=AdapterDisposition.ACTION_PALETTE,
            palette=True,
        ),
        implementation_state=ImplementationState.CANONICAL,
    ),
    _capability(
        "conversation.message.send",
        "Send message",
        "Send a message in an authorized conversation.",
        "conversation_commands",
        AuthorityScope.CHANNEL_MEMBER,
        telegram=_telegram(input_shape=CapabilityInputShape.MESSAGE),
        workshop=_workshop(
            WorkshopSurface.CONVERSATION,
            input_shape=CapabilityInputShape.MESSAGE,
        ),
        mutates_state=True,
        idempotency=IdempotencyPolicy.REQUIRED,
        requirements=_CONVERSATION,
    ),
    _capability(
        "channels.manage",
        "Channels",
        "Create, archive, restore, and inspect Workshop channels.",
        "channel_lifecycle",
        AuthorityScope.CHANNEL_OWNER,
        telegram=_telegram(disposition=AdapterDisposition.UNSUPPORTED_BY_DESIGN),
        workshop=_workshop(
            WorkshopSurface.CHANNELS,
            input_shape=CapabilityInputShape.CHANNEL_OPERATION,
        ),
        mutates_state=True,
        confirmation=ConfirmationPolicy.CONTEXT_DEPENDENT,
        idempotency=IdempotencyPolicy.REQUIRED,
        requirements=_AUTHENTICATED,
    ),
    _capability(
        "direct_messages.manage",
        "Direct messages",
        "Start, archive, restore, and inspect private direct conversations.",
        "direct_message_lifecycle",
        AuthorityScope.PRINCIPAL,
        telegram=_telegram(disposition=AdapterDisposition.UNSUPPORTED_BY_DESIGN),
        workshop=_workshop(
            WorkshopSurface.DIRECT_MESSAGES,
            input_shape=CapabilityInputShape.DIRECT_MESSAGE_PEER,
        ),
        mutates_state=True,
        confirmation=ConfirmationPolicy.CONTEXT_DEPENDENT,
        idempotency=IdempotencyPolicy.REQUIRED,
        requirements=_AUTHENTICATED,
    ),
    _capability(
        "agents.manage",
        "Agents",
        "Create, revise, activate, archive, and inspect principal-owned agents.",
        "agent_definitions",
        AuthorityScope.AGENT_OWNER,
        telegram=_telegram(disposition=AdapterDisposition.UNSUPPORTED_BY_DESIGN),
        workshop=_workshop(
            WorkshopSurface.AGENTS,
            input_shape=CapabilityInputShape.AGENT_DEFINITION,
        ),
        mutates_state=True,
        confirmation=ConfirmationPolicy.CONTEXT_DEPENDENT,
        idempotency=IdempotencyPolicy.REQUIRED,
        revision_check=True,
        requirements=_AUTHENTICATED,
    ),
    _capability(
        "channels.participants.manage",
        "Channel participants",
        "Add or remove human and agent participation in an owned channel.",
        "channel_membership",
        AuthorityScope.CHANNEL_OWNER,
        telegram=_telegram(disposition=AdapterDisposition.UNSUPPORTED_BY_DESIGN),
        workshop=_workshop(
            WorkshopSurface.CONVERSATION_CONTEXT,
            input_shape=CapabilityInputShape.CHANNEL_PARTICIPANT,
        ),
        mutates_state=True,
        confirmation=ConfirmationPolicy.CONTEXT_DEPENDENT,
        idempotency=IdempotencyPolicy.REQUIRED,
        requirements=(*_CONVERSATION, ContextRequirement.CHANNEL_OWNER),
    ),
    _capability(
        "channels.standing_participation.manage",
        "Standing participation",
        "Manage policy-bounded standing agent participation in a channel.",
        "standing_participation",
        AuthorityScope.CHANNEL_OWNER,
        telegram=_telegram(disposition=AdapterDisposition.UNSUPPORTED_BY_DESIGN),
        workshop=_workshop(
            WorkshopSurface.CONVERSATION_CONTEXT,
            input_shape=CapabilityInputShape.STANDING_PARTICIPATION_POLICY,
        ),
        mutates_state=True,
        confirmation=ConfirmationPolicy.CONTEXT_DEPENDENT,
        idempotency=IdempotencyPolicy.REQUIRED,
        requirements=(*_CONVERSATION, ContextRequirement.CHANNEL_OWNER),
    ),
    _capability(
        "threads.manage",
        "Threads",
        "Read, reply to, follow, unfollow, and update read state for threads.",
        "thread_service",
        AuthorityScope.CHANNEL_MEMBER,
        telegram=_telegram(disposition=AdapterDisposition.UNSUPPORTED_BY_DESIGN),
        workshop=_workshop(
            WorkshopSurface.THREADS,
            input_shape=CapabilityInputShape.THREAD_OPERATION,
        ),
        mutates_state=True,
        idempotency=IdempotencyPolicy.REQUIRED,
        requirements=_CONVERSATION,
    ),
    _capability(
        "reactions.manage",
        "Message reactions",
        "Add or remove reactions on messages visible to the principal.",
        "message_reactions",
        AuthorityScope.CHANNEL_MEMBER,
        telegram=_telegram(disposition=AdapterDisposition.UNSUPPORTED_BY_DESIGN),
        workshop=_workshop(
            WorkshopSurface.CONVERSATION,
            input_shape=CapabilityInputShape.REACTION,
        ),
        mutates_state=True,
        idempotency=IdempotencyPolicy.REQUIRED,
        requirements=_CONVERSATION,
    ),
    _capability(
        "activity.inbox.manage",
        "Activity",
        "Inspect notifications, mentions, followed threads, and unread state.",
        "activity_service",
        AuthorityScope.PRINCIPAL,
        telegram=_telegram(disposition=AdapterDisposition.UNSUPPORTED_BY_DESIGN),
        workshop=_workshop(
            WorkshopSurface.ACTIVITY,
            input_shape=CapabilityInputShape.ACTIVITY_STATE,
        ),
        mutates_state=True,
        idempotency=IdempotencyPolicy.REQUIRED,
        requirements=_AUTHENTICATED,
    ),
    _capability(
        "profile.manage",
        "Profile",
        "Inspect or change the principal's display name and avatar.",
        "human_profile",
        AuthorityScope.PRINCIPAL,
        telegram=_telegram(disposition=AdapterDisposition.UNSUPPORTED_BY_DESIGN),
        workshop=_workshop(
            WorkshopSurface.PROFILE,
            input_shape=CapabilityInputShape.PROFILE,
        ),
        mutates_state=True,
        idempotency=IdempotencyPolicy.CONTEXT_DEPENDENT,
        revision_check=True,
        requirements=_AUTHENTICATED,
    ),
    _capability(
        "appearance.manage",
        "Appearance",
        "Inspect or change personal Workshop appearance preferences.",
        "appearance_preferences",
        AuthorityScope.PRINCIPAL,
        telegram=_telegram(disposition=AdapterDisposition.UNSUPPORTED_BY_DESIGN),
        workshop=_workshop(
            WorkshopSurface.APPEARANCE,
            input_shape=CapabilityInputShape.APPEARANCE,
        ),
        mutates_state=True,
        idempotency=IdempotencyPolicy.CONTEXT_DEPENDENT,
        revision_check=True,
        requirements=_AUTHENTICATED,
    ),
    _capability(
        "context.principal_policy.manage",
        "Principal policy",
        "Inspect or change the principal-owned policy supplied to agents.",
        "principal_policies",
        AuthorityScope.PRINCIPAL,
        telegram=_telegram(disposition=AdapterDisposition.UNSUPPORTED_BY_DESIGN),
        workshop=_workshop(WorkshopSurface.RUN_INSPECTOR),
        mutates_state=True,
        idempotency=IdempotencyPolicy.CONTEXT_DEPENDENT,
        revision_check=True,
        requirements=_AUTHENTICATED,
    ),
    _capability(
        "runtime.routing_policy.manage",
        "Task routing policy",
        "Inspect or change policy-bounded task routing for an agent runtime.",
        "routing_policy",
        AuthorityScope.PRINCIPAL_AGENT,
        telegram=_telegram(disposition=AdapterDisposition.UNSUPPORTED_BY_DESIGN),
        workshop=_workshop(WorkshopSurface.AGENT_SETTINGS),
        mutates_state=True,
        idempotency=IdempotencyPolicy.CONTEXT_DEPENDENT,
        revision_check=True,
        requirements=(*_AUTHENTICATED, ContextRequirement.AGENT),
    ),
    _capability(
        "context.inspect",
        "Context inspector",
        "Inspect the redacted context sources used for an agent run.",
        "context_manifests",
        AuthorityScope.CHANNEL_MEMBER,
        telegram=_telegram(disposition=AdapterDisposition.UNSUPPORTED_BY_DESIGN),
        workshop=_workshop(WorkshopSurface.RUN_INSPECTOR),
        requirements=_AGENT_CONVERSATION,
    ),
    _capability(
        "artifacts.read",
        "Artifacts",
        "Open or download artifacts visible in an authorized conversation.",
        "artifact_service",
        AuthorityScope.CHANNEL_MEMBER,
        telegram=_telegram(disposition=AdapterDisposition.UNSUPPORTED_BY_DESIGN),
        workshop=_workshop(
            WorkshopSurface.ARTIFACTS,
            input_shape=CapabilityInputShape.ARTIFACT_REFERENCE,
        ),
        requirements=_CONVERSATION,
    ),
)


def validate_capability_registry(
    definitions: Iterable[CapabilityDefinition] = CAPABILITY_REGISTRY,
    *,
    telegram_commands: Iterable[str] | None = None,
) -> tuple[CapabilityDefinition, ...]:
    """Validate uniqueness, semantic consistency, and optional adapter coverage."""
    items = tuple(definitions)
    operation_ids = [item.operation_id for item in items]
    if len(operation_ids) != len(set(operation_ids)):
        raise ValueError("Capability operation IDs must be unique")
    if any(
        not item.operation_id or not item.label or not item.description or not item.canonical_service for item in items
    ):
        raise ValueError("Capability definitions require stable IDs and public metadata")

    # Imported lazily to keep boundary metadata free to import registry helpers
    # without creating a module-initialization cycle.
    from kai.capability_boundaries import CANONICAL_SERVICE_BOUNDARIES

    unknown_services = sorted({item.canonical_service for item in items} - set(CANONICAL_SERVICE_BOUNDARIES))
    if unknown_services:
        raise ValueError(f"Capability definitions name unknown canonical services: {unknown_services}")

    command_owners: dict[str, str] = {}
    for item in items:
        if set(item.presentations) != set(AdapterId):
            raise ValueError(f"Capability {item.operation_id} must define every adapter disposition")
        if not item.mutates_state and (
            item.confirmation != ConfirmationPolicy.NONE
            or item.idempotency != IdempotencyPolicy.NONE
            or item.revision_check
            or item.continuity_effect != ContinuityEffect.NONE
        ):
            raise ValueError(f"Read-only capability {item.operation_id} declares mutation semantics")
        if (
            item.implementation_state == ImplementationState.MIXED_COMPATIBILITY
            and item.compatibility_removal_gate is None
        ):
            raise ValueError(f"Mixed compatibility capability {item.operation_id} lacks a removal gate")
        if (
            item.implementation_state != ImplementationState.MIXED_COMPATIBILITY
            and item.compatibility_removal_gate is not None
        ):
            raise ValueError(f"Canonical capability {item.operation_id} declares a compatibility removal gate")
        if (
            item.authority_scope == AuthorityScope.ADMINISTRATOR
            and ContextRequirement.ADMINISTRATOR not in item.requirements
        ):
            raise ValueError(f"Administrator capability {item.operation_id} lacks an administrator requirement")
        for adapter, presentation in item.presentations.items():
            if adapter == AdapterId.TELEGRAM:
                if presentation.surface is not None or presentation.palette_entry:
                    raise ValueError(f"Telegram capability {item.operation_id} declares a Workshop surface")
                if presentation.primary_command is None and presentation.aliases:
                    raise ValueError(f"Telegram capability {item.operation_id} has aliases without a primary command")
                if presentation.help_entries and presentation.help_group is None:
                    raise ValueError(f"Telegram capability {item.operation_id} has help entries without a help group")
                if presentation.help_group is not None and not presentation.help_entries:
                    raise ValueError(f"Telegram capability {item.operation_id} has a help group without help entries")
                if presentation.help_discoverable and not presentation.help_entries:
                    raise ValueError(f"Telegram capability {item.operation_id} is discoverable without help entries")
                help_commands: set[str] = set()
                for entry in presentation.help_entries:
                    if not entry.usage.startswith("/") or not entry.description.strip() or not entry.commands:
                        raise ValueError(f"Capability {item.operation_id} has invalid Telegram help metadata")
                    help_commands.update(entry.commands)
                unknown_help_commands = help_commands - set(presentation.commands)
                if unknown_help_commands:
                    raise ValueError(
                        f"Capability {item.operation_id} advertises unregistered Telegram commands: "
                        f"{sorted(unknown_help_commands)}"
                    )
                if presentation.help_discoverable and help_commands != set(presentation.commands):
                    raise ValueError(
                        f"Capability {item.operation_id} does not advertise every registered Telegram command"
                    )
                for command in presentation.commands:
                    if not command.isidentifier() or command.lower() != command:
                        raise ValueError(f"Capability {item.operation_id} has an invalid Telegram command")
                    prior = command_owners.setdefault(command, item.operation_id)
                    if prior != item.operation_id:
                        raise ValueError(f"Telegram command {command} belongs to multiple capabilities")
            else:
                if presentation.primary_command is not None or presentation.aliases:
                    raise ValueError(f"Workshop capability {item.operation_id} declares Telegram commands")
                if presentation.help_group is not None or presentation.help_entries or presentation.help_discoverable:
                    raise ValueError(f"Workshop capability {item.operation_id} declares Telegram help metadata")
                if (
                    presentation.disposition
                    in {
                        AdapterDisposition.NATIVE_SURFACE,
                        AdapterDisposition.ACTION_PALETTE,
                        AdapterDisposition.ADMINISTRATOR_ONLY,
                    }
                    and presentation.surface is None
                ):
                    raise ValueError(f"Workshop capability {item.operation_id} lacks a surface")
            if presentation.disposition == AdapterDisposition.UNSUPPORTED_BY_DESIGN and (
                presentation.surface is not None
                or presentation.primary_command is not None
                or presentation.palette_entry
                or presentation.implemented
                or presentation.help_entries
            ):
                raise ValueError(f"Unsupported capability {item.operation_id} declares an adapter surface")
            if presentation.disposition == AdapterDisposition.ACTION_PALETTE and not presentation.palette_entry:
                raise ValueError(f"Palette capability {item.operation_id} is not marked for the palette")
            if presentation.disposition == AdapterDisposition.ADMINISTRATOR_ONLY and (
                item.authority_scope != AuthorityScope.ADMINISTRATOR
            ):
                raise ValueError(f"Non-administrator capability {item.operation_id} uses an administrator-only surface")

    if telegram_commands is not None:
        registered = frozenset(telegram_commands)
        inventoried = frozenset(command_owners)
        if registered != inventoried:
            missing = sorted(registered - inventoried)
            extra = sorted(inventoried - registered)
            raise ValueError(f"Telegram capability inventory mismatch: missing={missing}, extra={extra}")
    return items


def capability_by_id(operation_id: str) -> CapabilityDefinition:
    """Return one registered capability by its stable operation ID."""
    try:
        return _CAPABILITIES_BY_ID[operation_id]
    except KeyError as exc:
        raise KeyError(f"Unknown capability operation: {operation_id}") from exc


def telegram_command_inventory() -> Mapping[str, str]:
    """Map every Telegram command and presentation alias to one operation ID."""
    return _TELEGRAM_COMMANDS


def telegram_command_dispositions() -> Mapping[str, AdapterDisposition]:
    """Describe primary commands and aliases without duplicating operations."""
    return _TELEGRAM_COMMAND_DISPOSITIONS


def render_telegram_help(*, administrator: bool) -> str:
    """Render Telegram-native help from the canonical capability registry."""
    grouped: dict[int, list[str]] = {}
    for item in CAPABILITY_REGISTRY:
        presentation = item.presentations[AdapterId.TELEGRAM]
        if (
            not presentation.implemented
            or not presentation.help_discoverable
            or presentation.disposition == AdapterDisposition.UNSUPPORTED_BY_DESIGN
            or (item.authority_scope == AuthorityScope.ADMINISTRATOR and not administrator)
        ):
            continue
        assert presentation.help_group is not None
        grouped.setdefault(presentation.help_group, []).extend(
            f"{entry.usage} - {entry.description}" for entry in presentation.help_entries
        )
    return "\n\n".join("\n".join(grouped[group]) for group in sorted(grouped))


def capability_availability(
    adapter: AdapterId,
    context: CapabilityContext,
    *,
    include_unavailable: bool = False,
) -> tuple[CapabilityAvailability, ...]:
    """Project safe discovery metadata; canonical services must still authorize every call."""
    results: list[CapabilityAvailability] = []
    for item in CAPABILITY_REGISTRY:
        presentation = item.presentations[adapter]
        if not presentation.implemented or presentation.disposition == AdapterDisposition.UNSUPPORTED_BY_DESIGN:
            continue
        missing = tuple(requirement for requirement in item.requirements if not context.satisfies(requirement))
        if ContextRequirement.ADMINISTRATOR in missing:
            # Do not disclose administrator-only operations to ordinary principals.
            continue
        available = not missing
        if not available and not include_unavailable:
            continue
        reason = None if available else _availability_reason(missing)
        results.append(
            CapabilityAvailability(
                operation_id=item.operation_id,
                label=item.label,
                description=item.description,
                disposition=presentation.disposition,
                surface=presentation.surface,
                palette_entry=presentation.palette_entry,
                authority_scope=item.authority_scope,
                input_shape=presentation.input_shape,
                mutates_state=item.mutates_state,
                confirmation=item.confirmation,
                available=available,
                unavailable_reason=reason,
            )
        )
    return tuple(results)


def _availability_reason(missing: tuple[ContextRequirement, ...]) -> str:
    if ContextRequirement.AUTHENTICATED in missing:
        return "Sign in to use this action."
    if ContextRequirement.CHANNEL_OWNER in missing:
        return "This action requires channel-owner access."
    if ContextRequirement.AGENT_OWNER in missing:
        return "This action requires agent-owner access."
    if ContextRequirement.CONVERSATION in missing:
        return "Open a conversation to use this action."
    if ContextRequirement.AGENT in missing:
        return "Select an agent to use this action."
    if ContextRequirement.WORKSPACE in missing:
        return "Select an authorized workspace to use this action."
    return "This action is unavailable in the current context."


validate_capability_registry()
_CAPABILITIES_BY_ID = MappingProxyType({item.operation_id: item for item in CAPABILITY_REGISTRY})
_TELEGRAM_COMMANDS = MappingProxyType(
    {
        command: item.operation_id
        for item in CAPABILITY_REGISTRY
        for command in item.presentations[AdapterId.TELEGRAM].commands
    }
)
_TELEGRAM_COMMAND_DISPOSITIONS = MappingProxyType(
    {
        command: (
            item.presentations[AdapterId.TELEGRAM].disposition
            if command == item.presentations[AdapterId.TELEGRAM].primary_command
            else AdapterDisposition.PRESENTATION_ALIAS
        )
        for item in CAPABILITY_REGISTRY
        for command in item.presentations[AdapterId.TELEGRAM].commands
    }
)
