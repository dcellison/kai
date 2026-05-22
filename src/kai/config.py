"""
Application configuration loaded from environment variables.

Provides functionality to:
1. Define the Config dataclass with all application settings
2. Load and validate configuration from .env file
3. Resolve filesystem paths relative to the project root
4. Fail fast with clear error messages on misconfiguration

The main interface is through load_config(), which returns a frozen Config instance.
All paths are resolved relative to PROJECT_ROOT (the repository root), which is
derived from this file's location in the source tree: src/kai/config.py -> project root.
"""

import logging
import os
import pwd
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml
from dotenv import load_dotenv

log = logging.getLogger(__name__)


# ── Module-level paths and constants ─────────────────────────────────

# Derive project root from file location: src/kai/config.py -> src/kai -> src -> project root.
# In a pip-installed deployment (e.g., /opt/kai/venv/lib/.../site-packages/kai/), this
# resolves to site-packages/ instead of the install root. KAI_INSTALL_DIR overrides it.
_FILE_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = Path(os.environ.get("KAI_INSTALL_DIR") or str(_FILE_ROOT))

# Writable data directory for runtime artifacts (database, logs, crash flag).
# Defaults to PROJECT_ROOT for development. In a protected installation where
# source lives in read-only /opt/kai/, this points to user-owned /var/lib/kai/.
# Uses `or` so that an empty string also falls back (same pattern as CLAUDE_USER).
DATA_DIR = Path(os.environ.get("KAI_DATA_DIR") or str(PROJECT_ROOT))

# Valid agent backend choices. "claude" uses Claude Code CLI,
# "goose" uses Goose ACP, "codex" uses OpenAI Codex CLI's app-server
# JSON-RPC protocol. Shared between load_config() and install.py.
VALID_BACKENDS = {"claude", "goose", "codex"}

# Valid LLM providers per backend. Claude always uses Anthropic
# (hardcoded in get_effective_provider), so it has no entry here.
# Backends that don't appear in this dict accept no provider config.
# NOTE: Non-Claude entries in VALID_BACKENDS should have a matching
# key here. Missing entries silently skip provider validation.
VALID_PROVIDERS: dict[str, frozenset[str]] = {
    "goose": frozenset({"anthropic", "openai", "google", "openrouter", "ollama"}),
}

# Maps LLM provider to its API key environment variable name.
# Backend-agnostic - the API key for Anthropic is the same env var
# regardless of which backend is using it.
# Ollama is absent because it requires no API key (local inference).
PROVIDER_KEY_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# ── Provider-aware model registry ──────────────────────────────────────

# Maps provider name to a dict of model_key -> display_name. Keys are
# the identifiers passed to GOOSE_MODEL (for Goose) or --model (for
# Claude CLI). Values are display names for the Telegram keyboard.
# Goose passes model IDs through verbatim to the provider API - no
# aliasing layer - so these must be the exact strings the APIs accept.
PROVIDER_MODELS: dict[str, dict[str, str]] = {
    "anthropic": {
        "opus": "\U0001f9e0 Opus",
        "sonnet": "\u26a1 Sonnet",
        "haiku": "\U0001fab6 Haiku",
    },
    "openai": {
        "gpt-5.4": "\U0001f7e2 GPT-5.4",
        "gpt-5.4-mini": "\U0001f7e1 GPT-5.4 Mini",
        "gpt-5.4-nano": "\U0001f535 GPT-5.4 Nano",
    },
    "google": {
        "gemini-3.1-pro": "\u264a Gemini 3.1 Pro",
        "gemini-3-flash": "\u264a Gemini 3 Flash",
        "gemini-3-deep-think": "\u264a Gemini 3 Deep Think",
    },
}

# Default model for each provider, used when no model is explicitly
# configured. Open-ended providers (openrouter, ollama) have no
# default - users on those providers MUST have a model set.
PROVIDER_DEFAULTS: dict[str, str] = {
    "anthropic": "sonnet",
    "openai": "gpt-5.4",
    "google": "gemini-3-flash",
}

# Codex CLI model surface. Independent of PROVIDER_MODELS["openai"]:
# codex CLI exposes a different set than the OpenAI HTTP API surface
# goose drives, and treating them as one list lets a goose-only
# model (e.g. gpt-5.4-nano) leak onto a codex install where the CLI
# rejects it. Refresh when the codex CLI bumps; no auto-discovery.
CODEX_MODELS: dict[str, str] = {
    "gpt-5.5": "\U0001f7e2 GPT-5.5",
    "gpt-5.4": "\U0001f7e1 GPT-5.4",
    "gpt-5.4-mini": "\U0001f535 GPT-5.4 Mini",
    "gpt-5.3-codex": "\U0001f7e0 GPT-5.3 Codex",
    "gpt-5.3-codex-spark": "⚡ GPT-5.3 Codex Spark",
    "gpt-5.2": "\U0001f7e4 GPT-5.2",
}

# Codex's default model when DEFAULT_MODEL is unset on a codex install.
# Independent of PROVIDER_DEFAULTS["openai"] (goose-on-openai still
# consults that constant; shifting it would change goose's default).
CODEX_DEFAULT_MODEL = "gpt-5.5"

# Providers that accept arbitrary model IDs with no curated list.
# These show a text-based UI instead of an inline keyboard in bot.py.
OPEN_ENDED_PROVIDERS: frozenset[str] = frozenset({"openrouter", "ollama"})

# Union of all model keys from curated surfaces. Used for workspace
# config validation where the active backend is unknown at load time
# (workspaces can be used by users on different backends/providers).
# Includes CODEX_MODELS so codex-only IDs like gpt-5.5 are accepted
# in workspaces.yaml; each backend's change_workspace decides whether
# to actually USE the override for its surface.
_ALL_CURATED_MODELS: frozenset[str] = frozenset(
    list(model for models in PROVIDER_MODELS.values() for model in models) + list(CODEX_MODELS.keys())
)


# ── Per-role model registry ──────────────────────────────────────────
#
# Maps (backend, role) -> the model identifier each backend's CLI
# accepts as --model for that role. Centralizes per-function defaults
# so codex and any future backend can declare its mapping in one place
# instead of scattered _TRIAGE_MODEL / _REVIEW_MODEL / _DEFAULT_*
# constants across modules.
#
# Scope: only the four one-shot agent roles whose model strings move
# between backends in the codex epic. CONVERSATION is excluded
# deliberately: conversational model selection is owned by pool.py +
# DEFAULT_MODEL + PROVIDER_DEFAULTS, and routing it through the
# registry would collide with the per-user override layer that
# users.yaml / /settings already provide.


class ModelRole(StrEnum):
    """
    Logical role tag for a model-selection lookup.

    Each role corresponds to a specific agent-invocation site
    (triage.py, review.py, eval/behavioral.py) whose model string
    differs by backend. The registry MODEL_REGISTRY pairs each
    (backend, role) with the vendor-specific identifier to pass to
    that backend's CLI as --model.
    """

    PR_REVIEW = "pr_review"
    ISSUE_TRIAGE = "issue_triage"
    BEHAVIORAL_JUDGE = "behavioral_judge"
    BEHAVIORAL_GEN = "behavioral_gen"
    # Memory extraction roles. Resolved per-user at extraction time:
    # `get_model_for(MEMORY_EXTRACTION, effective_backend)` picks the
    # registry row matching the user's effective agent_backend. Codex
    # users get the codex registry value (gpt-5.4-mini); claude users
    # get the claude registry value. MEMORY_EPISODE is a distinct row
    # so operators can adjust one stage without touching the other; both
    # are runtime-only (no env-var override surface, no Config field).
    MEMORY_EXTRACTION = "memory_extraction"
    MEMORY_EPISODE = "memory_episode"


# Values are exactly what each backend's CLI accepts as --model.
# Claude rows match today's inline constants verbatim so the registry
# migration is no-behavior-change:
# - triage.py _TRIAGE_MODEL = "sonnet"
# - review.py _REVIEW_MODEL = "sonnet"
# - eval/behavioral.py _DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"
# - eval/behavioral.py _DEFAULT_GEN_MODEL = "sonnet"
# Codex rows reflect editorial tier mapping: nano for cheap high-volume
# work (judge), mini for balanced default generation (gen, triage, review).
# Goose entries are intentionally absent: goose's model identifier
# depends on llm_provider (the _GOOSE_AGENT_MODELS dicts in triage.py /
# review.py already encode that per-provider resolution), which would
# need a third axis the registry does not carry today.
MODEL_REGISTRY: dict[tuple[str, ModelRole], str] = {
    ("claude", ModelRole.PR_REVIEW): "sonnet",
    ("claude", ModelRole.ISSUE_TRIAGE): "sonnet",
    ("claude", ModelRole.BEHAVIORAL_JUDGE): "claude-haiku-4-5-20251001",
    ("claude", ModelRole.BEHAVIORAL_GEN): "sonnet",
    ("codex", ModelRole.PR_REVIEW): "gpt-5.4-mini",
    ("codex", ModelRole.ISSUE_TRIAGE): "gpt-5.4-mini",
    # gpt-5.4-mini for the judge role: the closest analog to the
    # claude haiku tier (small, fast, cheap). gpt-5.4-nano would have
    # matched the editorial "cheaper than mini" intent but the codex
    # CLI does not expose it, so it would fail at first behavioral
    # eval. _check_model_registry_complete validates registry rows
    # against CODEX_MODELS so future drift gets caught at startup.
    ("codex", ModelRole.BEHAVIORAL_JUDGE): "gpt-5.4-mini",
    ("codex", ModelRole.BEHAVIORAL_GEN): "gpt-5.4-mini",
    # Memory extraction rows. Claude entries match the historical
    # extraction default (claude-haiku-4-5-20251001), so any install
    # that ran under the old MEMORY_EXTRACTION_MODEL=<empty> path
    # sees identical model resolution post-migration. Codex entries
    # pick the smallest currently-exposed codex model (gpt-5.4-mini)
    # on the same editorial tier as Claude's haiku selection;
    # _check_model_registry_complete validates codex rows against
    # CODEX_MODELS at startup so a future SKU rename surfaces the
    # bad row before runtime.
    ("claude", ModelRole.MEMORY_EXTRACTION): "claude-haiku-4-5-20251001",
    ("claude", ModelRole.MEMORY_EPISODE): "claude-haiku-4-5-20251001",
    ("codex", ModelRole.MEMORY_EXTRACTION): "gpt-5.4-mini",
    ("codex", ModelRole.MEMORY_EPISODE): "gpt-5.4-mini",
}


def get_model_for(role: ModelRole, backend: str, override: str = "") -> str:
    """
    Resolve the model identifier for a (role, backend) pair.

    Override precedence: if `override` is truthy, it wins (used by
    per-role env vars like ISSUE_TRIAGE_MODEL_CODEX or by CLI flags
    like --judge-model). Otherwise the registry value is returned.

    Raises LookupError on a missing (backend, role) row. Total in the
    steady state because _check_model_registry_complete() runs at
    startup and fails fast (SystemExit) if any role is missing for
    the active backend. The runtime LookupError exists for defensive
    parsing: a request-path bug should surface as a 5xx, not a
    process-wide SystemExit.

    Args:
        role: The ModelRole the caller wants a model for.
        backend: The active backend string ("claude" or "codex").
        override: Caller-supplied override string. Empty disables.

    Returns:
        The model identifier to pass to the backend's CLI.
    """
    if override:
        return override
    try:
        return MODEL_REGISTRY[(backend, role)]
    except KeyError:
        raise LookupError(f"No registry entry for (backend={backend}, role={role.value})") from None


def _check_model_registry_complete(backend: str) -> None:
    """
    Verify MODEL_REGISTRY has a row for every role the active backend uses,
    AND that each codex row names a model the codex CLI actually exposes.

    Runs once at load_config() time. Raises SystemExit on a missing or
    invalid row so the bug surfaces at startup rather than at a per-
    request LookupError (which would otherwise crash a single agent
    invocation silently if a future role were added without its
    registry rows, or with a codex-incompatible model).

    Goose is exempt: goose-side model resolution lives in module-level
    _GOOSE_AGENT_MODELS dicts that this registry does not subsume.
    """
    if backend not in ("claude", "codex"):
        return
    missing = [role for role in ModelRole if (backend, role) not in MODEL_REGISTRY]
    if missing:
        names = ", ".join(role.value for role in missing)
        raise SystemExit(
            f"MODEL_REGISTRY is missing rows for backend '{backend}': {names}. Update MODEL_REGISTRY in config.py."
        )
    # Validate codex rows against the CLI's actual model surface so a
    # future drift (operator updates CODEX_MODELS without touching
    # MODEL_REGISTRY, or vice versa) fails fast at startup rather
    # than at the first behavioral / triage / review run.
    if backend == "codex":
        invalid = [
            (role, MODEL_REGISTRY[(backend, role)])
            for role in ModelRole
            if MODEL_REGISTRY[(backend, role)] not in CODEX_MODELS
        ]
        if invalid:
            valid_list = ", ".join(sorted(CODEX_MODELS.keys()))
            details = ", ".join(f"{role.value}={model}" for role, model in invalid)
            raise SystemExit(
                f"MODEL_REGISTRY has codex rows naming models the codex CLI does not expose: {details}. "
                f"Valid codex models: {valid_list}. Update MODEL_REGISTRY in config.py."
            )


def get_effective_provider(backend: str, llm_provider: str) -> str:
    """Derive the effective provider from backend + llm_provider.

    Claude is always anthropic; codex is always openai (codex CLI has
    no other provider surface, so a wizard-generated codex install with
    LLM_PROVIDER unset still needs the runtime to know its provider is
    openai). Goose is the only backend that consults the raw
    llm_provider, because it routes to whatever provider the user
    configured. Kept identical to the backend->provider rule
    get_user_backend_and_provider uses so the two cascade helpers do
    not drift; any caller resolving an effective provider for a
    backend gets the same answer regardless of which helper it uses.
    """
    if backend == "claude":
        return "anthropic"
    if backend == "codex":
        return "openai"
    return llm_provider


# Valid values for the inner Claude --effort flag, taken verbatim from
# `claude --help` output. If Anthropic adds a new tier (e.g. "ultra"),
# update EFFORT_LEVELS here or otherwise-valid configs will be rejected
# at config-load time. Two shapes intentionally:
#   - EFFORT_LEVELS (ordered tuple): operator-facing surface. Used in
#     error messages and wizard prompts so the listing reads as an
#     intensity progression (low -> max) rather than alphabetical
#     ('high','low','max','medium','xhigh'), which is what sorted()
#     on the frozenset would produce and would confuse an operator.
#   - _VALID_EFFORT_LEVELS (derived frozenset): internal membership
#     check, O(1). Derived from EFFORT_LEVELS so the two cannot drift.
# Validation against the frozenset keeps an invalid CLAUDE_EFFORT_LEVEL
# from reaching the inner-Claude subprocess: a bad value would otherwise
# cost a full session to discover (the subprocess would fail at startup,
# not at config load).
EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
_VALID_EFFORT_LEVELS: frozenset[str] = frozenset(EFFORT_LEVELS)


def validate_model_for_provider(model: str, provider: str) -> bool:
    """Check if a model is valid for a provider.

    Returns True if the provider is open-ended (accepts any model)
    or if the model is in the provider's curated list. Unknown providers
    (not in PROVIDER_MODELS or OPEN_ENDED_PROVIDERS) are accepted with
    a warning - this catches the case where VALID_PROVIDERS gains
    a new entry but PROVIDER_MODELS was not updated to match.

    Backend-aware callers should use `validate_model_for_backend`. This
    function stays as the implementation goose / claude delegate to.
    """
    if provider in OPEN_ENDED_PROVIDERS:
        return True
    models = PROVIDER_MODELS.get(provider)
    if models is None:
        # Provider is valid (passed VALID_PROVIDERS check) but has
        # no curated model list and is not explicitly open-ended. This
        # is a programming oversight - log it so it's visible.
        if provider:
            log.warning(
                "Provider '%s' has no entry in PROVIDER_MODELS or OPEN_ENDED_PROVIDERS; accepting model '%s' unchecked",
                provider,
                model,
            )
        return True
    return model in models


def validate_model_for_backend(model: str, backend: str, eff_provider: str) -> bool:
    """Check if a model is valid for the active backend.

    Codex installs validate against CODEX_MODELS only - no fallback to
    PROVIDER_MODELS["openai"] or any other provider surface. The
    backend determines the model surface 1:1 for codex (which speaks
    its own CLI's curated set); other backends delegate to the existing
    provider-only validator, which is unchanged for goose / claude.

    Canonical model validator: every model-selection site in the
    codebase routes through this function so codex and goose share
    no fallback path.
    """
    if backend == "codex":
        return model in CODEX_MODELS
    return validate_model_for_provider(model, eff_provider)


def models_for_backend(agent_backend: str, eff_provider: str) -> dict[str, str] | None:
    """Curated model list for the given (backend, provider) pair.

    Codex consults its own CODEX_MODELS surface; everything else falls
    back to PROVIDER_MODELS[eff_provider]. Returns None for open-ended
    providers (caller falls back to a free-text prompt rather than a
    fixed-choice keyboard). Used by both install.py (wizard prompt +
    apply-time validator) and bot.py (/model keyboard + selection
    validator).
    """
    if agent_backend == "codex":
        return CODEX_MODELS
    if eff_provider in OPEN_ENDED_PROVIDERS:
        return None
    return PROVIDER_MODELS.get(eff_provider)


def get_user_backend_and_provider(user_config: "UserConfig | None", config: "Config") -> tuple[str, str]:
    """Resolve (backend, provider) for a chat_id with the per-user cascade.

    Pool.py and bot.py already implement this cascade (user.agent_backend
    > config.agent_backend, similar for llm_provider); this function
    consolidates it in one place so every runtime model-validation site
    sees the same effective backend for a given user. Backend determines
    provider 1:1 for codex (openai) and claude (anthropic); goose's
    provider comes from the same cascade as its backend.
    """
    backend = user_config.agent_backend if user_config and user_config.agent_backend else config.agent_backend
    if backend == "claude":
        provider = "anthropic"
    elif backend == "codex":
        provider = "openai"
    else:
        provider = user_config.llm_provider if user_config and user_config.llm_provider else config.llm_provider
    return backend, provider


# Maximum context window size in tokens. Claude's hard ceiling.
# Shared between load_config() (env var validation) and
# _load_user_configs() (users.yaml validation).
MAX_CONTEXT_CEILING = 1_000_000

# Image file extensions that Telegram renders inline as photos.
# Shared between bot.py (inbound document handling) and webhook.py (send-file API).
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


# ── Self-sudo resolution ─────────────────────────────────────────────


def resolve_claude_user(claude_user: str | None) -> str | None:
    """
    Return None if claude_user matches the current process user.

    When the bot process runs as the same OS user specified in
    claude_user, sudo -u is both unnecessary and likely to fail
    (sudoers typically disallows self-sudo). This function detects
    that case and returns None so callers skip the sudo wrapper.

    Returns claude_user unchanged when it differs from the current
    user, or when the current user cannot be determined (e.g.,
    containers with unmapped UIDs).
    """
    if not claude_user:
        return None
    try:
        current_user = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        # UID has no passwd entry (e.g., containers with --user <uid>).
        # Can't determine if it's a self-sudo, so fall through to sudo.
        return claude_user
    if claude_user == current_user:
        log.warning(
            "claude_user %r matches bot process user; skipping sudo",
            claude_user,
        )
        return None
    return claude_user


# ── Per-workspace configuration ──────────────────────────────────────


@dataclass(frozen=True)
class WorkspaceConfig:
    """
    Per-workspace configuration loaded from workspaces.yaml.

    All fields except path are optional. When None, the global default
    from Config is used instead. This lets workspaces override only
    the settings they care about.

    Attributes:
        path: Canonical resolved workspace directory.
        model: Claude model override (haiku/sonnet/opus).
        budget: Per-session spending cap in USD.
        timeout: Per-readline timeout in seconds.
        env: Inline environment variables for the Claude subprocess.
        env_file: Path to a KEY=VALUE file to load as environment vars.
        system_prompt: Inline system prompt text.
        system_prompt_file: Path to a file containing the system prompt.
    """

    path: Path
    model: str | None = None
    budget: float | None = None
    timeout: int | None = None
    env: dict[str, str] | None = None
    env_file: Path | None = None
    system_prompt: str | None = None
    system_prompt_file: Path | None = None


# ── Per-user configuration ──────────────────────────────────────────


@dataclass(frozen=True)
class UserConfig:
    """
    Per-user configuration loaded from users.yaml.

    Defines a user's identity, authorization, and resource limits.
    Preferences that the user controls (active model, working budget)
    live in the settings table, not here. The fields below are
    admin-set baselines that users can override within boundaries.

    Attributes:
        telegram_id: Telegram user ID (authorization key).
        name: Display name for logs and notifications.
        role: "admin" or "user". Admins receive unattributed webhooks.
        github: GitHub username for webhook actor routing.
        os_user: OS username for subprocess isolation (Phase 3).
        home_workspace: Per-user home workspace directory.
        max_budget: Default budget in USD for this user. Used as the
            baseline when the user has not set their own via /settings
            budget. Does NOT serve as a ceiling - the ceiling is
            BUDGET_CEILING (global env var).
        model: Default model name (e.g., "opus", "sonnet", "haiku").
        timeout: Default timeout in seconds for Claude responses.
        context_window: Default context window size in tokens (0 = default).
        workspace_base: Base directory for /workspace new and name resolution.
            Falls back to global WORKSPACE_BASE env var if not set.
    """

    telegram_id: int
    name: str
    role: str = "user"
    github: str | None = None
    os_user: str | None = None
    home_workspace: Path | None = None
    max_budget: float | None = None
    model: str | None = None
    timeout: int | None = None
    context_window: int | None = None
    workspace_base: Path | None = None
    # Per-user allowed workspaces from users.yaml. Distinct from the
    # global `Config.allowed_workspaces` (env var / workspaces.yaml)
    # and from the per-chat DB `allowed_workspaces` table (set via
    # `/workspace allow`): this list is admin-set in users.yaml for
    # workspaces a specific user should access by name without having
    # to run `/workspace allow` first. Resolved at load time
    # (.expanduser().resolve()); paths that don't exist on the host
    # are dropped with a warning rather than failing the whole
    # config load. See issue #460.
    allowed_workspaces: list[Path] = field(default_factory=list)
    # Per-user backend/provider override. Admin-controlled via users.yaml,
    # not user-configurable via /settings. None = use global config.
    agent_backend: str | None = None
    llm_provider: str | None = None
    # GitHub notification routing fields. github_repos controls which
    # repos route webhook events to this user. pr_review and issue_triage
    # are tri-state: None = use global default, True/False = admin override.
    github_repos: list[str] = field(default_factory=list)
    github_notify_chat_id: int | None = None
    pr_review: bool | None = None
    issue_triage: bool | None = None


# ── Config dataclass ─────────────────────────────────────────────────


@dataclass(frozen=True)
class Config:
    """
    Immutable application configuration populated from environment variables.

    All fields map to environment variables defined in .env (see templates/.env for
    reference). Required fields raise SystemExit with descriptive messages if missing.
    Optional fields have sensible defaults for single-user local deployment.

    Attributes:
        telegram_bot_token: Bot token from @BotFather (required)
        telegram_webhook_url: Public URL where Telegram pushes updates via webhook.
            When set, Kai runs in webhook mode (Telegram POSTs updates here).
            When None, Kai falls back to long-polling (Kai pulls updates from Telegram).
        telegram_webhook_secret: Secret token for validating incoming Telegram updates.
            Sent by Telegram as X-Telegram-Bot-Api-Secret-Token header on each update.
            Only required in webhook mode. Defaults to WEBHOOK_SECRET if not explicitly set.
        allowed_user_ids: Set of Telegram user IDs permitted to interact with the bot (required)
        default_model: Default model name, provider-dependent (e.g. sonnet, gpt-5.4, gemini-3-flash)
        claude_timeout_seconds: Seconds before a Claude response is considered timed out
        budget_ceiling: Global budget ceiling in USD. Users cannot exceed
            this via /settings budget. Also serves as the fallback default
            for users without a per-user max_budget in users.yaml.
            Informational only on the claude backend (Max-plan OAuth makes
            the CLI's computed-cost ceiling a phantom signal; the
            --max-budget-usd flag is omitted from claude --print argv).
            Enforced on non-claude backends, which run on pay-per-token
            billing where the ceiling is a real cost gate.
        claude_max_session_hours: Hours before the inner Claude process is recycled. Prevents
            unbounded V8 memory growth that can trigger macOS Jetsam kernel panics. 0 = no limit.
        session_db_path: Path to the SQLite database for sessions, jobs, and settings
        webhook_port: Port for the local aiohttp server (webhooks + scheduling API)
        webhook_secret: HMAC secret for verifying incoming webhook payloads
        voice_enabled: Whether to transcribe Telegram voice notes via whisper-cpp
        whisper_model_path: Path to the whisper-cpp GGML model file
        tts_enabled: Whether to enable Piper text-to-speech for voice responses
        piper_model_dir: Directory containing Piper voice model files
        workspace_base: Base directory for workspace name resolution (/workspace <name>)
        allowed_workspaces: Additional workspace directories accessible by name, from config only.
            These appear as pinned workspaces in /workspaces and are reachable via /workspace <name>
            without being under WORKSPACE_BASE. Non-existent paths are skipped at startup.
        claude_user: OS user for the inner Claude process. When set, Claude is spawned
            via 'sudo -u <user>' for OS-level isolation. Must be a non-admin user.
            When None, Claude runs as the same user as the bot (development default).
    """

    # Required fields - no defaults, must be provided
    telegram_bot_token: str
    allowed_user_ids: set[int]

    # Telegram transport mode: set telegram_webhook_url to use webhook mode,
    # leave as None to fall back to long-polling. The secret is only needed
    # in webhook mode to authenticate incoming updates from Telegram.
    telegram_webhook_url: str | None = None
    telegram_webhook_secret: str | None = None

    # Claude Code process configuration
    default_model: str = "sonnet"
    claude_timeout_seconds: int = 120
    budget_ceiling: float = 10.0
    claude_max_session_hours: float = 0  # 0 = no limit
    claude_idle_timeout: int = 1800  # seconds before idle subprocess eviction; 0 = no eviction

    # Context window tuning. Both settings help control token usage and
    # reduce cache invalidation pressure on the inner Claude process.
    # 0 = use Claude Code defaults (1M window, ~83% autocompact threshold).
    claude_max_context_window: int = 0  # tokens; passed as --max-context-window CLI flag
    claude_autocompact_pct: int = 0  # 1-100; passed as CLAUDE_AUTOCOMPACT_PCT_OVERRIDE env var

    # Effort level passed to inner Claude as `--effort <value>`. Higher
    # settings spend more reasoning tokens per turn, improving answer
    # quality at the cost of latency and budget. Default "high" matches
    # the operator's outer-Claude default; switching to "medium" globally
    # would silently downgrade existing reasoning quality. Critical when
    # CLAUDE_USER / users.yaml `os_user` is set, because inner Claude
    # then runs as a different OS user and does NOT inherit the outer
    # operator's settings.json effort default - without this CLI value,
    # user-isolated installs would silently fall to whatever the claude
    # binary picks as its own default. Validated at config load against
    # _VALID_EFFORT_LEVELS so a typo fails fast rather than at the
    # subprocess startup of the next chat session.
    claude_effort_level: str = "high"

    # Codex backend auth mode. "subscription" (default): the codex CLI
    # uses ~/.codex/auth.json populated by an interactive `codex login`.
    # "api_key": codex reads OPENAI_API_KEY from the environment, the
    # same way goose+openai does. Only consulted when AGENT_BACKEND=codex;
    # ignored on every other backend.
    codex_auth_mode: str = "subscription"

    # Database - uses DATA_DIR so the db lands in the writable data directory
    session_db_path: Path = field(default_factory=lambda: DATA_DIR / "kai.db")

    # Webhook server
    webhook_port: int = 8080
    webhook_secret: str = ""

    # Voice input (speech-to-text via whisper-cpp)
    voice_enabled: bool = False
    whisper_model_path: Path = field(default_factory=lambda: PROJECT_ROOT / "models" / "ggml-base.en.bin")

    # Voice output (text-to-speech via Piper)
    tts_enabled: bool = False
    piper_model_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models" / "piper")

    # Workspace switching
    workspace_base: Path | None = None
    allowed_workspaces: list[Path] = field(default_factory=list)

    # Per-workspace configuration from workspaces.yaml. Keyed by
    # canonical resolved path. Empty dict if no config file exists.
    workspace_configs: dict[Path, WorkspaceConfig] = field(default_factory=dict)

    # User separation: run Claude as a different OS user for process isolation.
    # When set, the bot spawns Claude via 'sudo -u <user> claude ...'.
    # The user must exist on the system and must NOT have admin/sudo privileges.
    # When unset, Claude runs as the same user as the bot (development default).
    claude_user: str | None = None

    # PR review agent: automatically review PRs when GitHub webhooks fire.
    # Disabled by default so existing users are not surprised by automatic reviews.
    pr_review_enabled: bool = False
    # Minimum seconds between reviews of the same PR. Absorbs force-push bursts
    # so rapid pushes to an open PR don't trigger a review for each one.
    pr_review_cooldown: int = 300
    # Subprocess timeout for a single PR review, in seconds. Sonnet with
    # extended thinking on a large diff with prior review context can take
    # a long time; the default gives thinking-heavy reviews room while
    # still terminating genuinely stuck processes.
    pr_review_timeout_s: int = 900
    # Hard USD ceiling for one PR review subprocess. Omitted from claude
    # --print argv on the claude backend (Max-plan OAuth makes the CLI's
    # computed-cost ceiling a phantom signal); the field is consulted by
    # non-claude backends, where reviews run on pay-per-token billing
    # and the ceiling is a real cost gate. Field stays defined so
    # operators with hand-edited /etc/kai/env from a previous install
    # are not affected by upgrade.
    pr_review_budget_usd: float = 1.0
    # Deprecated: review agent now resolves repos via workspace config.
    # Kept for backwards compatibility with existing .env files; the value
    # is parsed but no longer used by webhook.py.
    github_repo: str = ""
    # Directory (relative to repo root) where spec files live for
    # branch-name matching. Does not affect body marker resolution,
    # which accepts any path relative to the repo root.
    spec_dir: str = "specs"

    # Issue triage agent: automatically triage new issues when webhooks fire.
    # Disabled by default so existing users are not surprised by automatic triage.
    issue_triage_enabled: bool = False

    # Global notification chat override. When set, acts as fallback for
    # users without a per-user github_notify_chat_id. Parsed from
    # GITHUB_NOTIFY_CHAT_ID env var.
    github_notify_chat_id: int | None = None

    # File retention: delete uploaded files older than this many days.
    # 0 = no cleanup (default). Cleanup runs once every 24 hours.
    file_retention_days: int = 0

    # Per-user configuration from users.yaml. Keyed by telegram_id.
    # None means users.yaml does not exist (fall back to allowed_user_ids).
    # Empty dict means users.yaml exists but has no valid entries.
    user_configs: dict[int, UserConfig] | None = None

    # TOTP two-factor authentication timing (only relevant when TOTP is enabled)
    totp_session_minutes: int = 30
    totp_challenge_seconds: int = 120
    totp_lockout_attempts: int = 3
    totp_lockout_minutes: int = 15

    # Agent backend selection: "claude" (default) uses Claude Code CLI,
    # "goose" uses Goose ACP (Agent Client Protocol) as the agent harness.
    agent_backend: str = "claude"

    # LLM provider for non-Claude backends (e.g. Goose). Determines
    # which API key env var the backend expects and whether Kai's
    # logical model names ("sonnet", "opus") are translated to Anthropic IDs.
    # Ignored when agent_backend="claude".
    llm_provider: str = ""

    # Semantic memory system (Mem0 + Qdrant + local embeddings).
    # When enabled, every conversation is embedded and searchable.
    # When disabled, all memory functions return empty results.
    memory_enabled: bool = False
    memory_search_limit: int = 10
    memory_token_budget: int = 2000
    memory_embedding_model: str = "all-MiniLM-L6-v2"

    # Memory extraction toggle. Sub-toggle of memory_enabled (extraction
    # only runs when memory is on); the kill switch is the same flag.
    # The reasoner and model for extraction are derived per-user from
    # the user's effective `agent_backend` at extraction time
    # (memory_extraction._build_memory_reasoner +
    # get_model_for(role, effective_backend)). There is no global
    # MEMORY_REASONER_BACKEND / MEMORY_EXTRACTION_MODEL / MEMORY_EPISODE_MODEL
    # config surface; the registry is the single source of truth for
    # the (role, backend) -> model mapping. An operator who wants a
    # different model for a (role, backend) pair edits MODEL_REGISTRY
    # in code.
    memory_extraction_enabled: bool = False
    # Per-call budget ceiling. Retained as inert compatibility config.
    # Both supported memory reasoners (claude and codex) are
    # subscription-backed in the operator deployment model and no code
    # path forwards this value to subprocess argv. Runaway protection
    # comes from memory_extraction_timeout_s instead. The field stays
    # in the dataclass so operators upgrading from a deployment with
    # MEMORY_EXTRACTION_BUDGET_USD already set in /etc/kai/env do not
    # see a startup error after upgrade.
    memory_extraction_budget_usd: float = 0.01
    # Timeout (seconds) for a single extraction subprocess. Haiku
    # typically finishes in 2-4s; 10s gives headroom without stranding
    # an executor thread on a hung subprocess.
    memory_extraction_timeout_s: int = 10

    # Number of prior (user, assistant) exchanges fed to the stage-1
    # extractor as PRIOR CONTEXT background for the episode classifier.
    # The classifier judges whether the CURRENT exchange (the one being
    # logged) is the closing turn of an episode; it needs the lead-up
    # to recognize closure that wasn't visible in a single-turn payload
    # (issue #392). Total payload window = N + 1 (current + N prior).
    # Set 0 to disable windowing entirely - the extractor reverts to
    # the single-turn payload that was production behavior before this
    # field shipped. The 0-10 range is enforced at load time; the cap
    # exists to prevent an operator-typo (3000 instead of 3) from
    # producing a single payload with ~3001 pairs in the PRIOR
    # CONTEXT block, which would exceed Haiku's per-call token
    # limit. Default 3 is empirically grounded: the live probe that
    # motivated this fix flipped the test episode true at 3 and 4
    # turns, but 4 turns leaked one borderline assistant-claim fact
    # that 3 turns suppressed; 3 is the cleaner pick for fact
    # extraction at equal classifier accuracy.
    episode_classifier_context_turns: int = 3

    # Number of existing facts surfaced to the extractor per call as
    # consolidation candidates (intent: update_of / skip_redundant).
    # Selected by semantic similarity to the assistant payload, capped
    # at this value. Set to 0 to disable consolidation entirely: the
    # candidate fetch is skipped, the EXISTING FACTS data block is
    # omitted from the Haiku payload, and the extractor falls back to
    # always emitting intent="new" (anchored by the CONSOLIDATION
    # prompt section, which is retained even when the data block is
    # empty). Storage then uses the existing _paraphrase_neighbor semantics.
    # Tradeoff: each candidate adds ~50-100 chars to the Haiku payload
    # and contributes to per-call cache-creation tokens. Default 8 is
    # 2x the per-call fact cap (5), which gives the model enough to
    # find 1-2 update targets per proposed fact without doubling the
    # per-call cost.
    memory_consolidation_candidates_n: int = 8

    # Stage-2 episode generation (issue #385). Conditional second extractor
    # that runs out-of-band on stage-1 positives (has_episode=true) to
    # produce one Sophia-shaped episode record per episode-worthy turn.
    # Honors memory_enabled; no dedicated kill switch. The model is
    # resolved per-user from the registry at episode-extraction time
    # via get_model_for(ModelRole.MEMORY_EPISODE, effective_backend);
    # there is no global override field.
    # Per-call budget ceiling (USD). Retained as inert compatibility
    # config. Same rationale as memory_extraction_budget_usd above:
    # both supported reasoners are subscription-backed and no code
    # path forwards this value to subprocess argv. Runaway protection
    # comes from memory_episode_timeout_s instead. The field stays in
    # the dataclass so operators upgrading from a deployment with
    # MEMORY_EPISODE_BUDGET_USD already set in /etc/kai/env do not see
    # a startup error after upgrade.
    memory_episode_budget_usd: float = 0.15
    # Subprocess timeout (seconds). Default 120 - twice the production-
    # tuned stage-1 value (60s) and 12x the stage-1 dataclass default
    # (10s). The asymmetry is intentional: stage 2 is fire-and-forget
    # off the user-facing turn, so a long-tailed timeout only delays
    # storage, never the user's reply. Floor of 10s prevents accidental
    # sub-Haiku-warm-up timeouts that would mask real model failure as
    # configuration error.
    memory_episode_timeout_s: int = 120

    # Minimum Mem0 similarity score for a memory to be returned by
    # search-driven paths: both `format_context` (context injection
    # at session start) and the `/memory search` UI surface in
    # `memory_command.py`. Values below the floor are dropped before
    # any ranking. Default 0.3 matches Mem0's built-in default and
    # the prior hard-coded constant; raise toward 0.5+ to reduce
    # false positives at the cost of recall. Spec 310 §7.5 documents
    # the "one knob, two paths" decision: keeping the UI floor and
    # the context-injection floor in lockstep prevents silent
    # divergence between "what the user sees in /memory search" and
    # "what Kai pulls into context" after a config change.
    memory_search_floor: float = 0.3

    # Write-time semantic-similarity dedup threshold. The extractor's
    # `_paraphrase_neighbor` helper compares each `intent="new"`
    # candidate against the top-1 nearest existing fact (per user) and
    # drops the candidate when the cosine score meets or exceeds this
    # value. Lowering the threshold catches more paraphrase clusters at
    # the cost of more false merges; raising it preserves more facts at
    # the cost of duplicate accumulation. Range [0.3, 1.01]: the lower
    # bound matches `memory_search_floor`'s operator floor; 1.0 fires
    # only on exact-cosine matches (effectively disabled), and 1.01 is
    # the unambiguous-disable sentinel. The dataclass default is the
    # operator-validated production value; upgraders inherit it
    # without re-running `make config`.
    memory_duplicate_threshold: float = 0.9

    def get_workspace_config(self, workspace: Path) -> WorkspaceConfig | None:
        """
        Get per-workspace config for a path, or None for global defaults.

        Resolves the path before lookup to handle symlinks and relative
        paths consistently.
        """
        return self.workspace_configs.get(workspace.resolve())

    def get_user_config(self, user_id: int) -> UserConfig | None:
        """Get per-user config by Telegram user ID, or None if not configured."""
        if self.user_configs is None:
            return None
        return self.user_configs.get(user_id)

    def get_user_by_github(self, github_login: str) -> UserConfig | None:
        """Look up a user by GitHub username. Used for webhook actor routing."""
        if self.user_configs is None:
            return None
        for uc in self.user_configs.values():
            if uc.github and uc.github.lower() == github_login.lower():
                return uc
        return None

    def get_admins(self) -> list[UserConfig]:
        """Get all admin users. Used for unattributed webhook fallback routing."""
        if self.user_configs is None:
            return []
        return [uc for uc in self.user_configs.values() if uc.role == "admin"]


# ── Config loading ───────────────────────────────────────────────────


def _read_protected_file(path: str) -> str | None:
    """
    Read a root-owned file via sudo cat.

    Used to load config from /etc/kai/ in a protected installation where the
    bot process runs as an unprivileged user. The -n flag ensures sudo fails
    immediately if no NOPASSWD rule exists (avoids blocking on a password prompt).

    Returns:
        File contents as a string, or None on any failure (missing file,
        no sudoers rule, timeout, etc.).
    """
    try:
        result = subprocess.run(
            ["sudo", "-n", "cat", path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, OSError):
        return None


# Sentinel returned by _read_protected_yaml when the file exists but
# contains invalid YAML. Distinct from None (file absent) so callers
# can stop on malformed config rather than falling through to a local file.
_YAML_MALFORMED = object()


def _read_protected_yaml(filename: str) -> dict | object | None:
    """
    Read a YAML file from /etc/kai/ via sudo.

    Returns:
        Parsed dict on success, None if the file does not exist or cannot
        be read, or the _YAML_MALFORMED sentinel if the file exists but
        is invalid. Callers must check ``is _YAML_MALFORMED`` before use.
    """
    content = _read_protected_file(f"/etc/kai/{filename}")
    if content is None:
        return None
    try:
        result = yaml.safe_load(content)
        if isinstance(result, dict):
            return result
        log.warning("/etc/kai/%s: expected a YAML dict, got %s", filename, type(result).__name__)
        return _YAML_MALFORMED
    except yaml.YAMLError as e:
        log.error("Invalid YAML in /etc/kai/%s: %s", filename, e)
        return _YAML_MALFORMED


def _strip_quotes(value: str) -> str:
    """
    Remove matched surrounding quotes from a value string.

    Only strips when the first and last characters are the same quote
    type (' or "). This avoids corrupting values that contain the
    opposite quote type internally (e.g., 'he said "hello"').
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def parse_env_file(path: Path) -> dict[str, str]:
    """
    Parse a KEY=VALUE file into a dict.

    Handles:
    - Lines with KEY=VALUE or KEY="VALUE" or KEY='VALUE'
    - Lines starting with 'export ' (stripped)
    - Comments (lines starting with #) and blank lines (skipped)
    - Surrounding quotes on values (stripped as matched pairs only -
      inner quotes of the opposite type are preserved)

    Same parsing logic as _read_protected_file() uses for /etc/kai/env.
    Re-reads the file each time to pick up changes without restart.
    """
    env: dict[str, str] = {}
    try:
        # utf-8-sig transparently strips a BOM if present. Windows
        # editors (Notepad, etc.) often save UTF-8 with a BOM, which
        # would silently prepend \ufeff to the first key name.
        text = path.read_text(encoding="utf-8-sig")
    except OSError as e:
        log.warning("Cannot read env file %s: %s", path, e)
        return env
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        # Handle `export KEY=VALUE` lines (common in shell-sourced env files)
        line = line.removeprefix("export ")
        key, _, value = line.partition("=")
        env[key.strip()] = _strip_quotes(value.strip())
    return env


def _load_workspace_configs() -> dict[Path, WorkspaceConfig]:
    """
    Load per-workspace configs from workspaces.yaml.

    Tries /etc/kai/workspaces.yaml first (protected installation),
    falls back to PROJECT_ROOT/workspaces.yaml (development). Returns
    an empty dict if neither file exists.

    Returns a dict keyed by canonical resolved path for O(1) lookup.
    """
    # Try protected file first, fall back to local. A malformed
    # protected file stops loading entirely rather than silently
    # falling through to a local file (which could contain stale
    # or dev config on a production system).
    data = _read_protected_yaml("workspaces.yaml")
    if data is _YAML_MALFORMED:
        # Fail open: return empty dict so the system continues without
        # workspace overrides. Workspace config is convenience, not
        # security-critical.
        log.warning("Skipping workspace config: /etc/kai/workspaces.yaml is malformed or empty")
        return {}
    if data is None:
        local_path = PROJECT_ROOT / "workspaces.yaml"
        if not local_path.exists():
            return {}
        try:
            with open(local_path) as f:
                data = yaml.safe_load(f)
        except (yaml.YAMLError, OSError) as e:
            log.error("Cannot load %s: %s", local_path, e)
            return {}
        if not isinstance(data, dict):
            log.warning("%s: expected a YAML dict, got %s", local_path, type(data).__name__)
            return {}

    entries = data.get("workspaces")
    if not isinstance(entries, list):
        if entries is not None:
            log.warning("workspaces.yaml: 'workspaces' must be a list, got %s", type(entries).__name__)
        return {}

    # Helper for coercing YAML env values to strings. Defined once
    # outside the loop rather than re-created per workspace entry.
    def _coerce_env_value(v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return str(v).lower()
        return str(v)

    configs: dict[Path, WorkspaceConfig] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            log.warning("workspaces.yaml: skipping non-dict entry: %s", entry)
            continue

        # Validate required path field
        raw_path = entry.get("path")
        if not raw_path:
            log.warning("workspaces.yaml: skipping entry without path")
            continue
        path = Path(str(raw_path)).expanduser().resolve()
        if not path.is_dir():
            log.warning("workspaces.yaml: skipping non-existent path: %s", path)
            continue

        # Duplicate check: first wins
        if path in configs:
            log.warning("workspaces.yaml: duplicate path %s; using first entry", path)
            continue

        # Parse the optional claude: section
        claude_section = entry.get("claude") or {}
        if not isinstance(claude_section, dict):
            log.warning("workspaces.yaml: invalid claude section for %s", path)
            continue

        # Validate model against the union of all curated provider
        # models. Workspaces don't have a provider field - they can be
        # used by users on different providers - so any curated model
        # key is accepted. Open-ended provider model IDs are also fine.
        model = claude_section.get("model")
        if model is not None:
            model = str(model)
            if model not in _ALL_CURATED_MODELS:
                log.warning(
                    "workspaces.yaml: unrecognized model '%s' for %s "
                    "(not in any curated provider list); proceeding anyway",
                    model,
                    path,
                )
                # Don't skip - could be an open-ended provider model ID

        # Validate budget (same bool guard as timeout - float(True) is 1.0)
        budget = claude_section.get("budget")
        if budget is not None:
            try:
                if isinstance(budget, bool):
                    raise ValueError("must be a number, not a boolean")
                budget = float(budget)
                if budget <= 0:
                    raise ValueError("must be positive")
            except (TypeError, ValueError) as e:
                log.warning("workspaces.yaml: invalid budget for %s: %s; skipping entry", path, e)
                continue

        # Validate timeout (must be a positive integer, not a float or bool).
        # bool is a subclass of int in Python, so `timeout: true` would
        # silently become 1 without an explicit check.
        timeout = claude_section.get("timeout")
        if timeout is not None:
            try:
                if isinstance(timeout, bool):
                    raise ValueError("must be an integer, not a boolean")
                if isinstance(timeout, float) and not timeout.is_integer():
                    raise ValueError("must be an integer, not a float")
                timeout = int(timeout)
                if timeout <= 0:
                    raise ValueError("must be positive")
            except (TypeError, ValueError) as e:
                log.warning("workspaces.yaml: invalid timeout for %s: %s; skipping entry", path, e)
                continue

        # Parse env vars (inline dict)
        env = claude_section.get("env")
        if env is not None:
            if not isinstance(env, dict):
                log.warning("workspaces.yaml: invalid env for %s; skipping entry", path)
                continue

            env = {str(k): _coerce_env_value(v) for k, v in env.items()}

        # Validate env_file
        env_file = claude_section.get("env_file")
        if env_file is not None:
            env_file = Path(str(env_file)).expanduser().resolve()
            if not env_file.is_file():
                log.warning("workspaces.yaml: env_file not found for %s: %s; skipping entry", path, env_file)
                continue

        # Validate system_prompt / system_prompt_file mutual exclusion
        system_prompt = claude_section.get("system_prompt")
        system_prompt_file = claude_section.get("system_prompt_file")
        if system_prompt is not None and system_prompt_file is not None:
            log.error(
                "workspaces.yaml: both system_prompt and system_prompt_file set for %s; skipping entry",
                path,
            )
            continue
        if system_prompt is not None:
            system_prompt = str(system_prompt)
        if system_prompt_file is not None:
            system_prompt_file = Path(str(system_prompt_file)).expanduser().resolve()
            if not system_prompt_file.is_file():
                log.warning(
                    "workspaces.yaml: system_prompt_file not found for %s: %s; skipping entry",
                    path,
                    system_prompt_file,
                )
                continue

        configs[path] = WorkspaceConfig(
            path=path,
            model=model,
            budget=budget,
            timeout=timeout,
            env=env,
            env_file=env_file,
            system_prompt=system_prompt,
            system_prompt_file=system_prompt_file,
        )

    return configs


# Valid user roles for users.yaml
_VALID_ROLES = {"admin", "user"}


def _compute_extraction_eligible_backends(
    agent_backend: str,
    user_configs: "dict[int, UserConfig] | None",
    memory_extraction_enabled: bool,
) -> set[str]:
    """
    Return the set of distinct effective backends across extraction-eligible users.

    Mirrors `_ingest_memory`'s extraction gate (effective backend in
    {"claude", "codex"}; goose users are filtered out because they
    have no `OneShotReasoner` implementation). With users.yaml: each
    user contributes its effective backend (per-user override or
    global default). Without users.yaml (legacy ALLOWED_USER_IDS auth):
    every authorized user inherits the global, so the set is
    `{agent_backend}` when that is extraction-eligible, else empty.

    Returns an empty set when `memory_extraction_enabled` is False;
    callers that gate codex plumbing or registry validation off the
    set should not fire on retrieval-only or memory-disabled installs.
    """
    if not memory_extraction_enabled:
        return set()
    eligible: set[str] = set()
    if user_configs is None:
        if agent_backend in ("claude", "codex"):
            eligible.add(agent_backend)
        return eligible
    for uc in user_configs.values():
        effective = uc.agent_backend or agent_backend
        if effective in ("claude", "codex"):
            eligible.add(effective)
    return eligible


def _load_user_configs(
    global_backend: str,
    global_llm_provider: str,
) -> dict[int, UserConfig] | None:
    """
    Load per-user configs from users.yaml.

    Tries /etc/kai/users.yaml first (protected), falls back to
    PROJECT_ROOT/users.yaml (dev). Returns None if neither exists
    (signals the caller to fall back to ALLOWED_USER_IDS).

    Args:
        global_backend: The global agent_backend from env config.
        global_llm_provider: The global llm_provider from env config.
            Both are needed to cascade per-user model validation:
            a user's effective provider determines which models are valid.

    Returns a dict keyed by telegram_id for O(1) lookup.
    """
    # Same dual-mode loading pattern as _load_workspace_configs.
    data = _read_protected_yaml("users.yaml")
    if data is _YAML_MALFORMED:
        # Fail closed: return None so the caller falls back to
        # ALLOWED_USER_IDS (or exits if that is also unset). Auth
        # config must not silently degrade.
        log.warning("Skipping user config: /etc/kai/users.yaml is malformed or empty")
        return None
    if data is None:
        local_path = PROJECT_ROOT / "users.yaml"
        if not local_path.exists():
            return None
        try:
            with open(local_path) as f:
                data = yaml.safe_load(f)
        except (yaml.YAMLError, OSError) as e:
            log.error("Cannot load %s: %s", local_path, e)
            return None
        if not isinstance(data, dict):
            log.warning("%s: expected a YAML dict, got %s", local_path, type(data).__name__)
            return None

    # After the _YAML_MALFORMED and None checks, data should be a dict
    # (protected path returns _YAML_MALFORMED for non-dicts, local path
    # checks isinstance explicitly). Guard defensively rather than assert
    # since assertions are stripped under Python -O.
    if not isinstance(data, dict):
        log.warning("users.yaml: expected a YAML dict, got %s", type(data).__name__)
        return None

    entries = data.get("users")
    if not isinstance(entries, list):
        # Warn for any non-list value, including None (missing key).
        # A users.yaml file without a 'users' key is almost certainly
        # a typo (e.g., 'user:' instead of 'users:').
        if entries is not None:
            log.warning("users.yaml: 'users' must be a list, got %s", type(entries).__name__)
        else:
            log.warning("users.yaml: no 'users' key found; check for typos")
        return None

    configs: dict[int, UserConfig] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            log.warning("users.yaml: skipping non-dict entry: %s", entry)
            continue

        # Validate required telegram_id (must be a positive integer, not a bool)
        raw_id = entry.get("telegram_id")
        if raw_id is None:
            log.warning("users.yaml: skipping entry without telegram_id")
            continue
        try:
            if isinstance(raw_id, bool):
                raise ValueError("must be an integer, not a boolean")
            telegram_id = int(raw_id)
            if telegram_id <= 0:
                raise ValueError("must be positive")
        except (TypeError, ValueError) as e:
            log.warning("users.yaml: invalid telegram_id %s: %s; skipping entry", raw_id, e)
            continue

        # Validate required name (strip first so whitespace-only is rejected)
        name = str(entry.get("name") or "").strip()
        if not name:
            log.warning("users.yaml: skipping entry for telegram_id %d without name", telegram_id)
            continue

        # Duplicate check: first wins
        if telegram_id in configs:
            log.warning("users.yaml: duplicate telegram_id %d; using first entry", telegram_id)
            continue

        # Validate role
        role = str(entry.get("role", "user")).strip().lower()
        if role not in _VALID_ROLES:
            log.warning(
                "users.yaml: invalid role '%s' for %s (must be one of %s); skipping entry",
                role,
                name,
                _VALID_ROLES,
            )
            continue

        # Optional fields
        github = entry.get("github")
        if github is not None:
            github = str(github).strip() or None

        os_user = entry.get("os_user")
        if os_user is not None:
            os_user = str(os_user).strip() or None

        # Validate home_workspace. Warn but don't skip the user if
        # the directory doesn't exist - it may be on an unmounted drive
        # or not yet created. The user keeps access; the workspace
        # falls back to the per-user default at runtime.
        home_workspace = entry.get("home_workspace")
        if home_workspace is not None:
            # Guard against empty strings: Path("").resolve() silently
            # returns CWD, which would give the user unintended access.
            home_workspace_str = str(home_workspace).strip()
            if not home_workspace_str:
                home_workspace = None
            else:
                home_workspace = Path(home_workspace_str).expanduser().resolve()
                if not home_workspace.is_dir():
                    # Null it out and let the runtime resolver
                    # (backend.resolve_home_workspace) fall through to
                    # DATA_DIR/home/<chat_id>/. There is no longer a
                    # shared "global default" directory - that was the
                    # multi-user privacy hazard #353 removed.
                    log.warning(
                        "users.yaml: home_workspace not found for %s: %s; falling back to per-user default",
                        name,
                        home_workspace,
                    )
                    home_workspace = None

        # Validate max_budget (same bool guard as workspace budget)
        max_budget = entry.get("max_budget")
        if max_budget is not None:
            try:
                if isinstance(max_budget, bool):
                    raise ValueError("must be a number, not a boolean")
                max_budget = float(max_budget)
                if max_budget <= 0:
                    raise ValueError("must be positive")
            except (TypeError, ValueError) as e:
                log.warning("users.yaml: invalid max_budget for %s: %s; skipping entry", name, e)
                continue

        # Validate optional agent_backend (must be valid if set)
        user_backend: str | None = None
        raw_backend = entry.get("agent_backend")
        if raw_backend is not None:
            backend_str = str(raw_backend).strip().lower()
            if backend_str not in VALID_BACKENDS:
                raise SystemExit(
                    f"users.yaml: user '{name}' has invalid agent_backend '{backend_str}' "
                    f"(must be one of: {', '.join(sorted(VALID_BACKENDS))})"
                )
            user_backend = backend_str

        # Validate optional llm_provider (must be valid for the user's backend)
        user_provider: str | None = None
        raw_provider = entry.get("llm_provider")
        if raw_provider is not None:
            provider_str = str(raw_provider).strip().lower()
            # Validate against the user's effective backend. If the user
            # has no explicit backend, validate against the global one.
            eff_backend_for_val = user_backend or global_backend
            valid = VALID_PROVIDERS.get(eff_backend_for_val)
            if valid is not None and provider_str not in valid:
                raise SystemExit(
                    f"users.yaml: user '{name}' has invalid llm_provider "
                    f"'{provider_str}' for backend '{eff_backend_for_val}' "
                    f"(must be one of: {', '.join(sorted(valid))})"
                )
            user_provider = provider_str

        # Resolve effective backend and provider for this user.
        # Used for both the provider-required check and model validation.
        eff_backend = user_backend or global_backend
        eff_provider_str = user_provider or global_llm_provider
        eff_provider = get_effective_provider(eff_backend, eff_provider_str)

        # If user's effective backend requires a provider but none can be
        # resolved (neither user-level nor global), that's a fatal config error.
        if eff_backend in VALID_PROVIDERS and not eff_provider_str:
            raise SystemExit(
                f"users.yaml: user '{name}' has agent_backend='{eff_backend}' but no "
                f"llm_provider is configured (set it in users.yaml or as "
                f"LLM_PROVIDER env var)"
            )

        # Warn if user is on an open-ended provider with no model set.
        # PROVIDER_DEFAULTS has no entry for openrouter/ollama, so the
        # pool would fall back to the global default_model (e.g., "sonnet")
        # which is almost certainly wrong for that provider.
        model = entry.get("model")
        if model is not None:
            model = str(model).strip().lower()
        if model is None and eff_provider in OPEN_ENDED_PROVIDERS:
            log.warning(
                "users.yaml: user '%s' is on open-ended provider '%s' with no "
                "model set; they must set one via /model or users.yaml",
                name,
                eff_provider,
            )

        # Validate optional model against the user's effective backend.
        # Codex installs consult CODEX_MODELS; other backends consult
        # PROVIDER_MODELS[eff_provider] via the provider-only delegate.
        # Cascade: user override -> global config, same as pool.py.
        if model is not None and not validate_model_for_backend(model, eff_backend, eff_provider):
            if eff_backend == "codex":
                valid = sorted(CODEX_MODELS.keys())
                surface_label = "codex"
            else:
                valid = sorted(PROVIDER_MODELS.get(eff_provider, {}).keys())
                surface_label = f"provider '{eff_provider}'"
            log.warning(
                "users.yaml: invalid model '%s' for %s (%s, must be one of %s); ignoring",
                model,
                name,
                surface_label,
                valid,
            )
            model = None

        # Validate optional timeout (positive integer)
        user_timeout = entry.get("timeout")
        if user_timeout is not None:
            try:
                if isinstance(user_timeout, bool):
                    raise ValueError("must be an integer, not a boolean")
                user_timeout = int(user_timeout)
                if user_timeout <= 0:
                    raise ValueError("must be positive")
            except (TypeError, ValueError) as e:
                log.warning("users.yaml: invalid timeout for %s: %s; ignoring", name, e)
                user_timeout = None

        # Validate optional context_window (0 or 50000-1000000)
        user_context_window = entry.get("context_window")
        if user_context_window is not None:
            try:
                if isinstance(user_context_window, bool):
                    raise ValueError("must be an integer, not a boolean")
                user_context_window = int(user_context_window)
                if user_context_window < 0:
                    raise ValueError("must be non-negative")
                if user_context_window != 0 and user_context_window < 50_000:
                    raise ValueError("must be at least 50000 (or 0 for default)")
                if user_context_window > MAX_CONTEXT_CEILING:
                    raise ValueError(f"must not exceed {MAX_CONTEXT_CEILING}")
            except (TypeError, ValueError) as e:
                log.warning("users.yaml: invalid context_window for %s: %s; ignoring", name, e)
                user_context_window = None

        # Validate optional workspace_base (absolute path to a directory).
        # Warn but don't skip the user if the directory doesn't exist.
        # Unlike home_workspace (which keeps the path and lets runtime
        # handle the fallback), workspace_base is set to None so name
        # resolution falls back to the global WORKSPACE_BASE cleanly.
        user_workspace_base = entry.get("workspace_base")
        if user_workspace_base is not None:
            ws_base_str = str(user_workspace_base).strip()
            if not ws_base_str:
                user_workspace_base = None
            else:
                user_workspace_base = Path(ws_base_str).expanduser().resolve()
                if not user_workspace_base.is_dir():
                    log.warning(
                        "users.yaml: workspace_base not found for %s: %s; using global default",
                        name,
                        user_workspace_base,
                    )
                    user_workspace_base = None

        # Parse optional per-user allowed_workspaces (#460). Distinct
        # from the global `allowed_workspaces` config field (env var
        # + workspaces.yaml) and from the per-chat DB
        # `allowed_workspaces` table (set via `/workspace allow`):
        # this is the admin's per-user yaml grant of additional
        # paths accessible by name.
        #
        # Each entry: strip, expanduser, resolve. Empty strings are
        # dropped silently. Paths that don't exist are dropped with
        # a warning - matches the `workspace_base` / `home_workspace`
        # precedent of "warn but don't fail the load" (a missing
        # directory may be on an unmounted drive or about to be
        # created; failing the whole config load would block the
        # operator from any access). Non-list values produce a
        # warning and an empty list. The runtime resolver
        # (`sessions.resolve_workspace_access`) reads this field
        # and unions it with the DB and global sources.
        user_allowed_workspaces: list[Path] = []
        raw_user_allowed = entry.get("allowed_workspaces", [])
        if isinstance(raw_user_allowed, list):
            for raw_ws in raw_user_allowed:
                ws_str = str(raw_ws).strip()
                if not ws_str:
                    continue
                ws_path = Path(ws_str).expanduser().resolve()
                if not ws_path.is_dir():
                    log.warning(
                        "users.yaml: allowed_workspaces entry for %s not found: %s; dropping",
                        name,
                        ws_path,
                    )
                    continue
                # Dedup at load time: a user listing the same path
                # twice is almost certainly a typo, and the union
                # in resolve_workspace_access dedupes anyway, but
                # catching it here keeps the per-user list clean
                # for downstream consumers (e.g., `/workspace allowed`
                # source attribution).
                if ws_path not in user_allowed_workspaces:
                    user_allowed_workspaces.append(ws_path)
        else:
            log.warning(
                "users.yaml: allowed_workspaces for %s must be a list (ignoring)",
                name,
            )

        # Parse optional github_repos list (list of "owner/repo" strings).
        # Validate format but don't verify repo existence (that would
        # require network calls during config loading).
        raw_repos = entry.get("github_repos", [])
        github_repos: list[str] = []
        if isinstance(raw_repos, list):
            for repo_entry in raw_repos:
                repo_str = str(repo_entry).strip()
                parts = repo_str.split("/")
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    log.warning(
                        "users.yaml: invalid github_repos entry for %s: %s (expected owner/repo format)",
                        name,
                        repo_str,
                    )
                    continue
                github_repos.append(repo_str)
        else:
            log.warning(
                "users.yaml: github_repos for %s must be a list (ignoring)",
                name,
            )

        # Parse optional github_notify_chat_id (integer, can be negative
        # for group chats). Follows the same pattern as telegram_id validation.
        github_notify_chat_id: int | None = None
        raw_notify = entry.get("github_notify_chat_id")
        if raw_notify is not None:
            try:
                github_notify_chat_id = int(raw_notify)
            except (ValueError, TypeError):
                log.warning(
                    "users.yaml: invalid github_notify_chat_id for %s: %s",
                    name,
                    raw_notify,
                )

        # Parse optional pr_review and issue_triage booleans.
        # None means "use global default". Explicit true/false overrides.
        pr_review: bool | None = None
        raw_pr = entry.get("pr_review")
        if raw_pr is not None:
            if isinstance(raw_pr, bool):
                pr_review = raw_pr
            else:
                log.warning(
                    "users.yaml: pr_review for %s must be true or false: %s",
                    name,
                    raw_pr,
                )

        issue_triage: bool | None = None
        raw_triage = entry.get("issue_triage")
        if raw_triage is not None:
            if isinstance(raw_triage, bool):
                issue_triage = raw_triage
            else:
                log.warning(
                    "users.yaml: issue_triage for %s must be true or false: %s",
                    name,
                    raw_triage,
                )

        configs[telegram_id] = UserConfig(
            telegram_id=telegram_id,
            name=name,
            role=role,
            github=github,
            os_user=os_user,
            home_workspace=home_workspace,
            max_budget=max_budget,
            model=model,
            timeout=user_timeout,
            context_window=user_context_window,
            workspace_base=user_workspace_base,
            allowed_workspaces=user_allowed_workspaces,
            agent_backend=user_backend,
            llm_provider=user_provider,
            github_repos=github_repos,
            github_notify_chat_id=github_notify_chat_id,
            pr_review=pr_review,
            issue_triage=issue_triage,
        )

    # Warn if no admin is defined - external webhooks will route to
    # an arbitrary user, which may be surprising.
    if configs and not any(uc.role == "admin" for uc in configs.values()):
        log.warning(
            "users.yaml: no admin users defined. External webhook notifications "
            "(GitHub, generic) will route to an arbitrary user."
        )

    return configs


def load_config() -> Config:
    """
    Load application configuration from environment variables.

    Reads from the .env file at the project root via python-dotenv, validates
    required fields, and returns a frozen Config instance. Calls SystemExit
    with descriptive messages on any misconfiguration so the bot fails fast
    at startup rather than encountering cryptic errors later.

    Returns:
        A frozen Config instance with all settings populated.

    Raises:
        SystemExit: If required environment variables are missing or invalid.
    """
    # Try protected config first (/etc/kai/env, root-owned). In a protected
    # installation, secrets live here instead of .env. Falls back to local
    # .env for development. Uses setdefault so explicitly set env vars
    # (e.g., from the launchd plist) take precedence - same as load_dotenv().
    protected_env = _read_protected_file("/etc/kai/env")
    if protected_env:
        for line in protected_env.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                # Handle `export KEY=VALUE` lines (common in shell-sourced env files)
                line = line.removeprefix("export ")
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), _strip_quotes(value.strip()))
    else:
        load_dotenv(PROJECT_ROOT / ".env")

    # Validate required: Telegram bot token
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is required in .env")

    # Telegram transport mode: if TELEGRAM_WEBHOOK_URL is set, use webhook mode
    # (Telegram POSTs updates to this URL). If unset, fall back to long-polling
    # (Kai pulls updates from Telegram). This lets users without a tunnel/proxy
    # run Kai out of the box.
    telegram_webhook_url: str | None = None
    telegram_webhook_secret: str | None = None
    raw_webhook_url = os.environ.get("TELEGRAM_WEBHOOK_URL", "").strip()
    if raw_webhook_url:
        telegram_webhook_url = raw_webhook_url

        # Webhook secret: validates incoming updates from Telegram. Falls back to
        # WEBHOOK_SECRET so existing installs work without a config change. Must be
        # non-empty in webhook mode; an empty secret would let anyone POST fake
        # updates to /webhook/telegram.
        telegram_webhook_secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip()
        if not telegram_webhook_secret:
            telegram_webhook_secret = os.environ.get("WEBHOOK_SECRET", "").strip()
        if not telegram_webhook_secret:
            raise SystemExit(
                "TELEGRAM_WEBHOOK_SECRET (or WEBHOOK_SECRET as fallback) is required in .env "
                "when TELEGRAM_WEBHOOK_URL is set. Without it, anyone could POST fake updates "
                "to the Telegram webhook endpoint."
            )
        log.info("Telegram transport: webhook (%s)", telegram_webhook_url)
    else:
        log.info("Telegram transport: polling (TELEGRAM_WEBHOOK_URL not set)")

    # Parse allowed user IDs. users.yaml is the primary authorization
    # source. ALLOWED_USER_IDS is a legacy fallback for installations
    # that haven't migrated yet.
    # Defer the ValueError until after users.yaml is checked so that
    # a malformed env var doesn't block startup when users.yaml is
    # authoritative and would make the env var irrelevant.
    raw_ids = os.environ.get("ALLOWED_USER_IDS", "")
    allowed_ids: set[int] = set()
    allowed_ids_error: str | None = None
    try:
        allowed_ids = {int(uid.strip()) for uid in raw_ids.split(",") if uid.strip()}
    except ValueError:
        allowed_ids_error = (
            "ALLOWED_USER_IDS contains non-numeric values. "
            "Consider migrating to users.yaml (run 'make config' or "
            "see README for the schema). If using ALLOWED_USER_IDS, values must be "
            "numeric Telegram user IDs - message @userinfobot to find yours."
        )

    # Validate optional: workspace base directory (must exist if provided)
    workspace_base = None
    raw_base = os.environ.get("WORKSPACE_BASE", "").strip()
    if raw_base:
        workspace_base = Path(raw_base).expanduser().resolve()
        if not workspace_base.is_dir():
            raise SystemExit(f"WORKSPACE_BASE is not an existing directory: {workspace_base}")

    # Parse optional: allowed workspaces (comma-separated absolute paths).
    # Paths are resolved to canonical form so /a/b and /a/../a/b deduplicate
    # to one entry. Non-existent paths are skipped with a warning rather than
    # crashing, so a stale entry (e.g. an unmounted drive) doesn't block startup.
    allowed_workspaces: list[Path] = []
    seen_allowed: set[Path] = set()
    raw_allowed = os.environ.get("ALLOWED_WORKSPACES", "").strip()
    if raw_allowed:
        for raw_path in raw_allowed.split(","):
            p = Path(raw_path.strip()).expanduser().resolve()
            if p in seen_allowed:
                continue
            seen_allowed.add(p)
            if p.is_dir():
                allowed_workspaces.append(p)
            else:
                log.warning("ALLOWED_WORKSPACES: skipping non-existent path: %s", p)

    # Validate numeric config - fail fast with clear messages rather than
    # cryptic ValueError tracebacks from int()/float() on bad input.
    # AGENT_TIMEOUT_SECONDS is the canonical key; CLAUDE_TIMEOUT_SECONDS
    # is a legacy alias kept for installs upgrading without re-running
    # the wizard. Apply-time migration in install.py rewrites
    # /etc/kai/env to the new key, so the legacy fallback only fires
    # on the first start after the upgrade.
    raw_timeout = os.environ.get("AGENT_TIMEOUT_SECONDS", "").strip()
    if not raw_timeout:
        legacy_timeout = os.environ.get("CLAUDE_TIMEOUT_SECONDS", "").strip()
        if legacy_timeout:
            log.warning(
                "CLAUDE_TIMEOUT_SECONDS is deprecated; use AGENT_TIMEOUT_SECONDS. "
                "Re-run 'make install' to migrate /etc/kai/env automatically."
            )
            raw_timeout = legacy_timeout
        else:
            raw_timeout = "120"
    try:
        claude_timeout_seconds = int(raw_timeout)
    except ValueError:
        raise SystemExit("AGENT_TIMEOUT_SECONDS must be an integer") from None
    # Budget ceiling: new var takes precedence, fall back to old var
    # for backward compat on upgrade (existing .env has old key name).
    raw_ceiling = os.environ.get("BUDGET_CEILING", "").strip()
    if not raw_ceiling:
        raw_ceiling = os.environ.get("CLAUDE_MAX_BUDGET_USD", "").strip()
    if not raw_ceiling:
        raw_ceiling = "10.0"
    try:
        budget_ceiling = float(raw_ceiling)
    except ValueError:
        raise SystemExit("BUDGET_CEILING must be a number") from None
    try:
        claude_max_session_hours = float(os.environ.get("CLAUDE_MAX_SESSION_HOURS", "0"))
    except ValueError:
        raise SystemExit("CLAUDE_MAX_SESSION_HOURS must be a number") from None
    try:
        claude_idle_timeout = int(os.environ.get("CLAUDE_IDLE_TIMEOUT", "1800"))
    except ValueError:
        raise SystemExit("CLAUDE_IDLE_TIMEOUT must be an integer") from None
    try:
        webhook_port = int(os.environ.get("WEBHOOK_PORT", "8080"))
    except ValueError:
        raise SystemExit("WEBHOOK_PORT must be an integer") from None
    try:
        file_retention_days = int(os.environ.get("FILE_RETENTION_DAYS", "0"))
    except ValueError:
        raise SystemExit("FILE_RETENTION_DAYS must be an integer") from None

    # Context window tuning - 0 means "use Claude Code defaults"
    try:
        claude_max_context_window = int(os.environ.get("CLAUDE_MAX_CONTEXT_WINDOW", "0"))
        if claude_max_context_window < 0 or claude_max_context_window > MAX_CONTEXT_CEILING:
            raise SystemExit(f"CLAUDE_MAX_CONTEXT_WINDOW must be 0-{MAX_CONTEXT_CEILING} (0 = use default)")
    except ValueError:
        raise SystemExit("CLAUDE_MAX_CONTEXT_WINDOW must be an integer") from None
    try:
        claude_autocompact_pct = int(os.environ.get("CLAUDE_AUTOCOMPACT_PCT", "0"))
        if claude_autocompact_pct < 0 or claude_autocompact_pct > 100:
            raise SystemExit("CLAUDE_AUTOCOMPACT_PCT must be 0-100 (0 = use default)")
    except ValueError:
        raise SystemExit("CLAUDE_AUTOCOMPACT_PCT must be an integer") from None

    # CLAUDE_EFFORT_LEVEL: validated against the documented `claude --help`
    # allow-list (_VALID_EFFORT_LEVELS at module top). Lowercase + strip so
    # "HIGH " or " medium" copy-pasted from docs do not become silent
    # operator footguns. Empty string / unset both fall back to the
    # dataclass default ("high") via the `or "high"` short-circuit on the
    # stripped value. Parsing is a simple membership check rather than the
    # surrounding numeric blocks' try/except ValueError pattern, because
    # str.strip().lower() cannot raise; the membership check is the only
    # way to fail. SystemExit on bad input mirrors how the surrounding
    # CLAUDE_* parsing blocks signal config-load failure, keeping the
    # operator-visible behavior consistent across the cluster.
    # Codex auth mode validation: "subscription" (default) or "api_key".
    # An unknown value is a typo in /etc/kai/env; fail fast at startup.
    codex_auth_mode = os.environ.get("CODEX_AUTH_MODE", "subscription").strip().lower() or "subscription"
    if codex_auth_mode not in ("subscription", "api_key"):
        raise SystemExit(f"CODEX_AUTH_MODE must be 'subscription' or 'api_key', got {codex_auth_mode!r}")

    claude_effort_level = os.environ.get("CLAUDE_EFFORT_LEVEL", "high").strip().lower() or "high"
    if claude_effort_level not in _VALID_EFFORT_LEVELS:
        # Use the ordered tuple, not sorted(_VALID_EFFORT_LEVELS) - the
        # latter would print alphabetically ('high','low','max','medium',
        # 'xhigh'), which mangles the intensity progression an operator
        # expects to see in an error string.
        raise SystemExit(f"CLAUDE_EFFORT_LEVEL must be one of {list(EFFORT_LEVELS)}, got {claude_effort_level!r}")

    # PR review agent config
    pr_review_enabled = os.environ.get("PR_REVIEW_ENABLED", "").lower() in ("1", "true", "yes")
    try:
        pr_review_cooldown = int(os.environ.get("PR_REVIEW_COOLDOWN", "300"))
    except ValueError:
        raise SystemExit("PR_REVIEW_COOLDOWN must be an integer") from None
    try:
        pr_review_timeout_s = int(os.environ.get("PR_REVIEW_TIMEOUT_S", "900"))
        if pr_review_timeout_s <= 0:
            raise SystemExit("PR_REVIEW_TIMEOUT_S must be a positive integer")
    except ValueError:
        raise SystemExit("PR_REVIEW_TIMEOUT_S must be an integer") from None
    try:
        pr_review_budget_usd = float(os.environ.get("PR_REVIEW_BUDGET_USD", "1.0"))
        if pr_review_budget_usd <= 0:
            raise SystemExit("PR_REVIEW_BUDGET_USD must be a positive number")
    except ValueError:
        raise SystemExit("PR_REVIEW_BUDGET_USD must be a number") from None

    # Issue triage agent config
    issue_triage_enabled = os.environ.get("ISSUE_TRIAGE_ENABLED", "").lower() in ("1", "true", "yes")

    # Global GitHub notification chat override. When set, acts as fallback
    # for users without a per-user github_notify_chat_id in users.yaml or DB.
    github_notify_raw = os.environ.get("GITHUB_NOTIFY_CHAT_ID", "")
    github_notify_chat_id: int | None = None
    if github_notify_raw:
        try:
            github_notify_chat_id = int(github_notify_raw)
        except ValueError:
            log.warning(
                "Invalid GITHUB_NOTIFY_CHAT_ID: %s (ignoring)",
                github_notify_raw,
            )

    try:
        totp_session_minutes = int(os.environ.get("TOTP_SESSION_MINUTES", "30"))
    except ValueError:
        raise SystemExit("TOTP_SESSION_MINUTES must be an integer") from None
    try:
        totp_challenge_seconds = int(os.environ.get("TOTP_CHALLENGE_SECONDS", "120"))
    except ValueError:
        raise SystemExit("TOTP_CHALLENGE_SECONDS must be an integer") from None
    try:
        totp_lockout_attempts = int(os.environ.get("TOTP_LOCKOUT_ATTEMPTS", "3"))
    except ValueError:
        raise SystemExit("TOTP_LOCKOUT_ATTEMPTS must be an integer") from None
    try:
        totp_lockout_minutes = int(os.environ.get("TOTP_LOCKOUT_MINUTES", "15"))
    except ValueError:
        raise SystemExit("TOTP_LOCKOUT_MINUTES must be an integer") from None

    # Agent backend selection - "claude" (default) or "goose"
    agent_backend = os.environ.get("AGENT_BACKEND", "claude").strip().lower()
    if agent_backend not in VALID_BACKENDS:
        raise SystemExit(
            f"AGENT_BACKEND '{agent_backend}' is not valid (must be one of: {', '.join(sorted(VALID_BACKENDS))})"
        )

    # Verify the per-role model registry has rows for every role the
    # active backend will look up. Goose is exempt (its model resolution
    # uses _GOOSE_AGENT_MODELS dicts, not the registry). Raises
    # SystemExit on a missing row so the bug surfaces at startup rather
    # than as a per-request LookupError.
    _check_model_registry_complete(agent_backend)

    # LLM provider - validated against the backend's supported set.
    # Only required for non-Claude backends.
    llm_provider = ""
    valid_providers = VALID_PROVIDERS.get(agent_backend)
    if valid_providers is not None:
        llm_provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
        if llm_provider not in valid_providers:
            raise SystemExit(
                f"LLM_PROVIDER '{llm_provider}' is not valid for backend "
                f"'{agent_backend}' (must be one of: "
                f"{', '.join(sorted(valid_providers))})"
            )

    # Semantic memory system - parse env vars with the same validation
    # pattern as other typed config fields (try/except, SystemExit on bad input).
    memory_enabled = os.environ.get("MEMORY_ENABLED", "").lower() in ("1", "true", "yes")
    try:
        memory_search_limit = int(os.environ.get("MEMORY_SEARCH_LIMIT", "10"))
        if memory_search_limit <= 0:
            raise SystemExit("MEMORY_SEARCH_LIMIT must be a positive integer")
    except ValueError:
        raise SystemExit("MEMORY_SEARCH_LIMIT must be an integer") from None
    try:
        memory_token_budget = int(os.environ.get("MEMORY_TOKEN_BUDGET", "2000"))
        if memory_token_budget <= 0:
            raise SystemExit("MEMORY_TOKEN_BUDGET must be a positive integer")
    except ValueError:
        raise SystemExit("MEMORY_TOKEN_BUDGET must be an integer") from None
    memory_embedding_model = os.environ.get("MEMORY_EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip() or "all-MiniLM-L6-v2"

    # Track 2 Haiku extraction config. Same validation pattern as the
    # other memory_* fields: try/except ValueError, SystemExit on bad
    # input, and reject negatives explicitly on numeric fields.
    memory_extraction_enabled = os.environ.get("MEMORY_EXTRACTION_ENABLED", "").lower() in ("1", "true", "yes")
    # memory_extraction_enabled is a sub-toggle of memory_enabled. The
    # dataclass docstring documents this dependency, but without
    # parse-time enforcement the extraction subprocess fires when
    # MEMORY_EXTRACTION_ENABLED=true is set with MEMORY_ENABLED=false,
    # burning ~$0.01/turn of Haiku budget whose result silently no-ops
    # in the `_memory is None` guard inside add_structured. Compose
    # here so the dependency is explicit and the budget-burn hole is
    # closed regardless of operator env-var ordering.
    memory_extraction_enabled = memory_extraction_enabled and memory_enabled

    # Deprecation warnings for the three retired memory env vars. The
    # reasoner and model used for memory extraction now derive entirely
    # from each user's effective `agent_backend` (per-user dispatch via
    # memory_extraction._build_memory_reasoner +
    # get_model_for(role, effective_backend)). A legacy /etc/kai/env
    # may still carry any of these keys; honor them as one-shot
    # deprecation hints (one log.warning per key seen), then ignore
    # the values. The next `sudo make install` rewrites /etc/kai/env
    # from install.conf with the wizard's emission blocks removed, so
    # the keys do not survive the next reinstall.
    _deprecated_memory_env = (
        ("MEMORY_REASONER_BACKEND", "memory reasoner is now derived per-user from agent_backend"),
        ("MEMORY_EXTRACTION_MODEL", "memory extraction model is now resolved per-user from the MODEL_REGISTRY"),
        ("MEMORY_EPISODE_MODEL", "memory episode model is now resolved per-user from the MODEL_REGISTRY"),
    )
    for _legacy_key, _reason in _deprecated_memory_env:
        if os.environ.get(_legacy_key):
            log.warning(
                "%s is deprecated and ignored; %s. Remove it from /etc/kai/env "
                "(the next 'sudo make install' will drop it automatically).",
                _legacy_key,
                _reason,
            )

    try:
        memory_extraction_budget_usd = float(os.environ.get("MEMORY_EXTRACTION_BUDGET_USD", "0.01"))
        if memory_extraction_budget_usd < 0:
            raise SystemExit("MEMORY_EXTRACTION_BUDGET_USD must be non-negative")
    except ValueError:
        raise SystemExit("MEMORY_EXTRACTION_BUDGET_USD must be a number") from None
    try:
        memory_extraction_timeout_s = int(os.environ.get("MEMORY_EXTRACTION_TIMEOUT_S", "10"))
        if memory_extraction_timeout_s <= 0:
            raise SystemExit("MEMORY_EXTRACTION_TIMEOUT_S must be a positive integer")
    except ValueError:
        raise SystemExit("MEMORY_EXTRACTION_TIMEOUT_S must be an integer") from None
    # Episode-classifier context-turn count. Zero is a valid disable
    # value (single-turn payload, current pre-#392 behavior). Upper
    # bound 10 caps the payload size to protect against an operator
    # typo (a 3000-turn window would blow Haiku's context). The cap is
    # enforced here AND inline at the wizard prompt; load_config is
    # the single source of truth for the daemon, the wizard inline
    # check exists only to give immediate operator feedback.
    try:
        episode_classifier_context_turns = int(os.environ.get("EPISODE_CLASSIFIER_CONTEXT_TURNS", "3"))
        if episode_classifier_context_turns < 0:
            raise SystemExit("EPISODE_CLASSIFIER_CONTEXT_TURNS must be non-negative")
        if episode_classifier_context_turns > 10:
            raise SystemExit("EPISODE_CLASSIFIER_CONTEXT_TURNS must be <= 10")
    except ValueError:
        raise SystemExit("EPISODE_CLASSIFIER_CONTEXT_TURNS must be an integer") from None
    # Consolidation candidate count. Zero is a valid kill-switch value
    # (consolidation disabled, extractor falls back to all-`new`); only
    # negatives are rejected. Same try/except pattern as the other
    # memory_* numeric vars.
    try:
        memory_consolidation_candidates_n = int(os.environ.get("MEMORY_CONSOLIDATION_CANDIDATES_N", "8"))
        if memory_consolidation_candidates_n < 0:
            raise SystemExit("MEMORY_CONSOLIDATION_CANDIDATES_N must be a non-negative integer")
    except ValueError:
        raise SystemExit("MEMORY_CONSOLIDATION_CANDIDATES_N must be an integer") from None

    # Stage-2 episode generation (issue #385). Same try/except pattern as
    # the other memory_* numeric vars. Model resolution is per-user at
    # extraction time (memory_extraction._resolve_episode_model uses
    # get_model_for(ModelRole.MEMORY_EPISODE, effective_backend)); the
    # global MEMORY_EPISODE_MODEL env var is deprecated and warned
    # about above. Budget is strictly positive (zero would silently
    # disable a real subprocess; the intentional kill switch is
    # MEMORY_ENABLED). Timeout floor is 10s to prevent accidentally
    # tightening it below Haiku's warm-up time.
    try:
        memory_episode_budget_usd = float(os.environ.get("MEMORY_EPISODE_BUDGET_USD", "0.15"))
        if memory_episode_budget_usd <= 0:
            raise SystemExit("MEMORY_EPISODE_BUDGET_USD must be positive")
    except ValueError:
        raise SystemExit("MEMORY_EPISODE_BUDGET_USD must be a number") from None
    try:
        memory_episode_timeout_s = int(os.environ.get("MEMORY_EPISODE_TIMEOUT_S", "120"))
        if memory_episode_timeout_s < 10:
            raise SystemExit("MEMORY_EPISODE_TIMEOUT_S must be at least 10")
    except ValueError:
        raise SystemExit("MEMORY_EPISODE_TIMEOUT_S must be an integer") from None

    # Search relevance floor. Float in [0.0, 1.0]; default 0.3 matches
    # Mem0's built-in default and the prior hard-coded constant. Same
    # try/except pattern as the other memory_* numeric vars: bad input
    # exits at startup rather than surfacing as a divide-by-zero or
    # mysterious "no results" symptom at first query. Range is bounded
    # at both ends because Mem0 cosine similarity is normalized to
    # [0.0, 1.0]; a floor outside that range silently filters
    # everything (>1.0) or nothing (<0.0), both of which are footguns.
    try:
        memory_search_floor = float(os.environ.get("MEMORY_SEARCH_FLOOR", "0.3"))
        if memory_search_floor < 0.0 or memory_search_floor > 1.0:
            raise SystemExit("MEMORY_SEARCH_FLOOR must be between 0.0 and 1.0")
    except ValueError:
        raise SystemExit("MEMORY_SEARCH_FLOOR must be a number") from None

    # Write-time paraphrase-dedup threshold. Range [0.3, 1.01]: the
    # lower bound matches the existing `memory_search_floor` operator
    # floor (a threshold below 0.3 produces near-everything-is-a-dup
    # behavior under cosine similarity), and the upper bound permits
    # 1.01 as the unambiguous-disable sentinel (since `score == 1.0`
    # can rarely fire on non-identical text under the embedding model,
    # 1.0 alone is "effectively disabled" but not "guaranteed
    # disabled"). Same try/except shape as the other memory_* numeric
    # vars so a bad env value fails fast at startup rather than
    # surfacing later as silent dedup misbehavior.
    try:
        memory_duplicate_threshold = float(os.environ.get("MEMORY_DUPLICATE_THRESHOLD", "0.9"))
        if memory_duplicate_threshold < 0.3 or memory_duplicate_threshold > 1.01:
            raise SystemExit("MEMORY_DUPLICATE_THRESHOLD must be between 0.3 and 1.01")
    except ValueError:
        raise SystemExit("MEMORY_DUPLICATE_THRESHOLD must be a number") from None

    # Per-workspace configuration. Loaded after ALLOWED_WORKSPACES so
    # YAML-defined workspaces can be merged into the allowed set.
    workspace_configs = _load_workspace_configs()

    # Merge YAML workspace paths into allowed_workspaces. Workspaces
    # defined in the config file are implicitly allowed.
    for p in workspace_configs:
        if p not in seen_allowed:
            seen_allowed.add(p)
            allowed_workspaces.append(p)

    # Per-user configuration. If users.yaml exists, it is authoritative;
    # ALLOWED_USER_IDS is ignored. If users.yaml does not exist,
    # ALLOWED_USER_IDS works as before (backward-compatible).
    user_configs = _load_user_configs(agent_backend, llm_provider)
    if user_configs is not None:
        if raw_ids:
            log.warning("ALLOWED_USER_IDS is set but users.yaml exists; using users.yaml")
        # Users in the YAML replace the ALLOWED_USER_IDS set.
        # Any ALLOWED_USER_IDS parse error is irrelevant since users.yaml is authoritative.
        allowed_ids = set(user_configs.keys())
        if not allowed_ids:
            raise SystemExit("users.yaml exists but contains no valid user entries")
    elif allowed_ids_error:
        # No users.yaml and ALLOWED_USER_IDS had a parse error
        raise SystemExit(allowed_ids_error)
    elif not allowed_ids:
        # No users.yaml and no ALLOWED_USER_IDS - can't start
        raise SystemExit(
            "No user authorization configured. "
            "Run 'make config' to generate users.yaml, "
            "or create one manually (see README for the schema)."
        )
    else:
        # ALLOWED_USER_IDS is valid and no users.yaml exists. This works
        # but is the legacy path - nudge toward users.yaml.
        log.info("Using ALLOWED_USER_IDS from env (legacy). Run 'make config' to migrate to users.yaml.")

    # Memory extraction preconditions, validated per the
    # extraction-eligible backend set. Reasoner and model both derive
    # per-user from each user's effective `agent_backend` at extraction
    # time, so the "is this install in a valid state for memory
    # extraction" question must read the full eligible set rather than
    # a single global backend value. The helper mirrors
    # `_ingest_memory`'s extraction gate in bot.py (goose users are
    # filtered out because they have no `OneShotReasoner` implementation).
    extraction_eligible_backends = _compute_extraction_eligible_backends(
        agent_backend, user_configs, memory_extraction_enabled
    )

    # Registry completeness and binary resolution per
    # eligible backend. With model env vars retired, get_model_for()
    # is the only check that a (role, backend) registry row exists;
    # the existing early _check_model_registry_complete(agent_backend)
    # call covers conversational + triage + review + behavioral-eval
    # rows for the global backend, but a per-user override to a
    # different backend would reach runtime without its memory-role
    # rows being validated. Loop over the eligible set after
    # _load_user_configs runs so per-user overrides are visible.
    # `_check_model_registry_complete` is idempotent so re-running
    # it for the global backend if it appears in the set is harmless.
    # `resolve_oneshot_binary` is gated on the same set so a missing
    # CODEX_BIN on a deployment where any user routes to codex fails
    # fast at startup instead of at the first extraction.
    if memory_extraction_enabled and extraction_eligible_backends:
        from kai.oneshot_binary import BinaryResolutionError, resolve_oneshot_binary

        for _backend in sorted(extraction_eligible_backends):
            _check_model_registry_complete(_backend)
            try:
                resolve_oneshot_binary(_backend)
            except BinaryResolutionError as e:
                raise SystemExit(
                    f"Memory extraction requires the {_backend!r} binary to be reachable "
                    f"at startup (at least one extraction-eligible user routes to it), but "
                    f"{e}. Set CODEX_BIN (for codex) or fix PATH (for either backend); "
                    f"rerun the wizard if you no longer want memory extraction enabled."
                ) from None

    # Deprecation warnings for env vars superseded by users.yaml.
    # The vars still work as global fallbacks, but users.yaml is the
    # primary configuration path. Warnings guide admins to migrate.
    # Only fire when users.yaml parsed successfully. If users.yaml is
    # malformed, the system falls back to ALLOWED_USER_IDS and the env
    # vars are actively needed (not deprecated in that context).
    if user_configs is not None:
        _deprecated_env_vars = {
            "CLAUDE_MODEL": "Renamed to DEFAULT_MODEL. Set per-user 'model' in users.yaml or use /settings model",
            "CLAUDE_MAX_BUDGET_USD": "Renamed to BUDGET_CEILING (global ceiling). Per-user defaults go in users.yaml 'max_budget'",
            "CLAUDE_TIMEOUT_SECONDS": "Set per-user 'timeout' in users.yaml or use /settings timeout",
            "CLAUDE_MAX_CONTEXT_WINDOW": "Set per-user 'context_window' in users.yaml or use /settings context",
            "CLAUDE_USER": "Set per-user 'os_user' in users.yaml",
            "WORKSPACE_BASE": "Set per-user 'workspace_base' in users.yaml",
            "ALLOWED_WORKSPACES": "Users can manage their own via /workspace allow",
            "PR_REVIEW_ENABLED": "Set per-user 'pr_review' in users.yaml or use /github reviews",
            "ISSUE_TRIAGE_ENABLED": "Set per-user 'issue_triage' in users.yaml or use /github triage",
            "GITHUB_NOTIFY_CHAT_ID": "Set per-user 'github_notify_chat_id' in users.yaml or use /github notify",
        }
        for var, guidance in _deprecated_env_vars.items():
            if os.environ.get(var, "").strip():
                log.warning(
                    "%s in env is deprecated (users.yaml exists). %s. "
                    "This env var will be removed in a future release.",
                    var,
                    guidance,
                )

        # Warn when GitHub agent features are enabled but no users have
        # github_repos configured. Events will never reach the agents
        # because _get_subscribed_users() returns empty for every
        # incoming webhook. The fallback path in _process_github_event()
        # delivers basic notifications but does not guarantee the agents
        # fire.
        no_repos = not any(uc.github_repos for uc in user_configs.values())
        if no_repos:
            review_on = pr_review_enabled or any(uc.pr_review is True for uc in user_configs.values())
            triage_on = issue_triage_enabled or any(uc.issue_triage is True for uc in user_configs.values())
            if review_on or triage_on:
                features = []
                if review_on:
                    features.append("PR review")
                if triage_on:
                    features.append("issue triage")
                log.warning(
                    "GitHub features enabled (%s) but no users have "
                    "github_repos configured. GitHub webhook events will "
                    "not be delivered to these features. Add 'github_repos' "
                    "to users.yaml entries. See: https://github.com/"
                    "dcellison/kai/wiki/Multi-User-Setup"
                    "#what-you-must-set-manually",
                    ", ".join(features),
                )

    # Read DEFAULT_MODEL, falling back to CLAUDE_MODEL for backward
    # compat with existing /etc/kai/env files that haven't been
    # regenerated via make config.
    default_model = os.environ.get(
        "DEFAULT_MODEL",
        os.environ.get("CLAUDE_MODEL", "sonnet"),
    )
    if os.environ.get("CLAUDE_MODEL") and not os.environ.get("DEFAULT_MODEL"):
        log.warning("CLAUDE_MODEL is deprecated, use DEFAULT_MODEL instead")

    # Validate DEFAULT_MODEL against the effective global backend.
    # Codex installs validate against CODEX_MODELS only - no fallback
    # to PROVIDER_MODELS["openai"]. Other backends still use the
    # provider-only validator. Catches typos at startup instead of
    # letting them propagate to a confusing runtime failure.
    global_provider = get_effective_provider(agent_backend, llm_provider)
    if not validate_model_for_backend(default_model, agent_backend, global_provider):
        if agent_backend == "codex":
            valid = sorted(CODEX_MODELS.keys())
            raise SystemExit(
                f"DEFAULT_MODEL '{default_model}' is not valid for codex (must be one of: {', '.join(valid)})"
            )
        valid = sorted(PROVIDER_MODELS.get(global_provider, {}).keys())
        raise SystemExit(
            f"DEFAULT_MODEL '{default_model}' is not valid for provider "
            f"'{global_provider}' (must be one of: {', '.join(valid)})"
        )

    return Config(
        telegram_bot_token=token,
        telegram_webhook_url=telegram_webhook_url,
        telegram_webhook_secret=telegram_webhook_secret,
        allowed_user_ids=allowed_ids,
        default_model=default_model,
        claude_timeout_seconds=claude_timeout_seconds,
        budget_ceiling=budget_ceiling,
        claude_max_session_hours=claude_max_session_hours,
        claude_idle_timeout=claude_idle_timeout,
        claude_max_context_window=claude_max_context_window,
        claude_autocompact_pct=claude_autocompact_pct,
        claude_effort_level=claude_effort_level,
        codex_auth_mode=codex_auth_mode,
        webhook_port=webhook_port,
        webhook_secret=os.environ.get("WEBHOOK_SECRET", ""),
        voice_enabled=os.environ.get("VOICE_ENABLED", "").lower() in ("1", "true", "yes"),
        tts_enabled=os.environ.get("TTS_ENABLED", "").lower() in ("1", "true", "yes"),
        workspace_base=workspace_base,
        allowed_workspaces=allowed_workspaces,
        workspace_configs=workspace_configs,
        claude_user=os.environ.get("CLAUDE_USER") or None,
        pr_review_enabled=pr_review_enabled,
        pr_review_cooldown=pr_review_cooldown,
        pr_review_timeout_s=pr_review_timeout_s,
        pr_review_budget_usd=pr_review_budget_usd,
        github_repo=os.getenv("GITHUB_REPO", ""),
        spec_dir=os.getenv("SPEC_DIR", "specs"),
        issue_triage_enabled=issue_triage_enabled,
        github_notify_chat_id=github_notify_chat_id,
        file_retention_days=file_retention_days,
        user_configs=user_configs,
        totp_session_minutes=totp_session_minutes,
        totp_challenge_seconds=totp_challenge_seconds,
        totp_lockout_attempts=totp_lockout_attempts,
        totp_lockout_minutes=totp_lockout_minutes,
        agent_backend=agent_backend,
        llm_provider=llm_provider,
        memory_enabled=memory_enabled,
        memory_search_limit=memory_search_limit,
        memory_token_budget=memory_token_budget,
        memory_embedding_model=memory_embedding_model,
        memory_extraction_enabled=memory_extraction_enabled,
        memory_extraction_budget_usd=memory_extraction_budget_usd,
        memory_extraction_timeout_s=memory_extraction_timeout_s,
        episode_classifier_context_turns=episode_classifier_context_turns,
        memory_consolidation_candidates_n=memory_consolidation_candidates_n,
        memory_episode_budget_usd=memory_episode_budget_usd,
        memory_episode_timeout_s=memory_episode_timeout_s,
        memory_search_floor=memory_search_floor,
        memory_duplicate_threshold=memory_duplicate_threshold,
    )
