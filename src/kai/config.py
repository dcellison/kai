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
import secrets
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml
from dotenv import load_dotenv

from kai.user_isolation import validate_protected_user_isolation

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
# Uses `or` so that an empty string also falls back to the project default.
DATA_DIR = Path(os.environ.get("KAI_DATA_DIR") or str(PROJECT_ROOT))

# Valid agent backend choices. "claude" uses Claude Code CLI,
# "goose" uses Goose ACP, "codex" uses OpenAI Codex CLI's app-server
# JSON-RPC protocol, "opencode" uses OpenCode's ACP (model selected
# via OPENCODE_CONFIG_CONTENT; auth managed by the operator through
# `opencode auth login`). Shared between load_config() and install.py.
VALID_BACKENDS = {"claude", "goose", "codex", "opencode"}

# Backends that ship a `OneShotReasoner` implementation in
# `src/kai/oneshot.py`. Every site that gates on "does this backend
# support one-shot agent dispatch" - memory extraction eligibility,
# PR review / triage dispatch, smoke validation, behavioral eval
# coverage, install-time MEMORY_* persistence - reads from this set
# rather than an ad-hoc literal tuple. Adding a new backend with a
# OneShotReasoner is a one-line change here; without the constant,
# widening costs a grep-and-extend dance across nine sites in
# `bot`, `install`, `smoke`, and `eval` (which is exactly what
# the opencode rollout cost across two follow-up PRs).
#
# `frozenset` because the only operation any caller needs is
# membership; the constant is immutable at module scope and the
# type pins that intent.
#
# Subset of VALID_BACKENDS by contract: every entry here is also a
# valid agent backend, and membership asserts that
# `src/kai/oneshot.py` ships a OneShotReasoner for it. Today every
# valid backend qualifies, so the two sets happen to be equal, but
# they stay distinct constants on purpose: a future backend lands in
# VALID_BACKENDS first and joins this set only when its reasoner
# exists, and every extraction-eligibility gate (bot dispatch, config
# validation, wizard prompts, smoke, the /memory retrieval-only
# note) keys off this set rather than VALID_BACKENDS so that gap
# stays representable.
ONESHOT_REASONER_BACKENDS: frozenset[str] = frozenset({"claude", "codex", "goose", "opencode"})

# The single authoritative (backend, provider) allowlist. Every site
# that needs to know "what providers can this backend talk to" reads
# from this map. Claude and codex are 1:1 with one provider each;
# opencode and goose are 1:N. load_config validates that every user's
# (backend, provider) pair is a member. Adding a new
# backend is one row here; adding a new provider to an existing
# backend is one tuple element. Tuple values (not frozensets) so the
# wizard offers providers in a stable, documented order; sorted
# alphabetically to match how operator-facing error messages render
# them.
BACKEND_PROVIDERS: dict[str, tuple[str, ...]] = {
    "claude": ("anthropic",),
    "codex": ("openai",),
    "opencode": ("anthropic", "deepseek", "google", "ollama", "openai", "openrouter"),
    "goose": ("anthropic", "deepseek", "google", "ollama", "openai", "openrouter"),
}

# Backends whose runtime configuration requires the operator (or
# users.yaml) to name a provider explicitly. Derived from the
# multiplicity of BACKEND_PROVIDERS: a single-provider backend
# (claude, codex) does not need a provider prompt because the choice
# is unambiguous and the per-backend code path hardcodes the provider
# (claude through get_effective_provider, codex always openai). A
# multi-provider backend (opencode, goose) does need an explicit
# provider so the (backend, provider, role) registry triple-key
# lookup can find a row.
#
# Kept distinct from BACKEND_PROVIDERS so claude / codex stay out of
# the "requires provider" gates that drive the wizard provider-prompt,
# the per-user `provider` validation in _load_user_configs, and
# the global DEFAULT_PROVIDER env-var validation in load_config. Membership
# in BACKEND_PROVIDERS describes what is allowed; membership here
# describes what is required.
BACKENDS_NEEDING_PROVIDER_PROMPT: frozenset[str] = frozenset(b for b, ps in BACKEND_PROVIDERS.items() if len(ps) > 1)

# Maps LLM provider to its API key environment variable name.
# Backend-agnostic - the API key for Anthropic is the same env var
# regardless of which backend is using it. DEEPSEEK_API_KEY is the
# var goose's declarative DeepSeek provider names as its api_key_env
# (and the one the per-backend env allowlists / preserve lists
# already forward), so deepseek-on-goose key collection routes
# through the same map as every other provider.
# Ollama is absent because it requires no API key (local inference).
PROVIDER_KEY_VARS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

# ── Provider-aware model registry ──────────────────────────────────────

# Maps provider name to a dict of model_key -> display_name. Keys are
# the identifiers passed to GOOSE_MODEL (for Goose) or --model (for
# Claude CLI). Values are display names for the Telegram keyboard.
# Goose passes model IDs through verbatim to the provider API - no
# aliasing layer - so these must be the exact strings the APIs accept.
PROVIDER_MODELS: dict[str, dict[str, str]] = {
    # Claude CLI accepts short aliases that auto-resolve to the
    # current SKU (`opus` -> claude-opus-4-8 as of 2026-06-09).
    # Listing the aliases here keeps the registry stable across
    # Anthropic version bumps; the CLI handles the resolution.
    # Claude Fable 5 is Anthropic's strongest widely-released model
    # today but is not yet wired as a Claude CLI short alias, so
    # the keyboard surface stays at opus/sonnet/haiku.
    "anthropic": {
        "opus": "\U0001f9e0 Opus",
        "sonnet": "\u26a1 Sonnet",
        "haiku": "\U0001fab6 Haiku",
    },
    # OpenAI current API surface (verified 2026-06-09 against
    # developers.openai.com/api/docs/models). gpt-5.5-pro is the
    # single strongest text model; gpt-5.5 is the standard frontier;
    # the 5.4 family covers cheaper / cost-tier workloads. 5.3 and
    # earlier are retired from this surface.
    "openai": {
        "gpt-5.5-pro": "\U0001f7e3 GPT-5.5 Pro",
        "gpt-5.5": "\U0001f7e2 GPT-5.5",
        "gpt-5.4-pro": "\U0001f7e1 GPT-5.4 Pro",
        "gpt-5.4": "\U0001f7e1 GPT-5.4",
        "gpt-5.4-mini": "\U0001f7e0 GPT-5.4 Mini",
        "gpt-5.4-nano": "\U0001f535 GPT-5.4 Nano",
    },
    # Google Gemini current API surface (verified 2026-06-09 against
    # ai.google.dev/gemini-api/docs/models). gemini-2.5-pro is the
    # most advanced for complex tasks per the docs; the 2.5 Flash
    # family covers speed and cost. The 3.x family exists but is
    # mostly Preview variants and specialized SKUs (image / live /
    # TTS); the curated keyboard stays on the stable 2.5 trio.
    "google": {
        "gemini-2.5-pro": "\u264a Gemini 2.5 Pro",
        "gemini-2.5-flash": "\u264a Gemini 2.5 Flash",
        "gemini-2.5-flash-lite": "\u264a Gemini 2.5 Flash Lite",
    },
    # DeepSeek's first-class OpenCode / Models.dev surface is exactly
    # these two V4 SKUs. The legacy `deepseek-chat` and
    # `deepseek-reasoner` aliases (which mapped to V4 Flash's
    # non-thinking and thinking modes respectively) are deprecated by
    # DeepSeek and scheduled to retire 2026-07-24; do not list them
    # here. Thinking-vs-non-thinking is a runtime mode flag handled
    # by opencode's `options.reasoning.enabled`, not a model-name
    # distinction at this layer.
    "deepseek": {
        "deepseek-v4-pro": "DeepSeek V4 Pro",
        "deepseek-v4-flash": "DeepSeek V4 Flash",
    },
}

# Default model for each provider, used as the wizard prompt
# suggestion for the conversational role (DEFAULT_MODEL / per-user
# `models.agent`). The conversational role does the most work and
# the heaviest work in the Kai system; defaulting to a balanced or
# speed-tier model on this surface underspends on the highest-value
# path. Each entry below is the strongest curated model the provider
# offers. Open-ended providers (openrouter, ollama) have no entry;
# users on those providers MUST set a model explicitly.
#
# Non-agent roles (PR review, issue triage, memory extraction,
# memory episode, behavioral judge, behavioral gen) use the tier
# scheme in `_BACKEND_PROVIDER_TIER_MODELS` instead, where balanced
# and cheap tiers split per-role per-(backend, provider).
PROVIDER_DEFAULTS: dict[str, str] = {
    "anthropic": "opus",
    "openai": "gpt-5.5-pro",
    "google": "gemini-2.5-pro",
    "deepseek": "deepseek-v4-pro",
}

# Codex CLI model surface. Independent of PROVIDER_MODELS["openai"]:
# codex CLI exposes a different set than the OpenAI HTTP API surface
# goose drives, and treating them as one list lets a goose-only
# model (e.g. gpt-5.4-nano) leak onto a codex install where the CLI
# rejects it. GPT-5.6 has three subscription-Codex model IDs; the
# family shorthand is not accepted by the ChatGPT-account service
# even though some Codex documentation and CLI examples use it.
# Verified against Codex app-server model/list on 2026-08-07. Refresh
# when the codex CLI bumps; no auto-discovery.
CODEX_MODELS: dict[str, str] = {
    "gpt-5.6-sol": "\U0001f7e2 GPT-5.6 Sol",
    "gpt-5.6-terra": "\U0001f7e1 GPT-5.6 Terra",
    "gpt-5.6-luna": "\U0001f535 GPT-5.6 Luna",
    "gpt-5.5": "\U0001f7e2 GPT-5.5",
    "gpt-5.4": "\U0001f7e1 GPT-5.4",
    "gpt-5.4-mini": "\U0001f535 GPT-5.4 Mini",
    "gpt-5.3-codex": "\U0001f7e0 GPT-5.3 Codex",
    "gpt-5.3-codex-spark": "⚡ GPT-5.3 Codex Spark",
    "gpt-5.2": "\U0001f7e4 GPT-5.2",
}

# One-release compatibility for the invalid GPT-5.6 family shorthand
# Kai briefly offered. Keep aliases out of CODEX_MODELS so new model
# pickers and validators expose only IDs accepted by Codex. Callers
# that ingest persisted operator state canonicalize before validating.
_LEGACY_CODEX_MODEL_ALIASES: dict[str, str] = {
    "gpt-5.6": "gpt-5.6-sol",
}


def canonicalize_model_for_backend(model: str, backend: str) -> str:
    """Map a retired model spelling to the exact ID its backend accepts."""
    if backend == "codex":
        return _LEGACY_CODEX_MODEL_ALIASES.get(model, model)
    return model


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
# to actually USE the override for its surface. Legacy Codex aliases
# remain loadable for one release and are canonicalized at apply time.
_ALL_CURATED_MODELS: frozenset[str] = frozenset(
    list(model for models in PROVIDER_MODELS.values() for model in models)
    + list(CODEX_MODELS.keys())
    + list(_LEGACY_CODEX_MODEL_ALIASES.keys())
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


# Per-role tier assignment. Roles that need reasoning depth (PR
# review, issue triage, behavioral gen) pick the "balanced" tier;
# roles that need cheap high-volume throughput (memory extraction,
# memory episode, behavioral judge) pick "cheap". Two tiers cover
# every current role; adding a third (e.g. "strong" for deeper
# reasoning) means one line here plus a per-(backend, provider)
# tier-map entry in lockstep.
_TIER_BY_ROLE: dict[ModelRole, str] = {
    ModelRole.PR_REVIEW: "balanced",
    ModelRole.ISSUE_TRIAGE: "balanced",
    ModelRole.MEMORY_EXTRACTION: "cheap",
    ModelRole.MEMORY_EPISODE: "cheap",
    ModelRole.BEHAVIORAL_JUDGE: "cheap",
    ModelRole.BEHAVIORAL_GEN: "balanced",
}

# Per-(backend, provider) tier-to-model map. Names are exactly what
# each backend's CLI accepts as --model. The backend axis matters
# because the same underlying model is named differently across
# backends:
#  - claude CLI takes anthropic aliases ("sonnet", "haiku")
#  - codex CLI takes its own model identifiers ("gpt-5.4-mini")
#  - opencode takes full "provider/model" strings, structurally
#    validated by is_opencode_model_shape
#  - goose takes the provider's native model name verbatim
# Cheap-tier picks must be in CODEX_MODELS for the codex backend
# (the startup completeness check validates this); the balanced
# tier picks the current frontier where the CLI surface allows.
# Every (backend, provider) pair in BACKEND_PROVIDERS must have a
# row here; _build_registry raises KeyError at module import time
# on any missing pair.
_BACKEND_PROVIDER_TIER_MODELS: dict[tuple[str, str], dict[str, str]] = {
    # Claude CLI accepts short aliases that auto-resolve to the
    # current SKU; `sonnet` -> claude-sonnet-4-6 and `haiku` ->
    # claude-haiku-4-5 as of 2026-06-09. Keep the dated haiku ID for
    # cheap-tier roles where a fully-pinned model version is
    # preferable so an automated stage-2 episode generation stays
    # reproducible across CLI version bumps.
    ("claude", "anthropic"): {"balanced": "sonnet", "cheap": "claude-haiku-4-5-20251001"},
    # Codex CLI accepts the OpenAI model surface. gpt-5.5 is the
    # current frontier reasoning model; gpt-5.4-mini stays as the
    # cheap tier for high-volume roles (memory extraction, episode
    # generation, behavioral judge).
    ("codex", "openai"): {"balanced": "gpt-5.5", "cheap": "gpt-5.4-mini"},
    # OpenCode/Anthropic surface uses the `anthropic/<model>`
    # provider-prefixed shape; the model halves match the current
    # Claude SKUs (Sonnet 4.6 and Haiku 4.5 as of 2026-06-09).
    ("opencode", "anthropic"): {"balanced": "anthropic/claude-sonnet-4-6", "cheap": "anthropic/claude-haiku-4-5"},
    # OpenCode/OpenAI: same tier split as codex/openai but in the
    # opencode provider-prefixed shape.
    ("opencode", "openai"): {"balanced": "openai/gpt-5.5", "cheap": "openai/gpt-5.4-mini"},
    # DeepSeek V4 first-class OpenCode surface (V4 Pro and V4 Flash).
    # The legacy `deepseek-chat` / `deepseek-reasoner` aliases (mode
    # flags on V4 Flash) retire 2026-07-24, so do not use them in
    # registry defaults. Pro for reasoning-heavy roles (the
    # balanced tier covers review / triage / behavioral_gen); Flash
    # for high-volume / latency-sensitive roles (the cheap tier
    # covers memory extraction / memory episode / behavioral_judge).
    # Same 1M context on both; the split is capability-vs-cost.
    ("opencode", "deepseek"): {"balanced": "deepseek/deepseek-v4-pro", "cheap": "deepseek/deepseek-v4-flash"},
    # Google's most advanced for complex tasks per the Gemini API
    # docs (2026-06-09) is gemini-2.5-pro; gemini-2.5-flash is the
    # speed/cost tier. The 3.x family is mostly Preview / specialized.
    ("opencode", "google"): {"balanced": "google/gemini-2.5-pro", "cheap": "google/gemini-2.5-flash"},
    # OpenRouter's "provider/model" shape nests under opencode's own
    # "provider/model" prefix; the structural check accepts this
    # because the outer slash is what matters to opencode.
    ("opencode", "openrouter"): {
        "balanced": "openrouter/anthropic/claude-sonnet-4-6",
        "cheap": "openrouter/anthropic/claude-haiku-4-5",
    },
    ("opencode", "ollama"): {"balanced": "ollama/llama4:70b", "cheap": "ollama/llama4:8b"},
    ("goose", "anthropic"): {"balanced": "claude-sonnet-4-6", "cheap": "claude-haiku-4-5"},
    ("goose", "openai"): {"balanced": "gpt-5.5", "cheap": "gpt-5.4-mini"},
    # goose-on-deepseek: bare model names; goose passes them through
    # to the provider API directly (no opencode-style "provider/"
    # prefix). Same Pro / Flash split as opencode-on-deepseek above;
    # same 2026-07-24 deprecation reason for skipping the legacy
    # `deepseek-chat` alias.
    ("goose", "deepseek"): {"balanced": "deepseek-v4-pro", "cheap": "deepseek-v4-flash"},
    ("goose", "google"): {"balanced": "gemini-2.5-pro", "cheap": "gemini-2.5-flash"},
    ("goose", "openrouter"): {
        "balanced": "openrouter/anthropic/claude-sonnet-4-6",
        "cheap": "openrouter/anthropic/claude-haiku-4-5",
    },
    ("goose", "ollama"): {"balanced": "llama4:70b", "cheap": "llama4:8b"},
}


def _build_registry() -> dict[tuple[str, str, ModelRole], str]:
    """Fan out the (backend, provider) tier maps over every ModelRole.

    Raises KeyError at module import time if any (backend, provider)
    pair in BACKEND_PROVIDERS lacks a tier map; the precise pair name
    appears in the exception message so the maintainer can fix the
    omission without grepping. This intentionally fails fast rather
    than silently skipping, because a partial registry would
    surface as a confusing per-request LookupError much later.
    """
    rows: dict[tuple[str, str, ModelRole], str] = {}
    for backend, providers in BACKEND_PROVIDERS.items():
        for provider in providers:
            try:
                tier_map = _BACKEND_PROVIDER_TIER_MODELS[(backend, provider)]
            except KeyError:
                raise KeyError(
                    f"_BACKEND_PROVIDER_TIER_MODELS missing entry for "
                    f"(backend={backend!r}, provider={provider!r}); update "
                    f"src/kai/config.py to include this pair."
                ) from None
            for role, tier in _TIER_BY_ROLE.items():
                rows[(backend, provider, role)] = tier_map[tier]
    return rows


# (backend, provider, role) -> model identifier passed to the
# backend's CLI as --model. Built mechanically from the tier scheme
# above so a new (backend, provider) pair extends the registry via
# one row in _BACKEND_PROVIDER_TIER_MODELS plus one tuple element in
# BACKEND_PROVIDERS.
MODEL_REGISTRY: dict[tuple[str, str, ModelRole], str] = _build_registry()


def get_model_for(role: ModelRole, backend: str, provider: str, override: str = "") -> str:
    """
    Resolve the model identifier for a (role, backend, provider) triple.

    Override precedence: if `override` is truthy, it wins (used by
    CLI flags like --judge-model in the behavioral eval). Otherwise
    the registry value for the exact triple is returned.

    For single-provider backends (claude, codex), an empty `provider`
    string is resolved to the implicit provider (anthropic / openai)
    via `get_effective_provider`. This lets callers that do not have
    a provider in scope (e.g. eval-time globals) still hit the
    registry without an explicit lookup; multi-provider backends
    (opencode, goose) require an explicit non-empty provider because
    the implicit value is undefined.

    Raises LookupError on a missing (backend, provider, role) row.
    Total in the steady state because _check_model_registry_complete()
    runs at startup and fails fast (SystemExit) if any (backend,
    provider) pair in BACKEND_PROVIDERS is missing a role.

    Args:
        role: The ModelRole the caller wants a model for.
        backend: The active backend string (one of VALID_BACKENDS).
        provider: The effective provider string; empty falls back to
            the backend's implicit provider on single-provider backends.
        override: Caller-supplied override string. Empty disables.

    Returns:
        The model identifier to pass to the backend's CLI.
    """
    if override:
        return override
    effective_provider = provider or get_effective_provider(backend, provider)
    try:
        return MODEL_REGISTRY[(backend, effective_provider, role)]
    except KeyError:
        raise LookupError(
            f"No registry entry for (backend={backend}, provider={effective_provider}, role={role.value})"
        ) from None


def _check_model_registry_complete() -> None:
    """
    Validate MODEL_REGISTRY against per-backend invariants at startup.

    Self-driving: walks every (backend, provider) pair in
    BACKEND_PROVIDERS. Runs once at load_config() time. Raises
    SystemExit on any invariant violation so the bug surfaces at
    startup rather than as a per-request LookupError or a per-request
    OneShot handshake failure.

    Two layers of validation, only one of which is load-bearing for
    production code:

    1. Triple-key completeness (defense-in-depth). `_build_registry`
       already guarantees every (backend, provider, role) triple in
       BACKEND_PROVIDERS has a row, raising KeyError at module import
       time if a `(backend, provider)` pair lacks a tier map or if
       any `_TIER_BY_ROLE` tier is missing. The completeness loop
       below catches post-construction mutation (test fixtures that
       monkeypatch `delitem` on MODEL_REGISTRY) and the
       theoretical case where a future refactor decouples
       `_build_registry` from `BACKEND_PROVIDERS`. In production this
       branch is unreachable.

    2. Per-backend value shape (the real production guarantee):
       - codex rows must name models the codex CLI exposes (validated
         against CODEX_MODELS); a drift between CODEX_MODELS and the
         tier map fails here, not at first behavioral run.
       - opencode rows must pass `is_opencode_model_shape`; bare names
         like "sonnet" that would survive registry construction but
         fail at the opencode handshake are caught here.
    """
    for backend, providers in BACKEND_PROVIDERS.items():
        for provider in providers:
            missing = [role for role in ModelRole if (backend, provider, role) not in MODEL_REGISTRY]
            if missing:
                names = ", ".join(role.value for role in missing)
                raise SystemExit(
                    f"MODEL_REGISTRY is missing rows for "
                    f"(backend='{backend}', provider='{provider}'): {names}. "
                    f"Update _BACKEND_PROVIDER_TIER_MODELS in config.py."
                )
            # Codex CLI model surface check. A future drift (operator
            # updates CODEX_MODELS without touching the tier map, or
            # vice versa) fails fast at startup.
            if backend == "codex":
                invalid_codex = [
                    (role, MODEL_REGISTRY[(backend, provider, role)])
                    for role in ModelRole
                    if MODEL_REGISTRY[(backend, provider, role)] not in CODEX_MODELS
                ]
                if invalid_codex:
                    valid_list = ", ".join(sorted(CODEX_MODELS.keys()))
                    details = ", ".join(f"{role.value}={model}" for role, model in invalid_codex)
                    raise SystemExit(
                        f"MODEL_REGISTRY has codex rows naming models the codex CLI does not expose: "
                        f"{details}. Valid codex models: {valid_list}. "
                        f"Update _BACKEND_PROVIDER_TIER_MODELS in config.py."
                    )
            # OpenCode "provider/model" shape check. OpenCode has no
            # canonical model allow-list (75+ providers via AI SDK /
            # Models.dev resolved at runtime against the operator's
            # `opencode auth login` state); what is checkable upfront
            # is the shape contract.
            if backend == "opencode":
                invalid_opencode = [
                    (role, MODEL_REGISTRY[(backend, provider, role)])
                    for role in ModelRole
                    if not is_opencode_model_shape(MODEL_REGISTRY[(backend, provider, role)])
                ]
                if invalid_opencode:
                    details = ", ".join(f"{role.value}={model}" for role, model in invalid_opencode)
                    raise SystemExit(
                        f"MODEL_REGISTRY has opencode rows that are not in `provider/model` "
                        f"shape: {details}. Update _BACKEND_PROVIDER_TIER_MODELS in config.py."
                    )


def get_effective_provider(backend: str, llm_provider: str) -> str:
    """Derive the effective provider from backend + llm_provider.

    Claude is always anthropic; codex is always openai (codex CLI has
    no other provider surface, so a wizard-generated codex install with
    LLM_PROVIDER unset still needs the runtime to know its provider is
    openai). Goose consults the raw llm_provider because it routes to
    whatever provider the user configured. OpenCode also returns the
    raw llm_provider: the provider lives inside the full `provider/model`
    string OpenCode resolves at runtime, so Kai's effective_provider
    is informational (used for shadow recall logging metadata and the
    /stats display) rather than load-bearing for OpenCode dispatch.

    Kept identical to the backend->provider rule
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

# Valid values for the codex `model_reasoning_effort` config override,
# taken verbatim from the upstream codex configuration reference
# (xhigh is model-dependent; the override applies to Responses-API
# models). Same two-shape pattern as EFFORT_LEVELS above: ordered
# tuple for operator-facing listings in intensity order, derived
# frozenset for O(1) membership checks. The vocabularies are
# CLI-specific and deliberately not shared: codex has "minimal" and
# no "max"; claude has "max" and no "minimal".
CODEX_EFFORT_LEVELS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh")
_VALID_CODEX_EFFORT_LEVELS: frozenset[str] = frozenset(CODEX_EFFORT_LEVELS)


def validate_model_for_provider(model: str, provider: str) -> bool:
    """Check if a model is valid for a provider.

    Returns True if the provider is open-ended (accepts any model)
    or if the model is in the provider's curated list. Unknown providers
    (not in PROVIDER_MODELS or OPEN_ENDED_PROVIDERS) are accepted with
    a warning; this catches the case where BACKEND_PROVIDERS gains a
    new entry but PROVIDER_MODELS was not updated to match.

    Backend-aware callers should use `validate_model_for_backend`. This
    function stays as the implementation goose / claude delegate to.
    """
    if provider in OPEN_ENDED_PROVIDERS:
        return True
    models = PROVIDER_MODELS.get(provider)
    if models is None:
        # Provider is in BACKEND_PROVIDERS but has no curated model
        # list and is not explicitly open-ended. This is a programming
        # oversight; log it so the gap is visible at runtime.
        if provider:
            log.warning(
                "Provider '%s' has no entry in PROVIDER_MODELS or OPEN_ENDED_PROVIDERS; accepting model '%s' unchecked",
                provider,
                model,
            )
        return True
    return model in models


def is_opencode_model_shape(model: str) -> bool:
    """Structural check: OpenCode model IDs are `provider_id/model_id`.

    OpenCode resolves models against its own provider registry at
    runtime; Kai cannot enumerate the supported set (75+ providers
    via AI SDK and Models.dev, varying by what the operator has
    authenticated through `opencode auth login`). What IS knowable
    upfront is the structural contract documented at
    https://opencode.ai/docs/models/: every accepted model string
    splits on `/` into exactly two non-empty segments.

    Rejecting strings that violate that contract catches the common
    operator footgun where a bare Anthropic name like `sonnet` or
    `opus` (correct for the claude / goose / codex surfaces) is
    typed into `/model` on an opencode install. Without this check,
    such a value persists through DB write -> pool restore ->
    OpenCodeBackend.build_env, where it becomes
    `OPENCODE_CONFIG_CONTENT='{"model": "sonnet"}'` and OpenCode
    fails model resolution at handshake time with no clear pointer
    back to the Kai-side typo.
    """
    if not model:
        return False
    # Split on every "/" and require at least two non-empty segments.
    # The first segment is opencode's provider prefix; remaining
    # segments form the model_id, which itself may contain slashes
    # for openrouter-style nesting like
    # "openrouter/anthropic/claude-sonnet-4-5". Empty segments
    # (leading slash, trailing slash, or "foo//bar" double-slash)
    # all fail because the opencode provider resolver expects every
    # path segment to be non-empty.
    parts = model.split("/")
    if len(parts) < 2:
        return False
    return all(parts)


def validate_model_for_backend(model: str, backend: str, eff_provider: str) -> bool:
    """Check if a model is valid for the active backend.

    Codex installs validate against CODEX_MODELS only - no fallback to
    PROVIDER_MODELS["openai"] or any other provider surface. The
    backend determines the model surface 1:1 for codex (which speaks
    its own CLI's curated set); other backends delegate to the existing
    provider-only validator, which is unchanged for goose / claude.

    OpenCode validates STRUCTURALLY: model strings must match
    `provider/model` (both segments non-empty). The supported provider/
    model SET is not curated here - OpenCode is the source of truth on
    which IDs resolve - but the structural contract blocks the obvious
    operator typo (bare Anthropic names like "opus" or "sonnet" that
    are correct on other backends but unusable on OpenCode).

    The claude backend and goose-on-anthropic additionally accept
    any `claude-*` ID structurally, beyond the curated alias trio.
    Both hand the model string verbatim to a surface that resolves
    full IDs (the claude CLI's --model flag; the Anthropic API via
    GOOSE_MODEL), so the curated aliases must never be a ceiling on
    which SKUs a user can reach: pinning a previous generation (e.g.
    claude-opus-4-7) is a real operator need when the newest SKU
    misbehaves. A bogus claude-* string fails at the CLI or provider
    with a clear error on the next message, the same contract
    opencode accepts for its structurally-validated IDs.

    Canonical model validator: every model-selection site in the
    codebase routes through this function so codex / goose / opencode
    share no fallback path.
    """
    if backend == "codex":
        return model in CODEX_MODELS
    if backend == "opencode":
        return is_opencode_model_shape(model)
    if backend == "claude" and model.startswith("claude-"):
        return True
    if backend == "goose" and eff_provider == "anthropic" and model.startswith("claude-"):
        return True
    return validate_model_for_provider(model, eff_provider)


# (context, matched legacy key) pairs that have already emitted a
# renamed-key deprecation warning. Keyed by both halves so a given
# source (the env file, a specific user's yaml entry, install.conf)
# warns at most once per legacy key per process instead of once per
# lookup, and a multi-key fallback chain warns distinctly for each
# legacy name it encounters.
_renamed_key_deprecation_warned: set[tuple[str, str]] = set()


def _resolve_renamed_key(
    get: Callable[[str], str | None],
    *,
    new_key: str,
    legacy_keys: list[str],
    context: str,
    default: str | None,
) -> str | None:
    """Read a config value, preferring the new key over legacy names.

    Centralizes the one-release back-compat window for a key rename.
    Every reader (env, install.conf dict, per-user yaml entry) passes
    its own lookup callable plus the key casing for its source so a
    single implementation serves all of them.

    `legacy_keys` is ordered most-recent-first: the first legacy key
    that is present wins. A key that has had more than one rename (the
    per-user backend went `agent_backend` -> `default_backend` ->
    `backend`) lists every prior name so an install carrying any of
    them still resolves.

    Absence semantics differ by caller and are made explicit through
    `default`: global readers pass `default="claude"` (the
    installation default); the per-user yaml reader passes
    `default=None` so a backend-less user still inherits the global
    backend via the caller's `user_backend or global_backend` cascade.
    Returning "claude" for an absent per-user key would wrongly pin
    that user to claude on a non-claude install.

    Args:
        get: Lookup callable taking a key and returning value-or-None
            (os.environ.get, env_dict.get, entry.get).
        new_key: The current key name (DEFAULT_BACKEND / backend).
        legacy_keys: Deprecated key names, most-recent first
            (e.g. ["default_backend", "agent_backend"]).
        context: Human-readable source name for the one-shot
            deprecation log line (e.g. "/etc/kai/env", "install.conf",
            "users.yaml entry for <user>").
        default: Value to return when no key is set.

    Returns:
        The resolved string, or `default` when no key is present. Does
        not validate the value; callers keep their existing validation.
    """
    new_value = get(new_key)
    if new_value is not None:
        return new_value
    for old_key in legacy_keys:
        old_value = get(old_key)
        if old_value is not None:
            warned = (context, old_key)
            if warned not in _renamed_key_deprecation_warned:
                _renamed_key_deprecation_warned.add(warned)
                log.warning(
                    "%s is deprecated; rename to %s (found in %s)",
                    old_key,
                    new_key,
                    context,
                )
            return old_value
    return default


def _resolve_eval_provider(context: str) -> str:
    """Eval-time provider, read from the process env.

    The developer eval tools (behavioral eval, memory backend gate) run
    against the operator's configured provider rather than a sandboxed
    user, so they read the same global env `load_config` does: the
    canonical DEFAULT_PROVIDER with a one-release fallback to the
    deprecated LLM_PROVIDER name. Single-provider backends (claude,
    codex) ignore the result because their provider is implicit. Shared
    so both eval entry points resolve the provider identically and
    neither bypasses the rename window. `context` names the caller for
    the one-shot deprecation log line.
    """
    return (
        (
            _resolve_renamed_key(
                os.environ.get,
                new_key="DEFAULT_PROVIDER",
                legacy_keys=["LLM_PROVIDER"],
                context=context,
                default="",
            )
            or ""
        )
        .strip()
        .lower()
    )


def models_for_backend(agent_backend: str, eff_provider: str) -> dict[str, str] | None:
    """Curated model list for the given (backend, provider) pair.

    Codex consults its own CODEX_MODELS surface; OpenCode and the
    open-ended providers return None (caller falls back to a free-text
    prompt rather than a fixed-choice keyboard). Used by both
    install.py (wizard prompt + apply-time validator) and bot.py
    (/model keyboard + selection validator).

    OpenCode returns None unconditionally: model strings are full
    `provider/model` IDs and the supported set depends on which
    providers the operator authenticated via `opencode auth login`.
    A curated keyboard would mislead users into picking IDs their
    OpenCode install cannot resolve.
    """
    if agent_backend == "codex":
        return CODEX_MODELS
    if agent_backend == "opencode":
        return None
    if eff_provider in OPEN_ENDED_PROVIDERS:
        return None
    return PROVIDER_MODELS.get(eff_provider)


def get_user_backend_and_provider(user_config: "UserConfig | None", config: "Config") -> tuple[str, str]:
    """Resolve (backend, provider) for a chat_id with the per-user cascade.

    Pool.py and bot.py already implement this cascade (user.backend
    > config.default_backend, similar for provider); this function
    consolidates it in one place so every runtime model-validation site
    sees the same effective backend for a given user. Backend determines
    provider 1:1 for codex (openai) and claude (anthropic); goose and
    opencode both consult the raw provider cascade (opencode's
    value is informational - the real provider/model resolution lives
    inside OpenCode's full `provider/model` string at runtime).
    """
    backend = user_config.backend if user_config and user_config.backend else config.default_backend
    if backend == "claude":
        provider = "anthropic"
    elif backend == "codex":
        provider = "openai"
    else:
        provider = user_config.provider if user_config and user_config.provider else config.default_provider
    return backend, provider


def resolve_user_model(
    role: ModelRole,
    user_config: "UserConfig | None",
    config: "Config",
    *,
    backend: str | None = None,
    provider: str | None = None,
) -> str:
    """Per-role model resolution with the per-user `models:` override.

    Precedence (highest first):
        1. user_config.models[role.value] when present (per-user from
           users.yaml; also seeded from legacy PR_REVIEW_MODEL_* /
           ISSUE_TRIAGE_MODEL_* env vars at load time)
        2. config.default_models[role.value] when present (global from
           DEFAULT_MODELS_JSON in /etc/kai/env; captured by the wizard)
        3. MODEL_REGISTRY[(backend, provider, role)] (curated default)

    The user's per-role override wins so an operator who sets
    `models.pr_review: deepseek/deepseek-coder` for one user does not
    inherit the install-wide value. The global default sits between
    per-user and the registry so the wizard's per-role customization
    applies across users who have not set their own override.
    Single canonical resolver used by every per-user dispatch site;
    callers that need a raw registry lookup (no user context) call
    `get_model_for` directly.

    `backend` and `provider` are optional pre-resolved values. Callers
    that already computed the effective backend / provider (memory
    extraction threads them through both stages of the dispatch
    pipeline) pass them as kwargs to skip the re-resolution. Either
    both are passed or neither; the helper falls through to
    `get_user_backend_and_provider` when they are absent.
    """
    if user_config is not None and user_config.models:
        override = user_config.models.get(role.value, "")
        if override:
            return override
    if config.default_models:
        global_override = config.default_models.get(role.value, "")
        if global_override:
            return global_override
    if backend is None or provider is None:
        backend, provider = get_user_backend_and_provider(user_config, config)
    return get_model_for(role, backend, provider)


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
        timeout: Per-readline timeout in seconds.
        env: Inline environment variables for the Claude subprocess.
        env_file: Path to a KEY=VALUE file to load as environment vars.
        system_prompt: Inline system prompt text.
        system_prompt_file: Path to a file containing the system prompt.
    """

    path: Path
    model: str | None = None
    timeout: int | None = None
    env: dict[str, str] | None = None
    env_file: Path | None = None
    system_prompt: str | None = None
    system_prompt_file: Path | None = None


# ── Memory project registry ──────────────────────────────────────────


@dataclass(frozen=True)
class MemoryProjectConfig:
    """
    Per-project memory configuration loaded from memory-projects.yaml.

    The memory project registry answers "what memory authority
    boundary does this directory belong to?" - a separate question
    from WorkspaceConfig's "how should the backend run in this
    directory?" The two surfaces overlap on paths but not on meaning;
    keeping them separate avoids accidentally enabling project memory
    just because a workspace has a prompt or model override.

    Attributes:
        project_id: Stable identifier used as the registry key and as
            the value stored in memory rows' `project_id` field.
            Normalized at load time by stripping surrounding
            whitespace.
        display_name: Human-readable name used in logs and future
            operator UI.
        workspace_roots: Canonical resolved absolute paths whose
            descendants belong to this project. Stored as a tuple
            because MemoryProjectConfig is frozen and equality should
            treat root order as significant for the longest-prefix
            tie-breaker.
        memory_enabled: When False, the project is still detectable
            for logs and diagnostics, but later retrieval and write
            paths treat it as global-only.
        default_scope_for_new_facts: Optional policy hint for the
            future write-scope routing path. Only `kai.memory.SCOPE_GLOBAL`
            or `kai.memory.SCOPE_PROJECT` are accepted; `SCOPE_TASK` is
            not because task scope is not a write target yet. The
            field is inert until the write-routing issue lands.
    """

    project_id: str
    display_name: str
    workspace_roots: tuple[Path, ...]
    memory_enabled: bool
    default_scope_for_new_facts: str | None = None


# ── Per-user configuration ──────────────────────────────────────────


@dataclass(frozen=True)
class UserConfig:
    """
    Per-user configuration loaded from users.yaml.

    Defines a user's identity, authorization, and resource limits.
    Preferences that the user controls (active model, timeout) live
    in the settings table, not here. The fields below are admin-set
    baselines that users can override within boundaries.

    Attributes:
        telegram_id: Telegram user ID (authorization key).
        name: Display name for logs and notifications.
        role: "admin" or "user". Admins receive unattributed webhooks.
        github: GitHub username for webhook actor routing.
        os_user: OS username for subprocess isolation (Phase 3).
        home_workspace: Per-user home workspace directory.
        model: Default model name (e.g., "opus", "sonnet", "haiku").
        timeout: Default timeout in seconds for Claude responses.
        workspace_base: Base directory for /workspace new and name resolution.
            Falls back to global WORKSPACE_BASE env var if not set.
        github_repos: Admin-controlled repositories this user may access
            through Kai's shared GitHub review and triage authority. The
            mutable notification subscription list cannot expand this set.
        allowed_services: External proxy service names this user's persistent
            agent may call. Empty by default, including for admins.
    """

    telegram_id: int
    name: str
    role: str = "user"
    github: str | None = None
    os_user: str | None = None
    home_workspace: Path | None = None
    model: str | None = None
    timeout: int | None = None
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
    backend: str | None = None
    provider: str | None = None
    # GitHub notification routing fields. github_repos controls which
    # repos initially route webhook events to this user and is also the
    # admin-controlled authorization boundary for review/triage operations.
    # Self-service notification subscriptions never expand that authority.
    # pr_review and issue_triage are tri-state: None = use global default,
    # True/False = admin override.
    github_repos: list[str] = field(default_factory=list)
    github_notify_chat_id: int | None = None
    pr_review: bool | None = None
    issue_triage: bool | None = None
    # External services the persistent agent may call through Kai's
    # credential-injecting proxy. Admin-controlled via users.yaml and
    # fail-closed: omission means no services, including for admins.
    allowed_services: list[str] = field(default_factory=list)
    # Per-role per-user model overrides loaded from users.yaml `models:`.
    # Keys are role identifiers: "agent" for the conversational role
    # plus every ModelRole.value ("pr_review", "issue_triage",
    # "memory_extraction", "memory_episode", "behavioral_judge",
    # "behavioral_gen"). Values are the model strings the user's
    # backend's CLI accepts; same shape rules as the global
    # MODEL_REGISTRY's (backend, provider, role) rows. Missing keys
    # fall through to the registry default.
    #
    # Back-compat: a user with legacy `model:` set and no `models:`
    # gets `models["agent"] = model_value` synthesized at load time
    # so existing users.yaml files keep working unchanged.
    models: dict[str, str] | None = None

    def authorizes_github_repo(self, repo: str) -> bool:
        """Return whether shared-identity GitHub operations may target ``repo``.

        Only the loaded, admin-controlled ``users.yaml`` baseline grants
        this authority. Database additions made through ``/github add`` are
        notification subscriptions and are deliberately not consulted here.
        GitHub owner/repository names are case-insensitive.
        """
        normalized = repo.strip().lower()
        return bool(normalized) and any(configured.strip().lower() == normalized for configured in self.github_repos)


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
            Only used in webhook mode. Generated for the process when not explicitly set.
        allowed_user_ids: Set of Telegram user IDs permitted to interact with the bot (required)
        default_model: Default model name, provider-dependent (e.g. sonnet, gpt-5.5-pro, gemini-2.5-pro)
        default_timeout: Seconds before an agent response is considered timed
            out, on every backend
        agent_max_session_hours: Hours before an agent subprocess is recycled, on every
            backend. Prevents unbounded memory growth in long-lived agent processes (the
            original motivator: V8 growth in the claude CLI triggering macOS Jetsam kernel
            panics). 0 = no limit.
        session_db_path: Path to the SQLite database for sessions, jobs, and settings
        webhook_port: Port for the local aiohttp server (webhooks + scheduling API)
        github_webhook_secret: HMAC secret for verifying GitHub webhook payloads.
        generic_webhook_secret: Header secret for the generic webhook endpoint.
        webhook_secret: Deprecated WEBHOOK_SECRET compatibility credential. During
            the migration window it is accepted only by the GitHub and generic
            external webhook routes, never by Telegram or internal APIs.
        voice_enabled: Whether to transcribe Telegram voice notes via whisper-cpp
        whisper_model_path: Path to the whisper-cpp GGML model file
        tts_enabled: Whether to enable Piper text-to-speech for voice responses
        piper_model_dir: Directory containing Piper voice model files
        workspace_base: Base directory for workspace name resolution (/workspace <name>)
        allowed_workspaces: Additional workspace directories accessible by name, from config only.
            These appear as pinned workspaces in /workspaces and are reachable via /workspace <name>
            without being under WORKSPACE_BASE. Non-existent paths are skipped at startup.
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
    # Global per-role model defaults. Parsed from DEFAULT_MODELS_JSON
    # env var at load time as a JSON object: keys are role identifiers
    # ("agent" plus every ModelRole.value), values are model strings.
    # Sits between per-user UserConfig.models (highest precedence) and
    # MODEL_REGISTRY[(backend, provider, role)] (lowest) in the
    # resolution chain that `resolve_user_model` implements.
    default_models: dict[str, str] = field(default_factory=dict)
    default_timeout: int = 120
    # Backend-generic pool lifecycle tunables, matching the AGENT_-
    # prefixed env vars they load from: every backend's subprocess is
    # recycled by age and evicted when idle.
    agent_max_session_hours: float = 0  # 0 = no limit
    agent_idle_timeout: int = 1800  # seconds before idle subprocess eviction; 0 = no eviction

    # Autocompact tuning helps control token usage on the claude
    # backend. 0 = use the Claude Code default (~83% threshold).
    claude_autocompact_pct: int = 0  # 1-100; passed as CLAUDE_AUTOCOMPACT_PCT_OVERRIDE env var

    # Effort level passed to inner Claude as `--effort <value>`. Higher
    # settings spend more reasoning tokens per turn, improving answer
    # quality at the cost of latency. Default "high" matches
    # the operator's outer-Claude default; switching to "medium" globally
    # would silently downgrade existing reasoning quality. Critical when
    # users.yaml `os_user` is set, because inner Claude then runs as a
    # different OS user and does NOT inherit the outer operator's
    # settings.json effort default - without this CLI value, user-isolated
    # installs would silently fall to whatever the claude binary picks
    # as its own default. Validated at config load against
    # _VALID_EFFORT_LEVELS so a typo fails fast rather than at the
    # subprocess startup of the next chat session.
    claude_effort_level: str = "high"

    # Codex backend auth mode. "subscription" (default): the codex CLI
    # uses ~/.codex/auth.json populated by an interactive `codex login`.
    # "api_key": codex reads OPENAI_API_KEY from the environment, the
    # same way goose+openai does. Only consulted when DEFAULT_BACKEND=codex;
    # ignored on every other backend.
    codex_auth_mode: str = "subscription"

    # Reasoning effort passed to the inner codex as a
    # `-c model_reasoning_effort=<value>` config override on the
    # app-server argv. Empty string (default) passes nothing: codex
    # then falls back to each OS user's own ~/.codex/config.toml or
    # the model default. That asymmetry with claude_effort_level
    # (which always passes a value) is deliberate: codex config is
    # per-OS-user and operator-owned, and xhigh availability is
    # model-dependent, so a global override is opt-in rather than
    # silently replacing a user's own setting. Validated at config
    # load against _VALID_CODEX_EFFORT_LEVELS so a typo fails fast.
    # Only consulted when the user's effective backend is codex.
    codex_effort_level: str = ""

    # Database - uses DATA_DIR so the db lands in the writable data directory
    session_db_path: Path = field(default_factory=lambda: DATA_DIR / "kai.db")

    # Webhook server
    webhook_port: int = 8080
    github_webhook_secret: str = ""
    generic_webhook_secret: str = ""
    # Deprecated: temporary external-only fallback for existing GitHub and
    # generic webhook callers. Never use this for Telegram or internal APIs.
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

    # Memory project registry from memory-projects.yaml. Keyed by
    # project_id. Empty dict if no config file exists. The detector
    # in kai.memory_projects consumes this; no retrieval or write
    # routing path reads it yet (those land in #544 and later).
    # Workspace access is still owned by allowed_workspaces and is
    # NOT extended by registry roots.
    memory_projects: dict[str, MemoryProjectConfig] = field(default_factory=dict)

    # Per-user OS isolation lives in `UserConfig.os_user` (loaded from
    # users.yaml). The bot spawns the inner agent via
    # `sudo -u <user>` when that field is set; when unset the agent
    # runs as the bot's own OS user.

    # PR review agent: per-user toggle lives in users.yaml `pr_review`
    # (or the per-chat /github reviews command). Global resource
    # controls stay on Config.
    # Minimum seconds between reviews of the same PR. Absorbs force-push bursts
    # so rapid pushes to an open PR don't trigger a review for each one.
    pr_review_cooldown: int = 300
    # Subprocess timeout for a single PR review, in seconds. Sonnet with
    # extended thinking on a large diff with prior review context can take
    # a long time; the default gives thinking-heavy reviews room while
    # still terminating genuinely stuck processes.
    pr_review_timeout_s: int = 900
    # Deprecated: review agent now resolves repos via workspace config.
    # Kept for backwards compatibility with existing .env files; the value
    # is parsed but no longer used by webhook.py.
    github_repo: str = ""
    # Directory (relative to repo root) where spec files live for
    # branch-name matching. Does not affect body marker resolution,
    # which accepts any path relative to the repo root.
    spec_dir: str = "specs"

    # Issue triage agent: per-user toggle lives in users.yaml
    # `issue_triage` (or the per-chat /github triage command).

    # GitHub notification routing is per-user: `github_notify_chat_id`
    # in users.yaml, with /github notify as the per-chat override. When
    # neither is set the runtime falls back to the user's own chat_id.

    # File retention: delete uploaded files older than this many days.
    # 0 = no cleanup (default). Cleanup runs once every 24 hours.
    file_retention_days: int = 0

    # Per-user configuration from users.yaml, keyed by telegram_id.
    # users.yaml is mandatory: _load_user_configs raises SystemExit
    # if the file is missing, malformed, or has no valid entries, so
    # this field is always a populated dict at runtime. The default
    # factory is for tests that construct Config directly; production
    # code goes through load_config.
    user_configs: dict[int, UserConfig] = field(default_factory=dict)

    # TOTP two-factor authentication timing (only relevant when TOTP is enabled)
    totp_session_minutes: int = 30
    totp_challenge_seconds: int = 120
    totp_lockout_attempts: int = 3
    totp_lockout_minutes: int = 15

    # Default backend selection: "claude" (default) uses Claude Code CLI,
    # "goose" uses Goose ACP (Agent Client Protocol) as the agent harness.
    default_backend: str = "claude"

    # LLM provider for non-Claude backends (e.g. Goose). Determines
    # which API key env var the backend expects and whether Kai's
    # logical model names ("sonnet", "opus") are translated to Anthropic IDs.
    # Ignored when default_backend="claude".
    default_provider: str = ""

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
        return self.user_configs.get(user_id)

    def get_user_by_github(self, github_login: str) -> UserConfig | None:
        """Look up a user by GitHub username. Used for webhook actor routing."""
        for uc in self.user_configs.values():
            if uc.github and uc.github.lower() == github_login.lower():
                return uc
        return None

    def get_admins(self) -> list[UserConfig]:
        """Get all admin users. Used for unattributed webhook fallback routing."""
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


def _xdg_config_home() -> Path:
    """Return `$XDG_CONFIG_HOME` (if set) or `$HOME/.config` otherwise.

    Used by single-user installs that run without root to locate
    `kai/users.yaml` under the operator's config tree. The XDG variable
    is honored when set so operators who have configured an alternate
    config home for other tools see the same convention apply here.
    """
    explicit = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".config"


def _resolve_users_yaml_path(protected_env_was_loaded: bool) -> Path:
    """Resolve the canonical users.yaml path for this deployment.

    Resolution order, first match wins:
      1. `KAI_USERS_YAML` env var. Test / explicit-development override
         only; the README does not document this as a normal operator
         path. Lets tests pin a tmp path without faking
         protected-env-loaded state.
      2. `/etc/kai/users.yaml` when `protected_env_was_loaded` is True.
         The signal is "_read_protected_file('/etc/kai/env') returned
         non-empty content during this load_config call." Ambient env
         vars like `KAI_INSTALL_DIR` and `KAI_DATA_DIR` deliberately
         do NOT participate: they are path overrides for data and
         install layout but do not imply protected deployment mode.
      3. `${XDG_CONFIG_HOME:-$HOME/.config}/kai/users.yaml` otherwise.
         The single-user repo install lives entirely under the
         operator's home; no `/etc/kai/` writes, no sudo at startup.

    Returning a Path (not a string) lets callers stat the file
    directly when reading without round-tripping through the
    protected-file sudo shim.
    """
    override = os.environ.get("KAI_USERS_YAML", "").strip()
    if override:
        return Path(override).expanduser()
    if protected_env_was_loaded:
        return Path("/etc/kai/users.yaml")
    return _xdg_config_home() / "kai" / "users.yaml"


def _read_users_yaml(path: Path) -> dict | object | None:
    """Read users.yaml at `path` with read mechanism keyed to location.

    `/etc/kai/users.yaml` is root-owned mode 0600 and goes through the
    `_read_protected_yaml` sudo-cat shim. Any other path (XDG
    single-user, `KAI_USERS_YAML` test override) is read directly with
    `Path.read_text` because the operator owns it and no privilege
    escalation is needed or appropriate.

    Returns the same tri-state as `_read_protected_yaml`: parsed dict
    on success, None when the file does not exist or cannot be read,
    `_YAML_MALFORMED` when the file exists but parses to an invalid
    shape.
    """
    if path == Path("/etc/kai/users.yaml"):
        return _read_protected_yaml("users.yaml")
    if not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("Cannot read %s: %s", path, e)
        return None
    try:
        result = yaml.safe_load(content)
    except yaml.YAMLError as e:
        log.error("Invalid YAML in %s: %s", path, e)
        return _YAML_MALFORMED
    if isinstance(result, dict):
        return result
    log.warning("%s: expected a YAML dict, got %s", path, type(result).__name__)
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

        # Budgets are no longer tracked; tolerate a lingering key from
        # an older workspaces.yaml rather than failing the entry, so an
        # un-migrated file keeps loading after upgrade.
        if claude_section.get("budget") is not None:
            log.warning("workspaces.yaml: 'budget' for %s is no longer supported; ignoring", path)

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
            timeout=timeout,
            env=env,
            env_file=env_file,
            system_prompt=system_prompt,
            system_prompt_file=system_prompt_file,
        )

    return configs


def _load_memory_project_configs() -> dict[str, MemoryProjectConfig]:
    """
    Load the memory project registry from memory-projects.yaml.

    Tries /etc/kai/memory-projects.yaml first (protected
    installation), falls back to PROJECT_ROOT/memory-projects.yaml
    (development). Returns an empty dict if neither file exists.

    The registry is keyed by project_id. Each MemoryProjectConfig
    holds the canonical resolved workspace_roots tuple plus the
    enablement flag and optional default-scope policy. Detection
    logic lives in kai.memory_projects and consumes this dict.

    Validation is fail-closed: any malformed entry is skipped (with
    a warning) so an unmatched cwd returns no active project rather
    than producing accidental project-scoped recall. Path existence
    is NOT required at load time; registry config may be authored
    before a checkout exists or while a mount is unavailable, and
    detection's longest-prefix match handles the absent-root case
    naturally by failing to match.

    Duplicate handling:
    - Duplicate project_id: first entry wins, later entries logged
      and skipped.
    - Duplicate workspace_root across distinct projects: the root is
      dropped from the later project only. If the later project ends
      up with no roots, the whole project is also skipped.
    """
    # Mirrors _load_workspace_configs: protected-first, fall back to
    # PROJECT_ROOT for dev. A malformed protected file does NOT fall
    # through to the local file (avoid silently using dev config on
    # a production system); fail closed with an empty registry
    # instead.
    data = _read_protected_yaml("memory-projects.yaml")
    if data is _YAML_MALFORMED:
        log.warning("Skipping memory project registry: /etc/kai/memory-projects.yaml is malformed or empty")
        return {}
    if data is None:
        local_path = PROJECT_ROOT / "memory-projects.yaml"
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

    entries = data.get("projects")
    if not isinstance(entries, list):
        if entries is not None:
            log.warning(
                "memory-projects.yaml: 'projects' must be a list, got %s",
                type(entries).__name__,
            )
        return {}

    # Lazy import: kai.memory imports kai.config at module load time,
    # so importing the scope constants at module scope here would
    # create a circular import. Importing inside the function body
    # breaks the cycle and keeps config.py's import surface lean.
    from kai.memory import SCOPE_GLOBAL, SCOPE_PROJECT

    valid_default_scopes = {SCOPE_GLOBAL, SCOPE_PROJECT}

    configs: dict[str, MemoryProjectConfig] = {}
    # Track which resolved root path is already owned by which
    # project_id so the "drop duplicate root from later project"
    # rule is enforceable across the loop.
    root_owners: dict[Path, str] = {}

    for entry in entries:
        if not isinstance(entry, dict):
            log.warning("memory-projects.yaml: skipping non-dict entry: %s", entry)
            continue

        # project_id: non-empty string after whitespace strip.
        raw_project_id = entry.get("project_id")
        if not isinstance(raw_project_id, str):
            log.warning(
                "memory-projects.yaml: skipping entry; project_id must be a string, got %s",
                type(raw_project_id).__name__,
            )
            continue
        project_id = raw_project_id.strip()
        if not project_id:
            log.warning("memory-projects.yaml: skipping entry with empty project_id")
            continue

        # First-wins on duplicate project_id.
        if project_id in configs:
            log.warning(
                "memory-projects.yaml: duplicate project_id %r; using first entry",
                project_id,
            )
            continue

        # display_name: non-empty string.
        display_name = entry.get("display_name")
        if not isinstance(display_name, str) or not display_name.strip():
            log.warning(
                "memory-projects.yaml: skipping project %r; display_name must be a non-empty string",
                project_id,
            )
            continue

        # memory_enabled: strict boolean. YAML truthy values like
        # "true", "yes", 1 must NOT be coerced; force a hard reject
        # so the operator notices the typo at load time. bool is a
        # subclass of int in Python, so even though isinstance(True,
        # int) is True, isinstance(1, bool) is False, which gives
        # the correct rejection here.
        memory_enabled = entry.get("memory_enabled")
        if not isinstance(memory_enabled, bool):
            log.warning(
                "memory-projects.yaml: skipping project %r; memory_enabled must be a real boolean (true/false), got %r",
                project_id,
                memory_enabled,
            )
            continue

        # workspace_roots: non-empty list of paths. Each is resolved
        # (expanduser + resolve) but NOT required to exist - registry
        # config may pre-date a checkout. Cross-project duplicates
        # are dropped from later projects via the root_owners gate.
        raw_roots = entry.get("workspace_roots")
        if not isinstance(raw_roots, list) or not raw_roots:
            log.warning(
                "memory-projects.yaml: skipping project %r; workspace_roots must be a non-empty list",
                project_id,
            )
            continue

        resolved_roots: list[Path] = []
        intra_project_seen: set[Path] = set()
        for raw_root in raw_roots:
            if not isinstance(raw_root, (str, Path)):
                log.warning(
                    "memory-projects.yaml: project %r: skipping non-string root %r",
                    project_id,
                    raw_root,
                )
                continue
            try:
                resolved = Path(str(raw_root)).expanduser().resolve()
            except (OSError, ValueError) as e:
                log.warning(
                    "memory-projects.yaml: project %r: cannot resolve root %r: %s",
                    project_id,
                    raw_root,
                    e,
                )
                continue
            # Within-project duplicate: keep first only.
            if resolved in intra_project_seen:
                log.warning(
                    "memory-projects.yaml: project %r: duplicate root %s within project; keeping first",
                    project_id,
                    resolved,
                )
                continue
            # Cross-project duplicate: drop from the LATER project.
            owner = root_owners.get(resolved)
            if owner is not None:
                log.warning(
                    "memory-projects.yaml: root %s already owned by project %r; dropping from %r",
                    resolved,
                    owner,
                    project_id,
                )
                continue
            intra_project_seen.add(resolved)
            resolved_roots.append(resolved)

        if not resolved_roots:
            # All roots were duplicates or malformed; drop the
            # project entirely so detection never returns a project
            # with no roots to match against.
            log.warning(
                "memory-projects.yaml: skipping project %r; no valid workspace_roots remain after deduplication",
                project_id,
            )
            continue

        # default_scope_for_new_facts: optional. When present, must
        # match SCOPE_GLOBAL or SCOPE_PROJECT. SCOPE_TASK is rejected
        # because task scope is not a write target in this issue.
        default_scope = entry.get("default_scope_for_new_facts")
        if default_scope is not None and (
            not isinstance(default_scope, str) or default_scope not in valid_default_scopes
        ):
            log.warning(
                "memory-projects.yaml: skipping project %r; default_scope_for_new_facts must be %r or %r, got %r",
                project_id,
                SCOPE_GLOBAL,
                SCOPE_PROJECT,
                default_scope,
            )
            continue

        # Record ownership only after every validation has passed so
        # a rejected project does not "claim" a root and lock it out
        # of a later valid project.
        for resolved in resolved_roots:
            root_owners[resolved] = project_id

        configs[project_id] = MemoryProjectConfig(
            project_id=project_id,
            display_name=display_name,
            workspace_roots=tuple(resolved_roots),
            memory_enabled=memory_enabled,
            default_scope_for_new_facts=default_scope,
        )

    return configs


# Valid user roles for users.yaml
_VALID_ROLES = {"admin", "user"}


def _compute_extraction_eligible_backends(
    agent_backend: str,
    user_configs: dict[int, UserConfig],
    memory_extraction_enabled: bool,
) -> set[str]:
    """
    Return the set of distinct effective backends across extraction-eligible users.

    Mirrors `_ingest_memory`'s extraction gate (effective backend in
    ONESHOT_REASONER_BACKENDS; a backend without a OneShotReasoner is
    filtered out). Each user contributes its effective backend
    (per-user override or global default).

    Returns an empty set when `memory_extraction_enabled` is False;
    callers that gate codex / opencode plumbing or registry validation
    off the set should not fire on retrieval-only or memory-disabled
    installs.
    """
    if not memory_extraction_enabled:
        return set()
    eligible: set[str] = set()
    for uc in user_configs.values():
        effective = uc.backend or agent_backend
        if effective in ONESHOT_REASONER_BACKENDS:
            eligible.add(effective)
    return eligible


def _apply_legacy_model_env_overrides(
    user_configs: dict[int, "UserConfig"],
    default_backend: str,
) -> dict[int, "UserConfig"]:
    """Seed UserConfig.models from the deprecated per-role env vars.

    Reads PR_REVIEW_MODEL_<EFFECTIVE_BACKEND> and
    ISSUE_TRIAGE_MODEL_<EFFECTIVE_BACKEND> from the process env for
    each user. EFFECTIVE_BACKEND is the user's per-user override
    (`uc.backend`) or `default_backend` when the user has no
    override. The resolution is inlined as a single fallback
    expression rather than calling `get_user_backend_and_provider`,
    because the full Config object is not yet built at this point in
    `load_config` (this pass runs immediately after `_load_user_configs`,
    before the Config constructor runs). Provider is not needed by
    this function; only `backend` drives the env-var suffix.

    Per-user `models:` entries always win over the env-var seed; only
    roles absent from the user's own map get seeded.

    Logs a one-shot deprecation warning the first time ANY user gets
    a seed value applied; one log line per `load_config` call, not
    per user. The warning names the env var, the role it seeds, and
    points the operator at the users.yaml `models:` migration path.

    Returns a new user_configs dict (UserConfig is a frozen dataclass;
    `dataclasses.replace` is required to update the models field).
    """
    import dataclasses

    warned = False
    out: dict[int, UserConfig] = {}
    for uid, uc in user_configs.items():
        backend = uc.backend or default_backend
        existing_models = dict(uc.models or {})
        for role_str, env_prefix in [
            ("pr_review", "PR_REVIEW_MODEL"),
            ("issue_triage", "ISSUE_TRIAGE_MODEL"),
        ]:
            if role_str in existing_models:
                continue  # user's own map wins
            env_value = os.environ.get(f"{env_prefix}_{backend.upper()}", "").strip()
            if not env_value:
                continue
            existing_models[role_str] = env_value
            if not warned:
                log.warning(
                    "%s_%s is deprecated; migrate the value into the "
                    "user's `models.%s` field in users.yaml. The env "
                    "var continues to work as a fallback for the "
                    "current major version.",
                    env_prefix,
                    backend.upper(),
                    role_str,
                )
                warned = True
        out[uid] = dataclasses.replace(uc, models=existing_models or None)
    return out


def _compute_extraction_eligible_backend_provider_pairs(
    agent_backend: str,
    agent_provider: str,
    user_configs: dict[int, UserConfig],
    memory_extraction_enabled: bool,
) -> set[tuple[str, str]]:
    """
    Return the set of distinct (effective_backend, effective_provider)
    pairs across extraction-eligible users.

    Same shape as `_compute_extraction_eligible_backends` but paired
    with the user's effective provider so callers can look up per-role
    models in the (backend, provider, role) MODEL_REGISTRY. Used by
    `memory.config` startup logging to render each eligible pair's
    extraction and episode model selection.
    """
    if not memory_extraction_enabled:
        return set()
    eligible: set[tuple[str, str]] = set()
    for uc in user_configs.values():
        eff_backend = uc.backend or agent_backend
        if eff_backend not in ONESHOT_REASONER_BACKENDS:
            continue
        eff_provider = uc.provider or agent_provider
        if eff_backend == "claude":
            eff_provider = "anthropic"
        elif eff_backend == "codex":
            eff_provider = "openai"
        eligible.add((eff_backend, eff_provider))
    return eligible


def _load_user_configs(
    global_backend: str,
    global_llm_provider: str,
    users_yaml_path: Path | None = None,
) -> dict[int, UserConfig]:
    """
    Load per-user configs from users.yaml at the given path.

    Reads via `_read_users_yaml`, which routes `/etc/kai/users.yaml`
    through the sudo-cat shim and any other path (XDG single-user,
    `KAI_USERS_YAML` override) through a direct `Path.read_text`.
    users.yaml is mandatory: any failure raises SystemExit with a
    message naming the resolved path. There is no None return path
    and no fallback to ALLOWED_USER_IDS at runtime. The fail-closed
    shape is load-bearing for the daemon's auth contract: a
    malformed or unreadable users file must not silently degrade to
    env-only auth, because the wizard and the runtime would then
    disagree about whose value wins.

    Failure cases (each raises SystemExit):
        - File absent: error names the path and points at `make config`.
          When ALLOWED_USER_IDS is set in env, appends a one-line
          migration hint mentioning that the env var is no longer
          honored as a fallback.
        - Malformed YAML, non-dict top-level, missing or non-list
          `users` key: error names the path and the parse/schema fault.
        - Zero valid user entries after per-entry validation: error
          names the path and refers the operator to the warnings
          logged above.

    Per-entry validation errors continue to log a warning and skip
    the entry rather than abort; the SystemExit at the end fires only
    when every entry was rejected.

    Args:
        global_backend: The global default_backend from env config.
        global_llm_provider: The global default_provider from env config.
            Both are needed to cascade per-user model validation:
            a user's effective provider determines which models are valid.
        users_yaml_path: Resolved users.yaml path for this deployment.
            Defaults to `/etc/kai/users.yaml` to preserve protected-install
            ergonomics for tests that do not exercise XDG resolution.
            Production callers in `load_config` pass the result of
            `_resolve_users_yaml_path(protected_env_was_loaded)`.

    Returns a dict keyed by telegram_id for O(1) lookup.
    """
    if users_yaml_path is None:
        users_yaml_path = Path("/etc/kai/users.yaml")
    data = _read_users_yaml(users_yaml_path)
    if data is _YAML_MALFORMED:
        raise SystemExit(
            f"{users_yaml_path} is malformed or has an invalid top-level YAML shape. "
            "Fix the file or re-run 'make config' to regenerate it."
        )
    if data is None:
        msg = (
            f"users.yaml is required for authorization and was not found at {users_yaml_path}. "
            "Run 'make config' to generate it; the wizard prompts for the admin Telegram ID."
        )
        if os.environ.get("ALLOWED_USER_IDS"):
            # Surface the breaking change explicitly so a legacy
            # env-only operator understands why the daemon stopped
            # starting. The hint does NOT claim the wizard auto-seeds
            # the new users.yaml from the env var; that auto-seed
            # behavior is not implemented here and the wizard will
            # prompt for the Telegram ID manually.
            msg += (
                " ALLOWED_USER_IDS is set in env but is no longer honored as an "
                "auth fallback; copy each ID into the wizard prompt when it asks."
            )
        raise SystemExit(msg)

    # After the _YAML_MALFORMED and None checks, `data` is guaranteed
    # to be a dict (`_read_protected_yaml` returns `_YAML_MALFORMED`
    # for any non-dict top-level value). Guard defensively rather than
    # assert since assertions are stripped under Python -O.
    if not isinstance(data, dict):
        raise SystemExit(f"{users_yaml_path}: expected a YAML dict, got {type(data).__name__}")

    entries = data.get("users")
    if not isinstance(entries, list):
        # A users.yaml file without a 'users' key is almost certainly
        # a typo (e.g., 'user:' instead of 'users:'). Either shape is
        # fatal at this point because there is no auth fallback.
        if entries is not None:
            raise SystemExit(f"{users_yaml_path}: 'users' must be a list, got {type(entries).__name__}")
        raise SystemExit(f"{users_yaml_path}: no 'users' key found; check for typos (e.g. 'user:' vs 'users:')")

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

        # Budgets are no longer tracked; tolerate a lingering key from
        # an older users.yaml rather than failing the entry, so an
        # un-migrated file keeps loading after upgrade.
        if entry.get("max_budget") is not None:
            log.warning("users.yaml: 'max_budget' for %s is no longer supported; ignoring", name)

        # Validate optional per-user backend (must be valid if set).
        # Reads `backend`, falling back to the deprecated `default_backend`
        # and then `agent_backend` keys for one release (the per-user key
        # was renamed twice: agent_backend -> default_backend -> backend).
        # default=None so a user with neither key inherits the global
        # backend below via `user_backend or global_backend`, rather than
        # being pinned to claude.
        user_backend: str | None = None
        raw_backend = _resolve_renamed_key(
            entry.get,
            new_key="backend",
            legacy_keys=["default_backend", "agent_backend"],
            context=f"users.yaml entry for {name}",
            default=None,
        )
        if raw_backend is not None:
            backend_str = str(raw_backend).strip().lower()
            if backend_str not in VALID_BACKENDS:
                raise SystemExit(
                    f"users.yaml: user '{name}' has invalid backend '{backend_str}' "
                    f"(must be one of: {', '.join(sorted(VALID_BACKENDS))})"
                )
            user_backend = backend_str

        # Validate optional per-user provider (must be valid for the
        # user's backend). Reads `provider`, falling back to the
        # deprecated `llm_provider` key for one release.
        user_provider: str | None = None
        raw_provider = _resolve_renamed_key(
            entry.get,
            new_key="provider",
            legacy_keys=["llm_provider"],
            context=f"users.yaml entry for {name}",
            default=None,
        )
        if raw_provider is not None:
            provider_str = str(raw_provider).strip().lower()
            # Validate against the user's effective backend. If the user
            # has no explicit backend, validate against the global one.
            eff_backend_for_val = user_backend or global_backend
            # Validate against BACKEND_PROVIDERS only when the backend
            # requires a provider prompt. Single-provider backends
            # (claude, codex) are absent from BACKENDS_NEEDING_PROVIDER_PROMPT,
            # so a per-user provider override on those backends is
            # accepted without a curated-list check (the provider is
            # implicit at runtime regardless of what users.yaml says).
            valid: tuple[str, ...] | None = (
                BACKEND_PROVIDERS.get(eff_backend_for_val)
                if eff_backend_for_val in BACKENDS_NEEDING_PROVIDER_PROMPT
                else None
            )
            if valid is not None and provider_str not in valid:
                raise SystemExit(
                    f"users.yaml: user '{name}' has invalid provider "
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
        if eff_backend in BACKENDS_NEEDING_PROVIDER_PROMPT and not eff_provider_str:
            raise SystemExit(
                f"users.yaml: user '{name}' has backend='{eff_backend}' but no "
                f"provider is configured (set it in users.yaml or as "
                f"DEFAULT_PROVIDER env var)"
            )

        # Warn if user is on an open-ended provider with no model set.
        # PROVIDER_DEFAULTS has no entry for openrouter/ollama, so the
        # pool would fall back to the global default_model (e.g., "sonnet")
        # which is almost certainly wrong for that provider.
        model = entry.get("model")
        if model is not None:
            model = str(model).strip().lower()
            model = canonicalize_model_for_backend(model, eff_backend)
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

        # The context_window setting was removed (the agent CLI default
        # applies). Tolerate the key in existing users.yaml files so
        # installs keep loading; warn so the operator knows the value
        # has no effect.
        if entry.get("context_window") is not None:
            log.warning("users.yaml: 'context_window' is no longer supported; ignoring (user %s)", name)

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

        # Parse the explicit per-user external-service allowlist. Service
        # existence is checked later by SubprocessPool, after services.yaml
        # has been loaded and missing-key services have been filtered out.
        raw_allowed_services = entry.get("allowed_services", [])
        allowed_services: list[str] = []
        if isinstance(raw_allowed_services, list):
            for raw_service in raw_allowed_services:
                if not isinstance(raw_service, str):
                    log.warning(
                        "users.yaml: invalid allowed_services entry for %s: %r (expected a service name)",
                        name,
                        raw_service,
                    )
                    continue
                service_name = raw_service.strip()
                if not service_name or service_name == "*" or "/" in service_name:
                    log.warning(
                        "users.yaml: invalid allowed_services entry for %s: %r (wildcards and paths are not allowed)",
                        name,
                        raw_service,
                    )
                    continue
                if service_name not in allowed_services:
                    allowed_services.append(service_name)
        else:
            log.warning(
                "users.yaml: allowed_services for %s must be a list (ignoring)",
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

        # Per-role per-user model overrides (`models:` sub-map). Keys
        # are "agent" plus any ModelRole.value; values are model strings
        # the user's backend's CLI accepts. Validation mirrors the
        # legacy `model:` field's check via validate_model_for_backend.
        # An invalid key or value raises SystemExit so the operator
        # sees the precise failure at startup rather than at first
        # dispatch.
        user_models: dict[str, str] | None = None
        raw_models = entry.get("models")
        if raw_models is not None:
            if not isinstance(raw_models, dict):
                raise SystemExit(
                    f"users.yaml: user '{name}' has `models` that is not a mapping; "
                    f"expected a sub-map of role -> model string."
                )
            valid_role_keys = {"agent", *(r.value for r in ModelRole)}
            checked: dict[str, str] = {}
            for raw_key, raw_value in raw_models.items():
                role_key = str(raw_key).strip().lower()
                if role_key not in valid_role_keys:
                    raise SystemExit(
                        f"users.yaml: user '{name}' has models.{role_key!r} which is not "
                        f"a recognized role; valid roles: {', '.join(sorted(valid_role_keys))}."
                    )
                value_str = str(raw_value).strip()
                if not value_str:
                    raise SystemExit(
                        f"users.yaml: user '{name}' has models.{role_key} with an empty value; "
                        f"remove the key or set a non-empty model string."
                    )
                value_str = canonicalize_model_for_backend(value_str, eff_backend)
                if not validate_model_for_backend(value_str, eff_backend, eff_provider):
                    raise SystemExit(
                        f"users.yaml: user '{name}' has models.{role_key}={value_str!r} which is "
                        f"not valid for (backend={eff_backend}, provider={eff_provider})."
                    )
                checked[role_key] = value_str
            user_models = checked if checked else None

        # Back-compat: if the operator only set the legacy `model:`
        # field, synthesize `models["agent"] = model_value` so the
        # in-memory shape matches users.yaml entries that use the
        # `models:` map. The on-disk file is untouched; the next
        # `make config` re-run writes the canonical shape.
        if user_models is None and model is not None:
            user_models = {"agent": model}

        configs[telegram_id] = UserConfig(
            telegram_id=telegram_id,
            name=name,
            role=role,
            github=github,
            os_user=os_user,
            home_workspace=home_workspace,
            model=model,
            timeout=user_timeout,
            workspace_base=user_workspace_base,
            allowed_workspaces=user_allowed_workspaces,
            backend=user_backend,
            provider=user_provider,
            github_repos=github_repos,
            github_notify_chat_id=github_notify_chat_id,
            pr_review=pr_review,
            issue_triage=issue_triage,
            allowed_services=allowed_services,
            models=user_models,
        )

    if not configs:
        raise SystemExit(
            f"{users_yaml_path}: no valid user entries; every entry was rejected "
            "by per-entry validation. See the warnings logged above for the "
            "individual rejection reasons."
        )

    # Warn if no admin is defined - external webhooks will route to
    # an arbitrary user, which may be surprising.
    if not any(uc.role == "admin" for uc in configs.values()):
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

        # Webhook secret: validates incoming updates from Telegram. A fresh
        # process-lifetime value is safe here because start() registers that same
        # value with Telegram before accepting traffic. Never reuse an external
        # webhook signing secret across protocol boundaries.
        telegram_webhook_secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip()
        if not telegram_webhook_secret:
            telegram_webhook_secret = secrets.token_urlsafe(32)
        log.info("Telegram transport: webhook (%s)", telegram_webhook_url)
    else:
        log.info("Telegram transport: polling (TELEGRAM_WEBHOOK_URL not set)")

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
    # DEFAULT_TIMEOUT is the canonical key; AGENT_TIMEOUT_SECONDS is a
    # legacy alias kept for installs upgrading without re-running the
    # wizard. Apply-time migration in install.py rewrites /etc/kai/env
    # to the new key, so the legacy fallback only fires on the first
    # start after the upgrade. The lookup treats an empty value as
    # absent so an unset/blank new key still falls through to the
    # legacy name and then the 120 default.
    raw_timeout = (
        _resolve_renamed_key(
            lambda k: (os.environ.get(k) or "").strip() or None,
            new_key="DEFAULT_TIMEOUT",
            legacy_keys=["AGENT_TIMEOUT_SECONDS"],
            context="/etc/kai/env",
            default=None,
        )
        or "120"
    )
    try:
        default_timeout = int(raw_timeout)
    except ValueError:
        raise SystemExit("DEFAULT_TIMEOUT must be an integer") from None
    # AGENT_MAX_SESSION_HOURS / AGENT_IDLE_TIMEOUT are the canonical
    # keys; the CLAUDE_-prefixed forms are legacy aliases kept for
    # installs upgrading without re-running the wizard. Apply-time
    # migration in install.py rewrites /etc/kai/env to the new keys,
    # so the fallback only fires on the first start after an upgrade
    # (the _renamed_env_vars warning below tells the operator to
    # migrate). Both govern the subprocess pool's session lifecycle
    # for every backend, which is why the claude prefix was retired.
    raw_session_hours = os.environ.get("AGENT_MAX_SESSION_HOURS", "").strip()
    if not raw_session_hours:
        raw_session_hours = os.environ.get("CLAUDE_MAX_SESSION_HOURS", "").strip() or "0"
    try:
        agent_max_session_hours = float(raw_session_hours)
    except ValueError:
        raise SystemExit("AGENT_MAX_SESSION_HOURS must be a number") from None
    raw_idle_timeout = os.environ.get("AGENT_IDLE_TIMEOUT", "").strip()
    if not raw_idle_timeout:
        raw_idle_timeout = os.environ.get("CLAUDE_IDLE_TIMEOUT", "").strip() or "1800"
    try:
        agent_idle_timeout = int(raw_idle_timeout)
    except ValueError:
        raise SystemExit("AGENT_IDLE_TIMEOUT must be an integer") from None
    try:
        webhook_port = int(os.environ.get("WEBHOOK_PORT", "8080"))
    except ValueError:
        raise SystemExit("WEBHOOK_PORT must be an integer") from None
    try:
        file_retention_days = int(os.environ.get("FILE_RETENTION_DAYS", "0"))
    except ValueError:
        raise SystemExit("FILE_RETENTION_DAYS must be an integer") from None

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

    # CODEX_EFFORT_LEVEL: same strip/lower + membership shape as the
    # claude block above, with one contract difference: empty / unset
    # stays empty (set-or-absent). Empty means CodexBackend passes no
    # `-c model_reasoning_effort` override and codex falls back to the
    # per-OS-user ~/.codex/config.toml or the model default.
    codex_effort_level = os.environ.get("CODEX_EFFORT_LEVEL", "").strip().lower()
    if codex_effort_level and codex_effort_level not in _VALID_CODEX_EFFORT_LEVELS:
        raise SystemExit(
            f"CODEX_EFFORT_LEVEL must be one of {list(CODEX_EFFORT_LEVELS)} or empty, got {codex_effort_level!r}"
        )

    # PR review agent config. The global `pr_review` toggle now lives
    # per-user in users.yaml; PR_REVIEW_COOLDOWN / PR_REVIEW_TIMEOUT_S
    # remain as global resource controls.
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

    # Default backend selection - "claude" (default) or "goose".
    # DEFAULT_BACKEND with a one-release fallback to the deprecated
    # AGENT_BACKEND name; absence means the installation default "claude".
    default_backend = (
        (
            _resolve_renamed_key(
                os.environ.get,
                new_key="DEFAULT_BACKEND",
                legacy_keys=["AGENT_BACKEND"],
                context="/etc/kai/env",
                default="claude",
            )
            or "claude"
        )
        .strip()
        .lower()
    )
    if default_backend not in VALID_BACKENDS:
        raise SystemExit(
            f"DEFAULT_BACKEND '{default_backend}' is not valid (must be one of: {', '.join(sorted(VALID_BACKENDS))})"
        )

    # Verify the per-role model registry has rows for every role the
    # active backend will look up. Goose is exempt (its model resolution
    # uses _GOOSE_AGENT_MODELS dicts, not the registry). Raises
    # SystemExit on a missing row so the bug surfaces at startup rather
    # than as a per-request LookupError.
    _check_model_registry_complete()

    # LLM provider - validated against the backend's supported set.
    # Single-provider backends (claude, codex) skip the prompt because
    # their provider is implicit; multi-provider backends (opencode,
    # goose) must name one so the (backend, provider, role) registry
    # lookup can find a row. DEFAULT_PROVIDER with a one-release
    # fallback to the deprecated LLM_PROVIDER name.
    default_provider = ""
    valid_providers: tuple[str, ...] | None = (
        BACKEND_PROVIDERS.get(default_backend) if default_backend in BACKENDS_NEEDING_PROVIDER_PROMPT else None
    )
    if valid_providers is not None:
        default_provider = (
            (
                _resolve_renamed_key(
                    os.environ.get,
                    new_key="DEFAULT_PROVIDER",
                    legacy_keys=["LLM_PROVIDER"],
                    context="/etc/kai/env",
                    default="",
                )
                or ""
            )
            .strip()
            .lower()
        )
        if default_provider not in valid_providers:
            raise SystemExit(
                f"DEFAULT_PROVIDER '{default_provider}' is not valid for backend "
                f"'{default_backend}' (must be one of: "
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
    # spawning a per-turn Haiku call whose result silently no-ops in
    # the `_memory is None` guard inside add_structured. Compose here
    # so the dependency is explicit and the wasted-subprocess hole is
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
        ("MEMORY_REASONER_BACKEND", "memory reasoner is now derived per-user from default_backend"),
        ("MEMORY_EXTRACTION_MODEL", "memory extraction model is now resolved per-user from the MODEL_REGISTRY"),
        ("MEMORY_EPISODE_MODEL", "memory episode model is now resolved per-user from the MODEL_REGISTRY"),
        (
            "GOOSE_MODEL",
            "goose model selection now flows through the (backend, provider, role) "
            "MODEL_REGISTRY and per-user `models.agent` in users.yaml",
        ),
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
    # about above. Timeout floor is 10s to prevent accidentally
    # tightening it below Haiku's warm-up time.
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

    # Memory project registry. Loaded independently of workspace_configs
    # because memory authority and backend overrides are different
    # concepts that happen to overlap on paths. Memory project roots
    # are NOT merged into allowed_workspaces: workspace access is
    # still owned by /workspace access configuration. Per-user
    # workspace permissions still gate which workspaces the operator
    # can enter; the registry only describes the memory boundary of
    # workspaces the user is otherwise allowed to enter.
    memory_projects = _load_memory_project_configs()

    # Per-user configuration. users.yaml is mandatory; the loader
    # raises SystemExit on any failure with a path-naming message
    # (and a one-line ALLOWED_USER_IDS migration hint when that env
    # var is set on a missing-file install). The legacy env-var
    # fallback is gone: a malformed users file used to silently
    # degrade to ALLOWED_USER_IDS, which the wizard and the loader
    # then disagreed about; fail-closed is the contract now.
    #
    # The path resolves based on whether `/etc/kai/env` had readable
    # content at the top of this load_config call (protected install)
    # or not (single-user install reads from PROJECT_ROOT/.env and
    # places users.yaml under XDG config home). KAI_USERS_YAML is an
    # explicit override for tests and ad-hoc development; the operator
    # path is always one of the two resolved defaults.
    users_yaml_path = _resolve_users_yaml_path(bool(protected_env))
    user_configs = _load_user_configs(default_backend, default_provider, users_yaml_path)
    if protected_env:
        # A protected install gives the outer service account narrowly
        # scoped sudo access to root-owned Kai configuration.  A persistent
        # conversational agent running as that same account would inherit
        # those capabilities even though its environment is sanitized.
        # Fail at startup if a hand-edited users.yaml weakens the boundary.
        try:
            service_user = pwd.getpwuid(os.geteuid()).pw_name
        except KeyError:
            raise SystemExit(
                f"Protected installation could not resolve the Kai service account for effective uid {os.geteuid()}."
            ) from None
        try:
            validate_protected_user_isolation(
                ((uc.telegram_id, uc.name, uc.os_user) for uc in user_configs.values()),
                service_user,
                account_uid=lambda name: pwd.getpwnam(name).pw_uid,
                service_uid=os.geteuid(),
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    # Seed UserConfig.models from the deprecated per-role env vars
    # (PR_REVIEW_MODEL_<BACKEND> / ISSUE_TRIAGE_MODEL_<BACKEND>). The
    # values reach dispatch through UserConfig.models rather than via
    # the historic dispatch-time env reads in review.py / triage.py
    # so the per-user `models:` map remains the single source of
    # truth for per-role selection. One-shot deprecation warning
    # fires inside the helper when any seed value applies.
    user_configs = _apply_legacy_model_env_overrides(user_configs, default_backend)
    allowed_ids = set(user_configs.keys())
    if os.environ.get("ALLOWED_USER_IDS", "").strip():
        log.warning(
            "ALLOWED_USER_IDS is set in env but is no longer honored; "
            "users.yaml is authoritative. Remove the env var from /etc/kai/env "
            "(or re-run 'make config') to clear this warning."
        )

    # Memory extraction preconditions, validated per the
    # extraction-eligible backend set. Reasoner and model both derive
    # per-user from each user's effective `agent_backend` at extraction
    # time, so the "is this install in a valid state for memory
    # extraction" question must read the full eligible set rather than
    # a single global backend value. The helper mirrors
    # `_ingest_memory`'s extraction gate in bot.py (membership in
    # ONESHOT_REASONER_BACKENDS).
    extraction_eligible_backends = _compute_extraction_eligible_backends(
        default_backend, user_configs, memory_extraction_enabled
    )

    # Per-eligible-backend binary resolution. The earlier
    # _check_model_registry_complete() call is self-driving over every
    # (backend, provider) pair in BACKEND_PROVIDERS, so registry rows
    # are already validated for every backend including per-user
    # overrides. `resolve_oneshot_binary` is per-backend (each backend
    # honors its *_BIN override, falling back to PATH) so it stays in
    # the loop; a missing binary on a deployment where any user routes
    # to that backend fails fast at startup instead of at the first
    # extraction.
    if memory_extraction_enabled and extraction_eligible_backends:
        from kai.oneshot_binary import BinaryResolutionError, resolve_oneshot_binary

        for _backend in sorted(extraction_eligible_backends):
            try:
                resolve_oneshot_binary(_backend)
            except BinaryResolutionError as e:
                raise SystemExit(
                    f"Memory extraction requires the {_backend!r} binary to be reachable "
                    f"at startup (at least one extraction-eligible user routes to it), but "
                    f"{e}. Set {_backend.upper()}_BIN to the binary's absolute path or "
                    f"fix PATH; rerun the wizard if you no longer want memory extraction "
                    f"enabled."
                ) from None

    # Renamed env vars: warn when the legacy name is present so the
    # operator knows to re-run `make config`. Only renames remain in
    # this map. Per-user fields (model, os_user, pr_review, etc.) are
    # not legacy renames; the runtime no longer reads them at all and
    # the corresponding wizard prompts have been retired.
    _renamed_env_vars = {
        "CLAUDE_MAX_SESSION_HOURS": "AGENT_MAX_SESSION_HOURS.",
        "CLAUDE_IDLE_TIMEOUT": "AGENT_IDLE_TIMEOUT.",
    }
    for var, replacement in _renamed_env_vars.items():
        if os.environ.get(var, "").strip():
            log.warning(
                "%s in env is deprecated; use %s Re-run 'make config' to migrate /etc/kai/env automatically.",
                var,
                replacement,
            )

    # Retired env vars: the setting no longer exists, so unlike the
    # renames above there is no replacement key to point at. Warn so
    # the operator knows the lingering value has no effect; the wizard
    # drops the key on the next regenerate. The reason clause finishes
    # the sentence "<KEY> is no longer supported; ...".
    _retired_env_vars = {
        "CLAUDE_MAX_CONTEXT_WINDOW": "the agent CLI's default context window applies",
        "BUDGET_CEILING": "budgets are no longer tracked",
        "CLAUDE_MAX_BUDGET_USD": "budgets are no longer tracked",
        "PR_REVIEW_BUDGET_USD": "budgets are no longer tracked",
        "MEMORY_EXTRACTION_BUDGET_USD": "budgets are no longer tracked",
        "MEMORY_EPISODE_BUDGET_USD": "budgets are no longer tracked",
        "MEMORY_SCOPED_RECALL_ENABLED": "scoped retrieval is the only live recall path",
        "MEMORY_RECALL_SHADOW_ENABLED": "the scoped-vs-legacy shadow comparator is gone",
    }
    for var, reason in _retired_env_vars.items():
        if os.environ.get(var, "").strip():
            log.warning(
                "%s is no longer supported; %s. Re-run 'make config' to clean /etc/kai/env.",
                var,
                reason,
            )

    # Warn when GitHub features are configured but no user has a
    # repo list. Events will never reach the agents because
    # `_get_subscribed_users()` returns empty for every incoming
    # webhook. The fallback path in `_process_github_event()` still
    # delivers basic notifications but does not guarantee the agents
    # fire.
    no_repos = not any(uc.github_repos for uc in user_configs.values())
    if no_repos:
        review_on = any(uc.pr_review is True for uc in user_configs.values())
        triage_on = any(uc.issue_triage is True for uc in user_configs.values())
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

    default_model = canonicalize_model_for_backend(
        os.environ.get("DEFAULT_MODEL", "sonnet"),
        default_backend,
    )

    # Parse DEFAULT_MODELS_JSON: global per-role defaults captured by
    # the wizard's per-role customization step. Empty / unset / invalid
    # JSON collapses to an empty dict, which leaves resolve_user_model
    # falling through to MODEL_REGISTRY for every role; the wizard
    # writes the key only when at least one captured value differed
    # from the registry default (delta-from-defaults), so the absence
    # of the key in env reflects "operator accepted every default".
    default_models_raw = os.environ.get("DEFAULT_MODELS_JSON", "").strip()
    default_models: dict[str, str] = {}
    if default_models_raw:
        import json

        try:
            parsed = json.loads(default_models_raw)
        except ValueError as e:
            raise SystemExit(
                f"DEFAULT_MODELS_JSON is not valid JSON: {e}. Re-run `make config` or remove the env var."
            ) from None
        if not isinstance(parsed, dict):
            raise SystemExit(f"DEFAULT_MODELS_JSON must be a JSON object; got {type(parsed).__name__}.")
        valid_role_keys = {"agent", *(r.value for r in ModelRole)}
        for raw_key, raw_value in parsed.items():
            role_key = str(raw_key).strip().lower()
            if role_key not in valid_role_keys:
                raise SystemExit(
                    f"DEFAULT_MODELS_JSON: unrecognized role '{role_key}'; "
                    f"valid roles: {', '.join(sorted(valid_role_keys))}."
                )
            value_str = str(raw_value).strip()
            if not value_str:
                raise SystemExit(f"DEFAULT_MODELS_JSON: empty value for role '{role_key}'.")
            value_str = canonicalize_model_for_backend(value_str, default_backend)
            default_models[role_key] = value_str

    # Validate DEFAULT_MODEL against the effective global backend.
    # Codex installs validate against CODEX_MODELS only - no fallback
    # to PROVIDER_MODELS["openai"]. Other backends still use the
    # provider-only validator. Catches typos at startup instead of
    # letting them propagate to a confusing runtime failure.
    global_provider = get_effective_provider(default_backend, default_provider)
    if not validate_model_for_backend(default_model, default_backend, global_provider):
        if default_backend == "codex":
            valid = sorted(CODEX_MODELS.keys())
            raise SystemExit(
                f"DEFAULT_MODEL '{default_model}' is not valid for codex (must be one of: {', '.join(valid)})"
            )
        valid = sorted(PROVIDER_MODELS.get(global_provider, {}).keys())
        raise SystemExit(
            f"DEFAULT_MODEL '{default_model}' is not valid for provider "
            f"'{global_provider}' (must be one of: {', '.join(valid)})"
        )

    github_webhook_secret = os.environ.get("GITHUB_WEBHOOK_SECRET", "").strip()
    generic_webhook_secret = os.environ.get("GENERIC_WEBHOOK_SECRET", "").strip()
    legacy_webhook_secret = os.environ.get("WEBHOOK_SECRET", "").strip()
    if legacy_webhook_secret:
        log.warning(
            "WEBHOOK_SECRET is deprecated; temporary compatibility authentication "
            "is enabled for /webhook/github and /webhook only. It never "
            "authenticates Telegram or /api/* routes."
        )

    configured_ingress_secrets = [
        ("TELEGRAM_WEBHOOK_SECRET", telegram_webhook_secret),
        ("GITHUB_WEBHOOK_SECRET", github_webhook_secret),
        ("GENERIC_WEBHOOK_SECRET", generic_webhook_secret),
    ]
    for index, (left_name, left_value) in enumerate(configured_ingress_secrets):
        if not left_value:
            continue
        for right_name, right_value in configured_ingress_secrets[index + 1 :]:
            if right_value and left_value == right_value:
                raise SystemExit(f"{left_name} and {right_name} must use different values")

    return Config(
        telegram_bot_token=token,
        telegram_webhook_url=telegram_webhook_url,
        telegram_webhook_secret=telegram_webhook_secret,
        allowed_user_ids=allowed_ids,
        default_model=default_model,
        default_models=default_models,
        default_timeout=default_timeout,
        agent_max_session_hours=agent_max_session_hours,
        agent_idle_timeout=agent_idle_timeout,
        claude_autocompact_pct=claude_autocompact_pct,
        claude_effort_level=claude_effort_level,
        codex_auth_mode=codex_auth_mode,
        codex_effort_level=codex_effort_level,
        webhook_port=webhook_port,
        github_webhook_secret=github_webhook_secret,
        generic_webhook_secret=generic_webhook_secret,
        webhook_secret=legacy_webhook_secret,
        voice_enabled=os.environ.get("VOICE_ENABLED", "").lower() in ("1", "true", "yes"),
        tts_enabled=os.environ.get("TTS_ENABLED", "").lower() in ("1", "true", "yes"),
        workspace_base=workspace_base,
        allowed_workspaces=allowed_workspaces,
        workspace_configs=workspace_configs,
        memory_projects=memory_projects,
        pr_review_cooldown=pr_review_cooldown,
        pr_review_timeout_s=pr_review_timeout_s,
        github_repo=os.getenv("GITHUB_REPO", ""),
        spec_dir=os.getenv("SPEC_DIR", "specs"),
        file_retention_days=file_retention_days,
        user_configs=user_configs,
        totp_session_minutes=totp_session_minutes,
        totp_challenge_seconds=totp_challenge_seconds,
        totp_lockout_attempts=totp_lockout_attempts,
        totp_lockout_minutes=totp_lockout_minutes,
        default_backend=default_backend,
        default_provider=default_provider,
        memory_enabled=memory_enabled,
        memory_search_limit=memory_search_limit,
        memory_token_budget=memory_token_budget,
        memory_embedding_model=memory_embedding_model,
        memory_extraction_enabled=memory_extraction_enabled,
        memory_extraction_timeout_s=memory_extraction_timeout_s,
        episode_classifier_context_turns=episode_classifier_context_turns,
        memory_consolidation_candidates_n=memory_consolidation_candidates_n,
        memory_episode_timeout_s=memory_episode_timeout_s,
        memory_search_floor=memory_search_floor,
        memory_duplicate_threshold=memory_duplicate_threshold,
    )
