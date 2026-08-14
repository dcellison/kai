"""Backend-neutral agent failure classification and user-safe rendering.

Backend harnesses expose different protocols and native error strings. This
module converts the subset Kai understands confidently into a stable contract
that the Telegram presentation layer consumes today and Workshop workers can
carry across an isolation boundary later.
"""

from enum import StrEnum

from kai.config import Config, get_user_backend_and_provider


class AgentFailureKind(StrEnum):
    """Backend-neutral reason an agent interaction failed."""

    AUTHENTICATION_EXPIRED = "authentication_expired"
    AUTHENTICATION_REQUIRED = "authentication_required"
    QUOTA_EXHAUSTED = "quota_exhausted"
    MODEL_UNAVAILABLE = "model_unavailable"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    TRANSIENT = "transient"
    BACKEND_CRASHED = "backend_crashed"
    UNKNOWN = "unknown"


def classify_agent_failure(error: str | None) -> AgentFailureKind:
    """Conservatively classify a native backend error string.

    Every supported harness has different error envelopes, but their visible
    authentication, quota, and model diagnostics use a small set of stable
    phrases. Matching is deliberately narrow: an unrecognized error retains
    the legacy user-visible behavior instead of receiving misleading recovery
    advice. More specific categories run before process/startup wrappers so
    ``startup failed: OAuth session expired`` remains an authentication error.
    """

    if not error:
        return AgentFailureKind.UNKNOWN

    normalized = " ".join(error.casefold().split())

    expired_markers = (
        "authentication expired",
        "credentials expired",
        "oauth session expired",
        "refresh token expired",
        "session expired and could not be refreshed",
        "token expired",
    )
    if any(marker in normalized for marker in expired_markers):
        return AgentFailureKind.AUTHENTICATION_EXPIRED

    authentication_markers = (
        "auth failed",
        "authentication failed",
        "credentials not found",
        "failed to authenticate",
        "login required",
        "missing credentials",
        "no credentials configured",
        "not authenticated",
        "not logged in",
        "please log in",
    )
    if any(marker in normalized for marker in authentication_markers):
        return AgentFailureKind.AUTHENTICATION_REQUIRED

    quota_markers = (
        "billing hard limit",
        "credits exhausted",
        "insufficient_quota",
        "no credits remaining",
        "quota exceeded",
        "usage limit reached",
    )
    if any(marker in normalized for marker in quota_markers):
        return AgentFailureKind.QUOTA_EXHAUSTED

    model_markers = (
        "model is not supported",
        "model not found",
        "model not supported",
        "unknown model",
        "unsupported model",
    )
    if any(marker in normalized for marker in model_markers):
        return AgentFailureKind.MODEL_UNAVAILABLE

    provider_markers = (
        "failed to connect to provider",
        "provider is unavailable",
        "provider unavailable",
    )
    if any(marker in normalized for marker in provider_markers):
        return AgentFailureKind.PROVIDER_UNAVAILABLE

    transient_markers = (
        "rate limit exceeded",
        "service temporarily unavailable",
        "service unavailable",
        "temporarily unavailable",
    )
    if any(marker in normalized for marker in transient_markers):
        return AgentFailureKind.TRANSIENT

    crashed_markers = (
        "cli not found",
        "process died",
        "process ended unexpectedly",
        "process exited during handshake",
        "startup failed",
    )
    if any(marker in normalized for marker in crashed_markers):
        return AgentFailureKind.BACKEND_CRASHED

    return AgentFailureKind.UNKNOWN


_BACKEND_DISPLAY_NAMES = {
    "claude": "Claude Code",
    "codex": "Codex",
    "goose": "Goose",
    "opencode": "OpenCode",
    "pi": "Pi",
}


def _agent_route_display(backend: str, provider: str) -> str:
    """Return a concise user-facing name for an active agent route."""
    backend_label = _BACKEND_DISPLAY_NAMES.get(backend, backend or "Agent backend")
    provider_label = provider.strip()
    if not provider_label or (backend, provider_label) in {("claude", "anthropic"), ("codex", "openai")}:
        return backend_label
    return f"{backend_label} ({provider_label})"


def _authentication_recovery(
    backend: str,
    provider: str,
    os_user: str | None,
    config: Config,
    failure_kind: AgentFailureKind,
) -> str:
    """Return backend-specific, credential-free sign-in guidance."""
    account = f"OS user {os_user}" if os_user else "the Kai service account"
    if backend == "codex":
        if config.codex_auth_mode == "api_key":
            return "Ask the Kai operator to refresh the OpenAI API credentials configured for Codex, then retry."
        return f"Run `codex login` as {account} on the Kai host, then retry."
    if backend == "opencode":
        return f"Run `opencode auth login` as {account} on the Kai host, then retry."
    if backend == "pi":
        return f"Open `pi` as {account} on the Kai host, use `/login`, then retry."
    if backend == "claude":
        if failure_kind is AgentFailureKind.AUTHENTICATION_EXPIRED:
            return f"Open Claude Code as {account} on the Kai host and sign in again, then retry."
        return f"Refresh the Claude Code credentials for {account} on the Kai host, then retry."
    provider_detail = f" for provider {provider}" if provider else ""
    return f"Ask the Kai operator to refresh the credentials{provider_detail} used by Goose, then retry."


def render_agent_failure(
    failure_kind: AgentFailureKind | None,
    error: str | None,
    config: Config,
    chat_id: int,
    *,
    runtime_route: tuple[str, str, str | None] | None = None,
) -> str:
    """Render a safe, actionable error for a failed backend turn.

    Recognized failures intentionally do not echo the native backend message:
    provider responses may contain account or endpoint details. Unknown errors
    preserve the legacy ``Error: <detail>`` behavior until an adapter-specific
    classifier can identify them confidently.
    """
    detail = error or "no error detail provided"
    kind = failure_kind or AgentFailureKind.UNKNOWN
    if kind is AgentFailureKind.UNKNOWN:
        return f"Error: {detail}"

    user_config = config.get_user_config(chat_id)
    if runtime_route is None:
        backend, provider = get_user_backend_and_provider(user_config, config)
        os_user = user_config.os_user if user_config else None
    else:
        backend, provider, os_user = runtime_route
    route = _agent_route_display(backend, provider)

    if kind in {AgentFailureKind.AUTHENTICATION_EXPIRED, AgentFailureKind.AUTHENTICATION_REQUIRED}:
        state = "has expired" if kind is AgentFailureKind.AUTHENTICATION_EXPIRED else "is required"
        recovery = _authentication_recovery(backend, provider, os_user, config, kind)
        return f"Error: Authentication for {route} {state}. Kai did not complete this request.\n\n{recovery}"
    if kind is AgentFailureKind.QUOTA_EXHAUSTED:
        return (
            f"Error: {route} reported that its credits or usage allowance are exhausted. "
            "Kai did not complete this request. Try again after the allowance resets, or ask the Kai operator "
            "to select another configured backend/provider."
        )
    if kind is AgentFailureKind.MODEL_UNAVAILABLE:
        return (
            f"Error: The configured model is unavailable through {route}. Kai did not complete this request. "
            "Choose another model with /model, or ask the Kai operator to update the configured route."
        )
    if kind is AgentFailureKind.PROVIDER_UNAVAILABLE:
        return (
            f"Error: {route} is currently unavailable. Kai did not complete this request. "
            "Retry later, or ask the Kai operator to select another configured backend/provider."
        )
    if kind is AgentFailureKind.TRANSIENT:
        return f"Error: {route} reported a temporary failure. Kai did not complete this request. Please retry later."
    if kind is AgentFailureKind.BACKEND_CRASHED:
        return (
            f"Error: {route} stopped unexpectedly. Kai did not complete this request. "
            "Retry once; if it happens again, ask the Kai operator to check the service logs."
        )
    return f"Error: {detail}"
