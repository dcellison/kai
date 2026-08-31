"""
Shared HTTP server for integrations, internal APIs, and the Workshop client.

Provides functionality to:
1. Receive and validate GitHub webhook events (push, PR, issues, comments, reviews)
2. Accept generic webhook notifications from any source
3. Expose scheduling and jobs APIs to authenticated internal agents
4. Proxy authenticated requests to external services
5. Publish proactive messages and artifacts to canonical Workshop channels
6. Redeem Workshop client enrollment grants and synchronize authorized timelines
7. Host explicitly configured adapter-owned routes through generic registrars

Routes are organized into these groups:
    - /webhook/github       - GitHub events with HMAC-SHA256 signature validation
    - /webhook              - Generic webhooks with shared-secret auth
    - /api/schedule         - Job creation API (used by persistent agents via curl)
    - /api/jobs             - Job listing and detail API
    - /api/jobs/{id}        - Job detail (GET), deletion (DELETE), and update (PATCH)
    - /api/services/{name}  - External service proxy (injects auth from .env)
    - /api/send-message     - Publish a proactive canonical text message
    - /api/send-file        - Publish a proactive canonical artifact
    - /api/agent-delegations - Run one bounded agent-to-agent delegation
    - /api/memory/add       - Store a structured memory (POST)
    - /api/memory/search    - Search memories by query (POST)
    - /api/memory/stats     - Memory statistics for a user (GET)
    - /api/memory/all       - Delete all memories for a user (DELETE, requires confirm token)
    - /v1/client/enrollment/redeem - Exchange an operator-issued Workshop grant
    - /v1/channels/{id}/timeline   - Read one authorized canonical timeline
    - /v1/channels/{id}/events     - Resume authorized canonical message events

Every core ingress domain has an independent credential. GitHub uses
GITHUB_WEBHOOK_SECRET, and the generic endpoint uses GENERIC_WEBHOOK_SECRET.
Internal API routes always use random, principal-bound process credentials and
do not depend on external webhook secrets. Adapter routes own their credentials
and request handling outside this module.

GitHub events are formatted into human-readable Markdown and routed to
canonical Workshop channels. External delivery is performed later by each
channel binding's outbox worker; GitHub ingress never sends through Telegram
directly. The formatter dispatch makes it easy to add new event types.
"""

import asyncio
import functools
import hashlib
import hmac
import json
import logging
import os
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from aiohttp import web

from kai import memory, services, sessions
from kai.application_host import KaiApplicationHost, KaiCoreServices
from kai.config import (
    DATA_DIR,
    Config,
)
from kai.internal_api_auth import InternalAPIAuth, InternalAPIPrincipal, InternalAPIScope
from kai.job_types import CANONICAL_JOB_TYPES, normalize_job_type
from kai.workshop.agent_delegation import (
    AgentDelegationAuthority,
    AgentDelegationConflict,
    AgentDelegationDenied,
)
from kai.workshop.agent_enablement import WorkshopAgentEnablementService
from kai.workshop.appearance_preferences import WorkshopAppearancePreferenceService
from kai.workshop.artifacts import MAX_ARTIFACT_BYTES, WorkshopArtifactService
from kai.workshop.channel_notification_policy import WorkshopChannelNotificationPolicyService
from kai.workshop.client_api import (
    WorkshopClientCommandSubmitter,
    WorkshopEnrollmentRateLimiter,
    WorkshopEventStreamLimiter,
    register_workshop_command_routes,
    register_workshop_enrollment_routes,
    register_workshop_read_routes,
)
from kai.workshop.client_preferences import WorkshopClientPreferenceService
from kai.workshop.client_sessions import (
    WorkshopBearerSessionAuthenticator,
    WorkshopClientEnrollmentManager,
    WorkshopClientSessionManager,
)
from kai.workshop.client_shell import register_workshop_shell_routes
from kai.workshop.github_automation import (
    GitHubSubscriptionRoute,
    WorkshopGitHubAutomationService,
)
from kai.workshop.github_settings import WorkshopGitHubSettingsService
from kai.workshop.integration_notifications import (
    DEFAULT_INTEGRATION_ROUTE,
    IntegrationNotification,
    WorkshopIntegrationNotificationService,
)
from kai.workshop.internal_api_contexts import WorkshopInternalAPIExecutionContext
from kai.workshop.memory_queries import WorkshopMemoryQueryService
from kai.workshop.notification_preferences import WorkshopNotificationPreferenceService
from kai.workshop.preferences import WorkshopPreferenceService
from kai.workshop.proactive_publication import (
    ProactivePublicationAuthority,
    ProactivePublicationResult,
)
from kai.workshop.routing_eligibility import WorkshopRoutingEligibilityService
from kai.workshop.routing_policy import WorkshopRoutingPolicyService
from kai.workshop.run_previews import WorkshopRunPreviewRegistry
from kai.workshop.scheduled_jobs import (
    WorkshopScheduledJobAuthority,
    WorkshopScheduledJobUpdate,
)
from kai.workshop.scheduler import WorkshopScheduledJobRegistrationError
from kai.workshop.settings_workspaces import WorkshopSettingsWorkspaceService
from kai.workshop.storage_namespaces import (
    WorkshopPrincipalStorageRegistry,
    WorkshopStorageNamespaceError,
)
from kai.workshop.store import WorkshopEventStore

log = logging.getLogger(__name__)


# ── Application state keys ──────────────────────────────────────────
#
# Typed aiohttp AppKey constants for every value this module sets or
# reads on the running Application. aiohttp 4.x is expected to make
# string-key access (`app["foo"]`) an error rather than the current
# NotAppKeyWarning; AppKey is the typed handle pattern that replaces
# string keys. These constants live at module scope so the test
# suite imports them from `kai.webhook` and accesses the same
# Application via the same typed keys, avoiding the same warning at
# test time.
#
# Keep the constants alphabetized by name. The runtime type argument
# is omitted for Union-typed keys (AppKey accepts `type[T] | None`,
# which cannot express `T | None` at runtime); the variable
# annotation carries the type narrowing the call sites rely on.

ALLOWED_WORKSPACES_KEY: web.AppKey[list[str]] = web.AppKey("allowed_workspaces", list)
CONFIG_KEY: web.AppKey[Config] = web.AppKey("config", Config)
CORE_HOST_KEY: web.AppKey[KaiApplicationHost] = web.AppKey("core_host", KaiApplicationHost)
GENERIC_WEBHOOK_SECRET_KEY: web.AppKey[str] = web.AppKey("generic_webhook_secret", str)
GITHUB_WEBHOOK_SECRET_KEY: web.AppKey[str] = web.AppKey("github_webhook_secret", str)
INTERNAL_API_AUTH_KEY: web.AppKey[InternalAPIAuth] = web.AppKey("internal_api_auth", InternalAPIAuth)
WORKSPACE_BASE_KEY: web.AppKey[str | None] = web.AppKey("workspace_base")
WORKSHOP_PRINCIPAL_STORAGE_KEY: web.AppKey[WorkshopPrincipalStorageRegistry] = web.AppKey(
    "workshop_principal_storage", WorkshopPrincipalStorageRegistry
)
WORKSHOP_INTEGRATION_NOTIFICATIONS_KEY: web.AppKey[WorkshopIntegrationNotificationService] = web.AppKey(
    "workshop_integration_notifications", WorkshopIntegrationNotificationService
)
WORKSHOP_GITHUB_AUTOMATION_KEY: web.AppKey[WorkshopGitHubAutomationService] = web.AppKey(
    "workshop_github_automation", WorkshopGitHubAutomationService
)


# Module-level server state, managed by start() and stop()
_app: web.Application | None = None
_runner: web.AppRunner | None = None
_workshop_lan_runner: web.AppRunner | None = None
_HTTP_RUNNER_SHUTDOWN_TIMEOUT = 5.0


async def _resolve_local_repo(repo_full_name: str, app: web.Application) -> str | None:
    """
    Resolve a GitHub repo name to a local filesystem path.

    Matches the repo part of the full name (e.g., "kai" from "dcellison/kai")
    against known workspace locations. Checks in priority order:

    1. WORKSPACE_BASE children
    2. ALLOWED_WORKSPACES entries
    3. workspace_history entries from the database

    Args:
        repo_full_name: Full GitHub repo name (e.g., "dcellison/kai").
        app: The aiohttp application with workspace config.

    Returns:
        Absolute path to the local repo checkout, or None if not found.
    """
    # Extract just the repo name from "owner/repo"
    repo_name = repo_full_name.split("/")[-1]

    # 1. WORKSPACE_BASE - scan immediate children for matching dir name
    workspace_base = app.get(WORKSPACE_BASE_KEY)
    if workspace_base:
        candidate = Path(workspace_base) / repo_name
        if candidate.is_dir():
            return str(candidate)

    # 2. ALLOWED_WORKSPACES - check each entry's directory name
    for allowed in app.get(ALLOWED_WORKSPACES_KEY, []):
        if Path(allowed).name == repo_name and Path(allowed).is_dir():
            return str(allowed)

    # 3. workspace_history - search all users' history since webhook
    # routing has no user context (server-to-server GitHub payload)
    history_paths = await sessions.get_all_workspace_paths(limit=50)
    for path_str in history_paths:
        path = Path(path_str)
        if path.name == repo_name and path.is_dir():
            return str(path)

    return None


def _require_generic_webhook_secret(handler):
    """Validate the dedicated generic webhook credential."""

    @functools.wraps(handler)
    async def wrapper(request: web.Request) -> web.Response:
        secret = request.app[GENERIC_WEBHOOK_SECRET_KEY]
        provided = request.headers.get("X-Webhook-Secret", "")
        if secret and hmac.compare_digest(provided, secret):
            return await handler(request)
        log.warning("Auth failure on %s from %s", request.path, request.remote)
        return web.Response(status=401, text="Invalid secret")

    return wrapper


def _require_internal_api(scope: InternalAPIScope):
    """Require a principal-bound internal API credential with one scope."""

    def decorator(handler):
        @functools.wraps(handler)
        async def wrapper(request: web.Request) -> web.Response:
            credential = request.headers.get("X-Webhook-Secret", "")
            principal = request.app[INTERNAL_API_AUTH_KEY].authenticate(credential)
            if principal is None:
                log.warning("Internal API auth failure on %s from %s", request.path, request.remote)
                return web.Response(status=401, text="Invalid credential")
            if not principal.allows(scope):
                log.warning(
                    "Internal API scope denial on %s for principal %s",
                    request.path,
                    principal.principal_id,
                )
                return web.Response(status=403, text="Credential is not authorized for this operation")
            return await handler(request, principal)

        return wrapper

    return decorator


_INTERNAL_API_IDENTITY_SELECTORS = frozenset(
    {
        "chat_id",
        "user_id",
        "principal_id",
        "channel_id",
        "agent_id",
        "runtime_profile_id",
        "runtime_config_id",
        "run_id",
        "parent_run_id",
        "delegation_id",
        "sponsor_principal_id",
        "requesting_principal_id",
    }
)


def _reject_internal_identity_selectors(payload: dict) -> None:
    """Reject identity fields whose authority is already bound to a credential."""
    supplied = sorted(_INTERNAL_API_IDENTITY_SELECTORS.intersection(payload))
    if supplied:
        raise ValueError(
            f"Identity selector {supplied[0]} is not accepted; the internal API credential already binds "
            "the canonical execution context"
        )


def _scheduled_job_authority(principal: InternalAPIPrincipal) -> WorkshopScheduledJobAuthority:
    return WorkshopScheduledJobAuthority(
        principal.principal_id,
        principal.channel_id,
        principal.agent_id,
        principal.runtime_profile_id,
    )


def _internal_execution_context(
    principal: InternalAPIPrincipal,
) -> WorkshopInternalAPIExecutionContext:
    """Recover the exact canonical lane already bound to one credential."""
    return WorkshopInternalAPIExecutionContext(
        principal.principal_id,
        principal.channel_id,
        principal.agent_id,
        principal.runtime_profile_id,
        principal.private_context,
    )


# ── GitHub event formatters ───────────────────────────────────────────
# Each formatter takes a GitHub webhook payload dict and returns a formatted
# Markdown string for Telegram, or None if the event should be silently ignored.


def _fmt_push(payload: dict) -> str | None:
    """Format a GitHub push event into a Markdown notification."""
    pusher = payload.get("pusher", {}).get("name", "Someone")
    ref = payload.get("ref", "").replace("refs/heads/", "")
    commits = payload.get("commits", [])
    repo = payload.get("repository", {}).get("full_name", "")
    compare = payload.get("compare", "")

    lines = [f"**Push** to `{repo}:{ref}` by {pusher}"]
    for c in commits[:5]:
        sha = c.get("id", "")[:7]
        msg = c.get("message", "").split("\n")[0]
        lines.append(f"  `{sha}` {msg}")
    if len(commits) > 5:
        lines.append(f"  ... and {len(commits) - 5} more")
    if compare:
        lines.append(f"[Compare]({compare})")
    return "\n".join(lines)


def _fmt_pull_request(payload: dict) -> str | None:
    """Format a GitHub pull_request event (opened/closed/merged/reopened)."""
    action = payload.get("action", "")
    if action not in ("opened", "closed", "reopened"):
        return None
    pr = payload.get("pull_request", {})
    merged = pr.get("merged", False)
    if action == "closed" and merged:
        action = "merged"
    title = pr.get("title", "")
    number = pr.get("number", "")
    author = pr.get("user", {}).get("login", "")
    url = pr.get("html_url", "")
    repo = payload.get("repository", {}).get("full_name", "")
    return f"**PR #{number} {action}** in `{repo}`\n{title}\nby {author}\n{url}"


def _fmt_issues(payload: dict) -> str | None:
    """Format a GitHub issues event (opened/closed/reopened)."""
    action = payload.get("action", "")
    if action not in ("opened", "closed", "reopened"):
        return None
    issue = payload.get("issue", {})
    title = issue.get("title", "")
    number = issue.get("number", "")
    author = issue.get("user", {}).get("login", "")
    url = issue.get("html_url", "")
    repo = payload.get("repository", {}).get("full_name", "")
    return f"**Issue #{number} {action}** in `{repo}`\n{title}\nby {author}\n{url}"


def _fmt_issue_comment(payload: dict) -> str | None:
    """Format a GitHub issue_comment event (new comments only)."""
    if payload.get("action") != "created":
        return None
    comment = payload.get("comment", {})
    body = comment.get("body", "")
    if len(body) > 200:
        body = body[:200] + "..."
    author = comment.get("user", {}).get("login", "")
    url = comment.get("html_url", "")
    issue = payload.get("issue", {})
    number = issue.get("number", "")
    repo = payload.get("repository", {}).get("full_name", "")
    return f"**Comment** on #{number} in `{repo}` by {author}\n{body}\n{url}"


def _fmt_pull_request_review(payload: dict) -> str | None:
    """Format a GitHub pull_request_review event (approvals and change requests)."""
    if payload.get("action") != "submitted":
        return None
    review = payload.get("review", {})
    state = review.get("state", "")
    if state not in ("approved", "changes_requested"):
        return None
    reviewer = review.get("user", {}).get("login", "")
    pr = payload.get("pull_request", {})
    number = pr.get("number", "")
    url = review.get("html_url", "")
    repo = payload.get("repository", {}).get("full_name", "")
    label = "approved" if state == "approved" else "requested changes on"
    return f"**{reviewer}** {label} PR #{number} in `{repo}`\n{url}"


# Dispatch table mapping GitHub event type header → formatter function
_GITHUB_FORMATTERS = {
    "push": _fmt_push,
    "pull_request": _fmt_pull_request,
    "issues": _fmt_issues,
    "issue_comment": _fmt_issue_comment,
    "pull_request_review": _fmt_pull_request_review,
}


# ── Signature validation ─────────────────────────────────────────────


def _verify_github_signature(secret: str, body: bytes, signature: str) -> bool:
    """
    Verify a GitHub webhook HMAC-SHA256 signature.

    GitHub signs each webhook payload with the configured secret using
    HMAC-SHA256 and sends the signature in the X-Hub-Signature-256 header.
    This function recomputes the signature and compares using constant-time
    comparison to prevent timing attacks.

    Args:
        secret: The dedicated GITHUB_WEBHOOK_SECRET configured in GitHub and Kai.
        body: The raw request body bytes.
        signature: The X-Hub-Signature-256 header value (e.g., "sha256=abc123...").

    Returns:
        True if the signature is valid, False otherwise.
    """
    if not signature.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


# ── Route handlers ───────────────────────────────────────────────────


async def _handle_health(request: web.Request) -> web.Response:
    """Return liveness plus non-sensitive core component readiness."""
    response: dict[str, object] = {
        "status": "ok",
        "memory_enabled": memory.is_enabled(),
    }
    service_generation = os.environ.get("KAI_SERVICE_GENERATION", "").strip()
    if service_generation.isdigit():
        response["service_generation"] = service_generation
    host = request.app.get(CORE_HOST_KEY)
    if host is not None:
        response["core"] = host.readiness.as_dict()
        response["adapters"] = host.adapter_readiness
    return web.json_response(response)


async def _handle_github(request: web.Request) -> web.Response:
    """
    Handle incoming GitHub webhook events.

    Validates the HMAC-SHA256 signature, parses the event payload, and routes
    it through canonical subscriptions, durable automation, and notification
    channels.

    Supported events: push, pull_request, issues, issue_comment, pull_request_review.
    Unsupported events are silently acknowledged with {"msg": "ignored"}.
    """
    secret = request.app[GITHUB_WEBHOOK_SECRET_KEY]

    body = await request.read()

    # Validate HMAC-SHA256 signature from GitHub
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not secret or not _verify_github_signature(secret, body, signature):
        log.warning("GitHub webhook: invalid signature")
        return web.Response(status=401, text="Invalid signature")

    event_type = request.headers.get("X-GitHub-Event", "")

    # Ping is a connectivity test — just acknowledge
    if event_type == "ping":
        return web.json_response({"msg": "pong"})

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return web.Response(status=400, text="Invalid JSON")

    # Catch unexpected exceptions so GitHub gets a clean 500 rather than an aiohttp traceback.
    try:
        return await _process_github_event(request, payload, event_type)
    except Exception:
        log.exception("Unhandled error processing GitHub %s event", event_type)
        return web.json_response({"msg": "internal_error"}, status=500)


async def _process_github_event(request: web.Request, payload: dict, event_type: str) -> web.Response:
    """Route one authenticated GitHub delivery through canonical services."""
    repo_full_name = str(payload.get("repository", {}).get("full_name", ""))
    delivery_id = request.headers.get("X-GitHub-Delivery", "")
    if not delivery_id:
        return web.json_response({"status": "invalid", "detail": "missing delivery identity"}, status=400)

    automation = request.app[WORKSHOP_GITHUB_AUTOMATION_KEY]
    routes = await automation.routes_for_repository(repo_full_name)
    formatter = _GITHUB_FORMATTERS.get(event_type)
    formatted = formatter(payload) if formatter is not None else None
    if not routes:
        log.warning("GitHub %s event for %s has no canonical subscriber", event_type, repo_full_name)
        if formatted is None:
            return web.json_response({"msg": "ignored"})
        if not await _record_github_default_notification(
            request,
            event_type=event_type,
            repository=repo_full_name,
            body=formatted,
        ):
            return web.json_response({"status": "unavailable"}, status=503)
        return web.json_response({"status": "ok"})

    action = str(payload.get("action", ""))
    local_repo_path: str | None = None
    failures = False
    for route in routes:
        try:
            queued = False
            if (
                event_type == "pull_request"
                and action in {"opened", "reopened", "synchronize"}
                and route.pr_review_enabled
                and route.operations_authorized
            ):
                if local_repo_path is None:
                    local_repo_path = await _resolve_local_repo(repo_full_name, request.app)
                await automation.enqueue(
                    delivery_id=delivery_id,
                    kind="pr_review",
                    event_type=event_type,
                    payload=payload,
                    route=route,
                    local_repo_path=local_repo_path,
                )
                queued = True
            elif (
                event_type == "issues"
                and action == "opened"
                and route.issue_triage_enabled
                and route.operations_authorized
            ):
                await automation.enqueue(
                    delivery_id=delivery_id,
                    kind="issue_triage",
                    event_type=event_type,
                    payload=payload,
                    route=route,
                )
                queued = True

            if not queued and formatted is not None:
                await _record_github_route_notification(
                    request,
                    route=route,
                    event_type=event_type,
                    repository=repo_full_name,
                    body=formatted,
                )
        except Exception:
            failures = True
            log.exception(
                "Canonical GitHub %s routing failed for principal %s",
                event_type,
                route.principal_id,
            )
    return web.json_response({"status": "unavailable" if failures else "ok"}, status=503 if failures else 200)


async def _record_github_route_notification(
    request: web.Request,
    *,
    route: GitHubSubscriptionRoute,
    event_type: str,
    repository: str,
    body: str,
) -> None:
    delivery_id = request.headers.get("X-GitHub-Delivery", "")
    if not delivery_id:
        raise RuntimeError("GitHub notification has no delivery identity")
    await request.app[WORKSHOP_INTEGRATION_NOTIFICATIONS_KEY].record_for_channel(
        IntegrationNotification(
            delivery_id=delivery_id,
            source="github",
            event_type=event_type,
            repository=repository,
            body=body,
            occurred_at=datetime.now(UTC),
        ),
        route.notification_channel_id,
    )


async def _record_github_default_notification(
    request: web.Request,
    *,
    event_type: str,
    repository: str,
    body: str,
) -> bool:
    service = request.app.get(WORKSHOP_INTEGRATION_NOTIFICATIONS_KEY)
    delivery_id = request.headers.get("X-GitHub-Delivery", "")
    if service is None or not delivery_id:
        return False
    try:
        recorded = await service.record_for_default_admin(
            IntegrationNotification(
                delivery_id=delivery_id,
                source="github",
                event_type=event_type,
                repository=repository,
                body=body,
                occurred_at=datetime.now(UTC),
            )
        )
    except ValueError:
        log.warning("GitHub %s delivery has an unsupported identity", event_type)
        return False
    log.info(
        "Recorded GitHub %s notification as canonical default-admin message %s (inserted=%s, deliveries=%d)",
        event_type,
        recorded.message_id,
        recorded.inserted,
        len(recorded.deliveries),
    )
    return True


@_require_generic_webhook_secret
async def _handle_generic(request: web.Request) -> web.Response:
    """
    Handle generic webhook notifications from any source.

    Extracts a "message" field from the JSON payload (or dumps the full
    payload), records it through the explicit canonical generic/default route,
    and requests delivery through any configured channel bindings.
    """
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return web.Response(status=400, text="Invalid JSON")

    # Reject non-object JSON shapes (null, lists, scalars, strings) before
    # any payload.get() call. JSON bodies like `null`, `[]`, `42`, or
    # `"string"` parse successfully (no JSONDecodeError) but raise
    # AttributeError on .get(), which would escape as an unstyled HTML
    # 500 traceback instead of a clean JSON 400.
    if not isinstance(payload, dict):
        return web.json_response({"error": "Request body must be a JSON object"}, status=400)

    # Use the "message" field if present (including empty string),
    # otherwise dump the full JSON. `is not None` avoids treating "" as absent.
    msg = payload.get("message")
    text = msg if msg is not None else json.dumps(payload, indent=2)
    if not isinstance(text, str):
        return web.json_response({"error": "message must be a string"}, status=400)
    if not text:
        return web.json_response({"error": "message must not be empty"}, status=400)
    if len(text) > 4_096_000:
        return web.json_response({"error": "message is too large"}, status=413)

    service = request.app[WORKSHOP_INTEGRATION_NOTIFICATIONS_KEY]
    delivery_id = (
        request.headers.get("Idempotency-Key")
        or request.headers.get("X-Kai-Delivery")
        or f"generated-{uuid.uuid4().hex}"
    )
    try:
        notification = IntegrationNotification(
            delivery_id=delivery_id,
            source="generic",
            event_type="notification",
            body=text,
            occurred_at=datetime.now(UTC),
        )
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    try:
        recorded = await service.record_for_route(
            notification,
            route_name=DEFAULT_INTEGRATION_ROUTE,
        )
    except Exception:
        log.exception("Failed to record generic webhook notification")
        return web.json_response({"status": "unavailable"}, status=503)

    return web.json_response(
        {
            "status": "ok",
            "message_id": str(recorded.message_id),
            "inserted": recorded.inserted,
        }
    )


# ── Scheduling API ───────────────────────────────────────────────────

# Valid schedule types accepted by the scheduling API
_VALID_SCHEDULE_TYPES = ("once", "daily", "interval")


def _validate_schedule_data(
    schedule_data: dict | str,
    schedule_type: str,
) -> tuple[str | None, str | None]:
    """
    Validate schedule_data and return the serialized JSON string.

    Handles both dict and pre-serialized string inputs. Validates
    that the JSON structure matches what the given schedule_type
    requires so the core scheduler never encounters malformed data at fire time.

    Args:
        schedule_data: Either a dict or a JSON string.
        schedule_type: One of "once", "daily", "interval".

    Returns:
        (json_string, None) on success, or (None, error_message) on failure.
    """
    # Parse string input into a dict for structural validation
    if isinstance(schedule_data, str):
        try:
            parsed = json.loads(schedule_data)
        except json.JSONDecodeError:
            return None, "schedule_data is not valid JSON"
        if not isinstance(parsed, dict):
            return None, "schedule_data must be a JSON object"
    else:
        if not isinstance(schedule_data, dict):
            return None, "schedule_data must be a JSON object"
        parsed = schedule_data

    # Structural validation per schedule_type
    if schedule_type == "once":
        if "run_at" not in parsed:
            return None, "schedule_data for 'once' requires 'run_at'"
        # Validate it parses as an ISO datetime string
        run_at = parsed["run_at"]
        if not isinstance(run_at, str):
            return None, "schedule_data 'run_at' must be a string"
        try:
            datetime.fromisoformat(run_at)
        except ValueError:
            return None, "schedule_data 'run_at' is not a valid ISO datetime"

    elif schedule_type == "daily":
        if "times" not in parsed:
            return None, "schedule_data for 'daily' requires 'times'"
        times = parsed["times"]
        if not isinstance(times, list) or len(times) == 0:
            return None, "schedule_data 'times' must be a non-empty list"
        for t in times:
            if not isinstance(t, str) or ":" not in t:
                return None, f"schedule_data 'times' entry '{t}' is not a valid HH:MM string"
            try:
                parts = t.split(":")
                if len(parts) != 2:
                    raise ValueError
                h, m = int(parts[0]), int(parts[1])
                if not (0 <= h <= 23 and 0 <= m <= 59):
                    raise ValueError
            except (ValueError, IndexError):
                return None, f"schedule_data 'times' entry '{t}' is not a valid HH:MM string"

    elif schedule_type == "interval":
        if "seconds" not in parsed:
            return None, "schedule_data for 'interval' requires 'seconds'"
        seconds = parsed["seconds"]
        # bool is a subclass of int in Python, so reject it explicitly
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds <= 0:
            return None, "schedule_data 'seconds' must be a positive number"

    else:
        return None, f"unknown schedule_type '{schedule_type}'"

    return json.dumps(parsed), None


@_require_internal_api(InternalAPIScope.JOBS_WRITE)
async def _handle_schedule(request: web.Request, principal: InternalAPIPrincipal) -> web.Response:
    """
    Create a new scheduled job via the HTTP API.

    This is the primary interface for persistent agent backends to create
    scheduled tasks from within the workspace.

    Required JSON fields: name, prompt, schedule_type, schedule_data.
    Optional fields: job_type (default "reminder"), auto_remove (default false),
        notify_on_check (default false).

    The job is persisted to the database and immediately registered with the
    core-owned scheduler so it starts firing without a restart.

    Returns:
        JSON with job_id and name on success, or an error message on failure.
    """
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    # Reject non-object JSON shapes (null, lists, scalars, strings) before
    # any payload.get() call. JSON bodies like `null`, `[]`, `42`, or
    # `"string"` parse successfully (no JSONDecodeError) but raise
    # AttributeError on .get(), which would escape as an unstyled HTML
    # 500 traceback instead of a clean JSON 400.
    if not isinstance(payload, dict):
        return web.json_response({"error": "Request body must be a JSON object"}, status=400)

    # Extract and validate required fields
    name = payload.get("name")
    prompt = payload.get("prompt")
    schedule_type = payload.get("schedule_type")
    schedule_data = payload.get("schedule_data")

    # Use `is None` checks so empty strings (e.g., prompt="") are not
    # rejected as missing. Truthiness would treat "" as absent.
    if name is None or prompt is None or schedule_type is None or schedule_data is None:
        return web.json_response(
            {"error": "Missing required fields: name, prompt, schedule_type, schedule_data"},
            status=400,
        )

    if schedule_type not in _VALID_SCHEDULE_TYPES:
        return web.json_response(
            {"error": f"schedule_type must be one of: {', '.join(_VALID_SCHEDULE_TYPES)}"},
            status=400,
        )

    # Normalize the retired job-type alias at ingress so canonical storage
    # never records a backend-specific scheduler identifier.
    try:
        job_type = normalize_job_type(payload.get("job_type", "reminder"))
    except ValueError:
        return web.json_response(
            {"error": f"job_type must be one of: {', '.join(CANONICAL_JOB_TYPES)}"},
            status=400,
        )
    auto_remove = payload.get("auto_remove", False)
    notify_on_check = payload.get("notify_on_check", False)
    try:
        _reject_internal_identity_selectors(payload)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)

    # Validate schedule_data structure before persisting. Catches malformed
    # JSON strings and wrong-shape payloads (e.g., interval keys on a daily
    # job) that would otherwise fail the core scheduler on every fire attempt.
    schedule_data_str, error = _validate_schedule_data(schedule_data, schedule_type)
    if error:
        return web.json_response({"error": error}, status=400)
    assert schedule_data_str is not None  # guaranteed when error is None

    scheduler = request.app[CORE_HOST_KEY].services.scheduler
    try:
        job_id = await scheduler.create_job(
            _scheduled_job_authority(principal),
            name=name,
            job_type=job_type,
            prompt=prompt,
            schedule_type=schedule_type,
            schedule_data=schedule_data_str,
            auto_remove=auto_remove,
            notify_on_check=notify_on_check,
        )
    except WorkshopScheduledJobRegistrationError:
        log.exception("Failed to register created job")
        return web.json_response({"error": "Failed to register job"}, status=500)
    except Exception:
        log.exception("Failed to create job")
        return web.json_response({"error": "Failed to create job"}, status=500)

    log.info("Scheduled job %d '%s' via API (%s)", job_id, name, schedule_type)
    return web.json_response({"job_id": job_id, "name": name})


# ── Jobs API ─────────────────────────────────────────────────────────


@_require_internal_api(InternalAPIScope.JOBS_READ)
async def _handle_get_jobs(request: web.Request, principal: InternalAPIPrincipal) -> web.Response:
    """
    List all active jobs for the configured chat.

    Used by persistent agents to check what jobs are currently scheduled
    without needing to parse Telegram bot command output.
    """

    try:
        _reject_internal_identity_selectors(dict(request.query))
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    jobs = await request.app[CORE_HOST_KEY].services.scheduler.list_jobs(_scheduled_job_authority(principal))
    return web.json_response(jobs)


@_require_internal_api(InternalAPIScope.JOBS_READ)
async def _handle_get_job(request: web.Request, principal: InternalAPIPrincipal) -> web.Response:
    """
    Get a single job by its database ID.

    Returns the full job record as JSON, or 404 if not found.
    """

    try:
        job_id = int(request.match_info["id"])
    except ValueError:
        return web.json_response({"error": "Invalid job ID"}, status=400)

    try:
        _reject_internal_identity_selectors(dict(request.query))
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    job = await request.app[CORE_HOST_KEY].services.scheduler.get_job(
        job_id,
        _scheduled_job_authority(principal),
    )
    if job is None:
        return web.json_response({"error": "Job not found"}, status=404)
    return web.json_response(job)


@_require_internal_api(InternalAPIScope.JOBS_WRITE)
async def _handle_delete_job(request: web.Request, principal: InternalAPIPrincipal) -> web.Response:
    """
    Delete a scheduled job by ID via the HTTP API.

    Removes the job from both the database and core scheduler.
    Returns 404 if the job doesn't exist.
    """

    try:
        job_id = int(request.match_info["id"])
    except ValueError:
        return web.json_response({"error": "Invalid job ID"}, status=400)

    try:
        _reject_internal_identity_selectors(dict(request.query))
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    scheduler = request.app[CORE_HOST_KEY].services.scheduler
    deleted = await scheduler.delete_job(job_id, _scheduled_job_authority(principal))
    if not deleted:
        return web.json_response({"error": "Job not found"}, status=404)

    log.info("Deleted job %d via API", job_id)
    return web.json_response({"deleted": job_id})


@_require_internal_api(InternalAPIScope.JOBS_WRITE)
async def _handle_update_job(request: web.Request, principal: InternalAPIPrincipal) -> web.Response:
    """
    Update a scheduled job's mutable fields via the HTTP API.

    Accepts a JSON body with any of: name, prompt, schedule_type,
    schedule_data, auto_remove, notify_on_check. Only provided fields
    are updated. If the schedule changes (type or data), the job is
    re-registered with the core scheduler to pick up the new timing.

    Returns 404 if the job doesn't exist or is inactive.
    """

    try:
        job_id = int(request.match_info["id"])
    except ValueError:
        return web.json_response({"error": "Invalid job ID"}, status=400)

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    # Reject non-object JSON shapes (null, lists, scalars, strings) before
    # any payload.get() call. JSON bodies like `null`, `[]`, `42`, or
    # `"string"` parse successfully (no JSONDecodeError) but raise
    # AttributeError on .get(), which would escape as an unstyled HTML
    # 500 traceback instead of a clean JSON 400.
    if not isinstance(payload, dict):
        return web.json_response({"error": "Request body must be a JSON object"}, status=400)

    existing_job: dict | None = None

    # Validate schedule_type if provided
    new_schedule_type = payload.get("schedule_type")
    if new_schedule_type and new_schedule_type not in _VALID_SCHEDULE_TYPES:
        return web.json_response(
            {"error": f"schedule_type must be one of: {', '.join(_VALID_SCHEDULE_TYPES)}"},
            status=400,
        )

    # Validate schedule_data if provided. For PATCH, schedule_type might come
    # from the payload (changing the schedule) or from the existing job (keeping
    # the same type but updating the data). We need the effective schedule_type
    # to validate the shape.
    schedule_data = payload.get("schedule_data")
    if schedule_data is not None:
        # If schedule_type is also being changed, validate against the new one.
        # Otherwise, fetch the current job to get its existing schedule_type.
        effective_type = new_schedule_type
        if effective_type is None:
            existing_job = await request.app[CORE_HOST_KEY].services.scheduler.get_job(
                job_id,
                _scheduled_job_authority(principal),
            )
            if existing_job is None:
                return web.json_response({"error": "Job not found or inactive"}, status=404)
            effective_type = existing_job["schedule_type"]
        schedule_data, error = _validate_schedule_data(schedule_data, effective_type)
        if error:
            return web.json_response({"error": error}, status=400)

    try:
        _reject_internal_identity_selectors(payload)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)

    # If the schedule changes, keep the previous row so a failed scheduler
    # re-registration can be compensated. Without this, PATCH can leave the DB
    # saying a job is active on the new schedule while the core scheduler has
    # no corresponding task.
    schedule_changed = new_schedule_type is not None or schedule_data is not None
    if schedule_changed:
        if existing_job is None:
            existing_job = await request.app[CORE_HOST_KEY].services.scheduler.get_job(
                job_id,
                _scheduled_job_authority(principal),
            )
        if existing_job is None:
            return web.json_response({"error": "Job not found or inactive"}, status=404)

    scheduler = request.app[CORE_HOST_KEY].services.scheduler
    try:
        updated = await scheduler.update_job(
            job_id,
            _scheduled_job_authority(principal),
            WorkshopScheduledJobUpdate(
                name=payload.get("name"),
                prompt=payload.get("prompt"),
                schedule_type=new_schedule_type,
                schedule_data=schedule_data,
                auto_remove=payload.get("auto_remove"),
                notify_on_check=payload.get("notify_on_check"),
            ),
        )
    except Exception:
        log.exception("Failed to update and re-register job %d", job_id)
        return web.json_response({"error": "Failed to register job"}, status=500)

    if not updated:
        return web.json_response({"error": "Job not found or inactive"}, status=404)

    log.info("Updated job %d via API", job_id)
    return web.json_response({"updated": job_id})


# ── Service proxy ────────────────────────────────────────────────────


@_require_internal_api(InternalAPIScope.SERVICES_CALL)
async def _handle_service_call(request: web.Request, principal: InternalAPIPrincipal) -> web.Response:
    """
    Proxy an authenticated request to an external service.

    This is how the inner Claude process calls external APIs without ever
    seeing API keys. Claude POSTs to /api/services/{name} with an optional
    JSON body containing `body`, `params`, and/or `path_suffix`. This handler
    resolves the service definition, injects auth from .env, makes the HTTP
    call, and returns the response.

    Request JSON fields (all optional):
        body: dict — JSON body forwarded to the external API
        params: dict — query parameters merged with static config params
        path_suffix: str — appended to the service's base URL

    Returns:
        JSON {"status": N, "body": "..."} on success, or
        JSON {"error": "..."} with HTTP 502 on failure.
    """

    # Extract service name from URL path
    service_name = request.match_info["name"]
    if not principal.allows_service(service_name):
        log.warning(
            "Internal API service denial for principal %s: %s",
            principal.principal_id,
            service_name,
        )
        return web.json_response({"error": "Service is not authorized for this principal"}, status=403)

    # Parse optional JSON body with request parameters. The body is
    # genuinely optional here (services with no JSON-body config still
    # work), so JSONDecodeError is silently treated as "no body provided"
    # and the field defaults stay in effect. A non-object JSON value
    # (null, lists, scalars, strings) is treated as a structural error
    # and rejected with 400, matching the other handlers - silently
    # accepting `null` as "no body" would mask client bugs that send a
    # malformed body intending to send fields.
    body = None
    params = None
    path_suffix = ""
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            return web.json_response({"error": "Request body must be a JSON object"}, status=400)
        body = payload.get("body")
        params = payload.get("params")
        path_suffix = payload.get("path_suffix", "")
    except json.JSONDecodeError:
        pass  # No body is fine — all fields are optional

    result = await services.call_service(
        service_name,
        body=body,
        params=params,
        path_suffix=path_suffix,
    )

    if result.success:
        return web.json_response({"status": result.status, "body": result.body})
    else:
        return web.json_response({"error": result.error}, status=502)


# ── Messaging ────────────────────────────────────────────────────────


def _proactive_authority(principal: InternalAPIPrincipal) -> ProactivePublicationAuthority:
    return ProactivePublicationAuthority(
        principal_id=principal.principal_id,
        channel_id=principal.channel_id,
        agent_id=principal.agent_id,
        runtime_profile_id=principal.runtime_profile_id,
    )


def _proactive_response(result: ProactivePublicationResult, **extra: object) -> web.Response:
    return web.json_response(
        {
            "status": "recorded",
            "delivery": result.delivery_status,
            "deliveries": len(result.deliveries),
            **extra,
        }
    )


@_require_internal_api(InternalAPIScope.MESSAGES_SEND)
async def _handle_send_message(request: web.Request, principal: InternalAPIPrincipal) -> web.Response:
    """
    Publish a proactive agent message to the credential's canonical channel.

    Called by the inner Claude process to proactively notify the user - e.g.,
    when a background task completes, or a scheduled job wants to report
    results without going through the full Claude prompt cycle.

    Optional adapter delivery is requested durably after canonical recording.
    An optional idempotency_key makes an uncertain caller retry resolve to the
    same message and delivery work.

    Returns:
        Recorded status plus queued/delivered/not-configured adapter state.
    """
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    # Reject non-object JSON shapes (null, lists, scalars, strings) before
    # any payload.get() call. JSON bodies like `null`, `[]`, `42`, or
    # `"string"` parse successfully (no JSONDecodeError) but raise
    # AttributeError on .get(), which would escape as an unstyled HTML
    # 500 traceback instead of a clean JSON 400.
    if not isinstance(payload, dict):
        return web.json_response({"error": "Request body must be a JSON object"}, status=400)

    raw_text = payload.get("text", "")
    if not isinstance(raw_text, str):
        return web.json_response({"error": "text must be a string"}, status=400)
    text = raw_text.strip()
    if not text:
        return web.json_response({"error": "Missing required field: text"}, status=400)

    try:
        _reject_internal_identity_selectors(payload)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)

    try:
        request_id = payload.get("idempotency_key")
        if request_id is None:
            request_id = uuid.uuid4().hex
        result = await request.app[CORE_HOST_KEY].services.proactive_publication.publish_text(
            _proactive_authority(principal),
            request_id=request_id,
            body=text,
            occurred_at=datetime.now(UTC),
        )
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except Exception:
        log.exception("Canonical proactive message publication failed")
        return web.json_response({"error": "Message publication failed"}, status=500)

    log.info("Recorded proactive canonical message (%d chars)", len(text))
    return _proactive_response(result)


@_require_internal_api(InternalAPIScope.AGENTS_DELEGATE)
async def _handle_agent_delegation(
    request: web.Request,
    principal: InternalAPIPrincipal,
) -> web.Response:
    """Run one explicit delegation from the credential's active attempt."""
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    if not isinstance(payload, dict):
        return web.json_response({"error": "Request body must be a JSON object"}, status=400)
    try:
        _reject_internal_identity_selectors(payload)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    unsupported = set(payload) - {
        "target_handle",
        "task",
        "context",
        "idempotency_key",
    }
    if unsupported:
        return web.json_response(
            {"error": f"Unsupported field: {sorted(unsupported)[0]}"},
            status=400,
        )
    missing = [key for key in ("target_handle", "task", "idempotency_key") if key not in payload]
    if missing:
        return web.json_response({"error": f"Missing required field: {missing[0]}"}, status=400)
    try:
        result = await request.app[CORE_HOST_KEY].services.agent_delegation.delegate(
            AgentDelegationAuthority(
                sponsor_principal_id=principal.principal_id,
                channel_id=principal.channel_id,
                caller_agent_id=principal.agent_id,
                runtime_profile_id=principal.runtime_profile_id,
            ),
            target_handle=payload["target_handle"],
            task=payload["task"],
            context=payload.get("context"),
            idempotency_key=payload["idempotency_key"],
        )
    except AgentDelegationDenied as exc:
        status = 400 if exc.code.startswith("invalid_") or exc.code.endswith("_too_large") else 409
        return web.json_response(
            {"error": str(exc), "code": exc.code},
            status=status,
        )
    except AgentDelegationConflict as exc:
        return web.json_response({"error": str(exc), "code": "idempotency_conflict"}, status=409)
    except Exception:
        log.exception("Canonical agent delegation failed")
        return web.json_response({"error": "Agent delegation failed"}, status=500)
    delegation = result.delegation
    return web.json_response(
        {
            "version": 1,
            "delegation_id": str(delegation.delegation_id),
            "child_run_id": str(delegation.child_run_id),
            "status": delegation.status,
            "outcome_code": delegation.outcome_code,
            "response": result.response,
            "limits": {
                "depth": delegation.depth,
                "target_handle": delegation.target_handle,
            },
        }
    )


# ── File exchange ────────────────────────────────────────────────────


@_require_internal_api(InternalAPIScope.FILES_SEND)
async def _handle_send_file(request: web.Request, principal: InternalAPIPrincipal) -> web.Response:
    """
    Publish a file from the filesystem as a canonical channel artifact.

    Called by the inner Claude process to deliver files back to the user.
    Accepts a JSON body with a required "path" field (absolute path) and
    optional "caption" and "idempotency_key". Optional adapters deliver the
    artifact later through durable channel-binding workers.

    Path confinement: the resolved path must be inside the authenticated
    principal's current workspace or its scoped upload directory. This
    prevents path traversal attacks via symlinks or "../" and prevents one
    principal from sending files from another principal's upload directory.

    Returns:
        Recorded status, adapter state, and canonical filename on success.
    """
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    # Reject non-object JSON shapes (null, lists, scalars, strings) before
    # any payload.get() call. JSON bodies like `null`, `[]`, `42`, or
    # `"string"` parse successfully (no JSONDecodeError) but raise
    # AttributeError on .get(), which would escape as an unstyled HTML
    # 500 traceback instead of a clean JSON 400.
    if not isinstance(payload, dict):
        return web.json_response({"error": "Request body must be a JSON object"}, status=400)

    file_path = payload.get("path")
    if not file_path:
        return web.json_response({"error": "Missing required field: path"}, status=400)
    if not isinstance(file_path, str):
        return web.json_response({"error": "path must be a string"}, status=400)

    # Identity and authority are bound to the credential. Caller-supplied
    # selectors are rejected before any filesystem lookup.
    try:
        _reject_internal_identity_selectors(payload)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)

    path = Path(file_path).resolve()

    # Confine to the requesting user's workspace to prevent path traversal.
    # Uses Path.relative_to() which raises ValueError on escape. Each user
    # has their own per-user home workspace resolved from the pool (#353).
    # When the pool is unavailable (transient startup state) we refuse the
    # request rather than opening a global fallback path.
    core_host = request.app.get(CORE_HOST_KEY)
    if core_host is None:
        return web.json_response({"error": "No workspace configured"}, status=403)
    try:
        workspace = str(
            await core_host.services.runtime_pool.get_effective_workspace(_internal_execution_context(principal))
        )
    except Exception:
        log.exception("Internal file workspace resolution failed")
        return web.json_response({"error": "No workspace configured"}, status=403)

    # Allow files from the effective workspace or this authenticated
    # principal's canonical upload directory. Numeric compatibility
    # directories are archives and are never protected runtime read roots.
    workspace_resolved = Path(workspace).resolve()
    storage_registry = request.app.get(WORKSHOP_PRINCIPAL_STORAGE_KEY)
    if storage_registry is None:
        return web.json_response({"error": "Principal storage unavailable"}, status=403)
    try:
        storage_namespace = storage_registry.for_runtime_profile(principal.runtime_profile_id)
    except WorkshopStorageNamespaceError:
        return web.json_response({"error": "Principal storage unavailable"}, status=403)
    if storage_namespace.principal_id != principal.principal_id:
        return web.json_response({"error": "Principal storage unavailable"}, status=403)
    allowed_roots = (
        workspace_resolved,
        storage_namespace.files_directory(DATA_DIR).resolve(),
    )
    if not any(path.is_relative_to(root) for root in allowed_roots):
        return web.json_response({"error": "Path outside allowed directories"}, status=403)

    if not path.is_file():
        return web.json_response({"error": f"File not found: {file_path}"}, status=404)

    caption = payload.get("caption", "")
    if not isinstance(caption, str):
        return web.json_response({"error": "caption must be a string"}, status=400)
    try:
        request_id = payload.get("idempotency_key")
        if request_id is None:
            request_id = uuid.uuid4().hex
        result = await request.app[CORE_HOST_KEY].services.proactive_publication.publish_file(
            _proactive_authority(principal),
            request_id=request_id,
            path=path,
            caption=caption,
            occurred_at=datetime.now(UTC),
        )
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except Exception:
        log.exception("Canonical proactive artifact publication failed")
        return web.json_response({"error": "File publication failed"}, status=500)

    log.info("Recorded proactive canonical artifact %s", path.name)
    return _proactive_response(result, file=path.name)


# ── Memory API ───────────────────────────────────────────────────────
# Four thin REST handlers wrapping memory.py primitives. Implementation
# notes that apply uniformly across all four:
#
#   - All four share the symmetric `is_enabled()` precheck. Returns 503
#     when memory is off, even on read endpoints whose primitives degrade
#     gracefully (search returns [], get_stats returns zeroed). Returning
#     the degraded value at the API would conflate "memory off" with
#     "no data" and inner Claude could not pick a retry policy from the
#     status code alone.
#
#   - Memory primitives take user_id as a string. Every handler passes the
#     credential-bound canonical principal ID; memory.py's configured
#     authority resolver adds and verifies the complete canonical provenance
#     tuple without creating a second namespace.
#
#   - Handlers import memory as a module (`from kai import memory`) so
#     tests can patch `kai.memory.<func>` uniformly. Function-level
#     imports would force tests to patch the local binding
#     (kai.webhook.add_structured), which is a known mocking pitfall.

# Static well-known token for the DELETE /api/memory/all confirmation
# field. Picked over a per-request UUID because the threat model is
# typo-resistance and prompt-injection defense, not replay protection;
# a static descriptive string is sufficient for both. The exact value
# also appears in the 400 error body so a stray curl gets a clear hint
# at what was missing.
_DELETE_ALL_CONFIRM_TOKEN = "delete-all-memories"


def _memory_disabled_response() -> web.Response:
    """503 response used by the symmetric is_enabled() precheck.

    Centralized so all four memory handlers return the identical body and
    status. The 503 contract (vs the 500 used for `add_structured` failure
    in /api/memory/add) is documented in CLAUDE.md: 503 means the
    operator must enable memory, so callers should NOT retry; 500 means
    the underlying store call failed and may be transient.
    """
    return web.json_response({"error": "Memory system disabled"}, status=503)


@_require_internal_api(InternalAPIScope.MEMORY_ADD)
async def _handle_memory_add(request: web.Request, principal: InternalAPIPrincipal) -> web.Response:
    """
    Store a structured memory via memory.add_structured().

    Wraps the existing add_structured() primitive used by the Haiku
    extraction path. Lets inner Claude (and other localhost clients)
    deliberately store facts without waiting for the extractor to pass
    over a conversation. The primitive itself is unchanged; this handler
    only adds the HTTP surface.

    Request body (JSON):
        content: str, required, non-empty after strip
        memory_type: str, optional, default "fact"
        tags: list[str], optional
        metadata: dict, optional (keys "type" and "tags" are reserved by
            add_structured; user values for those will be overwritten)
    Provenance is stamped server-side: `source` is always "explicit"
    (a caller-supplied value is overridden, so the convention cannot
    drift per caller), the scope fields are routed from the caller's
    detected workspace project exactly like the extraction path with
    no classifier hint, and `speaker`/`confidence` receive defaults
    ("assistant", 0.9) that the caller MAY override, since a caller
    relaying a user-stated fact legitimately knows the speaker.

    Responses:
        200 {"id": "<mem0-uuid>"} on success
        400 on bad input or a retired caller identity selector
        401 on bad/missing internal API credential (handled by the decorator)
        503 when memory is disabled (precheck; do not retry, escalate)
        500 when add_structured() returns None despite memory being enabled
            (the underlying store call failed; may be transient)
    """
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    # Reject non-object JSON shapes (null, lists, scalars, strings) before
    # any payload.get() call. JSON bodies like `null`, `[]`, `42`, or
    # `"string"` parse successfully (no JSONDecodeError) but raise
    # AttributeError on .get(), which would escape as an unstyled HTML
    # 500 traceback instead of a clean JSON 400.
    if not isinstance(payload, dict):
        return web.json_response({"error": "Request body must be a JSON object"}, status=400)

    # Required-field validation runs BEFORE the is_enabled() precheck so
    # callers learn about bad-input bugs even when memory is off. Without
    # this ordering, a caller debugging a missing-field bug while memory
    # happens to be disabled would chase a 503 red herring instead of
    # seeing the real 400.
    #
    # isinstance(...) guards before .strip() / iteration are deliberate.
    # JSON delivers numbers, booleans, lists, nulls, and dicts; calling
    # `.strip()` on a number raises AttributeError that escapes to
    # aiohttp as a framework 500 with no clean error body. Same logic
    # for the optional-field checks below: validate at the API boundary
    # so the 400 surface stays the contract that callers actually see.
    content = payload.get("content")
    if not isinstance(content, str) or not content.strip():
        return web.json_response({"error": "Missing required field: content"}, status=400)
    content = content.strip()

    # Two-step handling: collapse explicit None to the default first, then
    # type-check. This makes `{"memory_type": null}` behave the same as
    # omitting the field entirely (both fall back to "fact"), matching the
    # lenient None handling used for `tags` and `metadata` below.
    # `payload.get("memory_type", "fact")` alone would only apply the
    # default when the key is ABSENT - an explicit null would slip through
    # and fail isinstance, surprising callers who reasonably expect null
    # and missing-key to be equivalent in JSON.
    memory_type = payload.get("memory_type")
    if memory_type is None:
        memory_type = "fact"
    if not isinstance(memory_type, str):
        return web.json_response({"error": "memory_type must be a string"}, status=400)

    tags = payload.get("tags")
    if tags is not None and (not isinstance(tags, list) or not all(isinstance(t, str) for t in tags)):
        return web.json_response({"error": "tags must be a list of strings"}, status=400)

    metadata = payload.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        return web.json_response({"error": "metadata must be a JSON object"}, status=400)

    # The two caller-overridable provenance keys get the same 400
    # treatment as every other field: these values feed arithmetic
    # (ranking multiplies weight by confidence, the browser sorts by
    # it) and string formatting in now-visible UI rows, so a stored
    # bad type surfaces later as a broken facts browser rather than a
    # clean error here. bool is rejected for confidence because JSON
    # true/false pass isinstance(int) and would silently rank as 1/0.
    if metadata is not None:
        speaker = metadata.get("speaker")
        if speaker is not None and not isinstance(speaker, str):
            return web.json_response({"error": "metadata.speaker must be a string"}, status=400)
        confidence = metadata.get("confidence")
        if confidence is not None and (
            isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0
        ):
            return web.json_response({"error": "metadata.confidence must be a number in [0.0, 1.0]"}, status=400)

    try:
        _reject_internal_identity_selectors(payload)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)

    # Symmetric is_enabled() precheck. Runs after auth + 400-level
    # validation but before the primitive call. See the §Memory API
    # block comment above for why all four memory handlers share this.
    if not memory.is_enabled():
        return _memory_disabled_response()

    user_id = str(principal.principal_id)

    # Server-side provenance stamp. Lazy imports keep
    # memory_extraction (and its one-shot reasoner stack) out of this
    # module's import graph for callers that never write memories.
    from kai.memory_extraction import _route_write_scope
    from kai.memory_projects import detect_active_memory_project, merged_registry

    final_metadata = dict(metadata) if metadata else {}
    # Defaults the caller may override; a caller relaying a
    # user-stated fact legitimately knows the speaker.
    final_metadata.setdefault("speaker", "assistant")
    final_metadata.setdefault("confidence", 0.9)
    # The stamp the caller may NOT override: provenance conventions
    # drifted when callers self-reported this value from doc examples.
    final_metadata["source"] = "explicit"
    # Scope routed from the caller's detected workspace project, same
    # rules as the extraction path with no classifier hint. A missing
    # pool (transient startup state) collapses to no-workspace
    # semantics: global scope, the same posture scoped retrieval takes.
    # The workspace resolution is guarded because it is not a pure
    # read (settings DB, path validation); a transient failure there
    # must surface as the handler's clean JSON 500, not mis-scope the
    # write to global silently or escape as a framework HTML 500.
    try:
        api_config: Config = request.app[CONFIG_KEY]
        workspace = await request.app[CORE_HOST_KEY].services.runtime_pool.get_effective_workspace(
            _internal_execution_context(principal)
        )
        active_project = detect_active_memory_project(
            workspace,
            merged_registry(api_config.memory_projects),
        )
    except Exception:
        log.exception("Memory add workspace scope resolution failed")
        return web.json_response({"error": "Memory storage failed"}, status=500)
    final_metadata.update(_route_write_scope(None, active_project))

    # Defense-in-depth guard, matching the pattern used in the other
    # three memory handlers. add_structured catches its own internal
    # exceptions and returns None (which the None-check below maps to
    # 500), but if the call raises BEFORE reaching that internal try
    # (init failure, bad argument shape, network error during connection
    # setup), the exception would otherwise escape to aiohttp as an HTML
    # 500. The wider guard collapses both failure modes into the same
    # clean 500 JSON body.
    try:
        memory_id = memory.add_structured(
            content,
            user_id=user_id,
            memory_type=memory_type,
            tags=tags,
            metadata=final_metadata,
            runtime_profile_id=str(principal.runtime_profile_id),
        )
    except Exception:
        log.exception("memory.add_structured failed for canonical principal")
        return web.json_response({"error": "Memory storage failed"}, status=500)

    if memory_id is None:
        # Reaching here means is_enabled() was True at precheck (so we
        # know memory was up) and add_structured did not raise, but it
        # returned None anyway. The only remaining None path inside
        # add_structured (after the empty-content path which we filtered
        # above with the 400 check) is the caught Exception around
        # _memory.add(). Same 500 response body as the wider guard above
        # because callers cannot meaningfully distinguish "primitive
        # raised" from "primitive caught and returned None" - both are
        # transient store failures.
        return web.json_response({"error": "Memory storage failed"}, status=500)

    log.info("Stored memory %s through canonical internal API authority", memory_id)
    return web.json_response({"id": memory_id})


@_require_internal_api(InternalAPIScope.MEMORY_READ)
async def _handle_memory_search(request: web.Request, principal: InternalAPIPrincipal) -> web.Response:
    """
    Search memories via memory.search().

    POST chosen over GET because user-message-shaped queries can be long
    enough to bump against URL-length limits on some HTTP clients, and
    POST also matches the rest of the /api/* surface (POST except
    /api/jobs* GET).

    Request body (JSON):
        query: str, required, non-empty after strip
        limit: int, optional (defaults to config.memory_search_limit)
    Responses:
        200 {"results": [<MemoryResult-dict>, ...]} on success (empty list
            is a valid 200 result for "no matches")
        400 on bad input
        401 on bad secret
        503 when memory is disabled
    """
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    # Reject non-object JSON shapes (null, lists, scalars, strings) before
    # any payload.get() call. JSON bodies like `null`, `[]`, `42`, or
    # `"string"` parse successfully (no JSONDecodeError) but raise
    # AttributeError on .get(), which would escape as an unstyled HTML
    # 500 traceback instead of a clean JSON 400.
    if not isinstance(payload, dict):
        return web.json_response({"error": "Request body must be a JSON object"}, status=400)

    # Same isinstance-before-.strip() rationale as _handle_memory_add:
    # a non-string `query` from JSON would raise AttributeError on .strip()
    # and escape to aiohttp.
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        return web.json_response({"error": "Missing required field: query"}, status=400)
    query = query.strip()

    # Optional `limit`: must be a positive int when provided. The
    # `isinstance(limit, bool)` short-circuit is load-bearing because
    # bool is a subclass of int in Python (`isinstance(True, int)` is
    # True), so without the bool check `{"limit": true}` would be
    # accepted as `limit=1` and silently truncate results to one row.
    limit = payload.get("limit")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        return web.json_response({"error": "limit must be a positive integer"}, status=400)

    try:
        _reject_internal_identity_selectors(payload)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)

    # search() degrades to [] when memory is disabled, but returning []
    # at the API layer would be indistinguishable from "no matches".
    # Inner Claude needs to pick "log no relevant memories and continue"
    # vs "memory is off, surface to operator" - the only distinguishing
    # signal is the status code, so the precheck is required.
    if not memory.is_enabled():
        return _memory_disabled_response()

    user_id = str(principal.principal_id)

    # Defense-in-depth around the full risky path: primitive call AND
    # serialization. search() catches its own exceptions and returns []
    # today, and asdict() should always succeed on the frozen dataclass
    # MemoryResult, but extending the guard to cover the asdict+
    # json_response step keeps the 500-as-clean-JSON contract intact
    # even if a future refactor changes search()'s return type or if
    # MemoryResult.metadata ever contains non-JSON-native values.
    # Without the wider guard, asdict raising TypeError or json_response
    # raising ValueError would escape to aiohttp as an unstyled HTML 500.
    try:
        results = memory.search(
            query,
            user_id=user_id,
            limit=limit,
            runtime_profile_id=str(principal.runtime_profile_id),
        )
        # asdict() flattens each frozen dataclass to a plain dict. Every
        # value inside MemoryResult.metadata is JSON-native because Mem0
        # stores metadata in Qdrant as JSON, so json_response can
        # serialize the whole structure without a custom encoder.
        return web.json_response({"results": [asdict(r) for r in results]})
    except Exception:
        log.exception("memory.search failed for canonical principal")
        return web.json_response({"error": "Memory search failed"}, status=500)


@_require_internal_api(InternalAPIScope.MEMORY_READ)
async def _handle_memory_stats(request: web.Request, principal: InternalAPIPrincipal) -> web.Response:
    """
    Return memory statistics via memory.get_stats().

    GET endpoint derives its scope from the authenticated execution context.

    Response is the MemoryStats object at the top level, NOT wrapped in
    {"results": ...} - single-object reads return the object bare per
    the API's response-shape rule (matches /api/jobs/{id} returning
    `job` at top level).

    Note on null fields: confidence_min, confidence_median, confidence_max
    are JSON null when extracted_count == 0. This is correct for users
    with no extracted facts and is NOT a store-failure signal. Callers
    should treat null in those fields as "no data to summarize."

    Responses:
        200 <MemoryStats-dict> on success
        400 on a retired caller identity selector
        401 on bad secret
        503 when memory is disabled
    """
    # Query parameters cannot select identity; the credential owns routing.
    try:
        _reject_internal_identity_selectors(dict(request.query))
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)

    # get_stats() returns zeroed MemoryStats when memory is disabled,
    # but - same as search - "memory off" and "user has no facts yet"
    # would be indistinguishable at the API. The 503 precheck preserves
    # the distinction at the status-code layer.
    if not memory.is_enabled():
        return _memory_disabled_response()

    user_id = str(principal.principal_id)

    # Wider guard around primitive call AND serialization, matching the
    # search handler. get_stats() doesn't have its own try/except today
    # (aggregation errors on a malformed row could surface here), and
    # extending the guard to the asdict+json_response step closes the
    # remaining route by which an exception could escape to aiohttp as
    # an HTML 500.
    try:
        stats = memory.get_stats(
            user_id=user_id,
            runtime_profile_id=str(principal.runtime_profile_id),
        )
        # asdict() preserves None for the optional confidence_* fields,
        # which become JSON null on the wire. The CLAUDE.md
        # "Memory System" section documents this so inner Claude does
        # not misread null as a store failure.
        return web.json_response(asdict(stats))
    except Exception:
        log.exception("memory.get_stats failed for canonical principal")
        return web.json_response({"error": "Memory stats failed"}, status=500)


@_require_internal_api(InternalAPIScope.MEMORY_DELETE_ALL)
async def _handle_memory_delete_all(request: web.Request, principal: InternalAPIPrincipal) -> web.Response:
    """
    Delete all memories for a user via memory.delete_all().

    Requires an exact-match confirm token in the body. Static
    well-known token (not per-request UUID) because the threat model
    is typo + prompt-injection defense, not replay protection.

    Request body (JSON):
        confirm: str, required, must equal "delete-all-memories"

    Responses:
        200 {"status": "deleted"} on success
        400 on missing/wrong confirm or a retired caller identity selector
        401 on bad secret
        503 when memory is disabled

    Caveat: delete_all() swallows internal errors (memory.py:1066-1069).
    Partial failures inside Mem0 are logged but invisible at the API
    layer. Surfacing them would require widening memory.py's error
    contract, which is out of scope for the foundation issue.
    """
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    # Reject non-object JSON shapes (null, lists, scalars, strings) before
    # any payload.get() call. JSON bodies like `null`, `[]`, `42`, or
    # `"string"` parse successfully (no JSONDecodeError) but raise
    # AttributeError on .get(), which would escape as an unstyled HTML
    # 500 traceback instead of a clean JSON 400.
    if not isinstance(payload, dict):
        return web.json_response({"error": "Request body must be a JSON object"}, status=400)

    # Confirm-token check before canonical authority validation: a request with the
    # wrong token is structurally bad input regardless of which user it
    # was aimed at, and rejecting it first means we don't waste a
    # selector check on requests we will reject anyway.
    if payload.get("confirm") != _DELETE_ALL_CONFIRM_TOKEN:
        return web.json_response(
            {"error": f'Missing or incorrect confirm field; expected "{_DELETE_ALL_CONFIRM_TOKEN}"'},
            status=400,
        )

    try:
        _reject_internal_identity_selectors(payload)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)

    # Symmetric precheck. delete_all() is a no-op when disabled, but
    # callers asking the API to wipe their memories deserve to know the
    # operation didn't actually run because the system was off.
    if not memory.is_enabled():
        return _memory_disabled_response()

    user_id = str(principal.principal_id)
    # Defense-in-depth guard, matching the pattern used in
    # _handle_memory_search and _handle_memory_stats. delete_all()
    # catches its own internal errors today (memory.py:1066-1069), so
    # this try/except mostly never fires - but if a future refactor
    # ever lets an exception escape (or if the call raises before
    # reaching the inner try, e.g. on a TypeError from a bad argument
    # shape), the handler still returns a clean 500 JSON body instead
    # of an aiohttp HTML 500 page.
    try:
        memory.delete_all(
            user_id=user_id,
            runtime_profile_id=str(principal.runtime_profile_id),
        )
    except Exception:
        log.exception("memory.delete_all failed for canonical principal")
        return web.json_response({"error": "Memory delete failed"}, status=500)

    log.info("Deleted memories through canonical internal API authority")
    return web.json_response({"status": "deleted"})


# ── Lifecycle ────────────────────────────────────────────────────────


def _register_routes(
    app: web.Application,
    config: Config,
) -> None:
    """Register transport-independent integration and internal API routes."""
    app.router.add_get("/health", _handle_health)

    if config.github_webhook_secret:
        app[GITHUB_WEBHOOK_SECRET_KEY] = config.github_webhook_secret
        app.router.add_post("/webhook/github", _handle_github)
    else:
        log.warning("GITHUB_WEBHOOK_SECRET not set - GitHub webhook endpoint disabled")

    if config.generic_webhook_secret:
        app[GENERIC_WEBHOOK_SECRET_KEY] = config.generic_webhook_secret
        app.router.add_post("/webhook", _handle_generic)
    else:
        log.info("GENERIC_WEBHOOK_SECRET not set - generic webhook endpoint disabled")

    # These routes authenticate through INTERNAL_API_AUTH_KEY. Their
    # availability must not be coupled to any public ingress configuration.
    app.router.add_get("/api/jobs", _handle_get_jobs)
    app.router.add_get("/api/jobs/{id}", _handle_get_job)
    app.router.add_post("/api/services/{name}", _handle_service_call)
    app.router.add_post("/api/memory/add", _handle_memory_add)
    app.router.add_post("/api/memory/search", _handle_memory_search)
    app.router.add_get("/api/memory/stats", _handle_memory_stats)
    app.router.add_delete("/api/memory/all", _handle_memory_delete_all)
    app.router.add_post("/api/schedule", _handle_schedule)
    app.router.add_delete("/api/jobs/{id}", _handle_delete_job)
    app.router.add_patch("/api/jobs/{id}", _handle_update_job)
    app.router.add_post("/api/send-message", _handle_send_message)
    app.router.add_post("/api/send-file", _handle_send_file)
    app.router.add_post("/api/agent-delegations", _handle_agent_delegation)


async def _register_workshop_client_api(
    app: web.Application,
    store: WorkshopEventStore,
    *,
    command_submitter: WorkshopClientCommandSubmitter | None = None,
    run_previews: WorkshopRunPreviewRegistry | None = None,
    artifact_service: WorkshopArtifactService | None = None,
    settings_workspaces: WorkshopSettingsWorkspaceService | None = None,
    routing_eligibility: WorkshopRoutingEligibilityService | None = None,
    routing_policy: WorkshopRoutingPolicyService | None = None,
    memory_queries: WorkshopMemoryQueryService | None = None,
    preference_documents: WorkshopPreferenceService | None = None,
    github_settings: WorkshopGitHubSettingsService | None = None,
    notification_preferences: WorkshopNotificationPreferenceService | None = None,
    channel_notification_policy: WorkshopChannelNotificationPolicyService | None = None,
    client_preferences: WorkshopClientPreferenceService | None = None,
    appearance_preferences: WorkshopAppearancePreferenceService | None = None,
    agent_enablement: WorkshopAgentEnablementService | None = None,
) -> Callable[[web.Application], None]:
    """Register the client API against the core-owned canonical store.

    The returned registrar deliberately contains only the Workshop browser
    shell and authenticated client API.  It can therefore be applied to the
    loopback application and, when explicitly configured, to a second
    LAN-bound application without exposing webhook or internal-agent routes.
    Shared locks and rate limiters keep both listeners inside one security
    boundary rather than giving each listener an independent allowance.
    """
    request_lock = asyncio.Lock()
    sessions_manager = WorkshopClientSessionManager(store)
    enrollment_manager = WorkshopClientEnrollmentManager(store)
    authenticator = WorkshopBearerSessionAuthenticator(sessions_manager)
    enrollment_rate_limiter = WorkshopEnrollmentRateLimiter()
    event_stream_limiter = WorkshopEventStreamLimiter()

    def register(target: web.Application) -> None:
        register_workshop_enrollment_routes(
            target,
            enrollment_manager=enrollment_manager,
            rate_limiter=enrollment_rate_limiter,
            request_lock=request_lock,
        )
        register_workshop_read_routes(
            target,
            store=store,
            authenticator=authenticator,
            request_lock=request_lock,
            event_stream_limiter=event_stream_limiter,
            run_previews=run_previews,
            artifact_service=artifact_service,
            settings_workspaces=settings_workspaces,
            routing_eligibility=routing_eligibility,
            routing_policy=routing_policy,
            memory_queries=memory_queries,
            preference_documents=preference_documents,
            github_settings=github_settings,
            notification_preferences=notification_preferences,
            channel_notification_policy=channel_notification_policy,
            client_preferences=client_preferences,
            appearance_preferences=appearance_preferences,
            agent_enablement=agent_enablement,
        )
        if command_submitter is not None:
            register_workshop_command_routes(
                target,
                store=store,
                authenticator=authenticator,
                submitter=command_submitter,
                request_lock=request_lock,
                artifact_service=artifact_service,
                routing_policy=routing_policy,
            )
        register_workshop_shell_routes(target)

    register(app)
    return register


async def start(
    config: Config,
    *,
    core_host: KaiApplicationHost,
    core_services: KaiCoreServices,
    integration_notifications: WorkshopIntegrationNotificationService,
    workshop_enabled: bool = True,
    route_registrars: Iterable[Callable[[web.Application], None]] = (),
) -> None:
    """
    Start the shared HTTP server and configured route registrars.

    The HTTP server always starts for health, authenticated integration ingress,
    and transport-independent internal APIs. Workshop client routes are
    independently enabled. Optional adapters may contribute routes without
    exposing their SDK objects to this host.

    Args:
        config: The application Config instance.
        core_host: Core lifecycle owner used for health/readiness reporting.
        core_services: Typed core dependencies required by HTTP routes.
        integration_notifications: Core-owned canonical notification service.
        workshop_enabled: Whether to publish Workshop client routes.
        route_registrars: Explicitly configured adapter-owned HTTP routes.
    """
    global _app, _runner, _workshop_lan_runner

    _app = web.Application(client_max_size=MAX_ARTIFACT_BYTES + 128 * 1024)
    _app[CORE_HOST_KEY] = core_host

    pool = core_services.subprocess_pool
    internal_api_auth = getattr(pool, "internal_api_auth", None)
    if not isinstance(internal_api_auth, InternalAPIAuth):
        raise RuntimeError("Subprocess pool did not provide an internal API credential store")
    _app[INTERNAL_API_AUTH_KEY] = internal_api_auth

    # Retain loaded configuration for transport-neutral service policy.
    _app[CONFIG_KEY] = config

    # Workspace policy lets canonical review work resolve a local checkout for
    # _resolve_local_repo() match incoming PR webhook repos against
    # local checkouts without a hardcoded GITHUB_REPO setting.
    _app[WORKSPACE_BASE_KEY] = str(config.workspace_base) if config.workspace_base else None
    _app[ALLOWED_WORKSPACES_KEY] = [str(p) for p in config.allowed_workspaces]

    _register_routes(_app, config)
    for register_routes in route_registrars:
        register_routes(_app)
    _app[WORKSHOP_PRINCIPAL_STORAGE_KEY] = core_services.principal_storage
    _app[WORKSHOP_INTEGRATION_NOTIFICATIONS_KEY] = integration_notifications
    _app[WORKSHOP_GITHUB_AUTOMATION_KEY] = core_services.github_automation
    register_workshop_routes: Callable[[web.Application], None] | None = None
    if workshop_enabled:
        register_workshop_routes = await _register_workshop_client_api(
            _app,
            core_services.client_store,
            command_submitter=core_services.client_commands,
            run_previews=core_services.run_previews,
            artifact_service=core_services.artifacts,
            settings_workspaces=core_services.settings_workspaces,
            routing_eligibility=getattr(core_services, "routing_eligibility", None),
            routing_policy=getattr(core_services, "routing_policy", None),
            memory_queries=core_services.memory_queries,
            preference_documents=core_services.preference_documents,
            github_settings=getattr(core_services, "github_settings", None),
            notification_preferences=getattr(core_services, "notification_preferences", None),
            channel_notification_policy=getattr(core_services, "channel_notification_policy", None),
            client_preferences=getattr(core_services, "client_preferences", None),
            appearance_preferences=getattr(core_services, "appearance_preferences", None),
            agent_enablement=getattr(core_services, "agent_enablement", None),
        )

    _runner = web.AppRunner(
        _app,
        access_log=None,
        shutdown_timeout=_HTTP_RUNNER_SHUTDOWN_TIMEOUT,
    )
    await _runner.setup()
    # The mixed webhook/internal API application remains loopback-only.  A
    # separately configured LAN listener below receives only Workshop client
    # routes, so opting into browser access cannot expose /api or /webhook.
    site = web.TCPSite(_runner, "127.0.0.1", config.webhook_port)
    await site.start()
    log.info("Webhook server listening on 127.0.0.1:%d", config.webhook_port)

    if workshop_enabled and config.workshop_lan_host:
        assert register_workshop_routes is not None
        workshop_lan_app = web.Application(client_max_size=MAX_ARTIFACT_BYTES + 128 * 1024)
        register_workshop_routes(workshop_lan_app)
        _workshop_lan_runner = web.AppRunner(
            workshop_lan_app,
            access_log=None,
            shutdown_timeout=_HTTP_RUNNER_SHUTDOWN_TIMEOUT,
        )
        await _workshop_lan_runner.setup()
        workshop_lan_site = web.TCPSite(
            _workshop_lan_runner,
            config.workshop_lan_host,
            config.webhook_port,
        )
        await workshop_lan_site.start()
        log.warning(
            "Workshop client listening on trusted-LAN HTTP at http://%s:%d/workshop/",
            config.workshop_lan_host,
            config.webhook_port,
        )


async def stop() -> None:
    """Stop the shared HTTP listeners after adapter ingress has drained."""
    global _app, _runner, _workshop_lan_runner
    try:
        if _workshop_lan_runner:
            await _workshop_lan_runner.cleanup()
            log.info("Workshop LAN client server stopped")
        if _runner:
            await _runner.cleanup()
            log.info("Webhook server stopped")
    finally:
        _workshop_lan_runner = None
        _runner = None
        _app = None


def is_running() -> bool:
    """True if the webhook server is currently running."""
    return _runner is not None


def is_workshop_lan_running() -> bool:
    """True if the dedicated Workshop LAN listener is currently running."""
    return _workshop_lan_runner is not None
