"""
Webhook HTTP server for receiving external notifications, scheduling jobs, and
optionally serving as the Telegram update transport.

Provides functionality to:
1. Receive Telegram updates via webhook (when TELEGRAM_WEBHOOK_URL is configured)
2. Receive and validate GitHub webhook events (push, PR, issues, comments, reviews)
3. Accept generic webhook notifications from any source
4. Expose a scheduling API for creating cron-style jobs via HTTP
5. Expose a jobs query API for listing and fetching scheduled jobs
6. Proxy authenticated requests to external services (service layer)
7. Send text messages and files to the Telegram chat (messaging APIs)
8. Monitor webhook health and auto-recover from Telegram delivery failures
9. Redeem Workshop client enrollment grants and synchronize authorized timelines

The server always runs on aiohttp alongside the Telegram bot in the same event
loop, regardless of transport mode. In polling mode, Telegram updates arrive via
the Updater owned by ``TelegramAdapter``; this server still handles everything
else. ``HttpAdapter`` owns this module's listener lifecycle.

Routes are organized into these groups:
    - /webhook/telegram     - Telegram updates (webhook mode only, secret_token auth)
    - /webhook/github       - GitHub events with HMAC-SHA256 signature validation
    - /webhook              - Generic webhooks with shared-secret auth
    - /api/schedule         - Job creation API (used by persistent agents via curl)
    - /api/jobs             - Job listing and detail API
    - /api/jobs/{id}        - Job detail (GET), deletion (DELETE), and update (PATCH)
    - /api/services/{name}  - External service proxy (injects auth from .env)
    - /api/send-message     - Send a text message to the Telegram chat
    - /api/send-file        - Send a file from the filesystem to the Telegram chat
    - /api/memory/add       - Store a structured memory (POST)
    - /api/memory/search    - Search memories by query (POST)
    - /api/memory/stats     - Memory statistics for a user (GET)
    - /api/memory/all       - Delete all memories for a user (DELETE, requires confirm token)
    - /v1/client/enrollment/redeem - Exchange an operator-issued Workshop grant
    - /v1/channels/{id}/timeline   - Read one authorized canonical timeline
    - /v1/channels/{id}/events     - Resume authorized canonical message events

Every ingress domain has an independent credential. Telegram uses its configured
or process-generated secret token, GitHub uses GITHUB_WEBHOOK_SECRET, and the
generic endpoint uses GENERIC_WEBHOOK_SECRET. Internal API routes always use
random, principal-bound process credentials and do not depend on external webhook
secrets.

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
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from aiohttp import web
from telegram import Bot, Update
from telegram.ext import Application

from kai import memory, services, sessions
from kai.application_host import KaiApplicationHost, KaiCoreServices
from kai.config import (
    DATA_DIR,
    IMAGE_EXTENSIONS,
    Config,
)
from kai.internal_api_auth import InternalAPIAuth, InternalAPIPrincipal, InternalAPIScope
from kai.job_types import CANONICAL_JOB_TYPES, normalize_job_type
from kai.telegram_utils import chunk_text
from kai.workshop.artifacts import MAX_ARTIFACT_BYTES, WorkshopArtifactService
from kai.workshop.client_api import (
    WorkshopClientCommandSubmitter,
    WorkshopEnrollmentRateLimiter,
    WorkshopEventStreamLimiter,
    register_workshop_command_routes,
    register_workshop_enrollment_routes,
    register_workshop_read_routes,
)
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
from kai.workshop.integration_notifications import (
    DEFAULT_INTEGRATION_ROUTE,
    IntegrationNotification,
    WorkshopIntegrationNotificationService,
)
from kai.workshop.memory_queries import WorkshopMemoryQueryService
from kai.workshop.run_previews import WorkshopRunPreviewRegistry
from kai.workshop.scheduler import WorkshopCanonicalScheduler
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

ALLOWED_USER_IDS_KEY: web.AppKey[set[int]] = web.AppKey("allowed_user_ids", set)
ALLOWED_WORKSPACES_KEY: web.AppKey[list[str]] = web.AppKey("allowed_workspaces", list)
CHAT_ID_KEY: web.AppKey[int] = web.AppKey("chat_id", int)
CONFIG_KEY: web.AppKey[Config] = web.AppKey("config", Config)
CORE_HOST_KEY: web.AppKey[KaiApplicationHost] = web.AppKey("core_host", KaiApplicationHost)
GENERIC_WEBHOOK_SECRET_KEY: web.AppKey[str] = web.AppKey("generic_webhook_secret", str)
GITHUB_WEBHOOK_SECRET_KEY: web.AppKey[str] = web.AppKey("github_webhook_secret", str)
INTERNAL_API_AUTH_KEY: web.AppKey[InternalAPIAuth] = web.AppKey("internal_api_auth", InternalAPIAuth)
NOTIFICATION_CHAT_IDS_KEY: web.AppKey[set[int]] = web.AppKey("notification_chat_ids", set)
POOL_KEY: web.AppKey[object] = web.AppKey("pool", object)
TELEGRAM_APP_KEY: web.AppKey[Application] = web.AppKey("telegram_app", Application)
TELEGRAM_BOT_KEY: web.AppKey[Bot] = web.AppKey("telegram_bot", Bot)
TELEGRAM_WEBHOOK_SECRET_KEY: web.AppKey[str] = web.AppKey("telegram_webhook_secret", str)
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
# Tracks whether we registered a Telegram webhook with the API, so stop()
# knows whether to call delete_webhook(). Only True in webhook mode.
_webhook_registered: bool = False

# Background tasks for processing Telegram updates. Tasks are kept in this set
# to prevent garbage collection (Python only weakly references fire-and-forget
# tasks, so an unreferenced task can be silently collected mid-execution).
# Each task removes itself from the set via a done callback.
_background_tasks: set[asyncio.Task] = set()
_BACKGROUND_TASK_DRAIN_TIMEOUT = 30.0
_HTTP_RUNNER_SHUTDOWN_TIMEOUT = 5.0
_telegram_queue_worker_task: asyncio.Task | None = None
_telegram_queue_worker_active_row_id: int | None = None
_TELEGRAM_UPDATE_MAX_ATTEMPTS = 5

# Webhook health monitor task, started in start() and cancelled in stop().
# Periodically checks Telegram's getWebhookInfo for delivery errors and
# re-registers the webhook if needed to reset exponential backoff.
_health_monitor_task: asyncio.Task | None = None

# How often to check webhook health (seconds). Frequent enough to catch
# problems quickly, infrequent enough to avoid API rate limits.
_HEALTH_CHECK_INTERVAL = 300  # 5 minutes


async def _drain_background_tasks(timeout: float = _BACKGROUND_TASK_DRAIN_TIMEOUT) -> None:
    """
    Wait briefly for fire-and-forget webhook work during shutdown.

    Telegram update processing is launched as background work so inbound
    updates can be acknowledged promptly. During a
    controlled shutdown, let those tasks finish before tearing down the server.
    If they overrun the bounded timeout, cancel them rather than hanging
    shutdown indefinitely.
    """
    pending = {task for task in _background_tasks if not task.done()}
    if not pending:
        return

    log.info("Waiting for %d webhook background task(s) to finish", len(pending))
    done, still_pending = await asyncio.wait(pending, timeout=timeout)
    for task in done:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("Webhook background task failed during shutdown")
        finally:
            _background_tasks.discard(task)

    if not still_pending:
        return

    log.warning(
        "Cancelling %d webhook background task(s) after %.1fs shutdown timeout",
        len(still_pending),
        timeout,
    )
    for task in still_pending:
        task.cancel()
    await asyncio.gather(*still_pending, return_exceptions=True)
    for task in still_pending:
        _background_tasks.discard(task)


async def _process_queued_telegram_update(
    row: sessions.TelegramUpdateQueueRow,
    telegram_app: Application,
    bot: Bot,
) -> None:
    row_id = row["id"]
    try:
        data = json.loads(row["payload"])
        update = Update.de_json(data, bot)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log.exception("Discarding malformed queued Telegram update %s", row_id)
        await sessions.discard_telegram_update(row_id, error)
        return

    if update is None:
        await sessions.complete_telegram_update(row_id)
        return

    try:
        await telegram_app.process_update(update)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if row["attempt_count"] >= _TELEGRAM_UPDATE_MAX_ATTEMPTS:
            log.exception(
                "Discarding Telegram update queue row %s after %d failed attempt(s)",
                row_id,
                row["attempt_count"],
            )
            await sessions.discard_telegram_update(row_id, error)
            return
        log.exception("Telegram update queue row %s failed; retrying later", row_id)
        await sessions.retry_telegram_update(row_id, error)
        return

    await sessions.complete_telegram_update(row_id)


async def _telegram_update_queue_worker(telegram_app: Application, bot: Bot) -> None:
    global _telegram_queue_worker_active_row_id
    try:
        while True:
            row = await sessions.claim_next_telegram_update()
            if row is None:
                return
            _telegram_queue_worker_active_row_id = row["id"]
            try:
                await _process_queued_telegram_update(row, telegram_app, bot)
            finally:
                _telegram_queue_worker_active_row_id = None
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("Telegram update queue worker crashed")


def _telegram_worker_done(task: asyncio.Task) -> None:
    global _telegram_queue_worker_task
    if _telegram_queue_worker_task is task:
        _telegram_queue_worker_task = None
    _background_tasks.discard(task)


def _ensure_telegram_update_queue_worker(telegram_app: Application, bot: Bot) -> None:
    global _telegram_queue_worker_task
    if _telegram_queue_worker_task is not None and not _telegram_queue_worker_task.done():
        return
    task = asyncio.create_task(_telegram_update_queue_worker(telegram_app, bot))
    _telegram_queue_worker_task = task
    _background_tasks.add(task)
    task.add_done_callback(_telegram_worker_done)


def _is_telegram_stop_update(data: object) -> bool:
    """Return whether a raw Telegram update contains a /stop command message."""
    if not isinstance(data, dict):
        return False
    message = data.get("message")
    if not isinstance(message, dict):
        return False
    text = message.get("text")
    if not isinstance(text, str):
        return False
    return re.match(r"^/stop(?:@[A-Za-z0-9_]+)?(?:\s|$)", text) is not None


async def _process_priority_telegram_update(
    row_id: int,
    telegram_app: Application,
    bot: Bot,
) -> None:
    """Claim and process one persisted control update outside the FIFO worker."""
    try:
        row = await sessions.claim_telegram_update(row_id)
        if row is not None:
            await _process_queued_telegram_update(row, telegram_app, bot)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("Priority Telegram update queue task crashed for row %s", row_id)


def _dispatch_priority_telegram_stop(
    row_id: int,
    data: object,
    telegram_app: Application,
    bot: Bot,
) -> None:
    """Dispatch a persisted /stop concurrently when the FIFO worker is busy."""
    if _telegram_queue_worker_active_row_id is None or not _is_telegram_stop_update(data):
        return
    task = asyncio.create_task(_process_priority_telegram_update(row_id, telegram_app, bot))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


# If Telegram reports an error within this window, re-register the webhook.
# Slightly longer than the check interval so a single transient error
# in the previous cycle triggers a re-registration on the next check.
_ERROR_RECENCY_THRESHOLD = 600  # 10 minutes


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


def _strip_markdown(text: str) -> str:
    """
    Remove markdown syntax so text reads cleanly as plain Telegram text.

    Used as a fallback when Telegram's Markdown parser rejects a message
    (e.g., unbalanced backticks or brackets). Converts links to "text (url)"
    format and strips bold, italic, and code markers.

    Args:
        text: Markdown-formatted string.

    Returns:
        The same text with markdown syntax removed.
    """
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)  # [text](url) → text (url)
    text = text.replace("**", "").replace("__", "")  # bold
    text = text.replace("`", "")  # inline code
    text = re.sub(r"(?<!\w)_(\S.*?\S)_(?!\w)", r"\1", text)  # _italic_ but not snake_case
    return text


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
    }
)


def _resolve_internal_runtime_config_id(
    principal: InternalAPIPrincipal,
    payload: dict,
) -> int:
    """
    Resolve the private compatibility key from authenticated authority.

    Canonical identity is bound to the process-lifetime credential. Request
    bodies and query strings may not repeat or select the retired ``chat_id``
    identity field. The returned integer is confined to this server adapter
    while legacy persistence primitives are being replaced.

    Raises:
        ValueError: If the retired caller-supplied identity field is present.
    """
    supplied = sorted(_INTERNAL_API_IDENTITY_SELECTORS.intersection(payload))
    if supplied:
        raise ValueError(
            f"Identity selector {supplied[0]} is not accepted; the internal API credential already binds "
            "the canonical execution context"
        )
    return principal.compatibility_runtime_config_id()


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


async def _handle_telegram_update(request: web.Request) -> web.Response:
    """
    Receive a Telegram update pushed via webhook.

    Validates the X-Telegram-Bot-Api-Secret-Token header against the configured
    secret, persists the raw update to the durable inbound queue, and starts a
    background worker that dispatches queued updates to process_update().  A
    persisted /stop command may use a separately tracked task while that FIFO
    worker is busy so it can interrupt the active request; other updates remain
    serialized.

    IMPORTANT: queued updates are processed by a background task, not awaited.
    Claude responses can take 30+ seconds, and Telegram's webhook client times
    out after ~30-35s. If we awaited process_update(), Telegram would assume
    delivery failed and retry the same message, causing duplicate responses.
    By returning 200 after durable enqueue and processing in the background, we
    acknowledge receipt before Telegram's timeout while allowing restart replay.
    The per-chat lock in bot.py serializes concurrent messages, so ordering is
    preserved per chat.

    Returns 200 after the update is durably queued. Payloads without a usable
    update_id still return 200 to avoid permanent Telegram retries, but a local
    persistence failure returns 500 because no work was accepted.
    """
    secret = request.app[TELEGRAM_WEBHOOK_SECRET_KEY]
    provided = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not hmac.compare_digest(provided, secret):
        log.warning("Telegram update: invalid secret")
        return web.Response(status=401, text="Invalid secret")

    try:
        data = await request.json()
    except json.JSONDecodeError:
        log.warning("Telegram update: malformed JSON")
        return web.Response(status=200)

    telegram_app = request.app[TELEGRAM_APP_KEY]
    bot = request.app[TELEGRAM_BOT_KEY]
    update_id = data.get("update_id") if isinstance(data, dict) else None
    if isinstance(update_id, bool) or not isinstance(update_id, int):
        log.warning("Telegram update: missing or invalid update_id")
        return web.Response(status=200)

    try:
        row_id, _inserted = await sessions.enqueue_telegram_update(update_id, json.dumps(data))
    except Exception:
        log.exception("Failed to persist Telegram update %s", update_id)
        return web.Response(status=500, text="Failed to enqueue update")

    try:
        _ensure_telegram_update_queue_worker(telegram_app, bot)
    except Exception:
        log.exception("Failed to start Telegram update queue worker")
    else:
        try:
            _dispatch_priority_telegram_stop(row_id, data, telegram_app, bot)
        except Exception:
            log.exception("Failed to dispatch priority Telegram /stop row %s", row_id)

    return web.Response(status=200)


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

    # Normalize the compatibility alias at ingress so all new rows use the
    # backend-neutral identifier. sessions.create_job repeats this validation
    # as the persistence boundary for non-HTTP callers.
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
        chat_id = _resolve_internal_runtime_config_id(principal, payload)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)

    # Validate schedule_data structure before persisting. Catches malformed
    # JSON strings and wrong-shape payloads (e.g., interval keys on a daily
    # job) that would otherwise fail the core scheduler on every fire attempt.
    schedule_data_str, error = _validate_schedule_data(schedule_data, schedule_type)
    if error:
        return web.json_response({"error": error}, status=400)
    assert schedule_data_str is not None  # guaranteed when error is None

    # Persist to database
    try:
        job_id = await sessions.create_job(
            chat_id=chat_id,
            name=name,
            job_type=job_type,
            prompt=prompt,
            schedule_type=schedule_type,
            schedule_data=schedule_data_str,
            auto_remove=auto_remove,
            notify_on_check=notify_on_check,
        )
    except Exception:
        log.exception("Failed to create job")
        return web.json_response({"error": "Failed to create job"}, status=500)

    scheduler = request.app[CORE_HOST_KEY].services.scheduler
    try:
        registered = await scheduler.register_job(job_id)
    except Exception:
        log.exception("Failed to register job %d with scheduler", job_id)
        await sessions.deactivate_job(job_id, chat_id=chat_id)
        return web.json_response({"error": "Failed to register job"}, status=500)
    if not registered:
        log.error("Failed to register job %d with scheduler", job_id)
        await sessions.deactivate_job(job_id, chat_id=chat_id)
        return web.json_response({"error": "Failed to register job"}, status=500)

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

    # Query parameters cannot select identity; the credential owns routing.
    try:
        chat_id = _resolve_internal_runtime_config_id(principal, dict(request.query))
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    jobs = await sessions.get_jobs(chat_id)
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

    # Resolve caller identity for ownership check
    try:
        chat_id = _resolve_internal_runtime_config_id(principal, dict(request.query))
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)

    job = await sessions.get_job_by_id(job_id)
    # Return 404 for missing OR wrong-owner jobs (don't leak existence).
    # Both sides are ints: _resolve_internal_runtime_config_id always returns int, and
    # chat_id is stored as INTEGER in the jobs table.
    if not job or job["chat_id"] != chat_id:
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

    # Resolve compatibility storage only from server-owned authority.
    try:
        chat_id = _resolve_internal_runtime_config_id(principal, dict(request.query))
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    deleted = await sessions.delete_job(job_id, chat_id=chat_id)
    if not deleted:
        return web.json_response({"error": "Job not found"}, status=404)

    await request.app[CORE_HOST_KEY].services.scheduler.remove_job(job_id)

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
            existing_job = await sessions.get_job_by_id(job_id)
            if existing_job is None:
                return web.json_response({"error": "Job not found or inactive"}, status=404)
            effective_type = existing_job["schedule_type"]
        schedule_data, error = _validate_schedule_data(schedule_data, effective_type)
        if error:
            return web.json_response({"error": error}, status=400)

    # The credential selects authority; request data cannot choose identity.
    try:
        chat_id = _resolve_internal_runtime_config_id(principal, payload)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)

    # If the schedule changes, keep the previous row so a failed scheduler
    # re-registration can be compensated. Without this, PATCH can leave the DB
    # saying a job is active on the new schedule while the core scheduler has
    # no corresponding task.
    schedule_changed = new_schedule_type is not None or schedule_data is not None
    if schedule_changed:
        if existing_job is None:
            existing_job = await sessions.get_job_by_id(job_id)
        if existing_job is None or existing_job["chat_id"] != chat_id:
            return web.json_response({"error": "Job not found or inactive"}, status=404)

    updated = await sessions.update_job(
        job_id,
        chat_id=chat_id,
        name=payload.get("name"),
        prompt=payload.get("prompt"),
        schedule_type=new_schedule_type,
        schedule_data=schedule_data,
        auto_remove=payload.get("auto_remove"),
        notify_on_check=payload.get("notify_on_check"),
    )

    if not updated:
        return web.json_response({"error": "Job not found or inactive"}, status=404)

    if schedule_changed:
        assert existing_job is not None
        scheduler = request.app[CORE_HOST_KEY].services.scheduler
        try:
            registered = await scheduler.register_job(job_id)
        except Exception:
            log.exception("Failed to re-register updated job %d with scheduler", job_id)
            await _restore_job_after_failed_reschedule(scheduler, job_id, chat_id, existing_job)
            return web.json_response({"error": "Failed to register job"}, status=500)
        if not registered:
            log.error("Failed to re-register updated job %d with scheduler", job_id)
            await _restore_job_after_failed_reschedule(scheduler, job_id, chat_id, existing_job)
            return web.json_response({"error": "Failed to register job"}, status=500)

    log.info("Updated job %d via API", job_id)
    return web.json_response({"updated": job_id})


async def _restore_job_after_failed_reschedule(
    scheduler: WorkshopCanonicalScheduler,
    job_id: int,
    chat_id: int,
    previous_job: dict,
) -> None:
    """Best-effort rollback after PATCH committed but scheduler registration failed."""
    restored = await sessions.update_job(
        job_id,
        chat_id=chat_id,
        name=previous_job["name"],
        prompt=previous_job["prompt"],
        schedule_type=previous_job["schedule_type"],
        schedule_data=previous_job["schedule_data"],
        auto_remove=previous_job["auto_remove"],
        notify_on_check=previous_job["notify_on_check"],
    )
    if not restored:
        log.error("Failed to restore job %d after scheduler registration failure", job_id)
        return
    try:
        restored_registered = await scheduler.register_job(job_id)
    except Exception:
        log.exception("Failed to restore scheduler entry for job %d after rollback", job_id)
        return
    if not restored_registered:
        log.error("Failed to restore scheduler entry for job %d after rollback", job_id)


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


@_require_internal_api(InternalAPIScope.MESSAGES_SEND)
async def _handle_send_message(request: web.Request, principal: InternalAPIPrincipal) -> web.Response:
    """
    Send a text message to the Telegram chat.

    Called by the inner Claude process to proactively notify the user - e.g.,
    when a background task completes, or a scheduled job wants to report
    results without going through the full Claude prompt cycle.

    Accepts a JSON body with a required "text" field. Messages longer than
    Telegram's 4096-character limit are split into chunks.

    Returns:
        JSON {"status": "sent"} on success, or an appropriate HTTP error.
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

    text = payload.get("text", "").strip()
    if not text:
        return web.json_response({"error": "Missing required field: text"}, status=400)

    bot = request.app[TELEGRAM_BOT_KEY]
    try:
        chat_id = _resolve_internal_runtime_config_id(principal, payload)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)

    try:
        for part in chunk_text(text):
            await bot.send_message(chat_id, part)
    except Exception:
        log.exception("Failed to send message to chat %d via API", chat_id)
        return web.json_response({"error": "Failed to send message"}, status=500)

    log.info("Sent message to chat %d via API (%d chars)", chat_id, len(text))
    return web.json_response({"status": "sent"})


# ── File exchange ────────────────────────────────────────────────────


@_require_internal_api(InternalAPIScope.FILES_SEND)
async def _handle_send_file(request: web.Request, principal: InternalAPIPrincipal) -> web.Response:
    """
    Send a file from the filesystem to the Telegram chat.

    Called by the inner Claude process to deliver files back to the user.
    Accepts a JSON body with a required "path" field (absolute path) and
    an optional "caption". Images are sent as photos (rendered inline),
    everything else as document attachments.

    Path confinement: the resolved path must be inside the authenticated
    principal's current workspace or its scoped upload directory. This
    prevents path traversal attacks via symlinks or "../" and prevents one
    principal from sending files from another principal's upload directory.

    Returns:
        JSON {"status": "sent", "file": "<filename>"} on success, or an
        appropriate HTTP error (400/401/403/404).
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

    # Resolve chat_id first - needed for per-user workspace confinement
    try:
        chat_id = _resolve_internal_runtime_config_id(principal, payload)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)

    path = Path(file_path).resolve()

    # Confine to the requesting user's workspace to prevent path traversal.
    # Uses Path.relative_to() which raises ValueError on escape. Each user
    # has their own per-user home workspace resolved from the pool (#353).
    # When the pool is unavailable (transient startup state) we refuse the
    # request rather than opening a global fallback path.
    pool = request.app.get(POOL_KEY)
    if pool is None:
        return web.json_response({"error": "No workspace configured"}, status=403)
    workspace = str(await pool.get_effective_workspace(chat_id))
    # Allow files from either the workspace or this authenticated principal's
    # canonical upload directory. The prior configured-user directory remains
    # readable during migration, but the legacy shared root and sibling
    # principals' directories are excluded. Both scopes are resolved from the
    # authenticated credential, never from caller-selected request fields.
    workspace_resolved = Path(workspace).resolve()
    storage_registry = request.app.get(WORKSHOP_PRINCIPAL_STORAGE_KEY)
    if storage_registry is None:
        return web.json_response({"error": "Principal storage unavailable"}, status=403)
    try:
        storage_namespace = storage_registry.for_runtime_config_id(principal.compatibility_runtime_config_id())
    except WorkshopStorageNamespaceError:
        return web.json_response({"error": "Principal storage unavailable"}, status=403)
    allowed_roots = (
        workspace_resolved,
        storage_namespace.files_directory(DATA_DIR).resolve(),
        storage_namespace.legacy_files_directory(DATA_DIR).resolve(),
    )
    if not any(path.is_relative_to(root) for root in allowed_roots):
        return web.json_response({"error": "Path outside allowed directories"}, status=403)

    if not path.is_file():
        return web.json_response({"error": f"File not found: {file_path}"}, status=404)

    bot = request.app[TELEGRAM_BOT_KEY]
    caption = payload.get("caption", "")

    # Send images as photos (Telegram renders them inline) and everything
    # else as document attachments (preserves filename, allows any type).
    try:
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            with open(path, "rb") as f:
                await bot.send_photo(chat_id, f, caption=caption or None)
        else:
            with open(path, "rb") as f:
                await bot.send_document(chat_id, f, caption=caption or None, filename=path.name)
    except Exception:
        log.exception("Failed to send file %s to chat %d", path, chat_id)
        return web.json_response({"error": "Failed to send file"}, status=500)

    log.info("Sent file %s to chat %d via API", path.name, chat_id)
    return web.json_response({"status": "sent", "file": path.name})


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
#   - Memory primitives take user_id as a string; _resolve_internal_runtime_config_id returns
#     int. Stringify at the boundary in every handler. Removing the cast
#     is a load-bearing bug: Mem0 keys are stored under the string form,
#     so a missed cast would silently isolate the caller's memories from
#     their existing facts.
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
        chat_id = _resolve_internal_runtime_config_id(principal, payload)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)

    # Symmetric is_enabled() precheck. Runs after auth + 400-level
    # validation but before the primitive call. See the §Memory API
    # block comment above for why all four memory handlers share this.
    if not memory.is_enabled():
        return _memory_disabled_response()

    # int -> str at the memory boundary; do NOT remove the cast.
    user_id = str(chat_id)

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
        pool = request.app.get(POOL_KEY)
        active_project = None
        if pool is not None:
            api_config: Config = request.app[CONFIG_KEY]
            workspace = await pool.get_effective_workspace(chat_id)
            active_project = detect_active_memory_project(workspace, merged_registry(api_config.memory_projects))
    except Exception:
        log.exception("memory add: workspace scope resolution failed for chat %d", chat_id)
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
        )
    except Exception:
        log.exception("memory.add_structured failed for chat %d", chat_id)
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

    log.info("Stored memory %s for chat %d via API", memory_id, chat_id)
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
        chat_id = _resolve_internal_runtime_config_id(principal, payload)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)

    # search() degrades to [] when memory is disabled, but returning []
    # at the API layer would be indistinguishable from "no matches".
    # Inner Claude needs to pick "log no relevant memories and continue"
    # vs "memory is off, surface to operator" - the only distinguishing
    # signal is the status code, so the precheck is required.
    if not memory.is_enabled():
        return _memory_disabled_response()

    user_id = str(chat_id)

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
        results = memory.search(query, user_id=user_id, limit=limit)
        # asdict() flattens each frozen dataclass to a plain dict. Every
        # value inside MemoryResult.metadata is JSON-native because Mem0
        # stores metadata in Qdrant as JSON, so json_response can
        # serialize the whole structure without a custom encoder.
        return web.json_response({"results": [asdict(r) for r in results]})
    except Exception:
        log.exception("memory.search failed for chat %d", chat_id)
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
        chat_id = _resolve_internal_runtime_config_id(principal, dict(request.query))
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)

    # get_stats() returns zeroed MemoryStats when memory is disabled,
    # but - same as search - "memory off" and "user has no facts yet"
    # would be indistinguishable at the API. The 503 precheck preserves
    # the distinction at the status-code layer.
    if not memory.is_enabled():
        return _memory_disabled_response()

    user_id = str(chat_id)

    # Wider guard around primitive call AND serialization, matching the
    # search handler. get_stats() doesn't have its own try/except today
    # (aggregation errors on a malformed row could surface here), and
    # extending the guard to the asdict+json_response step closes the
    # remaining route by which an exception could escape to aiohttp as
    # an HTML 500.
    try:
        stats = memory.get_stats(user_id=user_id)
        # asdict() preserves None for the optional confidence_* fields,
        # which become JSON null on the wire. The CLAUDE.md
        # "Memory System" section documents this so inner Claude does
        # not misread null as a store failure.
        return web.json_response(asdict(stats))
    except Exception:
        log.exception("memory.get_stats failed for chat %d", chat_id)
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

    # Confirm-token check before compatibility storage resolution: a request with the
    # wrong token is structurally bad input regardless of which user it
    # was aimed at, and rejecting it first means we don't waste a
    # _resolve_internal_runtime_config_id call on requests we will reject anyway.
    if payload.get("confirm") != _DELETE_ALL_CONFIRM_TOKEN:
        return web.json_response(
            {"error": f'Missing or incorrect confirm field; expected "{_DELETE_ALL_CONFIRM_TOKEN}"'},
            status=400,
        )

    try:
        chat_id = _resolve_internal_runtime_config_id(principal, payload)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)

    # Symmetric precheck. delete_all() is a no-op when disabled, but
    # callers asking the API to wipe their memories deserve to know the
    # operation didn't actually run because the system was off.
    if not memory.is_enabled():
        return _memory_disabled_response()

    user_id = str(chat_id)
    # Defense-in-depth guard, matching the pattern used in
    # _handle_memory_search and _handle_memory_stats. delete_all()
    # catches its own internal errors today (memory.py:1066-1069), so
    # this try/except mostly never fires - but if a future refactor
    # ever lets an exception escape (or if the call raises before
    # reaching the inner try, e.g. on a TypeError from a bad argument
    # shape), the handler still returns a clean 500 JSON body instead
    # of an aiohttp HTML 500 page.
    try:
        memory.delete_all(user_id=user_id)
    except Exception:
        log.exception("memory.delete_all failed for chat %d", chat_id)
        return web.json_response({"error": "Memory delete failed"}, status=500)

    log.info("Deleted all memories for chat %d via API", chat_id)
    return web.json_response({"status": "deleted"})


# ── Webhook health monitor ───────────────────────────────────────────


async def _webhook_health_loop(bot, webhook_url: str, webhook_secret: str, chat_id: int) -> None:
    """
    Periodically check Telegram webhook health and re-register if needed.

    Telegram silently drops updates after repeated delivery failures (502s
    from Cloudflare tunnel hiccups, etc.) via exponential backoff. Once
    backed off far enough, the bot appears completely dead - no errors in
    our logs, no pending_update_count on Telegram's side.

    This loop calls getWebhookInfo every _HEALTH_CHECK_INTERVAL seconds
    and re-registers the webhook when any of these conditions are met:

    1. The webhook URL was cleared (manual intervention, competing instance)
    2. Telegram reports a recent delivery error (last_error_date within
       _ERROR_RECENCY_THRESHOLD)
    3. pending_update_count has been >0 for two consecutive checks,
       meaning Telegram is queuing updates it cannot deliver

    Condition 3 requires two consecutive checks to avoid false positives
    from normal message bursts (a single check catching in-flight updates).

    If 3 consecutive health checks fail, the admin is notified once via
    Telegram. The notification resets after a successful check.

    Args:
        bot: The Telegram bot instance.
        webhook_url: The configured webhook URL.
        webhook_secret: The Telegram webhook secret token.
        chat_id: Admin chat ID for failure notifications.
    """
    await asyncio.sleep(_HEALTH_CHECK_INTERVAL)  # skip the first check (just registered)

    # Track pending updates across consecutive checks. A single non-zero
    # reading is normal (messages in flight); two in a row means delivery
    # is stalled - Telegram is queuing but not successfully pushing.
    prev_pending: int = 0
    consecutive_failures: int = 0
    failure_notified: bool = False

    while True:
        try:
            info = await bot.get_webhook_info()
            needs_reregister = False
            reason = ""

            # Re-register if the URL was cleared (e.g., by manual intervention
            # or a competing bot instance calling deleteWebhook)
            if not info.url:
                needs_reregister = True
                reason = "webhook URL is empty"

            # Re-register if Telegram reports a recent delivery error.
            # last_error_date is a datetime (None if no errors).
            elif info.last_error_date:
                error_age = time.time() - info.last_error_date.timestamp()
                if error_age < _ERROR_RECENCY_THRESHOLD:
                    needs_reregister = True
                    reason = f"recent error ({int(error_age)}s ago): {info.last_error_message or 'unknown'}"

            # Re-register if pending updates have been non-zero for two
            # consecutive checks - Telegram is queuing but can't deliver.
            current_pending = info.pending_update_count or 0
            if not needs_reregister and current_pending > 0 and prev_pending > 0:
                needs_reregister = True
                reason = f"pending_update_count stuck at {current_pending} (was {prev_pending} on previous check)"
            prev_pending = current_pending

            if needs_reregister:
                log.warning("Webhook health: %s - re-registering", reason)
                await bot.delete_webhook()
                await bot.set_webhook(
                    url=webhook_url,
                    secret_token=webhook_secret,
                    allowed_updates=["message", "callback_query"],
                )
                log.info("Webhook re-registered (self-healing)")
                # Reset pending tracker after re-registration so we don't
                # immediately trigger again on the next check
                prev_pending = 0

            consecutive_failures = 0
            failure_notified = False

        except Exception:
            # Don't let a failed health check kill the monitor loop.
            # Network blips, API rate limits, etc. are transient.
            log.exception("Webhook health check failed")
            consecutive_failures += 1

            # Notify admin once after 3 consecutive failures (15 min of
            # downtime at the 5-minute check interval). Don't spam - only
            # notify once until a successful check resets the flag.
            if consecutive_failures >= 3 and not failure_notified:
                try:
                    await bot.send_message(
                        chat_id,
                        "Webhook health monitor has failed 3 consecutive checks. Self-healing may be degraded.",
                    )
                except Exception:
                    # If we can't even reach Telegram, just log it.
                    log.warning("Could not send health monitor failure notification")
                # Set regardless of whether the send succeeded. We tried
                # once per failure sequence - don't retry every 5 minutes.
                failure_notified = True

        await asyncio.sleep(_HEALTH_CHECK_INTERVAL)


# ── Lifecycle ────────────────────────────────────────────────────────


def _register_routes(
    app: web.Application,
    config: Config,
    *,
    telegram_enabled: bool = True,
) -> None:
    """Register gated adapter routes and transport-independent internal APIs."""
    app.router.add_get("/health", _handle_health)

    if telegram_enabled and config.telegram_webhook_url:
        if not config.telegram_webhook_secret:
            raise RuntimeError("Telegram webhook mode requires a non-empty secret")
        app[TELEGRAM_WEBHOOK_SECRET_KEY] = config.telegram_webhook_secret
        app.router.add_post("/webhook/telegram", _handle_telegram_update)

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
    if telegram_enabled:
        # These compatibility operations still use Telegram delivery.
        app.router.add_post("/api/send-message", _handle_send_message)
        app.router.add_post("/api/send-file", _handle_send_file)


async def _register_workshop_client_api(
    app: web.Application,
    store: WorkshopEventStore,
    *,
    command_submitter: WorkshopClientCommandSubmitter | None = None,
    run_previews: WorkshopRunPreviewRegistry | None = None,
    artifact_service: WorkshopArtifactService | None = None,
    settings_workspaces: WorkshopSettingsWorkspaceService | None = None,
    memory_queries: WorkshopMemoryQueryService | None = None,
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
            memory_queries=memory_queries,
        )
        if command_submitter is not None:
            register_workshop_command_routes(
                target,
                store=store,
                authenticator=authenticator,
                submitter=command_submitter,
                request_lock=request_lock,
                artifact_service=artifact_service,
            )
        register_workshop_shell_routes(target)

    register(app)
    return register


async def start(
    telegram_app: Application | None,
    config: Config,
    *,
    core_host: KaiApplicationHost,
    core_services: KaiCoreServices,
    integration_notifications: WorkshopIntegrationNotificationService,
    workshop_enabled: bool = True,
) -> None:
    """
    Start the HTTP server and optionally register the Telegram webhook.

    The HTTP server always starts for health, authenticated integration ingress,
    and transport-independent internal APIs. Workshop client routes are
    independently enabled. Only Telegram webhook and compatibility delivery
    routes require a Telegram application.

    In polling mode, the server still runs but Telegram updates arrive via
    the Updater's long-polling loop in ``TelegramAdapter`` instead.

    Args:
        config: The application Config instance.
        core_host: Core lifecycle owner used for health/readiness reporting.
        core_services: Typed core dependencies required by HTTP routes.
        telegram_app: Started Telegram application, or None when disabled.
        integration_notifications: Core-owned canonical notification service.
        workshop_enabled: Whether to publish Workshop client routes.
    """
    global _app, _runner, _workshop_lan_runner, _webhook_registered, _health_monitor_task

    _app = web.Application(client_max_size=MAX_ARTIFACT_BYTES + 128 * 1024)
    _app[CORE_HOST_KEY] = core_host
    telegram_enabled = telegram_app is not None
    if telegram_enabled:
        _app[TELEGRAM_APP_KEY] = telegram_app
        _app[TELEGRAM_BOT_KEY] = telegram_app.bot

    pool = core_services.subprocess_pool
    internal_api_auth = getattr(pool, "internal_api_auth", None)
    if not isinstance(internal_api_auth, InternalAPIAuth):
        raise RuntimeError("Subprocess pool did not provide an internal API credential store")
    _app[INTERNAL_API_AUTH_KEY] = internal_api_auth

    # Set the fallback destination for unattributed external webhook events.
    # Internal API calls never use this value for identity; their credential
    # resolves a principal before the handler runs.
    if telegram_enabled:
        admins = config.get_admins()
        if admins:
            _app[CHAT_ID_KEY] = admins[0].telegram_id
        else:
            fallback = next(iter(config.user_configs.values()))
            log.warning(
                "No admin users defined in users.yaml; using %s "
                "(telegram_id: %d) as default webhook target. "
                "External notifications may route unexpectedly.",
                fallback.name,
                fallback.telegram_id,
            )
            _app[CHAT_ID_KEY] = fallback.telegram_id

    # Keep notification destinations separate from Config.allowed_user_ids,
    # which is the immutable-at-runtime source for Telegram inbound auth. The
    # API no longer consults this set for identity; credentials resolve their
    # principal server-side.
    _app[ALLOWED_USER_IDS_KEY] = set(config.allowed_user_ids)
    _app[NOTIFICATION_CHAT_IDS_KEY] = set()

    # Retain the loaded application configuration for compatibility APIs.
    _app[CONFIG_KEY] = config

    # Store the subprocess pool for per-user workspace lookup in send-file.
    # Set by main.py after pool creation; may be None during init.
    _app[POOL_KEY] = pool

    # Workspace policy lets canonical review work resolve a local checkout for
    # _resolve_local_repo() match incoming PR webhook repos against
    # local checkouts without a hardcoded GITHUB_REPO setting.
    _app[WORKSPACE_BASE_KEY] = str(config.workspace_base) if config.workspace_base else None
    _app[ALLOWED_WORKSPACES_KEY] = [str(p) for p in config.allowed_workspaces]

    # Maintain the legacy live notification-destination registry from both
    # users.yaml and DB. This set is intentionally detached from Config's
    # inbound Telegram principals and is not consulted by internal API auth.
    for uc in config.user_configs.values():
        if telegram_enabled and uc.github_notify_chat_id is not None:
            _app[NOTIFICATION_CHAT_IDS_KEY].add(uc.github_notify_chat_id)
    # Also add any DB-stored notify chat IDs (set via /github notify).
    # webhook.start() is already async so the await is fine.
    for uid in config.user_configs if telegram_enabled else ():
        val = await sessions.get_setting(f"github_notify_chat:{uid}")
        if val:
            try:
                _app[NOTIFICATION_CHAT_IDS_KEY].add(int(val))
            except ValueError:
                log.warning(
                    "Invalid github_notify_chat for user %s in DB: %s (ignoring)",
                    uid,
                    val,
                )
    _register_routes(_app, config, telegram_enabled=telegram_enabled)
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
            memory_queries=core_services.memory_queries,
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

    # Register the webhook URL with Telegram's API if in webhook mode. This must
    # come after the server is listening so the endpoint is ready before Telegram
    # starts pushing. allowed_updates limits which update types Telegram sends -
    # Kai only handles messages and callback queries (inline keyboard taps).
    #
    # Retry with backoff because Telegram's API can time out transiently,
    # especially after a period of downtime when queued updates are flushing.
    # Without retries, a single timeout kills the whole startup and launchd
    # eventually gives up restarting.
    if telegram_app is not None and config.telegram_webhook_url:
        requeued = await sessions.requeue_processing_telegram_updates()
        if requeued:
            log.info("Requeued %d unfinished Telegram update(s) from previous run", requeued)
        _ensure_telegram_update_queue_worker(telegram_app, telegram_app.bot)

        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                await telegram_app.bot.set_webhook(
                    url=config.telegram_webhook_url,
                    secret_token=config.telegram_webhook_secret,
                    allowed_updates=["message", "callback_query"],
                )
                _webhook_registered = True
                log.info("Registered Telegram webhook: %s", config.telegram_webhook_url)
                break
            except Exception:
                if attempt == max_attempts:
                    log.exception("Failed to register webhook after %d attempts", max_attempts)
                    raise
                wait = 2**attempt  # 2, 4, 8, 16s
                log.warning(
                    "Webhook registration attempt %d/%d failed, retrying in %ds",
                    attempt,
                    max_attempts,
                    wait,
                )
                await asyncio.sleep(wait)

        # Start the background health monitor to detect and recover from
        # Telegram delivery failures (e.g., Cloudflare tunnel drops causing
        # 502s that trigger Telegram's exponential backoff).
        _health_monitor_task = asyncio.create_task(
            _webhook_health_loop(
                telegram_app.bot,
                config.telegram_webhook_url,
                config.telegram_webhook_secret,
                _app[CHAT_ID_KEY],
            )
        )


async def stop() -> None:
    """
    Deregister the Telegram webhook (if active) and stop the HTTP server.

    Called during shutdown by ``HttpAdapter``. In webhook mode,
    deregisters the webhook with Telegram first (so Telegram stops sending
    updates to an endpoint that's about to disappear). In polling mode,
    skips the delete_webhook call since no webhook was registered.

    The delete_webhook call is wrapped in try/except because it's not critical -
    if the network is down at shutdown time, Telegram will just overwrite the
    stale webhook on the next set_webhook call at startup.
    """
    global _app, _runner, _workshop_lan_runner, _webhook_registered, _health_monitor_task
    # Cancel the webhook health monitor before tearing down the server
    if _health_monitor_task is not None:
        _health_monitor_task.cancel()
        try:
            await _health_monitor_task
        except asyncio.CancelledError:
            pass
        _health_monitor_task = None

    # Only deregister if we registered a webhook (i.e., webhook mode was active)
    if _webhook_registered and _app is not None:
        telegram_bot = _app.get(TELEGRAM_BOT_KEY)
        if telegram_bot is not None:
            try:
                await telegram_bot.delete_webhook()
                log.info("Deregistered Telegram webhook")
            except Exception:
                log.warning("Failed to deregister Telegram webhook (will re-register on next start)")
        _webhook_registered = False
    await _drain_background_tasks(_BACKGROUND_TASK_DRAIN_TIMEOUT)
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


def add_notification_chat_id(chat_id: int) -> None:
    """
    Add a chat_id to the live notification-destination registry.

    Called by bot.py when /github notify sets a new notification
    destination. This registry never grants Telegram or internal API
    authority; those identities come from users.yaml and API credentials.
    """
    if _app is not None:
        notification_chat_ids = _app.get(NOTIFICATION_CHAT_IDS_KEY)
        if notification_chat_ids is not None:
            notification_chat_ids.add(chat_id)


def remove_notification_chat_id(chat_id: int) -> None:
    """
    Remove a chat_id from the live notification registry, but only
    if it does not belong to an actual authorized user.

    Called by bot.py when /github notify reset clears a notification
    destination. A user's own telegram_id must never be removed.

    Config.allowed_user_ids is deliberately a different set object so
    notification changes cannot mutate inbound Telegram authorization.
    user_configs remains the immutable source for the preservation guard.
    """
    if _app is not None:
        notification_chat_ids = _app.get(NOTIFICATION_CHAT_IDS_KEY)
        if notification_chat_ids is None:
            return
        # Never remove a chat_id that belongs to an actual user.
        # user_configs keys are telegram_ids of real users.
        config = _app.get(CONFIG_KEY)
        if config and chat_id in config.user_configs:
            return
        notification_chat_ids.discard(chat_id)
