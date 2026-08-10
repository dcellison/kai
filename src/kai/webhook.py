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

The server always runs on aiohttp alongside the Telegram bot in the same event
loop, regardless of transport mode. In polling mode, Telegram updates arrive via
the Updater in main.py; this server still handles everything else.

Routes are organized into these groups:
    - /webhook/telegram     - Telegram updates (webhook mode only, secret_token auth)
    - /webhook/github       - GitHub events with HMAC-SHA256 signature validation
    - /webhook              - Generic webhooks with shared-secret auth
    - /api/schedule         - Job creation API (used by inner Claude via curl)
    - /api/jobs             - Job listing and detail API
    - /api/jobs/{id}        - Job detail (GET), deletion (DELETE), and update (PATCH)
    - /api/services/{name}  - External service proxy (injects auth from .env)
    - /api/send-message     - Send a text message to the Telegram chat
    - /api/send-file        - Send a file from the filesystem to the Telegram chat
    - /api/memory/add       - Store a structured memory (POST)
    - /api/memory/search    - Search memories by query (POST)
    - /api/memory/stats     - Memory statistics for a user (GET)
    - /api/memory/all       - Delete all memories for a user (DELETE, requires confirm token)

Every ingress domain has an independent credential. Telegram uses its configured
or process-generated secret token, GitHub uses GITHUB_WEBHOOK_SECRET, and the
generic endpoint uses GENERIC_WEBHOOK_SECRET. During migration only, the
deprecated WEBHOOK_SECRET is an external-only fallback for the GitHub and generic
routes; successful fallback use is logged. Internal API routes always use random,
principal-bound process credentials and do not depend on external webhook secrets.

GitHub events are formatted into human-readable Markdown messages and sent
to the configured Telegram chat. The formatter pattern (dispatch dict mapping
event type to formatter function) makes it easy to add new event types.
"""

import asyncio
import functools
import hashlib
import hmac
import json
import logging
import re
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from aiohttp import web
from telegram import Bot, Update
from telegram.ext import Application

from kai import cron, memory, review, services, sessions, triage
from kai.config import DATA_DIR, IMAGE_EXTENSIONS, Config, ModelRole, UserConfig, resolve_user_model
from kai.internal_api_auth import InternalAPIAuth, InternalAPIPrincipal, InternalAPIScope
from kai.telegram_utils import chunk_text

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
GENERIC_WEBHOOK_SECRET_KEY: web.AppKey[str] = web.AppKey("generic_webhook_secret", str)
GITHUB_WEBHOOK_SECRET_KEY: web.AppKey[str] = web.AppKey("github_webhook_secret", str)
INTERNAL_API_AUTH_KEY: web.AppKey[InternalAPIAuth] = web.AppKey("internal_api_auth", InternalAPIAuth)
LEGACY_WEBHOOK_SECRET_KEY: web.AppKey[str] = web.AppKey("legacy_webhook_secret", str)
POOL_KEY: web.AppKey[object] = web.AppKey("pool", object)
PR_REVIEW_COOLDOWN_KEY: web.AppKey[int] = web.AppKey("pr_review_cooldown", int)
PR_REVIEW_TIMEOUT_S_KEY: web.AppKey[int] = web.AppKey("pr_review_timeout_s", int)
SPEC_DIR_KEY: web.AppKey[str] = web.AppKey("spec_dir", str)
TELEGRAM_APP_KEY: web.AppKey[Application] = web.AppKey("telegram_app", Application)
TELEGRAM_BOT_KEY: web.AppKey[Bot] = web.AppKey("telegram_bot", Bot)
TELEGRAM_WEBHOOK_SECRET_KEY: web.AppKey[str] = web.AppKey("telegram_webhook_secret", str)
WEBHOOK_PORT_KEY: web.AppKey[int] = web.AppKey("webhook_port", int)
WORKSPACE_BASE_KEY: web.AppKey[str | None] = web.AppKey("workspace_base")


class UnauthorizedChatIdError(Exception):
    """Raised when a request targets a chat_id other than its principal."""


# Module-level server state, managed by start() and stop()
_app: web.Application | None = None
_runner: web.AppRunner | None = None
# Tracks whether we registered a Telegram webhook with the API, so stop()
# knows whether to call delete_webhook(). Only True in webhook mode.
_webhook_registered: bool = False

# Background tasks for processing Telegram updates. Tasks are kept in this set
# to prevent garbage collection (Python only weakly references fire-and-forget
# tasks, so an unreferenced task can be silently collected mid-execution).
# Each task removes itself from the set via a done callback.
_background_tasks: set[asyncio.Task] = set()
_BACKGROUND_TASK_DRAIN_TIMEOUT = 30.0

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

    Telegram update processing, PR review, and issue triage are launched as
    background tasks so inbound webhooks can be acknowledged promptly. During a
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


# If Telegram reports an error within this window, re-register the webhook.
# Slightly longer than the check interval so a single transient error
# in the previous cycle triggers a re-registration on the next check.
_ERROR_RECENCY_THRESHOLD = 600  # 10 minutes


# ── PR review rate limiting ─────────────────────────────────────────
# In-memory cooldown dict: (repo_full_name, pr_number) -> last_review_timestamp.
# Resets on restart, which is acceptable - worst case is one extra review
# after a restart. No database table needed for stateless reviews.
_review_cooldowns: dict[tuple[str, int], float] = {}


def _prune_expired(cooldowns: dict[tuple[str, int], float], max_age: float) -> None:
    """
    Remove entries older than max_age seconds from a cooldown dict.

    Called during record operations to prevent unbounded growth.
    Builds a list of expired keys first to avoid mutating the dict
    during iteration.

    Args:
        cooldowns: The cooldown dict to prune.
        max_age: Maximum entry age in seconds.
    """
    now = time.time()
    expired = [k for k, ts in cooldowns.items() if (now - ts) >= max_age]
    for k in expired:
        del cooldowns[k]


def _should_skip_review(repo: str, pr_number: int, cooldown: int) -> bool:
    """
    Check if a PR was reviewed recently enough to skip this event.

    Uses the in-memory cooldown dict to absorb force-push bursts.
    Returns True if the PR should NOT be reviewed (still in cooldown).

    Args:
        repo: GitHub repo full name (e.g., "dcellison/kai").
        pr_number: The PR number.
        cooldown: Minimum seconds between reviews of the same PR.
    """
    key = (repo, pr_number)
    last_review = _review_cooldowns.get(key)
    if last_review is None:
        return False
    return (time.time() - last_review) < cooldown


def _record_review(repo: str, pr_number: int, cooldown: float) -> None:
    """
    Record that a PR review was just initiated, updating the cooldown timestamp.

    Prunes expired entries first to prevent unbounded dict growth.
    Called after a review is successfully launched (not after it completes,
    since the review runs as a background task and we want to prevent
    duplicate launches, not duplicate completions).

    Args:
        repo: GitHub repo full name (e.g., "dcellison/kai").
        pr_number: The PR number.
        cooldown: Cooldown period in seconds (used as pruning threshold).
    """
    _prune_expired(_review_cooldowns, cooldown)
    _review_cooldowns[(repo, pr_number)] = time.time()


# ── Issue triage rate limiting ─────────────────────────────────────────
# In-memory cooldown dict: (repo_full_name, issue_number) -> last_triage_timestamp.
# Prevents duplicate triage if GitHub sends multiple webhook deliveries
# for the same event (retries, duplicate deliveries). 60-second cooldown.
_triage_cooldowns: dict[tuple[str, int], float] = {}

# Fixed cooldown for triage - much shorter than PR review (300s) because
# issues don't have the force-push burst problem. This is purely for
# duplicate delivery protection.
_TRIAGE_COOLDOWN_SECONDS = 60


def _should_skip_triage(repo: str, issue_number: int) -> bool:
    """
    Check if an issue was triaged recently enough to skip this event.

    Uses the in-memory cooldown dict to absorb duplicate deliveries.
    Returns True if the issue should NOT be triaged (still in cooldown).

    Args:
        repo: GitHub repo full name (e.g., "dcellison/kai").
        issue_number: The issue number.
    """
    key = (repo, issue_number)
    last_triage = _triage_cooldowns.get(key)
    if last_triage is None:
        return False
    return (time.time() - last_triage) < _TRIAGE_COOLDOWN_SECONDS


def _record_triage(repo: str, issue_number: int) -> None:
    """
    Record that an issue triage was just initiated.

    Prunes expired entries first to prevent unbounded dict growth.
    Called after a triage is successfully launched (not after it completes,
    since the triage runs as a background task and we want to prevent
    duplicate launches, not duplicate completions).

    Args:
        repo: GitHub repo full name (e.g., "dcellison/kai").
        issue_number: The issue number.
    """
    _prune_expired(_triage_cooldowns, _TRIAGE_COOLDOWN_SECONDS)
    _triage_cooldowns[(repo, issue_number)] = time.time()


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
    """Validate the generic credential, with observable legacy fallback."""

    @functools.wraps(handler)
    async def wrapper(request: web.Request) -> web.Response:
        secret = request.app[GENERIC_WEBHOOK_SECRET_KEY]
        legacy_secret = request.app.get(LEGACY_WEBHOOK_SECRET_KEY, "")
        provided = request.headers.get("X-Webhook-Secret", "")
        if secret and hmac.compare_digest(provided, secret):
            return await handler(request)
        if legacy_secret and hmac.compare_digest(provided, legacy_secret):
            log.warning(
                "Legacy WEBHOOK_SECRET authenticated /webhook from %s; migrate this caller to GENERIC_WEBHOOK_SECRET",
                request.remote,
            )
            return await handler(request)
        else:
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
                    "Internal API scope denial on %s for principal %d",
                    request.path,
                    principal.chat_id,
                )
                return web.Response(status=403, text="Credential is not authorized for this operation")
            return await handler(request, principal)

        return wrapper

    return decorator


def _resolve_chat_id(principal: InternalAPIPrincipal, payload: dict) -> int:
    """
    Resolve identity from the authenticated principal, not request data.

    A caller may include chat_id for clarity and compatibility, but it must
    match the server-resolved credential owner. Omitting chat_id uses that
    owner directly; there is no admin or application-default fallback.

    Raises:
        ValueError: If chat_id is present but not a valid integer.
        UnauthorizedChatIdError: If chat_id differs from the principal.
    """
    explicit = payload.get("chat_id")
    if explicit is not None:
        # Reject bools (int(True) == 1) and non-integer floats
        if isinstance(explicit, bool):
            raise ValueError(f"chat_id must be an integer, got {explicit!r}")
        if isinstance(explicit, float) and not float(explicit).is_integer():
            raise ValueError(f"chat_id must be an integer, got {explicit!r}")
        try:
            resolved = int(explicit)
        except (TypeError, ValueError):
            raise ValueError(f"chat_id must be an integer, got {explicit!r}") from None
        if resolved != principal.chat_id:
            raise UnauthorizedChatIdError(
                f"chat_id {resolved} does not match authenticated principal {principal.chat_id}"
            )
        return resolved
    return principal.chat_id


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
    """Return liveness plus the non-sensitive memory feature mode."""
    return web.json_response({"status": "ok", "memory_enabled": memory.is_enabled()})


async def _handle_telegram_update(request: web.Request) -> web.Response:
    """
    Receive a Telegram update pushed via webhook.

    Validates the X-Telegram-Bot-Api-Secret-Token header against the configured
    secret, deserializes the JSON body into a python-telegram-bot Update object,
    and dispatches it to the existing handler system via process_update().

    IMPORTANT: process_update() is launched as a background task, not awaited.
    Claude responses can take 30+ seconds, and Telegram's webhook client times
    out after ~30-35s. If we awaited process_update(), Telegram would assume
    delivery failed and retry the same message, causing duplicate responses.
    By returning 200 immediately and processing in the background, we acknowledge
    receipt before Telegram's timeout. The per-chat lock in bot.py serializes
    concurrent messages, so ordering is preserved.

    Returns 200 after the update is queued. Payload errors still return 200 to
    avoid permanent Telegram retries, but a local enqueue failure returns 500
    because no work was accepted and a retry may succeed.
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

    try:
        update = Update.de_json(data, bot)
    except Exception:
        log.exception("Error deserializing Telegram update")
        return web.Response(status=200)
    if not update:
        return web.Response(status=200)

    # Fire-and-forget: return 200 to Telegram immediately while processing
    # continues in the background. Without this, Claude's response time would
    # exceed Telegram's webhook timeout.
    process_update = None
    try:
        process_update = telegram_app.process_update(update)
        task = asyncio.create_task(process_update)
    except Exception:
        if process_update is not None:
            close = getattr(process_update, "close", None)
            if close is not None:
                close()
        log.exception("Failed to enqueue Telegram update")
        return web.Response(status=500, text="Failed to enqueue update")
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return web.Response(status=200)


async def _handle_github(request: web.Request) -> web.Response:
    """
    Handle incoming GitHub webhook events.

    Validates the HMAC-SHA256 signature, parses the event payload, dispatches
    to the appropriate formatter, and sends the formatted message to Telegram.
    Falls back to plain text if Markdown parsing fails.

    Supported events: push, pull_request, issues, issue_comment, pull_request_review.
    Unsupported events are silently acknowledged with {"msg": "ignored"}.
    """
    secret = request.app[GITHUB_WEBHOOK_SECRET_KEY]
    legacy_secret = request.app.get(LEGACY_WEBHOOK_SECRET_KEY, "")

    body = await request.read()

    # Validate HMAC-SHA256 signature from GitHub
    signature = request.headers.get("X-Hub-Signature-256", "")
    if secret and _verify_github_signature(secret, body, signature):
        pass
    elif legacy_secret and _verify_github_signature(legacy_secret, body, signature):
        log.warning(
            "Legacy WEBHOOK_SECRET authenticated /webhook/github from %s; "
            "migrate the GitHub webhook to GITHUB_WEBHOOK_SECRET",
            request.remote,
        )
    else:
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


async def _get_subscribed_users(config: Config, repo_full_name: str) -> list[UserConfig]:
    """Find all users subscribed to a GitHub repo.

    Computes each user's effective repo list (yaml + DB-added - DB-removed)
    via sessions.get_effective_repos(), then matches against repo_full_name.
    Comparison is case-insensitive since GitHub repo names are
    case-insensitive.

    Admin users with empty effective repos are treated as wildcards: they
    receive all events for all repos, matching the pre-multi-user behavior
    where admins received GitHub events globally. An admin who wants to
    receive only specific repos should list them explicitly or use
    /github add - once any repos are in the effective list, the wildcard
    no longer applies.

    Non-admin users with empty effective repos receive nothing. Regular
    users never received global events before the multi-user buildout, so
    this is the correct backward-compatible default.

    Args:
        config: The application Config instance.
        repo_full_name: Full GitHub repo name (e.g., "dcellison/kai").

    Returns:
        List of UserConfig objects for users who should receive this event.
        May contain both explicitly-subscribed users and admin wildcards.
    """
    repo_lower = repo_full_name.lower()

    explicitly_subscribed: list[UserConfig] = []
    admin_wildcards: list[UserConfig] = []

    for uc in config.user_configs.values():
        # Compute effective repos: yaml baseline + DB-added - DB-removed.
        effective = await sessions.get_effective_repos(uc.telegram_id, uc.github_repos)

        if repo_lower in effective:
            explicitly_subscribed.append(uc)
        elif uc.role == "admin" and not effective:
            # Admin with no effective repos = wildcard (receives all events).
            admin_wildcards.append(uc)

    # No deduplication needed: admin_wildcards requires empty effective
    # repos, so no user can appear in both lists simultaneously.
    return explicitly_subscribed + admin_wildcards


async def _process_github_event(request: web.Request, payload: dict, event_type: str) -> web.Response:
    """Process a validated GitHub webhook event.

    Routes the event to all users subscribed to the triggering repo
    via _get_subscribed_users(). Each user gets their own feature flag
    check and notification destination via resolve_github_settings().

    Admin users with empty github_repos receive all events (wildcard).
    Only reaches the fallback path when no user is subscribed and no
    admin wildcard exists - for example, when no users.yaml is configured
    or when all admins have opted into specific repos only.
    """
    bot = request.app[TELEGRAM_BOT_KEY]
    config: Config = request.app[CONFIG_KEY]

    # Extract the repo that triggered this event
    repo_full_name = payload.get("repository", {}).get("full_name", "")

    # Find all users subscribed to this repo
    subscribed_users = await _get_subscribed_users(config, repo_full_name)

    if not subscribed_users:
        # No subscribers and no admin wildcards for this repo. With
        # users.yaml mandatory the cause is now narrowly that all
        # admins have explicit github_repos that don't include this
        # repo, or that no admin is configured at all (a warning at
        # config load time covers the latter).
        log.warning(
            "GitHub %s event for %s: no subscribed users. "
            "Add 'github_repos: [%s]' to users.yaml to receive "
            "events for this repo.",
            event_type,
            repo_full_name,
            repo_full_name,
        )
        # Fall back to the first admin in the app config so the event is
        # not silently dropped. Wrapped in try/except for consistency with
        # the fan-out path - a transient failure should return 200, not 500.
        fallback_chat_id = request.app[CHAT_ID_KEY]
        try:
            await _process_github_event_for_user(
                request,
                payload,
                event_type,
                bot,
                config,
                fallback_chat_id,
            )
        except Exception:
            log.exception(
                "Error processing %s event for fallback admin (chat %d)",
                event_type,
                fallback_chat_id,
            )
        return web.json_response({"status": "ok"})

    # Process the event for each subscribed user independently.
    # Each user has their own pr_review/issue_triage flags and
    # notification destination. Per-user try/except ensures a transient
    # failure (e.g., DB error) for one user does not block the others.
    for user_config in subscribed_users:
        chat_id = user_config.telegram_id
        try:
            await _process_github_event_for_user(
                request,
                payload,
                event_type,
                bot,
                config,
                chat_id,
            )
        except Exception:
            log.exception(
                "Error processing %s event for user %s (chat %d)",
                event_type,
                user_config.name,
                chat_id,
            )

    return web.json_response({"status": "ok"})


async def _process_github_event_for_user(
    request: web.Request,
    payload: dict,
    event_type: str,
    bot: Bot,
    config: Config,
    chat_id: int,
) -> None:
    """Process a GitHub event for a single user.

    Resolves the user's GitHub settings (pr_review, issue_triage,
    notify_chat_id) and dispatches accordingly. Called once per
    subscribed user for each incoming event.

    Args:
        request: The aiohttp request (for app config access).
        payload: The GitHub webhook payload dict.
        event_type: GitHub event type (e.g., "pull_request", "issues").
        bot: The Telegram bot instance.
        config: The application Config instance.
        chat_id: The user's Telegram chat ID.
    """
    # Resolve this user's effective GitHub settings
    settings = await sessions.resolve_github_settings(chat_id, config)
    target_chat_id = settings["notify_chat_id"]

    # Resolve per-user os_user for subprocess isolation. None falls
    # back to "run as the bot's own OS user" inside the agent backend.
    user_config = config.get_user_config(chat_id)
    claude_user = user_config.os_user if user_config and user_config.os_user else None
    repo_full_name = payload.get("repository", {}).get("full_name", "")
    github_operations_authorized = bool(user_config and user_config.authorizes_github_repo(repo_full_name))

    # Resolve per-user backend/provider for the review/triage agents.
    # Same resolution logic as pool.py _create_instance: per-user
    # config overrides global config.
    if user_config and user_config.backend:
        agent_backend = user_config.backend
    else:
        agent_backend = config.default_backend

    if user_config and user_config.provider:
        provider = user_config.provider
    else:
        provider = config.default_provider

    # Per-role model overrides for review and triage. `resolve_user_model`
    # implements the full precedence chain: per-user `models:` from
    # users.yaml > Config.default_models from DEFAULT_MODELS_JSON >
    # MODEL_REGISTRY's (backend, provider, role) default. review.py /
    # triage.py treat the empty string as "fall through to the registry
    # default" via the model_override parameter; the resolver always
    # returns a concrete model string, so the override path here is
    # load-bearing only when the resolved value differs from the
    # registry default (which is exactly when it should fire).
    pr_review_model_override = resolve_user_model(ModelRole.PR_REVIEW, user_config, config)
    issue_triage_model_override = resolve_user_model(ModelRole.ISSUE_TRIAGE, user_config, config)

    # ── PR review routing ────────────────────────────────────────
    # When PR review is enabled for this user, reviewable PR events
    # (opened, reopened, synchronize) are routed to the review pipeline
    # instead of the notification formatter. Non-reviewable actions
    # (closed, merged) still get the standard Telegram notification.
    if settings["pr_review"] and event_type == "pull_request":
        action = payload.get("action", "")
        if action in ("opened", "reopened", "synchronize") and github_operations_authorized:
            pr = payload.get("pull_request", {})
            pr_number = pr.get("number", 0)
            repo = repo_full_name
            cooldown = request.app.get(PR_REVIEW_COOLDOWN_KEY, 300)

            # Cooldown is server-level, shared across all users.
            # One review per PR per cooldown window regardless of
            # how many users are subscribed.
            if _should_skip_review(repo, pr_number, cooldown):
                log.info(
                    "Skipping review of %s PR #%d (cooldown)",
                    repo,
                    pr_number,
                )
                return

            _record_review(repo, pr_number, cooldown)

            # Resolve a local repo path for spec/convention loading.
            # Checks home workspace, WORKSPACE_BASE, ALLOWED_WORKSPACES,
            # and workspace history for a directory matching the repo name.
            local_repo_path = await _resolve_local_repo(repo, request.app)

            # Launch the review as a fire-and-forget background task.
            # Same pattern as Telegram update processing: create_task +
            # _background_tasks set to prevent GC during execution.
            # Always pass target_chat_id so the review notification
            # reaches the subscribing user, not the default admin.
            task = asyncio.create_task(
                review.review_pr(
                    payload,
                    webhook_port=request.app[WEBHOOK_PORT_KEY],
                    webhook_secret=request.app[INTERNAL_API_AUTH_KEY].notification_credential_for(target_chat_id),
                    claude_user=claude_user,
                    local_repo_path=local_repo_path,
                    spec_dir=request.app.get(SPEC_DIR_KEY, "specs"),
                    notify_chat_id=target_chat_id,
                    agent_backend=agent_backend,
                    provider=provider,
                    timeout_s=request.app[PR_REVIEW_TIMEOUT_S_KEY],
                    model_override=pr_review_model_override,
                )
            )
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

            log.info(
                "PR review triggered for %s PR #%d (%s) for user %d",
                repo,
                pr_number,
                action,
                chat_id,
            )
            return
        if action in ("opened", "reopened", "synchronize"):
            log.warning(
                "Denied automated GitHub review for user %d: repository %s is not admin-authorized",
                chat_id,
                repo_full_name,
            )

    # ── Issue triage routing ────────────────────────────────────
    # When issue triage is enabled for this user, opened issues are
    # routed to the triage pipeline. The triage Telegram summary
    # replaces the basic _fmt_issues() notification (richer content).
    # Non-triaged actions (closed, reopened) still fall through to
    # the standard formatter.
    if settings["issue_triage"] and event_type == "issues":
        action = payload.get("action", "")
        if action == "opened" and github_operations_authorized:
            issue = payload.get("issue", {})
            issue_number = issue.get("number", 0)
            repo = repo_full_name

            if _should_skip_triage(repo, issue_number):
                log.info(
                    "Skipping triage of %s issue #%d (cooldown)",
                    repo,
                    issue_number,
                )
                return

            _record_triage(repo, issue_number)

            task = asyncio.create_task(
                triage.triage_issue(
                    payload,
                    webhook_port=request.app[WEBHOOK_PORT_KEY],
                    webhook_secret=request.app[INTERNAL_API_AUTH_KEY].notification_credential_for(target_chat_id),
                    claude_user=claude_user,
                    notify_chat_id=target_chat_id,
                    agent_backend=agent_backend,
                    provider=provider,
                    model_override=issue_triage_model_override,
                    allowed_triage_projects=user_config.allowed_triage_projects if user_config else [],
                )
            )
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

            log.info(
                "Issue triage triggered for %s issue #%d for user %d",
                repo,
                issue_number,
                chat_id,
            )
            return
        if action == "opened":
            log.warning(
                "Denied automated GitHub issue triage for user %d: repository %s is not admin-authorized",
                chat_id,
                repo_full_name,
            )

    # ── Standard notification path ───────────────────────────────
    # Look up the formatter for this event type
    formatter = _GITHUB_FORMATTERS.get(event_type)
    if not formatter:
        return

    message = formatter(payload)
    if not message:
        return

    # Send to Telegram with Markdown, falling back to plain text on parse failure
    try:
        await bot.send_message(target_chat_id, message, parse_mode="Markdown")
    except Exception:
        try:
            await bot.send_message(target_chat_id, _strip_markdown(message))
        except Exception:
            log.exception(
                "Failed to send GitHub notification to chat %d",
                target_chat_id,
            )
    log.info(
        "Sent GitHub %s notification to chat %d (user %d)",
        event_type,
        target_chat_id,
        chat_id,
    )


@_require_generic_webhook_secret
async def _handle_generic(request: web.Request) -> web.Response:
    """
    Handle generic webhook notifications from any source.

    Extracts a "message" field from the JSON payload (or dumps the full
    payload) and forwards it to the Telegram chat. Truncates to Telegram's
    4096-char limit.
    """
    bot = request.app[TELEGRAM_BOT_KEY]
    chat_id = request.app[CHAT_ID_KEY]

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
    if len(text) > 4096:
        text = text[:4093] + "..."

    try:
        await bot.send_message(chat_id, text)
    except Exception:
        log.exception("Failed to send generic webhook notification")

    return web.json_response({"status": "ok"})


# ── Scheduling API ───────────────────────────────────────────────────

# Valid schedule types accepted by the scheduling API
_VALID_SCHEDULE_TYPES = ("once", "daily", "interval")

# Valid job types accepted by the scheduling API
_VALID_JOB_TYPES = ("reminder", "claude")


def _validate_schedule_data(
    schedule_data: dict | str,
    schedule_type: str,
) -> tuple[str | None, str | None]:
    """
    Validate schedule_data and return the serialized JSON string.

    Handles both dict and pre-serialized string inputs. Validates
    that the JSON structure matches what the given schedule_type
    requires so cron.py never encounters malformed data at fire time.

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

    This is the primary interface for the inner Claude Code process to create
    scheduled tasks. Claude uses curl to POST here from within the workspace.

    Required JSON fields: name, prompt, schedule_type, schedule_data.
    Optional fields: job_type (default "reminder"), auto_remove (default false),
        notify_on_check (default false).

    The job is persisted to the database and immediately registered with
    APScheduler so it starts firing without a restart.

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

    # Optional fields with defaults — validate job_type the same way as schedule_type
    job_type = payload.get("job_type", "reminder")
    if job_type not in _VALID_JOB_TYPES:
        return web.json_response(
            {"error": f"job_type must be one of: {', '.join(_VALID_JOB_TYPES)}"},
            status=400,
        )
    auto_remove = payload.get("auto_remove", False)
    notify_on_check = payload.get("notify_on_check", False)
    try:
        chat_id = _resolve_chat_id(principal, payload)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except UnauthorizedChatIdError as e:
        return web.json_response({"error": str(e)}, status=403)

    # Validate schedule_data structure before persisting. Catches malformed
    # JSON strings and wrong-shape payloads (e.g., interval keys on a daily
    # job) that would otherwise crash cron.py on every fire attempt.
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

    # Register with APScheduler immediately so it starts firing
    telegram_app = request.app[TELEGRAM_APP_KEY]
    try:
        registered = await cron.register_job_by_id(telegram_app, job_id)
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

    Used by the inner Claude to check what jobs are currently scheduled
    without needing to parse Telegram bot command output.
    """

    # Resolve the caller's identity. Query params are passed as the payload
    # dict so _resolve_chat_id can extract and validate chat_id the same
    # way it does for POST endpoints with JSON bodies.
    try:
        chat_id = _resolve_chat_id(principal, dict(request.query))
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except UnauthorizedChatIdError as e:
        return web.json_response({"error": str(e)}, status=403)
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
        chat_id = _resolve_chat_id(principal, dict(request.query))
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except UnauthorizedChatIdError as e:
        return web.json_response({"error": str(e)}, status=403)

    job = await sessions.get_job_by_id(job_id)
    # Return 404 for missing OR wrong-owner jobs (don't leak existence).
    # Both sides are ints: _resolve_chat_id always returns int, and
    # chat_id is stored as INTEGER in the jobs table.
    if not job or job["chat_id"] != chat_id:
        return web.json_response({"error": "Job not found"}, status=404)
    return web.json_response(job)


@_require_internal_api(InternalAPIScope.JOBS_WRITE)
async def _handle_delete_job(request: web.Request, principal: InternalAPIPrincipal) -> web.Response:
    """
    Delete a scheduled job by ID via the HTTP API.

    Removes the job from both the database and APScheduler's in-memory
    queue. Uses the same logic as the /canceljob Telegram command.
    Returns 404 if the job doesn't exist.
    """

    try:
        job_id = int(request.match_info["id"])
    except ValueError:
        return web.json_response({"error": "Invalid job ID"}, status=400)

    # Resolve caller identity from its server-owned principal. A query-string
    # chat_id is accepted only when it repeats that identity.
    try:
        chat_id = _resolve_chat_id(principal, dict(request.query))
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except UnauthorizedChatIdError as e:
        return web.json_response({"error": str(e)}, status=403)
    deleted = await sessions.delete_job(job_id, chat_id=chat_id)
    if not deleted:
        return web.json_response({"error": "Job not found"}, status=404)

    # Remove from APScheduler's in-memory queue. Daily jobs with multiple
    # times get suffixed names (e.g., cron_19_0, cron_19_1), so we match
    # both the exact name and the prefix pattern.
    telegram_app = request.app[TELEGRAM_APP_KEY]
    jq = telegram_app.job_queue
    assert jq is not None
    prefix = f"cron_{job_id}"
    for j in jq.jobs():
        if j.name == prefix or (j.name and j.name.startswith(f"{prefix}_")):
            j.schedule_removal()

    log.info("Deleted job %d via API", job_id)
    return web.json_response({"deleted": job_id})


@_require_internal_api(InternalAPIScope.JOBS_WRITE)
async def _handle_update_job(request: web.Request, principal: InternalAPIPrincipal) -> web.Response:
    """
    Update a scheduled job's mutable fields via the HTTP API.

    Accepts a JSON body with any of: name, prompt, schedule_type,
    schedule_data, auto_remove, notify_on_check. Only provided fields
    are updated. If the schedule changes (type or data), the job is
    re-registered with APScheduler to pick up the new timing.

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

    # Resolve caller identity from the JSON body (e.g., {"chat_id": 456, ...}).
    # Falls back to admin chat_id when omitted. The chat_id field is used
    # solely for authorization (WHERE clause filter in update_job), not as
    # an updatable column - update_job only writes explicitly named fields
    # (name, prompt, schedule_type, etc.) to the SET clause.
    try:
        chat_id = _resolve_chat_id(principal, payload)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except UnauthorizedChatIdError as e:
        return web.json_response({"error": str(e)}, status=403)
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

    # If schedule changed, re-register with APScheduler. Remove old entries
    # and re-register with the new schedule from the database.
    schedule_changed = new_schedule_type is not None or schedule_data is not None
    if schedule_changed:
        # Remove old APScheduler entries
        telegram_app = request.app[TELEGRAM_APP_KEY]
        jq = telegram_app.job_queue
        assert jq is not None
        prefix = f"cron_{job_id}"
        for j in jq.jobs():
            if j.name == prefix or (j.name and j.name.startswith(f"{prefix}_")):
                j.schedule_removal()
        # Re-register with new schedule
        await cron.register_job_by_id(telegram_app, job_id)

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
            "Internal API service denial for principal %d: %s",
            principal.chat_id,
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
        chat_id = _resolve_chat_id(principal, payload)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except UnauthorizedChatIdError as e:
        return web.json_response({"error": str(e)}, status=403)

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
        chat_id = _resolve_chat_id(principal, payload)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except UnauthorizedChatIdError as e:
        return web.json_response({"error": str(e)}, status=403)

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
    # upload directory. Telegram uploads are stored under
    # DATA_DIR/files/<chat_id>/; the legacy shared DATA_DIR/files/ root and
    # sibling principals' directories are deliberately excluded. The scope is
    # derived from the authenticated principal, never from a caller-selected
    # request field.
    workspace_resolved = Path(workspace).resolve()
    principal_files_resolved = (DATA_DIR / "files" / str(principal.chat_id)).resolve()
    try:
        path.relative_to(workspace_resolved)
    except ValueError:
        try:
            path.relative_to(principal_files_resolved)
        except ValueError:
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
#   - Memory primitives take user_id as a string; _resolve_chat_id returns
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
        chat_id: int, optional, defaults to the authenticated principal

    Responses:
        200 {"id": "<mem0-uuid>"} on success
        400 on bad input (invalid JSON, missing/empty content, bad chat_id)
        401 on bad/missing internal API credential (handled by the decorator)
        403 on a chat_id that differs from the authenticated principal
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

    try:
        chat_id = _resolve_chat_id(principal, payload)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except UnauthorizedChatIdError as e:
        return web.json_response({"error": str(e)}, status=403)

    # Symmetric is_enabled() precheck. Runs after auth + 400-level
    # validation but before the primitive call. See the §Memory API
    # block comment above for why all four memory handlers share this.
    if not memory.is_enabled():
        return _memory_disabled_response()

    # int -> str at the memory boundary; do NOT remove the cast.
    user_id = str(chat_id)

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
            metadata=metadata,
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
        chat_id: int, optional, defaults to the authenticated principal

    Responses:
        200 {"results": [<MemoryResult-dict>, ...]} on success (empty list
            is a valid 200 result for "no matches")
        400 on bad input
        401 on bad secret
        403 on unauthorized chat_id
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
        chat_id = _resolve_chat_id(principal, payload)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except UnauthorizedChatIdError as e:
        return web.json_response({"error": str(e)}, status=403)

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

    GET endpoint reads chat_id from query params, mirroring _handle_get_jobs.

    Query params:
        chat_id: int, optional (defaults to the authenticated principal)

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
        400 on bad chat_id
        401 on bad secret
        403 on unauthorized chat_id
        503 when memory is disabled
    """
    # GET handlers feed query params to _resolve_chat_id; established
    # pattern at _handle_get_jobs.
    try:
        chat_id = _resolve_chat_id(principal, dict(request.query))
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except UnauthorizedChatIdError as e:
        return web.json_response({"error": str(e)}, status=403)

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

    Convention divergence (deliberate): _handle_delete_job reads chat_id
    from query params, but this handler reads chat_id from the body
    alongside the confirm token, since the body already exists for the
    confirm field. Splitting confirm into the body and chat_id into the
    query would be inconsistent within the same endpoint.

    Request body (JSON):
        confirm: str, required, must equal "delete-all-memories"
        chat_id: int, optional, defaults to the authenticated principal

    Responses:
        200 {"status": "deleted"} on success
        400 on missing/wrong confirm or bad chat_id
        401 on bad secret
        403 on unauthorized chat_id
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

    # Confirm-token check before chat_id resolution: a request with the
    # wrong token is structurally bad input regardless of which user it
    # was aimed at, and rejecting it first means we don't waste a
    # _resolve_chat_id call on requests we will reject anyway.
    if payload.get("confirm") != _DELETE_ALL_CONFIRM_TOKEN:
        return web.json_response(
            {"error": f'Missing or incorrect confirm field; expected "{_DELETE_ALL_CONFIRM_TOKEN}"'},
            status=400,
        )

    try:
        chat_id = _resolve_chat_id(principal, payload)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except UnauthorizedChatIdError as e:
        return web.json_response({"error": str(e)}, status=403)

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


def _register_routes(app: web.Application, config: Config) -> None:
    """Register independently gated external routes and always-on internal APIs."""
    app.router.add_get("/health", _handle_health)

    if config.webhook_secret:
        app[LEGACY_WEBHOOK_SECRET_KEY] = config.webhook_secret

    if config.telegram_webhook_url:
        if not config.telegram_webhook_secret:
            raise RuntimeError("Telegram webhook mode requires a non-empty secret")
        app[TELEGRAM_WEBHOOK_SECRET_KEY] = config.telegram_webhook_secret
        app.router.add_post("/webhook/telegram", _handle_telegram_update)

    if config.github_webhook_secret or config.webhook_secret:
        app[GITHUB_WEBHOOK_SECRET_KEY] = config.github_webhook_secret
        app.router.add_post("/webhook/github", _handle_github)
    else:
        log.warning("GITHUB_WEBHOOK_SECRET not set - GitHub webhook endpoint disabled")

    if config.generic_webhook_secret or config.webhook_secret:
        app[GENERIC_WEBHOOK_SECRET_KEY] = config.generic_webhook_secret
        app.router.add_post("/webhook", _handle_generic)
    else:
        log.info("GENERIC_WEBHOOK_SECRET not set - generic webhook endpoint disabled")

    # These routes authenticate through INTERNAL_API_AUTH_KEY. Their
    # availability must not be coupled to any public ingress configuration.
    app.router.add_post("/api/schedule", _handle_schedule)
    app.router.add_get("/api/jobs", _handle_get_jobs)
    app.router.add_get("/api/jobs/{id}", _handle_get_job)
    app.router.add_delete("/api/jobs/{id}", _handle_delete_job)
    app.router.add_patch("/api/jobs/{id}", _handle_update_job)
    app.router.add_post("/api/services/{name}", _handle_service_call)
    app.router.add_post("/api/send-message", _handle_send_message)
    app.router.add_post("/api/send-file", _handle_send_file)
    app.router.add_post("/api/memory/add", _handle_memory_add)
    app.router.add_post("/api/memory/search", _handle_memory_search)
    app.router.add_get("/api/memory/stats", _handle_memory_stats)
    app.router.add_delete("/api/memory/all", _handle_memory_delete_all)


async def start(telegram_app, config) -> None:
    """
    Start the HTTP server and optionally register the Telegram webhook.

    The HTTP server always starts regardless of transport mode - it serves the
    scheduling API, GitHub webhooks, file exchange, and health check. The
    Telegram webhook route and set_webhook API call are only added/made when
    config.telegram_webhook_url is set (webhook mode).

    In polling mode, the server still runs but Telegram updates arrive via
    the Updater's long-polling loop in main.py instead.

    Args:
        telegram_app: The python-telegram-bot Application instance.
        config: The application Config instance.
    """
    global _app, _runner, _webhook_registered, _health_monitor_task

    _app = web.Application()
    _app[TELEGRAM_APP_KEY] = telegram_app
    _app[TELEGRAM_BOT_KEY] = telegram_app.bot

    pool = telegram_app.bot_data.get("pool")
    internal_api_auth = getattr(pool, "internal_api_auth", None)
    if not isinstance(internal_api_auth, InternalAPIAuth):
        raise RuntimeError("Subprocess pool did not provide an internal API credential store")
    _app[INTERNAL_API_AUTH_KEY] = internal_api_auth

    # Set the fallback destination for unattributed external webhook events.
    # Internal API calls never use this value for identity; their credential
    # resolves a principal before the handler runs.
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

    # Store config for GitHub actor routing in _handle_github()
    _app[CONFIG_KEY] = config

    # Store the subprocess pool for per-user workspace lookup in send-file.
    # Set by main.py after pool creation; may be None during init.
    _app[POOL_KEY] = pool

    # Server-level config for review/triage background tasks. Per-user
    # feature flags (pr_review, issue_triage) are resolved at event time
    # via sessions.resolve_github_settings(), not stored here.
    _app[PR_REVIEW_COOLDOWN_KEY] = config.pr_review_cooldown
    _app[PR_REVIEW_TIMEOUT_S_KEY] = config.pr_review_timeout_s
    _app[WEBHOOK_PORT_KEY] = config.webhook_port

    # Workspace config for review agent repo resolution. These let
    # _resolve_local_repo() match incoming PR webhook repos against
    # local checkouts without a hardcoded GITHUB_REPO setting.
    _app[WORKSPACE_BASE_KEY] = str(config.workspace_base) if config.workspace_base else None
    _app[ALLOWED_WORKSPACES_KEY] = [str(p) for p in config.allowed_workspaces]
    _app[SPEC_DIR_KEY] = config.spec_dir

    # Maintain the legacy live notification-destination registry from both
    # users.yaml and DB. This set is intentionally detached from Config's
    # inbound Telegram principals and is not consulted by internal API auth.
    for uc in config.user_configs.values():
        if uc.github_notify_chat_id is not None:
            _app[ALLOWED_USER_IDS_KEY].add(uc.github_notify_chat_id)
    # Also add any DB-stored notify chat IDs (set via /github notify).
    # webhook.start() is already async so the await is fine.
    for uid in config.user_configs:
        val = await sessions.get_setting(f"github_notify_chat:{uid}")
        if val:
            try:
                _app[ALLOWED_USER_IDS_KEY].add(int(val))
            except ValueError:
                log.warning(
                    "Invalid github_notify_chat for user %s in DB: %s (ignoring)",
                    uid,
                    val,
                )
    _register_routes(_app, config)

    _runner = web.AppRunner(_app, access_log=None)
    await _runner.setup()
    # Bind to localhost only - all external access routes through Cloudflare Tunnel,
    # so there's no reason to expose the server on the LAN.
    site = web.TCPSite(_runner, "127.0.0.1", config.webhook_port)
    await site.start()
    log.info("Webhook server listening on port %d", config.webhook_port)

    # Register the webhook URL with Telegram's API if in webhook mode. This must
    # come after the server is listening so the endpoint is ready before Telegram
    # starts pushing. allowed_updates limits which update types Telegram sends -
    # Kai only handles messages and callback queries (inline keyboard taps).
    #
    # Retry with backoff because Telegram's API can time out transiently,
    # especially after a period of downtime when queued updates are flushing.
    # Without retries, a single timeout kills the whole startup and launchd
    # eventually gives up restarting.
    if config.telegram_webhook_url:
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

    Called during shutdown from main.py's finally block. In webhook mode,
    deregisters the webhook with Telegram first (so Telegram stops sending
    updates to an endpoint that's about to disappear). In polling mode,
    skips the delete_webhook call since no webhook was registered.

    The delete_webhook call is wrapped in try/except because it's not critical -
    if the network is down at shutdown time, Telegram will just overwrite the
    stale webhook on the next set_webhook call at startup.
    """
    global _app, _runner, _webhook_registered, _health_monitor_task
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
    if _runner:
        await _runner.cleanup()
        log.info("Webhook server stopped")
    _runner = None
    _app = None


def is_running() -> bool:
    """True if the webhook server is currently running."""
    return _runner is not None


def add_allowed_chat_id(chat_id: int) -> None:
    """
    Add a chat_id to the legacy live notification-destination registry.

    Called by bot.py when /github notify sets a new notification
    destination. This registry no longer grants Telegram or internal API
    authority; those identities come from users.yaml and API credentials.
    """
    if _app is not None:
        allowed = _app.get(ALLOWED_USER_IDS_KEY)
        if allowed is not None:
            allowed.add(chat_id)


def remove_allowed_chat_id(chat_id: int) -> None:
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
        allowed = _app.get(ALLOWED_USER_IDS_KEY)
        if allowed is None:
            return
        # Never remove a chat_id that belongs to an actual user.
        # user_configs keys are telegram_ids of real users.
        config = _app.get(CONFIG_KEY)
        if config and chat_id in config.user_configs:
            return
        allowed.discard(chat_id)
