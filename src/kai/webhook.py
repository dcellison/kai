"""
Webhook HTTP server for receiving external notifications and scheduling jobs.

Provides functionality to:
1. Receive and validate GitHub webhook events (push, PR, issues, comments, reviews)
2. Accept generic webhook notifications from any source
3. Expose a scheduling API for creating cron-style jobs via HTTP
4. Expose a jobs query API for listing and fetching scheduled jobs

The server runs on aiohttp alongside the Telegram bot in the same event loop.
Routes are organized into three groups:
    - /webhook/github   — GitHub events with HMAC-SHA256 signature validation
    - /webhook          — Generic webhooks with shared-secret auth
    - /api/schedule     — Job creation API (used by inner Claude via curl)
    - /api/jobs         — Job listing and detail API

All webhook/API endpoints (except /health) require WEBHOOK_SECRET to be set.
When unset, only the health check endpoint is registered. This allows the
server to start cleanly in development without exposing unauthenticated routes.

GitHub events are formatted into human-readable Markdown messages and sent
to the configured Telegram chat. The formatter pattern (dispatch dict mapping
event type → formatter function) makes it easy to add new event types.
"""

import hashlib
import hmac
import json
import logging
import re

from aiohttp import web

from kai import cron, sessions

log = logging.getLogger(__name__)

# Module-level server state, managed by start() and stop()
_app: web.Application | None = None
_runner: web.AppRunner | None = None


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
        secret: The shared webhook secret configured in GitHub and .env.
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
    """Health check endpoint. Returns {"status": "ok"} for uptime monitoring."""
    return web.json_response({"status": "ok"})


async def _handle_github(request: web.Request) -> web.Response:
    """
    Handle incoming GitHub webhook events.

    Validates the HMAC-SHA256 signature, parses the event payload, dispatches
    to the appropriate formatter, and sends the formatted message to Telegram.
    Falls back to plain text if Markdown parsing fails.

    Supported events: push, pull_request, issues, issue_comment, pull_request_review.
    Unsupported events are silently acknowledged with {"msg": "ignored"}.
    """
    secret = request.app["webhook_secret"]
    bot = request.app["telegram_bot"]
    chat_id = request.app["chat_id"]

    body = await request.read()

    # Validate HMAC-SHA256 signature from GitHub
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not _verify_github_signature(secret, body, signature):
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

    # Look up the formatter for this event type
    formatter = _GITHUB_FORMATTERS.get(event_type)
    if not formatter:
        return web.json_response({"msg": "ignored", "event": event_type})

    message = formatter(payload)
    if not message:
        return web.json_response({"msg": "ignored", "event": event_type})

    # Send to Telegram with Markdown, falling back to plain text on parse failure
    try:
        await bot.send_message(chat_id, message, parse_mode="Markdown")
    except Exception:
        try:
            await bot.send_message(chat_id, _strip_markdown(message))
        except Exception:
            log.exception("Failed to send GitHub notification")
            return web.json_response({"msg": "error"})
    log.info("Sent GitHub %s notification to chat %d", event_type, chat_id)

    return web.json_response({"msg": "ok"})


async def _handle_generic(request: web.Request) -> web.Response:
    """
    Handle generic webhook notifications from any source.

    Validates the shared secret via X-Webhook-Secret header, extracts a
    "message" field from the JSON payload (or dumps the full payload), and
    forwards it to the Telegram chat. Truncates to Telegram's 4096-char limit.
    """
    secret = request.app["webhook_secret"]
    bot = request.app["telegram_bot"]
    chat_id = request.app["chat_id"]

    # Validate shared secret header (constant-time comparison)
    provided = request.headers.get("X-Webhook-Secret", "")
    if not hmac.compare_digest(provided, secret):
        log.warning("Generic webhook: invalid secret")
        return web.Response(status=401, text="Invalid secret")

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return web.Response(status=400, text="Invalid JSON")

    # Use the "message" field if present, otherwise dump the full JSON
    text = payload.get("message") or json.dumps(payload, indent=2)
    if len(text) > 4096:
        text = text[:4093] + "..."

    try:
        await bot.send_message(chat_id, text)
    except Exception:
        log.exception("Failed to send generic webhook notification")

    return web.json_response({"msg": "ok"})


# ── Scheduling API ───────────────────────────────────────────────────

# Valid schedule types accepted by the scheduling API
_VALID_SCHEDULE_TYPES = ("once", "daily", "interval")


async def _handle_schedule(request: web.Request) -> web.Response:
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
    secret = request.app["webhook_secret"]

    # Validate shared secret
    provided = request.headers.get("X-Webhook-Secret", "")
    if not hmac.compare_digest(provided, secret):
        return web.Response(status=401, text="Invalid secret")

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return web.Response(status=400, text="Invalid JSON")

    # Extract and validate required fields
    name = payload.get("name")
    prompt = payload.get("prompt")
    schedule_type = payload.get("schedule_type")
    schedule_data = payload.get("schedule_data")

    if not all([name, prompt, schedule_type, schedule_data]):
        return web.json_response(
            {"error": "Missing required fields: name, prompt, schedule_type, schedule_data"},
            status=400,
        )

    if schedule_type not in _VALID_SCHEDULE_TYPES:
        return web.json_response(
            {"error": f"schedule_type must be one of: {', '.join(_VALID_SCHEDULE_TYPES)}"},
            status=400,
        )

    # Optional fields with defaults
    job_type = payload.get("job_type", "reminder")
    auto_remove = payload.get("auto_remove", False)
    notify_on_check = payload.get("notify_on_check", False)
    chat_id = request.app["chat_id"]

    # schedule_data can arrive as a JSON object or a pre-serialized string
    if isinstance(schedule_data, dict):
        schedule_data_str = json.dumps(schedule_data)
    else:
        schedule_data_str = schedule_data

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
    telegram_app = request.app["telegram_app"]
    await cron.register_job_by_id(telegram_app, job_id)

    log.info("Scheduled job %d '%s' via API (%s)", job_id, name, schedule_type)
    return web.json_response({"job_id": job_id, "name": name})


# ── Jobs API ─────────────────────────────────────────────────────────


async def _handle_get_jobs(request: web.Request) -> web.Response:
    """
    List all active jobs for the configured chat.

    Used by the inner Claude to check what jobs are currently scheduled
    without needing to parse Telegram bot command output.
    """
    secret = request.app["webhook_secret"]

    provided = request.headers.get("X-Webhook-Secret", "")
    if not hmac.compare_digest(provided, secret):
        return web.Response(status=401, text="Invalid secret")

    chat_id = request.app["chat_id"]
    jobs = await sessions.get_jobs(chat_id)
    return web.json_response(jobs)


async def _handle_get_job(request: web.Request) -> web.Response:
    """
    Get a single job by its database ID.

    Returns the full job record as JSON, or 404 if not found.
    """
    secret = request.app["webhook_secret"]

    provided = request.headers.get("X-Webhook-Secret", "")
    if not hmac.compare_digest(provided, secret):
        return web.Response(status=401, text="Invalid secret")

    try:
        job_id = int(request.match_info["id"])
    except ValueError:
        return web.json_response({"error": "Invalid job ID"}, status=400)

    job = await sessions.get_job_by_id(job_id)
    if not job:
        return web.json_response({"error": "Job not found"}, status=404)
    return web.json_response(job)


# ── Lifecycle ────────────────────────────────────────────────────────


async def start(telegram_app, config) -> None:
    """
    Start the webhook HTTP server on the configured port.

    Sets up all routes and stores references to the Telegram app/bot and
    webhook secret in the aiohttp app dict so route handlers can access them.
    The first allowed user ID is used as the notification target chat.

    Webhook and scheduling routes are only registered if WEBHOOK_SECRET is
    set — otherwise only the /health endpoint is available.

    Args:
        telegram_app: The python-telegram-bot Application instance.
        config: The application Config instance.
    """
    global _app, _runner

    _app = web.Application()
    _app["telegram_app"] = telegram_app
    _app["telegram_bot"] = telegram_app.bot
    _app["webhook_secret"] = config.webhook_secret

    # Use first allowed user ID as the notification target
    _app["chat_id"] = next(iter(config.allowed_user_ids))

    _app.router.add_get("/health", _handle_health)

    if config.webhook_secret:
        _app.router.add_post("/webhook/github", _handle_github)
        _app.router.add_post("/webhook", _handle_generic)
        _app.router.add_post("/api/schedule", _handle_schedule)
        _app.router.add_get("/api/jobs", _handle_get_jobs)
        _app.router.add_get("/api/jobs/{id}", _handle_get_job)
    else:
        log.warning("WEBHOOK_SECRET not set — webhook and scheduling endpoints disabled")

    _runner = web.AppRunner(_app, access_log=None)
    await _runner.setup()
    site = web.TCPSite(_runner, "0.0.0.0", config.webhook_port)
    await site.start()
    log.info("Webhook server listening on port %d", config.webhook_port)


async def stop() -> None:
    """
    Stop the webhook server and clean up resources.

    Called during shutdown from main.py's finally block.
    """
    global _app, _runner
    if _runner:
        await _runner.cleanup()
        log.info("Webhook server stopped")
    _runner = None
    _app = None


def is_running() -> bool:
    """True if the webhook server is currently running."""
    return _runner is not None
