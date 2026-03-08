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
7. Send files from the workspace to the Telegram chat (file exchange API)

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
    - /api/send-file        - Send a file from the filesystem to the Telegram chat

The Telegram webhook route uses its own secret (TELEGRAM_WEBHOOK_SECRET) and is
only registered in webhook mode. All other webhook/API endpoints require
WEBHOOK_SECRET. When WEBHOOK_SECRET is unset, only /health is active (plus
/webhook/telegram if in webhook mode).

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
from pathlib import Path

from aiohttp import web
from telegram import Update

from kai import cron, services, sessions
from kai.config import IMAGE_EXTENSIONS

log = logging.getLogger(__name__)

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


def _require_secret(handler):
    """Decorator that validates the X-Webhook-Secret header before
    calling the route handler. Returns 401 on mismatch."""

    @functools.wraps(handler)
    async def wrapper(request: web.Request) -> web.Response:
        secret = request.app["webhook_secret"]
        provided = request.headers.get("X-Webhook-Secret", "")
        if not hmac.compare_digest(provided, secret):
            log.warning("Auth failure on %s from %s", request.path, request.remote)
            return web.Response(status=401, text="Invalid secret")
        return await handler(request)

    return wrapper


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

    Always returns 200 on valid-secret requests, even on errors. Telegram retries
    on non-200 responses, so surfacing internal errors as HTTP errors would cause
    an infinite retry loop. Errors are logged instead.
    """
    secret = request.app["telegram_webhook_secret"]
    provided = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not hmac.compare_digest(provided, secret):
        log.warning("Telegram update: invalid secret")
        return web.Response(status=401, text="Invalid secret")

    try:
        data = await request.json()
    except json.JSONDecodeError:
        log.warning("Telegram update: malformed JSON")
        return web.Response(status=200)

    telegram_app = request.app["telegram_app"]
    bot = request.app["telegram_bot"]

    try:
        update = Update.de_json(data, bot)
        if update:
            # Fire-and-forget: return 200 to Telegram immediately while
            # processing continues in the background. Without this, Claude's
            # response time would exceed Telegram's webhook timeout.
            task = asyncio.create_task(telegram_app.process_update(update))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
    except Exception:
        log.exception("Error processing Telegram update")

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

    return web.json_response({"status": "ok"})


@_require_secret
async def _handle_generic(request: web.Request) -> web.Response:
    """
    Handle generic webhook notifications from any source.

    Extracts a "message" field from the JSON payload (or dumps the full
    payload) and forwards it to the Telegram chat. Truncates to Telegram's
    4096-char limit.
    """
    bot = request.app["telegram_bot"]
    chat_id = request.app["chat_id"]

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return web.Response(status=400, text="Invalid JSON")

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


@_require_secret
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
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

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


@_require_secret
async def _handle_get_jobs(request: web.Request) -> web.Response:
    """
    List all active jobs for the configured chat.

    Used by the inner Claude to check what jobs are currently scheduled
    without needing to parse Telegram bot command output.
    """

    chat_id = request.app["chat_id"]
    jobs = await sessions.get_jobs(chat_id)
    return web.json_response(jobs)


@_require_secret
async def _handle_get_job(request: web.Request) -> web.Response:
    """
    Get a single job by its database ID.

    Returns the full job record as JSON, or 404 if not found.
    """

    try:
        job_id = int(request.match_info["id"])
    except ValueError:
        return web.json_response({"error": "Invalid job ID"}, status=400)

    job = await sessions.get_job_by_id(job_id)
    if not job:
        return web.json_response({"error": "Job not found"}, status=404)
    return web.json_response(job)


@_require_secret
async def _handle_delete_job(request: web.Request) -> web.Response:
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

    deleted = await sessions.delete_job(job_id)
    if not deleted:
        return web.json_response({"error": "Job not found"}, status=404)

    # Remove from APScheduler's in-memory queue. Daily jobs with multiple
    # times get suffixed names (e.g., cron_19_0, cron_19_1), so we match
    # both the exact name and the prefix pattern.
    telegram_app = request.app["telegram_app"]
    jq = telegram_app.job_queue
    assert jq is not None
    prefix = f"cron_{job_id}"
    for j in jq.jobs():
        if j.name == prefix or (j.name and j.name.startswith(f"{prefix}_")):
            j.schedule_removal()

    log.info("Deleted job %d via API", job_id)
    return web.json_response({"deleted": job_id})


@_require_secret
async def _handle_update_job(request: web.Request) -> web.Response:
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

    # Validate schedule_type if provided
    new_schedule_type = payload.get("schedule_type")
    if new_schedule_type and new_schedule_type not in _VALID_SCHEDULE_TYPES:
        return web.json_response(
            {"error": f"schedule_type must be one of: {', '.join(_VALID_SCHEDULE_TYPES)}"},
            status=400,
        )

    # Serialize schedule_data if provided as a dict (allow either string or dict)
    schedule_data = payload.get("schedule_data")
    if isinstance(schedule_data, dict):
        schedule_data = json.dumps(schedule_data)

    updated = await sessions.update_job(
        job_id,
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
        telegram_app = request.app["telegram_app"]
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


@_require_secret
async def _handle_service_call(request: web.Request) -> web.Response:
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

    # Parse optional JSON body with request parameters
    body = None
    params = None
    path_suffix = ""
    try:
        payload = await request.json()
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


# ── File exchange ────────────────────────────────────────────────────


@_require_secret
async def _handle_send_file(request: web.Request) -> web.Response:
    """
    Send a file from the filesystem to the Telegram chat.

    Called by the inner Claude process to deliver files back to the user.
    Accepts a JSON body with a required "path" field (absolute path) and
    an optional "caption". Images are sent as photos (rendered inline),
    everything else as document attachments.

    Path confinement: the resolved path must be inside the current workspace
    directory. This prevents path traversal attacks via symlinks or "../".

    Returns:
        JSON {"status": "sent", "file": "<filename>"} on success, or an
        appropriate HTTP error (400/401/403/404).
    """
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    file_path = payload.get("path")
    if not file_path:
        return web.json_response({"error": "Missing required field: path"}, status=400)

    path = Path(file_path).resolve()

    # Confine to the current workspace to prevent path traversal. Uses
    # Path.relative_to() which raises ValueError on escape, unlike string
    # prefix matching which is bypassable via symlinks.
    # Fail closed: if workspace is somehow unset, deny all file access
    # rather than allowing reads from anywhere on the filesystem.
    workspace = request.app.get("workspace")
    if not workspace:
        return web.json_response({"error": "No workspace configured"}, status=403)
    workspace_resolved = Path(workspace).resolve()
    try:
        path.relative_to(workspace_resolved)
    except ValueError:
        return web.json_response({"error": "Path outside workspace"}, status=403)

    if not path.is_file():
        return web.json_response({"error": f"File not found: {file_path}"}, status=404)

    bot = request.app["telegram_bot"]
    chat_id = request.app["chat_id"]
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


# ── Lifecycle ────────────────────────────────────────────────────────


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
    global _app, _runner, _webhook_registered

    _app = web.Application()
    _app["telegram_app"] = telegram_app
    _app["telegram_bot"] = telegram_app.bot
    _app["webhook_secret"] = config.webhook_secret

    # Use first allowed user ID as the notification target.
    # Config validation ensures allowed_user_ids is non-empty, but guard
    # against edge cases to avoid a StopIteration crash at startup.
    chat_id = next(iter(config.allowed_user_ids), None)
    if chat_id is None:
        raise SystemExit("No allowed user IDs configured; cannot start webhook server")
    _app["chat_id"] = chat_id

    # Store workspace path for send-file path confinement
    _app["workspace"] = str(config.claude_workspace)

    _app.router.add_get("/health", _handle_health)

    # Only register the Telegram webhook route in webhook mode. In polling mode,
    # there's no need for the endpoint and no secret to validate against.
    if config.telegram_webhook_url:
        _app["telegram_webhook_secret"] = config.telegram_webhook_secret
        _app.router.add_post("/webhook/telegram", _handle_telegram_update)

    if config.webhook_secret:
        _app.router.add_post("/webhook/github", _handle_github)
        _app.router.add_post("/webhook", _handle_generic)
        _app.router.add_post("/api/schedule", _handle_schedule)
        _app.router.add_get("/api/jobs", _handle_get_jobs)
        _app.router.add_get("/api/jobs/{id}", _handle_get_job)
        _app.router.add_delete("/api/jobs/{id}", _handle_delete_job)
        _app.router.add_patch("/api/jobs/{id}", _handle_update_job)
        _app.router.add_post("/api/services/{name}", _handle_service_call)
        _app.router.add_post("/api/send-file", _handle_send_file)
    else:
        log.warning("WEBHOOK_SECRET not set - webhook and scheduling endpoints disabled")

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
    if config.telegram_webhook_url:
        await telegram_app.bot.set_webhook(
            url=config.telegram_webhook_url,
            secret_token=config.telegram_webhook_secret,
            allowed_updates=["message", "callback_query"],
        )
        _webhook_registered = True
        log.info("Registered Telegram webhook: %s", config.telegram_webhook_url)


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
    global _app, _runner, _webhook_registered
    # Only deregister if we registered a webhook (i.e., webhook mode was active)
    if _webhook_registered and _app is not None:
        telegram_bot = _app.get("telegram_bot")
        if telegram_bot is not None:
            try:
                await telegram_bot.delete_webhook()
                log.info("Deregistered Telegram webhook")
            except Exception:
                log.warning("Failed to deregister Telegram webhook (will re-register on next start)")
        _webhook_registered = False
    if _runner:
        await _runner.cleanup()
        log.info("Webhook server stopped")
    _runner = None
    _app = None


def is_running() -> bool:
    """True if the webhook server is currently running."""
    return _runner is not None


def update_workspace(workspace: str) -> None:
    """
    Update the workspace path used by the send-file endpoint's confinement check.

    Called by _do_switch_workspace in bot.py whenever the user switches workspaces,
    and by main.py after startup if a non-default workspace was restored from the
    settings table. Without this, the confinement check keeps using the initial
    home workspace path set at startup, causing send-file to return 403 for any
    file in the current (switched) workspace.

    The server must already be running when this is called - callers in main.py
    should invoke this after webhook.start(), not before, since start() sets the
    path from config and would overwrite an earlier call.

    Args:
        workspace: Absolute path string of the new current workspace.
    """
    if _app is not None:
        _app["workspace"] = workspace
