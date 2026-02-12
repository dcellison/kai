"""
Scheduled job execution engine using APScheduler.

Provides functionality to:
1. Load active jobs from the database and register them with APScheduler at startup
2. Register new jobs on-the-fly when created via the scheduling API
3. Execute jobs when their triggers fire (reminders and Claude-processed tasks)
4. Handle conditional auto-remove jobs with CONDITION_MET/CONDITION_NOT_MET markers

Job types:
    - **reminder** — Sends the prompt text directly to Telegram as a message.
    - **claude** — Sends the prompt through the persistent Claude process and
      delivers Claude's response to Telegram. Supports auto-remove mode where
      the job deactivates itself when Claude indicates a condition is met.

Schedule types:
    - **once** — Fires at a specific ISO datetime, then deactivates.
    - **daily** — Fires at one or more UTC times every day ({"times": ["HH:MM", ...]}).
    - **interval** — Fires every N seconds ({"seconds": N}).

The APScheduler JobQueue is provided by python-telegram-bot and runs in the
same event loop as the Telegram bot.
"""

import json
import logging
from datetime import UTC, datetime
from datetime import time as dt_time

from telegram.constants import ChatAction
from telegram.error import Forbidden
from telegram.ext import Application, ContextTypes

from kai import sessions
from kai.history import log_message
from kai.locks import get_lock

log = logging.getLogger(__name__)

# Protocol markers for conditional auto-remove jobs.
# Claude is instructed to begin its response with one of these markers.
# Matching is case-insensitive and checks the start of the first non-empty line.
_CONDITION_MET_PREFIX = "CONDITION_MET:"
_CONDITION_NOT_MET_PREFIX = "CONDITION_NOT_MET"


# ── Job registration ─────────────────────────────────────────────────


def _ensure_utc(dt: datetime) -> datetime:
    """
    Attach UTC timezone if the datetime is naive.

    APScheduler requires timezone-aware datetimes. SQLite and ISO format
    strings may produce naive datetimes, so this normalizes them.

    Args:
        dt: A datetime that may or may not have timezone info.

    Returns:
        The same datetime with UTC attached if it was naive, unchanged otherwise.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


async def init_jobs(app: Application) -> None:
    """
    Load all active jobs from the database and register them with APScheduler.

    Called once at startup from main.py after the database is initialized.
    Expired one-shot jobs are automatically deactivated rather than registered.

    Args:
        app: The Telegram Application instance (provides the job queue).
    """
    await _register_new_jobs(app)


async def _register_new_jobs(app: Application) -> int:
    """
    Find active DB jobs not yet in the scheduler and register them.

    Skips jobs that are already registered (by checking APScheduler's job names)
    and deactivates one-shot jobs whose run_at time has already passed.

    Args:
        app: The Telegram Application instance.

    Returns:
        The number of newly registered jobs.
    """
    assert app.job_queue is not None
    jobs = await sessions.get_all_active_jobs()
    registered = {j.name for j in app.job_queue.jobs()}
    now = datetime.now(UTC)
    count = 0
    for job in jobs:
        job_name = f"cron_{job['id']}"
        # Daily jobs with multiple times get suffixed names (cron_13_0, cron_13_1)
        if any(name == job_name or name.startswith(f"{job_name}_") for name in registered):
            continue
        schedule = json.loads(job["schedule_data"])
        # Skip expired one-shot jobs rather than registering them
        if job["schedule_type"] == "once":
            run_at = _ensure_utc(datetime.fromisoformat(schedule["run_at"]))
            if run_at <= now:
                await sessions.deactivate_job(job["id"])
                log.info("Skipped expired one-shot job %d: %s", job["id"], job["name"])
                continue
        _register_job(app, job)
        count += 1
    return count


async def register_job_by_id(app: Application, job_id: int) -> bool:
    """
    Register a single job by its DB ID. Called by the scheduling API
    (webhook.py) immediately after creating a new job so it starts
    firing without waiting for a restart.

    Args:
        app: The Telegram Application instance.
        job_id: Database ID of the job to register.

    Returns:
        True if the job was found and registered, False otherwise.
    """
    job = await sessions.get_job_by_id(job_id)
    if not job:
        log.error("Job %d not found in DB", job_id)
        return False
    _register_job(app, job)
    return True


def _register_job(app: Application, job: dict) -> None:
    """
    Register a single job with the APScheduler JobQueue.

    Parses the schedule_data JSON and creates the appropriate APScheduler
    trigger (run_once, run_repeating, or run_daily). For daily jobs with
    multiple times, each time gets its own trigger with a suffixed name
    (e.g., cron_13_0, cron_13_1) while sharing the same callback data.

    Args:
        app: The Telegram Application instance.
        job: Job dict from the database (as returned by sessions.get_job_by_id).
    """
    jq = app.job_queue
    assert jq is not None
    schedule = json.loads(job["schedule_data"])
    job_name = f"cron_{job['id']}"

    # Data passed to the callback when the job fires
    callback_data = {
        "job_id": job["id"],
        "chat_id": job["chat_id"],
        "job_type": job["job_type"],
        "prompt": job["prompt"],
        "auto_remove": job["auto_remove"],
        "name": job["name"],
        "schedule_type": job["schedule_type"],
    }

    if job["schedule_type"] == "once":
        run_at = _ensure_utc(datetime.fromisoformat(schedule["run_at"]))
        jq.run_once(_job_callback, when=run_at, name=job_name, data=callback_data)
        log.info("Scheduled one-shot job %d '%s' at %s", job["id"], job["name"], run_at)

    elif job["schedule_type"] == "interval":
        seconds = schedule["seconds"]
        jq.run_repeating(_job_callback, interval=seconds, name=job_name, data=callback_data)
        log.info("Scheduled repeating job %d '%s' every %ds", job["id"], job["name"], seconds)

    elif job["schedule_type"] == "daily":
        times = schedule["times"]
        for i, time_str in enumerate(times):
            # Parse "HH:MM" string into hour and minute integers
            try:
                parts = time_str.split(":")
                hour, minute = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                log.error("Invalid time %s for job %d, skipping", time_str, job["id"])
                continue
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                log.error("Invalid time %s for job %d, skipping", time_str, job["id"])
                continue
            t = dt_time(hour, minute, tzinfo=UTC)
            # Suffix name for multi-time daily jobs to avoid APScheduler name collisions
            name_suffix = f"{job_name}_{i}" if len(times) > 1 else job_name
            jq.run_daily(_job_callback, time=t, name=name_suffix, data=callback_data)
            log.info("Scheduled daily job %d '%s' at %s UTC", job["id"], job["name"], time_str)

    else:
        log.warning("Unknown schedule type '%s' for job %d, skipping", job["schedule_type"], job["id"])


# ── Job execution ────────────────────────────────────────────────────


async def _job_callback(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Called by APScheduler when a scheduled job fires.

    Routes to the appropriate handler based on job_type:
    - "reminder" jobs send the prompt text directly to Telegram.
    - "claude" jobs send the prompt through the persistent Claude process,
      then handle the response (including conditional auto-remove logic).

    For Claude jobs with auto_remove=True, the response is inspected for
    protocol markers:
    - CONDITION_MET: <message> — delivers the message and deactivates the job.
    - CONDITION_NOT_MET — silently continues without notifying the user.
    - No marker — delivers the full response (non-auto-remove jobs always do this).

    Args:
        context: Telegram callback context containing job data and bot reference.
    """
    job = context.job
    assert job is not None
    assert isinstance(job.data, dict)
    data: dict = job.data
    chat_id = data["chat_id"]
    job_type = data["job_type"]
    prompt = data["prompt"]
    auto_remove = data["auto_remove"]
    job_id = data["job_id"]

    log.info("Job %d '%s' fired (type=%s)", job_id, data["name"], job_type)

    # ── Reminder jobs: send prompt text directly ──
    if job_type == "reminder":
        # Strip stray backslash escapes (e.g. \! from bash double-quoting in curl)
        prompt = prompt.replace("\\!", "!").replace("\\.", ".").replace("\\?", "?")
        try:
            log_message(direction="assistant", chat_id=chat_id, text=f"[Reminder: {data['name']}] {prompt}")
            await context.bot.send_message(chat_id=chat_id, text=prompt)
        except Forbidden:
            log.warning("Job %d: chat %d is gone, deactivating", job_id, chat_id)
            await sessions.deactivate_job(job_id)
            job.schedule_removal()
            return
        except Exception:
            log.exception("Failed to send reminder for job %d", job_id)
        # One-shot reminders auto-deactivate after firing
        if data["schedule_type"] == "once":
            await sessions.deactivate_job(job_id)
        return

    # ── Claude jobs: send prompt through the Claude process ──
    claude = context.bot_data.get("claude")
    if not claude:
        log.error("No Claude process available for job %d", job_id)
        return

    async with get_lock(chat_id):
        # Show typing indicator while Claude processes the prompt
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception:
            pass

        # Send prompt to Claude and collect the final response (no streaming to Telegram)
        try:
            final_response = None
            async for event in claude.send(prompt):
                if event.done:
                    final_response = event.response
                    break
        except Exception:
            log.exception("Job %d crashed during Claude interaction", job_id)
            return

        if final_response is None or not final_response.success:
            error = final_response.error if final_response else "No response"
            log.error("Job %d Claude error: %s", job_id, error)
            return

        response_text = final_response.text

        # Check first non-empty line for condition markers (auto-remove jobs only)
        first_line = response_text.strip().split("\n", 1)[0].strip().upper()

        if auto_remove and first_line.startswith(_CONDITION_MET_PREFIX.upper()):
            # Condition met — extract the message after the marker, send it, and deactivate
            lines = response_text.strip().split("\n", 1)
            after_marker = lines[0].strip()[len(_CONDITION_MET_PREFIX) :].strip()
            rest = lines[1].strip() if len(lines) > 1 else ""
            clean_text = f"{after_marker}\n{rest}".strip() if after_marker else rest
            msg = f"[Job: {data['name']}]\n{clean_text}" if clean_text else f"[Job: {data['name']}] Condition met."
            try:
                log_message(direction="assistant", chat_id=chat_id, text=msg)
                await context.bot.send_message(chat_id=chat_id, text=msg)
            except Forbidden:
                log.warning("Job %d: chat %d is gone, deactivating", job_id, chat_id)
            except Exception:
                log.exception("Failed to send job %d result", job_id)
            await sessions.deactivate_job(job_id)
            job.schedule_removal()
            log.info("Job %d condition met, deactivated", job_id)

        elif auto_remove and first_line.startswith(_CONDITION_NOT_MET_PREFIX.upper()):
            # Condition not met — silently continue, no notification to user
            log.info("Job %d condition not met, continuing", job_id)

        else:
            # Non-conditional or non-auto-remove: always deliver the response
            msg = f"[Job: {data['name']}]\n{response_text}"
            try:
                log_message(direction="assistant", chat_id=chat_id, text=msg)
                await context.bot.send_message(chat_id=chat_id, text=msg)
            except Forbidden:
                log.warning("Job %d: chat %d is gone, deactivating", job_id, chat_id)
                await sessions.deactivate_job(job_id)
                job.schedule_removal()
            except Exception:
                log.exception("Failed to send job %d result", job_id)
