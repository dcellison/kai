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
# Matching is case-insensitive and checks the start of any line.
_CONDITION_MET_PREFIX = "CONDITION_MET:"
_CONDITION_NOT_MET_PREFIX = "CONDITION_NOT_MET"


def _ensure_utc(dt: datetime) -> datetime:
    """Attach UTC timezone if the datetime is naive."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


async def init_jobs(app: Application) -> None:
    """Load all active jobs from DB and register them with the scheduler."""
    await _register_new_jobs(app)


async def _register_new_jobs(app: Application) -> int:
    """Find active DB jobs not yet in the scheduler and register them.

    Returns the number of newly registered jobs.
    """
    jobs = await sessions.get_all_active_jobs()
    registered = {j.name for j in app.job_queue.jobs()}
    now = datetime.now(UTC)
    count = 0
    for job in jobs:
        job_name = f"cron_{job['id']}"
        if job_name in registered:
            continue
        schedule = json.loads(job["schedule_data"])
        # Skip expired one-shot jobs
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
    """Register a single job by its DB ID. Called by the scheduling API."""
    job = await sessions.get_job_by_id(job_id)
    if not job:
        log.error("Job %d not found in DB", job_id)
        return False
    _register_job(app, job)
    return True


def _register_job(app: Application, job: dict) -> None:
    """Register a single job with the APScheduler JobQueue."""
    jq = app.job_queue
    schedule = json.loads(job["schedule_data"])
    job_name = f"cron_{job['id']}"
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
        try:
            parts = schedule["time"].split(":")
            hour, minute = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            log.error("Invalid time %s for job %d, skipping", schedule["time"], job["id"])
            return
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            log.error("Invalid time %s for job %d, skipping", schedule["time"], job["id"])
            return
        t = dt_time(hour, minute, tzinfo=UTC)
        jq.run_daily(_job_callback, time=t, name=job_name, data=callback_data)
        log.info("Scheduled daily job %d '%s' at %s UTC", job["id"], job["name"], schedule["time"])

    else:
        log.warning("Unknown schedule type '%s' for job %d, skipping", job["schedule_type"], job["id"])


async def _job_callback(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Called by APScheduler when a job fires."""
    data = context.job.data
    chat_id = data["chat_id"]
    job_type = data["job_type"]
    prompt = data["prompt"]
    auto_remove = data["auto_remove"]
    job_id = data["job_id"]

    log.info("Job %d '%s' fired (type=%s)", job_id, data["name"], job_type)

    if job_type == "reminder":
        # Strip stray backslash escapes (e.g. \! from bash double-quoting)
        prompt = prompt.replace("\\!", "!").replace("\\.", ".").replace("\\?", "?")
        try:
            log_message(direction="assistant", chat_id=chat_id, text=f"[Reminder: {data['name']}] {prompt}")
            await context.bot.send_message(chat_id=chat_id, text=prompt)
        except Forbidden:
            log.warning("Job %d: chat %d is gone, deactivating", job_id, chat_id)
            await sessions.deactivate_job(job_id)
            context.job.schedule_removal()
            return
        except Exception:
            log.exception("Failed to send reminder for job %d", job_id)
        # One-shot reminders auto-deactivate
        if data["schedule_type"] == "once":
            await sessions.deactivate_job(job_id)
        return

    # Claude-type job: send prompt through the Claude process
    claude = context.bot_data.get("claude")
    if not claude:
        log.error("No Claude process available for job %d", job_id)
        return

    async with get_lock(chat_id):
        try:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        except Exception:
            pass

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
        # Check first non-empty line for condition markers
        first_line = response_text.strip().split("\n", 1)[0].strip().upper()

        if auto_remove and first_line.startswith(_CONDITION_MET_PREFIX.upper()):
            # Condition met — send the rest (after the marker line) and deactivate
            lines = response_text.strip().split("\n", 1)
            # Text after the marker on the same line, plus any remaining lines
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
            context.job.schedule_removal()
            log.info("Job %d condition met, deactivated", job_id)

        elif auto_remove and first_line.startswith(_CONDITION_NOT_MET_PREFIX.upper()):
            # Silently continue
            log.info("Job %d condition not met, continuing", job_id)

        else:
            # Always send response for non-auto-remove jobs
            msg = f"[Job: {data['name']}]\n{response_text}"
            try:
                log_message(direction="assistant", chat_id=chat_id, text=msg)
                await context.bot.send_message(chat_id=chat_id, text=msg)
            except Forbidden:
                log.warning("Job %d: chat %d is gone, deactivating", job_id, chat_id)
                await sessions.deactivate_job(job_id)
                context.job.schedule_removal()
            except Exception:
                log.exception("Failed to send job %d result", job_id)
