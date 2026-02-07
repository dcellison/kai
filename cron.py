from __future__ import annotations

import json
import logging
from datetime import datetime, time as dt_time, timezone
from pathlib import Path

from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes

import sessions
from locks import get_lock

log = logging.getLogger(__name__)

CRON_DIR = Path(__file__).parent / "workspace" / ".cron"

# Protocol markers for conditional auto-remove jobs.
# Claude is instructed to begin its response with one of these markers.
# Matching is case-insensitive and checks the start of any line.
_CONDITION_MET_PREFIX = "CONDITION_MET:"
_CONDITION_NOT_MET_PREFIX = "CONDITION_NOT_MET"


def _ensure_utc(dt: datetime) -> datetime:
    """Attach UTC timezone if the datetime is naive."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


async def init_jobs(app: Application) -> None:
    """Load all active jobs from DB and register them with the JobQueue."""
    CRON_DIR.mkdir(parents=True, exist_ok=True)
    jobs = await sessions.get_all_active_jobs()
    now = datetime.now(timezone.utc)
    for job in jobs:
        schedule = json.loads(job["schedule_data"])
        # Skip expired one-shot jobs
        if job["schedule_type"] == "once":
            run_at = _ensure_utc(datetime.fromisoformat(schedule["run_at"]))
            if run_at <= now:
                await sessions.deactivate_job(job["id"])
                log.info("Skipped expired one-shot job %d: %s", job["id"], job["name"])
                continue
        _register_job(app, job)
    log.info("Loaded %d active jobs", len(jobs))


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
        parts = schedule["time"].split(":")
        hour, minute = int(parts[0]), int(parts[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            log.error("Invalid time %s for job %d, skipping", schedule["time"], job["id"])
            return
        t = dt_time(hour, minute, tzinfo=timezone.utc)
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
        try:
            await context.bot.send_message(chat_id=chat_id, text=prompt)
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

        accumulated = ""
        final_response = None
        async for event in claude.send(prompt):
            if event.done:
                final_response = event.response
                break
            accumulated = event.text_so_far

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
            after_marker = lines[0].strip()[len(_CONDITION_MET_PREFIX):].strip()
            rest = lines[1].strip() if len(lines) > 1 else ""
            clean_text = f"{after_marker}\n{rest}".strip() if after_marker else rest
            msg = f"[Job: {data['name']}]\n{clean_text}" if clean_text else f"[Job: {data['name']}] Condition met."
            try:
                await context.bot.send_message(chat_id=chat_id, text=msg)
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
                await context.bot.send_message(chat_id=chat_id, text=msg)
            except Exception:
                log.exception("Failed to send job %d result", job_id)


async def process_cron_files(app: Application, chat_id: int) -> list[dict]:
    """Check for new .cron/*.json files, register them, and return created jobs."""
    created = []
    if not CRON_DIR.exists():
        return created

    for f in CRON_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            job_id = await sessions.create_job(
                chat_id=chat_id,
                name=data["name"],
                job_type=data.get("job_type", "reminder"),
                prompt=data["prompt"],
                schedule_type=data["schedule_type"],
                schedule_data=json.dumps(data["schedule_data"]),
                auto_remove=data.get("auto_remove", False),
            )
            job = {
                "id": job_id,
                "chat_id": chat_id,
                "name": data["name"],
                "job_type": data.get("job_type", "reminder"),
                "prompt": data["prompt"],
                "schedule_type": data["schedule_type"],
                "schedule_data": json.dumps(data["schedule_data"]),
                "auto_remove": data.get("auto_remove", False),
            }
            _register_job(app, job)
            created.append(job)
            log.info("Registered new job %d '%s' from file %s", job_id, data["name"], f.name)
        except Exception:
            log.exception("Failed to process cron file %s", f.name)
        finally:
            f.unlink(missing_ok=True)

    return created
