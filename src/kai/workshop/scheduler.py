"""Core-owned scheduling for canonical Workshop reminders and agent runs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from kai.backend import AgentResponse
from kai.job_types import JOB_TYPE_AGENT, JOB_TYPE_REMINDER, normalize_job_type
from kai.workshop.compatibility_state import WorkshopCompatibilityStateWriter
from kai.workshop.conversation_commands import ConversationCommandDisposition
from kai.workshop.domain import (
    CanonicalMemoryProvenance,
    ChannelId,
    MessageId,
    PrincipalId,
    RunId,
    RuntimeProfileId,
)
from kai.workshop.execution_coordinator import (
    CanonicalExecutionDisposition,
    CanonicalSuccessOutcome,
)
from kai.workshop.inbound import ScheduledInboundMessage
from kai.workshop.private_text_execution import WorkshopPrivateTextExecutionService
from kai.workshop.scheduled_notifications import (
    ScheduledReminder,
    WorkshopScheduledReminderRecorder,
)
from kai.workshop.store import WorkshopEventStore

log = logging.getLogger(__name__)

_CONDITION_MET_PREFIX = "CONDITION_MET:"
_CONDITION_NOT_MET_PREFIX = "CONDITION_NOT_MET"
_RECONCILE_SECONDS = 5.0
_FIRING_DRAIN_SECONDS = 30.0


class WorkshopSchedulerState(StrEnum):
    NEW = "new"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class WorkshopSchedulerReadiness:
    state: WorkshopSchedulerState
    active_jobs: int
    active_firings: int

    @property
    def ready(self) -> bool:
        return self.state == WorkshopSchedulerState.READY


@dataclass(frozen=True, slots=True)
class _CanonicalJob:
    job_id: int
    name: str
    job_type: str
    prompt: str
    schedule_type: str
    schedule_data: dict[str, Any]
    created_at: datetime
    auto_remove: bool
    notify_on_check: bool
    principal_id: PrincipalId
    channel_id: ChannelId
    runtime_profile_id: RuntimeProfileId


@dataclass(frozen=True, slots=True)
class _FiringOutcome:
    status: str
    message_id: MessageId | None = None
    run_id: RunId | None = None
    terminal_code: str | None = None


class WorkshopCanonicalScheduler:
    """Own schedule timing and route every firing through canonical services."""

    def __init__(
        self,
        store: WorkshopEventStore,
        execution: WorkshopPrivateTextExecutionService,
        compatibility_state: WorkshopCompatibilityStateWriter,
    ) -> None:
        self._store = store
        self._execution = execution
        self._compatibility_state = compatibility_state
        self._reminders = WorkshopScheduledReminderRecorder(store)
        self._state = WorkshopSchedulerState.NEW
        self._stop_event = asyncio.Event()
        self._failure_event = asyncio.Event()
        self._failure: BaseException | None = None
        self._reconcile_task: asyncio.Task[None] | None = None
        self._scheduler = AsyncIOScheduler(timezone=UTC)
        self._registered_jobs: dict[int, tuple[str, ...]] = {}
        self._firing_tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    @classmethod
    async def open_and_start(
        cls,
        database_path: Path,
        execution: WorkshopPrivateTextExecutionService,
        compatibility_state: WorkshopCompatibilityStateWriter,
    ) -> WorkshopCanonicalScheduler:
        store = await WorkshopEventStore.open(database_path)
        service = cls(store, execution, compatibility_state)
        try:
            await service._recover_interrupted_firings()
            service._scheduler.start(paused=True)
            await service._reconcile()
            service._scheduler.resume()
            service._reconcile_task = asyncio.create_task(
                service._reconcile_loop(),
                name="kai-workshop-canonical-scheduler",
            )
            service._state = WorkshopSchedulerState.READY
            return service
        except BaseException:
            if service._scheduler.running:
                service._scheduler.shutdown(wait=False)
            await store.close()
            raise

    @property
    def readiness(self) -> WorkshopSchedulerReadiness:
        return WorkshopSchedulerReadiness(
            self._state,
            len(self._registered_jobs),
            len(self._firing_tasks),
        )

    async def register_job(self, job_id: int) -> bool:
        """Register or replace one active job after a committed mutation."""
        job = await self._load_job(job_id)
        async with self._lock:
            self._remove_registered_job(job_id)
            if job is None:
                return False
            self._register_loaded_job(job)
            return True

    async def remove_job(self, job_id: int) -> None:
        async with self._lock:
            self._remove_registered_job(job_id)

    async def wait(self) -> None:
        task = self._reconcile_task
        if task is None:
            raise RuntimeError("Workshop scheduler is not started")
        failure_wait = asyncio.create_task(self._failure_event.wait())
        try:
            done, _ = await asyncio.wait(
                (task, failure_wait),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if failure_wait in done:
                assert self._failure is not None
                raise self._failure
            await asyncio.shield(task)
        finally:
            failure_wait.cancel()
            await asyncio.gather(failure_wait, return_exceptions=True)

    async def stop(self) -> None:
        if self._state in {WorkshopSchedulerState.STOPPED, WorkshopSchedulerState.NEW}:
            self._state = WorkshopSchedulerState.STOPPED
            return
        self._state = WorkshopSchedulerState.STOPPING
        self._stop_event.set()
        reconcile = self._reconcile_task
        if reconcile is not None:
            await asyncio.shield(reconcile)
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        async with self._lock:
            self._registered_jobs.clear()
        if self._firing_tasks:
            firing_tasks = tuple(self._firing_tasks.values())
            _, pending = await asyncio.wait(
                firing_tasks,
                timeout=_FIRING_DRAIN_SECONDS,
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        await self._store.close()
        self._reconcile_task = None
        self._state = WorkshopSchedulerState.STOPPED

    async def _reconcile_loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=_RECONCILE_SECONDS)
                return
            except TimeoutError:
                pass
            try:
                await self._reconcile()
            except Exception as exc:
                self._record_failure(exc)
                log.exception("Workshop scheduler reconciliation failed")
                return

    async def _reconcile(self) -> None:
        if not await self._jobs_table_exists():
            return
        async with self._store.connection.execute("SELECT id FROM jobs WHERE active = 1 ORDER BY id") as cursor:
            active_ids = {int(row[0]) for row in await cursor.fetchall()}
        invalid_ids = [job_id for job_id in active_ids if await self._load_job(job_id) is None]
        if invalid_ids:
            rendered = ", ".join(str(job_id) for job_id in invalid_ids)
            raise RuntimeError(f"Active scheduled jobs do not have one complete canonical owner (job ids: {rendered})")
        async with self._lock:
            registered_ids = set(self._registered_jobs)
        for job_id in active_ids - registered_ids:
            await self.register_job(job_id)
        for job_id in registered_ids - active_ids:
            await self.remove_job(job_id)
        await self._resume_pending_firings()

    async def _recover_interrupted_firings(self) -> None:
        """Return work owned by the stopped predecessor to the pending queue."""
        await self._store.connection.execute(
            "UPDATE workshop_schedule_firings SET status = 'pending', "
            "last_error_code = 'process_restarted', "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
            "WHERE status = 'executing'"
        )
        await self._store.connection.commit()

    async def _resume_pending_firings(self) -> None:
        async with self._store.connection.execute(
            "SELECT firing_id, job_id, scheduled_for FROM workshop_schedule_firings "
            "WHERE status = 'pending' ORDER BY scheduled_for, firing_id"
        ) as cursor:
            rows = list(await cursor.fetchall())
        for row in rows:
            firing_id = str(row[0])
            if firing_id in self._firing_tasks:
                continue
            job = await self._load_job(int(row[1]), active_only=False)
            if job is None:
                await self._mark_firing(
                    firing_id,
                    _FiringOutcome("failed", terminal_code="job_definition_missing"),
                )
                continue
            self._start_existing_firing(
                firing_id,
                job,
                _ensure_utc(datetime.fromisoformat(str(row[2]))),
            )

    async def _jobs_table_exists(self) -> bool:
        async with self._store.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'jobs'"
        ) as cursor:
            return await cursor.fetchone() is not None

    async def _load_job(self, job_id: int, *, active_only: bool = True) -> _CanonicalJob | None:
        if not await self._jobs_table_exists():
            return None
        async with self._store.connection.execute(
            "SELECT j.id, j.name, j.job_type, j.prompt, j.schedule_type, j.schedule_data, "
            "j.auto_remove, j.notify_on_check, j.created_at, "
            "o.principal_id, o.channel_id, o.runtime_profile_id "
            "FROM jobs j JOIN workshop_job_owners o ON o.job_id = j.id "
            "JOIN principals p ON p.id = o.principal_id "
            "JOIN channels c ON c.id = o.channel_id "
            "JOIN channel_memberships cm ON cm.channel_id = c.id "
            "AND cm.principal_id = p.id "
            "JOIN agents a ON a.id = o.agent_id AND a.workshop_id = c.workshop_id "
            "JOIN channel_agent_runtime_assignments ra ON ra.channel_id = c.id "
            "AND ra.agent_id = a.id AND ra.runtime_profile_id = o.runtime_profile_id "
            f"WHERE j.id = ?{' AND j.active = 1' if active_only else ''}",
            (job_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        schedule_data = json.loads(str(row[5]))
        if not isinstance(schedule_data, dict):
            raise RuntimeError(f"Scheduled job {job_id} has invalid schedule data")
        return _CanonicalJob(
            job_id=int(row[0]),
            name=str(row[1]),
            job_type=normalize_job_type(str(row[2])),
            prompt=str(row[3]),
            schedule_type=str(row[4]),
            schedule_data=schedule_data,
            created_at=_ensure_utc(datetime.fromisoformat(str(row[8]))),
            auto_remove=bool(row[6]),
            notify_on_check=bool(row[7]),
            principal_id=PrincipalId(str(row[9])),
            channel_id=ChannelId(str(row[10])),
            runtime_profile_id=RuntimeProfileId(str(row[11])),
        )

    def _register_loaded_job(self, job: _CanonicalJob) -> None:
        options = {
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": None,
            "replace_existing": True,
        }
        scheduler_ids: list[str] = []
        if job.schedule_type == "once":
            run_at = _ensure_utc(datetime.fromisoformat(str(job.schedule_data["run_at"])))
            scheduler_id = f"kai-core-job:{job.job_id}"
            self._scheduler.add_job(
                self._scheduled_callback,
                DateTrigger(run_date=run_at, timezone=UTC),
                id=scheduler_id,
                args=(job.job_id, "once", run_at.isoformat()),
                **options,
            )
            scheduler_ids.append(scheduler_id)
        elif job.schedule_type == "interval":
            seconds = int(job.schedule_data["seconds"])
            if seconds <= 0:
                raise RuntimeError(f"Scheduled job {job.job_id} has a non-positive interval")
            scheduler_id = f"kai-core-job:{job.job_id}"
            self._scheduler.add_job(
                self._scheduled_callback,
                IntervalTrigger(
                    seconds=seconds,
                    start_date=_next_interval_fire(datetime.now(UTC), job.created_at, seconds),
                    timezone=UTC,
                ),
                id=scheduler_id,
                args=(job.job_id, "interval", job.created_at.isoformat()),
                **options,
            )
            scheduler_ids.append(scheduler_id)
        elif job.schedule_type == "daily":
            daily_times = _daily_times(job.schedule_data)
            for index, daily_time in enumerate(daily_times):
                scheduler_id = f"kai-core-job:{job.job_id}:{index}"
                self._scheduler.add_job(
                    self._scheduled_callback,
                    CronTrigger(
                        hour=daily_time.hour,
                        minute=daily_time.minute,
                        second=0,
                        timezone=UTC,
                    ),
                    id=scheduler_id,
                    args=(job.job_id, "daily", daily_time.isoformat()),
                    **options,
                )
                scheduler_ids.append(scheduler_id)
        else:
            raise RuntimeError(f"Scheduled job {job.job_id} has unknown schedule type {job.schedule_type!r}")
        self._registered_jobs[job.job_id] = tuple(scheduler_ids)

    def _remove_registered_job(self, job_id: int) -> None:
        for scheduler_id in self._registered_jobs.pop(job_id, ()):
            if self._scheduler.get_job(scheduler_id) is not None:
                self._scheduler.remove_job(scheduler_id)

    async def _scheduled_callback(
        self,
        job_id: int,
        schedule_type: str,
        marker: str,
    ) -> None:
        try:
            job = await self._load_job(job_id)
            if job is None:
                return
            now = datetime.now(UTC)
            if schedule_type == "once":
                scheduled_for = _ensure_utc(datetime.fromisoformat(marker))
            elif schedule_type == "interval":
                seconds = float(job.schedule_data["seconds"])
                scheduled_for = _current_interval_occurrence(
                    now,
                    _ensure_utc(datetime.fromisoformat(marker)),
                    seconds,
                )
            elif schedule_type == "daily":
                scheduled_for = _current_daily_occurrence(now, time.fromisoformat(marker))
            else:
                raise RuntimeError(f"Scheduled callback has unknown type {schedule_type!r}")
            await self._queue_firing(job, scheduled_for)
            if schedule_type == "once":
                await self._deactivate(job.job_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._record_failure(exc)
            log.exception("Core scheduler callback failed for job %d", job_id)
            raise

    async def _queue_firing(self, job: _CanonicalJob, scheduled_for: datetime) -> None:
        occurrence_id = _occurrence_id(scheduled_for)
        firing_id = _firing_id(job.job_id, occurrence_id)
        await self._store.connection.execute(
            "INSERT OR IGNORE INTO workshop_schedule_firings "
            "(firing_id, job_id, occurrence_id, scheduled_for, job_type, status) "
            "VALUES (?, ?, ?, ?, ?, 'pending')",
            (
                firing_id,
                job.job_id,
                occurrence_id,
                scheduled_for.astimezone(UTC).isoformat(),
                job.job_type,
            ),
        )
        await self._store.connection.commit()
        async with self._store.connection.execute(
            "SELECT status FROM workshop_schedule_firings WHERE firing_id = ?",
            (firing_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is not None and str(row[0]) == "pending":
            self._start_existing_firing(firing_id, job, scheduled_for)

    def _start_existing_firing(
        self,
        firing_id: str,
        job: _CanonicalJob,
        scheduled_for: datetime,
    ) -> None:
        if firing_id in self._firing_tasks:
            return
        task = asyncio.create_task(
            self._execute_firing(firing_id, job, scheduled_for),
            name=f"kai-workshop-scheduled-firing:{firing_id}",
        )
        self._firing_tasks[firing_id] = task
        task.add_done_callback(lambda completed, current=firing_id: self._firing_done(current, completed))

    async def _execute_firing(
        self,
        firing_id: str,
        job: _CanonicalJob,
        scheduled_for: datetime,
    ) -> None:
        cursor = await self._store.connection.execute(
            "UPDATE workshop_schedule_firings SET status = 'executing', "
            "attempt_count = attempt_count + 1, last_error_code = NULL, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
            "WHERE firing_id = ? AND status = 'pending'",
            (firing_id,),
        )
        await self._store.connection.commit()
        if cursor.rowcount != 1:
            return
        try:
            outcome = await self._fire(firing_id, job, scheduled_for)
            await self._mark_firing(firing_id, outcome)
        except asyncio.CancelledError:
            await self._store.connection.execute(
                "UPDATE workshop_schedule_firings SET status = 'pending', "
                "last_error_code = 'shutdown_interrupted', "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
                "WHERE firing_id = ? AND status = 'executing'",
                (firing_id,),
            )
            await self._store.connection.commit()
            raise
        except Exception as exc:
            await self._store.connection.execute(
                "UPDATE workshop_schedule_firings SET status = 'pending', last_error_code = ?, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
                "WHERE firing_id = ? AND status = 'executing'",
                (type(exc).__name__, firing_id),
            )
            await self._store.connection.commit()
            raise

    async def _mark_firing(self, firing_id: str, outcome: _FiringOutcome) -> None:
        await self._store.connection.execute(
            "UPDATE workshop_schedule_firings SET status = ?, canonical_message_id = ?, "
            "run_id = ?, terminal_code = ?, last_error_code = NULL, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE firing_id = ?",
            (
                outcome.status,
                outcome.message_id,
                outcome.run_id,
                outcome.terminal_code,
                firing_id,
            ),
        )
        await self._store.connection.commit()

    async def _fire(
        self,
        firing_id: str,
        job: _CanonicalJob,
        scheduled_for: datetime,
    ) -> _FiringOutcome:
        occurrence_id = _occurrence_id(scheduled_for)
        if job.job_type == JOB_TYPE_REMINDER:
            prompt = job.prompt.replace("\\!", "!").replace("\\.", ".").replace("\\?", "?")
            recorded = await self._reminders.record(ScheduledReminder(job.job_id, occurrence_id, prompt, scheduled_for))
            return _FiringOutcome("succeeded", message_id=recorded.message_id)
        if job.job_type != JOB_TYPE_AGENT:
            raise RuntimeError(f"Scheduled job {job.job_id} has unsupported type {job.job_type!r}")

        accepted = await self._execution.accept_scheduled(
            ScheduledInboundMessage(
                principal_id=job.principal_id,
                channel_id=job.channel_id,
                job_id=job.job_id,
                occurrence_id=occurrence_id,
                body=job.prompt,
                occurred_at=scheduled_for,
            )
        )
        if accepted.command.disposition not in {
            ConversationCommandDisposition.NEWLY_ACCEPTED,
            ConversationCommandDisposition.READY_REPLAY,
        }:
            if accepted.command.disposition == ConversationCommandDisposition.TERMINAL_REPLAY:
                status = "succeeded" if accepted.run.status.value == "completed" else "failed"
                if status == "succeeded" and await self._condition_met(firing_id):
                    await self._deactivate(job.job_id)
                return _FiringOutcome(
                    status,
                    message_id=accepted.run.result_message_id,
                    run_id=accepted.run.run_id,
                    terminal_code=accepted.run.terminal_code,
                )
            return _FiringOutcome("pending", run_id=accepted.run.run_id)
        await self._store.connection.execute(
            "UPDATE workshop_schedule_firings SET run_id = ?, "
            "canonical_message_id = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
            "WHERE firing_id = ?",
            (
                accepted.run.run_id,
                accepted.command.message.event.envelope.aggregate_id,
                firing_id,
            ),
        )
        await self._store.connection.commit()
        condition_met = False

        async def transform(response: AgentResponse) -> CanonicalSuccessOutcome:
            nonlocal condition_met
            outcome, condition_met = _scheduled_success_outcome(job, response.text)
            if condition_met:
                await self._store.connection.execute(
                    "UPDATE workshop_schedule_firings SET condition_met = 1, "
                    "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE firing_id = ?",
                    (firing_id,),
                )
                await self._store.connection.commit()
            return outcome

        result = await self._execution.execute(
            accepted.run.run_id,
            success_transformer=transform,
        )
        compatibility = self._compatibility_state.for_profile(accepted.runtime_profile_id)
        if result.session_id and result.selection is not None:
            await compatibility.save_session(result.session_id, result.selection.model)
        if result.disposition == CanonicalExecutionDisposition.COMPLETED and result.terminal is not None:
            result_message_id = result.terminal.finalization.message.event.envelope.aggregate_id
            inbound_message_id = accepted.command.message.event.envelope.aggregate_id
            if not isinstance(result_message_id, MessageId) or not isinstance(inbound_message_id, MessageId):
                raise RuntimeError("Scheduled execution did not identify canonical messages")
            if result.workspace is not None:
                prior_pairs = await self._execution.prior_conversation_pairs(
                    accepted.run.run_id,
                    limit=compatibility.memory_context_turns,
                )
                compatibility.schedule_memory_ingestion(
                    prompt=job.prompt,
                    assistant_text=result.terminal.body,
                    session_id=result.session_id,
                    workspace=result.workspace,
                    canonical_provenance=CanonicalMemoryProvenance(
                        run_id=accepted.run.run_id,
                        source_message_id=inbound_message_id,
                        result_message_id=result_message_id,
                    ),
                    canonical_prior_pairs=prior_pairs,
                )
        if condition_met:
            await self._deactivate(job.job_id)
        terminal_code = result.run.terminal_code
        status = "succeeded" if result.run.status.value == "completed" else "failed"
        if result.disposition in {
            CanonicalExecutionDisposition.ACTIVE_REPLAY,
            CanonicalExecutionDisposition.CANCELLATION_PENDING_REPLAY,
            CanonicalExecutionDisposition.PREPARATION_DEFERRED,
        }:
            status = "pending"
        return _FiringOutcome(
            status,
            message_id=result.run.result_message_id,
            run_id=result.run.run_id,
            terminal_code=terminal_code,
        )

    async def _condition_met(self, firing_id: str) -> bool:
        async with self._store.connection.execute(
            "SELECT condition_met FROM workshop_schedule_firings WHERE firing_id = ?",
            (firing_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return row is not None and bool(row[0])

    async def _deactivate(self, job_id: int) -> None:
        await self._store.connection.execute("UPDATE jobs SET active = 0 WHERE id = ?", (job_id,))
        await self._store.connection.commit()
        async with self._lock:
            self._remove_registered_job(job_id)

    def _firing_done(self, firing_id: str, task: asyncio.Task[None]) -> None:
        if self._firing_tasks.get(firing_id) is task:
            del self._firing_tasks[firing_id]
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self._record_failure(error)
            log.error(
                "Scheduled firing failed: %s",
                error,
                exc_info=(type(error), error, error.__traceback__),
            )

    def _record_failure(self, error: BaseException) -> None:
        if self._failure is None:
            self._failure = error
            self._state = WorkshopSchedulerState.FAILED
            self._failure_event.set()


def _ensure_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _occurrence_id(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _firing_id(job_id: int, occurrence_id: str) -> str:
    digest = hashlib.sha256(f"{job_id}:{occurrence_id}".encode()).hexdigest()[:32]
    return f"fir_{digest}"


def _daily_times(schedule_data: dict[str, Any]) -> tuple[time, ...]:
    values = schedule_data.get("times")
    if not isinstance(values, list) or not values:
        raise RuntimeError("Daily schedule requires non-empty times")
    result = []
    for value in values:
        hour_text, minute_text = str(value).split(":", 1)
        result.append(time(int(hour_text), int(minute_text), tzinfo=UTC))
    return tuple(sorted(result))


def _next_daily_fire(now: datetime, daily_times: tuple[time, ...]) -> datetime:
    for candidate_time in daily_times:
        candidate = datetime.combine(now.date(), candidate_time)
        if candidate > now:
            return candidate
    return datetime.combine(now.date() + timedelta(days=1), daily_times[0])


def _next_interval_fire(now: datetime, anchor: datetime, seconds: float) -> datetime:
    """Return the first stable interval boundary strictly after ``now``."""
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    anchor = _ensure_utc(anchor)
    now = _ensure_utc(now)
    if anchor > now:
        return anchor
    elapsed = (now - anchor).total_seconds()
    periods = int(elapsed // seconds) + 1
    return anchor + timedelta(seconds=periods * seconds)


def _current_interval_occurrence(now: datetime, anchor: datetime, seconds: float) -> datetime:
    """Resolve a delayed callback to its deterministic interval boundary."""
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    anchor = _ensure_utc(anchor)
    now = _ensure_utc(now)
    if now <= anchor:
        return anchor
    periods = max(1, int((now - anchor).total_seconds() // seconds))
    return anchor + timedelta(seconds=periods * seconds)


def _current_daily_occurrence(now: datetime, scheduled_time: time) -> datetime:
    """Resolve a delayed daily callback to the most recent matching UTC slot."""
    now = _ensure_utc(now)
    candidate = datetime.combine(now.date(), scheduled_time, tzinfo=scheduled_time.tzinfo or UTC)
    candidate = _ensure_utc(candidate)
    return candidate if candidate <= now else candidate - timedelta(days=1)


def _strip_condition_marker(text: str, marker: str) -> str:
    lines = text.strip().split("\n", 1)
    after = lines[0].strip()[len(marker) :].lstrip(":").strip()
    rest = lines[1].strip() if len(lines) > 1 else ""
    return f"{after}\n{rest}".strip() if after else rest


def _scheduled_success_outcome(
    job: _CanonicalJob,
    text: str,
) -> tuple[CanonicalSuccessOutcome, bool]:
    first_line = text.strip().split("\n", 1)[0].strip().upper()
    if job.auto_remove and first_line.startswith(_CONDITION_MET_PREFIX):
        clean = _strip_condition_marker(text, _CONDITION_MET_PREFIX)
        body = f"[Job: {job.name}]\n{clean}" if clean else f"[Job: {job.name}] Condition met."
        return CanonicalSuccessOutcome(body), True
    if job.auto_remove and first_line.startswith(_CONDITION_NOT_MET_PREFIX):
        clean = _strip_condition_marker(text, _CONDITION_NOT_MET_PREFIX)
        body = f"[Job: {job.name}]\n{clean}" if clean else f"[Job: {job.name}] Still checking..."
        return CanonicalSuccessOutcome(body, request_delivery=job.notify_on_check), False
    return CanonicalSuccessOutcome(f"[Job: {job.name}]\n{text}"), False
