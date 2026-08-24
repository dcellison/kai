"""Contracts for the core-owned canonical Workshop scheduler."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kai import sessions
from kai.backend import AgentResponse, StreamEvent
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.client_events import ClientTimelineMessageEvent, read_client_channel_events
from kai.workshop.conversation_commands import ConversationCommandDisposition
from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.domain import AgentId, ChannelId, PrincipalId, RuntimeProfileId
from kai.workshop.execution_coordinator import CanonicalExecutionDisposition
from kai.workshop.private_text_execution import WorkshopPrivateTextExecutionService
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.scheduled_jobs import (
    WorkshopScheduledJobAuthority,
    WorkshopScheduledJobStore,
)
from kai.workshop.scheduler import (
    WorkshopCanonicalScheduler,
    _CanonicalJob,
    _current_interval_occurrence,
    _firing_id,
    _next_daily_fire,
    _next_interval_fire,
    _occurrence_id,
    _scheduled_success_outcome,
)
from kai.workshop.store import WorkshopEventStore
from kai.workshop.timeline import read_channel_timeline
from tests.workshop_profiles import profile_id, profile_registry


class _UnusedExecution:
    pass


class _UnusedCompatibilityState:
    pass


class _AllowRead:
    async def can_read_channel(self, principal_id: PrincipalId, channel_id: ChannelId) -> bool:
        del principal_id, channel_id
        return True


class _AgentExecution:
    def __init__(self) -> None:
        run_id = "run_" + "4" * 32
        inbound_id = "msg_" + "5" * 32
        result_id = "msg_" + "6" * 32
        accepted_run = SimpleNamespace(
            run_id=run_id,
            status=SimpleNamespace(value="accepted"),
            result_message_id=None,
            terminal_code=None,
        )
        completed_run = SimpleNamespace(
            run_id=run_id,
            status=SimpleNamespace(value="completed"),
            result_message_id=result_id,
            terminal_code=None,
        )
        self.accept_scheduled = AsyncMock(
            return_value=SimpleNamespace(
                command=SimpleNamespace(
                    disposition=ConversationCommandDisposition.NEWLY_ACCEPTED,
                    message=SimpleNamespace(
                        event=SimpleNamespace(
                            envelope=SimpleNamespace(aggregate_id=inbound_id),
                        )
                    ),
                ),
                run=accepted_run,
                runtime_profile_id=profile_id(101),
            )
        )
        self.execute = AsyncMock(
            return_value=SimpleNamespace(
                disposition=CanonicalExecutionDisposition.COMPLETED,
                run=completed_run,
                terminal=None,
                session_id=None,
                selection=None,
            )
        )


class _CompatibilityState:
    def for_profile(self, runtime_profile_id):
        assert runtime_profile_id == profile_id(101)
        return SimpleNamespace(memory_context_turns=10)


class _CanonicalRuntime:
    def __init__(self, workspace: Path) -> None:
        self.selection = SimpleNamespace(
            backend="codex",
            provider="openai",
            model="gpt-5.6-sol",
        )
        self.workspace = workspace

    def stage_canonical_history(self, _history: str) -> None:
        pass

    def validate_current(self) -> None:
        pass

    async def cancel(self) -> None:
        pass

    async def stream(self, _prompt: str):
        yield StreamEvent(
            text_so_far="Scheduled agent answer",
            done=True,
            response=AgentResponse(success=True, text="Scheduled agent answer"),
        )


class _CanonicalCompatibility:
    memory_context_turns = 10

    async def save_session(self, _session_id: str, _model: str) -> None:
        pass

    def schedule_memory_ingestion(self, **_kwargs) -> None:
        pass


class _CanonicalCompatibilityState:
    def for_profile(self, runtime_profile_id):
        assert runtime_profile_id == profile_id(101)
        return _CanonicalCompatibility()


def _job(
    *,
    auto_remove: bool = False,
    notify_on_check: bool = False,
) -> _CanonicalJob:
    return _CanonicalJob(
        job_id=1,
        name="Monitor",
        job_type="agent",
        prompt="Check it",
        schedule_type="interval",
        schedule_data={"seconds": 60},
        created_at=datetime(2026, 8, 22, tzinfo=UTC),
        auto_remove=auto_remove,
        notify_on_check=notify_on_check,
        principal_id=PrincipalId("prn_" + "1" * 32),
        channel_id=ChannelId("chn_" + "2" * 32),
        runtime_profile_id=RuntimeProfileId("rtp_" + "3" * 32),
    )


async def _install_owned_job(
    database: Path,
    *,
    transport: str,
    run_at: datetime,
    job_type: str = "reminder",
    prompt: str = "Remember this",
) -> tuple[WorkshopEventStore, int]:
    await sessions.init_db(database)
    store = await WorkshopEventStore.open(database)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                display_name="Daniel",
                role="admin",
                transport=transport,
                external_subject="daniel",
                external_channel_id="daniel-direct",
                runtime_profile_id=profile_id(101),
            ),
        ),
    )
    async with store.connection.execute(
        "SELECT ei.principal_id, cb.channel_id FROM external_identities ei "
        "JOIN channel_bindings cb ON cb.transport = ei.provider "
        "AND cb.external_channel_id = ? WHERE ei.provider = ? AND ei.external_subject = ?",
        ("daniel-direct", transport, "daniel"),
    ) as cursor:
        principal_id, channel_id = await cursor.fetchone()
    async with store.connection.execute("SELECT id FROM agents") as cursor:
        agent_id = (await cursor.fetchone())[0]
    job = await WorkshopScheduledJobStore(store).create(
        WorkshopScheduledJobAuthority(
            PrincipalId(str(principal_id)),
            ChannelId(str(channel_id)),
            AgentId(str(agent_id)),
            profile_id(101),
        ),
        name="Canonical reminder",
        job_type=job_type,
        prompt=prompt,
        schedule_type="once",
        schedule_data=json.dumps({"run_at": run_at.isoformat()}),
    )
    return store, job.job_id


async def _job_owner(store: WorkshopEventStore, job_id: int) -> tuple[PrincipalId, ChannelId]:
    async with store.connection.execute(
        "SELECT principal_id, channel_id FROM workshop_scheduled_jobs WHERE id = ?",
        (job_id,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return PrincipalId(str(row[0])), ChannelId(str(row[1]))


@pytest.mark.parametrize(
    ("text", "notify", "expected_body", "expected_delivery", "condition_met"),
    [
        ("ordinary result", False, "[Job: Monitor]\nordinary result", True, False),
        ("CONDITION_MET: Ready", False, "[Job: Monitor]\nReady", True, True),
        ("CONDITION_MET:", False, "[Job: Monitor] Condition met.", True, True),
        ("CONDITION_NOT_MET: Waiting", False, "[Job: Monitor]\nWaiting", False, False),
        ("CONDITION_NOT_MET", True, "[Job: Monitor] Still checking...", True, False),
    ],
)
def test_conditional_agent_result_policy(
    text: str,
    notify: bool,
    expected_body: str,
    expected_delivery: bool,
    condition_met: bool,
) -> None:
    outcome, met = _scheduled_success_outcome(
        _job(auto_remove=True, notify_on_check=notify),
        text,
    )

    assert outcome.body == expected_body
    assert outcome.request_delivery is expected_delivery
    assert met is condition_met


def test_next_daily_fire_rolls_to_tomorrow_after_last_time() -> None:
    now = datetime(2026, 8, 22, 23, 0, tzinfo=UTC)

    result = _next_daily_fire(
        now,
        (
            datetime(2026, 8, 22, 9, 0, tzinfo=UTC).timetz(),
            datetime(2026, 8, 22, 17, 0, tzinfo=UTC).timetz(),
        ),
    )

    assert result == datetime(2026, 8, 23, 9, 0, tzinfo=UTC)


def test_interval_boundaries_are_stable_across_restart_and_delay() -> None:
    anchor = datetime(2026, 8, 22, 10, 0, tzinfo=UTC)
    now = datetime(2026, 8, 22, 10, 3, 5, tzinfo=UTC)

    assert _next_interval_fire(now, anchor, 60) == datetime(2026, 8, 22, 10, 4, tzinfo=UTC)
    assert _current_interval_occurrence(now, anchor, 60) == datetime(2026, 8, 22, 10, 3, tzinfo=UTC)


async def test_workshop_only_reminder_fires_canonically_without_delivery(tmp_path: Path) -> None:
    database = tmp_path / "kai.db"
    source_store, job_id = await _install_owned_job(
        database,
        transport="workshop",
        run_at=datetime.now(UTC) + timedelta(milliseconds=100),
    )
    scheduler = await WorkshopCanonicalScheduler.open_and_start(
        database,
        _UnusedExecution(),  # type: ignore[arg-type]
        _UnusedCompatibilityState(),  # type: ignore[arg-type]
    )
    try:
        await asyncio.sleep(0.25)
        async with source_store.connection.execute(
            "SELECT m.body, json_extract(e.metadata_json, '$.source') "
            "FROM messages m JOIN event_log e ON e.position = m.created_event_position"
        ) as cursor:
            assert tuple(await cursor.fetchone()) == ("Remember this", "scheduled_job")
        async with source_store.connection.execute("SELECT COUNT(*) FROM delivery_outbox") as cursor:
            assert (await cursor.fetchone())[0] == 0
        async with source_store.connection.execute(
            "SELECT active FROM workshop_scheduled_jobs WHERE id = ?",
            (job_id,),
        ) as cursor:
            assert (await cursor.fetchone())[0] == 0
        async with source_store.connection.execute(
            "SELECT status, attempt_count, canonical_message_id FROM workshop_schedule_firings WHERE job_id = ?",
            (job_id,),
        ) as cursor:
            firing = await cursor.fetchone()
        assert tuple(firing[:2]) == ("succeeded", 1)
        assert str(firing[2]).startswith("msg_")
        principal_id, channel_id = await _job_owner(source_store, job_id)
        page = await read_channel_timeline(
            source_store,
            principal_id=principal_id,
            channel_id=channel_id,
            authorizer=_AllowRead(),
        )
        assert [message.body for message in page.messages] == ["Remember this"]
        assert [message.author_kind for message in page.messages] == ["agent"]
    finally:
        await scheduler.stop()
        await source_store.close()
        await sessions.close_db()


async def test_telegram_bound_reminder_uses_notification_outbox(tmp_path: Path) -> None:
    database = tmp_path / "kai.db"
    source_store, _ = await _install_owned_job(
        database,
        transport="telegram",
        run_at=datetime.now(UTC) + timedelta(milliseconds=100),
    )
    scheduler = await WorkshopCanonicalScheduler.open_and_start(
        database,
        _UnusedExecution(),  # type: ignore[arg-type]
        _UnusedCompatibilityState(),  # type: ignore[arg-type]
    )
    try:
        await asyncio.sleep(0.25)
        async with source_store.connection.execute("SELECT purpose, status FROM delivery_outbox") as cursor:
            assert tuple(await cursor.fetchone()) == ("notification", "pending")
    finally:
        await scheduler.stop()
        await source_store.close()
        await sessions.close_db()


async def test_scheduler_starts_without_legacy_jobs_table(tmp_path: Path) -> None:
    database = tmp_path / "workshop-only.db"
    store = await WorkshopEventStore.open(database)
    await store.close()

    scheduler = await WorkshopCanonicalScheduler.open_and_start(
        database,
        _UnusedExecution(),  # type: ignore[arg-type]
        _UnusedCompatibilityState(),  # type: ignore[arg-type]
    )
    try:
        assert scheduler.readiness.ready is True
        assert scheduler.readiness.active_jobs == 0
    finally:
        await scheduler.stop()


async def test_scheduler_fails_closed_for_active_job_without_canonical_owner(
    tmp_path: Path,
) -> None:
    database = tmp_path / "kai.db"
    store, job_id = await _install_owned_job(
        database,
        transport="workshop",
        run_at=datetime.now(UTC) + timedelta(hours=1),
    )
    await store.connection.execute(
        "UPDATE workshop_scheduled_jobs SET runtime_profile_id = ? WHERE id = ?",
        (RuntimeProfileId("rtp_" + "9" * 32), job_id),
    )
    await store.connection.commit()

    with pytest.raises(RuntimeError, match="canonical execution lane"):
        await WorkshopCanonicalScheduler.open_and_start(
            database,
            _UnusedExecution(),  # type: ignore[arg-type]
            _UnusedCompatibilityState(),  # type: ignore[arg-type]
        )
    await store.close()
    await sessions.close_db()


async def test_once_daily_and_interval_jobs_reconcile_into_core_scheduler(
    tmp_path: Path,
) -> None:
    database = tmp_path / "kai.db"
    source_store, once_id = await _install_owned_job(
        database,
        transport="workshop",
        run_at=datetime.now(UTC) + timedelta(hours=1),
    )
    async with source_store.connection.execute(
        "SELECT principal_id, channel_id, agent_id, runtime_profile_id FROM workshop_scheduled_jobs WHERE id = ?",
        (once_id,),
    ) as cursor:
        owner = tuple(await cursor.fetchone())
    authority = WorkshopScheduledJobAuthority(
        PrincipalId(str(owner[0])),
        ChannelId(str(owner[1])),
        AgentId(str(owner[2])),
        RuntimeProfileId(str(owner[3])),
    )
    job_store = WorkshopScheduledJobStore(source_store)
    interval = await job_store.create(
        authority,
        name="Interval",
        job_type="reminder",
        prompt="Interval reminder",
        schedule_type="interval",
        schedule_data=json.dumps({"seconds": 60}),
    )
    daily = await job_store.create(
        authority,
        name="Daily",
        job_type="reminder",
        prompt="Daily reminder",
        schedule_type="daily",
        schedule_data=json.dumps({"times": ["09:00", "17:00"]}),
    )
    interval_id = interval.job_id
    daily_id = daily.job_id

    scheduler = await WorkshopCanonicalScheduler.open_and_start(
        database,
        _UnusedExecution(),  # type: ignore[arg-type]
        _UnusedCompatibilityState(),  # type: ignore[arg-type]
    )
    try:
        assert scheduler.readiness.active_jobs == 3
        assert len(scheduler._scheduler.get_jobs()) == 4
        assert set(scheduler._registered_jobs) == {once_id, interval_id, daily_id}
    finally:
        await scheduler.stop()
        await source_store.close()
        await sessions.close_db()


async def test_interrupted_reminder_firing_recovers_once(tmp_path: Path) -> None:
    database = tmp_path / "kai.db"
    scheduled_for = datetime.now(UTC) - timedelta(seconds=1)
    source_store, job_id = await _install_owned_job(
        database,
        transport="workshop",
        run_at=datetime.now(UTC) + timedelta(hours=1),
    )
    occurrence_id = _occurrence_id(scheduled_for)
    firing_id = _firing_id(job_id, occurrence_id)
    await source_store.connection.execute(
        "INSERT INTO workshop_schedule_firings "
        "(firing_id, job_id, occurrence_id, scheduled_for, job_type, status, attempt_count) "
        "VALUES (?, ?, ?, ?, 'reminder', 'executing', 1)",
        (firing_id, job_id, occurrence_id, scheduled_for.isoformat()),
    )
    await source_store.connection.commit()

    scheduler = await WorkshopCanonicalScheduler.open_and_start(
        database,
        _UnusedExecution(),  # type: ignore[arg-type]
        _UnusedCompatibilityState(),  # type: ignore[arg-type]
    )
    try:
        await asyncio.sleep(0.2)
        async with source_store.connection.execute(
            "SELECT status, attempt_count FROM workshop_schedule_firings WHERE firing_id = ?",
            (firing_id,),
        ) as cursor:
            assert tuple(await cursor.fetchone()) == ("succeeded", 2)
        async with source_store.connection.execute(
            "SELECT COUNT(*) FROM messages m JOIN event_log e "
            "ON e.position = m.created_event_position "
            "WHERE json_extract(e.metadata_json, '$.occurrence_id') = ?",
            (occurrence_id,),
        ) as cursor:
            assert (await cursor.fetchone())[0] == 1
    finally:
        await scheduler.stop()
        await source_store.close()
        await sessions.close_db()


async def test_workshop_only_agent_job_uses_canonical_execution(tmp_path: Path) -> None:
    database = tmp_path / "kai.db"
    source_store, job_id = await _install_owned_job(
        database,
        transport="workshop",
        run_at=datetime.now(UTC) + timedelta(milliseconds=100),
        job_type="agent",
        prompt="Perform the scheduled check",
    )
    execution = _AgentExecution()
    scheduler = await WorkshopCanonicalScheduler.open_and_start(
        database,
        execution,  # type: ignore[arg-type]
        _CompatibilityState(),  # type: ignore[arg-type]
    )
    try:
        await asyncio.sleep(0.25)
        execution.accept_scheduled.assert_awaited_once()
        execution.execute.assert_awaited_once()
        scheduled = execution.accept_scheduled.await_args.args[0]
        assert scheduled.job_id == job_id
        assert scheduled.body == "Perform the scheduled check"
        async with source_store.connection.execute(
            "SELECT status, run_id, attempt_count FROM workshop_schedule_firings WHERE job_id = ?",
            (job_id,),
        ) as cursor:
            assert tuple(await cursor.fetchone()) == (
                "succeeded",
                "run_" + "4" * 32,
                1,
            )
    finally:
        await scheduler.stop()
        await source_store.close()
        await sessions.close_db()


async def test_agent_firing_completes_through_real_canonical_coordinator(
    tmp_path: Path,
) -> None:
    database = tmp_path / "kai.db"
    source_store, job_id = await _install_owned_job(
        database,
        transport="workshop",
        run_at=datetime.now(UTC) + timedelta(milliseconds=100),
        job_type="agent",
        prompt="Run through the canonical coordinator",
    )
    await WorkshopConversationDeliveryAuthority(source_store).activate()
    runtime = _CanonicalRuntime(tmp_path)
    pool = SimpleNamespace(prepare_execution=AsyncMock(return_value=runtime))
    execution = await WorkshopPrivateTextExecutionService.open_and_start(
        database,
        WorkshopRuntimePool(pool, profile_registry(101)),  # type: ignore[arg-type]
        registered_backend_ids=frozenset({"codex"}),
    )
    scheduler = await WorkshopCanonicalScheduler.open_and_start(
        database,
        execution,
        _CanonicalCompatibilityState(),  # type: ignore[arg-type]
    )
    try:
        await asyncio.sleep(0.35)
        async with source_store.connection.execute(
            "SELECT f.status, r.status, result.body FROM workshop_schedule_firings f "
            "JOIN runs r ON r.id = f.run_id "
            "JOIN messages result ON result.id = r.result_message_id "
            "WHERE f.job_id = ?",
            (job_id,),
        ) as cursor:
            assert tuple(await cursor.fetchone()) == (
                "succeeded",
                "completed",
                "[Job: Canonical reminder]\nScheduled agent answer",
            )
        async with source_store.connection.execute("SELECT COUNT(*) FROM delivery_outbox") as cursor:
            assert (await cursor.fetchone())[0] == 0
        principal_id, channel_id = await _job_owner(source_store, job_id)
        page = await read_channel_timeline(
            source_store,
            principal_id=principal_id,
            channel_id=channel_id,
            authorizer=_AllowRead(),
            limit=1,
        )
        assert [message.body for message in page.messages] == ["[Job: Canonical reminder]\nScheduled agent answer"]
        hidden_only = await read_client_channel_events(
            source_store,
            principal_id=principal_id,
            channel_id=channel_id,
            authorizer=_AllowRead(),
            after_position=0,
            limit=1,
        )
        assert hidden_only.events == ()
        assert hidden_only.next_position > 0
        live = await read_client_channel_events(
            source_store,
            principal_id=principal_id,
            channel_id=channel_id,
            authorizer=_AllowRead(),
            after_position=0,
        )
        visible_messages = [
            event.message.body for event in live.events if isinstance(event, ClientTimelineMessageEvent)
        ]
        assert visible_messages == ["[Job: Canonical reminder]\nScheduled agent answer"]
    finally:
        await scheduler.stop()
        await execution.stop()
        await source_store.close()
        await sessions.close_db()


def test_core_scheduler_has_no_telegram_dependency() -> None:
    source = Path("src/kai/workshop/scheduler.py").read_text()

    assert "telegram" not in source.lower()
    assert "Bot" not in source
    assert "ContextTypes" not in source
