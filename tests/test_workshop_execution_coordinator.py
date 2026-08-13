"""End-to-end contracts for the production-unused Workshop coordinator."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from kai.agent_failure import AgentFailureKind
from kai.backend import AgentResponse, StreamEvent
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.conversation_commands import WorkshopConversationCommandService
from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.domain import RunExecutionOwnerId
from kai.workshop.execution_coordinator import (
    CanonicalCancellationDisposition,
    CanonicalExecutionDisposition,
    WorkshopCanonicalExecutionCoordinator,
)
from kai.workshop.inbound import InboundMessage
from kai.workshop.run_execution_authority import (
    RunAttemptStatus,
    RunExecutionSelection,
    WorkshopRunExecutionAuthority,
)
from kai.workshop.run_lifecycle import RunStatus, WorkshopRunLifecycle
from kai.workshop.store import WorkshopEventStore

_NOW = datetime(2026, 8, 12, 22, 0, tzinfo=UTC)


class _Prepared:
    def __init__(
        self,
        run,
        *,
        response: AgentResponse | None = None,
        wait: asyncio.Event | None = None,
        on_stream: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.run = run
        self.selection = RunExecutionSelection("codex", "gpt-5.6-sol")
        self.workspace = Path("/private/tmp/kai-workshop-test-workspace")
        self.response = response or AgentResponse(success=True, text="Canonical answer")
        self.wait = wait
        self.on_stream = on_stream
        self.prompts: list[str] = []
        self.validated = False
        self.cancelled = False
        self.reject_validation = False

    def validate_current(self) -> None:
        self.validated = True
        if self.reject_validation:
            raise RuntimeError("runtime drift")

    async def cancel(self) -> None:
        self.cancelled = True
        if self.wait is not None:
            self.wait.set()

    async def stream(self, prompt: str) -> AsyncIterator[StreamEvent]:
        self.prompts.append(prompt)
        if self.on_stream is not None:
            await self.on_stream()
        if self.wait is not None:
            await self.wait.wait()
            if self.cancelled:
                raise RuntimeError("runtime stopped")
        yield StreamEvent(text_so_far=self.response.text, done=True, response=self.response)


class _Preparation:
    def __init__(self, prepared: _Prepared) -> None:
        self.prepared = prepared
        self.calls = 0

    async def prepare(self, run_id):
        assert run_id == self.prepared.run.run_id
        self.calls += 1
        return self.prepared


async def _accepted(path: Path, *, suffix: str = "1"):
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                display_name="Workshop Human",
                role="admin",
                transport="telegram",
                external_subject="101",
                external_channel_id="101",
            ),
        ),
    )
    result = await WorkshopConversationCommandService(store).accept(
        InboundMessage(
            transport="telegram",
            update_id=f"command-{suffix}",
            message_id=f"message-{suffix}",
            sender_subject="101",
            channel_subject="101",
            body=f"Canonical prompt {suffix}",
            occurred_at=_NOW,
        )
    )
    await WorkshopConversationDeliveryAuthority(store).activate()
    return store, result.run


def _coordinator(store, preparation, *, lease_seconds: int = 60):
    return WorkshopCanonicalExecutionCoordinator(
        store,
        preparation,
        registered_backend_ids=frozenset({"codex"}),
        clock=lambda: _NOW + timedelta(seconds=10),
        lease_duration=timedelta(seconds=lease_seconds),
    )


async def _terminal_bodies(store: WorkshopEventStore) -> list[str]:
    async with store.connection.execute("SELECT body FROM messages ORDER BY created_event_position") as cursor:
        return [str(row[0]) for row in await cursor.fetchall()]


class TestCanonicalExecutionCoordinator:
    async def test_success_uses_stored_prompt_and_starts_before_exact_dispatch(self, tmp_path: Path):
        store, run = await _accepted(tmp_path / "kai.db")

        async def assert_started() -> None:
            assert (await WorkshopRunLifecycle(store).state(run.run_id)).status == RunStatus.STARTED

        prepared = _Prepared(run, on_stream=assert_started)
        coordinator = _coordinator(store, _Preparation(prepared))
        try:
            result = await coordinator.execute(run.run_id)

            assert result.disposition == CanonicalExecutionDisposition.COMPLETED
            assert result.run.status == RunStatus.COMPLETED
            assert prepared.validated is True
            assert prepared.prompts == ["Canonical prompt 1"]
            assert await _terminal_bodies(store) == ["Canonical prompt 1", "Canonical answer"]
        finally:
            await store.close()

    async def test_stream_observer_sees_only_nonterminal_events(self, tmp_path: Path):
        store, run = await _accepted(tmp_path / "kai.db")
        prepared = _Prepared(run)
        observed: list[StreamEvent] = []

        async def observe(event: StreamEvent) -> None:
            observed.append(event)

        original_stream = prepared.stream

        async def stream_with_preview(prompt: str) -> AsyncIterator[StreamEvent]:
            yield StreamEvent(text_so_far="Stable preview.", done=False)
            async for event in original_stream(prompt):
                yield event

        prepared.stream = stream_with_preview  # type: ignore[method-assign]
        try:
            result = await _coordinator(store, _Preparation(prepared)).execute(
                run.run_id,
                stream_observer=observe,
            )

            assert result.disposition == CanonicalExecutionDisposition.COMPLETED
            assert [event.text_so_far for event in observed] == ["Stable preview."]
        finally:
            await store.close()

    async def test_concurrent_duplicate_dispatches_backend_once(self, tmp_path: Path):
        store, run = await _accepted(tmp_path / "kai.db")
        release = asyncio.Event()
        prepared = _Prepared(run, wait=release)
        preparation = _Preparation(prepared)
        coordinator = _coordinator(store, preparation)
        try:
            first = asyncio.create_task(coordinator.execute(run.run_id))
            while not prepared.prompts:
                await asyncio.sleep(0)
            second = asyncio.create_task(coordinator.execute(run.run_id))
            release.set()
            first_result, second_result = await asyncio.gather(first, second)

            assert first_result.disposition == CanonicalExecutionDisposition.COMPLETED
            assert second_result.disposition == CanonicalExecutionDisposition.TERMINAL_REPLAY
            assert preparation.calls == 1
            assert prepared.prompts == ["Canonical prompt 1"]
        finally:
            await store.close()

    async def test_native_failure_is_replaced_by_bounded_canonical_text(self, tmp_path: Path):
        store, run = await _accepted(tmp_path / "kai.db")
        prepared = _Prepared(
            run,
            response=AgentResponse(
                success=False,
                text="",
                error="native secret-bearing payload",
                failure_kind=AgentFailureKind.AUTHENTICATION_REQUIRED,
            ),
        )
        try:
            result = await _coordinator(store, _Preparation(prepared)).execute(run.run_id)

            assert result.disposition == CanonicalExecutionDisposition.FAILED
            bodies = await _terminal_bodies(store)
            assert (
                bodies[-1] == "Authentication for the configured agent is required. Kai did not complete this request."
            )
            assert "native" not in bodies[-1]
        finally:
            await store.close()

    async def test_runtime_drift_leaves_retryable_grant_until_expiry(self, tmp_path: Path):
        store, run = await _accepted(tmp_path / "kai.db")
        prepared = _Prepared(run)
        prepared.reject_validation = True
        preparation = _Preparation(prepared)
        coordinator = _coordinator(store, preparation, lease_seconds=5)
        try:
            deferred = await coordinator.execute(run.run_id)
            assert deferred.disposition == CanonicalExecutionDisposition.PREPARATION_DEFERRED
            assert (await WorkshopRunLifecycle(store).state(run.run_id)).status == RunStatus.ACCEPTED

            recovered = await coordinator.recover_expired(occurred_at=_NOW + timedelta(seconds=16))
            assert recovered.expired_before_dispatch == 1
            prepared.reject_validation = False
            completed = await coordinator.execute(run.run_id)
            assert completed.disposition == CanonicalExecutionDisposition.COMPLETED
            assert preparation.calls == 2
        finally:
            await store.close()

    async def test_cancellation_is_confirmed_only_after_exact_runtime_stops(self, tmp_path: Path):
        store, run = await _accepted(tmp_path / "kai.db")
        release = asyncio.Event()
        prepared = _Prepared(run, wait=release)
        coordinator = _coordinator(store, _Preparation(prepared))
        try:
            execution = asyncio.create_task(coordinator.execute(run.run_id))
            while not prepared.prompts:
                await asyncio.sleep(0)
            cancellation = await coordinator.request_cancellation(run.run_id)
            result = await execution

            assert cancellation == CanonicalCancellationDisposition.REQUESTED
            assert prepared.cancelled is True
            assert result.disposition == CanonicalExecutionDisposition.CANCELLED
            assert result.run.status == RunStatus.CANCELLED
            assert (await _terminal_bodies(store))[-1] == "This request was cancelled."
        finally:
            await store.close()

    async def test_accepted_run_cancels_durably_before_backend_dispatch(self, tmp_path: Path):
        store, run = await _accepted(tmp_path / "kai.db")
        prepared = _Prepared(run)
        preparation = _Preparation(prepared)
        coordinator = _coordinator(store, preparation)
        try:
            cancellation = await coordinator.request_cancellation(run.run_id)
            current = await WorkshopRunLifecycle(store).state(run.run_id)

            assert cancellation == CanonicalCancellationDisposition.REQUESTED
            assert current.status == RunStatus.CANCELLED
            assert current.terminal_code == "requested_by_human"
            assert preparation.calls == 0
            assert await _terminal_bodies(store) == ["Canonical prompt 1"]
        finally:
            await store.close()

    async def test_expired_started_attempt_gets_visible_interruption_without_redispatch(self, tmp_path: Path):
        store, run = await _accepted(tmp_path / "kai.db")
        selection = RunExecutionSelection("codex", "gpt-5.6-sol")
        authority = WorkshopRunExecutionAuthority(
            store,
            selection_resolver=lambda _run: selection,
            registered_backend_ids=frozenset({"codex"}),
        )
        granted = await authority.grant(
            run.run_id,
            owner_id=RunExecutionOwnerId.new(),
            occurred_at=_NOW + timedelta(seconds=1),
            lease_expires_at=_NOW + timedelta(seconds=2),
        )
        await authority.start(granted.claim, occurred_at=_NOW + timedelta(seconds=1, milliseconds=500))
        prepared = _Prepared(run)
        preparation = _Preparation(prepared)
        coordinator = _coordinator(store, preparation)
        try:
            recovered = await coordinator.recover_expired(occurred_at=_NOW + timedelta(seconds=3))

            assert recovered.interrupted_after_dispatch == 1
            assert (await authority.attempt(granted.claim.attempt_id)).status == RunAttemptStatus.INTERRUPTED
            assert (await WorkshopRunLifecycle(store).state(run.run_id)).status == RunStatus.FAILED
            assert (await _terminal_bodies(store))[-1] == (
                "Kai was interrupted while the configured agent was working. This request was not retried."
            )
            replay = await coordinator.execute(run.run_id)
            assert replay.disposition == CanonicalExecutionDisposition.TERMINAL_REPLAY
            assert preparation.calls == 0
        finally:
            await store.close()

    async def test_coordinator_remains_absent_from_production_construction(self):
        source_root = Path(__file__).parents[1] / "src" / "kai"
        for relative_path in ("main.py", "bot.py", "sessions.py"):
            source = (source_root / relative_path).read_text(encoding="utf-8")
            assert "WorkshopCanonicalExecutionCoordinator" not in source
