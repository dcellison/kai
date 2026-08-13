"""Integrated contracts for the production private-text execution owner."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from kai.backend import AgentResponse, StreamEvent
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.domain import ChannelId, PrincipalId
from kai.workshop.execution_coordinator import (
    CanonicalCancellationDisposition,
    CanonicalExecutionDisposition,
)
from kai.workshop.inbound import ClientInboundMessage, InboundMessage
from kai.workshop.private_text_execution import (
    RecoverableClientRun,
    WorkshopPrivateTextExecutionService,
)
from kai.workshop.run_lifecycle import RunStatus
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id, profile_registry

_NOW = datetime(2026, 1, 1, 23, 0, tzinfo=UTC)


class _Runtime:
    def __init__(self, workspace: Path, *, wait: asyncio.Event | None = None) -> None:
        self.selection = SimpleNamespace(backend="codex", provider="openai", model="gpt-5.6-sol")
        self.workspace = workspace
        self.wait = wait
        self.validated = False
        self.cancelled = False

    def validate_current(self) -> None:
        self.validated = True

    async def cancel(self) -> None:
        self.cancelled = True
        if self.wait is not None:
            self.wait.set()

    async def stream(self, prompt: str):
        yield StreamEvent(text_so_far="Stable preview.", done=False)
        if self.wait is not None:
            await self.wait.wait()
            if self.cancelled:
                raise RuntimeError("runtime stopped")
        yield StreamEvent(
            text_so_far="Durable answer",
            done=True,
            response=AgentResponse(success=True, text="Durable answer", session_id="session-1"),
        )


async def _foundation(database: Path) -> None:
    store = await WorkshopEventStore.open(database)
    try:
        await bootstrap_default_workshop(
            store,
            (
                BootstrapHuman(
                    display_name="Workshop Human",
                    role="admin",
                    transport="telegram",
                    external_subject="101",
                    external_channel_id="101",
                    runtime_profile_id=profile_id(101),
                ),
            ),
        )
        await WorkshopConversationDeliveryAuthority(store).activate()
    finally:
        await store.close()


def _message(*, suffix: str = "1") -> InboundMessage:
    return InboundMessage(
        transport="telegram",
        update_id=f"update-{suffix}",
        message_id=f"message-{suffix}",
        sender_subject="101",
        channel_subject="101",
        body=f"Canonical prompt {suffix}",
        occurred_at=_NOW,
    )


async def test_owner_accepts_executes_and_atomically_enqueues_terminal_reply(tmp_path: Path):
    database = tmp_path / "kai.db"
    await _foundation(database)
    runtime = _Runtime(tmp_path)
    pool = SimpleNamespace(prepare_execution=AsyncMock(return_value=runtime))
    service = await WorkshopPrivateTextExecutionService.open_and_start(
        database,
        WorkshopRuntimePool(pool, profile_registry(101)),  # type: ignore[arg-type]
        registered_backend_ids=frozenset({"codex"}),
    )
    observed: list[str] = []
    try:
        accepted = await service.accept(_message())

        async def observe(event: StreamEvent) -> None:
            observed.append(event.text_so_far)

        result = await service.execute(accepted.run.run_id, stream_observer=observe)

        assert result.disposition == CanonicalExecutionDisposition.COMPLETED
        assert result.run.status == RunStatus.COMPLETED
        assert result.session_id == "session-1"
        assert result.workspace == str(tmp_path)
        assert observed == ["Stable preview."]
        assert runtime.validated is True
        pool.prepare_execution.assert_awaited_once_with(101)
    finally:
        await service.stop()

    inspection = await WorkshopEventStore.open(database)
    try:
        async with inspection.connection.execute(
            "SELECT m.body, d.status FROM messages m "
            "JOIN delivery_outbox d ON d.message_id = m.id "
            "WHERE m.reply_to_message_id IS NOT NULL"
        ) as cursor:
            assert tuple(await cursor.fetchone()) == ("Durable answer", "pending")
    finally:
        await inspection.close()


async def test_owner_routes_stop_to_exact_active_runtime_and_terminal_cancellation(tmp_path: Path):
    database = tmp_path / "kai.db"
    await _foundation(database)
    release = asyncio.Event()
    runtime = _Runtime(tmp_path, wait=release)
    pool = SimpleNamespace(prepare_execution=AsyncMock(return_value=runtime))
    service = await WorkshopPrivateTextExecutionService.open_and_start(
        database,
        WorkshopRuntimePool(pool, profile_registry(101)),  # type: ignore[arg-type]
        registered_backend_ids=frozenset({"codex"}),
    )
    try:
        accepted = await service.accept(_message())
        execution = asyncio.create_task(service.execute(accepted.run.run_id))
        while not runtime.validated:
            await asyncio.sleep(0)

        cancellation = await service.request_cancellation(telegram_user_id=101, telegram_chat_id=101)
        result = await execution

        assert cancellation == CanonicalCancellationDisposition.REQUESTED
        assert runtime.cancelled is True
        assert result.disposition == CanonicalExecutionDisposition.CANCELLED
        assert result.run.status == RunStatus.CANCELLED
    finally:
        await service.stop()


async def test_owner_discovers_only_durably_accepted_workshop_client_runs(tmp_path: Path):
    database = tmp_path / "kai.db"
    await _foundation(database)
    inspection = await WorkshopEventStore.open(database)
    try:
        async with inspection.connection.execute(
            "SELECT p.id, c.id FROM principals p "
            "JOIN external_identities ei ON ei.principal_id = p.id "
            "JOIN channel_memberships cm ON cm.principal_id = p.id "
            "JOIN channels c ON c.id = cm.channel_id AND c.kind = 'direct' "
            "WHERE ei.provider = 'telegram' AND ei.external_subject = '101'"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        principal_id = PrincipalId(str(row[0]))
        channel_id = ChannelId(str(row[1]))
    finally:
        await inspection.close()

    pool = SimpleNamespace(prepare_execution=AsyncMock())
    service = await WorkshopPrivateTextExecutionService.open_and_start(
        database,
        WorkshopRuntimePool(pool, profile_registry(101)),  # type: ignore[arg-type]
        registered_backend_ids=frozenset({"codex"}),
    )
    try:
        accepted = await service.accept_client(
            ClientInboundMessage(
                principal_id=principal_id,
                channel_id=channel_id,
                client_message_id="recoverable-browser-command",
                body="Resume this browser run after restart",
                occurred_at=_NOW,
            )
        )

        assert await service.recoverable_client_runs() == (
            RecoverableClientRun(
                accepted.run.run_id,
                profile_id(101),
                "Resume this browser run after restart",
            ),
        )
    finally:
        await service.stop()
