"""Durable run_traces persistence under execution-claim ownership."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kai.backend import TraceEntry
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.domain import MessageId, RunExecutionOwnerId
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.run_execution_authority import (
    RunExecutionClaim,
    RunExecutionSelection,
    StaleRunExecutionAuthorityError,
    WorkshopRunExecutionAuthority,
)
from kai.workshop.run_lifecycle import WorkshopRunLifecycle
from kai.workshop.run_traces import (
    _TRACE_MAX_ENTRIES,
    TRACE_TRUNCATION_KIND,
    WorkshopRunTraceStore,
)
from kai.workshop.store import WorkshopEventStore

_NOW = datetime(2026, 8, 17, 18, 0, tzinfo=UTC)


async def _open_store(path: Path) -> tuple[WorkshopEventStore, WorkshopRunExecutionAuthority, MessageId]:
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
    inbound = await record_inbound_message(
        store,
        InboundMessage(
            transport="telegram",
            update_id="command-1",
            message_id="message-1",
            sender_subject="101",
            channel_subject="101",
            body="Perform one durable unit of work",
            occurred_at=_NOW,
        ),
    )
    inbound_id = MessageId(str(inbound.event.envelope.aggregate_id))
    authority = WorkshopRunExecutionAuthority(
        store,
        selection_resolver=lambda _run: RunExecutionSelection(
            backend="codex",
            provider=None,
            model="gpt-5.6-sol",
        ),
        registered_backend_ids=frozenset({"codex"}),
    )
    await WorkshopConversationDeliveryAuthority(store).activate()
    return store, authority, inbound_id


async def _granted_claim(
    store: WorkshopEventStore,
    authority: WorkshopRunExecutionAuthority,
    inbound_id: MessageId,
) -> RunExecutionClaim:
    accepted = await WorkshopRunLifecycle(store).accept(inbound_id, occurred_at=_NOW + timedelta(seconds=1))
    granted = await authority.grant(
        accepted.run.run_id,
        owner_id=RunExecutionOwnerId.new(),
        occurred_at=_NOW + timedelta(seconds=2),
        lease_expires_at=_NOW + timedelta(seconds=62),
    )
    return RunExecutionClaim.from_attempt(granted.attempt)


def _trace(index: int) -> TraceEntry:
    return TraceEntry(
        kind="tool_call",
        tool_use_id=f"toolu_{index}",
        summary=f"Bash: step {index}",
        detail=f"step {index} detail",
        tool_name="Bash",
    )


async def _rows(store: WorkshopEventStore, run_id) -> list[tuple]:
    async with store.connection.execute(
        "SELECT seq, kind, tool_use_id, summary, detail, is_diff, is_error FROM run_traces "
        "WHERE run_id = ? ORDER BY seq",
        (run_id,),
    ) as cursor:
        return list(await cursor.fetchall())


class TestRunTracePersistence:
    async def test_entries_persist_in_order_with_dense_seq(self, tmp_path: Path):
        store, authority, inbound_id = await _open_store(tmp_path / "kai.db")
        try:
            claim = await _granted_claim(store, authority, inbound_id)
            traces = WorkshopRunTraceStore(store)
            for index in range(1, 4):
                await traces.append(claim, _trace(index), occurred_at=_NOW + timedelta(seconds=2 + index))
            await traces.append(
                claim,
                TraceEntry(
                    kind="tool_result",
                    tool_use_id="toolu_3",
                    summary="done",
                    detail="output",
                    is_error=True,
                ),
                occurred_at=_NOW + timedelta(seconds=6),
            )

            rows = await _rows(store, claim.run_id)
            assert [row[0] for row in rows] == [1, 2, 3, 4]
            assert [row[1] for row in rows] == ["tool_call", "tool_call", "tool_call", "tool_result"]
            assert rows[0][2] == "toolu_1"
            assert rows[0][3] == "Bash: step 1"
            assert rows[3][6] == 1
        finally:
            await store.close()

    async def test_superseded_attempt_writes_are_rejected(self, tmp_path: Path):
        store, authority, inbound_id = await _open_store(tmp_path / "kai.db")
        try:
            first_claim = await _granted_claim(store, authority, inbound_id)
            traces = WorkshopRunTraceStore(store)
            await traces.append(first_claim, _trace(1), occurred_at=_NOW + timedelta(seconds=3))

            await authority.expire_grant(first_claim, occurred_at=_NOW + timedelta(seconds=63))
            second = await authority.grant(
                first_claim.run_id,
                owner_id=RunExecutionOwnerId.new(),
                occurred_at=_NOW + timedelta(seconds=64),
                lease_expires_at=_NOW + timedelta(seconds=124),
            )
            second_claim = RunExecutionClaim.from_attempt(second.attempt)

            with pytest.raises(StaleRunExecutionAuthorityError):
                await traces.append(first_claim, _trace(2), occurred_at=_NOW + timedelta(seconds=65))

            await traces.append(second_claim, _trace(3), occurred_at=_NOW + timedelta(seconds=66))
            rows = await _rows(store, first_claim.run_id)
            assert [row[0] for row in rows] == [1, 2]
            assert [row[2] for row in rows] == ["toolu_1", "toolu_3"]
        finally:
            await store.close()

    async def test_overflow_drops_entry_and_writes_one_truncation_marker(self, tmp_path: Path):
        store, authority, inbound_id = await _open_store(tmp_path / "kai.db")
        try:
            claim = await _granted_claim(store, authority, inbound_id)
            traces = WorkshopRunTraceStore(store)
            for index in range(1, _TRACE_MAX_ENTRIES + 1):
                await traces.append(claim, _trace(index), occurred_at=_NOW + timedelta(seconds=3))

            await traces.append(claim, _trace(_TRACE_MAX_ENTRIES + 1), occurred_at=_NOW + timedelta(seconds=4))
            await traces.append(claim, _trace(_TRACE_MAX_ENTRIES + 2), occurred_at=_NOW + timedelta(seconds=5))

            rows = await _rows(store, claim.run_id)
            assert len(rows) == _TRACE_MAX_ENTRIES + 1
            markers = [row for row in rows if row[1] == TRACE_TRUNCATION_KIND]
            assert len(markers) == 1
            assert markers[0][0] == _TRACE_MAX_ENTRIES + 1
            assert markers[0][3] == f"trace truncated at {_TRACE_MAX_ENTRIES} steps"
            dropped = [row for row in rows if row[2] == f"toolu_{_TRACE_MAX_ENTRIES + 1}"]
            assert dropped == []
        finally:
            await store.close()
