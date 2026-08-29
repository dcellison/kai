"""Atomic terminal transaction contracts for Workshop runs."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiosqlite
import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.conversation_commands import WorkshopConversationCommandService
from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.delivery_outbox import STREAMING_FINALIZATION_CONTRACT
from kai.workshop.domain import ChannelId, MessageId, PrincipalId, RunExecutionOwnerId
from kai.workshop.inbound import ClientInboundMessage, InboundMessage, record_inbound_message
from kai.workshop.outbound import (
    OutboundMessage,
    record_outbound_message_with_streaming_finalization,
)
from kai.workshop.post_run_effects import WorkshopPostRunEffectService
from kai.workshop.run_execution_authority import (
    RunAttemptStatus,
    RunExecutionClaim,
    RunExecutionSelection,
    StaleRunExecutionAuthorityError,
    WorkshopRunExecutionAuthority,
)
from kai.workshop.run_lifecycle import RunStatus, WorkshopRunLifecycle
from kai.workshop.runtime_sessions import RuntimeSessionSettlement, load_runtime_session
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore
from kai.workshop.terminal_transactions import (
    TerminalFailureCode,
    TerminalOutcome,
    TerminalTransactionCommitUncertainError,
    TerminalTransactionStateConflictError,
)
from kai.workshop.terminal_transactions import (
    WorkshopRunTerminalTransactionCoordinator as _WorkshopRunTerminalTransactionCoordinator,
)
from tests.workshop_delivery import DISABLED_DELIVERY_POLICY, TELEGRAM_DELIVERY_POLICY
from tests.workshop_profiles import profile_id

_NOW = datetime(2026, 8, 12, 21, 0, tzinfo=UTC)


def WorkshopRunTerminalTransactionCoordinator(authority):
    return _WorkshopRunTerminalTransactionCoordinator(
        authority,
        delivery_policy=TELEGRAM_DELIVERY_POLICY,
    )


async def _started_run(
    path: Path,
) -> tuple[WorkshopEventStore, WorkshopRunExecutionAuthority, RunExecutionClaim]:
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
                runtime_profile_id=profile_id(101),
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
    accepted = await WorkshopRunLifecycle(store).accept(
        inbound_id,
        occurred_at=_NOW + timedelta(seconds=1),
    )
    authority = WorkshopRunExecutionAuthority(
        store,
        selection_resolver=lambda _run: RunExecutionSelection("codex", "gpt-5.6-sol"),
        registered_backend_ids=frozenset({"codex"}),
    )
    granted = await authority.grant(
        accepted.run.run_id,
        owner_id=RunExecutionOwnerId.new(),
        occurred_at=_NOW + timedelta(seconds=2),
        lease_expires_at=_NOW + timedelta(minutes=2),
    )
    started = await authority.start(granted.claim, occurred_at=_NOW + timedelta(seconds=3))
    await WorkshopConversationDeliveryAuthority(store).activate()
    return store, authority, started.claim


async def _started_client_run(
    path: Path,
) -> tuple[WorkshopEventStore, WorkshopRunExecutionAuthority, RunExecutionClaim]:
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
                runtime_profile_id=profile_id(101),
            ),
        ),
    )
    async with store.connection.execute(
        "SELECT e.principal_id, cb.channel_id FROM external_identities e "
        "JOIN channel_bindings cb ON cb.transport = e.provider "
        "AND cb.external_channel_id = e.external_subject "
        "WHERE e.provider = 'telegram' AND e.external_subject = '101'"
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    accepted = await WorkshopConversationCommandService(
        store,
        delivery_policy=TELEGRAM_DELIVERY_POLICY,
    ).accept_client(
        ClientInboundMessage(
            principal_id=PrincipalId(str(row[0])),
            channel_id=ChannelId(str(row[1])),
            client_message_id="browser-command-1",
            body="Perform one durable unit of work",
            occurred_at=_NOW,
        )
    )
    authority = WorkshopRunExecutionAuthority(
        store,
        selection_resolver=lambda _run: RunExecutionSelection("codex", "gpt-5.6-sol"),
        registered_backend_ids=frozenset({"codex"}),
    )
    granted = await authority.grant(
        accepted.run.run_id,
        owner_id=RunExecutionOwnerId.new(),
        occurred_at=_NOW + timedelta(seconds=2),
        lease_expires_at=_NOW + timedelta(minutes=2),
    )
    started = await authority.start(granted.claim, occurred_at=_NOW + timedelta(seconds=3))
    await WorkshopConversationDeliveryAuthority(store).activate()
    return store, authority, started.claim


async def _started_workshop_only_client_run(
    path: Path,
) -> tuple[WorkshopEventStore, WorkshopRunExecutionAuthority, RunExecutionClaim]:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                display_name="Workshop Human",
                role="admin",
                transport="desktop",
                external_subject="desktop-human",
                external_channel_id="desktop-human",
                runtime_profile_id=profile_id(101),
            ),
        ),
    )
    async with store.connection.execute(
        "SELECT cm.principal_id, c.id FROM channel_memberships cm "
        "JOIN channels c ON c.id = cm.channel_id AND c.kind = 'direct' "
        "WHERE cm.role = 'owner'"
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    accepted = await WorkshopConversationCommandService(
        store,
        delivery_policy=DISABLED_DELIVERY_POLICY,
    ).accept_client(
        ClientInboundMessage(
            principal_id=PrincipalId(str(row[0])),
            channel_id=ChannelId(str(row[1])),
            client_message_id="browser-command-1",
            body="Perform one durable unit of work",
            occurred_at=_NOW,
        )
    )
    assert accepted.delivery is None
    authority = WorkshopRunExecutionAuthority(
        store,
        selection_resolver=lambda _run: RunExecutionSelection("codex", "gpt-5.6-sol"),
        registered_backend_ids=frozenset({"codex"}),
    )
    granted = await authority.grant(
        accepted.run.run_id,
        owner_id=RunExecutionOwnerId.new(),
        occurred_at=_NOW + timedelta(seconds=2),
        lease_expires_at=_NOW + timedelta(minutes=2),
    )
    started = await authority.start(granted.claim, occurred_at=_NOW + timedelta(seconds=3))
    return store, authority, started.claim


async def _terminal_rows(store: WorkshopEventStore) -> tuple[int, int, int]:
    counts = []
    for table in ("messages", "delivery_outbox", "delivery_fragments"):
        async with store.connection.execute(f"SELECT COUNT(*) FROM {table}") as cursor:
            counts.append(int((await cursor.fetchone())[0]))
    # One inbound message already exists.
    return counts[0] - 1, counts[1], counts[2]


async def _post_run_effect_count(store: WorkshopEventStore) -> int:
    async with store.connection.execute("SELECT COUNT(*) FROM workshop_post_run_effects") as cursor:
        return int((await cursor.fetchone())[0])


class TestAtomicTerminalTransactions:
    async def test_success_commits_result_delivery_plan_and_fenced_settlement(self, tmp_path: Path):
        store, authority, claim = await _started_run(tmp_path / "kai.db")
        run = await WorkshopRunLifecycle(store).state(claim.run_id)
        runtime_session = RuntimeSessionSettlement(
            channel_id=run.channel_id,
            agent_id=run.agent_id,
            runtime_profile_id=profile_id(101),
            selection=RunExecutionSelection("codex", "gpt-5.6-sol"),
            workspace="/private/tmp/kai-workshop-test-workspace",
            provider_session_id="provider-session-1",
            run_id=run.run_id,
        )
        coordinator = WorkshopRunTerminalTransactionCoordinator(authority)
        try:
            result = await coordinator.complete(
                claim,
                body="Canonical result",
                occurred_at=_NOW + timedelta(seconds=4),
                runtime_session=runtime_session,
            )
            retry = await coordinator.complete(
                claim,
                body="Canonical result",
                occurred_at=_NOW + timedelta(seconds=30),
                runtime_session=runtime_session,
            )

            assert result.outcome == TerminalOutcome.COMPLETED
            assert result.changed is True
            assert result.execution.run.status == RunStatus.COMPLETED
            assert result.execution.attempt.status == RunAttemptStatus.COMPLETED
            assert result.execution.run.result_message_id == result.finalization.message.event.envelope.aggregate_id
            assert result.finalization.delivery.delivery.authority_epoch_id is not None
            assert result.finalization.plan is None
            assert result.finalization.message.event.envelope.payload["body"] == "Canonical result"
            assert retry.changed is False
            assert retry.finalization.message.event == result.finalization.message.event
            assert retry.finalization.delivery.delivery == result.finalization.delivery.delivery
            assert retry.finalization.plan is None
            assert await _terminal_rows(store) == (1, 1, 0)
            assert await _post_run_effect_count(store) == 1
            async with store.connection.execute(
                "SELECT status, runtime_profile_id, source_message_id, result_message_id, "
                "workspace, provider_session_id, attempt_count "
                "FROM workshop_post_run_effects WHERE run_id = ?",
                (run.run_id,),
            ) as cursor:
                effect = await cursor.fetchone()
            assert tuple(effect) == (
                "pending",
                str(profile_id(101)),
                str(run.inbound_message_id),
                str(result.execution.run.result_message_id),
                "/private/tmp/kai-workshop-test-workspace",
                "provider-session-1",
                0,
            )
            async with store.connection.execute(
                "SELECT event_type FROM event_log WHERE position >= ? ORDER BY position",
                (result.finalization.message.event.position,),
            ) as cursor:
                assert [str(row[0]) for row in await cursor.fetchall()] == [
                    "message.created",
                    "delivery.requested",
                    "run_attempt.completed",
                    "run.completed",
                ]
        finally:
            await store.close()

    async def test_projection_rebuild_preserves_post_run_effect_receipt(
        self,
        tmp_path: Path,
    ):
        store, authority, claim = await _started_run(tmp_path / "kai.db")
        run = await WorkshopRunLifecycle(store).state(claim.run_id)
        try:
            result = await WorkshopRunTerminalTransactionCoordinator(authority).complete(
                claim,
                body="Canonical result before projection rebuild",
                occurred_at=_NOW + timedelta(seconds=4),
                runtime_session=RuntimeSessionSettlement(
                    channel_id=run.channel_id,
                    agent_id=run.agent_id,
                    runtime_profile_id=profile_id(101),
                    selection=RunExecutionSelection("codex", "gpt-5.6-sol"),
                    workspace="/private/tmp/kai-workshop-test-workspace",
                    provider_session_id="provider-session-before-rebuild",
                    run_id=run.run_id,
                ),
            )
            assert await _post_run_effect_count(store) == 1

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

            assert await _post_run_effect_count(store) == 1
            rebuilt = await WorkshopRunLifecycle(store).state(run.run_id)
            assert rebuilt.status == RunStatus.COMPLETED
            assert rebuilt.result_message_id == result.execution.run.result_message_id
            async with store.connection.execute(
                "SELECT status, source_message_id, result_message_id FROM workshop_post_run_effects WHERE run_id = ?",
                (run.run_id,),
            ) as cursor:
                effect = await cursor.fetchone()
            assert effect is not None
            assert tuple(effect) == (
                "pending",
                str(run.inbound_message_id),
                str(result.execution.run.result_message_id),
            )
        finally:
            await store.close()

    async def test_version_thirty_one_receipt_survives_schema_upgrade(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        from kai.workshop import schema

        database = tmp_path / "kai.db"
        with monkeypatch.context() as migration_context:
            migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 31)
            migration_context.setattr(schema, "_MIGRATIONS", schema._MIGRATIONS[:31])
            store, authority, claim = await _started_run(database)
            run = await WorkshopRunLifecycle(store).state(claim.run_id)
            result = await WorkshopRunTerminalTransactionCoordinator(authority).complete(
                claim,
                body="Canonical result before schema upgrade",
                occurred_at=_NOW + timedelta(seconds=4),
                runtime_session=RuntimeSessionSettlement(
                    channel_id=run.channel_id,
                    agent_id=run.agent_id,
                    runtime_profile_id=profile_id(101),
                    selection=RunExecutionSelection("codex", "gpt-5.6-sol"),
                    workspace="/private/tmp/kai-workshop-test-workspace",
                    provider_session_id="provider-session-before-upgrade",
                    run_id=run.run_id,
                ),
            )
            assert await store.schema_version() == 31
            assert await _post_run_effect_count(store) == 1
            await store.close()

        upgraded = await WorkshopEventStore.open(database)
        try:
            assert await upgraded.schema_version() == 46
            assert await _post_run_effect_count(upgraded) == 1
            async with upgraded.connection.execute(
                "SELECT status, source_message_id, result_message_id FROM workshop_post_run_effects WHERE run_id = ?",
                (run.run_id,),
            ) as cursor:
                effect = await cursor.fetchone()
            assert effect is not None
            assert tuple(effect) == (
                "pending",
                str(run.inbound_message_id),
                str(result.execution.run.result_message_id),
            )
            async with upgraded.connection.execute("PRAGMA foreign_key_list(workshop_post_run_effects)") as cursor:
                assert await cursor.fetchall() == []
        finally:
            await upgraded.close()

    async def test_runtime_reassignment_does_not_discard_completed_answer(self, tmp_path: Path):
        store, authority, claim = await _started_run(tmp_path / "kai.db")
        run = await WorkshopRunLifecycle(store).state(claim.run_id)
        original_profile = profile_id(101)
        try:
            await store.connection.execute(
                "UPDATE channel_agent_runtime_assignments SET runtime_profile_id = ? "
                "WHERE channel_id = ? AND agent_id = ?",
                (profile_id(202), run.channel_id, run.agent_id),
            )
            await store.connection.commit()

            result = await WorkshopRunTerminalTransactionCoordinator(authority).complete(
                claim,
                body="Answer survives reassignment",
                occurred_at=_NOW + timedelta(seconds=4),
                runtime_session=RuntimeSessionSettlement(
                    channel_id=run.channel_id,
                    agent_id=run.agent_id,
                    runtime_profile_id=original_profile,
                    selection=RunExecutionSelection("codex", "gpt-5.6-sol"),
                    workspace="/private/tmp/kai-workshop-test-workspace",
                    provider_session_id="provider-session-before-reassignment",
                    run_id=run.run_id,
                ),
            )

            assert result.outcome == TerminalOutcome.COMPLETED
            assert result.execution.run.status == RunStatus.COMPLETED
            assert result.runtime_session is None
            assert await load_runtime_session(store, run.channel_id, run.agent_id) is None
            assert await _terminal_rows(store) == (1, 1, 0)
        finally:
            await store.close()

    async def test_conflicting_runtime_session_replay_does_not_veto_result_replay(self, tmp_path: Path):
        store, authority, claim = await _started_run(tmp_path / "kai.db")
        run = await WorkshopRunLifecycle(store).state(claim.run_id)
        settlement = RuntimeSessionSettlement(
            channel_id=run.channel_id,
            agent_id=run.agent_id,
            runtime_profile_id=profile_id(101),
            selection=RunExecutionSelection("codex", "gpt-5.6-sol"),
            workspace="/private/tmp/kai-workshop-test-workspace",
            provider_session_id="provider-session-original",
            run_id=run.run_id,
        )
        coordinator = WorkshopRunTerminalTransactionCoordinator(authority)
        try:
            completed = await coordinator.complete(
                claim,
                body="Canonical result",
                occurred_at=_NOW + timedelta(seconds=4),
                runtime_session=settlement,
            )
            replay = await coordinator.complete(
                claim,
                body="Canonical result",
                occurred_at=_NOW + timedelta(seconds=30),
                runtime_session=RuntimeSessionSettlement(
                    channel_id=run.channel_id,
                    agent_id=run.agent_id,
                    runtime_profile_id=profile_id(101),
                    selection=RunExecutionSelection("codex", "gpt-5.6-sol"),
                    workspace="/private/tmp/kai-workshop-test-workspace",
                    provider_session_id="provider-session-conflict",
                    run_id=run.run_id,
                ),
            )

            assert completed.runtime_session is not None
            assert replay.changed is False
            assert replay.runtime_session is None
            session = await load_runtime_session(store, run.channel_id, run.agent_id)
            assert session is not None
            assert session.provider_session_id == "provider-session-original"
            assert await _terminal_rows(store) == (1, 1, 0)
        finally:
            await store.close()

    async def test_workshop_client_success_sends_without_telegram_preview(self, tmp_path: Path):
        store, authority, claim = await _started_client_run(tmp_path / "kai.db")
        try:
            result = await WorkshopRunTerminalTransactionCoordinator(authority).complete(
                claim,
                body="WORKSHOP-COMMAND-OK",
                occurred_at=_NOW + timedelta(seconds=4),
            )

            assert result.outcome == TerminalOutcome.COMPLETED
            assert result.execution.run.status == RunStatus.COMPLETED
            assert result.finalization.delivery.delivery.execution_contract == STREAMING_FINALIZATION_CONTRACT
            assert result.finalization.plan is None
            assert result.finalization.message.event.envelope.payload["body"] == "WORKSHOP-COMMAND-OK"
            # The browser command's attributed echo precedes the assistant
            # finalization and both remain durable outbox work.
            assert await _terminal_rows(store) == (1, 2, 0)
        finally:
            await store.close()

    async def test_workshop_only_success_commits_without_any_transport_work(
        self,
        tmp_path: Path,
    ):
        store, authority, claim = await _started_workshop_only_client_run(tmp_path / "kai.db")
        try:
            result = await WorkshopRunTerminalTransactionCoordinator(authority).complete(
                claim,
                body="WORKSHOP-ONLY-OK",
                occurred_at=_NOW + timedelta(seconds=4),
            )

            assert result.outcome == TerminalOutcome.COMPLETED
            assert result.execution.run.status == RunStatus.COMPLETED
            assert result.finalization.delivery is None
            assert result.finalization.plan is None
            assert await _terminal_rows(store) == (1, 0, 0)
            async with store.connection.execute(
                "SELECT COUNT(*) FROM channel_bindings WHERE transport = 'telegram'"
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 0
            async with store.connection.execute(
                "SELECT COUNT(*) FROM external_identities WHERE provider = 'telegram'"
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 0
        finally:
            await store.close()

    async def test_failure_commits_only_typed_sanitized_visible_outcome(self, tmp_path: Path):
        store, authority, claim = await _started_run(tmp_path / "kai.db")
        try:
            result = await WorkshopRunTerminalTransactionCoordinator(authority).fail(
                claim,
                failure_code=TerminalFailureCode.AUTHENTICATION_EXPIRED,
                occurred_at=_NOW + timedelta(seconds=4),
            )

            assert result.outcome == TerminalOutcome.FAILED
            assert result.execution.run.status == RunStatus.FAILED
            assert result.execution.run.terminal_code == "authentication_expired"
            assert result.execution.attempt.status == RunAttemptStatus.FAILED
            assert result.finalization.plan is None
            assert result.finalization.message.event.envelope.payload["body"] == (
                "Authentication for the configured agent has expired. Kai did not complete this request."
            )
            assert await _post_run_effect_count(store) == 0
            with pytest.raises(ValueError, match="TerminalFailureCode"):
                await WorkshopRunTerminalTransactionCoordinator(authority).fail(
                    claim,
                    failure_code="native provider payload",  # type: ignore[arg-type]
                    occurred_at=_NOW + timedelta(seconds=5),
                )
        finally:
            await store.close()

    async def test_confirmed_cancellation_commits_request_code_and_fixed_outcome(self, tmp_path: Path):
        store, authority, claim = await _started_run(tmp_path / "kai.db")
        try:
            await authority.request_cancellation(
                claim.run_id,
                cancellation_code="requested_by_human",
                occurred_at=_NOW + timedelta(seconds=4),
            )
            result = await WorkshopRunTerminalTransactionCoordinator(authority).confirm_cancellation(
                claim,
                occurred_at=_NOW + timedelta(seconds=5),
            )

            assert result.outcome == TerminalOutcome.CANCELLED
            assert result.execution.run.status == RunStatus.CANCELLED
            assert result.execution.run.terminal_code == "requested_by_human"
            assert result.execution.attempt.status == RunAttemptStatus.CANCELLED
            assert result.finalization.plan is None
            assert result.finalization.message.event.envelope.payload["body"] == "This request was cancelled."
            assert await _post_run_effect_count(store) == 0
        finally:
            await store.close()

    async def test_post_run_worker_ingests_one_non_telegram_success_exactly_once(
        self,
        tmp_path: Path,
    ):
        database = tmp_path / "kai.db"
        store, authority, claim = await _started_workshop_only_client_run(database)
        run = await WorkshopRunLifecycle(store).state(claim.run_id)
        result = await WorkshopRunTerminalTransactionCoordinator(authority).complete(
            claim,
            body="Transport-neutral answer",
            occurred_at=_NOW + timedelta(seconds=4),
            runtime_session=RuntimeSessionSettlement(
                channel_id=run.channel_id,
                agent_id=run.agent_id,
                runtime_profile_id=profile_id(101),
                selection=RunExecutionSelection("codex", "gpt-5.6-sol"),
                workspace="/private/tmp/non-telegram-workspace",
                provider_session_id="non-telegram-provider-session",
                run_id=run.run_id,
            ),
        )
        await store.close()

        profile_state = SimpleNamespace(
            memory_context_turns=8,
            has_memory_for_run=AsyncMock(return_value=False),
            ingest_memory=AsyncMock(),
        )
        compatibility = SimpleNamespace(for_profile=lambda _profile_id: profile_state)
        service = await WorkshopPostRunEffectService.open_and_start(
            database,
            compatibility,  # type: ignore[arg-type]
        )
        try:
            for _ in range(100):
                if (await service.readiness()).succeeded == 1:
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("post-run effect did not settle")
        finally:
            await service.stop()

        profile_state.has_memory_for_run.assert_awaited_once_with(str(run.run_id))
        profile_state.ingest_memory.assert_awaited_once()
        call = profile_state.ingest_memory.await_args.kwargs
        assert call["prompt"] == "Perform one durable unit of work"
        assert call["assistant_text"] == "Transport-neutral answer"
        assert call["session_id"] == "non-telegram-provider-session"
        assert call["workspace"] == "/private/tmp/non-telegram-workspace"
        assert call["canonical_provenance"].run_id == run.run_id
        assert call["canonical_provenance"].source_message_id == run.inbound_message_id
        assert call["canonical_provenance"].result_message_id == result.execution.run.result_message_id

        restarted = await WorkshopPostRunEffectService.open_and_start(
            database,
            compatibility,  # type: ignore[arg-type]
        )
        try:
            await asyncio.sleep(0.05)
            assert (await restarted.readiness()).succeeded == 1
        finally:
            await restarted.stop()
        profile_state.ingest_memory.assert_awaited_once()

    async def test_recovered_post_run_effect_does_not_duplicate_committed_memory(
        self,
        tmp_path: Path,
    ):
        database = tmp_path / "kai.db"
        store, authority, claim = await _started_run(database)
        run = await WorkshopRunLifecycle(store).state(claim.run_id)
        await WorkshopRunTerminalTransactionCoordinator(authority).complete(
            claim,
            body="Committed before interruption",
            occurred_at=_NOW + timedelta(seconds=4),
            runtime_session=RuntimeSessionSettlement(
                channel_id=run.channel_id,
                agent_id=run.agent_id,
                runtime_profile_id=profile_id(101),
                selection=RunExecutionSelection("codex", "gpt-5.6-sol"),
                workspace="/private/tmp/kai-workshop-test-workspace",
                provider_session_id="provider-session-before-restart",
                run_id=run.run_id,
            ),
        )
        await store.connection.execute(
            "UPDATE workshop_post_run_effects SET status = 'executing' WHERE run_id = ?",
            (run.run_id,),
        )
        await store.connection.commit()
        await store.close()

        profile_state = SimpleNamespace(
            memory_context_turns=8,
            has_memory_for_run=AsyncMock(return_value=True),
            ingest_memory=AsyncMock(),
        )
        compatibility = SimpleNamespace(for_profile=lambda _profile_id: profile_state)
        service = await WorkshopPostRunEffectService.open_and_start(
            database,
            compatibility,  # type: ignore[arg-type]
        )
        try:
            for _ in range(100):
                if (await service.readiness()).succeeded == 1:
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("recovered post-run effect did not settle")
        finally:
            await service.stop()

        profile_state.has_memory_for_run.assert_awaited_once_with(str(run.run_id))
        profile_state.ingest_memory.assert_not_awaited()

    async def test_stale_fence_rolls_back_every_visible_outcome_row(self, tmp_path: Path):
        store, authority, stale_claim = await _started_run(tmp_path / "kai.db")
        run = await WorkshopRunLifecycle(store).state(stale_claim.run_id)
        try:
            renewed = await authority.renew(
                stale_claim,
                occurred_at=_NOW + timedelta(seconds=4),
                lease_expires_at=_NOW + timedelta(minutes=3),
            )
            with pytest.raises(StaleRunExecutionAuthorityError, match="stale lease version"):
                await WorkshopRunTerminalTransactionCoordinator(authority).complete(
                    stale_claim,
                    body="Must roll back",
                    occurred_at=_NOW + timedelta(seconds=5),
                    runtime_session=RuntimeSessionSettlement(
                        channel_id=run.channel_id,
                        agent_id=run.agent_id,
                        runtime_profile_id=profile_id(101),
                        selection=RunExecutionSelection("codex", "gpt-5.6-sol"),
                        workspace="/private/tmp/kai-workshop-test-workspace",
                        provider_session_id="stale-provider-session",
                        run_id=run.run_id,
                    ),
                )

            assert await _terminal_rows(store) == (0, 0, 0)
            assert await _post_run_effect_count(store) == 0
            assert (await WorkshopRunLifecycle(store).state(stale_claim.run_id)).status == RunStatus.STARTED
            assert (await authority.attempt(renewed.claim.attempt_id)).status == RunAttemptStatus.STARTED
        finally:
            await store.close()

    async def test_adapter_plan_storage_is_not_part_of_core_terminal_transaction(self, tmp_path: Path):
        store, authority, claim = await _started_run(tmp_path / "kai.db")
        try:
            await store.connection.execute(
                "CREATE TRIGGER reject_terminal_plan BEFORE INSERT ON delivery_fragments "
                "BEGIN SELECT RAISE(ABORT, 'terminal plan rejected'); END"
            )
            await store.connection.commit()

            result = await WorkshopRunTerminalTransactionCoordinator(authority).complete(
                claim,
                body="Core-owned result",
                occurred_at=_NOW + timedelta(seconds=4),
            )

            assert result.outcome == TerminalOutcome.COMPLETED
            assert result.finalization.plan is None
            assert await _terminal_rows(store) == (1, 1, 0)
        finally:
            await store.close()

    async def test_preexisting_visible_outcome_without_settlement_is_rejected(self, tmp_path: Path):
        store, authority, claim = await _started_run(tmp_path / "kai.db")
        try:
            run = await WorkshopRunLifecycle(store).state(claim.run_id)
            await record_outbound_message_with_streaming_finalization(
                store,
                OutboundMessage(
                    in_reply_to_message_id=run.inbound_message_id,
                    body="Half state",
                    occurred_at=_NOW + timedelta(seconds=4),
                ),
                delivery_policy=TELEGRAM_DELIVERY_POLICY,
            )

            with pytest.raises(TerminalTransactionStateConflictError, match="share one prior state"):
                await WorkshopRunTerminalTransactionCoordinator(authority).complete(
                    claim,
                    body="Half state",
                    occurred_at=_NOW + timedelta(seconds=5),
                )

            assert (await WorkshopRunLifecycle(store).state(claim.run_id)).status == RunStatus.STARTED
            assert (await authority.attempt(claim.attempt_id)).status == RunAttemptStatus.STARTED
            assert await _terminal_rows(store) == (1, 1, 0)
        finally:
            await store.close()

    async def test_first_terminal_outcome_wins_without_persisting_loser(self, tmp_path: Path):
        store, authority, claim = await _started_run(tmp_path / "kai.db")
        coordinator = WorkshopRunTerminalTransactionCoordinator(authority)
        try:
            completed = await coordinator.complete(
                claim,
                body="Winning result",
                occurred_at=_NOW + timedelta(seconds=4),
            )

            with pytest.raises(IdempotencyConflictError):
                await coordinator.fail(
                    claim,
                    failure_code=TerminalFailureCode.TRANSIENT,
                    occurred_at=_NOW + timedelta(seconds=5),
                )

            run = await WorkshopRunLifecycle(store).state(claim.run_id)
            assert run.status == RunStatus.COMPLETED
            assert run.result_message_id == completed.finalization.message.event.envelope.aggregate_id
            assert await _terminal_rows(store) == (1, 1, 0)
            async with store.connection.execute(
                "SELECT COUNT(*) FROM event_log WHERE event_type IN ('run.failed', 'run_attempt.failed')"
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 0
        finally:
            await store.close()

    async def test_commit_result_loss_is_resolved_by_deterministic_retry(self, tmp_path: Path, monkeypatch):
        store, authority, claim = await _started_run(tmp_path / "kai.db")
        coordinator = WorkshopRunTerminalTransactionCoordinator(authority)
        original_commit = aiosqlite.Connection.commit
        commit_calls = 0

        async def commit_then_lose_result(connection: aiosqlite.Connection) -> None:
            nonlocal commit_calls
            await original_commit(connection)
            commit_calls += 1
            if connection is store.connection and commit_calls == 1:
                raise aiosqlite.OperationalError("commit result lost")

        try:
            monkeypatch.setattr(aiosqlite.Connection, "commit", commit_then_lose_result)
            result = await coordinator.complete(
                claim,
                body="Committed exactly once",
                occurred_at=_NOW + timedelta(seconds=4),
            )

            assert result.changed is False
            assert commit_calls == 2
            assert await _terminal_rows(store) == (1, 1, 0)
            assert result.execution.run.status == RunStatus.COMPLETED
        finally:
            await store.close()

    async def test_second_commit_resolution_failure_is_explicitly_uncertain(self, tmp_path: Path, monkeypatch):
        store, authority, claim = await _started_run(tmp_path / "kai.db")
        run = await WorkshopRunLifecycle(store).state(claim.run_id)
        coordinator = WorkshopRunTerminalTransactionCoordinator(authority)

        async def unavailable_commit(_connection: aiosqlite.Connection) -> None:
            raise aiosqlite.OperationalError("commit unavailable")

        try:
            monkeypatch.setattr(aiosqlite.Connection, "commit", unavailable_commit)
            with pytest.raises(TerminalTransactionCommitUncertainError):
                await coordinator.complete(
                    claim,
                    body="Never committed",
                    occurred_at=_NOW + timedelta(seconds=4),
                    runtime_session=RuntimeSessionSettlement(
                        channel_id=run.channel_id,
                        agent_id=run.agent_id,
                        runtime_profile_id=profile_id(101),
                        selection=RunExecutionSelection("codex", "gpt-5.6-sol"),
                        workspace="/private/tmp/kai-workshop-test-workspace",
                        provider_session_id="uncertain-provider-session",
                        run_id=run.run_id,
                    ),
                )

            assert await _terminal_rows(store) == (0, 0, 0)
            assert await _post_run_effect_count(store) == 0
        finally:
            await store.close()

    def test_coordinator_remains_unregistered(self):
        source_root = Path(__file__).parents[1] / "src" / "kai"
        for relative_path in ("main.py", "bot.py", "sessions.py"):
            source = (source_root / relative_path).read_text(encoding="utf-8")
            assert "WorkshopRunTerminalTransactionCoordinator" not in source
