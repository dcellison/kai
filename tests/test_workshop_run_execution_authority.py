"""Contracts for production-unused fenced Workshop run execution authority."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.domain import MessageId, RunExecutionOwnerId
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.outbound import OutboundMessage, record_outbound_message
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.run_execution_authority import (
    RunAttemptStatus,
    RunExecutionAuthorityError,
    RunExecutionClaim,
    RunExecutionConflictError,
    RunExecutionSelection,
    StaleRunExecutionAuthorityError,
    WorkshopRunExecutionAuthority,
)
from kai.workshop.run_lifecycle import RunStatus, WorkshopRunLifecycle
from kai.workshop.store import WorkshopEventStore

_NOW = datetime(2026, 8, 12, 18, 0, tzinfo=UTC)


async def _open_authority(
    path: Path,
) -> tuple[WorkshopEventStore, WorkshopRunExecutionAuthority, MessageId]:
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
        registered_backend_ids=frozenset({"codex", "pi"}),
    )
    await WorkshopConversationDeliveryAuthority(store).activate()
    return store, authority, inbound_id


async def _accepted(
    store: WorkshopEventStore,
    inbound_id: MessageId,
    *,
    offset: int = 1,
):
    return await WorkshopRunLifecycle(store).accept(inbound_id, occurred_at=_NOW + timedelta(seconds=offset))


async def _grant(
    authority: WorkshopRunExecutionAuthority,
    run_id,
    *,
    owner_id: RunExecutionOwnerId | None = None,
    offset: int = 2,
):
    return await authority.grant(
        run_id,
        owner_id=owner_id or RunExecutionOwnerId.new(),
        occurred_at=_NOW + timedelta(seconds=offset),
        lease_expires_at=_NOW + timedelta(seconds=offset + 60),
    )


class TestRunExecutionGrant:
    @pytest.mark.parametrize("model", ["/usr/local/bin/codex", "model with prose", "../model"])
    def test_selection_rejects_paths_and_non_identifiers(self, model: str):
        with pytest.raises(ValueError, match="model identifier"):
            RunExecutionSelection("codex", model)

    async def test_grant_persists_protected_selection_and_fence(self, tmp_path: Path):
        store, authority, inbound_id = await _open_authority(tmp_path / "kai.db")
        try:
            accepted = await _accepted(store, inbound_id)
            granted = await _grant(authority, accepted.run.run_id)

            assert granted.changed is True
            assert granted.run.status == RunStatus.ACCEPTED
            assert granted.attempt.status == RunAttemptStatus.GRANTED
            assert granted.attempt.attempt_sequence == 1
            assert granted.attempt.fence_token == 1
            assert granted.attempt.lease_version == 1
            assert granted.attempt.selection == RunExecutionSelection("codex", "gpt-5.6-sol")
            assert granted.attempt.execution_contract == "trusted_host_compatibility_v1"
        finally:
            await store.close()

    async def test_same_owner_retries_but_second_owner_is_rejected(self, tmp_path: Path):
        store, authority, inbound_id = await _open_authority(tmp_path / "kai.db")
        try:
            accepted = await _accepted(store, inbound_id)
            owner = RunExecutionOwnerId.new()
            first = await _grant(authority, accepted.run.run_id, owner_id=owner)
            retry = await _grant(authority, accepted.run.run_id, owner_id=owner, offset=3)

            assert retry.changed is False
            assert retry.attempt == first.attempt
            with pytest.raises(RunExecutionConflictError, match="active execution owner"):
                await _grant(authority, accepted.run.run_id, offset=4)
        finally:
            await store.close()

    async def test_pre_dispatch_cancellation_is_atomic_and_blocks_grant(self, tmp_path: Path):
        store, authority, inbound_id = await _open_authority(tmp_path / "kai.db")
        try:
            accepted = await _accepted(store, inbound_id)
            cancelled = await authority.cancel_before_dispatch(
                accepted.run.run_id,
                cancellation_code="requested_by_human",
                occurred_at=_NOW + timedelta(seconds=2),
            )

            assert cancelled.status == RunStatus.CANCELLED
            assert cancelled.cancellation_code == "requested_by_human"
            assert cancelled.terminal_code == "requested_by_human"
            with pytest.raises(RunExecutionConflictError, match="uncancelled accepted"):
                await _grant(authority, accepted.run.run_id, offset=3)
        finally:
            await store.close()

    async def test_unregistered_resolved_backend_fails_closed(self, tmp_path: Path):
        store, _, inbound_id = await _open_authority(tmp_path / "kai.db")
        try:
            accepted = await _accepted(store, inbound_id)
            authority = WorkshopRunExecutionAuthority(
                store,
                selection_resolver=lambda _run: RunExecutionSelection("goose", "default"),
                registered_backend_ids=frozenset({"codex"}),
            )
            with pytest.raises(RunExecutionConflictError, match="protected registry"):
                await _grant(authority, accepted.run.run_id)
        finally:
            await store.close()


class TestRunExecutionFence:
    async def test_start_is_atomic_and_fenced(self, tmp_path: Path):
        store, authority, inbound_id = await _open_authority(tmp_path / "kai.db")
        try:
            accepted = await _accepted(store, inbound_id)
            granted = await _grant(authority, accepted.run.run_id)
            started = await authority.start(granted.claim, occurred_at=_NOW + timedelta(seconds=3))

            assert started.run.status == RunStatus.STARTED
            assert started.attempt.status == RunAttemptStatus.STARTED
            assert [event.envelope.event_type for event in started.events] == [
                "run_attempt.started",
                "run.started",
            ]
            retry = await authority.start(granted.claim, occurred_at=_NOW + timedelta(seconds=30))
            assert retry.changed is False

            stale = RunExecutionClaim(
                attempt_id=granted.claim.attempt_id,
                run_id=granted.claim.run_id,
                owner_id=RunExecutionOwnerId.new(),
                fence_token=granted.claim.fence_token,
                lease_version=granted.claim.lease_version,
            )
            with pytest.raises(StaleRunExecutionAuthorityError, match="fenced owner"):
                await authority.fail(
                    stale,
                    failure_code="backend_unavailable",
                    occurred_at=_NOW + timedelta(seconds=4),
                )
        finally:
            await store.close()

    async def test_renewal_advances_version_and_rejects_old_claim_for_new_work(self, tmp_path: Path):
        store, authority, inbound_id = await _open_authority(tmp_path / "kai.db")
        try:
            accepted = await _accepted(store, inbound_id)
            granted = await _grant(authority, accepted.run.run_id)
            new_expiry = _NOW + timedelta(seconds=100)
            renewed = await authority.renew(
                granted.claim,
                occurred_at=_NOW + timedelta(seconds=3),
                lease_expires_at=new_expiry,
            )
            retry = await authority.renew(
                granted.claim,
                occurred_at=_NOW + timedelta(seconds=4),
                lease_expires_at=new_expiry,
            )

            assert renewed.attempt.lease_version == 2
            assert retry.changed is False
            with pytest.raises(StaleRunExecutionAuthorityError, match="stale lease version"):
                await authority.start(granted.claim, occurred_at=_NOW + timedelta(seconds=5))
            started = await authority.start(renewed.claim, occurred_at=_NOW + timedelta(seconds=5))
            assert started.run.status == RunStatus.STARTED
        finally:
            await store.close()


class TestRunExecutionTerminalFacts:
    async def test_completion_references_canonical_agent_result(self, tmp_path: Path):
        store, authority, inbound_id = await _open_authority(tmp_path / "kai.db")
        try:
            accepted = await _accepted(store, inbound_id)
            granted = await _grant(authority, accepted.run.run_id)
            started = await authority.start(granted.claim, occurred_at=_NOW + timedelta(seconds=3))
            outbound = await record_outbound_message(
                store,
                OutboundMessage(
                    in_reply_to_message_id=inbound_id,
                    body="Canonical result",
                    occurred_at=_NOW + timedelta(seconds=4),
                ),
            )
            result_id = MessageId(str(outbound.event.envelope.aggregate_id))

            completed = await authority.complete(
                started.claim,
                result_message_id=result_id,
                occurred_at=_NOW + timedelta(seconds=5),
            )

            assert completed.run.status == RunStatus.COMPLETED
            assert completed.run.result_message_id == result_id
            assert completed.attempt.status == RunAttemptStatus.COMPLETED
        finally:
            await store.close()

    async def test_cancellation_intent_is_separate_from_confirmation(self, tmp_path: Path):
        store, authority, inbound_id = await _open_authority(tmp_path / "kai.db")
        try:
            accepted = await _accepted(store, inbound_id)
            granted = await _grant(authority, accepted.run.run_id)
            requested, _, changed = await authority.request_cancellation(
                accepted.run.run_id,
                cancellation_code="requested_by_human",
                occurred_at=_NOW + timedelta(seconds=3),
            )

            assert changed is True
            assert requested.status == RunStatus.ACCEPTED
            assert requested.cancellation_code == "requested_by_human"
            with pytest.raises(RunExecutionConflictError, match="cancellation was requested"):
                await authority.start(granted.claim, occurred_at=_NOW + timedelta(seconds=4))

            cancelled = await authority.confirm_cancellation(
                granted.claim,
                occurred_at=_NOW + timedelta(seconds=4),
            )
            assert cancelled.run.status == RunStatus.CANCELLED
            assert cancelled.run.terminal_code == "requested_by_human"
            assert cancelled.attempt.status == RunAttemptStatus.CANCELLED
        finally:
            await store.close()

    async def test_one_terminal_transition_wins_cancellation_completion_race(self, tmp_path: Path):
        store, authority, inbound_id = await _open_authority(tmp_path / "kai.db")
        try:
            accepted = await _accepted(store, inbound_id)
            granted = await _grant(authority, accepted.run.run_id)
            started = await authority.start(granted.claim, occurred_at=_NOW + timedelta(seconds=3))
            await authority.request_cancellation(
                accepted.run.run_id,
                cancellation_code="requested_by_human",
                occurred_at=_NOW + timedelta(seconds=4),
            )
            outbound = await record_outbound_message(
                store,
                OutboundMessage(inbound_id, "Already finished", _NOW + timedelta(seconds=5)),
            )
            completed = await authority.complete(
                started.claim,
                result_message_id=MessageId(str(outbound.event.envelope.aggregate_id)),
                occurred_at=_NOW + timedelta(seconds=6),
            )

            with pytest.raises(StaleRunExecutionAuthorityError, match="active authority"):
                await authority.confirm_cancellation(
                    started.claim,
                    occurred_at=_NOW + timedelta(seconds=7),
                )
            assert completed.run.status == RunStatus.COMPLETED
            assert (await WorkshopRunLifecycle(store).state(accepted.run.run_id)).status == RunStatus.COMPLETED
        finally:
            await store.close()

    async def test_terminal_retry_with_different_code_fails_closed(self, tmp_path: Path):
        store, authority, inbound_id = await _open_authority(tmp_path / "kai.db")
        try:
            accepted = await _accepted(store, inbound_id)
            granted = await _grant(authority, accepted.run.run_id)
            started = await authority.start(granted.claim, occurred_at=_NOW + timedelta(seconds=3))
            await authority.fail(
                started.claim,
                failure_code="backend_unavailable",
                occurred_at=_NOW + timedelta(seconds=4),
            )

            with pytest.raises(RunExecutionConflictError, match="conflicting facts"):
                await authority.fail(
                    started.claim,
                    failure_code="authentication_required",
                    occurred_at=_NOW + timedelta(seconds=5),
                )
        finally:
            await store.close()


class TestRunExecutionRecovery:
    async def test_expired_grant_can_retry_but_interrupted_execution_cannot(self, tmp_path: Path):
        store, authority, inbound_id = await _open_authority(tmp_path / "pre.db")
        try:
            accepted = await _accepted(store, inbound_id)
            first = await _grant(authority, accepted.run.run_id)
            recovery = await authority.recover_expired(occurred_at=_NOW + timedelta(seconds=63))
            second = await _grant(authority, accepted.run.run_id, offset=64)

            assert recovery.expired_before_dispatch == 1
            assert recovery.interrupted_after_dispatch == 0
            assert (await authority.attempt(first.claim.attempt_id)).status == RunAttemptStatus.EXPIRED
            assert second.attempt.attempt_sequence == 2
            assert second.attempt.fence_token == 2
            with pytest.raises(StaleRunExecutionAuthorityError, match="active authority"):
                await authority.fail(
                    first.claim,
                    failure_code="backend_unavailable",
                    occurred_at=_NOW + timedelta(seconds=65),
                )
        finally:
            await store.close()

        store, authority, inbound_id = await _open_authority(tmp_path / "post.db")
        try:
            accepted = await _accepted(store, inbound_id)
            granted = await _grant(authority, accepted.run.run_id)
            await authority.start(granted.claim, occurred_at=_NOW + timedelta(seconds=3))
            recovery = await authority.recover_expired(occurred_at=_NOW + timedelta(seconds=63))

            assert recovery.expired_before_dispatch == 0
            assert recovery.interrupted_after_dispatch == 1
            assert (await WorkshopRunLifecycle(store).state(accepted.run.run_id)).status == RunStatus.FAILED
            with pytest.raises(RunExecutionAuthorityError, match="accepted run"):
                await _grant(authority, accepted.run.run_id, offset=64)
        finally:
            await store.close()

    async def test_rebuild_restores_attempt_and_run_authority_state(self, tmp_path: Path):
        store, authority, inbound_id = await _open_authority(tmp_path / "kai.db")
        try:
            accepted = await _accepted(store, inbound_id)
            granted = await _grant(authority, accepted.run.run_id)
            started = await authority.start(granted.claim, occurred_at=_NOW + timedelta(seconds=3))
            await store.connection.execute("DELETE FROM run_attempts")
            await store.connection.execute("DELETE FROM runs")
            await store.connection.commit()

            checkpoint = await store.rebuild_projection(CanonicalConversationProjection())

            assert checkpoint.version == 8
            assert (await authority.attempt(started.claim.attempt_id)).status == RunAttemptStatus.STARTED
            assert (await WorkshopRunLifecycle(store).state(accepted.run.run_id)).status == RunStatus.STARTED
        finally:
            await store.close()


class TestRunExecutionMigration:
    async def test_version_fifteen_database_adds_execution_authority_projection(self, tmp_path: Path, monkeypatch):
        from kai.workshop import schema

        path = tmp_path / "kai.db"
        with monkeypatch.context() as migration_context:
            migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 15)
            migration_context.setattr(schema, "_MIGRATIONS", schema._MIGRATIONS[:15])
            version_fifteen = await WorkshopEventStore.open(path)
            await version_fifteen.close()

        upgraded = await WorkshopEventStore.open(path)
        try:
            assert await upgraded.schema_version() == 19
            assert "run_attempts" in await upgraded.schema_tables()
            async with upgraded.connection.execute("PRAGMA table_info(runs)") as cursor:
                columns = {str(row[1]) for row in await cursor.fetchall()}
            assert {"cancellation_requested_at", "cancellation_code", "result_message_id"} <= columns
        finally:
            await upgraded.close()
