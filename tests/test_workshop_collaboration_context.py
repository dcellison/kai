"""Security and snapshot contracts for bounded Workshop collaboration reads."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.collaboration_authority import (
    CollaborationBaseIdentity,
    CollaborationOperation,
    CollaborationOwnerPolicy,
    CollaborationProofError,
    WorkshopCollaborationAuthority,
)
from kai.workshop.collaboration_context import WorkshopCollaborationContextService
from kai.workshop.conversation_commands import WorkshopConversationCommandService
from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.domain import RunExecutionOwnerId, RuntimeProfileId
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.message_reactions import set_message_reaction
from kai.workshop.run_execution_authority import RunExecutionSelection, WorkshopRunExecutionAuthority
from kai.workshop.store import WorkshopEventStore
from kai.workshop.timeline import TimelineCursorError
from tests.test_workshop_wake_policy import _accepted_message_id, _message, _open_group_store

_NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
_PROFILE_ID = RuntimeProfileId.new()


class _Execution:
    def __init__(self, authority: WorkshopCollaborationAuthority) -> None:
        self.authority = authority

    async def authorize_collaboration(self, *args, **kwargs):
        return await self.authority.authorize(*args, **kwargs)


async def _running_context(path: Path):
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                "Workshop Human",
                "admin",
                "telegram",
                "101",
                "101",
                _PROFILE_ID,
            ),
        ),
    )
    for ordinal in range(24):
        await record_inbound_message(
            store,
            InboundMessage(
                "telegram",
                f"context-prior-{ordinal}",
                f"context-prior-message-{ordinal}",
                "101",
                "101",
                "x" * 5_000 if ordinal == 23 else f"Prior context {ordinal}",
                _NOW + timedelta(seconds=ordinal),
            ),
        )
    accepted = await WorkshopConversationCommandService(store).accept(
        InboundMessage(
            "telegram",
            "context-command",
            "context-command-message",
            "101",
            "101",
            "Current bounded request",
            _NOW + timedelta(seconds=30),
        )
    )
    await WorkshopConversationDeliveryAuthority(store).activate()
    execution_authority = WorkshopRunExecutionAuthority(
        store,
        selection_resolver=lambda _run: RunExecutionSelection("codex", "gpt-5.6-sol"),
        registered_backend_ids=frozenset({"codex"}),
    )
    granted = await execution_authority.grant(
        accepted.run.run_id,
        owner_id=RunExecutionOwnerId.new(),
        occurred_at=_NOW + timedelta(seconds=31),
        lease_expires_at=_NOW + timedelta(minutes=5),
    )
    started = await execution_authority.start(granted.claim, occurred_at=_NOW + timedelta(seconds=32))
    assert started.run.agent_definition_revision_id is not None
    await store.connection.execute(
        "UPDATE agent_definition_revisions SET collaboration_operations_json = ? WHERE id = ?",
        ('["context_read"]', started.run.agent_definition_revision_id),
    )
    await store.connection.commit()
    collaboration_authority = WorkshopCollaborationAuthority(
        store,
        owner_policy_resolver=lambda _revision: CollaborationOwnerPolicy(
            version=1,
            allowed_operations=frozenset({CollaborationOperation.CONTEXT_READ}),
        ),
        token_factory=lambda: "context-proof-000000000000000000000000000001",
    )
    await set_message_reaction(
        store,
        principal_id=started.run.requested_by_principal_id,
        channel_id=started.run.channel_id,
        message_id=started.run.inbound_message_id,
        reaction="thumbs_up",
        active=True,
        occurred_at=_NOW + timedelta(seconds=32, milliseconds=500),
    )
    grant, invocation = await collaboration_authority.issue(
        started.claim,
        occurred_at=_NOW + timedelta(seconds=33),
    )
    assert started.run.runtime_profile_id is not None
    identity = CollaborationBaseIdentity(
        started.run.requested_by_principal_id,
        started.run.channel_id,
        started.run.agent_id,
        started.run.runtime_profile_id,
    )
    return (
        store,
        started,
        grant,
        invocation,
        identity,
        WorkshopCollaborationContextService(
            store,
            _Execution(collaboration_authority),
            clock=lambda: _NOW + timedelta(seconds=34),
        ),
        collaboration_authority,
    )


async def test_context_read_is_bounded_to_grant_snapshot_and_pages_stably(tmp_path: Path) -> None:
    store, started, grant, invocation, identity, service, _authority = await _running_context(tmp_path / "kai.db")
    try:
        await record_inbound_message(
            store,
            InboundMessage(
                "telegram",
                "context-future",
                "context-future-message",
                "101",
                "101",
                "Message created after grant issuance",
                _NOW + timedelta(seconds=34),
            ),
        )
        await set_message_reaction(
            store,
            principal_id=started.run.requested_by_principal_id,
            channel_id=started.run.channel_id,
            message_id=started.run.inbound_message_id,
            reaction="thumbs_up",
            active=False,
            occurred_at=_NOW + timedelta(seconds=35),
        )
        first = await service.read(
            identity,
            proof=invocation.token,
            cursor=None,
            limit=10,
            idempotency_key="context-page-1",
        )
        assert first.payload["content_trust"] == "untrusted"
        assert first.payload["context_kind"] == "channel"
        assert first.payload["roster"]
        assert invocation.token not in str(first.payload)
        assert first.payload["snapshot_through_event_position"] == await _issued_position(store, grant.grant_id)
        messages = first.payload["messages"]
        assert isinstance(messages, list)
        assert len(messages) == 10
        assert messages[-1]["body"] == "Current bounded request"
        assert len(messages[-2]["body"]) == 4_000
        assert messages[-2]["body_truncated"] is True
        assert first.payload["truncated"] is True
        assert messages[-1]["reactions"] == [{"reaction": "thumbs_up", "count": 1, "reacted_by_requester": True}]
        assert all(item["body"] != "Message created after grant issuance" for item in messages)
        cursor = first.payload["next_cursor"]
        assert isinstance(cursor, str)

        second = await service.read(
            identity,
            proof=invocation.token,
            cursor=cursor,
            limit=10,
            idempotency_key="context-page-2",
        )
        second_messages = second.payload["messages"]
        assert isinstance(second_messages, list)
        assert len(second_messages) == 10
        assert {item["message_id"] for item in messages}.isdisjoint(item["message_id"] for item in second_messages)
        assert second.payload["snapshot_through_event_position"] == first.payload["snapshot_through_event_position"]

        # Simulate an interruption after the authority decision but before its
        # trace became durable. The exact replay restores the missing trace and
        # remains idempotent once it exists.
        await store.connection.execute(
            "DELETE FROM run_traces WHERE run_id = ? AND tool_use_id = ?",
            (started.run.run_id, "context-read:context-page-1"),
        )
        await store.connection.commit()
        replay = await service.read(
            identity,
            proof=invocation.token,
            cursor=None,
            limit=10,
            idempotency_key="context-page-1",
        )
        assert replay.payload == first.payload
        replay_again = await service.read(
            identity,
            proof=invocation.token,
            cursor=None,
            limit=10,
            idempotency_key="context-page-1",
        )
        assert replay_again.payload == first.payload

        async with store.connection.execute(
            "SELECT summary, detail FROM run_traces WHERE run_id = ? ORDER BY seq",
            (started.run.run_id,),
        ) as trace_cursor:
            trace_rows = list(await trace_cursor.fetchall())
        assert [str(row[0]) for row in trace_rows] == [
            "Read bounded Workshop context",
            "Read bounded Workshop context",
        ]
        assert all(str(grant.grant_id) in str(row[1]) for row in trace_rows)
        assert all(invocation.token not in str(row[1]) for row in trace_rows)
    finally:
        await store.close()


async def test_context_read_rejects_cursor_from_another_snapshot(tmp_path: Path) -> None:
    store, _started, _grant, invocation, identity, service, _authority = await _running_context(tmp_path / "kai.db")
    try:
        first = await service.read(
            identity,
            proof=invocation.token,
            cursor=None,
            limit=5,
            idempotency_key="context-cursor-source",
        )
        cursor = first.payload["next_cursor"]
        assert isinstance(cursor, str)
        # The opaque cursor is tied to both the exact channel and the immutable
        # grant snapshot; malformed or foreign state cannot select context.
        with pytest.raises(TimelineCursorError, match="cursor"):
            await service.read(
                identity,
                proof=invocation.token,
                cursor=cursor[:-1] + ("A" if cursor[-1] != "A" else "B"),
                limit=5,
                idempotency_key="context-cursor-forged",
            )
    finally:
        await store.close()


async def test_thread_context_returns_only_root_and_bounded_replies(tmp_path: Path) -> None:
    store, human_id, group_id, agent_ids = await _open_group_store(tmp_path / "kai.db")
    try:
        commands = WorkshopConversationCommandService(store)
        unrelated_root = _accepted_message_id(
            await commands.accept_client(_message(human_id, group_id, "unrelated-root", "Unrelated thread", _NOW))
        )
        root = _accepted_message_id(
            await commands.accept_client(_message(human_id, group_id, "context-root", "Authorized thread root", _NOW))
        )
        accepted = await commands.accept_client(
            _message(
                human_id,
                group_id,
                "context-thread-command",
                "@Kai read this thread",
                _NOW + timedelta(seconds=1),
                thread_root_id=root,
            )
        )
        run = accepted.run
        execution_authority = WorkshopRunExecutionAuthority(
            store,
            selection_resolver=lambda _run: RunExecutionSelection("codex", "gpt-5.6-sol"),
            registered_backend_ids=frozenset({"codex"}),
        )
        granted = await execution_authority.grant(
            run.run_id,
            owner_id=RunExecutionOwnerId.new(),
            occurred_at=_NOW + timedelta(seconds=2),
            lease_expires_at=_NOW + timedelta(minutes=5),
        )
        started = await execution_authority.start(granted.claim, occurred_at=_NOW + timedelta(seconds=3))
        assert started.run.agent_definition_revision_id is not None
        await store.connection.execute(
            "UPDATE agent_definition_revisions SET collaboration_operations_json = ? WHERE id = ?",
            ('["context_read"]', started.run.agent_definition_revision_id),
        )
        await store.connection.commit()
        authority = WorkshopCollaborationAuthority(
            store,
            owner_policy_resolver=lambda _revision: CollaborationOwnerPolicy(
                version=1,
                allowed_operations=frozenset({CollaborationOperation.CONTEXT_READ}),
            ),
            token_factory=lambda: "thread-context-proof-00000000000000000000000001",
        )
        _grant, invocation = await authority.issue(
            started.claim,
            occurred_at=_NOW + timedelta(seconds=4),
        )
        assert started.run.runtime_profile_id is not None
        identity = CollaborationBaseIdentity(
            human_id,
            group_id,
            agent_ids[0],
            started.run.runtime_profile_id,
        )
        result = await WorkshopCollaborationContextService(
            store,
            _Execution(authority),
            clock=lambda: _NOW + timedelta(seconds=5),
        ).read(
            identity,
            proof=invocation.token,
            cursor=None,
            limit=10,
            idempotency_key="thread-context-page",
        )

        assert result.payload["context_kind"] == "thread"
        assert result.payload["thread_root"]["message_id"] == str(root)  # type: ignore[index]
        messages = result.payload["messages"]
        assert isinstance(messages, list)
        assert [item["body"] for item in messages] == ["@Kai read this thread"]
        serialized = str(result.payload)
        assert str(unrelated_root) not in serialized
        assert "Unrelated thread" not in serialized
    finally:
        await store.close()


async def test_authorized_idempotent_read_cannot_replay_after_revocation(tmp_path: Path) -> None:
    store, _started, _grant, invocation, identity, service, authority = await _running_context(tmp_path / "kai.db")
    try:
        await service.read(
            identity,
            proof=invocation.token,
            cursor=None,
            limit=5,
            idempotency_key="context-before-revoke",
        )
        await authority.revoke(
            invocation,
            revocation_code="qualification_cancelled",
            occurred_at=_NOW + timedelta(seconds=40),
        )
        with pytest.raises(CollaborationProofError, match="proof"):
            await service.read(
                identity,
                proof=invocation.token,
                cursor=None,
                limit=5,
                idempotency_key="context-before-revoke",
            )
    finally:
        await store.close()


async def _issued_position(store: WorkshopEventStore, grant_id: object) -> int:
    async with store.connection.execute(
        "SELECT issued_event_position FROM collaboration_grants WHERE id = ?",
        (grant_id,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return int(row[0])
