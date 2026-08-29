"""Contracts for the canonical Workshop timeline reader."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.channel_lifecycle import WorkshopChannelLifecycleService
from kai.workshop.domain import (
    AgentId,
    ChannelId,
    EventEnvelope,
    MessageId,
    PrincipalId,
    RuntimeAssignmentId,
    RuntimeProfileId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.inbound import (
    ClientInboundMessage,
    InboundMessage,
    record_client_inbound_message_in_transaction,
    record_inbound_message,
)
from kai.workshop.outbound import OutboundMessage, record_outbound_message
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore
from kai.workshop.timeline import (
    TimelineAccessDeniedError,
    TimelineCursorError,
    read_channel_timeline,
    read_thread_timeline,
)

_NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)


@dataclass
class _Authorizer:
    allowed: set[tuple[PrincipalId, ChannelId]]
    calls: list[tuple[PrincipalId, ChannelId]] = field(default_factory=list)

    async def can_read_channel(self, principal_id: PrincipalId, channel_id: ChannelId) -> bool:
        self.calls.append((principal_id, channel_id))
        return (principal_id, channel_id) in self.allowed


async def _identity_for(
    store: WorkshopEventStore,
    external_subject: str,
) -> tuple[PrincipalId, ChannelId]:
    async with store.connection.execute(
        "SELECT e.principal_id, b.channel_id FROM external_identities e "
        "JOIN channel_bindings b ON b.transport = e.provider "
        "AND b.external_channel_id = e.external_subject "
        "WHERE e.provider = 'telegram' AND e.external_subject = ?",
        (external_subject,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return PrincipalId(str(row[0])), ChannelId(str(row[1]))


async def _open_store(path: Path) -> tuple[WorkshopEventStore, PrincipalId, ChannelId, PrincipalId, ChannelId]:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman("User One", "admin", "telegram", "101", "101"),
            BootstrapHuman("User Two", "member", "telegram", "202", "202"),
        ),
    )
    first_principal, first_channel = await _identity_for(store, "101")
    second_principal, second_channel = await _identity_for(store, "202")
    return store, first_principal, first_channel, second_principal, second_channel


async def _record_user_message(
    store: WorkshopEventStore,
    *,
    ordinal: int,
    body: str,
) -> None:
    await record_inbound_message(
        store,
        InboundMessage(
            "telegram",
            str(9000 + ordinal),
            str(40 + ordinal),
            "101",
            "101",
            body,
            _NOW + timedelta(seconds=ordinal),
        ),
    )


async def _create_group_channel(
    store: WorkshopEventStore,
    principal_id: PrincipalId,
    direct_channel_id: ChannelId,
) -> ChannelId:
    async with store.connection.execute(
        "SELECT ca.agent_id, c.workshop_id FROM channel_agents ca "
        "JOIN channels c ON c.id = ca.channel_id WHERE ca.channel_id = ?",
        (direct_channel_id,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    agent_id = AgentId(str(row[0]))
    workshop_id = WorkshopId(str(row[1]))
    await store.append(
        EventEnvelope.create(
            event_type=WorkshopEventType.RUNTIME_PROFILE_ASSIGNED,
            event_version=1,
            workshop_id=workshop_id,
            aggregate_type="runtime_assignment",
            aggregate_id=RuntimeAssignmentId.derived(direct_channel_id, f"runtime-profile:{agent_id}"),
            occurred_at=_NOW,
            idempotency_key=f"thread-test:runtime:{direct_channel_id}",
            payload={
                "channel_id": direct_channel_id,
                "agent_id": agent_id,
                "runtime_profile_id": RuntimeProfileId.new(),
            },
            metadata={"source": "test"},
        )
    )
    await store.project_pending(CanonicalConversationProjection())
    created = await WorkshopChannelLifecycleService(store).create_group(
        principal_id,
        name="Thread tests",
        agent_ids=[agent_id],
        origin_channel_id=direct_channel_id,
    )
    return created.channel_id


async def _record_client_message(
    store: WorkshopEventStore,
    principal_id: PrincipalId,
    channel_id: ChannelId,
    client_message_id: str,
    body: str,
    *,
    thread_root_id: MessageId | None = None,
    occurred_at: datetime = _NOW,
) -> MessageId:
    await store.connection.execute("BEGIN IMMEDIATE")
    try:
        result = await record_client_inbound_message_in_transaction(
            store,
            ClientInboundMessage(
                principal_id,
                channel_id,
                client_message_id,
                body,
                occurred_at,
                thread_root_id=thread_root_id,
            ),
        )
        await store.connection.commit()
    except Exception:
        await store.connection.rollback()
        raise
    return MessageId(str(result.event.envelope.aggregate_id))


class TestThreadTimelineQuery:
    async def test_channel_summary_and_thread_pages_preserve_structure_and_order(self, tmp_path: Path):
        store, principal_id, direct_channel, _, _ = await _open_store(tmp_path / "kai.db")
        try:
            channel_id = await _create_group_channel(store, principal_id, direct_channel)
            root_id = await _record_client_message(store, principal_id, channel_id, "root", "Root message")
            reply_id = await _record_client_message(
                store,
                principal_id,
                channel_id,
                "reply-one",
                "Human reply",
                thread_root_id=root_id,
                occurred_at=_NOW + timedelta(seconds=1),
            )
            await record_outbound_message(
                store,
                OutboundMessage(reply_id, "Agent reply", _NOW + timedelta(seconds=2)),
            )
            authorizer = _Authorizer({(principal_id, channel_id)})

            channel_page = await read_channel_timeline(
                store,
                principal_id=principal_id,
                channel_id=channel_id,
                authorizer=authorizer,
            )
            first = await read_thread_timeline(
                store,
                principal_id=principal_id,
                channel_id=channel_id,
                thread_root_id=root_id,
                authorizer=authorizer,
                limit=1,
            )
            assert [message.body for message in channel_page.messages] == ["Root message"]
            assert channel_page.messages[0].reply_count == 2
            assert channel_page.messages[0].latest_reply_at == _NOW + timedelta(seconds=2)
            assert first.root.message_id == root_id
            assert [message.body for message in first.messages] == ["Human reply"]
            assert first.next_cursor is not None
            second = await read_thread_timeline(
                store,
                principal_id=principal_id,
                channel_id=channel_id,
                thread_root_id=root_id,
                authorizer=authorizer,
                cursor=first.next_cursor,
                limit=1,
            )
            assert [message.body for message in second.messages] == ["Agent reply"]
            assert all(message.thread_root_id == root_id for message in (*first.messages, *second.messages))
            assert second.messages[0].reply_to_message_id == reply_id
            assert second.next_cursor is None
            await store.rebuild_projection(CanonicalConversationProjection())
            rebuilt = await read_thread_timeline(
                store,
                principal_id=principal_id,
                channel_id=channel_id,
                thread_root_id=root_id,
                authorizer=authorizer,
            )
            assert [message.body for message in rebuilt.messages] == ["Human reply", "Agent reply"]
            assert rebuilt.root.reply_count == 2
        finally:
            await store.close()

    async def test_rejects_foreign_or_nested_roots_without_mutation(self, tmp_path: Path):
        store, principal_id, direct_channel, _, _ = await _open_store(tmp_path / "kai.db")
        try:
            channel_id = await _create_group_channel(store, principal_id, direct_channel)
            root_id = await _record_client_message(store, principal_id, channel_id, "root", "Root")
            reply_id = await _record_client_message(
                store,
                principal_id,
                channel_id,
                "reply",
                "Reply",
                thread_root_id=root_id,
            )
            with pytest.raises(ValueError, match="top-level message"):
                await _record_client_message(
                    store,
                    principal_id,
                    channel_id,
                    "nested",
                    "Nested",
                    thread_root_id=reply_id,
                )
            with pytest.raises(ValueError, match="top-level message"):
                await _record_client_message(
                    store,
                    principal_id,
                    channel_id,
                    "foreign",
                    "Foreign",
                    thread_root_id=MessageId.new(),
                )
            async with store.connection.execute(
                "SELECT COUNT(*) FROM messages WHERE channel_id = ?",
                (channel_id,),
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 2
        finally:
            await store.close()


class TestCanonicalTimelineQuery:
    async def test_returns_server_resolved_mentions_without_client_reparsing(self, tmp_path: Path):
        store, principal_id, channel_id, _, _ = await _open_store(tmp_path / "kai.db")
        try:
            await _record_user_message(store, ordinal=1, body="Please ask @kAi about this")

            page = await read_channel_timeline(
                store,
                principal_id=principal_id,
                channel_id=channel_id,
                authorizer=_Authorizer({(principal_id, channel_id)}),
            )

            assert len(page.messages) == 1
            assert [(mention.kind, mention.start, mention.length) for mention in page.messages[0].mentions] == [
                ("agent", 11, 4)
            ]
            assert page.messages[0].body[11:15] == "@kAi"
        finally:
            await store.close()

    async def test_authorized_reader_returns_canonical_messages_in_event_order(self, tmp_path: Path):
        store, principal_id, channel_id, _, _ = await _open_store(tmp_path / "kai.db")
        try:
            inbound = await record_inbound_message(
                store,
                InboundMessage("telegram", "9001", "42", "101", "101", "Question", _NOW),
            )
            await record_outbound_message(
                store,
                OutboundMessage(
                    MessageId(str(inbound.event.envelope.aggregate_id)),
                    "Answer",
                    _NOW + timedelta(seconds=1),
                ),
            )
            authorizer = _Authorizer({(principal_id, channel_id)})

            page = await read_channel_timeline(
                store,
                principal_id=principal_id,
                channel_id=channel_id,
                authorizer=authorizer,
            )

            assert [message.body for message in page.messages] == ["Question", "Answer"]
            assert [message.author_kind for message in page.messages] == ["human", "agent"]
            assert [message.author_display_name for message in page.messages] == ["User One", "Kai"]
            assert page.messages[1].reply_to_message_id == page.messages[0].message_id
            assert all(message.channel_id == channel_id for message in page.messages)
            assert [message.event_position for message in page.messages] == sorted(
                message.event_position for message in page.messages
            )
            assert page.through_position == page.messages[-1].event_position
            assert page.next_cursor is None
            assert authorizer.calls == [(principal_id, channel_id)]
        finally:
            await store.close()

    async def test_denied_reader_receives_no_timeline(self, tmp_path: Path):
        store, principal_id, channel_id, _, _ = await _open_store(tmp_path / "kai.db")
        try:
            await _record_user_message(store, ordinal=1, body="Private content")
            authorizer = _Authorizer(set())

            with pytest.raises(TimelineAccessDeniedError):
                await read_channel_timeline(
                    store,
                    principal_id=principal_id,
                    channel_id=channel_id,
                    authorizer=authorizer,
                )

            assert authorizer.calls == [(principal_id, channel_id)]
        finally:
            await store.close()

    async def test_denial_happens_before_any_storage_lookup(self, tmp_path: Path):
        store, principal_id, channel_id, _, _ = await _open_store(tmp_path / "kai.db")
        await store.close()
        authorizer = _Authorizer(set())

        with pytest.raises(TimelineAccessDeniedError):
            await read_channel_timeline(
                store,
                principal_id=principal_id,
                channel_id=channel_id,
                authorizer=authorizer,
            )

        assert authorizer.calls == [(principal_id, channel_id)]

    async def test_authorized_but_unknown_channel_fails_closed(self, tmp_path: Path):
        store, principal_id, _, _, _ = await _open_store(tmp_path / "kai.db")
        unknown_channel = ChannelId.new()
        try:
            authorizer = _Authorizer({(principal_id, unknown_channel)})

            with pytest.raises(TimelineAccessDeniedError):
                await read_channel_timeline(
                    store,
                    principal_id=principal_id,
                    channel_id=unknown_channel,
                    authorizer=authorizer,
                )
        finally:
            await store.close()

    async def test_snapshot_cursor_excludes_messages_added_between_pages(self, tmp_path: Path):
        store, principal_id, channel_id, _, _ = await _open_store(tmp_path / "kai.db")
        try:
            for ordinal in range(1, 4):
                await _record_user_message(store, ordinal=ordinal, body=f"Message {ordinal}")
            authorizer = _Authorizer({(principal_id, channel_id)})

            first = await read_channel_timeline(
                store,
                principal_id=principal_id,
                channel_id=channel_id,
                authorizer=authorizer,
                limit=2,
            )
            assert [message.body for message in first.messages] == ["Message 1", "Message 2"]
            assert first.next_cursor is not None
            first_cursor = first.next_cursor

            repeated = await read_channel_timeline(
                store,
                principal_id=principal_id,
                channel_id=channel_id,
                authorizer=authorizer,
                limit=2,
            )
            assert repeated.next_cursor == first_cursor

            await _record_user_message(store, ordinal=4, body="Message 4")
            second = await read_channel_timeline(
                store,
                principal_id=principal_id,
                channel_id=channel_id,
                authorizer=authorizer,
                cursor=first_cursor,
                limit=2,
            )

            assert [message.body for message in second.messages] == ["Message 3"]
            assert second.through_position == first.through_position
            assert second.next_cursor is None

            fresh = await read_channel_timeline(
                store,
                principal_id=principal_id,
                channel_id=channel_id,
                authorizer=authorizer,
                limit=10,
            )
            assert [message.body for message in fresh.messages] == [
                "Message 1",
                "Message 2",
                "Message 3",
                "Message 4",
            ]
            assert fresh.through_position > first.through_position
        finally:
            await store.close()

    async def test_cursor_is_bound_to_its_channel(self, tmp_path: Path):
        store, principal_id, channel_id, second_principal, second_channel = await _open_store(tmp_path / "kai.db")
        try:
            for ordinal in range(1, 3):
                await _record_user_message(store, ordinal=ordinal, body=f"Message {ordinal}")
            authorizer = _Authorizer(
                {
                    (principal_id, channel_id),
                    (second_principal, second_channel),
                }
            )
            first = await read_channel_timeline(
                store,
                principal_id=principal_id,
                channel_id=channel_id,
                authorizer=authorizer,
                limit=1,
            )
            assert first.next_cursor is not None

            with pytest.raises(TimelineCursorError):
                await read_channel_timeline(
                    store,
                    principal_id=second_principal,
                    channel_id=second_channel,
                    authorizer=authorizer,
                    cursor=first.next_cursor,
                )
        finally:
            await store.close()

    @pytest.mark.parametrize("cursor", ["", "not-a-cursor", "v2.invalid", "v1.***"])
    async def test_malformed_cursor_fails_closed(self, tmp_path: Path, cursor: str):
        store, principal_id, channel_id, _, _ = await _open_store(tmp_path / "kai.db")
        try:
            with pytest.raises(TimelineCursorError):
                await read_channel_timeline(
                    store,
                    principal_id=principal_id,
                    channel_id=channel_id,
                    authorizer=_Authorizer({(principal_id, channel_id)}),
                    cursor=cursor,
                )
        finally:
            await store.close()


class TestTailTimelineQuery:
    async def test_tail_returns_newest_page_in_event_order(self, tmp_path: Path):
        store, principal_id, channel_id, _, _ = await _open_store(tmp_path / "kai.db")
        try:
            for ordinal in range(1, 6):
                await _record_user_message(store, ordinal=ordinal, body=f"Message {ordinal}")

            page = await read_channel_timeline(
                store,
                principal_id=principal_id,
                channel_id=channel_id,
                authorizer=_Authorizer({(principal_id, channel_id)}),
                tail=True,
                limit=2,
            )

            assert [message.body for message in page.messages] == ["Message 4", "Message 5"]
            assert page.through_position == page.messages[-1].event_position
            assert page.next_cursor is None
            assert page.previous_cursor is not None
        finally:
            await store.close()

    async def test_previous_cursor_walks_to_channel_start(self, tmp_path: Path):
        store, principal_id, channel_id, _, _ = await _open_store(tmp_path / "kai.db")
        try:
            for ordinal in range(1, 6):
                await _record_user_message(store, ordinal=ordinal, body=f"Message {ordinal}")
            authorizer = _Authorizer({(principal_id, channel_id)})

            pages = []
            cursor: str | None = None
            tail = True
            while True:
                page = await read_channel_timeline(
                    store,
                    principal_id=principal_id,
                    channel_id=channel_id,
                    authorizer=authorizer,
                    cursor=cursor,
                    tail=tail,
                    limit=2,
                )
                pages.append(page)
                tail = False
                cursor = page.previous_cursor
                if cursor is None:
                    break

            assert [[message.body for message in page.messages] for page in pages] == [
                ["Message 4", "Message 5"],
                ["Message 2", "Message 3"],
                ["Message 1"],
            ]
            assert len({page.through_position for page in pages}) == 1
            assert all(page.next_cursor is None for page in pages)
        finally:
            await store.close()

    async def test_tail_snapshot_excludes_messages_added_after_first_page(self, tmp_path: Path):
        store, principal_id, channel_id, _, _ = await _open_store(tmp_path / "kai.db")
        try:
            for ordinal in range(1, 4):
                await _record_user_message(store, ordinal=ordinal, body=f"Message {ordinal}")
            authorizer = _Authorizer({(principal_id, channel_id)})

            first = await read_channel_timeline(
                store,
                principal_id=principal_id,
                channel_id=channel_id,
                authorizer=authorizer,
                tail=True,
                limit=2,
            )
            assert [message.body for message in first.messages] == ["Message 2", "Message 3"]
            assert first.previous_cursor is not None

            await _record_user_message(store, ordinal=4, body="Message 4")
            earlier = await read_channel_timeline(
                store,
                principal_id=principal_id,
                channel_id=channel_id,
                authorizer=authorizer,
                cursor=first.previous_cursor,
                limit=2,
            )

            assert [message.body for message in earlier.messages] == ["Message 1"]
            assert earlier.through_position == first.through_position
            assert earlier.previous_cursor is None

            fresh = await read_channel_timeline(
                store,
                principal_id=principal_id,
                channel_id=channel_id,
                authorizer=authorizer,
                tail=True,
                limit=10,
            )
            assert [message.body for message in fresh.messages] == [
                "Message 1",
                "Message 2",
                "Message 3",
                "Message 4",
            ]
            assert fresh.through_position > first.through_position
        finally:
            await store.close()

    async def test_short_and_empty_channels_have_no_previous_cursor(self, tmp_path: Path):
        store, principal_id, channel_id, second_principal, second_channel = await _open_store(tmp_path / "kai.db")
        try:
            for ordinal in range(1, 3):
                await _record_user_message(store, ordinal=ordinal, body=f"Message {ordinal}")

            short = await read_channel_timeline(
                store,
                principal_id=principal_id,
                channel_id=channel_id,
                authorizer=_Authorizer({(principal_id, channel_id)}),
                tail=True,
                limit=10,
            )
            assert [message.body for message in short.messages] == ["Message 1", "Message 2"]
            assert short.previous_cursor is None

            empty = await read_channel_timeline(
                store,
                principal_id=second_principal,
                channel_id=second_channel,
                authorizer=_Authorizer({(second_principal, second_channel)}),
                tail=True,
                limit=10,
            )
            assert empty.messages == ()
            assert empty.previous_cursor is None
            assert empty.through_position == 0
        finally:
            await store.close()

    async def test_tail_with_cursor_is_rejected_before_authorization(self, tmp_path: Path):
        store, principal_id, channel_id, _, _ = await _open_store(tmp_path / "kai.db")
        try:
            authorizer = _Authorizer({(principal_id, channel_id)})
            with pytest.raises(ValueError, match="tail"):
                await read_channel_timeline(
                    store,
                    principal_id=principal_id,
                    channel_id=channel_id,
                    authorizer=authorizer,
                    cursor="v1.anything",
                    tail=True,
                )
            assert authorizer.calls == []
        finally:
            await store.close()

    async def test_tail_cursor_is_bound_to_its_channel(self, tmp_path: Path):
        store, principal_id, channel_id, second_principal, second_channel = await _open_store(tmp_path / "kai.db")
        try:
            for ordinal in range(1, 3):
                await _record_user_message(store, ordinal=ordinal, body=f"Message {ordinal}")
            authorizer = _Authorizer(
                {
                    (principal_id, channel_id),
                    (second_principal, second_channel),
                }
            )
            tail_page = await read_channel_timeline(
                store,
                principal_id=principal_id,
                channel_id=channel_id,
                authorizer=authorizer,
                tail=True,
                limit=1,
            )
            assert tail_page.previous_cursor is not None

            with pytest.raises(TimelineCursorError):
                await read_channel_timeline(
                    store,
                    principal_id=second_principal,
                    channel_id=second_channel,
                    authorizer=authorizer,
                    cursor=tail_page.previous_cursor,
                )
        finally:
            await store.close()

    @pytest.mark.parametrize(
        "payload",
        [
            # A boundary of zero cannot reference a returned message.
            {"before_position": 0, "channel_id": None, "through_position": 5},
            # Backward pages must stay inside the snapshot bound.
            {"before_position": 6, "channel_id": None, "through_position": 5},
            # A cursor cannot carry both directions at once.
            {"after_position": 1, "before_position": 2, "channel_id": None, "through_position": 5},
        ],
    )
    async def test_forged_tail_cursor_fails_closed(self, tmp_path: Path, payload: dict[str, object]):
        store, principal_id, channel_id, _, _ = await _open_store(tmp_path / "kai.db")
        try:
            payload["channel_id"] = str(channel_id)
            raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
            forged = "v1." + base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

            with pytest.raises(TimelineCursorError):
                await read_channel_timeline(
                    store,
                    principal_id=principal_id,
                    channel_id=channel_id,
                    authorizer=_Authorizer({(principal_id, channel_id)}),
                    cursor=forged,
                )
        finally:
            await store.close()


class TestTimelineRequestBounds:
    @pytest.mark.parametrize("limit", [0, -1, 101, True])
    async def test_limit_is_bounded(self, tmp_path: Path, limit: int):
        store, principal_id, channel_id, _, _ = await _open_store(tmp_path / "kai.db")
        try:
            with pytest.raises(ValueError, match="limit"):
                await read_channel_timeline(
                    store,
                    principal_id=principal_id,
                    channel_id=channel_id,
                    authorizer=_Authorizer({(principal_id, channel_id)}),
                    limit=limit,
                )
        finally:
            await store.close()
