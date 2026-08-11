"""Contracts for the production-unused canonical Workshop timeline reader."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.domain import ChannelId, MessageId, PrincipalId
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.outbound import OutboundMessage, record_outbound_message
from kai.workshop.store import WorkshopEventStore
from kai.workshop.timeline import (
    TimelineAccessDeniedError,
    TimelineCursorError,
    read_channel_timeline,
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


class TestCanonicalTimelineQuery:
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
