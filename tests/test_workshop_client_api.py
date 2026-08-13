"""HTTP contracts for the authenticated Workshop read API."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.client_api import (
    WorkshopEventStreamLimiter,
    register_workshop_command_routes,
    register_workshop_read_routes,
)
from kai.workshop.conversation_commands import ConversationCommandDisposition
from kai.workshop.domain import ChannelId, MessageId, PrincipalId, RunId
from kai.workshop.execution_coordinator import CanonicalExecutionDisposition
from kai.workshop.inbound import ClientInboundMessage, InboundMessage, record_inbound_message
from kai.workshop.run_lifecycle import RunStatus
from kai.workshop.store import WorkshopEventStore

_NOW = datetime(2026, 8, 11, 14, 0, tzinfo=UTC)


@dataclass
class _Authenticator:
    principals_by_token: dict[str, PrincipalId]
    calls: list[str] = field(default_factory=list)

    async def authenticate(self, request: web.Request) -> PrincipalId | None:
        authorization = request.headers.get("Authorization", "")
        self.calls.append(authorization)
        scheme, separator, token = authorization.partition(" ")
        if separator and scheme == "Bearer":
            return self.principals_by_token.get(token)
        return None


@dataclass
class _CommandSubmitter:
    messages: list[ClientInboundMessage] = field(default_factory=list)

    async def submit(self, message: ClientInboundMessage):
        self.messages.append(message)
        return SimpleNamespace(
            acceptance=SimpleNamespace(
                command=SimpleNamespace(
                    message=SimpleNamespace(
                        event=SimpleNamespace(envelope=SimpleNamespace(aggregate_id=MessageId.new()))
                    ),
                    disposition=ConversationCommandDisposition.NEWLY_ACCEPTED,
                ),
                run=SimpleNamespace(run_id=RunId.new()),
            ),
            execution=SimpleNamespace(
                disposition=CanonicalExecutionDisposition.COMPLETED,
                run=SimpleNamespace(status=RunStatus.COMPLETED),
            ),
        )


async def _identity_for(store: WorkshopEventStore, subject: str) -> tuple[PrincipalId, ChannelId]:
    async with store.connection.execute(
        "SELECT e.principal_id, b.channel_id FROM external_identities e "
        "JOIN channel_bindings b ON b.transport = e.provider "
        "AND b.external_channel_id = e.external_subject "
        "WHERE e.provider = 'telegram' AND e.external_subject = ?",
        (subject,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return PrincipalId(str(row[0])), ChannelId(str(row[1]))


async def _open_store(path: Path) -> tuple[WorkshopEventStore, PrincipalId, ChannelId, PrincipalId, ChannelId]:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman("Alice", "admin", "telegram", "101", "101"),
            BootstrapHuman("Bob", "member", "telegram", "202", "202"),
        ),
    )
    alice_id, alice_channel = await _identity_for(store, "101")
    bob_id, bob_channel = await _identity_for(store, "202")
    return store, alice_id, alice_channel, bob_id, bob_channel


async def _record_messages(store: WorkshopEventStore, count: int, *, start: int = 1) -> None:
    for ordinal in range(start, start + count):
        await record_inbound_message(
            store,
            InboundMessage(
                "telegram",
                str(9000 + ordinal),
                str(40 + ordinal),
                "101",
                "101",
                f"Message {ordinal}",
                _NOW + timedelta(seconds=ordinal),
            ),
        )


async def _open_client(
    store: WorkshopEventStore,
    authenticator: _Authenticator,
    *,
    event_poll_interval: float = 0.01,
    event_heartbeat_interval: float = 0.05,
    event_authentication_recheck_interval: float = 0.01,
    event_stream_limiter: WorkshopEventStreamLimiter | None = None,
) -> TestClient:
    app = web.Application()
    register_workshop_read_routes(
        app,
        store=store,
        authenticator=authenticator,
        request_lock=asyncio.Lock(),
        event_poll_interval=event_poll_interval,
        event_heartbeat_interval=event_heartbeat_interval,
        event_authentication_recheck_interval=event_authentication_recheck_interval,
        event_stream_limiter=event_stream_limiter,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


async def _open_command_client(
    store: WorkshopEventStore,
    authenticator: _Authenticator,
    submitter: _CommandSubmitter,
) -> TestClient:
    app = web.Application()
    register_workshop_command_routes(
        app,
        store=store,
        authenticator=authenticator,
        submitter=submitter,
        request_lock=asyncio.Lock(),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


async def _read_sse_event(response) -> dict[str, object]:
    event_name: str | None = None
    event_id: str | None = None
    data_lines: list[str] = []
    while True:
        raw_line = await asyncio.wait_for(response.content.readline(), timeout=1.0)
        assert raw_line, "Event stream ended before the next event"
        line = raw_line.decode().rstrip("\r\n")
        if not line:
            if event_name is None:
                continue
            return {
                "event": event_name,
                "id": event_id,
                "data": json.loads("\n".join(data_lines)),
            }
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "id":
            event_id = value
        elif field == "data":
            data_lines.append(value)


class TestWorkshopCommandHTTPContract:
    async def test_authenticated_member_submits_only_server_scoped_command_fields(
        self,
        tmp_path: Path,
    ):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        submitter = _CommandSubmitter()
        client = await _open_command_client(
            store,
            _Authenticator({"alice-token": alice_id}),
            submitter,
        )
        try:
            response = await client.post(
                f"/v1/channels/{alice_channel}/commands",
                headers={"Authorization": "Bearer alice-token"},
                json={
                    "client_message_id": "browser-command-1",
                    "body": "Hello from Workshop",
                },
            )
            payload = await response.json()

            assert response.status == 200
            assert payload["version"] == 1
            assert payload["acceptance"] == "newly_accepted"
            assert payload["execution"] == "completed"
            assert payload["run_status"] == "completed"
            assert str(payload["message_id"]).startswith("msg_")
            assert str(payload["run_id"]).startswith("run_")
            assert len(submitter.messages) == 1
            submitted = submitter.messages[0]
            assert submitted.principal_id == alice_id
            assert submitted.channel_id == alice_channel
            assert submitted.client_message_id == "browser-command-1"
            assert submitted.body == "Hello from Workshop"
            assert submitted.occurred_at.tzinfo is not None
        finally:
            await client.close()
            await store.close()

    async def test_authentication_precedes_parsing_and_cross_channel_is_denied(
        self,
        tmp_path: Path,
    ):
        store, alice_id, alice_channel, _, bob_channel = await _open_store(tmp_path / "kai.db")
        submitter = _CommandSubmitter()
        client = await _open_command_client(
            store,
            _Authenticator({"alice-token": alice_id}),
            submitter,
        )
        try:
            unauthenticated = await client.post(
                f"/v1/channels/{alice_channel}/commands",
                data="not json",
            )
            denied = await client.post(
                f"/v1/channels/{bob_channel}/commands",
                headers={"Authorization": "Bearer alice-token"},
                json={
                    "client_message_id": "browser-command-1",
                    "body": "Cross-channel command",
                },
            )

            assert unauthenticated.status == 401
            assert unauthenticated.headers["WWW-Authenticate"] == "Bearer"
            assert denied.status == 403
            assert submitter.messages == []
        finally:
            await client.close()
            await store.close()

    @pytest.mark.parametrize(
        "payload",
        [
            {"client_message_id": "browser-command-1"},
            {"client_message_id": "browser-command-1", "body": "Hello", "model": "gpt"},
            {"client_message_id": "bad id", "body": "Hello"},
            {"client_message_id": "browser-command-1", "body": "   "},
        ],
    )
    async def test_rejects_invalid_or_authority_expanding_input(
        self,
        tmp_path: Path,
        payload: dict[str, str],
    ):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        submitter = _CommandSubmitter()
        client = await _open_command_client(
            store,
            _Authenticator({"alice-token": alice_id}),
            submitter,
        )
        try:
            response = await client.post(
                f"/v1/channels/{alice_channel}/commands",
                headers={"Authorization": "Bearer alice-token"},
                json=payload,
            )

            assert response.status == 400
            assert submitter.messages == []
        finally:
            await client.close()
            await store.close()


class TestWorkshopTimelineHTTPContract:
    async def test_authenticated_member_receives_versioned_canonical_page(self, tmp_path: Path):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        await _record_messages(store, 1)
        authenticator = _Authenticator({"alice-token": alice_id})
        client = await _open_client(store, authenticator)
        try:
            response = await client.get(
                f"/v1/channels/{alice_channel}/timeline",
                headers={"Authorization": "Bearer alice-token"},
            )
            payload = await response.json()

            assert response.status == 200
            assert response.headers["Cache-Control"] == "private, no-store"
            assert response.headers["Content-Security-Policy"] == "default-src 'none'"
            assert response.headers["Referrer-Policy"] == "no-referrer"
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert payload == {
                "version": 1,
                "channel_id": alice_channel,
                "messages": [
                    {
                        "message_id": payload["messages"][0]["message_id"],
                        "channel_id": alice_channel,
                        "author_principal_id": alice_id,
                        "author_kind": "human",
                        "author_display_name": "Alice",
                        "reply_to_message_id": None,
                        "body": "Message 1",
                        "event_position": payload["messages"][0]["event_position"],
                        "created_at": "2026-08-11T14:00:01Z",
                    }
                ],
                "next_cursor": None,
                "through_position": payload["messages"][0]["event_position"],
            }
            assert payload["messages"][0]["message_id"].startswith("msg_")
            assert authenticator.calls == ["Bearer alice-token"]
        finally:
            await client.close()
            await store.close()

    async def test_unauthenticated_request_is_rejected_before_input_or_storage(self, tmp_path: Path):
        store, alice_id, _, _, _ = await _open_store(tmp_path / "kai.db")
        authenticator = _Authenticator({"alice-token": alice_id})
        client = await _open_client(store, authenticator)
        await store.close()
        try:
            response = await client.get("/v1/channels/not-an-id/timeline?limit=invalid")

            assert response.status == 401
            assert response.headers["WWW-Authenticate"] == "Bearer"
            assert await response.json() == {
                "error": {"code": "authentication_required", "message": "Authentication required"}
            }
        finally:
            await client.close()

    async def test_cross_channel_and_unknown_channel_have_same_denial(self, tmp_path: Path):
        store, alice_id, _, _, bob_channel = await _open_store(tmp_path / "kai.db")
        authenticator = _Authenticator({"alice-token": alice_id})
        client = await _open_client(store, authenticator)
        try:
            headers = {"Authorization": "Bearer alice-token"}
            cross_channel = await client.get(f"/v1/channels/{bob_channel}/timeline", headers=headers)
            unknown_channel = await client.get(f"/v1/channels/{ChannelId.new()}/timeline", headers=headers)

            assert cross_channel.status == unknown_channel.status == 403
            assert (
                await cross_channel.json()
                == await unknown_channel.json()
                == {"error": {"code": "access_denied", "message": "Access denied"}}
            )
        finally:
            await client.close()
            await store.close()

    @pytest.mark.parametrize(
        "query",
        [
            "?limit=0",
            "?limit=101",
            "?limit=not-a-number",
            "?limit=1&limit=2",
            "?cursor=not-a-cursor",
            "?cursor=one&cursor=two",
            "?chat_id=101",
        ],
    )
    async def test_invalid_pagination_input_returns_bounded_error(self, tmp_path: Path, query: str):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        client = await _open_client(store, _Authenticator({"alice-token": alice_id}))
        try:
            response = await client.get(
                f"/v1/channels/{alice_channel}/timeline{query}",
                headers={"Authorization": "Bearer alice-token"},
            )

            assert response.status == 400
            assert await response.json() == {
                "error": {"code": "invalid_request", "message": "Invalid timeline request"}
            }
        finally:
            await client.close()
            await store.close()

    async def test_cursor_resumes_same_snapshot_after_store_restart(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        store, alice_id, alice_channel, _, _ = await _open_store(db_path)
        await _record_messages(store, 3)
        first_client = await _open_client(store, _Authenticator({"alice-token": alice_id}))
        response = await first_client.get(
            f"/v1/channels/{alice_channel}/timeline?limit=2",
            headers={"Authorization": "Bearer alice-token"},
        )
        first_page = await response.json()
        await first_client.close()
        await store.close()

        restarted = await WorkshopEventStore.open(db_path)
        second_client = await _open_client(restarted, _Authenticator({"new-token": alice_id}))
        try:
            response = await second_client.get(
                f"/v1/channels/{alice_channel}/timeline",
                params={"cursor": first_page["next_cursor"], "limit": "2"},
                headers={"Authorization": "Bearer new-token"},
            )
            second_page = await response.json()

            assert response.status == 200
            assert [message["body"] for message in first_page["messages"]] == ["Message 1", "Message 2"]
            assert [message["body"] for message in second_page["messages"]] == ["Message 3"]
            assert second_page["through_position"] == first_page["through_position"]
            assert second_page["next_cursor"] is None
        finally:
            await second_client.close()
            await restarted.close()

    async def test_route_accepts_no_write_method(self, tmp_path: Path):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        client = await _open_client(store, _Authenticator({"alice-token": alice_id}))
        try:
            response = await client.post(
                f"/v1/channels/{alice_channel}/timeline",
                headers={"Authorization": "Bearer alice-token"},
                json={"body": "must not be accepted"},
            )

            assert response.status == 405
            events_response = await client.post(
                f"/v1/channels/{alice_channel}/events",
                headers={"Authorization": "Bearer alice-token"},
                json={"body": "must not be accepted"},
            )
            assert events_response.status == 405
            async with store.connection.execute("SELECT COUNT(*) FROM messages") as cursor:
                assert (await cursor.fetchone())[0] == 0
        finally:
            await client.close()
            await store.close()


class TestWorkshopTimelineEventStreamHTTPContract:
    async def test_replays_authorized_canonical_messages_as_versioned_sse(self, tmp_path: Path):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        await _record_messages(store, 2)
        client = await _open_client(store, _Authenticator({"alice-token": alice_id}))
        response = None
        try:
            response = await client.get(
                f"/v1/channels/{alice_channel}/events?after_position=0",
                headers={"Authorization": "Bearer alice-token"},
            )
            first = await _read_sse_event(response)
            second = await _read_sse_event(response)

            assert response.status == 200
            assert response.content_type == "text/event-stream"
            assert response.charset == "utf-8"
            assert response.headers["Cache-Control"] == "private, no-store"
            assert response.headers["Content-Security-Policy"] == "default-src 'none'"
            assert response.headers["Referrer-Policy"] == "no-referrer"
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert response.headers["X-Accel-Buffering"] == "no"
            assert [first["event"], second["event"]] == [
                "timeline.message.created",
                "timeline.message.created",
            ]
            assert int(str(first["id"])) < int(str(second["id"]))
            assert first["data"] == {
                "version": 1,
                "channel_id": alice_channel,
                "message": {
                    "message_id": first["data"]["message"]["message_id"],
                    "channel_id": alice_channel,
                    "author_principal_id": alice_id,
                    "author_kind": "human",
                    "author_display_name": "Alice",
                    "reply_to_message_id": None,
                    "body": "Message 1",
                    "event_position": int(str(first["id"])),
                    "created_at": "2026-08-11T14:00:01Z",
                },
            }
            assert second["data"]["message"]["body"] == "Message 2"
        finally:
            if response is not None:
                response.close()
            await client.close()
            await store.close()

    async def test_future_only_stream_receives_message_committed_after_connect(self, tmp_path: Path):
        database = tmp_path / "kai.db"
        writer, alice_id, alice_channel, _, _ = await _open_store(database)
        reader = await WorkshopEventStore.open(database)
        client = await _open_client(reader, _Authenticator({"alice-token": alice_id}))
        response = None
        try:
            response = await client.get(
                f"/v1/channels/{alice_channel}/events",
                headers={"Authorization": "Bearer alice-token"},
            )
            await _record_messages(writer, 1)

            event = await _read_sse_event(response)

            assert event["event"] == "timeline.message.created"
            assert event["data"]["message"]["body"] == "Message 1"
        finally:
            if response is not None:
                response.close()
            await client.close()
            await reader.close()
            await writer.close()

    async def test_last_event_id_resumes_after_store_restart_and_overrides_initial_query(self, tmp_path: Path):
        database = tmp_path / "kai.db"
        store, alice_id, alice_channel, _, _ = await _open_store(database)
        await _record_messages(store, 1)
        first_client = await _open_client(store, _Authenticator({"alice-token": alice_id}))
        first_response = await first_client.get(
            f"/v1/channels/{alice_channel}/events?after_position=0",
            headers={"Authorization": "Bearer alice-token"},
        )
        first_event = await _read_sse_event(first_response)
        first_response.close()
        await first_client.close()
        await _record_messages(store, 1, start=2)
        await store.close()

        restarted = await WorkshopEventStore.open(database)
        second_client = await _open_client(restarted, _Authenticator({"new-token": alice_id}))
        second_response = None
        try:
            second_response = await second_client.get(
                f"/v1/channels/{alice_channel}/events?after_position=0",
                headers={
                    "Authorization": "Bearer new-token",
                    "Last-Event-ID": str(first_event["id"]),
                },
            )
            second_event = await _read_sse_event(second_response)

            assert int(str(second_event["id"])) > int(str(first_event["id"]))
            assert second_event["data"]["message"]["body"] == "Message 2"
        finally:
            if second_response is not None:
                second_response.close()
            await second_client.close()
            await restarted.close()

    async def test_open_stream_rechecks_authentication_and_closes_after_revocation(self, tmp_path: Path):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        authenticator = _Authenticator({"alice-token": alice_id})
        client = await _open_client(store, authenticator)
        response = None
        try:
            response = await client.get(
                f"/v1/channels/{alice_channel}/events",
                headers={"Authorization": "Bearer alice-token"},
            )
            assert (await response.content.readline()).startswith(b": connected")
            assert await response.content.readline() == b"retry: 2000\n"
            assert await response.content.readline() == b"\n"

            authenticator.principals_by_token.clear()

            assert await asyncio.wait_for(response.content.read(), timeout=1.0) == b""
            assert len(authenticator.calls) >= 2
        finally:
            if response is not None:
                response.close()
            await client.close()
            await store.close()

    async def test_idle_polling_does_not_rewrite_session_last_seen_on_every_poll(self, tmp_path: Path):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        authenticator = _Authenticator({"alice-token": alice_id})
        client = await _open_client(
            store,
            authenticator,
            event_poll_interval=0.005,
            event_heartbeat_interval=0.01,
            event_authentication_recheck_interval=0.2,
        )
        response = None
        try:
            response = await client.get(
                f"/v1/channels/{alice_channel}/events",
                headers={"Authorization": "Bearer alice-token"},
            )
            await asyncio.sleep(0.05)

            assert authenticator.calls == ["Bearer alice-token"]
        finally:
            if response is not None:
                response.close()
            await client.close()
            await store.close()

    async def test_concurrent_stream_capacity_is_bounded_and_released_on_disconnect(self, tmp_path: Path):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        limiter = WorkshopEventStreamLimiter(per_principal_limit=1, global_limit=1)
        client = await _open_client(
            store,
            _Authenticator({"alice-token": alice_id}),
            event_poll_interval=0.005,
            event_heartbeat_interval=0.01,
            event_stream_limiter=limiter,
        )
        first_response = None
        replacement_response = None
        try:
            headers = {"Authorization": "Bearer alice-token"}
            path = f"/v1/channels/{alice_channel}/events"
            first_response = await client.get(path, headers=headers)
            rejected = await client.get(path, headers=headers)

            assert first_response.status == 200
            assert rejected.status == 429
            assert rejected.headers["Retry-After"] == "5"
            assert await rejected.json() == {
                "error": {
                    "code": "stream_capacity_exceeded",
                    "message": "Too many active event streams",
                }
            }

            first_response.close()
            first_response = None
            await asyncio.sleep(0.05)
            replacement_response = await client.get(path, headers=headers)
            assert replacement_response.status == 200
        finally:
            if first_response is not None:
                first_response.close()
            if replacement_response is not None:
                replacement_response.close()
            await client.close()
            await store.close()

    async def test_unauthenticated_event_stream_is_rejected_before_input_or_storage(self, tmp_path: Path):
        store, alice_id, _, _, _ = await _open_store(tmp_path / "kai.db")
        client = await _open_client(store, _Authenticator({"alice-token": alice_id}))
        await store.close()
        try:
            response = await client.get("/v1/channels/not-an-id/events?after_position=invalid")

            assert response.status == 401
            assert response.headers["WWW-Authenticate"] == "Bearer"
            assert await response.json() == {
                "error": {"code": "authentication_required", "message": "Authentication required"}
            }
        finally:
            await client.close()

    async def test_cross_channel_and_unknown_event_streams_have_same_denial(self, tmp_path: Path):
        store, alice_id, _, _, bob_channel = await _open_store(tmp_path / "kai.db")
        client = await _open_client(store, _Authenticator({"alice-token": alice_id}))
        try:
            headers = {"Authorization": "Bearer alice-token"}
            cross_channel = await client.get(f"/v1/channels/{bob_channel}/events", headers=headers)
            unknown_channel = await client.get(f"/v1/channels/{ChannelId.new()}/events", headers=headers)

            assert cross_channel.status == unknown_channel.status == 403
            assert (
                await cross_channel.json()
                == await unknown_channel.json()
                == {"error": {"code": "access_denied", "message": "Access denied"}}
            )
        finally:
            await client.close()
            await store.close()

    @pytest.mark.parametrize(
        ("query", "headers", "status", "error"),
        [
            ("?after_position=-1", {}, 400, "invalid_request"),
            ("?after_position=one", {}, 400, "invalid_request"),
            ("?after_position=1&after_position=2", {}, 400, "invalid_request"),
            ("?cursor=unused", {}, 400, "invalid_request"),
            ("", {"Last-Event-ID": "invalid"}, 400, "invalid_request"),
            ("?after_position=999999", {}, 409, "resynchronization_required"),
        ],
    )
    async def test_invalid_or_unresumable_event_position_is_bounded(
        self,
        tmp_path: Path,
        query: str,
        headers: dict[str, str],
        status: int,
        error: str,
    ):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        client = await _open_client(store, _Authenticator({"alice-token": alice_id}))
        try:
            response = await client.get(
                f"/v1/channels/{alice_channel}/events{query}",
                headers={"Authorization": "Bearer alice-token", **headers},
            )

            assert response.status == status
            assert (await response.json())["error"]["code"] == error
        finally:
            await client.close()
            await store.close()
