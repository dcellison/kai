"""HTTP contracts for the production-unregistered Workshop read API."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.client_api import register_workshop_read_routes
from kai.workshop.domain import ChannelId, PrincipalId
from kai.workshop.inbound import InboundMessage, record_inbound_message
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


async def _record_messages(store: WorkshopEventStore, count: int) -> None:
    for ordinal in range(1, count + 1):
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
) -> TestClient:
    app = web.Application()
    register_workshop_read_routes(app, store=store, authenticator=authenticator)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


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
            async with store.connection.execute("SELECT COUNT(*) FROM messages") as cursor:
                assert (await cursor.fetchone())[0] == 0
        finally:
            await client.close()
            await store.close()
