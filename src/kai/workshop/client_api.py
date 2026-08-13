"""Authenticated HTTP contracts for Workshop enrollment and conversation access."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import re
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from aiohttp import web

from kai.workshop.authorization import CanonicalChannelAuthorizer
from kai.workshop.client_commands import (
    ClientCommandExecutorUnavailableError,
    ClientCommandSubmission,
)
from kai.workshop.client_events import (
    ClientChannelEventBatch,
    ClientRunLifecycleEvent,
    ClientTimelineMessageEvent,
    read_client_channel_events,
)
from kai.workshop.client_sessions import EnrollmentGrantUnavailableError, WorkshopClientEnrollmentManager
from kai.workshop.conversation_commands import ConversationCommandAcceptanceError
from kai.workshop.domain import ChannelId, PrincipalId, RunId
from kai.workshop.execution_coordinator import CanonicalCancellationDisposition
from kai.workshop.inbound import ClientInboundMessage, InboundBindingNotFoundError
from kai.workshop.run_lifecycle import DurableRun, RunNotFoundError, RunStatus
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore
from kai.workshop.timeline import (
    TimelineAccessDeniedError,
    TimelineCursorError,
    TimelineMessage,
    TimelineResumeError,
    read_channel_timeline,
)

_TIMELINE_PATH = "/v1/channels/{channel_id}/timeline"
_TIMELINE_EVENTS_PATH = "/v1/channels/{channel_id}/events"
_CLIENT_NAVIGATION_PATH = "/v1/client/navigation"
_ENROLLMENT_REDEMPTION_PATH = "/v1/client/enrollment/redeem"
_COMMAND_SUBMISSION_PATH = "/v1/channels/{channel_id}/commands"
_RUN_STATE_PATH = "/v1/channels/{channel_id}/runs/{run_id}"
_RUN_CANCELLATION_PATH = "/v1/channels/{channel_id}/runs/{run_id}/cancel"
_ALLOWED_TIMELINE_QUERY_PARAMETERS = frozenset({"cursor", "limit"})
_ALLOWED_EVENT_QUERY_PARAMETERS = frozenset({"after_position"})
_ENROLLMENT_REQUEST_FIELDS = frozenset({"enrollment_token", "device_display_name"})
_COMMAND_REQUEST_FIELDS = frozenset({"client_message_id", "body"})
_DECIMAL_INTEGER = re.compile(r"^[0-9]+$")
_EVENT_BATCH_SIZE = 100
_SSE_RETRY_MILLISECONDS = 2000


class WorkshopClientAuthenticator(Protocol):
    """Resolve a human Workshop principal from a client request."""

    async def authenticate(self, request: web.Request) -> PrincipalId | None: ...


class WorkshopClientCommandSubmitter(Protocol):
    async def submit(self, message: ClientInboundMessage) -> ClientCommandSubmission: ...

    async def state(self, run_id: RunId) -> DurableRun: ...

    async def cancel(self, run_id: RunId) -> CanonicalCancellationDisposition: ...


class WorkshopEnrollmentRateLimiter:
    """Bound enrollment attempts by source and across the whole process.

    Cloudflare Tunnel connects to Kai over loopback, so ``request.remote`` is
    normally the tunnel process. ``CF-Connecting-IP`` is used only as a
    rate-limit partition when it contains one valid address; it never grants
    identity or authorization. A global ceiling still applies when a local
    caller spoofs or rotates that advisory header.
    """

    def __init__(
        self,
        *,
        per_source_limit: int = 10,
        global_limit: int = 120,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if per_source_limit < 1 or global_limit < per_source_limit or window_seconds <= 0:
            raise ValueError("Enrollment rate-limit bounds are invalid")
        self._per_source_limit = per_source_limit
        self._global_limit = global_limit
        self._window_seconds = window_seconds
        self._clock = clock
        self._global_attempts: deque[float] = deque()
        self._source_attempts: dict[str, deque[float]] = {}

    @staticmethod
    def _source(request: web.Request) -> str:
        forwarded = request.headers.getall("CF-Connecting-IP", [])
        if len(forwarded) == 1:
            try:
                return str(ipaddress.ip_address(forwarded[0].strip()))
            except ValueError:
                pass
        remote = request.remote
        if remote:
            try:
                return str(ipaddress.ip_address(remote))
            except ValueError:
                return "peer:unknown"
        return "peer:unknown"

    def check(self, request: web.Request) -> int | None:
        """Record one attempt, or return whole seconds until retry is allowed."""
        now = self._clock()
        cutoff = now - self._window_seconds
        while self._global_attempts and self._global_attempts[0] <= cutoff:
            self._global_attempts.popleft()

        for prior_source, prior_attempts in list(self._source_attempts.items()):
            while prior_attempts and prior_attempts[0] <= cutoff:
                prior_attempts.popleft()
            if not prior_attempts:
                del self._source_attempts[prior_source]

        source = self._source(request)
        attempts = self._source_attempts.setdefault(source, deque())

        blocked_until: float | None = None
        if len(self._global_attempts) >= self._global_limit:
            blocked_until = self._global_attempts[0] + self._window_seconds
        if len(attempts) >= self._per_source_limit:
            source_until = attempts[0] + self._window_seconds
            blocked_until = max(blocked_until or source_until, source_until)
        if blocked_until is not None:
            return max(1, math.ceil(blocked_until - now))

        self._global_attempts.append(now)
        attempts.append(now)
        return None


class WorkshopEventStreamLimiter:
    """Bound concurrent long-lived streams per principal and process."""

    def __init__(self, *, per_principal_limit: int = 4, global_limit: int = 32) -> None:
        if per_principal_limit < 1 or global_limit < per_principal_limit:
            raise ValueError("Event-stream concurrency bounds are invalid")
        self._per_principal_limit = per_principal_limit
        self._global_limit = global_limit
        self._active_total = 0
        self._active_by_principal: dict[PrincipalId, int] = {}

    def acquire(self, principal_id: PrincipalId) -> bool:
        if not isinstance(principal_id, PrincipalId):
            return False
        active_for_principal = self._active_by_principal.get(principal_id, 0)
        if self._active_total >= self._global_limit or active_for_principal >= self._per_principal_limit:
            return False
        self._active_total += 1
        self._active_by_principal[principal_id] = active_for_principal + 1
        return True

    def release(self, principal_id: PrincipalId) -> None:
        active_for_principal = self._active_by_principal.get(principal_id, 0)
        if active_for_principal < 1 or self._active_total < 1:
            raise RuntimeError("Event-stream capacity was released without an active claim")
        if active_for_principal == 1:
            del self._active_by_principal[principal_id]
        else:
            self._active_by_principal[principal_id] = active_for_principal - 1
        self._active_total -= 1


def _apply_client_security_headers(response: web.StreamResponse) -> None:
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["Content-Security-Policy"] = "default-src 'none'"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"


def _json_response(payload: dict[str, object], *, status: int) -> web.Response:
    response = web.json_response(payload, status=status)
    _apply_client_security_headers(response)
    return response


def _error_response(*, status: int, code: str, message: str) -> web.Response:
    return _json_response({"error": {"code": code, "message": message}}, status=status)


def _single_query_value(request: web.Request, name: str) -> str | None:
    values = request.query.getall(name, [])
    if len(values) > 1:
        raise ValueError(f"Duplicate {name} parameter")
    return values[0] if values else None


def _parse_timeline_request(request: web.Request) -> tuple[ChannelId, str | None, int]:
    if not set(request.query).issubset(_ALLOWED_TIMELINE_QUERY_PARAMETERS):
        raise ValueError("Unsupported query parameter")

    channel_id = ChannelId(request.match_info["channel_id"])
    cursor = _single_query_value(request, "cursor")
    limit_value = _single_query_value(request, "limit")
    if limit_value is None:
        limit = 50
    elif not _DECIMAL_INTEGER.fullmatch(limit_value):
        raise ValueError("Invalid limit")
    else:
        limit = int(limit_value)
    return channel_id, cursor, limit


def _parse_event_stream_request(request: web.Request) -> tuple[ChannelId, int | None]:
    if not set(request.query).issubset(_ALLOWED_EVENT_QUERY_PARAMETERS):
        raise ValueError("Unsupported query parameter")

    channel_id = ChannelId(request.match_info["channel_id"])
    last_event_ids = request.headers.getall("Last-Event-ID", [])
    if len(last_event_ids) > 1:
        raise ValueError("Duplicate Last-Event-ID header")
    value = last_event_ids[0] if last_event_ids else _single_query_value(request, "after_position")
    if value is None:
        return channel_id, None
    if not _DECIMAL_INTEGER.fullmatch(value):
        raise ValueError("Invalid timeline resume position")
    return channel_id, int(value)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _serialize_message(message: TimelineMessage) -> dict[str, object]:
    return {
        "message_id": str(message.message_id),
        "channel_id": str(message.channel_id),
        "author_principal_id": str(message.author_principal_id),
        "author_kind": message.author_kind,
        "author_display_name": message.author_display_name,
        "reply_to_message_id": (str(message.reply_to_message_id) if message.reply_to_message_id is not None else None),
        "body": message.body,
        "event_position": message.event_position,
        "created_at": _format_timestamp(message.created_at),
    }


def _serialize_run(run: DurableRun) -> dict[str, object]:
    return {
        "run_id": str(run.run_id),
        "channel_id": str(run.channel_id),
        "status": run.status.value,
        "accepted_at": _format_timestamp(run.accepted_at),
        "started_at": _format_timestamp(run.started_at) if run.started_at is not None else None,
        "terminal_at": _format_timestamp(run.terminal_at) if run.terminal_at is not None else None,
        "terminal_code": run.terminal_code,
        "cancellation_requested_at": (
            _format_timestamp(run.cancellation_requested_at) if run.cancellation_requested_at is not None else None
        ),
        "result_message_id": str(run.result_message_id) if run.result_message_id is not None else None,
    }


async def _handle_client_navigation(
    request: web.Request,
    *,
    store: WorkshopEventStore,
    authenticator: WorkshopClientAuthenticator,
) -> web.Response:
    """Return only the Workshops and channels explicitly visible to a human."""
    principal_id = await authenticator.authenticate(request)
    if not isinstance(principal_id, PrincipalId):
        response = _error_response(
            status=401,
            code="authentication_required",
            message="Authentication required",
        )
        response.headers["WWW-Authenticate"] = "Bearer"
        return response
    if request.query or request.can_read_body:
        return _error_response(
            status=400,
            code="invalid_request",
            message="Invalid navigation request",
        )

    async with store.connection.execute(
        "SELECT display_name FROM principals WHERE id = ? AND kind = 'human'",
        (principal_id,),
    ) as cursor:
        principal_row = await cursor.fetchone()
    if principal_row is None:
        return _error_response(status=403, code="access_denied", message="Access denied")

    async with store.connection.execute(
        "SELECT w.id, w.name, wm.role FROM workshop_memberships wm "
        "JOIN workshops w ON w.id = wm.workshop_id "
        "WHERE wm.principal_id = ? ORDER BY lower(w.name), w.id",
        (principal_id,),
    ) as cursor:
        workshop_rows = list(await cursor.fetchall())
    async with store.connection.execute(
        "SELECT c.workshop_id, c.id, c.kind, c.name, cm.role, a.id, a.name, "
        "CASE WHEN cara.id IS NULL THEN 0 ELSE 1 END "
        "FROM channel_memberships cm "
        "JOIN channels c ON c.id = cm.channel_id "
        "JOIN workshop_memberships wm ON wm.workshop_id = c.workshop_id "
        "AND wm.principal_id = cm.principal_id "
        "LEFT JOIN channel_agents ca ON ca.channel_id = c.id "
        "LEFT JOIN agents a ON a.id = ca.agent_id AND a.workshop_id = c.workshop_id "
        "LEFT JOIN channel_agent_runtime_assignments cara "
        "ON cara.channel_id = c.id AND cara.agent_id = a.id "
        "WHERE cm.principal_id = ? "
        "ORDER BY c.workshop_id, "
        "CASE c.kind WHEN 'direct' THEN 0 WHEN 'group' THEN 1 "
        "WHEN 'notification' THEN 2 ELSE 3 END, lower(coalesce(c.name, '')), c.id, a.name, a.id",
        (principal_id,),
    ) as cursor:
        channel_rows = list(await cursor.fetchall())
    async with store.connection.execute(
        "SELECT c.workshop_id, c.id, peer.id, peer.kind, peer.display_name "
        "FROM channel_memberships own_cm "
        "JOIN channels c ON c.id = own_cm.channel_id "
        "JOIN workshop_memberships wm ON wm.workshop_id = c.workshop_id "
        "AND wm.principal_id = own_cm.principal_id "
        "JOIN channel_memberships peer_cm ON peer_cm.channel_id = c.id "
        "AND peer_cm.principal_id != own_cm.principal_id "
        "JOIN principals peer ON peer.id = peer_cm.principal_id "
        "WHERE own_cm.principal_id = ? "
        "ORDER BY c.workshop_id, c.id, lower(peer.display_name), peer.id",
        (principal_id,),
    ) as cursor:
        participant_rows = list(await cursor.fetchall())

    channels_by_workshop: dict[str, dict[str, dict[str, object]]] = {}
    for row in channel_rows:
        workshop_id = str(row[0])
        channel_id = str(row[1])
        workshop_channels = channels_by_workshop.setdefault(workshop_id, {})
        channel = workshop_channels.setdefault(
            channel_id,
            {
                "channel_id": channel_id,
                "name": str(row[3]) if row[3] is not None else None,
                "kind": str(row[2]),
                "role": str(row[4]),
                "agents": [],
                "participants": [],
                "_runtime_assignments": [],
            },
        )
        if row[5] is not None:
            agents = channel["agents"]
            assignments = channel["_runtime_assignments"]
            if not isinstance(agents, list) or not isinstance(assignments, list):
                raise RuntimeError("Workshop navigation channel assembly failed")
            agents.append({"agent_id": str(row[5]), "name": str(row[6])})
            assignments.append(bool(row[7]))

    for row in participant_rows:
        workshop_channels = channels_by_workshop.get(str(row[0]))
        channel = workshop_channels.get(str(row[1])) if workshop_channels is not None else None
        if channel is None:
            raise RuntimeError("Workshop navigation participant assembly failed")
        participants = channel["participants"]
        if not isinstance(participants, list):
            raise RuntimeError("Workshop navigation participant assembly failed")
        participants.append(
            {
                "principal_id": str(row[2]),
                "kind": str(row[3]),
                "display_name": str(row[4]),
            }
        )

    workshops: list[dict[str, object]] = []
    for row in workshop_rows:
        workshop_id = str(row[0])
        visible_channels: list[dict[str, object]] = []
        for channel in channels_by_workshop.get(workshop_id, {}).values():
            assignments = channel.pop("_runtime_assignments")
            agents = channel["agents"]
            if not isinstance(assignments, list) or not isinstance(agents, list):
                raise RuntimeError("Workshop navigation capability assembly failed")
            channel["can_submit_commands"] = (
                channel["kind"] in {"direct", "group"} and len(agents) == 1 and assignments == [True]
            )
            visible_channels.append(channel)
        workshops.append(
            {
                "workshop_id": workshop_id,
                "name": str(row[1]),
                "role": str(row[2]),
                "channels": visible_channels,
            }
        )

    return _json_response(
        {
            "version": 1,
            "principal": {
                "principal_id": str(principal_id),
                "display_name": str(principal_row[0]),
            },
            "workshops": workshops,
        },
        status=200,
    )


async def _handle_channel_timeline(
    request: web.Request,
    *,
    store: WorkshopEventStore,
    authenticator: WorkshopClientAuthenticator,
) -> web.Response:
    principal_id = await authenticator.authenticate(request)
    if not isinstance(principal_id, PrincipalId):
        response = _error_response(
            status=401,
            code="authentication_required",
            message="Authentication required",
        )
        response.headers["WWW-Authenticate"] = "Bearer"
        return response

    try:
        channel_id, cursor, limit = _parse_timeline_request(request)
        page = await read_channel_timeline(
            store,
            principal_id=principal_id,
            channel_id=channel_id,
            authorizer=CanonicalChannelAuthorizer(store),
            cursor=cursor,
            limit=limit,
        )
    except TimelineAccessDeniedError:
        return _error_response(status=403, code="access_denied", message="Access denied")
    except (TimelineCursorError, TypeError, ValueError):
        return _error_response(
            status=400,
            code="invalid_request",
            message="Invalid timeline request",
        )

    return _json_response(
        {
            "version": 1,
            "channel_id": str(channel_id),
            "messages": [_serialize_message(message) for message in page.messages],
            "next_cursor": page.next_cursor,
            "through_position": page.through_position,
        },
        status=200,
    )


def _serialize_timeline_event(message: TimelineMessage) -> bytes:
    payload = json.dumps(
        {
            "version": 1,
            "channel_id": str(message.channel_id),
            "message": _serialize_message(message),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (f"id: {message.event_position}\nevent: timeline.message.created\ndata: {payload}\n\n").encode()


def _serialize_run_lifecycle_event(activity: ClientRunLifecycleEvent) -> bytes:
    payload = json.dumps(
        {
            "version": 1,
            "channel_id": str(activity.run.channel_id),
            "event_position": activity.event_position,
            "transition": activity.transition.value,
            "occurred_at": _format_timestamp(activity.occurred_at),
            "run": _serialize_run(activity.run),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (f"id: {activity.event_position}\nevent: run.lifecycle.changed\ndata: {payload}\n\n").encode()


async def _authorized_update_batch(
    request: web.Request,
    *,
    store: WorkshopEventStore,
    authenticator: WorkshopClientAuthenticator,
    request_lock: asyncio.Lock,
    expected_principal_id: PrincipalId | None,
    channel_id: ChannelId,
    after_position: int | None,
    reauthenticate: bool,
) -> tuple[PrincipalId | None, ClientChannelEventBatch | None]:
    async with request_lock:
        principal_id = expected_principal_id
        if reauthenticate:
            principal_id = await authenticator.authenticate(request)
            if not isinstance(principal_id, PrincipalId) or principal_id != expected_principal_id:
                return None, None
        if not isinstance(principal_id, PrincipalId):
            return None, None
        try:
            batch = await read_client_channel_events(
                store,
                principal_id=principal_id,
                channel_id=channel_id,
                authorizer=CanonicalChannelAuthorizer(store),
                after_position=after_position,
                limit=_EVENT_BATCH_SIZE,
            )
        except TimelineAccessDeniedError:
            return principal_id, None
    return principal_id, batch


async def _handle_channel_event_stream(
    request: web.Request,
    *,
    store: WorkshopEventStore,
    authenticator: WorkshopClientAuthenticator,
    request_lock: asyncio.Lock,
    poll_interval: float,
    heartbeat_interval: float,
    authentication_recheck_interval: float,
    stream_limiter: WorkshopEventStreamLimiter,
) -> web.StreamResponse:
    try:
        async with request_lock:
            principal_id = await authenticator.authenticate(request)
            if not isinstance(principal_id, PrincipalId):
                response = _error_response(
                    status=401,
                    code="authentication_required",
                    message="Authentication required",
                )
                response.headers["WWW-Authenticate"] = "Bearer"
                return response
            channel_id, after_position = _parse_event_stream_request(request)
            initial_batch = await read_client_channel_events(
                store,
                principal_id=principal_id,
                channel_id=channel_id,
                authorizer=CanonicalChannelAuthorizer(store),
                after_position=after_position,
                limit=_EVENT_BATCH_SIZE,
            )
    except TimelineAccessDeniedError:
        return _error_response(status=403, code="access_denied", message="Access denied")
    except TimelineResumeError:
        return _error_response(
            status=409,
            code="resynchronization_required",
            message="Timeline resynchronization required",
        )
    except (TypeError, ValueError):
        return _error_response(
            status=400,
            code="invalid_request",
            message="Invalid event-stream request",
        )

    if not stream_limiter.acquire(principal_id):
        response = _error_response(
            status=429,
            code="stream_capacity_exceeded",
            message="Too many active event streams",
        )
        response.headers["Retry-After"] = "5"
        return response

    response = web.StreamResponse(status=200)
    try:
        response.content_type = "text/event-stream"
        response.charset = "utf-8"
        _apply_client_security_headers(response)
        response.headers["X-Accel-Buffering"] = "no"
        await response.prepare(request)

        position = initial_batch.next_position
        batch = initial_batch
        last_heartbeat = time.monotonic()
        last_authentication_check = last_heartbeat
        await response.write(f": connected\nretry: {_SSE_RETRY_MILLISECONDS}\n\n".encode())
        while True:
            if batch.events:
                for event in batch.events:
                    if isinstance(event, ClientTimelineMessageEvent):
                        await response.write(_serialize_timeline_event(event.message))
                    else:
                        await response.write(_serialize_run_lifecycle_event(event))
                position = batch.next_position
                batch = ClientChannelEventBatch((), position)
                last_heartbeat = time.monotonic()
                continue

            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_interval:
                await response.write(b": keep-alive\n\n")
                last_heartbeat = now
            await asyncio.sleep(poll_interval)

            reauthenticate = time.monotonic() - last_authentication_check >= authentication_recheck_interval
            _, next_batch = await _authorized_update_batch(
                request,
                store=store,
                authenticator=authenticator,
                request_lock=request_lock,
                expected_principal_id=principal_id,
                channel_id=channel_id,
                after_position=position,
                reauthenticate=reauthenticate,
            )
            if reauthenticate:
                last_authentication_check = time.monotonic()
            if next_batch is None:
                break
            batch = next_batch
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        stream_limiter.release(principal_id)
    return response


async def _handle_enrollment_redemption(
    request: web.Request,
    *,
    enrollment_manager: WorkshopClientEnrollmentManager,
    rate_limiter: WorkshopEnrollmentRateLimiter,
    request_lock: asyncio.Lock,
) -> web.Response:
    """Exchange one opaque grant without accepting a client identity claim."""
    retry_after = rate_limiter.check(request)
    if retry_after is not None:
        response = _error_response(
            status=429,
            code="rate_limited",
            message="Too many enrollment attempts",
        )
        response.headers["Retry-After"] = str(retry_after)
        return response
    if request.content_type != "application/json":
        return _error_response(
            status=415,
            code="unsupported_media_type",
            message="Content-Type must be application/json",
        )

    try:
        payload = await request.json()
    except (UnicodeDecodeError, ValueError):
        payload = None
    if not isinstance(payload, dict) or set(payload) != _ENROLLMENT_REQUEST_FIELDS:
        return _error_response(
            status=400,
            code="invalid_request",
            message="Invalid enrollment request",
        )

    enrollment_token = payload["enrollment_token"]
    device_display_name = payload["device_display_name"]
    if not isinstance(enrollment_token, str) or not isinstance(device_display_name, str):
        return _error_response(
            status=400,
            code="invalid_request",
            message="Invalid enrollment request",
        )

    try:
        async with request_lock:
            redeemed = await enrollment_manager.redeem_grant(
                enrollment_token,
                device_display_name,
            )
    except EnrollmentGrantUnavailableError:
        # Malformed, unknown, expired, revoked, and reused grants deliberately
        # share one response so this endpoint is not a grant-enumeration oracle.
        return _error_response(
            status=401,
            code="enrollment_unavailable",
            message="Enrollment unavailable",
        )
    except (TypeError, ValueError):
        return _error_response(
            status=400,
            code="invalid_request",
            message="Invalid enrollment request",
        )

    return _json_response(
        {
            "version": 1,
            "device": {
                "device_id": str(redeemed.device.device_id),
                "display_name": redeemed.device.display_name,
            },
            "session": {
                "session_id": str(redeemed.session.session_id),
                "token": redeemed.session.token,
                "expires_at": _format_timestamp(redeemed.session.expires_at),
            },
        },
        status=201,
    )


async def _handle_command_submission(
    request: web.Request,
    *,
    store: WorkshopEventStore,
    authenticator: WorkshopClientAuthenticator,
    submitter: WorkshopClientCommandSubmitter,
    request_lock: asyncio.Lock,
) -> web.Response:
    """Authenticate, authorize, and durably enqueue one canonical command."""
    async with request_lock:
        principal_id = await authenticator.authenticate(request)
    if not isinstance(principal_id, PrincipalId):
        response = _error_response(
            status=401,
            code="authentication_required",
            message="Authentication required",
        )
        response.headers["WWW-Authenticate"] = "Bearer"
        return response
    if request.content_type != "application/json":
        return _error_response(
            status=415,
            code="unsupported_media_type",
            message="Content-Type must be application/json",
        )
    try:
        channel_id = ChannelId(request.match_info["channel_id"])
        payload = await request.json()
    except (UnicodeDecodeError, ValueError):
        return _error_response(status=400, code="invalid_request", message="Invalid command request")
    if not isinstance(payload, dict) or set(payload) != _COMMAND_REQUEST_FIELDS:
        return _error_response(status=400, code="invalid_request", message="Invalid command request")
    client_message_id = payload["client_message_id"]
    body = payload["body"]
    if not isinstance(client_message_id, str) or not isinstance(body, str):
        return _error_response(status=400, code="invalid_request", message="Invalid command request")

    async with request_lock:
        authorized = await CanonicalChannelAuthorizer(store).can_submit_command(principal_id, channel_id)
    if not authorized:
        return _error_response(status=403, code="access_denied", message="Access denied")

    try:
        command = ClientInboundMessage(
            principal_id=principal_id,
            channel_id=channel_id,
            client_message_id=client_message_id,
            body=body,
            occurred_at=datetime.now(UTC),
        )
    except (TypeError, ValueError):
        return _error_response(status=400, code="invalid_request", message="Invalid command request")

    try:
        result = await submitter.submit(command)
    except IdempotencyConflictError:
        return _error_response(
            status=409,
            code="idempotency_conflict",
            message="Command identity conflicts with an existing request",
        )
    except (ConversationCommandAcceptanceError, InboundBindingNotFoundError):
        return _error_response(
            status=409,
            code="command_state_conflict",
            message="Command could not be accepted in the current channel state",
        )
    except ClientCommandExecutorUnavailableError:
        response = _error_response(
            status=503,
            code="execution_unavailable",
            message="Kai cannot accept Workshop commands right now",
        )
        response.headers["Retry-After"] = "2"
        return response

    terminal = result.run.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
    return _json_response(
        {
            "version": 2,
            "message_id": str(result.acceptance.command.message.event.envelope.aggregate_id),
            "run_id": str(result.acceptance.run.run_id),
            "acceptance": result.acceptance.command.disposition.value,
            "run": _serialize_run(result.run),
        },
        status=200 if terminal else 202,
    )


async def _authorized_run(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    submitter: WorkshopClientCommandSubmitter,
    request_lock: asyncio.Lock,
) -> tuple[PrincipalId, ChannelId, DurableRun] | web.Response:
    async with request_lock:
        principal_id = await authenticator.authenticate(request)
    if not isinstance(principal_id, PrincipalId):
        response = _error_response(
            status=401,
            code="authentication_required",
            message="Authentication required",
        )
        response.headers["WWW-Authenticate"] = "Bearer"
        return response
    try:
        channel_id = ChannelId(request.match_info["channel_id"])
        run_id = RunId(request.match_info["run_id"])
    except (TypeError, ValueError):
        return _error_response(status=400, code="invalid_request", message="Invalid run request")
    try:
        run = await submitter.state(run_id)
    except RunNotFoundError:
        return _error_response(status=403, code="access_denied", message="Access denied")
    if run.channel_id != channel_id or run.requested_by_principal_id != principal_id:
        return _error_response(status=403, code="access_denied", message="Access denied")
    return principal_id, channel_id, run


async def _handle_run_state(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    submitter: WorkshopClientCommandSubmitter,
    request_lock: asyncio.Lock,
) -> web.Response:
    if request.query:
        return _error_response(status=400, code="invalid_request", message="Invalid run request")
    authorized = await _authorized_run(
        request,
        authenticator=authenticator,
        submitter=submitter,
        request_lock=request_lock,
    )
    if isinstance(authorized, web.Response):
        return authorized
    return _json_response({"version": 1, "run": _serialize_run(authorized[2])}, status=200)


async def _handle_run_cancellation(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    submitter: WorkshopClientCommandSubmitter,
    request_lock: asyncio.Lock,
) -> web.Response:
    if request.query or request.can_read_body:
        return _error_response(status=400, code="invalid_request", message="Invalid cancellation request")
    authorized = await _authorized_run(
        request,
        authenticator=authenticator,
        submitter=submitter,
        request_lock=request_lock,
    )
    if isinstance(authorized, web.Response):
        return authorized
    _, _, run = authorized
    disposition = await submitter.cancel(run.run_id)
    current = await submitter.state(run.run_id)
    return _json_response(
        {
            "version": 1,
            "cancellation": disposition.value,
            "run": _serialize_run(current),
        },
        status=(
            202
            if disposition == CanonicalCancellationDisposition.REQUESTED
            and current.status not in {RunStatus.CANCELLED, RunStatus.COMPLETED, RunStatus.FAILED}
            else 200
        ),
    )


def register_workshop_read_routes(
    app: web.Application,
    *,
    store: WorkshopEventStore,
    authenticator: WorkshopClientAuthenticator,
    request_lock: asyncio.Lock,
    event_poll_interval: float = 1.0,
    event_heartbeat_interval: float = 15.0,
    event_authentication_recheck_interval: float = 15.0,
    event_stream_limiter: WorkshopEventStreamLimiter | None = None,
) -> None:
    """Register the read-only contract on an explicitly supplied application."""
    if event_poll_interval <= 0 or event_heartbeat_interval <= 0 or event_authentication_recheck_interval <= 0:
        raise ValueError("Event-stream intervals must be positive")
    stream_limiter = event_stream_limiter or WorkshopEventStreamLimiter()

    async def handle_channel_timeline(request: web.Request) -> web.Response:
        async with request_lock:
            return await _handle_channel_timeline(
                request,
                store=store,
                authenticator=authenticator,
            )

    async def handle_client_navigation(request: web.Request) -> web.Response:
        async with request_lock:
            return await _handle_client_navigation(
                request,
                store=store,
                authenticator=authenticator,
            )

    async def handle_channel_event_stream(request: web.Request) -> web.StreamResponse:
        return await _handle_channel_event_stream(
            request,
            store=store,
            authenticator=authenticator,
            request_lock=request_lock,
            poll_interval=event_poll_interval,
            heartbeat_interval=event_heartbeat_interval,
            authentication_recheck_interval=event_authentication_recheck_interval,
            stream_limiter=stream_limiter,
        )

    app.router.add_get(_CLIENT_NAVIGATION_PATH, handle_client_navigation)
    app.router.add_get(_TIMELINE_PATH, handle_channel_timeline)
    app.router.add_get(_TIMELINE_EVENTS_PATH, handle_channel_event_stream)


def register_workshop_enrollment_routes(
    app: web.Application,
    *,
    enrollment_manager: WorkshopClientEnrollmentManager,
    rate_limiter: WorkshopEnrollmentRateLimiter,
    request_lock: asyncio.Lock,
) -> None:
    """Register grant redemption; grant issuance remains operator-only."""

    async def handle_enrollment_redemption(request: web.Request) -> web.Response:
        return await _handle_enrollment_redemption(
            request,
            enrollment_manager=enrollment_manager,
            rate_limiter=rate_limiter,
            request_lock=request_lock,
        )

    app.router.add_post(_ENROLLMENT_REDEMPTION_PATH, handle_enrollment_redemption)


def register_workshop_command_routes(
    app: web.Application,
    *,
    store: WorkshopEventStore,
    authenticator: WorkshopClientAuthenticator,
    submitter: WorkshopClientCommandSubmitter,
    request_lock: asyncio.Lock,
) -> None:
    """Register the authenticated command boundary on a supplied application."""

    async def handle_command_submission(request: web.Request) -> web.Response:
        return await _handle_command_submission(
            request,
            store=store,
            authenticator=authenticator,
            submitter=submitter,
            request_lock=request_lock,
        )

    async def handle_run_state(request: web.Request) -> web.Response:
        return await _handle_run_state(
            request,
            authenticator=authenticator,
            submitter=submitter,
            request_lock=request_lock,
        )

    async def handle_run_cancellation(request: web.Request) -> web.Response:
        return await _handle_run_cancellation(
            request,
            authenticator=authenticator,
            submitter=submitter,
            request_lock=request_lock,
        )

    app.router.add_post(_COMMAND_SUBMISSION_PATH, handle_command_submission)
    app.router.add_get(_RUN_STATE_PATH, handle_run_state)
    app.router.add_post(_RUN_CANCELLATION_PATH, handle_run_cancellation)
