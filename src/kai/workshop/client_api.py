"""Authenticated HTTP contracts for Workshop enrollment and read-only synchronization."""

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
from kai.workshop.client_sessions import EnrollmentGrantUnavailableError, WorkshopClientEnrollmentManager
from kai.workshop.domain import ChannelId, PrincipalId
from kai.workshop.store import WorkshopEventStore
from kai.workshop.timeline import (
    TimelineAccessDeniedError,
    TimelineCursorError,
    TimelineMessage,
    TimelineResumeError,
    TimelineUpdateBatch,
    read_channel_timeline,
    read_channel_timeline_updates,
)

_TIMELINE_PATH = "/v1/channels/{channel_id}/timeline"
_TIMELINE_EVENTS_PATH = "/v1/channels/{channel_id}/events"
_ENROLLMENT_REDEMPTION_PATH = "/v1/client/enrollment/redeem"
_ALLOWED_TIMELINE_QUERY_PARAMETERS = frozenset({"cursor", "limit"})
_ALLOWED_EVENT_QUERY_PARAMETERS = frozenset({"after_position"})
_ENROLLMENT_REQUEST_FIELDS = frozenset({"enrollment_token", "device_display_name"})
_DECIMAL_INTEGER = re.compile(r"^[0-9]+$")
_EVENT_BATCH_SIZE = 100
_SSE_RETRY_MILLISECONDS = 2000


class WorkshopClientAuthenticator(Protocol):
    """Resolve a human Workshop principal from a client request."""

    async def authenticate(self, request: web.Request) -> PrincipalId | None: ...


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
) -> tuple[PrincipalId | None, TimelineUpdateBatch | None]:
    async with request_lock:
        principal_id = expected_principal_id
        if reauthenticate:
            principal_id = await authenticator.authenticate(request)
            if not isinstance(principal_id, PrincipalId) or principal_id != expected_principal_id:
                return None, None
        if not isinstance(principal_id, PrincipalId):
            return None, None
        try:
            batch = await read_channel_timeline_updates(
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
            initial_batch = await read_channel_timeline_updates(
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
            if batch.messages:
                for message in batch.messages:
                    await response.write(_serialize_timeline_event(message))
                position = batch.next_position
                batch = TimelineUpdateBatch((), position)
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
