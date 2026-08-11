"""Production-unregistered HTTP contract for canonical Workshop timeline reads."""

from __future__ import annotations

import re
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
    read_channel_timeline,
)

_TIMELINE_PATH = "/v1/channels/{channel_id}/timeline"
_ENROLLMENT_REDEMPTION_PATH = "/v1/client/enrollment/redeem"
_ALLOWED_QUERY_PARAMETERS = frozenset({"cursor", "limit"})
_ENROLLMENT_REQUEST_FIELDS = frozenset({"enrollment_token", "device_display_name"})
_DECIMAL_INTEGER = re.compile(r"^[0-9]+$")


class WorkshopClientAuthenticator(Protocol):
    """Resolve a human Workshop principal from a client request."""

    async def authenticate(self, request: web.Request) -> PrincipalId | None: ...


def _json_response(payload: dict[str, object], *, status: int) -> web.Response:
    response = web.json_response(payload, status=status)
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def _error_response(*, status: int, code: str, message: str) -> web.Response:
    return _json_response({"error": {"code": code, "message": message}}, status=status)


def _single_query_value(request: web.Request, name: str) -> str | None:
    values = request.query.getall(name, [])
    if len(values) > 1:
        raise ValueError(f"Duplicate {name} parameter")
    return values[0] if values else None


def _parse_timeline_request(request: web.Request) -> tuple[ChannelId, str | None, int]:
    if not set(request.query).issubset(_ALLOWED_QUERY_PARAMETERS):
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


async def _handle_enrollment_redemption(
    request: web.Request,
    *,
    enrollment_manager: WorkshopClientEnrollmentManager,
) -> web.Response:
    """Exchange one opaque grant without accepting a client identity claim."""
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
) -> None:
    """Register the read-only contract on an explicitly supplied application.

    Production does not call this function yet. A later milestone will supply
    revocable human-client sessions before registering the route there.
    """

    async def handle_channel_timeline(request: web.Request) -> web.Response:
        return await _handle_channel_timeline(
            request,
            store=store,
            authenticator=authenticator,
        )

    app.router.add_get(_TIMELINE_PATH, handle_channel_timeline)


def register_workshop_enrollment_routes(
    app: web.Application,
    *,
    enrollment_manager: WorkshopClientEnrollmentManager,
) -> None:
    """Register grant redemption on an explicitly supplied application.

    Production does not call this function yet. Grant issuance remains a
    trusted operator/server capability and is intentionally absent here.
    """

    async def handle_enrollment_redemption(request: web.Request) -> web.Response:
        return await _handle_enrollment_redemption(
            request,
            enrollment_manager=enrollment_manager,
        )

    app.router.add_post(_ENROLLMENT_REDEMPTION_PATH, handle_enrollment_redemption)
