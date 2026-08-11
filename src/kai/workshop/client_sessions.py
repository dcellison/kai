"""Durable, revocable authentication sessions for human Workshop clients."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from aiohttp import web

from kai.workshop.domain import ClientSessionId, DeviceId, PrincipalId
from kai.workshop.store import WorkshopEventStore

_TOKEN_PREFIX = "kai_ws_v1"
_TOKEN_SECRET_PATTERN = re.compile(r"^[A-Za-z0-9_-]{43,128}$")
_MAX_TOKEN_LENGTH = 512
_DEFAULT_SESSION_LIFETIME = timedelta(days=30)
_MAX_SESSION_LIFETIME = timedelta(days=90)


class ClientDeviceUnavailableError(PermissionError):
    """A client device is absent, revoked, or not owned by the principal."""


@dataclass(frozen=True, slots=True)
class TrackedClientDevice:
    device_id: DeviceId
    principal_id: PrincipalId
    display_name: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class IssuedWorkshopClientSession:
    session_id: ClientSessionId
    device_id: DeviceId
    principal_id: PrincipalId
    token: str = field(repr=False)
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class AuthenticatedWorkshopClientSession:
    session_id: ClientSessionId
    device_id: DeviceId
    principal_id: PrincipalId


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Workshop client-session times must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _parse_token(token: object) -> tuple[ClientSessionId, str] | None:
    if not isinstance(token, str) or not token or len(token) > _MAX_TOKEN_LENGTH:
        return None
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != _TOKEN_PREFIX or not _TOKEN_SECRET_PATTERN.fullmatch(parts[2]):
        return None
    try:
        session_id = ClientSessionId(parts[1])
    except (TypeError, ValueError):
        return None
    return session_id, token


class WorkshopClientSessionManager:
    """Issue and resolve human-client sessions without storing bearer tokens."""

    def __init__(
        self,
        store: WorkshopEventStore,
        *,
        clock: Callable[[], datetime] | None = None,
        token_secret_factory: Callable[[], str] | None = None,
    ) -> None:
        self._store = store
        self._clock = clock or (lambda: datetime.now(UTC))
        self._token_secret_factory = token_secret_factory or (lambda: secrets.token_urlsafe(32))

    def _now(self) -> datetime:
        value = self._clock()
        _format_timestamp(value)
        return value.astimezone(UTC)

    async def register_device(self, principal_id: PrincipalId, display_name: str) -> TrackedClientDevice:
        if not isinstance(principal_id, PrincipalId):
            raise ValueError("principal_id must be a PrincipalId")
        if not isinstance(display_name, str) or not display_name.strip() or len(display_name) > 200:
            raise ValueError("display_name must contain 1 through 200 characters")
        normalized_name = display_name.strip()
        device_id = DeviceId.new()
        now = self._now()
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            async with connection.execute(
                "SELECT 1 FROM principals WHERE id = ? AND kind = 'human'",
                (principal_id,),
            ) as cursor:
                human_exists = await cursor.fetchone() is not None
            if not human_exists:
                raise ClientDeviceUnavailableError("Human principal is unavailable")
            await connection.execute(
                "INSERT INTO workshop_client_devices (id, principal_id, display_name, created_at) VALUES (?, ?, ?, ?)",
                (device_id, principal_id, normalized_name, _format_timestamp(now)),
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        return TrackedClientDevice(device_id, principal_id, normalized_name, now)

    async def issue_session(
        self,
        principal_id: PrincipalId,
        device_id: DeviceId,
        *,
        lifetime: timedelta = _DEFAULT_SESSION_LIFETIME,
    ) -> IssuedWorkshopClientSession:
        if not isinstance(principal_id, PrincipalId):
            raise ValueError("principal_id must be a PrincipalId")
        if not isinstance(device_id, DeviceId):
            raise ValueError("device_id must be a DeviceId")
        if not isinstance(lifetime, timedelta) or not timedelta(0) < lifetime <= _MAX_SESSION_LIFETIME:
            raise ValueError("lifetime must be greater than zero and at most 90 days")

        session_id = ClientSessionId.new()
        secret = self._token_secret_factory()
        if not isinstance(secret, str) or not _TOKEN_SECRET_PATTERN.fullmatch(secret):
            raise RuntimeError("Workshop client-session token generator returned an invalid secret")
        token = f"{_TOKEN_PREFIX}.{session_id}.{secret}"
        now = self._now()
        expires_at = now + lifetime
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            async with connection.execute(
                "SELECT 1 FROM workshop_client_devices WHERE id = ? AND principal_id = ? AND revoked_at IS NULL",
                (device_id, principal_id),
            ) as cursor:
                device_available = await cursor.fetchone() is not None
            if not device_available:
                raise ClientDeviceUnavailableError("Client device is unavailable")
            await connection.execute(
                "INSERT INTO workshop_client_sessions "
                "(id, device_id, principal_id, token_hash, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    device_id,
                    principal_id,
                    _token_hash(token),
                    _format_timestamp(now),
                    _format_timestamp(expires_at),
                ),
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        return IssuedWorkshopClientSession(session_id, device_id, principal_id, token, expires_at)

    async def authenticate_token(self, token: object) -> AuthenticatedWorkshopClientSession | None:
        parsed = _parse_token(token)
        if parsed is None:
            return None
        session_id, normalized_token = parsed
        now = self._now()
        now_text = _format_timestamp(now)
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            async with connection.execute(
                "SELECT s.principal_id, s.device_id, s.token_hash, s.expires_at, "
                "s.revoked_at, d.revoked_at FROM workshop_client_sessions s "
                "JOIN workshop_client_devices d ON d.id = s.device_id "
                "AND d.principal_id = s.principal_id WHERE s.id = ?",
                (session_id,),
            ) as cursor:
                row = await cursor.fetchone()
            valid = (
                row is not None
                and hmac.compare_digest(_token_hash(normalized_token), str(row[2]))
                and row[4] is None
                and row[5] is None
                and now < _parse_timestamp(str(row[3]))
            )
            if not valid or row is None:
                await connection.rollback()
                return None

            principal_id = PrincipalId(str(row[0]))
            device_id = DeviceId(str(row[1]))
            await connection.execute(
                "UPDATE workshop_client_sessions SET last_seen_at = ? WHERE id = ? AND revoked_at IS NULL",
                (now_text, session_id),
            )
            await connection.execute(
                "UPDATE workshop_client_devices SET last_seen_at = ? "
                "WHERE id = ? AND principal_id = ? AND revoked_at IS NULL",
                (now_text, device_id, principal_id),
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        return AuthenticatedWorkshopClientSession(session_id, device_id, principal_id)

    async def revoke_session(self, principal_id: PrincipalId, session_id: ClientSessionId) -> bool:
        if not isinstance(principal_id, PrincipalId) or not isinstance(session_id, ClientSessionId):
            raise ValueError("principal_id and session_id must be typed Workshop identifiers")
        now_text = _format_timestamp(self._now())
        connection = self._store.connection
        try:
            cursor = await connection.execute(
                "UPDATE workshop_client_sessions SET revoked_at = ? "
                "WHERE id = ? AND principal_id = ? AND revoked_at IS NULL",
                (now_text, session_id, principal_id),
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        return cursor.rowcount == 1

    async def revoke_device(self, principal_id: PrincipalId, device_id: DeviceId) -> bool:
        if not isinstance(principal_id, PrincipalId) or not isinstance(device_id, DeviceId):
            raise ValueError("principal_id and device_id must be typed Workshop identifiers")
        now_text = _format_timestamp(self._now())
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            cursor = await connection.execute(
                "UPDATE workshop_client_devices SET revoked_at = ? "
                "WHERE id = ? AND principal_id = ? AND revoked_at IS NULL",
                (now_text, device_id, principal_id),
            )
            if cursor.rowcount != 1:
                await connection.rollback()
                return False
            await connection.execute(
                "UPDATE workshop_client_sessions SET revoked_at = ? "
                "WHERE device_id = ? AND principal_id = ? AND revoked_at IS NULL",
                (now_text, device_id, principal_id),
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        return True


class WorkshopBearerSessionAuthenticator:
    """Adapt one durable human-client bearer session to the read API contract."""

    def __init__(self, sessions: WorkshopClientSessionManager) -> None:
        self._sessions = sessions

    async def authenticate(self, request: web.Request) -> PrincipalId | None:
        values = request.headers.getall("Authorization", [])
        if len(values) != 1:
            return None
        parts = values[0].split(" ")
        if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
            return None
        authenticated = await self._sessions.authenticate_token(parts[1])
        return authenticated.principal_id if authenticated is not None else None
