"""SQLite event-store and projection contracts for Kai Workshop."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import aiosqlite

from kai.workshop.domain import EventEnvelope, EventId, PrincipalId, WorkshopId, parse_opaque_id
from kai.workshop.schema import migrate_workshop_schema

_PROJECTION_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,127}$")


class IdempotencyConflictError(RuntimeError):
    """An event identity was reused for different semantic content."""


class EventIntegrityError(RuntimeError):
    """A stored event no longer matches its recorded semantic hash."""


@dataclass(frozen=True, slots=True)
class StoredEvent:
    position: int
    envelope: EventEnvelope
    recorded_at: datetime


@dataclass(frozen=True, slots=True)
class AppendResult:
    event: StoredEvent
    inserted: bool


@dataclass(frozen=True, slots=True)
class ProjectionCheckpoint:
    name: str
    version: int
    last_position: int


class Projection(Protocol):
    """A deterministic projection that can be rebuilt from the event log."""

    name: str
    version: int

    async def reset(self, connection: aiosqlite.Connection) -> None: ...

    async def apply(self, connection: aiosqlite.Connection, event: StoredEvent) -> None: ...


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _restrict_database_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)


class WorkshopEventStore:
    """Append-only Workshop event storage with deterministic replay."""

    def __init__(self, connection: aiosqlite.Connection, *, owns_connection: bool) -> None:
        self._connection = connection
        self._owns_connection = owns_connection

    @classmethod
    async def open(cls, path: Path) -> WorkshopEventStore:
        _restrict_database_path(path)
        connection = await aiosqlite.connect(str(path))
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA busy_timeout=5000")
        try:
            await migrate_workshop_schema(connection)
        except Exception:
            await connection.close()
            raise
        return cls(connection, owns_connection=True)

    @classmethod
    async def attach(cls, connection: aiosqlite.Connection) -> WorkshopEventStore:
        """Attach to an existing Kai connection without taking ownership."""
        connection.row_factory = aiosqlite.Row
        await migrate_workshop_schema(connection)
        return cls(connection, owns_connection=False)

    @classmethod
    def from_initialized_connection(cls, connection: aiosqlite.Connection) -> WorkshopEventStore:
        """Use a connection whose Workshop schema was initialized by Kai startup."""
        return cls(connection, owns_connection=False)

    @property
    def connection(self) -> aiosqlite.Connection:
        return self._connection

    async def close(self) -> None:
        if self._owns_connection:
            await self._connection.close()

    async def schema_version(self) -> int:
        async with self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) FROM workshop_schema_migrations"
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise RuntimeError("Workshop schema version query returned no row")
        return int(row[0])

    async def schema_tables(self) -> set[str]:
        async with self._connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ) as cursor:
            return {str(row[0]) for row in await cursor.fetchall()}

    async def append(self, envelope: EventEnvelope) -> AppendResult:
        """Append once, returning the prior event for a semantic retry."""
        try:
            await self._connection.execute("BEGIN IMMEDIATE")
            result = await self.append_in_transaction(envelope)
            await self._connection.commit()
            return result
        except Exception:
            await self._connection.rollback()
            raise

    async def append_in_transaction(self, envelope: EventEnvelope) -> AppendResult:
        """Append without committing, so a caller can atomically persist related state."""
        if not self._connection.in_transaction:
            raise RuntimeError("append_in_transaction requires an active transaction")
        try:
            cursor = await self._connection.execute(
                """
                INSERT INTO event_log (
                    event_id, envelope_version, event_type, event_version,
                    workshop_id, aggregate_type, aggregate_id,
                    actor_principal_id, occurred_at, idempotency_key,
                    payload_json, metadata_json, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope.event_id,
                    envelope.envelope_version,
                    envelope.event_type,
                    envelope.event_version,
                    envelope.workshop_id,
                    envelope.aggregate_type,
                    envelope.aggregate_id,
                    envelope.actor_principal_id,
                    _format_timestamp(envelope.occurred_at),
                    envelope.idempotency_key,
                    envelope.payload_json,
                    envelope.metadata_json,
                    envelope.content_hash,
                ),
            )
        except aiosqlite.IntegrityError as exc:
            existing = await self._find_existing(envelope)
            if existing is None:
                raise
            if existing.envelope.content_hash != envelope.content_hash:
                identity = envelope.idempotency_key or str(envelope.event_id)
                raise IdempotencyConflictError(
                    f"Event identity {identity!r} was reused with different content"
                ) from exc
            return AppendResult(event=existing, inserted=False)

        position = cursor.lastrowid
        if position is None:
            raise RuntimeError("SQLite did not return an event position")
        return AppendResult(event=await self._event_at_position(int(position)), inserted=True)

    async def _find_existing(self, envelope: EventEnvelope) -> StoredEvent | None:
        clauses = ["event_id = ?"]
        parameters: list[str] = [str(envelope.event_id)]
        if envelope.idempotency_key is not None:
            clauses.append("idempotency_key = ?")
            parameters.append(envelope.idempotency_key)
        async with self._connection.execute(
            f"SELECT * FROM event_log WHERE {' OR '.join(clauses)} ORDER BY position",
            parameters,
        ) as cursor:
            rows = list(await cursor.fetchall())
        if len(rows) > 1:
            raise IdempotencyConflictError("event_id and idempotency_key resolve to different existing events")
        return self._stored_event_from_row(rows[0]) if rows else None

    async def _event_at_position(self, position: int) -> StoredEvent:
        async with self._connection.execute("SELECT * FROM event_log WHERE position = ?", (position,)) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise RuntimeError(f"Event position {position} disappeared before commit")
        return self._stored_event_from_row(row)

    def _stored_event_from_row(self, row: aiosqlite.Row) -> StoredEvent:
        actor = row["actor_principal_id"]
        envelope = EventEnvelope(
            event_id=EventId(row["event_id"]),
            envelope_version=int(row["envelope_version"]),
            event_type=row["event_type"],
            event_version=int(row["event_version"]),
            workshop_id=WorkshopId(row["workshop_id"]),
            aggregate_type=row["aggregate_type"],
            aggregate_id=parse_opaque_id(row["aggregate_id"]),
            actor_principal_id=PrincipalId(actor) if actor is not None else None,
            occurred_at=_parse_timestamp(row["occurred_at"]),
            idempotency_key=row["idempotency_key"],
            payload=json.loads(row["payload_json"]),
            metadata=json.loads(row["metadata_json"]),
        )
        if envelope.content_hash != row["content_hash"]:
            raise EventIntegrityError(f"Stored event {envelope.event_id} failed its content integrity check")
        return StoredEvent(
            position=int(row["position"]),
            envelope=envelope,
            recorded_at=_parse_timestamp(row["recorded_at"]),
        )

    async def read_events(self, *, after_position: int = 0, limit: int | None = None) -> list[StoredEvent]:
        if after_position < 0:
            raise ValueError("after_position must be non-negative")
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive when supplied")
        sql = "SELECT * FROM event_log WHERE position > ? ORDER BY position"
        parameters: list[int] = [after_position]
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(limit)
        async with self._connection.execute(sql, parameters) as cursor:
            rows = await cursor.fetchall()
        return [self._stored_event_from_row(row) for row in rows]

    async def event_by_idempotency_key(self, idempotency_key: str) -> StoredEvent | None:
        if not idempotency_key.strip():
            raise ValueError("idempotency_key must be non-empty")
        async with self._connection.execute(
            "SELECT * FROM event_log WHERE idempotency_key = ?",
            (idempotency_key,),
        ) as cursor:
            row = await cursor.fetchone()
        return self._stored_event_from_row(row) if row is not None else None

    async def rebuild_projection(self, projection: Projection) -> ProjectionCheckpoint:
        """Atomically reset and replay one projection from position zero."""
        if not _PROJECTION_NAME_PATTERN.fullmatch(projection.name):
            raise ValueError("Projection name must be a lowercase identifier")
        if not isinstance(projection.version, int) or isinstance(projection.version, bool) or projection.version < 1:
            raise ValueError("Projection version must be a positive integer")

        try:
            await self._connection.execute("BEGIN IMMEDIATE")
            await projection.reset(self._connection)
            events = await self.read_events()
            for event in events:
                await projection.apply(self._connection, event)
            last_position = events[-1].position if events else 0
            updated_at = _format_timestamp(datetime.now(UTC))
            await self._connection.execute(
                """
                INSERT INTO projection_checkpoints (name, version, last_position, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    version = excluded.version,
                    last_position = excluded.last_position,
                    updated_at = excluded.updated_at
                """,
                (projection.name, projection.version, last_position, updated_at),
            )
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise
        return ProjectionCheckpoint(
            name=projection.name,
            version=projection.version,
            last_position=last_position,
        )

    async def project_pending(self, projection: Projection) -> ProjectionCheckpoint:
        """Atomically apply events after a projection checkpoint."""
        if not _PROJECTION_NAME_PATTERN.fullmatch(projection.name):
            raise ValueError("Projection name must be a lowercase identifier")
        if not isinstance(projection.version, int) or isinstance(projection.version, bool) or projection.version < 1:
            raise ValueError("Projection version must be a positive integer")

        try:
            await self._connection.execute("BEGIN IMMEDIATE")
            async with self._connection.execute(
                "SELECT version, last_position FROM projection_checkpoints WHERE name = ?",
                (projection.name,),
            ) as cursor:
                checkpoint_row = await cursor.fetchone()

            if checkpoint_row is None or int(checkpoint_row[0]) < projection.version:
                await projection.reset(self._connection)
                after_position = 0
            elif int(checkpoint_row[0]) > projection.version:
                raise RuntimeError(
                    f"Projection {projection.name!r} is newer than this build: "
                    f"stored={int(checkpoint_row[0])}, supported={projection.version}"
                )
            else:
                after_position = int(checkpoint_row[1])

            events = await self.read_events(after_position=after_position)
            for event in events:
                await projection.apply(self._connection, event)
            last_position = events[-1].position if events else after_position
            updated_at = _format_timestamp(datetime.now(UTC))
            await self._connection.execute(
                """
                INSERT INTO projection_checkpoints (name, version, last_position, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    version = excluded.version,
                    last_position = excluded.last_position,
                    updated_at = excluded.updated_at
                """,
                (projection.name, projection.version, last_position, updated_at),
            )
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise
        return ProjectionCheckpoint(
            name=projection.name,
            version=projection.version,
            last_position=last_position,
        )
