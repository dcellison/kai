"""Canonical scheduled-job persistence and ownership authorization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from kai.job_types import normalize_job_type
from kai.workshop.domain import (
    AgentId,
    ChannelId,
    EventEnvelope,
    PrincipalId,
    RuntimeProfileId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.store import WorkshopEventStore


class WorkshopScheduledJobAuthorityError(RuntimeError):
    """A scheduled-job owner does not resolve to one canonical execution lane."""


@dataclass(frozen=True, slots=True)
class WorkshopScheduledJobAuthority:
    """Canonical authority allowed to own and manage scheduled work."""

    principal_id: PrincipalId
    channel_id: ChannelId
    agent_id: AgentId
    runtime_profile_id: RuntimeProfileId


@dataclass(frozen=True, slots=True)
class WorkshopScheduledJob:
    """One canonical scheduled-job definition."""

    job_id: int
    authority: WorkshopScheduledJobAuthority
    name: str
    job_type: str
    prompt: str
    schedule_type: str
    schedule_data: str
    created_at: str
    active: bool
    auto_remove: bool
    notify_on_check: bool

    def as_dict(self) -> dict[str, Any]:
        """Return the stable client-facing job representation."""
        return {
            "id": self.job_id,
            "name": self.name,
            "job_type": self.job_type,
            "prompt": self.prompt,
            "schedule_type": self.schedule_type,
            "schedule_data": self.schedule_data,
            "created_at": self.created_at,
            "active": self.active,
            "auto_remove": self.auto_remove,
            "notify_on_check": self.notify_on_check,
        }


@dataclass(frozen=True, slots=True)
class WorkshopScheduledJobUpdate:
    """Mutable fields accepted by canonical scheduled-job CRUD."""

    name: str | None = None
    prompt: str | None = None
    schedule_type: str | None = None
    schedule_data: str | None = None
    auto_remove: bool | None = None
    notify_on_check: bool | None = None


@dataclass(frozen=True, slots=True)
class WorkshopScheduledJobView:
    """Principal-facing job details without exposing internal authority IDs."""

    job: WorkshopScheduledJob
    agent_name: str
    channel_name: str
    delivery_transports: tuple[str, ...]

    def as_dict(self, *, next_run_at: str | None) -> dict[str, Any]:
        return {
            **self.job.as_dict(),
            "agent_name": self.agent_name,
            "channel_name": self.channel_name,
            "delivery_transports": list(self.delivery_transports),
            "next_run_at": next_run_at,
        }


@dataclass(frozen=True, slots=True)
class WorkshopScheduledJobCancellation:
    """Replay-safe result of a principal-scoped cancellation."""

    changed: bool
    replayed: bool
    event_position: int


_JOB_COLUMNS = (
    "id, principal_id, channel_id, agent_id, runtime_profile_id, name, "
    "job_type, prompt, schedule_type, schedule_data, created_at, active, "
    "auto_remove, notify_on_check"
)
_QUALIFIED_JOB_COLUMNS = (
    "j.id, j.principal_id, j.channel_id, j.agent_id, j.runtime_profile_id, j.name, "
    "j.job_type, j.prompt, j.schedule_type, j.schedule_data, j.created_at, j.active, "
    "j.auto_remove, j.notify_on_check"
)


class WorkshopScheduledJobStore:
    """Persist jobs using canonical ownership only."""

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store

    async def validate_authority(self, authority: WorkshopScheduledJobAuthority) -> None:
        """Require one complete direct-channel execution assignment."""
        await self._workshop_id(authority)

    async def _workshop_id(self, authority: WorkshopScheduledJobAuthority) -> WorkshopId:
        async with self._store.connection.execute(
            "SELECT c.workshop_id FROM channel_agent_runtime_assignments ra "
            "JOIN channels c ON c.id = ra.channel_id AND c.kind = 'direct' AND c.archived_at IS NULL "
            "JOIN agents a ON a.id = ra.agent_id AND a.workshop_id = c.workshop_id "
            "JOIN channel_agents ca ON ca.channel_id = c.id AND ca.agent_id = a.id "
            "JOIN channel_memberships cm ON cm.channel_id = c.id "
            "AND cm.principal_id = ? AND cm.role = 'owner' "
            "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
            "JOIN workshop_memberships wm ON wm.workshop_id = c.workshop_id "
            "AND wm.principal_id = p.id "
            "WHERE ra.channel_id = ? AND ra.agent_id = ? AND ra.runtime_profile_id = ?",
            (
                authority.principal_id,
                authority.channel_id,
                authority.agent_id,
                authority.runtime_profile_id,
            ),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if len(rows) != 1:
            raise WorkshopScheduledJobAuthorityError(
                "Scheduled-job authority does not resolve to one canonical execution lane"
            )
        return WorkshopId(str(rows[0][0]))

    async def _append_change_in_transaction(
        self,
        *,
        authority: WorkshopScheduledJobAuthority,
        event_type: WorkshopEventType,
        job_id: int,
        actor_principal_id: PrincipalId | None,
        idempotency_key: str | None = None,
    ) -> int:
        workshop_id = await self._workshop_id(authority)
        result = await self._store.append_in_transaction(
            EventEnvelope.create(
                event_type=event_type,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="principal",
                aggregate_id=authority.principal_id,
                actor_principal_id=actor_principal_id,
                occurred_at=datetime.now(UTC),
                idempotency_key=idempotency_key,
                payload={
                    "job_id": job_id,
                    "principal_id": str(authority.principal_id),
                    "channel_id": str(authority.channel_id),
                    "agent_id": str(authority.agent_id),
                    "runtime_profile_id": str(authority.runtime_profile_id),
                },
            )
        )
        return result.event.position

    async def create(
        self,
        authority: WorkshopScheduledJobAuthority,
        *,
        name: str,
        job_type: str,
        prompt: str,
        schedule_type: str,
        schedule_data: str,
        auto_remove: bool = False,
        notify_on_check: bool = False,
    ) -> WorkshopScheduledJob:
        await self.validate_authority(authority)
        canonical_job_type = normalize_job_type(job_type)
        connection = self._store.connection
        try:
            cursor = await connection.execute(
                "INSERT INTO workshop_scheduled_jobs ("
                "principal_id, channel_id, agent_id, runtime_profile_id, name, job_type, "
                "prompt, schedule_type, schedule_data, auto_remove, notify_on_check"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    authority.principal_id,
                    authority.channel_id,
                    authority.agent_id,
                    authority.runtime_profile_id,
                    name,
                    canonical_job_type,
                    prompt,
                    schedule_type,
                    schedule_data,
                    int(auto_remove),
                    int(notify_on_check),
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("Canonical scheduled-job insert returned no ID")
            job_id = int(cursor.lastrowid)
            await self._append_change_in_transaction(
                authority=authority,
                event_type=WorkshopEventType.SCHEDULED_JOB_CREATED,
                job_id=job_id,
                actor_principal_id=authority.principal_id,
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        job = await self.get(job_id, authority, active_only=False)
        if job is None:
            raise RuntimeError("Canonical scheduled job disappeared after creation")
        return job

    async def list_active(
        self,
        authority: WorkshopScheduledJobAuthority,
    ) -> tuple[WorkshopScheduledJob, ...]:
        await self.validate_authority(authority)
        async with self._store.connection.execute(
            f"SELECT {_JOB_COLUMNS} FROM workshop_scheduled_jobs "
            "WHERE principal_id = ? AND channel_id = ? AND agent_id = ? "
            "AND runtime_profile_id = ? AND active = 1 ORDER BY id",
            self._owner_values(authority),
        ) as cursor:
            return tuple(self._from_row(row) for row in await cursor.fetchall())

    async def list_active_for_principal(
        self,
        principal_id: PrincipalId,
    ) -> tuple[WorkshopScheduledJobView, ...]:
        """List every active job owned by one principal across canonical lanes."""
        async with self._store.connection.execute(
            f"SELECT {_QUALIFIED_JOB_COLUMNS}"
            ", a.name, COALESCE(c.name, a.name), "
            "COALESCE(GROUP_CONCAT(DISTINCT cb.transport), '') "
            "FROM workshop_scheduled_jobs j "
            "JOIN channels c ON c.id = j.channel_id "
            "JOIN agents a ON a.id = j.agent_id AND a.workshop_id = c.workshop_id "
            "LEFT JOIN channel_bindings cb ON cb.channel_id = c.id "
            "WHERE j.principal_id = ? AND j.active = 1 "
            "GROUP BY j.id ORDER BY j.id",
            (principal_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        views: list[WorkshopScheduledJobView] = []
        for row in rows:
            job = self._from_row(row)
            await self.validate_authority(job.authority)
            transports = tuple(sorted(item for item in str(row[16]).split(",") if item))
            views.append(
                WorkshopScheduledJobView(
                    job=job,
                    agent_name=str(row[14]),
                    channel_name=str(row[15]),
                    delivery_transports=transports,
                )
            )
        return tuple(views)

    async def get_for_principal(
        self,
        job_id: int,
        principal_id: PrincipalId,
        *,
        active_only: bool = True,
    ) -> WorkshopScheduledJob | None:
        clause = " AND active = 1" if active_only else ""
        async with self._store.connection.execute(
            f"SELECT {_JOB_COLUMNS} FROM workshop_scheduled_jobs WHERE id = ? AND principal_id = ?{clause}",
            (job_id, principal_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        job = self._from_row(row)
        await self.validate_authority(job.authority)
        return job

    async def get(
        self,
        job_id: int,
        authority: WorkshopScheduledJobAuthority,
        *,
        active_only: bool = True,
    ) -> WorkshopScheduledJob | None:
        await self.validate_authority(authority)
        async with self._store.connection.execute(
            f"SELECT {_JOB_COLUMNS} FROM workshop_scheduled_jobs WHERE id = ? "
            "AND principal_id = ? AND channel_id = ? AND agent_id = ? "
            f"AND runtime_profile_id = ?{' AND active = 1' if active_only else ''}",
            (job_id, *self._owner_values(authority)),
        ) as cursor:
            row = await cursor.fetchone()
        return None if row is None else self._from_row(row)

    async def get_for_scheduler(
        self,
        job_id: int,
        *,
        active_only: bool = True,
    ) -> WorkshopScheduledJob | None:
        async with self._store.connection.execute(
            f"SELECT {_JOB_COLUMNS} FROM workshop_scheduled_jobs WHERE id = ?"
            f"{' AND active = 1' if active_only else ''}",
            (job_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        job = self._from_row(row)
        await self.validate_authority(job.authority)
        return job

    async def active_ids(self) -> set[int]:
        async with self._store.connection.execute(
            "SELECT id FROM workshop_scheduled_jobs WHERE active = 1 ORDER BY id"
        ) as cursor:
            return {int(row[0]) for row in await cursor.fetchall()}

    async def update(
        self,
        job_id: int,
        authority: WorkshopScheduledJobAuthority,
        update: WorkshopScheduledJobUpdate,
    ) -> bool:
        await self.validate_authority(authority)
        fields: list[str] = []
        values: list[object] = []
        for column, value in (
            ("name", update.name),
            ("prompt", update.prompt),
            ("schedule_type", update.schedule_type),
            ("schedule_data", update.schedule_data),
        ):
            if value is not None:
                fields.append(f"{column} = ?")
                values.append(value)
        for column, value in (
            ("auto_remove", update.auto_remove),
            ("notify_on_check", update.notify_on_check),
        ):
            if value is not None:
                fields.append(f"{column} = ?")
                values.append(int(value))
        if not fields:
            return False
        connection = self._store.connection
        try:
            cursor = await connection.execute(
                f"UPDATE workshop_scheduled_jobs SET {', '.join(fields)} WHERE id = ? "
                "AND principal_id = ? AND channel_id = ? AND agent_id = ? "
                "AND runtime_profile_id = ? AND active = 1",
                (*values, job_id, *self._owner_values(authority)),
            )
            if cursor.rowcount == 1:
                await self._append_change_in_transaction(
                    authority=authority,
                    event_type=WorkshopEventType.SCHEDULED_JOB_UPDATED,
                    job_id=job_id,
                    actor_principal_id=authority.principal_id,
                )
            await connection.commit()
            return cursor.rowcount == 1
        except Exception:
            await connection.rollback()
            raise

    async def delete(self, job_id: int, authority: WorkshopScheduledJobAuthority) -> bool:
        await self.validate_authority(authority)
        connection = self._store.connection
        try:
            cursor = await connection.execute(
                "DELETE FROM workshop_scheduled_jobs WHERE id = ? AND principal_id = ? "
                "AND channel_id = ? AND agent_id = ? AND runtime_profile_id = ?",
                (job_id, *self._owner_values(authority)),
            )
            if cursor.rowcount == 1:
                await self._append_change_in_transaction(
                    authority=authority,
                    event_type=WorkshopEventType.SCHEDULED_JOB_CANCELLED,
                    job_id=job_id,
                    actor_principal_id=authority.principal_id,
                )
            await connection.commit()
            return cursor.rowcount == 1
        except Exception:
            await connection.rollback()
            raise

    async def cancel_for_principal(
        self,
        job_id: int,
        principal_id: PrincipalId,
        client_operation_id: str,
    ) -> WorkshopScheduledJobCancellation | None:
        """Deactivate one job with replay-safe principal authority."""
        key = f"scheduled-job-cancel:{principal_id}:{client_operation_id}"
        existing = await self._store.event_by_idempotency_key(key)
        if existing is not None:
            payload = existing.envelope.payload
            if (
                existing.envelope.event_type != WorkshopEventType.SCHEDULED_JOB_CANCELLED
                or existing.envelope.aggregate_id != principal_id
                or payload.get("job_id") != job_id
            ):
                raise ValueError("client_operation_id was already used for another cancellation")
            return WorkshopScheduledJobCancellation(False, True, existing.position)
        job = await self.get_for_principal(job_id, principal_id)
        if job is None:
            return None
        connection = self._store.connection
        try:
            cursor = await connection.execute(
                "UPDATE workshop_scheduled_jobs SET active = 0 WHERE id = ? AND principal_id = ? AND active = 1",
                (job_id, principal_id),
            )
            if cursor.rowcount != 1:
                await connection.rollback()
                return None
            position = await self._append_change_in_transaction(
                authority=job.authority,
                event_type=WorkshopEventType.SCHEDULED_JOB_CANCELLED,
                job_id=job_id,
                actor_principal_id=principal_id,
                idempotency_key=key,
            )
            await connection.commit()
            return WorkshopScheduledJobCancellation(True, False, position)
        except Exception:
            await connection.rollback()
            raise

    async def deactivate(
        self,
        job_id: int,
        authority: WorkshopScheduledJobAuthority | None = None,
    ) -> bool:
        resolved = await self.get_for_scheduler(job_id) if authority is None else await self.get(job_id, authority)
        if resolved is None:
            return False
        connection = self._store.connection
        try:
            cursor = await connection.execute(
                "UPDATE workshop_scheduled_jobs SET active = 0 WHERE id = ? AND active = 1",
                (job_id,),
            )
            if cursor.rowcount == 1:
                await self._append_change_in_transaction(
                    authority=resolved.authority,
                    event_type=WorkshopEventType.SCHEDULED_JOB_DEACTIVATED,
                    job_id=job_id,
                    actor_principal_id=None,
                )
            await connection.commit()
            return cursor.rowcount == 1
        except Exception:
            await connection.rollback()
            raise

    @staticmethod
    def _owner_values(authority: WorkshopScheduledJobAuthority) -> tuple[str, str, str, str]:
        return (
            str(authority.principal_id),
            str(authority.channel_id),
            str(authority.agent_id),
            str(authority.runtime_profile_id),
        )

    @staticmethod
    def _from_row(row: Any) -> WorkshopScheduledJob:
        return WorkshopScheduledJob(
            job_id=int(row[0]),
            authority=WorkshopScheduledJobAuthority(
                PrincipalId(str(row[1])),
                ChannelId(str(row[2])),
                AgentId(str(row[3])),
                RuntimeProfileId(str(row[4])),
            ),
            name=str(row[5]),
            job_type=normalize_job_type(str(row[6])),
            prompt=str(row[7]),
            schedule_type=str(row[8]),
            schedule_data=str(row[9]),
            created_at=str(row[10]),
            active=bool(row[11]),
            auto_remove=bool(row[12]),
            notify_on_check=bool(row[13]),
        )
