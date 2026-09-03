"""Canonical self-service profile settings for Workshop humans."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime

from kai.workshop.domain import EventEnvelope, PrincipalId, WorkshopEventType, WorkshopId
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import StoredEvent, WorkshopEventStore


class WorkshopHumanProfileError(RuntimeError):
    """A canonical human profile operation failed."""


class WorkshopHumanProfileAccessDenied(WorkshopHumanProfileError):
    """The authenticated principal cannot manage a human profile."""


class WorkshopHumanProfileValidationError(WorkshopHumanProfileError):
    """A requested human profile value is invalid."""


class WorkshopHumanProfileConflict(WorkshopHumanProfileError):
    """The human profile changed after the caller loaded it."""


class WorkshopHumanProfileStorageError(WorkshopHumanProfileError):
    """Canonical human profile state is unavailable."""


@dataclass(frozen=True, slots=True)
class HumanProfileSnapshot:
    principal_id: PrincipalId
    display_name: str
    handle: str
    state_version: int
    mutation_changed: bool | None = None
    replayed: bool = False


def normalize_human_display_name(value: object) -> str:
    """Preserve intentional casing while rejecting unsafe or unusable names."""
    if not isinstance(value, str):
        raise WorkshopHumanProfileValidationError("Display name must be a string")
    normalized = value.strip()
    if not normalized or len(normalized) > 200:
        raise WorkshopHumanProfileValidationError("Display name must contain 1 through 200 characters")
    if any(unicodedata.category(character).startswith("C") for character in normalized):
        raise WorkshopHumanProfileValidationError("Display name cannot contain control characters")
    return normalized


class WorkshopHumanProfileService:
    """Read and change only the authenticated human's canonical display name."""

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store

    async def inspect(self, principal_id: PrincipalId) -> HumanProfileSnapshot:
        return await self._snapshot(principal_id)

    async def update_display_name(
        self,
        principal_id: PrincipalId,
        display_name: object,
        *,
        expected_state_version: object,
        client_operation_id: object,
    ) -> HumanProfileSnapshot:
        normalized_name = normalize_human_display_name(display_name)
        if (
            not isinstance(expected_state_version, int)
            or isinstance(expected_state_version, bool)
            or expected_state_version < 0
        ):
            raise WorkshopHumanProfileValidationError("expected_state_version must be a non-negative integer")
        if (
            not isinstance(client_operation_id, str)
            or not client_operation_id.strip()
            or len(client_operation_id) > 200
        ):
            raise WorkshopHumanProfileValidationError("client_operation_id must be a non-empty string")
        operation_id = client_operation_id.strip()
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            before = await self._snapshot(principal_id)
            idempotency_key = f"workshop-client:human-profile:{principal_id}:{operation_id}"
            existing = await self._store.event_by_idempotency_key(idempotency_key)
            if existing is not None:
                self._validate_replay(existing, principal_id, normalized_name, expected_state_version)
                await connection.rollback()
                current = await self._snapshot(principal_id)
                return HumanProfileSnapshot(
                    current.principal_id,
                    current.display_name,
                    current.handle,
                    current.state_version,
                    mutation_changed=True,
                    replayed=True,
                )
            if before.state_version != expected_state_version:
                raise WorkshopHumanProfileConflict("Human profile changed since it was loaded")
            if before.display_name == normalized_name:
                await connection.rollback()
                return HumanProfileSnapshot(
                    before.principal_id,
                    before.display_name,
                    before.handle,
                    before.state_version,
                    mutation_changed=False,
                )
            workshop_id = await self._workshop_id(principal_id)
            event = EventEnvelope.create(
                event_type=WorkshopEventType.PRINCIPAL_DISPLAY_NAME_CHANGED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="principal_profile",
                aggregate_id=principal_id,
                actor_principal_id=principal_id,
                occurred_at=datetime.now(UTC),
                idempotency_key=idempotency_key,
                payload={
                    "display_name": normalized_name,
                    "expected_state_version": expected_state_version,
                },
                metadata={"source": "workshop_client"},
            )
            result = await self._store.append_in_transaction(event)
            if not result.inserted:
                raise WorkshopHumanProfileError("New human profile event already exists")
            await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            after = await self._snapshot(principal_id)
            await connection.commit()
            return HumanProfileSnapshot(
                after.principal_id,
                after.display_name,
                after.handle,
                after.state_version,
                mutation_changed=True,
            )
        except WorkshopHumanProfileError:
            await connection.rollback()
            raise
        except Exception as exc:
            await connection.rollback()
            raise WorkshopHumanProfileStorageError("Human profile could not be saved") from exc

    async def _snapshot(self, principal_id: PrincipalId) -> HumanProfileSnapshot:
        if not isinstance(principal_id, PrincipalId):
            raise WorkshopHumanProfileAccessDenied("Human profile access denied")
        async with self._store.connection.execute(
            "SELECT p.display_name, p.display_name_state_version, hh.handle "
            "FROM principals p JOIN human_handles hh ON hh.principal_id = p.id "
            "WHERE p.id = ? AND p.kind = 'human'",
            (principal_id,),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if len(rows) != 1:
            raise WorkshopHumanProfileAccessDenied("Human profile access denied")
        return HumanProfileSnapshot(
            principal_id,
            str(rows[0][0]),
            str(rows[0][2]),
            int(rows[0][1]),
        )

    async def _workshop_id(self, principal_id: PrincipalId) -> WorkshopId:
        async with self._store.connection.execute(
            "SELECT workshop_id FROM workshop_memberships WHERE principal_id = ?",
            (principal_id,),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if len(rows) != 1:
            raise WorkshopHumanProfileAccessDenied("Human profile access denied")
        return WorkshopId(str(rows[0][0]))

    @staticmethod
    def _validate_replay(
        stored: StoredEvent,
        principal_id: PrincipalId,
        display_name: str,
        expected_state_version: int,
    ) -> None:
        envelope = stored.envelope
        if (
            envelope.event_type != WorkshopEventType.PRINCIPAL_DISPLAY_NAME_CHANGED
            or envelope.event_version != 1
            or envelope.aggregate_type != "principal_profile"
            or envelope.aggregate_id != principal_id
            or envelope.actor_principal_id != principal_id
            or envelope.payload
            != {
                "display_name": display_name,
                "expected_state_version": expected_state_version,
            }
        ):
            raise WorkshopHumanProfileConflict(
                "client_operation_id is already bound to a different human profile operation"
            )
