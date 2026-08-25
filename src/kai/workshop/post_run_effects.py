"""Durable, transport-neutral effects after successful Workshop runs."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kai.workshop.compatibility_state import WorkshopCompatibilityStateWriter
from kai.workshop.conversation_context import assemble_canonical_prior_pairs
from kai.workshop.domain import CanonicalMemoryProvenance, MessageId, RunId, RuntimeProfileId
from kai.workshop.run_lifecycle import RunStatus, WorkshopRunLifecycle
from kai.workshop.runtime_sessions import RuntimeSessionSettlement
from kai.workshop.store import WorkshopEventStore

log = logging.getLogger(__name__)

SEMANTIC_MEMORY_EFFECT = "semantic_memory_ingestion"
_POLL_SECONDS = 0.5
_MAX_ATTEMPTS = 3


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("occurred_at must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


async def enqueue_post_run_effect_in_transaction(
    store: WorkshopEventStore,
    *,
    run_id: RunId,
    source_message_id: MessageId,
    result_message_id: MessageId,
    runtime_session: RuntimeSessionSettlement,
    occurred_at: datetime,
) -> bool:
    """Queue the one common success effect in the terminal transaction."""
    if not store.connection.in_transaction:
        raise RuntimeError("enqueue_post_run_effect_in_transaction requires an active transaction")
    if runtime_session.run_id != run_id:
        raise ValueError("runtime session and post-run effect must identify the same run")
    now = _timestamp(occurred_at)
    cursor = await store.connection.execute(
        "INSERT OR IGNORE INTO workshop_post_run_effects "
        "(run_id, effect_type, runtime_profile_id, source_message_id, result_message_id, "
        "workspace, provider_session_id, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
        (
            run_id,
            SEMANTIC_MEMORY_EFFECT,
            runtime_session.runtime_profile_id,
            source_message_id,
            result_message_id,
            runtime_session.workspace,
            runtime_session.provider_session_id,
            now,
            now,
        ),
    )
    return cursor.rowcount == 1


@dataclass(frozen=True, slots=True)
class PostRunEffectReadiness:
    ready: bool
    pending: int
    executing: int
    succeeded: int
    failed: int


@dataclass(frozen=True, slots=True)
class _ClaimedEffect:
    run_id: RunId
    runtime_profile_id: RuntimeProfileId
    source_message_id: MessageId
    result_message_id: MessageId
    workspace: str
    provider_session_id: str | None
    attempt_count: int


class WorkshopPostRunEffectService:
    """Own semantic-memory ingestion for every successful execution trigger."""

    def __init__(
        self,
        store: WorkshopEventStore,
        compatibility_state: WorkshopCompatibilityStateWriter,
    ) -> None:
        self._store = store
        self._compatibility_state = compatibility_state
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    @classmethod
    async def open_and_start(
        cls,
        database_path: Path,
        compatibility_state: WorkshopCompatibilityStateWriter,
    ) -> WorkshopPostRunEffectService:
        store = await WorkshopEventStore.open(database_path)
        service = cls(store, compatibility_state)
        try:
            await store.connection.execute(
                "UPDATE workshop_post_run_effects SET status = 'pending', "
                "last_error_code = 'process_restarted', "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
                "WHERE status = 'executing'"
            )
            await store.connection.commit()
            service._task = asyncio.create_task(
                service._run(),
                name="kai-workshop-post-run-effects",
            )
            return service
        except BaseException:
            await store.close()
            raise

    @property
    def ready(self) -> bool:
        task = self._task
        return not self._closed and task is not None and not task.done()

    async def readiness(self) -> PostRunEffectReadiness:
        counts = {"pending": 0, "executing": 0, "succeeded": 0, "failed": 0}
        async with self._store.connection.execute(
            "SELECT status, COUNT(*) FROM workshop_post_run_effects GROUP BY status"
        ) as cursor:
            for row in await cursor.fetchall():
                counts[str(row[0])] = int(row[1])
        return PostRunEffectReadiness(self.ready, **counts)

    async def wait(self) -> None:
        task = self._task
        if task is None:
            raise RuntimeError("Workshop post-run effect service is not started")
        await asyncio.shield(task)

    async def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        task = self._task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._task = None
        await self._store.close()

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            claimed = await self._claim_next()
            if claimed is None:
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=_POLL_SECONDS)
                except TimeoutError:
                    continue
                return
            await self._execute(claimed)

    async def _claim_next(self) -> _ClaimedEffect | None:
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            async with connection.execute(
                "SELECT run_id, runtime_profile_id, source_message_id, result_message_id, "
                "workspace, provider_session_id, attempt_count "
                "FROM workshop_post_run_effects WHERE status = 'pending' "
                "ORDER BY created_at, run_id LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                await connection.commit()
                return None
            run_id = RunId(str(row[0]))
            cursor = await connection.execute(
                "UPDATE workshop_post_run_effects SET status = 'executing', "
                "attempt_count = attempt_count + 1, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
                "WHERE run_id = ? AND status = 'pending'",
                (run_id,),
            )
            if cursor.rowcount != 1:
                await connection.rollback()
                return None
            await connection.commit()
            return _ClaimedEffect(
                run_id=run_id,
                runtime_profile_id=RuntimeProfileId(str(row[1])),
                source_message_id=MessageId(str(row[2])),
                result_message_id=MessageId(str(row[3])),
                workspace=str(row[4]),
                provider_session_id=str(row[5]) if row[5] is not None else None,
                attempt_count=int(row[6]) + 1,
            )
        except BaseException:
            await connection.rollback()
            raise

    async def _execute(self, effect: _ClaimedEffect) -> None:
        try:
            run = await WorkshopRunLifecycle(self._store).state(effect.run_id)
            if (
                run.status != RunStatus.COMPLETED
                or run.inbound_message_id != effect.source_message_id
                or run.result_message_id != effect.result_message_id
            ):
                raise RuntimeError("Post-run effect no longer matches one successful canonical run")
            profile_state = self._compatibility_state.for_profile(effect.runtime_profile_id)
            if await profile_state.has_memory_for_run(str(effect.run_id)):
                await self._settle(effect.run_id)
                return
            async with self._store.connection.execute(
                "SELECT source.body, result.body FROM messages source, messages result "
                "WHERE source.id = ? AND result.id = ?",
                (effect.source_message_id, effect.result_message_id),
            ) as cursor:
                bodies = await cursor.fetchone()
            if bodies is None:
                raise RuntimeError("Post-run effect messages are unavailable")
            prior_pairs = await assemble_canonical_prior_pairs(
                self._store,
                run,
                limit=profile_state.memory_context_turns,
            )
            await profile_state.ingest_memory(
                prompt=str(bodies[0]),
                assistant_text=str(bodies[1]),
                session_id=effect.provider_session_id,
                workspace=effect.workspace,
                canonical_provenance=CanonicalMemoryProvenance(
                    run_id=effect.run_id,
                    source_message_id=effect.source_message_id,
                    result_message_id=effect.result_message_id,
                ),
                canonical_prior_pairs=prior_pairs,
            )
            await self._settle(effect.run_id)
        except asyncio.CancelledError:
            await self._retry(effect, "shutdown_interrupted")
            raise
        except Exception as exc:
            log.warning("Post-run effect failed for %s", effect.run_id, exc_info=True)
            await self._retry(effect, type(exc).__name__)

    async def _retry(self, effect: _ClaimedEffect, error_code: str) -> None:
        terminal = effect.attempt_count >= _MAX_ATTEMPTS
        await self._store.connection.execute(
            "UPDATE workshop_post_run_effects SET status = ?, last_error_code = ?, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), completed_at = ? "
            "WHERE run_id = ? AND status = 'executing'",
            (
                "failed" if terminal else "pending",
                error_code,
                _timestamp(datetime.now(UTC)) if terminal else None,
                effect.run_id,
            ),
        )
        await self._store.connection.commit()

    async def _settle(self, run_id: RunId) -> None:
        await self._store.connection.execute(
            "UPDATE workshop_post_run_effects SET status = ?, last_error_code = NULL, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), "
            "completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
            "WHERE run_id = ? AND status = 'executing'",
            ("succeeded", run_id),
        )
        await self._store.connection.commit()
