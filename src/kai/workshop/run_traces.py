"""Durable per-run trace persistence for the Workshop run inspector.

Traces deliberately do not reuse the ephemeral TTL'd preview registry:
they must survive for runs that finished long ago. Rows live exactly as
long as the run row does; run pruning deletes them by run_id in the same
pass, so there is no separate TTL.
"""

from __future__ import annotations

from datetime import datetime

from kai.backend import TraceEntry
from kai.workshop.run_execution_authority import (
    RunExecutionClaim,
    StaleRunExecutionAuthorityError,
)
from kai.workshop.store import WorkshopEventStore

# Per-run row cap. Tool calls are orders of magnitude rarer than text
# deltas, so per-entry SQLite writes are fine; the cap only bounds a
# pathological run. On overflow the new entry is dropped and a single
# synthetic marker row records the truncation, so the card can say the
# trace is incomplete instead of silently lying about completeness.
_TRACE_MAX_ENTRIES = 500
TRACE_TRUNCATION_KIND = "truncated"
_TRACE_TRUNCATION_SUMMARY = f"trace truncated at {_TRACE_MAX_ENTRIES} steps"

_INSERT_TRACE = (
    "INSERT INTO run_traces (run_id, seq, kind, tool_name, tool_use_id, "
    "summary, detail, is_diff, is_error, created_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


class WorkshopRunTraceStore:
    """Append-only run_traces writer gated by execution-claim ownership."""

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store

    async def append(
        self,
        claim: RunExecutionClaim,
        trace: TraceEntry,
        *,
        occurred_at: datetime,
    ) -> bool:
        """Append one trace row for the claim's run.

        The claim must still match the run's active fenced attempt; the
        preview registry gets the same guarantee structurally by being
        published only from the owned execution path. A superseded
        attempt raises StaleRunExecutionAuthorityError and writes
        nothing. seq is dense per run and assigned here, under the write
        transaction.

        Returns True when the entry was appended; False once the run's
        cap is reached (the marker row is written on the first overflow,
        further calls write nothing), so the caller can stop paying a
        write transaction per event for the rest of the run.
        """
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            async with connection.execute(
                "SELECT status FROM run_attempts WHERE id = ? AND run_id = ? AND owner_id = ? AND fence_token = ?",
                (claim.attempt_id, claim.run_id, claim.owner_id, claim.fence_token),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None or str(row[0]) not in ("granted", "started"):
                raise StaleRunExecutionAuthorityError("Execution claim no longer holds active authority")
            async with connection.execute(
                "SELECT COALESCE(MAX(seq), 0), COUNT(*) FROM run_traces WHERE run_id = ?",
                (claim.run_id,),
            ) as cursor:
                seq_row = await cursor.fetchone()
            assert seq_row is not None
            next_seq = int(seq_row[0]) + 1
            if int(seq_row[1]) >= _TRACE_MAX_ENTRIES:
                async with connection.execute(
                    "SELECT 1 FROM run_traces WHERE run_id = ? AND kind = ? LIMIT 1",
                    (claim.run_id, TRACE_TRUNCATION_KIND),
                ) as cursor:
                    already_marked = await cursor.fetchone()
                if already_marked is None:
                    await connection.execute(
                        _INSERT_TRACE,
                        (
                            claim.run_id,
                            next_seq,
                            TRACE_TRUNCATION_KIND,
                            None,
                            None,
                            _TRACE_TRUNCATION_SUMMARY,
                            "",
                            0,
                            0,
                            occurred_at.isoformat(),
                        ),
                    )
                await connection.commit()
                return False
            await connection.execute(
                _INSERT_TRACE,
                (
                    claim.run_id,
                    next_seq,
                    trace.kind,
                    # Absent optional fields store as NULL, never "";
                    # one representation keeps readers from having to
                    # treat the two as synonyms.
                    trace.tool_name or None,
                    trace.tool_use_id or None,
                    trace.summary,
                    trace.detail,
                    int(trace.is_diff),
                    int(trace.is_error),
                    occurred_at.isoformat(),
                ),
            )
            await connection.commit()
            return True
        except Exception:
            await connection.rollback()
            raise
