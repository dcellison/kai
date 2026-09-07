"""Operational diagnostics for attempt-scoped Workshop collaboration."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from kai.workshop.collaboration_authority import (
    CollaborationDenied,
    CollaborationHostPolicy,
    CollaborationOperation,
    WorkshopCollaborationAuthority,
)
from kai.workshop.diagnostics import workshop_collaboration_authority_status
from tests.test_workshop_collaboration_authority import (
    _NOW,
    _base_identity,
    _running_attempt,
)


def test_status_is_pending_before_collaboration_schema_exists(tmp_path: Path) -> None:
    assert workshop_collaboration_authority_status(tmp_path / "missing.db") == (
        "Workshop collaboration authority: pending; attempt-scoped authority schema unavailable"
    )


async def test_status_reports_activity_and_clean_projection_integrity(tmp_path: Path) -> None:
    path = tmp_path / "kai.db"
    store, _execution, started = await _running_attempt(path)
    try:
        authority = WorkshopCollaborationAuthority(
            store,
            host_policy=CollaborationHostPolicy(
                allowed_operations=frozenset({CollaborationOperation.AGENT_DELEGATION}),
                quotas={CollaborationOperation.AGENT_DELEGATION: 1},
            ),
            token_factory=lambda: "diagnostic-proof-0000000000000000000000000001",
        )
        _grant, invocation = await authority.issue(
            started.claim,
            occurred_at=_NOW + timedelta(seconds=3),
        )
        await authority.authorize(
            invocation.token,
            CollaborationOperation.AGENT_DELEGATION,
            base_identity=_base_identity(started),
            idempotency_key="diagnostic-authorized",
            request_hash="a" * 64,
            occurred_at=_NOW + timedelta(seconds=4),
        )
        with pytest.raises(CollaborationDenied, match="exhausted"):
            await authority.authorize(
                invocation.token,
                CollaborationOperation.AGENT_DELEGATION,
                base_identity=_base_identity(started),
                idempotency_key="diagnostic-denied",
                request_hash="b" * 64,
                occurred_at=_NOW + timedelta(seconds=5),
            )
        await authority.revoke(
            invocation,
            revocation_code="qualification_complete",
            occurred_at=_NOW + timedelta(seconds=6),
        )
    finally:
        await store.close()

    status = workshop_collaboration_authority_status(path)

    assert status.startswith("Workshop collaboration authority: active;")
    assert "grants=1 (active=0, revoked=1, expired=0)" in status
    assert "operations=2 (authorized=1, denied=1, quota=1, timeouts=0)" in status
    assert "receipts=0 (succeeded=0, denied=0)" in status
    assert "adapter deliveries=0" in status
    assert "integrity gaps=0, replay gaps=0" in status


async def test_status_fails_closed_on_quota_projection_drift(tmp_path: Path) -> None:
    path = tmp_path / "kai.db"
    store, _execution, started = await _running_attempt(path)
    try:
        authority = WorkshopCollaborationAuthority(
            store,
            host_policy=CollaborationHostPolicy(
                allowed_operations=frozenset({CollaborationOperation.AGENT_DELEGATION}),
                quotas={CollaborationOperation.AGENT_DELEGATION: 1},
            ),
            token_factory=lambda: "diagnostic-proof-0000000000000000000000000002",
        )
        _grant, invocation = await authority.issue(
            started.claim,
            occurred_at=_NOW + timedelta(seconds=3),
        )
        await authority.authorize(
            invocation.token,
            CollaborationOperation.AGENT_DELEGATION,
            base_identity=_base_identity(started),
            idempotency_key="diagnostic-drift",
            request_hash="c" * 64,
            occurred_at=_NOW + timedelta(seconds=4),
        )
        await store.connection.execute(
            "UPDATE collaboration_operation_decisions SET quota_ordinal = 2 WHERE idempotency_key = 'diagnostic-drift'"
        )
        await store.connection.commit()
    finally:
        await store.close()

    status = workshop_collaboration_authority_status(path)

    assert status.startswith("Workshop collaboration authority: INCOMPLETE;")
    assert "integrity gaps=1" in status
