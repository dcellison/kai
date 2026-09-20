"""Contracts for durable canonical manual pull-request reviews."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.oneshot import OneShotSubprocessError
from kai.review import CollectionWarning, PRReviewResult
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.execution_state import WorkshopExecutionStateRegistry
from kai.workshop.review_jobs import (
    WorkshopReviewJobAccessDenied,
    WorkshopReviewJobService,
    WorkshopReviewJobValidationError,
)
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore
from tests.workshop_profiles import profile_id, profile_registry


async def _seed(path: Path) -> WorkshopExecutionStateRegistry:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                "Daniel",
                "admin",
                "telegram",
                "101",
                "101",
                runtime_profile_id=profile_id(101),
            ),
        ),
    )
    registry = await WorkshopExecutionStateRegistry.from_store(store, profile_registry(101))
    principal_id = registry.namespaces[0].principal_id
    await store.connection.execute(
        "INSERT INTO principal_github_subscriptions ("
        "principal_id, baseline_repos_json, added_repos_json, removed_repos_json, "
        "pr_review_enabled, issue_triage_enabled, pr_review_source, issue_triage_source, github_token"
        ") VALUES (?, '[\"owner/repo\"]', '[\"added/only\"]', '[]', 1, 1, "
        "'operator', 'operator', 'ghp_test')",
        (principal_id,),
    )
    await store.connection.commit()
    await store.close()
    return registry


def _dependencies(workspace: Path):
    runtime_pool = MagicMock()
    runtime_pool.get_effective_workspace = AsyncMock(return_value=workspace)
    runtime_pool.runtime_profile.return_value = SimpleNamespace(os_user="daniel")
    runtime_pool.get_backend_provider.return_value = ("codex", "openai")
    runtime_pool.get_role_model.return_value = "gpt-test"
    profile_state = SimpleNamespace(github_token=AsyncMock(return_value="ghp_test"))
    runtime_state = MagicMock()
    runtime_state.for_profile.return_value = profile_state
    return runtime_pool, runtime_state


def _result() -> PRReviewResult:
    return PRReviewResult(
        repo="owner/repo",
        pr_number=42,
        pr_title="Canonical review",
        pr_url="https://github.com/owner/repo/pull/42",
        review_text="No blocking findings.",
        collection_warnings=(CollectionWarning("related_context", "Unavailable"),),
    )


async def _open(path: Path, registry: WorkshopExecutionStateRegistry, workspace: Path):
    runtime_pool, runtime_state = _dependencies(workspace)
    service = await WorkshopReviewJobService.open_and_start(
        path,
        runtime_pool,
        registry,
        runtime_state,
        spec_dir="specs",
        review_timeout_seconds=30,
    )
    return service, runtime_pool


class TestReviewJobAuthority:
    async def test_executes_durably_and_returns_principal_scoped_artifact(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        registry = await _seed(path)
        service, runtime_pool = await _open(path, registry, tmp_path / "repo")
        try:
            authority = await service.authority_for_external_identity("telegram", "101")
            with (
                patch(
                    "kai.workshop.review_jobs.review._resolve_workspace_remote_repo",
                    new=AsyncMock(return_value="owner/repo"),
                ),
                patch(
                    "kai.workshop.review_jobs.review.generate_pr_review",
                    new=AsyncMock(return_value=_result()),
                ) as generate,
            ):
                submitted = await service.submit(
                    authority,
                    repository=None,
                    pull_request_number=42,
                    idempotency_key="telegram:101:9001",
                )
                terminal = await asyncio.wait_for(
                    service.wait_for_terminal(authority, submitted.review_job_id),
                    timeout=2,
                )

            assert terminal.status == "succeeded"
            artifact = await service.artifact(authority, terminal.review_job_id)
            assert artifact.filename == "owner-repo-pr-42-review.md"
            assert b"No blocking findings" in artifact.body
            assert artifact.warnings == (CollectionWarning("related_context", "Unavailable"),)
            recent = await service.list_recent(authority)
            assert recent[0].review_job_id == terminal.review_job_id
            assert recent[0].artifact_filename == artifact.filename
            assert recent[0].warning_count == 1
            assert generate.call_args.kwargs["local_repo_path"] == str(tmp_path / "repo")
            assert generate.call_args.kwargs["github_token"] == "ghp_test"
            assert runtime_pool.get_role_model.call_count == 1
        finally:
            await service.stop()

        reopened = await WorkshopEventStore.open(path)
        try:
            row = await (
                await reopened.connection.execute("SELECT status, COUNT(*) FROM workshop_review_jobs GROUP BY status")
            ).fetchone()
            assert tuple(row) == ("succeeded", 1)
        finally:
            await reopened.close()

    async def test_idempotent_replay_does_not_duplicate_job(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        registry = await _seed(path)
        service, _ = await _open(path, registry, tmp_path)
        try:
            authority = service.authority_for_principal(registry.namespaces[0].principal_id)
            with (
                patch(
                    "kai.workshop.review_jobs.review._resolve_workspace_remote_repo",
                    new=AsyncMock(return_value=""),
                ),
                patch(
                    "kai.workshop.review_jobs.review.generate_pr_review",
                    new=AsyncMock(return_value=_result()),
                ),
            ):
                first = await service.submit(
                    authority,
                    repository="owner/repo",
                    pull_request_number=42,
                    idempotency_key="same-key",
                )
                replay = await service.submit(
                    authority,
                    repository="owner/repo",
                    pull_request_number=42,
                    idempotency_key="same-key",
                )
                with pytest.raises(IdempotencyConflictError):
                    await service.submit(
                        authority,
                        repository="owner/repo",
                        pull_request_number=43,
                        idempotency_key="same-key",
                    )
            assert replay.review_job_id == first.review_job_id
            assert replay.replayed is True
        finally:
            await service.stop()

    async def test_self_service_subscription_never_grants_execution(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        registry = await _seed(path)
        service, _ = await _open(path, registry, tmp_path)
        try:
            authority = service.authority_for_principal(registry.namespaces[0].principal_id)
            with (
                patch(
                    "kai.workshop.review_jobs.review._resolve_workspace_remote_repo",
                    new=AsyncMock(return_value=""),
                ),
                pytest.raises(WorkshopReviewJobAccessDenied, match="not operator-authorized"),
            ):
                await service.submit(
                    authority,
                    repository="added/only",
                    pull_request_number=42,
                    idempotency_key="denied",
                )
        finally:
            await service.stop()

    async def test_timeout_is_terminal_and_explicit_retry_succeeds(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        registry = await _seed(path)
        service, _ = await _open(path, registry, tmp_path)
        try:
            authority = service.authority_for_principal(registry.namespaces[0].principal_id)
            with (
                patch(
                    "kai.workshop.review_jobs.review._resolve_workspace_remote_repo",
                    new=AsyncMock(return_value=""),
                ),
                patch(
                    "kai.workshop.review_jobs.review.generate_pr_review",
                    new=AsyncMock(side_effect=TimeoutError),
                ),
            ):
                submitted = await service.submit(
                    authority,
                    repository="owner/repo",
                    pull_request_number=42,
                    idempotency_key="timeout",
                )
                timed_out = await asyncio.wait_for(
                    service.wait_for_terminal(authority, submitted.review_job_id),
                    timeout=2,
                )
            assert timed_out.status == "timed_out"
            assert timed_out.last_error_code == "review_timeout"

            with (
                patch(
                    "kai.workshop.review_jobs.review._resolve_workspace_remote_repo",
                    new=AsyncMock(return_value=""),
                ),
                patch(
                    "kai.workshop.review_jobs.review.generate_pr_review",
                    new=AsyncMock(return_value=_result()),
                ),
            ):
                retried = await service.retry(authority, submitted.review_job_id)
                assert retried.status == "pending"
                completed = await asyncio.wait_for(
                    service.wait_for_terminal(authority, submitted.review_job_id),
                    timeout=2,
                )
            assert completed.status == "succeeded"
            assert completed.attempt_count == 2
        finally:
            await service.stop()

    async def test_transient_codex_transport_failure_retries_then_succeeds(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        registry = await _seed(path)
        service, _ = await _open(path, registry, tmp_path)
        transport_error = RuntimeError("Codex review failed")
        transport_error.__cause__ = OneShotSubprocessError(
            returncode=1,
            stderr=b"Reading prompt from stdin",
            stdout=b'{"type":"error","message":"failed to connect to websocket"}',
        )
        try:
            authority = service.authority_for_principal(registry.namespaces[0].principal_id)
            generate = AsyncMock(side_effect=(transport_error, transport_error, _result()))
            with (
                patch(
                    "kai.workshop.review_jobs.review._resolve_workspace_remote_repo",
                    new=AsyncMock(return_value=""),
                ),
                patch("kai.workshop.review_jobs.review.generate_pr_review", new=generate),
                patch("kai.workshop.review_jobs._TRANSPORT_RETRY_DELAYS_SECONDS", (0.0, 0.0)),
            ):
                submitted = await service.submit(
                    authority,
                    repository="owner/repo",
                    pull_request_number=42,
                    idempotency_key="transport-recovers",
                )
                completed = await asyncio.wait_for(
                    service.wait_for_terminal(authority, submitted.review_job_id),
                    timeout=2,
                )
            assert completed.status == "succeeded"
            assert completed.attempt_count == 3
            assert generate.await_count == 3
        finally:
            await service.stop()

    async def test_exhausted_transport_failure_has_stable_code(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        registry = await _seed(path)
        service, _ = await _open(path, registry, tmp_path)
        transport_error = RuntimeError("Codex review failed")
        transport_error.__cause__ = OneShotSubprocessError(
            returncode=1,
            stderr=b"failed to connect to websocket: task was cancelled",
        )
        try:
            authority = service.authority_for_principal(registry.namespaces[0].principal_id)
            with (
                patch(
                    "kai.workshop.review_jobs.review._resolve_workspace_remote_repo",
                    new=AsyncMock(return_value=""),
                ),
                patch(
                    "kai.workshop.review_jobs.review.generate_pr_review",
                    new=AsyncMock(side_effect=transport_error),
                ),
                patch("kai.workshop.review_jobs._TRANSPORT_RETRY_DELAYS_SECONDS", (0.0, 0.0)),
            ):
                submitted = await service.submit(
                    authority,
                    repository="owner/repo",
                    pull_request_number=42,
                    idempotency_key="transport-exhausted",
                )
                failed = await asyncio.wait_for(
                    service.wait_for_terminal(authority, submitted.review_job_id),
                    timeout=2,
                )
            assert failed.status == "failed"
            assert failed.attempt_count == 3
            assert failed.last_error_code == "review_transport_unavailable"
        finally:
            await service.stop()

    async def test_backend_failure_has_stable_code(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        registry = await _seed(path)
        service, _ = await _open(path, registry, tmp_path)
        try:
            authority = service.authority_for_principal(registry.namespaces[0].principal_id)
            with (
                patch(
                    "kai.workshop.review_jobs.review._resolve_workspace_remote_repo",
                    new=AsyncMock(return_value=""),
                ),
                patch(
                    "kai.workshop.review_jobs.review.generate_pr_review",
                    new=AsyncMock(side_effect=RuntimeError("provider rejected request")),
                ),
            ):
                submitted = await service.submit(
                    authority,
                    repository="owner/repo",
                    pull_request_number=42,
                    idempotency_key="backend-failed",
                )
                failed = await asyncio.wait_for(
                    service.wait_for_terminal(authority, submitted.review_job_id),
                    timeout=2,
                )
            assert failed.status == "failed"
            assert failed.attempt_count == 1
            assert failed.last_error_code == "review_backend_failed"
        finally:
            await service.stop()

    async def test_pending_job_can_be_cancelled_without_execution(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        registry = await _seed(path)
        store = await WorkshopEventStore.open(path)
        runtime_pool, runtime_state = _dependencies(tmp_path)
        service = WorkshopReviewJobService(
            store,
            runtime_pool,
            registry,
            runtime_state,
            spec_dir="specs",
            review_timeout_seconds=30,
        )
        try:
            authority = service.authority_for_principal(registry.namespaces[0].principal_id)
            with patch(
                "kai.workshop.review_jobs.review._resolve_workspace_remote_repo",
                new=AsyncMock(return_value=""),
            ):
                submitted = await service.submit(
                    authority,
                    repository="owner/repo",
                    pull_request_number=42,
                    idempotency_key="cancel",
                )
            cancelled = await service.cancel(authority, submitted.review_job_id)
            assert cancelled.status == "cancelled"
            assert cancelled.attempt_count == 0
        finally:
            await service.stop()

    async def test_ambiguous_inference_fails_closed(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        registry = await _seed(path)
        store = await WorkshopEventStore.open(path)
        await store.connection.execute(
            'UPDATE principal_github_subscriptions SET baseline_repos_json = \'["owner/repo","owner/other"]\''
        )
        await store.connection.commit()
        await store.close()
        service, _ = await _open(path, registry, tmp_path)
        try:
            authority = service.authority_for_principal(registry.namespaces[0].principal_id)
            with (
                patch(
                    "kai.workshop.review_jobs.review._resolve_workspace_remote_repo",
                    new=AsyncMock(return_value="unrelated/repo"),
                ),
                pytest.raises(WorkshopReviewJobValidationError, match="Could not infer"),
            ):
                await service.submit(
                    authority,
                    repository=None,
                    pull_request_number=42,
                    idempotency_key="ambiguous",
                )
        finally:
            await service.stop()

    async def test_restart_requeues_interrupted_read_only_review(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        registry = await _seed(path)
        store = await WorkshopEventStore.open(path)
        principal_id = registry.namespaces[0].principal_id
        await store.connection.execute(
            "INSERT INTO workshop_review_jobs ("
            "review_job_id, principal_id, runtime_profile_id, repository, pull_request_number, "
            "idempotency_key, request_fingerprint, status, attempt_count"
            ") VALUES ('rvj_interrupted', ?, ?, 'owner/repo', 42, 'restart', ?, 'executing', 1)",
            (principal_id, profile_id(101), "a" * 64),
        )
        await store.connection.commit()
        await store.close()

        recovery_store = await WorkshopEventStore.open(path)
        runtime_pool, runtime_state = _dependencies(tmp_path)
        service = WorkshopReviewJobService(
            recovery_store,
            runtime_pool,
            registry,
            runtime_state,
            spec_dir="specs",
            review_timeout_seconds=30,
        )
        try:
            await service._recover_interrupted()
            authority = service.authority_for_principal(principal_id)
            snapshot = await service.inspect(authority, "rvj_interrupted")
            assert snapshot.status == "pending"
            assert snapshot.last_error_code == "process_restarted_during_execution"
        finally:
            await service.stop()
