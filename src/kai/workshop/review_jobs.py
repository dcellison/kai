"""Durable principal-scoped authority for on-demand pull-request reviews."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import secrets
from dataclasses import dataclass
from pathlib import Path

from kai import review
from kai.config import ModelRole
from kai.oneshot import OneShotSubprocessError
from kai.workshop.domain import PrincipalId, RuntimeProfileId
from kai.workshop.execution_state import WorkshopExecutionStateRegistry
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.runtime_state import WorkshopRuntimeStateWriter
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore

log = logging.getLogger(__name__)

_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_POLL_SECONDS = 0.1
_DRAIN_SECONDS = 30.0
_MAX_ARTIFACT_BYTES = 1024 * 1024
_MAX_WARNINGS_BYTES = 64 * 1024
_MAX_TRANSPORT_ATTEMPTS = 3
_TRANSPORT_RETRY_DELAYS_SECONDS = (1.0, 3.0)
_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled", "timed_out"})


def _is_transient_review_transport_failure(exc: BaseException) -> bool:
    """Recognize only bounded Codex transport failures without exposing output."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OneShotSubprocessError):
            detail = b"\n".join((current.stderr, current.stdout)).decode(errors="replace").lower()
            return any(
                marker in detail
                for marker in (
                    "failed to connect to websocket",
                    "websocket connection is unavailable",
                    "websocket connection failed",
                )
            )
        current = current.__cause__ or current.__context__
    return False


class WorkshopReviewJobError(RuntimeError):
    """Base error for canonical manual review operations."""


class WorkshopReviewJobAccessDenied(WorkshopReviewJobError):
    """The requester does not own the requested review authority or artifact."""


class WorkshopReviewJobValidationError(WorkshopReviewJobError):
    """The requested repository or pull request is invalid or unauthorized."""


@dataclass(frozen=True, slots=True)
class ReviewJobAuthority:
    principal_id: PrincipalId
    runtime_profile_id: RuntimeProfileId


@dataclass(frozen=True, slots=True)
class ReviewJobSnapshot:
    review_job_id: str
    principal_id: PrincipalId
    runtime_profile_id: RuntimeProfileId
    repository: str
    pull_request_number: int
    status: str
    attempt_count: int
    replayed: bool
    last_error_code: str | None


@dataclass(frozen=True, slots=True)
class ReviewArtifact:
    artifact_id: str
    review_job_id: str
    filename: str
    media_type: str
    body: bytes
    sha256: str
    warnings: tuple[review.CollectionWarning, ...]


@dataclass(frozen=True, slots=True)
class _ReviewWork:
    review_job_id: str
    principal_id: PrincipalId
    runtime_profile_id: RuntimeProfileId
    repository: str
    pull_request_number: int
    local_repo_path: str


class WorkshopReviewJobService:
    """Execute and retain manual reviews independently of any client adapter."""

    def __init__(
        self,
        store: WorkshopEventStore,
        runtime_pool: WorkshopRuntimePool,
        execution_state: WorkshopExecutionStateRegistry,
        runtime_state: WorkshopRuntimeStateWriter,
        *,
        spec_dir: str,
        review_timeout_seconds: int,
    ) -> None:
        self._store = store
        self._runtime_pool = runtime_pool
        self._execution_state = execution_state
        self._runtime_state = runtime_state
        self._spec_dir = spec_dir
        self._review_timeout_seconds = review_timeout_seconds
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._active: asyncio.Task[None] | None = None
        self._active_job_id: str | None = None
        self._closed = False

    @classmethod
    async def open_and_start(
        cls,
        database_path: Path,
        runtime_pool: WorkshopRuntimePool,
        execution_state: WorkshopExecutionStateRegistry,
        runtime_state: WorkshopRuntimeStateWriter,
        *,
        spec_dir: str,
        review_timeout_seconds: int,
    ) -> WorkshopReviewJobService:
        service = cls(
            await WorkshopEventStore.open(database_path),
            runtime_pool,
            execution_state,
            runtime_state,
            spec_dir=spec_dir,
            review_timeout_seconds=review_timeout_seconds,
        )
        try:
            await service._recover_interrupted()
            service._task = asyncio.create_task(service._run(), name="kai-workshop-review-jobs")
            return service
        except BaseException:
            await service._store.close()
            raise

    @property
    def ready(self) -> bool:
        return not self._closed and self._task is not None and not self._task.done()

    async def authority_for_external_identity(self, provider: str, subject: str) -> ReviewJobAuthority:
        async with self._store.connection.execute(
            "SELECT principal_id FROM external_identities WHERE provider = ? AND external_subject = ?",
            (provider, subject),
        ) as cursor:
            rows = tuple(await cursor.fetchall())
        if len(rows) != 1:
            raise WorkshopReviewJobAccessDenied("The client identity has no unique canonical principal")
        return self.authority_for_principal(str(rows[0][0]))

    def authority_for_principal(self, principal_id: str | PrincipalId) -> ReviewJobAuthority:
        namespace = self._execution_state.maybe_for_principal_id(str(principal_id))
        if namespace is None:
            raise WorkshopReviewJobAccessDenied("The principal does not own one protected runtime")
        return ReviewJobAuthority(namespace.principal_id, namespace.runtime_profile_id)

    async def submit(
        self,
        authority: ReviewJobAuthority,
        *,
        repository: str | None,
        pull_request_number: int,
        idempotency_key: str,
    ) -> ReviewJobSnapshot:
        self._validate_authority(authority)
        if (
            isinstance(pull_request_number, bool)
            or not isinstance(pull_request_number, int)
            or pull_request_number <= 0
        ):
            raise WorkshopReviewJobValidationError("Pull request number must be a positive integer")
        if not isinstance(idempotency_key, str) or not (1 <= len(idempotency_key) <= 128):
            raise WorkshopReviewJobValidationError("Idempotency key must be a bounded non-empty string")

        baseline = await self._baseline_repositories(authority.principal_id)
        workspace = await self._runtime_pool.get_effective_workspace(authority.runtime_profile_id)
        workspace_remote_raw = await review._resolve_workspace_remote_repo(str(workspace))
        workspace_remote = workspace_remote_raw.strip().lower() if workspace_remote_raw else ""
        if repository is None:
            if workspace_remote in baseline:
                normalized_repository = workspace_remote
            elif len(baseline) == 1:
                normalized_repository = next(iter(baseline))
            else:
                raise WorkshopReviewJobValidationError(
                    "Could not infer the repository from the active workspace or operator-authorized repositories"
                )
        else:
            normalized_repository = repository.strip().lower()
            if _REPOSITORY_PATTERN.fullmatch(normalized_repository) is None:
                raise WorkshopReviewJobValidationError("Repository must use owner/name format")
        if normalized_repository not in baseline:
            raise WorkshopReviewJobAccessDenied(
                f"Repository {normalized_repository!r} is not operator-authorized for GitHub review"
            )

        local_repo_path = str(workspace) if workspace_remote == normalized_repository else ""
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "repository": normalized_repository,
                    "pull_request_number": pull_request_number,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        review_job_id = f"rvj_{secrets.token_hex(16)}"
        cursor = await self._store.connection.execute(
            "INSERT OR IGNORE INTO workshop_review_jobs ("
            "review_job_id, principal_id, runtime_profile_id, repository, "
            "pull_request_number, local_repo_path, idempotency_key, request_fingerprint, status"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
            (
                review_job_id,
                authority.principal_id,
                authority.runtime_profile_id,
                normalized_repository,
                pull_request_number,
                local_repo_path,
                idempotency_key,
                fingerprint,
            ),
        )
        await self._store.connection.commit()
        replayed = cursor.rowcount != 1
        if replayed:
            async with self._store.connection.execute(
                "SELECT review_job_id, request_fingerprint FROM workshop_review_jobs "
                "WHERE principal_id = ? AND idempotency_key = ?",
                (authority.principal_id, idempotency_key),
            ) as existing_cursor:
                existing = await existing_cursor.fetchone()
            if existing is None or str(existing[1]) != fingerprint:
                raise IdempotencyConflictError("Review idempotency key was reused with a different request")
            review_job_id = str(existing[0])
        self._wake.set()
        return await self.inspect(authority, review_job_id, replayed=replayed)

    async def inspect(
        self,
        authority: ReviewJobAuthority,
        review_job_id: str,
        *,
        replayed: bool = False,
    ) -> ReviewJobSnapshot:
        self._validate_authority(authority)
        async with self._store.connection.execute(
            "SELECT review_job_id, principal_id, runtime_profile_id, repository, "
            "pull_request_number, status, attempt_count, last_error_code "
            "FROM workshop_review_jobs WHERE review_job_id = ? AND principal_id = ?",
            (review_job_id, authority.principal_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise WorkshopReviewJobAccessDenied("Review job is not visible to this principal")
        return ReviewJobSnapshot(
            review_job_id=str(row[0]),
            principal_id=PrincipalId(str(row[1])),
            runtime_profile_id=RuntimeProfileId(str(row[2])),
            repository=str(row[3]),
            pull_request_number=int(row[4]),
            status=str(row[5]),
            attempt_count=int(row[6]),
            replayed=replayed,
            last_error_code=str(row[7]) if row[7] else None,
        )

    async def wait_for_terminal(
        self,
        authority: ReviewJobAuthority,
        review_job_id: str,
    ) -> ReviewJobSnapshot:
        while True:
            snapshot = await self.inspect(authority, review_job_id)
            if snapshot.status in _TERMINAL_STATUSES:
                return snapshot
            await asyncio.sleep(_POLL_SECONDS)

    async def artifact(
        self,
        authority: ReviewJobAuthority,
        review_job_id: str,
    ) -> ReviewArtifact:
        self._validate_authority(authority)
        async with self._store.connection.execute(
            "SELECT a.artifact_id, a.review_job_id, a.filename, a.media_type, "
            "a.body, a.sha256, a.warnings_json "
            "FROM workshop_review_artifacts a "
            "JOIN workshop_review_jobs j ON j.review_job_id = a.review_job_id "
            "WHERE a.review_job_id = ? AND a.principal_id = ? AND j.principal_id = ?",
            (review_job_id, authority.principal_id, authority.principal_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise WorkshopReviewJobAccessDenied("Review artifact is not visible to this principal")
        raw_warnings = json.loads(str(row[6]))
        return ReviewArtifact(
            artifact_id=str(row[0]),
            review_job_id=str(row[1]),
            filename=str(row[2]),
            media_type=str(row[3]),
            body=bytes(row[4]),
            sha256=str(row[5]),
            warnings=tuple(
                review.CollectionWarning(source=str(item["source"]), message=str(item["message"]))
                for item in raw_warnings
            ),
        )

    async def cancel(self, authority: ReviewJobAuthority, review_job_id: str) -> ReviewJobSnapshot:
        self._validate_authority(authority)
        cursor = await self._store.connection.execute(
            "UPDATE workshop_review_jobs SET cancellation_requested_at = "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), "
            "status = CASE WHEN status = 'pending' THEN 'cancelled' ELSE status END, "
            "terminal_at = CASE WHEN status = 'pending' THEN "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now') ELSE terminal_at END, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
            "WHERE review_job_id = ? AND principal_id = ? "
            "AND status IN ('pending', 'executing')",
            (review_job_id, authority.principal_id),
        )
        await self._store.connection.commit()
        if cursor.rowcount == 0:
            return await self.inspect(authority, review_job_id)
        if self._active_job_id == review_job_id and self._active is not None:
            self._active.cancel()
        return await self.inspect(authority, review_job_id)

    async def retry(self, authority: ReviewJobAuthority, review_job_id: str) -> ReviewJobSnapshot:
        self._validate_authority(authority)
        cursor = await self._store.connection.execute(
            "UPDATE workshop_review_jobs SET status = 'pending', cancellation_requested_at = NULL, "
            "last_error_code = NULL, started_at = NULL, terminal_at = NULL, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
            "WHERE review_job_id = ? AND principal_id = ? "
            "AND status IN ('failed', 'cancelled', 'timed_out')",
            (review_job_id, authority.principal_id),
        )
        await self._store.connection.commit()
        if cursor.rowcount == 1:
            self._wake.set()
        return await self.inspect(authority, review_job_id)

    async def wait(self) -> None:
        if self._task is None:
            raise RuntimeError("Review job worker is not started")
        await asyncio.shield(self._task)

    async def stop(self) -> None:
        if self._closed:
            return
        self._stop.set()
        self._wake.set()
        if self._active is not None and not self._active.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._active), timeout=_DRAIN_SECONDS)
            except TimeoutError:
                self._active.cancel()
                await asyncio.gather(self._active, return_exceptions=True)
            except Exception:
                # The worker loop owns operation failure reporting. Shutdown
                # must still drain and close its database after a terminal
                # review exception races this call.
                pass
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
        self._closed = True
        self._task = None
        await self._store.close()

    def _validate_authority(self, authority: ReviewJobAuthority) -> None:
        namespace = self._execution_state.maybe_for_runtime_profile_id(authority.runtime_profile_id)
        if namespace is None or namespace.principal_id != authority.principal_id:
            raise WorkshopReviewJobAccessDenied("The principal does not own this review runtime")

    async def _baseline_repositories(self, principal_id: PrincipalId) -> frozenset[str]:
        async with self._store.connection.execute(
            "SELECT baseline_repos_json FROM principal_github_subscriptions WHERE principal_id = ?",
            (principal_id,),
        ) as cursor:
            rows = tuple(await cursor.fetchall())
        if len(rows) != 1:
            raise WorkshopReviewJobAccessDenied("The principal has no unique GitHub execution policy")
        try:
            decoded = json.loads(str(rows[0][0]))
        except json.JSONDecodeError as exc:
            raise WorkshopReviewJobAccessDenied("GitHub execution policy is corrupt") from exc
        if not isinstance(decoded, list) or any(not isinstance(item, str) for item in decoded):
            raise WorkshopReviewJobAccessDenied("GitHub execution policy is corrupt")
        return frozenset(item.strip().lower() for item in decoded if _REPOSITORY_PATTERN.fullmatch(item.strip()))

    async def _recover_interrupted(self) -> None:
        await self._store.connection.execute(
            "UPDATE workshop_review_jobs SET status = 'pending', "
            "last_error_code = 'process_restarted_during_execution', started_at = NULL, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE status = 'executing'"
        )
        await self._store.connection.commit()

    async def _run(self) -> None:
        while not self._stop.is_set():
            item = await self._next_pending()
            if item is None:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=_POLL_SECONDS)
                except TimeoutError:
                    pass
                continue
            self._active_job_id = item.review_job_id
            self._active = asyncio.create_task(
                self._execute(item),
                name=f"kai-workshop-review-job:{item.review_job_id}",
            )
            try:
                await self._active
            except asyncio.CancelledError:
                if not self._stop.is_set():
                    log.info("Canonical review job %s was cancelled", item.review_job_id)
            except Exception:
                log.exception("Canonical review job %s failed", item.review_job_id)
            finally:
                self._active = None
                self._active_job_id = None

    async def _next_pending(self) -> _ReviewWork | None:
        async with self._store.connection.execute(
            "SELECT review_job_id, principal_id, runtime_profile_id, repository, "
            "pull_request_number, local_repo_path FROM workshop_review_jobs "
            "WHERE status = 'pending' ORDER BY created_at, review_job_id LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return _ReviewWork(
            review_job_id=str(row[0]),
            principal_id=PrincipalId(str(row[1])),
            runtime_profile_id=RuntimeProfileId(str(row[2])),
            repository=str(row[3]),
            pull_request_number=int(row[4]),
            local_repo_path=str(row[5]),
        )

    async def _execute(self, item: _ReviewWork) -> None:
        cursor = await self._store.connection.execute(
            "UPDATE workshop_review_jobs SET status = 'executing', "
            "attempt_count = attempt_count + 1, started_at = "
            "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), terminal_at = NULL, "
            "last_error_code = NULL, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
            "WHERE review_job_id = ? AND status = 'pending' AND cancellation_requested_at IS NULL",
            (item.review_job_id,),
        )
        await self._store.connection.commit()
        if cursor.rowcount != 1:
            return
        try:
            namespace = self._execution_state.maybe_for_principal_id(item.principal_id)
            if namespace is None or namespace.runtime_profile_id != item.runtime_profile_id:
                await self._mark(item.review_job_id, "failed", "runtime_authority_changed")
                return
            baseline = await self._baseline_repositories(item.principal_id)
            if item.repository not in baseline:
                await self._mark(item.review_job_id, "failed", "repository_authority_changed")
                return
            profile = self._runtime_pool.runtime_profile(item.runtime_profile_id)
            runtime_state = self._runtime_state.for_profile(item.runtime_profile_id)
            github_token = await runtime_state.github_token()
            if not github_token:
                await self._mark(item.review_job_id, "failed", "github_token_missing")
                return
            backend, provider = self._runtime_pool.get_backend_provider(item.runtime_profile_id)
            local_repo_path = item.local_repo_path or None
            if local_repo_path is not None:
                execution_remote = await review._resolve_workspace_remote_repo(local_repo_path)
                if not execution_remote or execution_remote.strip().lower() != item.repository:
                    # A checkout may be moved or have its remote changed while a
                    # durable job is queued. Never inject context from a path
                    # whose repository identity is no longer exact.
                    local_repo_path = None
            async with asyncio.timeout(self._review_timeout_seconds + 30):
                result = await review.generate_pr_review(
                    item.repository,
                    item.pull_request_number,
                    local_repo_path=local_repo_path,
                    spec_dir=self._spec_dir,
                    include_prior_comments=True,
                    claude_user=profile.os_user,
                    agent_backend=backend,
                    provider=provider,
                    timeout_s=self._review_timeout_seconds,
                    model_override=self._runtime_pool.get_role_model(
                        item.runtime_profile_id,
                        ModelRole.PR_REVIEW,
                    ),
                    github_token=github_token,
                )
            if not result.review_text.strip():
                await self._mark(item.review_job_id, "failed", "empty_review")
                return
            await self._store_artifact(item, result)
        except TimeoutError:
            await self._mark(item.review_job_id, "timed_out", "review_timeout")
        except asyncio.CancelledError:
            if self._stop.is_set():
                await self._mark(item.review_job_id, "pending", "shutdown_interrupted", terminal=False)
            else:
                await self._mark(item.review_job_id, "cancelled", "cancelled")
            raise
        except Exception as exc:
            if _is_transient_review_transport_failure(exc):
                attempt_count = await self._attempt_count(item.review_job_id)
                if attempt_count < _MAX_TRANSPORT_ATTEMPTS:
                    await self._mark(
                        item.review_job_id,
                        "pending",
                        "review_transport_retry",
                        terminal=False,
                    )
                    delay = _TRANSPORT_RETRY_DELAYS_SECONDS[attempt_count - 1]
                    log.warning(
                        "Canonical review job %s will retry after a transient transport failure (attempt %d/%d)",
                        item.review_job_id,
                        attempt_count,
                        _MAX_TRANSPORT_ATTEMPTS,
                    )
                    await asyncio.sleep(delay)
                    self._wake.set()
                    return
                await self._mark(item.review_job_id, "failed", "review_transport_unavailable")
            else:
                await self._mark(item.review_job_id, "failed", "review_backend_failed")
            raise

    async def _attempt_count(self, review_job_id: str) -> int:
        async with self._store.connection.execute(
            "SELECT attempt_count FROM workshop_review_jobs WHERE review_job_id = ?",
            (review_job_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise WorkshopReviewJobAccessDenied("Review job disappeared during execution")
        return int(row[0])

    async def _store_artifact(self, item: _ReviewWork, result: review.PRReviewResult) -> None:
        body = (
            f"# PR #{item.pull_request_number} review\n\n"
            f"Repository: {item.repository}\n"
            f"URL: {result.pr_url}\n\n{result.review_text}\n"
        ).encode()
        if not (1 <= len(body) <= _MAX_ARTIFACT_BYTES):
            await self._mark(item.review_job_id, "failed", "artifact_size_exceeded")
            return
        warnings_json = json.dumps(
            [{"source": warning.source, "message": warning.message} for warning in result.collection_warnings],
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(warnings_json.encode()) > _MAX_WARNINGS_BYTES:
            await self._mark(item.review_job_id, "failed", "warnings_size_exceeded")
            return
        filename_repo = item.repository.replace("/", "-")
        filename = f"{filename_repo}-pr-{item.pull_request_number}-review.md"
        artifact_id = f"rva_{secrets.token_hex(16)}"
        try:
            await self._store.connection.execute("BEGIN IMMEDIATE")
            await self._store.connection.execute(
                "INSERT INTO workshop_review_artifacts ("
                "artifact_id, review_job_id, principal_id, filename, media_type, "
                "body, byte_size, sha256, warnings_json) VALUES (?, ?, ?, ?, 'text/markdown', ?, ?, ?, ?)",
                (
                    artifact_id,
                    item.review_job_id,
                    item.principal_id,
                    filename,
                    body,
                    len(body),
                    hashlib.sha256(body).hexdigest(),
                    warnings_json,
                ),
            )
            await self._store.connection.execute(
                "UPDATE workshop_review_jobs SET status = 'succeeded', terminal_at = "
                "strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), last_error_code = NULL, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
                "WHERE review_job_id = ? AND status = 'executing'",
                (item.review_job_id,),
            )
            await self._store.connection.commit()
        except Exception:
            await self._store.connection.rollback()
            raise

    async def _mark(
        self,
        review_job_id: str,
        status: str,
        error_code: str | None,
        *,
        terminal: bool = True,
    ) -> None:
        await self._store.connection.execute(
            "UPDATE workshop_review_jobs SET status = ?, last_error_code = ?, "
            "terminal_at = CASE WHEN ? THEN strftime('%Y-%m-%dT%H:%M:%fZ', 'now') ELSE NULL END, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE review_job_id = ?",
            (status, error_code, int(terminal), review_job_id),
        )
        await self._store.connection.commit()
