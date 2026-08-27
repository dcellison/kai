"""Canonical subscription routing and durable GitHub automation work."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kai import review, triage
from kai.config import ModelRole
from kai.workshop.domain import ChannelId, MessageId, PrincipalId, RuntimeProfileId
from kai.workshop.execution_state import WorkshopExecutionStateRegistry
from kai.workshop.integration_notifications import (
    IntegrationNotification,
    WorkshopIntegrationNotificationService,
)
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.runtime_state import WorkshopRuntimeStateWriter
from kai.workshop.store import WorkshopEventStore

log = logging.getLogger(__name__)

_POLL_SECONDS = 2.0
_DRAIN_SECONDS = 30.0


class GitHubAutomationRoutingError(RuntimeError):
    """Canonical GitHub policy or destination ownership is unsafe."""


@dataclass(frozen=True, slots=True)
class GitHubSubscriptionRoute:
    """One canonical subscriber and its protected execution/delivery lanes."""

    principal_id: PrincipalId
    execution_channel_id: ChannelId
    notification_channel_id: ChannelId
    runtime_profile_id: RuntimeProfileId
    pr_review_enabled: bool
    issue_triage_enabled: bool
    operations_authorized: bool


@dataclass(frozen=True, slots=True)
class GitHubAutomationEnqueueResult:
    work_id: str
    inserted: bool


@dataclass(frozen=True, slots=True)
class _WorkItem:
    work_id: str
    delivery_id: str
    kind: str
    event_type: str
    action: str
    repository: str
    item_number: int
    principal_id: PrincipalId
    execution_channel_id: ChannelId
    notification_channel_id: ChannelId
    runtime_profile_id: RuntimeProfileId
    local_repo_path: str
    payload: dict[str, Any]


class WorkshopGitHubAutomationService:
    """Own GitHub routing and restart-visible review/triage execution."""

    def __init__(
        self,
        store: WorkshopEventStore,
        runtime_pool: WorkshopRuntimePool,
        execution_state: WorkshopExecutionStateRegistry,
        runtime_state: WorkshopRuntimeStateWriter,
        notifications: WorkshopIntegrationNotificationService,
        *,
        spec_dir: str,
        review_timeout_seconds: int,
    ) -> None:
        self._store = store
        self._runtime_pool = runtime_pool
        self._execution_state = execution_state
        self._runtime_state = runtime_state
        self._notifications = notifications
        self._spec_dir = spec_dir
        self._review_timeout_seconds = review_timeout_seconds
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._active: asyncio.Task[None] | None = None
        self._closed = False

    @classmethod
    async def open_and_start(
        cls,
        database_path: Path,
        runtime_pool: WorkshopRuntimePool,
        execution_state: WorkshopExecutionStateRegistry,
        runtime_state: WorkshopRuntimeStateWriter,
        notifications: WorkshopIntegrationNotificationService,
        *,
        spec_dir: str,
        review_timeout_seconds: int,
    ) -> WorkshopGitHubAutomationService:
        service = cls(
            await WorkshopEventStore.open(database_path),
            runtime_pool,
            execution_state,
            runtime_state,
            notifications,
            spec_dir=spec_dir,
            review_timeout_seconds=review_timeout_seconds,
        )
        try:
            await service._recover_interrupted()
            service._task = asyncio.create_task(
                service._run(),
                name="kai-workshop-github-automation",
            )
            return service
        except BaseException:
            await service._store.close()
            raise

    @property
    def ready(self) -> bool:
        return not self._closed and self._task is not None and not self._task.done()

    async def routes_for_repository(self, repository: str) -> tuple[GitHubSubscriptionRoute, ...]:
        """Resolve canonical subscription policy without consulting users.yaml keys."""
        repo = repository.strip().lower()
        if not repo or "/" not in repo:
            return ()
        async with self._store.connection.execute(
            "SELECT g.principal_id, g.baseline_repos_json, g.added_repos_json, "
            "g.removed_repos_json, g.pr_review_enabled, g.issue_triage_enabled, "
            "EXISTS (SELECT 1 FROM workshop_memberships wm "
            "WHERE wm.principal_id = p.id AND wm.role = 'admin') "
            "FROM principal_github_subscriptions g "
            "JOIN principals p ON p.id = g.principal_id AND p.kind = 'human' "
            "ORDER BY g.principal_id"
        ) as cursor:
            rows = tuple(await cursor.fetchall())

        routes: list[GitHubSubscriptionRoute] = []
        for row in rows:
            principal_id = PrincipalId(str(row[0]))
            baseline = _repository_set(row[1], "baseline")
            added = _repository_set(row[2], "added")
            removed = _repository_set(row[3], "removed")
            effective = (baseline | added) - removed
            subscribed = repo in effective or (bool(row[6]) and not effective)
            if not subscribed:
                continue
            namespace = self._execution_state.maybe_for_principal_id(principal_id)
            if namespace is None:
                raise GitHubAutomationRoutingError("Canonical GitHub subscriber does not resolve one protected runtime")
            notification_channel_id = await self._notification_channel(
                principal_id,
                namespace.channel_id,
            )
            routes.append(
                GitHubSubscriptionRoute(
                    principal_id=principal_id,
                    execution_channel_id=namespace.channel_id,
                    notification_channel_id=notification_channel_id,
                    runtime_profile_id=namespace.runtime_profile_id,
                    pr_review_enabled=bool(row[4]),
                    issue_triage_enabled=bool(row[5]),
                    operations_authorized=repo in baseline,
                )
            )
        return tuple(routes)

    async def enqueue(
        self,
        *,
        delivery_id: str,
        kind: str,
        event_type: str,
        payload: dict[str, Any],
        route: GitHubSubscriptionRoute,
        local_repo_path: str | None = None,
    ) -> GitHubAutomationEnqueueResult:
        """Durably accept one authorized automation exactly once."""
        if kind not in {"pr_review", "issue_triage"}:
            raise ValueError("kind must be pr_review or issue_triage")
        if not route.operations_authorized:
            raise GitHubAutomationRoutingError("GitHub mutation is not operator-authorized")
        repository = str(payload.get("repository", {}).get("full_name", "")).strip()
        action = str(payload.get("action", "")).strip()
        item = payload.get("pull_request" if kind == "pr_review" else "issue", {})
        item_number = item.get("number") if isinstance(item, dict) else None
        if not repository or not isinstance(item_number, int) or isinstance(item_number, bool) or item_number <= 0:
            raise ValueError("GitHub automation payload has no valid repository item")
        work_id = _work_id(delivery_id, kind, route.principal_id)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        cursor = await self._store.connection.execute(
            "INSERT OR IGNORE INTO workshop_github_automation_work ("
            "work_id, delivery_id, kind, event_type, action, repository, item_number, "
            "principal_id, execution_channel_id, notification_channel_id, runtime_profile_id, "
            "local_repo_path, payload_json, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')",
            (
                work_id,
                delivery_id,
                kind,
                event_type,
                action,
                repository,
                item_number,
                route.principal_id,
                route.execution_channel_id,
                route.notification_channel_id,
                route.runtime_profile_id,
                local_repo_path or "",
                encoded,
            ),
        )
        await self._store.connection.commit()
        inserted = cursor.rowcount == 1
        if not inserted:
            async with self._store.connection.execute(
                "SELECT event_type, action, repository, item_number, principal_id, "
                "execution_channel_id, notification_channel_id, runtime_profile_id, "
                "local_repo_path, payload_json "
                "FROM workshop_github_automation_work WHERE work_id = ?",
                (work_id,),
            ) as existing_cursor:
                existing = await existing_cursor.fetchone()
            expected = (
                event_type,
                action,
                repository,
                item_number,
                str(route.principal_id),
                str(route.execution_channel_id),
                str(route.notification_channel_id),
                str(route.runtime_profile_id),
                local_repo_path or "",
                encoded,
            )
            if existing is None or tuple(existing) != expected:
                raise GitHubAutomationRoutingError("GitHub delivery identity was reused with different work")
        self._wake.set()
        return GitHubAutomationEnqueueResult(work_id, inserted)

    async def wait(self) -> None:
        task = self._task
        if task is None:
            raise RuntimeError("GitHub automation worker is not started")
        await asyncio.shield(task)

    async def stop(self) -> None:
        if self._closed:
            return
        self._stop.set()
        self._wake.set()
        active = self._active
        if active is not None and not active.done():
            try:
                await asyncio.wait_for(asyncio.shield(active), timeout=_DRAIN_SECONDS)
            except TimeoutError:
                active.cancel()
                await asyncio.gather(active, return_exceptions=True)
        task = self._task
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        self._closed = True
        self._task = None
        await self._store.close()

    async def _notification_channel(
        self,
        principal_id: PrincipalId,
        direct_channel_id: ChannelId,
    ) -> ChannelId:
        async with self._store.connection.execute(
            "SELECT c.id FROM channels c "
            "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.principal_id = ? "
            "JOIN channel_agents ca ON ca.channel_id = c.id "
            "WHERE c.kind = 'notification' ORDER BY c.id",
            (principal_id,),
        ) as cursor:
            rows = tuple(await cursor.fetchall())
        if not rows:
            return direct_channel_id
        if len(rows) != 1:
            raise GitHubAutomationRoutingError("Canonical GitHub subscriber has multiple notification destinations")
        return ChannelId(str(rows[0][0]))

    async def _recover_interrupted(self) -> None:
        await self._store.connection.execute(
            "UPDATE workshop_github_automation_work SET status = 'uncertain', "
            "last_error_code = 'process_restarted_during_execution', "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE status = 'executing'"
        )
        await self._store.connection.commit()

    async def _run(self) -> None:
        await self._report_uncertain()
        while not self._stop.is_set():
            item = await self._next_pending()
            if item is None:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=_POLL_SECONDS)
                except TimeoutError:
                    pass
                continue
            self._active = asyncio.create_task(
                self._execute(item),
                name=f"kai-workshop-github-work:{item.work_id}",
            )
            try:
                await self._active
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Canonical GitHub automation failed for %s", item.work_id)
            finally:
                self._active = None

    async def _next_pending(self) -> _WorkItem | None:
        async with self._store.connection.execute(
            "SELECT work_id, delivery_id, kind, event_type, action, repository, item_number, "
            "principal_id, execution_channel_id, notification_channel_id, runtime_profile_id, "
            "local_repo_path, payload_json FROM workshop_github_automation_work "
            "WHERE status = 'pending' ORDER BY created_at, work_id LIMIT 1"
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        payload = json.loads(str(row[12]))
        if not isinstance(payload, dict):
            await self._mark(str(row[0]), "failed", "invalid_payload")
            return None
        return _WorkItem(
            work_id=str(row[0]),
            delivery_id=str(row[1]),
            kind=str(row[2]),
            event_type=str(row[3]),
            action=str(row[4]),
            repository=str(row[5]),
            item_number=int(row[6]),
            principal_id=PrincipalId(str(row[7])),
            execution_channel_id=ChannelId(str(row[8])),
            notification_channel_id=ChannelId(str(row[9])),
            runtime_profile_id=RuntimeProfileId(str(row[10])),
            local_repo_path=str(row[11]),
            payload=payload,
        )

    async def _execute(self, item: _WorkItem) -> None:
        cursor = await self._store.connection.execute(
            "UPDATE workshop_github_automation_work SET status = 'executing', "
            "attempt_count = attempt_count + 1, last_error_code = NULL, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
            "WHERE work_id = ? AND status = 'pending'",
            (item.work_id,),
        )
        await self._store.connection.commit()
        if cursor.rowcount != 1:
            return
        namespace = self._execution_state.maybe_for_principal_channel(
            item.principal_id,
            item.execution_channel_id,
        )
        if namespace is None or namespace.runtime_profile_id != item.runtime_profile_id:
            await self._mark(item.work_id, "failed", "runtime_authority_changed")
            return
        profile = self._runtime_pool.runtime_profile(item.runtime_profile_id)
        effective_backend, effective_provider = self._runtime_pool.get_backend_provider(item.runtime_profile_id)
        runtime_state = self._runtime_state.for_profile(item.runtime_profile_id)
        token = await runtime_state.github_token()
        if not token:
            await self._notify(
                item,
                f"GitHub {item.kind.replace('_', ' ')} skipped for {item.repository}: "
                "the protected runtime has no stored GitHub token.",
                suffix="missing-token",
            )
            await self._mark(item.work_id, "failed", "github_token_missing")
            return

        async def sink(body: str) -> None:
            message_id = await self._notify(item, body, suffix="result")
            await self._store.connection.execute(
                "UPDATE workshop_github_automation_work SET canonical_message_id = ?, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE work_id = ?",
                (message_id, item.work_id),
            )
            await self._store.connection.commit()

        try:
            if item.kind == "pr_review":
                success = await review.review_pr(
                    item.payload,
                    0,
                    "",
                    agent_backend=effective_backend,
                    claude_user=profile.os_user,
                    local_repo_path=item.local_repo_path or None,
                    spec_dir=self._spec_dir,
                    provider=effective_provider,
                    timeout_s=self._review_timeout_seconds,
                    model_override=self._runtime_pool.get_role_model(
                        item.runtime_profile_id,
                        ModelRole.PR_REVIEW,
                    ),
                    github_token=token,
                    notification_sink=sink,
                )
            else:
                success = await triage.triage_issue(
                    item.payload,
                    0,
                    "",
                    agent_backend=effective_backend,
                    claude_user=profile.os_user,
                    provider=effective_provider,
                    model_override=self._runtime_pool.get_role_model(
                        item.runtime_profile_id,
                        ModelRole.ISSUE_TRIAGE,
                    ),
                    allowed_triage_projects=list(runtime_state.allowed_triage_projects),
                    github_token=token,
                    notification_sink=sink,
                )
            await self._mark(
                item.work_id, "succeeded" if success else "failed", None if success else "operation_failed"
            )
        except asyncio.CancelledError:
            await self._mark(item.work_id, "uncertain", "shutdown_interrupted")
            raise
        except Exception as exc:
            await self._mark(item.work_id, "failed", type(exc).__name__)
            raise

    async def _notify(self, item: _WorkItem, body: str, *, suffix: str) -> MessageId:
        record = await self._notifications.record_for_channel(
            IntegrationNotification(
                delivery_id=_notification_delivery_id(item, suffix),
                source="github",
                event_type=f"{item.kind}_{suffix.replace('-', '_')}",
                repository=item.repository,
                body=body,
                occurred_at=datetime.now(UTC),
            ),
            item.notification_channel_id,
        )
        return record.message_id

    async def _report_uncertain(self) -> None:
        async with self._store.connection.execute(
            "SELECT work_id, delivery_id, kind, event_type, action, repository, item_number, "
            "principal_id, execution_channel_id, notification_channel_id, runtime_profile_id, "
            "local_repo_path, payload_json FROM workshop_github_automation_work "
            "WHERE status = 'uncertain' AND last_error_code = 'process_restarted_during_execution'"
        ) as cursor:
            rows = tuple(await cursor.fetchall())
        for row in rows:
            item = _WorkItem(
                str(row[0]),
                str(row[1]),
                str(row[2]),
                str(row[3]),
                str(row[4]),
                str(row[5]),
                int(row[6]),
                PrincipalId(str(row[7])),
                ChannelId(str(row[8])),
                ChannelId(str(row[9])),
                RuntimeProfileId(str(row[10])),
                str(row[11]),
                json.loads(str(row[12])),
            )
            await self._notify(
                item,
                f"GitHub {item.kind.replace('_', ' ')} for {item.repository}#{item.item_number} "
                "was interrupted by a restart; its external side-effect state is uncertain and was not retried.",
                suffix="restart-uncertain",
            )

    async def _mark(self, work_id: str, status: str, error_code: str | None) -> None:
        await self._store.connection.execute(
            "UPDATE workshop_github_automation_work SET status = ?, last_error_code = ?, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE work_id = ?",
            (status, error_code, work_id),
        )
        await self._store.connection.commit()


def _repository_set(value: object, label: str) -> set[str]:
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise GitHubAutomationRoutingError(f"Canonical GitHub {label} repository policy is corrupt") from exc
    if not isinstance(decoded, list) or any(not isinstance(item, str) for item in decoded):
        raise GitHubAutomationRoutingError(f"Canonical GitHub {label} repository policy is corrupt")
    return {item.strip().lower() for item in decoded if item.strip()}


def _work_id(delivery_id: str, kind: str, principal_id: PrincipalId) -> str:
    token = hashlib.sha256(f"{delivery_id}\0{kind}\0{principal_id}".encode()).hexdigest()[:32]
    return f"ghw_{token}"


def _notification_delivery_id(item: _WorkItem, suffix: str) -> str:
    identity = f"{item.delivery_id}\0{item.kind}\0{item.principal_id}\0{suffix}"
    return f"github-automation-{hashlib.sha256(identity.encode()).hexdigest()[:40]}"
