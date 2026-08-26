"""Canonical ownership and backfill for scheduled jobs and GitHub policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import aiosqlite

from kai.workshop.execution_state import (
    WorkshopExecutionStateNamespace,
    WorkshopExecutionStateRegistry,
)

if TYPE_CHECKING:
    from kai.config import Config
    from kai.workshop.runtime_profiles import ProtectedRuntimeProfile, WorkshopRuntimeProfileRegistry


class WorkshopOperationalStateError(RuntimeError):
    """Protected operational state cannot resolve one canonical owner."""


@dataclass(frozen=True, slots=True)
class WorkshopOperationalStateMigration:
    """Aggregate result of one idempotent operational-state reconciliation."""

    profiles: int
    newly_migrated: int
    jobs: int
    github_subscriptions: int


@dataclass(frozen=True, slots=True)
class _GitHubSubscriptionSeed:
    baseline_repos_json: str
    added_repos_json: str
    removed_repos_json: str
    pr_review_enabled: int
    issue_triage_enabled: int
    pr_review_source: str
    issue_triage_source: str
    github_token: str | None
    allowed_triage_projects_json: str
    has_legacy_policy: bool

    def database_values(self) -> tuple[object, ...]:
        return (
            self.baseline_repos_json,
            self.added_repos_json,
            self.removed_repos_json,
            self.pr_review_enabled,
            self.issue_triage_enabled,
            self.pr_review_source,
            self.issue_triage_source,
            self.github_token,
            self.allowed_triage_projects_json,
        )


@dataclass(frozen=True, slots=True)
class _OperatorGitHubPolicy:
    baseline_repos_json: str
    pr_review_enabled: int
    issue_triage_enabled: int
    pr_review_source: str
    issue_triage_source: str
    allowed_triage_projects_json: str
    has_user_config: bool

    def database_values(self) -> tuple[object, ...]:
        return (
            self.baseline_repos_json,
            self.pr_review_enabled,
            self.issue_triage_enabled,
            self.pr_review_source,
            self.issue_triage_source,
            self.allowed_triage_projects_json,
        )


async def reconcile_workshop_operational_state(
    connection: aiosqlite.Connection,
    registry: WorkshopExecutionStateRegistry,
    config: Config,
    runtime_profiles: WorkshopRuntimeProfileRegistry | None,
) -> WorkshopOperationalStateMigration:
    """Make canonical owners authoritative for protected jobs and GitHub policy."""
    pending = tuple(
        item for item in await _pending_namespaces(connection, registry) if item.legacy_runtime_key is not None
    )
    seeds: dict[str, _GitHubSubscriptionSeed] = {}
    if pending:
        for namespace in pending:
            seed = await _github_seed(connection, namespace, config)
            existing = seeds.get(str(namespace.principal_id))
            if (
                existing is not None
                and existing.has_legacy_policy
                and seed.has_legacy_policy
                and existing.database_values() != seed.database_values()
            ):
                raise WorkshopOperationalStateError(
                    "Multiple protected runtime profiles for one human contain conflicting "
                    "legacy GitHub subscription policy; reconcile the legacy settings before migration"
                )
            if existing is None or (seed.has_legacy_policy and not existing.has_legacy_policy):
                seeds[str(namespace.principal_id)] = seed

    operator_policies: dict[str, _OperatorGitHubPolicy] = {}
    for namespace in registry.namespaces:
        if runtime_profiles is None:
            legacy_key = namespace.require_legacy_runtime_key()
            user = config.get_user_config(legacy_key)
            policy = _operator_github_policy_from_user(user)
        else:
            policy = _operator_github_policy(runtime_profiles.resolve(namespace.runtime_profile_id))
        principal_id = str(namespace.principal_id)
        existing = operator_policies.get(principal_id)
        if (
            existing is not None
            and existing.has_user_config
            and policy.has_user_config
            and existing.database_values() != policy.database_values()
        ):
            raise WorkshopOperationalStateError(
                "Multiple protected runtime profiles for one human contain conflicting "
                "operator GitHub policy; reconcile users.yaml before startup"
            )
        if existing is None or (policy.has_user_config and not existing.has_user_config):
            operator_policies[principal_id] = policy

    jobs = github_subscriptions = newly_migrated = 0
    try:
        await connection.execute("BEGIN IMMEDIATE")
        for namespace in registry.namespaces:
            policy = operator_policies[str(namespace.principal_id)]
            if namespace.legacy_runtime_key is None:
                github_subscriptions += await _ensure_canonical_github_subscription(
                    connection,
                    namespace,
                    policy,
                )
                await _sync_operator_github_policy(connection, namespace, policy)
                await _verify_namespace(connection, namespace)
                continue
            scheduled_jobs_migrated = await _scheduled_job_migration_owner(connection, namespace)
            migrated = await _migration_owner(connection, namespace)
            if migrated:
                if not scheduled_jobs_migrated:
                    job_count = await _backfill_jobs(connection, namespace)
                    await _verify_legacy_jobs_for_cutover(connection, namespace)
                    await _record_scheduled_job_migration(connection, namespace, job_count)
                    jobs += job_count
                await _migrate_runtime_key_github_token(connection, namespace)
                await _sync_operator_github_policy(
                    connection,
                    namespace,
                    policy,
                )
                await _verify_namespace(connection, namespace)
                continue

            job_count = 0
            if not scheduled_jobs_migrated:
                job_count = await _backfill_jobs(connection, namespace)
                await _verify_legacy_jobs_for_cutover(connection, namespace)
                await _record_scheduled_job_migration(connection, namespace, job_count)
            github_count = await _backfill_github_subscription(
                connection,
                namespace,
                seeds[str(namespace.principal_id)],
            )
            await _sync_operator_github_policy(
                connection,
                namespace,
                policy,
            )
            await _verify_namespace(connection, namespace)
            await connection.execute(
                "INSERT INTO workshop_operational_state_migrations ("
                "runtime_profile_id, runtime_config_id, principal_id, channel_id, agent_id, "
                "jobs_count, github_subscription_count"
                ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    namespace.runtime_profile_id,
                    namespace.require_legacy_runtime_key(),
                    namespace.principal_id,
                    namespace.channel_id,
                    namespace.agent_id,
                    job_count,
                    github_count,
                ),
            )
            newly_migrated += 1
            jobs += job_count
            github_subscriptions += github_count
        await connection.commit()
    except Exception:
        await connection.rollback()
        raise
    return WorkshopOperationalStateMigration(
        profiles=len(registry.namespaces),
        newly_migrated=newly_migrated,
        jobs=jobs,
        github_subscriptions=github_subscriptions,
    )


async def _pending_namespaces(
    connection: aiosqlite.Connection,
    registry: WorkshopExecutionStateRegistry,
) -> tuple[WorkshopExecutionStateNamespace, ...]:
    async with connection.execute("SELECT runtime_profile_id FROM workshop_operational_state_migrations") as cursor:
        migrated = {str(row[0]) for row in await cursor.fetchall()}
    return tuple(namespace for namespace in registry.namespaces if str(namespace.runtime_profile_id) not in migrated)


async def _migration_owner(
    connection: aiosqlite.Connection,
    namespace: WorkshopExecutionStateNamespace,
) -> bool:
    async with connection.execute(
        "SELECT runtime_config_id, principal_id, channel_id, agent_id "
        "FROM workshop_operational_state_migrations WHERE runtime_profile_id = ?",
        (namespace.runtime_profile_id,),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        return False
    recorded = (int(row[0]), str(row[1]), str(row[2]), str(row[3]))
    current = (
        namespace.require_legacy_runtime_key(),
        str(namespace.principal_id),
        str(namespace.channel_id),
        str(namespace.agent_id),
    )
    if recorded != current:
        raise WorkshopOperationalStateError(
            "Canonical operational-state migration conflicts with current protected ownership "
            f"for runtime profile {namespace.runtime_profile_id}; restore its recorded canonical "
            "assignment or restore the database from backup"
        )
    return True


async def _scheduled_job_migration_owner(
    connection: aiosqlite.Connection,
    namespace: WorkshopExecutionStateNamespace,
) -> bool:
    async with connection.execute(
        "SELECT runtime_config_id, principal_id, channel_id, agent_id "
        "FROM workshop_scheduled_job_migrations WHERE runtime_profile_id = ?",
        (namespace.runtime_profile_id,),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        return False
    recorded = (int(row[0]), str(row[1]), str(row[2]), str(row[3]))
    current = (
        namespace.require_legacy_runtime_key(),
        str(namespace.principal_id),
        str(namespace.channel_id),
        str(namespace.agent_id),
    )
    if recorded != current:
        raise WorkshopOperationalStateError(
            "Canonical scheduled-job migration conflicts with current protected ownership "
            f"for runtime profile {namespace.runtime_profile_id}; restore its recorded canonical "
            "assignment or restore the database from backup"
        )
    return True


async def _record_scheduled_job_migration(
    connection: aiosqlite.Connection,
    namespace: WorkshopExecutionStateNamespace,
    legacy_jobs_count: int,
) -> None:
    await connection.execute(
        "INSERT INTO workshop_scheduled_job_migrations ("
        "runtime_profile_id, runtime_config_id, principal_id, channel_id, agent_id, "
        "legacy_jobs_count) VALUES (?, ?, ?, ?, ?, ?)",
        (
            namespace.runtime_profile_id,
            namespace.require_legacy_runtime_key(),
            namespace.principal_id,
            namespace.channel_id,
            namespace.agent_id,
            legacy_jobs_count,
        ),
    )


async def _legacy_setting(
    connection: aiosqlite.Connection,
    key: str,
) -> str | None:
    async with connection.execute("SELECT value FROM settings WHERE key = ?", (key,)) as cursor:
        row = await cursor.fetchone()
    return str(row[0]) if row is not None else None


def _repos(value: str | None, *, key: str) -> list[str]:
    if value is None:
        return []
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise WorkshopOperationalStateError(f"Legacy {key} is not valid JSON") from exc
    if not isinstance(decoded, list) or any(not isinstance(item, str) for item in decoded):
        raise WorkshopOperationalStateError(f"Legacy {key} must be a JSON string array")
    return sorted({item.strip().lower() for item in decoded if item.strip()})


def _toggle(value: str | None, *, key: str) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise WorkshopOperationalStateError(f"Legacy {key} must be true or false")


async def _github_seed(
    connection: aiosqlite.Connection,
    namespace: WorkshopExecutionStateNamespace,
    config: Config,
) -> _GitHubSubscriptionSeed:
    legacy_id = namespace.require_legacy_runtime_key()
    user = config.get_user_config(legacy_id)
    baseline = sorted({repo.strip().lower() for repo in (user.github_repos if user else []) if repo.strip()})
    added = await _legacy_setting(connection, f"github_repos_added:{legacy_id}")
    removed = await _legacy_setting(connection, f"github_repos_removed:{legacy_id}")
    pr_review = _toggle(await _legacy_setting(connection, f"pr_review:{legacy_id}"), key="pr_review")
    issue_triage = _toggle(
        await _legacy_setting(connection, f"issue_triage:{legacy_id}"),
        key="issue_triage",
    )
    github_token = await _legacy_setting(connection, f"github_token:{legacy_id}")
    operator_pr_review = user.pr_review if user is not None else None
    operator_issue_triage = user.issue_triage if user is not None else None
    has_legacy_policy = user is not None or any(
        value is not None for value in (added, removed, pr_review, issue_triage)
    )

    return _GitHubSubscriptionSeed(
        baseline_repos_json=json.dumps(baseline, separators=(",", ":")),
        added_repos_json=json.dumps(
            _repos(added, key="github_repos_added"),
            separators=(",", ":"),
        ),
        removed_repos_json=json.dumps(
            _repos(removed, key="github_repos_removed"),
            separators=(",", ":"),
        ),
        pr_review_enabled=int(pr_review if pr_review is not None else (operator_pr_review or False)),
        issue_triage_enabled=int(issue_triage if issue_triage is not None else (operator_issue_triage or False)),
        pr_review_source=(
            "user" if pr_review is not None else "operator" if operator_pr_review is not None else "default"
        ),
        issue_triage_source=(
            "user" if issue_triage is not None else "operator" if operator_issue_triage is not None else "default"
        ),
        github_token=github_token,
        allowed_triage_projects_json=json.dumps(
            sorted(set(getattr(user, "allowed_triage_projects", []) if user is not None else [])),
            separators=(",", ":"),
        ),
        has_legacy_policy=has_legacy_policy,
    )


def _operator_github_policy(
    profile: ProtectedRuntimeProfile,
) -> _OperatorGitHubPolicy:
    baseline = sorted(set(profile.github_repos))
    pr_review = profile.pr_review
    issue_triage = profile.issue_triage
    return _OperatorGitHubPolicy(
        baseline_repos_json=json.dumps(baseline, separators=(",", ":")),
        pr_review_enabled=int(pr_review or False),
        issue_triage_enabled=int(issue_triage or False),
        pr_review_source="operator" if pr_review is not None else "default",
        issue_triage_source="operator" if issue_triage is not None else "default",
        allowed_triage_projects_json=json.dumps(
            sorted(set(profile.allowed_triage_projects)),
            separators=(",", ":"),
        ),
        has_user_config=True,
    )


def _operator_github_policy_from_user(user: object | None) -> _OperatorGitHubPolicy:
    """Compatibility only for direct unit/dev callers without protected policy."""
    repos = getattr(user, "github_repos", []) if user is not None else []
    pr_review = getattr(user, "pr_review", None) if user is not None else None
    issue_triage = getattr(user, "issue_triage", None) if user is not None else None
    allowed = getattr(user, "allowed_triage_projects", []) if user is not None else []
    return _OperatorGitHubPolicy(
        baseline_repos_json=json.dumps(sorted({repo.strip().lower() for repo in repos}), separators=(",", ":")),
        pr_review_enabled=int(pr_review or False),
        issue_triage_enabled=int(issue_triage or False),
        pr_review_source="operator" if pr_review is not None else "default",
        issue_triage_source="operator" if issue_triage is not None else "default",
        allowed_triage_projects_json=json.dumps(sorted(set(allowed)), separators=(",", ":")),
        has_user_config=user is not None,
    )


async def _backfill_jobs(
    connection: aiosqlite.Connection,
    namespace: WorkshopExecutionStateNamespace,
) -> int:
    await connection.execute(
        "INSERT OR IGNORE INTO workshop_job_owners ("
        "job_id, principal_id, channel_id, agent_id, runtime_profile_id"
        ") SELECT id, ?, ?, ?, ? FROM jobs WHERE chat_id = ?",
        (
            namespace.principal_id,
            namespace.channel_id,
            namespace.agent_id,
            namespace.runtime_profile_id,
            namespace.require_legacy_runtime_key(),
        ),
    )
    job_cursor = await connection.execute(
        "INSERT OR IGNORE INTO workshop_scheduled_jobs ("
        "id, principal_id, channel_id, agent_id, runtime_profile_id, name, job_type, "
        "prompt, schedule_type, schedule_data, created_at, active, auto_remove, notify_on_check"
        ") SELECT j.id, o.principal_id, o.channel_id, o.agent_id, o.runtime_profile_id, "
        "j.name, j.job_type, j.prompt, j.schedule_type, j.schedule_data, "
        "strftime('%Y-%m-%dT%H:%M:%fZ', j.created_at || 'Z'), j.active, "
        "j.auto_remove, j.notify_on_check FROM jobs j "
        "JOIN workshop_job_owners o ON o.job_id = j.id "
        "JOIN principals p ON p.id = o.principal_id AND p.kind = 'human' "
        "JOIN channels c ON c.id = o.channel_id AND c.kind = 'direct' "
        "JOIN channel_memberships cm ON cm.channel_id = c.id "
        "AND cm.principal_id = p.id AND cm.role = 'owner' "
        "JOIN agents a ON a.id = o.agent_id AND a.workshop_id = c.workshop_id "
        "JOIN channel_agent_runtime_assignments ra ON ra.channel_id = c.id "
        "AND ra.agent_id = a.id AND ra.runtime_profile_id = o.runtime_profile_id "
        "WHERE j.chat_id = ?",
        (namespace.require_legacy_runtime_key(),),
    )
    return job_cursor.rowcount


async def _backfill_github_subscription(
    connection: aiosqlite.Connection,
    namespace: WorkshopExecutionStateNamespace,
    seed: _GitHubSubscriptionSeed,
) -> int:
    async with connection.execute(
        "SELECT 1 FROM workshop_operational_state_migrations WHERE principal_id = ? LIMIT 1",
        (namespace.principal_id,),
    ) as cursor:
        sibling_migrated = await cursor.fetchone() is not None
    if sibling_migrated:
        async with connection.execute(
            "SELECT 1 FROM principal_github_subscriptions WHERE principal_id = ?",
            (namespace.principal_id,),
        ) as cursor:
            if await cursor.fetchone() is None:
                raise WorkshopOperationalStateError(
                    "Canonical GitHub subscription receipt exists without its principal state"
                )
        return 0

    cursor = await connection.execute(
        "INSERT OR IGNORE INTO principal_github_subscriptions ("
        "principal_id, baseline_repos_json, added_repos_json, removed_repos_json, "
        "pr_review_enabled, issue_triage_enabled, pr_review_source, issue_triage_source, "
        "github_token, allowed_triage_projects_json"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (namespace.principal_id, *seed.database_values()),
    )
    if cursor.rowcount == 0:
        async with connection.execute(
            "SELECT baseline_repos_json, added_repos_json, removed_repos_json, "
            "pr_review_enabled, issue_triage_enabled, pr_review_source, issue_triage_source, "
            "github_token, allowed_triage_projects_json "
            "FROM principal_github_subscriptions WHERE principal_id = ?",
            (namespace.principal_id,),
        ) as result_cursor:
            row = await result_cursor.fetchone()
        if row is None or tuple(row) != seed.database_values():
            raise WorkshopOperationalStateError(
                "Canonical GitHub subscription state conflicts with unmigrated legacy policy"
            )
    return cursor.rowcount


async def _ensure_canonical_github_subscription(
    connection: aiosqlite.Connection,
    namespace: WorkshopExecutionStateNamespace,
    policy: _OperatorGitHubPolicy,
) -> int:
    """Provision canonical state for a profile that has no legacy archive."""
    cursor = await connection.execute(
        "INSERT OR IGNORE INTO principal_github_subscriptions ("
        "principal_id, baseline_repos_json, added_repos_json, removed_repos_json, "
        "pr_review_enabled, issue_triage_enabled, pr_review_source, issue_triage_source, "
        "allowed_triage_projects_json"
        ") VALUES (?, ?, '[]', '[]', ?, ?, ?, ?, ?)",
        (namespace.principal_id, *policy.database_values()),
    )
    return cursor.rowcount


async def _sync_operator_github_policy(
    connection: aiosqlite.Connection,
    namespace: WorkshopExecutionStateNamespace,
    policy: _OperatorGitHubPolicy,
) -> None:
    cursor = await connection.execute(
        "UPDATE principal_github_subscriptions SET baseline_repos_json = ?, "
        "pr_review_enabled = CASE WHEN pr_review_source = 'user' "
        "THEN pr_review_enabled ELSE ? END, "
        "issue_triage_enabled = CASE WHEN issue_triage_source = 'user' "
        "THEN issue_triage_enabled ELSE ? END, "
        "pr_review_source = CASE WHEN pr_review_source = 'user' "
        "THEN pr_review_source ELSE ? END, "
        "issue_triage_source = CASE WHEN issue_triage_source = 'user' "
        "THEN issue_triage_source ELSE ? END, "
        "allowed_triage_projects_json = ?, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
        "WHERE principal_id = ?",
        (*policy.database_values(), namespace.principal_id),
    )
    if cursor.rowcount != 1:
        raise WorkshopOperationalStateError(
            "Protected GitHub subscription policy has no unique canonical principal owner"
        )


async def _migrate_runtime_key_github_token(
    connection: aiosqlite.Connection,
    namespace: WorkshopExecutionStateNamespace,
) -> None:
    """Carry the mutable token across schema v33 before sealing the archive.

    Operational-state receipts predate the canonical token column. Existing
    installations therefore need one final archived read on their first v33
    startup. The immutable runtime-key cutover receipt is the removal gate:
    once present, later legacy changes are never re-imported.
    """
    async with connection.execute(
        "SELECT 1 FROM workshop_runtime_key_cutovers WHERE runtime_profile_id = ?",
        (namespace.runtime_profile_id,),
    ) as cursor:
        if await cursor.fetchone() is not None:
            return
    legacy_token = await _legacy_setting(
        connection,
        f"github_token:{namespace.require_legacy_runtime_key()}",
    )
    if legacy_token is None:
        return
    await connection.execute(
        "UPDATE principal_github_subscriptions SET github_token = COALESCE(github_token, ?), "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE principal_id = ?",
        (legacy_token, namespace.principal_id),
    )


async def _verify_namespace(
    connection: aiosqlite.Connection,
    namespace: WorkshopExecutionStateNamespace,
) -> None:
    async with connection.execute(
        "SELECT COUNT(*) FROM principal_github_subscriptions WHERE principal_id = ?",
        (namespace.principal_id,),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None or int(row[0]) != 1:
        raise WorkshopOperationalStateError(
            "Protected GitHub subscription policy has no unique canonical principal owner"
        )


async def _verify_legacy_jobs_for_cutover(
    connection: aiosqlite.Connection,
    namespace: WorkshopExecutionStateNamespace,
) -> None:
    async with connection.execute(
        "SELECT j.id FROM jobs j LEFT JOIN workshop_job_owners o ON o.job_id = j.id "
        "WHERE j.chat_id = ? AND (o.job_id IS NULL OR o.principal_id != ? OR o.channel_id != ? "
        "OR o.agent_id != ? OR o.runtime_profile_id != ?) ORDER BY j.id LIMIT 5",
        (
            namespace.require_legacy_runtime_key(),
            namespace.principal_id,
            namespace.channel_id,
            namespace.agent_id,
            namespace.runtime_profile_id,
        ),
    ) as cursor:
        rows = await cursor.fetchall()
    if rows:
        job_ids = ", ".join(str(row[0]) for row in rows)
        raise WorkshopOperationalStateError(
            "Protected scheduled jobs have conflicting canonical ownership "
            f"(job ids: {job_ids}); restore workshop_job_owners from the same backup "
            "or remove and recreate the affected jobs"
        )
    async with connection.execute(
        "SELECT j.id FROM jobs j JOIN workshop_job_owners o ON o.job_id = j.id "
        "LEFT JOIN workshop_scheduled_jobs c ON c.id = j.id "
        "WHERE j.chat_id = ? AND (c.id IS NULL OR c.principal_id != o.principal_id "
        "OR c.channel_id != o.channel_id OR c.agent_id != o.agent_id "
        "OR c.runtime_profile_id != o.runtime_profile_id OR c.name != j.name "
        "OR c.job_type != j.job_type OR c.prompt != j.prompt "
        "OR c.schedule_type != j.schedule_type OR c.schedule_data != j.schedule_data "
        "OR c.active != j.active OR c.auto_remove != j.auto_remove "
        "OR c.notify_on_check != j.notify_on_check) ORDER BY j.id LIMIT 5",
        (namespace.require_legacy_runtime_key(),),
    ) as cursor:
        canonical_rows = await cursor.fetchall()
    if canonical_rows:
        job_ids = ", ".join(str(row[0]) for row in canonical_rows)
        raise WorkshopOperationalStateError(
            "Protected scheduled jobs conflict with canonical definitions "
            f"(job ids: {job_ids}); restore the database from the same pre-migration backup"
        )
