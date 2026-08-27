"""Canonical personal GitHub settings authority shared by client adapters."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from kai.workshop.domain import PrincipalId, RuntimeProfileId
from kai.workshop.execution_state import WorkshopExecutionStateRegistry
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileRegistry

_REPOSITORY_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")
_MAX_TOKEN_BYTES = 8_192


class WorkshopGitHubSettingsError(RuntimeError):
    """Base failure for personal GitHub settings."""


class WorkshopGitHubSettingsAccessDenied(WorkshopGitHubSettingsError):
    """The authenticated principal does not own this settings record."""


class WorkshopGitHubSettingsValidationError(WorkshopGitHubSettingsError):
    """A requested GitHub setting is invalid."""


class WorkshopGitHubSettingsConflict(WorkshopGitHubSettingsError):
    """The settings changed after the caller loaded them."""


class WorkshopGitHubSettingsStorageError(WorkshopGitHubSettingsError):
    """Canonical GitHub settings are unavailable or corrupt."""


@dataclass(frozen=True, slots=True)
class GitHubSettingsAuthority:
    principal_id: PrincipalId
    runtime_profile_id: RuntimeProfileId


@dataclass(frozen=True, slots=True)
class GitHubRepositorySetting:
    repository: str
    source: str
    automation_authorized: bool


@dataclass(frozen=True, slots=True)
class GitHubToggleSetting:
    enabled: bool
    source: str
    resettable: bool


@dataclass(frozen=True, slots=True)
class GitHubSettingsMutation:
    operation: str
    changed: bool


@dataclass(frozen=True, slots=True)
class GitHubSettingsSnapshot:
    github_login: str | None
    repositories: tuple[GitHubRepositorySetting, ...]
    repositories_resettable: bool
    pr_review: GitHubToggleSetting
    issue_triage: GitHubToggleSetting
    token_stored: bool
    revision: str
    mutation: GitHubSettingsMutation | None = None


@dataclass(frozen=True, slots=True)
class _GitHubRow:
    baseline: tuple[str, ...]
    added: tuple[str, ...]
    removed: tuple[str, ...]
    pr_review_enabled: bool
    issue_triage_enabled: bool
    pr_review_source: str
    issue_triage_source: str
    token_stored: bool

    @property
    def effective_repositories(self) -> tuple[str, ...]:
        return tuple(sorted((set(self.baseline) | set(self.added)) - set(self.removed)))

    @property
    def revision(self) -> str:
        encoded = json.dumps(
            {
                "baseline": self.baseline,
                "added": self.added,
                "removed": self.removed,
                "pr_review_enabled": self.pr_review_enabled,
                "issue_triage_enabled": self.issue_triage_enabled,
                "pr_review_source": self.pr_review_source,
                "issue_triage_source": self.issue_triage_source,
                "token_stored": self.token_stored,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return "ghs_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


def _restrict_database_path(path: Path) -> None:
    if not path.is_file():
        raise WorkshopGitHubSettingsStorageError("GitHub settings database is unavailable")


def _repository_list(value: object, *, field: str) -> tuple[str, ...]:
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError) as exc:
        raise WorkshopGitHubSettingsStorageError(f"Canonical GitHub {field} policy is corrupt") from exc
    if not isinstance(decoded, list) or any(not isinstance(item, str) for item in decoded):
        raise WorkshopGitHubSettingsStorageError(f"Canonical GitHub {field} policy is corrupt")
    normalized = tuple(sorted({item.strip().lower() for item in decoded if item.strip()}))
    if len(normalized) != len(decoded) or any(_REPOSITORY_PATTERN.fullmatch(item) is None for item in normalized):
        raise WorkshopGitHubSettingsStorageError(f"Canonical GitHub {field} policy is corrupt")
    return normalized


class WorkshopGitHubSettingsService:
    """Own principal GitHub settings without transport-shaped identifiers."""

    def __init__(
        self,
        connection: aiosqlite.Connection,
        execution_state: WorkshopExecutionStateRegistry,
        runtime_profiles: WorkshopRuntimeProfileRegistry,
    ) -> None:
        self._connection = connection
        self._execution_state = execution_state
        self._runtime_profiles = runtime_profiles
        self._locks: dict[PrincipalId, asyncio.Lock] = {}

    @classmethod
    async def open(
        cls,
        path: Path,
        execution_state: WorkshopExecutionStateRegistry,
        runtime_profiles: WorkshopRuntimeProfileRegistry,
    ) -> WorkshopGitHubSettingsService:
        _restrict_database_path(path)
        connection = await aiosqlite.connect(str(path))
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA busy_timeout=5000")
        await connection.execute("PRAGMA foreign_keys=ON")
        return cls(connection, execution_state, runtime_profiles)

    async def close(self) -> None:
        await self._connection.close()

    def authority_for_principal(
        self,
        principal_id: str | PrincipalId,
    ) -> GitHubSettingsAuthority:
        namespace = self._execution_state.maybe_for_principal_id(str(principal_id))
        if namespace is None:
            raise WorkshopGitHubSettingsAccessDenied("The principal does not own GitHub settings")
        return GitHubSettingsAuthority(namespace.principal_id, namespace.runtime_profile_id)

    def authority_for_principal_profile(
        self,
        principal_id: str | PrincipalId,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> GitHubSettingsAuthority:
        namespace = self._execution_state.maybe_for_runtime_profile_id(runtime_profile_id)
        try:
            canonical_principal = principal_id if isinstance(principal_id, PrincipalId) else PrincipalId(principal_id)
        except (TypeError, ValueError) as exc:
            raise WorkshopGitHubSettingsAccessDenied("The principal does not own GitHub settings") from exc
        if namespace is None or namespace.principal_id != canonical_principal:
            raise WorkshopGitHubSettingsAccessDenied("The principal does not own GitHub settings")
        return GitHubSettingsAuthority(namespace.principal_id, namespace.runtime_profile_id)

    def _lock(self, authority: GitHubSettingsAuthority) -> asyncio.Lock:
        self._validate_authority(authority)
        return self._locks.setdefault(authority.principal_id, asyncio.Lock())

    def _validate_authority(self, authority: GitHubSettingsAuthority) -> None:
        namespace = self._execution_state.maybe_for_runtime_profile_id(authority.runtime_profile_id)
        if namespace is None or namespace.principal_id != authority.principal_id:
            raise WorkshopGitHubSettingsAccessDenied("The principal does not own GitHub settings")

    async def inspect(self, authority: GitHubSettingsAuthority) -> GitHubSettingsSnapshot:
        async with self._lock(authority):
            return self._snapshot(authority, await self._read_row(authority.principal_id))

    async def set_repository_subscription(
        self,
        authority: GitHubSettingsAuthority,
        repository: str,
        *,
        subscribed: bool,
        expected_revision: str | None = None,
    ) -> GitHubSettingsSnapshot:
        normalized = repository.strip().lower()
        if _REPOSITORY_PATTERN.fullmatch(normalized) is None:
            raise WorkshopGitHubSettingsValidationError("Repository must use owner/name format")
        if not isinstance(subscribed, bool):
            raise WorkshopGitHubSettingsValidationError("Subscription state must be true or false")
        async with self._lock(authority):
            await self._connection.execute("BEGIN IMMEDIATE")
            try:
                current = await self._read_row(authority.principal_id)
                self._check_revision(current.revision, expected_revision)
                added = set(current.added)
                removed = set(current.removed)
                before = (set(added), set(removed))
                if subscribed:
                    removed.discard(normalized)
                    if normalized not in current.baseline:
                        added.add(normalized)
                else:
                    added.discard(normalized)
                    if normalized in current.baseline:
                        removed.add(normalized)
                changed = before != (added, removed)
                if changed:
                    await self._update_repositories(
                        authority.principal_id,
                        tuple(sorted(added)),
                        tuple(sorted(removed)),
                    )
                await self._connection.commit()
            except Exception:
                await self._connection.rollback()
                raise
            row = await self._read_row(authority.principal_id)
            operation = "subscribe_github_repository" if subscribed else "unsubscribe_github_repository"
            return self._snapshot(authority, row, GitHubSettingsMutation(operation, changed))

    async def set_toggle(
        self,
        authority: GitHubSettingsAuthority,
        field: str,
        enabled: bool | None,
        *,
        expected_revision: str | None = None,
    ) -> GitHubSettingsSnapshot:
        columns = {
            "pr_review": ("pr_review_enabled", "pr_review_source"),
            "issue_triage": ("issue_triage_enabled", "issue_triage_source"),
        }
        if field not in columns:
            raise WorkshopGitHubSettingsValidationError("Unsupported GitHub automation setting")
        if enabled is not None and not isinstance(enabled, bool):
            raise WorkshopGitHubSettingsValidationError("Automation setting must be true, false, or reset")
        profile = self._runtime_profiles.resolve(authority.runtime_profile_id)
        protected = profile.pr_review if field == "pr_review" else profile.issue_triage
        desired = bool(protected) if enabled is None else enabled
        source = ("operator" if protected is not None else "default") if enabled is None else "user"
        value_column, source_column = columns[field]
        async with self._lock(authority):
            await self._connection.execute("BEGIN IMMEDIATE")
            try:
                current = await self._read_row(authority.principal_id)
                self._check_revision(current.revision, expected_revision)
                current_value = current.pr_review_enabled if field == "pr_review" else current.issue_triage_enabled
                current_source = current.pr_review_source if field == "pr_review" else current.issue_triage_source
                changed = current_value != desired or current_source != source
                if changed:
                    await self._connection.execute(
                        f"UPDATE principal_github_subscriptions SET {value_column} = ?, "
                        f"{source_column} = ?, updated_at = "
                        "strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE principal_id = ?",
                        (int(desired), source, authority.principal_id),
                    )
                await self._connection.commit()
            except Exception:
                await self._connection.rollback()
                raise
            row = await self._read_row(authority.principal_id)
            operation = f"reset_github_{field}" if enabled is None else f"set_github_{field}"
            return self._snapshot(authority, row, GitHubSettingsMutation(operation, changed))

    async def set_token(
        self,
        authority: GitHubSettingsAuthority,
        token: str | None,
        *,
        expected_revision: str | None = None,
    ) -> GitHubSettingsSnapshot:
        normalized = None if token is None else token.strip()
        if token is not None and (not normalized or len(normalized.encode("utf-8")) > _MAX_TOKEN_BYTES):
            raise WorkshopGitHubSettingsValidationError("GitHub token is empty or too large")
        async with self._lock(authority):
            await self._connection.execute("BEGIN IMMEDIATE")
            try:
                current = await self._read_row(authority.principal_id)
                self._check_revision(current.revision, expected_revision)
                changed = current.token_stored != (normalized is not None)
                # Replacement is always a write even when the redacted status
                # is unchanged. Never compare, log, or return token material.
                if normalized is not None or current.token_stored:
                    await self._connection.execute(
                        "UPDATE principal_github_subscriptions SET github_token = ?, updated_at = "
                        "strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE principal_id = ?",
                        (normalized, authority.principal_id),
                    )
                    changed = True
                await self._connection.commit()
            except Exception:
                await self._connection.rollback()
                raise
            row = await self._read_row(authority.principal_id)
            operation = "remove_github_token" if normalized is None else "replace_github_token"
            return self._snapshot(authority, row, GitHubSettingsMutation(operation, changed))

    async def reset_repository_subscriptions(
        self,
        authority: GitHubSettingsAuthority,
        *,
        expected_revision: str | None = None,
    ) -> GitHubSettingsSnapshot:
        async with self._lock(authority):
            await self._connection.execute("BEGIN IMMEDIATE")
            try:
                current = await self._read_row(authority.principal_id)
                self._check_revision(current.revision, expected_revision)
                changed = bool(current.added or current.removed)
                if changed:
                    await self._update_repositories(authority.principal_id, (), ())
                await self._connection.commit()
            except Exception:
                await self._connection.rollback()
                raise
            row = await self._read_row(authority.principal_id)
            return self._snapshot(
                authority,
                row,
                GitHubSettingsMutation("reset_github_repositories", changed),
            )

    async def _read_row(self, principal_id: PrincipalId) -> _GitHubRow:
        async with self._connection.execute(
            "SELECT baseline_repos_json, added_repos_json, removed_repos_json, "
            "pr_review_enabled, issue_triage_enabled, pr_review_source, "
            "issue_triage_source, github_token IS NOT NULL "
            "FROM principal_github_subscriptions WHERE principal_id = ?",
            (principal_id,),
        ) as cursor:
            rows = tuple(await cursor.fetchall())
        if len(rows) != 1:
            raise WorkshopGitHubSettingsStorageError("Canonical GitHub settings have no unique owner")
        row = rows[0]
        pr_source = str(row[5])
        triage_source = str(row[6])
        if pr_source not in {"default", "operator", "user"} or triage_source not in {
            "default",
            "operator",
            "user",
        }:
            raise WorkshopGitHubSettingsStorageError("Canonical GitHub automation policy is corrupt")
        return _GitHubRow(
            baseline=_repository_list(row[0], field="baseline repository"),
            added=_repository_list(row[1], field="added repository"),
            removed=_repository_list(row[2], field="removed repository"),
            pr_review_enabled=bool(row[3]),
            issue_triage_enabled=bool(row[4]),
            pr_review_source=pr_source,
            issue_triage_source=triage_source,
            token_stored=bool(row[7]),
        )

    async def _update_repositories(
        self,
        principal_id: PrincipalId,
        added: tuple[str, ...],
        removed: tuple[str, ...],
    ) -> None:
        cursor = await self._connection.execute(
            "UPDATE principal_github_subscriptions SET added_repos_json = ?, removed_repos_json = ?, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE principal_id = ?",
            (
                json.dumps(added, separators=(",", ":")),
                json.dumps(removed, separators=(",", ":")),
                principal_id,
            ),
        )
        if cursor.rowcount != 1:
            raise WorkshopGitHubSettingsStorageError("Canonical GitHub settings have no unique owner")

    @staticmethod
    def _check_revision(current: str, expected: str | None) -> None:
        if expected is not None and expected != current:
            raise WorkshopGitHubSettingsConflict("GitHub settings changed in another session")

    def _snapshot(
        self,
        authority: GitHubSettingsAuthority,
        row: _GitHubRow,
        mutation: GitHubSettingsMutation | None = None,
    ) -> GitHubSettingsSnapshot:
        profile = self._runtime_profiles.resolve(authority.runtime_profile_id)
        baseline = set(row.baseline)
        added = set(row.added)
        repositories = tuple(
            GitHubRepositorySetting(
                repository=repo,
                source="user" if repo in added and repo not in baseline else "operator",
                automation_authorized=repo in baseline,
            )
            for repo in row.effective_repositories
        )
        return GitHubSettingsSnapshot(
            github_login=profile.github_login,
            repositories=repositories,
            repositories_resettable=bool(row.added or row.removed),
            pr_review=GitHubToggleSetting(
                row.pr_review_enabled,
                row.pr_review_source,
                row.pr_review_source == "user",
            ),
            issue_triage=GitHubToggleSetting(
                row.issue_triage_enabled,
                row.issue_triage_source,
                row.issue_triage_source == "user",
            ),
            token_stored=row.token_stored,
            revision=row.revision,
            mutation=mutation,
        )
