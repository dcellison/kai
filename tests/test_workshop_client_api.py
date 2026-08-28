"""HTTP contracts for the authenticated Workshop read API."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from kai.workshop.appearance_preferences import (
    WORKSHOP_APPEARANCE_THEMES,
    WorkshopAppearancePreferenceService,
)
from kai.workshop.artifacts import (
    StagedArtifact,
    WorkshopArtifactService,
    record_inbound_artifact,
)
from kai.workshop.bootstrap import (
    BootstrapHuman,
    BootstrapNotificationChannel,
    bootstrap_default_workshop,
)
from kai.workshop.client_api import (
    WorkshopEventStreamLimiter,
    register_workshop_command_routes,
    register_workshop_read_routes,
)
from kai.workshop.client_events import (
    ClientRunLifecycleEvent,
    ClientTimelineMessageEvent,
    read_client_channel_events,
)
from kai.workshop.client_preferences import (
    ClientVoiceCapability,
    WorkshopClientPreferenceService,
)
from kai.workshop.client_sessions import (
    WorkshopBearerSessionAuthenticator,
    WorkshopClientSessionManager,
)
from kai.workshop.conversation_commands import ConversationCommandDisposition
from kai.workshop.domain import (
    AgentId,
    ChannelId,
    ChannelMembershipId,
    MessageId,
    PrincipalId,
    RunId,
    WorkshopEventType,
)
from kai.workshop.execution_coordinator import CanonicalCancellationDisposition
from kai.workshop.execution_state import WorkshopExecutionStateRegistry
from kai.workshop.github_settings import (
    GitHubRepositorySetting,
    GitHubSettingsAuthority,
    GitHubSettingsMutation,
    GitHubSettingsSnapshot,
    GitHubToggleSetting,
    WorkshopGitHubSettingsAccessDenied,
    WorkshopGitHubSettingsConflict,
)
from kai.workshop.inbound import ClientInboundMessage, InboundMessage, record_inbound_message
from kai.workshop.memory_queries import (
    MemoryCreationSnapshot,
    MemoryEditSnapshot,
    MemoryMutationBatch,
    MemoryMutationResult,
    MemoryProjectOption,
    MemoryQueryAuthority,
    MemoryRecordDetail,
    MemoryRecordPage,
    MemoryRecordSummary,
    MemoryScopeSnapshot,
    MemorySearchHit,
    MemorySearchSnapshot,
    MemorySourceContext,
    MemoryStatsSnapshot,
    WorkshopMemoryAccessDenied,
    WorkshopMemoryConflict,
    WorkshopMemoryNotFound,
)
from kai.workshop.model_catalogue import (
    ModelCatalogueEntry,
    ModelCatalogueEntryStatus,
    ModelCatalogueProvenance,
    ModelCatalogueRefreshResult,
    ModelCatalogueRefreshState,
    ModelCatalogueRefreshStatus,
    ModelCatalogueSnapshot,
)
from kai.workshop.notification_preferences import WorkshopNotificationPreferenceService
from kai.workshop.run_lifecycle import (
    DurableRun,
    RunNotFoundError,
    RunStatus,
    WorkshopRunLifecycle,
)
from kai.workshop.run_previews import WorkshopRunPreviewRegistry
from kai.workshop.settings_workspaces import (
    BackendOption,
    EditableCapability,
    EffectiveValue,
    ModelOption,
    SettingsWorkspaceSnapshot,
    WorkshopSettingsWorkspaceAccessDenied,
    WorkshopSettingsWorkspaceConflict,
    WorkspaceConfigSnapshot,
    WorkspaceOption,
)
from kai.workshop.storage_namespaces import (
    WorkshopChannelHistoryRegistry,
    WorkshopPrincipalStorageRegistry,
)
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id, profile_registry

_NOW = datetime(2026, 8, 11, 14, 0, tzinfo=UTC)


@dataclass
class _Authenticator:
    principals_by_token: dict[str, PrincipalId]
    calls: list[str] = field(default_factory=list)

    async def authenticate(self, request: web.Request) -> PrincipalId | None:
        authorization = request.headers.get("Authorization", "")
        self.calls.append(authorization)
        scheme, separator, token = authorization.partition(" ")
        if separator and scheme == "Bearer":
            return self.principals_by_token.get(token)
        return None

    async def authenticate_token(self, token: str) -> PrincipalId | None:
        self.calls.append(f"Form {token}")
        return self.principals_by_token.get(token)


@dataclass
class _CommandSubmitter:
    messages: list[ClientInboundMessage] = field(default_factory=list)
    artifacts: list[StagedArtifact | None] = field(default_factory=list)
    runs: dict[RunId, DurableRun] = field(default_factory=dict)

    async def submit(
        self,
        message: ClientInboundMessage,
        *,
        artifact: StagedArtifact | None = None,
    ):
        self.messages.append(message)
        self.artifacts.append(artifact)
        message_id = MessageId.new()
        run_id = RunId.new()
        run = DurableRun(
            run_id=run_id,
            workshop_id=SimpleNamespace(),
            channel_id=message.channel_id,
            requested_by_principal_id=message.principal_id,
            agent_id=SimpleNamespace(),
            inbound_message_id=message_id,
            status=RunStatus.ACCEPTED,
            accepted_at=message.occurred_at,
            started_at=None,
            terminal_at=None,
            terminal_code=None,
            cancellation_requested_at=None,
            cancellation_code=None,
            result_message_id=None,
            last_event_position=1,
        )
        self.runs[run_id] = run
        return SimpleNamespace(
            acceptance=SimpleNamespace(
                command=SimpleNamespace(
                    message=SimpleNamespace(event=SimpleNamespace(envelope=SimpleNamespace(aggregate_id=message_id))),
                    disposition=ConversationCommandDisposition.NEWLY_ACCEPTED,
                ),
                run=run,
            ),
            run=run,
        )

    async def state(self, run_id: RunId) -> DurableRun:
        try:
            return self.runs[run_id]
        except KeyError as exc:
            raise RunNotFoundError("missing") from exc

    async def cancel(self, run_id: RunId) -> CanonicalCancellationDisposition:
        run = await self.state(run_id)
        self.runs[run_id] = replace(
            run,
            status=RunStatus.CANCELLED,
            terminal_at=_NOW,
            terminal_code="requested_by_human",
            cancellation_requested_at=_NOW,
            cancellation_code="requested_by_human",
        )
        return CanonicalCancellationDisposition.REQUESTED


class _AllowChannelRead:
    async def can_read_channel(self, principal_id: PrincipalId, channel_id: ChannelId) -> bool:
        return True


@dataclass
class _SettingsWorkspaces:
    principal_id: PrincipalId
    channel_id: ChannelId
    switched: list[str] = field(default_factory=list)
    workspace_config_changes: list[tuple[str, str]] = field(default_factory=list)
    runtime_changes: list[tuple[str, object]] = field(default_factory=list)
    catalogue_calls: list[tuple[str, str | None]] = field(default_factory=list)

    def authority_for_principal_channel(self, principal_id, channel_id):
        if principal_id != self.principal_id or channel_id != self.channel_id:
            raise WorkshopSettingsWorkspaceAccessDenied("denied")
        return SimpleNamespace(principal_id=principal_id, channel_id=channel_id)

    @staticmethod
    def _check_revision(expected, current: str) -> None:
        if expected != current:
            raise WorkshopSettingsWorkspaceConflict("stale settings")

    async def inspect(self, _authority):
        return self._snapshot()

    async def inspect_model_catalogue(self, _authority, option_id=None):
        self.catalogue_calls.append(("inspect", option_id))
        return self._catalogue_snapshot(option_id)

    async def refresh_model_catalogue(self, _authority, option_id=None):
        self.catalogue_calls.append(("refresh", option_id))
        return self._refresh_result(option_id)

    async def refresh_all_model_catalogues_as_operator(self):
        self.catalogue_calls.append(("refresh_all", None))
        return (self._refresh_result("codex:openai"),)

    async def upsert_operator_model(
        self,
        _authority,
        option_id,
        *,
        model_id,
        display_label,
    ):
        self.catalogue_calls.append((f"upsert:{model_id}:{display_label}", option_id))
        return self._catalogue_snapshot(option_id)

    async def deactivate_operator_model(self, _authority, option_id, *, model_id):
        self.catalogue_calls.append((f"deactivate:{model_id}", option_id))
        return self._catalogue_snapshot(option_id)

    async def switch_workspace(self, _authority, path: str, *, expected_revision=None):
        self._check_revision(expected_revision, "sws_current")
        self.switched.append(path)
        return self._snapshot(workspace=path)

    async def set_model(
        self,
        _authority,
        _model: str,
        *,
        expected_revision=None,
        clear_workspace_override=True,
    ):
        self._check_revision(expected_revision, "sws_current")
        self.runtime_changes.append(("model", _model))
        return self._snapshot()

    async def set_backend(self, _authority, backend: str, *, expected_revision=None):
        self._check_revision(expected_revision, "sws_current")
        self.runtime_changes.append(("backend", backend))
        return self._snapshot()

    async def set_timeout(self, _authority, _timeout: int, *, expected_revision=None):
        self._check_revision(expected_revision, "sws_current")
        self.runtime_changes.append(("timeout", _timeout))
        return self._snapshot()

    async def reset_settings(self, _authority, _field=None, *, expected_revision=None):
        self._check_revision(expected_revision, "sws_current")
        self.runtime_changes.append(("reset", _field or "all"))
        return self._snapshot()

    async def workspace_config(self, _authority):
        return self._workspace_config_snapshot()

    async def set_workspace_config(
        self,
        _authority,
        *,
        field: str,
        value: str,
        workspace_path: str | None = None,
        expected_revision: str | None = None,
    ):
        self._check_revision(expected_revision, "sws_workspace")
        self.workspace_config_changes.append((field, value))
        return self._workspace_config_snapshot()

    async def reset_workspace_config(
        self,
        _authority,
        *,
        field: str | None = None,
        workspace_path: str | None = None,
        expected_revision: str | None = None,
    ):
        self._check_revision(expected_revision, "sws_workspace")
        self.workspace_config_changes.append(("reset", field or "all"))
        return self._workspace_config_snapshot()

    async def set_self_service_workspace_config(self, *args, **kwargs):
        return await self.set_workspace_config(*args, **kwargs)

    async def reset_self_service_workspace_config(self, *args, **kwargs):
        return await self.reset_workspace_config(*args, **kwargs)

    def _snapshot(self, *, workspace: str = "/srv/kai"):
        return SettingsWorkspaceSnapshot(
            principal_id=self.principal_id,
            channel_id=self.channel_id,
            runtime_profile_id=profile_id(101),
            backend_option_id="codex:openai",
            backend="codex",
            provider="openai",
            backend_options=(
                BackendOption("codex:openai", "codex", "openai", True),
                BackendOption("claude:anthropic", "claude", "anthropic", False),
            ),
            model=EffectiveValue("gpt-5.6-sol", "runtime policy", "gpt-5.6-sol"),
            timeout_seconds=EffectiveValue(120, "runtime policy", 120),
            workspace=workspace,
            model_options=(ModelOption("gpt-5.6-sol", "GPT-5.6 Sol"),),
            workspaces=(WorkspaceOption(workspace, "kai", True, False),),
            revision="sws_current",
            capabilities=(EditableCapability("model", "runtime", "model_id", True),),
        )

    def _catalogue_snapshot(self, option_id: str | None = None) -> ModelCatalogueSnapshot:
        refresh = ModelCatalogueRefreshState(
            ModelCatalogueRefreshStatus.SUCCEEDED,
            2,
            _NOW,
            _NOW,
            _NOW + timedelta(hours=1),
            None,
            None,
        )
        return ModelCatalogueSnapshot(
            principal_id=self.principal_id,
            runtime_profile_id=profile_id(101),
            option_id=option_id or "codex:openai",
            cache_key="catalogue-cache-key",
            entries=(
                ModelCatalogueEntry(
                    "gpt-5.6-sol",
                    "GPT-5.6 Sol",
                    ModelCatalogueEntryStatus.AVAILABLE,
                    True,
                    True,
                    (
                        ModelCatalogueProvenance(
                            "discovered:fixture",
                            ModelCatalogueEntryStatus.AVAILABLE,
                            "GPT-5.6 Sol",
                            {},
                        ),
                    ),
                ),
            ),
            refresh=refresh,
            stale=False,
            last_known_good=False,
        )

    @staticmethod
    def _refresh_result(option_id: str | None = None) -> ModelCatalogueRefreshResult:
        return ModelCatalogueRefreshResult(
            profile_id(101),
            option_id or "codex:openai",
            "catalogue-cache-key",
            ModelCatalogueRefreshStatus.SUCCEEDED,
            2,
            1,
            False,
        )

    @staticmethod
    def _workspace_config_snapshot() -> WorkspaceConfigSnapshot:
        return WorkspaceConfigSnapshot(
            workspace="/srv/kai",
            model=EffectiveValue("gpt-5.6-sol", "runtime policy", "gpt-5.6-sol"),
            timeout_seconds=EffectiveValue(120, "runtime policy", 120),
            environment_keys=("SAFE_KEY",),
            prompt=None,
            has_prompt=False,
            prompt_source=None,
            override_fields=(),
            revision="sws_workspace",
            capabilities=(EditableCapability("prompt", "workspace", "text", True),),
        )


@dataclass
class _MemoryQueries:
    principal_id: PrincipalId
    mutations: list[tuple[str, tuple[str, ...], str | None, str | None]] = field(default_factory=list)
    content_mutations: list[tuple[str, object]] = field(default_factory=list)

    def authority_for_principal(self, principal_id):
        if principal_id != self.principal_id:
            raise WorkshopMemoryAccessDenied("denied")
        return MemoryQueryAuthority(principal_id, None)

    async def stats(self, _authority):
        return MemoryStatsSnapshot(
            total=1,
            facts=1,
            episodes=0,
            by_source={"extracted": 1},
            by_type={"fact": 1},
            by_scope={"global": 1},
            allowed_projects=(MemoryProjectOption("kai", "Kai"),),
        )

    async def list_records(self, _authority, *, filters, limit, cursor, order):
        assert filters.source == "extracted"
        assert limit == 1
        assert cursor is None
        assert order == "oldest"
        return MemoryRecordPage(
            records=(
                MemoryRecordSummary(
                    memory_id="memory-1",
                    kind="fact",
                    source="extracted",
                    memory_type="fact",
                    preview="Remember this",
                    tags=("preference",),
                    speaker="user",
                    confidence=1.0,
                    created_at="2026-08-24T10:00:00Z",
                    updated_at="2026-08-24T10:00:00Z",
                    revision="mr1_test-revision",
                    scope=MemoryScopeSnapshot(
                        scope="global",
                        project_id=None,
                        scope_confidence=1.0,
                        scope_source="operator",
                        legacy_defaulted=False,
                        invalid_defaulted=False,
                        retrievable=True,
                        exclusion_reason=None,
                    ),
                ),
            ),
            next_cursor="next-page",
        )

    async def detail(self, _authority, memory_id):
        if memory_id != "memory-1":
            raise WorkshopMemoryNotFound("missing")
        record = (
            await self.list_records(
                _authority,
                filters=SimpleNamespace(source="extracted"),
                limit=1,
                cursor=None,
                order="oldest",
            )
        ).records[0]
        return MemoryRecordDetail(
            record=record,
            content="Remember this",
            compact_recall='{"record_type":"memory"}',
            confirmation_quote=None,
            prompt_version="v1",
            episode=None,
        )

    async def create_fact(self, authority, **kwargs):
        self.content_mutations.append(("create", kwargs))
        return MemoryCreationSnapshot(await self.detail(authority, "memory-1"), True)

    async def edit(self, authority, memory_id, **kwargs):
        self.content_mutations.append(("edit", {"memory_id": memory_id, **kwargs}))
        return MemoryEditSnapshot(
            await self.detail(authority, memory_id),
            ("content", "tags"),
            False,
        )

    async def search(self, _authority, query, *, filters, limit):
        assert query == "remember"
        assert limit == 10
        record = (
            await self.list_records(
                _authority,
                filters=SimpleNamespace(source="extracted"),
                limit=1,
                cursor=None,
                order="oldest",
            )
        ).records[0]
        return MemorySearchSnapshot(
            hits=(MemorySearchHit(record, 0.9, 0.9, '{"record_type":"memory"}'),),
            active_project_id=None,
            reason="ok",
        )

    async def source_context(self, _authority, memory_id):
        if memory_id != "memory-1":
            raise WorkshopMemoryNotFound("missing")
        return MemorySourceContext("unavailable", "legacy_source", None, None, None)

    async def move_scope(self, _authority, memory_ids, *, scope, project_id=None):
        self.mutations.append(("move_scope", tuple(memory_ids), scope, project_id))
        return MemoryMutationBatch(
            "move_scope",
            tuple(MemoryMutationResult(memory_id, "succeeded", None, None) for memory_id in memory_ids),
        )

    async def delete(self, _authority, memory_ids):
        self.mutations.append(("delete", tuple(memory_ids), None, None))
        return MemoryMutationBatch(
            "delete",
            tuple(MemoryMutationResult(memory_id, "succeeded", None, None) for memory_id in memory_ids),
        )


@dataclass
class _GitHubSettings:
    principal_id: PrincipalId
    calls: list[tuple[str, object]] = field(default_factory=list)

    def authority_for_principal(self, principal_id):
        if principal_id != self.principal_id:
            raise WorkshopGitHubSettingsAccessDenied("denied")
        return GitHubSettingsAuthority(principal_id, profile_id(101))

    @staticmethod
    def _snapshot(mutation=None):
        return GitHubSettingsSnapshot(
            github_login="alice",
            repositories=(GitHubRepositorySetting("owner/repo", "operator", True),),
            repositories_resettable=True,
            pr_review=GitHubToggleSetting(True, "operator", False),
            issue_triage=GitHubToggleSetting(False, "user", True),
            token_stored=True,
            revision="ghs_current",
            mutation=mutation,
        )

    async def inspect(self, authority):
        self.calls.append(("inspect", authority))
        return self._snapshot()

    async def set_repository_subscription(
        self,
        authority,
        repository,
        *,
        subscribed,
        expected_revision,
    ):
        self.calls.append(("repository", (authority, repository, subscribed, expected_revision)))
        return self._snapshot(GitHubSettingsMutation("subscribe_github_repository", True))

    async def set_toggle(self, authority, setting, enabled, *, expected_revision):
        if expected_revision == "ghs_stale":
            raise WorkshopGitHubSettingsConflict("GitHub settings changed in another session")
        self.calls.append(("toggle", (authority, setting, enabled, expected_revision)))
        return self._snapshot(GitHubSettingsMutation("set_github_issue_triage", True))

    async def reset_repository_subscriptions(self, authority, *, expected_revision):
        self.calls.append(("repository_reset", (authority, expected_revision)))
        return self._snapshot(GitHubSettingsMutation("reset_github_repositories", True))

    async def set_token(self, authority, token, *, expected_revision):
        assert token == "new-secret"
        self.calls.append(("token", (authority, "redacted", expected_revision)))
        return self._snapshot(GitHubSettingsMutation("replace_github_token", True))


async def _identity_for(store: WorkshopEventStore, subject: str) -> tuple[PrincipalId, ChannelId]:
    async with store.connection.execute(
        "SELECT e.principal_id, b.channel_id FROM external_identities e "
        "JOIN channel_bindings b ON b.transport = e.provider "
        "AND b.external_channel_id = e.external_subject "
        "WHERE e.provider = 'telegram' AND e.external_subject = ?",
        (subject,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return PrincipalId(str(row[0])), ChannelId(str(row[1]))


async def _open_store(path: Path) -> tuple[WorkshopEventStore, PrincipalId, ChannelId, PrincipalId, ChannelId]:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                "Alice",
                "admin",
                "telegram",
                "101",
                "101",
                profile_id(101),
            ),
            BootstrapHuman(
                "Bob",
                "member",
                "telegram",
                "202",
                "202",
                profile_id(202),
            ),
        ),
        notification_channels=(BootstrapNotificationChannel("telegram", "-100123", ("101", "202")),),
    )
    alice_id, alice_channel = await _identity_for(store, "101")
    bob_id, bob_channel = await _identity_for(store, "202")
    return store, alice_id, alice_channel, bob_id, bob_channel


async def _record_messages(store: WorkshopEventStore, count: int, *, start: int = 1) -> None:
    for ordinal in range(start, start + count):
        await record_inbound_message(
            store,
            InboundMessage(
                "telegram",
                str(9000 + ordinal),
                str(40 + ordinal),
                "101",
                "101",
                f"Message {ordinal}",
                _NOW + timedelta(seconds=ordinal),
            ),
        )


async def _open_client(
    store: WorkshopEventStore,
    authenticator: _Authenticator,
    *,
    event_poll_interval: float = 0.01,
    event_heartbeat_interval: float = 0.05,
    event_authentication_recheck_interval: float = 0.01,
    event_stream_limiter: WorkshopEventStreamLimiter | None = None,
    run_previews: WorkshopRunPreviewRegistry | None = None,
    artifact_service: WorkshopArtifactService | None = None,
    settings_workspaces=None,
    memory_queries=None,
    github_settings=None,
    notification_preferences=None,
    client_preferences=None,
    appearance_preferences=None,
) -> TestClient:
    app = web.Application()
    register_workshop_read_routes(
        app,
        store=store,
        authenticator=authenticator,
        request_lock=asyncio.Lock(),
        event_poll_interval=event_poll_interval,
        event_heartbeat_interval=event_heartbeat_interval,
        event_authentication_recheck_interval=event_authentication_recheck_interval,
        event_stream_limiter=event_stream_limiter,
        run_previews=run_previews,
        artifact_service=artifact_service,
        settings_workspaces=settings_workspaces,
        memory_queries=memory_queries,
        github_settings=github_settings,
        notification_preferences=notification_preferences,
        client_preferences=client_preferences,
        appearance_preferences=appearance_preferences,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


async def _open_command_client(
    store: WorkshopEventStore,
    authenticator: _Authenticator,
    submitter: _CommandSubmitter,
    artifact_service: WorkshopArtifactService | None = None,
) -> TestClient:
    app = web.Application(client_max_size=21 * 1024 * 1024)
    register_workshop_command_routes(
        app,
        store=store,
        authenticator=authenticator,
        submitter=submitter,
        request_lock=asyncio.Lock(),
        artifact_service=artifact_service,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


async def _artifact_service(
    store: WorkshopEventStore,
    data_dir: Path,
) -> WorkshopArtifactService:
    profiles = profile_registry(101, 202)
    storage = await WorkshopPrincipalStorageRegistry.from_store(store, profiles)
    return WorkshopArtifactService(
        store,
        data_dir=data_dir,
        principal_storage=storage,
        runtime_profiles=profiles,
    )


async def _read_sse_event(response) -> dict[str, object]:
    event_name: str | None = None
    event_id: str | None = None
    data_lines: list[str] = []
    while True:
        raw_line = await asyncio.wait_for(response.content.readline(), timeout=1.0)
        assert raw_line, "Event stream ended before the next event"
        line = raw_line.decode().rstrip("\r\n")
        if not line:
            if event_name is None:
                continue
            return {
                "event": event_name,
                "id": event_id,
                "data": json.loads("\n".join(data_lines)),
            }
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "id":
            event_id = value
        elif field == "data":
            data_lines.append(value)


@pytest.mark.asyncio
async def test_memory_api_uses_bearer_principal_and_stable_read_schema(
    tmp_path: Path,
) -> None:
    store, alice_id, _, _, _ = await _open_store(tmp_path / "kai.db")
    client = await _open_client(
        store,
        _Authenticator({"alice-token": alice_id}),
        memory_queries=_MemoryQueries(alice_id),
    )
    try:
        unauthorized = await client.get("/v1/memory/stats")
        assert unauthorized.status == 401

        stats = await client.get(
            "/v1/memory/stats",
            headers={"Authorization": "Bearer alice-token"},
        )
        assert stats.status == 200
        assert await stats.json() == {
            "version": 1,
            "stats": {
                "total": 1,
                "facts": 1,
                "episodes": 0,
                "by_source": {"extracted": 1},
                "by_type": {"fact": 1},
                "by_scope": {"global": 1},
                "allowed_projects": [
                    {"project_id": "kai", "display_name": "Kai"},
                ],
            },
        }

        page = await client.get(
            "/v1/memory/records?source=extracted&limit=1&order=oldest",
            headers={"Authorization": "Bearer alice-token"},
        )
        assert page.status == 200
        payload = await page.json()
        assert payload["version"] == 1
        assert payload["next_cursor"] == "next-page"
        assert payload["records"][0]["memory_id"] == "memory-1"
        assert payload["records"][0]["scope"]["retrievable"] is True
        assert "principal_id" not in payload

        search = await client.get(
            "/v1/memory/search?q=remember",
            headers={"Authorization": "Bearer alice-token"},
        )
        assert search.status == 200
        assert (await search.json())["hits"][0]["raw_score"] == 0.9

        detail = await client.get(
            "/v1/memory/records/memory-1",
            headers={"Authorization": "Bearer alice-token"},
        )
        assert detail.status == 200
        assert (await detail.json())["record"]["compact_recall"] == '{"record_type":"memory"}'

        source = await client.get(
            "/v1/memory/records/memory-1/source",
            headers={"Authorization": "Bearer alice-token"},
        )
        assert source.status == 200
        assert (await source.json())["source_context"]["reason"] == "legacy_source"

        foreign = await client.get(
            "/v1/memory/records/foreign-memory-id",
            headers={"Authorization": "Bearer alice-token"},
        )
        assert foreign.status == 404
        assert await foreign.json() == {"error": {"code": "memory_not_found", "message": "Memory not found"}}
    finally:
        await client.close()
        await store.close()


@pytest.mark.asyncio
async def test_memory_api_creates_and_edits_only_typed_content(
    tmp_path: Path,
) -> None:
    store, alice_id, _, _, _ = await _open_store(tmp_path / "kai.db")
    queries = _MemoryQueries(alice_id)
    client = await _open_client(store, _Authenticator({"alice-token": alice_id}), memory_queries=queries)
    headers = {"Authorization": "Bearer alice-token", "Content-Type": "application/json"}
    try:
        created = await client.post(
            "/v1/memory/records",
            headers=headers,
            json={
                "kind": "fact",
                "content": "An explicit fact",
                "tags": ["explicit"],
                "scope": "project",
                "project_id": "kai",
                "request_id": "create-1",
            },
        )
        assert created.status == 201
        created_payload = await created.json()
        assert created_payload["created"] is True
        assert created_payload["record"]["revision"] == "mr1_test-revision"

        edited = await client.patch(
            "/v1/memory/records/memory-1",
            headers=headers,
            json={
                "kind": "fact",
                "revision": "mr1_test-revision",
                "request_id": "edit-1",
                "content": "Corrected fact",
                "tags": ["corrected"],
            },
        )
        assert edited.status == 200
        assert (await edited.json())["changed_fields"] == ["content", "tags"]
        assert [operation for operation, _ in queries.content_mutations] == ["create", "edit"]

        arbitrary = await client.patch(
            "/v1/memory/records/memory-1",
            headers=headers,
            json={
                "kind": "fact",
                "revision": "mr1_test-revision",
                "request_id": "edit-2",
                "content": "Attack",
                "tags": [],
                "metadata": {"user_id": "someone-else"},
            },
        )
        assert arbitrary.status == 400
        assert len(queries.content_mutations) == 2
    finally:
        await client.close()
        await store.close()


@pytest.mark.asyncio
async def test_memory_api_returns_current_revision_on_edit_conflict(
    tmp_path: Path,
) -> None:
    store, alice_id, _, _, _ = await _open_store(tmp_path / "kai.db")
    queries = _MemoryQueries(alice_id)

    async def conflict(*_args, **_kwargs):
        raise WorkshopMemoryConflict("mr1_current")

    queries.edit = conflict  # type: ignore[method-assign]
    client = await _open_client(store, _Authenticator({"alice-token": alice_id}), memory_queries=queries)
    try:
        response = await client.patch(
            "/v1/memory/records/memory-1",
            headers={"Authorization": "Bearer alice-token", "Content-Type": "application/json"},
            json={
                "kind": "fact",
                "revision": "mr1_stale",
                "request_id": "edit-1",
                "content": "Correction",
                "tags": [],
            },
        )
        assert response.status == 409
        assert await response.json() == {
            "error": {
                "code": "memory_revision_conflict",
                "message": "Memory changed since it was opened",
                "current_revision": "mr1_current",
            }
        }
    finally:
        await client.close()
        await store.close()


@pytest.mark.asyncio
async def test_memory_api_rejects_owner_parameters_and_duplicate_values(
    tmp_path: Path,
) -> None:
    store, alice_id, _, _, _ = await _open_store(tmp_path / "kai.db")
    client = await _open_client(
        store,
        _Authenticator({"alice-token": alice_id}),
        memory_queries=_MemoryQueries(alice_id),
    )
    headers = {"Authorization": "Bearer alice-token"}
    try:
        for path in (
            "/v1/memory/records?principal_id=prn_" + "9" * 32,
            "/v1/memory/records?source=extracted&source=episode",
            "/v1/memory/search?q=hello&q=other",
        ):
            response = await client.get(path, headers=headers)
            assert response.status == 400
    finally:
        await client.close()
        await store.close()


@pytest.mark.asyncio
async def test_memory_api_exposes_typed_individual_and_bulk_mutations(
    tmp_path: Path,
) -> None:
    store, alice_id, _, _, _ = await _open_store(tmp_path / "kai.db")
    service = _MemoryQueries(alice_id)
    client = await _open_client(
        store,
        _Authenticator({"alice-token": alice_id}),
        memory_queries=service,
    )
    headers = {"Authorization": "Bearer alice-token"}
    try:
        moved = await client.patch(
            "/v1/memory/records/memory-1/scope",
            headers=headers,
            json={"scope": "project", "project_id": "kai"},
        )
        deleted = await client.delete(
            "/v1/memory/records/memory-1",
            headers=headers,
        )
        bulk_moved = await client.post(
            "/v1/memory/actions/scope",
            headers=headers,
            json={"memory_ids": ["memory-1", "memory-2"], "scope": "global"},
        )
        bulk_deleted = await client.post(
            "/v1/memory/actions/delete",
            headers=headers,
            json={"memory_ids": ["memory-1", "memory-2"]},
        )

        assert [response.status for response in (moved, deleted, bulk_moved, bulk_deleted)] == [200] * 4
        assert (await bulk_deleted.json())["results"] == [
            {"memory_id": "memory-1", "outcome": "succeeded", "prior_scope": None, "new_scope": None},
            {"memory_id": "memory-2", "outcome": "succeeded", "prior_scope": None, "new_scope": None},
        ]
        assert service.mutations == [
            ("move_scope", ("memory-1",), "project", "kai"),
            ("delete", ("memory-1",), None, None),
            ("move_scope", ("memory-1", "memory-2"), "global", None),
            ("delete", ("memory-1", "memory-2"), None, None),
        ]

        invalid = await client.post(
            "/v1/memory/actions/delete",
            headers=headers,
            json={"memory_ids": ["memory-1"], "principal_id": str(alice_id)},
        )
        body_on_delete = await client.delete(
            "/v1/memory/records/memory-1",
            headers=headers,
            json={"confirm": True},
        )
        assert invalid.status == 400
        assert body_on_delete.status == 400
    finally:
        await client.close()
        await store.close()


@pytest.mark.asyncio
async def test_concurrent_memory_reads_share_the_client_request_transaction_boundary(
    tmp_path: Path,
) -> None:
    store, alice_id, _, _, _ = await _open_store(tmp_path / "kai.db")
    sessions = WorkshopClientSessionManager(store)
    device = await sessions.register_device(alice_id, "Alice browser")
    issued = await sessions.issue_session(alice_id, device.device_id)
    client = await _open_client(
        store,
        WorkshopBearerSessionAuthenticator(sessions),
        memory_queries=_MemoryQueries(alice_id),
    )
    headers = {"Authorization": f"Bearer {issued.token}"}
    try:
        stats, records = await asyncio.gather(
            client.get("/v1/memory/stats", headers=headers),
            client.get(
                "/v1/memory/records?source=extracted&limit=1&order=oldest",
                headers=headers,
            ),
        )

        assert stats.status == 200
        assert records.status == 200
    finally:
        await client.close()
        await store.close()


class TestWorkshopNavigationHTTPContract:
    async def test_lists_only_explicit_memberships_and_marks_outbound_channels_read_only(
        self,
        tmp_path: Path,
    ):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        client = await _open_client(store, _Authenticator({"alice": alice_id}))
        try:
            response = await client.get(
                "/v1/client/navigation",
                headers={"Authorization": "Bearer alice"},
            )

            assert response.status == 200
            payload = await response.json()
            assert payload["principal"] == {
                "principal_id": alice_id,
                "display_name": "Alice",
            }
            assert len(payload["workshops"]) == 1
            workshop = payload["workshops"][0]
            assert workshop["name"] == "Kai Workshop"
            assert workshop["role"] == "admin"
            assert [channel["kind"] for channel in workshop["channels"]] == [
                "direct",
                "notification",
            ]
            direct, notification = workshop["channels"]
            assert direct == {
                "channel_id": alice_channel,
                "name": "Direct",
                "kind": "direct",
                "role": "owner",
                "agents": [
                    {
                        "agent_id": direct["agents"][0]["agent_id"],
                        "name": "Kai",
                    }
                ],
                "participants": [
                    {
                        "principal_id": direct["participants"][0]["principal_id"],
                        "kind": "agent",
                        "display_name": "Kai",
                    }
                ],
                "can_submit_commands": True,
            }
            assert notification["name"] == "Notifications"
            assert notification["role"] == "participant"
            assert notification["can_submit_commands"] is False

            async with store.connection.execute(
                "SELECT c.id FROM channels c JOIN channel_bindings cb ON cb.channel_id = c.id "
                "WHERE cb.external_channel_id = '202'"
            ) as cursor:
                bob_channel = await cursor.fetchone()
            assert bob_channel is not None
            assert str(bob_channel[0]) not in {channel["channel_id"] for channel in workshop["channels"]}
        finally:
            await client.close()
            await store.close()

    async def test_authentication_precedes_navigation_validation(self, tmp_path: Path):
        store, alice_id, _, _, _ = await _open_store(tmp_path / "kai.db")
        client = await _open_client(store, _Authenticator({"alice": alice_id}))
        try:
            unauthenticated = await client.get("/v1/client/navigation?unsupported=1")
            malformed = await client.get(
                "/v1/client/navigation?unsupported=1",
                headers={"Authorization": "Bearer alice"},
            )

            assert unauthenticated.status == 401
            assert malformed.status == 400
            assert (await malformed.json())["error"]["code"] == "invalid_request"
        finally:
            await client.close()
            await store.close()

    async def test_direct_channel_participants_include_the_other_human(self, tmp_path: Path):
        store, alice_id, _, bob_id, _ = await _open_store(tmp_path / "kai.db")
        async with store.connection.execute("SELECT id FROM workshops LIMIT 1") as cursor:
            workshop_id = str((await cursor.fetchone())[0])
        human_direct_channel = ChannelId.new()
        await store.connection.execute(
            "INSERT INTO channels (id, workshop_id, kind, name, created_at) VALUES (?, ?, ?, ?, ?)",
            (human_direct_channel, workshop_id, "direct", "Direct", _NOW.isoformat()),
        )
        await store.connection.executemany(
            "INSERT INTO channel_memberships (id, channel_id, principal_id, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                (ChannelMembershipId.new(), human_direct_channel, alice_id, "owner", _NOW.isoformat()),
                (ChannelMembershipId.new(), human_direct_channel, bob_id, "owner", _NOW.isoformat()),
            ),
        )
        await store.connection.commit()
        client = await _open_client(store, _Authenticator({"alice": alice_id}))
        try:
            response = await client.get(
                "/v1/client/navigation",
                headers={"Authorization": "Bearer alice"},
            )

            assert response.status == 200
            payload = await response.json()
            direct = next(
                channel
                for channel in payload["workshops"][0]["channels"]
                if channel["channel_id"] == human_direct_channel
            )
            assert direct["participants"] == [
                {
                    "principal_id": bob_id,
                    "kind": "human",
                    "display_name": "Bob",
                }
            ]
            assert direct["agents"] == []
            assert direct["can_submit_commands"] is False
        finally:
            await client.close()
            await store.close()


class TestWorkshopChannelLifecycleHTTPContract:
    async def test_creates_one_private_canonical_group_without_transport_binding(
        self,
        tmp_path: Path,
    ) -> None:
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        async with store.connection.execute(
            "SELECT ca.agent_id FROM channel_agents ca WHERE ca.channel_id = ?",
            (alice_channel,),
        ) as cursor:
            agent_row = await cursor.fetchone()
        assert agent_row is not None
        agent_id = AgentId(str(agent_row[0]))
        async with store.connection.execute("SELECT * FROM event_log ORDER BY position") as cursor:
            bootstrap_events = [tuple(row) for row in await cursor.fetchall()]
        async with store.connection.execute("SELECT * FROM channels ORDER BY id") as cursor:
            bootstrap_channels = {str(row[0]): tuple(row) for row in await cursor.fetchall()}
        async with store.connection.execute("SELECT * FROM channel_memberships ORDER BY id") as cursor:
            bootstrap_memberships = {str(row[0]): tuple(row) for row in await cursor.fetchall()}
        async with store.connection.execute(
            "SELECT * FROM channels WHERE id = ?",
            (alice_channel,),
        ) as cursor:
            origin_before = tuple(await cursor.fetchone())

        client = await _open_client(
            store,
            _Authenticator({"alice": alice_id}),
        )
        try:
            response = await client.post(
                "/v1/channels",
                headers={"Authorization": "Bearer alice"},
                json={
                    "name": "  Release planning  ",
                    "agent_ids": [agent_id],
                    "origin_channel_id": alice_channel,
                },
            )

            assert response.status == 201
            payload = await response.json()
            channel = payload["channel"]
            channel_id = ChannelId(channel["channel_id"])
            assert channel == {
                "channel_id": channel_id,
                "workshop_id": payload["channel"]["workshop_id"],
                "name": "Release planning",
                "kind": "group",
                "visibility": "private",
                "origin_channel_id": alice_channel,
                "role": "owner",
                "agent_ids": [agent_id],
            }

            async with store.connection.execute(
                "SELECT event_type, payload_json FROM event_log WHERE position > ? ORDER BY position",
                (len(bootstrap_events),),
            ) as cursor:
                created_events = list(await cursor.fetchall())
            assert [str(row[0]) for row in created_events] == [
                WorkshopEventType.CHANNEL_CREATED,
                WorkshopEventType.CHANNEL_MEMBER_ADDED,
                WorkshopEventType.CHANNEL_MEMBER_ADDED,
                WorkshopEventType.CHANNEL_AGENT_ATTACHED,
                WorkshopEventType.RUNTIME_PROFILE_ASSIGNED,
            ]
            assert json.loads(str(created_events[0][1])) == {
                "kind": "group",
                "name": "Release planning",
                "origin_channel_id": alice_channel,
                "visibility": "private",
            }
            async with store.connection.execute(
                "SELECT COUNT(*) FROM channel_bindings WHERE channel_id = ?",
                (channel_id,),
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 0
            async with store.connection.execute(
                "SELECT * FROM event_log WHERE position <= ? ORDER BY position",
                (len(bootstrap_events),),
            ) as cursor:
                assert [tuple(row) for row in await cursor.fetchall()] == bootstrap_events
            async with store.connection.execute(
                "SELECT * FROM channels WHERE id = ?",
                (alice_channel,),
            ) as cursor:
                assert tuple(await cursor.fetchone()) == origin_before
            async with store.connection.execute("SELECT * FROM channels ORDER BY id") as cursor:
                current_channels = {str(row[0]): tuple(row) for row in await cursor.fetchall()}
            assert {
                channel_key: current_channels[channel_key] for channel_key in bootstrap_channels
            } == bootstrap_channels
            async with store.connection.execute("SELECT * FROM channel_memberships ORDER BY id") as cursor:
                current_memberships = {str(row[0]): tuple(row) for row in await cursor.fetchall()}
            assert {
                membership_key: current_memberships[membership_key] for membership_key in bootstrap_memberships
            } == bootstrap_memberships

            navigation_response = await client.get(
                "/v1/client/navigation",
                headers={"Authorization": "Bearer alice"},
            )
            assert navigation_response.status == 200
            navigation = await navigation_response.json()
            visible = next(item for item in navigation["workshops"][0]["channels"] if item["channel_id"] == channel_id)
            assert visible["kind"] == "group"
            assert visible["name"] == "Release planning"
            assert visible["role"] == "owner"
            assert visible["agents"] == [{"agent_id": agent_id, "name": "Kai"}]
            assert visible["can_submit_commands"] is True

            history = await WorkshopChannelHistoryRegistry.from_store(
                store,
                profile_registry(101, 202),
            )
            assert channel_id in {namespace.channel_id for namespace in history.namespaces}
            assert history.for_compatibility_chat_id(101).channel_id == alice_channel
        finally:
            await client.close()
            await store.close()

    async def test_unknown_agent_rejects_the_entire_event_set(
        self,
        tmp_path: Path,
    ) -> None:
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
            event_count = int((await cursor.fetchone())[0])
        async with store.connection.execute("SELECT COUNT(*) FROM channels") as cursor:
            channel_count = int((await cursor.fetchone())[0])
        client = await _open_client(
            store,
            _Authenticator({"alice": alice_id}),
        )
        try:
            response = await client.post(
                "/v1/channels",
                headers={"Authorization": "Bearer alice"},
                json={
                    "name": "Must not exist",
                    "agent_ids": [AgentId.new()],
                    "origin_channel_id": alice_channel,
                },
            )

            assert response.status == 400
            assert (await response.json())["error"]["code"] == "invalid_request"
            async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
                assert int((await cursor.fetchone())[0]) == event_count
            async with store.connection.execute("SELECT COUNT(*) FROM channels") as cursor:
                assert int((await cursor.fetchone())[0]) == channel_count
        finally:
            await client.close()
            await store.close()

    async def test_projection_failure_rolls_back_every_creation_event(
        self,
        tmp_path: Path,
    ) -> None:
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        async with store.connection.execute(
            "SELECT agent_id FROM channel_agents WHERE channel_id = ?",
            (alice_channel,),
        ) as cursor:
            agent_id = AgentId(str((await cursor.fetchone())[0]))
        async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
            event_count = int((await cursor.fetchone())[0])
        async with store.connection.execute("SELECT COUNT(*) FROM channels") as cursor:
            channel_count = int((await cursor.fetchone())[0])
        await store.connection.execute(
            "CREATE TRIGGER reject_new_channel_agent "
            "BEFORE INSERT ON channel_agents "
            "BEGIN SELECT RAISE(ABORT, 'forced projection failure'); END"
        )
        await store.connection.commit()
        client = await _open_client(
            store,
            _Authenticator({"alice": alice_id}),
        )
        try:
            response = await client.post(
                "/v1/channels",
                headers={"Authorization": "Bearer alice"},
                json={
                    "name": "Must roll back",
                    "agent_ids": [agent_id],
                    "origin_channel_id": alice_channel,
                },
            )

            assert response.status == 503
            assert (await response.json())["error"] == {
                "code": "channel_creation_unavailable",
                "message": "Channel creation is temporarily unavailable",
            }
            async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
                assert int((await cursor.fetchone())[0]) == event_count
            async with store.connection.execute("SELECT COUNT(*) FROM channels") as cursor:
                assert int((await cursor.fetchone())[0]) == channel_count
        finally:
            await client.close()
            await store.close()

    async def test_authentication_precedes_channel_request_validation(
        self,
        tmp_path: Path,
    ) -> None:
        store, alice_id, _, _, _ = await _open_store(tmp_path / "kai.db")
        client = await _open_client(
            store,
            _Authenticator({"alice": alice_id}),
        )
        try:
            unauthenticated = await client.post(
                "/v1/channels?unsupported=1",
                data="not-json",
            )
            malformed = await client.post(
                "/v1/channels?unsupported=1",
                headers={"Authorization": "Bearer alice"},
                data="not-json",
            )

            assert unauthenticated.status == 401
            assert malformed.status == 400
        finally:
            await client.close()
            await store.close()


class TestWorkshopSettingsWorkspaceHTTPContract:
    async def test_model_catalogue_read_refresh_and_operator_actions_are_canonical(
        self,
        tmp_path: Path,
    ) -> None:
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        service = _SettingsWorkspaces(alice_id, alice_channel)
        client = await _open_client(
            store,
            _Authenticator({"alice-token": alice_id}),
            settings_workspaces=service,
        )
        headers = {"Authorization": "Bearer alice-token"}
        path = f"/v1/channels/{alice_channel}/models"
        try:
            loaded = await client.get(
                path,
                headers=headers,
                params={"option_id": "codex:openai"},
            )
            refreshed = await client.post(
                path,
                headers=headers,
                json={"option_id": "codex:openai"},
            )
            upserted = await client.put(
                path,
                headers=headers,
                json={
                    "option_id": "codex:openai",
                    "model_id": "gpt-5.6-sol",
                    "display_label": "GPT-5.6 Sol",
                },
            )
            deactivated = await client.delete(
                path,
                headers=headers,
                json={"option_id": "codex:openai", "model_id": "gpt-5.6-sol"},
            )
            all_refreshed = await client.post(
                "/v1/settings/model-catalogue/refresh-all",
                headers=headers,
            )

            assert [response.status for response in (loaded, refreshed, upserted, deactivated, all_refreshed)] == [
                200,
                200,
                200,
                200,
                200,
            ]
            payload = await loaded.json()
            assert payload["principal_id"] == str(alice_id)
            assert payload["option_id"] == "codex:openai"
            assert payload["models"] == [
                {
                    "model_id": "gpt-5.6-sol",
                    "display_name": "GPT-5.6 Sol",
                    "status": "available",
                    "selectable": True,
                    "retained": True,
                    "sources": ["discovered:fixture"],
                }
            ]
            assert await all_refreshed.json() == {
                "version": 1,
                "contexts": 1,
                "statuses": {"succeeded": 1},
                "selection_changed": False,
            }
            assert service.runtime_changes == []
            assert service.catalogue_calls == [
                ("inspect", "codex:openai"),
                ("refresh", "codex:openai"),
                ("inspect", "codex:openai"),
                ("upsert:gpt-5.6-sol:GPT-5.6 Sol", "codex:openai"),
                ("deactivate:gpt-5.6-sol", "codex:openai"),
                ("refresh_all", None),
            ]
        finally:
            await client.close()
            await store.close()

    async def test_model_catalogue_isolation_and_admin_boundary_precede_discovery(
        self,
        tmp_path: Path,
    ) -> None:
        store, alice_id, alice_channel, bob_id, bob_channel = await _open_store(tmp_path / "kai.db")
        service = _SettingsWorkspaces(alice_id, alice_channel)
        client = await _open_client(
            store,
            _Authenticator({"bob-token": bob_id}),
            settings_workspaces=service,
        )
        headers = {"Authorization": "Bearer bob-token"}
        try:
            foreign = await client.get(
                f"/v1/channels/{alice_channel}/models",
                headers=headers,
            )
            own_operator_write = await client.put(
                f"/v1/channels/{bob_channel}/models",
                headers=headers,
                json={
                    "option_id": "codex:openai",
                    "model_id": "forbidden",
                    "display_label": "Forbidden",
                },
            )
            refresh_all = await client.post(
                "/v1/settings/model-catalogue/refresh-all",
                headers=headers,
            )

            assert foreign.status == 403
            assert own_operator_write.status == 403
            assert refresh_all.status == 403
            assert service.catalogue_calls == []
            assert service.runtime_changes == []
        finally:
            await client.close()
            await store.close()

    async def test_owner_reads_and_switches_canonical_runtime_workspace(
        self,
        tmp_path: Path,
    ) -> None:
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        service = _SettingsWorkspaces(alice_id, alice_channel)
        client = await _open_client(
            store,
            _Authenticator({"alice-token": alice_id}),
            settings_workspaces=service,
        )
        try:
            settings = await client.get(
                f"/v1/channels/{alice_channel}/settings",
                headers={"Authorization": "Bearer alice-token"},
            )
            switched = await client.post(
                f"/v1/channels/{alice_channel}/workspace",
                headers={
                    "Authorization": "Bearer alice-token",
                    "Content-Type": "application/json",
                },
                json={"path": "/srv/other", "revision": "sws_current"},
            )
            changed = await client.patch(
                f"/v1/channels/{alice_channel}/settings",
                headers={
                    "Authorization": "Bearer alice-token",
                    "Content-Type": "application/json",
                },
                json={"timeout_seconds": 180, "revision": "sws_current"},
            )
            backend_changed = await client.patch(
                f"/v1/channels/{alice_channel}/settings",
                headers={
                    "Authorization": "Bearer alice-token",
                    "Content-Type": "application/json",
                },
                json={"backend": "claude", "revision": "sws_current"},
            )

            assert settings.status == 200
            settings_payload = await settings.json()
            assert settings_payload["model"] == {
                "value": "gpt-5.6-sol",
                "source": "runtime policy",
                "default_value": "gpt-5.6-sol",
            }
            assert settings_payload["revision"] == "sws_current"
            assert settings_payload["capabilities"] == [
                {
                    "field": "model",
                    "scope": "runtime",
                    "value_type": "model_id",
                    "resettable": True,
                    "choices": None,
                    "minimum": None,
                    "maximum": None,
                }
            ]
            assert settings_payload["mutation"] is None
            assert switched.status == 200
            assert (await switched.json())["workspace"] == "/srv/other"
            assert changed.status == 200
            assert backend_changed.status == 200
            assert service.switched == ["/srv/other"]
            assert service.runtime_changes == [
                ("timeout", 180),
                ("backend", "claude"),
            ]
        finally:
            await client.close()
            await store.close()

    async def test_workspace_config_uses_the_same_canonical_authority(
        self,
        tmp_path: Path,
    ) -> None:
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        service = _SettingsWorkspaces(alice_id, alice_channel)
        client = await _open_client(
            store,
            _Authenticator({"alice-token": alice_id}),
            settings_workspaces=service,
        )
        try:
            current = await client.get(
                f"/v1/channels/{alice_channel}/workspace-config",
                headers={"Authorization": "Bearer alice-token"},
            )
            changed = await client.patch(
                f"/v1/channels/{alice_channel}/workspace-config",
                headers={
                    "Authorization": "Bearer alice-token",
                    "Content-Type": "application/json",
                },
                json={
                    "field": "timeout",
                    "value": "180",
                    "revision": "sws_workspace",
                },
            )

            assert current.status == 200
            assert (await current.json())["environment_keys"] == ["SAFE_KEY"]
            assert changed.status == 200
            assert service.workspace_config_changes == [("timeout", "180")]
        finally:
            await client.close()
            await store.close()

    async def test_cross_principal_settings_access_fails_before_service_mutation(
        self,
        tmp_path: Path,
    ) -> None:
        store, alice_id, alice_channel, bob_id, _ = await _open_store(tmp_path / "kai.db")
        service = _SettingsWorkspaces(alice_id, alice_channel)
        client = await _open_client(
            store,
            _Authenticator({"bob-token": bob_id}),
            settings_workspaces=service,
        )
        try:
            response = await client.patch(
                f"/v1/channels/{alice_channel}/settings",
                headers={
                    "Authorization": "Bearer bob-token",
                    "Content-Type": "application/json",
                },
                json={"backend": "claude", "revision": "sws_current"},
            )

            assert response.status == 403
            assert (await response.json())["error"]["code"] == "access_denied"
            assert service.switched == []
            assert service.runtime_changes == []
        finally:
            await client.close()
            await store.close()

    async def test_cross_channel_settings_access_fails_before_service_mutation(
        self,
        tmp_path: Path,
    ) -> None:
        store, alice_id, alice_channel, _, bob_channel = await _open_store(tmp_path / "kai.db")
        service = _SettingsWorkspaces(alice_id, alice_channel)
        client = await _open_client(
            store,
            _Authenticator({"alice-token": alice_id}),
            settings_workspaces=service,
        )
        try:
            response = await client.post(
                f"/v1/channels/{bob_channel}/workspace",
                headers={
                    "Authorization": "Bearer alice-token",
                    "Content-Type": "application/json",
                },
                json={"path": "/srv/other"},
            )

            assert response.status == 403
            assert (await response.json())["error"]["code"] == "access_denied"
            assert service.switched == []
        finally:
            await client.close()
            await store.close()

    async def test_stale_revision_and_environment_edit_fail_without_mutation(
        self,
        tmp_path: Path,
    ) -> None:
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        service = _SettingsWorkspaces(alice_id, alice_channel)
        client = await _open_client(
            store,
            _Authenticator({"alice-token": alice_id}),
            settings_workspaces=service,
        )
        headers = {
            "Authorization": "Bearer alice-token",
            "Content-Type": "application/json",
        }
        try:
            stale = await client.patch(
                f"/v1/channels/{alice_channel}/settings",
                headers=headers,
                json={"timeout_seconds": 180, "revision": "sws_stale"},
            )
            environment = await client.patch(
                f"/v1/channels/{alice_channel}/workspace-config",
                headers=headers,
                json={
                    "field": "env",
                    "value": '{"SECRET":"must-not-be-returned"}',
                    "revision": "sws_workspace",
                },
            )

            assert stale.status == 409
            assert (await stale.json())["error"]["code"] == "settings_conflict"
            assert environment.status == 400
            environment_payload = await environment.json()
            assert environment_payload["error"]["code"] == "invalid_setting"
            assert "must-not-be-returned" not in repr(environment_payload)
            assert service.workspace_config_changes == []
        finally:
            await client.close()
            await store.close()


class TestWorkshopCommandHTTPContract:
    async def test_authenticated_multipart_upload_is_staged_for_canonical_acceptance(
        self,
        tmp_path: Path,
    ):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        service = await _artifact_service(store, tmp_path)
        submitter = _CommandSubmitter()
        client = await _open_command_client(
            store,
            _Authenticator({"alice-token": alice_id}),
            submitter,
            service,
        )
        form = FormData()
        form.add_field("client_message_id", "browser-artifact-1")
        form.add_field("body", "Please inspect the attachment")
        form.add_field(
            "file",
            b"Workshop artifact content",
            filename="notes.txt",
            content_type="text/plain",
        )
        try:
            response = await client.post(
                f"/v1/channels/{alice_channel}/commands",
                headers={"Authorization": "Bearer alice-token"},
                data=form,
            )

            response_body = await response.text()
            assert response.status == 202, response_body
            assert len(submitter.messages) == 1
            assert len(submitter.artifacts) == 1
            artifact = submitter.artifacts[0]
            assert artifact is not None
            assert artifact.storage_path.read_bytes() == b"Workshop artifact content"
            assert artifact.storage_path.parent == tmp_path / "files" / str(alice_id)
            assert artifact.original_filename == "notes.txt"
            assert submitter.messages[0].artifact_source_unique_id == "browser-artifact-1"
        finally:
            await client.close()
            await store.close()

    async def test_upload_authorization_precedes_multipart_parsing(self, tmp_path: Path):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        service = await _artifact_service(store, tmp_path)
        submitter = _CommandSubmitter()
        client = await _open_command_client(
            store,
            _Authenticator({"alice-token": alice_id}),
            submitter,
            service,
        )
        try:
            response = await client.post(
                f"/v1/channels/{alice_channel}/commands",
                data=b"untrusted body that must not be parsed",
                headers={"Content-Type": "multipart/form-data"},
            )

            assert response.status == 401
            assert submitter.messages == []
            assert not (tmp_path / "files").exists()
        finally:
            await client.close()
            await store.close()

    async def test_authenticated_member_submits_only_server_scoped_command_fields(
        self,
        tmp_path: Path,
    ):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        submitter = _CommandSubmitter()
        client = await _open_command_client(
            store,
            _Authenticator({"alice-token": alice_id}),
            submitter,
        )
        try:
            response = await client.post(
                f"/v1/channels/{alice_channel}/commands",
                headers={"Authorization": "Bearer alice-token"},
                json={
                    "client_message_id": "browser-command-1",
                    "body": "Hello from Workshop",
                },
            )
            payload = await response.json()

            assert response.status == 202
            assert payload["version"] == 2
            assert payload["acceptance"] == "newly_accepted"
            assert payload["run"]["status"] == "accepted"
            assert payload["run"]["channel_id"] == alice_channel
            assert str(payload["message_id"]).startswith("msg_")
            assert str(payload["run_id"]).startswith("run_")
            assert len(submitter.messages) == 1
            submitted = submitter.messages[0]
            assert submitted.principal_id == alice_id
            assert submitted.channel_id == alice_channel
            assert submitted.client_message_id == "browser-command-1"
            assert submitted.body == "Hello from Workshop"
            assert submitted.occurred_at.tzinfo is not None
        finally:
            await client.close()
            await store.close()

    async def test_notification_channel_rejects_command_submission(self, tmp_path: Path):
        store, alice_id, _, _, _ = await _open_store(tmp_path / "kai.db")
        async with store.connection.execute(
            "SELECT c.id FROM channels c JOIN channel_bindings cb ON cb.channel_id = c.id "
            "WHERE c.kind = 'notification' AND cb.external_channel_id = '-100123'"
        ) as cursor:
            notification_row = await cursor.fetchone()
        assert notification_row is not None
        notification_channel = ChannelId(str(notification_row[0]))
        submitter = _CommandSubmitter()
        client = await _open_command_client(
            store,
            _Authenticator({"alice-token": alice_id}),
            submitter,
        )
        try:
            response = await client.post(
                f"/v1/channels/{notification_channel}/commands",
                headers={"Authorization": "Bearer alice-token"},
                json={
                    "client_message_id": "browser-command-1",
                    "body": "Do not execute from the notification channel",
                },
            )

            assert response.status == 403
            assert await response.json() == {"error": {"code": "access_denied", "message": "Access denied"}}
            assert submitter.messages == []
        finally:
            await client.close()
            await store.close()

    async def test_owner_can_inspect_and_cancel_accepted_run(self, tmp_path: Path):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        submitter = _CommandSubmitter()
        client = await _open_command_client(
            store,
            _Authenticator({"alice-token": alice_id}),
            submitter,
        )
        try:
            accepted = await client.post(
                f"/v1/channels/{alice_channel}/commands",
                headers={"Authorization": "Bearer alice-token"},
                json={"client_message_id": "browser-command-1", "body": "Long task"},
            )
            run_id = (await accepted.json())["run_id"]

            state = await client.get(
                f"/v1/channels/{alice_channel}/runs/{run_id}",
                headers={"Authorization": "Bearer alice-token"},
            )
            invalid = await client.post(
                f"/v1/channels/{alice_channel}/runs/{run_id}/cancel",
                headers={"Authorization": "Bearer alice-token"},
                data="unexpected",
            )
            cancelled = await client.post(
                f"/v1/channels/{alice_channel}/runs/{run_id}/cancel",
                headers={"Authorization": "Bearer alice-token"},
            )

            assert state.status == 200
            assert (await state.json())["run"]["status"] == "accepted"
            assert invalid.status == 400
            assert cancelled.status == 200
            cancellation_payload = await cancelled.json()
            assert cancellation_payload["cancellation"] == "requested"
            assert cancellation_payload["run"]["status"] == "cancelled"
        finally:
            await client.close()
            await store.close()

    async def test_run_state_does_not_leak_across_principals_or_unknown_ids(self, tmp_path: Path):
        store, alice_id, alice_channel, bob_id, _ = await _open_store(tmp_path / "kai.db")
        submitter = _CommandSubmitter()
        client = await _open_command_client(
            store,
            _Authenticator({"alice-token": alice_id, "bob-token": bob_id}),
            submitter,
        )
        try:
            accepted = await client.post(
                f"/v1/channels/{alice_channel}/commands",
                headers={"Authorization": "Bearer alice-token"},
                json={"client_message_id": "browser-command-1", "body": "Private work"},
            )
            run_id = (await accepted.json())["run_id"]
            headers = {"Authorization": "Bearer bob-token"}

            cross_principal = await client.get(
                f"/v1/channels/{alice_channel}/runs/{run_id}",
                headers=headers,
            )
            unknown = await client.get(
                f"/v1/channels/{alice_channel}/runs/{RunId.new()}",
                headers=headers,
            )

            assert cross_principal.status == 403
            assert unknown.status == 403
            assert await cross_principal.json() == await unknown.json()
        finally:
            await client.close()
            await store.close()

    async def test_authentication_precedes_parsing_and_cross_channel_is_denied(
        self,
        tmp_path: Path,
    ):
        store, alice_id, alice_channel, _, bob_channel = await _open_store(tmp_path / "kai.db")
        submitter = _CommandSubmitter()
        client = await _open_command_client(
            store,
            _Authenticator({"alice-token": alice_id}),
            submitter,
        )
        try:
            unauthenticated = await client.post(
                f"/v1/channels/{alice_channel}/commands",
                data="not json",
            )
            denied = await client.post(
                f"/v1/channels/{bob_channel}/commands",
                headers={"Authorization": "Bearer alice-token"},
                json={
                    "client_message_id": "browser-command-1",
                    "body": "Cross-channel command",
                },
            )

            assert unauthenticated.status == 401
            assert unauthenticated.headers["WWW-Authenticate"] == "Bearer"
            assert denied.status == 403
            assert submitter.messages == []
        finally:
            await client.close()
            await store.close()

    @pytest.mark.parametrize(
        "payload",
        [
            {"client_message_id": "browser-command-1"},
            {"client_message_id": "browser-command-1", "body": "Hello", "model": "gpt"},
            {"client_message_id": "bad id", "body": "Hello"},
            {"client_message_id": "browser-command-1", "body": "   "},
        ],
    )
    async def test_rejects_invalid_or_authority_expanding_input(
        self,
        tmp_path: Path,
        payload: dict[str, str],
    ):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        submitter = _CommandSubmitter()
        client = await _open_command_client(
            store,
            _Authenticator({"alice-token": alice_id}),
            submitter,
        )
        try:
            response = await client.post(
                f"/v1/channels/{alice_channel}/commands",
                headers={"Authorization": "Bearer alice-token"},
                json=payload,
            )

            assert response.status == 400
            assert submitter.messages == []
        finally:
            await client.close()
            await store.close()


class TestWorkshopArtifactHTTPContract:
    async def test_member_previews_and_natively_downloads_an_opaque_artifact(
        self,
        tmp_path: Path,
    ):
        store, alice_id, alice_channel, bob_id, bob_channel = await _open_store(tmp_path / "kai.db")
        service = await _artifact_service(store, tmp_path)

        async def content():
            yield b"private attachment"

        staged = await service.stage_client_upload(
            principal_id=alice_id,
            channel_id=alice_channel,
            client_message_id="browser-download-1",
            filename="qualification.aiff",
            claimed_media_type="audio/aiff",
            chunks=content(),
            occurred_at=_NOW,
        )
        await _record_messages(store, 1)
        async with store.connection.execute(
            "SELECT id FROM messages ORDER BY created_event_position DESC LIMIT 1"
        ) as cursor:
            message_id = MessageId(str((await cursor.fetchone())[0]))
        recorded = await record_inbound_artifact(
            store,
            staged.for_message(message_id),
            storage_root=tmp_path / "files",
        )
        artifact_id = recorded.event.envelope.aggregate_id
        client = await _open_client(
            store,
            _Authenticator({"alice-token": alice_id, "bob-token": bob_id}),
            artifact_service=service,
        )
        try:
            response = await client.get(
                f"/v1/channels/{alice_channel}/artifacts/{artifact_id}/content",
                headers={"Authorization": "Bearer alice-token"},
            )
            download = await client.post(
                f"/v1/channels/{alice_channel}/artifacts/{artifact_id}/download",
                data={"session_token": "alice-token"},
            )
            unauthenticated_download = await client.post(
                f"/v1/channels/{alice_channel}/artifacts/{artifact_id}/download",
                data={"session_token": "unknown-token"},
            )
            cross_principal_download = await client.post(
                f"/v1/channels/{alice_channel}/artifacts/{artifact_id}/download",
                data={"session_token": "bob-token"},
            )
            cross_principal = await client.get(
                f"/v1/channels/{alice_channel}/artifacts/{artifact_id}/content",
                headers={"Authorization": "Bearer bob-token"},
            )
            wrong_channel = await client.get(
                f"/v1/channels/{bob_channel}/artifacts/{artifact_id}/content",
                headers={"Authorization": "Bearer alice-token"},
            )
            timeline = await client.get(
                f"/v1/channels/{alice_channel}/timeline",
                headers={"Authorization": "Bearer alice-token"},
            )
            timeline_payload = await timeline.json()
            artifact_payload = timeline_payload["messages"][0]["artifacts"][0]

            assert response.status == 200
            assert await response.read() == b"private attachment"
            assert response.headers["Cache-Control"] == "private, no-store"
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert response.headers["Content-Disposition"].startswith("inline;")
            assert download.status == 200
            assert await download.read() == b"private attachment"
            assert download.headers["Content-Disposition"].startswith("attachment;")
            assert 'filename="qualification.aiff"' in download.headers["Content-Disposition"]
            assert unauthenticated_download.status == 401
            assert cross_principal_download.status == 403
            assert cross_principal.status == 403
            assert wrong_channel.status == 403
            assert await cross_principal.json() == await wrong_channel.json()
            assert artifact_payload["artifact_id"] == artifact_id
            assert artifact_payload["original_filename"] == "qualification.aiff"
            assert artifact_payload["media_type"] == "audio/aiff"
            assert "storage_path" not in artifact_payload
            assert "source_unique_id" not in artifact_payload
        finally:
            await client.close()
            await store.close()


class TestWorkshopTimelineHTTPContract:
    async def test_authenticated_member_receives_versioned_canonical_page(self, tmp_path: Path):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        await _record_messages(store, 1)
        authenticator = _Authenticator({"alice-token": alice_id})
        client = await _open_client(store, authenticator)
        try:
            response = await client.get(
                f"/v1/channels/{alice_channel}/timeline",
                headers={"Authorization": "Bearer alice-token"},
            )
            payload = await response.json()

            assert response.status == 200
            assert response.headers["Cache-Control"] == "private, no-store"
            assert response.headers["Content-Security-Policy"] == "default-src 'none'"
            assert response.headers["Referrer-Policy"] == "no-referrer"
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert payload == {
                "version": 1,
                "channel_id": alice_channel,
                "messages": [
                    {
                        "message_id": payload["messages"][0]["message_id"],
                        "channel_id": alice_channel,
                        "author_principal_id": alice_id,
                        "author_kind": "human",
                        "author_display_name": "Alice",
                        "reply_to_message_id": None,
                        "body": "Message 1",
                        "event_position": payload["messages"][0]["event_position"],
                        "created_at": "2026-08-11T14:00:01Z",
                        "artifacts": [],
                    }
                ],
                "next_cursor": None,
                "previous_cursor": None,
                "through_position": payload["messages"][0]["event_position"],
            }
            assert payload["messages"][0]["message_id"].startswith("msg_")
            assert authenticator.calls == ["Bearer alice-token"]
        finally:
            await client.close()
            await store.close()

    async def test_unauthenticated_request_is_rejected_before_input_or_storage(self, tmp_path: Path):
        store, alice_id, _, _, _ = await _open_store(tmp_path / "kai.db")
        authenticator = _Authenticator({"alice-token": alice_id})
        client = await _open_client(store, authenticator)
        await store.close()
        try:
            response = await client.get("/v1/channels/not-an-id/timeline?limit=invalid")

            assert response.status == 401
            assert response.headers["WWW-Authenticate"] == "Bearer"
            assert await response.json() == {
                "error": {"code": "authentication_required", "message": "Authentication required"}
            }
        finally:
            await client.close()

    async def test_cross_channel_and_unknown_channel_have_same_denial(self, tmp_path: Path):
        store, alice_id, _, _, bob_channel = await _open_store(tmp_path / "kai.db")
        authenticator = _Authenticator({"alice-token": alice_id})
        client = await _open_client(store, authenticator)
        try:
            headers = {"Authorization": "Bearer alice-token"}
            cross_channel = await client.get(f"/v1/channels/{bob_channel}/timeline", headers=headers)
            unknown_channel = await client.get(f"/v1/channels/{ChannelId.new()}/timeline", headers=headers)

            assert cross_channel.status == unknown_channel.status == 403
            assert (
                await cross_channel.json()
                == await unknown_channel.json()
                == {"error": {"code": "access_denied", "message": "Access denied"}}
            )
        finally:
            await client.close()
            await store.close()

    @pytest.mark.parametrize(
        "query",
        [
            "?limit=0",
            "?limit=101",
            "?limit=not-a-number",
            "?limit=1&limit=2",
            "?cursor=not-a-cursor",
            "?cursor=one&cursor=two",
            "?chat_id=101",
            "?tail=0",
            "?tail=true",
            "?tail=1&tail=1",
            "?tail=1&cursor=not-a-cursor",
        ],
    )
    async def test_invalid_pagination_input_returns_bounded_error(self, tmp_path: Path, query: str):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        client = await _open_client(store, _Authenticator({"alice-token": alice_id}))
        try:
            response = await client.get(
                f"/v1/channels/{alice_channel}/timeline{query}",
                headers={"Authorization": "Bearer alice-token"},
            )

            assert response.status == 400
            assert await response.json() == {
                "error": {"code": "invalid_request", "message": "Invalid timeline request"}
            }
        finally:
            await client.close()
            await store.close()

    async def test_cursor_resumes_same_snapshot_after_store_restart(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        store, alice_id, alice_channel, _, _ = await _open_store(db_path)
        await _record_messages(store, 3)
        first_client = await _open_client(store, _Authenticator({"alice-token": alice_id}))
        response = await first_client.get(
            f"/v1/channels/{alice_channel}/timeline?limit=2",
            headers={"Authorization": "Bearer alice-token"},
        )
        first_page = await response.json()
        await first_client.close()
        await store.close()

        restarted = await WorkshopEventStore.open(db_path)
        second_client = await _open_client(restarted, _Authenticator({"new-token": alice_id}))
        try:
            response = await second_client.get(
                f"/v1/channels/{alice_channel}/timeline",
                params={"cursor": first_page["next_cursor"], "limit": "2"},
                headers={"Authorization": "Bearer new-token"},
            )
            second_page = await response.json()

            assert response.status == 200
            assert [message["body"] for message in first_page["messages"]] == ["Message 1", "Message 2"]
            assert [message["body"] for message in second_page["messages"]] == ["Message 3"]
            assert second_page["through_position"] == first_page["through_position"]
            assert second_page["next_cursor"] is None
        finally:
            await second_client.close()
            await restarted.close()

    async def test_tail_request_pages_backwards_over_http(self, tmp_path: Path):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        await _record_messages(store, 3)
        client = await _open_client(store, _Authenticator({"alice-token": alice_id}))
        try:
            headers = {"Authorization": "Bearer alice-token"}
            response = await client.get(
                f"/v1/channels/{alice_channel}/timeline?tail=1&limit=2",
                headers=headers,
            )
            tail_page = await response.json()

            assert response.status == 200
            assert [message["body"] for message in tail_page["messages"]] == ["Message 2", "Message 3"]
            assert tail_page["next_cursor"] is None
            assert tail_page["previous_cursor"] is not None

            response = await client.get(
                f"/v1/channels/{alice_channel}/timeline",
                params={"cursor": tail_page["previous_cursor"], "limit": "2"},
                headers=headers,
            )
            earlier_page = await response.json()

            assert response.status == 200
            assert [message["body"] for message in earlier_page["messages"]] == ["Message 1"]
            assert earlier_page["previous_cursor"] is None
            assert earlier_page["next_cursor"] is None
            assert earlier_page["through_position"] == tail_page["through_position"]
        finally:
            await client.close()
            await store.close()

    async def test_route_accepts_no_write_method(self, tmp_path: Path):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        client = await _open_client(store, _Authenticator({"alice-token": alice_id}))
        try:
            response = await client.post(
                f"/v1/channels/{alice_channel}/timeline",
                headers={"Authorization": "Bearer alice-token"},
                json={"body": "must not be accepted"},
            )

            assert response.status == 405
            events_response = await client.post(
                f"/v1/channels/{alice_channel}/events",
                headers={"Authorization": "Bearer alice-token"},
                json={"body": "must not be accepted"},
            )
            assert events_response.status == 405
            async with store.connection.execute("SELECT COUNT(*) FROM messages") as cursor:
                assert (await cursor.fetchone())[0] == 0
        finally:
            await client.close()
            await store.close()


class TestWorkshopTimelineEventStreamHTTPContract:
    async def test_run_activity_is_private_even_when_channel_read_is_allowed(self, tmp_path: Path):
        store, alice_id, alice_channel, bob_id, _ = await _open_store(tmp_path / "kai.db")
        inbound = await record_inbound_message(
            store,
            InboundMessage(
                "telegram",
                "private-run-1",
                "private-message-1",
                "101",
                "101",
                "Private run state",
                _NOW,
            ),
        )
        message_id = inbound.event.envelope.aggregate_id
        assert isinstance(message_id, MessageId)
        accepted = await WorkshopRunLifecycle(store).accept(
            message_id,
            occurred_at=_NOW + timedelta(seconds=1),
        )
        try:
            alice = await read_client_channel_events(
                store,
                principal_id=alice_id,
                channel_id=alice_channel,
                authorizer=_AllowChannelRead(),
                after_position=0,
            )
            bob = await read_client_channel_events(
                store,
                principal_id=bob_id,
                channel_id=alice_channel,
                authorizer=_AllowChannelRead(),
                after_position=0,
            )
            future_only = await read_client_channel_events(
                store,
                principal_id=alice_id,
                channel_id=alice_channel,
                authorizer=_AllowChannelRead(),
                after_position=None,
            )

            assert any(isinstance(event, ClientRunLifecycleEvent) for event in alice.events)
            assert not any(isinstance(event, ClientRunLifecycleEvent) for event in bob.events)
            assert any(isinstance(event, ClientTimelineMessageEvent) for event in bob.events)
            assert bob.next_position == inbound.event.position
            assert accepted.event.position > bob.next_position
            assert future_only.events == ()
            assert future_only.next_position == accepted.event.position
        finally:
            await store.close()

    async def test_replays_requester_run_lifecycle_as_versioned_sse(self, tmp_path: Path):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        inbound = await record_inbound_message(
            store,
            InboundMessage(
                "telegram",
                "run-update-1",
                "run-message-1",
                "101",
                "101",
                "Perform one task",
                _NOW,
            ),
        )
        message_id = inbound.event.envelope.aggregate_id
        assert isinstance(message_id, MessageId)
        accepted = await WorkshopRunLifecycle(store).accept(
            message_id,
            occurred_at=_NOW + timedelta(seconds=1),
        )
        client = await _open_client(store, _Authenticator({"alice-token": alice_id}))
        response = None
        try:
            response = await client.get(
                f"/v1/channels/{alice_channel}/events",
                params={"after_position": str(accepted.event.position - 1)},
                headers={"Authorization": "Bearer alice-token"},
            )
            event = await _read_sse_event(response)

            assert event["event"] == "run.lifecycle.changed"
            assert event["id"] == str(accepted.event.position)
            assert event["data"] == {
                "version": 1,
                "channel_id": alice_channel,
                "event_position": accepted.event.position,
                "transition": "run.accepted",
                "occurred_at": "2026-08-11T14:00:01Z",
                "run": {
                    "run_id": accepted.run.run_id,
                    "channel_id": alice_channel,
                    "status": "accepted",
                    "accepted_at": "2026-08-11T14:00:01Z",
                    "started_at": None,
                    "terminal_at": None,
                    "terminal_code": None,
                    "cancellation_requested_at": None,
                    "result_message_id": None,
                },
            }
        finally:
            if response is not None:
                response.close()
            await client.close()
            await store.close()

    async def test_replays_authorized_canonical_messages_as_versioned_sse(self, tmp_path: Path):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        await _record_messages(store, 2)
        client = await _open_client(store, _Authenticator({"alice-token": alice_id}))
        response = None
        try:
            response = await client.get(
                f"/v1/channels/{alice_channel}/events?after_position=0",
                headers={"Authorization": "Bearer alice-token"},
            )
            first = await _read_sse_event(response)
            second = await _read_sse_event(response)

            assert response.status == 200
            assert response.content_type == "text/event-stream"
            assert response.charset == "utf-8"
            assert response.headers["Cache-Control"] == "private, no-store"
            assert response.headers["Content-Security-Policy"] == "default-src 'none'"
            assert response.headers["Referrer-Policy"] == "no-referrer"
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert response.headers["X-Accel-Buffering"] == "no"
            assert [first["event"], second["event"]] == [
                "timeline.message.created",
                "timeline.message.created",
            ]
            assert int(str(first["id"])) < int(str(second["id"]))
            assert first["data"] == {
                "version": 1,
                "channel_id": alice_channel,
                "message": {
                    "message_id": first["data"]["message"]["message_id"],
                    "channel_id": alice_channel,
                    "author_principal_id": alice_id,
                    "author_kind": "human",
                    "author_display_name": "Alice",
                    "reply_to_message_id": None,
                    "body": "Message 1",
                    "event_position": int(str(first["id"])),
                    "created_at": "2026-08-11T14:00:01Z",
                    "artifacts": [],
                },
            }
            assert second["data"]["message"]["body"] == "Message 2"
        finally:
            if response is not None:
                response.close()
            await client.close()
            await store.close()

    async def test_future_only_stream_receives_message_committed_after_connect(self, tmp_path: Path):
        database = tmp_path / "kai.db"
        writer, alice_id, alice_channel, _, _ = await _open_store(database)
        reader = await WorkshopEventStore.open(database)
        client = await _open_client(reader, _Authenticator({"alice-token": alice_id}))
        response = None
        try:
            response = await client.get(
                f"/v1/channels/{alice_channel}/events",
                headers={"Authorization": "Bearer alice-token"},
            )
            await _record_messages(writer, 1)

            event = await _read_sse_event(response)

            assert event["event"] == "timeline.message.created"
            assert event["data"]["message"]["body"] == "Message 1"
        finally:
            if response is not None:
                response.close()
            await client.close()
            await reader.close()
            await writer.close()

    async def test_last_event_id_resumes_after_store_restart_and_overrides_initial_query(self, tmp_path: Path):
        database = tmp_path / "kai.db"
        store, alice_id, alice_channel, _, _ = await _open_store(database)
        await _record_messages(store, 1)
        first_client = await _open_client(store, _Authenticator({"alice-token": alice_id}))
        first_response = await first_client.get(
            f"/v1/channels/{alice_channel}/events?after_position=0",
            headers={"Authorization": "Bearer alice-token"},
        )
        first_event = await _read_sse_event(first_response)
        first_response.close()
        await first_client.close()
        await _record_messages(store, 1, start=2)
        await store.close()

        restarted = await WorkshopEventStore.open(database)
        second_client = await _open_client(restarted, _Authenticator({"new-token": alice_id}))
        second_response = None
        try:
            second_response = await second_client.get(
                f"/v1/channels/{alice_channel}/events?after_position=0",
                headers={
                    "Authorization": "Bearer new-token",
                    "Last-Event-ID": str(first_event["id"]),
                },
            )
            second_event = await _read_sse_event(second_response)

            assert int(str(second_event["id"])) > int(str(first_event["id"]))
            assert second_event["data"]["message"]["body"] == "Message 2"
        finally:
            if second_response is not None:
                second_response.close()
            await second_client.close()
            await restarted.close()

    async def test_open_stream_rechecks_authentication_and_closes_after_revocation(self, tmp_path: Path):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        authenticator = _Authenticator({"alice-token": alice_id})
        client = await _open_client(store, authenticator)
        response = None
        try:
            response = await client.get(
                f"/v1/channels/{alice_channel}/events",
                headers={"Authorization": "Bearer alice-token"},
            )
            assert (await response.content.readline()).startswith(b": connected")
            assert await response.content.readline() == b"retry: 2000\n"
            assert await response.content.readline() == b"\n"

            authenticator.principals_by_token.clear()

            assert await asyncio.wait_for(response.content.read(), timeout=1.0) == b""
            assert len(authenticator.calls) >= 2
        finally:
            if response is not None:
                response.close()
            await client.close()
            await store.close()

    async def test_idle_polling_does_not_rewrite_session_last_seen_on_every_poll(self, tmp_path: Path):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        authenticator = _Authenticator({"alice-token": alice_id})
        client = await _open_client(
            store,
            authenticator,
            event_poll_interval=0.005,
            event_heartbeat_interval=0.01,
            event_authentication_recheck_interval=0.2,
        )
        response = None
        try:
            response = await client.get(
                f"/v1/channels/{alice_channel}/events",
                headers={"Authorization": "Bearer alice-token"},
            )
            await asyncio.sleep(0.05)

            assert authenticator.calls == ["Bearer alice-token"]
        finally:
            if response is not None:
                response.close()
            await client.close()
            await store.close()

    async def test_concurrent_stream_capacity_is_bounded_and_released_on_disconnect(self, tmp_path: Path):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        limiter = WorkshopEventStreamLimiter(per_principal_limit=1, global_limit=1)
        client = await _open_client(
            store,
            _Authenticator({"alice-token": alice_id}),
            event_poll_interval=0.005,
            event_heartbeat_interval=0.01,
            event_stream_limiter=limiter,
        )
        first_response = None
        replacement_response = None
        try:
            headers = {"Authorization": "Bearer alice-token"}
            path = f"/v1/channels/{alice_channel}/events"
            first_response = await client.get(path, headers=headers)
            rejected = await client.get(path, headers=headers)

            assert first_response.status == 200
            assert rejected.status == 429
            assert rejected.headers["Retry-After"] == "5"
            assert await rejected.json() == {
                "error": {
                    "code": "stream_capacity_exceeded",
                    "message": "Too many active event streams",
                }
            }

            first_response.close()
            first_response = None
            await asyncio.sleep(0.05)
            replacement_response = await client.get(path, headers=headers)
            assert replacement_response.status == 200
        finally:
            if first_response is not None:
                first_response.close()
            if replacement_response is not None:
                replacement_response.close()
            await client.close()
            await store.close()

    async def test_server_shutdown_closes_an_open_event_stream_promptly(self, tmp_path: Path):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        limiter = WorkshopEventStreamLimiter(per_principal_limit=1, global_limit=1)
        client = await _open_client(
            store,
            _Authenticator({"alice-token": alice_id}),
            event_poll_interval=30.0,
            event_heartbeat_interval=30.0,
            event_stream_limiter=limiter,
        )
        response = None
        try:
            response = await client.get(
                f"/v1/channels/{alice_channel}/events",
                headers={"Authorization": "Bearer alice-token"},
            )
            assert await response.content.readline() == b": connected\n"

            await asyncio.wait_for(client.server.close(), timeout=1.0)

            assert limiter.acquire(alice_id) is True
            limiter.release(alice_id)
        finally:
            if response is not None:
                response.close()
            await client.close()
            await store.close()

    async def test_unauthenticated_event_stream_is_rejected_before_input_or_storage(self, tmp_path: Path):
        store, alice_id, _, _, _ = await _open_store(tmp_path / "kai.db")
        client = await _open_client(store, _Authenticator({"alice-token": alice_id}))
        await store.close()
        try:
            response = await client.get("/v1/channels/not-an-id/events?after_position=invalid")

            assert response.status == 401
            assert response.headers["WWW-Authenticate"] == "Bearer"
            assert await response.json() == {
                "error": {"code": "authentication_required", "message": "Authentication required"}
            }
        finally:
            await client.close()

    async def test_cross_channel_and_unknown_event_streams_have_same_denial(self, tmp_path: Path):
        store, alice_id, _, _, bob_channel = await _open_store(tmp_path / "kai.db")
        client = await _open_client(store, _Authenticator({"alice-token": alice_id}))
        try:
            headers = {"Authorization": "Bearer alice-token"}
            cross_channel = await client.get(f"/v1/channels/{bob_channel}/events", headers=headers)
            unknown_channel = await client.get(f"/v1/channels/{ChannelId.new()}/events", headers=headers)

            assert cross_channel.status == unknown_channel.status == 403
            assert (
                await cross_channel.json()
                == await unknown_channel.json()
                == {"error": {"code": "access_denied", "message": "Access denied"}}
            )
        finally:
            await client.close()
            await store.close()

    @pytest.mark.parametrize(
        ("query", "headers", "status", "error"),
        [
            ("?after_position=-1", {}, 400, "invalid_request"),
            ("?after_position=one", {}, 400, "invalid_request"),
            ("?after_position=1&after_position=2", {}, 400, "invalid_request"),
            ("?cursor=unused", {}, 400, "invalid_request"),
            ("", {"Last-Event-ID": "invalid"}, 400, "invalid_request"),
            ("?after_position=999999", {}, 409, "resynchronization_required"),
        ],
    )
    async def test_invalid_or_unresumable_event_position_is_bounded(
        self,
        tmp_path: Path,
        query: str,
        headers: dict[str, str],
        status: int,
        error: str,
    ):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        client = await _open_client(store, _Authenticator({"alice-token": alice_id}))
        try:
            response = await client.get(
                f"/v1/channels/{alice_channel}/events{query}",
                headers={"Authorization": "Bearer alice-token", **headers},
            )

            assert response.status == status
            assert (await response.json())["error"]["code"] == error
        finally:
            await client.close()
            await store.close()


class TestWorkshopRunPreviewEventStream:
    async def test_preview_events_carry_no_id_and_advance_by_sequence(self, tmp_path: Path):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        previews = WorkshopRunPreviewRegistry()
        run_id = RunId.new()
        previews.publish(run_id, ChannelId(alice_channel), "First sentence.")
        client = await _open_client(
            store,
            _Authenticator({"alice-token": alice_id}),
            run_previews=previews,
        )
        response = None
        try:
            response = await client.get(
                f"/v1/channels/{alice_channel}/events",
                headers={"Authorization": "Bearer alice-token"},
            )
            event = await _read_sse_event(response)
            while event["event"] != "run.preview.updated":
                event = await _read_sse_event(response)

            assert event["id"] is None
            assert event["data"] == {
                "version": 1,
                "channel_id": alice_channel,
                "run_id": str(run_id),
                "sequence": 1,
                "text": "First sentence.",
            }

            previews.publish(run_id, ChannelId(alice_channel), "First sentence. Second sentence.")
            event = await _read_sse_event(response)
            while event["event"] != "run.preview.updated":
                event = await _read_sse_event(response)

            # An unchanged preview is never re-sent, so the very next preview
            # event on the wire is the sequence-2 update.
            assert event["id"] is None
            assert event["data"]["sequence"] == 2
            assert event["data"]["text"] == "First sentence. Second sentence."
        finally:
            if response is not None:
                response.close()
            await client.close()
            await store.close()

    async def test_preview_events_are_scoped_to_their_channel(self, tmp_path: Path):
        store, _, alice_channel, bob_id, bob_channel = await _open_store(tmp_path / "kai.db")
        previews = WorkshopRunPreviewRegistry()
        previews.publish(RunId.new(), ChannelId(alice_channel), "Private to the other channel.")
        client = await _open_client(
            store,
            _Authenticator({"bob-token": bob_id}),
            run_previews=previews,
        )
        response = None
        try:
            response = await client.get(
                f"/v1/channels/{bob_channel}/events",
                headers={"Authorization": "Bearer bob-token"},
            )
            bob_run = RunId.new()
            previews.publish(bob_run, ChannelId(bob_channel), "Visible in this channel.")
            event = await _read_sse_event(response)
            while event["event"] != "run.preview.updated":
                event = await _read_sse_event(response)

            # The other channel's earlier preview was live for the whole
            # connection; the first preview this stream ever sees must be the
            # one published for its own channel.
            assert event["data"]["run_id"] == str(bob_run)
            assert event["data"]["channel_id"] == bob_channel
        finally:
            if response is not None:
                response.close()
            await client.close()
            await store.close()


# ── Run trace endpoint and doorbell ─────────────────────────────────


async def _insert_trace_rows(store: WorkshopEventStore, run_id: str, count: int, *, start: int = 1) -> None:
    for seq in range(start, start + count):
        await store.connection.execute(
            "INSERT INTO run_traces (run_id, seq, kind, tool_name, tool_use_id, "
            "summary, detail, is_diff, is_error, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                seq,
                "tool_call",
                "Bash",
                f"toolu_{seq}",
                f"summary {seq}",
                f"detail {seq}",
                0,
                0,
                "2026-08-11T14:00:00+00:00",
            ),
        )
    await store.connection.commit()


class TestWorkshopRunTraceHTTPContract:
    async def test_trace_access_is_denied_across_principals(self, tmp_path: Path):
        store, alice_id, alice_channel, bob_id, _ = await _open_store(tmp_path / "kai.db")
        submitter = _CommandSubmitter()
        client = await _open_command_client(
            store,
            _Authenticator({"alice-token": alice_id, "bob-token": bob_id}),
            submitter,
        )
        try:
            accepted = await client.post(
                f"/v1/channels/{alice_channel}/commands",
                headers={"Authorization": "Bearer alice-token"},
                json={"client_message_id": "browser-command-1", "body": "Private work"},
            )
            run_id = (await accepted.json())["run_id"]
            await _insert_trace_rows(store, run_id, 1)

            denied = await client.get(
                f"/v1/channels/{alice_channel}/runs/{run_id}/trace",
                headers={"Authorization": "Bearer bob-token"},
            )
            allowed = await client.get(
                f"/v1/channels/{alice_channel}/runs/{run_id}/trace",
                headers={"Authorization": "Bearer alice-token"},
            )

            assert denied.status == 403
            assert allowed.status == 200
            payload = await allowed.json()
            assert payload["run_id"] == run_id
            assert payload["channel_id"] == alice_channel
            assert [entry["seq"] for entry in payload["entries"]] == [1]
        finally:
            await client.close()
            await store.close()

    async def test_after_seq_paging_returns_disjoint_ordered_slices(self, tmp_path: Path):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        submitter = _CommandSubmitter()
        client = await _open_command_client(store, _Authenticator({"alice-token": alice_id}), submitter)
        try:
            accepted = await client.post(
                f"/v1/channels/{alice_channel}/commands",
                headers={"Authorization": "Bearer alice-token"},
                json={"client_message_id": "browser-command-1", "body": "Long task"},
            )
            run_id = (await accepted.json())["run_id"]
            await _insert_trace_rows(store, run_id, 201)
            headers = {"Authorization": "Bearer alice-token"}

            first = await client.get(f"/v1/channels/{alice_channel}/runs/{run_id}/trace", headers=headers)
            first_payload = await first.json()
            second = await client.get(
                f"/v1/channels/{alice_channel}/runs/{run_id}/trace",
                params={"after_seq": str(first_payload["entries"][-1]["seq"])},
                headers=headers,
            )
            second_payload = await second.json()

            assert first.status == 200
            assert [entry["seq"] for entry in first_payload["entries"]] == list(range(1, 201))
            assert first_payload["has_more"] is True
            assert second.status == 200
            assert [entry["seq"] for entry in second_payload["entries"]] == [201]
            assert second_payload["has_more"] is False
            assert second_payload["entries"][0]["summary"] == "summary 201"
        finally:
            await client.close()
            await store.close()

    async def test_truncation_marker_row_serves_as_stored(self, tmp_path: Path):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        submitter = _CommandSubmitter()
        client = await _open_command_client(store, _Authenticator({"alice-token": alice_id}), submitter)
        try:
            accepted = await client.post(
                f"/v1/channels/{alice_channel}/commands",
                headers={"Authorization": "Bearer alice-token"},
                json={"client_message_id": "browser-command-1", "body": "Long task"},
            )
            run_id = (await accepted.json())["run_id"]
            await _insert_trace_rows(store, run_id, 1)
            await store.connection.execute(
                "INSERT INTO run_traces (run_id, seq, kind, tool_name, tool_use_id, "
                "summary, detail, is_diff, is_error, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    2,
                    "truncated",
                    None,
                    None,
                    "trace truncated at 500 steps",
                    "",
                    0,
                    0,
                    "2026-08-11T14:00:01+00:00",
                ),
            )
            await store.connection.commit()

            response = await client.get(
                f"/v1/channels/{alice_channel}/runs/{run_id}/trace",
                headers={"Authorization": "Bearer alice-token"},
            )

            assert response.status == 200
            marker = (await response.json())["entries"][1]
            assert marker == {
                "seq": 2,
                "kind": "truncated",
                "tool_name": None,
                "tool_use_id": None,
                "summary": "trace truncated at 500 steps",
                "detail": "",
                "is_diff": False,
                "is_error": False,
                "created_at": "2026-08-11T14:00:01+00:00",
            }
        finally:
            await client.close()
            await store.close()


class TestWorkshopRunTraceEventStream:
    async def test_trace_doorbell_carries_no_id_and_advances_by_seq(self, tmp_path: Path):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        inbound = await record_inbound_message(
            store,
            InboundMessage(
                "telegram",
                "trace-update-1",
                "trace-message-1",
                "101",
                "101",
                "Perform one task",
                _NOW,
            ),
        )
        message_id = inbound.event.envelope.aggregate_id
        assert isinstance(message_id, MessageId)
        accepted = await WorkshopRunLifecycle(store).accept(message_id, occurred_at=_NOW + timedelta(seconds=1))
        run_id = str(accepted.run.run_id)
        # The doorbell targets the channel's started run: appends only
        # happen between start and settlement, so a queued accepted run
        # must never mask the executing one.
        await store.connection.execute(
            "UPDATE runs SET status = 'started', started_at = ? WHERE id = ?",
            ((_NOW + timedelta(seconds=2)).isoformat(), run_id),
        )
        await store.connection.commit()
        await _insert_trace_rows(store, run_id, 3)
        client = await _open_client(store, _Authenticator({"alice-token": alice_id}))
        response = None
        try:
            response = await client.get(
                f"/v1/channels/{alice_channel}/events",
                headers={"Authorization": "Bearer alice-token"},
            )
            event = await _read_sse_event(response)
            while event["event"] != "run.trace.updated":
                event = await _read_sse_event(response)

            assert event["id"] is None
            assert event["data"] == {
                "version": 1,
                "channel_id": alice_channel,
                "run_id": run_id,
                "seq": 3,
            }

            await _insert_trace_rows(store, run_id, 1, start=4)
            event = await _read_sse_event(response)
            while event["event"] != "run.trace.updated":
                event = await _read_sse_event(response)
            assert event["id"] is None
            assert event["data"]["seq"] == 4
        finally:
            if response is not None:
                response.close()
            await client.close()
            await store.close()

    async def test_invalid_trace_queries_return_bounded_errors(self, tmp_path: Path):
        store, alice_id, alice_channel, _, _ = await _open_store(tmp_path / "kai.db")
        submitter = _CommandSubmitter()
        client = await _open_command_client(store, _Authenticator({"alice-token": alice_id}), submitter)
        try:
            accepted = await client.post(
                f"/v1/channels/{alice_channel}/commands",
                headers={"Authorization": "Bearer alice-token"},
                json={"client_message_id": "browser-command-1", "body": "Long task"},
            )
            run_id = (await accepted.json())["run_id"]
            headers = {"Authorization": "Bearer alice-token"}
            path = f"/v1/channels/{alice_channel}/runs/{run_id}/trace"

            for query in ("after_seq=abc", "after_seq=-1", "after_seq=+5", "unknown=1", "after_seq=1&after_seq=2"):
                response = await client.get(f"{path}?{query}", headers=headers)
                assert response.status == 400, query
                assert (await response.json())["error"]["code"] == "invalid_request", query
        finally:
            await client.close()
            await store.close()


@pytest.mark.asyncio
async def test_github_settings_api_binds_principal_and_redacts_token(tmp_path: Path) -> None:
    store, alice_id, _, _, _ = await _open_store(tmp_path / "kai.db")
    service = _GitHubSettings(alice_id)
    client = await _open_client(
        store,
        _Authenticator({"alice-token": alice_id}),
        github_settings=service,
    )
    try:
        response = await client.get(
            "/v1/settings/github",
            headers={"Authorization": "Bearer alice-token"},
        )
        payload = await response.json()

        assert response.status == 200
        assert payload == {
            "version": 1,
            "github_login": "alice",
            "repositories_resettable": True,
            "repositories": [
                {
                    "repository": "owner/repo",
                    "source": "operator",
                    "automation_authorized": True,
                }
            ],
            "pr_review": {"enabled": True, "source": "operator", "resettable": False},
            "issue_triage": {"enabled": False, "source": "user", "resettable": True},
            "token_stored": True,
            "revision": "ghs_current",
            "mutation": None,
        }
        assert "secret" not in json.dumps(payload)
        assert service.calls[0][0] == "inspect"
    finally:
        await client.close()
        await store.close()


@pytest.mark.asyncio
async def test_github_settings_api_accepts_one_strict_write_at_a_time(tmp_path: Path) -> None:
    store, alice_id, _, _, _ = await _open_store(tmp_path / "kai.db")
    service = _GitHubSettings(alice_id)
    client = await _open_client(
        store,
        _Authenticator({"alice-token": alice_id}),
        github_settings=service,
    )
    headers = {"Authorization": "Bearer alice-token"}
    try:
        repository = await client.patch(
            "/v1/settings/github",
            headers=headers,
            json={
                "revision": "ghs_current",
                "repository": {"name": "other/repo", "subscribed": True},
            },
        )
        toggle = await client.patch(
            "/v1/settings/github",
            headers=headers,
            json={
                "revision": "ghs_current",
                "toggle": {"field": "issue_triage", "enabled": None},
            },
        )
        token = await client.patch(
            "/v1/settings/github",
            headers=headers,
            json={"revision": "ghs_current", "token": "new-secret"},
        )
        reset = await client.patch(
            "/v1/settings/github",
            headers=headers,
            json={"revision": "ghs_current", "reset_repositories": True},
        )

        assert repository.status == toggle.status == token.status == reset.status == 200
        assert service.calls[0][0] == "repository"
        assert service.calls[1][0] == "toggle"
        assert service.calls[2] == (
            "token",
            (GitHubSettingsAuthority(alice_id, profile_id(101)), "redacted", "ghs_current"),
        )
        assert service.calls[3] == (
            "repository_reset",
            (GitHubSettingsAuthority(alice_id, profile_id(101)), "ghs_current"),
        )
        assert "new-secret" not in json.dumps(await token.json())
    finally:
        await client.close()
        await store.close()


@pytest.mark.asyncio
async def test_github_settings_api_rejects_unauthorized_invalid_and_stale_writes(
    tmp_path: Path,
) -> None:
    store, alice_id, _, bob_id, _ = await _open_store(tmp_path / "kai.db")
    service = _GitHubSettings(alice_id)
    client = await _open_client(
        store,
        _Authenticator({"alice-token": alice_id, "bob-token": bob_id}),
        github_settings=service,
    )
    try:
        unauthorized = await client.get("/v1/settings/github")
        cross_principal = await client.get(
            "/v1/settings/github",
            headers={"Authorization": "Bearer bob-token"},
        )
        invalid = await client.patch(
            "/v1/settings/github",
            headers={"Authorization": "Bearer alice-token"},
            json={
                "revision": "ghs_current",
                "token": "new-secret",
                "toggle": {"field": "issue_triage", "enabled": True},
            },
        )
        stale = await client.patch(
            "/v1/settings/github",
            headers={"Authorization": "Bearer alice-token"},
            json={
                "revision": "ghs_stale",
                "toggle": {"field": "issue_triage", "enabled": True},
            },
        )

        assert unauthorized.status == 401
        assert cross_principal.status == 403
        assert invalid.status == 400
        assert stale.status == 409
    finally:
        await client.close()
        await store.close()


@pytest.mark.asyncio
async def test_notification_preferences_api_uses_opaque_authorized_choices(
    tmp_path: Path,
) -> None:
    store, alice_id, _, bob_id, _ = await _open_store(tmp_path / "kai.db")
    registry = await WorkshopExecutionStateRegistry.from_store(
        store,
        profile_registry(101, 202),
    )
    service = WorkshopNotificationPreferenceService(store.connection, registry)
    client = await _open_client(
        store,
        _Authenticator({"alice-token": alice_id, "bob-token": bob_id}),
        notification_preferences=service,
    )
    alice_headers = {"Authorization": "Bearer alice-token"}
    bob_headers = {"Authorization": "Bearer bob-token"}
    try:
        loaded = await client.get(
            "/v1/settings/notifications",
            headers=alice_headers,
        )
        payload = await loaded.json()

        assert loaded.status == 200
        assert payload["version"] == 1
        assert [item["kind"] for item in payload["destinations"]] == [
            "direct",
            "notification",
        ]
        assert all(item["choice_id"].startswith("ndst_") for item in payload["destinations"])
        encoded = json.dumps(payload)
        assert "-100123" not in encoded
        assert '"101"' not in encoded
        assert "telegram" not in encoded.lower()

        shared = next(item for item in payload["destinations"] if item["kind"] == "notification")
        selected = await client.patch(
            "/v1/settings/notifications",
            headers=alice_headers,
            json={
                "revision": payload["revision"],
                "integration_class": "github",
                "destination_choice_id": shared["choice_id"],
            },
        )
        selected_payload = await selected.json()

        assert selected.status == 200
        assert selected_payload["preferences"][0]["destination_name"] == "Notifications"
        assert selected_payload["preferences"][0]["source"] == "personal override"
        assert selected_payload["mutation"] == {
            "operation": "select_github_notification_destination",
            "changed": True,
        }

        stale = await client.patch(
            "/v1/settings/notifications",
            headers=alice_headers,
            json={
                "revision": payload["revision"],
                "integration_class": "github",
                "reset": True,
            },
        )
        forged = await client.patch(
            "/v1/settings/notifications",
            headers=bob_headers,
            json={
                "revision": (
                    await (
                        await client.get(
                            "/v1/settings/notifications",
                            headers=bob_headers,
                        )
                    ).json()
                )["revision"],
                "integration_class": "github",
                "destination_choice_id": shared["choice_id"],
            },
        )
        raw_authority = await client.patch(
            "/v1/settings/notifications",
            headers=alice_headers,
            json={
                "revision": selected_payload["revision"],
                "integration_class": "github",
                "channel_id": "-100123",
            },
        )

        assert stale.status == 409
        assert forged.status == 403
        assert raw_authority.status == 400
    finally:
        await client.close()
        await store.close()


@pytest.mark.asyncio
async def test_client_voice_preferences_api_is_binding_scoped_and_revision_checked(
    tmp_path: Path,
) -> None:
    store, alice_id, _, bob_id, _ = await _open_store(tmp_path / "kai.db")
    await store.connection.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    service = WorkshopClientPreferenceService(
        store.connection,
        (
            ClientVoiceCapability(
                "telegram",
                "Telegram",
                True,
                (("alan", "Alan"), ("jenny", "Jenny")),
                "alan",
            ),
        ),
    )
    await service.reconcile_legacy_preferences()
    client = await _open_client(
        store,
        _Authenticator({"alice-token": alice_id, "bob-token": bob_id}),
        client_preferences=service,
    )
    alice_headers = {"Authorization": "Bearer alice-token"}
    bob_headers = {"Authorization": "Bearer bob-token"}
    try:
        loaded = await client.get("/v1/settings/clients", headers=alice_headers)
        payload = await loaded.json()
        assert loaded.status == 200
        assert payload["voice_output"]["available"] is True
        assert payload["voice_output"]["bindings"][0]["client_name"] == "Telegram"
        assert payload["voice_output"]["bindings"][0]["choice_id"].startswith("cbd_")
        assert set(payload["voice_output"]["bindings"][0]) == {
            "choice_id",
            "client_name",
            "mode",
            "voice",
            "voice_name",
            "editable",
        }

        binding_choice = payload["voice_output"]["bindings"][0]["choice_id"]
        changed = await client.patch(
            "/v1/settings/clients",
            headers=alice_headers,
            json={
                "revision": payload["revision"],
                "binding_choice_id": binding_choice,
                "mode": "text_and_voice",
            },
        )
        changed_payload = await changed.json()
        assert changed.status == 200
        assert changed_payload["voice_output"]["bindings"][0]["mode"] == "text_and_voice"

        stale = await client.patch(
            "/v1/settings/clients",
            headers=alice_headers,
            json={
                "revision": payload["revision"],
                "binding_choice_id": binding_choice,
                "voice": "jenny",
            },
        )
        bob_loaded = await client.get("/v1/settings/clients", headers=bob_headers)
        bob_payload = await bob_loaded.json()
        forged = await client.patch(
            "/v1/settings/clients",
            headers=alice_headers,
            json={
                "revision": changed_payload["revision"],
                "binding_choice_id": bob_payload["voice_output"]["bindings"][0]["choice_id"],
                "voice": "jenny",
            },
        )
        assert stale.status == 409
        assert forged.status == 403
    finally:
        await client.close()
        await store.close()


@pytest.mark.asyncio
async def test_appearance_preferences_api_is_principal_scoped_and_revision_checked(
    tmp_path: Path,
) -> None:
    store, alice_id, _, bob_id, _ = await _open_store(tmp_path / "kai.db")
    service = WorkshopAppearancePreferenceService(store.connection)
    client = await _open_client(
        store,
        _Authenticator({"alice-token": alice_id, "bob-token": bob_id}),
        appearance_preferences=service,
    )
    alice_headers = {"Authorization": "Bearer alice-token"}
    bob_headers = {"Authorization": "Bearer bob-token"}
    try:
        loaded = await client.get("/v1/settings/appearance", headers=alice_headers)
        payload = await loaded.json()
        assert loaded.status == 200
        assert payload["theme_id"] == "atom-one-dark"
        assert payload["themes"] == [
            {
                "theme_id": item.theme_id,
                "display_name": item.display_name,
                "color_scheme": item.color_scheme,
            }
            for item in WORKSHOP_APPEARANCE_THEMES
        ]

        unchanged = await client.patch(
            "/v1/settings/appearance",
            headers=alice_headers,
            json={"revision": payload["revision"], "theme_id": "atom-one-dark"},
        )
        assert unchanged.status == 200
        assert (await unchanged.json())["mutation"] == {
            "operation": "set_theme",
            "changed": False,
        }

        invalid = await client.patch(
            "/v1/settings/appearance",
            headers=alice_headers,
            json={"revision": payload["revision"], "theme_id": "../../custom.css"},
        )
        stale_principal = await client.patch(
            "/v1/settings/appearance",
            headers=bob_headers,
            json={"revision": payload["revision"], "theme_id": "atom-one-dark"},
        )
        anonymous = await client.get("/v1/settings/appearance")
        assert invalid.status == 400
        assert stale_principal.status == 409
        assert anonymous.status == 401
        assert anonymous.headers["WWW-Authenticate"] == "Bearer"
    finally:
        await client.close()
        await store.close()
