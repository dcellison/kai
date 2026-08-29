"""Contracts for protected Workshop execution preparation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from kai.config import Config, UserConfig
from kai.pool import SubprocessPool
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.conversation_commands import WorkshopConversationCommandService
from kai.workshop.inbound import InboundMessage
from kai.workshop.protected_execution import (
    ProtectedExecutionPreparationError,
    WorkshopProtectedExecutionPreparationService,
)
from kai.workshop.routing_policy import (
    RoutingDecisionDisposition,
    RunRoutingDecision,
)
from kai.workshop.run_execution_authority import RunExecutionSelection, WorkshopRunExecutionAuthority
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id, profile_registry

_NOW = datetime(2026, 8, 12, 21, 0, tzinfo=UTC)


class _RoutingPolicy:
    async def decide_for_run(self, run, runtime_profile_id):
        return RunRoutingDecision(
            run_id=run.run_id,
            runtime_profile_id=runtime_profile_id,
            requested_task_class=None,
            requested_backend_option_id=None,
            selected_backend_option_id="codex:openai",
            disposition=RoutingDecisionDisposition.SELECTED_DEFAULT,
            reason_code="task_class_not_requested",
            policy_revision=None,
            selection=RunExecutionSelection("codex", "gpt-5.6-sol", "openai"),
            evidence_version=1,
            decided_at=_NOW,
        )


async def _accepted_run(path: Path, home: Path):
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                display_name="Workshop Human",
                role="admin",
                transport="telegram",
                external_subject="101",
                external_channel_id="101",
                runtime_profile_id=profile_id(101),
            ),
        ),
    )
    accepted = await WorkshopConversationCommandService(store).accept(
        InboundMessage(
            transport="telegram",
            update_id="command-1",
            message_id="message-1",
            sender_subject="101",
            channel_subject="101",
            body="Prepare one protected execution",
            occurred_at=_NOW,
        )
    )
    config = Config(
        telegram_bot_token="test",
        allowed_user_ids={101},
        default_backend="codex",
        default_provider="openai",
        default_model="gpt-5.5",
        default_timeout=30,
        agent_max_session_hours=0,
        agent_idle_timeout=1800,
        webhook_port=8080,
        user_configs={
            101: UserConfig(
                telegram_id=101,
                name="human",
                backend="claude",
                model="sonnet",
                home_workspace=home,
            )
        },
    )
    profiles = profile_registry(101)
    return (
        store,
        accepted.run,
        SubprocessPool(config=config, services_info=[], runtime_profiles=profiles),
        profiles,
    )


class TestProtectedExecutionPreparation:
    async def test_resolves_effective_registered_selection_and_hides_transport_key(self, tmp_path: Path):
        home = tmp_path / "home"
        home.mkdir()
        store, run, pool, profiles = await _accepted_run(tmp_path / "kai.db", home)
        try:
            with (
                patch(
                    "kai.pool.sessions.get_canonical_execution_settings",
                    new_callable=AsyncMock,
                    return_value={},
                ),
            ):
                prepared = await WorkshopProtectedExecutionPreparationService(
                    store,
                    WorkshopRuntimePool(pool, profiles),
                    _RoutingPolicy(),  # type: ignore[arg-type]
                    registered_backend_ids=frozenset({"codex"}),
                ).prepare(run.run_id)

            assert prepared.run == run
            assert prepared.selection.backend == "codex"
            assert prepared.selection.provider == "openai"
            assert prepared.selection.model == "gpt-5.6-sol"
            assert prepared.workspace == pool.get_home_workspace(profile_id(101))
            assert "_runtime" not in repr(prepared)
            assert not hasattr(prepared, "chat_id")
        finally:
            await pool.shutdown()
            await store.close()

    async def test_unregistered_effective_backend_fails_closed(self, tmp_path: Path):
        home = tmp_path / "home"
        home.mkdir()
        store, run, pool, profiles = await _accepted_run(tmp_path / "kai.db", home)
        try:
            with (
                patch(
                    "kai.pool.sessions.get_canonical_execution_settings",
                    new_callable=AsyncMock,
                    return_value={},
                ),
                pytest.raises(ProtectedExecutionPreparationError, match="protected registry"),
            ):
                await WorkshopProtectedExecutionPreparationService(
                    store,
                    WorkshopRuntimePool(pool, profiles),
                    _RoutingPolicy(),  # type: ignore[arg-type]
                    registered_backend_ids=frozenset({"claude"}),
                ).prepare(run.run_id)
        finally:
            await pool.shutdown()
            await store.close()

    async def test_cancellation_pending_run_never_prepares_a_runtime(self, tmp_path: Path):
        home = tmp_path / "home"
        home.mkdir()
        store, run, pool, profiles = await _accepted_run(tmp_path / "kai.db", home)
        try:
            authority = WorkshopRunExecutionAuthority(
                store,
                selection_resolver=lambda _run: RunExecutionSelection("claude", "sonnet", "anthropic"),
                registered_backend_ids=frozenset({"claude"}),
            )
            await authority.request_cancellation(
                run.run_id,
                cancellation_code="requested_by_human",
                occurred_at=_NOW,
            )
            with (
                patch.object(pool, "prepare_execution", new_callable=AsyncMock) as prepare,
                pytest.raises(ProtectedExecutionPreparationError, match="uncancelled accepted"),
            ):
                await WorkshopProtectedExecutionPreparationService(
                    store,
                    WorkshopRuntimePool(pool, profiles),
                    _RoutingPolicy(),  # type: ignore[arg-type]
                    registered_backend_ids=frozenset({"claude"}),
                ).prepare(run.run_id)
            prepare.assert_not_awaited()
        finally:
            await pool.shutdown()
            await store.close()

    def test_service_is_registered_only_through_private_text_runtime_owner(self):
        source_root = Path(__file__).parents[1] / "src" / "kai"
        main_source = (source_root / "main.py").read_text(encoding="utf-8")
        host_source = (source_root / "application_host.py").read_text(encoding="utf-8")
        owner_source = (source_root / "workshop" / "private_text_execution.py").read_text(encoding="utf-8")
        assert "WorkshopPrivateTextExecutionService.open_and_start" in host_source
        assert "WorkshopPrivateTextExecutionService.open_and_start" not in main_source
        assert "WorkshopProtectedExecutionPreparationService" in owner_source
        assert "WorkshopProtectedExecutionPreparationService" not in (source_root / "bot.py").read_text(
            encoding="utf-8"
        )
