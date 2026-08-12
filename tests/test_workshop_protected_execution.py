"""Contracts for production-unused protected Workshop execution preparation."""

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
from kai.workshop.run_execution_authority import RunExecutionSelection, WorkshopRunExecutionAuthority
from kai.workshop.store import WorkshopEventStore

_NOW = datetime(2026, 8, 12, 21, 0, tzinfo=UTC)


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
    return store, accepted.run, SubprocessPool(config=config, services_info=[])


class TestProtectedExecutionPreparation:
    async def test_resolves_effective_registered_selection_and_hides_transport_key(self, tmp_path: Path):
        home = tmp_path / "home"
        home.mkdir()
        store, run, pool = await _accepted_run(tmp_path / "kai.db", home)
        try:
            with (
                patch("kai.pool.sessions.get_setting", new_callable=AsyncMock, return_value=None),
                patch("kai.pool.sessions.get_user_settings", new_callable=AsyncMock, return_value={}),
            ):
                prepared = await WorkshopProtectedExecutionPreparationService(
                    store,
                    pool,
                    registered_backend_ids=frozenset({"claude"}),
                ).prepare(run.run_id)

            assert prepared.run == run
            assert prepared.selection.backend == "claude"
            assert prepared.selection.provider == "anthropic"
            assert prepared.selection.model == "sonnet"
            assert prepared.workspace == home
            assert "101" not in repr(prepared)
            assert not hasattr(prepared, "chat_id")
        finally:
            await pool.shutdown()
            await store.close()

    async def test_unregistered_effective_backend_fails_closed(self, tmp_path: Path):
        home = tmp_path / "home"
        home.mkdir()
        store, run, pool = await _accepted_run(tmp_path / "kai.db", home)
        try:
            with (
                patch("kai.pool.sessions.get_setting", new_callable=AsyncMock, return_value=None),
                patch("kai.pool.sessions.get_user_settings", new_callable=AsyncMock, return_value={}),
                pytest.raises(ProtectedExecutionPreparationError, match="protected registry"),
            ):
                await WorkshopProtectedExecutionPreparationService(
                    store,
                    pool,
                    registered_backend_ids=frozenset({"codex"}),
                ).prepare(run.run_id)
        finally:
            await pool.shutdown()
            await store.close()

    async def test_cancellation_pending_run_never_prepares_a_runtime(self, tmp_path: Path):
        home = tmp_path / "home"
        home.mkdir()
        store, run, pool = await _accepted_run(tmp_path / "kai.db", home)
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
                    pool,
                    registered_backend_ids=frozenset({"claude"}),
                ).prepare(run.run_id)
            prepare.assert_not_awaited()
        finally:
            await pool.shutdown()
            await store.close()

    def test_service_remains_unregistered(self):
        source_root = Path(__file__).parents[1] / "src" / "kai"
        for relative_path in ("main.py", "bot.py", "sessions.py"):
            source = (source_root / relative_path).read_text(encoding="utf-8")
            assert "WorkshopProtectedExecutionPreparationService" not in source
