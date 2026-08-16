"""Ephemeral run-preview registry and browser-lane observer wiring."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from kai.workshop.client_commands import WorkshopClientCommandExecutor
from kai.workshop.domain import ChannelId, RunId
from kai.workshop.execution_coordinator import (
    CanonicalExecutionDisposition,
    CanonicalExecutionResult,
)
from kai.workshop.run_previews import WorkshopRunPreviewRegistry


class TestRunPreviewRegistry:
    def test_publish_bumps_sequence_only_on_changed_text(self):
        registry = WorkshopRunPreviewRegistry()
        run_id = RunId.new()
        channel_id = ChannelId.new()

        registry.publish(run_id, channel_id, "First sentence.")
        registry.publish(run_id, channel_id, "First sentence.")
        first = registry.channel_preview(channel_id)
        registry.publish(run_id, channel_id, "First sentence. Second sentence.")
        second = registry.channel_preview(channel_id)

        assert first is not None and first.sequence == 1
        assert second is not None and second.sequence == 2
        assert second.text == "First sentence. Second sentence."

    def test_clear_removes_the_preview(self):
        registry = WorkshopRunPreviewRegistry()
        run_id = RunId.new()
        channel_id = ChannelId.new()
        registry.publish(run_id, channel_id, "Partial text.")

        registry.clear(run_id)

        assert registry.channel_preview(channel_id) is None

    def test_channel_lookup_is_isolated_and_expires_stale_entries(self):
        clock = SimpleNamespace(now=0.0)
        registry = WorkshopRunPreviewRegistry(clock=lambda: clock.now)
        stale_run = RunId.new()
        live_run = RunId.new()
        stale_channel = ChannelId.new()
        live_channel = ChannelId.new()

        registry.publish(stale_run, stale_channel, "Stale text.")
        clock.now = 700.0
        registry.publish(live_run, live_channel, "Live text.")

        live = registry.channel_preview(live_channel)
        assert live is not None and live.run_id == live_run
        assert registry.channel_preview(stale_channel) is None

    def test_sequence_never_restarts_after_expiry_or_clear(self):
        clock = SimpleNamespace(now=0.0)
        registry = WorkshopRunPreviewRegistry(clock=lambda: clock.now)
        run_id = RunId.new()
        channel_id = ChannelId.new()

        registry.publish(run_id, channel_id, "Before the quiet period.")
        before = registry.channel_preview(channel_id)
        # A long silent tool call outlives the TTL; the entry expires while
        # the run is still alive, then publishing resumes.
        clock.now = 700.0
        assert registry.channel_preview(channel_id) is None
        registry.publish(run_id, channel_id, "After the quiet period.")
        after = registry.channel_preview(channel_id)

        # A reader keeping a per-run high-water mark must never see the
        # resumed preview sort below the pre-expiry one.
        assert before is not None and after is not None
        assert after.sequence > before.sequence

    def test_newest_preview_wins_within_a_channel(self):
        clock = SimpleNamespace(now=0.0)
        registry = WorkshopRunPreviewRegistry(clock=lambda: clock.now)
        channel_id = ChannelId.new()
        older = RunId.new()
        newer = RunId.new()

        registry.publish(older, channel_id, "Older run text.")
        clock.now = 1.0
        registry.publish(newer, channel_id, "Newer run text.")

        preview = registry.channel_preview(channel_id)
        assert preview is not None and preview.run_id == newer


async def test_executor_publishes_stable_prefixes_and_clears_at_settlement():
    registry = WorkshopRunPreviewRegistry()
    run_id = RunId.new()
    channel_id = ChannelId.new()
    observed_during_run: list[str] = []

    async def execute(_run_id, *, stream_observer=None):
        assert stream_observer is not None
        # A dangling fragment has no publishable prefix and must not surface.
        await stream_observer(SimpleNamespace(text_so_far="One complete"))
        assert registry.channel_preview(channel_id) is None
        await stream_observer(SimpleNamespace(text_so_far="One complete sentence. And a partial"))
        preview = registry.channel_preview(channel_id)
        assert preview is not None
        observed_during_run.append(preview.text)
        return CanonicalExecutionResult(
            CanonicalExecutionDisposition.CANCELLED,
            SimpleNamespace(status="cancelled"),
        )

    execution = SimpleNamespace(
        execute=AsyncMock(side_effect=execute),
        run_state=AsyncMock(return_value=SimpleNamespace(channel_id=channel_id)),
        recoverable_client_runs=AsyncMock(return_value=()),
        request_run_cancellation=AsyncMock(),
    )
    executor = WorkshopClientCommandExecutor(
        execution,
        SimpleNamespace(for_profile=Mock()),
        run_previews=registry,
    )

    await executor.start()
    try:
        await executor._schedule(
            SimpleNamespace(run_id=run_id, runtime_profile_id=None, inbound_message_id=None, body="Hi", user_log=None)
        )
        await asyncio.sleep(0.05)
    finally:
        await executor.stop()

    assert observed_during_run == ["One complete sentence."]
    assert registry.channel_preview(channel_id) is None
