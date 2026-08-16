"""Ephemeral streaming previews for in-flight Workshop runs.

Previews are advisory display state only. They are never written to the
event store, carry no delivery authority, and vanish on process restart;
the durable run lifecycle and the canonical result message remain the
source of truth. The registry exists so the browser event stream can show
progressively growing assistant text for a run the executor is streaming
in the same process.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from kai.workshop.domain import ChannelId, RunId

# Previews for runs that never settle (a crashed task, an expired lease)
# must not linger forever in the reader's view. The TTL comfortably
# exceeds the coordinator's lease duration so a live, renewing run is
# never expired while it is still streaming.
_PREVIEW_TTL_SECONDS = 600.0


@dataclass(frozen=True, slots=True)
class RunPreview:
    """One run's newest publishable partial text."""

    run_id: RunId
    channel_id: ChannelId
    text: str
    sequence: int
    updated_at: float


class WorkshopRunPreviewRegistry:
    """In-memory publishable-preview state shared by writer and readers.

    The executor publishes from the run's owned execution path, so a
    superseded attempt cannot write here. Readers (the client event
    stream) look up by channel; lane serialization means at most one run
    per channel streams at a time.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._previews: dict[RunId, RunPreview] = {}
        # Registry-global monotonic sequence. A per-run counter would restart
        # at 1 whenever an entry is re-created after TTL expiry, and readers
        # that keep a per-run high-water mark would then discard every later
        # update; a process-global counter can never hand out a lower
        # sequence than one already delivered from this process.
        self._sequence = 0

    def publish(self, run_id: RunId, channel_id: ChannelId, text: str) -> None:
        """Record `text` as the run's newest preview if it changed."""
        current = self._previews.get(run_id)
        if current is not None and current.text == text:
            return
        self._sequence += 1
        self._previews[run_id] = RunPreview(
            run_id=run_id,
            channel_id=channel_id,
            text=text,
            sequence=self._sequence,
            updated_at=self._clock(),
        )

    def clear(self, run_id: RunId) -> None:
        """Drop the run's preview; called at terminal settlement."""
        self._previews.pop(run_id, None)

    def channel_preview(self, channel_id: ChannelId) -> RunPreview | None:
        """Return the channel's newest live preview, expiring stale ones."""
        now = self._clock()
        expired = [
            run_id for run_id, preview in self._previews.items() if now - preview.updated_at > _PREVIEW_TTL_SECONDS
        ]
        for run_id in expired:
            del self._previews[run_id]
        newest: RunPreview | None = None
        for preview in self._previews.values():
            if preview.channel_id != channel_id:
                continue
            if newest is None or preview.updated_at > newest.updated_at:
                newest = preview
        return newest
