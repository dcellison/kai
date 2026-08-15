"""Compatibility writes shared by canonical conversation ingress clients."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kai.config import ONESHOT_REASONER_BACKENDS, Config
from kai.history import LogEntry

if TYPE_CHECKING:
    from kai.workshop.domain import MessageId, RunId

log = logging.getLogger(__name__)

_pending_memory_tasks: set[asyncio.Task[None]] = set()


@dataclass(frozen=True, slots=True)
class CanonicalMemoryProvenance:
    """Canonical messages and run consumed by one memory extraction."""

    run_id: RunId
    source_message_id: MessageId
    result_message_id: MessageId

    def metadata(self) -> dict[str, object]:
        from kai import memory

        return {
            memory.WORKSHOP_RUN_ID_KEY: str(self.run_id),
            memory.WORKSHOP_SOURCE_MESSAGE_ID_KEY: str(self.source_message_id),
            memory.WORKSHOP_RESULT_MESSAGE_ID_KEY: str(self.result_message_id),
        }


def reader_user(config: Config, chat_id: int) -> str | None:
    """Return the mapped OS reader for one compatibility history tree."""
    user = config.get_user_config(chat_id)
    return user.os_user if user is not None else None


def schedule_memory_ingestion(
    *,
    prompt: str | list,
    assistant_text: str,
    chat_id: int,
    session_id: str | None,
    config: Config,
    workspace: str,
    user_log: LogEntry | None,
    assistant_log: LogEntry | None,
    canonical_provenance: CanonicalMemoryProvenance | None = None,
    canonical_prior_pairs: tuple[tuple[str, str], ...] | None = None,
    reasoner_backends: frozenset[str] = ONESHOT_REASONER_BACKENDS,
    effective_backend: str | None = None,
) -> None:
    """Preserve the existing fire-and-forget semantic-memory write path."""
    from kai.memory import is_enabled as memory_is_enabled

    if memory_is_enabled():

        async def ingest_memory() -> None:
            try:
                from kai import memory_extraction

                if isinstance(prompt, str):
                    user_text = prompt
                else:
                    user_text = next(
                        (block["text"] for block in prompt if block.get("type") == "text"),
                        "",
                    )
                if not user_text:
                    return
                user_config = config.get_user_config(chat_id)
                backend = effective_backend or (
                    user_config.backend if user_config and user_config.backend else config.default_backend
                )
                if config.memory_extraction_enabled and backend in reasoner_backends:
                    prior_pairs: list[tuple[str, str]] = []
                    if config.episode_classifier_context_turns > 0:
                        if canonical_provenance is not None:
                            # Canonical private text must never consult the
                            # compatibility JSONL transcript. Its caller
                            # supplies pairs strictly before the current
                            # inbound message from the Workshop timeline.
                            prior_pairs = list(canonical_prior_pairs or ())
                        else:
                            from kai.history import get_recent_pairs

                            fetched = get_recent_pairs(chat_id, config.episode_classifier_context_turns + 1)
                            prior_pairs = fetched[:-1]
                    await memory_extraction.extract_and_store(
                        user_text=user_text,
                        assistant_text=assistant_text,
                        user_id=str(chat_id),
                        session_id=session_id,
                        config=config,
                        prior_pairs=prior_pairs,
                        workspace=workspace,
                        user_log=user_log,
                        assistant_log=assistant_log,
                        canonical_provenance=(
                            canonical_provenance.metadata() if canonical_provenance is not None else None
                        ),
                    )
            except Exception:
                log.warning("Memory ingestion failed", exc_info=True)

        task = asyncio.create_task(ingest_memory())
        _pending_memory_tasks.add(task)
        task.add_done_callback(_pending_memory_tasks.discard)
