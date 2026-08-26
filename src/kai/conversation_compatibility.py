"""Compatibility writes shared by canonical conversation ingress clients."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from kai.config import ONESHOT_REASONER_BACKENDS, Config
from kai.history import LogEntry

# Type-only: Workshop modules import this module at runtime, so a runtime
# import of anything under kai.workshop from here would be circular.
if TYPE_CHECKING:
    from kai.workshop.domain import CanonicalMemoryProvenance

log = logging.getLogger(__name__)

_pending_memory_tasks: set[asyncio.Task[None]] = set()


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
        task = asyncio.create_task(
            ingest_conversation_memory(
                prompt=prompt,
                assistant_text=assistant_text,
                chat_id=chat_id,
                session_id=session_id,
                config=config,
                workspace=workspace,
                user_log=user_log,
                assistant_log=assistant_log,
                canonical_provenance=canonical_provenance,
                canonical_prior_pairs=canonical_prior_pairs,
                reasoner_backends=reasoner_backends,
                effective_backend=effective_backend,
            )
        )
        _pending_memory_tasks.add(task)
        task.add_done_callback(_pending_memory_tasks.discard)


async def ingest_conversation_memory(
    *,
    prompt: str | list,
    assistant_text: str,
    chat_id: int | None,
    session_id: str | None,
    config: Config,
    workspace: str,
    user_log: LogEntry | None,
    assistant_log: LogEntry | None,
    canonical_provenance: CanonicalMemoryProvenance | None = None,
    canonical_prior_pairs: tuple[tuple[str, str], ...] | None = None,
    reasoner_backends: frozenset[str] = ONESHOT_REASONER_BACKENDS,
    effective_backend: str | None = None,
    canonical_user_id: str | None = None,
    runtime_profile_id: str | None = None,
    os_user_override: str | None = None,
    effective_provider: str | None = None,
) -> None:
    """Run one memory ingestion to completion for a canonical owner.

    Legacy callers may still wrap this coroutine in a fire-and-forget task.
    The canonical post-run worker awaits it, including episode generation, so
    its durable receipt represents the complete transport-neutral effect.
    """
    from kai.memory import is_enabled as memory_is_enabled

    if memory_is_enabled():
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
            user_config = config.get_user_config(chat_id) if chat_id is not None else None
            backend = effective_backend or (
                user_config.backend if user_config and user_config.backend else config.default_backend
            )
            if config.memory_extraction_enabled and backend in reasoner_backends:
                prior_pairs: list[tuple[str, str]] = []
                if config.episode_classifier_context_turns > 0:
                    if canonical_provenance is not None:
                        # Canonical callers supply completed exchanges from the
                        # exact principal/channel/agent run lane. Never fall back
                        # to compatibility JSONL for this path.
                        prior_pairs = list(canonical_prior_pairs or ())
                    else:
                        from kai.history import get_recent_pairs

                        if chat_id is None:
                            raise RuntimeError("Canonical memory ingestion requires canonical prior pairs")
                        fetched = get_recent_pairs(chat_id, config.episode_classifier_context_turns + 1)
                        prior_pairs = fetched[:-1]
                await memory_extraction.extract_and_store(
                    user_text=user_text,
                    assistant_text=assistant_text,
                    user_id=canonical_user_id or str(chat_id),
                    session_id=session_id,
                    config=config,
                    prior_pairs=prior_pairs,
                    workspace=workspace,
                    user_log=user_log,
                    assistant_log=assistant_log,
                    canonical_provenance=(
                        canonical_provenance.metadata() if canonical_provenance is not None else None
                    ),
                    runtime_profile_id=runtime_profile_id,
                    os_user_override=os_user_override,
                    effective_backend_override=backend,
                    effective_provider_override=effective_provider,
                    await_episode=True,
                )
        except Exception:
            log.warning("Memory ingestion failed", exc_info=True)
