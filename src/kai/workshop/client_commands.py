"""Authorized Workshop client command execution through canonical authority."""

from __future__ import annotations

from dataclasses import dataclass

from kai import sessions
from kai.config import Config
from kai.conversation_compatibility import reader_user, schedule_memory_ingestion
from kai.history import log_message
from kai.workshop.conversation_commands import (
    ClientConversationCommandAcceptance,
    ConversationCommandDisposition,
)
from kai.workshop.execution_coordinator import (
    CanonicalExecutionDisposition,
    CanonicalExecutionResult,
)
from kai.workshop.inbound import ClientInboundMessage
from kai.workshop.private_text_execution import WorkshopPrivateTextExecutionService


@dataclass(frozen=True, slots=True)
class ClientCommandResult:
    acceptance: ClientConversationCommandAcceptance
    execution: CanonicalExecutionResult


class WorkshopClientCommandExecutor:
    """Bridge an authorized client command into the existing protected lane."""

    def __init__(
        self,
        execution: WorkshopPrivateTextExecutionService,
        config: Config,
    ) -> None:
        self._execution = execution
        self._config = config

    async def submit(self, message: ClientInboundMessage) -> ClientCommandResult:
        accepted = await self._execution.accept_client(message)
        runtime_config_id = self._execution.runtime_config_id(accepted.runtime_profile_id)
        mapped_reader = reader_user(self._config, runtime_config_id)
        user_log = (
            log_message(
                direction="user",
                chat_id=runtime_config_id,
                text=message.body,
                reader_user=mapped_reader,
            )
            if accepted.command.disposition == ConversationCommandDisposition.NEWLY_ACCEPTED
            else None
        )

        result = await self._execution.execute(accepted.run.run_id)
        if result.terminal is not None:
            assistant_log = log_message(
                direction="assistant",
                chat_id=runtime_config_id,
                text=result.terminal.body,
                reader_user=mapped_reader,
            )
            if result.session_id and result.selection is not None:
                await sessions.save_session(runtime_config_id, result.session_id, result.selection.model)
            if result.disposition == CanonicalExecutionDisposition.COMPLETED and result.workspace is not None:
                schedule_memory_ingestion(
                    prompt=message.body,
                    assistant_text=result.terminal.body,
                    chat_id=runtime_config_id,
                    session_id=result.session_id,
                    config=self._config,
                    workspace=result.workspace,
                    user_log=user_log,
                    assistant_log=assistant_log,
                )
        return ClientCommandResult(accepted, result)
