"""Authorized Workshop client command execution through canonical authority."""

from __future__ import annotations

from dataclasses import dataclass

from kai.workshop.compatibility_state import WorkshopCompatibilityStateWriter
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
        compatibility_state: WorkshopCompatibilityStateWriter,
    ) -> None:
        self._execution = execution
        self._compatibility_state = compatibility_state

    async def submit(self, message: ClientInboundMessage) -> ClientCommandResult:
        accepted = await self._execution.accept_client(message)
        compatibility_state = self._compatibility_state.for_profile(accepted.runtime_profile_id)
        user_log = (
            compatibility_state.append_history(
                direction="user",
                text=message.body,
            )
            if accepted.command.disposition == ConversationCommandDisposition.NEWLY_ACCEPTED
            else None
        )

        result = await self._execution.execute(accepted.run.run_id)
        if result.terminal is not None:
            assistant_log = compatibility_state.append_history(
                direction="assistant",
                text=result.terminal.body,
            )
            if result.session_id and result.selection is not None:
                await compatibility_state.save_session(
                    result.session_id,
                    result.selection.model,
                )
            if result.disposition == CanonicalExecutionDisposition.COMPLETED and result.workspace is not None:
                compatibility_state.schedule_memory_ingestion(
                    prompt=message.body,
                    assistant_text=result.terminal.body,
                    session_id=result.session_id,
                    workspace=result.workspace,
                    user_log=user_log,
                    assistant_log=assistant_log,
                )
        return ClientCommandResult(accepted, result)
