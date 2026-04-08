"""
Agent backend abstraction and shared context injection.

Provides:
1. AgentBackend ABC - the interface that pool.py programs against
2. AgentResponse / StreamEvent - backend-agnostic protocol types
3. ApiContext - groups webhook/service info for context injection
4. Context injection functions - build the identity/memory/history/API
   prefix that gets prepended to the first message of each session

The ABC defines the minimal surface that SubprocessPool needs. Concrete
backends (ClaudeCodeBackend, future GooseBackend, etc.) implement the
process management; the context functions here handle the Kai-specific
prompt assembly that is identical across all backends.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from kai.config import WorkspaceConfig
from kai.history import get_recent_history

log = logging.getLogger(__name__)


# ── Protocol types ──────────────────────────────────────────────────


@dataclass
class AgentResponse:
    """
    Final response from an agent backend interaction.

    Attributes:
        success: True if the backend returned a valid response, False on error.
        text: The full response text (accumulated from streaming chunks).
        session_id: Session identifier (used for session continuity).
        cost_usd: Cost of this interaction in USD.
        duration_ms: Wall-clock duration of the interaction in milliseconds.
        error: Error message if success is False, None otherwise.
    """

    success: bool
    text: str
    session_id: str | None = None
    cost_usd: float = 0.0
    duration_ms: int = 0
    error: str | None = None


@dataclass
class StreamEvent:
    """
    A partial update emitted during a backend's streaming response.

    Yielded by AgentBackend.send() as the backend generates text. The
    final event has done=True and includes the complete AgentResponse.

    Attributes:
        text_so_far: Accumulated response text up to this point.
        done: True if this is the final event (response complete or error).
        response: The complete AgentResponse, set only when done=True.
    """

    text_so_far: str
    done: bool = False
    response: AgentResponse | None = None


# ── API context ─────────────────────────────────────────────────────


@dataclass
class ApiContext:
    """
    Webhook and service proxy info for context injection.

    Groups the three API-related values that build_session_context()
    needs, keeping the function signature manageable.

    Attributes:
        webhook_port: Local port the webhook server listens on.
        webhook_secret: Shared secret for authenticating API requests.
        services_info: List of dicts describing available external services.
    """

    webhook_port: int
    webhook_secret: str
    services_info: list[dict] = field(default_factory=list)


# ── Abstract backend ────────────────────────────────────────────────


class AgentBackend(ABC):
    """
    Abstract base for agent subprocess backends.

    Backends manage a persistent subprocess that accepts prompts and
    streams responses. The pool (pool.py) owns backend instances and
    calls these methods; it does not care which agent harness is behind
    them.

    Mutable attributes on the ABC (set by pool.py during workspace
    restore, /settings, and /model commands). These are generic to any
    backend:
        model, workspace, home_workspace, max_budget_usd,
        timeout_seconds, max_context_window, workspace_config, provider

    Backend-specific attributes (NOT on the ABC) stay on the concrete
    class. For ClaudeCodeBackend these include: claude_user,
    max_session_hours, autocompact_pct. The pool never touches these
    directly; they are set at construction by _create_instance().
    """

    # These are plain instance attributes, not abstract properties,
    # because pool.py reads AND writes them directly (e.g.,
    # instance.model = "opus"). Abstract properties with setters are
    # verbose for no benefit here. Backends must initialize all of
    # these in __init__.
    model: str
    workspace: Path
    home_workspace: Path
    max_budget_usd: float
    timeout_seconds: int
    max_context_window: int
    workspace_config: WorkspaceConfig | None
    provider: str

    @abstractmethod
    async def send(self, prompt: str | list, chat_id: int | None = None) -> AsyncIterator[StreamEvent]:
        """Send a message and yield streaming events.

        Implementations should be async generators (use yield, not
        return). The AsyncIterator return type is the consumer-facing
        interface; pool.py iterates over the result, it does not send
        values into the generator.
        """
        ...

    @abstractmethod
    async def change_workspace(self, new_workspace: Path, workspace_config: WorkspaceConfig | None = None) -> None:
        """Switch working directory. Kills current process; next send() restarts."""
        ...

    @abstractmethod
    async def restart(self) -> None:
        """Kill process; next send() starts fresh."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Graceful shutdown with save-prompt."""
        ...

    @abstractmethod
    def force_kill(self) -> None:
        """Immediate kill, no cleanup. Safe without lock."""
        ...

    @property
    @abstractmethod
    def is_alive(self) -> bool:
        """True if the subprocess is running."""
        ...

    @property
    @abstractmethod
    def session_id(self) -> str | None:
        """Current session identifier."""
        ...


# ── Context injection functions ─────────────────────────────────────


def build_session_context(
    *,
    workspace: Path,
    home_workspace: Path,
    api: ApiContext,
    workspace_config: WorkspaceConfig | None,
    chat_id: int | None,
    data_dir: Path,
) -> str:
    """
    Build the context prefix for the first message of a new session.

    Always returns a non-empty string because the memory section
    unconditionally appends to parts (exists, empty, or missing).
    The caller prepends this to the prompt before sending.

    Extracted from claude.py _send_locked() lines 414-548.

    Args:
        workspace: The backend's current working directory.
        home_workspace: Kai's home workspace (identity + memory source).
        api: Webhook port, secret, and services info.
        workspace_config: Per-workspace config (for system_prompt).
        chat_id: Telegram chat ID for history scoping and API routing.
        data_dir: Root data directory (memory, history, files live here).
    """
    parts: list[str] = []

    # When in a foreign workspace, inject Kai's identity from home.
    # try/except guards against race (file deleted between exists()
    # and read_text()) and permission errors, matching the pattern
    # in get_workspace_system_prompt().
    if workspace != home_workspace:
        identity_path = home_workspace / ".claude" / "CLAUDE.md"
        try:
            identity = identity_path.read_text().strip()
            if identity:
                parts.append(f"[Your core identity and instructions:]\n{identity}")
        except OSError:
            pass

    # Always inject Kai's personal memory from DATA_DIR. This file
    # lives outside the install tree (/var/lib/kai/memory/ in production)
    # so it survives make install. Available regardless of which
    # workspace the inner Claude is operating in. try/except guards
    # against race and permission errors (same pattern as identity).
    memory_path = data_dir / "memory" / "MEMORY.md"
    try:
        memory = memory_path.read_text().strip()
        if memory:
            parts.append(f"[Your persistent memory (file: {memory_path}):]\n{memory}")
        else:
            parts.append(f"[Your persistent memory (file: {memory_path}):]\n(currently empty)")
    except OSError:
        parts.append(f"[Your persistent memory (file: {memory_path}):]\n(not yet created)")

    # Per-workspace system prompt from workspaces.yaml. Injected
    # between the identity/memory block and conversation history,
    # so it acts as workspace-specific instructions.
    ws_prompt = get_workspace_system_prompt(workspace_config)
    if ws_prompt:
        parts.append(f"## Workspace Instructions\n\n{ws_prompt}")

    # Always inject the per-user history directory path so the inner
    # Claude's grep/jq searches are naturally scoped to this user.
    history_dir = str(data_dir / "history" / str(chat_id)) if chat_id is not None else str(data_dir / "history")

    # Inject recent conversation history for continuity.
    # Filter by chat_id so each user's session only sees their
    # own messages (Phase 2 per-user data isolation).
    recent = get_recent_history(chat_id=chat_id)
    if recent:
        parts.append(f"[Recent conversations (search {history_dir}/ for full logs):]\n{recent}")
    else:
        parts.append(
            f"[Chat history is stored in {history_dir}/ as daily JSONL files. Search with grep or jq when asked about past conversations.]"
        )

    # Inject scheduling API info (always, so cron works from any workspace).
    # The secret is passed via $KAI_WEBHOOK_SECRET env var (not embedded
    # in prompt text) to prevent leakage through session logs.
    if api.webhook_secret:
        api_note = (
            f"[Scheduling API: To create jobs, use curl (NEVER WebFetch) to POST JSON to "
            f"http://localhost:{api.webhook_port}/api/schedule "
            f"with header 'X-Webhook-Secret: $KAI_WEBHOOK_SECRET' (environment variable). "
            f"Required fields: name, prompt, schedule_type, schedule_data. "
            f"Optional: job_type (reminder|claude), auto_remove (bool). "
            f"To list jobs: GET /api/jobs. To update: PATCH /api/jobs/{{id}}. "
            f"To delete: DELETE /api/jobs/{{id}}.]"
        )
        if workspace != home_workspace:
            api_note = (
                f"[Workspace context: You are working in {workspace}. "
                f"Your home workspace is {home_workspace}.]\n{api_note}"
            )
        parts.append(api_note)

    # Inject messaging and file exchange API info so Claude can
    # proactively send text or files to the user (e.g., when a
    # background task completes or a scheduled job has results).
    if api.webhook_secret:
        parts.append(
            f"[Messaging API: To send a text message to the user proactively "
            f"(e.g., background task results), use curl (NEVER WebFetch) to POST JSON to "
            f"http://localhost:{api.webhook_port}/api/send-message "
            f"with header 'X-Webhook-Secret: $KAI_WEBHOOK_SECRET' (environment variable). "
            f'Required: "text" (the message content). '
            f"Long messages are automatically split at Telegram's 4096-char limit.]"
        )
        files_path = f"{data_dir}/files/{chat_id}/" if chat_id else f"{data_dir}/files/"
        parts.append(
            f"[File API: To send a file to the user, use curl (NEVER WebFetch) to POST JSON to "
            f"http://localhost:{api.webhook_port}/api/send-file "
            f"with header 'X-Webhook-Secret: $KAI_WEBHOOK_SECRET' (environment variable). "
            f'Required: "path" (absolute file path within the current workspace {workspace}). '
            f'Optional: "caption". Images are sent as photos, '
            f"everything else as documents.\n"
            f"Incoming files from the user are auto-saved to "
            f"{files_path} and their paths are included in the message.]"
        )

    # Inject available external services info (only if services are configured)
    if api.services_info and api.webhook_secret:
        svc_lines = [
            "[External Services: To call external APIs, use curl (NEVER WebFetch) to POST JSON to "
            f"http://localhost:{api.webhook_port}/api/services/{{name}} "
            f"with header 'X-Webhook-Secret: $KAI_WEBHOOK_SECRET' (environment variable). "
            "Request JSON fields (all optional): "
            '"body" (dict - forwarded as JSON), '
            '"params" (dict - query parameters), '
            '"path_suffix" (str - appended to base URL).',
            "",
            "Available services:",
        ]
        for svc in api.services_info:
            svc_lines.append(f"  - {svc['name']} ({svc['method']}): {svc['description']}")
            if svc.get("notes"):
                svc_lines.append(f"    Notes: {svc['notes']}")
        svc_lines.append("")
        svc_lines.append(
            "Example (Perplexity web search):\n"
            f"  curl -s -X POST http://localhost:{api.webhook_port}/api/services/perplexity "
            f"-H 'Content-Type: application/json' "
            f"""-H "X-Webhook-Secret: $KAI_WEBHOOK_SECRET" """
            """-d '{"body": {"model": "sonar", "messages": [{"role": "user", "content": "your query"}]}}'"""
        )
        svc_lines.append(
            "Prefer external services over built-in WebSearch/WebFetch when available - they provide better results.]"
        )
        parts.append("\n".join(svc_lines))

    # Include chat_id so inner Claude can pass it back in API
    # calls for correct multi-user routing. Without this, all
    # API calls route to the default admin user.
    if chat_id is not None:
        parts.append(
            f"[Your chat_id for API calls: {chat_id}. Include "
            f'"chat_id": {chat_id} in the JSON body of all '
            f"POST requests to /api/schedule, /api/send-message, "
            f"and /api/send-file so responses route to the "
            f"correct user.]"
        )

    # No trailing \n\n here - prepend_to_prompt() adds the separator
    # between the context block and the user's message.
    return "\n\n".join(parts)


def build_foreign_workspace_reminder(workspace: Path, home_workspace: Path) -> str | None:
    """
    Build the per-message reminder for foreign workspaces.

    Returns the reminder string if workspace != home_workspace, else
    None. Applied on EVERY message, not just the first.

    Extracted from claude.py _send_locked() lines 557-569.
    """
    if workspace == home_workspace:
        return None
    return (
        "[IMPORTANT: This message is from a user via Telegram. "
        "Respond ONLY to what they wrote below. Do NOT continue, "
        "resume, or start any previous work, plans, or tasks.]"
    )


def prepend_to_prompt(prompt: str | list, prefix: str) -> str | list:
    """
    Prepend a text prefix to a prompt.

    Handles both str prompts and list-of-content-blocks prompts.
    No-ops if prefix is empty.
    """
    if not prefix:
        return prompt
    if isinstance(prompt, str):
        # String prompts get a \n\n separator between prefix and prompt.
        return prefix + "\n\n" + prompt
    # List prompts: insert a separate text block at the front. No \n\n
    # needed because content blocks are independent elements in the API.
    return [{"type": "text", "text": prefix}] + prompt


def get_workspace_system_prompt(
    workspace_config: WorkspaceConfig | None,
) -> str | None:
    """
    Read the system prompt for a workspace config.

    Returns the inline system_prompt if set, reads from
    system_prompt_file on each call to pick up changes, or returns
    None if neither is configured.
    """
    if not workspace_config:
        return None
    if workspace_config.system_prompt:
        return workspace_config.system_prompt
    if workspace_config.system_prompt_file:
        # File path was validated at config load time (fail-fast on typos).
        # Read content here so updates are picked up without restart.
        try:
            return workspace_config.system_prompt_file.read_text()
        except OSError:
            log.warning("Cannot read system_prompt_file: %s", workspace_config.system_prompt_file)
            return None
    return None
