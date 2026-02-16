"""
Persistent Claude Code subprocess manager.

Provides functionality to:
1. Manage a long-running Claude Code subprocess with stream-json I/O
2. Inject identity, memory, history, and API context on each new session
3. Stream partial responses for real-time Telegram message updates
4. Handle workspace switching, model changes, and graceful shutdown

This is the core bridge between Telegram (bot.py) and Claude Code. Instead of
launching a new Claude process per message, a single persistent process is kept
alive and communicated with via newline-delimited JSON on stdin/stdout. This
preserves Claude's conversation context across messages within a session.

The stream-json protocol:
    Input:  {"type": "user", "message": {"role": "user", "content": [...]}}
    Output: {"type": "system", ...}      — session metadata
            {"type": "assistant", ...}   — partial text (streaming)
            {"type": "result", ...}      — final response with cost/session info

Context injection on first message of each session:
    1. Identity (CLAUDE.md from home workspace, when in a foreign workspace)
    2. Personal memory (MEMORY.md from home workspace)
    3. Workspace memory (MEMORY.md from current workspace, if different from home)
    4. Recent conversation history (last 20 messages from JSONL logs)
    5. Scheduling API endpoint info (URL, secret, field reference)
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from kai.history import get_recent_history

log = logging.getLogger(__name__)


@dataclass
class ClaudeResponse:
    """
    Final response from a Claude Code interaction.

    Attributes:
        success: True if Claude returned a valid response, False on error.
        text: The full response text (accumulated from streaming chunks).
        session_id: Claude's session identifier (used for session continuity).
        cost_usd: Cost of this interaction in USD (from Claude's billing).
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
    A partial update emitted during Claude's streaming response.

    Yielded by PersistentClaude.send() as Claude generates text. The final
    event has done=True and includes the complete ClaudeResponse.

    Attributes:
        text_so_far: Accumulated response text up to this point.
        done: True if this is the final event (response complete or error).
        response: The complete ClaudeResponse, set only when done=True.
    """

    text_so_far: str
    done: bool = False
    response: ClaudeResponse | None = None


class PersistentClaude:
    """
    A long-running Claude Code subprocess using stream-json I/O for multi-turn chat.

    Manages the lifecycle of the Claude process: starting, sending messages,
    streaming responses, killing/restarting, and workspace switching. All message
    sends are serialized via an internal asyncio lock to prevent interleaving.

    The process runs with --permission-mode bypassPermissions (required for headless
    operation via Telegram) and --max-budget-usd to cap per-session spending.
    """

    def __init__(
        self,
        *,
        model: str = "sonnet",
        workspace: Path = Path("workspace"),
        home_workspace: Path | None = None,
        webhook_port: int = 8080,
        webhook_secret: str = "",
        max_budget_usd: float = 1.0,
        timeout_seconds: int = 120,
        services_info: list[dict] | None = None,
    ):
        self.model = model
        self.workspace = workspace
        self.home_workspace = home_workspace or workspace
        self.webhook_port = webhook_port
        self.webhook_secret = webhook_secret
        self.max_budget_usd = max_budget_usd
        self.timeout_seconds = timeout_seconds
        self.services_info = services_info or []
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()  # Serializes all message sends
        self._session_id: str | None = None
        self._fresh_session = True  # True until the first message is sent
        self._stderr_task: asyncio.Task | None = None  # Background stderr drain

    @property
    def is_alive(self) -> bool:
        """True if the Claude subprocess is running and hasn't exited."""
        return self._proc is not None and self._proc.returncode is None

    @property
    def session_id(self) -> str | None:
        """The current Claude session ID, or None if no session is active."""
        return self._session_id

    async def _ensure_started(self) -> None:
        """
        Start the Claude Code subprocess if not already running.

        Launches claude with stream-json I/O, bypassPermissions mode (required
        for headless operation), and the configured model and budget. The process
        runs in the current workspace directory and persists across messages.

        The stdout buffer limit is raised to 1 MiB (from the default 64 KiB)
        because large tool results from Claude can exceed the default.
        """
        if self.is_alive:
            return

        cmd = [
            "claude",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            self.model,
            "--permission-mode",
            "bypassPermissions",
            "--max-budget-usd",
            str(self.max_budget_usd),
        ]
        log.info("Starting persistent Claude process (model=%s)", self.model)

        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workspace),
            limit=1024 * 1024,  # 1 MiB; default 64 KiB too small for large tool results
        )
        self._session_id = None
        self._fresh_session = True

        # Drain stderr in background to prevent pipe buffer deadlock
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        """
        Continuously read and discard stderr from the Claude process.

        Without this, the stderr pipe buffer fills up and the process deadlocks.
        Lines are logged at DEBUG level (truncated to 200 chars) for diagnostics.
        """
        while self._proc and self._proc.stderr:
            try:
                line = await self._proc.stderr.readline()
                if not line:
                    break
                text = line.decode().strip()
                if text:
                    log.debug("Claude stderr: %s", text[:200])
            except Exception:
                break

    async def send(self, prompt: str | list) -> AsyncIterator[StreamEvent]:
        """
        Send a message to Claude and yield streaming events.

        This is the main public interface. All sends are serialized via an
        internal lock so concurrent callers (e.g., a user message arriving
        while a cron job is running) queue rather than interleave.

        Args:
            prompt: Either a text string or a list of content blocks (for
                multi-modal messages like images).

        Yields:
            StreamEvent objects with accumulated text. The final event has
            done=True and includes the complete ClaudeResponse.
        """
        async with self._lock:
            async for event in self._send_locked(prompt):
                yield event

    async def _send_locked(self, prompt: str | list) -> AsyncIterator[StreamEvent]:
        """
        Core message-sending logic (must be called while holding self._lock).

        Handles the full lifecycle of a single Claude interaction:
        1. Ensure the subprocess is alive (start if needed)
        2. On the first message of a new session, prepend identity, memory,
           conversation history, and scheduling API context to the prompt
        3. When in a foreign workspace, prepend a per-message reminder to
           prevent Claude from acting on workspace context autonomously
        4. Write the JSON message to stdin and stream stdout line-by-line
        5. Parse stream-json events and yield StreamEvents to the caller

        Args:
            prompt: Either a text string or a list of content blocks (for
                multi-modal messages like images).

        Yields:
            StreamEvent objects with accumulated text. The final event has
            done=True and includes the complete ClaudeResponse.
        """
        try:
            await self._ensure_started()
        except FileNotFoundError:
            yield StreamEvent(
                text_so_far="",
                done=True,
                response=ClaudeResponse(success=False, text="", error="claude CLI not found"),
            )
            return

        # Inject identity and memory on the first message of a new session
        if self._fresh_session:
            self._fresh_session = False
            parts = []

            # When in a foreign workspace, inject Kai's identity from home
            if self.workspace != self.home_workspace:
                identity_path = self.home_workspace / ".claude" / "CLAUDE.md"
                if identity_path.exists():
                    identity = identity_path.read_text().strip()
                    if identity:
                        parts.append(f"[Your core identity and instructions:]\n{identity}")

            # Always inject Kai's personal memory from home workspace
            memory_path = self.home_workspace / ".claude" / "MEMORY.md"
            if memory_path.exists():
                memory = memory_path.read_text().strip()
                if memory:
                    parts.append(f"[Your persistent memory from home workspace:]\n{memory}")

            # Inject current workspace memory if different from home
            if self.workspace != self.home_workspace:
                ws_memory_path = self.workspace / ".claude" / "MEMORY.md"
                if ws_memory_path.exists():
                    ws_memory = ws_memory_path.read_text().strip()
                    if ws_memory:
                        parts.append(f"[Your memory for this workspace ({self.workspace.name}):]\n{ws_memory}")

            # Inject recent conversation history for continuity
            recent = get_recent_history()
            if recent:
                parts.append(f"[Recent conversations (search .claude/history/ for full logs):]\n{recent}")

            # Inject scheduling API info (always, so cron works from any workspace)
            if self.webhook_secret:
                api_note = (
                    f"[Scheduling API: To schedule jobs, POST JSON to "
                    f"http://localhost:{self.webhook_port}/api/schedule "
                    f"with header 'X-Webhook-Secret: {self.webhook_secret}'. "
                    f"Required fields: name, prompt, schedule_type, schedule_data. "
                    f"Optional: job_type (reminder|claude), auto_remove (bool).]"
                )
                if self.workspace != self.home_workspace:
                    api_note = (
                        f"[Workspace context: You are working in {self.workspace}. "
                        f"Your home workspace is {self.home_workspace}.]\n{api_note}"
                    )
                parts.append(api_note)

            # Inject available external services info (only if services are configured)
            if self.services_info and self.webhook_secret:
                svc_lines = [
                    "[External Services: To call external APIs, POST JSON to "
                    f"http://localhost:{self.webhook_port}/api/services/{{name}} "
                    f"with header 'X-Webhook-Secret: {self.webhook_secret}'. "
                    "Request JSON fields (all optional): "
                    '"body" (dict — forwarded as JSON), '
                    '"params" (dict — query parameters), '
                    '"path_suffix" (str — appended to base URL).',
                    "",
                    "Available services:",
                ]
                for svc in self.services_info:
                    svc_lines.append(f"  - {svc['name']} ({svc['method']}): {svc['description']}")
                    if svc.get("notes"):
                        svc_lines.append(f"    Notes: {svc['notes']}")
                svc_lines.append("")
                svc_lines.append(
                    "Example (Perplexity web search):\n"
                    f"  curl -s -X POST http://localhost:{self.webhook_port}/api/services/perplexity "
                    f"-H 'Content-Type: application/json' "
                    f"-H 'X-Webhook-Secret: {self.webhook_secret}' "
                    """-d '{"body": {"model": "sonar", "messages": [{"role": "user", "content": "your query"}]}}'"""
                )
                svc_lines.append(
                    "Prefer external services over built-in WebSearch/WebFetch when available "
                    "— they provide better results.]"
                )
                parts.append("\n".join(svc_lines))

            if parts:
                prefix = "\n\n".join(parts) + "\n\n"
                if isinstance(prompt, str):
                    prompt = prefix + prompt
                elif isinstance(prompt, list):
                    prompt = [{"type": "text", "text": prefix}] + prompt

        # When in a foreign workspace, remind on every message to only respond
        # to what the user asks — workspace context (CLAUDE.md, git branch,
        # auto-memory) can otherwise trigger autonomous action.
        if self.workspace != self.home_workspace:
            reminder = (
                "[IMPORTANT: This message is from a user via Telegram. "
                "Respond ONLY to what they wrote below. Do NOT continue, "
                "resume, or start any previous work, plans, or tasks.]"
            )
            if isinstance(prompt, str):
                prompt = reminder + "\n\n" + prompt
            elif isinstance(prompt, list):
                prompt = [{"type": "text", "text": reminder}] + prompt

        content = prompt if isinstance(prompt, list) else [{"type": "text", "text": prompt}]
        msg = (
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": content,
                    },
                }
            )
            + "\n"
        )

        assert self._proc is not None
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        try:
            self._proc.stdin.write(msg.encode())
            await self._proc.stdin.drain()
        except OSError as e:
            log.error("Failed to write to Claude process: %s", e)
            await self._kill()
            yield StreamEvent(
                text_so_far="",
                done=True,
                response=ClaudeResponse(
                    success=False, text="", error="Claude process died, restarting on next message"
                ),
            )
            return

        accumulated_text = ""
        try:
            while True:
                try:
                    # Opus with tool use can go minutes between output lines
                    timeout = self.timeout_seconds * 3
                    line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=timeout)
                except TimeoutError:
                    log.error("Claude response timed out")
                    await self._kill()
                    yield StreamEvent(
                        text_so_far=accumulated_text,
                        done=True,
                        response=ClaudeResponse(success=False, text=accumulated_text, error="Claude timed out"),
                    )
                    return

                if not line:
                    # Process died unexpectedly
                    log.error("Claude process EOF")
                    await self._kill()
                    yield StreamEvent(
                        text_so_far=accumulated_text,
                        done=True,
                        response=ClaudeResponse(
                            success=bool(accumulated_text),
                            text=accumulated_text,
                            error=None if accumulated_text else "Claude process ended unexpectedly",
                        ),
                    )
                    return

                try:
                    event = json.loads(line.decode())
                except json.JSONDecodeError:
                    continue

                etype = event.get("type")

                if etype == "system":
                    sid = event.get("session_id")
                    if sid:
                        self._session_id = sid

                elif etype == "result":
                    # The result event's text may only contain the final
                    # assistant message; accumulated_text has everything
                    # (including text before tool use).
                    result_text = event.get("result", "")
                    text = accumulated_text if len(accumulated_text) > len(result_text) else result_text
                    response = ClaudeResponse(
                        success=not event.get("is_error", False),
                        text=text,
                        session_id=event.get("session_id", self._session_id),
                        cost_usd=event.get("total_cost_usd", 0.0),
                        duration_ms=event.get("duration_ms", 0),
                        error=event.get("result") if event.get("is_error") else None,
                    )
                    yield StreamEvent(text_so_far=response.text, done=True, response=response)
                    return

                elif etype == "assistant" and "message" in event:
                    msg_data = event["message"]
                    if isinstance(msg_data, dict) and "content" in msg_data:
                        for block in msg_data["content"]:
                            if block.get("type") == "text":
                                new_text = block.get("text", "")
                                if accumulated_text and new_text and not accumulated_text.endswith("\n"):
                                    accumulated_text += "\n\n"
                                accumulated_text += new_text
                                yield StreamEvent(text_so_far=accumulated_text)

        except Exception as e:
            log.exception("Unexpected error reading Claude stream")
            await self._kill()
            yield StreamEvent(
                text_so_far=accumulated_text,
                done=True,
                response=ClaudeResponse(success=False, text=accumulated_text, error=str(e)),
            )

    def force_kill(self) -> None:
        """
        Kill the subprocess immediately. Safe to call without holding the lock.

        Called by /stop to abort an in-flight response. The streaming loop
        in _send_locked() will see EOF on stdout and clean up via its
        existing error-handling path. Does not await process termination
        (that happens in the streaming loop).
        """
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass

    async def change_workspace(self, new_workspace: Path) -> None:
        """
        Switch the working directory for future Claude sessions.

        Kills the current process so the next send() call will restart
        Claude in the new directory. Called by /workspace command.

        Args:
            new_workspace: Path to the new working directory.
        """
        self.workspace = new_workspace
        await self._kill()

    async def restart(self) -> None:
        """
        Kill the current process so the next send() starts fresh.
        Called by /new command and model switches.
        """
        await self._kill()

    async def _kill(self) -> None:
        """
        Kill the subprocess and clean up resources.

        Sends SIGKILL, waits for the process to exit, clears the session ID,
        and cancels the stderr drain task. Idempotent — safe to call even if
        the process has already exited.
        """
        if self._proc:
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass
            try:
                await self._proc.wait()
            except Exception:
                pass
            self._proc = None
            self._session_id = None
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None

    async def shutdown(self) -> None:
        """
        Gracefully shut down the Claude process.

        Sends SIGTERM first and waits up to 5 seconds for clean exit.
        Falls back to SIGKILL if the process doesn't terminate in time.
        Called during bot shutdown from main.py.
        """
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        self._proc = None
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None
