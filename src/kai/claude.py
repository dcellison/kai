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
    3. Recent conversation history (last 20 messages from JSONL logs)
    4. Scheduling API endpoint info (URL, secret, field reference)
"""

import asyncio
import json
import logging
import os
import signal
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from kai.history import get_recent_history

log = logging.getLogger(__name__)


# ── Protocol types ───────────────────────────────────────────────────


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


# ── Persistent Claude process ────────────────────────────────────────


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
        claude_user: str | None = None,
        max_session_hours: float = 0,
    ):
        self.model = model
        self.workspace = workspace
        self.home_workspace = home_workspace or workspace
        self.webhook_port = webhook_port
        self.webhook_secret = webhook_secret
        self.max_budget_usd = max_budget_usd
        self.timeout_seconds = timeout_seconds
        self.services_info = services_info or []
        self.claude_user = claude_user
        self.max_session_hours = max_session_hours
        self._proc: asyncio.subprocess.Process | None = None
        self._pgid: int | None = None  # Process group ID for reliable signal delivery
        self._lock = asyncio.Lock()  # Serializes all message sends
        self._session_id: str | None = None
        self._fresh_session = True  # True until the first message is sent
        self._stderr_task: asyncio.Task | None = None  # Background stderr drain
        self._session_started_at: float | None = None  # time.monotonic() at process start

    @property
    def is_alive(self) -> bool:
        """True if the Claude subprocess is running and hasn't exited."""
        return self._proc is not None and self._proc.returncode is None

    @property
    def session_id(self) -> str | None:
        """The current Claude session ID, or None if no session is active."""
        return self._session_id

    def _session_age_hours(self) -> float:
        """Hours elapsed since the current session started."""
        if self._session_started_at is None:
            return 0.0
        return (time.monotonic() - self._session_started_at) / 3600

    def _should_recycle(self) -> bool:
        """True if the session has exceeded the configured age limit."""
        return self.max_session_hours > 0 and self.is_alive and self._session_age_hours() >= self.max_session_hours

    async def _ensure_started(self) -> None:
        """
        Start the Claude Code subprocess if not already running.

        Launches claude with stream-json I/O, bypassPermissions mode (required
        for headless operation), and the configured model and budget. The process
        runs in the current workspace directory and persists across messages.

        When claude_user is set, the process is spawned via sudo -u to run as
        a different OS user. The subprocess is started in its own process group
        (start_new_session=True) so the entire tree (sudo + claude) can be
        killed reliably via os.killpg().

        The stdout buffer limit is raised to 1 MiB (from the default 64 KiB)
        because large tool results from Claude can exceed the default.
        """
        if self.is_alive:
            return

        # Build the Claude command arguments.
        claude_cmd = [
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

        # When running as a different user, spawn via sudo -u.
        # The subprocess runs with the target user's UID, home directory,
        # and environment - completely isolated from the bot user.
        if self.claude_user:
            cmd = ["sudo", "-u", self.claude_user, "--"] + claude_cmd
        else:
            cmd = claude_cmd

        log.info(
            "Starting persistent Claude process (model=%s, user=%s)",
            self.model,
            self.claude_user or "(same as bot)",
        )

        # Pass the webhook secret via environment variable so it never
        # appears in prompt text, Claude Code session logs, or Anthropic's
        # API logs. The inner Claude references $KAI_WEBHOOK_SECRET in curl
        # commands instead of a literal secret value.
        env = os.environ.copy()
        if self.webhook_secret:
            env["KAI_WEBHOOK_SECRET"] = self.webhook_secret

        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workspace),
            env=env,
            limit=1024 * 1024,  # 1 MiB; default 64 KiB too small for large tool results
            # When spawned via sudo, start in a new process group so we can
            # kill the entire tree (sudo + claude) via os.killpg(). Without
            # this, killing sudo may orphan the claude process.
            start_new_session=bool(self.claude_user),
        )
        self._session_id = None
        self._fresh_session = True
        self._session_started_at = time.monotonic()

        # Save the process group ID for reliable signal delivery.
        # When claude_user is set, start_new_session=True creates a new group
        # with PGID == PID (session leader). Save it now because os.getpgid()
        # fails after the process exits, but os.killpg() works as long as any
        # group member is still alive (i.e., the actual claude process).
        if self.claude_user:
            self._pgid = self._proc.pid  # PGID == PID for session leaders
        else:
            self._pgid = None

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
                log.warning("Unexpected error in stderr drain", exc_info=True)
                break

    def _send_signal(self, sig: int) -> None:
        """
        Send a signal to the Claude process (or process group if claude_user).

        Deliberately does NOT check self._proc.returncode. When claude_user
        is set, self._proc tracks the sudo wrapper, not the actual claude
        process. If sudo exits before claude (e.g., from SIGTERM), checking
        returncode would skip sending further signals - leaving claude
        orphaned. Instead, we always attempt delivery and let OSError handle
        the already-dead case cleanly:
        - claude_user path: os.killpg() raises OSError if the group is gone
        - direct path: send_signal() calls os.kill() which raises OSError

        Args:
            sig: Signal to send (e.g., signal.SIGTERM, signal.SIGKILL).
        """
        if self._pgid is not None:
            # claude_user mode: signal the entire process group (sudo + claude)
            # using the PGID saved at spawn time
            try:
                os.killpg(self._pgid, sig)
            except OSError:
                pass
        elif self._proc is not None:
            try:
                self._proc.send_signal(sig)
            except OSError:
                pass

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
        # Recycle the session if it has exceeded the age limit. This prevents
        # unbounded memory growth in the inner Claude process (Node.js/V8),
        # which can cause macOS kernel panics via Jetsam on memory-constrained
        # machines. Only checked before starting a new interaction, never
        # during one, so in-flight responses complete normally.
        if self._should_recycle():
            log.info(
                "Session age %.1f hours exceeds limit of %.1f hours; recycling",
                self._session_age_hours(),
                self.max_session_hours,
            )
            await self._kill()

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

            # NOTE: Foreign workspace memory (.claude/MEMORY.md) is NOT injected
            # here. Claude Code reads it natively from its cwd (built-in auto-memory).
            # Bot-side reads are redundant and can cause PermissionError on Linux
            # when the workspace is owned by a different user (CLAUDE_USER).

            # Inject recent conversation history for continuity
            recent = get_recent_history()
            if recent:
                parts.append(f"[Recent conversations (search .claude/history/ for full logs):]\n{recent}")

            # Inject scheduling API info (always, so cron works from any workspace).
            # The secret is passed via $KAI_WEBHOOK_SECRET env var (not embedded
            # in prompt text) to prevent leakage through session logs.
            if self.webhook_secret:
                api_note = (
                    f"[Scheduling API: To create jobs, POST JSON to "
                    f"http://localhost:{self.webhook_port}/api/schedule "
                    f"with header 'X-Webhook-Secret: $KAI_WEBHOOK_SECRET' (environment variable). "
                    f"Required fields: name, prompt, schedule_type, schedule_data. "
                    f"Optional: job_type (reminder|claude), auto_remove (bool). "
                    f"To list jobs: GET /api/jobs. To update: PATCH /api/jobs/{{id}}. "
                    f"To delete: DELETE /api/jobs/{{id}}.]"
                )
                if self.workspace != self.home_workspace:
                    api_note = (
                        f"[Workspace context: You are working in {self.workspace}. "
                        f"Your home workspace is {self.home_workspace}.]\n{api_note}"
                    )
                parts.append(api_note)

            # Inject messaging and file exchange API info so Claude can
            # proactively send text or files to the user (e.g., when a
            # background task completes or a scheduled job has results).
            if self.webhook_secret:
                parts.append(
                    f"[Messaging API: To send a text message to the user proactively "
                    f"(e.g., background task results), POST JSON to "
                    f"http://localhost:{self.webhook_port}/api/send-message "
                    f"with header 'X-Webhook-Secret: $KAI_WEBHOOK_SECRET' (environment variable). "
                    f'Required: "text" (the message content). '
                    f"Long messages are automatically split at Telegram's 4096-char limit.]"
                )
                parts.append(
                    f"[File API: To send a file to the user, POST JSON to "
                    f"http://localhost:{self.webhook_port}/api/send-file "
                    f"with header 'X-Webhook-Secret: $KAI_WEBHOOK_SECRET' (environment variable). "
                    f'Required: "path" (absolute file path within the current workspace {self.workspace}). '
                    f'Optional: "caption". Images are sent as photos, '
                    f"everything else as documents.\n"
                    f"Incoming files from the user are auto-saved to "
                    f"{self.workspace}/files/ and their paths are included "
                    f"in the message.]"
                )

            # Inject available external services info (only if services are configured)
            if self.services_info and self.webhook_secret:
                svc_lines = [
                    "[External Services: To call external APIs, POST JSON to "
                    f"http://localhost:{self.webhook_port}/api/services/{{name}} "
                    f"with header 'X-Webhook-Secret: $KAI_WEBHOOK_SECRET' (environment variable). "
                    "Request JSON fields (all optional): "
                    '"body" (dict - forwarded as JSON), '
                    '"params" (dict - query parameters), '
                    '"path_suffix" (str - appended to base URL).',
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
                    f"""-H "X-Webhook-Secret: $KAI_WEBHOOK_SECRET" """
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
                    log.debug("Skipping non-JSON stdout line: %s", line.decode().strip()[:200])
                    continue

                etype = event.get("type")

                if etype == "system":
                    sid = event.get("session_id")
                    if sid:
                        self._session_id = sid

                elif etype == "result":
                    # Prefer accumulated_text (which includes text before tool
                    # use) over the result event's text (which may only contain
                    # the final assistant message). Fall back to result_text
                    # when nothing was accumulated (e.g., system-only responses).
                    result_text = event.get("result", "")
                    text = accumulated_text if accumulated_text else result_text
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

        Called by /stop to abort an in-flight response. There is a race window
        between _ensure_started() and the stdin write in _send_locked(), but
        it is safe: killing the process causes EOF on stdout, which the
        streaming loop handles by yielding a done event and calling _kill()
        to clean up. No lock acquisition is needed here because _send_signal()
        only sends a signal and is itself idempotent.
        """
        self._send_signal(signal.SIGKILL)

    async def change_workspace(self, new_workspace: Path) -> None:
        """
        Switch the working directory for future Claude sessions.

        Kills the current process so the next send() call will restart
        Claude in the new directory. Called by /workspace command.

        Args:
            new_workspace: Path to the new working directory.
        """
        # No lock needed: _kill() terminates the process, and the next send()
        # call will start fresh in the new workspace via _ensure_started().
        # Any in-flight send() will see EOF on stdout and clean up.
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
        Kill the subprocess immediately and clean up resources.

        Sends SIGKILL, waits up to 5 seconds for exit, then clears all
        process state. The timeout prevents hanging on zombie processes
        (the previous bare wait() had no timeout). Idempotent - safe to
        call even if the process has already exited.
        """
        if self._proc:
            self._send_signal(signal.SIGKILL)
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except TimeoutError:
                pass
            self._proc = None
            self._pgid = None
            self._session_id = None
            self._session_started_at = None
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None

    async def shutdown(self) -> None:
        """
        Gracefully shut down the Claude process.

        Sends SIGTERM first and waits up to 5 seconds for clean exit.
        Falls back to SIGKILL if the process doesn't terminate in time.
        Called during bot shutdown from main.py.

        Unlike the old implementation, this does NOT check returncode before
        sending signals. When claude_user is set, self._proc tracks sudo,
        not claude - if sudo exits from SIGTERM before claude does, the
        returncode guard would skip the SIGKILL fallback, orphaning claude.
        _send_signal() handles already-dead processes via OSError instead.
        """
        if self._proc:
            self._send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except TimeoutError:
                self._send_signal(signal.SIGKILL)
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=5)
                except TimeoutError:
                    log.warning("Process did not exit after SIGKILL; abandoning")
        # Clean up state regardless of how (or whether) the process exited
        self._proc = None
        self._pgid = None
        self._session_started_at = None
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None
