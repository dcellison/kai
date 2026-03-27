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

from kai.config import DATA_DIR, WorkspaceConfig, parse_env_file, resolve_claude_user
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
        workspace: Path = Path("home"),
        home_workspace: Path | None = None,
        webhook_port: int = 8080,
        webhook_secret: str = "",
        max_budget_usd: float = 1.0,
        timeout_seconds: int = 120,
        services_info: list[dict] | None = None,
        claude_user: str | None = None,
        max_session_hours: float = 0,
        workspace_config: WorkspaceConfig | None = None,
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
        self.workspace_config = workspace_config

        # Global defaults, preserved so we can restore them when
        # switching away from a configured workspace.
        self._default_model = model
        self._default_budget = max_budget_usd
        self._default_timeout = timeout_seconds

        # Apply per-workspace overrides (if configured). These become
        # the "effective" values for this workspace. The /model command
        # can still override model within a session.
        if workspace_config:
            if workspace_config.model:
                self.model = workspace_config.model
            if workspace_config.budget is not None:
                self.max_budget_usd = workspace_config.budget
            if workspace_config.timeout is not None:
                self.timeout_seconds = workspace_config.timeout

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

        # Resolve self-sudo: skip sudo when claude_user matches the bot
        # process user. The shared utility logs a warning when skipping.
        effective_claude_user = resolve_claude_user(self.claude_user)

        if effective_claude_user:
            cmd = ["sudo", "-u", effective_claude_user, "--"] + claude_cmd
        else:
            cmd = claude_cmd

        log.info(
            "Starting persistent Claude process (model=%s, user=%s)",
            self.model,
            effective_claude_user or "(same as bot)",
        )

        # Build the subprocess environment. Merge order:
        # 1. Base environment (inherited from parent process)
        # 2. Per-workspace env_file values (shared .env file)
        # 3. Per-workspace inline env values (override env_file)
        # 4. Webhook secret (LAST - workspace env can't override it)
        env = os.environ.copy()
        if self.workspace_config:
            if self.workspace_config.env_file:
                env.update(parse_env_file(self.workspace_config.env_file))
            if self.workspace_config.env:
                env.update(self.workspace_config.env)
        # Webhook secret last - ensures workspace env can't override it.
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
            start_new_session=bool(effective_claude_user),
        )
        self._session_id = None
        self._fresh_session = True
        self._session_started_at = time.monotonic()

        # Save the process group ID for reliable signal delivery.
        # When claude_user is set, start_new_session=True creates a new group
        # with PGID == PID (session leader). Save it now because os.getpgid()
        # fails after the process exits, but os.killpg() works as long as any
        # group member is still alive (i.e., the actual claude process).
        if effective_claude_user:
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

    async def send(self, prompt: str | list, chat_id: int | None = None) -> AsyncIterator[StreamEvent]:
        """
        Send a message to Claude and yield streaming events.

        This is the main public interface. All sends are serialized via an
        internal lock so concurrent callers (e.g., a user message arriving
        while a cron job is running) queue rather than interleave.

        Args:
            prompt: Either a text string or a list of content blocks (for
                multi-modal messages like images).
            chat_id: Optional Telegram chat ID of the user. When provided
                on the first message of a session, it's included in the
                context so inner Claude can route API calls to the correct
                user. Forward-compatible with Phase 3 per-user subprocesses.

        Yields:
            StreamEvent objects with accumulated text. The final event has
            done=True and includes the complete ClaudeResponse.
        """
        async with self._lock:
            async for event in self._send_locked(prompt, chat_id=chat_id):
                yield event

    async def _send_locked(self, prompt: str | list, chat_id: int | None = None) -> AsyncIterator[StreamEvent]:
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
            # Shorter timeout for recycle since the user is waiting
            await self._save_prompt(timeout=10)
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

            # Always inject Kai's personal memory from DATA_DIR. This file
            # lives outside the install tree (/var/lib/kai/memory/ in production)
            # so it survives make install. Available regardless of which
            # workspace the inner Claude is operating in.
            memory_path = DATA_DIR / "memory" / "MEMORY.md"
            if memory_path.exists():
                memory = memory_path.read_text().strip()
                if memory:
                    parts.append(f"[Your persistent memory (file: {memory_path}):]\n{memory}")
                else:
                    parts.append(f"[Your persistent memory (file: {memory_path}):]\n(currently empty)")
            else:
                parts.append(f"[Your persistent memory (file: {memory_path}):]\n(not yet created)")

            # Per-workspace system prompt from workspaces.yaml. Injected
            # between the identity/memory block and conversation history,
            # so it acts as workspace-specific instructions.
            ws_prompt = self._get_workspace_system_prompt()
            if ws_prompt:
                parts.append(f"## Workspace Instructions\n\n{ws_prompt}")

            # Always inject the per-user history directory path so the inner
            # Claude's grep/jq searches are naturally scoped to this user.
            history_dir = str(DATA_DIR / "history" / str(chat_id)) if chat_id is not None else str(DATA_DIR / "history")

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
                files_path = f"{DATA_DIR}/files/{chat_id}/" if chat_id else f"{DATA_DIR}/files/"
                parts.append(
                    f"[File API: To send a file to the user, POST JSON to "
                    f"http://localhost:{self.webhook_port}/api/send-file "
                    f"with header 'X-Webhook-Secret: $KAI_WEBHOOK_SECRET' (environment variable). "
                    f'Required: "path" (absolute file path within the current workspace {self.workspace}). '
                    f'Optional: "caption". Images are sent as photos, '
                    f"everything else as documents.\n"
                    f"Incoming files from the user are auto-saved to "
                    f"{files_path} and their paths are included in the message.]"
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
        # Wall-clock limit for the entire interaction. The per-readline
        # timeout below catches dead processes (no output for 6 min), but
        # a process stuck in a tool-use loop emits progress events that
        # reset the readline timer indefinitely. This outer limit catches
        # that case: if the total interaction exceeds N minutes regardless
        # of output, kill the process.
        interaction_start = time.monotonic()
        max_interaction_seconds = self.timeout_seconds * 5  # 10 min at default 120s
        try:
            while True:
                # Check wall-clock limit before each readline
                elapsed = time.monotonic() - interaction_start
                if elapsed > max_interaction_seconds:
                    log.error(
                        "Interaction exceeded wall-clock limit (%.0fs > %ds)",
                        elapsed,
                        max_interaction_seconds,
                    )
                    await self._kill()
                    yield StreamEvent(
                        text_so_far=accumulated_text,
                        done=True,
                        response=ClaudeResponse(
                            success=False,
                            text=accumulated_text,
                            error="Claude interaction timed out (too long)",
                        ),
                    )
                    return

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
        # Cancel the stderr drain so it does not outlive the process.
        # For /stop, _kill() will see _stderr_task=None and skip its own
        # cancel. For eviction, this is the only cleanup point.
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None

    async def _save_prompt(self, timeout: float = 30) -> None:
        """
        Send a save prompt to the inner Claude before shutdown.

        Gives the subprocess a chance to persist useful context to MEMORY.md
        before it is killed. Best-effort: if the prompt times out or the
        process is unresponsive, the caller proceeds with shutdown anyway.

        Args:
            timeout: Maximum seconds to wait for the save operation.
                Defaults to 30s for idle eviction/graceful shutdown.
                Use a shorter value (e.g., 10s) for user-visible paths
                like session recycle where latency matters.

        Prerequisites:
          - self._proc is alive and responsive
          - No other interaction is in flight (caller holds exclusive pipe access)
          - self._fresh_session is False (at least one real message was processed)
        """
        if self._fresh_session or not self.is_alive:
            return

        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            return

        save_msg = (
            "You are about to be shut down. Save anything worth remembering "
            "from this session to your memory file. Only save genuinely "
            "useful information - user preferences, personal facts, decisions, "
            "corrections, or important context. Do not save session-specific "
            "details like current task progress or temporary debugging state."
        )

        content = [{"type": "text", "text": save_msg}]
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

        try:
            self._proc.stdin.write(msg.encode())
            await self._proc.stdin.drain()
        except (OSError, RuntimeError):
            # OSError: pipe broken. RuntimeError: transport closed during
            # drain(). Either way, the process is dying. Nothing to save.
            log.debug("Save prompt write failed; process already dying")
            return

        # Read and discard response lines until we see the result event or
        # timeout. The timeout covers the entire save operation (read current
        # memory, decide what to save, write file, respond).
        try:
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    log.warning("Save prompt timed out; proceeding with shutdown")
                    break
                try:
                    line = await asyncio.wait_for(self._proc.stdout.readline(), timeout=remaining)
                except TimeoutError:
                    log.warning("Save prompt timed out; proceeding with shutdown")
                    break
                if not line:
                    # EOF - process died during save
                    break
                # Parse the line to check for completion. The inner Claude
                # responds via stream-json, so look for "result" type events.
                try:
                    event = json.loads(line)
                    if event.get("type") == "result":
                        # Final event - save interaction complete
                        log.info("Save prompt completed successfully")
                        break
                except (json.JSONDecodeError, ValueError):
                    # Non-JSON line (stderr bleed, etc.) - skip
                    continue
        except Exception:
            # Any error reading the response - log and move on.
            # The subprocess will be killed momentarily regardless.
            log.debug("Save prompt response read failed; proceeding with shutdown")

    def _get_workspace_system_prompt(self) -> str | None:
        """
        Get the system prompt for the current workspace config.

        Returns the inline system_prompt if set, or reads from
        system_prompt_file on each invocation to pick up changes.
        Returns None if neither is configured.
        """
        if not self.workspace_config:
            return None
        if self.workspace_config.system_prompt:
            return self.workspace_config.system_prompt
        if self.workspace_config.system_prompt_file:
            # File path was validated at config load time (fail-fast on typos).
            # Read content here so updates are picked up without restart.
            try:
                return self.workspace_config.system_prompt_file.read_text()
            except OSError:
                log.warning("Cannot read system_prompt_file: %s", self.workspace_config.system_prompt_file)
                return None
        return None

    async def change_workspace(
        self,
        new_workspace: Path,
        workspace_config: WorkspaceConfig | None = None,
    ) -> None:
        """
        Switch to a new workspace directory and apply its config.

        Kills the current subprocess. The next send() restarts Claude
        in the new directory with the new config applied.

        Args:
            new_workspace: Path to the new working directory.
            workspace_config: Per-workspace config for the target, or
                None to use global defaults.
        """
        # Kill first, then mutate. An in-flight _send_locked() reads
        # self.workspace, self.timeout_seconds, and self.workspace_config
        # at various await points during streaming. If we mutate before
        # killing, the stream sees new config values while still running
        # the old workspace's process. Killing first ensures the stream
        # hits EOF and exits before any state changes.
        await self._kill()

        self.workspace = new_workspace
        self.workspace_config = workspace_config

        # Always revert to global defaults first, then apply overrides.
        # This prevents stale values when switching from a fully-configured
        # workspace to a partially-configured one (e.g., workspace A has
        # budget=15.0 but workspace B only sets model - without the reset,
        # budget would carry over from A instead of reverting to default).
        self.model = self._default_model
        self.max_budget_usd = self._default_budget
        self.timeout_seconds = self._default_timeout

        if workspace_config:
            if workspace_config.model:
                self.model = workspace_config.model
            if workspace_config.budget is not None:
                self.max_budget_usd = workspace_config.budget
            if workspace_config.timeout is not None:
                self.timeout_seconds = workspace_config.timeout

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
        process state. After clearing, sends one final SIGKILL to the
        saved process group to catch any orphaned children that survived
        the initial signal (e.g., claude reparented to init after sudo
        died). The timeout prevents hanging on zombie processes.
        Idempotent - safe to call even if the process has already exited.

        Note: _stderr_task cancellation is inside the `if self._proc` guard
        because _stderr_task is only created alongside _proc in _ensure_started().
        If _proc is None, there is no stderr task to cancel.
        """
        if self._proc:
            # Save pgid before clearing - the EOF handler in _send_locked()
            # may call _kill() again after we clear self._pgid, but we need
            # to ensure the process group gets signaled at least once more
            # after the wait completes (belt-and-suspenders for the race
            # where sudo dies but claude survives the initial SIGKILL).
            saved_pgid = self._pgid

            self._send_signal(signal.SIGKILL)
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except TimeoutError:
                pass

            # Cancel the stderr drain BEFORE clearing self._proc.
            # _drain_stderr's while-loop checks self._proc on each iteration;
            # if we clear proc first, the drain task could observe None in a
            # state that was never intended to be visible to it. Cancelling
            # the task first ensures it stops reading before its dependencies
            # are destroyed.
            if self._stderr_task:
                self._stderr_task.cancel()
                self._stderr_task = None

            self._proc = None
            self._pgid = None
            self._session_id = None
            self._session_started_at = None

            # Final cleanup: signal the saved process group one more time.
            # If claude was reparented to init during the wait, this catches
            # it. If everything is already dead, killpg raises OSError which
            # we ignore. Only applies to claude_user mode (pgid is None
            # otherwise).
            if saved_pgid is not None:
                try:
                    os.killpg(saved_pgid, signal.SIGKILL)
                except OSError:
                    pass

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
            # Give the inner Claude a chance to save context before dying.
            # Best-effort: timeout or failure skips silently.
            await self._save_prompt()

            saved_pgid = self._pgid

            self._send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except TimeoutError:
                self._send_signal(signal.SIGKILL)
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=5)
                except TimeoutError:
                    log.warning("Process did not exit after SIGKILL; abandoning")
        else:
            saved_pgid = None

        # Cancel stderr drain before clearing proc (same invariant as _kill:
        # the drain task checks self._proc, so cancel it first).
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None

        # Clean up state regardless of how (or whether) the process exited
        self._proc = None
        self._pgid = None
        self._session_started_at = None

        # Final cleanup: signal the saved process group one more time
        # to catch any orphaned children that survived the initial signals.
        if saved_pgid is not None:
            try:
                os.killpg(saved_pgid, signal.SIGKILL)
            except OSError:
                pass
