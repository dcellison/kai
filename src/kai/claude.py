"""
Claude Code subprocess backend.

Implements the AgentBackend ABC for Claude Code's stream-json protocol.
Manages a long-running subprocess that accepts prompts on stdin and
streams responses on stdout as newline-delimited JSON.

This is the concrete backend that pool.py instantiates by default.
Process management (spawn, stream, kill, restart) lives here; context
injection (identity, memory, history, API docs) lives in backend.py
as shared functions usable by any backend.

The stream-json protocol:
    Input:  {"type": "user", "message": {"role": "user", "content": [...]}}
    Output: {"type": "system", ...}      - session metadata
            {"type": "assistant", ...}   - partial text (streaming)
            {"type": "result", ...}      - final response with cost/session info
"""

import asyncio
import json
import logging
import os
import signal
import time
from collections.abc import AsyncIterator
from pathlib import Path

from kai.backend import (
    AgentBackend,
    AgentResponse,
    ApiContext,
    StreamEvent,
    build_foreign_workspace_reminder,
    build_session_context,
    prepend_to_prompt,
)
from kai.config import DATA_DIR, WorkspaceConfig, parse_env_file, resolve_claude_user

log = logging.getLogger(__name__)


# ── Claude Code backend ─────────────────────────────────────────────


class ClaudeCodeBackend(AgentBackend):
    """
    AgentBackend implementation for Claude Code's stream-json protocol.

    Manages the lifecycle of a Claude Code subprocess: starting, sending
    messages, streaming responses, killing/restarting, and workspace
    switching. All message sends are serialized via an internal asyncio
    lock to prevent interleaving.

    The process runs with --permission-mode bypassPermissions (required
    for headless operation via Telegram) and --max-budget-usd to cap
    per-session spending.
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
        max_context_window: int = 0,
        autocompact_pct: int = 0,
    ):
        # ABC-required attributes (pool.py reads/writes these)
        self.model = model
        self.workspace = workspace
        self.home_workspace = home_workspace or workspace
        self.max_budget_usd = max_budget_usd
        self.timeout_seconds = timeout_seconds
        self.workspace_config = workspace_config
        self.max_context_window = max_context_window
        self.provider = "anthropic"  # Claude CLI always uses Anthropic

        # Claude-Code-specific attributes (not on the ABC)
        self.claude_user = claude_user
        self.max_session_hours = max_session_hours
        self.autocompact_pct = autocompact_pct

        # API context for session injection (passed to build_session_context)
        self._api_context = ApiContext(
            webhook_port=webhook_port,
            webhook_secret=webhook_secret,
            services_info=services_info or [],
        )

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

        # Limit context window size to reduce token usage and cache
        # invalidation pressure. Passed via --settings (not a standalone
        # CLI flag). 0 = use Claude Code's default (1M).
        if self.max_context_window > 0:
            settings = {"preferences": {"maxContextWindow": self.max_context_window}}
            claude_cmd += ["--settings", json.dumps(settings)]

        # Resolve self-sudo: skip sudo when claude_user matches the bot
        # process user. The shared utility logs a warning when skipping.
        effective_claude_user = resolve_claude_user(self.claude_user)

        if effective_claude_user:
            # --preserve-env=VAR passes exactly one env var through sudo's
            # env_reset. Requires sudo >= 1.8.11 (2014) and a SETENV: tag
            # in the sudoers rule (generated by install.py).
            cmd = [
                "sudo",
                "-u",
                effective_claude_user,
                "--preserve-env=KAI_WEBHOOK_SECRET",
                "--",
            ] + claude_cmd
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
        if self._api_context.webhook_secret:
            env["KAI_WEBHOOK_SECRET"] = self._api_context.webhook_secret

        # Set autocompact threshold so Claude compacts earlier, reducing
        # token usage. Passed as an env var (not a CLI flag) because
        # Claude Code reads it from its process environment.
        if self.autocompact_pct > 0:
            env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = str(self.autocompact_pct)

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
            done=True and includes the complete AgentResponse.
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
            done=True and includes the complete AgentResponse.
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
                response=AgentResponse(success=False, text="", error="claude CLI not found"),
            )
            return

        # Inject identity, memory, history, and API context on the
        # first message of a new session. Context injection logic lives
        # in backend.py as shared functions usable by any backend.
        if self._fresh_session:
            self._fresh_session = False
            session_ctx = build_session_context(
                workspace=self.workspace,
                home_workspace=self.home_workspace,
                api=self._api_context,
                workspace_config=self.workspace_config,
                chat_id=chat_id,
                data_dir=DATA_DIR,
            )
            prompt = prepend_to_prompt(prompt, session_ctx)

        # When in a foreign workspace, remind on every message to only
        # respond to what the user asks - workspace context (CLAUDE.md,
        # git branch, auto-memory) can otherwise trigger autonomous action.
        reminder = build_foreign_workspace_reminder(self.workspace, self.home_workspace)
        if reminder:
            prompt = prepend_to_prompt(prompt, reminder)

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
                response=AgentResponse(success=False, text="", error="Claude process died, restarting on next message"),
            )
            return

        accumulated_text = ""
        # Idle-activity timeout for the interaction. Resets every time the
        # process emits a line of output. If the process goes silent for
        # this long, it is likely dead or wedged. The per-readline timeout
        # below (timeout_seconds * 3) is the primary dead-process detector;
        # this is a secondary safety net measured across the whole interaction.
        last_activity = time.monotonic()
        max_idle_seconds = self.timeout_seconds * 5  # 10 min of silence at default 120s
        try:
            while True:
                # Check idle timeout before each readline
                idle_seconds = time.monotonic() - last_activity
                if idle_seconds > max_idle_seconds:
                    log.error(
                        "Interaction idle timeout (%.0fs with no output, limit %ds)",
                        idle_seconds,
                        max_idle_seconds,
                    )
                    await self._kill()
                    yield StreamEvent(
                        text_so_far=accumulated_text,
                        done=True,
                        response=AgentResponse(
                            success=False,
                            text=accumulated_text,
                            error="Claude interaction timed out (no output)",
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
                        response=AgentResponse(success=False, text=accumulated_text, error="Claude timed out"),
                    )
                    return

                # Reset idle timer - process is alive and producing output.
                # Do NOT reset on empty line (EOF); that means the process died.
                if line:
                    last_activity = time.monotonic()

                if not line:
                    # Process died unexpectedly
                    log.error("Claude process EOF")
                    await self._kill()
                    yield StreamEvent(
                        text_so_far=accumulated_text,
                        done=True,
                        response=AgentResponse(
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
                    response = AgentResponse(
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
                response=AgentResponse(success=False, text=accumulated_text, error=str(e)),
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
