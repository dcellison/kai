"""
Goose ACP (Agent Client Protocol) subprocess backend.

Implements the AgentBackend ABC for Goose's JSON-RPC 2.0 protocol.
Manages a persistent `goose acp` subprocess that stays alive across
multiple prompts, communicating via stdin/stdout newline-delimited
JSON-RPC messages.

This is structurally equivalent to ClaudeCodeBackend - one subprocess
per user, kept alive across messages, restarted on /new or workspace
switch. The wire protocol differs (JSON-RPC 2.0 vs Claude's
stream-json), but the lifecycle maps directly to the AgentBackend ABC.

The ACP protocol:
    Startup:  initialize → session/new (handshake)
    Input:    session/prompt (JSON-RPC request with sessionId + prompt)
    Output:   session/update (streaming notifications, no id field)
    Finish:   JSON-RPC result with matching id + stopReason
"""

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path

import kai
from kai.backend import (
    AgentBackend,
    AgentResponse,
    ApiContext,
    StreamEvent,
    build_foreign_workspace_reminder,
    build_session_context,
    prepend_to_prompt,
)
from kai.config import DATA_DIR, WorkspaceConfig, parse_env_file

log = logging.getLogger(__name__)


# Map Kai logical model names to Anthropic model IDs for GOOSE_MODEL.
# These IDs will go stale as new model versions are released.
# The .get(key, key) fallback passes unrecognized values through
# unchanged, so full model IDs (e.g. "claude-opus-4-6") work without
# being in the map.
_ANTHROPIC_MODEL_MAP: dict[str, str] = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}


# ── Goose ACP backend ─────────────────────────────────────────────


class GooseBackend(AgentBackend):
    """
    AgentBackend implementation for Goose's ACP (JSON-RPC 2.0) protocol.

    Manages the lifecycle of a persistent `goose acp` subprocess:
    starting with a two-step handshake (initialize + session/new),
    sending prompts via session/prompt, streaming responses from
    session/update notifications, and killing/restarting on demand.

    All message sends are serialized via an internal asyncio lock to
    prevent interleaving. The developer builtin extension auto-approves
    all tool calls, so no permission write-back is needed.
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
        workspace_config: WorkspaceConfig | None = None,
        max_context_window: int = 0,
        goose_provider: str = "",
    ):
        # ABC-required attributes (pool.py reads/writes these)
        self.model = model
        self.workspace = workspace
        self.home_workspace = home_workspace or workspace
        self.max_budget_usd = max_budget_usd
        self.timeout_seconds = timeout_seconds
        self.workspace_config = workspace_config
        self.max_context_window = max_context_window
        self.provider = goose_provider  # ABC-mandated; bot.py reads this
        self.goose_provider = goose_provider

        # API context for session injection (passed to build_session_context)
        self._api_context = ApiContext(
            webhook_port=webhook_port,
            webhook_secret=webhook_secret,
            services_info=services_info or [],
        )

        # Global defaults, preserved so we can restore them when
        # switching away from a configured workspace.
        self._default_model = model
        self._default_timeout = timeout_seconds

        # Apply per-workspace overrides (if configured). These become
        # the "effective" values for this workspace.
        if workspace_config:
            if workspace_config.model:
                self.model = workspace_config.model
            if workspace_config.timeout is not None:
                self.timeout_seconds = workspace_config.timeout

        # Subprocess and session state
        self._proc: asyncio.subprocess.Process | None = None
        self._session_id: str | None = None
        self._next_id: int = 1  # Monotonically increasing JSON-RPC request ID
        self._lock = asyncio.Lock()  # Serializes all message sends
        self._fresh_session: bool = True  # True until the first message is sent
        self._stderr_task: asyncio.Task | None = None  # Background stderr drain

    @property
    def is_alive(self) -> bool:
        """True if the Goose subprocess is running and hasn't exited."""
        return self._proc is not None and self._proc.returncode is None

    @property
    def session_id(self) -> str | None:
        """The current Goose session ID, or None if no session is active."""
        return self._session_id

    # ── Process lifecycle ──────────────────────────────────────────

    async def _ensure_started(self) -> None:
        """
        Start the Goose ACP subprocess if not already running.

        Launches `goose acp --with-builtin developer` and runs the
        two-step handshake:
        1. initialize - establishes protocol version and capabilities
        2. session/new - creates a session with cwd and empty mcpServers

        The process persists across prompts. Model selection happens via
        the GOOSE_MODEL environment variable (translated from Kai's
        logical model names via _ANTHROPIC_MODEL_MAP).
        """
        if self.is_alive:
            return

        # Build the subprocess environment. Merge order:
        # 1. Base environment (inherited from parent process)
        # 2. GOOSE_MODEL (translated from self.model)
        # 3. Per-workspace env_file values
        # 4. Per-workspace inline env values (override env_file)
        # 5. Webhook secret (LAST - workspace env can't override it)
        env = os.environ.copy()
        # Kai's logical model names ("sonnet", "opus", "haiku") only apply
        # to the Anthropic provider. Other providers require full model IDs
        # set via user config (users.yaml or /settings). Pass through unchanged.
        if self.goose_provider == "anthropic":
            mapped = _ANTHROPIC_MODEL_MAP.get(self.model, self.model)
        else:
            mapped = self.model
        if mapped:
            env["GOOSE_MODEL"] = mapped
        if self.workspace_config:
            if self.workspace_config.env_file:
                env.update(parse_env_file(self.workspace_config.env_file))
            if self.workspace_config.env:
                env.update(self.workspace_config.env)
        if self._api_context.webhook_secret:
            env["KAI_WEBHOOK_SECRET"] = self._api_context.webhook_secret

        log.info(
            "Starting persistent Goose ACP process (model=%s, mapped=%s)",
            self.model,
            mapped,
        )

        self._proc = await asyncio.create_subprocess_exec(
            "goose",
            "acp",
            "--with-builtin",
            "developer",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workspace),
            env=env,
            limit=1024 * 1024,  # 1MB per-line buffer
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())

        # Step 1: initialize - establish protocol version
        self._next_id = 1
        await self._write_rpc(
            "initialize",
            {
                "protocolVersion": "v1",
                "clientInfo": {"name": "kai", "version": kai.__version__},
            },
        )
        await self._read_result(expected_id=1)

        # Step 2: session/new - create a session with workspace cwd.
        # mcpServers must be an array (even if empty) - an object
        # causes a deserialization error in Goose.
        await self._write_rpc(
            "session/new",
            {
                "cwd": str(self.workspace),
                "mcpServers": [],
            },
        )
        result = await self._read_result(expected_id=2)
        self._session_id = result["sessionId"]
        self._fresh_session = True

    async def _drain_stderr(self) -> None:
        """
        Continuously read and discard stderr from the Goose process.

        Without this, the stderr pipe buffer fills up and the process
        deadlocks. Lines are logged at DEBUG level for diagnostics.
        """
        while self._proc and self._proc.stderr:
            try:
                line = await self._proc.stderr.readline()
                if not line:
                    break
                text = line.decode().strip()
                if text:
                    log.debug("Goose stderr: %s", text[:200])
            except Exception:
                log.warning("Unexpected error in Goose stderr drain", exc_info=True)
                break

    # ── JSON-RPC helpers ───────────────────────────────────────────

    async def _write_rpc(self, method: str, params: dict) -> None:
        """
        Write a JSON-RPC 2.0 request to the subprocess stdin.

        Increments the monotonic request ID, serializes the message
        with a trailing newline, and flushes. Raises RuntimeError if
        the process or its stdin pipe is gone.
        """
        assert self._proc is not None and self._proc.stdin is not None
        request_id = self._next_id
        self._next_id += 1
        msg = (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": method,
                    "id": request_id,
                    "params": params,
                }
            )
            + "\n"
        )
        self._proc.stdin.write(msg.encode())
        await self._proc.stdin.drain()

    async def _read_result(self, expected_id: int) -> dict:
        """
        Read stdout lines until a JSON-RPC result with the expected id appears.

        Discards session/update notifications that Goose may emit during
        startup. Raises on JSON-RPC error responses or timeout.
        """
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            try:
                line = await asyncio.wait_for(
                    self._proc.stdout.readline(),
                    timeout=self.timeout_seconds,
                )
            except TimeoutError as exc:
                raise TimeoutError(f"Goose ACP handshake timed out waiting for response id={expected_id}") from exc

            if not line:
                raise RuntimeError("Goose process exited during handshake")

            try:
                msg = json.loads(line.decode())
            except json.JSONDecodeError:
                # Non-JSON output during startup (e.g., progress bars)
                continue

            # Check for matching response
            if msg.get("id") == expected_id:
                if "error" in msg:
                    err = msg["error"]
                    raise RuntimeError(f"Goose ACP error (id={expected_id}): {err.get('message', 'unknown error')}")
                return msg.get("result", {})

            # Discard notifications (session/update during startup)

    # ── Sending prompts ────────────────────────────────────────────

    async def send(self, prompt: str | list, chat_id: int | None = None) -> AsyncIterator[StreamEvent]:
        """
        Send a message to Goose and yield streaming events.

        Serialized via an internal lock so concurrent callers queue
        rather than interleave. Context injection (identity, memory,
        history, API docs) is prepended on the first message of each
        session, identical to ClaudeCodeBackend.

        Args:
            prompt: Either a text string or a list of content blocks.
            chat_id: Optional Telegram chat ID for history scoping
                and API routing.

        Yields:
            StreamEvent objects with accumulated text. The final event
            has done=True and includes the complete AgentResponse.
        """
        async with self._lock:
            async for event in self._send_locked(prompt, chat_id):
                yield event

    async def _send_locked(self, prompt: str | list, chat_id: int | None = None) -> AsyncIterator[StreamEvent]:
        """
        Core send logic (must be called while holding self._lock).

        Handles context injection, prompt coercion to ACP format,
        JSON-RPC session/prompt dispatch, and streaming response
        accumulation from session/update notifications.
        """
        try:
            await self._ensure_started()
        except (OSError, RuntimeError, TimeoutError) as exc:
            yield StreamEvent(
                text_so_far="",
                done=True,
                response=AgentResponse(success=False, text="", error=f"Goose startup failed: {exc}"),
            )
            return

        # Inject identity, memory, history, and API context on the
        # first message of a new session. Context injection logic
        # lives in backend.py as shared functions.
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
        # respond to what the user asks.
        reminder = build_foreign_workspace_reminder(self.workspace, self.home_workspace)
        if reminder:
            prompt = prepend_to_prompt(prompt, reminder)

        # Coerce prompt to ACP format (array of content blocks).
        # Goose supports images but the initial implementation handles
        # text only. Non-text blocks are logged and skipped.
        acp_prompt: list[dict] = []
        if isinstance(prompt, str):
            acp_prompt = [{"type": "text", "text": prompt}]
        else:
            for block in prompt:
                if block.get("type") == "text":
                    acp_prompt.append({"type": "text", "text": block["text"]})
                else:
                    log.warning(
                        "GooseBackend: dropping non-text content block type=%s",
                        block.get("type"),
                    )
            if not acp_prompt:
                acp_prompt = [{"type": "text", "text": "(empty prompt)"}]

        # Send the session/prompt request
        assert self._proc is not None
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None

        try:
            prompt_id = self._next_id
            await self._write_rpc(
                "session/prompt",
                {
                    "sessionId": self._session_id,
                    "prompt": acp_prompt,
                },
            )
        except OSError as e:
            log.error("Failed to write to Goose process: %s", e)
            await self._kill()
            yield StreamEvent(
                text_so_far="",
                done=True,
                response=AgentResponse(
                    success=False,
                    text="",
                    error="Goose process died, restarting on next message",
                ),
            )
            return

        # Stream response: read stdout lines, accumulate
        # agent_message_chunk text, and yield StreamEvents.
        accumulated = ""
        last_activity = time.monotonic()
        max_idle_seconds = self.timeout_seconds * 5

        try:
            while True:
                # Check idle timeout before each readline
                idle = time.monotonic() - last_activity
                if idle > max_idle_seconds:
                    log.error(
                        "Goose idle timeout (%.0fs with no output, limit %ds)",
                        idle,
                        max_idle_seconds,
                    )
                    await self._kill()
                    yield StreamEvent(
                        text_so_far=accumulated,
                        done=True,
                        response=AgentResponse(
                            success=False,
                            text=accumulated,
                            error="Goose interaction timed out (no output)",
                        ),
                    )
                    return

                try:
                    line = await asyncio.wait_for(
                        self._proc.stdout.readline(),
                        timeout=self.timeout_seconds * 3,
                    )
                except TimeoutError:
                    log.error("Goose response timed out")
                    await self._kill()
                    yield StreamEvent(
                        text_so_far=accumulated,
                        done=True,
                        response=AgentResponse(
                            success=False,
                            text=accumulated,
                            error="Goose timed out",
                        ),
                    )
                    return

                # Reset idle timer on non-empty output
                if line:
                    last_activity = time.monotonic()
                else:
                    # EOF - process died
                    log.error("Goose process EOF")
                    await self._kill()
                    yield StreamEvent(
                        text_so_far=accumulated,
                        done=True,
                        response=AgentResponse(
                            success=bool(accumulated),
                            text=accumulated,
                            error=None if accumulated else "Goose process ended unexpectedly",
                        ),
                    )
                    return

                try:
                    msg = json.loads(line.decode())
                except json.JSONDecodeError:
                    log.debug("Skipping non-JSON line: %s", line[:200])
                    continue

                # Streaming notifications (no id field) - accumulate
                # agent_message_chunk text, skip everything else.
                if msg.get("method") == "session/update":
                    update = msg.get("params", {}).get("update", {})
                    if update.get("sessionUpdate") == "agent_message_chunk":
                        text = update.get("content", {}).get("text", "")
                        if text:
                            accumulated += text
                            yield StreamEvent(text_so_far=accumulated)
                    # agent_thought_chunk, tool_call, tool_call_update: skip

                # Final result for our prompt (has matching id)
                elif msg.get("id") == prompt_id and "result" in msg:
                    response = AgentResponse(
                        success=True,
                        text=accumulated,
                        session_id=self._session_id,
                        cost_usd=0.0,  # Goose ACP does not report cost
                        duration_ms=0,
                    )
                    yield StreamEvent(
                        text_so_far=accumulated,
                        done=True,
                        response=response,
                    )
                    return

                # JSON-RPC error for our prompt
                elif msg.get("id") == prompt_id and "error" in msg:
                    err = msg["error"].get("message", "unknown ACP error")
                    yield StreamEvent(
                        text_so_far=accumulated,
                        done=True,
                        response=AgentResponse(
                            success=False,
                            text=accumulated,
                            error=err,
                        ),
                    )
                    return

        except Exception as e:
            log.exception("Unexpected error reading Goose stream")
            await self._kill()
            yield StreamEvent(
                text_so_far=accumulated,
                done=True,
                response=AgentResponse(
                    success=False,
                    text=accumulated,
                    error=str(e),
                ),
            )

    # ── Kill / restart / shutdown ──────────────────────────────────

    def force_kill(self) -> None:
        """
        Kill the subprocess immediately via SIGKILL.

        Safe to call without holding the lock. Called by /stop to abort
        an in-flight response. Does NOT null _proc - full cleanup
        happens in _kill() when the read loop detects EOF.
        """
        if self._proc is not None:
            try:
                self._proc.kill()
            except OSError:
                pass
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None

    async def _kill(self) -> None:
        """
        Kill the subprocess and clean up all process state.

        Sends SIGKILL, waits up to 5 seconds for exit, then nulls
        all process references. Idempotent.
        """
        if self._proc:
            self.force_kill()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except TimeoutError:
                pass
            # force_kill() already cancelled _stderr_task and set it to
            # None, so no additional cleanup needed here.
            self._proc = None
            self._session_id = None

    async def restart(self) -> None:
        """
        Kill the current process so the next send() starts fresh.

        Called by /new command and model switches. The next send()
        calls _ensure_started() which re-runs the full handshake
        with a new session.
        """
        await self._kill()
        self._fresh_session = True

    async def change_workspace(
        self,
        new_workspace: Path,
        workspace_config: WorkspaceConfig | None = None,
    ) -> None:
        """
        Switch to a new workspace directory and apply its config.

        Kills the current subprocess. The next send() restarts Goose
        in the new directory with the new config applied.
        """
        await self._kill()
        self.workspace = new_workspace
        self.workspace_config = workspace_config

        # Revert to global defaults, then apply overrides. Prevents
        # stale values when switching from a fully-configured workspace
        # to a partially-configured one.
        self.model = self._default_model
        self.timeout_seconds = self._default_timeout
        if workspace_config:
            if workspace_config.model:
                self.model = workspace_config.model
            if workspace_config.timeout is not None:
                self.timeout_seconds = workspace_config.timeout
        # _fresh_session is set True by _ensure_started() on next send()

    async def shutdown(self) -> None:
        """
        Graceful shutdown. No save prompt - Goose ACP has no equivalent.

        Sends SIGTERM first and waits up to 5 seconds for clean exit.
        Falls back to SIGKILL if the process doesn't terminate in time.
        """
        if self._proc:
            try:
                self._proc.terminate()
            except OSError:
                # Process already exited between the _proc check and
                # terminate(). Fall through to cleanup below.
                pass
            else:
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=5)
                except TimeoutError:
                    self.force_kill()
                    try:
                        await asyncio.wait_for(self._proc.wait(), timeout=5)
                    except TimeoutError:
                        log.warning("GooseBackend: process did not exit after SIGKILL")
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None
        self._proc = None
        self._session_id = None
