"""
OpenAI Codex CLI subprocess backend.

Implements the AgentBackend ABC for Codex's JSON-RPC 2.0 protocol over
a persistent `codex app-server` subprocess. Structurally equivalent to
GooseBackend: one subprocess per user, kept alive across messages,
restarted on /new or workspace switch. The wire protocol is JSON-RPC
2.0 (same family as goose's ACP), but the message types and event
vocabulary are codex-specific.

The codex protocol (per pinned codex CLI version):
    Startup:  initialize -> session/new (handshake)
    Input:    session/prompt (JSON-RPC request with sessionId + prompt)
    Output:   session/update (streaming notifications, no id field)
    Finish:   JSON-RPC result with matching id

Schema-drift posture: the documented codex event names are known to be
out of sync with actual CLI output (openai/codex#4776). This module
matches the goose ACP envelope (session/update notifications carrying
an agent_message_chunk payload) because that envelope is what JSON-RPC
2.0 servers conventionally emit; the first smoke test against a real
codex binary will reveal whatever variations exist for the pinned
version, and the unknown-event branches log at DEBUG and skip rather
than aborting. The codex CLI version is captured in install metadata
so a future bump triggers a re-pin pass.

This module does NOT take a dependency on OpenAI's Codex Python SDK.
The decision matches goose.py: keeping the wire protocol in our own
code keeps Kai's release schedule decoupled from a vendor SDK's
versioning and avoids inheriting transitive dependencies.
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
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
    ensure_user_memory,
    ensure_user_preferences,
    prepend_to_prompt,
)
from kai.config import DATA_DIR, WorkspaceConfig, parse_env_file, resolve_claude_user

log = logging.getLogger(__name__)


# Codex is openai-only in v1. Pool.py resolves self.model via
# PROVIDER_DEFAULTS["openai"] (currently "gpt-5.4") for codex-backed
# users, so the value is already in OpenAI's native form ("gpt-5.4",
# "gpt-5.4-mini", "gpt-5.4-nano"). No logical-name remapping is
# needed at this layer, unlike goose.py which translates "sonnet" /
# "opus" / "haiku" to claude-sonnet-4-6 etc for the Anthropic case.
# If a future codex version accepts a Kai-internal alias, add a map
# here and apply it before setting CODEX_MODEL on the subprocess env.


# ── Codex CLI backend ─────────────────────────────────────────────


class CodexBackend(AgentBackend):
    """
    AgentBackend implementation for OpenAI Codex CLI's JSON-RPC protocol.

    Manages the lifecycle of a persistent `codex app-server` subprocess:
    starting with a two-step handshake (initialize + session/new),
    sending prompts via session/prompt, streaming responses from
    session/update notifications, and killing/restarting on demand.

    All message sends are serialized via an internal asyncio lock to
    prevent interleaving. Tool auto-approval is handled by codex's own
    config; this backend does not inject permission decisions.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-5.4",
        workspace: Path = Path("home"),
        home_workspace: Path | None = None,
        webhook_port: int = 8080,
        webhook_secret: str = "",
        max_budget_usd: float = 1.0,
        timeout_seconds: int = 120,
        services_info: list[dict] | None = None,
        workspace_config: WorkspaceConfig | None = None,
        max_context_window: int = 0,
        provider: str = "openai",
        # Optional OS user to run codex as via `sudo -H -u <user>`. When
        # set, the codex subprocess runs as <codex_user> and reads its
        # auth token from ~<codex_user>/.codex/auth.json, NOT the kai
        # service user's home. This is the multi-user OAuth-isolation
        # lever: a kai install that runs as service user "kai" but has
        # users.yaml entries with os_user="daniel" / os_user="scott"
        # spawns a codex subprocess as the per-user os_user, picking
        # up that user's own codex login. Mirrors ClaudeCodeBackend's
        # claude_user kwarg. Default None means "run as the service
        # user" (single-user install or test fixture).
        codex_user: str | None = None,
        # Operator-intent flag for the memory subsystem (Config.
        # memory_enabled). Drives the [Memory subsystem: ...] marker
        # emission and gates MEMORY.md inject in build_session_context.
        # Default False so direct test instantiations need not plumb
        # the kwarg; production callers (pool.py) always pass an
        # explicit value. Codex installs run with memory disabled
        # (the install wizard's guard forces MEMORY_ENABLED=false on
        # codex), so this flag is effectively always False in v1.
        memory_enabled: bool = False,
    ):
        # ABC-required attributes (pool.py reads/writes these)
        self.model = model
        self.workspace = workspace
        self.home_workspace = home_workspace or workspace
        self.max_budget_usd = max_budget_usd
        self.timeout_seconds = timeout_seconds
        self.workspace_config = workspace_config
        self.max_context_window = max_context_window
        self.provider = provider  # ABC-mandated; bot.py reads this
        self.codex_user = codex_user
        self.memory_enabled = memory_enabled

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

        # Cross-user teardown apparatus (#456-equivalent for codex).
        # When effective_codex_user is set in _ensure_started, the
        # subprocess is the sudo wrapper, not codex itself; signalling
        # only the wrapper orphans the codex grandchild because POSIX
        # signal permission rules forbid a service-user-owned killpg
        # from reaching a target-user grandchild. The escalation path
        # uses `sudo -n -u <target> /bin/kill ...` against the cached
        # inner-codex PID. See _send_signal for the full rationale and
        # claude.py for the original implementation this mirrors.
        self._pgid: int | None = None
        self._inner_codex_pid: int | None = None
        self._effective_codex_user: str | None = None

    @property
    def is_alive(self) -> bool:
        """True if the codex subprocess is running and hasn't exited."""
        return self._proc is not None and self._proc.returncode is None

    @property
    def session_id(self) -> str | None:
        """The current codex session ID, or None if no session is active."""
        return self._session_id

    # ── Process lifecycle ──────────────────────────────────────────

    async def _ensure_started(self) -> None:
        """
        Start the codex app-server subprocess if not already running.

        Launches `codex app-server` and runs the two-step JSON-RPC
        handshake:
        1. initialize - establishes protocol version and capabilities
        2. session/new - creates a session with cwd

        The process persists across prompts. Model selection happens
        via the CODEX_MODEL environment variable (a passthrough of
        self.model; no logical-alias mapping needed at this layer).
        """
        if self.is_alive:
            return

        # Build the subprocess environment. Merge order:
        # 1. Base environment (inherited from parent process)
        # 2. CODEX_MODEL (mirrors self.model)
        # 3. CODEX_PROVIDER (forward-compat placeholder; "openai" today)
        # 4. Per-workspace env_file values
        # 5. Per-workspace inline env values (override env_file)
        # 6. Webhook secret (LAST - workspace env can't override it)
        env = os.environ.copy()
        if self.model:
            env["CODEX_MODEL"] = self.model
        if self.provider:
            env["CODEX_PROVIDER"] = self.provider
        if self.workspace_config:
            if self.workspace_config.env_file:
                env.update(parse_env_file(self.workspace_config.env_file))
            if self.workspace_config.env:
                env.update(self.workspace_config.env)
        if self._api_context.webhook_secret:
            env["KAI_WEBHOOK_SECRET"] = self._api_context.webhook_secret

        # Build the argv. Codex runs either as the bot's process user
        # (single-user install or self-sudo) or as a per-user os_user
        # (multi-user install where each users.yaml entry has its own
        # ~/.codex/auth.json). resolve_claude_user is named claude-
        # historically but its body is backend-agnostic: it returns
        # None when codex_user matches the bot's own user (skipping
        # an unnecessary self-sudo) and the value unchanged otherwise.
        effective_codex_user = resolve_claude_user(self.codex_user)

        codex_argv = ["codex", "app-server"]
        if effective_codex_user:
            # -H sets HOME to <codex_user>'s pw entry so codex reads
            # auth from ~<codex_user>/.codex/auth.json, not the bot's
            # home. --preserve-env passes KAI_WEBHOOK_SECRET and
            # TMPDIR through sudo's env_reset (the SETENV: sudoers
            # rule allows this). Mirrors claude.py's sudo construction.
            argv: list[str] = [
                "sudo",
                "-H",
                "-u",
                effective_codex_user,
                "--preserve-env=KAI_WEBHOOK_SECRET,TMPDIR",
                "--",
            ] + codex_argv
        else:
            argv = codex_argv

        log.info(
            "Starting persistent Codex app-server process (model=%s, provider=%s, user=%s)",
            self.model,
            self.provider,
            effective_codex_user or "(same as bot)",
        )

        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workspace),
            env=env,
            # Cross-user mode: new session group so the sudo wrapper
            # is the session leader (PGID == PID). os.killpg later
            # reaps the wrapper after the inner-codex sudo escalation
            # in _send_signal / _async_send_signal_for_close has
            # signalled the grandchild directly. Without the
            # escalation, killpg alone would orphan the codex
            # grandchild (POSIX permission rules: service user
            # cannot signal a process whose UIDs are all the
            # target user). See _send_signal for the full chain.
            start_new_session=bool(effective_codex_user),
            limit=1024 * 1024,  # 1MB per-line buffer
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())

        # Save the process group ID for reliable signal delivery.
        # When effective_codex_user is set, start_new_session=True
        # made the wrapper a session leader so PGID == PID. We save
        # it now because os.getpgid() fails after the process exits,
        # but os.killpg() works as long as any group member survives.
        if effective_codex_user:
            self._pgid = self._proc.pid
        else:
            self._pgid = None

        # Remember the resolved sudo target for the #456-equivalent
        # cross-user signal escalation. _inner_codex_pid is reset to
        # None here so a recycled instance does not carry a stale
        # PID from a previous spawn into the fresh tree.
        self._effective_codex_user = effective_codex_user
        self._inner_codex_pid = None

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
        # Field shape mirrors the goose ACP convention; if the pinned
        # codex version's actual schema differs, the smoke test will
        # surface the error and this call needs adjustment.
        await self._write_rpc(
            "session/new",
            {
                "cwd": str(self.workspace),
            },
        )
        result = await self._read_result(expected_id=2)
        # Accept either camelCase or snake_case to tolerate codex CLI
        # schema variants observed across versions. Both absent is a
        # loud failure: a None session_id would otherwise flow into
        # the next session/prompt as "sessionId": None and surface as
        # a confusing downstream prompt error rather than a clear
        # handshake mismatch. Fail at the boundary instead.
        session_id = result.get("sessionId") or result.get("session_id")
        if not session_id:
            raise RuntimeError(
                "Codex session/new returned no session id "
                "(expected 'sessionId' or 'session_id' in result); "
                "pinned codex CLI schema may differ from this build."
            )
        self._session_id = session_id
        self._fresh_session = True

    async def _drain_stderr(self) -> None:
        """
        Continuously read and discard stderr from the codex process.

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
                    log.debug("Codex stderr: %s", text[:200])
            except Exception:
                log.warning("Unexpected error in Codex stderr drain", exc_info=True)
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

        Discards session/update notifications that codex may emit
        during startup. Raises on JSON-RPC error responses or timeout.
        """
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            try:
                line = await asyncio.wait_for(
                    self._proc.stdout.readline(),
                    timeout=self.timeout_seconds,
                )
            except TimeoutError as exc:
                raise TimeoutError(f"Codex handshake timed out waiting for response id={expected_id}") from exc

            if not line:
                raise RuntimeError("Codex process exited during handshake")

            try:
                msg = json.loads(line.decode())
            except json.JSONDecodeError:
                # Non-JSON output during startup (e.g., progress bars)
                continue

            # Check for matching response
            if msg.get("id") == expected_id:
                if "error" in msg:
                    err = msg["error"]
                    raise RuntimeError(f"Codex error (id={expected_id}): {err.get('message', 'unknown error')}")
                return msg.get("result", {})

            # Discard notifications during handshake

    # ── Sending prompts ────────────────────────────────────────────

    async def send(self, prompt: str | list, chat_id: int | None = None) -> AsyncIterator[StreamEvent]:
        """
        Send a message to codex and yield streaming events.

        Serialized via an internal lock so concurrent callers queue
        rather than interleave. Context injection (identity, memory,
        history, API docs) is prepended on the first message of each
        session, identical to ClaudeCodeBackend and GooseBackend.

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

        Handles context injection, prompt coercion to the JSON-RPC
        content-block format, session/prompt dispatch, and streaming
        response accumulation from session/update notifications.
        """
        try:
            await self._ensure_started()
        except (OSError, RuntimeError, TimeoutError) as exc:
            yield StreamEvent(
                text_so_far="",
                done=True,
                response=AgentResponse(success=False, text="", error=f"Codex startup failed: {exc}"),
            )
            return

        # Inject identity, memory, history, and API context on the
        # first message of a new session. Context injection logic
        # lives in backend.py as shared functions.
        if self._fresh_session:
            self._fresh_session = False
            # Mirror the ClaudeCodeBackend / GooseBackend send() path:
            # ensure the per-user MEMORY.md and PREFERENCES.md surfaces
            # exist before building the session context. The codex
            # backend ships with memory_enabled=False by default, so
            # the MEMORY.md path is created but the subsystem marker
            # in the injected context is "disabled" until backend-
            # agnostic semantic memory lands.
            ensure_user_memory(chat_id, DATA_DIR)
            ensure_user_preferences(chat_id, DATA_DIR)
            session_ctx = build_session_context(
                workspace=self.workspace,
                home_workspace=self.home_workspace,
                api=self._api_context,
                workspace_config=self.workspace_config,
                chat_id=chat_id,
                data_dir=DATA_DIR,
                memory_enabled=self.memory_enabled,
            )
            prompt = prepend_to_prompt(prompt, session_ctx)

        # When in a foreign workspace, remind on every message to only
        # respond to what the user asks.
        reminder = build_foreign_workspace_reminder(self.workspace, self.home_workspace)
        if reminder:
            prompt = prepend_to_prompt(prompt, reminder)

        # Coerce prompt to the JSON-RPC content-block format.
        # The codex CLI accepts text content blocks; image / audio
        # support is deferred until the smoke test confirms which
        # block types the pinned version handles.
        rpc_prompt: list[dict] = []
        if isinstance(prompt, str):
            rpc_prompt = [{"type": "text", "text": prompt}]
        else:
            for block in prompt:
                if block.get("type") == "text":
                    rpc_prompt.append({"type": "text", "text": block["text"]})
                else:
                    log.warning(
                        "CodexBackend: dropping non-text content block type=%s",
                        block.get("type"),
                    )
            if not rpc_prompt:
                rpc_prompt = [{"type": "text", "text": "(empty prompt)"}]

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
                    "prompt": rpc_prompt,
                },
            )
        except OSError as e:
            log.error("Failed to write to Codex process: %s", e)
            await self._kill()
            yield StreamEvent(
                text_so_far="",
                done=True,
                response=AgentResponse(
                    success=False,
                    text="",
                    error="Codex process died, restarting on next message",
                ),
            )
            return

        # Stream response: read stdout lines, accumulate text chunks,
        # and yield StreamEvents. Unknown event types are logged at
        # DEBUG and skipped (schema-drift defense). The smoke test
        # against a real codex binary will reveal which event names
        # the pinned version actually emits; this loop tolerates
        # variation by only acting on events it recognizes.
        accumulated = ""
        last_activity = time.monotonic()
        max_idle_seconds = self.timeout_seconds * 5

        try:
            while True:
                # Check idle timeout before each readline
                idle = time.monotonic() - last_activity
                if idle > max_idle_seconds:
                    log.error(
                        "Codex idle timeout (%.0fs with no output, limit %ds)",
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
                            error="Codex interaction timed out (no output)",
                        ),
                    )
                    return

                try:
                    line = await asyncio.wait_for(
                        self._proc.stdout.readline(),
                        timeout=self.timeout_seconds * 3,
                    )
                except TimeoutError:
                    log.error("Codex response timed out")
                    await self._kill()
                    yield StreamEvent(
                        text_so_far=accumulated,
                        done=True,
                        response=AgentResponse(
                            success=False,
                            text=accumulated,
                            error="Codex timed out",
                        ),
                    )
                    return

                # Reset idle timer on non-empty output
                if line:
                    last_activity = time.monotonic()
                else:
                    # EOF - process died
                    log.error("Codex process EOF")
                    await self._kill()
                    yield StreamEvent(
                        text_so_far=accumulated,
                        done=True,
                        response=AgentResponse(
                            success=bool(accumulated),
                            text=accumulated,
                            error=None if accumulated else "Codex process ended unexpectedly",
                        ),
                    )
                    return

                try:
                    msg = json.loads(line.decode())
                except json.JSONDecodeError:
                    log.debug("Skipping non-JSON line: %s", line[:200])
                    continue

                # Streaming notifications (no id field) - accumulate
                # message chunks; log everything else at DEBUG and
                # skip. The agent_message_chunk envelope mirrors goose
                # ACP; the smoke test confirms or corrects this name.
                if msg.get("method") == "session/update":
                    update = msg.get("params", {}).get("update", {})
                    event_type = update.get("sessionUpdate") or update.get("event") or update.get("type")
                    if event_type == "agent_message_chunk":
                        text = update.get("content", {}).get("text", "")
                        if text:
                            accumulated += text
                            yield StreamEvent(text_so_far=accumulated)
                    else:
                        # Unknown event - log and skip. Schema-drift
                        # defense: a future codex version emitting a
                        # new event type (e.g. tool_call_update) does
                        # not break the conversational stream.
                        log.debug("Codex: skipping session/update event=%s", event_type)

                # Final result for our prompt (has matching id)
                elif msg.get("id") == prompt_id and "result" in msg:
                    response = AgentResponse(
                        success=True,
                        text=accumulated,
                        session_id=self._session_id,
                        cost_usd=0.0,  # subscription auth; codex reports no per-call cost
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
                    err = msg["error"].get("message", "unknown codex error")
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

                else:
                    # Unknown top-level message shape. Schema-drift
                    # defense: log and skip rather than abort.
                    log.debug("Codex: skipping unrecognized message id=%s method=%s", msg.get("id"), msg.get("method"))

        except Exception as e:
            log.exception("Unexpected error reading Codex stream")
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

    def _lookup_inner_codex_pid(self) -> int | None:
        """
        Find the inner codex subprocess PID in cross-user mode.

        In cross-user mode the bot spawns `sudo -u <target> -- codex
        app-server` and self._proc tracks the sudo wrapper. sudo
        fork+exec's codex as its sole child, so pgrep -P <sudo_pid>
        returns the codex PID. Returns None when no sudo wrapper is
        alive, sudo has not yet forked, or pgrep fails. The result
        is cached on self._inner_codex_pid; callers that need to
        signal after sudo has been killed must rely on the cache.
        Synchronous variant used by the sync force_kill path.
        Mirrors claude.py's _lookup_inner_claude_pid (#456/#459).
        """
        if self._proc is None:
            return None
        try:
            result = subprocess.run(
                ["pgrep", "-P", str(self._proc.pid)],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        if result.returncode != 0:
            return None
        first_line = result.stdout.strip().split("\n", 1)[0] if result.stdout.strip() else ""
        if not first_line:
            return None
        try:
            return int(first_line)
        except ValueError:
            return None

    async def _async_lookup_inner_codex_pid(self) -> int | None:
        """
        Async pgrep equivalent of `_lookup_inner_codex_pid`.

        Same semantics, but uses asyncio.create_subprocess_exec so
        the event loop is not blocked while pgrep runs. Called from
        the async _kill / shutdown paths; the sync variant remains
        for force_kill. 2s ceiling matches the sync version. Mirrors
        claude.py's _async_lookup_inner_claude_pid (#459).
        """
        if self._proc is None:
            return None
        try:
            proc = await asyncio.create_subprocess_exec(
                "pgrep",
                "-P",
                str(self._proc.pid),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except (OSError, FileNotFoundError):
            return None
        try:
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=2)
        except TimeoutError:
            # pgrep hung; reap it so we do not leak a child. The
            # kill() guard handles the race where pgrep exited
            # between wait_for timing out and the kill() call:
            # ProcessLookupError is a benign already-dead signal.
            try:
                proc.kill()
            except OSError:
                pass
            try:
                await proc.wait()
            except OSError:
                pass
            return None
        if proc.returncode != 0:
            return None
        first_line = stdout_bytes.decode(errors="replace").strip().split("\n", 1)[0]
        if not first_line:
            return None
        try:
            return int(first_line)
        except ValueError:
            return None

    async def _async_sudo_kill(self, target_user: str, pid: int, sig: int) -> None:
        """
        Run `sudo -n -u <target> /bin/kill -<sig> <pid>` without blocking.

        Equivalent to the synchronous subprocess.run inside _send_signal,
        but uses asyncio.create_subprocess_exec so a hung sudo or kill
        does not stall other coroutines. Logs at WARNING on non-zero
        exit and on timeout so a missing sudoers rule or other failure
        mode surfaces in the operator log instead of silently leaking
        an orphan. Mirrors claude.py's _async_sudo_kill (#459).
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "sudo",
                "-n",
                "-u",
                target_user,
                "/bin/kill",
                f"-{int(sig)}",
                str(pid),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (OSError, FileNotFoundError):
            return
        try:
            _, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=5)
        except TimeoutError:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                await proc.wait()
            except OSError:
                pass
            log.warning(
                "async sudo kill timed out after 5s (target=%s, pid=%d, sig=%d); inner codex may orphan",
                target_user,
                pid,
                int(sig),
            )
            return
        if proc.returncode != 0:
            log.warning(
                "sudo kill escalation failed (rc=%d, stderr=%r); inner codex pid=%d may orphan",
                proc.returncode,
                stderr_bytes[:200].decode(errors="replace") if stderr_bytes else "",
                pid,
            )

    def _send_signal(self, sig: int) -> None:
        """
        Send a signal to the codex process (or process group + sudo
        escalation when running cross-user).

        Cross-user mode: os.killpg from the service user cannot signal
        a target-user codex grandchild because POSIX signal permission
        rules require real or effective UID to match real or saved
        set-user-ID of the target. killpg succeeds for the sudo
        wrapper (real UID is the service user) but the inner codex
        (UIDs = target) survives and gets orphaned to init. We
        escalate via `sudo -n -u <target> /bin/kill -<sig> <pid>` for
        the inner codex BEFORE killpg reaps the sudo wrapper.
        install.py's _generate_sudoers emits a per-target
        NOPASSWD: /bin/kill rule for this path.

        Mirrors claude.py's _send_signal (#456/#458). Deliberately
        does NOT check self._proc.returncode: when codex_user is set
        self._proc tracks the sudo wrapper, not codex, and checking
        returncode would skip delivery if sudo exited first.
        """
        if self._pgid is not None:
            if self._effective_codex_user is not None:
                if self._inner_codex_pid is None:
                    self._inner_codex_pid = self._lookup_inner_codex_pid()
                if self._inner_codex_pid is not None:
                    try:
                        # Absolute /bin/kill path so sudo's secure_path
                        # resolution cannot silently pick a different
                        # binary than the sudoers rule names. The
                        # install.py rule also pins /bin/kill; the two
                        # must match exactly. int(sig) explicit because
                        # IntEnum f-string behavior changed in 3.11;
                        # cast is the contract regardless of version.
                        result = subprocess.run(
                            [
                                "sudo",
                                "-n",
                                "-u",
                                self._effective_codex_user,
                                "/bin/kill",
                                f"-{int(sig)}",
                                str(self._inner_codex_pid),
                            ],
                            capture_output=True,
                            timeout=5,
                            check=False,
                        )
                        if result.returncode != 0:
                            log.warning(
                                "sudo kill escalation failed (rc=%d, stderr=%r); inner codex may orphan",
                                result.returncode,
                                result.stderr[:200].decode(errors="replace") if result.stderr else "",
                            )
                    except (subprocess.TimeoutExpired, OSError):
                        # Inner codex already dead, sudoers rule
                        # missing on an old install, or kill timed
                        # out. Fall through to killpg; the wrapper
                        # still needs reaping.
                        pass
            try:
                os.killpg(self._pgid, sig)
            except OSError:
                pass
        elif self._proc is not None:
            try:
                self._proc.send_signal(sig)
            except OSError:
                pass

    async def _async_send_signal_for_close(self, sig: int) -> None:
        """
        Async equivalent of `_send_signal` used by `_kill` and `shutdown`.

        Same three-step structure as the sync version: prime the
        inner-codex-PID cache (async pgrep), signal the inner codex
        through sudo (async subprocess), then reap the wrapper via
        os.killpg (a non-blocking syscall, sync is fine). The
        single-user fallback uses _proc.send_signal (also non-
        blocking). Mirrors claude.py's _async_send_signal_for_close.
        """
        if self._pgid is not None:
            if self._effective_codex_user is not None:
                if self._inner_codex_pid is None:
                    self._inner_codex_pid = await self._async_lookup_inner_codex_pid()
                if self._inner_codex_pid is not None:
                    await self._async_sudo_kill(
                        self._effective_codex_user,
                        self._inner_codex_pid,
                        int(sig),
                    )
            try:
                os.killpg(self._pgid, sig)
            except OSError:
                pass
        elif self._proc is not None:
            try:
                self._proc.send_signal(sig)
            except OSError:
                pass

    def force_kill(self) -> None:
        """
        Kill the subprocess immediately via SIGKILL.

        Safe to call without holding the lock. Called by /stop to abort
        an in-flight response. Cross-user mode goes through
        _send_signal so the codex grandchild is sudo-escalated before
        the wrapper is killpg'd, avoiding the #456 orphan leak.
        Single-user mode falls back to _proc.kill() via the
        _send_signal else-branch. Does NOT null _proc - full cleanup
        happens in _kill() when the read loop detects EOF.
        """
        if self._proc is not None:
            self._send_signal(signal.SIGKILL)
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None

    async def _kill(self) -> None:
        """
        Kill the subprocess and clean up all process state.

        Sends SIGKILL via the async escalation path (so cross-user
        installs do not orphan the codex grandchild), waits up to 5
        seconds for exit, then nulls all process references.
        Idempotent.
        """
        if self._proc:
            await self._async_send_signal_for_close(signal.SIGKILL)
            if self._stderr_task:
                self._stderr_task.cancel()
                self._stderr_task = None
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except TimeoutError:
                pass
            self._proc = None
            self._session_id = None
            self._pgid = None
            self._inner_codex_pid = None
            self._effective_codex_user = None

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

        Kills the current subprocess. The next send() restarts codex
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
        Graceful shutdown. No save prompt - codex CLI has no equivalent.

        Sends SIGTERM first via the async escalation path so the
        target-user codex grandchild gets sudo-signalled before the
        wrapper is reaped; waits up to 5 seconds for clean exit.
        Falls back to SIGKILL (also through the escalation path) if
        the process doesn't terminate in time.
        """
        if self._proc:
            await self._async_send_signal_for_close(signal.SIGTERM)
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except TimeoutError:
                await self._async_send_signal_for_close(signal.SIGKILL)
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=5)
                except TimeoutError:
                    log.warning("CodexBackend: process did not exit after SIGKILL")
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None
        self._proc = None
        self._session_id = None
        self._pgid = None
        self._inner_codex_pid = None
        self._effective_codex_user = None
