"""Persistent conversational backend for Pi's JSONL RPC mode."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

from kai.acp import _kill_target_user_tree, _kill_target_user_tree_sync
from kai.backend import (
    TRACE_DETAIL_MAX_CHARS,
    TRACE_SUMMARY_MAX_CHARS,
    AgentBackend,
    AgentResponse,
    AgentRuntimeIdentity,
    ApiContext,
    StreamEvent,
    TraceEntry,
    apply_workspace_model,
    assemble_turn_context,
    build_foreign_workspace_reminder,
    build_session_context,
    ensure_user_context_files,
    sanitize_agent_environment,
    scrub_trace_text,
    trace_secret_values,
)
from kai.backend_registry import resolve_backend_command
from kai.config import DATA_DIR, WorkspaceConfig, parse_env_file, resolve_claude_user
from kai.pi_rpc import (
    PI_RPC_STREAM_LIMIT,
    PiRpcError,
    PiRpcProtocolError,
    PiRpcTimeoutError,
    PiRpcTransport,
    pi_rpc_extension_error,
    pi_rpc_is_settled,
    pi_rpc_text_delta,
    require_pi_rpc_response,
)
from kai.subprocess_identity import subprocess_spawn_cwd, wrap_command_for_target_user

log = logging.getLogger(__name__)

_PI_THINKING_LEVELS = frozenset({"off", "minimal", "low", "medium", "high", "xhigh"})
_PI_PROVIDER_ENV_VARS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_OAUTH_TOKEN"),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "ollama": ("OLLAMA_HOST",),
    "openai": ("OPENAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    # Subscription credentials for these providers live in the target
    # user's ~/.pi/agent/auth.json and therefore need no environment key.
    "openai-codex": (),
    "github-copilot": (),
}
_PI_ALL_PROVIDER_ENV_VARS = frozenset(name for names in _PI_PROVIDER_ENV_VARS.values() for name in names)


def _pi_provider_env_vars(provider: str) -> tuple[str, ...]:
    """Return only the environment credentials the selected provider needs."""

    return _PI_PROVIDER_ENV_VARS.get(provider, ())


def _split_pi_model(model: str, provider: str = "") -> tuple[str, str, str | None]:
    """Return provider, model id, and optional thinking suffix.

    Pi accepts either ``provider/model`` or a bare model when ``--provider``
    is supplied separately.  Kai always has an effective provider at runtime,
    so both documented forms remain usable from ``/model``.
    """

    model_provider, separator, model_id = model.partition("/")
    if separator:
        if not model_provider or not model_id:
            raise ValueError("Pi models must use provider/model shape or a bare model with an explicit provider")
        if provider and model_provider != provider:
            raise ValueError(f"Pi model provider {model_provider!r} does not match configured provider {provider!r}")
        provider = model_provider
    else:
        model_id = model_provider
        if not provider or not model_id:
            raise ValueError("Pi models must use provider/model shape or a bare model with an explicit provider")
    thinking: str | None = None
    base, suffix_separator, suffix = model_id.rpartition(":")
    if suffix_separator and base and suffix in _PI_THINKING_LEVELS:
        model_id = base
        thinking = suffix
    return provider, model_id, thinking


def _assistant_message_text(message: Any) -> str | None:
    """Extract authoritative visible text from a completed Pi message."""

    if not isinstance(message, Mapping) or message.get("role") != "assistant":
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if not isinstance(block, Mapping) or block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _tool_result_text(result: Any) -> str | None:
    """
    Join the text content blocks of a pi tool result, or None when the
    shape carries no identifiable content list (the per-tool `details`
    object is not surfaced; the text blocks are the human-readable
    output).
    """
    if not isinstance(result, Mapping):
        return None
    content = result.get("content")
    if not isinstance(content, list):
        return None
    return "\n".join(
        part.get("text", "") for part in content if isinstance(part, Mapping) and part.get("type") == "text"
    )


def _assistant_message_error(message: Any) -> str | None:
    """Extract a terminal model error from a completed assistant message."""

    if not isinstance(message, Mapping) or message.get("role") != "assistant":
        return None
    stop_reason = message.get("stopReason")
    error_message = message.get("errorMessage")
    if stop_reason == "error":
        return error_message if isinstance(error_message, str) and error_message else "Pi model request failed"
    return None


class PiBackend(AgentBackend):
    """A persistent Pi process using the documented JSONL RPC protocol."""

    backend_name = "pi"

    def __init__(
        self,
        *,
        model: str,
        workspace: Path,
        home_workspace: Path | None = None,
        webhook_port: int = 8080,
        webhook_secret: str = "",
        timeout_seconds: int = 300,
        services_info: list[dict] | None = None,
        workspace_config: WorkspaceConfig | None = None,
        provider: str,
        memory_enabled: bool = False,
        os_user: str | None = None,
        max_session_hours: float = 0,
        defer_user_file_reads: bool = False,
        # Total bound on one turn, in seconds. The per-receive timeout
        # resets whenever any RPC record arrives; this backstop never
        # does. Validated positive at config load (see
        # Config.turn_deadline_seconds).
        turn_deadline_seconds: int = 3600,
    ) -> None:
        self.model = model
        self.workspace = workspace
        self.home_workspace = home_workspace or workspace
        self.timeout_seconds = timeout_seconds
        self.workspace_config = workspace_config
        self.provider = provider
        self.memory_enabled = memory_enabled
        self.os_user = os_user
        self.max_session_hours = max_session_hours
        self.defer_user_file_reads = defer_user_file_reads
        self.turn_deadline_seconds = turn_deadline_seconds

        self._api_context = ApiContext(
            webhook_port=webhook_port,
            webhook_secret=webhook_secret,
            services_info=services_info or [],
        )
        self._default_model = model
        self._default_timeout = timeout_seconds
        if workspace_config:
            self.model = apply_workspace_model(workspace_config, self.backend_name, self.provider, self.model)
            if workspace_config.timeout is not None:
                self.timeout_seconds = workspace_config.timeout

        self._proc: asyncio.subprocess.Process | None = None
        self._transport: PiRpcTransport | None = None
        self._session_id: str | None = None
        self._session_started_at: float | None = None
        self._effective_os_user: str | None = None
        self._pgid: int | None = None
        self._supports_image_input = False
        self._next_id = 1
        self._fresh_session = True
        self._stderr_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        # Serializes _kill (shutdown delegates to it). Teardown paths
        # await between reading process state and using it; two
        # teardowns interleaving across those awaits would let the
        # second read attributes the first already nulled. Always
        # acquired after _lock when both are held, never before it.
        # The sync force_kill stays outside this gate: it cannot be
        # interleaved mid-function, and nothing it nulls is re-read
        # unguarded by a gated body afterwards.
        self._teardown_lock = asyncio.Lock()
        # Secret values scrubbed from trace text. Snapshotted from the
        # fully built subprocess environment at spawn so constructor-only
        # credentials (the per-principal webhook secret) are covered.
        self._trace_secrets: tuple[str, ...] = ()

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def _new_request_id(self, command: str) -> str:
        request_id = f"kai-{command}-{self._next_id}"
        self._next_id += 1
        return request_id

    def _build_argv(self) -> list[str]:
        command = resolve_backend_command("pi", allow_bare_fallback=True)
        return [
            command,
            "--mode",
            "rpc",
            "--provider",
            self.provider,
            "--model",
            self.model,
            "--no-approve",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-context-files",
        ]

    def _build_env(self, effective_os_user: str | None) -> dict[str, str]:
        env = os.environ.copy()
        if self.workspace_config:
            if self.workspace_config.env_file:
                env.update(parse_env_file(self.workspace_config.env_file))
            if self.workspace_config.env:
                env.update(self.workspace_config.env)
        env = sanitize_agent_environment(env)
        selected_provider_vars = frozenset(_pi_provider_env_vars(self.provider))
        for name in _PI_ALL_PROVIDER_ENV_VARS - selected_provider_vars:
            env.pop(name, None)
        if self._api_context.webhook_secret:
            env["KAI_WEBHOOK_SECRET"] = self._api_context.webhook_secret
        if effective_os_user:
            env["TMPDIR"] = str(DATA_DIR / "tmp" / effective_os_user)
        return env

    async def _ensure_started(self) -> None:
        if self.is_alive:
            return

        effective_os_user = resolve_claude_user(self.os_user)
        argv = self._build_argv()
        env = self._build_env(effective_os_user)
        self._trace_secrets = trace_secret_values(env)
        if effective_os_user:
            argv = wrap_command_for_target_user(
                argv,
                target_user=effective_os_user,
                working_directory=self.workspace,
                preserve_env=("KAI_WEBHOOK_SECRET", "TMPDIR", *_pi_provider_env_vars(self.provider)),
            )

        log.info(
            "Starting persistent Pi RPC process (model=%s, user=%s)",
            self.model,
            effective_os_user or "(same as bot)",
        )
        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=subprocess_spawn_cwd(
                self.workspace,
                target_user=effective_os_user,
            ),
            env=env,
            limit=PI_RPC_STREAM_LIMIT,
            start_new_session=bool(effective_os_user),
        )
        self._effective_os_user = effective_os_user
        self._pgid = self._proc.pid if effective_os_user else None
        self._session_started_at = time.monotonic()
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        self._transport = PiRpcTransport(self._proc.stdin, self._proc.stdout)
        self._next_id = 1

        state = await self._command("get_state")
        if not isinstance(state, Mapping):
            raise PiRpcProtocolError("Pi get_state response requires an object")
        session_id = state.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            raise PiRpcProtocolError("Pi get_state response requires a sessionId")
        active_model = state.get("model")
        if not isinstance(active_model, Mapping):
            raise PiRpcProtocolError(f"Pi did not activate configured model {self.model!r}")
        expected_provider, expected_model_id, expected_thinking = _split_pi_model(self.model, self.provider)
        if active_model.get("provider") != expected_provider or active_model.get("id") != expected_model_id:
            raise PiRpcProtocolError(
                "Pi activated a different model than configured: "
                f"expected {expected_provider}/{expected_model_id}, "
                f"got {active_model.get('provider')}/{active_model.get('id')}"
            )
        if expected_thinking is not None and state.get("thinkingLevel") != expected_thinking:
            raise PiRpcProtocolError(
                f"Pi did not activate configured thinking level {expected_thinking!r}; "
                f"got {state.get('thinkingLevel')!r}"
            )
        inputs = active_model.get("input")
        self._supports_image_input = isinstance(inputs, list) and "image" in inputs
        self._session_id = session_id
        self._fresh_session = True

    async def _command(self, command: str, **params: Any) -> Any:
        if self._transport is None:
            raise PiRpcProtocolError("Pi RPC transport is not started")
        request_id = self._new_request_id(command)
        await self._transport.send({"id": request_id, "type": command, **params})
        while True:
            message = await self._transport.receive(timeout_seconds=self.timeout_seconds)
            if message.get("type") != "response":
                raise PiRpcProtocolError(
                    f"Pi emitted unexpected {message.get('type')!r} event while waiting for {command!r}"
                )
            if message.get("id") != request_id:
                raise PiRpcProtocolError(f"Pi emitted response for unexpected request id {message.get('id')!r}")
            return require_pi_rpc_response(message, request_id=request_id, command=command)

    async def _drain_stderr(self) -> None:
        while self._proc is not None and self._proc.stderr is not None:
            try:
                line = await self._proc.stderr.readline()
            except (ConnectionError, RuntimeError):
                return
            if not line:
                return
            text = line.decode(errors="replace").strip()
            if text:
                log.debug("Pi stderr: %s", text[:500])

    async def send(
        self,
        prompt: str | list,
        chat_id: int | None = None,
        *,
        runtime_identity: AgentRuntimeIdentity | None = None,
    ) -> AsyncIterator[StreamEvent]:
        async with self._lock:
            async for event in self._send_locked(prompt, chat_id, runtime_identity=runtime_identity):
                yield event

    async def _send_locked(
        self,
        prompt: str | list,
        chat_id: int | None,
        *,
        runtime_identity: AgentRuntimeIdentity | None = None,
    ) -> AsyncIterator[StreamEvent]:
        started_at = time.monotonic()
        if self._should_recycle():
            log.info(
                "Pi session age %.1f hours exceeds limit of %.1f hours; recycling",
                self._session_age_hours(),
                self.max_session_hours,
            )
            await self._kill()

        try:
            await self._ensure_started()
        except (OSError, ValueError, PiRpcError) as exc:
            await self._kill()
            yield self._final_event("", started_at, error=f"Pi startup failed: {exc}")
            return

        session_context = ""
        if self._fresh_session:
            self._fresh_session = False
            ensure_user_context_files(
                chat_id,
                DATA_DIR,
                defer_user_file_reads=self.defer_user_file_reads,
            )
            session_context = build_session_context(
                workspace=self.workspace,
                home_workspace=self.home_workspace,
                api=self._api_context,
                workspace_config=self.workspace_config,
                chat_id=chat_id,
                runtime_identity=runtime_identity,
                data_dir=DATA_DIR,
                backend_name=self.backend_name,
                memory_enabled=self.memory_enabled,
                defer_user_file_reads=self.defer_user_file_reads,
                canonical_history=self.consume_canonical_history(),
            )
        reminder = build_foreign_workspace_reminder(self.workspace, self.home_workspace) or ""

        images: list[dict[str, str]] = []
        dropped_images = 0
        had_user_text = isinstance(prompt, str) and bool(prompt.strip())
        if isinstance(prompt, list):
            normalized: list[dict[str, str]] = []
            for block in prompt:
                if not isinstance(block, Mapping):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    normalized.append({"type": "text", "text": block["text"]})
                    if block["text"].strip():
                        had_user_text = True
                    continue
                if block.get("type") == "image":
                    source = block.get("source")
                    if self._supports_image_input and isinstance(source, Mapping):
                        data = source.get("data")
                        mime_type = source.get("media_type")
                        if source.get("type") == "base64" and isinstance(data, str) and isinstance(mime_type, str):
                            images.append({"type": "image", "data": data, "mimeType": mime_type})
                            continue
                    dropped_images += 1
            prompt = normalized or [{"type": "text", "text": "(empty prompt)"}]

        prompt = await assemble_turn_context(
            prompt,
            chat_id=chat_id if had_user_text else None,
            runtime_identity=runtime_identity if had_user_text else None,
            session_context=session_context,
            workspace_reminder=reminder,
            workspace=self.workspace,
            backend_name=self.backend_name,
            job_type="interactive",
            session_id=self._session_id,
        )
        if isinstance(prompt, str):
            message_text = prompt
        else:
            message_text = "\n\n".join(
                block["text"]
                for block in prompt
                if isinstance(block, Mapping) and block.get("type") == "text" and isinstance(block.get("text"), str)
            )
        if not message_text:
            message_text = "(empty prompt)"

        notice = ""
        if dropped_images:
            noun = "image" if dropped_images == 1 else "images"
            notice = (
                f"[Note: {dropped_images} attached {noun} could not be passed to Pi; "
                "this reply is based on the message text only.]\n\n"
            )

        assert self._transport is not None
        request_id = self._new_request_id("prompt")
        command: dict[str, Any] = {"id": request_id, "type": "prompt", "message": message_text}
        if images:
            command["images"] = images
        visible_text = notice
        try:
            await self._transport.send(command)
            async for event in self._read_turn(request_id, notice, started_at):
                visible_text = event.text_so_far
                yield event
        except PiRpcError as exc:
            await self._kill()
            yield self._final_event(visible_text, started_at, error=str(exc))

    async def _read_turn(
        self,
        request_id: str,
        notice: str,
        started_at: float,
    ) -> AsyncIterator[StreamEvent]:
        assert self._transport is not None
        # Local reference: a /stop-shaped teardown nulls self._transport
        # mid-turn, and this loop must keep reading the dead process's
        # buffered records to their end (surfaced as PiRpcEOFError into
        # the standard failure path) instead of crashing on a None
        # re-read.
        transport = self._transport
        accepted = False
        streamed = ""
        completed_messages: list[str] = []
        terminal_error: str | None = None

        while True:
            # Check the total-turn deadline before each receive. The
            # per-receive timeout resets whenever any record arrives,
            # so a turn that trickles retry notices without completing
            # ends here instead; the raise rides the same
            # PiRpcError-to-failed-final-event path as a receive
            # timeout, killing the subprocess in the send handler.
            elapsed = time.monotonic() - started_at
            if elapsed > self.turn_deadline_seconds:
                log.error(
                    "Pi turn deadline exceeded (%.0fs elapsed, limit %ds)",
                    elapsed,
                    self.turn_deadline_seconds,
                )
                raise PiRpcTimeoutError("Pi turn exceeded its deadline")
            message = await transport.receive(timeout_seconds=self.timeout_seconds)
            message_type = message.get("type")

            if message_type == "response":
                if accepted:
                    raise PiRpcProtocolError("Pi emitted more than one response for a prompt")
                if message.get("id") != request_id:
                    raise PiRpcProtocolError(f"Pi emitted response for unexpected request id {message.get('id')!r}")
                require_pi_rpc_response(message, request_id=request_id, command="prompt")
                accepted = True
                continue

            delta = pi_rpc_text_delta(message)
            if delta is not None:
                streamed += delta
                yield StreamEvent(text_so_far=notice + streamed)
                continue

            if message_type == "message_end":
                authoritative = _assistant_message_text(message.get("message"))
                if authoritative:
                    completed_messages.append(authoritative)
                terminal_error = _assistant_message_error(message.get("message")) or terminal_error
                continue

            if message_type == "auto_retry_end" and message.get("success") is False:
                detail = message.get("finalError") or message.get("error")
                terminal_error = detail if isinstance(detail, str) and detail else "Pi exhausted automatic retries"
                continue

            extension_error = pi_rpc_extension_error(message)
            if extension_error is not None:
                raise PiRpcProtocolError(f"Pi extension failed despite extensions being disabled: {extension_error}")

            if message_type in {"tool_execution_start", "tool_execution_update", "tool_execution_end"}:
                log.debug(
                    "Pi tool event type=%s tool=%s call_id=%s",
                    message_type,
                    message.get("toolName"),
                    message.get("toolCallId"),
                )
                # tool_execution_update streams accumulated partial
                # output; the card shows call then result, so progress
                # churn stays invisible, same as the ACP lanes.
                if message_type == "tool_execution_start":
                    trace = self._tool_start_trace(message)
                    if trace is not None:
                        yield StreamEvent(text_so_far="", trace=trace)
                elif message_type == "tool_execution_end":
                    trace = self._tool_end_trace(message)
                    if trace is not None:
                        yield StreamEvent(text_so_far="", trace=trace)
                continue

            if pi_rpc_is_settled(message):
                if not accepted:
                    raise PiRpcProtocolError("Pi settled a prompt before acknowledging it")
                authoritative_text = "\n\n".join(text for text in completed_messages if text)
                final_text = notice + (authoritative_text or streamed)
                yield self._final_event(final_text, started_at, error=terminal_error)
                return

            if message_type == "extension_ui_request":
                raise PiRpcProtocolError("Pi requested extension UI despite extensions being disabled")

    def _tool_start_trace(self, message: Mapping) -> TraceEntry | None:
        """
        Build a tool_call TraceEntry from a tool_execution_start event.

        toolName feeds the summary and the args object the detail; a
        payload whose args cannot be identified carries the raw event
        JSON as detail instead, so no information is lost. Returns None
        with a debug log on a malformed event; trace emission must never
        break the text stream.
        """
        try:
            tool_call_id = message["toolCallId"]
            tool_name = message["toolName"]
            if not isinstance(tool_call_id, str) or not isinstance(tool_name, str):
                raise TypeError("unexpected tool event field type")
            args = message.get("args")
            if isinstance(args, Mapping):
                detail = json.dumps(args, ensure_ascii=False)
            else:
                detail = json.dumps(dict(message), ensure_ascii=False)
            return TraceEntry(
                kind="tool_call",
                tool_use_id=tool_call_id,
                summary=scrub_trace_text(tool_name, self._trace_secrets, TRACE_SUMMARY_MAX_CHARS),
                detail=scrub_trace_text(detail, self._trace_secrets, TRACE_DETAIL_MAX_CHARS),
                tool_name=tool_name,
            )
        except Exception:
            log.debug("Skipping malformed tool_execution_start for trace", exc_info=True)
            return None

    def _tool_end_trace(self, message: Mapping) -> TraceEntry | None:
        """
        Build a tool_result TraceEntry from a tool_execution_end event.

        tool_use_id carries the toolCallId, the same value the tool_call
        trace carried, so client call/result pairing works unchanged.
        The result's text content blocks feed the detail; a result whose
        content cannot be identified carries the raw event JSON instead.
        Returns None with a debug log on a malformed event; trace
        emission must never break the text stream.
        """
        try:
            tool_call_id = message["toolCallId"]
            tool_name = message["toolName"]
            if not isinstance(tool_call_id, str) or not isinstance(tool_name, str):
                raise TypeError("unexpected tool event field type")
            detail = _tool_result_text(message.get("result"))
            if detail is None:
                detail = json.dumps(dict(message), ensure_ascii=False)
            return TraceEntry(
                kind="tool_result",
                tool_use_id=tool_call_id,
                summary=scrub_trace_text(tool_name, self._trace_secrets, TRACE_SUMMARY_MAX_CHARS),
                detail=scrub_trace_text(detail, self._trace_secrets, TRACE_DETAIL_MAX_CHARS),
                is_error=bool(message.get("isError", False)),
            )
        except Exception:
            log.debug("Skipping malformed tool_execution_end for trace", exc_info=True)
            return None

    def _final_event(self, text: str, started_at: float, *, error: str | None) -> StreamEvent:
        response = AgentResponse(
            success=error is None,
            text=text,
            session_id=self._session_id,
            duration_ms=int((time.monotonic() - started_at) * 1000),
            error=error,
        )
        return StreamEvent(text_so_far=text, done=True, response=response)

    def force_kill(self) -> None:
        if self._proc is not None:
            if self._effective_os_user is not None and self._pgid is not None:
                _kill_target_user_tree_sync(
                    target_user=self._effective_os_user,
                    pgid=self._pgid,
                    purpose="chat",
                    backend=self.backend_name,
                )
                self._pgid = None
            try:
                self._proc.kill()
            except OSError:
                pass
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            self._stderr_task = None

    async def _kill(self) -> None:
        async with self._teardown_lock:
            if self._proc is not None:
                if self._effective_os_user is not None and self._pgid is not None:
                    await _kill_target_user_tree(
                        target_user=self._effective_os_user,
                        pgid=self._pgid,
                        purpose="chat",
                        backend=self.backend_name,
                    )
                    self._pgid = None
                self.force_kill()
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=5)
                except TimeoutError:
                    pass
            self._proc = None
            self._transport = None
            self._session_id = None
            self._effective_os_user = None
            self._pgid = None
            self._session_started_at = None
            self._supports_image_input = False

    async def restart(self) -> None:
        await self._kill()
        self._fresh_session = True

    async def change_workspace(
        self,
        new_workspace: Path,
        workspace_config: WorkspaceConfig | None = None,
    ) -> None:
        await self._kill()
        self.workspace = new_workspace
        self.workspace_config = workspace_config
        self.model = self._default_model
        self.timeout_seconds = self._default_timeout
        if workspace_config:
            self.model = apply_workspace_model(workspace_config, self.backend_name, self.provider, self.model)
            if workspace_config.timeout is not None:
                self.timeout_seconds = workspace_config.timeout
        self._fresh_session = True

    async def shutdown(self) -> None:
        # The tracked process may be a sudo wrapper. Killing only that wrapper
        # can orphan the target user's Pi grandchild, so use the same
        # cross-user process-group teardown as restart and error recovery.
        await self._kill()
