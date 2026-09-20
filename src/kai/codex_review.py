"""Bounded Codex app-server transport for automated review jobs.

The conversational Codex backend already proves the app-server JSON-RPC path
against each principal's OAuth state. Reviews use the same transport in a
short-lived process, but never reuse a conversational thread: the job starts a
read-only thread in Kai's neutral one-shot directory, runs one turn, deletes
that thread, and closes the process.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import kai
from kai.acp import _kill_target_user_tree
from kai.config import DATA_DIR, resolve_claude_user
from kai.oneshot import (
    _CODEX_ENV_ALLOWLIST,
    _CODEX_ONESHOT_DISABLED_FEATURES,
    _CODEX_ONESHOT_PERMISSION_PROFILE,
    _CODEX_PRESERVED_AUTH_VARS,
    _EXTRACTOR_CWD,
    OneShotOutputError,
    OneShotResult,
    OneShotRoutingError,
    OneShotSubprocessError,
    OneShotTimeout,
    _os_user_log_field,
    _render_codex_stdin,
    _wrap_cmd_for_user,
)
from kai.oneshot_binary import BinaryResolutionError, resolve_oneshot_binary
from kai.subprocess_identity import subprocess_spawn_cwd

log = logging.getLogger(__name__)

_STDERR_LIMIT = 64 * 1024
_PROCESS_EXIT_TIMEOUT_SECONDS = 2.0
_THREAD_DELETE_TIMEOUT_SECONDS = 2.0

# The conversational app-server transport succeeds without forcing an upstream
# Responses transport. Keep that proven behavior. The remaining feature
# switches remove tool surfaces from this bounded renderer.
_CODEX_REVIEW_DISABLED_FEATURES = tuple(
    feature
    for feature in _CODEX_ONESHOT_DISABLED_FEATURES
    if feature not in {"responses_websockets", "responses_websockets_v2"}
)


def _backend_error(message: str, payload: dict[str, Any] | None = None) -> OneShotSubprocessError:
    """Represent an app-server request/turn failure on the shared error surface."""
    rendered = message.strip() or "Codex app-server request failed"
    stdout = b"" if payload is None else json.dumps(payload, sort_keys=True).encode("utf-8")
    return OneShotSubprocessError(returncode=1, stderr=rendered.encode("utf-8"), stdout=stdout)


class _CodexReviewProtocol:
    """Minimal JSON-RPC client for one initialized app-server connection."""

    def __init__(self, proc: asyncio.subprocess.Process, stderr_buffer: bytearray) -> None:
        self._proc = proc
        self._stderr_buffer = stderr_buffer
        self._next_id = 1
        self.thread_id: str | None = None

    async def _write(self, message: dict[str, Any]) -> None:
        stdin = self._proc.stdin
        if stdin is None:
            raise _backend_error("Codex app-server stdin is unavailable")
        stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        await stdin.drain()

    async def _read(self) -> dict[str, Any]:
        stdout = self._proc.stdout
        if stdout is None:
            raise _backend_error("Codex app-server stdout is unavailable")
        line = await stdout.readline()
        if not line:
            returncode = self._proc.returncode
            if returncode is None:
                returncode = await self._proc.wait()
            raise OneShotSubprocessError(
                returncode=returncode if returncode is not None else -1,
                stderr=bytes(self._stderr_buffer),
            )
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OneShotOutputError(f"Codex app-server emitted invalid JSON: {exc}") from None
        if not isinstance(message, dict):
            raise OneShotOutputError("Codex app-server emitted a non-object JSON message")
        return message

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        await self._write({"method": method, "params": params or {}})

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        await self._write({"method": method, "id": request_id, "params": params})
        while True:
            message = await self._read()
            if message.get("id") != request_id:
                continue
            error = message.get("error")
            if isinstance(error, dict):
                raise _backend_error(str(error.get("message") or f"Codex {method} failed"), message)
            result = message.get("result")
            if not isinstance(result, dict):
                raise OneShotOutputError(f"Codex {method} returned a malformed result")
            return result

    async def initialize(self, *, cwd: Path, model: str | None) -> None:
        await self.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "kai_review_job",
                    "title": "Kai review job",
                    "version": kai.__version__,
                },
                "capabilities": {
                    "optOutNotificationMethods": [
                        "remoteControl/status/changed",
                        "mcpServer/startupStatus/updated",
                        "thread/started",
                        "thread/tokenUsage/updated",
                        "item/agentMessage/delta",
                    ]
                },
            },
        )
        await self.notify("initialized")
        # The process-level `default_permissions` selects Kai's shared
        # bounded one-shot profile. Do not also send either legacy `sandbox`
        # or turn-level `sandboxPolicy`: profiles do not compose with those
        # fields, and Codex 0.147 rejects the former restricted-read shape.
        thread_params: dict[str, Any] = {
            "cwd": str(cwd),
            "approvalPolicy": "never",
            "serviceName": "kai_review_job",
            "config": {
                "project_doc_max_bytes": 0,
                "project_doc_fallback_filenames": [],
            },
        }
        if model:
            thread_params["model"] = model
        result = await self.request("thread/start", thread_params)
        thread = result.get("thread")
        if not isinstance(thread, dict) or not isinstance(thread.get("id"), str):
            raise OneShotOutputError("Codex thread/start returned no thread.id")
        self.thread_id = thread["id"]

    async def run_turn(self, *, prompt: str, cwd: Path, model: str | None, join_items: bool) -> str:
        if self.thread_id is None:
            raise OneShotOutputError("Codex review thread was not initialized")
        request_id = self._next_id
        self._next_id += 1
        params: dict[str, Any] = {
            "threadId": self.thread_id,
            "input": [{"type": "text", "text": prompt}],
            "cwd": str(cwd),
            "approvalPolicy": "never",
        }
        if model:
            params["model"] = model
        await self._write({"method": "turn/start", "id": request_id, "params": params})

        visible: list[str] = []
        commentary: list[str] = []
        while True:
            message = await self._read()
            if message.get("id") == request_id and isinstance(message.get("error"), dict):
                error = message["error"]
                raise _backend_error(str(error.get("message") or "Codex turn/start failed"), message)

            method = message.get("method")
            if method == "item/completed":
                item = message.get("params", {}).get("item", {})
                if isinstance(item, dict) and item.get("type") == "agentMessage":
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        if item.get("phase") == "commentary":
                            commentary.append(text)
                        else:
                            visible.append(text)
                continue

            if method == "error":
                error = message.get("params", {}).get("error", {})
                detail = error.get("message") if isinstance(error, dict) else None
                raise _backend_error(str(detail or "Codex app-server turn failed"), message)

            if method == "turn/completed":
                turn = message.get("params", {}).get("turn", {})
                status = turn.get("status") if isinstance(turn, dict) else None
                if status != "completed":
                    error = turn.get("error") if isinstance(turn, dict) else None
                    detail = error.get("message") if isinstance(error, dict) else None
                    raise _backend_error(str(detail or f"Codex turn ended with status={status}"), message)
                break

        messages = visible or commentary
        if not messages:
            raise OneShotOutputError("Codex app-server produced no final agent message")
        return "\n\n".join(messages) if join_items else messages[-1]

    async def delete_thread(self) -> None:
        if self.thread_id is None:
            return
        thread_id = self.thread_id
        self.thread_id = None
        await self.request("thread/delete", {"threadId": thread_id})


class CodexAppServerReviewReasoner:
    """One-turn, read-only Codex renderer for automated PR review jobs."""

    def __init__(self, *, cwd: Path | None = None, os_user: str | None = None, join_items: bool = True) -> None:
        self._cwd = cwd if cwd is not None else _EXTRACTOR_CWD
        self._os_user = os_user
        self._join_items = join_items

    async def run(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        purpose: str,
        json_schema: dict[str, Any] | None = None,
    ) -> OneShotResult:
        if json_schema is not None:
            raise OneShotRoutingError("Codex app-server review calls do not accept a JSON schema")
        self._cwd.mkdir(parents=True, exist_ok=True)
        self._cwd.chmod(0o755)
        try:
            resolved_binary = resolve_oneshot_binary("codex")
        except BinaryResolutionError as exc:
            raise OneShotRoutingError(str(exc)) from exc

        cmd = [
            resolved_binary,
            "--config",
            'approval_policy="never"',
            # Reuse the exec reasoner's canonical permission profile rather
            # than maintaining a second app-server protocol representation.
            "--config",
            'default_permissions="kai-oneshot"',
            "--config",
            _CODEX_ONESHOT_PERMISSION_PROFILE,
            "--config",
            'web_search="disabled"',
            "--config",
            "mcp_servers={}",
        ]
        for feature in _CODEX_REVIEW_DISABLED_FEATURES:
            cmd.extend(["--disable", feature])
        cmd.append("app-server")

        effective_user = resolve_claude_user(self._os_user)
        subprocess_env = {key: os.environ[key] for key in _CODEX_ENV_ALLOWLIST if key in os.environ}
        if effective_user is not None:
            subprocess_env["TMPDIR"] = str(DATA_DIR / "tmp" / effective_user)
            cmd = _wrap_cmd_for_user(
                cmd,
                effective_user,
                "codex",
                working_directory=self._cwd,
                preserve_vars=(*_CODEX_PRESERVED_AUTH_VARS, "TMPDIR"),
            )

        start = time.monotonic()
        stderr_buffer = bytearray()
        proc: asyncio.subprocess.Process | None = None
        stderr_task: asyncio.Task[None] | None = None
        protocol: _CodexReviewProtocol | None = None
        os_user_field = _os_user_log_field(effective_user)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=subprocess_spawn_cwd(self._cwd, target_user=effective_user),
                env=subprocess_env,
                start_new_session=bool(effective_user),
                limit=16 * 1024 * 1024,
            )
            stderr_task = asyncio.create_task(self._drain_stderr(proc, stderr_buffer))
            protocol = _CodexReviewProtocol(proc, stderr_buffer)
            rendered_prompt = _render_codex_stdin(system_prompt, prompt)

            async def exchange() -> str:
                await protocol.initialize(cwd=self._cwd, model=model)
                try:
                    return await protocol.run_turn(
                        prompt=rendered_prompt,
                        cwd=self._cwd,
                        model=model,
                        join_items=self._join_items,
                    )
                finally:
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(
                            protocol.delete_thread(),
                            timeout=_THREAD_DELETE_TIMEOUT_SECONDS,
                        )

            try:
                final_text = await asyncio.wait_for(exchange(), timeout=timeout)
            except TimeoutError:
                duration_ms = int((time.monotonic() - start) * 1000)
                log.info(
                    "oneshot_reasoner purpose=%s backend=codex transport=app_server model=%s "
                    "duration_ms=%d outcome=timeout error_category=timeout os_user=%s",
                    purpose,
                    model,
                    duration_ms,
                    os_user_field,
                )
                raise OneShotTimeout() from None

            duration_ms = int((time.monotonic() - start) * 1000)
            log.info(
                "oneshot_reasoner purpose=%s backend=codex transport=app_server model=%s "
                "duration_ms=%d outcome=success returncode=0 os_user=%s",
                purpose,
                model,
                duration_ms,
                os_user_field,
            )
            return OneShotResult(
                text=final_text,
                backend="codex",
                model=model,
                raw_metadata={
                    "returncode": 0,
                    "stderr": bytes(stderr_buffer),
                    "cwd": str(self._cwd),
                    "cmd": list(cmd),
                    "resolved_binary": resolved_binary,
                    "transport": "app_server",
                },
                duration_ms=duration_ms,
            )
        finally:
            if proc is not None:
                await self._stop_process(proc, effective_user=effective_user, purpose=purpose)
            if stderr_task is not None:
                if not stderr_task.done():
                    stderr_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                    await asyncio.wait_for(stderr_task, timeout=1.0)

    @staticmethod
    async def _drain_stderr(proc: asyncio.subprocess.Process, buffer: bytearray) -> None:
        stderr = proc.stderr
        if stderr is None:
            return
        while True:
            line = await stderr.readline()
            if not line:
                return
            buffer.extend(line)
            if len(buffer) > _STDERR_LIMIT:
                del buffer[:-_STDERR_LIMIT]

    @staticmethod
    async def _stop_process(
        proc: asyncio.subprocess.Process,
        *,
        effective_user: str | None,
        purpose: str,
    ) -> None:
        stdin = proc.stdin
        if stdin is not None:
            stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), timeout=_PROCESS_EXIT_TIMEOUT_SECONDS)
            return
        except TimeoutError:
            pass
        if effective_user is not None:
            await _kill_target_user_tree(
                target_user=effective_user,
                pgid=proc.pid,
                purpose=purpose,
                backend="codex",
            )
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=_PROCESS_EXIT_TIMEOUT_SECONDS)
