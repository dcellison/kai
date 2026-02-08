from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class ClaudeResponse:
    success: bool
    text: str
    session_id: str | None = None
    cost_usd: float = 0.0
    duration_ms: int = 0
    error: str | None = None


@dataclass
class StreamEvent:
    """A partial update during streaming."""
    text_so_far: str
    done: bool = False
    response: ClaudeResponse | None = None  # set when done=True


class PersistentClaude:
    """A long-running Claude process using stream-json I/O for multi-turn chat."""

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
    ):
        self.model = model
        self.workspace = workspace
        self.home_workspace = home_workspace or workspace
        self.webhook_port = webhook_port
        self.webhook_secret = webhook_secret
        self.max_budget_usd = max_budget_usd
        self.timeout_seconds = timeout_seconds
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._session_id: str | None = None
        self._fresh_session = True
        # Background task to drain stderr
        self._stderr_task: asyncio.Task | None = None

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def _ensure_started(self) -> None:
        """Start the claude process if not already running."""
        if self.is_alive:
            return

        cmd = [
            "claude",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--model", self.model,
            "--permission-mode", "bypassPermissions",
            "--max-budget-usd", str(self.max_budget_usd),
        ]
        log.info("Starting persistent Claude process (model=%s)", self.model)

        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workspace),
        )
        self._session_id = None
        self._fresh_session = True

        # Drain stderr in background to prevent buffer deadlock
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self) -> None:
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
        """Send a message and yield streaming events. Serialized via lock."""
        async with self._lock:
            async for event in self._send_locked(prompt):
                yield event

    async def _send_locked(self, prompt: str | list) -> AsyncIterator[StreamEvent]:
        try:
            await self._ensure_started()
        except FileNotFoundError:
            yield StreamEvent(
                text_so_far="", done=True,
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
                    parts.append(f"[Your persistent memory from previous sessions:]\n{memory}")

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

            if parts:
                prefix = "\n\n".join(parts) + "\n\n"
                if isinstance(prompt, str):
                    prompt = prefix + prompt
                elif isinstance(prompt, list):
                    prompt = [{"type": "text", "text": prefix}] + prompt

        content = prompt if isinstance(prompt, list) else [{"type": "text", "text": prompt}]
        msg = json.dumps({
            "type": "user",
            "message": {
                "role": "user",
                "content": content,
            },
        }) + "\n"

        try:
            self._proc.stdin.write(msg.encode())
            await self._proc.stdin.drain()
        except OSError as e:
            log.error("Failed to write to Claude process: %s", e)
            await self._kill()
            yield StreamEvent(
                text_so_far="", done=True,
                response=ClaudeResponse(success=False, text="", error="Claude process died, restarting on next message"),
            )
            return

        accumulated_text = ""
        try:
            while True:
                try:
                    # Opus with tool use can go minutes between output lines
                    timeout = self.timeout_seconds * 3
                    line = await asyncio.wait_for(
                        self._proc.stdout.readline(), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    log.error("Claude response timed out")
                    await self._kill()
                    yield StreamEvent(
                        text_so_far=accumulated_text, done=True,
                        response=ClaudeResponse(success=False, text=accumulated_text, error="Claude timed out"),
                    )
                    return

                if not line:
                    # Process died unexpectedly
                    log.error("Claude process EOF")
                    await self._kill()
                    yield StreamEvent(
                        text_so_far=accumulated_text, done=True,
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
                text_so_far=accumulated_text, done=True,
                response=ClaudeResponse(success=False, text=accumulated_text, error=str(e)),
            )

    def force_kill(self) -> None:
        """Kill the subprocess immediately. Safe to call without holding the lock.

        The streaming loop in _send_locked() will see EOF on stdout and
        clean up via the existing error-handling path.
        """
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.kill()
            except ProcessLookupError:
                pass

    async def change_workspace(self, new_workspace: Path) -> None:
        """Switch the working directory. Kills the current process;
        next send() will restart in the new directory."""
        self.workspace = new_workspace
        await self._kill()

    async def restart(self) -> None:
        """Kill and restart the process (for /new command)."""
        await self._kill()

    async def _kill(self) -> None:
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
        """Cleanly shut down."""
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        self._proc = None
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None
