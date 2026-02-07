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
        max_budget_usd: float = 1.0,
        timeout_seconds: int = 120,
    ):
        self.model = model
        self.workspace = workspace
        self.max_budget_usd = max_budget_usd
        self.timeout_seconds = timeout_seconds
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._session_id: str | None = None
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

    async def send(self, prompt: str) -> AsyncIterator[StreamEvent]:
        """Send a message and yield streaming events. Serialized via lock."""
        async with self._lock:
            async for event in self._send_locked(prompt):
                yield event

    async def _send_locked(self, prompt: str) -> AsyncIterator[StreamEvent]:
        try:
            await self._ensure_started()
        except FileNotFoundError:
            yield StreamEvent(
                text_so_far="", done=True,
                response=ClaudeResponse(success=False, text="", error="claude CLI not found"),
            )
            return

        msg = json.dumps({
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
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
                    response = ClaudeResponse(
                        success=not event.get("is_error", False),
                        text=event.get("result", accumulated_text),
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
                                accumulated_text += block.get("text", "")
                                yield StreamEvent(text_so_far=accumulated_text)

        except Exception as e:
            log.exception("Unexpected error reading Claude stream")
            await self._kill()
            yield StreamEvent(
                text_so_far=accumulated_text, done=True,
                response=ClaudeResponse(success=False, text=accumulated_text, error=str(e)),
            )

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
